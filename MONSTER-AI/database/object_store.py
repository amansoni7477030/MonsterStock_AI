"""External object storage for off-box backups and signed model artifacts.

``ObjectStore`` is a minimal blob interface with three backends:

* ``LocalDirStore``: a directory (dev machines, tests, a mounted volume).
* ``HttpReadStore``: read-only HTTPS access to published artifacts (public bucket or CDN URL).
* ``S3Store``: the S3 REST API with a hand-written AWS Signature Version 4 on top of ``requests`` (AWS,
  Cloudflare R2 with region ``auto``, Backblaze B2, MinIO, DigitalOcean Spaces, Wasabi). No boto3.

Rules shared by every backend:

* Keys are relative and use a deliberately small character set (see ``validate_key``), so a key that is valid
  on one backend is valid on all of them and can never traverse out of a directory.
* Large objects are never held in RAM: uploads hash and stream from the file, downloads stream to disk.
* Contract: ``KeyError`` for a missing key (``exists`` returns ``False``), ``ValueError`` for an invalid key or
  argument, ``DataUnavailable`` for any other failure (its ``retryable`` flag says whether trying later may
  help; ``ObjectTooLarge`` when a ``max_bytes`` bound is exceeded).
* Errors never contain credentials, signatures or the Authorization header; text derived from exceptions goes
  through ``app.logging_setup.redact``.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import ipaddress
import logging
import os
import random
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET  # noqa: S405  (only fed through _parse_xml, which rejects DTDs)
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TypeVar
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests

from app.clock import UTC, Clock, get_clock, parse_ts
from app.errors import ConfigError, DataUnavailable
from app.logging_setup import redact, register_secret

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T")

MB = 1024 * 1024
HASH_CHUNK = MB
IO_CHUNK = 256 * 1024
DEFAULT_MAX_BYTES = 512 * MB
DEFAULT_GET_BYTES = 32 * MB
MAX_SINGLE_PUT = 5 * 1024 * MB
MAX_XML_BYTES = 8 * MB
MAX_ERROR_BODY = 64 * 1024
MAX_KEY_LEN = 1024
MAX_KEY_SEGMENTS = 16
MAX_LIST_PAGES = 10_000
DEFAULT_CONTENT_TYPE = "application/octet-stream"
USER_AGENT = "monster-ai-objectstore/1"

ObjectKey = str


@dataclass(frozen=True)
class ObjectInfo:
    """One listed object. ``key`` is relative to the store's prefix; ``last_modified`` is aware UTC."""

    key: str
    size: int
    last_modified: datetime | None = None


ObjectList = list[ObjectInfo]


class ObjectTooLarge(DataUnavailable):
    """An object exceeded the caller's ``max_bytes`` bound (never retried)."""

    def __init__(self, source: str, message: str = "") -> None:
        super().__init__(source, message or "object exceeds the size limit", retryable=False)


# ------------------------------------------------------------------------------------------------ keys
_SEGMENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._~=@+-]{0,254}")
_RESERVED_NAME = re.compile(r"(?i)(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?")


def _key_problem(key: object) -> str | None:
    if not isinstance(key, str) or not key:
        return "must be a non-empty string"
    if len(key) > MAX_KEY_LEN:
        return "is too long"
    if "\x00" in key:
        return "contains a NUL byte"
    if "\\" in key:
        return "contains a backslash"
    if key.startswith("/"):
        return "must be relative (no leading '/')"
    segments = key.split("/")
    if len(segments) > MAX_KEY_SEGMENTS:
        return "has too many path segments"
    for seg in segments:
        if seg in {"", ".", ".."}:
            return "contains an empty, '.' or '..' path segment"
        if not _SEGMENT.fullmatch(seg) or seg.endswith(".") or _RESERVED_NAME.fullmatch(seg):
            return "contains characters or names that are not allowed"
    return None


def validate_key(key: str) -> str:
    """Return ``key`` if it is a safe relative object key.

    Segments are ``[A-Za-z0-9_-]`` followed by ``[A-Za-z0-9._~=@+-]``; no empty, dot or dot-dot segments, no
    leading slash, backslash, NUL, ':' or control characters, at most 1024 chars and 16 segments.
    """
    problem = _key_problem(key)
    if problem:
        raise ValueError(f"invalid object key: {problem}")
    return key


def validate_prefix(prefix: str) -> str:
    """Like ``validate_key`` for listing prefixes: may be empty, end with '/', or end in a partial name."""
    if not isinstance(prefix, str):
        raise ValueError("invalid prefix: must be a string")
    if prefix == "":
        return prefix
    head, _, tail = prefix.rpartition("/")
    problem = None
    if prefix.startswith("/"):
        problem = "must be relative (no leading '/')"
    elif head:
        problem = _key_problem(head)
    if problem is None and tail and not _SEGMENT.fullmatch(tail):
        problem = "contains characters that are not allowed"
    if problem:
        raise ValueError(f"invalid prefix: {problem}")
    return prefix


def _check_content_type(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[\x21-\x7e][\x20-\x7e]{0,254}", value):
        raise ValueError("invalid content type")
    return value


# ----------------------------------------------------------------------------------------- file helpers
def _hash_file(path: Path) -> tuple[str, int]:
    """SHA-256 hex digest and size of a file, read in ``HASH_CHUNK`` pieces."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


_RETRY_REPLACE = os.name == "nt"  # Windows reports short "access denied" spells while a scanner holds a file
_REPLACE_ATTEMPTS = 6
_replace_sleep = time.sleep


def replace_path(src: str | Path, dst: str | Path) -> None:
    """``os.replace`` that, on Windows only, rides out the brief sharing violations caused by virus scanners and
    indexers (bounded: at most ~0.75 s in total). A directory destination is a permanent error, never retried."""
    attempt = 0
    while True:
        attempt += 1
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not _RETRY_REPLACE or attempt >= _REPLACE_ATTEMPTS or Path(dst).is_dir():
                raise
            _replace_sleep(0.05 * attempt)


def remove_tree(path: str | Path) -> None:
    """Best-effort recursive delete that never raises (retries briefly on Windows, like ``replace_path``)."""
    target = Path(path)
    attempt = 0
    while True:
        attempt += 1
        try:
            shutil.rmtree(target)
            return
        except FileNotFoundError:
            return
        except OSError:
            if not _RETRY_REPLACE or attempt >= _REPLACE_ATTEMPTS:
                shutil.rmtree(target, ignore_errors=True)
                return
            _replace_sleep(0.05 * attempt)


@contextlib.contextmanager
def atomic_write(dest: str | Path) -> Iterator[BinaryIO]:
    """Yield a binary file that is fsynced and moved onto ``dest`` only after a clean exit."""
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        replace_path(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _payload(data: object) -> bytes | Path:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, Path):
        if not data.is_file():
            raise FileNotFoundError(f"upload source is not a file: {data.name}")
        return data
    raise TypeError("data must be bytes or a pathlib.Path")


def _local_error(exc: OSError) -> str:
    return redact(exc.strerror or type(exc).__name__)


# --------------------------------------------------------------------------------------------- interface
class ObjectStore(ABC):
    """Blob store used for backups and signed model artifacts. Keys are relative (see ``validate_key``)."""

    @abstractmethod
    def put(
        self, key: str, data: bytes | Path, *, content_type: str = DEFAULT_CONTENT_TYPE
    ) -> None:
        """Store ``data`` (bytes, or a file that is streamed) under ``key``, replacing any previous object."""

    @abstractmethod
    def get(self, key: str, *, max_bytes: int = DEFAULT_GET_BYTES) -> bytes:
        """Return the object; meant for small objects (pointers, manifests). ``KeyError`` when missing."""

    @abstractmethod
    def get_to_file(self, key: str, dest: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> int:
        """Stream the object to ``dest`` (atomically) and return its size. ``KeyError`` when missing."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True if the object exists."""

    @abstractmethod
    def list(self, prefix: str = "") -> ObjectList:
        """Objects whose key starts with ``prefix``, sorted by key."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the object; deleting a missing key is not an error."""

    def close(self) -> None:  # noqa: B027  (optional hook)
        """Release any held resources."""

    def put_file(self, key: str, path: str | Path, *, content_type: str = DEFAULT_CONTENT_TYPE) -> None:
        """Upload a file (streamed, never read into memory): shorthand for ``put(key, Path(path))``."""
        self.put(key, Path(path), content_type=content_type)


# ---------------------------------------------------------------------------------------- local directory
class LocalDirStore(ObjectStore):
    """``ObjectStore`` on a directory. Writes are atomic (temp file + ``os.replace``)."""

    _source = "object_store:local"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._root = self.root.resolve()

    def _path(self, key: str) -> Path:
        validate_key(key)
        path = self._root.joinpath(*key.split("/"))
        try:
            path.resolve().relative_to(self._root)
        except ValueError:
            raise ValueError("invalid object key: resolves outside of the store") from None
        return path

    def _existing(self, key: str) -> Path:
        path = self._path(key)
        if not path.is_file():
            raise KeyError(key)
        return path

    def put(self, key: str, data: bytes | Path, *, content_type: str = DEFAULT_CONTENT_TYPE) -> None:
        path = self._path(key)
        _check_content_type(content_type)
        payload = _payload(data)
        try:
            with atomic_write(path) as out:
                if isinstance(payload, bytes):
                    out.write(payload)
                else:
                    with open(payload, "rb") as src:
                        while chunk := src.read(IO_CHUNK):
                            out.write(chunk)
        except OSError as exc:
            raise DataUnavailable(self._source, f"cannot write {key}: {_local_error(exc)}", retryable=False) from None

    def get(self, key: str, *, max_bytes: int = DEFAULT_GET_BYTES) -> bytes:
        path = self._existing(key)
        try:
            with open(path, "rb") as handle:
                data = handle.read(max_bytes + 1)
        except FileNotFoundError:
            raise KeyError(key) from None
        except OSError as exc:
            raise DataUnavailable(self._source, f"cannot read {key}: {_local_error(exc)}", retryable=False) from None
        if len(data) > max_bytes:
            raise ObjectTooLarge(self._source, f"{key} exceeds the {max_bytes} byte limit")
        return data

    def get_to_file(self, key: str, dest: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> int:
        path = self._existing(key)
        total = 0
        try:
            with open(path, "rb") as src, atomic_write(dest) as out:
                while chunk := src.read(IO_CHUNK):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ObjectTooLarge(self._source, f"{key} exceeds the {max_bytes} byte limit")
                    out.write(chunk)
        except FileNotFoundError:
            raise KeyError(key) from None
        except OSError as exc:
            raise DataUnavailable(self._source, f"cannot copy {key}: {_local_error(exc)}", retryable=False) from None
        return total

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> ObjectList:
        validate_prefix(prefix)
        head = prefix.rpartition("/")[0]
        start = self._root.joinpath(*head.split("/")) if head else self._root
        found: list[ObjectInfo] = []
        if not start.is_dir():
            return found
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                full = Path(dirpath, name)
                if name.startswith(".") or full.is_symlink():
                    continue
                key = full.relative_to(self._root).as_posix()
                if not key.startswith(prefix) or _key_problem(key):
                    continue
                stat = full.stat()
                found.append(ObjectInfo(key, stat.st_size, datetime.fromtimestamp(stat.st_mtime, UTC)))
        found.sort(key=lambda info: info.key)
        return found

    def delete(self, key: str) -> None:
        path = self._path(key)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DataUnavailable(self._source, f"cannot delete {key}: {_local_error(exc)}", retryable=False) from None


# --------------------------------------------------------------------------------------------- AWS SigV4
ALGORITHM = "AWS4-HMAC-SHA256"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def uri_encode(value: str, *, keep_slash: bool = False) -> str:
    """SigV4 ``UriEncode``: percent-encode every UTF-8 byte except ``A-Za-z0-9-._~`` (and '/' if kept)."""
    return quote(value, safe="/" if keep_slash else "")


def canonical_query_string(params: Mapping[str, str] | Iterable[tuple[str, str]]) -> str:
    """Encoded ``name=value`` pairs sorted by name then value, joined with '&' (empty value -> ``name=``)."""
    items = params.items() if isinstance(params, Mapping) else params
    encoded = sorted((uri_encode(str(name)), uri_encode(str(value))) for name, value in items)
    return "&".join(f"{name}={value}" for name, value in encoded)


def canonical_headers(headers: Mapping[str, str]) -> tuple[str, str]:
    """Return ``(canonical header block, signed header names)``: lower-cased, trimmed, sorted by name."""
    normalised = {name.strip().lower(): " ".join(str(value).split()) for name, value in headers.items()}
    names = sorted(normalised)
    return "".join(f"{name}:{normalised[name]}\n" for name in names), ";".join(names)


def canonical_request(
    method: str, canonical_uri: str, canonical_query: str, headers_block: str, signed_headers: str,
    payload_hash: str,
) -> str:
    return "\n".join([method, canonical_uri, canonical_query, headers_block, signed_headers, payload_hash])


def credential_scope(date_stamp: str, region: str, service: str = "s3") -> str:
    return f"{date_stamp}/{region}/{service}/aws4_request"


def string_to_sign(amz_date: str, scope: str, canonical_request_text: str) -> str:
    digest = hashlib.sha256(canonical_request_text.encode("utf-8")).hexdigest()
    return "\n".join([ALGORITHM, amz_date, scope, digest])


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def derive_signing_key(secret_access_key: str, date_stamp: str, region: str, service: str = "s3") -> bytes:
    key = _hmac(("AWS4" + secret_access_key).encode("utf-8"), date_stamp)
    return _hmac(_hmac(_hmac(key, region), service), "aws4_request")


def calculate_signature(signing_key: bytes, string_to_sign_text: str) -> str:
    return hmac.new(signing_key, string_to_sign_text.encode("utf-8"), hashlib.sha256).hexdigest()


def authorization_header(access_key_id: str, scope: str, signed_headers: str, signature: str) -> str:
    return f"{ALGORITHM} Credential={access_key_id}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"


def sign_request(
    *, method: str, canonical_uri: str, query: Mapping[str, str] | Iterable[tuple[str, str]],
    headers: Mapping[str, str], payload_hash: str, access_key_id: str, secret_access_key: str, region: str,
    amz_date: str, service: str = "s3",
) -> str:
    """``Authorization`` header value for a request. ``headers`` must hold exactly the headers to sign."""
    block, signed = canonical_headers(headers)
    request_text = canonical_request(
        method, canonical_uri, canonical_query_string(query), block, signed, payload_hash
    )
    scope = credential_scope(amz_date[:8], region, service)
    key = derive_signing_key(secret_access_key, amz_date[:8], region, service)
    signature = calculate_signature(key, string_to_sign(amz_date, scope, request_text))
    return authorization_header(access_key_id, scope, signed, signature)


# ------------------------------------------------------------------------------------------ HTTP plumbing
_RETRYABLE_STATUS = frozenset({408, 429})
_RETRYABLE_CODES = frozenset(
    {"RequestTimeout", "SlowDown", "InternalError", "ServiceUnavailable", "Throttling", "ThrottlingException",
     "RequestLimitExceeded"}
)
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_ASCII_INT = re.compile(r"[0-9]{1,18}")
_BASE_HEADERS = {"Accept-Encoding": "identity", "User-Agent": USER_AGENT}
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_SIGNATURE_TEXT = re.compile(r"(?i)(signature[=:]\s*)[0-9a-f]{16,}")
_AUTH_TEXT = re.compile(r"(?i)authorization[\"']?\s*[:=].*")


class _Transient(Exception):
    """Internal: a failure worth retrying. ``reason`` is already free of secrets."""

    def __init__(self, reason: str, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


class _NoAuth:
    """Stops requests from applying ~/.netrc credentials, which would replace the SigV4 header."""

    def __call__(self, request: Any) -> Any:
        return request


_NO_AUTH = _NoAuth()


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _split_url(url: str, *, what: str) -> tuple[str, str, str]:
    """Validate an http(s) base URL; return ``(scheme, Host header value, path)``."""
    try:
        parts = urlsplit((url or "").strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise ConfigError(f"{what} is not a valid URL") from None
    scheme = parts.scheme.lower()
    if scheme not in {"https", "http"} or not host:
        raise ConfigError(f"{what} must be an absolute http(s) URL")
    if parts.username is not None or parts.password is not None:
        raise ConfigError(f"{what} must not contain credentials")
    if parts.query or parts.fragment:
        raise ConfigError(f"{what} must not contain a query string or fragment")
    if scheme == "http" and not _is_loopback(host):
        raise ConfigError(f"{what} must use https (plain http is only allowed for localhost)")
    if not host.isascii():
        raise ConfigError(f"{what} must use an ASCII (punycode) host name")
    host_header = f"[{host}]" if ":" in host else host
    if port is not None and port != (443 if scheme == "https" else 80):
        host_header += f":{port}"
    return scheme, host_header, parts.path


def _content_length(response: Any) -> int | None:
    raw = str(response.headers.get("Content-Length") or "").strip()
    return int(raw) if _ASCII_INT.fullmatch(raw) else None


def _close(response: Any) -> None:
    with contextlib.suppress(Exception):
        response.close()


def _iter_body(response: Any) -> Iterator[bytes]:
    try:
        yield from response.iter_content(chunk_size=IO_CHUNK)
    except requests.exceptions.RequestException as exc:
        raise _Transient(f"response interrupted ({type(exc).__name__})") from None


def _iter_limited(response: Any, limit: int) -> Iterator[bytes]:
    """Yield at most ``limit`` bytes of the body (used for error documents)."""
    remaining = limit
    for chunk in _iter_body(response):
        yield chunk[:remaining]
        remaining -= len(chunk)
        if remaining <= 0:
            return


def _check_complete(response: Any, received: int) -> None:
    declared = _content_length(response)
    encoding = str(response.headers.get("Content-Encoding") or "identity").lower()
    if declared is not None and encoding == "identity" and received != declared:
        raise _Transient(f"truncated response ({received} of {declared} bytes)")


def _read_limited(response: Any, limit: int, source: str) -> bytes:
    declared = _content_length(response)
    if declared is not None and declared > limit:
        raise ObjectTooLarge(source, f"object is {declared} bytes, the limit is {limit}")
    chunks: list[bytes] = []
    total = 0
    for chunk in _iter_body(response):
        total += len(chunk)
        if total > limit:
            raise ObjectTooLarge(source, f"object exceeds the {limit} byte limit")
        chunks.append(chunk)
    _check_complete(response, total)
    return b"".join(chunks)


def _stream_to_file(response: Any, dest: Path, limit: int, source: str) -> int:
    declared = _content_length(response)
    if declared is not None and declared > limit:
        raise ObjectTooLarge(source, f"object is {declared} bytes, the limit is {limit}")
    total = 0
    with atomic_write(dest) as out:
        for chunk in _iter_body(response):
            total += len(chunk)
            if total > limit:
                raise ObjectTooLarge(source, f"object exceeds the {limit} byte limit")
            out.write(chunk)
        _check_complete(response, total)
    return total


def _retry_after(response: Any) -> float | None:
    raw = str(response.headers.get("Retry-After") or "").strip()
    return float(raw) if _ASCII_INT.fullmatch(raw) and len(raw) <= 4 else None


_XML_FORBIDDEN = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


def _parse_xml(data: bytes) -> ET.Element:
    """Parse a small XML document, refusing DTDs and entity declarations (billion-laughs / XXE)."""
    if len(data) > MAX_XML_BYTES:
        raise ValueError("XML document too large")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("XML document is not UTF-8") from None
    if "\x00" in text or _XML_FORBIDDEN.search(text):
        raise ValueError("XML DTDs and entities are not allowed")
    try:
        # A str is always parsed as UTF-8 by expat, whatever the XML declaration claims.
        return ET.fromstring(text)  # noqa: S314
    except ET.ParseError:
        raise ValueError("malformed XML") from None


def _local(tag: str) -> str:
    return tag.rpartition("}")[2]


def _child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local(child.tag) == name:
            return child.text or ""
    return ""


class _HttpStore(ObjectStore):
    """Shared retry / error-mapping machinery for the HTTP backends."""

    _source = "object_store"
    _denied_hint = ""

    def __init__(
        self, *, session: Any, clock: Clock | None, max_attempts: int, timeout: tuple[float, float],
        backoff_base: float, backoff_max: float, rng: Callable[[], float] | None,
    ) -> None:
        self._owns_session = session is None
        self._session = requests.Session() if session is None else session
        self._clock = clock
        self._max_attempts = max(1, int(max_attempts))
        self._timeout = timeout
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._rng = rng or random.random

    @property
    def clock(self) -> Clock:
        return self._clock or get_clock()

    def close(self) -> None:
        if self._owns_session:
            with contextlib.suppress(Exception):
                self._session.close()

    # ------------------------------------------------------------------------------------ errors / retries
    def _scrub(self, text: str) -> str:
        text = _AUTH_TEXT.sub("Authorization: <redacted>", text)
        return _SIGNATURE_TEXT.sub(r"\1<redacted>", redact(text))

    def _error(self, message: str, *, retryable: bool) -> DataUnavailable:
        return DataUnavailable(self._source, self._scrub(message), retryable=retryable)

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        ceiling = min(self._backoff_max, self._backoff_base * (2 ** (attempt - 1)))
        delay = ceiling * (0.5 + 0.5 * self._rng())
        if retry_after is not None:
            delay = max(delay, min(retry_after, self._backoff_max))
        return delay

    def _run(self, what: str, operation: Callable[[], T]) -> T:
        attempt = 0
        while True:
            attempt += 1
            try:
                return operation()
            except _Transient as exc:
                if attempt >= self._max_attempts:
                    raise self._error(
                        f"{what} failed after {attempt} attempt(s): {exc.reason}", retryable=True
                    ) from None
                delay = self._backoff(attempt, exc.retry_after)
                log.warning("%s: %s: %s; retry %d/%d in %.1fs", self._source, what, exc.reason, attempt,
                            self._max_attempts - 1, delay)
                self.clock.sleep(delay)

    def _from_request_error(self, exc: requests.exceptions.RequestException) -> Exception:
        config_errors = (
            requests.exceptions.InvalidURL, requests.exceptions.InvalidSchema,
            requests.exceptions.MissingSchema, requests.exceptions.InvalidHeader,
            requests.exceptions.URLRequired,
        )
        if isinstance(exc, config_errors):
            return self._error(f"invalid request ({type(exc).__name__})", retryable=False)
        if isinstance(exc, requests.exceptions.SSLError):
            kind = "TLS error"
        elif isinstance(exc, requests.exceptions.Timeout):
            kind = "timeout"
        elif isinstance(exc, requests.exceptions.ConnectionError):
            kind = "connection error"
        else:
            kind = type(exc).__name__
        return _Transient(f"{kind}: {self._scrub(str(exc))[:200]}")

    def _error_code(self, response: Any) -> str | None:
        return None

    def _redirect_hint(self, response: Any) -> str:
        return ""

    def _classify(self, response: Any, key: str) -> None:
        """Return for 2xx; raise ``KeyError`` / ``_Transient`` / ``DataUnavailable`` otherwise."""
        status = response.status_code
        if 200 <= status < 300:
            return
        code = self._error_code(response)
        detail = f"HTTP {status}" + (f" {code}" if code else "")
        if 300 <= status < 400:
            raise self._error(f"unexpected redirect ({detail}){self._redirect_hint(response)}", retryable=False)
        if status in {401, 403}:
            raise self._error(f"access denied ({detail}){self._denied_hint}", retryable=False)
        if status in {404, 410}:
            if code == "NoSuchBucket":
                raise self._error("bucket not found (HTTP 404 NoSuchBucket); check bucket and endpoint",
                                  retryable=False)
            raise KeyError(key)
        if status in _RETRYABLE_STATUS or status >= 500 or code in _RETRYABLE_CODES:
            raise _Transient(detail, _retry_after(response))
        raise self._error(f"request rejected ({detail})", retryable=False)

    def _to_file(self, response: Any, key: str, dest: Path, max_bytes: int) -> int:
        self._classify(response, key)
        try:
            return _stream_to_file(response, Path(dest), max_bytes, self._source)
        except OSError as exc:
            raise self._error(f"cannot write {Path(dest).name}: {_local_error(exc)}", retryable=False) from None


# --------------------------------------------------------------------------------------------- S3 store
_BUCKET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]")
_REGION = re.compile(r"[a-z0-9-]{1,32}")
_ACCESS_KEY = re.compile(r"[\x21-\x2e\x30-\x7e]{1,128}")
_REGION_HINT = re.compile(r"[a-z0-9-]{1,32}")


def _normalise_prefix(prefix: str | None) -> str:
    text = (prefix or "").strip().strip("/")
    if not text:
        return ""
    problem = _key_problem(text)
    if problem:
        raise ConfigError(f"invalid S3 prefix: {problem}")
    return text + "/"


class S3Store(_HttpStore):
    """S3-compatible store (header-auth SigV4, path-style for custom endpoints, virtual-hosted on AWS)."""

    _source = "object_store:s3"
    _denied_hint = "; check the access key, secret key, bucket permissions and the server clock"

    def __init__(
        self, endpoint_url: str | None, bucket: str, region: str, access_key_id: str, secret_access_key: str,
        prefix: str = "", *, session: Any = None, clock: Clock | None = None, max_attempts: int = 5,
        timeout: tuple[float, float] = (10.0, 60.0), backoff_base: float = 0.5, backoff_max: float = 20.0,
        rng: Callable[[], float] | None = None,
    ) -> None:
        if not _BUCKET.fullmatch(bucket or ""):
            raise ConfigError("invalid S3 bucket name")
        if not _REGION.fullmatch(region or ""):
            raise ConfigError("invalid S3 region")
        if not _ACCESS_KEY.fullmatch(access_key_id or ""):
            raise ConfigError("invalid S3 access key id")
        if not secret_access_key:
            raise ConfigError("missing S3 secret access key")
        prefix = _normalise_prefix(prefix)
        if endpoint_url:
            scheme, host, path = _split_url(endpoint_url, what="S3 endpoint URL")
            if path not in {"", "/"}:
                raise ConfigError("S3 endpoint URL must not contain a path")
            path_style = True
        else:
            domain = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
            scheme, path_style = "https", "." in bucket
            host = f"s3.{region}.{domain}" if path_style else f"{bucket}.s3.{region}.{domain}"
        super().__init__(session=session, clock=clock, max_attempts=max_attempts, timeout=timeout,
                         backoff_base=backoff_base, backoff_max=backoff_max, rng=rng)
        self._bucket = bucket
        self._region = region
        self._access_key_id = access_key_id
        self._secret = secret_access_key
        self._prefix = prefix
        self._scheme, self._host, self._path_style = scheme, host, path_style
        register_secret(secret_access_key)

    def __repr__(self) -> str:
        return f"S3Store(bucket={self._bucket!r}, host={self._host!r}, prefix={self._prefix!r})"

    def _scrub(self, text: str) -> str:
        text = super()._scrub(text)
        for secret in (self._secret, self._access_key_id):
            if len(secret) >= 4:
                text = text.replace(secret, "<redacted>")
        return text

    # ------------------------------------------------------------------------------------------- requests
    def _full_key(self, key: str) -> str:
        return self._prefix + validate_key(key)

    def _object_path(self, full_key: str) -> str:
        encoded = uri_encode(full_key, keep_slash=True)
        return f"/{uri_encode(self._bucket)}/{encoded}" if self._path_style else f"/{encoded}"

    def _bucket_path(self) -> str:
        return f"/{uri_encode(self._bucket)}" if self._path_style else "/"

    def _headers(
        self, method: str, path: str, query: Mapping[str, str], payload_hash: str, content_type: str | None
    ) -> dict[str, str]:
        amz_date = self.clock.now().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        signed = {"host": self._host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
        if content_type:
            signed["content-type"] = content_type
        authorization = sign_request(
            method=method, canonical_uri=path, query=query, headers=signed, payload_hash=payload_hash,
            access_key_id=self._access_key_id, secret_access_key=self._secret, region=self._region,
            amz_date=amz_date,
        )
        headers = {"Host": self._host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date,
                   "Authorization": authorization, **_BASE_HEADERS}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _request(
        self, method: str, path: str, query: Mapping[str, str] | None = None, *, payload_hash: str = EMPTY_SHA256,
        data: Any = None, content_type: str | None = None,
    ) -> Any:
        params = dict(query or {})
        headers = self._headers(method, path, params, payload_hash, content_type)
        url = f"{self._scheme}://{self._host}{path}"
        if params:
            url += "?" + canonical_query_string(params)
        try:
            return self._session.request(
                method, url, headers=headers, data=data, stream=True, timeout=self._timeout,
                allow_redirects=False, auth=_NO_AUTH,
            )
        except requests.exceptions.RequestException as exc:
            raise self._from_request_error(exc) from None

    def _error_code(self, response: Any) -> str | None:
        try:
            body = b"".join(_iter_limited(response, MAX_ERROR_BODY))
            code = _child_text(_parse_xml(body), "Code").strip()
        except (ValueError, _Transient):
            return None
        return code if _ERROR_CODE.fullmatch(code) else None

    def _redirect_hint(self, response: Any) -> str:
        region = str(response.headers.get("x-amz-bucket-region") or "")
        hint = "; check S3_ENDPOINT_URL and S3_REGION"
        return f"{hint} (the bucket reports region {region})" if _REGION_HINT.fullmatch(region) else hint

    # ------------------------------------------------------------------------------------------ operations
    def put(self, key: str, data: bytes | Path, *, content_type: str = DEFAULT_CONTENT_TYPE) -> None:
        path = self._object_path(self._full_key(key))
        _check_content_type(content_type)
        payload = _payload(data)
        if isinstance(payload, bytes):
            payload_hash, body = hashlib.sha256(payload).hexdigest(), payload
        else:
            payload_hash, size = _hash_file(payload)
            if size > MAX_SINGLE_PUT:
                raise ValueError("object is too large for a single PUT")
            # requests would send an empty file object chunked, which S3 rejects; send real bytes instead
            body = b"" if size == 0 else None

        def attempt() -> None:
            if body is not None:
                response = self._request("PUT", path, payload_hash=payload_hash, data=body,
                                         content_type=content_type)
            else:
                with open(payload, "rb") as handle:
                    response = self._request("PUT", path, payload_hash=payload_hash, data=handle,
                                             content_type=content_type)
            try:
                self._classify(response, key)
            finally:
                _close(response)

        self._run(f"PUT {key}", attempt)

    def get(self, key: str, *, max_bytes: int = DEFAULT_GET_BYTES) -> bytes:
        path = self._object_path(self._full_key(key))

        def attempt() -> bytes:
            response = self._request("GET", path)
            try:
                self._classify(response, key)
                return _read_limited(response, max_bytes, self._source)
            finally:
                _close(response)

        return self._run(f"GET {key}", attempt)

    def get_to_file(self, key: str, dest: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> int:
        path = self._object_path(self._full_key(key))

        def attempt() -> int:
            response = self._request("GET", path)
            try:
                return self._to_file(response, key, dest, max_bytes)
            finally:
                _close(response)

        return self._run(f"GET {key}", attempt)

    def exists(self, key: str) -> bool:
        path = self._object_path(self._full_key(key))

        def attempt() -> bool:
            response = self._request("HEAD", path)
            try:
                if response.status_code == 404:
                    return False
                self._classify(response, key)
                return True
            finally:
                _close(response)

        return self._run(f"HEAD {key}", attempt)

    def delete(self, key: str) -> None:
        path = self._object_path(self._full_key(key))

        def attempt() -> None:
            response = self._request("DELETE", path)
            try:
                if response.status_code != 404:
                    self._classify(response, key)
            finally:
                _close(response)

        self._run(f"DELETE {key}", attempt)

    def list(self, prefix: str = "") -> ObjectList:
        validate_prefix(prefix)
        full_prefix = self._prefix + prefix
        found: list[ObjectInfo] = []
        seen_tokens: set[str] = set()
        token: str | None = None
        for _page in range(MAX_LIST_PAGES):
            query = {"list-type": "2", "max-keys": "1000", "encoding-type": "url"}
            if full_prefix:
                query["prefix"] = full_prefix
            if token:
                query["continuation-token"] = token
            root = self._run("LIST", partial(self._list_page, query, prefix))
            token = self._collect(root, found)
            if token is None:
                break
            if token in seen_tokens:
                raise self._error("LIST returned a repeating continuation token", retryable=False)
            seen_tokens.add(token)
        else:
            raise self._error("LIST returned too many pages", retryable=False)
        found.sort(key=lambda info: info.key)
        return found

    def _list_page(self, query: Mapping[str, str], prefix: str) -> ET.Element:
        response = self._request("GET", self._bucket_path(), query)
        try:
            self._classify(response, prefix)
            body = _read_limited(response, MAX_XML_BYTES, self._source)
        finally:
            _close(response)
        try:
            return _parse_xml(body)
        except ValueError as exc:
            raise self._error(f"unexpected LIST response: {exc}", retryable=False) from None

    def _collect(self, root: ET.Element, out: ObjectList) -> str | None:
        """Append the page's objects to ``out``; return the next continuation token (None when done)."""
        if _local(root.tag) != "ListBucketResult":
            raise self._error("unexpected LIST response", retryable=False)
        url_encoded = _child_text(root, "EncodingType").lower() == "url"
        for item in root:
            if _local(item.tag) != "Contents":
                continue
            full_key = _child_text(item, "Key")
            if url_encoded:
                full_key = unquote(full_key)
            if not full_key.startswith(self._prefix):
                continue
            relative = full_key[len(self._prefix):]
            if _key_problem(relative):
                continue  # not something this layer could have written (marker objects, odd names)
            size_text = _child_text(item, "Size").strip()
            out.append(ObjectInfo(
                relative, int(size_text) if _ASCII_INT.fullmatch(size_text) else 0,
                parse_ts(_child_text(item, "LastModified")),
            ))
        if _child_text(root, "IsTruncated").strip().lower() != "true":
            return None
        return _child_text(root, "NextContinuationToken").strip() or None


# --------------------------------------------------------------------------------------- HTTPS read-only
def _redirect_allowed(url: str) -> bool:
    parts = urlsplit(url)
    host = parts.hostname
    if parts.scheme != "https" or not host or parts.username is not None or parts.password is not None:
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return host != "localhost" and not host.endswith(".localhost")


class HttpReadStore(_HttpStore):
    """Read-only store over HTTPS (public bucket or CDN). ``get`` / ``get_to_file`` / ``exists`` only.

    Plain http is accepted for localhost only; redirects are followed manually, at most ``max_redirects``
    times, and only to https targets that are not loopback / private addresses.
    """

    _source = "object_store:http"

    def __init__(
        self, base_url: str, *, session: Any = None, clock: Clock | None = None, max_bytes: int = DEFAULT_MAX_BYTES,
        max_redirects: int = 3, max_attempts: int = 3, timeout: tuple[float, float] = (10.0, 60.0),
        backoff_base: float = 0.5, backoff_max: float = 10.0, rng: Callable[[], float] | None = None,
    ) -> None:
        scheme, host, path = _split_url(base_url, what="artifact base URL")
        super().__init__(session=session, clock=clock, max_attempts=max_attempts, timeout=timeout,
                         backoff_base=backoff_base, backoff_max=backoff_max, rng=rng)
        self._base = f"{scheme}://{host}{path.rstrip('/')}/"
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects

    def __repr__(self) -> str:
        return f"HttpReadStore(base={self._base!r})"

    def _open(self, method: str, key: str) -> Any:
        url = self._base + uri_encode(validate_key(key), keep_slash=True)
        hops = 0
        while True:
            try:
                response = self._session.request(
                    method, url, headers=dict(_BASE_HEADERS), stream=True, timeout=self._timeout,
                    allow_redirects=False, auth=_NO_AUTH,
                )
            except requests.exceptions.RequestException as exc:
                raise self._from_request_error(exc) from None
            if response.status_code not in {301, 302, 303, 307, 308}:
                return response
            location = str(response.headers.get("Location") or "")
            _close(response)
            hops += 1
            if hops > self._max_redirects or not location:
                raise self._error("too many redirects or a redirect without a target", retryable=False)
            try:
                url = urljoin(url, location)
                allowed = _redirect_allowed(url)
            except ValueError:  # e.g. an unbalanced IPv6 bracket in the Location header
                allowed = False
            if not allowed:
                raise self._error("redirect to a disallowed location", retryable=False)

    def put(self, key: str, data: bytes | Path, *, content_type: str = DEFAULT_CONTENT_TYPE) -> None:
        raise NotImplementedError("HttpReadStore is read-only")

    def delete(self, key: str) -> None:
        raise NotImplementedError("HttpReadStore is read-only")

    def list(self, prefix: str = "") -> ObjectList:
        raise NotImplementedError("HttpReadStore cannot list objects")

    def get(self, key: str, *, max_bytes: int = DEFAULT_GET_BYTES) -> bytes:
        limit = min(max_bytes, self._max_bytes)

        def attempt() -> bytes:
            response = self._open("GET", key)
            try:
                self._classify(response, key)
                return _read_limited(response, limit, self._source)
            finally:
                _close(response)

        return self._run(f"GET {key}", attempt)

    def get_to_file(self, key: str, dest: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> int:
        limit = min(max_bytes, self._max_bytes)

        def attempt() -> int:
            response = self._open("GET", key)
            try:
                return self._to_file(response, key, dest, limit)
            finally:
                _close(response)

        return self._run(f"GET {key}", attempt)

    def exists(self, key: str) -> bool:
        def attempt() -> bool:
            response = self._open("HEAD", key)
            try:
                if response.status_code in {404, 410}:
                    return False
                self._classify(response, key)
                return True
            finally:
                _close(response)

        return self._run(f"HEAD {key}", attempt)


# ----------------------------------------------------------------------------------------------- factories
def build_object_store(settings: Settings) -> ObjectStore | None:
    """``S3Store`` from the S3_* settings, or ``None`` when off-box storage is not configured."""
    if not settings.s3_configured:
        return None
    return S3Store(
        settings.s3_endpoint_url or "", settings.s3_bucket or "", settings.s3_region,
        settings.s3_access_key_id or "", settings.s3_secret_access_key or "", settings.s3_prefix,
    )


def _artifact_base_url(url: str) -> str:
    """Accept the artifact base URL, or a URL of its ``latest.json`` / ``manifest.json`` (stripped to the base)."""
    parts = urlsplit(url.strip())
    path = parts.path
    if path.rsplit("/", 1)[-1].lower().endswith(".json"):
        path = path.rsplit("/", 1)[0]
        path = re.sub(r"/models/[a-z][a-z0-9_-]*$", "", path)
    return parts._replace(path=path, query="", fragment="").geturl()


def build_artifact_source(settings: Settings) -> ObjectStore | None:
    """Where signed model bundles are read from: ``MODEL_ARTIFACT_URL`` (read-only HTTPS) when set, else S3."""
    if settings.model_artifact_url:
        return HttpReadStore(_artifact_base_url(settings.model_artifact_url))
    return build_object_store(settings)


__all__ = [
    "DEFAULT_GET_BYTES", "DEFAULT_MAX_BYTES", "HttpReadStore", "LocalDirStore", "ObjectInfo", "ObjectKey",
    "ObjectList", "ObjectStore", "ObjectTooLarge", "S3Store", "atomic_write", "build_artifact_source",
    "build_object_store", "canonical_headers", "canonical_query_string", "canonical_request", "calculate_signature",
    "derive_signing_key", "remove_tree", "replace_path", "sign_request", "string_to_sign", "uri_encode",
    "validate_key", "validate_prefix",
]
