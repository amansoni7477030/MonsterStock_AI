"""Resilient HTTP for data providers.

Every call to a scraped / public data source (NSE, Yahoo, publisher RSS, GDELT ...) goes through ``HttpClient``:

* **transport** - ``curl_cffi`` (libcurl with a real browser TLS / HTTP-2 fingerprint, so CDNs and bot managers such as the
  ones in front of NSE and Yahoo treat the server as an ordinary browser) with an automatic fallback to ``requests`` when
  curl_cffi cannot be imported or rejects the impersonation target. A caller-supplied session (tests) is used as is.
* **politeness** - per-provider spacing with *adaptive* back-off (a 429 slows that provider down, successes speed it up
  again), jitter, ``Retry-After`` handling and a circuit breaker so a dead provider fails fast instead of burning the
  09:12 cutoff.
* **safety** - bounded retries with exponential back-off, a hard response-size cap (RAM guard), HTTPS only, and
  secret-safe errors and logs (host + path only, never the query string).
* **traceability** - one DEBUG line per request, WARNING lines for retries / give-ups, and per-source counters that the
  job runner logs as an HTTP summary (``http_summary``).

Impersonation sets the browser's own ``User-Agent`` / ``Accept-Encoding`` / ``sec-ch-*`` headers. Layering a different
``User-Agent`` on top would *contradict* the TLS fingerprint (an easy bot signal), so those headers are never forwarded
when impersonating.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests

from app.clock import Clock, get_clock
from app.errors import CircuitOpen, DataUnavailable
from app.logging_setup import redact

if TYPE_CHECKING:  # pragma: no cover
    from app.config import Settings

try:  # a binary wheel: a missing / broken native library must degrade to ``requests``, never break an import
    from curl_cffi import requests as cffi_requests
    from curl_cffi.requests import exceptions as cffi_exc
except Exception as _import_error:  # noqa: BLE001
    cffi_requests: Any = None
    cffi_exc: Any = None
    CFFI_IMPORT_ERROR = f"{type(_import_error).__name__}: {_import_error}"
else:
    CFFI_IMPORT_ERROR = ""

log = logging.getLogger(__name__)

DEFAULT_RETRY_STATUSES = (408, 425, 429, 500, 502, 503, 504)
USER_AGENT_SIMPLE = "Mozilla/5.0"
USER_AGENT_BROWSER = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

TRANSPORT_AUTO, TRANSPORT_CURL, TRANSPORT_REQUESTS = "auto", "curl_cffi", "requests"
TRANSPORTS = (TRANSPORT_AUTO, TRANSPORT_CURL, TRANSPORT_REQUESTS)
TRANSPORT_INJECTED = "injected"
DEFAULT_IMPERSONATE = "chrome"  # alias: the newest Chrome profile the installed curl_cffi knows
MAX_REDIRECTS = 8
SLOW_REQUEST_SEC = 15.0
# Headers the browser profile owns; forwarding our own copy while impersonating would contradict the TLS fingerprint.
FINGERPRINT_HEADERS = frozenset({"user-agent", "accept-encoding", "connection"})


# ------------------------------------------------------------------------------------------ transports
def curl_cffi_info() -> dict[str, Any]:
    """Availability of the curl_cffi transport (for ``doctor`` and start-up logs)."""
    if cffi_requests is None:
        return {"available": False, "error": CFFI_IMPORT_ERROR}
    try:
        import curl_cffi

        return {"available": True, "version": getattr(curl_cffi, "__version__", "?"),
                "libcurl": str(getattr(curl_cffi, "__curl_version__", "?"))}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


_REQUESTS_TRANSIENT: tuple[type[BaseException], ...] = (
    requests.exceptions.Timeout, requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError, requests.exceptions.SSLError,
)


def _cffi_types(*names: str) -> tuple[type[BaseException], ...]:
    if cffi_exc is None:
        return ()
    return tuple(getattr(cffi_exc, n) for n in names if isinstance(getattr(cffi_exc, n, None), type))


# Malformed requests: retrying cannot help.
_CFFI_FATAL = _cffi_types("InvalidURL", "InvalidSchema", "MissingSchema", "InvalidHeader", "InvalidProxyURL",
                          "TooManyRedirects", "UnrewindableBodyError", "URLRequired", "SessionClosed", "CookieConflict")
_CFFI_IMPERSONATE = _cffi_types("ImpersonateError")
_CFFI_ANY = _cffi_types("RequestException")


def classify_error(exc: BaseException) -> str | None:
    """How to treat an exception raised by an HTTP library: ``"transient"`` (retry), ``"fatal"`` (do not retry),
    ``"impersonate"`` (the browser profile is unsupported) or ``None`` (not an HTTP library error: propagate).

    libcurl protocol hiccups (HTTP/2 stream resets, partial reads, TLS resets) surface as curl_cffi ``HTTPError`` /
    ``IncompleteRead`` / ``SSLError`` and are retried; only malformed requests are fatal."""
    if _CFFI_IMPERSONATE and isinstance(exc, _CFFI_IMPERSONATE):
        return "impersonate"
    if _CFFI_FATAL and isinstance(exc, _CFFI_FATAL):
        return "fatal"
    if _CFFI_ANY and isinstance(exc, _CFFI_ANY):
        return "transient"
    if isinstance(exc, _REQUESTS_TRANSIENT):
        return "transient"
    if isinstance(exc, requests.exceptions.RequestException):
        return "fatal"
    return None


_LOGGED_ONCE: set[str] = set()
_WARNED = _LOGGED_ONCE  # kept for tests that reset it


def _log_once(level: int, key: str, message: str, *args: Any) -> None:
    if key not in _LOGGED_ONCE:
        _LOGGED_ONCE.add(key)
        log.log(level, message, *args)


def _warn_once(key: str, message: str, *args: Any) -> None:
    _log_once(logging.WARNING, key, message, *args)


# ------------------------------------------------------------------------------------------ data types
class Headers(Mapping[str, str]):
    """Read-only, case-insensitive header mapping (``response.headers.get("etag")`` works whatever the server sent)."""

    def __init__(self, items: Mapping[str, Any] | Iterable[tuple[Any, Any]] = ()) -> None:
        pairs = items.items() if isinstance(items, Mapping) else items
        self._data = {str(k).lower(): str(v) for k, v in pairs}

    def __getitem__(self, key: str) -> str:
        return self._data[str(key).lower()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Headers({self._data!r})"


@dataclass
class HttpResponse:
    status: int
    content: bytes
    headers: Mapping[str, str]
    host: str
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self, encoding: str | None = None) -> str:
        return self.content.decode(encoding or "utf-8", errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.content)
        except ValueError as exc:
            raise DataUnavailable(self.host, "response is not valid JSON", retryable=True) from exc


@dataclass
class HttpStats:
    """Counters for one provider within this process (a job run)."""

    requests: int = 0       # attempts that reached the network
    ok: int = 0
    retries: int = 0
    rate_limited: int = 0   # 429 / Retry-After answers
    not_modified: int = 0   # 304 answers to conditional requests
    failures: int = 0       # calls that ended in an error
    bytes: int = 0
    total_ms: float = 0.0
    slowest_ms: float = 0.0
    last_error: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {**self.__dict__, "total_ms": round(self.total_ms, 1), "slowest_ms": round(self.slowest_ms, 1)}

    def reset(self) -> None:
        for name, value in HttpStats().__dict__.items():
            setattr(self, name, value)


_STATS: dict[str, HttpStats] = {}
_STATS_LOCK = threading.Lock()


def stats_for(source: str) -> HttpStats:
    with _STATS_LOCK:
        return _STATS.setdefault(source, HttpStats())


def http_stats() -> dict[str, dict[str, Any]]:
    """Snapshot of every provider's counters in this process (providers without any traffic are omitted)."""
    with _STATS_LOCK:
        return {name: s.snapshot() for name, s in sorted(_STATS.items()) if s.requests or s.failures}


def reset_http_stats() -> None:
    """Zero every counter in place (clients keep a reference to their provider's counters)."""
    with _STATS_LOCK:
        for stats in _STATS.values():
            stats.reset()


def http_summary() -> str:
    """One log line: ``yahoo 36 req / 0 retry / 0 x429 / 1.2 MB / 41 s; nse_api ...`` (empty when nothing was fetched)."""
    parts = []
    for name, s in http_stats().items():
        extras = "".join(f" / {s[k]} {label}" for k, label in
                         (("failures", "failed"), ("not_modified", "not-modified")) if s[k])
        parts.append(f"{name} {s['requests']} req / {s['retries']} retry / {s['rate_limited']} x429{extras} / "
                     f"{s['bytes'] / 1e6:.1f} MB / {s['total_ms'] / 1000:.0f} s")
    return "; ".join(parts)


# ----------------------------------------------------------------------------------------- politeness
class RateLimiter:
    """Minimum spacing between consecutive requests (thread-safe) that adapts to the provider's mood.

    ``slow_down`` (called on 429 / ``Retry-After``) doubles the spacing up to ``max_interval``; every success multiplies it
    by 0.9 until it is back at the configured base. ``jitter`` adds up to that fraction of random extra spacing so
    requests are not perfectly periodic."""

    def __init__(self, min_interval: float, clock: Clock | None = None, *, jitter: float = 0.0,
                 rng: random.Random | None = None, max_interval: float | None = None) -> None:
        self.base_interval = max(0.0, float(min_interval))
        self.min_interval = self.base_interval
        self.max_interval = max(float(max_interval), self.base_interval) if max_interval else max(self.base_interval * 8.0, 10.0)
        self.jitter = max(0.0, float(jitter))
        self._rng = rng or random.Random()
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        clock = self._clock or get_clock()
        with self._lock:
            now = clock.monotonic()
            if self._last is not None:
                spacing = self.min_interval * (1.0 + self.jitter * self._rng.random())
                remaining = spacing - (now - self._last)
                if remaining > 0:
                    clock.sleep(remaining)
                    now = clock.monotonic()
            self._last = now

    def slow_down(self, factor: float = 2.0, floor: float = 0.5) -> float:
        with self._lock:
            self.min_interval = min(self.max_interval, max(self.min_interval * factor, floor))
            return self.min_interval

    def speed_up(self, factor: float = 0.9) -> float:
        with self._lock:
            if self.min_interval > self.base_interval:
                self.min_interval = max(self.base_interval, self.min_interval * factor)
            return self.min_interval


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; half-opens after ``cooldown`` seconds."""

    def __init__(self, threshold: int = 6, cooldown: float = 300.0, clock: Clock | None = None) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self.failures = 0
        self._opened_at: float | None = None

    def _now(self) -> float:
        return (self._clock or get_clock()).monotonic()

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if self._now() - self._opened_at >= self.cooldown:
            return True  # half-open: let one probe through
        return False

    def record_success(self) -> None:
        self.failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self._opened_at = self._now()

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None and not self.allow()


# ------------------------------------------------------------------------------------------- client
@dataclass
class HttpClient:
    source: str
    headers: Mapping[str, str] = field(default_factory=lambda: {"User-Agent": USER_AGENT_SIMPLE, "Accept": "*/*"})
    min_interval: float = 0.0
    timeout: tuple[float, float] = (10.0, 30.0)
    max_attempts: int = 4
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    max_bytes: int = 40 * 1024 * 1024
    session: Any = None
    clock: Clock | None = None
    breaker: CircuitBreaker | None = None
    retry_statuses: tuple[int, ...] = DEFAULT_RETRY_STATUSES
    allow_http_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    rng: random.Random = field(default_factory=random.Random)
    on_auth_failure: Callable[[], None] | None = None
    transport: str = TRANSPORT_AUTO
    impersonate: str | None = DEFAULT_IMPERSONATE
    jitter: float = 0.0

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.min_interval, self.clock, jitter=self.jitter, rng=self.rng)
        if self.breaker is None:
            self.breaker = CircuitBreaker(clock=self.clock)
        self.stats = stats_for(self.source)
        self.impersonate = self.impersonate or None
        if self.session is None:
            self._own_session = True
            self._transport = self._resolve_transport()
            self.session = self._new_session()
            _log_once(logging.INFO, f"transport:{self._transport}:{self.impersonate}", "http transport: %s%s (impersonate=%s)",
                      self._transport, f" {curl_cffi_info().get('version', '')}" if self._transport == TRANSPORT_CURL else "",
                      self.impersonate or "off")
        else:
            self._own_session = False
            self._transport = TRANSPORT_INJECTED

    @classmethod
    def from_settings(cls, source: str, settings: Settings, **overrides: Any) -> HttpClient:
        """A client configured from ``Settings`` (transport, impersonation, timeout, attempts, size cap); any keyword
        argument overrides the settings-derived value."""
        options: dict[str, Any] = {
            "transport": settings.http_transport, "impersonate": settings.http_impersonate or None, "jitter": 0.2,
            "timeout": (10.0, settings.http_timeout_sec), "max_attempts": settings.http_max_attempts,
            "max_bytes": settings.http_max_response_mb * 1024 * 1024,
        }
        options.update(overrides)
        return cls(source, **options)

    # ------------------------------------------------------------------------------------- plumbing
    def _clock(self) -> Clock:
        return self.clock or get_clock()

    def _resolve_transport(self) -> str:
        wanted = (self.transport or TRANSPORT_AUTO).lower()
        if wanted not in TRANSPORTS:
            _warn_once(f"transport:{wanted}", "unknown HTTP transport %r; using auto", wanted)
            wanted = TRANSPORT_AUTO
        if wanted in (TRANSPORT_AUTO, TRANSPORT_CURL):
            if cffi_requests is not None:
                return TRANSPORT_CURL
            _warn_once("curl_cffi-missing", "curl_cffi is not usable (%s): falling back to requests; some providers "
                       "may block this client", CFFI_IMPORT_ERROR or "not installed")
        return TRANSPORT_REQUESTS

    def _session_headers(self) -> dict[str, str]:
        return self._without_fingerprint(self.headers)

    def _without_fingerprint(self, headers: Mapping[str, str] | None) -> dict[str, str]:
        if not headers:
            return {}
        if self._transport == TRANSPORT_CURL and self.impersonate:
            return {k: v for k, v in headers.items() if k.lower() not in FINGERPRINT_HEADERS}
        return dict(headers)

    def _new_session(self) -> Any:
        if self._transport == TRANSPORT_CURL:
            try:
                session = cffi_requests.Session(**({"impersonate": self.impersonate} if self.impersonate else {}))
            except Exception as exc:  # noqa: BLE001 - e.g. the native library refuses to load
                _warn_once(f"curl_cffi-session:{self.source}", "http %s: curl_cffi session failed (%s: %s); using requests",
                           self.source, type(exc).__name__, exc)
                self._transport = TRANSPORT_REQUESTS
                return self._new_session()
            session.headers.update(self._session_headers())
            return session
        session = requests.Session()
        session.headers.update(dict(self.headers))
        return session

    def reset_session(self) -> None:
        """Drop cookies / connection state (used for NSE re-handshakes)."""
        if self._own_session:
            with contextlib.suppress(Exception):
                self.session.close()
            self.session = self._new_session()

    def close(self) -> None:
        if self._own_session:
            with contextlib.suppress(Exception):
                self.session.close()

    def info(self) -> dict[str, Any]:
        return {"source": self.source, "transport": self._transport, "impersonate": self.impersonate,
                "min_interval": self._limiter.min_interval}

    def _recover_transport(self) -> bool:
        """The browser profile was rejected: drop impersonation, then drop curl_cffi. ``True`` = state changed, retry now."""
        if not self._own_session or self._transport != TRANSPORT_CURL:
            return False
        if self.impersonate:
            _warn_once(f"impersonate:{self.impersonate}", "http %s: impersonation target %r is not supported by this "
                       "curl_cffi; continuing without impersonation", self.source, self.impersonate)
            self.impersonate = None
        else:
            _warn_once("curl_cffi-degraded", "http %s: curl_cffi keeps failing; falling back to requests", self.source)
            self._transport = TRANSPORT_REQUESTS
        self.reset_session()
        return True

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), 60.0)
        base = min(self.backoff_cap, self.backoff_base * (2 ** (attempt - 1)))
        return base * (0.5 + self.rng.random() * 0.5)

    @staticmethod
    def _retry_after(headers: Mapping[str, str]) -> float | None:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _read_body(self, response: Any) -> bytes:
        declared = response.headers.get("Content-Length") if hasattr(response, "headers") else None
        try:
            if declared is not None and int(declared) > self.max_bytes:
                raise DataUnavailable(self.source, "response exceeds size cap", retryable=False)
        except (TypeError, ValueError):
            pass
        chunks: list[bytes] = []
        total = 0
        # curl_cffi chooses its own chunking and warns when asked for a size; requests needs one (default is 1 byte)
        pieces = response.iter_content() if self._transport == TRANSPORT_CURL else response.iter_content(chunk_size=65536)
        for chunk in pieces:
            if not chunk:
                continue
            total += len(chunk)
            if total > self.max_bytes:
                raise DataUnavailable(self.source, "response exceeds size cap", retryable=False)
            chunks.append(chunk)
        return b"".join(chunks)

    def _send(self, method: str, url: str, *, params: Mapping[str, Any] | None, headers: Mapping[str, str] | None,
              data: Any, json_body: Any, timeout: tuple[float, float] | None, deadline: float | None = None) -> Any:
        wait = timeout or self.timeout
        if deadline is not None:  # never let one attempt outlive the caller's deadline (connect + read are summed by libcurl)
            left = max(deadline - self._clock().monotonic(), 0.2)
            wait = (min(wait[0], left / 2.0), min(wait[1], left / 2.0))
        kwargs: dict[str, Any] = {
            "params": params, "headers": self._without_fingerprint(headers) or None, "data": data, "json": json_body,
            "timeout": wait, "stream": True, "allow_redirects": True,
        }
        if self._transport == TRANSPORT_CURL:
            kwargs["max_redirects"] = MAX_REDIRECTS
        return self.session.request(method, url, **kwargs)

    # ---------------------------------------------------------------------------------------- request
    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        data: Any = None,
        json_body: Any = None,
        soft_statuses: Iterable[int] = (),
        timeout: tuple[float, float] | None = None,
        deadline: float | None = None,
    ) -> HttpResponse:
        """Perform a request. Statuses in ``soft_statuses`` (e.g. 404 for "file not published yet", 304 for a conditional
        GET) are returned to the caller; other non-2xx statuses raise ``DataUnavailable``.

        ``deadline`` is a ``clock.monotonic()`` instant: no attempt starts after it, each attempt's timeout is capped to what
        is left, and a back-off that would cross it ends the call. Time-critical callers (the 09:00-09:12 pre-open polls) use
        it so a slow provider can never push a report past its cutoff."""
        parsed = urlparse(url)
        host = parsed.hostname or "?"
        if parsed.scheme != "https" and host not in self.allow_http_hosts:
            raise DataUnavailable(self.source, "refusing non-HTTPS URL", retryable=False)
        where = f"{host}{parsed.path[:100]}"  # never the query string: it may carry an API key
        assert self.breaker is not None
        if not self.breaker.allow():
            raise CircuitOpen(self.source, "circuit open after repeated failures", retryable=True)
        soft = set(soft_statuses)
        stats = self.stats
        last_error = "unknown error"
        retryable = True
        attempt = 0

        while attempt < self.max_attempts:
            attempt += 1
            self._limiter.wait()
            if deadline is not None and self._clock().monotonic() >= deadline:
                stats.failures += 1
                stats.last_error = "deadline reached"
                log.warning("http %s %s %s: deadline reached before attempt %d/%d", self.source, method, where, attempt,
                            self.max_attempts)
                raise DataUnavailable(self.source, "deadline reached", retryable=True)
            started = self._clock().monotonic()
            retry_after: float | None = None
            stats.requests += 1
            try:
                response = self._send(method, url, params=params, headers=headers, data=data, json_body=json_body,
                                      timeout=timeout, deadline=deadline)
            except Exception as exc:  # noqa: BLE001 - classified below; anything that is not an HTTP-library error propagates
                kind = classify_error(exc)
                if kind is None:
                    raise
                if kind == "impersonate" and self._recover_transport():
                    attempt -= 1
                    continue
                if kind == "fatal" or kind == "impersonate":
                    stats.failures += 1
                    stats.last_error = f"request failed: {type(exc).__name__}"
                    self.breaker.record_failure()
                    log.warning("http %s: request to %s rejected (%s)", self.source, where, type(exc).__name__)
                    raise DataUnavailable(self.source, f"request failed: {type(exc).__name__}", retryable=False) from None
                last_error = type(exc).__name__
                retryable = True
            else:
                try:
                    status = int(response.status_code)
                    if status in self.retry_statuses and attempt < self.max_attempts:
                        retry_after = self._retry_after(response.headers)
                        last_error = f"HTTP {status}"
                        retryable = True
                        if status == 429 or retry_after is not None:
                            stats.rate_limited += 1
                            self._limiter.slow_down()
                        response.close()
                    else:
                        try:
                            body = self._read_body(response)
                        except (requests.exceptions.RequestException, OSError) as exc:
                            last_error = f"body read failed: {type(exc).__name__}"
                            retryable = True
                            body = None
                        if body is not None:
                            elapsed = (self._clock().monotonic() - started) * 1000.0
                            result = HttpResponse(status, body, Headers(response.headers), host, elapsed)
                            stats.bytes += len(body)
                            stats.total_ms += elapsed
                            stats.slowest_ms = max(stats.slowest_ms, elapsed)
                            if 200 <= status < 300 or status in soft:
                                if status == 304:
                                    stats.not_modified += 1
                                stats.ok += 1
                                self.breaker.record_success()
                                self._limiter.speed_up()
                                level = logging.INFO if elapsed > SLOW_REQUEST_SEC * 1000 else logging.DEBUG
                                log.log(level, "http %s %s %s -> %d in %.0f ms, %d bytes (attempt %d/%d, %s)", self.source,
                                        method, where, status, elapsed, len(body), attempt, self.max_attempts, self._transport)
                                return result
                            if status in (401, 403) and self.on_auth_failure and attempt < self.max_attempts:
                                self.on_auth_failure()
                                last_error, retryable = f"HTTP {status}", True
                            else:
                                if status == 429:
                                    stats.rate_limited += 1
                                    self._limiter.slow_down()
                                if status >= 500 or status in self.retry_statuses:
                                    self.breaker.record_failure()
                                stats.failures += 1
                                stats.last_error = f"HTTP {status}"
                                if status in self.retry_statuses:
                                    log.warning("http %s %s %s: giving up after %d attempts (HTTP %d)", self.source, method,
                                                where, attempt, status)
                                else:
                                    log.warning("http %s %s %s -> HTTP %d (not retried)", self.source, method, where, status)
                                raise DataUnavailable(self.source, f"HTTP {status}",
                                                      retryable=status in self.retry_statuses or status >= 500)
                finally:
                    with contextlib.suppress(Exception):
                        response.close()
            if attempt < self.max_attempts:
                delay = self._delay(attempt, retry_after)
                if deadline is not None and self._clock().monotonic() + delay >= deadline:
                    last_error = f"{last_error}; deadline reached"
                    break  # the next attempt could not finish in time: fail now so the caller can fall back
                stats.retries += 1
                log.warning("http %s %s %s: %s (attempt %d/%d), retrying in %.1fs", self.source, method, where,
                            redact(last_error), attempt, self.max_attempts, delay)
                self._clock().sleep(delay)

        self.breaker.record_failure()
        stats.failures += 1
        stats.last_error = last_error
        log.warning("http %s %s %s: giving up after %d attempts (%s)", self.source, method, where, attempt,
                    redact(last_error))
        raise DataUnavailable(self.source, f"{last_error} after {attempt} attempts", retryable=retryable)

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("GET", url, **kwargs)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.get(url, **kwargs).json()
