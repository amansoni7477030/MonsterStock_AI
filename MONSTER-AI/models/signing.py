"""HMAC-SHA256 signing and manifests for model artifacts.

Artifacts are LightGBM text models and JSON documents only (never pickle). A manifest lists every file with its
size and SHA-256 and carries an HMAC over the canonical JSON (``app.jsonutil.dumps``) of everything else, so a
bundle is accepted only if it was produced by a holder of ``MODEL_SIGNING_KEY`` and is exactly what was signed.
Verification failures raise ``ModelArtifactError`` with a precise reason and never include key material.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app import jsonutil
from app.clock import Clock, get_clock, iso
from app.errors import ModelArtifactError
from app.logging_setup import redact

MANIFEST_FORMAT = 1
MANIFEST_KIND = "monster-model-manifest"
MANIFEST_NAME = "manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFEST_FILES = 256
CHUNK = 1024 * 1024

NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,80}")
_VERSION_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,78}[A-Za-z0-9])?")
_STAGE_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FILE_ENTRY_KEYS = frozenset({"size", "sha256"})


def _shown(value: object) -> str:
    """Short, redacted rendering of an untrusted value for error messages (containers show only their type)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return redact(repr(value)[:90])
    return repr(type(value).__name__)


# ------------------------------------------------------------------------------------------- primitives
def _key_bytes(key: str | bytes | None) -> bytes:
    if isinstance(key, str):
        raw = key.encode("utf-8")
    elif isinstance(key, (bytes, bytearray)):
        raw = bytes(key)
    else:
        raw = b""
    if not raw:
        raise ModelArtifactError("model signing key is not configured")
    return raw


def sign_bytes(key: str | bytes, data: bytes) -> str:
    """Hex HMAC-SHA256 of ``data``."""
    return hmac.new(_key_bytes(key), data, hashlib.sha256).hexdigest()


def verify_signature(key: str | bytes, data: bytes, signature: str) -> bool:
    """Constant-time check of a hex HMAC-SHA256 signature (malformed signatures are simply invalid)."""
    expected = sign_bytes(key, data)
    if not isinstance(signature, str) or not _SHA256_RE.fullmatch(signature):
        return False
    return hmac.compare_digest(expected.encode("ascii"), signature.encode("ascii"))


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """The bytes that are signed: canonical JSON (sorted keys, compact, NaN-free)."""
    return jsonutil.dumps(document).encode("utf-8")


def sign_document(document: Mapping[str, Any], key: str | bytes) -> dict[str, Any]:
    """Return a canonicalised copy of ``document`` with its ``signature`` (any previous one is replaced)."""
    body = jsonutil.to_jsonable({k: v for k, v in document.items() if k != "signature"})
    return {**body, "signature": sign_bytes(key, canonical_bytes(body))}


def document_signature_ok(document: Mapping[str, Any], key: str | bytes) -> bool:
    signature = document.get("signature")
    if not isinstance(signature, str):
        return False
    try:
        body = canonical_bytes({k: v for k, v in document.items() if k != "signature"})
    except (TypeError, ValueError, RecursionError):
        return False
    return verify_signature(key, body, signature)


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in pairs:
        if name in out:
            raise ValueError("duplicate JSON key")
        out[name] = value
    return out


def _no_constants(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")


def parse_json_document(raw: bytes, *, limit: int, what: str = "document") -> dict[str, Any]:
    """Strict, size-bounded JSON object parser (no duplicate keys, no NaN/Infinity)."""
    if len(raw) > limit:
        raise ModelArtifactError(f"{what} too large ({len(raw)} bytes, limit {limit})")
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_no_constants)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ModelArtifactError(f"{what} is not valid JSON") from None
    if not isinstance(document, dict):
        raise ModelArtifactError(f"{what} must be a JSON object")
    return document


# ------------------------------------------------------------------------------------------- validation
def check_version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise ModelArtifactError(f"invalid model version {_shown(value)}")
    return value


def check_stage(value: object) -> str:
    if not isinstance(value, str) or not _STAGE_RE.fullmatch(value):
        raise ModelArtifactError(f"invalid model stage {_shown(value)}")
    return value


def check_file_name(name: object) -> str:
    """Bundle file names are plain basenames: ``[A-Za-z0-9._-]{1,80}``, never ``.``/``..``/``manifest.json``."""
    if not isinstance(name, str):
        raise ModelArtifactError(f"invalid file name {_shown(name)}")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ModelArtifactError(f"path traversal in file name {_shown(name)}")
    if not NAME_RE.fullmatch(name):
        raise ModelArtifactError(f"invalid file name {_shown(name)}")
    if name == MANIFEST_NAME:
        raise ModelArtifactError(f"{MANIFEST_NAME} is reserved and cannot be a bundle file")
    return name


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


# ---------------------------------------------------------------------------------------------- manifests
def build_manifest(
    files: Mapping[str, str | Path], *, version: str, stage: str, meta: Mapping[str, Any] | None = None,
    key: str | bytes, clock: Clock | None = None,
) -> dict[str, Any]:
    """Describe ``files`` (bundle name -> path) and sign the result.

    The manifest holds ``format``, ``kind``, ``version``, ``stage``, ``created_at`` (from ``clock``), ``files``
    (``{name: {"size", "sha256"}}``, the layout ``models.artifacts.verify_directory`` reads), ``meta`` and the
    ``signature``.
    """
    check_version(version)
    check_stage(stage)
    if not files:
        raise ModelArtifactError("a model bundle needs at least one file")
    if len(files) > MAX_MANIFEST_FILES:
        raise ModelArtifactError(f"too many files (limit {MAX_MANIFEST_FILES})")
    for name in files:
        check_file_name(name)
    entries: dict[str, dict[str, Any]] = {}
    for name in sorted(files):
        path = Path(files[name])
        if not _regular_file(path):
            raise ModelArtifactError(f"bundle file {_shown(name)} is not a regular file")
        entries[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    body = {
        "format": MANIFEST_FORMAT,
        "kind": MANIFEST_KIND,
        "version": version,
        "stage": stage,
        "created_at": iso((clock or get_clock()).now()),
        "files": entries,
        "meta": dict(meta or {}),
    }
    manifest = sign_document(body, key)
    if len(canonical_bytes(manifest)) > MAX_MANIFEST_BYTES:
        raise ModelArtifactError("manifest too large")
    return manifest


def check_manifest_structure(manifest: Any) -> list[dict[str, Any]]:
    """Validate the shape of a manifest (not its signature); return its file entries as ``{name, size, sha256}``
    dicts sorted by name."""
    if not isinstance(manifest, dict):
        raise ModelArtifactError("manifest must be a JSON object")
    fmt = manifest.get("format")
    if type(fmt) is not int or fmt != MANIFEST_FORMAT:
        raise ModelArtifactError(f"unsupported manifest format version {_shown(fmt)}")
    if manifest.get("kind") != MANIFEST_KIND:
        raise ModelArtifactError("not a model manifest")
    check_version(manifest.get("version"))
    check_stage(manifest.get("stage"))
    created = manifest.get("created_at")
    if not isinstance(created, str) or not 0 < len(created) <= 40:
        raise ModelArtifactError("invalid created_at in manifest")
    if not isinstance(manifest.get("meta"), dict):
        raise ModelArtifactError("invalid meta in manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelArtifactError("manifest lists no files")
    if len(files) > MAX_MANIFEST_FILES:
        raise ModelArtifactError(f"manifest lists too many files (limit {MAX_MANIFEST_FILES})")
    entries: list[dict[str, Any]] = []
    for raw_name in sorted(files, key=str):
        name = check_file_name(raw_name)
        entry = files[name]
        if not isinstance(entry, dict) or set(entry) != _FILE_ENTRY_KEYS:
            raise ModelArtifactError(f"malformed file entry for {_shown(name)} in manifest")
        size = entry["size"]
        if type(size) is not int or size < 0:
            raise ModelArtifactError(f"invalid size for {_shown(name)} in manifest")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ModelArtifactError(f"invalid sha256 for {_shown(name)} in manifest")
        entries.append({"name": name, "size": size, "sha256": digest})
    return entries


def validate_manifest(manifest: Any, key: str | bytes) -> list[dict[str, Any]]:
    """Size, format, signature and structure checks (in that order); return the file entries.

    This is the gate to run before any file of an untrusted bundle is extracted or trusted.
    """
    if not isinstance(manifest, dict):
        raise ModelArtifactError("manifest must be a JSON object")
    try:
        size = len(canonical_bytes(manifest))
    except (TypeError, ValueError, RecursionError):
        raise ModelArtifactError("manifest is not valid canonical JSON") from None
    if size > MAX_MANIFEST_BYTES:
        raise ModelArtifactError(f"manifest too large ({size} bytes, limit {MAX_MANIFEST_BYTES})")
    fmt = manifest.get("format")
    if type(fmt) is not int or fmt != MANIFEST_FORMAT:
        raise ModelArtifactError(f"unsupported manifest format version {_shown(fmt)}")
    if not isinstance(manifest.get("signature"), str):
        raise ModelArtifactError("manifest is unsigned")
    if not document_signature_ok(manifest, key):
        raise ModelArtifactError("manifest signature mismatch (tampered, or signed with a different key)")
    return check_manifest_structure(manifest)


def verify_manifest(manifest: Any, key: str | bytes, directory: str | Path) -> None:
    """Verify ``manifest`` and that ``directory`` holds exactly its files (``manifest.json`` itself is allowed).

    Raises ``ModelArtifactError`` with the precise reason: bad signature, unsupported format, manifest too
    large, path traversal in a name, missing file, size or SHA-256 mismatch, unexpected extra file or symlink.
    """
    entries = validate_manifest(manifest, key)
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ModelArtifactError("bundle directory not found")
    expected = {entry["name"]: entry for entry in entries}
    with os.scandir(root) as listing:
        for child in listing:
            if child.name == MANIFEST_NAME and child.is_file(follow_symlinks=False):
                continue
            if child.name not in expected:
                raise ModelArtifactError(f"unexpected extra file {_shown(child.name)} in bundle")
    for name, entry in expected.items():
        path = root / name
        if path.is_symlink():
            raise ModelArtifactError(f"bundle file {_shown(name)} is a symlink")
        if not path.is_file():
            raise ModelArtifactError(f"missing file {_shown(name)}")
        actual = path.stat().st_size
        if actual != entry["size"]:
            raise ModelArtifactError(f"size mismatch for {_shown(name)}: expected {entry['size']}, found {actual}")
        if sha256_file(path) != entry["sha256"]:
            raise ModelArtifactError(f"sha256 mismatch for {_shown(name)}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest file (bounded; strict JSON). The signature is *not* checked here."""
    target = Path(path)
    try:
        size = target.stat().st_size
        if size > MAX_MANIFEST_BYTES:
            raise ModelArtifactError(f"manifest too large ({size} bytes, limit {MAX_MANIFEST_BYTES})")
        with open(target, "rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError:
        raise ModelArtifactError("manifest not found or unreadable") from None
    return parse_json_document(raw, limit=MAX_MANIFEST_BYTES, what="manifest")
