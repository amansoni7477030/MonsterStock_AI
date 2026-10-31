"""Signed model bundles: pack, publish and fetch.

Object layout (relative to the object store's prefix)::

    models/<stage>/<version>.zip   manifest.json + the files it lists (flat; no directories)
    models/<stage>/latest.json     signed pointer: stage, version, sha256 + size of the zip

Everything is signature-gated. ``fetch_bundle`` downloads to a staging directory inside ``dest_dir``, verifies
the pointer (for ``'latest'``) and the manifest signature *before* extracting anything, extracts with zip-slip
and zip-bomb protection, re-verifies every file against the manifest and only then renames the finished
directory to ``dest_dir/<version>`` (atomic). On any failure the staging area is removed, so ``dest_dir`` never
holds a partial or unverified bundle. A replayed (older) pointer is accepted but must verify like any other.
"""

from __future__ import annotations

import contextlib
import errno
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app import jsonutil
from app.clock import Clock, get_clock, iso
from app.errors import ModelArtifactError, ResourceExhausted
from app.logging_setup import redact
from database.object_store import (
    ObjectKey,
    ObjectStore,
    ObjectTooLarge,
    atomic_write,
    remove_tree,
    replace_path,
    validate_key,
)
from models.signing import (
    CHUNK,
    MANIFEST_NAME,
    MAX_MANIFEST_BYTES,
    MAX_MANIFEST_FILES,
    build_manifest,
    check_file_name,
    check_manifest_structure,
    check_stage,
    check_version,
    document_signature_ok,
    load_manifest,
    parse_json_document,
    sha256_file,
    sign_document,
    validate_manifest,
    verify_manifest,
)

BUNDLE_ROOT = "models"
POINTER_NAME = "latest.json"
POINTER_FORMAT = 1
POINTER_KIND = "monster-model-pointer"
DEFAULT_MAX_BYTES = 200 * 1024 * 1024
MAX_POINTER_BYTES = 16 * 1024
MAX_MEMBERS = MAX_MANIFEST_FILES + 1
STALE_AFTER_SEC = 24 * 3600
_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _shown(value: object) -> str:
    return redact(repr(value)[:90])


# ------------------------------------------------------------------------------------------------- keys
def _not_reserved(version: str) -> str:
    if version == "latest":
        raise ModelArtifactError("'latest' is reserved for the pointer and cannot be a model version")
    return version


def bundle_key(stage: str, version: str) -> ObjectKey:
    """Object key of a bundle archive."""
    key = f"{BUNDLE_ROOT}/{check_stage(stage)}/{check_version(_not_reserved(version))}.zip"
    return _checked_key(key)


def pointer_key(stage: str) -> ObjectKey:
    """Object key of the signed ``latest.json`` pointer of a stage."""
    return _checked_key(f"{BUNDLE_ROOT}/{check_stage(stage)}/{POINTER_NAME}")


def _checked_key(key: str) -> str:
    try:
        return validate_key(key)
    except ValueError as exc:
        raise ModelArtifactError(str(exc)) from None


# ------------------------------------------------------------------------------------------------ packing
def _add_member(archive: zipfile.ZipFile, name: str, source: Path) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.file_size = source.stat().st_size
    with open(source, "rb") as src, archive.open(info, "w") as dst:
        shutil.copyfileobj(src, dst, CHUNK)


def pack_bundle(
    directory: str | Path, manifest_path: str | Path, out_zip: str | Path, *, key: str | bytes | None = None
) -> Path:
    """Write ``manifest.json`` plus the files it lists (taken from ``directory``) into ``out_zip``.

    The archive is deterministic (fixed timestamps, sorted members) and written atomically. With ``key`` the
    manifest and files are fully verified first; without it only the manifest's shape and the files' presence.
    """
    root, manifest_file, target = Path(directory), Path(manifest_path), Path(out_zip)
    manifest = load_manifest(manifest_file)
    entries = check_manifest_structure(manifest)
    if key is not None:
        verify_manifest(manifest, key, root)
    for entry in entries:
        path = root / entry["name"]
        if path.is_symlink() or not path.is_file():
            raise ModelArtifactError(f"missing file {_shown(entry['name'])}")
    with atomic_write(target) as raw, zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _add_member(archive, MANIFEST_NAME, manifest_file)
        for entry in sorted(entries, key=lambda e: e["name"]):
            _add_member(archive, entry["name"], root / entry["name"])
    return target


def publish_bundle(
    store: ObjectStore, version: str, directory: str | Path, key: str | bytes, *, stage: str | None = None,
    meta: Mapping[str, Any] | None = None, clock: Clock | None = None, tmp_dir: str | Path | None = None,
) -> ObjectKey:
    """Pack ``directory`` and publish the bundle and a signed ``latest.json`` pointer; return the bundle's key.

    ``directory`` holds the model files and optionally a ``manifest.json``. A *signed* manifest is used (and
    verified) as it is. Without one, or when the manifest is unsigned (the draft written by the training
    pipeline), a fresh signed manifest is built from the directory's files, which are hashed again here
    (``stage`` required; ``meta`` is stored in the manifest). Nothing is uploaded unless the manifest verifies;
    the archive is uploaded first and the pointer last, so a pointer never refers to a missing bundle.
    """
    check_version(_not_reserved(version))
    if stage is not None:
        check_stage(stage)
    root = Path(directory)
    if not root.is_dir():
        raise ModelArtifactError("bundle directory not found")
    scratch = Path(tempfile.mkdtemp(prefix=".publish-", dir=tmp_dir))
    try:
        manifest_file = root / MANIFEST_NAME
        manifest = None
        if manifest_file.is_file():
            existing = load_manifest(manifest_file)
            if "signature" in existing:
                manifest = existing
            elif existing.get("version") not in {None, version}:
                raise ModelArtifactError("the draft manifest.json is for a different version")
        if manifest is None:
            if stage is None:
                raise ModelArtifactError("stage is required when the directory has no signed manifest.json")
            files = {item.name: item for item in sorted(root.iterdir()) if item.name != MANIFEST_NAME}
            manifest = build_manifest(files, version=version, stage=stage, meta=meta, key=key, clock=clock)
            manifest_file = scratch / MANIFEST_NAME
            manifest_file.write_text(jsonutil.dumps(manifest, pretty=True), encoding="utf-8")
        verify_manifest(manifest, key, root)
        if manifest["version"] != version or (stage is not None and manifest["stage"] != stage):
            raise ModelArtifactError("manifest version or stage does not match the bundle being published")
        stage = manifest["stage"]
        archive = pack_bundle(root, manifest_file, scratch / "bundle.zip")
        pointer = sign_document(
            {
                "format": POINTER_FORMAT, "kind": POINTER_KIND, "stage": stage, "version": version,
                "sha256": sha256_file(archive), "size": archive.stat().st_size,
                "created_at": iso((clock or get_clock()).now()),
            },
            key,
        )
        object_key = bundle_key(stage, version)
        store.put(object_key, archive, content_type="application/zip")
        store.put(pointer_key(stage), jsonutil.dumps(pointer, pretty=True).encode("utf-8"),
                  content_type="application/json")
    finally:
        remove_tree(scratch)
    return object_key


# ---------------------------------------------------------------------------------------------- fetching
def read_pointer(source: ObjectStore, key: str | bytes, *, stage: str = "final") -> dict[str, Any]:
    """Fetch and verify ``latest.json`` for ``stage`` (cheap: lets callers skip a download of the same version)."""
    check_stage(stage)
    try:
        raw = source.get(pointer_key(stage), max_bytes=MAX_POINTER_BYTES)
    except KeyError:
        raise ModelArtifactError(f"no published model bundle for stage {_shown(stage)}") from None
    except ObjectTooLarge:
        raise ModelArtifactError("latest.json is too large") from None
    pointer = parse_json_document(raw, limit=MAX_POINTER_BYTES, what="latest.json")
    fmt = pointer.get("format")
    if type(fmt) is not int or fmt != POINTER_FORMAT:
        raise ModelArtifactError(f"unsupported latest.json format version {_shown(fmt)}")
    if not isinstance(pointer.get("signature"), str):
        raise ModelArtifactError("latest.json is unsigned")
    if not document_signature_ok(pointer, key):
        raise ModelArtifactError("latest.json signature mismatch (tampered, or signed with a different key)")
    if pointer.get("kind") != POINTER_KIND:
        raise ModelArtifactError("latest.json is not a model pointer")
    if pointer.get("stage") != stage:
        raise ModelArtifactError("latest.json is for a different stage")
    check_version(pointer.get("version"))
    digest, size = pointer.get("sha256"), pointer.get("size")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or type(size) is not int or size <= 0:
        raise ModelArtifactError("latest.json has an invalid sha256 or size")
    return pointer


class _Budget:
    """Running total of unpacked bytes; raises once the allowance is exceeded."""

    def __init__(self, limit: int) -> None:
        self.remaining = limit

    def take(self, amount: int) -> None:
        self.remaining -= amount
        if self.remaining < 0:
            raise ModelArtifactError("bundle expands beyond the allowed size (zip bomb?)")


def _check_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Reject anything but a flat set of plain, unencrypted, regular files (zip-slip, symlinks, nested dirs)."""
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ModelArtifactError(f"bundle has too many members (limit {MAX_MEMBERS})")
    members: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        name = info.filename
        if name != MANIFEST_NAME:
            check_file_name(name)
        if name in members:
            raise ModelArtifactError(f"duplicate member {_shown(name)} in bundle")
        if info.flag_bits & 0x1:
            raise ModelArtifactError(f"encrypted member {_shown(name)} in bundle")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ModelArtifactError(f"unsupported compression for member {_shown(name)}")
        mode = info.external_attr >> 16
        if mode and not stat.S_ISREG(mode):
            raise ModelArtifactError(f"symlink or special file {_shown(name)} in bundle")
        members[name] = info
    if MANIFEST_NAME not in members:
        raise ModelArtifactError("bundle has no manifest.json")
    return members


def _read_manifest_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.file_size > MAX_MANIFEST_BYTES:
        raise ModelArtifactError("manifest too large")
    with archive.open(info) as handle:
        data = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ModelArtifactError("manifest too large")
    return data


def _copy_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path, declared: int, budget: _Budget
) -> None:
    """Extract one member, counting the bytes actually produced (not just the declared size)."""
    written = 0
    with archive.open(info) as src, open(dest, "xb") as dst:
        while chunk := src.read(CHUNK):
            written += len(chunk)
            budget.take(len(chunk))
            if written > declared:
                raise ModelArtifactError(f"member {_shown(info.filename)} is larger than declared")
            dst.write(chunk)
    if written != declared:
        raise ModelArtifactError(f"member {_shown(info.filename)} is shorter than declared")


def _extract_verified(
    archive_path: Path, content_dir: Path, key: str | bytes, *, version: str, stage: str, limit: int
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _check_members(archive)
            manifest_bytes = _read_manifest_member(archive, members[MANIFEST_NAME])
            manifest = parse_json_document(manifest_bytes, limit=MAX_MANIFEST_BYTES, what="manifest")
            entries = validate_manifest(manifest, key)
            if manifest["version"] != version or manifest["stage"] != stage:
                raise ModelArtifactError("bundle manifest is for a different version or stage")
            if set(members) != {entry["name"] for entry in entries} | {MANIFEST_NAME}:
                raise ModelArtifactError("bundle members do not match the manifest (extra or missing files)")
            for entry in entries:
                if members[entry["name"]].file_size != entry["size"]:
                    raise ModelArtifactError(f"declared size of {_shown(entry['name'])} differs from the manifest")
            _Budget(limit).take(sum(entry["size"] for entry in entries))
            budget = _Budget(limit)
            content_dir.mkdir()
            (content_dir / MANIFEST_NAME).write_bytes(manifest_bytes)
            for entry in entries:
                _copy_member(archive, members[entry["name"]], content_dir / entry["name"], entry["size"], budget)
    except (zipfile.BadZipFile, zlib.error, EOFError):
        raise ModelArtifactError("bundle is not a valid zip archive (corrupt, truncated or unsupported)") from None
    verify_manifest(manifest, key, content_dir)
    return manifest


def _already_installed(target: Path, manifest: dict[str, Any], key: str | bytes) -> bool:
    try:
        verify_manifest(manifest, key, target)
    except ModelArtifactError:
        return False
    return True


def _install(content: Path, target: Path, manifest: dict[str, Any], key: str | bytes) -> Path:
    """Atomically move the verified directory to ``target`` (an identical existing install is kept)."""
    if target.exists() or target.is_symlink():
        if _already_installed(target, manifest, key):
            return target
        stale = target.with_name(f".stale-{uuid.uuid4().hex}")
        replace_path(target, stale)
        try:
            replace_path(content, target)
        except OSError:
            with contextlib.suppress(OSError):
                replace_path(stale, target)
            raise
        remove_tree(stale)
        return target
    try:
        replace_path(content, target)
    except OSError:
        if _already_installed(target, manifest, key):  # a concurrent fetch installed the same bundle first
            return target
        raise
    return target


def _sweep_stale(root: Path) -> None:
    """Remove staging leftovers of crashed fetches (older than a day)."""
    cutoff = time.time() - STALE_AFTER_SEC
    with contextlib.suppress(OSError):
        for child in root.iterdir():
            if child.name.startswith((".staging-", ".stale-")) and child.is_dir() and child.stat().st_mtime < cutoff:
                remove_tree(child)


def _local_failure(action: str, exc: OSError) -> Exception:
    if exc.errno == errno.ENOSPC:
        return ResourceExhausted(f"disk full while trying to {action}")
    return ModelArtifactError(f"cannot {action}: {redact(exc.strerror or type(exc).__name__)}")


def fetch_bundle(
    source: ObjectStore, version: str, dest_dir: str | Path, key: str | bytes, *, stage: str = "final",
    max_bytes: int = DEFAULT_MAX_BYTES, max_unpacked_bytes: int | None = None,
) -> Path:
    """Download, verify and atomically install a bundle; return ``dest_dir/<version>``.

    ``version`` is a version string or ``'latest'`` (resolved through the signed pointer). ``max_bytes`` bounds
    the download, ``max_unpacked_bytes`` (default ``4 * max_bytes``) the extracted size. Raises
    ``ModelArtifactError`` for anything wrong with the artifact (missing, unsigned, tampered, oversized, unsafe
    archive), ``DataUnavailable`` when the store cannot deliver, ``ResourceExhausted`` when the disk is full.
    """
    check_stage(stage)
    pointer = None
    if version == "latest":
        pointer = read_pointer(source, key, stage=stage)
        version = pointer["version"]
        if pointer["size"] > max_bytes:
            raise ModelArtifactError(f"bundle is larger than the {max_bytes} byte limit")
    else:
        check_version(version)
    object_key = bundle_key(stage, version)
    limit = max_unpacked_bytes if max_unpacked_bytes is not None else 4 * max_bytes
    root = Path(dest_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        _sweep_stale(root)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    except OSError as exc:
        raise _local_failure("prepare the model directory", exc) from None
    try:
        archive = staging / "bundle.zip"
        try:
            source.get_to_file(object_key, archive, max_bytes=max_bytes)
        except KeyError:
            raise ModelArtifactError(f"bundle {object_key} not found") from None
        except ObjectTooLarge:
            raise ModelArtifactError(f"bundle {object_key} exceeds the {max_bytes} byte limit") from None
        if pointer is not None and (
            archive.stat().st_size != pointer["size"] or sha256_file(archive) != pointer["sha256"]
        ):
            raise ModelArtifactError(
                "bundle does not match the signed latest.json pointer (truncated, replaced or tampered)"
            )
        content = staging / "content"
        manifest = _extract_verified(archive, content, key, version=version, stage=stage, limit=limit)
        return _install(content, root / version, manifest, key)
    except OSError as exc:
        raise _local_failure("fetch the model bundle", exc) from None
    finally:
        remove_tree(staging)


def verify_bundle_dir(directory: str | Path, key: str | bytes) -> dict[str, Any]:
    """Re-verify an installed bundle directory (e.g. at model load time); return its verified manifest."""
    root = Path(directory)
    manifest = load_manifest(root / MANIFEST_NAME)
    verify_manifest(manifest, key, root)
    return manifest


__all__ = [
    "BUNDLE_ROOT", "DEFAULT_MAX_BYTES", "POINTER_NAME", "bundle_key", "fetch_bundle", "pack_bundle",
    "pointer_key", "publish_bundle", "read_pointer", "verify_bundle_dir",
]
