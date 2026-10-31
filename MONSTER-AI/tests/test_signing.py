"""Signing, manifest and signed-bundle tests. Hermetic: temp dirs, FakeClock, no network, no real sleeping."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import shutil
import stat
import time
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from requests.structures import CaseInsensitiveDict

from app import jsonutil
from app.errors import DataUnavailable, ModelArtifactError, ResourceExhausted
from database import object_store as osx
from database.object_store import HttpReadStore, LocalDirStore
from models import bundle as bd
from models import signing as sg

KEY = "unit-test-signing-key-0123456789"
OTHER_KEY = "a-completely-different-key-987654"
FILES = {
    "model.txt": b"tree\nnum_leaves=31\nsplit_feature=0 1 2\n" * 40,
    "calibrator.json": b'{"kind": "isotonic", "x": [0.1, 0.5, 0.9]}',
    "schema.json": b'{"features": ["a", "b"]}',
}


def make_dir(root: Path, files: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, data in (files or FILES).items():
        (root / name).write_bytes(data)
    return root


def signed_manifest(root: Path, **kwargs) -> dict:
    kwargs.setdefault("version", "v1")
    kwargs.setdefault("stage", "final")
    kwargs.setdefault("meta", {"auc": 0.61})
    files = {p.name: p for p in sorted(root.iterdir()) if p.name != sg.MANIFEST_NAME}
    return sg.build_manifest(files, key=KEY, **kwargs)


def forge(manifest: dict, mutate) -> dict:
    """A manifest with a *valid* signature over deliberately bad content."""
    body = {k: v for k, v in manifest.items() if k != "signature"}
    mutate(body)
    return sg.sign_document(body, KEY)


def flip(data: bytes) -> bytes:
    return data[:5] + bytes([data[5] ^ 1]) + data[6:]


# ------------------------------------------------------------------------------------------- primitives
def test_hmac_known_answers_rfc4231():
    assert sg.sign_bytes(b"\x0b" * 20, b"Hi There") == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    assert sg.sign_bytes("Jefe", b"what do ya want for nothing?") == (
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")
    assert sg.sign_bytes("k\u00e9y", b"x") == sg.sign_bytes("k\u00e9y".encode(), b"x")  # str keys are UTF-8


def test_verify_signature_is_strict_and_constant_time(monkeypatch):
    calls = []
    real = sg.hmac.compare_digest
    monkeypatch.setattr(sg.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    signature = sg.sign_bytes(KEY, b"data")
    assert sg.verify_signature(KEY, b"data", signature) is True and calls
    assert sg.verify_signature(KEY, b"other", signature) is False
    assert sg.verify_signature(OTHER_KEY, b"data", signature) is False
    for bad in (signature.upper(), signature[:-1], signature + "0", "z" * 64, "", None, 123, b"x" * 64):
        assert sg.verify_signature(KEY, b"data", bad) is False  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["", b"", None, 5])
def test_empty_or_invalid_keys_are_refused(key):
    with pytest.raises(ModelArtifactError, match="not configured"):
        sg.sign_bytes(key, b"x")
    with pytest.raises(ModelArtifactError, match="not configured"):
        sg.verify_signature(key, b"x", "0" * 64)


def test_sha256_file_streams_in_chunks(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abcdefghij" * 100)
    monkeypatch.setattr(sg, "CHUNK", 3)
    assert sg.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sign_document_replaces_signatures_and_tolerates_junk():
    doc = sg.sign_document({"a": 1, "signature": "stale"}, KEY)
    assert doc["a"] == 1 and sg.document_signature_ok(doc, KEY) and not sg.document_signature_ok(doc, OTHER_KEY)
    assert not sg.document_signature_ok({"a": 1}, KEY) and not sg.document_signature_ok({"signature": 5}, KEY)
    cyclic: dict = {"signature": "0" * 64}
    cyclic["self"] = cyclic
    assert sg.document_signature_ok(cyclic, KEY) is False  # not canonicalisable: invalid, never an exception


@pytest.mark.parametrize("version", ["v1", "2026-11-02.1", "a", "final-20261102_abc", "A" * 80])
def test_valid_versions(version):
    assert sg.check_version(version) == version


@pytest.mark.parametrize("version", ["", ".v1", "v1.", "-v", "a/b", "..", "a b", "A" * 81, None, 3, "v\n1"])
def test_invalid_versions(version):
    with pytest.raises(ModelArtifactError, match="invalid model version"):
        sg.check_version(version)


@pytest.mark.parametrize("stage", ["", "Final", "9x", "a/b", "x" * 33, None])
def test_invalid_stages(stage):
    with pytest.raises(ModelArtifactError, match="invalid model stage"):
        sg.check_stage(stage)


# -------------------------------------------------------------------------------------------- manifests
def test_build_manifest_content_and_canonical_signature(tmp_path, fake_clock):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root, meta={"auc": 0.6, "bad": float("nan"), "pair": (1, 2)})
    assert (manifest["format"], manifest["kind"]) == (1, sg.MANIFEST_KIND)
    assert (manifest["version"], manifest["stage"]) == ("v1", "final")
    assert manifest["created_at"] == "2026-11-02T03:00:00Z"  # from the injected clock
    assert list(manifest["files"]) == ["calibrator.json", "model.txt", "schema.json"]
    for name, entry in manifest["files"].items():
        data = FILES[name]
        assert entry == {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert manifest["meta"] == {"auc": 0.6, "bad": None, "pair": [1, 2]}
    body = {k: v for k, v in manifest.items() if k != "signature"}
    assert manifest["signature"] == sg.sign_bytes(KEY, jsonutil.dumps(body).encode("utf-8"))
    sg.verify_manifest(manifest, KEY, root)
    sg.verify_manifest(json.loads(json.dumps(manifest)), KEY, root)  # what actually travels in the bundle
    assert signed_manifest(root, meta={"auc": 0.6, "bad": None, "pair": [1, 2]}) == manifest  # deterministic


def test_build_manifest_uses_the_clock_argument(tmp_path, fake_clock):
    root = make_dir(tmp_path / "m")
    fake_clock.advance(3600)
    assert signed_manifest(root, clock=fake_clock)["created_at"] == "2026-11-02T04:00:00Z"


@pytest.mark.parametrize("name", [
    "../x", "a/b", "a\\b", "", ".", "..", "x" * 81, "has space", "manifest.json", "caf\u00e9", "a:b", None, 7,
])
def test_build_manifest_rejects_unsafe_file_names(tmp_path, name):
    root = make_dir(tmp_path / "m")
    with pytest.raises(ModelArtifactError):
        sg.build_manifest({name: root / "model.txt"}, version="v1", stage="final", key=KEY)


def test_build_manifest_rejects_bad_inputs(tmp_path):
    root = make_dir(tmp_path / "m")
    good = {"model.txt": root / "model.txt"}
    with pytest.raises(ModelArtifactError, match="at least one file"):
        sg.build_manifest({}, version="v1", stage="final", key=KEY)
    with pytest.raises(ModelArtifactError, match="not a regular file"):
        sg.build_manifest({"missing.txt": root / "missing.txt"}, version="v1", stage="final", key=KEY)
    with pytest.raises(ModelArtifactError, match="not a regular file"):
        (root / "sub").mkdir()
        sg.build_manifest({"sub": root / "sub"}, version="v1", stage="final", key=KEY)
    with pytest.raises(ModelArtifactError, match="invalid model version"):
        sg.build_manifest(good, version="../v1", stage="final", key=KEY)
    with pytest.raises(ModelArtifactError, match="invalid model stage"):
        sg.build_manifest(good, version="v1", stage="Final", key=KEY)
    with pytest.raises(ModelArtifactError, match="not configured"):
        sg.build_manifest(good, version="v1", stage="final", key="")
    many = {f"f{i}.txt": root / "model.txt" for i in range(sg.MAX_MANIFEST_FILES + 1)}
    with pytest.raises(ModelArtifactError, match="too many files"):
        sg.build_manifest(many, version="v1", stage="final", key=KEY)
    with pytest.raises(ModelArtifactError, match="manifest too large"):
        sg.build_manifest(good, version="v1", stage="final", key=KEY, meta={"blob": "x" * (2 * 1024 * 1024)})


def test_build_manifest_rejects_symlinked_files(tmp_path):
    root = make_dir(tmp_path / "m")
    try:
        os.symlink(root / "model.txt", root / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    with pytest.raises(ModelArtifactError, match="not a regular file"):
        sg.build_manifest({"link.txt": root / "link.txt"}, version="v1", stage="final", key=KEY)


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(version="v2"),
    lambda m: m.update(stage="pre"),
    lambda m: m.update(created_at="2030-01-01T00:00:00Z"),
    lambda m: m["meta"].update(auc=0.99),
    lambda m: m["files"]["model.txt"].update(size=1),
    lambda m: m["files"]["model.txt"].update(sha256="0" * 64),
    lambda m: m["files"].pop("schema.json"),
    lambda m: m.update(extra="field"),
    lambda m: m.pop("meta"),
    lambda m: m.update(signature="0" * 64),
    lambda m: m.update(signature=m["signature"].upper()),
], ids=["version", "stage", "created_at", "meta", "size", "sha", "file-dropped", "extra-key", "meta-dropped",
        "zeroed-signature", "uppercased-signature"])
def test_any_edit_of_a_signed_manifest_breaks_the_signature(tmp_path, mutate):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    mutate(manifest)
    with pytest.raises(ModelArtifactError, match="signature mismatch"):
        sg.verify_manifest(manifest, KEY, root)


def test_wrong_key_and_unsigned_manifests_are_rejected(tmp_path):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    with pytest.raises(ModelArtifactError, match="signature mismatch"):
        sg.verify_manifest(manifest, OTHER_KEY, root)
    with pytest.raises(ModelArtifactError, match="not configured"):
        sg.verify_manifest(manifest, "", root)
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    with pytest.raises(ModelArtifactError, match="unsigned"):
        sg.verify_manifest(unsigned, KEY, root)
    with pytest.raises(ModelArtifactError, match="unsigned"):
        sg.verify_manifest({**manifest, "signature": 7}, KEY, root)


def test_file_level_tampering_is_detected(tmp_path):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    sg.verify_manifest(manifest, KEY, root)
    model = root / "model.txt"
    original = model.read_bytes()

    model.write_bytes(flip(original))  # one flipped byte, same size
    with pytest.raises(ModelArtifactError, match=r"sha256 mismatch for 'model.txt'"):
        sg.verify_manifest(manifest, KEY, root)
    model.write_bytes(original + b"x")
    with pytest.raises(ModelArtifactError, match=r"size mismatch for 'model.txt': expected \d+, found \d+"):
        sg.verify_manifest(manifest, KEY, root)
    model.write_bytes(original[:-1])
    with pytest.raises(ModelArtifactError, match="size mismatch"):
        sg.verify_manifest(manifest, KEY, root)
    model.unlink()
    with pytest.raises(ModelArtifactError, match=r"missing file 'model.txt'"):
        sg.verify_manifest(manifest, KEY, root)
    model.write_bytes(original)
    sg.verify_manifest(manifest, KEY, root)


def test_extra_files_directories_and_symlinks_are_rejected(tmp_path):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    write = (root / sg.MANIFEST_NAME)
    write.write_text(jsonutil.dumps(manifest))
    sg.verify_manifest(manifest, KEY, root)  # manifest.json itself is allowed in the directory
    (root / "extra.bin").write_bytes(b"x")
    with pytest.raises(ModelArtifactError, match=r"unexpected extra file 'extra.bin'"):
        sg.verify_manifest(manifest, KEY, root)
    (root / "extra.bin").unlink()
    (root / "sub").mkdir()
    with pytest.raises(ModelArtifactError, match="unexpected extra file 'sub'"):
        sg.verify_manifest(manifest, KEY, root)
    (root / "sub").rmdir()
    write.unlink()
    write.mkdir()  # a directory named manifest.json is not the manifest
    with pytest.raises(ModelArtifactError, match="unexpected extra file"):
        sg.verify_manifest(manifest, KEY, root)
    write.rmdir()
    with pytest.raises(ModelArtifactError, match="directory not found"):
        sg.verify_manifest(manifest, KEY, tmp_path / "nope")
    with pytest.raises(ModelArtifactError, match="directory not found"):
        sg.verify_manifest(manifest, KEY, root / "model.txt")


def test_symlinked_bundle_files_are_rejected(tmp_path):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_bytes(FILES["schema.json"])
    (root / "schema.json").unlink()
    try:
        os.symlink(elsewhere, root / "schema.json")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    with pytest.raises(ModelArtifactError, match="symlink"):
        sg.verify_manifest(manifest, KEY, root)


def rename_file(body: dict, new_name) -> None:
    body["files"][new_name] = body["files"].pop("model.txt")


@pytest.mark.parametrize("name", ["../evil", "/etc/passwd", "a/b", "..", "a\\b", "C:\\x"])
def test_signed_manifests_with_path_traversal_names_are_rejected(tmp_path, name):
    root = make_dir(tmp_path / "m")
    manifest = forge(signed_manifest(root), lambda b: rename_file(b, name))
    with pytest.raises(ModelArtifactError, match="path traversal"):
        sg.verify_manifest(manifest, KEY, root)


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda b: rename_file(b, "has space"), "invalid file name"),
    (lambda b: rename_file(b, "x" * 81), "invalid file name"),
    (lambda b: rename_file(b, "manifest.json"), "reserved"),
    (lambda b: b["files"]["model.txt"].update(extra=1), "malformed file entry"),
    (lambda b: b["files"]["model.txt"].pop("size"), "malformed file entry"),
    (lambda b: b["files"].update({"other.txt": "string-entry"}), "malformed file entry"),
    (lambda b: b["files"]["model.txt"].update(size=-1), "invalid size"),
    (lambda b: b["files"]["model.txt"].update(size=True), "invalid size"),
    (lambda b: b["files"]["model.txt"].update(size="12"), "invalid size"),
    (lambda b: b["files"]["model.txt"].update(size=1.5), "invalid size"),
    (lambda b: b["files"]["model.txt"].update(sha256="ABC"), "invalid sha256"),
    (lambda b: b["files"]["model.txt"].update(sha256="G" * 64), "invalid sha256"),
    (lambda b: b["files"]["model.txt"].update(sha256=None), "invalid sha256"),
    (lambda b: b.update(files={}), "lists no files"),
    (lambda b: b.update(files="nope"), "lists no files"),
    (lambda b: b.update(files=[{"name": "model.txt", "size": 1, "sha256": "0" * 64}]), "lists no files"),
    (lambda b: b.update(files={f"f{i}.txt": dict(b["files"]["model.txt"]) for i in range(257)}), "too many files"),
    (lambda b: b.update(kind="something-else"), "not a model manifest"),
    (lambda b: b.update(version="../v"), "invalid model version"),
    (lambda b: b.update(stage="Final"), "invalid model stage"),
    (lambda b: b.update(created_at=""), "created_at"),
    (lambda b: b.update(created_at=5), "created_at"),
    (lambda b: b.update(meta=[1]), "invalid meta"),
])
def test_signed_but_malformed_manifests_are_rejected_precisely(tmp_path, mutate, message):
    root = make_dir(tmp_path / "m")
    forged = forge(signed_manifest(root), mutate)
    with pytest.raises(ModelArtifactError, match=message):
        sg.verify_manifest(forged, KEY, root)


@pytest.mark.parametrize("fmt", [2, 0, "1", True, None, 1.0, [1], {"v": 1}])
def test_unsupported_format_versions(tmp_path, fmt):
    root = make_dir(tmp_path / "m")
    manifest = {**signed_manifest(root), "format": fmt}  # rejected before the signature is even looked at
    with pytest.raises(ModelArtifactError, match="unsupported manifest format version") as err:
        sg.verify_manifest(manifest, KEY, root)
    if isinstance(fmt, (list, dict)):
        assert str(err.value).endswith(f"'{type(fmt).__name__}'")  # containers are never echoed
    with pytest.raises(ModelArtifactError, match="unsupported manifest format version"):
        sg.verify_manifest(forge(manifest, lambda b: b.update(format=fmt)), KEY, root)


def test_oversized_and_non_object_manifests(tmp_path):
    root = make_dir(tmp_path / "m")
    huge = {**signed_manifest(root), "meta": {"blob": "x" * (sg.MAX_MANIFEST_BYTES + 10)}}
    with pytest.raises(ModelArtifactError, match="manifest too large"):
        sg.verify_manifest(huge, KEY, root)
    for junk in ([], "text", None, 5):
        with pytest.raises(ModelArtifactError, match="must be a JSON object"):
            sg.verify_manifest(junk, KEY, root)
    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(ModelArtifactError, match="canonical JSON"):
        sg.verify_manifest(cyclic, KEY, root)


def test_load_manifest_is_bounded_and_strict(tmp_path):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert sg.load_manifest(path) == manifest
    big = tmp_path / "big.json"
    big.write_bytes(b" " * (sg.MAX_MANIFEST_BYTES + 1))
    for bad, message in ((b"{not json", "not valid JSON"), (b'{"a": NaN}', "not valid JSON"),
                         (b'{"a": 1, "a": 2}', "not valid JSON"), (b"[1]", "must be a JSON object"),
                         (b"\xff\xfe{}", "not valid JSON"), (b'{"a": Infinity}', "not valid JSON")):
        path.write_bytes(bad)
        with pytest.raises(ModelArtifactError, match=message):
            sg.load_manifest(path)
    with pytest.raises(ModelArtifactError, match="manifest too large"):
        sg.load_manifest(big)
    with pytest.raises(ModelArtifactError, match="not found"):
        sg.load_manifest(tmp_path / "missing.json")
    with pytest.raises(ModelArtifactError, match="too large"):
        sg.parse_json_document(b"x" * 10, limit=5, what="thing")


# ------------------------------------------------------------------------------------- bundle helpers
@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """The Windows-only retry of replace/remove never really sleeps in tests."""
    monkeypatch.setattr(osx, "_replace_sleep", lambda seconds: None)


@pytest.fixture()
def env(tmp_path, fake_clock):
    return SimpleNamespace(tmp=tmp_path, store=LocalDirStore(tmp_path / "remote"), dest=tmp_path / "models",
                           clock=fake_clock)


def publish(env, version="v1", stage="final", files=None, meta=None, key=KEY):
    directory = make_dir(env.tmp / f"src-{stage}-{version}", files)
    return bd.publish_bundle(env.store, version, directory, key, stage=stage, meta=meta or {"trained": "2026-11-01"},
                             clock=env.clock)


def write_zip(path: Path, members, *, compression=zipfile.ZIP_DEFLATED, attrs=None) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about duplicate names, which some tests create on purpose
        with zipfile.ZipFile(path, "w", compression) as archive:
            for name, data in members:
                info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = compression
                info.external_attr = (attrs or {}).get(name, (stat.S_IFREG | 0o644) << 16)
                archive.writestr(info, data)
    return path


def read_members(store, key) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(store.get(key, max_bytes=50_000_000))) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def put_zip(env, key, members, **kwargs):
    items = list(members.items()) if isinstance(members, dict) else list(members)
    env.store.put(key, write_zip(env.tmp / "crafted.zip", items, **kwargs))


def assert_clean(dest: Path) -> None:
    """Nothing - no version directory, no staging leftovers - may remain after a rejected fetch."""
    assert not dest.exists() or list(dest.rglob("*")) == []


# ------------------------------------------------------------------------------------ pack / publish
def test_pack_bundle_is_deterministic_flat_and_atomic(tmp_path, fake_clock):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    (root / sg.MANIFEST_NAME).write_text(jsonutil.dumps(manifest, pretty=True))
    first = bd.pack_bundle(root, root / sg.MANIFEST_NAME, tmp_path / "a.zip", key=KEY)
    second = bd.pack_bundle(root, root / sg.MANIFEST_NAME, tmp_path / "b.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["manifest.json", "calibrator.json", "model.txt", "schema.json"]
        for info in archive.infolist():
            assert info.date_time == (2020, 1, 1, 0, 0, 0)
            assert stat.S_ISREG(info.external_attr >> 16) and info.compress_type == zipfile.ZIP_DEFLATED
        assert archive.testzip() is None


def test_pack_bundle_refuses_bad_directories(tmp_path, fake_clock):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    (root / sg.MANIFEST_NAME).write_text(jsonutil.dumps(manifest))
    with pytest.raises(ModelArtifactError, match="signature mismatch"):
        bd.pack_bundle(root, root / sg.MANIFEST_NAME, tmp_path / "x.zip", key=OTHER_KEY)
    assert not (tmp_path / "x.zip").exists() and not list(tmp_path.glob(".x.zip*"))
    (root / "schema.json").unlink()
    with pytest.raises(ModelArtifactError, match="missing file"):
        bd.pack_bundle(root, root / sg.MANIFEST_NAME, tmp_path / "x.zip")
    with pytest.raises(ModelArtifactError, match="manifest not found"):
        bd.pack_bundle(root, root / "nope.json", tmp_path / "x.zip")


def test_publish_uploads_the_bundle_first_then_a_signed_pointer(env):
    class Recording(LocalDirStore):
        order: list = []

        def put(self, key, data, **kwargs):
            Recording.order.append(key)
            super().put(key, data, **kwargs)

    env.store = Recording(env.tmp / "remote")
    assert publish(env) == "models/final/v1.zip"
    assert Recording.order == ["models/final/v1.zip", "models/final/latest.json"]
    pointer = json.loads(env.store.get("models/final/latest.json"))
    archive = env.store.get("models/final/v1.zip")
    assert pointer == bd.read_pointer(env.store, KEY, stage="final")
    assert (pointer["format"], pointer["kind"], pointer["stage"], pointer["version"]) == (
        1, bd.POINTER_KIND, "final", "v1")
    assert pointer["sha256"] == hashlib.sha256(archive).hexdigest() and pointer["size"] == len(archive)
    assert pointer["created_at"] == "2026-11-02T03:00:00Z"
    assert sg.document_signature_ok(pointer, KEY) and not sg.document_signature_ok(pointer, OTHER_KEY)
    members = read_members(env.store, "models/final/v1.zip")
    assert list(members)[0] == "manifest.json" and set(members) == {"manifest.json", *FILES}
    assert json.loads(members["manifest.json"])["meta"] == {"trained": "2026-11-01"}


def test_publish_with_a_prebuilt_manifest(env):
    root = make_dir(env.tmp / "src")
    (root / sg.MANIFEST_NAME).write_text(jsonutil.dumps(signed_manifest(root, version="v7", stage="pre")))
    assert bd.publish_bundle(env.store, "v7", root, KEY) == "models/pre/v7.zip"
    assert env.store.exists("models/pre/latest.json") and not env.store.exists("models/final/latest.json")
    with pytest.raises(ModelArtifactError, match="does not match the bundle being published"):
        bd.publish_bundle(env.store, "v8", root, KEY)
    with pytest.raises(ModelArtifactError, match="does not match the bundle being published"):
        bd.publish_bundle(env.store, "v7", root, KEY, stage="final")


def draft_manifest(root: Path, version="v1") -> dict:
    """The unsigned ``manifest.json`` the training pipeline writes next to a model: file -> {size, sha256}."""
    files = {p.name: {"size": p.stat().st_size, "sha256": sg.sha256_file(p)} for p in sorted(root.iterdir())}
    draft = {"format": 1, "version": version, "files": files}
    (root / "manifest.json").write_text(json.dumps(draft, indent=2))
    return draft


def test_an_unsigned_draft_manifest_is_resigned_from_the_directory_contents(env):
    root = make_dir(env.tmp / "src")
    draft = draft_manifest(root)
    draft_bytes = (root / "manifest.json").read_bytes()
    assert bd.publish_bundle(env.store, "v1", root, KEY, stage="final", meta={"auc": 0.6}, clock=env.clock) == (
        "models/final/v1.zip")
    assert (root / "manifest.json").read_bytes() == draft_bytes  # the source directory is never modified
    shipped = json.loads(read_members(env.store, "models/final/v1.zip")["manifest.json"])
    assert shipped["signature"] and shipped["kind"] == sg.MANIFEST_KIND and shipped["meta"] == {"auc": 0.6}
    assert shipped["files"] == draft["files"]  # same layout, files re-hashed from the directory itself
    path = bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert bd.verify_bundle_dir(path, KEY)["stage"] == "final"


def test_draft_manifests_are_not_trusted_for_content_or_version(env):
    root = make_dir(env.tmp / "src")
    draft = draft_manifest(root)
    draft["files"]["model.txt"]["sha256"] = "0" * 64  # a stale/forged hash in the draft changes nothing...
    (root / "manifest.json").write_text(json.dumps(draft))
    bd.publish_bundle(env.store, "v1", root, KEY, stage="final", clock=env.clock)
    shipped = json.loads(read_members(env.store, "models/final/v1.zip")["manifest.json"])
    assert shipped["files"]["model.txt"]["sha256"] == hashlib.sha256(FILES["model.txt"]).hexdigest()
    other = make_dir(env.tmp / "other")
    draft_manifest(other, version="v9")
    with pytest.raises(ModelArtifactError, match="draft manifest.json is for a different version"):
        bd.publish_bundle(env.store, "v1", other, KEY, stage="final")
    with pytest.raises(ModelArtifactError, match="stage is required"):
        bd.publish_bundle(env.store, "v9", other, KEY)
    signed_elsewhere = make_dir(env.tmp / "elsewhere")
    manifest = sg.build_manifest({p.name: p for p in sorted(signed_elsewhere.iterdir())}, version="v1", stage="final",
                                 key=OTHER_KEY)
    (signed_elsewhere / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ModelArtifactError, match="signature mismatch"):  # a *signed* manifest is never replaced
        bd.publish_bundle(env.store, "v1", signed_elsewhere, KEY, stage="final")


def test_fetched_bundles_satisfy_the_registry_verifier(env):
    artifacts = pytest.importorskip("models.artifacts")
    publish(env)
    path = bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    manifest = artifacts.verify_directory(path)  # the model registry's own pre-load check accepts the signed manifest
    assert manifest["version"] == "v1" and set(manifest["files"]) == set(FILES)
    (path / "model.txt").write_bytes(flip(FILES["model.txt"]))
    with pytest.raises(ModelArtifactError):
        artifacts.verify_directory(path)
    with pytest.raises(ModelArtifactError, match="sha256 mismatch"):
        bd.verify_bundle_dir(path, KEY)


def test_publish_rejects_invalid_bundles_and_uploads_nothing(env):
    root = make_dir(env.tmp / "src")
    (root / sg.MANIFEST_NAME).write_text(jsonutil.dumps(signed_manifest(root)))
    (root / "model.txt").write_bytes(flip(FILES["model.txt"]))  # tampered after signing
    scratch = {"tmp_dir": env.tmp}
    with pytest.raises(ModelArtifactError, match="sha256 mismatch"):
        bd.publish_bundle(env.store, "v1", root, KEY, **scratch)
    with pytest.raises(ModelArtifactError, match="signature mismatch"):
        bd.publish_bundle(env.store, "v1", root, OTHER_KEY, **scratch)
    bare = make_dir(env.tmp / "bare")
    with pytest.raises(ModelArtifactError, match="stage is required"):
        bd.publish_bundle(env.store, "v1", bare, KEY, **scratch)
    (bare / "subdir").mkdir()
    with pytest.raises(ModelArtifactError, match="not a regular file"):
        bd.publish_bundle(env.store, "v1", bare, KEY, stage="final", **scratch)
    with pytest.raises(ModelArtifactError, match="directory not found"):
        bd.publish_bundle(env.store, "v1", env.tmp / "nope", KEY, stage="final", **scratch)
    with pytest.raises(ModelArtifactError, match="invalid model version"):
        bd.publish_bundle(env.store, "../v1", bare, KEY, stage="final", **scratch)
    with pytest.raises(ModelArtifactError, match="invalid model stage"):
        bd.publish_bundle(env.store, "v1", bare, KEY, stage="FINAL", **scratch)
    assert env.store.list() == []
    assert not [p for p in env.tmp.iterdir() if p.name.startswith(".publish-")]  # scratch space is cleaned up


def test_bundle_and_pointer_keys_are_validated():
    assert bd.bundle_key("final", "v1") == "models/final/v1.zip"
    assert bd.pointer_key("pre") == "models/pre/latest.json"
    for stage, version in (("Final", "v1"), ("final", "../x"), ("final", "nul"), ("final", ""), ("final", "latest")):
        with pytest.raises(ModelArtifactError):
            bd.bundle_key(stage, version)
    with pytest.raises(ModelArtifactError):
        bd.pointer_key("a/b")


def test_the_version_name_latest_is_reserved_for_the_pointer(env):
    root = make_dir(env.tmp / "src")
    with pytest.raises(ModelArtifactError, match="'latest' is reserved"):
        bd.publish_bundle(env.store, "latest", root, KEY, stage="final")
    assert env.store.list() == []


# ---------------------------------------------------------------------------------------- fetch: happy path
def test_fetch_latest_roundtrip_installs_a_verified_directory(env):
    publish(env)
    path = bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert path == env.dest / "v1"
    assert sorted(p.name for p in path.iterdir()) == sorted([*FILES, "manifest.json"])
    for name, data in FILES.items():
        assert (path / name).read_bytes() == data
    manifest = bd.verify_bundle_dir(path, KEY)
    assert manifest["version"] == "v1" and manifest["meta"] == {"trained": "2026-11-01"}
    assert [p.name for p in env.dest.iterdir()] == ["v1"]  # no staging directory is left behind
    assert bd.fetch_bundle(env.store, "v1", env.dest, KEY) == path  # explicit version, same result


def test_fetch_honours_the_stage(env):
    publish(env, "v1", "final")
    publish(env, "p1", "pre", files={"model.txt": b"pre-model", "schema.json": b"{}"})
    assert bd.fetch_bundle(env.store, "latest", env.dest, KEY, stage="pre").name == "p1"
    assert bd.fetch_bundle(env.store, "latest", env.dest, KEY).name == "v1"  # default stage is 'final'
    assert bd.read_pointer(env.store, KEY, stage="pre")["version"] == "p1"
    shutil.rmtree(env.dest)
    with pytest.raises(ModelArtifactError, match="no published model bundle for stage 'oops'"):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY, stage="oops")
    assert_clean(env.dest)


def test_fetch_keeps_an_identical_existing_install_and_repairs_a_corrupt_one(env):
    publish(env)
    path = bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    before = (path / "model.txt").stat().st_mtime_ns
    assert bd.fetch_bundle(env.store, "latest", env.dest, KEY) == path
    assert (path / "model.txt").stat().st_mtime_ns == before  # not rewritten
    (path / "model.txt").write_bytes(b"corrupted on disk")
    assert bd.fetch_bundle(env.store, "latest", env.dest, KEY) == path
    assert (path / "model.txt").read_bytes() == FILES["model.txt"]
    assert [p.name for p in env.dest.iterdir()] == ["v1"]  # the old directory was removed, not left as .stale-*


def test_fetch_over_http_read_store(env):
    publish(env)

    class Response:
        def __init__(self, status, body=b""):
            self.status_code = status
            self.headers = CaseInsensitiveDict({"Content-Length": str(len(body))} if body else {})
            self._body = body

        def iter_content(self, chunk_size=1):
            for start in range(0, len(self._body), 4096):
                yield self._body[start:start + 4096]

        def close(self):
            pass

    class DirSession:
        def request(self, method, url, **kwargs):
            target = env.tmp / "remote" / urlsplit(url).path[len("/art/"):]
            if not target.is_file():
                return Response(404)
            return Response(200, target.read_bytes() if method == "GET" else b"")

    http = HttpReadStore("https://cdn.example.test/art/", session=DirSession())
    path = bd.fetch_bundle(http, "latest", env.dest, KEY)
    assert (path / "model.txt").read_bytes() == FILES["model.txt"]
    with pytest.raises(ModelArtifactError, match="not found"):
        bd.fetch_bundle(http, "v404", env.dest, KEY)


# ------------------------------------------------------------------------------------ fetch: rejection
def test_wrong_key_is_rejected_and_nothing_is_written(env):
    publish(env)
    with pytest.raises(ModelArtifactError, match="latest.json signature mismatch"):
        bd.fetch_bundle(env.store, "latest", env.dest, OTHER_KEY)
    with pytest.raises(ModelArtifactError, match="manifest signature mismatch"):
        bd.fetch_bundle(env.store, "v1", env.dest, OTHER_KEY)
    with pytest.raises(ModelArtifactError, match="not configured"):
        bd.fetch_bundle(env.store, "v1", env.dest, "")
    assert_clean(env.dest)


def test_unsigned_bundles_and_pointers_are_rejected(env):
    publish(env)
    members = read_members(env.store, "models/final/v1.zip")
    manifest = json.loads(members["manifest.json"])
    manifest.pop("signature")
    put_zip(env, "models/final/v1.zip", {**members, "manifest.json": json.dumps(manifest).encode()})
    with pytest.raises(ModelArtifactError, match="manifest is unsigned"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    pointer = json.loads(env.store.get("models/final/latest.json"))
    pointer.pop("signature")
    env.store.put("models/final/latest.json", json.dumps(pointer).encode())
    with pytest.raises(ModelArtifactError, match="latest.json is unsigned"):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert_clean(env.dest)


ARCHIVE_CASES = {
    "flipped model byte": "sha256 mismatch",
    "flipped calibrator byte": "sha256 mismatch",
    "truncated model": "differs from the manifest",
    "grown model": "differs from the manifest",
    "edited manifest meta": "signature mismatch",
    "edited manifest version": "signature mismatch",
    "flipped signature": "signature mismatch",
    "removed signature": "unsigned",
    "extra member": "do not match the manifest",
    "missing member": "do not match the manifest",
    "no manifest": "no manifest.json",
    "garbage manifest": "not valid JSON",
}


def tampered_archive(label: str, original: dict) -> dict:
    manifest = json.loads(original["manifest.json"])

    def edited(**changes):
        return json.dumps({**manifest, **changes}).encode()

    cases = {
        "flipped model byte": {**original, "model.txt": flip(original["model.txt"])},
        "flipped calibrator byte": {**original, "calibrator.json": flip(original["calibrator.json"])},
        "truncated model": {**original, "model.txt": original["model.txt"][:-1]},
        "grown model": {**original, "model.txt": original["model.txt"] + b"x"},
        "edited manifest meta": {**original, "manifest.json": edited(meta={"trained": "never"})},
        "edited manifest version": {**original, "manifest.json": edited(version="v2")},
        "flipped signature": {**original, "manifest.json": edited(signature="0" * 64)},
        "removed signature": {**original, "manifest.json": json.dumps(
            {k: v for k, v in manifest.items() if k != "signature"}).encode()},
        "extra member": {**original, "extra.txt": b"x"},
        "missing member": {k: v for k, v in original.items() if k != "schema.json"},
        "no manifest": {k: v for k, v in original.items() if k != "manifest.json"},
        "garbage manifest": {**original, "manifest.json": b"{not json"},
    }
    return cases[label]


@pytest.mark.parametrize(("label", "message"), list(ARCHIVE_CASES.items()), ids=list(ARCHIVE_CASES))
def test_tampering_with_any_part_of_the_archive_is_detected(env, label, message):
    publish(env)
    key = "models/final/v1.zip"
    put_zip(env, key, tampered_archive(label, read_members(env.store, key)))
    with pytest.raises(ModelArtifactError, match=message):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert_clean(env.dest)


def test_corrupt_or_truncated_archives(env):
    publish(env)
    key = "models/final/v1.zip"
    good = env.store.get(key)
    for blob in (good[:-60], good[: len(good) // 2], b"definitely not a zip file", b""):
        env.store.put(key, blob)
        with pytest.raises(ModelArtifactError, match="not a valid zip archive"):
            bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert_clean(env.dest)


def test_crc_failure_inside_a_stored_member_is_rejected(env):
    publish(env)
    key = "models/final/v1.zip"
    members = read_members(env.store, key)
    put_zip(env, key, members, compression=zipfile.ZIP_STORED)
    blob = bytearray(env.store.get(key))
    position = blob.index(FILES["schema.json"])
    blob[position + 3] ^= 0xFF  # corrupt the stored payload but keep every size field intact
    env.store.put(key, bytes(blob))
    with pytest.raises(ModelArtifactError, match="not a valid zip archive"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert_clean(env.dest)


POINTER_CASES = {
    "version swapped": "signature mismatch",
    "sha edited": "signature mismatch",
    "size edited": "signature mismatch",
    "stage edited": "signature mismatch",
    "format 2": "unsupported latest.json format",
    "other stage, re-signed": "different stage",
    "other kind, re-signed": "not a model pointer",
    "bad version, re-signed": "invalid model version",
    "bad sha, re-signed": "invalid sha256 or size",
    "zero size, re-signed": "invalid sha256 or size",
    "bool size, re-signed": "invalid sha256 or size",
}


def tampered_pointer(label: str, good: dict) -> dict:
    def resign(**changes):
        return sg.sign_document({**good, **changes}, KEY)

    cases = {
        "version swapped": {**good, "version": "v9"},
        "sha edited": {**good, "sha256": "0" * 64},
        "size edited": {**good, "size": 1},
        "stage edited": {**good, "stage": "pre"},
        "format 2": {**good, "format": 2},
        "other stage, re-signed": resign(stage="pre"),
        "other kind, re-signed": resign(kind="monster-model-manifest"),
        "bad version, re-signed": resign(version="../x"),
        "bad sha, re-signed": resign(sha256="xyz"),
        "zero size, re-signed": resign(size=0),
        "bool size, re-signed": resign(size=True),
    }
    return cases[label]


@pytest.mark.parametrize(("label", "message"), list(POINTER_CASES.items()), ids=list(POINTER_CASES))
def test_pointer_tampering_is_detected(env, label, message):
    publish(env)
    key = "models/final/latest.json"
    env.store.put(key, json.dumps(tampered_pointer(label, json.loads(env.store.get(key)))).encode())
    with pytest.raises(ModelArtifactError, match=message):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert_clean(env.dest)


def test_unusable_pointers_are_rejected(env):
    publish(env)
    key = "models/final/latest.json"
    for raw, message in ((b"not json", "not valid JSON"), (b"[1, 2]", "must be a JSON object"),
                         (b" " * 20_000, "too large")):
        env.store.put(key, raw)
        with pytest.raises(ModelArtifactError, match=message):
            bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    env.store.delete(key)
    with pytest.raises(ModelArtifactError, match="no published model bundle"):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert_clean(env.dest)


def test_replayed_older_pointer_is_accepted_but_must_still_verify(env):
    publish(env, "v1")
    old_pointer = env.store.get("models/final/latest.json")
    old_zip = env.store.get("models/final/v1.zip")
    publish(env, "v2", files={"model.txt": b"newer model", "schema.json": b"{}"})
    assert bd.read_pointer(env.store, KEY)["version"] == "v2"
    env.store.put("models/final/latest.json", old_pointer)  # rollback / replay of a genuine old pointer
    path = bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert path.name == "v1" and bd.verify_bundle_dir(path, KEY)["version"] == "v1"
    shutil.rmtree(env.dest)
    # ...but the replayed pointer cannot vouch for a different archive
    env.store.put("models/final/v1.zip", env.store.get("models/final/v2.zip"))
    with pytest.raises(ModelArtifactError, match="does not match the signed latest.json pointer"):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    env.store.put("models/final/v1.zip", old_zip[:-30])  # truncated copy of the right archive
    with pytest.raises(ModelArtifactError, match="does not match the signed latest.json pointer"):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert_clean(env.dest)


def test_a_bundle_cannot_be_substituted_for_another_version_or_stage(env):
    publish(env, "v1", "final")
    env.store.put("models/final/v2.zip", env.store.get("models/final/v1.zip"))
    env.store.put("models/pre/v1.zip", env.store.get("models/final/v1.zip"))
    with pytest.raises(ModelArtifactError, match="different version or stage"):
        bd.fetch_bundle(env.store, "v2", env.dest, KEY)
    with pytest.raises(ModelArtifactError, match="different version or stage"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY, stage="pre")
    assert_clean(env.dest)


def test_oversized_archives_are_refused(env):
    publish(env)
    size = len(env.store.get("models/final/v1.zip"))
    with pytest.raises(ModelArtifactError, match="larger than the 100 byte limit"):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY, max_bytes=100)  # known from the signed pointer
    with pytest.raises(ModelArtifactError, match="exceeds the 100 byte limit"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY, max_bytes=100)  # enforced while downloading
    assert_clean(env.dest)
    assert bd.fetch_bundle(env.store, "v1", env.dest, KEY, max_bytes=size).name == "v1"


def test_zip_bombs_are_refused_by_declared_and_actual_size(env, monkeypatch):
    publish(env, files={"model.txt": b"\x00" * 5_000_000, "schema.json": b"{}"})
    assert len(env.store.get("models/final/v1.zip")) < 20_000  # tiny on the wire, 5 MB unpacked
    with pytest.raises(ModelArtifactError, match="expands beyond the allowed size"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY, max_unpacked_bytes=1_000_000)
    with pytest.raises(ModelArtifactError, match="expands beyond the allowed size"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY, max_bytes=1_000_000)  # default allowance is 4 x max_bytes
    assert_clean(env.dest)
    assert bd.fetch_bundle(env.store, "v1", env.dest, KEY, max_unpacked_bytes=6_000_000).name == "v1"

    shutil.rmtree(env.dest)
    real_open = zipfile.ZipFile.open

    def lying_open(self, name, mode="r", *args, **kwargs):
        stream = real_open(self, name, mode, *args, **kwargs)
        if mode == "r" and getattr(name, "filename", name) == "schema.json":
            data = stream.read()
            stream.close()
            return io.BytesIO(data + b"EXTRA-BYTES-BEYOND-THE-DECLARED-SIZE")
        return stream

    monkeypatch.setattr(zipfile.ZipFile, "open", lying_open)
    with pytest.raises(ModelArtifactError, match="larger than declared"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert_clean(env.dest)

    def short_open(self, name, mode="r", *args, **kwargs):
        stream = real_open(self, name, mode, *args, **kwargs)
        if mode == "r" and getattr(name, "filename", name) == "schema.json":
            data = stream.read()
            stream.close()
            return io.BytesIO(data[:-1])
        return stream

    monkeypatch.setattr(zipfile.ZipFile, "open", short_open)
    with pytest.raises(ModelArtifactError, match="shorter than declared"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert_clean(env.dest)


@pytest.mark.parametrize("name", [
    "../evil.txt", "/abs/evil.txt", "..\\evil.txt", "sub/file.txt", "C:\\evil.txt", "dir/", "C:evil.txt", "..", ".",
    "a" * 81, "has space.txt",
])
def test_zip_slip_and_nested_paths_are_rejected(env, name):
    publish(env)
    members = read_members(env.store, "models/final/v1.zip")
    put_zip(env, "models/final/evil.zip", {**members, name: b"pwned"})
    with pytest.raises(ModelArtifactError, match="path traversal|invalid file name"):
        bd.fetch_bundle(env.store, "evil", env.dest, KEY)
    assert not (env.tmp / "evil.txt").exists() and not (env.dest.parent / "evil.txt").exists()
    assert not (env.tmp / "abs").exists() and not Path("/abs/evil.txt").exists()
    assert_clean(env.dest)


def test_symlink_special_file_and_duplicate_members_are_rejected(env):
    publish(env)
    members = read_members(env.store, "models/final/v1.zip")
    link_attrs = {"model.txt": (stat.S_IFLNK | 0o777) << 16}
    put_zip(env, "models/final/evil.zip", members, attrs=link_attrs)
    with pytest.raises(ModelArtifactError, match="symlink or special file"):
        bd.fetch_bundle(env.store, "evil", env.dest, KEY)
    put_zip(env, "models/final/evil.zip", members, attrs={"model.txt": (stat.S_IFIFO | 0o644) << 16})
    with pytest.raises(ModelArtifactError, match="symlink or special file"):
        bd.fetch_bundle(env.store, "evil", env.dest, KEY)
    put_zip(env, "models/final/evil.zip", [*members.items(), ("model.txt", b"second copy")])
    with pytest.raises(ModelArtifactError, match="duplicate member"):
        bd.fetch_bundle(env.store, "evil", env.dest, KEY)
    assert_clean(env.dest)


def test_too_many_members_and_unsupported_compression(env):
    publish(env)
    members = read_members(env.store, "models/final/v1.zip")
    put_zip(env, "models/final/evil.zip", {**members, **{f"extra{i}.txt": b"x" for i in range(sg.MAX_MANIFEST_FILES + 5)}})
    with pytest.raises(ModelArtifactError, match="too many members"):
        bd.fetch_bundle(env.store, "evil", env.dest, KEY)
    put_zip(env, "models/final/evil.zip", members, compression=zipfile.ZIP_BZIP2)
    with pytest.raises(ModelArtifactError, match="unsupported compression"):
        bd.fetch_bundle(env.store, "evil", env.dest, KEY)
    assert_clean(env.dest)


def test_encrypted_members_are_rejected():
    encrypted = zipfile.ZipInfo("model.txt")
    encrypted.flag_bits = 0x1

    class Archive:
        def infolist(self):
            return [zipfile.ZipInfo("manifest.json"), encrypted]

    with pytest.raises(ModelArtifactError, match="encrypted member"):
        bd._check_members(Archive())


# --------------------------------------------------------------------------------------------- atomicity
def test_failure_in_the_middle_of_extraction_leaves_nothing_behind(env, monkeypatch):
    publish(env)
    real, count = bd._copy_member, {"n": 0}

    def flaky(*args, **kwargs):
        count["n"] += 1
        if count["n"] == 2:
            raise OSError(errno.EIO, "Input/output error")
        return real(*args, **kwargs)

    monkeypatch.setattr(bd, "_copy_member", flaky)
    with pytest.raises(ModelArtifactError, match="cannot fetch the model bundle: Input/output error"):
        bd.fetch_bundle(env.store, "latest", env.dest, KEY)
    assert count["n"] == 2  # one member had already been written when it failed
    assert_clean(env.dest)
    monkeypatch.undo()
    assert bd.fetch_bundle(env.store, "latest", env.dest, KEY).name == "v1"  # and the next attempt succeeds


def test_disk_full_is_reported_as_resource_exhaustion(env, monkeypatch):
    publish(env)

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(bd, "_copy_member", full)
    with pytest.raises(ResourceExhausted, match="disk full"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert_clean(env.dest)


def test_unwritable_destination_is_a_model_artifact_error(env):
    publish(env)
    env.dest.parent.mkdir(exist_ok=True)
    env.dest.write_text("a file where the directory should be")
    with pytest.raises(ModelArtifactError, match="cannot prepare the model directory"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)


def test_failed_swap_restores_the_previous_directory(env, monkeypatch):
    publish(env)
    target = env.dest / "v1"
    target.mkdir(parents=True)
    (target / "junk.txt").write_text("previous contents")
    real = os.replace

    def failing(src, dst):
        if Path(src).name == "content":
            raise OSError(errno.EACCES, "Permission denied")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(ModelArtifactError, match="cannot fetch the model bundle"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    monkeypatch.undo()
    assert (target / "junk.txt").read_text() == "previous contents"  # rolled back
    assert [p.name for p in env.dest.iterdir()] == ["v1"]  # no staging / .stale leftovers


def test_concurrent_identical_install_is_accepted(env, monkeypatch):
    publish(env)
    installed = bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    shutil.copytree(installed, env.tmp / "good")
    shutil.rmtree(installed)
    real = os.replace

    def raced(src, dst):
        if Path(src).name == "content":
            shutil.copytree(env.tmp / "good", dst)  # another process installed the same bundle first
            raise OSError(errno.ENOTEMPTY, "Directory not empty")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", raced)
    assert bd.fetch_bundle(env.store, "v1", env.dest, KEY) == installed
    monkeypatch.undo()
    assert bd.verify_bundle_dir(installed, KEY)["version"] == "v1"
    assert [p.name for p in env.dest.iterdir()] == ["v1"]


def test_stale_staging_directories_of_crashed_fetches_are_swept(env):
    publish(env)
    env.dest.mkdir()
    old, fresh, other = env.dest / ".staging-old", env.dest / ".staging-fresh", env.dest / "keep-me"
    stale = env.dest / ".stale-old"
    for directory in (old, fresh, other, stale):
        directory.mkdir()
        (directory / "f").write_bytes(b"x")
    ancient = time.time() - 3 * 24 * 3600
    for directory in (old, stale, other):
        os.utime(directory, (ancient, ancient))
    bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert not old.exists() and not stale.exists() and fresh.exists() and other.exists()


def test_store_errors_propagate_as_data_unavailable_not_artifact_errors(env):
    publish(env)

    class Down(LocalDirStore):
        def get_to_file(self, key, dest, **kwargs):
            raise DataUnavailable("object_store:test", "network down", retryable=True)

    down = Down(env.tmp / "remote")
    with pytest.raises(DataUnavailable, match="network down"):
        bd.fetch_bundle(down, "v1", env.dest, KEY)
    assert_clean(env.dest)


def test_artifacts_are_text_and_json_only_never_pickle():
    root = Path(bd.__file__).resolve().parent.parent
    for relative in ("models/signing.py", "models/bundle.py", "database/object_store.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+(pickle|cPickle|marshal|shelve|dill|joblib)\b", source, re.M)


def test_manifest_members_beyond_the_size_limit_are_refused(env, monkeypatch):
    publish(env)
    key = "models/final/v1.zip"
    members = read_members(env.store, key)
    put_zip(env, key, {**members, "manifest.json": b" " * (sg.MAX_MANIFEST_BYTES + 10)})
    with pytest.raises(ModelArtifactError, match="manifest too large"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)  # the declared size is already over the limit
    publish(env)
    real_open = zipfile.ZipFile.open

    def long_open(self, name, mode="r", *args, **kwargs):
        stream = real_open(self, name, mode, *args, **kwargs)
        if mode == "r" and getattr(name, "filename", name) == "manifest.json":
            stream.close()
            return io.BytesIO(b" " * (sg.MAX_MANIFEST_BYTES + 10))  # the stream outgrows its declared size
        return stream

    monkeypatch.setattr(zipfile.ZipFile, "open", long_open)
    with pytest.raises(ModelArtifactError, match="manifest too large"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    assert_clean(env.dest)


def test_install_failure_without_a_competing_install_is_an_error(env, monkeypatch):
    publish(env)
    real = os.replace

    def failing(src, dst):
        if Path(src).name == "content":
            raise OSError(errno.EACCES, "Permission denied")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", failing)
    with pytest.raises(ModelArtifactError, match="cannot fetch the model bundle: Permission denied"):
        bd.fetch_bundle(env.store, "v1", env.dest, KEY)
    monkeypatch.undo()
    assert_clean(env.dest)


def test_manifest_structure_checks_work_on_their_own(tmp_path):
    with pytest.raises(ModelArtifactError, match="must be a JSON object"):
        sg.check_manifest_structure([])
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    (root / "manifest.json").write_text(json.dumps({**manifest, "format": 2}))
    with pytest.raises(ModelArtifactError, match="unsupported manifest format version 2"):
        bd.pack_bundle(root, root / "manifest.json", tmp_path / "x.zip")


def test_symlinks_are_rejected_even_where_the_platform_cannot_create_them(tmp_path, monkeypatch):
    root = make_dir(tmp_path / "m")
    manifest = signed_manifest(root)
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "schema.json")
    with pytest.raises(ModelArtifactError, match="'schema.json' is a symlink"):
        sg.verify_manifest(manifest, KEY, root)
    with pytest.raises(ModelArtifactError, match="not a regular file"):
        sg.build_manifest({"schema.json": root / "schema.json"}, version="v1", stage="final", key=KEY)
