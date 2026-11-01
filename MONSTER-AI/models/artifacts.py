"""Model artifact bundle: LightGBM text models + JSON (never pickle) with a sha256 manifest.

A bundle is a directory:
  meta.json            versions, thresholds, feature columns per specialist, base rates, metrics, gating, scales
  sp_<spec>_<tgt>.txt  specialist boosters          meta_<head>.txt  meta-model boosters
  calibration.json     one calibrator per probability head (+ reliability tables)
  feature_stats.json   per-feature p1/p50/p99 for drift monitoring
  manifest.json        file -> {size, sha256}; verified before every load (and signed when published externally)
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import lightgbm as lgb

from app import jsonutil
from app.errors import ModelArtifactError
from database.object_store import remove_tree, replace_path
from models.calibration import Calibrator

MANIFEST = "manifest.json"
MAX_FILE_BYTES = 60 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._:-]{1,120}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_name(key: str) -> str:
    return key.replace(":", "_") + ".txt"


def write_bundle(
    directory: Path, *, meta: dict[str, Any], boosters: dict[str, lgb.Booster], calibrators: dict[str, Calibrator],
    reliability: dict[str, Any], feature_stats: dict[str, Any],
) -> dict[str, Any]:
    """Write all artifacts atomically (temp dir + rename) and return the manifest."""
    directory = Path(directory)
    staging = directory.with_name(directory.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}

    def record(name: str) -> None:
        path = staging / name
        files[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}

    for key, booster in boosters.items():
        if not _SAFE_NAME.match(key):
            raise ModelArtifactError(f"unsafe booster key {key!r}")
        booster.save_model(str(staging / _file_name(key)))
        record(_file_name(key))
    (staging / "calibration.json").write_text(
        jsonutil.dumps({"calibrators": {k: c.to_dict() for k, c in calibrators.items()}, "reliability": reliability}),
        encoding="utf-8")
    record("calibration.json")
    (staging / "feature_stats.json").write_text(jsonutil.dumps(feature_stats), encoding="utf-8")
    record("feature_stats.json")
    (staging / "meta.json").write_text(jsonutil.dumps(meta, pretty=True), encoding="utf-8")
    record("meta.json")
    manifest = {"format": 1, "version": meta.get("version"), "files": files}
    (staging / MANIFEST).write_text(jsonutil.dumps(manifest, pretty=True), encoding="utf-8")
    if directory.exists():
        remove_tree(directory)
    replace_path(staging, directory)  # rides out the short "access denied" spells a Windows virus scanner can cause
    return manifest


def verify_directory(directory: Path) -> dict[str, Any]:
    """Check every file against the manifest; raises ``ModelArtifactError`` on any discrepancy."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise ModelArtifactError(f"{directory.name}: manifest missing")
    if manifest_path.stat().st_size > 1_000_000:
        raise ModelArtifactError("manifest too large")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ModelArtifactError("manifest is not valid JSON") from exc
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelArtifactError("manifest lists no files")
    on_disk = {p.name for p in directory.iterdir() if p.is_file()} - {MANIFEST}
    if on_disk != set(files):
        raise ModelArtifactError(f"file set differs from manifest: {sorted(on_disk ^ set(files))[:5]}")
    for name, info in files.items():
        if not _SAFE_NAME.match(name.replace(".", "_")):
            raise ModelArtifactError(f"unsafe file name {name!r}")
        path = directory / name
        if path.stat().st_size > MAX_FILE_BYTES or path.stat().st_size != info.get("size"):
            raise ModelArtifactError(f"{name}: size mismatch")
        if sha256_file(path) != info.get("sha256"):
            raise ModelArtifactError(f"{name}: checksum mismatch")
    return manifest


def read_bundle(directory: Path, *, verify: bool = True) -> tuple[dict[str, Any], dict[str, lgb.Booster], dict[str, Calibrator], dict[str, Any], dict[str, Any]]:
    directory = Path(directory)
    if verify:
        verify_directory(directory)
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        calib = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        stats = json.loads((directory / "feature_stats.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"{directory.name}: unreadable artifact ({type(exc).__name__})") from exc
    boosters: dict[str, lgb.Booster] = {}
    for key in meta.get("booster_keys", []):
        path = directory / _file_name(key)
        try:
            boosters[key] = lgb.Booster(model_file=str(path))
        except lgb.basic.LightGBMError as exc:
            raise ModelArtifactError(f"{path.name}: not a valid LightGBM model") from exc
    calibrators = {k: Calibrator.from_dict(v) for k, v in calib.get("calibrators", {}).items()}
    return meta, boosters, calibrators, stats, calib.get("reliability", {})
