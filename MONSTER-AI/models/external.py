"""External training mode (optional): a worker with plenty of RAM trains and publishes SIGNED model bundles; the small
production server only pulls, verifies and registers them. The server never trusts an unsigned or tampered artifact:
``fetch_bundle`` checks the HMAC-signed pointer and manifest before anything is extracted, and the registry re-verifies
every file at load time. Registered bundles go through the same champion/challenger gates as locally trained models."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from app.clock import Clock, get_clock, ist_date
from app.config import Settings
from app.errors import ModelArtifactError, MonsterError
from app.logging_setup import redact
from database.object_store import ObjectStore, build_artifact_source, build_object_store
from database.store import Store
from features.schema import FEATURE_VERSION
from models.bundle import fetch_bundle, publish_bundle, read_pointer
from models.registry import ModelRegistry
from models.training import TrainResult

log = logging.getLogger(__name__)

STAGES = ("pre", "final")
PUBLISHABLE = frozenset({"shadow", "validated"})  # a model that failed its gates is never published or registered


def _signing_key(settings: Settings) -> str:
    if not settings.model_signing_key:
        raise ModelArtifactError("MODEL_SIGNING_KEY is required to verify or sign model bundles")
    return settings.model_signing_key


def _result_from_bundle(directory: Path, stage: str, version: str) -> TrainResult:
    """Describe an already-trained, signature-verified bundle in the shape the registry expects."""
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"{version}: meta.json is unreadable ({type(exc).__name__})") from None
    if not isinstance(meta, dict) or meta.get("version") != version or meta.get("stage") != stage:
        raise ModelArtifactError(f"{version}: meta.json does not describe this version and stage")
    if meta.get("status") not in PUBLISHABLE:
        raise ModelArtifactError(f"{version}: status {meta.get('status')!r} is not publishable")
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ModelArtifactError(
            f"{version}: built with feature version {meta.get('feature_version')} but this server runs {FEATURE_VERSION}; "
            "update the server code")
    return TrainResult(
        version=version, stage=stage, status=meta["status"], directory=directory, metrics=meta.get("metrics") or {},
        gates=meta.get("gates") or {}, meta=meta, seconds=0.0, notes=["trained by the external worker"],
    )


def sync_models(settings: Settings, store: Store, registry: ModelRegistry, *, clock: Clock | None = None,
                source: ObjectStore | None = None) -> dict[str, Any]:
    """Pull the newest signed bundle of every stage and register unseen versions (the registry's champion/challenger
    policy decides activation). One bad stage never blocks the other: failures are reported and alerted."""
    src = source or build_artifact_source(settings)
    if src is None:
        return {"message": "no artifact source configured (MODEL_ARTIFACT_URL or S3_*)", "synced": [], "current": [],
                "failed": []}
    key = _signing_key(settings)
    synced: list[str] = []
    current: list[str] = []
    failed: list[str] = []
    for stage in STAGES:
        try:
            known = read_pointer(src, key, stage=stage)["version"]
            if store.get_model(known) is not None:
                current.append(f"{stage}:{known}")
                continue
            # "latest" makes fetch_bundle check the archive against the signed pointer as well as the signed manifest
            directory = fetch_bundle(src, "latest", settings.models_dir, key, stage=stage)
            version = directory.name
            if store.get_model(version) is not None:  # the pointer moved to a version we already know between the two reads
                current.append(f"{stage}:{version}")
                continue
            decision = registry.register(_result_from_bundle(directory, stage, version))
            outcome = "activated" if decision["activate"] else "kept as challenger"
            synced.append(f"{stage}:{version} ({decision['status']}, {outcome})")
        except (MonsterError, OSError, ValueError) as exc:
            reason = redact(str(exc))[:160]
            failed.append(f"{stage}: {reason}")
            log.warning("model sync for stage %s failed: %s", stage, reason)
            store.add_alert("warning", "model_sync_failed", f"stage {stage}: {reason}", dedupe_key=f"model-sync:{stage}",
                            dedupe_minutes=24 * 60)
    parts = [*(f"synced {s}" for s in synced), *(f"up to date {c}" for c in current), *(f"FAILED {f}" for f in failed)]
    return {"message": "; ".join(parts) or "nothing to do", "synced": synced, "current": current, "failed": failed}


def publish_models(settings: Settings, versions: dict[str, str], *, sink: ObjectStore | None = None,
                   clock: Clock | None = None) -> dict[str, str]:
    """Sign and upload trained versions that are already on disk (stage -> version); returns stage -> object key."""
    target = sink or build_object_store(settings)
    if target is None:
        raise ModelArtifactError("no S3 object store configured (S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY)")
    key = _signing_key(settings)
    published: dict[str, str] = {}
    for stage, version in versions.items():
        directory = settings.models_dir / version
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelArtifactError(f"{version}: meta.json is unreadable ({type(exc).__name__})") from None
        info = {"status": meta.get("status"), "train_end": meta.get("train_end"), "feature_version": meta.get("feature_version")}
        published[stage] = publish_bundle(target, version, directory, key, stage=stage, meta=info, clock=clock,
                                          tmp_dir=settings.tmp_dir)
    return published


MIN_SYMBOLS_WITH_HISTORY = 30


def _has_market_data(settings: Settings) -> bool:
    store = Store.open(settings.db_path)
    try:
        return len(store.latest_bar_dates(universe_only=True)) >= MIN_SYMBOLS_WITH_HISTORY
    finally:
        store.close()


def _obtain_market_data(settings: Settings, source: ObjectStore, run_key: str) -> None:
    """A fresh CI workspace has no history: restore the server's newest off-box backup, else bootstrap the data."""
    from automation.runner import execute_job
    from database.remote_backup import restore_from_remote

    print("no market data in this workspace: restoring the newest database backup from object storage", flush=True)
    if restore_from_remote(settings, source) is not None:
        return
    print("no usable remote backup: collecting the market data from scratch (this takes a while)", flush=True)
    outcome = execute_job(settings, "bootstrap", run_key, options={"data_only": "1"})
    print(f"bootstrap: {outcome.status} - {outcome.message}", flush=True)


def train_and_publish(settings: Settings, *, stage: str = "both", sink: ObjectStore | None = None) -> int:
    """CI entry point (``train --publish``): make sure market data exists, train, then sign and publish exactly the models
    trained now. Models that fail their gates are reported and not published (that is the system working, not an
    error). Exit code 1 on any training or publishing failure."""
    from automation.runner import execute_job

    stages = STAGES if stage == "both" else (stage,)
    target = sink or build_object_store(settings)
    if target is None or not settings.model_signing_key:
        print("publishing needs S3_* object storage and MODEL_SIGNING_KEY", flush=True)
        return 1
    worker = dataclasses.replace(settings, training_mode="local-light")  # this process trains whatever the server's mode is
    run_key = ist_date(get_clock().now()).isoformat()
    if not _has_market_data(worker):
        _obtain_market_data(worker, target, run_key)
    result = execute_job(worker, "train", run_key, force=True, timeout_min=worker.job_timeout_minutes,
                         options={"stages": ",".join(stages)})
    print(f"train: {result.status} - {result.message}", flush=True)
    outcomes = result.metrics or {}
    versions = {n: outcomes[n]["version"] for n in stages if (outcomes.get(n) or {}).get("status") in PUBLISHABLE}
    errors = [n for n in stages if "error" in (outcomes.get(n) or {"error": "not trained"})]
    for name in stages:
        if name not in versions:
            info = outcomes.get(name) or {}
            print(f"{name}: not published ({info.get('error') or info.get('reason') or info.get('status') or 'not trained'})",
                  flush=True)
    if result.status != "success":
        return 1
    try:
        published = publish_models(worker, versions, sink=target)
    except (MonsterError, OSError) as exc:
        print(f"publish failed: {redact(str(exc))}", flush=True)
        return 1
    for name, location in published.items():
        print(f"published {name}: {location}", flush=True)
    return 1 if errors else 0
