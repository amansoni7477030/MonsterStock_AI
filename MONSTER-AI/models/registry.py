"""Model registry and governance (blueprint section 19): versioned artifacts, champion/challenger activation, integrity
checks at load, automatic fallback and rollback. Daily outcomes never rewrite a model; only a controlled, validated
retraining can promote a new version, and the previous known-good model is always preserved for rollback."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from app.config import Settings
from app.errors import ModelArtifactError
from app.logging_setup import redact
from database.store import Store
from features.schema import FEATURE_VERSION
from models.artifacts import sha256_file, verify_directory
from models.inference import BaselineBundle, Bundle, ModelBundle
from models.training import TrainResult

log = logging.getLogger(__name__)

SKILL_TOLERANCE = 0.005
EDGE_TOLERANCE = 0.01
STALE_CHAMPION_DAYS = 120
KEEP_VERSIONS_PER_STAGE = 5


class ModelRegistry:
    def __init__(self, store: Store, settings: Settings) -> None:
        self.store, self.settings = store, settings

    # ------------------------------------------------------------------------------------- queries
    def baseline(self, stage: str) -> BaselineBundle:
        return BaselineBundle(stage, self.settings.range_thresholds, self.settings.primary_threshold)

    def active_record(self, stage: str) -> dict[str, Any] | None:
        return self.store.active_model(stage)

    def artifact_dir(self, version: str) -> Path:
        return self.settings.models_dir / version

    # -------------------------------------------------------------------------------- registration
    def register(self, result: TrainResult, *, code_commit: str | None = None) -> dict[str, Any]:
        """Record a trained candidate with its gate outcome; activate it only if the champion policy allows."""
        manifest_sha = sha256_file(result.directory / "manifest.json")
        self.store.register_model(
            model_version=result.version, stage=result.stage, status="candidate", train_start=result.meta.get("train_start"),
            train_end=result.meta.get("train_end"), feature_schema_hash=result.meta.get("feature_schema_hash"),
            feature_version=result.meta.get("feature_version"), code_commit=code_commit,
            metrics={"heads": result.metrics.get("heads"), "ranking": result.metrics.get("ranking"),
                     "quantiles": result.metrics.get("quantiles"), "gates": result.gates},
            artifact_path=str(result.directory), artifact_sha256=manifest_sha, notes=result.gates.get("reason"),
            parent_version=(self.active_record(result.stage) or {}).get("model_version"),
        )
        decision = self._decide(result)
        self.store.set_model_status(result.version, decision["status"], notes=decision["reason"])
        if decision["activate"]:
            self.store.activate_model(result.version)
        log.info("model %s -> %s (%s)", result.version, decision["status"], decision["reason"])
        self.prune(result.stage)
        return decision

    def _decide(self, result: TrainResult) -> dict[str, Any]:
        status = result.status
        if status == "rejected":
            return {"status": "rejected", "activate": False, "reason": f"gates failed: {result.gates.get('reason')}"}
        champion = self.active_record(result.stage)
        if champion is None:
            return {"status": status, "activate": True, "reason": "first approved model for this stage"}
        champ_ok = self._champion_healthy(champion)
        if not champ_ok:
            return {"status": status, "activate": True, "reason": "champion unhealthy (artifact invalid or too old)"}
        rank = {"shadow": 1, "validated": 2}
        if rank.get(status, 0) > rank.get(champion["status"], 0):
            return {"status": status, "activate": True, "reason": f"upgrades champion from {champion['status']} to {status}"}
        if rank.get(status, 0) < rank.get(champion["status"], 0):
            return {"status": status, "activate": False, "reason": "challenger is weaker than the validated champion; kept in registry"}
        cm = (champion.get("metrics") or {}).get("gates") or {}
        skill_ok = _num(result.gates.get("brier_skill")) >= _num(cm.get("brier_skill")) - SKILL_TOLERANCE
        edge_new = _num((result.gates.get("vs_best_baseline") or {}).get("mean_diff"))
        edge_old = _num((cm.get("vs_best_baseline") or {}).get("mean_diff"))
        if skill_ok and edge_new >= edge_old - EDGE_TOLERANCE:
            return {"status": status, "activate": True, "reason": "challenger matches or beats the champion's gate metrics"}
        return {"status": status, "activate": False, "reason": "challenger worse than champion on fixed gate metrics; champion kept"}

    def verify_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Integrity check of one registered version: every file against the manifest, and the manifest itself against the
        checksum recorded at registration (so a rewritten manifest cannot vouch for tampered model files)."""
        directory = Path(record["artifact_path"])
        manifest = verify_directory(directory)
        expected = record.get("artifact_sha256")
        if expected and sha256_file(directory / "manifest.json") != expected:
            raise ModelArtifactError(f"{directory.name}: manifest does not match the checksum recorded at registration")
        return manifest

    def _champion_healthy(self, champion: dict[str, Any]) -> bool:
        try:
            self.verify_record(champion)
        except (ModelArtifactError, OSError):
            return False
        from app.clock import get_clock, parse_ts

        created = parse_ts(champion.get("created_at"))
        if created is None:
            return True
        return (get_clock().now() - created).days <= STALE_CHAMPION_DAYS

    # ------------------------------------------------------------------------------------- loading
    def active_bundle(self, stage: str) -> Bundle:
        """The active model, verified. Falls back to the previous good version, then to the heuristic baseline."""
        record = self.active_record(stage)
        tried: set[str] = set()
        while record is not None and record["model_version"] not in tried:
            tried.add(record["model_version"])
            try:
                self.verify_record(record)
                bundle = ModelBundle.load(Path(record["artifact_path"]), verify=False)
                if bundle.meta.get("feature_version") != FEATURE_VERSION:
                    raise ModelArtifactError(
                        f"feature version {bundle.meta.get('feature_version')} != {FEATURE_VERSION}")
                bundle.meta["status"] = record["status"]
                return bundle
            except (ModelArtifactError, OSError) as exc:
                msg = redact(str(exc))
                log.error("active model %s unusable: %s", record["model_version"], msg)
                self.store.add_alert("error", "model_artifact_invalid", f"{record['model_version']}: {msg}",
                                     dedupe_key=f"model-invalid:{record['model_version']}", dedupe_minutes=720)
                self.store.set_model_status(record["model_version"], "retired", notes=f"artifact invalid: {msg}")
                previous = self.rollback(stage, quiet=True)
                record = self.active_record(stage) if previous else None
        return self.baseline(stage)

    # ------------------------------------------------------------------------------- governance
    def approve(self, version: str, status: str) -> None:
        """Manual operator decision: mark an existing, verifiable version shadow/validated and activate it."""
        if status not in {"shadow", "validated"}:
            raise ValueError("status must be shadow or validated")
        record = self.store.get_model(version)
        if record is None:
            raise KeyError(version)
        self.verify_record(record)
        self.store.set_model_status(version, status, notes="manual approval")
        self.store.activate_model(version)

    def activate(self, version: str) -> None:
        record = self.store.get_model(version)
        if record is None:
            raise KeyError(version)
        if record["status"] not in {"shadow", "validated"}:
            raise ValueError(f"{version} is {record['status']}; approve it first")
        self.verify_record(record)
        self.store.activate_model(version)

    def rollback(self, stage: str, *, quiet: bool = False) -> str | None:
        """Re-activate the most recent previously active, healthy version of ``stage`` (the current one is retired)."""
        current = self.active_record(stage)
        candidates = [m for m in self.store.list_models(stage) if m["status"] in {"shadow", "validated"}
                      and (current is None or m["model_version"] != current["model_version"]) and m.get("activated_at")]
        candidates.sort(key=lambda m: m["activated_at"] or "", reverse=True)
        for model in candidates:
            try:
                self.verify_record(model)
            except (ModelArtifactError, OSError):
                continue
            self.store.activate_model(model["model_version"])
            if current is not None:
                self.store.set_model_status(current["model_version"], "retired", notes="rolled back")
            if not quiet:
                log.warning("rolled back %s: %s -> %s", stage, current and current["model_version"], model["model_version"])
            return model["model_version"]
        if current is not None and not quiet:
            log.error("no previous healthy model to roll back to for stage %s", stage)
        return None

    def prune(self, stage: str, keep: int = KEEP_VERSIONS_PER_STAGE) -> int:
        """Delete artifact directories of old, non-active versions (keeps the newest ``keep`` plus the active one)."""
        models = self.store.list_models(stage)
        keep_set = {m["model_version"] for m in models[:keep]}
        active = self.active_record(stage)
        if active:
            keep_set.add(active["model_version"])
        removed = 0
        for model in models:
            if model["model_version"] in keep_set:
                continue
            path = Path(model["artifact_path"] or "")
            if path.exists() and path.parent == self.settings.models_dir:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for m in self.store.list_models():
            gates = (m.get("metrics") or {}).get("gates") or {}
            rows.append({"version": m["model_version"], "stage": m["stage"], "status": m["status"], "active": bool(m["is_active"]),
                         "created_at": m["created_at"], "train_end": m["train_end"], "ece": gates.get("ece"),
                         "brier_skill": gates.get("brier_skill"), "reason": m.get("notes")})
        return rows


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return out if out == out else float("-inf")
