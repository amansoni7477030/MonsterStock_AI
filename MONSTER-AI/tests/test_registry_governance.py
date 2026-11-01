"""Model governance without training anything: first model, champion/challenger rules, integrity-driven replacement, manual approval,
rollback and retention. Each version is a tiny but real, signed-by-manifest artifact."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pytest

from app.errors import ModelArtifactError
from features.schema import FEATURE_VERSION
from models.artifacts import write_bundle
from models.registry import KEEP_VERSIONS_PER_STAGE, STALE_CHAMPION_DAYS, ModelRegistry
from models.training import TrainResult

_rng = np.random.default_rng(3)
_X = _rng.normal(size=(120, 3))
_BOOSTER = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 5},
                     lgb.Dataset(_X, label=(_X[:, 0] > 0).astype(float), free_raw_data=False), num_boost_round=3)


@pytest.fixture()
def reg(store, settings, fake_clock):
    fake_clock.advance(1)
    return ModelRegistry(store, settings)


def make_result(reg, version, *, stage="final", status="shadow", skill=0.10, edge=0.05, reason="calibrated and skilful") -> TrainResult:
    meta = {"version": version, "stage": stage, "status": status, "feature_version": FEATURE_VERSION, "feature_schema_hash": "h",
            "train_start": "2021-01-01", "train_end": "2026-10-30", "booster_keys": ["sp:x:unusual"]}
    directory = reg.settings.models_dir / version
    write_bundle(directory, meta=meta, boosters={"sp:x:unusual": _BOOSTER}, calibrators={}, reliability={}, feature_stats={})
    gates = {"brier_skill": skill, "ece": 0.02, "vs_best_baseline": {"mean_diff": edge}, "reason": reason}
    return TrainResult(version, stage, status, directory, {"heads": {}, "ranking": {}, "quantiles": {}}, gates, meta, 1.0, [])


def register(reg, version, tick=1, **kwargs):
    reg.store._clock.advance(tick)  # model ordering follows the registration time
    return reg.register(make_result(reg, version, **kwargs))


# --------------------------------------------------------------------------------------- first model
def test_the_first_approved_model_of_a_stage_becomes_the_champion(reg):
    decision = register(reg, "m-1-final", status="shadow")
    assert decision == {"status": "shadow", "activate": True, "reason": "first approved model for this stage"}
    assert reg.active_record("final")["model_version"] == "m-1-final" and reg.active_record("pre") is None
    assert reg.artifact_dir("m-1-final") == reg.settings.models_dir / "m-1-final"
    assert reg.store.get_model("m-1-final")["artifact_sha256"]


def test_a_rejected_model_is_recorded_but_never_activated(reg):
    decision = register(reg, "m-bad-final", status="rejected", reason="not calibrated")
    assert decision["activate"] is False and decision["status"] == "rejected" and "gates failed: not calibrated" in decision["reason"]
    assert reg.active_record("final") is None and reg.store.get_model("m-bad-final")["status"] == "rejected"
    register(reg, "m-ok-final", status="shadow")
    register(reg, "m-bad2-final", status="rejected")
    assert reg.active_record("final")["model_version"] == "m-ok-final"  # a rejected challenger never displaces the champion


# ------------------------------------------------------------------------------- champion vs challenger
def test_a_validated_champion_is_not_replaced_by_a_shadow_challenger(reg):
    register(reg, "m-champ-final", status="validated")
    decision = register(reg, "m-weak-final", status="shadow", skill=0.5, edge=0.5)
    assert decision["activate"] is False and "weaker than the validated champion" in decision["reason"]
    assert reg.active_record("final")["model_version"] == "m-champ-final"


def test_a_validated_challenger_upgrades_a_shadow_champion(reg):
    register(reg, "m-champ-final", status="shadow", skill=0.3)
    decision = register(reg, "m-better-final", status="validated", skill=0.05)
    assert decision["activate"] is True and "upgrades champion from shadow to validated" in decision["reason"]


@pytest.mark.parametrize("skill, edge, activates", [
    (0.12, 0.06, True),   # better on both
    (0.095, 0.045, True),  # within the tolerances: matches
    (0.05, 0.05, False),  # clearly less skilful
    (0.12, 0.00, False),  # skilful but lost its edge over the baselines
])
def test_equal_status_challengers_must_match_the_champions_gate_metrics(reg, skill, edge, activates):
    register(reg, "m-champ-final", status="validated", skill=0.10, edge=0.05)
    decision = register(reg, "m-chal-final", status="validated", skill=skill, edge=edge)
    assert decision["activate"] is activates
    expected = "m-chal-final" if activates else "m-champ-final"
    assert reg.active_record("final")["model_version"] == expected
    assert ("matches or beats" in decision["reason"]) is activates


def test_a_tampered_champion_is_replaced_by_any_approved_challenger(reg):
    register(reg, "m-champ-final", status="validated")
    (reg.settings.models_dir / "m-champ-final" / "sp_x_unusual.txt").write_text("tampered", encoding="utf-8")
    decision = register(reg, "m-new-final", status="shadow")
    assert decision["activate"] is True and "champion unhealthy" in decision["reason"]


def test_a_stale_champion_is_replaced(reg, fake_clock):
    register(reg, "m-old-final", status="validated")
    fake_clock.advance((STALE_CHAMPION_DAYS + 1) * 86400)
    assert register(reg, "m-fresh-final", status="shadow")["activate"] is True


def test_missing_gate_numbers_never_win_a_comparison(reg):
    register(reg, "m-champ-final", status="validated", skill=0.1, edge=0.05)
    broken = make_result(reg, "m-nan-final", status="validated")
    broken.gates.update(brier_skill=None, vs_best_baseline={"mean_diff": "n/a"})
    assert reg.register(broken)["activate"] is False


# -------------------------------------------------------------------------------------- manual governance
def test_approve_validates_input_integrity_and_activates(reg):
    register(reg, "m-1-final", status="shadow")
    register(reg, "m-2-final", status="rejected")
    with pytest.raises(ValueError, match="shadow or validated"):
        reg.approve("m-2-final", "retired")
    with pytest.raises(KeyError):
        reg.approve("m-ghost-final", "shadow")
    reg.approve("m-2-final", "validated")
    record = reg.active_record("final")
    assert record["model_version"] == "m-2-final" and record["status"] == "validated" and record["notes"] == "manual approval"

    (reg.settings.models_dir / "m-1-final" / "calibration.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        reg.approve("m-1-final", "shadow")  # an artifact that fails its integrity check can never be approved


def test_activate_requires_an_approved_and_intact_version(reg):
    register(reg, "m-1-final", status="shadow")
    register(reg, "m-bad-final", status="rejected")
    register(reg, "m-2-final", status="shadow", skill=0.0)  # weaker: stays a challenger
    with pytest.raises(KeyError):
        reg.activate("m-ghost-final")
    with pytest.raises(ValueError, match="approve it first"):
        reg.activate("m-bad-final")
    reg.activate("m-2-final")
    assert reg.active_record("final")["model_version"] == "m-2-final"
    (reg.settings.models_dir / "m-1-final" / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        reg.activate("m-1-final")


# ----------------------------------------------------------------------------------------------- rollback
def test_rollback_reactivates_the_previous_healthy_version_and_retires_the_current(reg):
    register(reg, "m-1-final", status="shadow")
    register(reg, "m-2-final", status="validated")
    assert reg.rollback("final") == "m-1-final"
    assert reg.active_record("final")["model_version"] == "m-1-final"
    assert reg.store.get_model("m-2-final")["status"] == "retired"
    assert reg.rollback("final") is None  # nothing earlier that is still approved (m-2 is retired now)


def test_rollback_skips_versions_that_no_longer_verify(reg):
    for i in (1, 2, 3):
        register(reg, f"m-{i}-final", status="validated", skill=0.1 * i, edge=0.05 * i)
    (reg.settings.models_dir / "m-2-final" / "sp_x_unusual.txt").write_text("corrupt", encoding="utf-8")
    assert reg.rollback("final", quiet=True) == "m-1-final"  # m-2 is the most recent previous one, but broken


def test_rollback_without_any_model_is_a_quiet_no_op(reg):
    assert reg.rollback("pre") is None and reg.rollback("pre", quiet=True) is None
    register(reg, "m-only-pre", stage="pre", status="shadow")
    assert reg.rollback("pre") is None and reg.active_record("pre")["model_version"] == "m-only-pre"


# -------------------------------------------------------------------------------------- retention / views
def test_old_versions_are_pruned_but_the_champion_and_the_newest_are_kept(reg):
    for i in range(KEEP_VERSIONS_PER_STAGE + 3):
        register(reg, f"m-{i:02d}-final", status="validated", skill=0.5 - 0.01 * i, edge=0.1)
    assert reg.active_record("final")["model_version"] == "m-00-final"  # later, weaker models never displaced it
    survivors = sorted(p.name for p in reg.settings.models_dir.iterdir() if p.is_dir())
    assert len(survivors) == KEEP_VERSIONS_PER_STAGE + 1 and "m-00-final" in survivors and "m-07-final" in survivors
    assert "m-01-final" not in survivors  # the oldest non-active versions went first
    outside = reg.settings.data_dir / "elsewhere"
    outside.mkdir()
    reg.store.register_model(model_version="m-x-final", stage="final", status="retired", artifact_path=str(outside))
    reg.prune("final", keep=1)
    assert outside.exists()  # only directories directly inside the models directory are ever deleted


def test_the_summary_lists_every_version_with_its_gate_numbers(reg):
    register(reg, "m-1-final", status="shadow")
    register(reg, "m-2-pre", stage="pre", status="validated")
    rows = {r["version"]: r for r in reg.summary()}
    assert set(rows) == {"m-1-final", "m-2-pre"} and rows["m-1-final"]["active"] is True
    assert rows["m-2-pre"]["status"] == "validated" and rows["m-2-pre"]["train_end"] == "2026-10-30"


def test_the_baseline_bundle_follows_the_configured_thresholds(reg):
    baseline = reg.baseline("final")
    assert baseline.stage == "final" and baseline.status == "baseline"
