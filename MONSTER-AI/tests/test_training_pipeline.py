"""Training -> registry -> scan with a real (small) model, plus governance: rollback, corrupt artifact, fallback to baseline."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone

import pytest

from automation import jobs
from automation.scan import run_scan
from models import training
from models.inference import ModelBundle
from models.registry import ModelRegistry
from tests.seed_data import SMALL_TRAIN as SMALL
from tests.seed_data import make_job_context, panel_ending, populate_store

UTC = timezone.utc
PRELIM = datetime(2026, 11, 2, 3, 15, tzinfo=UTC)
FINAL = datetime(2026, 11, 2, 3, 40, tzinfo=UTC)


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=470, n_symbols=40, seed=21)


@pytest.fixture()
def env(settings, store, fake_clock, panel, monkeypatch):
    real = training.TrainConfig
    monkeypatch.setattr(training, "TrainConfig", lambda **kw: real(**{**kw, **SMALL}))
    cfg = settings.__class__(**{**settings.__dict__, "gate_max_ece": 0.5, "gate_min_brier_skill": -1.0, "telegram_bot_token": "123:TEST",
                                "telegram_chat_id": "42", "send_prelim_report": True})
    populate_store(store, panel)
    return type("Env", (), {"settings": cfg, "store": store, "clock": fake_clock, "panel": panel})


def train_stages(env, stages, run_key="2026-W44"):
    ctx = make_job_context(env.settings, env.store, env.clock, run_key=run_key)
    ctx.options = {"stages": stages}
    result = jobs.train(ctx)
    assert result.status == "success", result.message
    return result


@pytest.fixture()
def trained_pre(env):
    return train_stages(env, "pre")


@pytest.fixture()
def trained_final(env):
    return train_stages(env, "final")


def test_training_registers_both_stages_with_artifacts(env):
    result = train_stages(env, "pre,final")
    registry = ModelRegistry(env.store, env.settings)
    for stage in ("pre", "final"):
        record = registry.active_record(stage)
        assert record is not None and record["status"] in {"shadow", "validated"}
        assert (env.settings.models_dir / record["model_version"] / "manifest.json").exists()
        bundle = registry.active_bundle(stage)
        assert isinstance(bundle, ModelBundle) and bundle.version == record["model_version"] and bundle.stage == stage
    assert result.metrics["pre"]["activated"] is True and result.metrics["final"]["activated"] is True


def test_scan_with_a_trained_model_labels_status_and_never_exceeds_watch_unless_validated(env, trained_pre):
    env.clock.set(PRELIM)
    ctx = make_job_context(env.settings, env.store, env.clock)
    result = run_scan(ctx, "pre")
    assert result.status == "success", result.message
    snap = env.store.latest_snapshot("2026-11-02", "pre")
    report = snap["payload"]["report"]
    record = ModelRegistry(env.store, env.settings).active_record("pre")
    assert report["model_version"] == record["model_version"] == snap["model_version"]
    assert report["model_status"] == record["status"]
    if record["status"] != "validated":
        assert {c["tier"] for c in report["candidates"]} <= {"WATCH"}
    for c in report["candidates"]:
        assert 0.0 <= c["p"]["hit_3"] <= 1.0 and c["range"]["q10"] <= c["range"]["q50"] <= c["range"]["q90"]
        assert c["drivers"], "model explanations should be attached"
    assert any(not p["is_baseline"] for p in env.store.predictions_for("2026-11-02", stage="pre", baselines=None))


def test_final_scan_uses_the_gap_model_for_names_with_a_preopen_price(env, trained_final):
    from tests.seed_data import preopen_rows

    env.clock.set(FINAL)
    env.store.add_preopen_quotes("2026-11-02", preopen_rows(env.panel, env.panel.symbols), "02-Nov-2026 09:08:30")
    env.clock.advance(5)
    result = run_scan(make_job_context(env.settings, env.store, env.clock), "final")
    assert result.status == "success", result.message
    snap = env.store.latest_snapshot("2026-11-02", "final")
    assert snap["payload"]["meta"]["n_fallback"] == 0
    assert snap["model_version"].endswith("-final")
    assert "p_gap_pct" in snap["payload"]["features"]["columns"]


def test_tampered_model_file_or_rewritten_manifest_falls_back_to_baseline_and_alerts(env, trained_pre):
    registry = ModelRegistry(env.store, env.settings)
    record = registry.active_record("pre")
    directory = env.settings.models_dir / record["model_version"]
    manifest = directory / "manifest.json"
    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")  # a rewritten manifest no longer matches the registry
    bundle = ModelRegistry(env.store, env.settings).active_bundle("pre")
    assert bundle.version == "baseline-pre" and bundle.status == "baseline"
    assert any(a["code"] == "model_artifact_invalid" for a in env.store.pending_alerts(20))
    assert env.store.get_model(record["model_version"])["status"] == "retired"


def test_modified_booster_file_is_detected_by_the_manifest(env, trained_pre):
    registry = ModelRegistry(env.store, env.settings)
    record = registry.active_record("pre")
    victim = next((env.settings.models_dir / record["model_version"]).glob("meta_*.txt"))
    victim.write_text(victim.read_text() + "\n", encoding="utf-8")
    assert registry.active_bundle("pre").status == "baseline"


def test_missing_artifact_directory_is_handled(env, trained_pre):
    registry = ModelRegistry(env.store, env.settings)
    record = registry.active_record("pre")
    shutil.rmtree(env.settings.models_dir / record["model_version"])
    assert registry.active_bundle("pre").status == "baseline"


def test_rollback_restores_the_previous_healthy_model(env, trained_pre):
    env.clock.advance(3600)
    train_stages(env, "pre", run_key="2026-W45")
    registry = ModelRegistry(env.store, env.settings)
    versions = {m["model_version"] for m in env.store.list_models("pre")}
    assert len(versions) == 2
    first = trained_pre.metrics["pre"]["version"]
    if registry.active_record("pre")["model_version"] == first:
        pytest.skip("the challenger was not promoted; nothing to roll back")
    assert registry.rollback("pre") == first and registry.active_record("pre")["model_version"] == first


def test_manual_approval_requires_an_existing_verifiable_model(env, trained_pre):
    registry = ModelRegistry(env.store, env.settings)
    with pytest.raises(KeyError):
        registry.approve("does-not-exist", "validated")
    version = registry.active_record("pre")["model_version"]
    with pytest.raises(ValueError):
        registry.approve(version, "bogus")
    registry.approve(version, "validated")
    assert registry.active_record("pre")["status"] == "validated"
