"""The feature version gates which models may score live features. Feature SEMANTICS changed (relative volume, similar-day pool,
stale globals, results horizon, sector dispersion, c_n removed), so models trained under the previous version must not be served:
the registry refuses them, the scan falls back to the heuristic baseline and a retrained model takes over."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from app.errors import ModelArtifactError
from features import schema
from features.builder import build_feature_set
from features.schema import FEATURE_VERSION, schema_hash
from models.inference import ModelBundle, require_schema
from models.registry import ModelRegistry
from tests import test_registry_governance as governance
from tests.synthetic import make_panel

PREVIOUS = "fv1"


def test_the_feature_version_moved_past_the_one_models_were_trained_on():
    assert FEATURE_VERSION != PREVIOUS and FEATURE_VERSION == "fv2"


def test_the_schema_hash_is_tied_to_the_version():
    columns = ["t_ret1", "g_us_ret", "x_sim_n"]
    old_payload = "|".join(f"{c}:{schema.group_of(c)}" for c in columns) + f"#{PREVIOUS}"
    assert schema_hash(columns) != hashlib.sha256(old_payload.encode()).hexdigest()[:16]


def test_feature_sets_carry_the_current_version_and_no_longer_the_dead_feature():
    panel = make_panel(n_dates=150, n_symbols=6, with_fno=False, with_meetings=False)
    fs = build_feature_set(panel, stage="pre", thresholds=(1.0, 2.0, 3.0, 5.0), unusual_k=1.5, cutoff_hhmm="08:45",
                           with_labels=False, row_slice=slice(120, 150))
    assert fs.version == FEATURE_VERSION and "c_n" not in fs.columns
    assert {"x_sim_n", "x_sim_nvalid", "x_sim_dist_mean", "t_rvol1", "s_disp1", "c_days_to_results"} <= set(fs.columns)


def test_a_bundle_of_another_feature_version_fails_the_schema_check():
    bundle = SimpleNamespace(meta={"feature_version": PREVIOUS})
    with pytest.raises(ModelArtifactError, match="fv1 != code version fv2"):
        require_schema(bundle, FEATURE_VERSION)
    require_schema(SimpleNamespace(meta={"feature_version": FEATURE_VERSION}), FEATURE_VERSION)


# ------------------------------------------------------------------ models trained before the bump
@pytest.fixture()
def reg(store, settings, fake_clock):
    fake_clock.advance(1)
    return ModelRegistry(store, settings)


def register_under(reg, monkeypatch, version, name, **kwargs):
    """Register a tiny but real artifact whose metadata says it was built with feature version ``version``."""
    reg.store._clock.advance(1)
    with monkeypatch.context() as patch:
        patch.setattr(governance, "FEATURE_VERSION", version)
        return reg.register(governance.make_result(reg, name, **kwargs))


def test_models_trained_before_the_bump_fall_back_to_the_baseline_until_retrained(reg, monkeypatch):
    register_under(reg, monkeypatch, PREVIOUS, "m-old-pre", stage="pre", status="validated")
    assert reg.active_record("pre")["model_version"] == "m-old-pre"  # registered and active, as it was before the upgrade

    bundle = reg.active_bundle("pre")
    assert bundle.status == "baseline" and bundle.version == "baseline-pre"
    assert reg.store.get_model("m-old-pre")["status"] == "retired"
    alert = next(a for a in reg.store.pending_alerts(20) if a["code"] == "model_artifact_invalid")
    assert "m-old-pre" in alert["message"] and "feature version fv1 != fv2" in alert["message"]

    register_under(reg, monkeypatch, FEATURE_VERSION, "m-new-pre", stage="pre", status="shadow")
    retrained = reg.active_bundle("pre")
    assert isinstance(retrained, ModelBundle) and retrained.version == "m-new-pre" and retrained.status == "shadow"


def test_every_previous_version_model_is_retired_not_just_the_active_one(reg, monkeypatch):
    for name in ("m-a-final", "m-b-final"):
        register_under(reg, monkeypatch, PREVIOUS, name, stage="final", status="shadow")
    assert reg.active_bundle("final").status == "baseline"
    assert {m["status"] for m in reg.store.list_models("final")} == {"retired"}


def test_a_current_model_next_to_old_ones_is_the_one_served(reg, monkeypatch):
    register_under(reg, monkeypatch, PREVIOUS, "m-old-final", stage="final", status="shadow")
    register_under(reg, monkeypatch, FEATURE_VERSION, "m-new-final", stage="final", status="validated")
    assert reg.active_bundle("final").version == "m-new-final"
