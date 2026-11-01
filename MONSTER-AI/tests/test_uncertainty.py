"""Uncertainty: the shared formula, the drift share, and the probability-conditional reference learned at training time."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from automation.scan import _uncertainty_limits
from models.inference import (
    BaselineBundle,
    combine_uncertainty,
    disagreement_limit,
    drift_share,
    pick_uncertainty_ceilings,
)
from models.training import _pick_uncertainty_reference, daily_top_rows, uncertainty_curve

SCALES = {"spec_std_p95": 0.1, "spread_p95": 2.0, "dist_median": 1.0, "dist_p95": 3.0, "drift_p95": 0.2}
CURVE = {"p": [0.4, 0.6, 0.9], "p50": [0.40, 0.50, 0.70], "p85": [0.50, 0.60, 0.80], "n": 900}


def arr(*values: float) -> np.ndarray:
    return np.array(values, dtype=float)


# --------------------------------------------------------------------------------------- the formula
def test_everything_at_its_95th_percentile_is_fully_uncertain_and_everything_typical_is_reliable():
    assert combine_uncertainty(arr(0.1), arr(2.0), arr(3.0), arr(0.2), SCALES)[0] == pytest.approx(1.0)
    assert combine_uncertainty(arr(0.0), arr(0.0), arr(1.0), arr(0.0), SCALES)[0] == pytest.approx(0.0)


def test_weights_are_specialist_disagreement_interval_width_similar_day_distance_and_drift():
    zero = dict(spec_std=arr(0.0), spread=arr(0.0), dist=arr(1.0), drift=arr(0.0))
    for part, value, weight in [("spec_std", 0.1, 0.35), ("spread", 2.0, 0.25), ("dist", 3.0, 0.20), ("drift", 0.2, 0.20)]:
        one = {**zero, part: arr(value)}
        assert combine_uncertainty(one["spec_std"], one["spread"], one["dist"], one["drift"], SCALES)[0] == pytest.approx(weight)


def test_values_beyond_the_scale_are_clipped_and_missing_ones_count_as_the_middle():
    assert combine_uncertainty(arr(5.0), arr(50.0), arr(99.0), arr(1.0), SCALES)[0] == pytest.approx(1.0)
    missing = combine_uncertainty(arr(np.nan), arr(0.0), arr(np.nan), arr(0.0), SCALES)[0]
    assert missing == pytest.approx(0.35 * 0.5 + 0.20 * 0.5)


def test_scales_missing_from_an_old_model_fall_back_to_safe_defaults():
    assert combine_uncertainty(arr(0.1), arr(2.0), arr(1.0), arr(0.2), {})[0] == pytest.approx(1.0)


def test_it_is_vectorised_over_names():
    out = combine_uncertainty(arr(0.0, 0.1), arr(0.0, 2.0), arr(1.0, 3.0), arr(0.0, 0.2), SCALES)
    assert out.tolist() == pytest.approx([0.0, 1.0])


# ------------------------------------------------------------------------------------------- drift
def test_drift_is_the_share_of_known_values_outside_the_training_band():
    values = np.array([[0.0, 5.0, np.nan], [10.0, 5.0, 1.0], [np.nan, np.nan, np.nan], [5.0, 5.0, 5.0]])
    share = drift_share(values, arr(1, 1, 1), arr(9, 9, 9))
    assert share.tolist() == pytest.approx([0.5, 1 / 3, 0.0, 0.0])


# --------------------------------------------------------------------- ceilings read from the curve
def test_ceilings_follow_the_names_own_probability_by_linear_interpolation():
    strong, moderate = pick_uncertainty_ceilings({"unc_scales": {"pick_curve": CURVE}}, [0.4, 0.5, 0.6, 0.75, 0.9])
    assert strong.tolist() == pytest.approx([0.40, 0.45, 0.50, 0.60, 0.70])
    assert moderate.tolist() == pytest.approx([0.50, 0.55, 0.60, 0.70, 0.80])


def test_beyond_the_learned_range_the_nearest_level_applies_and_a_missing_probability_stays_missing():
    strong, moderate = pick_uncertainty_ceilings({"unc_scales": {"pick_curve": CURVE}}, pd.Series([0.1, 0.99, np.nan]))
    assert strong[:2].tolist() == pytest.approx([0.40, 0.70]) and moderate[:2].tolist() == pytest.approx([0.50, 0.80])
    assert np.isnan(strong[2]) and np.isnan(moderate[2])


def test_a_single_level_curve_is_a_constant():
    meta = {"unc_scales": {"pick_curve": {"p": [0.7], "p50": [0.5], "p85": [0.6], "n": 40}}}
    strong, moderate = pick_uncertainty_ceilings(meta, [0.1, 0.7, 0.95])
    assert strong.tolist() == pytest.approx([0.5] * 3) and moderate.tolist() == pytest.approx([0.6] * 3)


def broken(**changes):
    return {"unc_scales": {"pick_curve": {**CURVE, **changes}}}


@pytest.mark.parametrize("meta", [
    {}, {"unc_scales": None}, {"unc_scales": {}}, {"unc_scales": {"pick_curve": None}}, {"unc_scales": {"pick_curve": [0.5]}},
    broken(p=[]), broken(p50=[0.4, 0.5]), broken(p85=None), broken(p=[0.4, 0.4, 0.9]), broken(p=[0.9, 0.6, 0.4]),
    broken(p=[0.4, float("nan"), 0.9]), broken(p50=[0.4, 0.5, float("inf")]), broken(p50=["0.4", "0.5", "0.7"]),
    broken(p50=[True, True, True], p85=[True, True, True]), broken(p50=[0.0, 0.5, 0.7]), broken(p50=[0.55, 0.5, 0.7]),
    broken(p85=[0.5, 0.6, 1.2]), broken(p="0.4,0.6,0.9"),
])
def test_an_absent_or_implausible_curve_means_use_the_configured_constants(meta):
    assert pick_uncertainty_ceilings(meta, [0.5, 0.8]) is None


def test_the_heuristic_baseline_has_no_learned_reference():
    assert pick_uncertainty_ceilings(BaselineBundle("pre", (3.0,), 3.0).meta, [0.5]) is None


# --------------------------------------------------------- the specialist disagreement the model calls abnormal
def test_the_disagreement_limit_is_the_models_own_95th_percentile():
    assert disagreement_limit({"unc_scales": SCALES}) == 0.1
    assert disagreement_limit({"unc_scales": {"spec_std_p95": 3}}) == 3.0  # an integer is fine, and comes back as a float


@pytest.mark.parametrize("meta", [
    {}, {"unc_scales": None}, {"unc_scales": {}}, {"unc_scales": {"spec_std_p95": None}}, {"unc_scales": {"spec_std_p95": 0.0}},
    {"unc_scales": {"spec_std_p95": -0.1}}, {"unc_scales": {"spec_std_p95": float("nan")}},
    {"unc_scales": {"spec_std_p95": float("inf")}}, {"unc_scales": {"spec_std_p95": "0.1"}},
    {"unc_scales": {"spec_std_p95": True}}, {"unc_scales": [0.1]},
])
def test_no_or_implausible_limit_means_the_caller_keeps_its_default(meta):
    assert disagreement_limit(meta) is None


def test_the_heuristic_baseline_has_no_disagreement_limit():
    assert disagreement_limit(BaselineBundle("pre", (3.0,), 3.0).meta) is None


# ------------------------------------------------------------------------- limits per scored symbol
def test_each_scored_name_gets_the_limits_of_its_own_probability():
    bundle = SimpleNamespace(meta={"unc_scales": {"pick_curve": CURVE}})
    preds = pd.DataFrame({"p_primary": [0.4, 0.9, np.nan]}, index=["AAA", "BBB", "CCC"])
    assert _uncertainty_limits(bundle, preds) == {"AAA": (0.40, 0.50), "BBB": (0.70, 0.80)}


def test_a_model_without_a_curve_gives_no_limits_so_the_constants_apply():
    preds = pd.DataFrame({"p_primary": [0.4]}, index=["AAA"])
    assert _uncertainty_limits(SimpleNamespace(meta={"unc_scales": {}}), preds) == {}
    assert _uncertainty_limits(BaselineBundle("pre", (3.0,), 3.0), preds) == {}


# ---------------------------------------------------------------------------------- training: the curve
def test_the_curve_describes_uncertainty_at_rising_probability_levels():
    p = np.linspace(0.3, 0.9, 240)
    curve = uncertainty_curve(p, p / 2)  # uncertainty is exactly half the probability
    assert len(curve["p"]) == 8 and curve["p"] == sorted(curve["p"]) and curve["n"] == 240
    assert curve["p50"] == pytest.approx([x / 2 for x in curve["p"]], abs=0.01)
    assert all(a <= b for a, b in zip(curve["p50"], curve["p85"], strict=True))
    assert curve["p50"] == sorted(curve["p50"])


def test_too_few_rows_give_fewer_bins_or_none():
    rng = np.random.default_rng(1)
    assert uncertainty_curve(rng.random(29), rng.random(29)) is None
    assert len(uncertainty_curve(rng.random(30), rng.random(30))["p"]) == 1
    assert len(uncertainty_curve(rng.random(95), rng.random(95))["p"]) == 3


def test_missing_values_are_ignored_and_equal_probability_levels_are_merged():
    p = np.r_[np.full(60, 0.5), np.nan, 0.7]
    unc = np.r_[np.linspace(0.3, 0.6, 60), 0.5, np.nan]
    curve = uncertainty_curve(p, unc)  # a calibrated probability is a step function: many identical values
    assert curve["n"] == 60 and curve["p"] == [0.5] and 0.3 < curve["p50"][0] < curve["p85"][0] < 0.6


def test_the_best_scoring_rows_of_every_day_are_selected_within_the_mask():
    score = arr(0.9, 0.5, 0.7, 0.1, 0.8, 0.6, np.nan, 0.4)
    day = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    mask = np.array([True, True, True, True, True, True, True, False])
    assert sorted(daily_top_rows(score, day, mask, 2).tolist()) == [0, 2, 4, 5]
    assert sorted(daily_top_rows(score, day, mask, 1).tolist()) == [0, 4]
    assert sorted(daily_top_rows(score, day, mask, 10).tolist()) == [0, 1, 2, 3, 4, 5]  # NaN and masked-out rows never selected


def test_no_selectable_rows_gives_an_empty_selection():
    out = daily_top_rows(arr(0.1, 0.2), np.array([1, 1]), np.array([False, False]), 3)
    assert out.size == 0


def reference_inputs(*, n_days: int = 40, per_day: int = 6):
    """Rows ordered by day, scores 0..5 within a day. With top_n=1 the pool is the 3 best (scores 3, 4, 5, i.e. probability
    0.6 / 0.8 / 1.0, uncertainty rising with it); the other rows are decoys with extreme uncertainty."""
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day)
    score = np.tile(np.arange(per_day, dtype=float), n_days)
    pick = score >= 3
    Xv = np.column_stack([np.where(pick, 0.5, 100.0), np.full(n, 0.5)]).astype(np.float32)
    return dict(
        score=score, to_p=lambda raw: raw / 5.0, day=day, hold=np.ones(n, dtype=bool), std=np.where(pick, 0.02 * score, 0.5),
        spread=np.where(pick, 1.0, 9.0), dist_all=np.where(pick, 2.6, 50.0), meta_rows=np.arange(n), Xv=Xv, names=["a", "b"],
        stats={"features": {"a": {"p01": 0.0, "p99": 1.0}, "b": {"p01": 0.0, "p99": 1.0}}}, scales=SCALES, top_n=1,
    )


def pick_uncertainty(std: float, *, dist: float = 2.6, drift: float = 0.0) -> float:
    return float(combine_uncertainty(arr(std), arr(1.0), arr(dist), arr(drift), SCALES)[0])


def test_the_reference_is_the_uncertainty_of_the_daily_top_names_by_probability_level_and_ignores_the_rest():
    curve = _pick_uncertainty_reference(**reference_inputs())["pick_curve"]
    assert curve["p"] == [0.6, 0.8, 1.0] and curve["n"] == 120
    assert curve["p50"] == sorted(curve["p50"]) and curve["p50"][0] < curve["p50"][-1]  # more uncertain at higher probability
    assert curve["p50"][0] == pytest.approx(pick_uncertainty(0.06)) and curve["p50"][-1] == pytest.approx(pick_uncertainty(0.10))
    assert max(curve["p85"]) < 0.7 < pick_uncertainty(0.5, dist=50.0, drift=0.5)  # the extreme decoys never entered


def test_drift_of_the_picks_is_measured_against_the_training_band():
    kw = reference_inputs()
    kw["Xv"] = np.column_stack([np.full(240, 5.0), np.full(240, 0.5)]).astype(np.float32)  # feature "a" is outside [0, 1] everywhere
    curve = _pick_uncertainty_reference(**kw)["pick_curve"]
    assert curve["p50"][0] == pytest.approx(pick_uncertainty(0.06, drift=0.5))


def test_a_missing_similar_day_feature_or_feature_band_does_not_break_the_reference():
    kw = reference_inputs()
    kw["dist_all"], kw["stats"] = arr(np.nan), {"features": {}}  # no x_sim_dist_mean column and no feature band
    curve = _pick_uncertainty_reference(**kw)["pick_curve"]
    assert curve["p50"][0] == pytest.approx(pick_uncertainty(0.06, dist=np.nan))


def test_nothing_is_learned_without_enough_out_of_sample_rows():
    kw = reference_inputs()
    assert _pick_uncertainty_reference(**{**kw, "hold": np.zeros(240, dtype=bool)}) == {}
    assert _pick_uncertainty_reference(**reference_inputs(n_days=5)) == {}  # 15 rows: too few for one stable level
