"""Conviction tiers, learned uncertainty ceilings and the reasons given when the system abstains."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ranking.ranker import (
    RankInput,
    decide_tier,
    describe_blockers,
    enrichment_pool,
    rank,
    why_below_moderate,
)
from ranking.types import Catalyst

LEARNED = (0.60, 0.75)  # what a model might learn for names of a given probability: far above the configured 0.35 / 0.55


def build(probs, uncs, *, status="validated", ceilings=None, spec_std=None, catalysts=None, stage="pre") -> RankInput:
    symbols = [f"S{i}" for i in range(len(probs))]
    features = pd.DataFrame({"t_atr14_pct": 3.0, "t_range_mean20": 3.5, "u_hist_days": 500.0, "t_tv_med20_cr": 200.0}, index=symbols)
    columns = {"p_primary": probs, "p_hit_3": probs, "rng_q50": 4.0, "uncertainty": uncs}
    if spec_std is not None:
        columns["sp_std_unusual"] = spec_std
    return RankInput(
        stage=stage, trade_date="2026-11-02", model_version="m-test", model_status=status, primary_threshold=3.0, features=features,
        preds=pd.DataFrame(columns, index=symbols), instruments={s: {"name": s, "sector": "IT", "is_fno": False} for s in symbols},
        last_close={s: 900.0 for s in symbols}, catalysts=catalysts or {},
        uncertainty_ceilings={s: ceilings for s in symbols} if ceilings else {},
    )


# ------------------------------------------------------------------------------------------- decide_tier
@pytest.mark.parametrize("p, unc, tier", [
    (0.60, 0.30, "STRONG"), (0.60, 0.50, "MODERATE"), (0.60, 0.70, "WATCH"), (0.45, 0.30, "MODERATE"), (0.45, 0.60, "WATCH"),
    (0.30, 0.10, "WATCH"), (0.20, 0.10, None), (float("nan"), 0.10, None), (0.0, 0.10, None),
])
def test_configured_ceilings_apply_when_the_model_learned_none(settings, p, unc, tier):
    assert decide_tier(p, unc, 0, False, "validated", settings) == tier


@pytest.mark.parametrize("unc, tier", [(0.58, "STRONG"), (0.60, "STRONG"), (0.70, "MODERATE"), (0.75, "MODERATE"), (0.80, "WATCH")])
def test_learned_ceilings_replace_the_configured_constants(settings, unc, tier):
    assert decide_tier(0.9, unc, 0, False, "validated", settings, LEARNED) == tier


def test_downgrade_rules_still_apply_on_top_of_the_learned_ceilings(settings):
    assert decide_tier(0.9, 0.5, 2, False, "validated", settings, LEARNED) == "MODERATE"  # two contradictions: one tier down
    assert decide_tier(0.9, 0.7, 2, False, "validated", settings, LEARNED) == "WATCH"
    assert decide_tier(0.9, 0.5, 1, False, "validated", settings, LEARNED) == "STRONG"  # one contradiction is tolerated
    assert decide_tier(0.9, 0.5, 0, True, "validated", settings, LEARNED) == "MODERATE"  # a data-quality warning caps at MODERATE
    for status in ("shadow", "baseline"):
        assert decide_tier(0.9, 0.1, 0, False, status, settings, LEARNED) == "WATCH"


@pytest.mark.parametrize("p, unc, contradictions, cause", [
    (0.30, 0.10, 0, "probability"), (0.45, 0.80, 0, "uncertainty"), (0.45, 0.80, 3, "uncertainty"), (0.45, 0.30, 2, "contradictions"),
])
def test_the_reason_a_name_stays_at_watch_is_the_first_failing_condition(settings, p, unc, contradictions, cause):
    assert decide_tier(p, unc, contradictions, False, "validated", settings, (0.5, 0.6)) == "WATCH"
    assert why_below_moderate(p, unc, contradictions, settings, (0.5, 0.6)) == cause


def test_blockers_are_described_most_common_first_with_counts(settings):
    text = describe_blockers({"contradictions": 1, "uncertainty": 3, "probability": 1}, 5, settings)
    assert text == ("model uncertainty above the usual range for names of similar probability (3 of 5); "
                    "probability below the 40% MODERATE threshold (1 of 5); conflicting evidence (1 of 5)")


# ------------------------------------------------------------------------------------------------ rank()
def tiers(result) -> list[str]:
    return [c.tier for c in result.candidates]


def test_a_validated_model_with_learned_ceilings_can_reach_strong_where_the_constants_never_could(settings):
    probs, uncs = [0.92, 0.88, 0.85, 0.80, 0.78], [0.56, 0.60, 0.62, 0.70, 0.77]  # live-like: top picks are volatile names
    stuck = rank(build(probs, uncs), settings)
    assert tiers(stuck) == ["WATCH"] * 5 and stuck.no_setup
    learned = rank(build(probs, uncs, ceilings=(0.62, 0.78)), settings)
    assert tiers(learned) == ["STRONG", "STRONG", "STRONG", "MODERATE", "MODERATE"]
    assert not learned.no_setup and learned.no_setup_reason is None
    assert learned.table["tier"].tolist() == tiers(learned)


def test_the_ranking_itself_is_still_by_probability_only(settings):
    result = rank(build([0.50, 0.90, 0.70], [0.10, 0.99, 0.50], ceilings=LEARNED), settings)
    assert [c.symbol for c in result.candidates] == ["S1", "S2", "S0"]


def test_abstention_names_uncertainty_when_every_pick_is_above_the_usual_range_for_its_probability(settings):
    result = rank(build([0.92, 0.88, 0.85, 0.80, 0.78], [0.85, 0.9, 0.8, 0.95, 0.82], ceilings=LEARNED), settings)
    assert result.no_setup
    assert result.no_setup_reason == ("no candidate reached MODERATE conviction (best P = 92%): "
                                      "model uncertainty above the usual range for names of similar probability (5 of 5)")


def test_abstention_names_a_low_probability(settings):
    result = rank(build([0.35, 0.33, 0.30], [0.2, 0.2, 0.2], ceilings=LEARNED), settings)
    assert result.no_setup_reason == ("no candidate reached MODERATE conviction (best P = 35%): "
                                      "probability below the 40% MODERATE threshold (3 of 3)")


def test_abstention_names_conflicting_evidence(settings):
    weak = Catalyst(headline="rumour", confidence=0.2)
    catalysts = {f"S{i}": [weak] for i in range(2)}
    result = rank(build([0.5, 0.45], [0.3, 0.3], ceilings=LEARNED, spec_std=[0.2, 0.2], catalysts=catalysts), settings)
    assert result.no_setup
    assert result.no_setup_reason.endswith("conflicting evidence (2 of 2)")


def test_a_mix_of_causes_is_reported_with_counts(settings):
    result = rank(build([0.9, 0.8, 0.3], [0.9, 0.9, 0.1], ceilings=LEARNED), settings)
    assert result.no_setup_reason == ("no candidate reached MODERATE conviction (best P = 90%): "
                                      "model uncertainty above the usual range for names of similar probability (2 of 3); "
                                      "probability below the 40% MODERATE threshold (1 of 3)")


def test_nothing_is_listed_below_the_watch_probability(settings):
    result = rank(build([0.2, 0.1], [0.1, 0.1], ceilings=LEARNED), settings)
    assert result.candidates == [] and result.no_setup
    assert result.no_setup_reason == "no eligible name reached the minimum probability 25% (best 20%)"


@pytest.mark.parametrize("status", ["shadow", "baseline"])
def test_an_unvalidated_model_says_so_instead_of_blaming_uncertainty(settings, status):
    result = rank(build([0.9, 0.8], [0.1, 0.1], status=status, ceilings=LEARNED), settings)
    assert tiers(result) == ["WATCH", "WATCH"]
    assert result.no_setup_reason.startswith(f"model is {status.upper()}")


def test_a_symbol_without_a_learned_reference_falls_back_to_the_constants(settings):
    inp = build([0.9, 0.9], [0.5, 0.5], ceilings=None)
    inp.uncertainty_ceilings = {"S0": LEARNED}  # S1 was scored by a model that learned nothing (e.g. the heuristic baseline)
    result = rank(inp, settings)
    assert dict(zip((c.symbol for c in result.candidates), tiers(result), strict=True)) == {"S0": "STRONG", "S1": "MODERATE"}


def test_each_candidate_reports_its_own_uncertainty(settings):
    result = rank(build([0.9, 0.8], [0.4123, 0.5], ceilings=LEARNED), settings)
    assert [c.uncertainty for c in result.candidates] == [0.412, 0.5]
    assert np.isfinite(result.table["uncertainty"]).all()


# ------------------------------------------------------------------------------- enrichment pool
def test_the_enrichment_pool_skips_names_the_ranking_would_drop(settings):
    """The best-scored names are often illiquid: enriching them wastes the news budget and leaves the listed names bare."""
    inp = build([0.95, 0.94, 0.93, 0.60, 0.55, 0.50], [0.3] * 6, ceilings=LEARNED)
    inp.features.loc[["S0", "S1"], "t_tv_med20_cr"] = 1.0  # far below the liquidity floor
    inp.last_close = {**inp.last_close, "S2": 10.0}  # below the minimum price
    assert enrichment_pool(inp.features, inp.preds, inp.last_close, settings, 3) == ["S3", "S4", "S5"]
    assert enrichment_pool(inp.features, inp.preds, inp.last_close, settings, 2) == ["S3", "S4"]
    assert [c.symbol for c in rank(inp, settings).candidates] == ["S3", "S4", "S5"]  # exactly what the ranking can list


def test_the_enrichment_pool_matches_the_ranking_order_and_tolerates_missing_probabilities(settings):
    inp = build([0.5, 0.9, float("nan"), 0.7], [0.3] * 4, ceilings=LEARNED)
    pool = enrichment_pool(inp.features, inp.preds, inp.last_close, settings, 10)
    assert pool == ["S1", "S3", "S0"] == [c.symbol for c in rank(inp, settings, top_n=10).candidates]
    assert enrichment_pool(inp.features, inp.preds, inp.last_close, settings, 0) == []


# ------------------------------------------------------------------------- per-model disagreement limit
def test_the_disagreement_limit_of_the_scoring_model_decides_the_conflict(settings):
    """Specialists of this model normally disagree by ~0.04, so 0.08 is a conflict for it; the default limit (0.12) would not see it."""
    probs, uncs, spread = [0.5, 0.5], [0.3, 0.3], [0.08, 0.08]
    default = rank(build(probs, uncs, ceilings=LEARNED, spec_std=spread), settings)
    assert [len(c.contradictions) for c in default.candidates] == [0, 0]
    inp = build(probs, uncs, ceilings=LEARNED, spec_std=spread)
    inp.spec_std_limits = {"S0": 0.05}  # S1 was scored by a model that learned nothing: it keeps the default
    result = rank(inp, settings)
    assert {c.symbol: len(c.contradictions) for c in result.candidates} == {"S0": 1, "S1": 0}


def test_a_failing_explainer_is_logged_and_the_pick_is_still_listed(settings, caplog):
    import logging

    inp = build([0.9], [0.3], ceilings=LEARNED)

    def broken(symbol):
        raise RuntimeError("shap exploded")

    inp.explainer = broken
    with caplog.at_level(logging.WARNING, logger="ranking.ranker"):
        result = rank(inp, settings)
    assert [c.symbol for c in result.candidates] == ["S0"] and result.candidates[0].drivers == []
    assert any("explanation unavailable for S0" in r.message and "shap exploded" in r.message for r in caplog.records)
