"""Honest labelling: heuristic baseline output is never presented as a calibrated probability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.calibration import Calibrator, fit_calibrator
from models.inference import BaselineBundle
from notifications.report import HEURISTIC_LABEL, render_final_report
from ranking.ranker import p_and_range
from ranking.types import Candidate, ReportContext


def context(status: str) -> ReportContext:
    cand = Candidate(symbol="ABC", rank=1, tier="WATCH", p={"hit_1": 0.9, "hit_3": 0.6, "unusual": 0.2}, primary_threshold=3.0,
                     range={"q10": 2.0, "q50": 3.4, "q90": 6.0})
    return ReportContext(trade_date="2026-11-02", stage="final", snapshot_ist="2026-11-02 09:12:00 IST", model_version="m1",
                         model_status=status, candidates=[cand])


def text_of(ctx: ReportContext) -> str:
    return "\n".join(render_final_report(ctx))


def test_baseline_probabilities_are_labelled_heuristic_not_calibrated():
    text = text_of(context("baseline"))
    assert HEURISTIC_LABEL in text and "(calibrated)" not in text
    assert "P(up close) n/a / P(down close) n/a" in text


@pytest.mark.parametrize("status", ["shadow", "validated"])
def test_trained_models_report_calibrated_probabilities(status):
    text = text_of(context(status))
    assert "(calibrated)" in text and HEURISTIC_LABEL not in text


def test_baseline_bundle_is_monotone_bounded_and_has_no_direction_view():
    bundle = BaselineBundle("final", (1.0, 2.0, 3.0, 5.0), 3.0)
    X = pd.DataFrame({"t_range_mean20": [1.0, 2.5, 5.0, np.nan], "t_atr14_pct": [1.2, 2.5, 4.5, 3.0],
                      "p_gap_atr": [0.0, 1.0, 3.0, 0.0]}, index=list("ABCD"))
    out = bundle.predict(X)
    assert (out["p_primary"].iloc[:3].diff().dropna() > 0).all()  # a wider usual range means a higher probability
    assert (out["p_hit_1"] >= out["p_hit_2"]).all() and (out["p_hit_2"] >= out["p_hit_3"]).all() and (out["p_hit_3"] >= out["p_hit_5"]).all()
    assert out[["p_up", "p_down"]].isna().all().all()
    assert out["p_primary"].between(0, 1).all() and (out["uncertainty"] == 0.8).all()
    assert out["rng_q10"].lt(out["rng_q50"]).all() and out["rng_q50"].lt(out["rng_q90"]).all()
    p, rng = p_and_range(out.iloc[1])
    assert "up" not in p and "down" not in p and set(rng) >= {"q10", "q50", "q90"}
    assert bundle.explain(X, 0)[0].startswith("heuristic baseline")
    gap_wider = bundle.predict(X.assign(p_gap_atr=[0.0, 3.0, 3.0, 0.0]))["rng_q50"].iloc[1]
    assert gap_wider > out["rng_q50"].iloc[1]  # a larger opening gap widens the expected range


def test_isotonic_calibrator_is_monotone_and_reduces_overconfidence():
    rng = np.random.default_rng(0)
    p_raw = rng.uniform(0.0, 1.0, 5000)
    y = (rng.uniform(size=5000) < 0.2 + 0.5 * p_raw).astype(float)  # true probability is a flattened version of the score
    cal = fit_calibrator(p_raw[:4000], y[:4000])
    grid = np.linspace(0, 1, 101)
    mapped = cal.transform(grid)
    assert (np.diff(mapped) >= -1e-12).all() and mapped.min() >= 0 and mapped.max() <= 1
    err_raw = np.mean((p_raw[4000:] - y[4000:]) ** 2)
    err_cal = np.mean((cal.transform(p_raw[4000:]) - y[4000:]) ** 2)
    assert err_cal < err_raw
    assert Calibrator.from_dict(cal.to_dict()).transform(grid) == pytest.approx(mapped)


def test_calibrator_degrades_to_identity_on_tiny_or_degenerate_samples():
    tiny = fit_calibrator(np.array([0.2, 0.8]), np.array([0.0, 1.0]))
    assert tiny.kind == "identity" and tiny.transform(np.array([0.3, 0.7])) == pytest.approx([0.3, 0.7])
    constant = fit_calibrator(np.linspace(0, 1, 500), np.zeros(500))  # no positive labels: nothing to learn from
    assert constant.kind == "identity" and constant.transform(np.array([0.1, 0.9])) == pytest.approx([0.1, 0.9])
