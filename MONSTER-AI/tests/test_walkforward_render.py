"""The walk-forward markdown report prints the holdout's range-forecast error, interval coverage and direction accuracy with n."""

from __future__ import annotations

import re
from types import SimpleNamespace

from backtesting import walkforward as wf


def holdout_result(**over):
    metrics = {
        "heads": {"hit_3": {"n": 12345, "auc": 0.7, "brier": 0.2, "brier_skill": 0.1, "ece": 0.02, "base_rate": 0.4},
                  "up": {"n": 11000, "accuracy": 0.5234, "base_rate": 0.51, "auc": 0.52, "brier": 0.25, "brier_skill": 0.0,
                         "ece": 0.01}},
        "quantiles": {"range_q10": {"pinball": 0.1234, "below_rate": 0.1, "nominal": 0.1},
                      "range_q50": {"pinball": 0.4567, "below_rate": 0.5, "nominal": 0.5},
                      "range_q90": {"pinball": 0.1301, "below_rate": 0.9, "nominal": 0.9},
                      "range_80pct_interval_coverage": 0.7921, "range_mae_median": 1.2345},
        "ranking": {"top_n": 5, "scorers": {}, "best_baseline": "atr", "vs_best_baseline": {}},
    }
    metrics.update(over.pop("metrics", {}))
    base = {"status": "shadow", "gates": {"reason": "ok", "limits": {}, "holdout_days": 60}, "metrics": metrics,
            "meta": {"primary_threshold": 3.0}}
    base.update(over)
    return SimpleNamespace(**base)


def table_rows(text: str) -> dict[str, list[str]]:
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if line.startswith("|") and len(cells) == 3:
            rows[cells[0]] = cells[1:]
    return rows


def test_the_forecast_quality_table_shows_every_error_measure_with_its_sample_size():
    text = wf.render_markdown("final", holdout_result(), 5)
    assert "| forecast quality (holdout) | value | n |" in text
    rows = table_rows(text)
    assert rows["MAE of the median range forecast (pp)"] == ["1.234", "12345"]
    assert rows["Pinball loss of the range forecast, q10 / q50 / q90"] == ["0.123 / 0.457 / 0.130", "12345"]
    assert rows["80% interval coverage (nominal 0.80)"] == ["0.792", "12345"]
    assert rows["Direction accuracy, up vs not-up close (secondary; majority class would score 0.510)"] == ["0.523", "11000"]


def test_a_sample_size_recorded_by_the_trainer_for_the_quantile_heads_is_preferred():
    result = holdout_result(metrics={"quantiles": {"n": 12000, "range_mae_median": 1.0, "range_80pct_interval_coverage": 0.8,
                                                   "range_q10": {"pinball": 0.1}, "range_q50": {"pinball": 0.2},
                                                   "range_q90": {"pinball": 0.3}}})
    rows = table_rows(wf.render_markdown("final", result, 5))
    assert rows["MAE of the median range forecast (pp)"][1] == "12000" and rows["80% interval coverage (nominal 0.80)"][1] == "12000"


def test_missing_measures_print_n_a_and_never_a_number():
    result = SimpleNamespace(status="shadow", gates={"reason": "x", "limits": {}}, metrics={})
    rows = table_rows(wf.render_markdown("pre", result, 3))
    assert rows["MAE of the median range forecast (pp)"] == ["n/a", "n/a"]
    assert rows["Pinball loss of the range forecast, q10 / q50 / q90"] == ["n/a / n/a / n/a", "n/a"]
    assert rows["80% interval coverage (nominal 0.80)"] == ["n/a", "n/a"]
    assert rows["Direction accuracy, up vs not-up close (secondary)"] == ["n/a", "n/a"]


def test_an_interval_that_could_not_be_estimated_prints_n_a_not_nan():
    assert wf._fmt(float("nan")) == "n/a" and wf._fmt(float("inf")) == "n/a" and wf._fmt(-0.5) == "-0.500"
    ranking = {"top_n": 5, "scorers": {"model": {"hit_rate": 0.6, "ci_low": float("nan"), "ci_high": float("nan"),
                                                 "mean_range": 3.2, "n_selected": 10}}}
    text = wf.render_markdown("final", holdout_result(metrics={"ranking": ranking}), 5)
    assert "| model | 0.600 | n/a - n/a | 3.20 | 10 |" in text and not re.search(r"\bnan\b", text)


def test_the_n_falls_back_to_any_head_when_the_primary_is_unknown():
    result = holdout_result(meta={})
    assert table_rows(wf.render_markdown("final", result, 5))["MAE of the median range forecast (pp)"][1] == "12345"
