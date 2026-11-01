"""Evaluation statistics against analytically known values, and the walk-forward backtest report (the ``backtest`` command)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from backtesting import metrics as M
from backtesting import walkforward as wf
from models import training
from tests.seed_data import SMALL_TRAIN, panel_ending, populate_store

nan = float("nan")


def isnan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


# ------------------------------------------------------------------------------------------------ intervals
def test_the_wilson_interval_is_sane_at_small_samples_and_extreme_rates():
    assert all(isnan(v) for v in M.wilson_interval(0, 0))
    low, high = M.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    low, high = M.wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)  # never a degenerate [0, 0]
    low, high = M.wilson_interval(10, 10)
    assert high == 1.0 and low == pytest.approx(0.7225, abs=1e-3)


def test_mean_ci_handles_nan_and_tiny_samples():
    assert all(isnan(v) for v in M.mean_ci([nan, nan]))
    mean, low, high = M.mean_ci([1.0, nan])
    assert mean == 1.0 and isnan(low) and isnan(high)
    mean, low, high = M.mean_ci([1.0, 2.0, 3.0, 4.0])
    half = 1.96 * np.std([1, 2, 3, 4], ddof=1) / 2
    assert mean == 2.5 and low == pytest.approx(2.5 - half) and high == pytest.approx(2.5 + half)


# ------------------------------------------------------------------------------------- probabilistic scores
def test_brier_logloss_and_skill_have_the_textbook_values():
    p, y = np.array([1.0, 0.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0, 0.0])
    assert M.brier(p, y) == 0.0 and M.brier_skill(p, y) == 1.0
    base = np.full(4, 0.5)
    assert M.brier(base, y) == 0.25 and M.brier_skill(base, y) == pytest.approx(0.0)  # climatology has no skill
    worse = np.array([0.0, 1.0, 0.0, 1.0])
    assert M.brier_skill(worse, y) == pytest.approx(-3.0)
    assert M.brier_skill(np.array([0.8, 0.8]), np.array([1.0, 1.0])) != M.brier_skill(np.array([0.8, 0.8]), np.array([1.0, 1.0]), 0.5)
    assert M.log_loss(np.array([0.9]), np.array([1.0])) == pytest.approx(-math.log(0.9))
    assert math.isfinite(M.log_loss(np.array([0.0, 1.0]), np.array([1.0, 0.0])))  # clipped, never infinite


def test_nan_pairs_are_ignored_and_empty_inputs_give_nan():
    p, y = np.array([0.5, nan, 0.5]), np.array([1.0, 1.0, nan])
    assert M.brier(p, y) == 0.25
    for fn in (M.brier, M.brier_skill, M.log_loss, M.auc, M.ece):
        assert isnan(fn(np.array([nan]), np.array([nan])))
    assert isnan(M.brier_skill(np.array([0.5, 0.5]), np.array([1.0, 1.0])))  # constant outcome: no reference to beat


def test_reliability_table_and_ece_measure_calibration_by_equal_mass_bins():
    p = np.array([0.2] * 10 + [0.8] * 10)
    y = np.array([1] * 4 + [0] * 6 + [1] * 6 + [0] * 4, dtype=float)
    table = M.reliability_table(p, y, bins=2)
    assert list(table["n"]) == [10, 10] and list(table["observed"]) == [0.4, 0.6]
    assert M.ece(p, y, bins=2) == pytest.approx(0.2)
    perfect = np.array([0.0] * 5 + [1.0] * 5)
    assert M.ece(perfect, perfect, bins=2) == 0.0
    assert M.reliability_table(np.array([0.1, 0.2]), np.array([0.0, 1.0]), bins=10).empty and isnan(M.ece(np.array([0.1]), np.array([1.0])))


def test_auc_covers_perfect_reversed_tied_and_single_class_cases():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert M.auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert M.auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert M.auc(np.full(4, 0.5), y) == 0.5
    assert isnan(M.auc(np.array([0.1, 0.9]), np.array([1.0, 1.0])))


def test_quantile_and_interval_scores():
    y = np.array([1.0, 2.0, 3.0])
    assert M.pinball(y, np.array([2.0, 2.0, 2.0]), 0.5) == pytest.approx(1.0 / 3.0)
    assert M.pinball(y, np.array([2.0, 2.0, 2.0]), 0.9) == pytest.approx((0.9 * 1 + 0 + 0.1 * 1) / 3 * 1.0, abs=0.3)  # asymmetric
    assert M.pinball(y, y, 0.5) == 0.0 and isnan(M.pinball(np.array([]), np.array([]), 0.5))
    assert M.mae(y, y + 1) == 1.0 and isnan(M.mae(np.array([nan]), np.array([1.0])))
    assert M.interval_coverage(y, y - 1, y + 1) == 1.0
    assert M.interval_coverage(y, np.array([2.5, 2.5, 2.5]), np.array([3.5, 3.5, 3.5])) == pytest.approx(1 / 3)
    assert isnan(M.interval_coverage(np.array([nan]), np.array([0.0]), np.array([1.0])))


def test_summarize_binary_collects_the_headline_numbers():
    p = np.array([0.9, 0.8, 0.2, 0.1, nan])
    y = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    out = M.summarize_binary(p, y)
    assert out["n"] == 4 and out["base_rate"] == 0.5 and out["auc"] == 1.0 and out["brier_skill"] > 0.8
    assert M.summarize_binary(p, y, base_rate=0.5)["brier_skill"] == out["brier_skill"]
    assert M.summarize_binary(np.array([nan]), np.array([nan]))["n"] == 0


# ------------------------------------------------------------------------------------------------- ranking
def test_ndcg_rewards_the_right_order_and_handles_degenerate_inputs():
    relevance = np.array([3.0, 2.0, 1.0, 0.0])
    assert M.ndcg_at_k(np.array([4, 3, 2, 1.0]), relevance, 4) == pytest.approx(1.0)
    assert M.ndcg_at_k(np.array([1, 2, 3, 4.0]), relevance, 4) < 0.75
    assert M.ndcg_at_k(np.array([4, 3, 2, 1.0]), relevance, 99) == pytest.approx(1.0)  # k is capped at the list length
    assert isnan(M.ndcg_at_k(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 2)) and isnan(M.ndcg_at_k(np.array([]), np.array([]), 3))


def ranking_frame(days: int = 40, names: int = 10, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        for n in range(names):
            signal = rng.normal()
            rows.append({"date": f"2026-01-{d + 1:02d}", "symbol": f"S{n}", "good": signal, "noise": rng.normal(),
                         "hit": float(signal + 0.5 * rng.normal() > 0.5), "range": signal + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_daily_topn_and_the_pooled_hit_rate():
    frame = ranking_frame()
    good = M.daily_topn(frame, "good", "hit", 3)
    noise = M.daily_topn(frame, "noise", "hit", 3)
    assert len(good) == 40 and (good["n"] == 3).all()
    rate, low, high, n = M.topn_hit_rate(good)
    assert n == 120 and low < rate < high and rate > M.topn_hit_rate(noise)[0]
    assert all(isnan(v) for v in M.topn_hit_rate(pd.DataFrame())[:3]) and M.topn_hit_rate(pd.DataFrame())[3] == 0
    short = pd.DataFrame({"date": ["d1", "d1"], "s": [nan, nan], "v": [1.0, 2.0]})
    assert M.daily_topn(short, "s", "v", 3).empty  # a day without any score contributes nothing


def test_the_paired_bootstrap_detects_a_real_edge_and_not_a_null_one():
    frame = ranking_frame(days=80)
    model, baseline, noise = (M.daily_topn(frame, c, "hit", 3) for c in ("good", "noise", "noise"))
    real = M.paired_day_bootstrap(model, M.daily_topn(frame, "noise", "hit", 3), n_boot=400)
    assert real["n_days"] == 80 and real["mean_diff"] > 0.15 and real["p10"] > 0 and real["ci_low"] < real["mean_diff"] < real["ci_high"]
    null = M.paired_day_bootstrap(noise, baseline, n_boot=400)
    assert null["mean_diff"] == 0.0 and null["ci_low"] <= 0.0 <= null["ci_high"]
    assert M.paired_day_bootstrap(model, model.iloc[0:0])["n_days"] == 0
    nan_days = model.assign(mean=nan)
    assert M.paired_day_bootstrap(nan_days, baseline)["n_days"] == 0


def test_rank_ic_is_positive_for_a_real_signal_and_blank_when_days_are_too_small():
    frame = ranking_frame(days=30)
    informative = M.spearman_ic(frame, "good", "range")
    assert informative["n_days"] == 30 and informative["ic_mean"] > 0.5 and informative["t_stat"] > 10
    pure_noise = M.spearman_ic(frame, "noise", "range")
    assert abs(pure_noise["ic_mean"]) < 0.25
    tiny = frame[frame["symbol"].isin(["S0", "S1", "S2"])]
    assert M.spearman_ic(tiny, "good", "range")["n_days"] == 0 and isnan(M.spearman_ic(tiny, "good", "range")["ic_mean"])
    one_day = frame[frame["date"] == "2026-01-01"]
    single = M.spearman_ic(one_day, "good", "range")
    assert single["n_days"] == 1 and isnan(single["ic_std"]) and isnan(single["t_stat"])


# ---------------------------------------------------------------------------------------- walk-forward report
def test_the_report_formatter_tolerates_missing_numbers():
    assert wf._fmt(0.123456) == "0.123" and wf._fmt(None) == "n/a" and wf._fmt("x") == "n/a" and wf._fmt(2.0, 1) == "2.0"
    result = type("R", (), {"status": "shadow", "gates": {"reason": "not calibrated", "limits": {}}, "metrics": {}})()
    text = wf.render_markdown("final", result, 5)
    assert "status **SHADOW**" in text and "n/a" in text and "not calibrated" in text


@pytest.fixture()
def env(settings, store, fake_clock, monkeypatch):
    real = training.TrainConfig
    monkeypatch.setattr(training, "TrainConfig", lambda **kw: real(**{**kw, **SMALL_TRAIN}))
    cfg = settings.__class__(**{**settings.__dict__, "gate_max_ece": 0.5, "gate_min_brier_skill": -1.0})
    populate_store(store, panel_ending(n_dates=470, n_symbols=40, seed=21))
    cfg.ensure_dirs()
    return type("Env", (), {"settings": cfg, "store": store, "clock": fake_clock})


def test_backtest_writes_an_honest_report_with_limitations(env, tmp_path, capsys):
    out = tmp_path / "reports"
    code = wf.run_backtest(env.settings, env.store, stages=("pre",), years=3, out_dir=out)
    assert code == 0
    markdown = (out / "backtest-2026-10-30.md").read_text(encoding="utf-8")
    assert "# MONSTER AI walk-forward backtest (data through 2026-10-30)" in markdown and "## Stage `pre`" in markdown
    assert "| scorer | hit rate |" in markdown and "| head | AUC |" in markdown
    assert "### Limitations" in markdown and "survivorship bias" in markdown and "not a tradable edge" in markdown
    summary = json.loads((out / "backtest-2026-10-30.json").read_text(encoding="utf-8"))
    assert summary["through"] == "2026-10-30" and summary["stages"]["pre"]["status"] in {"shadow", "validated", "rejected"}
    assert "report written to" in capsys.readouterr().out
    # a backtest never registers or activates anything
    assert env.store.list_models() == [] and not list(env.settings.models_dir.glob("m-*"))


def test_backtest_reports_a_stage_it_cannot_evaluate_and_exits_non_zero(env, tmp_path):
    out = tmp_path / "reports"
    env.store.conn.execute("DELETE FROM bars_daily WHERE trade_date < '2026-08-01'")  # far too little history
    code = wf.run_backtest(env.settings, env.store, stages=("pre",), years=3, out_dir=out)
    assert code == 1
    assert "Not evaluated:" in (out / "backtest-2026-10-30.md").read_text(encoding="utf-8")
    assert "error" in json.loads((out / "backtest-2026-10-30.json").read_text(encoding="utf-8"))["stages"]["pre"]
