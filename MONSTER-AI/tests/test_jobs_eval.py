"""Outcome collection and evaluation: frozen predictions -> realised outcomes -> summary, metrics and the evening report."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from automation import jobs
from automation.scan import run_scan
from backtesting import evaluate as ev
from tests.seed_data import make_job_context, panel_ending, populate_store

UTC = timezone.utc
PRELIM = datetime(2026, 11, 2, 3, 15, tzinfo=UTC)  # 08:45 IST
EVENING = datetime(2026, 11, 2, 13, 0, tzinfo=UTC)  # 18:30 IST


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=300, n_symbols=24, seed=13)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42",
                                "send_prelim_report": True})
    populate_store(store, panel)
    return type("Env", (), {"settings": cfg, "store": store, "clock": fake_clock, "panel": panel})


def add_monday_bars(env, *, source="nse_bhavcopy", rank=2, wide=4.5, narrow=1.2):
    """Monday's realised session for every symbol: even symbols have a wide range, odd ones a narrow one."""
    last = env.panel.c.iloc[-1]
    for j, symbol in enumerate(env.panel.symbols):
        open_ = float(last[symbol]) * 1.002
        span = wide if j % 2 == 0 else narrow
        env.store.upsert_bars(env.store.instrument_id(symbol), [{
            "trade_date": "2026-11-02", "open": open_, "high": open_ * (1 + span / 100 * 0.6), "low": open_ * (1 - span / 100 * 0.4),
            "close": open_ * 1.004, "volume": 2e6}], source=source, source_rank=rank, quality="official" if rank > 1 else "ok")


def frozen_prelim(env):
    env.clock.set(PRELIM)
    ctx = make_job_context(env.settings, env.store, env.clock)
    assert run_scan(ctx, "pre").status == "success"
    return ctx


# -------------------------------------------------------------------------------------------- pure functions
def test_outcome_metrics_follow_the_blueprint_definitions():
    m = ev.outcome_metrics(100.0, 104.0, 99.0, 101.0)
    assert m["range_pct"] == pytest.approx(5.0)  # (H-L)/O
    assert m["exc_pct"] == pytest.approx(4.0)  # larger excursion from the open
    assert m["dir_ret_pct"] == pytest.approx(1.0)
    assert m["mfe_pct"] == pytest.approx(4.0) and m["mae_pct"] == pytest.approx(1.0)


def test_feature_payload_round_trips_and_handles_nan():
    import pandas as pd

    frame = pd.DataFrame({"a": [1.23456789, np.nan], "b": [100.0, 2.0]}, index=["X", "Y"])
    payload = ev.feature_payload(frame)
    assert payload["columns"] == ["a", "b"]
    assert ev.feature_value(payload, "X", "a") == pytest.approx(1.2346)
    assert ev.feature_value(payload, "Y", "a") is None
    assert ev.feature_value(payload, "Z", "a") is None and ev.feature_value(None, "X", "a") is None


# ----------------------------------------------------------------------------------------- end to end
def test_collect_outcomes_attaches_official_results_without_touching_predictions(env):
    frozen_prelim(env)
    before = env.store.predictions_for("2026-11-02", stage="pre", baselines=None)
    assert ev.collect_outcomes(env.store, env.settings, "2026-11-02")["missing"] == len(before)  # no bars yet
    add_monday_bars(env)
    counts = ev.collect_outcomes(env.store, env.settings, "2026-11-02")
    assert counts["attached"] == len(before) and counts["validated"] == len(before) and counts["missing"] == 0
    after = env.store.predictions_for("2026-11-02", stage="pre", baselines=None)
    assert [(p["prediction_id"], p["p"], p["tier"]) for p in before] == [(p["prediction_id"], p["p"], p["tier"]) for p in after]
    rows = env.store.evaluated_rows(since="2026-11-02", until="2026-11-02")
    assert rows and all(r["o_validated"] == 1 for r in rows)
    assert "unusual" in env.store.conn.execute("SELECT hit_json FROM outcomes LIMIT 1").fetchone()["hit_json"]
    assert ev.collect_outcomes(env.store, env.settings, "2026-11-02")["attached"] == 0  # nothing left to upgrade


def test_vendor_outcomes_are_provisional_until_the_official_file_arrives(env):
    frozen_prelim(env)
    add_monday_bars(env, source="yahoo", rank=1)
    counts = ev.collect_outcomes(env.store, env.settings, "2026-11-02")
    assert counts["validated"] == 0 and counts["attached"] > 0
    summary = ev.evaluate_day(env.store, env.settings, "2026-11-02")
    assert summary is not None and summary.provisional is True
    add_monday_bars(env)  # official bars replace the vendor ones (higher rank wins)
    upgraded = ev.collect_outcomes(env.store, env.settings, "2026-11-02")
    assert upgraded["validated"] == upgraded["attached"] > 0
    assert ev.evaluate_day(env.store, env.settings, "2026-11-02").provisional is False


def test_evaluate_job_scores_persists_metrics_and_sends_one_evening_report(env):
    ctx = frozen_prelim(env)
    add_monday_bars(env)
    env.clock.set(EVENING)
    eval_ctx = make_job_context(env.settings, env.store, env.clock)
    result = jobs.evaluate(eval_ctx)
    assert result.status == "success", result.message
    summary = ev.evaluate_day(env.store, env.settings, "2026-11-02")
    assert summary.n == 5 and summary.model_version == "baseline-pre" and summary.stage == "pre"
    assert 0.0 <= summary.model_hit_rate <= 1.0 and summary.model_hit_ci[0] <= summary.model_hit_rate <= summary.model_hit_ci[1]
    assert {"atr", "range20", "volume", "random"} <= set(summary.baseline_hit_rates)
    names = {m["name"] for m in env.store.latest_metrics("daily")}
    assert {"topn_hit_rate", "topn_mean_range_pct", "baseline_atr_hit_rate"} <= names
    sent = len(eval_ctx.telegram.sent)
    assert sent >= 1 and any("2026-11-02" in text for _, text in eval_ctx.telegram.sent)
    again = jobs.evaluate(make_job_context(env.settings, env.store, env.clock))
    assert again.status == "success"
    del ctx


def test_evaluate_waits_for_data_and_skips_days_without_predictions(env):
    frozen_prelim(env)
    env.clock.set(EVENING)
    result = jobs.evaluate(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "failed" and "no outcomes" in result.message
    empty = jobs.evaluate(make_job_context(env.settings, env.store, env.clock, run_key="2026-10-30"))
    assert empty.status == "skipped" and "no predictions" in empty.message


def test_missing_bar_for_a_symbol_is_listed_not_fatal(env):
    frozen_prelim(env)
    add_monday_bars(env)
    victim = env.store.predictions_for("2026-11-02", stage="pre")[0]["symbol"]
    env.store.conn.execute("DELETE FROM bars_daily WHERE trade_date = '2026-11-02' AND instrument_id = ?",
                           (env.store.instrument_id(victim),))
    counts = ev.collect_outcomes(env.store, env.settings, "2026-11-02")
    assert counts["missing"] >= 1
    summary = ev.evaluate_day(env.store, env.settings, "2026-11-02")
    assert victim in summary.missing_symbols and summary.n == 4


# --------------------------------------------------------------------------------------------- rolling stats
def test_calibration_from_frozen_tables_scores_informative_probabilities(env):
    days = [d.date().isoformat() for d in env.panel.dates[-12:]]
    thr = env.settings.primary_threshold
    for day in days:
        table = {}
        for symbol in env.panel.symbols:
            bar = env.store.get_bars(env.store.instrument_id(symbol), day, day).iloc[0]
            realised = (bar["high"] - bar["low"]) / bar["open"] * 100
            table[symbol] = 0.8 if realised > thr else 0.1
        env.store.insert_snapshot(trade_date=day, stage="final", cutoff_ts=f"{day}T03:38:00Z", feature_version="fv1", model_version="m",
                                  code_version="t", payload={"table": table}, data_quality={})
    out = ev.calibration_from_snapshots(env.store, env.settings, days)
    assert out["brier"] is not None and out["brier_skill"] > 0.5 and 0 <= out["ece"] <= 0.25
    assert ev.calibration_from_snapshots(env.store, env.settings, days[:3]) == {"brier": None, "ece": None, "brier_skill": None}


def test_rolling_summary_pools_days_and_baselines(env):
    frozen_prelim(env)
    add_monday_bars(env)
    ev.collect_outcomes(env.store, env.settings, "2026-11-02")
    rolling = ev.rolling_summary(env.store, env.settings, "2026-11-02")
    assert rolling["n_days_30"] == 1.0 and rolling["hit_rate_30d"] is not None
    assert "baseline_atr_30d" in rolling and "baseline_random_90d" in rolling
