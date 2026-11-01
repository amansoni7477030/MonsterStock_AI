"""Evaluation integrity: a forced re-scan freezes a second snapshot, but only the effective (latest) one is ever scored,
and every day is scored on a single stage."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from automation import jobs
from automation.scan import run_scan
from backtesting import evaluate as ev
from backtesting import live_record
from dashboard import app as dash
from database.store import Store
from tests.record_support import Pick, freeze_day, names, seed_universe
from tests.seed_data import make_job_context, panel_ending, populate_store

DAY = "2026-11-02"
UTC = timezone.utc
HIT, MISS = (4.5, 1.0), (1.0, -0.3)  # (realised range %, close-vs-open %): above / below the 3 % primary threshold


@pytest.fixture()
def universe(store):
    symbols = names(12)
    seed_universe(store, symbols)
    return symbols


def table_of(symbols, p=0.5):
    return {s: p for s in symbols}


# ------------------------------------------------------------------------------------------ one freeze counts once
def test_a_rescanned_day_is_scored_once_and_on_its_latest_snapshot(store, settings, universe):
    realised = {s: HIT if i < 4 else MISS for i, s in enumerate(universe)}
    freeze_day(store, DAY, picks=[Pick(s) for s in universe[:5]], table=table_of(universe), realised=realised, tag="first")
    freeze_day(store, DAY, picks=[Pick(s) for s in universe[5:10]], table=table_of(universe), realised=realised, tag="rescan")
    ev.collect_outcomes(store, settings, DAY)
    summary = ev.evaluate_day(store, settings, DAY)
    assert summary.n == 5 and [r.symbol for r in summary.rows] == universe[5:10]  # not 10 rows, not the superseded picks
    assert summary.model_hit_rate == 0.0 and summary.model_hit_ci[0] == 0.0
    assert ev.rolling_summary(store, settings, DAY)["hit_rate_30d"] == 0.0  # the double count would give 4 / 10


def test_outcomes_are_attached_to_the_effective_predictions_only(store, settings, universe):
    realised = {s: HIT for s in universe}
    first, _ = freeze_day(store, DAY, picks=[Pick(s) for s in universe[:5]], table=table_of(universe), realised=realised,
                          baselines={"atr": universe[:5]}, tag="first")
    second, ids = freeze_day(store, DAY, picks=[Pick(s) for s in universe[5:10]], table=table_of(universe), realised=realised,
                             baselines={"atr": universe[5:10]}, tag="rescan")
    counts = ev.collect_outcomes(store, settings, DAY)
    assert counts["attached"] == len(ids) == 10 and counts["missing"] == 0
    assert {r["snapshot_id"] for r in store.evaluated_rows(all_snapshots=True)} == {second}
    assert first < second and not [p for p in store.predictions_without_validated_outcome(DAY, all_snapshots=True)
                                   if p["snapshot_id"] == second]


def test_collect_outcomes_loads_each_snapshot_once_not_once_per_prediction(store, settings, universe, monkeypatch):
    freeze_day(store, DAY, picks=[Pick(s) for s in universe[:5]], table=table_of(universe), realised={s: HIT for s in universe},
               baselines={"atr": universe[:5], "random": universe[5:10]}, tag="only")
    loads = []
    real = Store._snapshot_dict
    monkeypatch.setattr(Store, "_snapshot_dict", staticmethod(lambda row: loads.append(row["snapshot_id"]) or real(row)))
    assert ev.collect_outcomes(store, settings, DAY)["attached"] == 15
    assert len(loads) == 1  # the payload (a gzip of every feature of the universe) is decompressed once


# ------------------------------------------------------------------------------------------- one stage per day
def test_each_day_is_scored_on_its_final_snapshot_else_its_preliminary_one(store, settings, universe):
    both, only_pre = "2026-10-29", "2026-10-30"
    realised = {s: HIT if i < 5 else MISS for i, s in enumerate(universe)}
    for day, stage, picks in ((both, "pre", universe[:5]), (both, "final", universe[5:10]), (only_pre, "pre", universe[:5])):
        freeze_day(store, day, stage=stage, picks=[Pick(s) for s in picks], table=table_of(universe), realised=realised)
        ev.collect_outcomes(store, settings, day)
    rolling = ev.rolling_summary(store, settings, only_pre)
    assert rolling["n_days_30"] == 2.0
    assert rolling["hit_rate_30d"] == pytest.approx(0.5)  # final of day 1 (0 / 5) + the only (preliminary) snapshot of day 2 (5 / 5)
    assert ev.evaluate_day(store, settings, both).stage == "final" and ev.evaluate_day(store, settings, only_pre).stage == "pre"


def test_baselines_follow_the_same_single_stage_rule(store, settings, universe):
    realised = {s: HIT if i < 5 else MISS for i, s in enumerate(universe)}
    freeze_day(store, DAY, stage="pre", picks=[Pick(s) for s in universe[:5]], table=table_of(universe), realised=realised,
               baselines={"atr": universe[:5]})
    freeze_day(store, DAY, stage="final", picks=[Pick(s) for s in universe[5:10]], table=table_of(universe), realised=realised,
               baselines={"atr": universe[5:10]})
    ev.collect_outcomes(store, settings, DAY)
    assert ev.rolling_summary(store, settings, DAY)["baseline_atr_30d"] == 0.0  # the preliminary baseline pick is not pooled in


# ---------------------------------------------------------------------------------------- calibration tables
def informative(realised, settings, p_hit=0.8, p_miss=0.1):
    return {s: p_hit if r > settings.primary_threshold else p_miss for s, (r, _) in realised.items()}


def test_calibration_scores_the_effective_snapshot_of_the_scored_stage(store, settings):
    symbols = names(20)
    seed_universe(store, symbols)
    days = [f"2026-10-{d:02d}" for d in range(1, 13)]
    for i, day in enumerate(days):
        realised = {s: HIT if (i + j) % 2 else MISS for j, s in enumerate(symbols)}
        good, flat = informative(realised, settings), table_of(symbols, 0.4)
        freeze_day(store, day, stage="pre", picks=[Pick(symbols[0])], table=good, realised=realised, tag="pre")
        freeze_day(store, day, stage="final", picks=[Pick(symbols[0])], table=flat, realised=realised, tag="superseded")
        freeze_day(store, day, stage="final", picks=[Pick(symbols[0])], table=good, realised=realised, tag="rescan")
    scored = ev.calibration_from_snapshots(store, settings, days)
    assert scored["brier_skill"] > 0.5 and scored["ece"] < 0.25  # the re-scanned (informative) final table, not the flat one


def test_calibration_falls_back_to_the_preliminary_table_only_when_there_is_no_final_snapshot(store, settings):
    symbols = names(20)
    seed_universe(store, symbols)
    days = [f"2026-10-{d:02d}" for d in range(1, 13)]
    for i, day in enumerate(days):
        realised = {s: HIT if (i + j) % 2 else MISS for j, s in enumerate(symbols)}
        freeze_day(store, day, stage="pre", picks=[Pick(symbols[0])], table=informative(realised, settings), realised=realised)
    assert ev.calibration_from_snapshots(store, settings, days)["brier_skill"] > 0.5


# ---------------------------------------------------------------------------------------------- dashboard
def test_the_dashboard_shows_and_counts_only_effective_snapshots(store, universe):
    realised = {s: HIT for s in universe}
    stale, stale_ids = freeze_day(store, DAY, picks=[Pick(s) for s in universe[:3]], table=table_of(universe), realised=realised,
                                  tag="first")
    fresh, fresh_ids = freeze_day(store, DAY, picks=[Pick(s) for s in universe[3:6]], table=table_of(universe), realised=realised,
                                  tag="rescan")
    for prediction_id, symbol in list(zip(stale_ids, universe[:3], strict=True)) + list(zip(fresh_ids, universe[3:6], strict=True)):
        store.upsert_outcome(prediction_id, DAY, symbol, source="nse_bhavcopy", validated=True, ohlcv={"open": 100, "high": 104,
                             "low": 99, "close": 103}, metrics={"range_pct": 5.0})
    data = dash.predictions_data(store, day=DAY, stage="final")
    assert [p["symbol"] for p in data["predictions"]] == universe[3:6] and data["n"] == 3
    assert dash.performance_data(store)["outcomes"] == {"validated": 3, "provisional": 0, "days": 1}
    assert stale < fresh


# ------------------------------------------------------------------------------- a real scan, forced twice
PRELIM = datetime(2026, 11, 2, 3, 15, tzinfo=UTC)  # 08:45 IST


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=300, n_symbols=24, seed=13)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42",
                                "send_prelim_report": True, "send_evening_report": True})
    populate_store(store, panel)
    return SimpleNamespace(settings=cfg, store=store, clock=fake_clock, panel=panel)


def test_scan_force_then_evaluate_scores_every_pick_once(env):
    env.clock.set(PRELIM)
    assert run_scan(make_job_context(env.settings, env.store, env.clock), "pre").status == "success"
    env.clock.advance(300)  # a later forced re-scan has a later cutoff, hence a different frozen snapshot
    assert run_scan(make_job_context(env.settings, env.store, env.clock, force=True), "pre").status == "success"
    assert env.store.conn.execute("SELECT COUNT(*) AS n FROM snapshots WHERE stage = 'pre'").fetchone()["n"] == 2
    last = env.panel.c.iloc[-1]
    for j, symbol in enumerate(env.panel.symbols):
        open_ = float(last[symbol]) * 1.002
        span = 4.5 if j % 2 == 0 else 1.2
        env.store.upsert_bars(env.store.instrument_id(symbol), [{
            "trade_date": DAY, "open": open_, "high": open_ * (1 + span / 100 * 0.6), "low": open_ * (1 - span / 100 * 0.4),
            "close": open_ * 1.004, "volume": 2e6}], source="nse_bhavcopy", source_rank=2, quality="official")
    env.clock.set(datetime(2026, 11, 2, 13, 0, tzinfo=UTC))  # 18:30 IST
    result = jobs.evaluate(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "success" and result.message.startswith("5 picks scored"), result.message
    summary = ev.evaluate_day(env.store, env.settings, DAY)
    assert summary.n == 5 and len({r.symbol for r in summary.rows}) == 5


def test_a_real_snapshot_payload_supports_the_regime_liquidity_and_ranking_slices(env):
    env.clock.set(PRELIM)
    assert run_scan(make_job_context(env.settings, env.store, env.clock), "pre").status == "success"
    payload = env.store.latest_snapshot(DAY, "pre")["payload"]
    day = live_record.day_context(payload)
    assert len(day.table) > 10 and day.regime and all(label.startswith("regime:") for label in day.regime)
    assert "regime:expiry-day" in day.regime or "regime:non-expiry-day" in day.regime
    assert set(day.liquidity.values()) == {"liquidity:low", "liquidity:mid", "liquidity:high"}
    last = env.panel.c.iloc[-1]
    for j, symbol in enumerate(env.panel.symbols):
        open_ = float(last[symbol]) * 1.002
        span = 4.5 if j % 2 == 0 else 1.2
        env.store.upsert_bars(env.store.instrument_id(symbol), [{
            "trade_date": DAY, "open": open_, "high": open_ * (1 + span / 100 * 0.6), "low": open_ * (1 - span / 100 * 0.4),
            "close": open_ * 1.004, "volume": 2e6}], source="nse_bhavcopy", source_rank=2, quality="official")
    ev.collect_outcomes(env.store, env.settings, DAY)
    record = ev.evaluate_day(env.store, env.settings, DAY).live_record["20d"]
    slices = {r["slice"] for r in record}
    assert "all" in slices and any(s.startswith("regime:") for s in slices)  # all five picks share the day's regime: n = 5
    by_name = {r["name"]: r for r in record if r["slice"] == "all"}
    assert by_name["range_mae_q50"]["n"] == 5 and by_name["ndcg_at_5"]["n"] == 1
