"""The rolling live record (blueprint acceptance 23.4): range-forecast error, interval coverage, direction accuracy, realised range
versus baselines and top-k ranking quality of the FROZEN picks, each with n and a day-block interval, plus performance slices."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from backtesting import evaluate as ev
from backtesting import live_record as lr
from backtesting import metrics as M
from tests.record_support import Pick, names, seed_days, seed_universe

UNIVERSE = names(12)

# the five picks S00..S04: realised (range %, close-vs-open %) and the forecast P(up close)
RANGES = [2.0, 3.5, 5.0, 7.0, 0.5]
DIRECTIONS = [0.5, -0.5, 1.0, -1.0, 0.0]
P_UP = [0.7, 0.3, 0.4, 0.8, 0.9]
FORECAST = {"q10": 1.5, "q50": 3.0, "q90": 6.0}  # the Pick defaults


@pytest.fixture()
def universe(store):
    seed_universe(store, UNIVERSE)
    return UNIVERSE


def table_of(p: float = 0.5):
    return {s: p for s in UNIVERSE}


def fixed_plan(i, day, *, up=True, drop=()):
    realised = {s: (RANGES[k], DIRECTIONS[k]) if k < 5 else (1.0, 0.1) for k, s in enumerate(UNIVERSE) if s not in drop}
    picks = [Pick(UNIVERSE[k], up=P_UP[k] if up else None) for k in range(5)]
    return {"picks": picks, "table": table_of(), "realised": realised,
            "baselines": {"atr": UNIVERSE[:5], "random": UNIVERSE[5:10]}}


def record_of(store, settings, count, plan=fixed_plan):
    days = seed_days(store, count, plan)
    summary = ev.evaluate_day(store, settings, days[-1])
    assert summary is not None
    return summary.live_record


def by_name(rows, slice_="all"):
    return {r["name"]: r for r in rows if r["slice"] == slice_}


# ---------------------------------------------------------------------------------- range-forecast errors
def test_range_forecast_errors_equal_the_hand_computed_values_with_their_sample_sizes(store, settings, universe):
    record = record_of(store, settings, 10)
    assert set(record) == {"20d"}  # ten evaluated days: the 60-day and all-time windows would repeat it
    rows = by_name(record["20d"])
    assert rows["range_mae_q50"]["value"] == pytest.approx(2.0)  # |range - 3.0| = 1, .5, 2, 4, 2.5
    assert rows["range_pinball_q50"]["value"] == pytest.approx(1.0)  # half the MAE at the median
    assert rows["range_pinball_q10"]["value"] == pytest.approx(0.41)
    assert rows["range_pinball_q90"]["value"] == pytest.approx(0.44)
    assert rows["range_coverage_80"]["value"] == pytest.approx(0.6) and rows["range_coverage_80"]["extra"]["nominal"] == 0.8
    assert rows["topn_hit_rate"]["value"] == pytest.approx(0.6)
    assert rows["pick_range_mean"]["value"] == pytest.approx(3.6) and rows["pick_range_median"]["value"] == pytest.approx(3.5)
    for name in ("range_mae_q50", "range_pinball_q10", "range_pinball_q50", "range_pinball_q90", "range_coverage_80",
                 "topn_hit_rate", "pick_range_mean", "pick_range_median"):
        assert rows[name]["n"] == 50 and rows[name]["extra"]["n_days"] == 10  # 5 picks x 10 days, never a bare percentage
        assert rows[name]["slice"] == "all"


def test_hit_rates_are_reported_at_every_configured_threshold(store, settings, universe):
    ranges = [1.5, 2.5, 4.0, 6.0, 0.5]  # clear of every threshold (1, 2, 3, 5 %)

    def plan(i, day):
        base = fixed_plan(i, day)
        base["realised"] = {s: (ranges[k], 0.2) if k < 5 else (1.0, 0.1) for k, s in enumerate(UNIVERSE)}
        return base

    rows = by_name(record_of(store, settings, 10, plan)["20d"])
    assert rows["topn_hit_rate"]["value"] == pytest.approx(0.4)  # the primary threshold (3 %) keeps its name
    assert [rows[f"topn_hit_rate_{t}"]["value"] for t in (1, 2, 5)] == pytest.approx([0.8, 0.6, 0.2])
    assert all(rows[f"topn_hit_rate_{t}"]["n"] == 50 and rows[f"topn_hit_rate_{t}"]["extra"]["threshold"] == float(t) for t in (1, 2, 5))
    assert "topn_hit_rate_3" not in rows


def test_direction_accuracy_counts_non_flat_sessions_and_calls_only(store, settings, universe):
    rows = by_name(record_of(store, settings, 10)["20d"])
    direction = rows["direction_accuracy"]
    assert direction["value"] == pytest.approx(0.5) and direction["n"] == 40  # 4 of the 5 picks closed away from the open
    assert direction["extra"]["flat_excluded"] == 10 and direction["extra"]["up_share"] == pytest.approx(0.5)


def test_a_model_without_a_direction_view_reports_no_direction_accuracy(store, settings, universe):
    rows = by_name(record_of(store, settings, 10, lambda i, day: fixed_plan(i, day, up=False))["20d"])
    assert "direction_accuracy" not in rows and "range_mae_q50" in rows  # the heuristic baseline has no P(up)


def test_picks_without_an_outcome_do_not_count(store, settings, universe):
    rows = by_name(record_of(store, settings, 10, lambda i, day: fixed_plan(i, day, drop=("S04",)))["20d"])
    assert rows["range_mae_q50"]["n"] == 40 and rows["topn_hit_rate"]["value"] == pytest.approx(0.75)  # S04 had no bar


def test_every_interval_brackets_its_value_once_there_are_enough_days(store, settings, universe):
    def varied(i, day):
        scale = 1 + 0.1 * ((i * 3) % 7 - 3)
        plan = fixed_plan(i, day)
        plan["realised"] = {s: (r * scale, d * scale) for s, (r, d) in plan["realised"].items()}
        return plan

    rows = by_name(record_of(store, settings, 30, varied)["20d"])
    for name in ("range_mae_q50", "range_pinball_q10", "range_coverage_80", "topn_hit_rate", "pick_range_mean", "pick_range_median"):
        row = rows[name]
        assert row["ci_low"] <= row["value"] <= row["ci_high"], name
    assert rows["range_mae_q50"]["ci_low"] < rows["range_mae_q50"]["ci_high"] and rows["range_mae_q50"]["extra"]["n_days"] == 20


def test_too_few_days_give_the_value_and_the_sample_size_but_no_interval(store, settings, universe):
    rows = by_name(record_of(store, settings, M.MIN_BOOT_DAYS - 1)["20d"])
    assert rows["range_mae_q50"]["n"] == 5 * (M.MIN_BOOT_DAYS - 1)
    assert rows["range_mae_q50"]["ci_low"] is None and rows["range_mae_q50"]["ci_high"] is None


def test_hit_rate_intervals_treat_a_days_picks_as_one_unit(store, settings, universe):
    def clustered(i, day):  # hot spells and cold spells: all five picks of a day share its fate
        hot = (i // 5) % 2 == 0
        plan = fixed_plan(i, day)
        plan["realised"] = {s: (5.0, 0.5) if hot else (1.0, 0.1) for s in UNIVERSE}
        return plan

    row = by_name(record_of(store, settings, 20, clustered)["20d"])["topn_hit_rate"]
    wilson_low, wilson_high = M.wilson_interval(row["value"] * row["n"], row["n"])
    assert row["value"] == pytest.approx(0.5) and row["n"] == 100
    assert (row["ci_high"] - row["ci_low"]) > 2.0 * (wilson_high - wilson_low)  # the pick-level interval is far too narrow


# ---------------------------------------------------------------------------------------------- windows
def test_windows_are_the_last_20_60_and_all_evaluated_days_and_never_repeat_each_other(store, settings, universe):
    days = seed_days(store, 70, fixed_plan)
    record = ev.evaluate_day(store, settings, days[-1]).live_record
    assert {label: by_name(rows)["topn_hit_rate"]["extra"]["n_days"] for label, rows in record.items()} == {
        "20d": 20, "60d": 60, "all": 70}
    shorter = ev.evaluate_day(store, settings, days[24]).live_record  # as of the 25th day
    assert {label: by_name(rows)["topn_hit_rate"]["extra"]["n_days"] for label, rows in shorter.items()} == {"20d": 20, "60d": 25}


def test_an_empty_record_is_empty(store, settings, universe):
    assert lr.build(store, settings, "2026-06-30", {}) == {}


# ------------------------------------------------------------------------------------- versus baselines
def test_realised_ranges_and_hit_rates_are_compared_with_every_baseline(store, settings, universe):
    rows = by_name(record_of(store, settings, 10)["20d"])
    assert rows["baseline_atr_hit_rate"]["value"] == pytest.approx(0.6) and rows["baseline_atr_range_mean"]["value"] == pytest.approx(3.6)
    assert rows["baseline_atr_range_median"]["value"] == pytest.approx(3.5)
    assert rows["baseline_random_hit_rate"]["value"] == 0.0 and rows["baseline_random_range_mean"]["value"] == pytest.approx(1.0)
    assert rows["baseline_random_range_median"]["value"] == pytest.approx(1.0)
    assert all(rows[f"baseline_{b}_hit_rate"]["n"] == 50 for b in ("atr", "random"))


# ---------------------------------------------------------------------------------------- ranking quality
def ranked_plan(direction: int):
    """The frozen probability of name j rises (direction=+1) or falls (-1) with its realised range."""
    def plan(i, day):
        realised = {s: (1.0 + 0.5 * j + 0.1 * (i % 3), 0.1) for j, s in enumerate(UNIVERSE)}
        table = {s: 0.05 + 0.07 * (j if direction > 0 else len(UNIVERSE) - 1 - j) for j, s in enumerate(UNIVERSE)}
        picks = [Pick(s) for s in sorted(UNIVERSE, key=table.get, reverse=True)[:5]]
        return {"picks": picks, "table": table, "realised": realised}
    return plan


def test_a_ranking_that_follows_the_realised_ranges_scores_perfectly(store, settings, universe):
    rows = by_name(record_of(store, settings, 12, ranked_plan(+1))["20d"])
    assert rows["ndcg_at_5"]["value"] == pytest.approx(1.0) and rows["rank_ic"]["value"] == pytest.approx(1.0)
    assert rows["ndcg_at_5"]["n"] == rows["rank_ic"]["n"] == 12  # days, not picks
    assert rows["ndcg_at_5"]["extra"]["random"] < 0.8 and rows["ndcg_at_5"]["extra"]["k"] == 5  # the no-skill reference
    assert rows["rank_ic"]["ci_low"] is not None


def test_a_reversed_ranking_scores_worse_than_chance(store, settings, universe):
    rows = by_name(record_of(store, settings, 12, ranked_plan(-1))["20d"])
    assert rows["rank_ic"]["value"] == pytest.approx(-1.0) and rows["ndcg_at_5"]["value"] < rows["ndcg_at_5"]["extra"]["random"]


def test_tied_scores_are_ranked_neutrally_not_by_symbol_order(store, settings, universe):
    def tied(i, day):
        realised = {s: (1.0 + 0.5 * j, 0.1) for j, s in enumerate(UNIVERSE)}  # the best names sit last in symbol order
        return {"picks": [Pick(s) for s in UNIVERSE[:5]], "table": table_of(0.4), "realised": realised}

    rows = by_name(record_of(store, settings, 30, tied)["20d"])
    assert abs(rows["ndcg_at_5"]["value"] - rows["ndcg_at_5"]["extra"]["random"]) < 0.08  # a stable sort would give ~0.3 here
    assert "rank_ic" not in rows  # a constant score has no rank correlation


def test_ranking_quality_needs_the_frozen_probability_table(store, settings, universe):
    def no_table(i, day):
        plan = fixed_plan(i, day)
        plan["table"] = {}
        return plan

    rows = by_name(record_of(store, settings, 10, no_table)["20d"])
    assert "ndcg_at_5" not in rows and "rank_ic" not in rows and "range_mae_q50" in rows


# ------------------------------------------------------------------------------------------------ slices
SLICE_PICKS = ["S00", "S05", "S10", "S03", "S08"]  # ranks 1..5
TURNOVER = {s: 10.0 * (k + 1) for k, s in enumerate(UNIVERSE)}  # terciles: S00-S03 low, S04-S07 mid, S08-S11 high
TIERS = ["STRONG", "MODERATE", "WATCH", "WATCH", "WATCH"]


def realised_range(i: int, symbol: str) -> float:
    return 1.5 + 0.4 * UNIVERSE.index(symbol) + 0.5 * (i % 3)


def slice_plan(i, day):
    market = {"m_trend": 1.0 if i < 8 else -1.0, "m_volreg": 2.0 if i < 8 else 3.0, "m_risk": 1.0,
              "m_expiry_share": 1.0 if i == 11 else 0.0}
    events = {0: "results" if i % 2 == 0 else None, 1: "ipo" if i < 4 else None, 2: "block deal" if i < 5 else None}
    picks = [Pick(s, tier=TIERS[k], event=events.get(k), up=0.6) for k, s in enumerate(SLICE_PICKS)]
    return {"picks": picks, "table": table_of(), "realised": {s: (realised_range(i, s), 0.2) for s in UNIVERSE},
            "market": market, "turnover": TURNOVER, "baselines": {"atr": UNIVERSE[:5]}}


def oracle(days: int = 12) -> dict[str, list[bool]]:
    """slice -> hit flags of its picks, computed from the scenario without any of the code under test."""
    sectors = ("BANKS", "IT", "AUTO", "PHARMA")
    out: dict[str, list[bool]] = {}
    for i in range(days):
        labels_of_day = ["regime:bull" if i < 8 else "regime:bear", "regime:high-vol" if i < 8 else "regime:crisis", "regime:risk-on",
                         "regime:expiry-day" if i == 11 else "regime:non-expiry-day"]
        events = {0: "results" if i % 2 == 0 else None, 1: "ipo" if i < 4 else None, 2: "block deal" if i < 5 else None}
        for k, symbol in enumerate(SLICE_PICKS):
            j = UNIVERSE.index(symbol)
            bucket = "low" if j < 4 else "mid" if j < 8 else "high"
            labels = [f"rank:{k + 1}", f"tier:{TIERS[k]}", f"sector:{sectors[j % 4]}", f"liquidity:{bucket}", *labels_of_day]
            if events.get(k):
                labels.append(f"event:{events[k]}")
            for label in labels:
                out.setdefault(label, []).append(realised_range(i, symbol) > 3.0)
    return out


def test_slices_cover_rank_regime_sector_liquidity_event_and_tier(store, settings, universe):
    rows = record_of(store, settings, 12, slice_plan)["20d"]
    expected = {label: hits for label, hits in oracle().items() if len(hits) >= lr.MIN_SLICE_N}
    got = {r["slice"]: r for r in rows if r["name"] == "topn_hit_rate" and r["slice"] != "all"}
    assert set(got) == set(expected) and len(got) > 20
    for label, hits in expected.items():
        assert got[label]["n"] == len(hits) and got[label]["value"] == pytest.approx(np.mean(hits)), label
    assert {"rank:1", "rank:5", "regime:bull", "regime:crisis", "sector:BANKS", "liquidity:low", "liquidity:high",
            "event:results", "tier:STRONG"} <= set(got)


def test_slices_with_fewer_than_five_picks_are_skipped_and_five_is_kept(store, settings, universe):
    slices = {r["slice"] for r in record_of(store, settings, 12, slice_plan)["20d"]}
    assert "event:ipo" not in slices  # 4 picks
    assert "event:block deal" in slices and "regime:expiry-day" in slices  # exactly 5 picks each
    assert lr.MIN_SLICE_N == 5


def test_each_slice_reports_its_range_error_coverage_and_intervals(store, settings, universe):
    rows = record_of(store, settings, 12, slice_plan)["20d"]
    one = by_name(rows, "rank:1")
    assert set(one) == {"topn_hit_rate", "pick_range_mean", "range_mae_q50", "range_coverage_80"}
    assert one["pick_range_mean"]["n"] == 12 and one["pick_range_mean"]["ci_low"] is not None  # one pick a day, 12 days
    liquid = by_name(rows, "liquidity:high")
    assert liquid["range_mae_q50"]["n"] == 24  # S10 and S08 on 12 days
    assert all(r["n"] >= lr.MIN_SLICE_N for r in rows if r["slice"] != "all")


def test_the_headline_slice_is_all_picks_and_slices_are_only_for_model_picks(store, settings, universe):
    rows = record_of(store, settings, 12, slice_plan)["20d"]
    assert by_name(rows)["topn_hit_rate"]["n"] == 60 and by_name(rows)["baseline_atr_hit_rate"]["n"] == 60
    assert not [r for r in rows if r["slice"] != "all" and r["name"].startswith("baseline_")]


# -------------------------------------------------------------------------------------------- persistence
def test_persisting_the_record_twice_never_duplicates_a_metric(store, settings, universe):
    days = seed_days(store, 12, slice_plan)
    first = ev.persist_metrics(store, ev.evaluate_day(store, settings, days[-1]))
    count = store.conn.execute("SELECT COUNT(*) AS n FROM eval_metrics").fetchone()["n"]
    assert first == count and count > 100
    assert ev.persist_metrics(store, ev.evaluate_day(store, settings, days[-1])) == first
    assert store.conn.execute("SELECT COUNT(*) AS n FROM eval_metrics").fetchone()["n"] == count
    duplicated = store.conn.execute(
        "SELECT COUNT(*) AS n FROM (SELECT 1 FROM eval_metrics GROUP BY as_of_date, scope, slice, name HAVING COUNT(*) > 1)"
    ).fetchone()["n"]
    assert duplicated == 0
    scopes = {r["scope"] for r in store.conn.execute("SELECT DISTINCT scope FROM eval_metrics").fetchall()}
    assert {"daily", "rolling", "rolling_20d"} <= scopes
    slices = {r["slice"] for r in store.conn.execute("SELECT DISTINCT slice FROM eval_metrics WHERE scope = 'rolling_20d'").fetchall()}
    assert {"all", "rank:1", "regime:bull"} <= slices


def test_persisted_rolling_numbers_include_the_window_sizes_the_health_check_reads(store, settings, universe):
    days = seed_days(store, 12, fixed_plan)
    ev.persist_metrics(store, ev.evaluate_day(store, settings, days[-1]))
    rolling = {m["name"]: m for m in store.latest_metrics("rolling")}
    assert rolling["n_days_30"]["value"] == 12.0 and rolling["n_days_30"]["n"] == 12
    assert rolling["n_days_90"]["value"] == 12.0 and rolling["hit_rate_30d"]["n"] == 12


def test_the_dashboard_reads_the_persisted_slices_with_their_sample_sizes(store, settings, universe):
    from dashboard import app as dash

    days = seed_days(store, 12, slice_plan)
    ev.persist_metrics(store, ev.evaluate_day(store, settings, days[-1]))
    scope = dash.performance_data(store)["scopes"]["rolling_20d"]
    assert scope["as_of_date"] == days[-1] and {"all", "rank:1", "regime:bull", "liquidity:high"} <= {m["slice"] for m in scope["metrics"]}
    top = next(m for m in scope["metrics"] if m["slice"] == "rank:1" and m["name"] == "topn_hit_rate")
    assert top["n"] == 12 and top["low_n"] is True and top["ci_low"] is not None  # a thin slice is flagged, never hidden


def test_the_health_check_sees_calibration_drift_once_the_window_size_is_persisted(store, settings, fake_clock):
    from automation.health import evaluate_health
    from ranking.types import EveningSummary

    summary = EveningSummary("2026-11-02", "final", 3.0, "m", rolling={"n_days_90": 25.0, "ece_90": 0.2, "brier_skill_90": 0.1})
    ev.persist_metrics(store, summary)
    assert "calibration_drift" in {p.code for p in evaluate_health(store, settings).problems}


# ------------------------------------------------------------------------------------ the evening summary
def test_the_evening_summary_carries_the_record_and_survives_a_failing_record(store, settings, universe, monkeypatch, caplog):
    days = seed_days(store, 10, fixed_plan)
    assert ev.evaluate_day(store, settings, days[-1]).live_record["20d"]
    monkeypatch.setattr(lr, "build", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.WARNING, logger="backtesting.evaluate"):
        summary = ev.evaluate_day(store, settings, days[-1])
    assert summary is not None and summary.live_record == {} and summary.n == 5  # the report itself is not lost
    assert any("live record" in r.message and r.levelno == logging.WARNING for r in caplog.records)
