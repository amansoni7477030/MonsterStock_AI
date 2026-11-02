"""Outcomes and the live record of the trade ideas: the exit rules, the cautious reading of a daily bar, and the validation gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting import ideas as bt
from backtesting.ideas import (
    MAX_GAP_PCT,
    VALIDATE_MIN_DAYS,
    VALIDATE_MIN_IDEAS,
    bracket_outcome,
    build_record,
    group_of,
    idea_outcome,
    summarize_group,
    validation_hint,
    watch_outcome,
)

LONG = dict(side="LONG", entry=100.0, stop=98.0, target1=102.0, target2=104.0)
SHORT = dict(side="SHORT", entry=100.0, stop=102.0, target1=98.0, target2=96.0)


def walk(plan, *sessions, horizon=1):
    return bracket_outcome(plan["side"], plan["entry"], plan["stop"], plan["target1"], plan["target2"], list(sessions), horizon)


def bars(*sessions, start="2026-11-03"):
    return pd.DataFrame(list(sessions), index=pd.bdate_range(start, periods=len(sessions)), columns=["open", "high", "low", "close"])


def frozen(**over):
    return {"horizon": "BTST", "side": "LONG", "symbol": "ABC", "entry": 100.0, "stop": 98.0, "target1": 102.0, "target2": 104.0,
            "est_cost_pct": 0.3, **over}


# ------------------------------------------------------------------------------------------ brackets
def test_a_session_that_reaches_the_first_target_without_the_stop_exits_there():
    out = walk(LONG, (100.5, 102.5, 100.0, 101.2))
    assert (out["status"], out["exit_price"], out["ret_pct"], out["sessions"]) == ("target1", 102.0, 2.0, 1)
    assert out["t1_hit"] is True and out["t2_hit"] is False
    assert (out["gap_pct"], out["close_pct"], out["mfe_pct"], out["mae_pct"]) == (0.5, 1.2, 2.5, 0.0)


def test_a_gap_above_the_first_target_exits_at_the_open_a_limit_order_fills_there():
    out = walk(LONG, (103.0, 103.5, 102.8, 103.2))
    assert (out["status"], out["exit_price"], out["ret_pct"]) == ("target1", 103.0, 3.0)


def test_a_session_that_touches_both_the_stop_and_the_target_counts_as_stopped():
    out = walk(LONG, (100.0, 102.5, 97.5, 101.0))
    assert (out["status"], out["exit_price"], out["ret_pct"], out["t1_hit"], out["t2_hit"]) == ("stop", 98.0, -2.0, False, False)


def test_a_gap_down_through_the_stop_exits_at_the_open_not_at_the_stop():
    out = walk(LONG, (97.0, 98.5, 96.0, 97.5))
    assert (out["status"], out["exit_price"], out["ret_pct"], out["gap_pct"]) == ("stop", 97.0, -3.0, -3.0)


def test_without_a_stop_or_target_the_position_is_left_at_the_close_of_the_horizon():
    out = walk(LONG, (100.0, 101.0, 99.5, 100.8))
    assert (out["status"], out["exit_price"], out["ret_pct"], out["t1_hit"]) == ("time", 100.8, 0.8, False)


def test_the_second_target_counts_as_reached_when_the_same_session_got_there_without_the_stop():
    out = walk(LONG, (100.0, 104.5, 100.2, 104.0))
    assert (out["status"], out["exit_price"], out["t1_hit"], out["t2_hit"]) == ("target1", 102.0, True, True)  # the exit is still the first target


def test_a_short_is_the_mirror_image():
    win = walk(SHORT, (99.5, 100.0, 97.5, 98.2))
    assert (win["status"], win["exit_price"], win["ret_pct"], win["mfe_pct"]) == ("target1", 98.0, 2.0, 2.5)
    loss = walk(SHORT, (100.0, 102.5, 99.5, 101.0))
    assert (loss["status"], loss["exit_price"], loss["ret_pct"]) == ("stop", 102.0, -2.0)
    gap = walk(SHORT, (103.0, 103.5, 102.0, 102.5))
    assert (gap["status"], gap["exit_price"], gap["ret_pct"]) == ("stop", 103.0, -3.0)
    held = walk(SHORT, (100.0, 100.8, 99.2, 99.6))
    assert (held["status"], held["ret_pct"], held["gap_pct"]) == ("time", 0.4, 0.0)


def test_a_swing_walks_session_by_session_and_ignores_what_comes_after_its_exit():
    quiet, hit, crash = (100.0, 101.0, 99.5, 100.5), (100.5, 102.4, 100.2, 102.0), (97.0, 97.5, 90.0, 91.0)
    out = walk(LONG, quiet, quiet, hit, crash, crash, horizon=5)
    assert (out["status"], out["sessions"], out["exit_price"]) == ("target1", 3, 102.0)
    stopped = walk(LONG, quiet, (100.0, 100.4, 97.9, 98.5), hit, horizon=5)
    assert (stopped["status"], stopped["sessions"], stopped["exit_price"]) == ("stop", 2, 98.0)


def test_a_swing_with_nothing_hit_is_left_at_the_fifth_close_whatever_comes_later():
    quiet = [(100.0, 101.0, 99.5, 100.0 + 0.1 * n) for n in range(1, 8)]
    out = walk(LONG, *quiet, horizon=5)
    assert (out["status"], out["sessions"], out["exit_price"]) == ("time", 5, 100.5)


def test_a_horizon_not_yet_over_is_judged_on_what_exists_so_far():
    out = walk(LONG, (100.0, 101.0, 99.5, 100.4), (100.4, 101.0, 99.6, 100.2), horizon=5)
    assert (out["status"], out["sessions"], out["exit_price"]) == ("time", 2, 100.2)


def test_the_best_and_worst_excursions_are_measured_over_the_sessions_actually_held():
    out = walk(LONG, (100.0, 101.5, 99.0, 100.0), (100.0, 101.0, 98.5, 100.0), (100.0, 109.0, 100.0, 108.0), horizon=3)
    assert (out["mfe_pct"], out["mae_pct"]) == (9.0, -1.5)


# --------------------------------------------------------------------------------------- the watchlist
WATCH = {"horizon": "WATCH", "side": "BOTH", "trigger_long": 101.0, "trigger_short": 99.0,
         "metrics": {"long_target1": 102.2, "short_target1": 97.8}}


def test_a_watch_level_that_broke_is_judged_by_where_the_session_closed_against_it():
    out = watch_outcome(WATCH, (100.2, 102.5, 99.8, 102.0))
    assert (out["status"], out["long_triggered"], out["short_triggered"], out["long_t1_hit"]) == ("long", True, False, True)
    assert out["long_close_pct"] == pytest.approx(0.99, abs=0.01) and out["range_pct"] == pytest.approx(2.69, abs=0.01)
    down = watch_outcome(WATCH, (99.5, 99.9, 97.0, 97.5))
    assert (down["status"], down["short_t1_hit"], down["short_close_pct"]) == ("short", True, pytest.approx(1.54, abs=0.01))


def test_a_session_that_broke_both_levels_or_neither_is_reported_as_such():
    assert watch_outcome(WATCH, (100.0, 101.5, 98.5, 100.0))["status"] == "both"
    quiet = watch_outcome(WATCH, (100.0, 100.5, 99.5, 100.0))
    assert quiet["status"] == "none" and quiet["long_triggered"] is False and "long_close_pct" not in quiet
    assert watch_outcome({"horizon": "WATCH", "trigger_long": 101.0}, (100.0, 102.0, 99.0, 100.0))["long_t1_hit"] is False  # no target stored


# --------------------------------------------------------------------------------- one frozen idea
def test_an_idea_is_judged_net_of_the_cost_the_report_showed():
    out = idea_outcome(frozen(), bars((100.0, 101.0, 99.5, 100.8)))
    assert (out["status"], out["ret_pct"], out["net_pct"], out["complete"]) == ("time", 0.8, 0.5, True)


def test_nothing_is_judged_without_a_complete_session_or_a_known_horizon_or_a_plan():
    assert idea_outcome(frozen(), bars()) is None
    assert idea_outcome(frozen(horizon="MONTHLY"), bars((100.0, 101.0, 99.5, 100.8))) is None
    assert idea_outcome(frozen(stop=None), bars((100.0, 101.0, 99.5, 100.8))) is None
    assert idea_outcome(frozen(entry=0.0), bars((100.0, 101.0, 99.5, 100.8))) is None
    assert idea_outcome(frozen(target1="n/a"), bars((100.0, 101.0, 99.5, 100.8))) is None


def test_a_session_without_a_full_bar_ends_what_can_be_judged_and_is_never_skipped_over():
    out = idea_outcome(frozen(horizon="SWING"), bars((100.0, 101.0, 99.5, 100.4), (np.nan, 99.0, 90.0, 91.0), (100.0, 101.0, 99.0, 100.0)))
    assert out["sessions"] == 1 and out["complete"] is False
    assert idea_outcome(frozen(), bars((np.nan, np.nan, np.nan, np.nan))) is None


def test_a_swing_is_complete_after_five_sessions_or_at_its_exit_not_before():
    quiet = (100.0, 101.0, 99.5, 100.2)
    assert idea_outcome(frozen(horizon="SWING"), bars(quiet, quiet))["complete"] is False
    assert idea_outcome(frozen(horizon="SWING"), bars(*[quiet] * 5))["complete"] is True
    assert idea_outcome(frozen(horizon="SWING"), bars(quiet, (100.0, 100.4, 97.0, 98.0)))["complete"] is True  # stopped on the 2nd


def test_a_first_open_far_from_the_reference_price_is_a_basis_change_not_an_outcome():
    out = idea_outcome(frozen(), bars((100.0 * (1 + (MAX_GAP_PCT + 5) / 100), 135.0, 128.0, 130.0)))
    assert out == {"status": "invalid", "reason": "price basis changed (split or bonus?)", "complete": True, "sessions": 0}


def test_a_watch_idea_is_judged_on_the_next_session_alone():
    out = idea_outcome(WATCH, bars((100.2, 102.5, 99.8, 102.0), (90.0, 91.0, 89.0, 90.0)))
    assert out["status"] == "long" and out["complete"] is True and out["sessions"] == 1


# ----------------------------------------------------------------------------------------- the record
def judged(day, net, *, status=None, t1=None, gap=0.1):
    return day, {"complete": True, "status": status or ("target1" if net > 0 else "stop"), "net_pct": net,
                 "t1_hit": net > 0 if t1 is None else t1, "gap_pct": gap}


def run(n_days, per_day, wins):
    days = [(pd.Timestamp("2026-08-03") + pd.Timedelta(days=d)).date().isoformat() for d in range(n_days)]
    return [judged(day, 1.0 if wins(d * per_day + k) else -0.6) for d, day in enumerate(days) for k in range(per_day)]


def test_the_group_summary_counts_the_judged_ideas_and_the_ones_still_inside_their_horizon():
    items = [judged("2026-10-01", 1.0), judged("2026-10-01", -0.5, status="stop"), judged("2026-10-02", 0.4, status="time", t1=False),
             ("2026-10-02", {"complete": False, "status": "time"}), ("2026-10-03", {"complete": True, "status": "invalid"})]
    out = summarize_group("BTST", items)
    assert (out["n"], out["days"], out["pending"], out["validated"]) == (3, 2, 1, False)
    assert out["win_rate"] == pytest.approx(2 / 3, abs=1e-4) and out["mean_net_pct"] == pytest.approx(0.3, abs=1e-3)
    assert out["t1_rate"] == pytest.approx(1 / 3, abs=1e-4) and out["stop_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert out["mean_gap_pct"] == pytest.approx(0.1) and out["win_ci"] is None  # 2 days: too few for an interval


def test_a_group_with_nothing_judged_has_only_its_counts():
    out = summarize_group("SWING", [("2026-10-01", {"complete": False, "status": "time"})])
    assert out == {"group": "SWING", "n": 0, "days": 0, "pending": 1, "validated": False}


def test_a_strong_record_over_enough_days_passes_the_gate():
    out = summarize_group("BTST", run(25, 3, lambda i: i % 4 != 0))  # 56 of 75 ideas win over 25 days, +1.0 / -0.6
    assert (out["n"], out["days"]) == (75, 25) and out["validated"] is True and validation_hint(out) == ""
    assert out["win_ci"][0] > 0.5 and out["mean_net_ci"][0] > 0.0 and out["win_rate"] == pytest.approx(56 / 75, abs=1e-3)


def test_a_coin_flip_record_never_passes_however_many_ideas_it_has():
    out = summarize_group("BTST", run(30, 4, lambda i: i % 2 == 0))
    assert out["n"] == 120 and out["validated"] is False
    assert validation_hint(out) == "the interval still includes no edge"


def test_a_good_looking_but_small_record_is_not_enough():
    few = summarize_group("BTST", run(10, 3, lambda i: i % 6 != 0))
    assert few["validated"] is False and validation_hint(few) == f"needs {VALIDATE_MIN_IDEAS} judged ideas over {VALIDATE_MIN_DAYS} days (has 30 over 10)"
    short = summarize_group("BTST", run(5, 20, lambda i: i % 6 != 0))  # 100 ideas but on 5 days: one market, not many
    assert short["validated"] is False and "has 100 over 5" in validation_hint(short)


def test_the_hint_for_a_record_that_has_the_size_but_no_interval():
    assert validation_hint({"n": 80, "days": 30, "validated": False, "win_ci": None}) == "too few days for an interval"
    assert validation_hint({"validated": True}) == "" and validation_hint({}) == f"needs {VALIDATE_MIN_IDEAS} judged ideas over {VALIDATE_MIN_DAYS} days (has 0 over 0)"


def test_the_watchlist_record_is_about_levels_not_trades():
    def watch(day, status, long_close=None, short_close=None, rng=2.0):
        out = {"complete": True, "status": status, "range_pct": rng}
        if long_close is not None:
            out["long_close_pct"] = long_close
        if short_close is not None:
            out["short_close_pct"] = short_close
        return day, out

    summary = summarize_group("WATCH", [watch("2026-10-01", "long", long_close=0.5), watch("2026-10-01", "short", short_close=-0.3, rng=3.0),
                                        watch("2026-10-02", "none"), watch("2026-10-02", "both", long_close=0.2, short_close=0.4)])
    assert summary["n"] == 4 and summary["triggered_rate"] == 0.75 and summary["mean_range_pct"] == pytest.approx(2.25)
    assert summary["n_triggers"] == 4 and summary["followed_through_rate"] == 0.75 and "win_rate" not in summary and summary["validated"] is False


def test_the_bearish_ideas_of_both_horizons_share_one_list():
    assert group_of({"horizon": "BTST", "side": "SHORT"}) == "SHORT" and group_of({"horizon": "SWING", "side": "SHORT"}) == "SHORT"
    assert group_of({"horizon": "WATCH", "side": "SHORT"}) == "WATCH" and group_of({"horizon": "SWING", "side": "LONG"}) == "SWING"
    assert group_of({}) == ""


def test_the_record_joins_the_frozen_ideas_to_their_outcomes_by_day_and_key():
    ideas = {"2026-10-01": [{"horizon": "BTST", "side": "LONG", "symbol": "AAA"}, {"horizon": "BTST", "side": "SHORT", "symbol": "BBB"},
                            {"horizon": "SWING", "side": "LONG", "symbol": "CCC"}],
             "2026-10-02": [{"horizon": "BTST", "side": "LONG", "symbol": "AAA"}]}
    outcomes = {"2026-10-01": {"BTST:LONG:AAA": judged("", 1.0)[1], "BTST:SHORT:BBB": judged("", -0.5)[1]}, "2026-10-09": {"x": {}}}
    record = build_record(outcomes, ideas)
    assert set(record) == {"BTST", "SHORT"}  # the swing idea and the second day's idea have no outcome yet; nothing is invented
    assert record["BTST"]["n"] == 1 and record["SHORT"]["n"] == 1 and record["BTST"]["days"] == 1
    assert build_record({}, ideas) == {} and build_record(outcomes, {}) == {}


def test_every_group_of_the_record_is_one_of_the_four_lists():
    assert bt.GROUPS == ("BTST", "SWING", "SHORT", "WATCH") and set(bt.HORIZON_SESSIONS) == {"BTST", "SWING", "WATCH"}
