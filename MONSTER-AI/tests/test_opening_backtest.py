"""Outcomes of the 09:30 opening read and the live record (``backtesting.opening``): hand-computed cases for every rule, so the record
that stands behind the model can be trusted to say what happened and nothing else."""

from __future__ import annotations

import pytest

from backtesting import opening as bt

IDEA = {
    "symbol": "AAA", "kind": "PICK", "open": 100.0, "or_high": 101.0, "or_low": 99.5, "ltp": 100.4, "p3": 0.62, "est_cost_pct": 0.2,
    "day_range": [1.4, 2.1, 3.4], "ahead": [1.0, 2.0, 3.5],
    "levels": {"trigger_long": 101.0, "long_stop": 100.5, "long_target1": 101.6, "long_target2": 102.1,
               "trigger_short": 99.0, "short_stop": 99.5, "short_target1": 98.4, "short_target2": 97.9},
}


def bar5(hhmm, o, h, low, c):
    return {"hhmm": hhmm, "open": o, "high": h, "low": low, "close": c}


# ------------------------------------------------------------------------------------------------ the day's bar
def test_day_outcome_reports_the_range_the_breaks_and_the_band():
    got = bt.day_outcome(IDEA, {"open": 100.2, "high": 105.0, "low": 99.0, "close": 103.0})
    assert got["status"] == "judged" and got["complete"] is True
    assert got["range_pct"] == pytest.approx(5.988, abs=0.001) and got["hit3"] is True and got["hit5"] is True
    assert got["close_pct"] == pytest.approx(2.794, abs=0.001) and got["after_scan_pct"] == pytest.approx(2.590, abs=0.001)
    assert got["broke_up"] is True and got["broke_down"] is True
    assert got["ext_up_pct"] == pytest.approx(3.984, abs=0.001) and got["ext_down_pct"] == pytest.approx(0.498, abs=0.001)
    assert got["in_band"] is False  # 5.99 % is outside the stated 80 % band of 1.4-3.4 %


def test_a_range_inside_the_band_that_never_left_the_opening_range_is_recorded_as_such():
    got = bt.day_outcome(IDEA, {"open": 100.0, "high": 100.9, "low": 99.6, "close": 100.3})
    assert got["hit3"] is False and got["hit5"] is False and got["broke_up"] is False and got["broke_down"] is False
    assert got["ext_up_pct"] == 0.0 and got["ext_down_pct"] == 0.0 and got["in_band"] is False  # 1.3 % is below the band's 1.4 % floor
    assert bt.day_outcome(IDEA, {"open": 100.0, "high": 101.7, "low": 99.6, "close": 100.3})["in_band"] is True


def test_the_three_percent_line_is_strict():
    assert bt.day_outcome(IDEA, {"open": 100.0, "high": 103.1, "low": 100.0, "close": 101})["hit3"] is True
    assert bt.day_outcome(IDEA, {"open": 100.0, "high": 102.9, "low": 100.0, "close": 101})["hit3"] is False


def test_a_name_whose_range_had_already_passed_3_percent_at_the_scan_is_flagged():
    bar = {"open": 100.2, "high": 105.0, "low": 99.0, "close": 103.0}
    assert bt.day_outcome({**IDEA, "metrics": {"or_raw_pct": 3.4}}, bar)["decided3"] is True
    assert bt.day_outcome({**IDEA, "metrics": {"or_raw_pct": 3.0}}, bar)["decided3"] is False  # exactly 3 % has not passed 3 %
    assert bt.day_outcome({**IDEA, "metrics": {"or_raw_pct": 2.2}}, bar)["decided3"] is False
    assert bt.day_outcome({**IDEA, "metrics": {"clv": 0.9}, "or_pct": 3.5}, bar)["decided3"] is True  # no raw range stored: the normalised one stands in
    assert bt.day_outcome(IDEA, bar)["decided3"] is False  # unknown: the stated chance is the model's own claim, so it is judged


def test_a_split_or_bonus_is_not_an_outcome():
    got = bt.day_outcome(IDEA, {"open": 50.0, "high": 52.0, "low": 49.0, "close": 51.0})
    assert got == {"status": "invalid", "reason": "price basis changed (split or bonus?)", "complete": True}


@pytest.mark.parametrize("bar", [None, {}, {"open": 100, "high": float("nan"), "low": 99, "close": 100}, {"open": 100, "high": 98, "low": 99, "close": 99},
                                 {"open": 0, "high": 1, "low": 0.5, "close": 1}, {"open": 100, "high": 101, "low": 99}])
def test_without_a_usable_bar_there_is_no_outcome(bar):
    assert bt.day_outcome(IDEA, bar) is None


def test_a_frozen_name_with_missing_figures_has_no_outcome():
    assert bt.day_outcome({**IDEA, "ltp": None}, {"open": 100, "high": 101, "low": 99, "close": 100}) is None
    assert "in_band" not in bt.day_outcome({**IDEA, "day_range": []}, {"open": 100, "high": 101, "low": 99, "close": 100})


# ------------------------------------------------------------------------------------------------ the range ahead
def test_the_range_ahead_counts_only_bars_from_the_scan_to_the_square_off():
    bars = [bar5("09:25", 100, 109, 91, 100), bar5("09:30", 100, 101.0, 99.6, 100.5), bar5("12:00", 100.5, 102.5, 100.2, 102.0),
            bar5("15:10", 102, 102.2, 101.5, 101.8), bar5("15:15", 101.8, 120, 80, 101)]
    got = bt.ahead_outcome(IDEA, bars, "09:30")
    assert got == {"ahead_pct": pytest.approx(2.888, abs=0.001), "ahead_bars": 3, "ahead_in_band": True}  # 2.9 on the price of 100.4 at the scan
    assert bt.ahead_outcome(IDEA, [], "09:30") is None and bt.ahead_outcome({**IDEA, "ltp": 0}, bars, "09:30") is None
    assert bt.ahead_outcome(IDEA, [{"hhmm": "09:30", "open": None, "high": 1, "low": 1, "close": 1}], "09:30") is None


# ------------------------------------------------------------------------------------------------ the levels as a plan
def test_a_long_break_that_reaches_the_first_target():
    bars = [bar5("09:30", 100.2, 100.8, 100.0, 100.6), bar5("09:35", 100.6, 101.2, 100.7, 101.1), bar5("09:40", 101.1, 101.8, 101.0, 101.5)]
    got = bt.plan_outcome(IDEA, bars, "09:30")
    assert got["status"] == "target1" and got["side"] == "LONG" and got["t1_hit"] is True and got["entry"] == 101.0 and got["exit"] == 101.6
    assert got["ret_pct"] == pytest.approx(0.594, abs=0.001) and got["net_pct"] == pytest.approx(0.394, abs=0.001)


def test_a_bar_that_reaches_both_the_stop_and_the_target_counts_as_stopped():
    bars = [bar5("09:30", 100.2, 100.8, 100.0, 100.6), bar5("09:35", 100.6, 101.9, 100.4, 101.0)]
    got = bt.plan_outcome(IDEA, bars, "09:30")  # triggers at 101, then the same bar trades through 100.5 and 101.6: the order is unknown
    assert got["status"] == "stop" and got["exit"] == 100.5 and got["t1_hit"] is False
    assert got["ret_pct"] == pytest.approx(-0.495, abs=0.001) and got["net_pct"] == pytest.approx(-0.695, abs=0.001)


def test_a_gap_through_the_trigger_fills_at_the_open_not_at_the_trigger():
    bars = [bar5("09:30", 101.5, 102.0, 101.4, 101.9), bar5("09:35", 101.9, 102.4, 101.8, 102.3)]
    got = bt.plan_outcome(IDEA, bars, "09:30")
    assert got["entry"] == 101.5 and got["status"] == "target1" and got["exit"] == 101.6  # the levels are the report's; the fill is the open's


def test_a_later_gap_through_the_stop_exits_at_the_open():
    bars = [bar5("09:30", 100.6, 101.2, 100.7, 101.1), bar5("09:35", 100.2, 100.4, 99.9, 100.0)]
    got = bt.plan_outcome(IDEA, bars, "09:30")
    assert got["status"] == "stop" and got["exit"] == 100.2 and got["ret_pct"] == pytest.approx(-0.792, abs=0.001)


def test_a_short_break_that_reaches_the_first_target():
    bars = [bar5("09:30", 99.6, 99.8, 99.3, 99.4), bar5("09:35", 99.3, 99.4, 98.9, 99.0), bar5("09:40", 99.0, 99.1, 98.2, 98.5)]
    got = bt.plan_outcome(IDEA, bars, "09:30")
    assert got["side"] == "SHORT" and got["status"] == "target1" and got["entry"] == 99.0 and got["exit"] == 98.4
    assert got["ret_pct"] == pytest.approx(0.606, abs=0.001) and got["net_pct"] == pytest.approx(0.406, abs=0.001)


def test_a_short_that_is_stopped_and_a_short_that_gaps_through_its_trigger():
    stopped = bt.plan_outcome(IDEA, [bar5("09:30", 99.4, 99.5, 98.9, 99.1), bar5("09:35", 99.1, 99.6, 99.0, 99.5)], "09:30")
    assert stopped["status"] == "stop" and stopped["exit"] == 99.5 and stopped["ret_pct"] == pytest.approx(-0.505, abs=0.001)
    gapped = bt.plan_outcome(IDEA, [bar5("09:30", 98.5, 98.8, 98.3, 98.6)], "09:30")
    assert gapped["entry"] == 98.5 and gapped["status"] == "target1" and gapped["exit"] == pytest.approx(98.4)


def test_the_first_side_to_trigger_wins_and_the_other_is_cancelled():
    bars = [bar5("09:30", 100.0, 100.5, 98.9, 99.2), bar5("09:35", 99.2, 101.3, 99.1, 101.2)]
    got = bt.plan_outcome(IDEA, bars, "09:30")
    assert got["side"] == "SHORT"  # 98.9 trades through 99.0 in the first bar; the later break above 101 is not a second trade


def test_both_triggers_in_one_bar_cannot_be_ordered_and_are_left_out_of_the_record():
    assert bt.plan_outcome(IDEA, [bar5("09:30", 100.0, 101.5, 98.5, 100.0)], "09:30") == {"status": "both", "complete": True}


def test_no_break_means_no_trade_and_a_trade_that_runs_out_of_day_closes_at_the_last_bar_before_the_square_off():
    quiet = bt.plan_outcome(IDEA, [bar5("09:30", 100.0, 100.9, 99.2, 100.3), bar5("15:10", 100.3, 100.8, 99.5, 100.1)], "09:30")
    assert quiet == {"status": "none", "complete": True}
    drift = [bar5("09:30", 100.6, 101.1, 100.7, 101.0), bar5("12:00", 101.0, 101.5, 100.6, 101.2), bar5("15:10", 101.2, 101.3, 100.9, 101.1),
             bar5("15:15", 101.1, 90.0, 80.0, 85.0)]
    got = bt.plan_outcome(IDEA, drift, "09:30")
    assert got["status"] == "time" and got["exit"] == 101.1 and got["t1_hit"] is False  # the 15:15 bar is after the square-off


def test_bars_before_the_scan_cannot_trigger_the_plan():
    bars = [bar5("09:25", 100.0, 102.0, 99.9, 101.8), bar5("09:30", 101.0, 101.3, 100.8, 101.1)]
    assert bt.plan_outcome(IDEA, bars, "09:30")["entry"] == 101.0 and bt.plan_outcome(IDEA, bars[:1], "09:30") is None
    assert bt.plan_outcome(IDEA, bars, "09:35") is None  # nothing left to judge from 09:35


def test_without_levels_or_a_cost_the_plan_is_still_judged_or_declined():
    assert bt.plan_outcome({**IDEA, "levels": {}}, [bar5("09:30", 100, 101, 99, 100)], "09:30") is None
    assert bt.plan_outcome({**IDEA, "levels": {"trigger_long": 101.0}}, [bar5("09:30", 100, 101, 99, 100)], "09:30") is None
    free = bt.plan_outcome({**IDEA, "est_cost_pct": None}, [bar5("09:30", 100.6, 101.2, 100.7, 101.1), bar5("09:35", 101.1, 101.8, 101.0, 101.5)], "09:30")
    assert free["net_pct"] == free["ret_pct"]


# ------------------------------------------------------------------------------------------------ the whole table
def state_row(i, p3, *, q10=0.5, q90=3.5, symbol=None):
    return {"symbol": symbol or f"S{i}", "open": 100.0, "p3": p3, "p5": p3 / 4, "day_q10": q10, "day_q90": q90}


def day_bars(rng_of):
    return {f"S{i}": {"open": 100.0, "high": 100.0 + r, "low": 100.0, "close": 100.0} for i, r in enumerate(rng_of)}


def test_the_model_is_judged_on_every_name_it_scored():
    rows = [state_row(i, (i + 0.5) / 30) for i in range(30)]
    bars = day_bars([1.0] * 20 + [4.0] * 10)  # the ten names with the highest stated chances are the ones that made a wide day
    got = bt.universe_outcome(rows, bars)
    assert got["n"] == 30 and got["hit3"] == pytest.approx(1 / 3, abs=1e-4) and got["mean_p3"] == pytest.approx(0.5, abs=1e-4)
    assert got["top_n"] == 6 and got["top_hit3"] == 1.0 and got["top_mean_range"] == 4.0 and got["mean_range"] == pytest.approx(2.0)
    assert got["band_cover"] == pytest.approx(20 / 30, abs=1e-4)  # 1 % is inside 0.5-3.5 %, 4 % is not
    assert sum(b[0] for b in got["calib"]) == 30 and sum(b[2] for b in got["calib"]) == 10 and len(got["calib"]) == 10
    assert sum(b[1] for b in got["calib"]) == pytest.approx(15.0, abs=0.01)


def test_names_without_a_bar_or_with_a_changed_basis_are_not_judged_and_too_few_names_make_no_statement():
    rows = [state_row(i, 0.3) for i in range(30)]
    bars = day_bars([2.0] * 30)
    del bars["S0"]
    bars["S1"]["open"] = 150.0  # a split
    assert bt.universe_outcome(rows, bars)["n"] == 28
    assert bt.universe_outcome(rows[:19], day_bars([2.0] * 19)) is None
    assert bt.universe_outcome([], {}) is None


def test_the_top_names_are_taken_by_stated_chance_with_ties_broken_by_position():
    rows = [state_row(i, 0.5) for i in range(25)]
    got = bt.universe_outcome(rows, day_bars([4.0] * 6 + [1.0] * 19))
    assert got["top_hit3"] == 1.0  # a stable sort: the first six rows are the top six when every chance is equal


def test_names_already_past_3_percent_are_counted_apart_and_never_test_the_model():
    rows = [state_row(i, (i + 0.5) / 30) for i in range(30)]
    for row in rows[25:]:
        row.update(or_raw_pct=3.4, p3=1.0)  # the five highest-rated names had already printed a range above 3 % when the scan read them
    for row in rows[:25]:
        row["or_raw_pct"] = 1.2
    bars = day_bars([1.0] * 20 + [4.0] * 10)  # ... and made one, as they had to
    got = bt.universe_outcome(rows, bars)
    assert got["n"] == 25 and got["n_decided"] == 5 and got["hit3"] == pytest.approx(5 / 25) and got["mean_p3"] == pytest.approx(0.41667, abs=1e-4)
    assert got["top_hit3"] == pytest.approx(5 / 6, abs=1e-4)  # the top six of the others (not the trivial 100 % of the names that had already passed it)
    assert sum(b[0] for b in got["calib"]) == 25 and sum(b[2] for b in got["calib"]) == 5
    assert got["band_n"] == 30 and got["band_cover"] == pytest.approx(20 / 30, abs=1e-4)  # the band is judged on every name
    assert bt.universe_outcome(rows[:15] + [{**r, "or_raw_pct": 3.4} for r in rows[15:]], bars) is None  # fewer than 20 undecided names: no statement


def test_the_normalised_range_stands_in_when_a_row_has_no_raw_range_and_exactly_3_percent_has_not_passed():
    rows = [{**state_row(i, 0.3), "or_pct": 3.0} for i in range(25)]
    got = bt.universe_outcome(rows, day_bars([2.0] * 25))
    assert got["n"] == 25 and got["n_decided"] == 0
    rows = [{**state_row(i, 0.3), "or_pct": 3.2} for i in range(25)]
    assert bt.universe_outcome(rows, day_bars([4.0] * 25)) is None  # all past 3 %: nothing to judge


# ------------------------------------------------------------------------------------------------ the record
def universe_day(i, *, hit3=0.27, mean_p3=0.27, top_hit3=0.9, n=100):
    wobble = ((i % 5) - 2) * 0.005
    return {"n": n, "hit3": hit3 + wobble, "mean_p3": mean_p3, "top_n": 6, "top_hit3": top_hit3 - abs(wobble), "top_mean_range": 5.0,
            "mean_range": 2.5, "band_cover": 0.8, "hit5": 0.05, "mean_p5": 0.05, "calib": []}


def days_of(count, **kw):
    return {f"2026-10-{d:02d}": universe_day(d, **kw) for d in range(1, count + 1)}


def test_a_calibrated_model_whose_top_names_beat_the_base_rate_passes_the_gate_after_enough_sessions():
    got = bt.summarize_universe(days_of(25))
    assert got["days"] == 25 and got["n"] == 2500 and got["validated"] is True and bt.validation_hint(got) == ""
    assert abs(got["gap"]) < 0.01 and got["top_hit"] > 0.85 and got["top_ci"][0] > got["hit3"] and got["band_cover"] == pytest.approx(0.8)


def test_too_few_sessions_miscalibration_or_no_edge_each_fail_the_gate_with_the_reason():
    short = bt.summarize_universe(days_of(10))
    assert short["validated"] is False and bt.validation_hint(short) == "needs 20 judged sessions (has 10)"
    off = bt.summarize_universe(days_of(25, hit3=0.40))
    assert off["validated"] is False and bt.validation_hint(off) == "the stated chance is off by 13 points"
    flat = bt.summarize_universe(days_of(25, top_hit3=0.27))
    assert flat["validated"] is False and bt.validation_hint(flat) == "its top names do not yet clearly beat the day's base rate"
    assert bt.summarize_universe({}) == {"days": 0, "n": 0, "validated": False} and bt.validation_hint({}) == "needs 20 judged sessions (has 0)"


def test_the_pooled_record_counts_decided_names_apart_and_weights_the_band_by_the_names_it_judged():
    a = {**universe_day(1, n=100), "n_decided": 4, "band_cover": 0.9, "band_n": 104}
    b = {**universe_day(2, n=50), "n_decided": 6, "band_cover": 0.5, "band_n": 56}
    got = bt.summarize_universe({"2026-10-01": a, "2026-10-02": b})
    assert got["n"] == 150 and got["n_decided"] == 10 and got["band_cover"] == pytest.approx((0.9 * 104 + 0.5 * 56) / 160, abs=1e-4)
    assert bt.summarize_universe({"2026-10-01": universe_day(1)})["n_decided"] == 0  # an older day without the count


def judged(range_pct=4.0, p3=0.6):
    return {"status": "judged", "complete": True, "range_pct": range_pct, "hit3": range_pct > 3.0, "broke_up": True, "broke_down": False,
            "in_band": True}


def test_a_lists_chance_is_judged_only_on_names_that_had_not_passed_3_percent():
    items = [("2026-10-01", {"kind": "MOVER", "symbol": "A", "p3": 0.6}, {**judged(4.0), "decided3": False}),
             ("2026-10-01", {"kind": "MOVER", "symbol": "B", "p3": 0.4}, {**judged(2.0), "decided3": False}),
             ("2026-10-01", {"kind": "MOVER", "symbol": "C", "p3": 1.0}, {**judged(5.0), "decided3": True})]
    got = bt.summarize_listed("MOVER", items)
    assert got["n"] == 3 and got["n_open"] == 2 and got["n_decided"] == 1
    assert got["hit3"] == 0.5 and got["mean_p3"] == pytest.approx(0.5) and got["mean_range_pct"] == pytest.approx(3.667, abs=0.001)
    only = bt.summarize_listed("MOVER", items[2:])
    assert only["n"] == 1 and only["n_open"] == 0 and only["n_decided"] == 1 and only["hit3"] is None and only["mean_p3"] is None
    assert only["mean_range_pct"] == 5.0 and bt.summarize_listed("MOVER", []) == {"kind": "MOVER", "n": 0, "days": 0, "pending": 0, "n_open": 0, "n_decided": 0}


def test_the_record_pools_the_lists_the_model_and_the_plan_and_leaves_provisional_days_waiting():
    ideas = {
        "2026-10-01": [{"kind": "PICK", "symbol": "A", "p3": 0.6}, {"kind": "MOVER", "symbol": "B", "p3": 0.7}],
        "2026-10-02": [{"kind": "PICK", "symbol": "A", "p3": 0.5}],
        "2026-10-03": [{"kind": "MOVER", "symbol": "C", "p3": 0.7}],  # provisional bar: not final yet
    }
    plan_win = {"status": "target1", "t1_hit": True, "net_pct": 0.4, "complete": True}
    plan_loss = {"status": "stop", "t1_hit": False, "net_pct": -0.7, "complete": True}
    outcomes = {
        "2026-10-01": {"official": True, "universe": universe_day(1), "ideas": {"PICK:A": {**judged(5.0), "plan": plan_win, "ahead_pct": 2.0, "ahead_in_band": True},
                                                                              "MOVER:B": {**judged(2.0), "plan": plan_loss}}},
        "2026-10-02": {"official": True, "universe": universe_day(2), "ideas": {"PICK:A": judged(1.0)}},
        "2026-10-03": {"official": False, "universe": universe_day(3), "ideas": {"MOVER:C": judged(4.0)}},
    }
    record = bt.build_record(outcomes, ideas)
    assert record["model"]["days"] == 2 and record["model"]["n"] == 200
    assert record["PICK"]["n"] == 2 and record["PICK"]["hit3"] == 0.5 and record["PICK"]["mean_p3"] == pytest.approx(0.55)
    assert record["PICK"]["n_ahead"] == 1 and record["PICK"]["ahead_cover"] == 1.0 and record["PICK"]["broke_up_rate"] == 1.0
    assert record["MOVER"]["n"] == 1 and record["MOVER"]["pending"] == 1  # the provisional day's mover waits for the official bar
    assert record["plan"]["n"] == 2 and record["plan"]["judged"] == 2 and record["plan"]["t1_rate"] == 0.5 and record["plan"]["stop_rate"] == 0.5
    assert record["plan"]["mean_net_pct"] == pytest.approx(-0.15) and record["plan"]["triggered_rate"] == 1.0


def test_plan_outcomes_that_never_traded_are_counted_as_judged_but_not_as_trades():
    items = [("2026-10-01", {"status": "none", "complete": True}), ("2026-10-01", {"status": "both", "complete": True}),
             ("2026-10-02", {"status": "target1", "t1_hit": True, "net_pct": 0.3, "complete": True})]
    got = bt.summarize_plan(items)
    assert got["judged"] == 3 and got["n"] == 1 and got["triggered_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert bt.summarize_plan([("2026-10-01", {"status": "none", "complete": True})]) == {"judged": 1, "n": 0, "days": 0}
    assert bt.summarize_plan([]) == {"judged": 0, "n": 0, "days": 0}


def test_an_empty_record_has_only_the_empty_model_entry():
    assert bt.build_record({}, {}) == {"model": {"days": 0, "n": 0, "validated": False}}
    only_invalid = bt.build_record({"2026-10-01": {"official": True, "ideas": {"PICK:A": {"status": "invalid", "complete": True}}}},
                                   {"2026-10-01": [{"kind": "PICK", "symbol": "A", "p3": 0.5}]})
    assert only_invalid["PICK"]["n"] == 0
