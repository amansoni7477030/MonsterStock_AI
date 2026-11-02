"""The closing-hour idea engine: setups, gates, mechanical ATR levels, the market regime and the honest-by-construction rules."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ranking import ideas as engine
from ranking.costs import CostModel, estimate_delivery_round_trip_cost_pct, estimate_round_trip_cost_pct
from ranking.ideas import BTST_PLAN, SWING_PLAN, plan_levels, rank_ideas, watch_levels
from tests.ideas_data import (
    BTST_LONG,
    BTST_SHORT,
    CALM,
    FILLER,
    SWING_LONG,
    WATCH_ONLY,
    catalyst,
    config,
    make_table,
    with_fillers,
)


def keys(ideas):
    return {i.key: i for i in ideas}


def listed(rows, market=CALM, cfg=None, **kw):
    return keys(rank_ideas(with_fillers(rows), market, cfg or config(), **kw))


# ------------------------------------------------------------------------------------------- the lists
def test_a_strong_close_on_volume_is_a_btst_long_with_mechanical_atr_levels():
    idea = listed({"UP": BTST_LONG})["BTST:LONG:UP"]
    assert (idea.rank, idea.grade, idea.horizon, idea.side, idea.is_fno) == (1, "A", "BTST", "LONG", True)
    assert (idea.entry, idea.max_entry, idea.stop, idea.target1, idea.target2) == (1000.0, 1003.0, 983.0, 1020.0, 1036.0)
    assert idea.risk_pct == pytest.approx(1.7) and idea.reward1_pct == pytest.approx(2.0) and idea.reward2_pct == pytest.approx(3.6)
    assert idea.rr1 == pytest.approx(1.18, abs=0.01) and idea.rr2 == pytest.approx(2.12, abs=0.01)
    assert idea.est_cost_pct == pytest.approx(0.37, abs=0.01) and idea.turnover_cr == 100.0
    assert idea.setup == "closing strength with a breakout" and idea.hold == BTST_PLAN.hold
    assert "closing near the day's high" in idea.reasons[0] and idea.metrics["clv"] == 0.92 and idea.cautions == []


def test_the_score_is_a_ranking_aid_between_zero_and_a_hundred():
    ideas = rank_ideas(with_fillers({"UP": BTST_LONG, "PULL": SWING_LONG}), CALM, config())
    assert ideas and all(0.0 <= i.score <= 100.0 and i.grade in ("A", "B", "C") for i in ideas)
    assert all("probab" not in " ".join([i.setup, *i.reasons, *i.cautions]).lower() for i in ideas)


def test_the_mirror_image_is_a_btst_short_for_an_fno_name():
    idea = listed({"DOWN": BTST_SHORT})["BTST:SHORT:DOWN"]
    assert (idea.entry, idea.max_entry, idea.stop, idea.target1, idea.target2) == (1000.0, 997.0, 1017.0, 980.0, 964.0)
    assert idea.setup == "closing weakness with a breakdown" and "closing near the day's low" in idea.reasons[0]
    assert "market regime is bullish: this side is discounted" in idea.cautions  # CALM says bullish: a short fights it


def test_a_cash_market_name_is_never_a_short_idea_and_shorts_can_be_switched_off():
    cash = listed({"DOWN": {**BTST_SHORT, "is_fno": False}})
    assert not any(key.split(":")[1] == "SHORT" and horizon != "WATCH" for key in cash for horizon in [key.split(":")[0]])
    assert "BTST:SHORT:DOWN" in listed({"DOWN": BTST_SHORT})
    off = listed({"DOWN": BTST_SHORT}, cfg=config(include_shorts=False))
    assert not any(i.side == "SHORT" for i in off.values())


def test_a_pullback_in_an_uptrend_is_a_swing_long_with_a_wider_stop_and_the_swing_hold():
    idea = listed({"PULL": SWING_LONG})["SWING:LONG:PULL"]
    assert idea.setup == "pullback to the 20-day average in an uptrend" and idea.hold == SWING_PLAN.hold
    assert (idea.stop, idea.target1, idea.target2) == (971.0, 1040.0, 1072.0)  # the week's low less a buffer, 2 and 3.6 ATR
    assert idea.risk_pct == pytest.approx(2.9) and idea.rr2 == pytest.approx(2.48, abs=0.01)


def test_a_symbol_is_listed_once_and_the_overnight_list_wins():
    ideas = rank_ideas(with_fillers({"UP": BTST_LONG}), CALM, config())
    assert [i.symbol for i in ideas].count("UP") == 1 and ideas[0].horizon == "BTST"  # it would also pass the swing gate


def test_the_watchlist_has_both_triggers_and_a_one_sided_lean_carries_its_levels():
    found = listed({"COIL": WATCH_ONLY})
    assert set(found) == {"WATCH:LONG:COIL"}
    idea = found["WATCH:LONG:COIL"]
    assert (idea.trigger_long, idea.trigger_short) == (1007.0, None)
    assert (idea.stop, idea.target1, idea.target2) == (997.0, 1019.0, 1029.0) and idea.entry is None
    assert idea.metrics["short_stop"] == 1005.0 and idea.metrics["trigger_short"] == 995.0
    assert idea.setup == "tight range below the 20-day high" and idea.hold == engine.WATCH_PLAN.hold
    assert any("coiled" in r for r in idea.reasons) and idea.cautions[-1].startswith("a gap at the open")


@pytest.mark.parametrize(("over", "side", "long_trigger", "short_trigger"), [
    ({"ema20": 1005.0, "ema50": 1040.0, "clv": 0.4}, "SHORT", None, 995.0),   # both averages down, weak close
    ({"clv": 0.4}, "BOTH", 1007.0, 995.0),                                      # uptrend but a weak close: no lean
    ({"ema20": 990.0, "ema50": 1040.0, "clv": 0.6}, "BOTH", 1007.0, 995.0),    # mixed averages
])
def test_the_watchlist_leans_with_the_trend_and_the_close_or_stays_two_sided(over, side, long_trigger, short_trigger):
    idea = listed({"COIL": {**WATCH_ONLY, **over}})[f"WATCH:{side}:COIL"]
    assert (idea.trigger_long, idea.trigger_short) == (long_trigger, short_trigger)
    if side == "BOTH":
        assert idea.stop is None and idea.target1 is None  # two sets of levels: they live in the metrics


def test_a_long_only_user_gets_long_watch_triggers_even_against_the_trend():
    idea = listed({"COIL": {**WATCH_ONLY, "ema20": 1005.0, "ema50": 1040.0, "clv": 0.4}}, cfg=config(include_shorts=False))
    assert set(idea) == {"WATCH:LONG:COIL"} and idea["WATCH:LONG:COIL"].trigger_short is None


@pytest.mark.parametrize("over", [{"atr_pct": 1.1}, {"range_comp": 1.3, "dist_hi20": -8.0, "dist_lo20": 9.0, "rvol": 0.8}])
def test_a_quiet_or_unremarkable_name_is_not_an_intraday_watch(over):
    assert listed({"COIL": {**WATCH_ONLY, **over}}) == {}


def test_ranks_restart_in_every_list_and_follow_the_score_within_it():
    rows = {f"L{i}": {**BTST_LONG, "clv": 0.72 + 0.03 * i, "rvol": 1.2 + 0.1 * i} for i in range(8)}
    ideas = rank_ideas(with_fillers(rows), CALM, config(top_n=3))
    btst = [i for i in ideas if i.horizon == "BTST"]
    assert [i.rank for i in btst] == [1, 2, 3] and [i.symbol for i in btst] == ["L7", "L6", "L5"]
    assert [i.score for i in btst] == sorted((i.score for i in btst), reverse=True)


def test_the_pool_is_three_times_the_list_for_the_news_shortlist():
    rows = {f"L{i}": {**BTST_LONG, "clv": 0.72 + 0.02 * i, "rvol": 1.2 + 0.05 * i} for i in range(12)}
    table = with_fillers(rows)
    assert len([i for i in rank_ideas(table, CALM, config(top_n=2)) if i.horizon == "BTST"]) == 2
    assert len([i for i in rank_ideas(table, CALM, config(top_n=2), pool=True) if i.horizon == "BTST"]) == 6


def test_bearish_ideas_come_after_swing_and_before_the_watchlist_in_report_order():
    ideas = rank_ideas(with_fillers({"UP": BTST_LONG, "PULL": SWING_LONG, "DOWN": BTST_SHORT, "COIL": WATCH_ONLY}), CALM, config())
    assert [i.key for i in ideas] == ["BTST:LONG:UP", "SWING:LONG:PULL", "BTST:SHORT:DOWN", "WATCH:LONG:COIL"]


# --------------------------------------------------------------------------------------------- gates
@pytest.mark.parametrize("over", [
    {"rvol": 0.8}, {"ret1": 8.0}, {"ret1": 0.1}, {"clv": 0.6}, {"rsi14": 85.0}, {"dist_ema20_atr": 4.0},
    {"ltp": 940.0, "breakout20": 0.0},
])
def test_a_btst_long_must_pass_every_gate(over):
    assert "BTST:LONG:UP" in listed({"UP": BTST_LONG})
    assert "BTST:LONG:UP" not in listed({"UP": {**BTST_LONG, **over}})


@pytest.mark.parametrize("over", [{"rvol": 0.8}, {"ret1": -8.0}, {"ret1": -0.1}, {"clv": 0.4}, {"rsi14": 15.0}, {"dist_ema20_atr": -4.0}])
def test_a_btst_short_must_pass_every_gate(over):
    assert "BTST:SHORT:DOWN" not in listed({"DOWN": {**BTST_SHORT, **over}})


@pytest.mark.parametrize("over", [
    {"tv_med20_cr": 5.0}, {"ltp": 40.0}, {"hist_days": 50.0}, {"excluded": "price basis differs"}, {"atr_abs": np.nan}, {"atr_abs": 0.0},
])
def test_an_illiquid_cheap_young_or_excluded_name_is_never_listed(over):
    assert not [k for k in listed({"UP": {**BTST_LONG, **over}}) if k.endswith(":UP")]


def test_eligibility_reads_the_exclusion_marks_of_the_scan():
    table = make_table({"OK": BTST_LONG, "OFF": {**BTST_LONG, "excluded": "no stored previous close"}, "THIN": {**BTST_LONG, "tv_med20_cr": 3.0}})
    assert engine.eligibility(table, config()).to_dict() == {"OK": True, "OFF": False, "THIN": False}
    assert engine.eligibility(table.drop(columns="excluded"), config()).to_dict() == {"OK": True, "OFF": True, "THIN": False}


def test_a_wide_stop_or_a_thin_reward_to_risk_or_a_high_cost_keeps_an_idea_off_the_list():
    assert "BTST:LONG:UP" in listed({"UP": BTST_LONG}, cfg=config(max_risk_pct=2.0, min_rr=2.0))
    assert "BTST:LONG:UP" not in listed({"UP": BTST_LONG}, cfg=config(max_risk_pct=1.5))  # risk is 1.7%
    assert "BTST:LONG:UP" not in listed({"UP": BTST_LONG}, cfg=config(min_rr=2.5))  # reward to the second target is 2.1x the risk
    assert "BTST:LONG:UP" not in listed({"UP": {**BTST_LONG, "tv_med20_cr": 26.0}})  # costs about 0.53% against a 2% first target


def test_nothing_is_listed_below_the_minimum_score():
    assert rank_ideas(with_fillers({"UP": BTST_LONG}), CALM, config(min_score=99.0)) == []


@pytest.mark.parametrize(("score", "grade"), [(100.0, "A"), (78.0, "A"), (77.9, "B"), (68.0, "B"), (67.9, "C"), (0.0, "C")])
def test_the_grade_bands_of_the_score(score, grade):
    assert engine.grade_for(score) == grade


# ------------------------------------------------------------------------------------ events and market
def test_a_results_date_tomorrow_keeps_an_overnight_trade_off_but_the_swing_idea_carries_the_warning():
    found = listed({"UP": {**BTST_LONG, "event_days": 1.0}})
    assert "BTST:LONG:UP" not in found
    assert "results / board meeting next session: gap risk" in found["SWING:LONG:UP"].cautions


def test_a_results_date_further_out_only_warns_and_a_swing_runs_through_it_at_a_discount():
    plain = listed({"PULL": SWING_LONG})["SWING:LONG:PULL"]
    event = listed({"PULL": {**SWING_LONG, "event_days": 3.0}})["SWING:LONG:PULL"]
    assert event.score == pytest.approx(plain.score * 0.85, abs=0.2)
    assert "results / board meeting in 3 sessions: gap risk" in event.cautions
    assert "BTST:LONG:UP" in listed({"UP": {**BTST_LONG, "event_days": 3.0}})
    today = listed({"UP": {**BTST_LONG, "event_days": 0.0}})
    assert not [k for k in today if k.startswith("BTST")]  # announced today after the close: the overnight gap is the coin flip


def test_a_market_that_fell_sharply_withholds_overnight_longs_but_not_shorts():
    crash = {"nifty_chg": -2.5, "median_chg": -2.0}
    found = listed({"UP": BTST_LONG, "DOWN": BTST_SHORT}, market=crash)
    assert "BTST:LONG:UP" not in found and "BTST:SHORT:DOWN" in found
    assert engine.overnight_block(crash) == ("LONG", "the market is down 2.5% today: overnight long ideas (BTST) are withheld")


def test_a_market_that_jumped_withholds_overnight_shorts_but_not_longs():
    surge = {"nifty_chg": 2.2}
    found = listed({"UP": BTST_LONG, "DOWN": BTST_SHORT}, market=surge)
    assert "BTST:LONG:UP" in found and "BTST:SHORT:DOWN" not in found
    assert engine.overnight_block(surge)[0] == "SHORT"


def test_the_median_stock_stands_in_for_a_missing_nifty_and_nothing_known_blocks_nothing():
    assert engine.overnight_block({"median_chg": -2.1})[0] == "LONG"
    assert engine.overnight_block({}) == (None, None) and engine.overnight_block({"nifty_chg": 1.0}) == (None, None)
    assert engine.overnight_block({"nifty_chg": float("nan")}) == (None, None)


@pytest.mark.parametrize(("market", "regime"), [
    ({"nifty_chg": -1.2}, "bearish"), ({"median_chg": -0.9}, "bearish"), ({"nifty_above_ema20": False, "nifty_ema20_above_ema50": False}, "bearish"),
    ({"nifty_chg": 0.5}, "bullish"), ({"nifty_chg": 0.5, "nifty_above_ema20": False}, "neutral"), ({"nifty_chg": 0.1}, "neutral"),
    ({"median_chg": 0.4}, "bullish"), ({}, "neutral"), ({"nifty_chg": float("nan")}, "neutral"), ({"nifty_chg": None}, "neutral"),
])
def test_the_market_regime_from_what_is_known(market, regime):
    assert engine.market_regime(market) == regime


def test_a_market_against_the_side_discounts_the_score():
    bullish = listed({"DOWN": BTST_SHORT}, market={"nifty_chg": 0.5})["BTST:SHORT:DOWN"]
    bearish = listed({"DOWN": BTST_SHORT}, market={"nifty_chg": -0.5, "regime": "bearish"})["BTST:SHORT:DOWN"]
    assert bearish.score > bullish.score
    assert "market regime is bullish: this side is discounted" in bullish.cautions and not bearish.cautions


# ---------------------------------------------------------------------------------------------- news
def test_a_strong_confirmed_report_against_the_side_vetoes_the_idea():
    found = listed({"UP": BTST_LONG}, catalysts={"UP": [catalyst("Regulator bans product", polarity=-0.8, severity=0.7)]})
    assert "BTST:LONG:UP" not in found
    assert engine.news_view([catalyst("Fraud probe", polarity=-0.8)], "LONG")[4] is True


def test_news_for_a_long_is_news_against_a_short():
    good = [catalyst("Big order", polarity=0.8, severity=0.7)]
    assert "BTST:SHORT:DOWN" not in listed({"DOWN": BTST_SHORT}, catalysts={"DOWN": good})
    assert engine.news_view(good, "SHORT")[4] is True


def test_milder_news_against_the_side_discounts_the_score_and_says_so():
    base = listed({"UP": BTST_LONG})["BTST:LONG:UP"]
    mild = listed({"UP": BTST_LONG}, catalysts={"UP": [catalyst("Weak guidance", polarity=-0.4, severity=0.5)]})["BTST:LONG:UP"]
    assert mild.score == pytest.approx(base.score * 0.8, abs=0.1)
    assert any(c.startswith("news against this side: Weak guidance") for c in mild.cautions)


def test_confirmed_good_news_adds_a_few_points_and_a_reason_and_is_attached_to_the_idea():
    base = listed({"UP": BTST_LONG})["BTST:LONG:UP"]
    news = [catalyst("Order win", polarity=0.7, severity=0.6), catalyst("Second", polarity=0.1), catalyst("Third")]
    idea = listed({"UP": BTST_LONG}, catalysts={"UP": news})["BTST:LONG:UP"]
    assert idea.score == pytest.approx(min(100.0, base.score + 4.0), abs=0.1) and idea.reasons[0] == "news in its favour: Order win"
    assert [c.headline for c in idea.catalysts] == ["Order win", "Second"]


def test_a_rumour_a_weak_item_or_an_unsure_one_never_moves_the_score():
    base = listed({"UP": BTST_LONG})["BTST:LONG:UP"].score
    for item in (catalyst("Hearsay", polarity=-0.9, severity=0.9, rumour=True), catalyst("Minor", polarity=-0.9, severity=0.2),
                 catalyst("Unsure", polarity=-0.9, severity=0.9, confidence=0.3), catalyst("No view", polarity=None)):
        assert listed({"UP": BTST_LONG}, catalysts={"UP": [item]})["BTST:LONG:UP"].score == base
    rumour = engine.news_view([catalyst("Hearsay", polarity=0.9, rumour=True)], "LONG")
    assert rumour[:2] == (1.0, 0.0) and rumour[3] == ["unconfirmed report: Hearsay"] and rumour[4] is False


def test_a_headline_is_clipped_in_the_cautions():
    caution = engine.news_view([catalyst("x" * 200, rumour=True)], "LONG")[3][0]
    assert caution == "unconfirmed report: " + "x" * 90


# ---------------------------------------------------------------------------------------------- levels
def test_the_stop_is_the_nearer_of_an_atr_multiple_and_the_days_low_and_never_closer_than_the_minimum():
    row = {"ltp": 1000.0, "atr_abs": 20.0, "day_low": 985.0, "day_high": 1015.0, "low5": 975.0, "high5": 1030.0}
    long = plan_levels("LONG", BTST_PLAN, row, horizon="BTST")
    assert (long["entry"], long["max_entry"], long["stop"], long["target1"], long["target2"]) == (1000.0, 1003.0, 983.0, 1020.0, 1036.0)
    assert plan_levels("LONG", BTST_PLAN, {**row, "day_low": 999.5}, horizon="BTST")["stop"] == 988.0  # 0.6 ATR at least
    assert plan_levels("LONG", BTST_PLAN, {**row, "day_low": float("nan")}, horizon="BTST")["stop"] == 980.0
    assert plan_levels("LONG", BTST_PLAN, {**row, "day_low": 900.0}, horizon="BTST")["stop"] == 980.0  # the ATR stop is nearer
    short = plan_levels("SHORT", BTST_PLAN, row, horizon="BTST")
    assert (short["max_entry"], short["stop"], short["target1"], short["target2"]) == (997.0, 1017.0, 980.0, 964.0)
    swing = plan_levels("LONG", SWING_PLAN, row, horizon="SWING")
    assert (swing["stop"], swing["target1"], swing["target2"]) == (971.0, 1040.0, 1072.0)
    assert plan_levels("SHORT", SWING_PLAN, row, horizon="SWING")["stop"] == 1034.0  # the week's high plus a buffer, nearer than 1.8 ATR


def test_levels_sit_on_the_price_tick_and_report_risk_and_reward_in_percent():
    out = plan_levels("LONG", BTST_PLAN, {"ltp": 1234.57, "atr_abs": 17.31, "day_low": 1220.0}, horizon="BTST")
    assert all(round(out[k] / 0.05) * 0.05 == pytest.approx(out[k], abs=1e-9) for k in ("entry", "max_entry", "stop", "target1", "target2"))
    assert out["risk_pct"] == pytest.approx((out["entry"] - out["stop"]) / out["entry"] * 100.0)
    assert out["rr1"] == pytest.approx(out["reward1_pct"] / out["risk_pct"])


@pytest.mark.parametrize("row", [{"ltp": 1000.0}, {"ltp": 1000.0, "atr_abs": 0.0}, {"ltp": 1000.0, "atr_abs": float("nan")},
                                 {"atr_abs": 20.0}, {"ltp": -5.0, "atr_abs": 20.0}, {"ltp": 1000.0, "atr_abs": 0.02}])
def test_levels_are_never_invented_without_a_price_and_an_atr_or_with_no_room_for_a_stop(row):
    assert plan_levels("LONG", BTST_PLAN, row, horizon="BTST") is None


def test_both_watch_triggers_sit_beyond_the_days_range_with_a_stop_and_two_targets_each():
    levels = watch_levels({"atr_abs": 20.0, "day_high": 1006.0, "day_low": 996.0})
    assert levels == {"trigger_long": 1007.0, "long_stop": 997.0, "long_target1": 1019.0, "long_target2": 1029.0,
                      "trigger_short": 995.0, "short_stop": 1005.0, "short_target1": 983.0, "short_target2": 973.0}
    assert watch_levels({"atr_abs": 20.0, "day_high": 1006.0}) is None and watch_levels({"atr_abs": 0.0, "day_high": 1.0, "day_low": 1.0}) is None


# ------------------------------------------------------------------------------------------ the score
def test_a_score_needs_most_of_its_components_to_be_known():
    comp = pd.DataFrame({
        "strength": [1.0, 1.0, 0.5, np.nan], "momentum": [1.0, np.nan, 0.5, np.nan], "volume": [1.0, 0.0, 0.5, np.nan],
        "trend": [1.0, 1.0, 0.5, np.nan], "structure": [1.0, np.nan, 0.5, np.nan], "relstrength": [1.0, np.nan, 0.5, 1.0],
        "tailwind": [1.0, np.nan, 0.5, 1.0],
    }, index=["ALL", "SOME", "HALF", "FEW"])
    score = engine.weighted_score(comp, engine.BTST_WEIGHTS)
    assert score["ALL"] == pytest.approx(100.0) and score["HALF"] == pytest.approx(50.0)
    assert score["SOME"] == pytest.approx(100.0 * (0.25 + 0.15) / 0.60)  # strength, volume (0) and trend carry 0.60 of the weights
    assert np.isnan(score["FEW"])  # a stock with only 0.15 of the weights known is not scored


def test_the_f_and_o_flag_of_an_idea_is_the_one_the_gates_read():
    record = {"UP": {"is_fno": True}}
    table = with_fillers({"UP": {**BTST_LONG, "is_fno": False}})
    assert keys(rank_ideas(table, CALM, config(), instruments=record))["BTST:LONG:UP"].is_fno is False  # the table column wins
    no_column = with_fillers({"UP": BTST_LONG}).drop(columns="is_fno")
    assert keys(rank_ideas(no_column, CALM, config(), instruments=record))["BTST:LONG:UP"].is_fno is True  # else the record
    unknown = with_fillers({"DOWN": {**BTST_SHORT, "is_fno": np.nan}})
    assert "BTST:SHORT:DOWN" not in keys(rank_ideas(unknown, CALM, config()))  # an unknown flag is not F&O: no short


def test_missing_optional_columns_read_as_unknown_not_as_zero():
    table = with_fillers({"UP": BTST_LONG}).drop(columns=["sector_chg", "event_days", "is_fno", "excluded"])
    idea = keys(rank_ideas(table, CALM, config()))["BTST:LONG:UP"]
    assert idea.is_fno is False and idea.score > 70


def test_a_row_of_nothing_but_unknowns_and_an_empty_table_give_no_ideas_and_no_errors():
    unknown = {key: np.nan for key in BTST_LONG if key not in ("is_fno", "excluded")} | {"is_fno": False, "excluded": ""}
    assert listed({"NAN": unknown}) == {}
    assert rank_ideas(make_table({}), CALM, config()) == [] and rank_ideas(with_fillers({}), CALM, config()) == []


def test_the_words_of_an_idea_carry_the_figures_and_the_cautions_name_what_argues_against_it():
    row = {**BTST_LONG, "rsi14": 74.0, "dist_ema20_atr": 2.8, "ret1": 5.4, "event_days": 4.0, "tv_med20_cr": 30.0}
    cautions = engine.cautions_for(row, "LONG", "bearish", 0.45)
    assert "RSI 74: already stretched" in cautions and "already +5.4% today" in cautions
    assert "results / board meeting in 4 sessions: gap risk" in cautions and "market regime is bearish: this side is discounted" in cautions
    assert any("2.8 ATR above its 20-day average" in c for c in cautions) and "round-trip cost about 0.45% of the position" in cautions
    assert engine.cautions_for({**BTST_SHORT, "rsi14": 25.0}, "SHORT", "neutral", None) == ["RSI 25: already stretched"]
    assert len(engine.reasons_for(BTST_LONG, "LONG", "a named pattern")) <= 5 and engine.reasons_for({}, "LONG") == []
    assert engine.reasons_for(BTST_LONG, "LONG", "a named pattern")[2] == "a named pattern"


def test_the_config_comes_from_the_settings(settings):
    cfg = engine.IdeaConfig.from_settings(dataclasses.replace(settings, ideas_top_n=3, ideas_min_score=70.0, ideas_include_shorts=False))
    assert (cfg.top_n, cfg.min_score, cfg.include_shorts) == (3, 70.0, False)
    assert (cfg.min_price, cfg.min_turnover_cr, cfg.max_risk_pct) == (settings.min_price, settings.min_median_turnover_cr, settings.ideas_max_risk_pct)


def test_filler_names_are_not_ideas():
    assert rank_ideas(make_table({f"F{i}": FILLER for i in range(5)}), CALM, config()) == []


# ---------------------------------------------------------------------------------------- delivery cost
def test_holding_overnight_costs_more_than_a_day_trade_and_less_liquid_names_cost_more():
    model = CostModel()
    assert model.delivery_round_trip_pct(200_000.0) > model.round_trip_pct(200_000.0)
    assert np.isnan(model.delivery_round_trip_pct(0.0))
    for turnover in (25.0, 100.0, 400.0):
        assert estimate_delivery_round_trip_cost_pct(turnover) > estimate_round_trip_cost_pct(turnover)
    assert estimate_delivery_round_trip_cost_pct(25.0) > estimate_delivery_round_trip_cost_pct(100.0) > estimate_delivery_round_trip_cost_pct(400.0)
    assert 0.2 < estimate_delivery_round_trip_cost_pct(400.0) < 0.6  # tax on both sides alone is 0.2%
    assert estimate_delivery_round_trip_cost_pct(None) > estimate_delivery_round_trip_cost_pct(100.0)


def test_delivery_fees_are_the_tax_on_both_sides_plus_stamp_duty_and_the_usual_charges():
    model = CostModel(brokerage_flat=0.0, brokerage_pct=0.0, exch_txn_pct=0.0, sebi_pct=0.0, gst_pct=0.0)
    assert model.delivery_fees(100_000.0, 110_000.0) == pytest.approx(0.001 * 210_000.0 + 0.00015 * 100_000.0)
