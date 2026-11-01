"""Evidence helpers behind each report line: confirmations, contradictions, "priced in" and data-quality notes. The wording is what the
reader sees, so the tests pin it down (and the absence of causal or directional claims)."""

from __future__ import annotations

import math

import pytest

from ranking import signals as sg
from ranking.types import Catalyst


def cat(polarity=0.6, confidence=0.8, headline="Order win"):
    return Catalyst(headline=headline, polarity=polarity, confidence=confidence)


# ------------------------------------------------------------------------------------------ confirmations
def test_an_empty_or_unusable_row_confirms_nothing():
    assert sg.confirmations({}) == []
    junk = {"t_atr_ratio": "abc", "t_range_exp": math.nan, "t_rvol1": math.inf, "s_ret1": None, "f_iv_rank120": [1]}
    assert sg.confirmations(junk) == []


@pytest.mark.parametrize("row, expected", [
    ({"t_atr_ratio": 1.3, "t_atr5_pct": 2.4}, "technical: volatility expanding (5d ATR 2.4% = 1.30x the 14d ATR)"),
    ({"t_range_exp": 2.0}, "technical: yesterday's range was 2.0x its 20-day average"),
    ({"t_nr7": 1.0}, "technical: volatility compression (narrowest range in 7 sessions) - often precedes expansion"),
    ({"t_rvol1": 1.8}, "volume: 1.8x the 20-day average yesterday"),
    ({"t_breakout_up": 1.0}, "technical: closed beyond the prior 20-day range"),
    ({"t_breakdown": 1.0}, "technical: closed beyond the prior 20-day range"),
    ({"s_ret1": -1.5}, "sector: peers moved -1.5% yesterday"),
    ({"s_ret1": 1.2, "s_breadth1": 0.8}, "sector: peers moved +1.2% yesterday (breadth 80%)"),
    ({"f_iv_rank120": 0.85}, "F&O: implied volatility at the 85% percentile of its 120-day range"),
    ({"f_exp_move_1d": 3.0, "t_atr14_pct": 3.0}, "F&O: options imply a ~3.0% one-day move (ATR 3.0%)"),
    ({"f_oi_price_code": 2, "f_fut_oi_chg_pct": 5.2}, "F&O: short build-up (futures OI +5.2%) - heuristic, not proof of intent"),
    ({"f_oi_price_code": 9, "f_fut_oi_chg_pct": -6.0}, "F&O: positioning change (futures OI -6.0%) - heuristic, not proof of intent"),
    ({"c_expiry_today": 1.0}, "calendar: F&O expiry day"),
    ({"c_results_prev": 1.0}, "calendar: results / board-meeting catalyst today or yesterday"),
    ({"p_gap_pct": 2.5, "p_gap_atr": 1.2}, "pre-open: indicative gap +2.5% (1.2x ATR)"),
])
def test_each_activity_signal_produces_its_sentence(row, expected):
    assert sg.confirmations(row) == [expected]


@pytest.mark.parametrize("row", [
    {"t_atr_ratio": 1.1, "t_atr5_pct": 2.0}, {"t_range_exp": 1.2}, {"t_nr7": 0.0}, {"t_rvol1": 1.2}, {"s_ret1": 0.4},
    {"f_iv_rank120": 0.5}, {"f_exp_move_1d": 1.0, "t_atr14_pct": 3.0}, {"f_oi_price_code": 1, "f_fut_oi_chg_pct": 2.0},
    {"p_gap_pct": 0.5, "p_gap_atr": 0.3}, {"c_expiry_today": 0.0},
])
def test_ordinary_values_are_not_reported_as_evidence(row):
    assert sg.confirmations(row) == []


def test_at_most_six_confirmations_are_returned_in_a_fixed_order():
    row = {"t_atr_ratio": 1.5, "t_atr5_pct": 3.0, "t_range_exp": 2.0, "t_nr7": 1.0, "t_rvol1": 2.0, "t_breakout_up": 1.0,
           "s_ret1": 2.0, "f_iv_rank120": 0.9, "c_expiry_today": 1.0}
    out = sg.confirmations(row)
    assert len(out) == 6 and out[0].startswith("technical: volatility expanding") and out[-1].startswith("sector: peers moved")


# ---------------------------------------------------------------------------------------- contradictions
@pytest.mark.parametrize("row, catalysts, fragment", [
    ({"s_ret1": -2.0}, [cat(0.6)], "positive company news but the sector fell -2.0% yesterday"),
    ({"s_ret1": 2.0}, [cat(-0.6)], "negative company news but the sector rose +2.0% yesterday"),
    ({"p_gap_pct": -1.0}, [cat(0.6)], "positive news but the stock is indicated -1.0% lower at the open"),
    ({"p_gap_pct": 1.0}, [cat(-0.6)], "negative news but the stock is indicated +1.0% higher at the open"),
    ({"f_oi_price_code": 2.0, "f_fut_oi_chg_pct": 5.0, "p_gap_pct": 1.4}, [],
     "F&O positioning reads bearish (short build-up, OI +5.0%) but the stock is indicated +1.4% higher at the open"),
    ({"f_oi_price_code": 4.0, "f_fut_oi_chg_pct": -4.0, "t_ret5": 4.5}, [],
     "F&O positioning reads bearish (long unwinding, OI -4.0%) but its 5-day return is +4.5%"),
    ({"f_oi_price_code": 1.0, "f_fut_oi_chg_pct": 4.0, "p_gap_pct": -1.0}, [],
     "F&O positioning reads bullish (long build-up, OI +4.0%) but the stock is indicated -1.0% lower at the open"),
    ({"f_oi_price_code": 3.0, "f_fut_oi_chg_pct": -3.5, "t_ret5": -3.5}, [],
     "F&O positioning reads bullish (short covering, OI -3.5%) but its 5-day return is -3.5%"),
    ({"g_us_ret": -1.5}, [cat(0.6)], "positive stock-specific news against a weak global tone (US -1.5%)"),
    ({"g_oil_ret": 4.0, "g_exp_oil": 0.05}, [], "large oil move but this stock shows little historical oil sensitivity"),
])
def test_conflicting_evidence_is_named(row, catalysts, fragment):
    found = sg.contradictions(row, catalysts, None, None)
    assert found.count == 1 and fragment in found.items[0]


def test_model_disagreement_and_weak_catalysts_lower_confidence():
    found = sg.contradictions({}, [cat(0.0, confidence=0.3)], None, 0.2)
    assert found.items == ["specialist models disagree materially (low signal agreement)",
                           "catalyst evidence is low-confidence (possible rumour / single source)"]
    assert sg.contradictions({}, [cat(0.0, confidence=None)], None, 0.05).count == 0


def test_neutral_or_aligned_evidence_raises_no_contradiction():
    aligned = {"s_ret1": 2.0, "p_gap_pct": 1.5, "g_us_ret": 1.0}
    assert sg.contradictions(aligned, [cat(0.6)], None, None).count == 0
    assert sg.contradictions({"s_ret1": -3.0}, [cat(None), cat(0.0)], None, None).count == 0  # no polarity: nothing to contradict
    mixed = [cat(0.6), cat(-0.6)]  # a balanced set of stories has no net view
    assert sg.contradictions({"s_ret1": -3.0}, mixed, None, None).count == 0


def test_at_most_four_contradictions_are_kept():
    row = {"s_ret1": -2.0, "p_gap_pct": -2.0, "g_us_ret": -2.0, "g_oil_ret": 5.0, "g_exp_oil": 0.0}
    found = sg.contradictions(row, [cat(0.6, confidence=0.2)], None, 0.5)
    assert found.count == 4


# ---------------------------------------------- futures positioning against price action (heuristic)
@pytest.mark.parametrize("code, label, read, oi_chg", [
    (1, "long build-up", "bullish", 4.0), (2, "short build-up", "bearish", 4.0),
    (3, "short covering", "bullish", -4.0), (4, "long unwinding", "bearish", -4.0),
])
def test_each_oi_reading_is_checked_against_price_action_pointing_the_other_way(code, label, read, oi_chg):
    sign = 1 if read == "bullish" else -1
    for evidence in ({"p_gap_pct": -1.2 * sign}, {"t_ret5": -4.0 * sign}):
        row = {"f_oi_price_code": float(code), "f_fut_oi_chg_pct": oi_chg, **evidence}
        found = sg.contradictions(row, [], None, None)
        assert found.count == 1 and f"reads {read} ({label}, OI {oi_chg:+.1f}%)" in found.items[0]
        assert found.items[0].endswith("- heuristic")  # a reading of positioning, not a statement of fact


@pytest.mark.parametrize("row", [
    {"f_oi_price_code": 1.0, "f_fut_oi_chg_pct": 4.0, "p_gap_pct": 1.5, "t_ret5": 5.0},       # bullish reading, bullish price action
    {"f_oi_price_code": 2.0, "f_fut_oi_chg_pct": 4.0, "p_gap_pct": -1.5, "t_ret5": -5.0},     # bearish reading, bearish price action
    {"f_oi_price_code": 3.0, "f_fut_oi_chg_pct": -4.0, "p_gap_pct": -0.5, "t_ret5": -2.0},    # opposite price action, but too small
    {"f_oi_price_code": 1.0, "f_fut_oi_chg_pct": 2.5, "p_gap_pct": -2.0, "t_ret5": -6.0},     # a small OI change reads as nothing
    {"f_oi_price_code": 1.0, "f_fut_oi_chg_pct": 4.0, "p_gap_pct": -0.8, "t_ret5": -3.0},     # exactly at the limits, not beyond them
    {"f_oi_price_code": 9.0, "f_fut_oi_chg_pct": 5.0, "p_gap_pct": -2.0},                     # an unknown code
    {"f_oi_price_code": 1.0, "p_gap_pct": -2.0},                                              # no OI change
    {"f_fut_oi_chg_pct": 5.0, "p_gap_pct": -2.0},                                             # no code
    {"f_oi_price_code": 1.0, "f_fut_oi_chg_pct": 4.0, "t_ret1": -2.0},                        # yesterday's move alone is no evidence
])
def test_agreeing_weak_or_incomplete_positioning_evidence_is_not_a_contradiction(row):
    assert sg.contradictions(row, [], None, None).count == 0


def test_the_indicated_gap_is_named_before_the_five_day_trend_and_only_once():
    row = {"f_oi_price_code": 2.0, "f_fut_oi_chg_pct": 5.0, "p_gap_pct": 1.4, "t_ret5": 6.0}
    (item,) = sg.contradictions(row, [], None, None).items
    assert "indicated +1.4% higher at the open" in item and "5-day" not in item


def test_the_message_fits_the_report_line_so_the_heuristic_label_is_never_clipped():
    report_item_limit = 120  # the Telegram report clips each item at this many characters
    for code, oi in ((1, 9.9), (4, -9.9)):
        for evidence in ({"p_gap_pct": -9.9 if code == 1 else 9.9}, {"t_ret5": -9.9 if code == 1 else 9.9}):
            row = {"f_oi_price_code": float(code), "f_fut_oi_chg_pct": oi, **evidence}
            (item,) = sg.contradictions(row, [], None, None).items
            assert len(item) <= report_item_limit and item.endswith("- heuristic"), item


# ------------------------------------------------------------------------------ specialist disagreement
def test_the_disagreement_limit_defaults_to_0_12_and_a_model_derived_limit_can_replace_it():
    assert sg.SPEC_STD_LIMIT == 0.12
    assert sg.contradictions({}, [], None, 0.13).count == 1 and sg.contradictions({}, [], None, 0.12).count == 0
    assert sg.contradictions({}, [], None, 0.13, spec_std_limit=0.2).count == 0
    assert sg.contradictions({}, [], None, 0.25, spec_std_limit=0.2).items == [
        "specialist models disagree materially (low signal agreement)"]
    assert sg.contradictions({}, [], None, 0.05, spec_std_limit=0.04).count == 1  # a model whose agreement is normally tighter


@pytest.mark.parametrize("limit", [None, 0.0, -0.1, float("nan"), float("inf")])
def test_an_unusable_limit_falls_back_to_the_default(limit):
    assert sg.contradictions({}, [], None, 0.13, spec_std_limit=limit).count == 1
    assert sg.contradictions({}, [], None, 0.11, spec_std_limit=limit).count == 0


# --------------------------------------------------------------------------------------------- priced in
def test_priced_in_without_a_pre_open_price_says_so():
    text, score = sg.priced_in({}, 0, 0)
    assert text == "pre-open price not yet available" and score == 0.0
    assert sg.priced_in({"p_gap_pct": 1.0}, 0, 0) == ("not assessed", 0.0)  # a gap without its ATR scale cannot be judged


@pytest.mark.parametrize("gap, atr, score, fragment", [
    (5.0, 2.5, 0.5, "large indicated gap +5.0% (2.5x ATR): part of the move may already be in the opening price"),
    (2.0, 1.2, 0.25, "moderate gap +2.0% (1.2x ATR)"),
    (0.8, 0.4, 0.0, "small gap +0.8%"),
])
def test_the_opening_gap_relative_to_atr_drives_the_score(gap, atr, score, fragment):
    text, value = sg.priced_in({"p_gap_pct": gap, "p_gap_atr": atr}, 0, 0)
    assert text == fragment and value == pytest.approx(score)


def test_range_expansion_and_market_reaction_add_to_the_score_and_it_is_capped():
    text, score = sg.priced_in({"p_gap_pct": 1.0, "p_gap_atr": 2.5, "t_range_exp": 2.5}, 4, 4)
    assert score == pytest.approx(1.0) and "yesterday's range already 2.5x normal" in text and "4/4 catalyst(s)" in text
    text, score = sg.priced_in({"p_gap_pct": 0.2, "p_gap_atr": 0.1}, 2, 4)
    assert score == pytest.approx(0.15) and "2/4 catalyst(s) already show a market reaction" in text
    text, score = sg.priced_in({"p_gap_pct": 0.2, "p_gap_atr": 0.1}, 0, 3)
    assert score == 0.0 and "no reaction yet to the catalyst(s)" in text


# -------------------------------------------------------------------------------------------- data quality
def test_data_quality_notes_name_each_gap():
    assert sg.data_quality_notes({"t_atr14_pct": 2.0, "t_range_mean20": 2.0, "f_iv_atm": 0.3, "g_cover": 0.9}, True, "pre") == []
    notes = sg.data_quality_notes({"g_cover": 0.5}, True, "final")
    assert notes == ["insufficient price history", "F&O data missing for this stock",
                     "no pre-open price: preliminary model used", "global overnight coverage below 70%"]
    # a cash-only stock is not expected to have F&O data, and a preliminary scan has no pre-open price by design
    assert sg.data_quality_notes({"t_atr14_pct": 1.0, "t_range_mean20": 1.0}, False, "pre") == []
