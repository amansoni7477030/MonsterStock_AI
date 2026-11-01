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
    ({"f_oi_price_code": 2.0, "t_ret1": 2.0, "f_fut_oi_chg_pct": 4.0}, [], "price rose yesterday but futures OI build-up looks bearish"),
    ({"f_oi_price_code": 1.0, "t_ret1": -2.0, "f_fut_oi_chg_pct": 4.0}, [], "price fell yesterday while futures OI build-up looks bullish"),
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
