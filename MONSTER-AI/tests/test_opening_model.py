"""The 09:30 opening model (``ranking.opening``): pure functions on small hand-made and synthetic inputs.

The golden cases were computed by an independent numpy script during the research (the same four inputs, the same fitted coefficients), so
they pin the shipped model to what was measured, not to itself. Two logical floors were added after the real-data replay and are pinned
here separately: a level the range so far has already passed has a chance of exactly 1, and the day's range band never starts below
the range so far."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ranking import opening as op
from ranking.types import Candidate, Catalyst
from tests.seed_data import panel_ending

# (gap %, opening range %, 20-session mean range %, ATR %) -> P(>2%), P(>3%), P(>5%), day q10/q50/q90, ahead q10/q50/q90
# (the research's unfloored fit, but for the case whose opening range had already passed a level or sits above the band: "explosive" prints
# 2.0 % so its day band starts at 2.00 instead of 1.89; "volatile gapper" prints 2.5 %, so P(>2%) is exactly 1 and its day band, which
# starts at 2.57, is already above the range so far)
GOLDEN = {
    "typical": ((0.3, 1.0, 2.3, 2.2), (0.533, 0.159, 0.013), (1.39, 2.06, 3.34), (1.03, 1.64, 2.80)),
    "quiet": ((0.3, 0.5, 2.3, 2.2), (0.199, 0.042, 0.002), (1.03, 1.52, 2.46), (0.87, 1.39, 2.36)),
    "explosive": ((0.3, 2.0, 2.3, 2.2), (0.840, 0.447, 0.073), (2.00, 2.79, 4.53), (1.22, 1.95, 3.32)),
    "volatile gapper": ((1.5, 2.5, 3.5, 3.0), (1.0, 0.764, 0.258), (2.57, 3.79, 6.16), (1.65, 2.65, 4.51)),
    "calm stock": ((0.0, 0.5, 1.5, 1.4), (0.068, 0.017, 0.001), (0.83, 1.23, 1.99), (0.65, 1.05, 1.78)),
    "gap down": ((-2.0, 1.6, 2.0, 2.0), (0.803, 0.374, 0.057), (1.78, 2.63, 4.27), (1.16, 1.86, 3.17)),
}


def hist_frame(rows: dict[str, tuple[float, float]], *, tv: float = 100.0, days: float = 200.0) -> pd.DataFrame:
    """The history table the model reads: per symbol (20-session mean range %, ATR %), previous close 100."""
    return pd.DataFrame({
        "prev_close": [100.0] * len(rows), "rm20": [r[0] for r in rows.values()], "atr_abs": [r[1] for r in rows.values()],
        "atr_pct": [r[1] for r in rows.values()], "tv_med20_cr": [tv] * len(rows), "hist_days": [days] * len(rows),
    }, index=list(rows))


def live_row(gap: float, or_pct: float, *, last_at: float = 0.5, prev: float = 100.0) -> dict[str, float]:
    """A live row: opened ``gap`` % from the previous close, the day's range so far is ``or_pct`` % of the open, the last price at
    ``last_at`` of that range (0 = the low, 1 = the high)."""
    opened = prev * (1 + gap / 100.0)
    high, low = opened * (1 + or_pct / 200.0), opened * (1 - or_pct / 200.0)
    return {"open": opened, "high": high, "low": low, "last": low + last_at * (high - low), "prev_close": prev, "volume": 1e6}


def table_for(cases: dict[str, tuple[float, float, float, float]], elapsed: float = 15.0):
    live = {name: live_row(gap, orp) for name, (gap, orp, _rm, _atr) in cases.items()}
    hist = hist_frame({name: (rm, atr) for name, (_g, _o, rm, atr) in cases.items()})
    return op.opening_table(live, hist, elapsed)


# ------------------------------------------------------------------------------------------------ the shipped model
@pytest.mark.parametrize("name", list(GOLDEN))
def test_the_model_reproduces_the_research_figures(name):
    inputs, chances, day, ahead = GOLDEN[name]
    table, rejected = table_for({name: inputs})
    assert not rejected
    row = op.score(table).loc[name]
    assert [row["p2"], row["p3"], row["p5"]] == pytest.approx(chances, abs=0.0015)
    assert [row["day_q10"], row["day_q50"], row["day_q90"]] == pytest.approx(day, abs=0.006)
    assert [row["ahead_q10"], row["ahead_q50"], row["ahead_q90"]] == pytest.approx(ahead, abs=0.006)


def test_chances_and_quantiles_are_ordered_for_every_input():
    grid = {f"s{i}": (gap, orp, rm, atr) for i, (gap, orp, rm, atr) in enumerate(
        (g, o, r, a) for g in (-6.0, 0.0, 0.4, 3.0) for o in (0.05, 0.4, 1.2, 4.0, 9.0) for r in (0.5, 1.5, 3.0, 8.0) for a in (0.6, 2.0, 6.0))}
    row = op.score(table_for(grid)[0])
    assert row["p2"].between(0, 1).all() and row["p5"].between(0, 1).all()
    assert (row["p2"] >= row["p3"]).all() and (row["p3"] >= row["p5"]).all()
    assert (row["ahead_q10"] < row["ahead_q50"]).all() and (row["ahead_q50"] < row["ahead_q90"]).all()
    assert (row["day_q10"] <= row["day_q50"]).all() and (row["day_q50"] <= row["day_q90"]).all()  # equal where the range so far bounds them
    assert row[["day_q10", "day_q90", "ahead_q10", "ahead_q90"]].notna().all().all()


def test_a_level_the_range_so_far_has_passed_has_a_chance_of_exactly_one():
    cases = {"under": (0.3, 1.5, 2.3, 2.2), "at2": (0.3, 2.0, 2.3, 2.2), "over2": (0.3, 2.5, 2.3, 2.2), "over3": (0.3, 3.5, 2.3, 2.2),
             "at5": (0.3, 5.0, 2.3, 2.2), "over5": (0.3, 5.6, 2.3, 2.2)}
    row = op.score(table_for(cases)[0])
    assert row.loc["under", ["p2", "p3", "p5"]].lt(1.0).all()
    assert row.loc["at2", "p2"] < 1.0  # a range of exactly 2 % has not passed 2 %
    assert row.loc["over2", "p2"] == 1.0 and row.loc["over2", "p3"] < 1.0
    assert row.loc["over3", ["p2", "p3"]].eq(1.0).all() and row.loc["over3", "p5"] < 1.0
    assert row.loc["at5", "p5"] < 1.0 and row.loc["over5", ["p2", "p3", "p5"]].eq(1.0).all()


def test_the_days_range_band_never_starts_below_the_range_so_far():
    grid = {f"s{i}": (gap, orp, rm, atr) for i, (gap, orp, rm, atr) in enumerate(
        (g, o, r, a) for g in (0.0, 2.0) for o in (0.3, 1.5, 3.0, 6.0, 9.0) for r in (0.8, 2.0, 4.0, 10.0) for a in (1.0, 3.0))}
    table, _ = table_for(grid)
    row = op.score(table)
    for label in ("q10", "q50", "q90"):
        assert (row[f"day_{label}"] >= table["or_raw_pct"] - 1e-9).all(), label
    big = row.loc[table["or_raw_pct"] > 8.0]
    assert len(big) and np.allclose(big["day_q10"], table.loc[big.index, "or_raw_pct"])  # past 8 % with a 10th percentile far below: bounded by it
    assert (big["p2"] == 1.0).all() and (big["p3"] == 1.0).all() and (big["p5"] == 1.0).all()


def test_the_floors_leave_the_range_still_ahead_and_names_below_the_level_untouched():
    plain = {"a": (0.3, 1.0, 2.3, 2.2)}
    wide = {"b": (0.3, 9.0, 2.3, 2.2)}
    a, b = op.score(table_for(plain)[0]).loc["a"], op.score(table_for(wide)[0]).loc["b"]
    assert b["ahead_q10"] < b["ahead_q50"] < b["ahead_q90"] and b["ahead_q50"] > a["ahead_q50"]  # the range ahead is still estimated, not clipped
    assert a["day_q10"] > 1.0 and a[["p2", "p3", "p5"]].lt(1.0).all()


def test_a_wider_opening_never_lowers_the_estimates():
    cases = {f"s{i}": (0.3, orp, 2.3, 2.2) for i, orp in enumerate((0.3, 0.6, 1.0, 1.5, 2.2, 3.5))}
    row = op.score(table_for(cases)[0])
    for column in ("p2", "p3", "p5", "day_q50", "ahead_q50"):
        assert row[column].is_monotonic_increasing, column


def test_a_missing_input_gives_nan_estimates_never_a_default():
    table, _ = table_for({"a": (0.3, 1.0, 2.3, 2.2), "b": (0.3, 1.0, 2.3, 2.2)})
    table.loc["b", "rm20"] = np.nan
    table.loc["b", "x"] = np.nan
    row = op.score(table)
    assert row.loc["a", "p3"] > 0 and math.isnan(row.loc["b", "p3"]) and math.isnan(row.loc["b", "day_q50"])


def test_an_empty_table_scores_to_an_empty_frame_with_the_columns():
    scored = op.score(op.opening_table({}, hist_frame({"a": (2.0, 2.0)}), 15.0)[0])
    assert scored.empty and {"p3", "day_q50", "ahead_q90", "strength"} <= set(scored.columns)
    assert op.eligibility(scored, op.OpeningConfig()).empty


# ------------------------------------------------------------------------------------------------ the table
def test_the_table_computes_gap_range_move_and_position():
    table, _ = op.opening_table({"a": live_row(1.0, 2.0, last_at=0.75)}, hist_frame({"a": (2.0, 2.0)}), 15.0)
    row = table.loc["a"]
    assert row["gap_pct"] == pytest.approx(1.0) and row["or_pct"] == pytest.approx(2.0) and row["clv"] == pytest.approx(0.75)
    assert row["x"] == pytest.approx(1.0) and row["gapx"] == pytest.approx(0.5)
    assert row["move_pct"] == pytest.approx((row["ltp"] / row["open"] - 1) * 100) and row["or_high"] > row["ltp"] > row["or_low"]


def test_a_late_scan_is_scaled_back_to_the_fifteen_minute_basis():
    live, hist = {"a": live_row(0.3, 1.2)}, hist_frame({"a": (2.3, 2.2)})
    on_time, late = op.opening_table(live, hist, 15.0)[0].loc["a"], op.opening_table(live, hist, 25.0)[0].loc["a"]
    assert on_time["or_pct"] == pytest.approx(1.2) and late["or_raw_pct"] == pytest.approx(1.2)
    assert late["or_pct"] == pytest.approx(1.2 / 1.131) and late["x"] < on_time["x"]


def test_rows_the_model_cannot_trust_are_left_out_with_a_reason():
    live = {
        "ok": live_row(0.2, 1.0), "nohist": live_row(0.2, 1.0), "zero": {**live_row(0.2, 1.0), "open": 0.0},
        "missing": {k: v for k, v in live_row(0.2, 1.0).items() if k != "low"}, "inverted": {**live_row(0.2, 1.0), "high": 90.0, "low": 95.0},
        "wild_last": {**live_row(0.2, 1.0), "last": 140.0}, "nan": {**live_row(0.2, 1.0), "high": float("nan")},
    }
    hist = hist_frame({k: (2.0, 2.0) for k in live if k != "nohist"})
    table, rejected = op.opening_table(live, hist, 15.0)
    assert list(table.index) == ["ok"]
    assert rejected["nohist"] == op.NO_HISTORY and rejected["wild_last"] == op.INCONSISTENT
    assert {rejected[k] for k in ("zero", "missing", "inverted", "nan")} == {op.INCOMPLETE}


def test_a_last_price_a_hair_outside_the_range_widens_the_range_instead_of_dropping_the_name():
    row = live_row(0.2, 1.0)
    row["last"] = row["high"] * 1.001
    table, rejected = op.opening_table({"a": row}, hist_frame({"a": (2.0, 2.0)}), 15.0)
    assert not rejected and table.loc["a", "or_high"] == pytest.approx(row["last"]) and table.loc["a", "clv"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------ history
def test_history_stats_match_a_direct_computation():
    panel = panel_ending(n_dates=90, n_symbols=5, seed=5)
    stats = op.history_stats(panel.o, panel.h, panel.low, panel.c, panel.v)
    for symbol in panel.symbols:
        o, h, low, c, v = (getattr(panel, k)[symbol] for k in ("o", "h", "low", "c", "v"))
        rng = ((h - low) / o * 100).iloc[-20:]
        tr = np.maximum(h - low, np.maximum((h - c.shift(1)).abs(), (low - c.shift(1)).abs()))  # the research's formula: NaN-propagating
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
        row = stats.loc[symbol]
        assert row["prev_close"] == pytest.approx(c.iloc[-1]) and row["rm20"] == pytest.approx(rng.mean())
        assert row["atr_abs"] == pytest.approx(atr) and row["atr_pct"] == pytest.approx(atr / c.iloc[-1] * 100)
        assert row["tv_med20_cr"] == pytest.approx((c * v).iloc[-20:].median() / 1e7) and row["hist_days"] == 90


def test_history_needs_fifteen_of_twenty_sessions_for_the_range_and_the_traded_value():
    panel = panel_ending(n_dates=90, n_symbols=2, seed=5)
    a, b = panel.symbols
    o, h, low, c, v = (getattr(panel, k).copy() for k in ("o", "h", "low", "c", "v"))
    for frame in (o, h, low, c, v):
        frame.iloc[-14:, frame.columns.get_loc(b)] = np.nan  # only 6 of the last 20 sessions are known
    stats = op.history_stats(o, h, low, c, v)
    assert stats.loc[a, ["rm20", "tv_med20_cr"]].notna().all() and stats.loc[b, ["rm20", "tv_med20_cr"]].isna().all()
    assert math.isnan(stats.loc[b, "prev_close"])  # no close of the previous session: the scan leaves the name out
    assert op.history_stats(o.iloc[:0], h.iloc[:0], low.iloc[:0], c.iloc[:0], v.iloc[:0]).empty


# ------------------------------------------------------------------------------------------------ words and bands
@pytest.mark.parametrize(("x", "band"), [(0.0, "QUIET"), (0.299, "QUIET"), (0.30, "NORMAL"), (0.599, "NORMAL"), (0.60, "HOT"), (0.849, "HOT"),
                                         (0.85, "EXPLOSIVE"), (4.0, "EXPLOSIVE"), (None, ""), (float("nan"), ""), ("x", "")])
def test_strength_bands(x, band):
    assert op.strength_of(x) == band


@pytest.mark.parametrize(("median_x", "kind"), [(0.30, "calm"), (0.377, "calm"), (0.378, "average"), (0.41, "average"), (0.442, "average"), (0.443, "wide"), (None, "")])
def test_day_type_bands(median_x, kind):
    assert op.day_type(median_x) == kind


def test_range_growth_is_one_up_to_fifteen_minutes_then_grows_and_is_capped():
    assert [op.range_growth(m) for m in (0, 14, 15)] == [1.0, 1.0, 1.0]
    assert op.range_growth(20) == pytest.approx(1.069) and op.range_growth(22.5) == pytest.approx((1.069 + 1.131) / 2)
    assert op.range_growth(30) == pytest.approx(1.183) and op.range_growth(500) == pytest.approx(1.273)
    assert op.range_growth(float("nan")) == 1.0 and op.range_growth(25) > op.range_growth(20)


def test_the_strength_history_covers_every_band_and_rises_with_it():
    assert set(op.STRENGTH_HISTORY) == {name for _floor, name in op.STRENGTH_BANDS}
    order = ("QUIET", "NORMAL", "HOT", "EXPLOSIVE")
    rates = [op.STRENGTH_HISTORY[name][1] for name in order]
    ahead = [op.STRENGTH_HISTORY[name][2] for name in order]  # the median range after 09:30 in each band: wider openings do leave more of the day
    assert rates == sorted(rates) and ahead == sorted(ahead) and sum(op.STRENGTH_HISTORY[n][0] for n in op.STRENGTH_HISTORY) == op.MODEL_INFO["stock_days"]
    assert all(0.0 < a < 5.0 for a in ahead)


# ------------------------------------------------------------------------------------------------ eligibility and choosing
def candidate(symbol, rank, tier="STRONG", p3=0.6, threshold=3.0, catalysts=()):
    return Candidate(symbol=symbol, rank=rank, tier=tier, p={"hit_3": p3, "hit_2": 0.8, "hit_5": 0.1}, primary_threshold=threshold,
                     range={"q10": 1.0, "q50": 2.5, "q90": 5.0}, catalysts=list(catalysts))


def scored_set(cases):
    return op.score(table_for(cases)[0])


def test_the_gates_leave_out_illiquid_cheap_short_history_and_wild_gap_names():
    cases = {n: (0.3, 2.0, 2.3, 2.2) for n in ("good", "illiquid", "cheap", "young", "gapped", "zero_range")}
    table, _ = table_for(cases)
    table.loc["illiquid", "tv_med20_cr"] = 5.0
    table.loc["cheap", "ltp"] = 40.0
    table.loc["young", "hist_days"] = 30.0
    table.loc["gapped", "gap_pct"] = 25.0
    table.loc["zero_range", "or_pct"] = 0.0
    ok = op.eligibility(op.score(table), op.OpeningConfig())
    assert ok.to_dict() == {"good": True, "illiquid": False, "cheap": False, "young": False, "gapped": False, "zero_range": False}


def test_a_name_whose_cost_swallows_the_range_ahead_is_left_out():
    table, _ = table_for({"a": (0.3, 2.0, 2.3, 2.2), "b": (0.3, 2.0, 2.3, 2.2)})
    table.loc["b", "tv_med20_cr"] = 26.0  # barely liquid: wide slippage
    scored = op.score(table)
    strict = op.OpeningConfig(max_cost_ratio=0.12)
    assert bool(op.eligibility(scored, strict)["a"]) and not bool(op.eligibility(scored, strict)["b"])


def test_picks_keep_their_morning_order_and_movers_are_ranked_by_the_range_still_expected():
    cases = {"m1": (0.3, 2.6, 2.3, 2.2), "m2": (0.3, 1.9, 2.3, 2.2), "m3": (0.3, 0.7, 2.3, 2.2), "p1": (0.3, 0.6, 2.3, 2.2), "p2": (0.3, 2.4, 2.3, 2.2)}
    scored = scored_set(cases)
    info = {s: {"name": f"{s} Ltd", "sector": "Banks", "is_fno": True} for s in cases}
    picks, movers, missing = op.build_ideas(scored, op.OpeningConfig(min_x=0.6), instruments=info, stage="final",
                                            candidates=[candidate("p2", 2), candidate("p1", 1), candidate("gone", 3)])
    assert [p.symbol for p in picks] == ["p1", "p2"] and [p.rank for p in picks] == [1, 2] and missing == ["gone"]
    assert [m.symbol for m in movers] == ["m1", "m2"] and [m.rank for m in movers] == [1, 2]  # m3 opened at 0.30x its normal range: not a mover
    assert movers[0].ahead[1] > movers[1].ahead[1] and all(m.x >= 0.6 for m in movers)
    assert not ({p.symbol for p in picks} & {m.symbol for m in movers})
    assert picks[0].kind == "PICK" and movers[0].kind == "MOVER" and picks[0].key == "PICK:p1"
    assert picks[0].pre == {"rank": 1, "tier": "STRONG", "p": 0.6, "threshold": 3.0, "stage": "final"} and movers[0].pre == {}
    assert picks[0].name == "p1 Ltd" and picks[0].is_fno


def test_movers_are_ranked_by_the_range_ahead_not_by_the_chance_of_a_range_already_printed():
    # "big" has already printed 4 % (a range above 3 % is certain: nothing to estimate) in a stock whose normal range is wide, so little is left;
    # "fresh" printed 2 % in a calmer stock and has more range expected ahead. The list must put the one with the most range ahead first.
    scored = scored_set({"big": (0.0, 4.0, 3.0, 2.0), "fresh": (0.0, 2.0, 2.6, 3.4)})
    assert scored.loc["big", "p3"] == 1.0 and scored.loc["fresh", "p3"] < 1.0
    _, movers, _ = op.build_ideas(scored, op.OpeningConfig(min_x=0.6, max_cost_ratio=0.9), instruments={})
    by_ahead = sorted(scored.index, key=lambda s: -scored.loc[s, "ahead_q50"])
    assert [m.symbol for m in movers] == by_ahead and len(movers) == 2


def test_movers_need_the_minimum_opening_strength_and_the_list_length_is_capped():
    scored = scored_set({f"s{i}": (0.3, 1.0 + i * 0.25, 2.3, 2.2) for i in range(12)})
    info = {s: {} for s in scored.index}
    _, strict, _ = op.build_ideas(scored, op.OpeningConfig(min_x=2.0), instruments=info)
    _, capped, _ = op.build_ideas(scored, op.OpeningConfig(min_x=0.3, top_n=3), instruments=info)
    _, hot, _ = op.build_ideas(scored, op.OpeningConfig(min_x=1.0), instruments=info)
    assert strict == [] and len(capped) == 3 and [m.symbol for m in capped] == ["s11", "s10", "s9"]
    assert [m.symbol for m in hot] == ["s11", "s10", "s9", "s8", "s7", "s6"] and all(m.x >= 1.0 for m in hot)  # s5 opened at 0.98x: just short


def test_a_pick_is_listed_even_when_it_would_not_pass_the_mover_gates():
    scored = scored_set({"quiet": (0.0, 0.3, 2.3, 2.2)})
    picks, movers, missing = op.build_ideas(scored, op.OpeningConfig(), instruments={}, candidates=[candidate("quiet", 1)], stage="pre")
    assert [p.symbol for p in picks] == ["quiet"] and picks[0].strength == "QUIET" and picks[0].pre["stage"] == "pre" and not movers and not missing


def test_only_the_selected_movers_are_looked_up_for_news():
    scored = scored_set({"a": (0.3, 2.6, 2.3, 2.2), "b": (0.3, 0.4, 2.3, 2.2), "p": (0.3, 2.4, 2.3, 2.2)})
    asked = []

    def news(symbols):
        asked.append(list(symbols))
        return {"a": [Catalyst(headline="  Order   win\nannounced ", source="Exchange")]}

    own = Catalyst(headline="Morning headline", source="Wire")
    picks, movers, _ = op.build_ideas(scored, op.OpeningConfig(min_x=0.6), instruments={}, news=news,
                                      candidates=[candidate("p", 1, catalysts=[own])])
    assert asked == [["a"]]
    assert any(n == "news: Order win announced (Exchange)" for n in movers[0].notes)
    assert any(n.startswith("news: Morning headline") for n in picks[0].notes)


def test_levels_sit_beyond_the_opening_range_in_atr_steps():
    table, _ = table_for({"a": (0.3, 2.0, 2.3, 2.2)})
    row = op.score(table).loc["a"]
    levels = op.levels_for(row)
    atr = row["atr_abs"]
    assert levels["trigger_long"] > row["or_high"] and levels["trigger_short"] < row["or_low"]
    assert levels["long_stop"] < levels["trigger_long"] < levels["long_target1"] < levels["long_target2"]
    assert levels["short_stop"] > levels["trigger_short"] > levels["short_target1"] > levels["short_target2"]
    assert levels["trigger_long"] - levels["long_stop"] == pytest.approx(0.5 * atr, abs=0.06)
    assert levels["long_target1"] - levels["trigger_long"] == pytest.approx(0.6 * atr, abs=0.06)
    assert op.levels_for({"atr_abs": float("nan"), "or_high": 10, "or_low": 9}) == {}


def test_notes_say_what_the_gap_and_the_position_in_the_range_are_doing():
    live = {"filled": live_row(1.0, 3.0, last_at=0.95), "holding": live_row(1.0, 0.8, last_at=0.4), "tiny": live_row(0.2, 0.8, last_at=0.5),
            "down": live_row(-1.2, 2.5, last_at=0.05)}
    hist = hist_frame({k: (2.0, 2.0) for k in live})
    scored = op.score(op.opening_table(live, hist, 15.0)[0])
    picks, _, _ = op.build_ideas(scored, op.OpeningConfig(), instruments={}, candidates=[candidate(s, i) for i, s in enumerate(live, 1)])
    notes = {p.symbol: " | ".join(p.notes) for p in picks}
    assert "gap up +1.0%: filled inside the opening range" in notes["filled"] and "top of its opening range" in notes["filled"]
    assert "gap up +1.0%: still holding" in notes["holding"] and "gap" not in notes["tiny"]
    assert "gap down -1.2%" in notes["down"] and "bottom of its opening range" in notes["down"]


# ------------------------------------------------------------------------------------------------ the market and the state
def test_the_market_pulse_summarises_breadth_intensity_and_sectors():
    live = {f"s{i}": live_row(0.1, 1.2, last_at=0.7 if i % 3 else 0.2) for i in range(12)}
    hist = hist_frame({k: (2.4, 2.2) for k in live})
    scored = op.score(op.opening_table(live, hist, 15.0)[0])
    sectors = {s: {"sector": ("Banks", "IT", "Auto", "Pharma")[i % 4]} for i, s in enumerate(live)}  # four sectors of three names
    pulse = op.market_pulse(scored, {"NIFTY 50": {"change_pct": 0.31, "open": 100.0, "high": 100.6, "low": 100.0},
                                     "INDIA VIX": {"last": 14.2, "change_pct": -1.04}}, instruments=sectors, nifty_range20=1.0)
    assert pulse["nifty_chg"] == 0.31 and pulse["nifty_or_pct"] == 0.6 and pulse["nifty_x"] == 0.6
    assert pulse["vix"] == 14.2 and pulse["vix_chg"] == -1.0 and pulse["advancers"] + pulse["decliners"] == 12
    assert pulse["day_type"] == "wide" and pulse["median_x"] == pytest.approx(0.5, abs=0.01) and pulse["hot_share"] == 0.0
    up, down = pulse["sectors_up"], pulse["sectors_down"]
    assert len(up) == 3 and len(down) == 3 and up[0][1] >= up[1][1] >= up[2][1] and down[0][1] <= down[1][1] <= down[2][1]
    assert {name for name, _ in up + down} == {"Banks", "IT", "Auto", "Pharma"}


def test_the_market_pulse_is_partial_when_the_index_quotes_are_missing():
    scored = scored_set({"a": (0.3, 1.0, 2.3, 2.2)})
    pulse = op.market_pulse(scored, {}, instruments={})
    assert "nifty_chg" not in pulse and "vix" not in pulse and pulse["advancers"] + pulse["decliners"] <= 1 and "sectors_up" not in pulse
    assert op.market_pulse(op.score(op.opening_table({}, hist_frame({"a": (2, 2)}), 15)[0]), {}, instruments={}) == {}


def test_the_frozen_state_holds_exactly_the_eligible_rows_with_the_figures_the_record_needs():
    scored = scored_set({"a": (0.3, 2.0, 2.3, 2.2), "b": (0.3, 1.0, 2.3, 2.2)})
    ok = pd.Series({"a": True, "b": False})
    state = op.state_rows(scored, ok)
    assert [r[0] for r in state["rows"]] == ["a"] and state["cols"][0] == "symbol"
    row = dict(zip(state["cols"], state["rows"][0], strict=True))
    assert row["p3"] == pytest.approx(scored.loc["a", "p3"], abs=1e-4) and row["day_q10"] < row["day_q50"] < row["day_q90"] and row["open"] > 0
    assert row["or_raw_pct"] == pytest.approx(2.0, abs=1e-3)  # what the record needs to tell a name already past 3 % from one that was not
    assert op.state_rows(scored.iloc[:0], pd.Series(dtype=bool))["rows"] == []


def test_the_pre_view_names_the_threshold_it_speaks_for():
    assert op.pre_view(candidate("a", 2, tier="MODERATE", p3=0.4123), "final") == {"rank": 2, "tier": "MODERATE", "p": 0.412, "threshold": 3.0, "stage": "final"}
    odd = op.pre_view(candidate("a", 1, threshold=2.5), "pre")
    assert odd["p"] is None and odd["threshold"] == 2.5  # no "hit_2p5" probability was stored: nothing is invented


def test_every_default_matches_the_settings_catalogue(settings):
    cfg = op.OpeningConfig.from_settings(settings)
    assert cfg == op.OpeningConfig(top_n=settings.opening_top_n, min_x=settings.opening_min_x, min_turnover_cr=settings.min_median_turnover_cr,
                                   min_price=settings.min_price, max_cost_ratio=settings.max_cost_to_range_ratio)
    assert op.OpeningConfig().top_n == settings.opening_top_n and op.OpeningConfig().min_x == settings.opening_min_x
    assert op.OpeningConfig().min_x == op.STRENGTH_BANDS[1][0]  # the default mover is a HOT opening or more
    assert op.MODEL_INFO["id"] == op.MODEL_ID and op.MIN_ELAPSED_MIN < op.OPEN_MINUTES < op.MAX_ELAPSED_MIN
