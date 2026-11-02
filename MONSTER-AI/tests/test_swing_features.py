"""The closing-hour features: the intraday volume profile, no look-ahead, the live row's volume projection and the ATR of the
previous close."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.swing import (
    MIN_VOLUME_FRACTION,
    SESSION_MINUTES,
    VOLUME_PROFILE,
    last_row,
    swing_frames,
    volume_fraction,
)
from tests.synthetic import make_panel


@pytest.fixture(scope="module")
def panel():
    return make_panel(n_dates=330, n_symbols=5, seed=3)


def frames_of(panel, upto=None, **kw):
    n = len(panel.dates) if upto is None else upto
    nifty = panel.nifty["close"].iloc[:n]
    return swing_frames(panel.o.iloc[:n], panel.h.iloc[:n], panel.low.iloc[:n], panel.c.iloc[:n], panel.v.iloc[:n], nifty, **kw)


# ----------------------------------------------------------------------------------------- the profile
def test_the_volume_profile_is_a_cumulative_share_from_zero_to_one():
    shares = [share for _minute, share in VOLUME_PROFILE]
    assert shares[0] == 0.0 and shares[-1] == 1.0 and shares == sorted(shares)
    assert [minute for minute, _share in VOLUME_PROFILE][-1] == SESSION_MINUTES == 375


@pytest.mark.parametrize(("minutes", "expected"), [(-30.0, MIN_VOLUME_FRACTION), (0.0, MIN_VOLUME_FRACTION), (375.0, 1.0), (500.0, 1.0)])
def test_the_fraction_is_floored_before_the_open_and_one_from_the_close_on(minutes, expected):
    assert volume_fraction(minutes) == expected


def test_the_fraction_never_decreases_and_at_the_closing_hour_most_of_the_day_is_done():
    values = [volume_fraction(m) for m in range(0, 400, 5)]
    assert values == sorted(values) and all(MIN_VOLUME_FRACTION <= v <= 1.0 for v in values)
    assert 0.75 < volume_fraction(315.0) < 0.83  # 14:30 IST is 315 minutes after 09:15
    assert volume_fraction(315.0) == pytest.approx(0.7875)  # the middle of 0.755 at 300 and 0.82 at 330


# -------------------------------------------------------------------------------------------- frames
def test_every_feature_is_a_dates_by_symbols_frame_on_the_inputs_index(panel):
    frames = frames_of(panel)
    for name, frame in frames.items():
        assert frame.shape == panel.c.shape and frame.index.equals(panel.c.index), name
    assert {"ltp", "ret1", "gap", "clv", "atr_abs", "rsi14", "adx14", "breakout20", "rvol", "tv_med20_cr", "rs1", "rs20", "hist_days"} <= set(frames)


def test_a_features_value_on_a_day_does_not_change_when_later_days_exist(panel):
    cut = 250
    short, full = frames_of(panel, upto=cut), frames_of(panel)
    for name, frame in short.items():
        np.testing.assert_allclose(frame.iloc[cut - 1].to_numpy(float), full[name].iloc[cut - 1].to_numpy(float), equal_nan=True, err_msg=name)


def test_the_returns_the_gap_and_the_close_position_follow_their_definitions(panel):
    frames = frames_of(panel)
    t = 200
    sym = panel.symbols[0]
    prev, close = panel.c[sym].iloc[t - 1], panel.c[sym].iloc[t]
    assert frames["ret1"][sym].iloc[t] == pytest.approx((close / prev - 1) * 100)
    assert frames["ret5"][sym].iloc[t] == pytest.approx((close / panel.c[sym].iloc[t - 5] - 1) * 100)
    assert frames["gap"][sym].iloc[t] == pytest.approx((panel.o[sym].iloc[t] / prev - 1) * 100)
    high, low = panel.h[sym].iloc[t], panel.low[sym].iloc[t]
    assert frames["clv"][sym].iloc[t] == pytest.approx((close - low) / (high - low))
    assert frames["prev_close"][sym].iloc[t] == prev and frames["ltp"][sym].iloc[t] == close


def test_a_session_with_no_range_has_no_close_position_instead_of_a_division_by_zero():
    idx = pd.bdate_range("2026-01-01", periods=40)
    flat = pd.DataFrame(100.0, index=idx, columns=["A"])
    frames = swing_frames(flat, flat, flat, flat, flat * 1000.0)
    assert frames["clv"].isna().all().all()


def test_the_20_day_breakout_compares_with_the_sessions_before_it_not_the_session_itself():
    idx = pd.bdate_range("2026-01-01", periods=60)
    base = pd.DataFrame({"A": np.linspace(100.0, 110.0, 60)}, index=idx)
    high = base * 1.01
    close = base.copy()
    close.iloc[-1] = float(high.iloc[-21:-1].max().iloc[0]) + 1.0  # closes beyond the highs of the 20 sessions before it
    high.iloc[-1] = close.iloc[-1] + 0.5
    frames = swing_frames(base, high, base * 0.99, close, base * 0 + 1e6)
    assert frames["breakout20"]["A"].iloc[-1] == 1.0 and frames["dist_hi20"]["A"].iloc[-1] > 0
    below = close.copy()
    below.iloc[-1] = float(high.iloc[-21:-1].max().iloc[0]) - 1.0
    high_hit = high.copy()
    high_hit.iloc[-1] = float(high.iloc[-21:-1].max().iloc[0]) + 3.0  # the day's HIGH went beyond it but the price is back under
    assert swing_frames(base, high_hit, base * 0.99, below, base * 0 + 1e6)["breakout20"]["A"].iloc[-1] == 0.0


def test_too_little_history_leaves_the_long_windows_unknown_and_hist_days_counts_the_bars(panel):
    frames = frames_of(panel, upto=100)
    assert frames["sma200"].iloc[-1].isna().all() and frames["dist_hi252"].iloc[-1].isna().all()
    assert frames["hist_days"].iloc[-1].eq(100.0).all() and frames["rsi14"].iloc[-1].notna().all()


# --------------------------------------------------------------------------------------- the live row
def test_the_live_rows_volume_is_projected_to_a_full_day_and_earlier_rows_are_untouched(panel):
    plain, live = frames_of(panel), frames_of(panel, live_fraction=0.5)
    pd.testing.assert_frame_equal(plain["rvol"].iloc[:-1], live["rvol"].iloc[:-1])
    np.testing.assert_allclose(live["rvol"].iloc[-1].to_numpy(float), plain["rvol"].iloc[-1].to_numpy(float) * 2.0)
    pd.testing.assert_frame_equal(plain["tv_med20_cr"], live["tv_med20_cr"])  # the baseline is the 20 sessions BEFORE this one


@pytest.mark.parametrize("fraction", [None, 1.0, 0.0, -0.2, 1.5])
def test_no_projection_without_a_session_in_progress(panel, fraction):
    pd.testing.assert_frame_equal(frames_of(panel)["rvol"], frames_of(panel, live_fraction=fraction)["rvol"])


def test_the_atr_that_sizes_stops_is_the_one_of_the_previous_close_so_a_half_formed_range_cannot_shrink_it(panel):
    before = frames_of(panel)
    blown = panel.h.copy()
    blown.iloc[-1] = blown.iloc[-1] * 1.5
    after = swing_frames(panel.o, blown, panel.low, panel.c, panel.v, panel.nifty["close"])
    pd.testing.assert_series_equal(before["atr_abs"].iloc[-1], after["atr_abs"].iloc[-1])
    pd.testing.assert_series_equal(before["atr_pct"].iloc[-1], after["atr_pct"].iloc[-1])
    assert (after["rng_pct"].iloc[-1] > before["rng_pct"].iloc[-1]).all() and (after["range_exp"].iloc[-1] > before["range_exp"].iloc[-1]).all()


def test_the_52_week_extremes_include_the_running_session():
    idx = pd.bdate_range("2025-01-01", periods=300)
    price = pd.DataFrame({"A": np.full(300, 100.0)}, index=idx)
    high, low = price * 1.01, price * 0.99
    high.iloc[-1] = 130.0
    close = price.copy()
    close.iloc[-1] = 125.0
    frames = swing_frames(price, high, low, close, price * 0 + 1e6)
    assert frames["dist_hi252"]["A"].iloc[-1] == pytest.approx((125.0 / 130.0 - 1) * 100)  # never beyond +0%: the day's high counts


# ----------------------------------------------------------------------------- relative strength, last_row
def test_relative_strength_is_the_stocks_move_less_the_nifty_move_and_unknown_without_a_nifty(panel):
    frames = frames_of(panel)
    t, sym = 200, panel.symbols[1]
    nifty = panel.nifty["close"]
    assert frames["rs1"][sym].iloc[t] == pytest.approx(frames["ret1"][sym].iloc[t] - (nifty.iloc[t] / nifty.iloc[t - 1] - 1) * 100)
    without = swing_frames(panel.o, panel.h, panel.low, panel.c, panel.v)
    assert without["rs1"].isna().all().all() and without["rs20"].isna().all().all() and without["ret1"].notna().any().any()


def test_last_row_is_one_table_with_a_row_per_symbol_and_a_column_per_feature(panel):
    frames = frames_of(panel)
    table = last_row(frames)
    assert list(table.index) == panel.symbols and set(table.columns) == set(frames)
    assert table.loc[panel.symbols[2], "ltp"] == panel.c[panel.symbols[2]].iloc[-1]
    earlier = last_row(frames, position=-3)
    assert earlier.loc[panel.symbols[2], "ltp"] == panel.c[panel.symbols[2]].iloc[-3]
