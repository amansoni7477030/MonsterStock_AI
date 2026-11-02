"""Features of the closing-hour scan: as of the END of each row's session, not shifted.

The morning builder (``features.builder``) shifts every price feature by one session, because a 09:12 scan has seen no bar of the
day. The closing scan runs at 14:30 and has yesterday's final bar AND the day so far: open, high, low, last price and the volume
traded. The last row of the matrices therefore holds a session that is not over yet, and every feature here is a function of the rows
up to and including it. Two things differ from a finished session and are handled here, once, for the live scan and the backtest alike:

* the volume of the last row is projected to a full day (``volume_fraction``: the share of a session's volume that is normally done
  by that time of day), so "volume is 2x its average" means the same at 14:30 as at the close;
* the ATR that sizes stops and targets is the one of the PREVIOUS close, so a half-formed range never shrinks it.

Run over the whole history (the final bar of each day standing in for the 14:30 state) the same function is what
``backtesting.ideas`` evaluates: one code path for the live scan and for its backtest.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from features.technical import TRADING_DAYS, adx, ema, rsi, true_range, wilder

SESSION_MINUTES = 375  # 09:15 - 15:30
MIN_VOLUME_FRACTION = 0.05
# Cumulative share of a regular NSE session's volume by minutes after 09:15 (piecewise linear). An approximation of the U-shaped
# profile of liquid stocks - a heavy open, a quiet middle, a heavy close - not a measurement: it only scales "volume so far" to a
# full-day equivalent, and the live scan says so in its notes.
VOLUME_PROFILE: tuple[tuple[int, float], ...] = (
    (0, 0.0), (15, 0.10), (30, 0.17), (60, 0.27), (90, 0.35), (120, 0.415), (150, 0.475), (180, 0.53), (210, 0.585),
    (240, 0.64), (270, 0.695), (300, 0.755), (330, 0.82), (360, 0.90), (375, 1.0),
)
_PROFILE_MINUTES = np.array([m for m, _ in VOLUME_PROFILE], dtype=float)
_PROFILE_SHARE = np.array([s for _, s in VOLUME_PROFILE], dtype=float)


def volume_fraction(minutes_since_open: float) -> float:
    """The share of a regular session's volume normally traded ``minutes_since_open`` minutes after 09:15: never below 5 % (it is
    a divisor) and 1.0 from the close on."""
    if minutes_since_open >= SESSION_MINUTES:
        return 1.0
    return max(MIN_VOLUME_FRACTION, float(np.interp(max(minutes_since_open, 0.0), _PROFILE_MINUTES, _PROFILE_SHARE)))


def _pct_change(frame: pd.DataFrame | pd.Series, periods: int) -> pd.DataFrame | pd.Series:
    return frame.pct_change(periods, fill_method=None) * 100.0


def swing_frames(
    o: pd.DataFrame, h: pd.DataFrame, low: pd.DataFrame, c: pd.DataFrame, v: pd.DataFrame, nifty_close: pd.Series | None = None,
    *, live_fraction: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Wide (dates x symbols) feature frames as of the end of each row. ``live_fraction`` marks the LAST row as a session in
    progress: its volume is divided by it (the share of the day's volume done so far) to a full-day projection. ``nifty_close``
    (aligned to the same dates) feeds the relative-strength features; without it they are NaN."""
    prev = c.shift(1)
    out: dict[str, pd.DataFrame] = {"ltp": c, "prev_close": prev}

    v_full = v.copy()
    if live_fraction is not None and 0.0 < live_fraction < 1.0 and len(v_full):
        v_full.iloc[-1] = v.iloc[-1] / live_fraction

    ret1 = _pct_change(c, 1)
    out["ret1"] = ret1
    for n in (3, 5, 20, 60):
        out[f"ret{n}"] = _pct_change(c, n)
    out["gap"] = (o / prev - 1.0) * 100.0

    span = h - low
    rng_pct = span / o * 100.0
    out["rng_pct"] = rng_pct
    out["clv"] = ((c - low) / span.where(span > 0)).clip(0.0, 1.0)

    atr_abs = wilder(true_range(h, low, c), 14).shift(1)  # the ATR of the previous close
    out["atr_abs"] = atr_abs
    out["atr_pct"] = atr_abs / prev * 100.0

    ema20, ema50 = ema(c, 20), ema(c, 50)
    out.update(ema20=ema20, ema50=ema50, sma200=c.rolling(200, min_periods=120).mean())
    out["dist_ema20_atr"] = (c - ema20) / atr_abs.where(atr_abs > 0)
    out["ema20_slope"] = (ema20 / ema20.shift(5) - 1.0) * 100.0
    out["rsi14"] = rsi(c, 14)
    out["adx14"] = adx(h, low, c, 14)

    hi20 = h.shift(1).rolling(20, min_periods=15).max()  # the 20 sessions BEFORE this one: a breakout is a move beyond them
    lo20 = low.shift(1).rolling(20, min_periods=15).min()
    out["dist_hi20"] = (c / hi20 - 1.0) * 100.0
    out["dist_lo20"] = (c / lo20 - 1.0) * 100.0
    out["breakout20"] = (c > hi20).astype(float).where(hi20.notna())
    out["breakdown20"] = (c < lo20).astype(float).where(lo20.notna())
    peak = np.maximum(h.shift(1).rolling(TRADING_DAYS, min_periods=120).max(), h)  # the 52-week extremes include today's session
    trough = np.minimum(low.shift(1).rolling(TRADING_DAYS, min_periods=120).min(), low)
    out["dist_hi252"] = (c / peak - 1.0) * 100.0
    out["dist_lo252"] = (c / trough - 1.0) * 100.0
    out["low3"] = low.rolling(3, min_periods=3).min()
    out["high3"] = h.rolling(3, min_periods=3).max()
    out["low5"] = low.rolling(5, min_periods=5).min()
    out["high5"] = h.rolling(5, min_periods=5).max()
    out.update(day_open=o, day_high=h, day_low=low)

    volume_base = v_full.shift(1).rolling(20, min_periods=10).mean()
    out["rvol"] = v_full / volume_base.where(volume_base > 0)
    out["tv_med20_cr"] = (c * v_full).shift(1).rolling(20, min_periods=10).median() / 1e7
    range_mean20 = rng_pct.shift(1).rolling(20, min_periods=10).mean()
    out["range_comp"] = rng_pct.shift(1).rolling(5, min_periods=3).mean() / range_mean20.where(range_mean20 > 0)
    out["range_exp"] = rng_pct / range_mean20.where(range_mean20 > 0)
    out["hist_days"] = c.notna().cumsum().astype(float)

    nifty = nifty_close.reindex(c.index) if nifty_close is not None else pd.Series(np.nan, index=c.index)
    nifty_ret1 = _pct_change(nifty, 1)
    out["rs1"] = ret1.sub(nifty_ret1, axis=0)
    for n in (5, 20, 60):
        out[f"rs{n}"] = out[f"ret{n}"].sub(_pct_change(nifty, n), axis=0)
    return out


def last_row(frames: Mapping[str, pd.DataFrame], position: int = -1) -> pd.DataFrame:
    """One session of the frames as a table: the symbols as the index, one column per feature (the last session by default)."""
    return pd.DataFrame({name: frame.iloc[position] for name, frame in frames.items()})
