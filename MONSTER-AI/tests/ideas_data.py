"""Hand-built closing-hour tables for the trade-idea tests: one row per stock with the columns of ``features.swing.last_row`` and the
ones the scan adds, so a test changes exactly the figure it is about."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ranking.ideas import IdeaConfig
from ranking.types import Catalyst

# A textbook BTST long at Rs 1000 (ATR Rs 20): a +2.6% day closing near the high on 1.8x volume, through its 20-day high, in an
# uptrend, liquid, no event ahead.
BTST_LONG: dict[str, Any] = dict(
    ltp=1000.0, prev_close=975.0, ret1=2.56, clv=0.92, rvol=1.8, ema20=980.0, ema50=950.0, sma200=900.0, ema20_slope=1.2, rsi14=62.0,
    dist_ema20_atr=1.0, range_comp=0.9, dist_hi20=0.5, dist_lo20=12.0, dist_hi252=-1.0, dist_lo252=40.0, breakout20=1.0,
    breakdown20=0.0, rs1=1.5, rs5=1.2, rs20=3.0, sector_chg=0.8, atr_abs=20.0, atr_pct=2.0, adx14=28.0, tv_med20_cr=100.0,
    hist_days=300.0, day_high=1001.0, day_low=985.0, low5=960.0, high5=1005.0, event_days=np.nan, is_fno=True, excluded="",
)
# Its mirror image: a -2.6% day closing near the low on volume, through the 20-day low, in a downtrend.
BTST_SHORT: dict[str, Any] = {
    **BTST_LONG, "ltp": 1000.0, "prev_close": 1026.0, "ret1": -2.53, "clv": 0.08, "ema20": 1020.0, "ema50": 1050.0, "sma200": 1100.0,
    "ema20_slope": -1.2, "rsi14": 38.0, "dist_ema20_atr": -1.0, "dist_hi20": -12.0, "dist_lo20": -0.5, "dist_hi252": -40.0,
    "dist_lo252": 1.0, "breakout20": 0.0, "breakdown20": 1.0, "rs1": -1.5, "rs5": -1.2, "rs20": -3.0, "sector_chg": -0.8,
    "day_high": 1015.0, "day_low": 999.0, "low5": 995.0, "high5": 1040.0,
}
# A pullback to the 20-day average in an uptrend: quiet day, not a BTST.
SWING_LONG: dict[str, Any] = {
    **BTST_LONG, "ret1": 0.4, "clv": 0.6, "rvol": 0.9, "ema20": 998.0, "ema50": 960.0, "sma200": 900.0, "rsi14": 52.0,
    "dist_ema20_atr": 0.1, "dist_hi20": -4.0, "breakout20": 0.0, "day_high": 1004.0, "day_low": 990.0, "low5": 975.0, "high5": 1020.0,
}
# A tight range under the 20-day high with room to move: an intraday watch for tomorrow, no swing or BTST gate passes.
WATCH_ONLY: dict[str, Any] = {
    **BTST_LONG, "ret1": 0.1, "clv": 0.5, "rvol": 1.4, "range_comp": 0.6, "ema20": 995.0, "ema50": 960.0, "rsi14": 50.0, "adx14": 12.0,
    "dist_hi20": -1.0, "dist_lo20": 9.0, "breakout20": 0.0, "atr_pct": 2.2, "day_high": 1006.0, "day_low": 996.0,
}
FILLER: dict[str, Any] = {
    **BTST_LONG, "ret1": 0.0, "clv": 0.5, "rvol": 1.0, "rs5": 0.0, "rs20": 0.0, "rs1": 0.0, "breakout20": 0.0, "dist_hi20": -8.0,
    "dist_lo20": 8.0, "rsi14": 50.0, "adx14": 10.0, "atr_pct": 1.0, "range_comp": 1.2, "ema20": 1000.0, "ema50": 1000.0,
    "ema20_slope": 0.0, "sma200": 1000.0, "dist_ema20_atr": 0.0, "sector_chg": 0.0,
}
CALM = {"nifty_chg": 0.4, "median_chg": 0.2, "regime": "bullish"}


def make_table(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """A table (symbols as the index) from ``{symbol: feature dict}``, columns typed as the scan's own table has them."""
    return pd.DataFrame.from_dict(rows, orient="index")


def with_fillers(rows: dict[str, dict[str, Any]], n: int = 6) -> pd.DataFrame:
    """The given rows next to ``n`` neutral names (percentile ranks inside the day's universe need a universe)."""
    return make_table({**rows, **{f"FIL{i}": {**FILLER, "rs5": (i - n / 2) * 0.1, "rs20": (i - n / 2) * 0.2} for i in range(n)}})


def config(**over: Any) -> IdeaConfig:
    return IdeaConfig(**{"top_n": 5, "min_score": 55.0, **over})


def catalyst(headline: str = "Demo order win", polarity: float = 0.7, severity: float = 0.7, confidence: float = 0.8,
             **over: Any) -> Catalyst:
    return Catalyst(headline=headline, polarity=polarity, severity=severity, confidence=confidence, source="Demo Wire",
                    url="https://example.com/demo", **over)
