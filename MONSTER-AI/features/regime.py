"""Market regime engine (blueprint section 11). Regime labels are derived only from data available at prediction time
(the builder shifts them by one session; the risk regime uses the already-aligned overnight global moves)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.technical import TRADING_DAYS, adx, ema, rolling_rank, true_range, wilder

TREND_NAMES = {1: "bull", 0: "sideways", -1: "bear"}
VOL_NAMES = {0: "low-vol", 1: "normal-vol", 2: "high-vol", 3: "crisis"}
RISK_NAMES = {1: "risk-on", 0: "neutral", -1: "risk-off"}


def market_features(
    nifty: pd.DataFrame, vix: pd.Series, ret1: pd.DataFrame, rng: pd.DataFrame, rvol1: pd.DataFrame,
    dist_ema50: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Market-level series as of each session's close. ``nifty`` has open/high/low/close indexed by date."""
    o, h, low, c = (nifty[k] for k in ("open", "high", "low", "close"))
    out: dict[str, pd.Series] = {}
    ret = c.pct_change(fill_method=None) * 100.0
    out["m_ret1"] = ret
    for n in (5, 20, 60):
        out[f"m_ret{n}"] = c.pct_change(n, fill_method=None) * 100.0
    atr = wilder(true_range(h.to_frame("x"), low.to_frame("x"), c.to_frame("x")), 14)["x"]
    out["m_atr_pct"] = atr / c * 100.0
    out["m_rv20"] = ret.rolling(20, min_periods=10).std() * np.sqrt(TRADING_DAYS)
    out["m_range1"] = (h - low) / o * 100.0
    vix = vix.reindex(c.index)
    out["m_vix"] = vix
    out["m_vix_chg1"] = vix.pct_change(fill_method=None) * 100.0
    out["m_vix_rank"] = rolling_rank(vix, TRADING_DAYS, 60)
    sma200 = c.rolling(200, min_periods=120).mean()
    out["m_dist_sma200"] = (c / sma200 - 1.0) * 100.0
    out["m_dist_hi252"] = (c / h.rolling(TRADING_DAYS, min_periods=120).max() - 1.0) * 100.0
    adx14 = adx(h.to_frame("x"), low.to_frame("x"), c.to_frame("x"), 14)["x"]
    out["m_adx14"] = adx14
    out["m_breadth_up"] = (ret1 > 0).astype(float).where(ret1.notna()).mean(axis=1, skipna=True)
    out["m_breadth_ma50"] = (dist_ema50 > 0).astype(float).where(dist_ema50.notna()).mean(axis=1, skipna=True)
    out["m_disp"] = ret1.std(axis=1, ddof=0)
    out["m_mean_range"] = rng.mean(axis=1, skipna=True)
    out["m_mean_rvol"] = rvol1.mean(axis=1, skipna=True)

    ema50 = ema(c, 50)
    bull = (c > sma200) & (ema50 > sma200) & (adx14 > 20)
    bear = (c < sma200) & (ema50 < sma200) & (adx14 > 20)
    trend = pd.Series(0.0, index=c.index).where(~bull, 1.0).where(~bear, -1.0)
    out["m_trend"] = trend.where(sma200.notna())

    fallback = rolling_rank(out["m_rv20"], TRADING_DAYS, 60)
    rank = out["m_vix_rank"].fillna(fallback)
    vol = pd.Series(1.0, index=c.index)
    vol = vol.where(rank >= 0.25, 0.0).where(rank < 0.75, 2.0).where(rank < 0.93, 3.0).where(~(vix > 30), 3.0)
    out["m_volreg"] = vol.where(rank.notna())
    return out


def risk_regime(us_ret: np.ndarray, vix_chg: np.ndarray, vix_lvl: np.ndarray) -> np.ndarray:
    """+1 risk-on, 0 neutral, -1 risk-off from the overnight US move and VIX (available before the Indian open)."""
    with np.errstate(invalid="ignore"):
        off = (us_ret < -1.0) | (vix_chg > 10.0) | (vix_lvl > 28.0)
        on = (us_ret > 0.3) & (vix_chg < 0.0) & ~off
    out = np.where(off, -1.0, np.where(on, 1.0, 0.0))
    return np.where(np.isfinite(us_ret) | np.isfinite(vix_chg), out, np.nan)


def describe(trend: float | None, volreg: float | None, risk: float | None, expiry: float | None = None) -> dict[str, str]:
    """Human-readable regime labels for reports and performance slices."""
    out: dict[str, str] = {}
    if trend is not None and trend == trend:
        out["trend"] = TREND_NAMES.get(int(trend), "n/a")
    if volreg is not None and volreg == volreg:
        out["volatility"] = VOL_NAMES.get(int(volreg), "n/a")
    if risk is not None and risk == risk:
        out["risk"] = RISK_NAMES.get(int(risk), "n/a")
    if expiry is not None and expiry == expiry and expiry > 0.5:
        out["event"] = "F&O expiry day"
    return out
