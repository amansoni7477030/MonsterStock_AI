"""Technical, volatility and volume features.

All functions take wide matrices (rows = trading dates, columns = symbols) and return values *as of the end of
each row's session*. The builder shifts every output by one session, so a feature at date t only ever uses data
through t-1 (point-in-time safety by construction, verified by the leakage tests).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252
# Frames other feature groups (sector, regime, F&O, labels) read back from ``stock_features``.
BUILDER_KEEP = ("t_ret1", "t_ret5", "t_ret20", "t_range_prev", "t_rvol1", "t_dist_ema50", "t_rv20", "t_atr14_pct",
                "t_range_mean20")


def wilder(x: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def ema(x: pd.DataFrame | pd.Series, n: int) -> pd.DataFrame | pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev = close.shift(1)
    return np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), n)
    loss = wilder((-delta).clip(lower=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss != 0.0, 100.0).where(gain.notna() & loss.notna())


def adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_n = wilder(true_range(high, low, close), n)
    plus_di = 100.0 * wilder(plus_dm, n) / atr_n
    minus_di = 100.0 * wilder(minus_dm, n) / atr_n
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return wilder(dx, n)


def rolling_rank(x: pd.DataFrame | pd.Series, window: int, min_periods: int) -> pd.DataFrame | pd.Series:
    """Percentile rank (0..1) of the latest value within its trailing window."""
    return x.rolling(window, min_periods=min_periods).rank(pct=True)


def stock_features(
    o: pd.DataFrame, h: pd.DataFrame, low: pd.DataFrame, c: pd.DataFrame, v: pd.DataFrame, mkt_ret1: pd.Series,
    mkt_range: pd.Series, sink: Callable[[str, pd.DataFrame], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per-stock technical / volatility / volume features as of each session's close.

    Without ``sink`` the complete dict of frames is returned. With a ``sink`` every feature is handed over the moment it is
    computed and only the few frames other feature groups need (``BUILDER_KEEP``) are returned, so the ~60 float64 frames
    never exist at once (that peak alone would not fit a 1 GB server at 10 years x 220 stocks)."""
    out: dict[str, pd.DataFrame] = {}

    def put(name: str, frame: pd.DataFrame) -> pd.DataFrame:
        if sink is None or name in BUILDER_KEEP:
            out[name] = frame
        if sink is not None:
            sink(name, frame)
        return frame

    ret1 = put("t_ret1", c.pct_change(fill_method=None) * 100.0)
    ret5 = ret20 = None
    for n in (2, 5, 10, 20, 60):
        frame = put(f"t_ret{n}", c.pct_change(n, fill_method=None) * 100.0)
        ret5, ret20 = (frame if n == 5 else ret5), (frame if n == 20 else ret20)
    put("t_gap_prev", (o / c.shift(1) - 1.0) * 100.0)
    put("t_oc_prev", (c / o - 1.0) * 100.0)
    rng = put("t_range_prev", (h - low) / o * 100.0)
    put("t_exc_prev", np.maximum(h - o, o - low) / o * 100.0)

    tr = true_range(h, low, c)
    atr14, atr5 = wilder(tr, 14), wilder(tr, 5)
    put("t_atr14_pct", atr14 / c * 100.0)
    put("t_atr5_pct", atr5 / c * 100.0)
    put("t_atr_ratio", atr5 / atr14)
    del tr, atr14, atr5

    range_mean20 = put("t_range_mean20", rng.rolling(20, min_periods=10).mean())
    put("t_range_med20", rng.rolling(20, min_periods=10).median())
    put("t_range_std20", rng.rolling(20, min_periods=10).std())
    put("t_range_p90_60", rng.rolling(60, min_periods=30).quantile(0.9))
    put("t_range_exp", rng / range_mean20)
    put("t_range_rank", rolling_rank(rng, TRADING_DAYS, 60))
    put("t_nr7", (rng <= rng.rolling(7, min_periods=7).min()).astype(float).where(rng.rolling(7, min_periods=7).min().notna()))

    rv20 = None
    for n in (10, 20, 60):
        frame = put(f"t_rv{n}", ret1.rolling(n, min_periods=max(5, n // 2)).std() * np.sqrt(TRADING_DAYS))
        rv20 = frame if n == 20 else rv20
    put("t_rv_rank", rolling_rank(rv20, TRADING_DAYS, 60))
    del rv20

    ema20, ema50 = ema(c, 20), ema(c, 50)
    sma200 = c.rolling(200, min_periods=120).mean()
    put("t_dist_ema20", (c / ema20 - 1.0) * 100.0)
    put("t_dist_ema50", (c / ema50 - 1.0) * 100.0)
    put("t_dist_sma200", (c / sma200 - 1.0) * 100.0)
    put("t_ema20_slope", (ema20 / ema20.shift(5) - 1.0) * 100.0)
    del ema20, ema50, sma200
    put("t_adx14", adx(h, low, c, 14))
    sd20, ma20 = c.rolling(20, min_periods=10).std(), c.rolling(20, min_periods=10).mean()
    put("t_boll_width", 4.0 * sd20 / ma20 * 100.0)
    put("t_boll_pos", (c - ma20) / (2.0 * sd20.replace(0.0, np.nan)))
    del sd20, ma20
    put("t_rsi14", rsi(c, 14))
    macd = ema(c, 12) - ema(c, 26)
    put("t_macd_hist", (macd - ema(macd, 9)) / c * 100.0)
    del macd

    hi20, lo20 = h.rolling(20, min_periods=10).max(), low.rolling(20, min_periods=10).min()
    hi252, lo252 = h.rolling(TRADING_DAYS, min_periods=120).max(), low.rolling(TRADING_DAYS, min_periods=120).min()
    put("t_dist_hi20", (c / hi20 - 1.0) * 100.0)
    put("t_dist_lo20", (c / lo20 - 1.0) * 100.0)
    put("t_dist_hi252", (c / hi252 - 1.0) * 100.0)
    put("t_dist_lo252", (c / lo252 - 1.0) * 100.0)
    del hi20, lo20, hi252, lo252
    prior_hi, prior_lo = h.shift(1).rolling(20, min_periods=10).max(), low.shift(1).rolling(20, min_periods=10).min()
    put("t_breakout_up", (c > prior_hi).astype(float).where(prior_hi.notna()))
    put("t_breakdown", (c < prior_lo).astype(float).where(prior_lo.notna()))
    del prior_hi, prior_lo
    typical = (h + low + c) / 3.0
    put("t_vwap_dist", (c / typical - 1.0) * 100.0)  # typical-price proxy: identical in training and live
    del typical

    # Relative volume = current volume / comparable HISTORICAL volume (blueprint Appendix A): the baseline is the mean of
    # the 20 sessions BEFORE the session, so a spike does not dilute its own yardstick (10x would read as 6.9x otherwise).
    vol_base20 = v.shift(1).rolling(20, min_periods=10).mean()
    rvol1 = put("t_rvol1", v / vol_base20)
    rvol5 = v.rolling(5, min_periods=3).mean() / vol_base20
    put("t_rvol5", rvol5)
    put("t_vol_acc", rvol1 - rvol5)
    del vol_base20, rvol5
    traded_value = c * v
    tv_med = traded_value.rolling(20, min_periods=10).median() / 1e7
    put("t_tv_med20_cr", tv_med)
    put("t_log_tv", np.log10(tv_med.clip(lower=1e-3)))
    del traded_value, tv_med

    mkt1 = mkt_ret1.reindex(c.index)
    put("t_rs5", ret5.sub(mkt1.rolling(5, min_periods=5).apply(lambda a: (np.prod(1 + a / 100.0) - 1) * 100.0, raw=True), axis=0))
    put("t_rs20", ret20.sub(mkt1.rolling(20, min_periods=20).apply(lambda a: (np.prod(1 + a / 100.0) - 1) * 100.0, raw=True), axis=0))
    cov60 = ret1.rolling(60, min_periods=40).cov(mkt1)
    var60 = mkt1.rolling(60, min_periods=40).var()
    beta60 = cov60.div(var60, axis=0)
    del cov60
    put("t_beta60", beta60)
    put("t_corr60", ret1.rolling(60, min_periods=40).corr(mkt1))
    resid = ret1 - beta60.mul(mkt1, axis=0)
    put("t_idio_vol20", resid.rolling(20, min_periods=10).std())
    put("t_range_vs_mkt", rng.div(mkt_range.reindex(c.index).replace(0.0, np.nan), axis=0))
    return out
