"""Synthetic but structured market data for feature/model tests (no network, deterministic)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from features.builder import Panel
from ingestion.series import SERIES

SECTORS = ["Financial Services", "Information Technology", "Oil Gas & Consumable Fuels", "Healthcare"]


def make_panel(
    n_dates: int = 420, n_symbols: int = 24, seed: int = 7, start: str = "2023-01-02", with_fno: bool = True,
    with_meetings: bool = True, results_boost: float = 1.9,
) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_dates)
    symbols = [f"S{i:02d}" for i in range(n_symbols)]

    # market volatility regime (clustered) and an overnight US factor that really transmits to India next day
    regime = np.zeros(n_dates)
    for t in range(1, n_dates):
        regime[t] = 0.97 * regime[t - 1] + rng.normal(0, 0.12)
    mkt_vol = 0.007 * np.exp(0.8 * regime)
    us_ret = rng.normal(0, 0.009, n_dates)
    mkt_ret = 0.25 * np.concatenate([[0.0], us_ret[:-1]]) + rng.normal(0, 1, n_dates) * mkt_vol

    meet = np.zeros((n_dates, n_symbols))
    if with_meetings:
        for j in range(n_symbols):
            for t in range(20 + 3 * j, n_dates - 3, 63):
                meet[t, j] = 1.0

    beta = rng.uniform(0.6, 1.4, n_symbols)
    base_vol = rng.uniform(0.010, 0.018, n_symbols)
    sector_of = {s: SECTORS[j % len(SECTORS)] for j, s in enumerate(symbols)}
    sector_shock = {sec: rng.normal(0, 0.004, n_dates) for sec in SECTORS}

    o = np.zeros((n_dates, n_symbols))
    h = np.zeros_like(o)
    lo = np.zeros_like(o)
    c = np.zeros_like(o)
    v = np.zeros_like(o)
    price = rng.uniform(300, 3000, n_symbols)
    for t in range(n_dates):
        for j, s in enumerate(symbols):
            vol = base_vol[j] * np.exp(0.8 * regime[t])
            day_boost = results_boost if t > 0 and meet[t - 1, j] else 1.0
            gap = rng.normal(0, 0.25 * vol) + 0.1 * mkt_ret[t]
            open_ = price[j] * (1 + gap)
            drift = beta[j] * mkt_ret[t] + sector_shock[sector_of[s]][t] + rng.normal(0, vol * day_boost)
            close = open_ * (1 + drift)
            wick_up = abs(rng.normal(0, 0.5 * vol * day_boost))
            wick_dn = abs(rng.normal(0, 0.5 * vol * day_boost))
            o[t, j], c[t, j] = open_, close
            h[t, j] = max(open_, close) * (1 + wick_up)
            lo[t, j] = min(open_, close) * (1 - wick_dn)
            v[t, j] = rng.lognormal(13.0, 0.3) * (1 + 3 * abs(drift) / vol) * (1.5 if day_boost > 1 else 1.0)
            price[j] = close

    cols = symbols
    idx = pd.DatetimeIndex(dates)
    O, H, L, C, V = (pd.DataFrame(a, index=idx, columns=cols) for a in (o, h, lo, c, v))
    nifty_c = (1 + pd.Series(mkt_ret, index=idx)).cumprod() * 20000
    nifty = pd.DataFrame({"open": nifty_c.shift(1).fillna(20000) * (1 + rng.normal(0, 0.002, n_dates)),
                          "high": nifty_c * 1.004, "low": nifty_c * 0.996, "close": nifty_c})
    nifty["high"] = nifty[["open", "high", "close"]].max(axis=1)
    nifty["low"] = nifty[["open", "low", "close"]].min(axis=1)
    vix = pd.Series(13 + 40 * mkt_vol + rng.normal(0, 0.6, n_dates), index=idx)

    gdates = pd.bdate_range(end=dates[-1] + pd.Timedelta(days=1), periods=n_dates + 8)
    closes: dict[str, pd.Series] = {}
    for spec in SERIES:
        if spec.key in ("^NSEI", "^NSEBANK", "^INDIAVIX"):
            continue
        base = {"^GSPC": 4500, "^IXIC": 14000, "^TNX": 4.0, "^VIX": 16.0}.get(spec.key, 100.0)
        steps = rng.normal(0, 0.008, len(gdates))
        if spec.key == "^GSPC":
            steps = np.concatenate([np.full(len(gdates) - len(us_ret), 0.0), us_ret])
            steps = steps[-len(gdates):]
        series = base * np.exp(np.cumsum(steps))
        if spec.key == "^TNX":
            series = base + np.cumsum(rng.normal(0, 0.03, len(gdates)))
        closes[spec.key] = pd.Series(series, index=gdates)

    fno = None
    if with_fno:
        rows = []
        for t, d in enumerate(dates):
            for j in range(n_symbols // 2):
                s = symbols[j]
                vol = base_vol[j] * np.exp(0.8 * regime[t])
                expiry = (d + pd.offsets.BMonthEnd(0)).date()
                dte = (expiry - d.date()).days
                oi = 1e6 * (1 + 0.2 * np.sin(t / 9 + j))
                rows.append({
                    "symbol": s, "trade_date": d, "underlying": c[t, j], "lot_size": 500, "fut_close": c[t, j] * 1.004,
                    "fut_settle": c[t, j] * 1.004, "fut_oi": oi, "fut_oi_chg": oi * rng.normal(0, 0.03), "fut_volume": oi * 0.01,
                    "fut_oi_all": oi * 1.1, "fut_oi_chg_all": oi * rng.normal(0, 0.03), "basis_pct": 0.4 + rng.normal(0, 0.1),
                    "rollover_pct": max(0.0, 100 - dte * 3.0) * 0.1, "call_oi": oi * 0.7, "put_oi": oi * 0.6,
                    "call_oi_chg": oi * rng.normal(0, 0.02), "put_oi_chg": oi * rng.normal(0, 0.02), "pcr_oi": 0.85 + rng.normal(0, 0.1),
                    "pcr_vol": 0.7 + rng.normal(0, 0.1), "iv_atm": 100 * vol * np.sqrt(252) * (1 + rng.normal(0, 0.05)),
                    "iv_skew": rng.normal(1.0, 0.5), "max_call_oi_strike": c[t, j] * 1.05, "max_put_oi_strike": c[t, j] * 0.95, "dte": dte,
                })
        fno = pd.DataFrame(rows)

    meetings = None
    cov_start = None
    if with_meetings:
        mrows = [{"symbol": symbols[j], "meeting_date": dates[t].date().isoformat(), "purpose": "Financial Results", "is_results": 1}
                 for t in range(n_dates) for j in range(n_symbols) if meet[t, j]]
        meetings = pd.DataFrame(mrows)
        cov_start = dates[0].date()

    next_after = (dates[-1] + pd.offsets.BDay(1)).date()
    future = [(dates[-1] + pd.offsets.BDay(k)).date() for k in range(1, 6)]
    return Panel(
        dates=idx, symbols=symbols, o=O, h=H, low=L, c=C, v=V, nifty=nifty, vix=vix, closes=closes, sectors=sector_of,
        is_fno={s: int(j < n_symbols // 2) for j, s in enumerate(symbols)}, fno=fno, meetings=meetings,
        meetings_coverage_start=cov_start, next_after_last=next_after, future_dates=future,
    )


def copy_panel(panel: Panel) -> Panel:
    import copy

    return Panel(
        dates=panel.dates, symbols=list(panel.symbols), o=panel.o.copy(), h=panel.h.copy(), low=panel.low.copy(),
        c=panel.c.copy(), v=panel.v.copy(), nifty=panel.nifty.copy(), vix=panel.vix.copy(),
        closes={k: s.copy() for k, s in panel.closes.items()}, sectors=dict(panel.sectors), is_fno=dict(panel.is_fno),
        fno=None if panel.fno is None else panel.fno.copy(), meetings=None if panel.meetings is None else panel.meetings.copy(),
        meetings_coverage_start=panel.meetings_coverage_start, next_after_last=panel.next_after_last,
        future_dates=copy.copy(panel.future_dates),
    )


def day_after(d: date) -> date:
    return d + timedelta(days=1)
