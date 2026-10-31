"""Calendar, expiry and scheduled-results features. Everything here is known ahead of time (exchange calendar, contract
expiry dates, announced board meetings), so it is point-in-time safe without shifting."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def calendar_market_features(dates: pd.DatetimeIndex, next_after_last: date | None = None) -> dict[str, np.ndarray]:
    """Market-level calendar features for each trading date (``next_after_last``: next trading day after the final date)."""
    n = len(dates)
    py = [d.date() for d in dates]
    nxt = py[1:] + [next_after_last]
    prev = [None] + py[:-1]
    gap_prev = np.array([(d - p).days if p else 1 for d, p in zip(py, prev, strict=True)], dtype=float)
    gap_next = np.array([(x - d).days if x else np.nan for d, x in zip(py, nxt, strict=True)], dtype=float)
    month_end = np.array([float(x.month != d.month) if x else np.nan for d, x in zip(py, nxt, strict=True)], dtype=float)
    month_start = np.array([float(p is None or p.month != d.month) if p else 0.0 for d, p in zip(py, prev, strict=True)], dtype=float)
    return {
        "c_dow": np.array([d.weekday() for d in py], dtype=float),
        "c_dom": np.array([d.day for d in py], dtype=float),
        "c_month_end": month_end,
        "c_month_start": month_start,
        "c_gap_prev": gap_prev,
        "c_gap_next": gap_next,
        "c_n": np.full(n, float(n)),
    }


def results_features(
    meetings: pd.DataFrame | None, dates: pd.DatetimeIndex, future: list[date], symbols: list[str], coverage_start: date | None,
) -> dict[str, pd.DataFrame]:
    """Scheduled quarterly-results features from board-meeting dates.

    Dates before ``coverage_start`` are unknown (NaN) rather than "no results", so a model is never taught that an
    uncovered period had no catalysts."""
    n = len(dates)
    extended = [d.date() for d in dates] + list(future)
    pos = {d: i for i, d in enumerate(extended)}
    flags = np.zeros((len(extended), len(symbols)))
    sym_pos = {s: i for i, s in enumerate(symbols)}
    if meetings is not None and len(meetings):
        for row in meetings.itertuples(index=False):
            if not getattr(row, "is_results", 0):
                continue
            i = pos.get(date.fromisoformat(str(row.meeting_date)[:10]))
            j = sym_pos.get(row.symbol)
            if i is not None and j is not None:
                flags[i, j] = 1.0
    if coverage_start is None:
        known = np.zeros(n, dtype=bool)
    else:
        known = np.array([d.date() >= coverage_start for d in dates])
    frame = pd.DataFrame(flags, index=range(len(extended)), columns=symbols)

    def view(shifted: pd.DataFrame) -> pd.DataFrame:
        out = shifted.iloc[:n].copy()
        out.index = dates
        out[~known] = np.nan
        return out

    today = view(frame)
    prev = view(frame.shift(1))
    next2 = view(pd.concat([frame.shift(-1), frame.shift(-2)]).groupby(level=0).max())
    idx = np.arange(len(extended))[:, None]
    marker = pd.DataFrame(np.where(flags > 0, idx, np.nan), columns=symbols)
    since = view(idx - marker.ffill().shift(1))
    until = view(marker.bfill().shift(-1) - idx)
    return {
        "c_results_today": today,
        "c_results_prev": prev,
        "c_results_next2": next2,
        "c_days_since_results": since.clip(upper=120),
        "c_days_to_results": until.clip(upper=60),
    }


def expiry_features(exp_ord: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """Expiry-day regime from the near-expiry date known at the previous session (date + dte)."""
    ordinals = np.array([d.toordinal() for d in dates], dtype=float)[:, None]
    known = exp_ord.shift(1)
    expiring = (known == ordinals).astype(float).where(known.notna())
    days_left = (known - ordinals).where(known >= ordinals)
    post = (known < ordinals).astype(float).where(known.notna())
    share = expiring.mean(axis=1, skipna=True).to_numpy(dtype=float)
    return {"c_expiry_today": expiring, "c_dte_cal": days_left, "c_post_expiry": post}, share


def next_trading_days(after: date, count: int, is_trading) -> list[date]:
    out: list[date] = []
    cursor = after
    while len(out) < count:
        cursor += timedelta(days=1)
        if is_trading(cursor):
            out.append(cursor)
    return out
