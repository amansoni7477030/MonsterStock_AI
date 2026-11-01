"""Tiny deterministic bar builders shared by the split-basis, F&O point-in-time, survivorship and validation tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from database.store import Store

LAST_SESSION = "2026-10-30"


def sessions(n: int, last: str = LAST_SESSION) -> list[str]:
    """``n`` consecutive business days ending on ``last`` (ISO strings)."""
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=last, periods=n)]


def economic_path(n: int, seed: int = 5) -> dict[str, np.ndarray]:
    """One continuous economic path on the POST-split basis: calm, a one-percent-ish range, volume around a million shares."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.008, n))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(0.8e6, 1.2e6, n)}


def old_scale(n: int, splits: Sequence[tuple[int, float]]) -> np.ndarray:
    """Per session, how many times larger the RAW price is than the post-split basis: the product of the ratios of the splits
    (position of the ex-date, ratio) that come later. A raw volume is the post-split volume divided by it."""
    factor = np.ones(n)
    for ex_pos, ratio in splits:
        factor[:ex_pos] *= ratio
    return factor


def bar_rows(days: Sequence[str], path: dict[str, np.ndarray], positions: Sequence[int], factor: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Rows for ``positions`` as an exchange (or vendor) would store them on a scale ``factor`` times the economic one."""
    scale = np.ones(len(days)) if factor is None else factor
    return [
        {"trade_date": days[i], "open": float(path["open"][i] * scale[i]), "high": float(path["high"][i] * scale[i]),
         "low": float(path["low"][i] * scale[i]), "close": float(path["close"][i] * scale[i]),
         "volume": float(path["volume"][i] / scale[i])}
        for i in positions
    ]


def add_equity(store: Store, symbol: str = "AAA", **attrs: Any) -> int:
    attrs.setdefault("in_universe", 1)
    return store.upsert_instrument(symbol, "equity", yahoo_symbol=f"{symbol}.NS", **attrs)


def put_official(store: Store, iid: int, rows: list[dict[str, Any]]) -> None:
    store.upsert_bars(iid, rows, source="nse_bhavcopy", source_rank=2, quality="official")


def put_vendor(store: Store, iid: int, rows: list[dict[str, Any]]) -> None:
    store.upsert_bars(iid, rows, source="yahoo", source_rank=1)


def bars_snapshot(store: Store) -> list[tuple[Any, ...]]:
    """Every stored bar, verbatim: proves a read never rewrites the raw rows."""
    return [tuple(r) for r in store.conn.execute("SELECT * FROM bars_daily ORDER BY instrument_id, trade_date").fetchall()]
