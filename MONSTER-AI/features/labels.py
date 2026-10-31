"""Prediction targets (blueprint section 13 / Appendix A).

Primary: intraday range % of open, maximum excursion from open, threshold exceedance (1/2/3/5 %), and a
stock-specific volatility-adjusted "unusual" flag. Secondary: close-vs-open direction. A wide range does not imply
a directional close or a profitable trade; those are evaluated separately (backtesting.execution).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(
    o: pd.DataFrame, h: pd.DataFrame, low: pd.DataFrame, c: pd.DataFrame, *, thresholds: tuple[float, ...],
    unusual_k: float, baseline: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Labels for the session of each row. ``baseline`` is the trailing mean range known BEFORE the session
    (so "unusual" is relative to what was normal for that stock at prediction time)."""
    valid = o.notna() & h.notna() & low.notna() & c.notna() & (o > 0)
    rng = ((h - low) / o * 100.0).where(valid)
    exc = (np.maximum(h - o, o - low) / o * 100.0).where(valid)
    direction = ((c - o) / o * 100.0).where(valid)
    out: dict[str, pd.DataFrame] = {"y_range": rng, "y_exc": exc, "y_dir": direction}
    out["y_up"] = (c > o).astype(float).where(valid)
    for t in thresholds:
        out[f"y_hit_{_tname(t)}"] = (rng > t).astype(float).where(valid)
    ok = valid & baseline.notna() & (baseline > 0)
    out["y_range_z"] = (rng / baseline).where(ok)
    out["y_unusual"] = (rng > unusual_k * baseline).astype(float).where(ok)
    return out


def _tname(threshold: float) -> str:
    return str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "p")


def threshold_name(threshold: float) -> str:
    """Column suffix for a threshold: 3.0 -> '3', 2.5 -> '2p5' (used for y_hit_<name> and p['hit_<name>'])."""
    return _tname(threshold)
