"""Sector engine: leave-one-out sector aggregates so a stock's own move never leaks into its sector context.

Identifies whether a move is idiosyncratic or part of a broad sector move; a weak index can coexist with a strong
sector. Outputs are as of each session's close; the builder applies the one-session shift.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _loo_mean(block: pd.DataFrame) -> pd.DataFrame:
    values = block.to_numpy(dtype=float)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    total = filled.sum(axis=1, keepdims=True)
    count = finite.sum(axis=1, keepdims=True)
    denom = count - finite.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        loo = (total - filled) / denom
    loo[denom <= 0] = np.nan
    return pd.DataFrame(loo, index=block.index, columns=block.columns)


def _loo_std(block: pd.DataFrame) -> pd.DataFrame:
    """Per member, the population standard deviation of the OTHER members' values (NaN with fewer than two others)."""
    values = block.to_numpy(dtype=float)
    finite = np.isfinite(values)
    count = finite.sum(axis=1, keepdims=True)
    centre = np.where(finite, values, 0.0).sum(axis=1, keepdims=True) / np.maximum(count, 1)
    dev = np.where(finite, values - centre, 0.0)  # centred: the sums of squares below do not cancel catastrophically
    others = count - finite.astype(float)
    dev_sum = dev.sum(axis=1, keepdims=True) - dev
    sq_sum = (dev ** 2).sum(axis=1, keepdims=True) - dev ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        var = sq_sum / others - (dev_sum / others) ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    std[others < 2] = np.nan
    return pd.DataFrame(std, index=block.index, columns=block.columns)


def sector_loo(frame: pd.DataFrame, sectors: Mapping[str, str | None]) -> pd.DataFrame:
    """Leave-one-out sector mean of an arbitrary matrix (used for relative-gap features)."""
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns, dtype=float)
    groups: dict[str, list[str]] = defaultdict(list)
    for symbol in frame.columns:
        groups[sectors.get(symbol) or "unknown"].append(symbol)
    for members in groups.values():
        if len(members) >= 2:
            out[members] = _loo_mean(frame[members])
    return out


def sector_features(
    ret1: pd.DataFrame, ret5: pd.DataFrame, ret20: pd.DataFrame, rng: pd.DataFrame, rvol1: pd.DataFrame,
    sectors: Mapping[str, str | None], mkt_ret20: pd.Series,
) -> dict[str, pd.DataFrame]:
    names = ("s_ret1", "s_ret5", "s_ret20", "s_breadth1", "s_disp1", "s_range1", "s_rvol1", "s_rs1", "s_rs20", "s_rot20", "s_n")
    out = {n: pd.DataFrame(np.nan, index=ret1.index, columns=ret1.columns, dtype=float) for n in names}
    groups: dict[str, list[str]] = defaultdict(list)
    for symbol in ret1.columns:
        groups[sectors.get(symbol) or "unknown"].append(symbol)

    for members in groups.values():
        if len(members) < 2:
            continue
        r1, r5, r20 = ret1[members], ret5[members], ret20[members]
        out["s_ret1"][members] = _loo_mean(r1)
        out["s_ret5"][members] = _loo_mean(r5)
        out["s_ret20"][members] = _loo_mean(r20)
        up = (r1 > 0).astype(float).where(r1.notna())
        out["s_breadth1"][members] = _loo_mean(up)
        out["s_disp1"][members] = _loo_std(r1)
        out["s_range1"][members] = _loo_mean(rng[members])
        out["s_rvol1"][members] = _loo_mean(rvol1[members])
        out["s_rs1"][members] = r1 - out["s_ret1"][members]
        out["s_rs20"][members] = r20 - out["s_ret20"][members]
        out["s_rot20"][members] = out["s_ret20"][members].sub(mkt_ret20.reindex(r1.index), axis=0)
        out["s_n"][members] = float(len(members))
    return out
