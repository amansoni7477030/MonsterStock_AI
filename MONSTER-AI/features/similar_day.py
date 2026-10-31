"""Similar-day engine (blueprint section 12): point-in-time matching of the current market state to earlier sessions.

For each session t the market vector (overnight global moves, India context, regime) is standardised with an
expanding window (statistics through t-1 only) and compared with every EARLIER session; the K nearest sessions'
stock-level outcomes (range distribution, threshold hit rates, direction frequency) become features. Similarity is a
feature, not proof that history repeats: sample size and distance are reported so the model can discount weak matches.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DIMS_PRE: tuple[str, ...] = (
    "g_us_ret", "g_semis_ret", "g_eu_ret", "g_asia_ret", "g_oil_ret", "g_gold_ret", "g_usdinr_ret", "g_dxy_ret",
    "g_us10y_bp", "g_vix_chg", "g_vix_lvl", "m_ret1", "m_ret5", "m_vix", "m_breadth_up", "m_disp", "m_mean_range",
)
DIMS_FINAL: tuple[str, ...] = (*DIMS_PRE, "p_mkt_gap")
LABELS_USED = ("y_range", "y_exc", "y_hit_2", "y_hit_3", "y_up", "y_unusual")
WARMUP = 60


def standardize(vectors: pd.DataFrame, min_periods: int = WARMUP) -> np.ndarray:
    """Expanding z-score using statistics through the previous row; missing/warm-up values become 0 (neutral)."""
    mean = vectors.expanding(min_periods=min_periods).mean().shift(1)
    std = vectors.expanding(min_periods=min_periods).std().shift(1).replace(0.0, np.nan)
    z = ((vectors - mean) / std).clip(-5.0, 5.0)
    return np.nan_to_num(z.to_numpy(dtype=np.float32), nan=0.0)


def similar_day_features(
    vectors: pd.DataFrame, labels: Mapping[str, np.ndarray], *, k: int = 30, min_history: int = 120,
    volreg: np.ndarray | None = None, penalty: float = 1.5, rows: Sequence[int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (stock-level features (T x N), market-level features (T,)). ``rows`` limits computation (live: last row)."""
    z = standardize(vectors)
    t_total = z.shape[0]
    n_stocks = next(iter(labels.values())).shape[1]
    sq = (z.astype(np.float64) ** 2).sum(axis=1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * z.astype(np.float64) @ z.astype(np.float64).T, 0.0)).astype(np.float32)
    if volreg is not None:
        vr = np.nan_to_num(volreg.astype(np.float32), nan=1.0)
        dist = dist + penalty * np.abs(vr[:, None] - vr[None, :])

    names = {
        "x_sim_range_mean": "y_range", "x_sim_exc_mean": "y_exc", "x_sim_hit2": "y_hit_2", "x_sim_hit3": "y_hit_3",
        "x_sim_up": "y_up", "x_sim_unusual": "y_unusual",
    }
    stock_out = {n: np.full((t_total, n_stocks), np.nan, dtype=np.float32) for n in (*names, "x_sim_range_q90", "x_sim_nvalid")}
    market_out = {"x_sim_dist_mean": np.full(t_total, np.nan, dtype=np.float32), "x_sim_n": np.full(t_total, np.nan, dtype=np.float32)}

    targets = range(min_history, t_total) if rows is None else [r for r in rows if r >= min_history]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (stocks without history yet) are expected
        for t in targets:
            candidates = np.arange(WARMUP, t)
            if len(candidates) < k:
                continue
            d = dist[t, candidates]
            pick = candidates[np.argpartition(d, k - 1)[:k]]
            market_out["x_sim_dist_mean"][t] = float(dist[t, pick].mean())
            market_out["x_sim_n"][t] = float(k)
            for out_name, label in names.items():
                values = labels[label][pick]
                stock_out[out_name][t] = np.nanmean(values, axis=0)
            rng_vals = labels["y_range"][pick]
            stock_out["x_sim_range_q90"][t] = np.nanquantile(rng_vals, 0.9, axis=0)
            stock_out["x_sim_nvalid"][t] = np.isfinite(rng_vals).sum(axis=0)
    return stock_out, market_out
