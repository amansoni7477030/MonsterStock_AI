"""Similar-day engine (blueprint section 12): point-in-time matching of the current market state to earlier sessions.

For each session t the market vector (overnight global moves, India context, regime) is standardised over the previous
``STD_WINDOW`` sessions (statistics through t-1 only) and compared with the most recent ``POOL`` sessions BEFORE t; the K
nearest sessions' stock-level outcomes (range distribution, threshold hit rates, direction frequency) become features.
Similarity is a feature, not proof that history repeats: the number of matches, their distance and the number with a known
outcome for the stock are reported so a model can discount weak matches.

Training and live scans must see exactly the same sessions or the features (and the uncertainty that uses the match
distance) drift apart between them. Both pool and statistics are therefore fixed-length windows, not "everything so far",
and ``HISTORY_SESSIONS`` is the history a caller must supply for a date to give the same answer as the full history: the
live loader (features.live) sizes its window from it.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from features.technical import TRADING_DAYS

DIMS_PRE: tuple[str, ...] = (
    "g_us_ret", "g_semis_ret", "g_eu_ret", "g_asia_ret", "g_oil_ret", "g_gold_ret", "g_usdinr_ret", "g_dxy_ret",
    "g_us10y_bp", "g_vix_chg", "g_vix_lvl", "m_ret1", "m_ret5", "m_vix", "m_breadth_up", "m_disp", "m_mean_range",
)
DIMS_FINAL: tuple[str, ...] = (*DIMS_PRE, "p_mkt_gap")
LABELS_USED = ("y_range", "y_exc", "y_hit_2", "y_hit_3", "y_up", "y_unusual")
K = 30  # nearest sessions used
MIN_HISTORY = 120  # first row that gets features
WARMUP = 60  # sessions before a row has statistics to be standardised with (and so before it can be a candidate)
POOL = 250  # candidates: the most recent POOL sessions strictly before the date
STD_WINDOW = 250  # z-scores of a session use the STD_WINDOW sessions before it
PENALTY = 1.5  # added distance per step of difference between the volatility regimes (m_volreg) of two sessions
# Sessions before a date that its features can depend on: the pool, plus whichever is deeper of the statistics behind the oldest
# candidate and the volatility regime of that candidate (it ranks the VIX over a year, and its fallback a 20-session volatility)
# - with a few sessions of margin for the 5-session changes inside the vectors.
HISTORY_SESSIONS = POOL + max(STD_WINDOW, TRADING_DAYS + 20) + 10


def standardize(vectors: pd.DataFrame, min_periods: int = WARMUP, window: int = STD_WINDOW) -> np.ndarray:
    """Rolling z-score using the ``window`` rows before each row (statistics through the previous row); missing/warm-up
    values become 0 (neutral)."""
    rolling = vectors.rolling(window, min_periods=min_periods)
    mean = rolling.mean().shift(1)
    std = rolling.std().shift(1).replace(0.0, np.nan)
    z = ((vectors - mean) / std).clip(-5.0, 5.0)
    return np.nan_to_num(z.to_numpy(dtype=np.float32), nan=0.0)


def similar_day_features(
    vectors: pd.DataFrame, labels: Mapping[str, np.ndarray], *, k: int = K, min_history: int = MIN_HISTORY,
    volreg: np.ndarray | None = None, penalty: float = PENALTY, rows: Sequence[int] | None = None, pool: int = POOL,
    window: int = STD_WINDOW, warmup: int = WARMUP,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (stock-level features (T x N), market-level features (T,)). ``rows`` limits computation (live: last row).

    ``x_sim_n`` is the number of matches used and ``x_sim_nvalid`` (per stock) how many of them have a known outcome for the
    stock, i.e. the sample behind its x_sim_* statistics."""
    z = standardize(vectors, min(warmup, window), window).astype(np.float64)
    t_total = z.shape[0]
    n_stocks = next(iter(labels.values())).shape[1]
    vr = None if volreg is None else np.nan_to_num(np.asarray(volreg, dtype=np.float64), nan=1.0)

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
            first = max(warmup, t - pool)
            if t - first < k:
                continue
            candidates = np.arange(first, t)
            d = np.sqrt(((z[candidates] - z[t]) ** 2).sum(axis=1))
            if vr is not None:
                d = d + penalty * np.abs(vr[candidates] - vr[t])
            nearest = np.argpartition(d, k - 1)[:k]
            pick = candidates[nearest]
            market_out["x_sim_dist_mean"][t] = float(d[nearest].mean())
            market_out["x_sim_n"][t] = float(k)
            for out_name, label in names.items():
                values = labels[label][pick]
                stock_out[out_name][t] = np.nanmean(values, axis=0)
            rng_vals = labels["y_range"][pick]
            stock_out["x_sim_range_q90"][t] = np.nanquantile(rng_vals, 0.9, axis=0)
            stock_out["x_sim_nvalid"][t] = np.isfinite(rng_vals).sum(axis=0)
    return stock_out, market_out
