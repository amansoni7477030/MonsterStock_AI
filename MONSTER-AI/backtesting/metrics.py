"""Evaluation statistics (blueprint section 15): calibration, ranking quality, intervals. Pure numpy/pandas.

Every headline number comes with a sample size and an interval; a bare percentage is never reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well behaved at small n and extreme rates)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _clean(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    return p[ok], y[ok]


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = _clean(p, y)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def brier_skill(p: np.ndarray, y: np.ndarray, base_rate: float | None = None) -> float:
    """1 - Brier / Brier(climatology). > 0 means better than always predicting the base rate."""
    p, y = _clean(p, y)
    if not len(p):
        return float("nan")
    base = float(np.mean(y)) if base_rate is None else base_rate
    ref = float(np.mean((base - y) ** 2))
    return float(1.0 - np.mean((p - y) ** 2) / ref) if ref > 0 else float("nan")


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p, y = _clean(p, y)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Equal-mass bins: mean predicted, observed frequency, count (the calibration curve)."""
    p, y = _clean(p, y)
    if len(p) < bins:
        return pd.DataFrame(columns=["bin", "mean_pred", "observed", "n"])
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, bins)
    rows = [{"bin": i, "mean_pred": float(p[idx].mean()), "observed": float(y[idx].mean()), "n": int(len(idx))}
            for i, idx in enumerate(chunks) if len(idx)]
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins."""
    table = reliability_table(p, y, bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_pred"] - table["observed"]).abs()).sum())


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via the rank-sum statistic (ties handled with average ranks)."""
    p, y = _clean(p, y)
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    y, q = _clean(y, q)
    diff = y - q
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff))) if len(y) else float("nan")


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = _clean(y, pred)
    return float(np.mean(np.abs(y - pred))) if len(y) else float("nan")


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of outcomes falling inside the predicted [lo, hi] interval (should match the nominal level)."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else float("nan")


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """Normalised discounted cumulative gain of ranking ``scores`` against graded ``relevance``."""
    scores, relevance = np.asarray(scores, dtype=float), np.asarray(relevance, dtype=float)
    ok = np.isfinite(scores) & np.isfinite(relevance)
    scores, relevance = scores[ok], relevance[ok]
    if len(scores) == 0:
        return float("nan")
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="mergesort")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((relevance[order] * discounts).sum())
    ideal = float((np.sort(relevance)[::-1][:k] * discounts).sum())
    return dcg / ideal if ideal > 0 else float("nan")


def daily_topn(frame: pd.DataFrame, score_col: str, value_col: str, n: int, *, date_col: str = "date") -> pd.DataFrame:
    """For each date select the ``n`` highest ``score_col`` rows; return per-date mean/sum of ``value_col`` and count."""
    rows = []
    for day, group in frame.groupby(date_col, sort=True):
        g = group[[score_col, value_col]].dropna(subset=[score_col])
        if len(g) < 1:
            continue
        top = g.nlargest(min(n, len(g)), score_col)
        vals = top[value_col].dropna()
        rows.append({"date": day, "mean": float(vals.mean()) if len(vals) else float("nan"), "sum": float(vals.sum()),
                     "n": int(len(vals))})
    return pd.DataFrame(rows)


def topn_hit_rate(daily: pd.DataFrame) -> tuple[float, float, float, int]:
    """Pooled hit rate over all selected names with a Wilson interval: (rate, ci_low, ci_high, n_selected)."""
    if daily.empty:
        return (float("nan"), float("nan"), float("nan"), 0)
    hits, n = float(daily["sum"].sum()), int(daily["n"].sum())
    low, high = wilson_interval(hits, n)
    return (hits / n if n else float("nan"), low, high, n)


def paired_day_bootstrap(
    model_daily: pd.DataFrame, baseline_daily: pd.DataFrame, *, n_boot: int = 1000, seed: int = 7
) -> dict[str, float]:
    """Day-level paired bootstrap of (model - baseline) mean top-N value. Resampling days respects within-day
    dependence; the 10th percentile is the one-sided 90% lower bound."""
    merged = model_daily.merge(baseline_daily, on="date", suffixes=("_m", "_b"))
    if merged.empty:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p10": float("nan"), "n_days": 0}
    diff = (merged["mean_m"] - merged["mean_b"]).to_numpy(dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p10": float("nan"), "n_days": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return {
        "mean_diff": float(diff.mean()), "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)),
        "p10": float(np.percentile(means, 10)), "n_days": int(len(diff)),
    }


def spearman_ic(frame: pd.DataFrame, score_col: str, outcome_col: str, *, date_col: str = "date") -> dict[str, float]:
    """Mean cross-sectional Spearman correlation between score and realised outcome per day (rank IC)."""
    ics = []
    for _day, group in frame.groupby(date_col):
        g = group[[score_col, outcome_col]].dropna()
        if len(g) >= 8 and g[score_col].nunique() > 1 and g[outcome_col].nunique() > 1:
            ics.append(g[score_col].rank().corr(g[outcome_col].rank()))
    if not ics:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "n_days": 0, "t_stat": float("nan")}
    arr = np.asarray(ics, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    t = float(arr.mean() / (std / math.sqrt(len(arr)))) if std and std == std else float("nan")
    return {"ic_mean": float(arr.mean()), "ic_std": std, "n_days": int(len(arr)), "t_stat": t}


def summarize_binary(p: np.ndarray, y: np.ndarray, base_rate: float | None = None) -> dict[str, float]:
    p, y = _clean(p, y)
    return {
        "n": int(len(p)), "base_rate": float(np.mean(y)) if len(y) else float("nan"), "brier": brier(p, y),
        "brier_skill": brier_skill(p, y, base_rate), "ece": ece(p, y), "auc": auc(p, y), "log_loss": log_loss(p, y),
    }


def mean_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if v == v], dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if len(arr) < 2:
        return (mean, float("nan"), float("nan"))
    half = z * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return (mean, mean - half, mean + half)
