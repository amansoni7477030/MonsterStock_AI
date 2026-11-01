"""Re-estimating graph edges from returns, with multiple-testing and sign-stability guards.

Pure numpy/pandas/scipy. Only returns up to ``as_of`` are used. For each candidate edge the driver is regressed on the
target at lags ``min_lag..max_lag`` and the lag with the best out-of-sample (blocked cross-validation) fit wins.
An estimate only replaces a prior when it passes every gate: enough observations, a stable sign across rolling
sub-windows, positive out-of-sample R^2 and Benjamini-Hochberg significance across *all* tested edges. A significant
estimate with the opposite sign to the prior is recorded as a ``relationship_break`` instead of silently flipping it. A break
asserts no direction (direction 0, lowest confidence): the prior sign is no longer true, the opposite one is not trusted yet.
It recovers its prior direction when a later estimate agrees with the prior again.

Alignment convention: ``driver_returns`` must be indexed by the date on which the driver value was KNOWN before the
target's session (shift US closes accordingly); lag 0 then means "known before the session".
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from database.store import Store
from event_graph.repository import BROKEN_SOURCE, MEASURED_SOURCE, Edge, GraphRepository

log = logging.getLogger(__name__)

EST_PREFIX = "graph.est:"
DECAY = 0.98
CONFIDENCE_FLOOR = 0.05
CONFIDENCE_CEILING = 0.9
LEAD_LAG_DAYS = 0.75  # a prior lag at least this long means the driver leads: lag 0 is not tested


@dataclass
class EdgeEstimate:
    best_lag: int | None
    betas: dict[int, float] = field(default_factory=dict)
    beta: float = 0.0
    std_err: float = float("inf")
    t_stat: float = 0.0
    p_value: float = 1.0  # robust p-value, Sidak-adjusted for the number of lags searched
    r2: float = 0.0
    n_obs: int = 0
    sign_stability: float = 0.0
    oos_r2: float = float("-inf")
    regime_betas: dict[str, float] = field(default_factory=dict)
    lags_tested: tuple[int, ...] = ()
    reason: str = ""
    p_raw: float = 1.0  # the unadjusted p-value of the selected lag

    @property
    def estimable(self) -> bool:
        return self.best_lag is not None


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, float] | None:
    xm, ym = float(x.mean()), float(y.mean())
    xc = x - xm
    sxx = float(xc @ xc)
    if sxx <= 1e-18:
        return None
    beta = float(xc @ (y - ym)) / sxx
    alpha = ym - beta * xm
    return alpha, beta, y - alpha - beta * x, sxx


def _robust_se(x: np.ndarray, resid: np.ndarray, sxx: float) -> float:
    """Heteroskedasticity-robust (HC1) standard error of the OLS slope."""
    n = len(x)
    xc = x - x.mean()
    var = float((xc**2) @ (resid**2)) / (sxx**2) * n / max(n - 2, 1)
    return math.sqrt(var) if var > 0 else float("inf")


def _oos_r2(x: np.ndarray, y: np.ndarray, folds: int) -> float:
    """Blocked (contiguous) K-fold out-of-sample R^2 of y ~ x against the training-mean forecast."""
    n = len(x)
    if n < folds * 20:
        return float("-inf")
    sse_model = sse_naive = 0.0
    for block in np.array_split(np.arange(n), folds):
        mask = np.ones(n, dtype=bool)
        mask[block] = False
        fit = _ols(x[mask], y[mask])
        if fit is None:
            return float("-inf")
        alpha, beta, _resid, _sxx = fit
        sse_model += float(((y[block] - alpha - beta * x[block]) ** 2).sum())
        sse_naive += float(((y[block] - y[mask].mean()) ** 2).sum())
    return 1.0 - sse_model / sse_naive if sse_naive > 0 else float("-inf")


def _sign_stability(x: np.ndarray, y: np.ndarray, sign: int, windows: Sequence[int]) -> float:
    shares = []
    for width in windows:
        if width < 30 or len(x) < width + 1:
            continue
        step = max(width // 4, 1)
        agree = total = 0
        for start in range(0, len(x) - width + 1, step):
            fit = _ols(x[start : start + width], y[start : start + width])
            if fit is None:
                continue
            total += 1
            agree += int((fit[1] > 0) == (sign > 0))
        if total >= 3:
            shares.append(agree / total)
    return float(np.mean(shares)) if shares else 0.0


def _regime_betas(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    masks = {
        "driver_up": x > 0,
        "driver_down": x < 0,
        "high_vol": np.abs(x) >= np.median(np.abs(x)),
        "low_vol": np.abs(x) < np.median(np.abs(x)),
    }
    for name, mask in masks.items():
        if int(mask.sum()) >= 60:
            fit = _ols(x[mask], y[mask])
            if fit is not None:
                out[name] = round(fit[1], 6)
    return out


def _aligned(driver: Any, target: Any) -> pd.DataFrame:
    d = driver if isinstance(driver, pd.Series) else pd.Series(np.asarray(driver, dtype=float))
    t = target if isinstance(target, pd.Series) else pd.Series(np.asarray(target, dtype=float))
    frame = pd.concat([d.rename("x"), t.rename("y")], axis=1, join="inner")
    return frame.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def estimate_edge(
    driver_returns: Any,
    target_returns: Any,
    *,
    max_lag: int = 3,
    min_lag: int = 0,
    min_obs: int = 250,
    windows: Sequence[int] = (120,),
    folds: int = 4,
    regimes: bool = True,
) -> EdgeEstimate:
    """Lagged OLS of target on driver returns; the best lag is chosen by out-of-sample fit, not in-sample fit."""
    frame = _aligned(driver_returns, target_returns)
    lags = tuple(range(max(0, min_lag), max(max_lag, min_lag) + 1))
    best: tuple[float, int, np.ndarray, np.ndarray, tuple[float, float, np.ndarray, float]] | None = None
    betas: dict[int, float] = {}
    for lag in lags:
        x = frame["x"].shift(lag)
        pair = pd.concat([x, frame["y"]], axis=1).dropna()
        if len(pair) < min_obs:
            continue
        xs, ys = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
        fit = _ols(xs, ys)
        if fit is None:
            continue
        betas[lag] = round(fit[1], 6)
        score = _oos_r2(xs, ys, folds)
        if best is None or score > best[0] + 1e-12:
            best = (score, lag, xs, ys, fit)
    if best is None:
        return EdgeEstimate(None, betas, lags_tested=lags, reason=f"fewer than {min_obs} usable observations")
    score, lag, xs, ys, (_alpha, beta, resid, sxx) = best
    n = len(xs)
    std_err = _robust_se(xs, resid, sxx)
    t_stat = beta / std_err if math.isfinite(std_err) and std_err > 0 else 0.0
    p_raw = float(2.0 * stats.t.sf(abs(t_stat), max(n - 2, 1)))
    p_value = 1.0 if p_raw >= 1.0 else -math.expm1(max(len(betas), 1) * math.log1p(-p_raw))  # Sidak over the lags searched
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 0.0
    sign = 1 if beta >= 0 else -1
    return EdgeEstimate(
        best_lag=lag, betas=betas, beta=beta, std_err=std_err, t_stat=t_stat, p_value=p_value, r2=r2, n_obs=n,
        sign_stability=_sign_stability(xs, ys, sign, windows), oos_r2=score,
        regime_betas=_regime_betas(xs, ys) if regimes else {}, lags_tested=lags, p_raw=p_raw,
    )


def shrink_towards_prior(
    estimate: EdgeEstimate, prior_sensitivity: float | None, prior_confidence: float, *, prior_sd: float | None = None
) -> tuple[float, float]:
    """Precision-weighted blend of the measured beta and the expert prior -> ``(sensitivity, confidence)``.

    The prior's spread widens as its confidence falls; the measured beta's precision comes from its robust t-stat.
    Confidence rises with the data's share of the precision and the sign stability, and never exceeds 0.9.
    """
    prior = 0.0 if prior_sensitivity is None else float(prior_sensitivity)
    if not estimate.estimable or not math.isfinite(estimate.std_err) or estimate.std_err <= 0:
        return prior, prior_confidence
    tau = prior_sd if prior_sd else max(0.5 * abs(prior), 0.15) * (1.6 - min(max(prior_confidence, 0.0), 1.0))
    data_precision, prior_precision = 1.0 / estimate.std_err**2, 1.0 / max(tau, 1e-9) ** 2
    weight = data_precision / (data_precision + prior_precision)
    sensitivity = weight * estimate.beta + (1.0 - weight) * prior
    evidence = weight * estimate.sign_stability * min(1.0, estimate.n_obs / 500.0)
    confidence = prior_confidence + (CONFIDENCE_CEILING - prior_confidence) * evidence
    return float(sensitivity), float(min(CONFIDENCE_CEILING, max(CONFIDENCE_FLOOR, confidence)))


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.10) -> list[bool]:
    """Reject mask controlling the false discovery rate at ``alpha`` across all tested edges."""
    return [bool(q <= alpha) for q in bh_adjust(pvalues)]


def _as_of_timestamp(as_of: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(as_of)
    return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp


def _prepare(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    out = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    out.index = idx.tz_convert(None) if idx.tz is not None else idx
    return out.loc[out.index.normalize() <= as_of.normalize()].sort_index()


def _label(rationale: str) -> str:
    return rationale.split(" | measured", 1)[0].split(" | RELATIONSHIP BREAK", 1)[0]


def update_graph_from_returns(
    store: Store,
    driver_returns: pd.DataFrame,
    target_returns: pd.DataFrame,
    *,
    as_of: str | date | datetime,
    max_lag: int = 3,
    min_obs: int = 250,
    alpha: float = 0.10,
    min_sign_stability: float = 0.7,
    windows: Sequence[int] = (120,),
) -> dict[str, Any]:
    """Test every edge whose ends are columns of the two frames and update the graph; returns a summary."""
    stamp = _as_of_timestamp(as_of)
    drivers = _prepare(driver_returns, stamp)
    targets = _prepare(target_returns, stamp)
    repo = GraphRepository(store)
    candidates = [
        e for e in repo.get_edges()
        if e.relation != "membership" and e.src in drivers.columns and e.dst in targets.columns
    ]
    summary: dict[str, Any] = {
        "as_of": stamp.strftime("%Y-%m-%d"), "tested": 0, "passed": 0, "updated": [], "breaks": [], "decayed": [],
        "failed_gates": [], "skipped_insufficient": [], "alpha": alpha,
    }
    records: list[tuple[Edge, EdgeEstimate, dict[str, Any]]] = []
    for edge in candidates:
        record = store.kv_get_json(EST_PREFIX + edge.key) or {}
        prior = record.get("prior") or {
            "sensitivity": edge.sensitivity, "confidence": edge.confidence, "direction": edge.direction,
            "lag_days_mean": edge.lag_days_mean,
        }
        lead = prior.get("lag_days_mean") is not None and float(prior["lag_days_mean"]) >= LEAD_LAG_DAYS
        estimate = estimate_edge(
            drivers[edge.src], targets[edge.dst], max_lag=max_lag, min_lag=1 if lead else 0, min_obs=min_obs, windows=windows
        )
        record["prior"] = prior
        records.append((edge, estimate, record))
    testable = [i for i, (_e, est, _r) in enumerate(records) if est.estimable]
    qvalues = bh_adjust([records[i][1].p_value for i in testable])
    qmap = {i: float(q) for i, q in zip(testable, qvalues, strict=True)}

    with store.tx():
        for i, (edge, est, record) in enumerate(records):
            summary["tested"] += 1
            prior = record["prior"]
            if not est.estimable:
                summary["skipped_insufficient"].append(edge.key)
                continue
            q = qmap[i]
            passed = (
                est.n_obs >= min_obs and est.sign_stability >= min_sign_stability and est.oos_r2 > 0 and q <= alpha
            )
            previous_as_of = record.get("as_of")
            record.update({
                "as_of": summary["as_of"], "best_lag": est.best_lag, "beta": round(est.beta, 6),
                "t": round(est.t_stat, 3), "p": est.p_value, "q": q, "r2": round(est.r2, 5),
                "oos_r2": round(est.oos_r2, 5), "n_obs": est.n_obs, "sign_stability": round(est.sign_stability, 3),
                "regime_betas": est.regime_betas, "passed": passed,
            })
            if not passed:
                if previous_as_of != summary["as_of"]:
                    repo.upsert_edge(
                        edge.src, edge.dst, edge.relation, direction=edge.direction, sensitivity=edge.sensitivity,
                        lag_days_mean=edge.lag_days_mean, lag_days_sd=edge.lag_days_sd,
                        confidence=max(CONFIDENCE_FLOOR, edge.confidence * DECAY), regime_dependence=edge.regime_dependence,
                        rationale=edge.rationale, source=edge.source, evidence_count=edge.evidence_count,
                    )
                    summary["decayed"].append(edge.key)
                summary["failed_gates"].append(edge.key)
                store.kv_set_json(EST_PREFIX + edge.key, record)
                continue
            summary["passed"] += 1
            prior_dir = int(prior.get("direction") or 0)
            measured_sign = 1 if est.beta > 0 else -1
            base = _label(edge.rationale)
            detail = f"beta {est.beta:+.2f} at lag {est.best_lag}d, t={est.t_stat:.1f}, n={est.n_obs}, as of {summary['as_of']}"
            if prior_dir != 0 and measured_sign != prior_dir:
                # the relationship stopped working: keep the evidence, but assert no direction at all (neither the stale
                # prior sign nor a silent flip) and the lowest confidence; the prior stays on record for a later recovery
                repo.upsert_edge(
                    edge.src, edge.dst, edge.relation, direction=0, sensitivity=None,
                    lag_days_mean=edge.lag_days_mean, lag_days_sd=edge.lag_days_sd, confidence=CONFIDENCE_FLOOR,
                    regime_dependence=edge.regime_dependence,
                    rationale=f"{base} | RELATIONSHIP BREAK: prior sign {prior_dir:+d} but measured {detail}; no directional claim",
                    source=BROKEN_SOURCE, evidence_count=est.n_obs,
                )
                summary["breaks"].append(edge.key)
                log.warning(
                    "graph relationship break %s: prior sign %+d but measured %s; the edge asserts no direction", edge.key,
                    prior_dir, detail,
                )
            else:
                sensitivity, confidence = shrink_towards_prior(
                    est, prior.get("sensitivity"), float(prior.get("confidence") or edge.confidence)
                )
                repo.upsert_edge(
                    edge.src, edge.dst, edge.relation, direction=prior_dir or measured_sign, sensitivity=round(sensitivity, 6),
                    lag_days_mean=float(est.best_lag or 0), lag_days_sd=max(0.5, edge.lag_days_sd or 0.5),
                    confidence=round(confidence, 4), regime_dependence=edge.regime_dependence,
                    rationale=f"{base} | measured {detail}", source=MEASURED_SOURCE, evidence_count=est.n_obs,
                )
                summary["updated"].append(edge.key)
            store.kv_set_json(EST_PREFIX + edge.key, record)
    return summary
