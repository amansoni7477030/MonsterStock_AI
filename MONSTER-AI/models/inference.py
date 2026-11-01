"""Inference: trained bundle (specialists -> meta -> calibration) and a transparent heuristic baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from app.errors import ModelArtifactError
from features.labels import threshold_name
from models import gbm
from models.artifacts import read_bundle
from models.calibration import Calibrator

STATUS_BASELINE = "baseline"


def _matrix(X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Columns in the model's order; features absent from ``X`` become NaN (a missing feed never crashes inference)."""
    cols = [X[c].to_numpy(dtype=np.float32) if c in X.columns else np.full(len(X), np.nan, dtype=np.float32) for c in columns]
    return np.column_stack(cols) if cols else np.empty((len(X), 0), dtype=np.float32)


def make_meta_frame(sp: dict[str, np.ndarray], X: pd.DataFrame, meta_context: list[str]) -> pd.DataFrame:
    """Meta-model inputs: specialist outputs + their agreement + market context (shared by training and inference)."""
    frame = pd.DataFrame(sp, index=X.index)
    probs = [v for k, v in sp.items() if k.endswith("_unusual")]
    if probs:
        stack = np.vstack(probs)
        with np.errstate(all="ignore"):
            frame["sp_mean_unusual"] = np.nanmean(stack, axis=0)
            counts = np.isfinite(stack).sum(axis=0)
            std = np.nanstd(stack, axis=0)
        frame["sp_std_unusual"] = np.where(counts >= 2, std, np.nan)
    for col in meta_context:
        frame[col] = X[col].to_numpy(dtype=np.float64) if col in X.columns else np.nan
    return frame


def drift_share(values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per row, the share of known feature values outside the training band [lo, hi] (0 when nothing is known)."""
    known = np.isfinite(values)
    outside = (values < lo) | (values > hi)
    with np.errstate(all="ignore"):
        return (outside & known).sum(axis=1) / np.maximum(known.sum(axis=1), 1)


def combine_uncertainty(
    spec_std: np.ndarray, spread: np.ndarray, dist: np.ndarray, drift: np.ndarray, scales: dict[str, Any],
) -> np.ndarray:
    """0 (reliable) .. 1 (unreliable); NOT a probability. 0.35 x specialist disagreement + 0.25 x forecast-interval width
    + 0.20 x similar-day distance + 0.20 x feature drift, each divided by its out-of-sample 95th percentile (``scales``) and
    clipped to [0, 1]. A missing disagreement or distance counts as the middle (0.5). Shared by inference and training, which
    uses it to learn what is normal for the model's own top picks."""
    a = np.where(np.isfinite(spec_std), np.clip(spec_std / max(scales.get("spec_std_p95", 0.1), 1e-6), 0, 1), 0.5)
    b = np.clip(spread / max(scales.get("spread_p95", 2.0), 1e-6), 0, 1)
    lo, hi = scales.get("dist_median", 0.0), scales.get("dist_p95", 1.0)
    c = np.where(np.isfinite(dist), np.clip((dist - lo) / max(hi - lo, 1e-6), 0, 1), 0.5)
    d = np.clip(drift / max(scales.get("drift_p95", 0.2), 1e-6), 0, 1)
    return np.clip(0.35 * a + 0.25 * b + 0.20 * c + 0.20 * d, 0.0, 1.0)


def pick_uncertainty_ceilings(meta: dict[str, Any]) -> tuple[float, float] | None:
    """(STRONG, MODERATE) uncertainty ceilings learned from this model's own out-of-sample daily top picks, or ``None`` for a
    model without them (older artifacts, the heuristic baseline): the caller then falls back to the configured constants."""
    scales = meta.get("unc_scales") or {}
    strong, moderate = scales.get("pick_p50"), scales.get("pick_p85")
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (strong, moderate))
    if numeric and 0.0 < strong <= moderate <= 1.0:
        return float(strong), float(moderate)
    return None


class ModelBundle:
    """A trained, calibrated model for one stage. ``predict`` is pure and deterministic."""

    def __init__(self, meta: dict[str, Any], boosters: dict[str, Any], calibrators: dict[str, Calibrator],
                 stats: dict[str, Any], directory: Path | None = None) -> None:
        self.meta, self.boosters, self.calibrators, self.stats, self.directory = meta, boosters, calibrators, stats, directory

    @classmethod
    def load(cls, directory: Path, *, verify: bool = True) -> ModelBundle:
        meta, boosters, calibrators, stats, _ = read_bundle(directory, verify=verify)
        return cls(meta, boosters, calibrators, stats, Path(directory))

    # ------------------------------------------------------------------------------------- properties
    @property
    def version(self) -> str:
        return str(self.meta["version"])

    @property
    def stage(self) -> str:
        return str(self.meta["stage"])

    @property
    def thresholds(self) -> list[float]:
        return [float(t) for t in self.meta["thresholds"]]

    @property
    def primary_threshold(self) -> float:
        return float(self.meta["primary_threshold"])

    @property
    def primary_head(self) -> str:
        return f"hit_{threshold_name(self.primary_threshold)}"

    @property
    def status(self) -> str:
        return str(self.meta.get("status", "shadow"))

    # ------------------------------------------------------------------------------------------ predict
    def specialist_outputs(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, spec in self.meta["specialists"].items():
            matrix = _matrix(X, spec["columns"])
            mask = np.ones(len(X), dtype=bool)
            for required in spec.get("requires", []):
                mask &= X[required].notna().to_numpy() if required in X.columns else np.zeros(len(X), dtype=bool)
            for target in spec["targets"]:
                pred = gbm.predict(self.boosters[f"sp:{name}:{target}"], matrix).astype(np.float64)
                pred[~mask] = np.nan
                out[f"sp_{name}_{target}"] = pred
        return out

    def _meta_frame(self, X: pd.DataFrame, sp: dict[str, np.ndarray]) -> pd.DataFrame:
        return make_meta_frame(sp, X, self.meta["meta_context"])

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        sp = self.specialist_outputs(X)
        mf = self._meta_frame(X, sp)
        meta_matrix = _matrix(mf, self.meta["meta_columns"])
        out = pd.DataFrame(index=X.index)

        heads = self.meta["heads"]["cls"]
        raw = {h: gbm.predict(self.boosters[f"meta:{h}"], meta_matrix).astype(np.float64) for h in heads}
        cal = {h: self.calibrators[h].transform(raw[h]) for h in heads}
        hit_names = [f"hit_{threshold_name(t)}" for t in self.thresholds]
        hits = np.vstack([cal[h] for h in hit_names])
        hits = np.minimum.accumulate(hits, axis=0)  # P(range > 5%) can never exceed P(range > 3%)
        for i, h in enumerate(hit_names):
            out[f"p_{h}"] = hits[i]
        out["p_unusual"] = cal["unusual"]
        out["p_primary"] = out[f"p_{self.primary_head}"]
        out["p_raw_primary"] = raw[self.primary_head]
        if "sp_std_unusual" in mf:
            out["sp_std_unusual"] = mf["sp_std_unusual"].to_numpy()
            out["sp_mean_unusual"] = mf["sp_mean_unusual"].to_numpy()

        up_raw = sp.get("sp_direction_up")
        if up_raw is not None and "up" in self.calibrators:
            p_up = self.calibrators["up"].transform(np.where(np.isfinite(up_raw), up_raw, 0.5))
            p_up = np.where(np.isfinite(up_raw), p_up, np.nan)
        else:
            p_up = np.full(len(X), np.nan)
        flat = float(self.meta.get("flat_rate", 0.0))
        out["p_up"] = p_up
        out["p_down"] = np.where(np.isfinite(p_up), np.clip(1.0 - p_up - flat, 0.0, 1.0), np.nan)

        q = {}
        for group, alphas in self.meta["heads"]["quant"].items():
            vals = np.vstack([np.exp(gbm.predict(self.boosters[f"meta:{group}_q{int(a * 100)}"], meta_matrix)) for a in alphas])
            vals = np.sort(vals, axis=0)
            for i, a in enumerate(alphas):
                q[f"{group}_q{int(a * 100)}"] = vals[i]
        out["rng_q10"], out["rng_q50"], out["rng_q90"] = q["range_q10"], q["range_q50"], q["range_q90"]
        out["exc_q50"], out["exc_q90"] = q["exc_q50"], q["exc_q90"]

        out["uncertainty"] = self._uncertainty(X, mf, out)
        for key, values in sp.items():
            out[key] = values
        return out

    # ----------------------------------------------------------------------------------- uncertainty
    def _uncertainty(self, X: pd.DataFrame, mf: pd.DataFrame, out: pd.DataFrame) -> np.ndarray:
        """0 (reliable) .. 1 (unreliable). NOT a probability; see ``combine_uncertainty`` for the formula."""
        std = mf["sp_std_unusual"].to_numpy(dtype=float) if "sp_std_unusual" in mf else np.full(len(X), np.nan)
        spread = ((out["rng_q90"] - out["rng_q10"]) / out["rng_q50"].clip(lower=1e-6)).to_numpy(dtype=float)
        dist = X["x_sim_dist_mean"].to_numpy(dtype=float) if "x_sim_dist_mean" in X.columns else np.full(len(X), np.nan)
        return combine_uncertainty(std, spread, dist, self._drift(X), self.meta.get("unc_scales", {}))

    def _drift(self, X: pd.DataFrame) -> np.ndarray:
        stats = self.stats.get("features", {})
        names = [n for n in stats if n in X.columns and stats[n].get("p99") is not None]
        if not names:
            return np.zeros(len(X))
        lo = np.array([stats[n]["p01"] for n in names])
        hi = np.array([stats[n]["p99"] for n in names])
        return drift_share(X[names].to_numpy(dtype=np.float64), lo, hi)

    # ------------------------------------------------------------------------------------------ explain
    def explain(self, X: pd.DataFrame, position: int, top: int = 3) -> list[str]:
        """Diagnostic drivers: which specialists and raw features moved the primary probability (not causal proof)."""
        sp = self.specialist_outputs(X.iloc[[position]])
        mf = self._meta_frame(X.iloc[[position]], sp)
        cols = self.meta["meta_columns"]
        contrib = gbm.contributions(self.boosters[f"meta:{self.primary_head}"], _matrix(mf, cols))[0][:-1]
        order = np.argsort(-np.abs(contrib))[:top]
        drivers = []
        for idx in order:
            name = cols[idx]
            if abs(contrib[idx]) < 1e-3:
                continue
            drivers.append(f"{_pretty(name)} {'raises' if contrib[idx] > 0 else 'lowers'} the estimate ({contrib[idx]:+.2f} logit)")
        best = next((cols[i] for i in order if cols[i].startswith("sp_") and cols[i].endswith("_unusual")), None)
        if best:
            spec = best[3:-8]
            spec_meta = self.meta["specialists"].get(spec)
            if spec_meta:
                matrix = _matrix(X.iloc[[position]], spec_meta["columns"])
                c2 = gbm.contributions(self.boosters[f"sp:{spec}:unusual"], matrix)[0][:-1]
                names = [spec_meta["columns"][i] for i in np.argsort(-np.abs(c2))[:2]]
                drivers.append(f"within {spec}: " + ", ".join(names))
        return drivers[: top + 1]


def _pretty(column: str) -> str:
    if column.startswith("sp_"):
        parts = column.split("_")
        return f"{parts[1]} specialist ({'/'.join(parts[2:])})"
    return column


class BaselineBundle:
    """Transparent heuristic used before any model is trained or validated (always labelled BASELINE, tiers capped).

    Range is modelled as lognormal around ``m * mean20_range`` with sigma 0.45; a pre-open gap widens it. These are
    uncalibrated heuristics, shown to the user as such - they exist so the pipeline stays alive and honest.
    """

    SIGMA = 0.45

    def __init__(self, stage: str, thresholds: tuple[float, ...], primary_threshold: float) -> None:
        self.meta = {"version": f"baseline-{stage}", "stage": stage, "thresholds": list(thresholds),
                     "primary_threshold": primary_threshold, "status": STATUS_BASELINE}
        self.directory = None

    version = property(lambda self: self.meta["version"])
    stage = property(lambda self: self.meta["stage"])
    status = property(lambda self: STATUS_BASELINE)
    thresholds = property(lambda self: [float(t) for t in self.meta["thresholds"]])
    primary_threshold = property(lambda self: float(self.meta["primary_threshold"]))
    primary_head = property(lambda self: f"hit_{threshold_name(self.primary_threshold)}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        base = X["t_range_mean20"] if "t_range_mean20" in X else pd.Series(np.nan, index=X.index)
        atr = X["t_atr14_pct"] if "t_atr14_pct" in X else pd.Series(np.nan, index=X.index)
        normal = base.fillna(atr * 0.9).clip(lower=0.2)
        center = normal
        if "p_gap_atr" in X and self.stage == "final":
            center = normal * (1.0 + 0.25 * X["p_gap_atr"].abs().clip(upper=3.0).fillna(0.0))
        mu = np.log(center.to_numpy(dtype=float))
        mu_normal = np.log(normal.to_numpy(dtype=float))
        out = pd.DataFrame(index=X.index)
        for t in self.thresholds:
            out[f"p_hit_{threshold_name(t)}"] = 1.0 - ndtr((math.log(t) - mu) / self.SIGMA)
        out["p_unusual"] = 1.0 - ndtr((math.log(1.5) + mu_normal - mu) / self.SIGMA)
        out["p_primary"] = out[f"p_{self.primary_head}"]
        out["p_up"] = np.nan  # the heuristic has no directional view: report n/a rather than a fake 50/50
        out["p_down"] = np.nan
        mean = np.exp(mu)
        out["rng_q10"], out["rng_q50"], out["rng_q90"] = mean * 0.55, mean * 0.97, mean * 1.9
        out["exc_q50"], out["exc_q90"] = out["rng_q50"] * 0.7, out["rng_q90"] * 0.7
        out["uncertainty"] = 0.8
        return out

    def explain(self, X: pd.DataFrame, position: int, top: int = 3) -> list[str]:
        return ["heuristic baseline: recent average range" + (" widened by the pre-open gap" if self.stage == "final" else "")]


Bundle = ModelBundle | BaselineBundle


def check_columns(bundle: ModelBundle, X: pd.DataFrame) -> list[str]:
    """Names of model input columns missing from ``X`` (each is treated as NaN by the model)."""
    wanted = set(bundle.meta.get("feature_columns", []))
    return sorted(wanted - set(X.columns))


def require_schema(bundle: ModelBundle, feature_version: str) -> None:
    if bundle.meta.get("feature_version") != feature_version:
        raise ModelArtifactError(
            f"model feature version {bundle.meta.get('feature_version')} != code version {feature_version}")
