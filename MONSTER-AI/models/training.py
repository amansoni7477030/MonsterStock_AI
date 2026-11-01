"""Model training (blueprint sections 13-15, 19).

Pipeline per stage ("pre" = 08:45 scan without the pre-open gap, "final" = 09:08+ scan with it):
  1. walk-forward folds with purge/embargo -> out-of-fold predictions of every specialist (no row is ever predicted by
     a model that saw its own period or any later period);
  2. meta-model trained with an expanding window over those out-of-fold predictions (again strictly past -> future);
  3. probability calibration fit on the EARLIER part of the meta out-of-fold region, evaluated on the latest holdout;
  4. gates: calibration, skill versus climatology and a paired day-block bootstrap against the best simple baseline;
  5. final refit on all data and a signed-ready artifact bundle.
Status: ``validated`` (all gates incl. ranking edge), ``shadow`` (calibrated, beats climatology, edge unproven),
``rejected`` (not even calibrated/skilful). Only the daily labels of past sessions are ever used.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.clock import iso
from app.resources import release_memory
from backtesting import metrics as M
from features.builder import FeatureSet
from features.labels import threshold_name
from features.schema import FEATURE_VERSION, schema_hash
from models import gbm
from models.artifacts import write_bundle
from models.calibration import Calibrator, fit_calibrator
from models.inference import make_meta_frame
from models.specialists import META_CONTEXT, specialists_for

log = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Training cannot proceed (insufficient history, degenerate labels, interrupted)."""


@dataclass
class TrainConfig:
    stage: str
    thresholds: tuple[float, ...]
    primary_threshold: float
    unusual_k: float
    top_n: int = 5
    min_hist_days: int = 120
    n_folds: int = 8
    fold_days: int = 63
    min_train_days: int = 450
    embargo_days: int = 5
    max_rows_per_fit: int = 120_000
    holdout_days: int = 150
    gate_min_days: int = 120
    gate_max_ece: float = 0.06
    gate_min_brier_skill: float = 0.0
    n_boot: int = 500
    seed: int = 7
    min_rows: int = 20_000
    spec_min_rows: int | None = None  # tests only: override every specialist's minimum training rows
    gbm: gbm.GbmConfig = field(default_factory=gbm.GbmConfig)


@dataclass
class TrainResult:
    version: str
    stage: str
    status: str
    directory: Path
    metrics: dict[str, Any]
    gates: dict[str, Any]
    meta: dict[str, Any]
    seconds: float
    notes: list[str]


ProgressFn = Callable[[str, float], None]


def walk_forward_blocks(n_days: int, cfg: TrainConfig) -> list[tuple[int, int, int]]:
    """(test_start, test_end, train_end_exclusive) in day-id space; training excludes ``embargo_days`` before the test."""
    blocks = []
    for k in range(cfg.n_folds):
        test_end = n_days - (cfg.n_folds - 1 - k) * cfg.fold_days
        test_start = test_end - cfg.fold_days
        train_end = test_start - cfg.embargo_days
        if train_end >= cfg.min_train_days and test_start > 0:
            blocks.append((test_start, test_end, train_end))
    return blocks


def _tname(t: float) -> str:
    return threshold_name(t)


def _subsample(rows: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(rows) <= cap:
        return rows
    return np.sort(rng.choice(rows, size=cap, replace=False))


def _quantile_stats(Xv: np.ndarray, names: list[str], rows: np.ndarray) -> dict[str, Any]:
    """p01 / p50 / p99 per feature, computed a few columns at a time (a full copy of the sample would add ~100 MB)."""
    out: dict[str, Any] = {}
    step = 16
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are expected and skipped below
        for start in range(0, len(names), step):
            cols = np.arange(start, min(start + step, len(names)))
            q = np.nanpercentile(Xv[np.ix_(rows, cols)], [1, 50, 99], axis=0)
            for j, i in enumerate(cols):
                if np.isfinite(q[:, j]).all():
                    out[names[i]] = {"p01": float(q[0, j]), "p50": float(q[1, j]), "p99": float(q[2, j])}
    return {"features": out}


def _fit_specialists(
    specs: dict[str, dict[str, Any]], Xv: np.ndarray, labels: dict[str, np.ndarray], train_rows: np.ndarray,
    cfg: TrainConfig, rng: np.random.Generator, names: list[str], stop: Callable[[], bool] | None,
) -> dict[str, Any]:
    boosters: dict[str, Any] = {}
    for name, spec in specs.items():
        if stop and stop():
            raise TrainingError("interrupted")
        idx = np.array(spec["col_idx"])
        rows = train_rows
        for required in spec["requires"]:
            if required in names:
                rows = rows[np.isfinite(Xv[rows, names.index(required)])]
        rows = _subsample(rows, cfg.max_rows_per_fit, rng)
        if len(rows) < spec["min_rows"]:
            continue
        sub = Xv[np.ix_(rows, idx)]
        feat = [names[i] for i in idx]
        for target in spec["targets"]:
            if target == "unusual":
                y, params = labels["y_unusual"][rows], gbm.params_for("binary", cfg.gbm)
            elif target == "up":
                y, params = labels["y_up"][rows], gbm.params_for("binary", cfg.gbm)
            else:
                y, params = np.log(np.clip(labels["y_range"][rows], 0.05, None)), gbm.params_for("l2", cfg.gbm)
            ok = np.isfinite(y)
            if target != "lr" and (y[ok].min() == y[ok].max()):
                continue
            fit_x = sub if ok.all() else sub[ok]  # a boolean mask always copies; skip it when every label is known
            boosters[f"sp:{name}:{target}"] = gbm.fit_booster(fit_x, y[ok], params, cfg.gbm.n_estimators, feat)
            del fit_x
    return boosters


def _predict_specialists(
    boosters: dict[str, Any], specs: dict[str, dict[str, Any]], Xv: np.ndarray, rows: np.ndarray, names: list[str],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        idx = np.array(spec["col_idx"])
        sub = Xv[np.ix_(rows, idx)]
        mask = np.ones(len(rows), dtype=bool)
        for required in spec["requires"]:
            if required in names:
                mask &= np.isfinite(Xv[rows, names.index(required)])
        for target in spec["targets"]:
            key = f"sp:{name}:{target}"
            if key not in boosters:
                continue
            pred = gbm.predict(boosters[key], sub).astype(np.float64)
            pred[~mask] = np.nan
            out[f"sp_{name}_{target}"] = pred
    return out


def _meta_targets(cfg: TrainConfig) -> tuple[list[str], dict[str, list[float]]]:
    cls = [f"hit_{_tname(t)}" for t in cfg.thresholds] + ["unusual"]
    return cls, {"range": [0.1, 0.5, 0.9], "exc": [0.5, 0.9]}


def _fit_meta(
    Mv: np.ndarray, cols: list[str], labels: dict[str, np.ndarray], rows: np.ndarray, cfg: TrainConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    cls_heads, quant = _meta_targets(cfg)
    rows = _subsample(rows, cfg.max_rows_per_fit, rng)
    sub = Mv[rows]
    boosters: dict[str, Any] = {}
    for head in cls_heads:
        y = labels["y_unusual" if head == "unusual" else f"y_{head}"][rows]
        ok = np.isfinite(y)
        if ok.sum() < 500 or y[ok].min() == y[ok].max():
            raise TrainingError(f"degenerate labels for head {head}")
        boosters[f"meta:{head}"] = gbm.fit_booster(sub[ok], y[ok], gbm.params_for("binary", cfg.gbm, meta=True),
                                                    cfg.gbm.meta_estimators, cols)
    for group, alphas in quant.items():
        y_raw = labels["y_range" if group == "range" else "y_exc"][rows]
        y = np.log(np.clip(y_raw, 0.05, None))
        ok = np.isfinite(y)
        for a in alphas:
            boosters[f"meta:{group}_q{int(a * 100)}"] = gbm.fit_booster(
                sub[ok], y[ok], gbm.params_for("quantile", cfg.gbm, meta=True, alpha=a), cfg.gbm.meta_estimators, cols)
    return boosters


def _predict_meta(boosters: dict[str, Any], Mv: np.ndarray, rows: np.ndarray, cfg: TrainConfig) -> dict[str, np.ndarray]:
    cls_heads, quant = _meta_targets(cfg)
    sub = Mv[rows]
    out = {f"cls:{h}": gbm.predict(boosters[f"meta:{h}"], sub).astype(np.float64) for h in cls_heads}
    for group, alphas in quant.items():
        stack = np.sort(np.vstack([np.exp(gbm.predict(boosters[f"meta:{group}_q{int(a * 100)}"], sub)) for a in alphas]), axis=0)
        for i, a in enumerate(alphas):
            out[f"q:{group}_q{int(a * 100)}"] = stack[i]
    return out


def train_stage(
    fs: FeatureSet, cfg: TrainConfig, out_dir: Path, *, progress: ProgressFn | None = None, now: datetime | None = None,
    code_commit: str | None = None, stop: Callable[[], bool] | None = None,
) -> TrainResult:
    started = time.monotonic()
    rng = np.random.default_rng(cfg.seed)
    notes: list[str] = []
    say = progress or (lambda _s, _f: None)

    # ---------------------------------------------------------------------------------- dataset
    X, Y = fs.release()  # takes ownership: the caller must not keep a second copy of the matrix alive during training
    if Y is None:
        raise TrainingError("feature set has no labels")
    required = ["t_atr14_pct", "t_range_mean20"]
    ok = (X["u_hist_days"] >= cfg.min_hist_days) & Y["y_range"].notna() & Y["y_unusual"].notna()
    for col in required:
        ok &= X[col].notna()
    X, Y = X[ok.to_numpy()], Y[ok.to_numpy()]
    if len(X) < cfg.min_rows:
        raise TrainingError(f"insufficient training rows ({len(X)})")
    names = list(X.columns)
    dates = X.index.get_level_values("date")
    day_ids, uniq = pd.factorize(dates, sort=True)
    n_days = len(uniq)
    Xv = X.to_numpy(dtype=np.float32)
    labels = {k: Y[k].to_numpy(dtype=np.float64) for k in Y.columns}
    del Y
    blocks = walk_forward_blocks(n_days, cfg)
    if len(blocks) < 3:
        raise TrainingError(f"insufficient history for walk-forward ({n_days} days, {len(blocks)} usable folds)")

    specs: dict[str, dict[str, Any]] = {}
    for s in specialists_for(cfg.stage):
        cols = s.columns(names)
        if not cols:
            continue
        specs[s.name] = {"col_idx": [names.index(c) for c in cols], "columns": cols, "targets": list(s.targets),
                         "requires": list(s.requires),
                         "min_rows": cfg.spec_min_rows if cfg.spec_min_rows is not None else s.min_rows}

    # ------------------------------------------------------------------------ specialists: out-of-fold
    oof: dict[str, np.ndarray] = {}
    test_rows_blocks: list[np.ndarray] = []
    for b, (t0, t1, train_end) in enumerate(blocks):
        say(f"fold {b + 1}/{len(blocks)} specialists", 0.05 + 0.55 * b / len(blocks))
        train_rows = np.flatnonzero(day_ids < train_end)
        test_rows = np.flatnonzero((day_ids >= t0) & (day_ids < t1))
        boosters = _fit_specialists(specs, Xv, labels, train_rows, cfg, rng, names, stop)
        for key, pred in _predict_specialists(boosters, specs, Xv, test_rows, names).items():
            arr = oof.setdefault(key, np.full(len(Xv), np.nan, dtype=np.float32))  # float32: ~30 of these live at once
            arr[test_rows] = pred
        test_rows_blocks.append(test_rows)
        del boosters, train_rows
        release_memory()
    if not oof:
        raise TrainingError("no specialist could be trained")
    active_specs = {n: s for n, s in specs.items() if any(k.startswith(f"sp_{n}_") for k in oof)}
    block_of_row = np.full(len(Xv), -1, dtype=int)
    for b, rows in enumerate(test_rows_blocks):
        block_of_row[rows] = b
    meta_rows = np.flatnonzero(block_of_row >= 0)

    # ----------------------------------------------------------------------------- meta-model OOF
    ctx_cols = [c for c in META_CONTEXT if c in names]
    ctx_frame = pd.DataFrame(Xv[np.ix_(meta_rows, [names.index(c) for c in ctx_cols])], columns=ctx_cols)
    meta_ctx_index = X.index[meta_rows]
    ctx_frame.index = meta_ctx_index
    sp_for_meta = {k: v[meta_rows] for k, v in oof.items()}
    del oof  # the per-row out-of-fold arrays are only needed at the meta rows from here on
    meta_df = make_meta_frame(sp_for_meta, ctx_frame, ctx_cols)
    del ctx_frame
    meta_columns = list(meta_df.columns)
    Mv = meta_df.to_numpy(dtype=np.float32)
    spec_std = meta_df["sp_std_unusual"].to_numpy(dtype=float)
    del meta_df
    p_up_oof = sp_for_meta.get("sp_direction_up")
    del sp_for_meta
    mblock = block_of_row[meta_rows]
    mlabels = {k: v[meta_rows] for k, v in labels.items()}
    cls_heads, quant = _meta_targets(cfg)
    n_meta = len(meta_rows)
    meta_oof: dict[str, np.ndarray] = {}
    for b in range(1, len(blocks)):
        say(f"meta fold {b}/{len(blocks) - 1}", 0.62 + 0.2 * b / len(blocks))
        tr = np.flatnonzero(mblock < b)
        te = np.flatnonzero(mblock == b)
        if len(tr) < 3000:
            continue
        boosters = _fit_meta(Mv, meta_columns, mlabels, tr, cfg, rng)
        for key, pred in _predict_meta(boosters, Mv, te, cfg).items():
            meta_oof.setdefault(key, np.full(n_meta, np.nan))[te] = pred
    if not meta_oof:
        raise TrainingError("meta out-of-fold region is empty")

    # ------------------------------------------------------------------------- calibration + holdout
    say("calibration and gates", 0.85)
    mday = day_ids[meta_rows]
    valid_meta = np.isfinite(meta_oof[f"cls:{cls_heads[0]}"])
    oof_days = np.unique(mday[valid_meta])
    hold_days = oof_days[-cfg.holdout_days:]
    hold = valid_meta & np.isin(mday, hold_days)
    fit_part = valid_meta & ~hold
    if fit_part.sum() < 2000:
        fit_part = valid_meta & (mday < hold_days[0]) if len(oof_days) > cfg.holdout_days else valid_meta
        notes.append("calibration sample is small; holdout overlaps")

    cal_deploy: dict[str, Calibrator] = {}
    reliability: dict[str, Any] = {}
    head_metrics: dict[str, Any] = {}
    base_rates: dict[str, float] = {}
    for head in cls_heads:
        y_all = mlabels["y_unusual" if head == "unusual" else f"y_{head}"]
        p = meta_oof[f"cls:{head}"]
        base_rates[head] = float(np.nanmean(labels["y_unusual" if head == "unusual" else f"y_{head}"]))
        cal_eval = fit_calibrator(p[fit_part], y_all[fit_part])
        head_metrics[head] = M.summarize_binary(cal_eval.transform(p[hold]), y_all[hold], base_rates[head])
        head_metrics[head]["calibrator"] = cal_eval.kind
        reliability[head] = M.reliability_table(cal_eval.transform(p[hold]), y_all[hold]).to_dict("records")
        cal_deploy[head] = fit_calibrator(p[valid_meta], y_all[valid_meta])

    if p_up_oof is not None:
        y_up = mlabels["y_up"]
        p_up = p_up_oof
        known = np.isfinite(p_up) & np.isfinite(y_up)
        f2, h2 = known & fit_part, known & hold
        cal_up = fit_calibrator(p_up[f2], y_up[f2]) if f2.sum() >= 300 else Calibrator("identity")
        head_metrics["up"] = M.summarize_binary(cal_up.transform(p_up[h2]), y_up[h2], float(np.nanmean(labels["y_up"])))
        head_metrics["up"]["accuracy"] = float(np.mean((cal_up.transform(p_up[h2]) > 0.5) == (y_up[h2] > 0.5))) if h2.any() else float("nan")
        cal_deploy["up"] = fit_calibrator(p_up[known], y_up[known])
        reliability["up"] = M.reliability_table(cal_up.transform(p_up[h2]), y_up[h2]).to_dict("records")
        base_rates["up"] = float(np.nanmean(labels["y_up"]))

    quant_metrics: dict[str, Any] = {}
    for key, pred in meta_oof.items():
        if key.startswith("q:"):
            group, alpha = key[2:].rsplit("_q", 1)
            truth = mlabels["y_range" if group == "range" else "y_exc"]
            quant_metrics[f"{group}_q{alpha}"] = {
                "pinball": M.pinball(truth[hold], pred[hold], int(alpha) / 100.0),
                "below_rate": float(np.nanmean(truth[hold] <= pred[hold])), "nominal": int(alpha) / 100.0,
            }
    q10, q90 = meta_oof["q:range_q10"], meta_oof["q:range_q90"]
    quant_metrics["range_80pct_interval_coverage"] = M.interval_coverage(mlabels["y_range"][hold], q10[hold], q90[hold])
    quant_metrics["range_mae_median"] = M.mae(mlabels["y_range"][hold], meta_oof["q:range_q50"][hold])

    # ---------------------------------------------------------- ranking edge vs simple baselines (holdout)
    primary = f"hit_{_tname(cfg.primary_threshold)}"
    baseline_cols = [c for c in BASELINE_COLUMNS if c in X.columns]  # select columns first: avoids copying all ~190 of them
    ranking = _ranking_edge(X[baseline_cols].iloc[meta_rows][hold], mlabels, meta_oof, hold, mday, primary, cfg, names)

    gates = _gates(head_metrics[primary], ranking, cfg, n_hold_days=int(len(hold_days)))
    status = gates["status"]

    # ------------------------------------------------------------------------------------ final refit
    say("final refit", 0.9)
    all_rows = np.arange(len(Xv))
    final_sp = _fit_specialists(active_specs, Xv, labels, all_rows, cfg, rng, names, stop)
    final_meta = _fit_meta(Mv, meta_columns, mlabels, np.arange(n_meta), cfg, rng)
    boosters = {**final_sp, **final_meta}

    feature_cols = sorted({c for s in active_specs.values() for c in s["columns"]} | set(ctx_cols), key=names.index)
    stats = _quantile_stats(Xv, names, _subsample(all_rows, 150_000, rng))
    stats["features"] = {k: v for k, v in stats["features"].items() if k in feature_cols}
    std = spec_std
    spread = (q90 - q10) / np.clip(meta_oof["q:range_q50"], 1e-6, None)
    dist = Xv[:, names.index("x_sim_dist_mean")] if "x_sim_dist_mean" in names else np.array([np.nan])
    unc_scales = {
        "spec_std_p95": _p(std[valid_meta], 95, 0.1), "spread_p95": _p(spread[valid_meta], 95, 2.0),
        "dist_median": _p(dist, 50, 0.0), "dist_p95": _p(dist, 95, 1.0), "drift_p95": 0.15,
    }

    created = iso(now) if now else iso(datetime.now().astimezone())
    version = f"m-{(now or datetime.now().astimezone()).strftime('%Y%m%d-%H%M%S')}-{cfg.stage}"
    flat_rate = float(np.nanmean(np.abs(labels["y_dir"]) < 1e-9)) if "y_dir" in labels else 0.0
    meta = {
        "version": version, "stage": cfg.stage, "status": status, "created_at": created, "feature_version": FEATURE_VERSION,
        "feature_schema_hash": schema_hash(feature_cols), "feature_columns": feature_cols,
        "thresholds": list(cfg.thresholds), "primary_threshold": cfg.primary_threshold, "unusual_k": cfg.unusual_k,
        "specialists": {n: {"columns": s["columns"], "targets": [t for t in s["targets"] if f"sp:{n}:{t}" in boosters],
                            "requires": s["requires"]} for n, s in active_specs.items()},
        "meta_columns": meta_columns, "meta_context": ctx_cols,
        "heads": {"cls": cls_heads, "quant": {k: list(v) for k, v in quant.items()}},
        "booster_keys": sorted(boosters), "base_rates": base_rates, "flat_rate": flat_rate, "unc_scales": unc_scales,
        "train_start": str(pd.Timestamp(uniq[0]).date()), "train_end": str(pd.Timestamp(uniq[-1]).date()),
        "n_rows": int(len(Xv)), "n_days": int(n_days), "holdout_days": int(len(hold_days)), "folds": len(blocks),
        "code_commit": code_commit, "seed": cfg.seed,
    }
    metrics = {"heads": head_metrics, "quantiles": quant_metrics, "ranking": ranking}
    meta["metrics"] = metrics
    meta["gates"] = gates
    write_bundle(out_dir / version, meta=meta, boosters=boosters, calibrators=cal_deploy, reliability=reliability,
                 feature_stats=stats)
    say("done", 1.0)
    return TrainResult(version, cfg.stage, status, out_dir / version, metrics, gates, meta, time.monotonic() - started, notes)


def _p(values: np.ndarray, pct: float, default: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, pct)) if len(v) else default


BASELINE_SCORERS = {"atr": "t_atr14_pct", "range20": "t_range_mean20", "volume": "t_rvol1", "recent_range": "t_range_prev",
                    "gap": "p_gap_abs"}
BASELINE_COLUMNS = tuple(BASELINE_SCORERS.values())


def _ranking_edge(
    hold_X: pd.DataFrame, mlabels: dict[str, np.ndarray], meta_oof: dict[str, np.ndarray], hold: np.ndarray,
    mday: np.ndarray, primary: str, cfg: TrainConfig, names: list[str],
) -> dict[str, Any]:
    """Daily top-N hit rate of the model versus simple baselines, with a paired day-block bootstrap."""
    y = mlabels[f"y_{primary}"][hold]
    frame = pd.DataFrame({"date": mday[hold], "hit": y, "model": meta_oof[f"cls:{primary}"][hold],
                          "range": mlabels["y_range"][hold]})
    rng = np.random.default_rng(cfg.seed + 99)
    frame["random"] = rng.random(len(frame))
    candidates = BASELINE_SCORERS
    for label, col in candidates.items():
        if col in hold_X.columns:
            frame[label] = hold_X[col].to_numpy(dtype=float)
    scorers = ["model", "random"] + [k for k in candidates if k in frame.columns]
    daily = {s: M.daily_topn(frame, s, "hit", cfg.top_n) for s in scorers}
    daily_rng = {s: M.daily_topn(frame, s, "range", cfg.top_n) for s in scorers}
    summary: dict[str, Any] = {}
    for s in scorers:
        rate, lo, hi, n = M.topn_hit_rate(daily[s])
        summary[s] = {"hit_rate": rate, "ci_low": lo, "ci_high": hi, "n_selected": n,
                      "mean_range": float(daily_rng[s]["mean"].mean()) if len(daily_rng[s]) else float("nan")}
    baselines = [s for s in scorers if s not in {"model", "random"}]
    best = max(baselines, key=lambda s: summary[s]["hit_rate"] if summary[s]["hit_rate"] == summary[s]["hit_rate"] else -1) if baselines else "random"
    boot = M.paired_day_bootstrap(daily["model"], daily[best], n_boot=cfg.n_boot, seed=cfg.seed)
    return {"top_n": cfg.top_n, "scorers": summary, "best_baseline": best, "vs_best_baseline": boot,
            "n_days": int(len(daily["model"]))}


def _gates(primary_metrics: dict[str, Any], ranking: dict[str, Any], cfg: TrainConfig, *, n_hold_days: int) -> dict[str, Any]:
    ece = primary_metrics.get("ece", float("nan"))
    skill = primary_metrics.get("brier_skill", float("nan"))
    boot = ranking["vs_best_baseline"]
    calibrated = ece == ece and ece <= cfg.gate_max_ece
    skilful = skill == skill and skill > cfg.gate_min_brier_skill
    edge = boot["mean_diff"] == boot["mean_diff"] and boot["mean_diff"] > 0 and boot["p10"] >= 0
    enough = n_hold_days >= cfg.gate_min_days
    if calibrated and skilful and edge and enough:
        status, reason = "validated", "calibrated, skilful and ahead of the best simple baseline (90% one-sided)"
    elif calibrated and skilful:
        status, reason = "shadow", ("ranking edge over simple baselines not demonstrated" if enough
                                    else "holdout shorter than the validation requirement")
    else:
        status, reason = "rejected", "not calibrated and/or no skill versus climatology on the holdout"
    return {"status": status, "reason": reason, "calibrated": bool(calibrated), "skilful": bool(skilful), "edge": bool(edge),
            "enough_days": bool(enough), "ece": ece, "brier_skill": skill, "holdout_days": n_hold_days,
            "vs_best_baseline": boot, "best_baseline": ranking["best_baseline"],
            "limits": {"max_ece": cfg.gate_max_ece, "min_brier_skill": cfg.gate_min_brier_skill, "min_days": cfg.gate_min_days}}
