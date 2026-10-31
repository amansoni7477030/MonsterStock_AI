"""LightGBM helpers tuned for a 1 GB / 1 vCPU machine. Models are stored in LightGBM's text format (never pickle)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np


@dataclass(frozen=True)
class GbmConfig:
    n_estimators: int = 140
    learning_rate: float = 0.07
    num_leaves: int = 15
    min_data_in_leaf: int = 300
    feature_fraction: float = 0.6
    bagging_fraction: float = 0.7
    lambda_l2: float = 20.0
    max_bin: int = 63
    meta_estimators: int = 120
    meta_leaves: int = 7
    meta_min_data: int = 400
    seed: int = 7


def _base(cfg: GbmConfig, *, leaves: int, min_data: int, seed_offset: int = 0) -> dict[str, Any]:
    return {
        "boosting_type": "gbdt", "learning_rate": cfg.learning_rate, "num_leaves": leaves, "min_data_in_leaf": min_data,
        "feature_fraction": cfg.feature_fraction, "bagging_fraction": cfg.bagging_fraction, "bagging_freq": 1,
        "lambda_l2": cfg.lambda_l2, "max_bin": cfg.max_bin, "num_threads": 1, "verbosity": -1, "force_col_wise": True,
        "deterministic": True, "feature_pre_filter": False, "seed": cfg.seed + seed_offset,
        "bagging_seed": cfg.seed + seed_offset, "feature_fraction_seed": cfg.seed + seed_offset,
    }


def params_for(kind: str, cfg: GbmConfig, *, meta: bool = False, alpha: float | None = None) -> dict[str, Any]:
    leaves, min_data = (cfg.meta_leaves, cfg.meta_min_data) if meta else (cfg.num_leaves, cfg.min_data_in_leaf)
    params = _base(cfg, leaves=leaves, min_data=min_data)
    if kind == "binary":
        params["objective"] = "binary"
    elif kind == "l2":
        params["objective"] = "regression"
    elif kind == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = alpha if alpha is not None else 0.5
    else:
        raise ValueError(kind)
    return params


def fit_booster(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any], rounds: int, feature_names: list[str]
) -> lgb.Booster:
    data = lgb.Dataset(np.ascontiguousarray(X, dtype=np.float32), label=np.asarray(y, dtype=np.float64),
                       feature_name=list(feature_names), free_raw_data=True,
                       params={"max_bin": params["max_bin"], "feature_pre_filter": False, "bin_construct_sample_cnt": 50_000})
    return lgb.train(params, data, num_boost_round=rounds)


def predict(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return booster.predict(np.ascontiguousarray(X, dtype=np.float32), num_threads=1)


def contributions(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Per-feature additive contributions (SHAP-style) incl. bias as the last column - diagnostic, not causal."""
    return booster.predict(np.ascontiguousarray(X, dtype=np.float32), pred_contrib=True, num_threads=1)
