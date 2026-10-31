"""Probability calibration (blueprint section 14): isotonic regression (pool-adjacent-violators) and Platt scaling.

A "70%" must mean ~70% observed frequency. Calibrators are JSON-serialisable (no pickle) and conservative: with too few
samples they fall back to Platt scaling or the identity instead of overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_platt(p: np.ndarray, y: np.ndarray, iterations: int = 50) -> tuple[float, float]:
    """Logistic regression y ~ sigmoid(a * logit(p) + b) by Newton iterations (2 parameters, ridge-stabilised)."""
    x = _logit(np.asarray(p, dtype=float))
    y = np.asarray(y, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iterations):
        mu = _sigmoid(a * x + b)
        w = np.maximum(mu * (1 - mu), 1e-9)
        g = np.array([np.sum((mu - y) * x) + 1e-3 * (a - 1.0), np.sum(mu - y)])
        h = np.array([[np.sum(w * x * x) + 1e-3, np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(a), float(b)


def fit_isotonic(p: np.ndarray, y: np.ndarray, n_bins: int = 80) -> tuple[list[float], list[float]]:
    """Monotone non-decreasing map p -> E[y] via PAV on equal-mass bins (bounded knots, smooth enough not to overfit)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(p, kind="mergesort")
    p, y = p[order], y[order]
    chunks = np.array_split(np.arange(len(p)), min(n_bins, max(2, len(p) // 40)))
    xs = np.array([p[c].mean() for c in chunks if len(c)])
    ys = np.array([y[c].mean() for c in chunks if len(c)])
    ws = np.array([len(c) for c in chunks if len(c)], dtype=float)
    # pool adjacent violators
    vals, weights, spans = list(ys), list(ws), [[i] for i in range(len(ys))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            total = weights[i] + weights[i + 1]
            vals[i] = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / total
            weights[i] = total
            spans[i] += spans[i + 1]
            del vals[i + 1], weights[i + 1], spans[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out_x, out_y = [], []
    for value, span in zip(vals, spans, strict=True):
        out_x.append(float(xs[span].mean()))
        out_y.append(float(value))
    return out_x, out_y


@dataclass
class Calibrator:
    kind: str = "identity"  # identity | platt | isotonic
    params: dict[str, Any] = field(default_factory=dict)
    n_fit: int = 0

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.kind == "platt":
            return np.clip(_sigmoid(self.params["a"] * _logit(p) + self.params["b"]), EPS, 1 - EPS)
        if self.kind == "isotonic":
            xs, ys = np.asarray(self.params["x"]), np.asarray(self.params["y"])
            return np.clip(np.interp(p, xs, ys), EPS, 1 - EPS)
        return np.clip(p, EPS, 1 - EPS)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.params, "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibrator:
        return cls(kind=data.get("kind", "identity"), params=dict(data.get("params", {})), n_fit=int(data.get("n_fit", 0)))


def fit_calibrator(p: np.ndarray, y: np.ndarray, *, min_isotonic: int = 2000, min_platt: int = 300) -> Calibrator:
    """Choose the calibrator by sample size: isotonic (n >= 2000 and enough positives), Platt, else identity."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n, pos = len(p), int(y.sum())
    if n >= min_isotonic and min(pos, n - pos) >= 100:
        xs, ys = fit_isotonic(p, y)
        return Calibrator("isotonic", {"x": xs, "y": ys}, n)
    if n >= min_platt and min(pos, n - pos) >= 20:
        a, b = fit_platt(p, y)
        return Calibrator("platt", {"a": a, "b": b}, n)
    return Calibrator("identity", {}, n)
