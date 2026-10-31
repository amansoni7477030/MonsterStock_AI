"""Outcome collection and evaluation (blueprint sections 15, 19): attach realised outcomes to frozen predictions WITHOUT
editing them, then score the model against same-day baselines with intervals and sample sizes."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app import jsonutil
from app.config import Settings
from backtesting import metrics as M
from database.store import Store
from features.labels import threshold_name
from ranking.types import EveningRow, EveningSummary

log = logging.getLogger(__name__)

BASELINES = ("atr", "range20", "volume", "gap", "random")


def outcome_metrics(o: float, h: float, low: float, c: float) -> dict[str, float]:
    """Appendix A definitions. mfe/mae: favourable = the larger excursion, adverse = the smaller one (direction-agnostic)."""
    up, down = (h - o) / o * 100.0, (o - low) / o * 100.0
    return {"range_pct": (h - low) / o * 100.0, "exc_pct": max(up, down), "dir_ret_pct": (c - o) / o * 100.0,
            "mfe_pct": max(up, down), "mae_pct": min(up, down)}


def collect_outcomes(store: Store, settings: Settings, trade_date: str) -> dict[str, int]:
    """Attach realised outcomes (official bhavcopy when available, otherwise vendor bars flagged unvalidated)."""
    pending = store.predictions_without_validated_outcome(trade_date)
    counts = {"attached": 0, "validated": 0, "missing": 0}
    snapshots: dict[str, dict[str, Any]] = {}
    for pred in pending:
        iid = store.instrument_id(pred["symbol"])
        bars = store.get_bars(iid, trade_date, trade_date) if iid is not None else pd.DataFrame()
        if bars.empty or not np.isfinite(bars.iloc[0][["open", "high", "low", "close"]].to_numpy(dtype=float)).all():
            counts["missing"] += 1
            continue
        bar = bars.iloc[0]
        o, h, low, c = (float(bar[k]) for k in ("open", "high", "low", "close"))
        if o <= 0:
            counts["missing"] += 1
            continue
        m = outcome_metrics(o, h, low, c)
        hits = {f"hit_{threshold_name(t)}": bool(m["range_pct"] > t) for t in settings.range_thresholds}
        snap = snapshots.setdefault(pred["stage"], store.latest_snapshot(trade_date, pred["stage"]) or {})
        baseline_range = feature_value((snap.get("payload") or {}).get("features"), pred["symbol"], "t_range_mean20")
        if baseline_range:
            hits["unusual"] = bool(m["range_pct"] > settings.unusual_k * float(baseline_range))
        validated = int(bar["source_rank"]) >= 2
        store.upsert_outcome(pred["prediction_id"], trade_date, pred["symbol"], source=str(bar["source"]), validated=validated,
                             ohlcv={"open": o, "high": h, "low": low, "close": c, "volume": bar["volume"]}, metrics=m, hits=hits)
        counts["attached"] += 1
        counts["validated"] += int(validated)
    return counts


def _day_rows(store: Store, trade_date: str, stage: str) -> pd.DataFrame:
    rows = [r for r in store.evaluated_rows(since=trade_date, until=trade_date) if r["stage"] == stage]
    return pd.DataFrame(rows)


def evaluate_day(store: Store, settings: Settings, trade_date: str) -> EveningSummary | None:
    """Per-day summary of the model's frozen top-N versus same-day baseline picks."""
    stage = "final" if store.predictions_for(trade_date, stage="final", baselines=None) else "pre"
    frame = _day_rows(store, trade_date, stage)
    if frame.empty:
        return None
    primary = f"hit_{threshold_name(settings.primary_threshold)}"
    model = frame[frame["is_baseline"] == 0].sort_values("rank")
    base = frame[frame["is_baseline"] == 1]
    if model.empty:
        return None

    def hit(row: Any) -> bool | None:
        value = row["range_pct"]
        return None if value is None or value != value else bool(value > settings.primary_threshold)

    rows = []
    for _, r in model.iterrows():
        rows.append(EveningRow(symbol=r["symbol"], rank=int(r["rank"]), tier=r["tier"], p_primary=r["p"].get(primary),
                               range_pct=r["range_pct"], exc_pct=r["exc_pct"], dir_ret_pct=r["dir_ret_pct"], hit_primary=hit(r),
                               source=r["o_source"], validated=bool(r["o_validated"])))
    hits = [x.hit_primary for x in rows if x.hit_primary is not None]
    lo, hi = M.wilson_interval(sum(hits), len(hits)) if hits else (float("nan"), float("nan"))
    baseline_rates: dict[str, float] = {}
    baseline_range: dict[str, float] = {}
    for name, group in base.groupby("baseline_name"):
        outcomes = [(r["range_pct"] > settings.primary_threshold) for _, r in group.iterrows() if r["range_pct"] == r["range_pct"]]
        if outcomes:
            baseline_rates[name] = float(np.mean(outcomes))
            baseline_range[name] = float(group["range_pct"].mean())
    missing = sorted(set(model["symbol"]) - {r.symbol for r in rows if r.range_pct is not None})
    meta_missing = sorted({p["symbol"] for p in store.predictions_for(trade_date, stage=stage)} - set(model["symbol"]))
    validated = bool(len(model)) and bool(model["o_validated"].all())
    rolling = rolling_summary(store, settings, trade_date)
    version = str(model["model_version"].iloc[0])
    return EveningSummary(
        trade_date=trade_date, stage=stage, primary_threshold=settings.primary_threshold, model_version=version, rows=rows,
        n=len(hits), model_hit_rate=float(np.mean(hits)) if hits else None, model_hit_ci=(lo, hi) if hits else None,
        baseline_hit_rates=baseline_rates, mean_range_pct=float(model["range_pct"].mean()),
        baseline_mean_range_pct=baseline_range, rolling=rolling,
        calibration_note=_calibration_note(rolling), provisional=not validated, missing_symbols=missing + meta_missing,
    )


def _calibration_note(rolling: dict[str, float | None]) -> str:
    n = rolling.get("n_days_90") or 0
    if n < 20:
        return f"only {int(n)} evaluated day(s) so far: too early to judge calibration"
    ece, skill = rolling.get("ece_90"), rolling.get("brier_skill_90")
    parts = []
    if ece is not None:
        parts.append(f"calibration error {ece:.3f} over {int(n)} days")
    if skill is not None:
        parts.append(f"Brier skill vs base rate {skill:+.3f}")
    return "; ".join(parts)


def rolling_summary(store: Store, settings: Settings, trade_date: str, windows: tuple[int, ...] = (30, 90)) -> dict[str, float | None]:
    """Pooled top-N hit rates and calibration over the last evaluated days (final-stage predictions only)."""
    start = (date.fromisoformat(trade_date) - timedelta(days=int(max(windows) * 1.6))).isoformat()
    rows = [r for r in store.evaluated_rows(since=start, until=trade_date) if r["stage"] == "final" or r["stage"] == "pre"]
    out: dict[str, float | None] = {}
    if not rows:
        return out
    frame = pd.DataFrame(rows)
    frame["hit"] = (frame["range_pct"] > settings.primary_threshold).astype(float).where(frame["range_pct"].notna())
    days = sorted(frame["trade_date"].unique())
    for w in windows:
        subset_days = days[-w:]
        sub = frame[frame["trade_date"].isin(subset_days)]
        model = sub[sub["is_baseline"] == 0]
        vals = model["hit"].dropna()
        out[f"n_days_{w}"] = float(len(subset_days))
        out[f"hit_rate_{w}d"] = float(vals.mean()) if len(vals) else None
        for name in BASELINES:
            b = sub[(sub["is_baseline"] == 1) & (sub["baseline_name"] == name)]["hit"].dropna()
            out[f"baseline_{name}_{w}d"] = float(b.mean()) if len(b) else None
        cal = calibration_from_snapshots(store, settings, subset_days)
        out[f"brier_{w}"], out[f"ece_{w}"], out[f"brier_skill_{w}"] = cal.get("brier"), cal.get("ece"), cal.get("brier_skill")
    return out


def calibration_from_snapshots(store: Store, settings: Settings, days: list[str]) -> dict[str, float | None]:
    """Brier / ECE of the frozen primary probability for ALL eligible names (not only the picks) versus realised outcomes."""
    ps: list[float] = []
    ys: list[float] = []
    for day in days:
        snap = store.latest_snapshot(day, "final") or store.latest_snapshot(day, "pre")
        if not snap:
            continue
        table = (snap.get("payload") or {}).get("table") or {}
        if not table:
            continue
        bars = store.conn.execute(
            "SELECT i.symbol AS symbol, b.open AS o, b.high AS h, b.low AS l FROM bars_daily b JOIN instruments i USING (instrument_id) "
            "WHERE b.trade_date = ? AND i.kind = 'equity'", (day,)).fetchall()
        realised = {r["symbol"]: (r["h"] - r["l"]) / r["o"] * 100.0 for r in bars if r["o"] and r["h"] and r["l"]}
        for symbol, p in table.items():
            if symbol in realised and p is not None:
                ps.append(float(p))
                ys.append(float(realised[symbol] > settings.primary_threshold))
    if len(ps) < 200:
        return {"brier": None, "ece": None, "brier_skill": None}
    p, y = np.array(ps), np.array(ys)
    return {"brier": M.brier(p, y), "ece": M.ece(p, y), "brier_skill": M.brier_skill(p, y)}


def persist_metrics(store: Store, summary: EveningSummary) -> int:
    """Write the day's and rolling numbers to ``eval_metrics`` (append-only history, never edits predictions)."""
    daily = [{"name": "topn_hit_rate", "value": summary.model_hit_rate, "n": summary.n,
              "ci_low": (summary.model_hit_ci or (None, None))[0], "ci_high": (summary.model_hit_ci or (None, None))[1]},
             {"name": "topn_mean_range_pct", "value": summary.mean_range_pct, "n": summary.n}]
    for name, value in summary.baseline_hit_rates.items():
        daily.append({"name": f"baseline_{name}_hit_rate", "value": value, "n": summary.n})
    written = store.insert_metrics(summary.trade_date, "daily", daily)
    rolling = [{"name": k, "value": v, "n": int(summary.rolling.get(f"n_days_{k.rsplit('_', 1)[-1].rstrip('d')}") or 0)}
               for k, v in summary.rolling.items() if v is not None and not k.startswith("n_days")]
    written += store.insert_metrics(summary.trade_date, "rolling", rolling)
    store.kv_set_json(f"evaluation:{summary.trade_date}", {"hit_rate": summary.model_hit_rate, "n": summary.n,
                                                        "provisional": summary.provisional})
    return written


def snapshot_table(table: pd.DataFrame) -> dict[str, float]:
    """Compact map symbol -> frozen primary probability for every eligible name (used for calibration scoring)."""
    pool = table.loc[table["eligible"], "p_primary"]
    return {s: round(float(v), 5) for s, v in pool.items() if v == v}


def feature_payload(features: pd.DataFrame) -> dict[str, Any]:
    """Frozen model inputs for every scored symbol in a compact columnar form (5 significant digits)."""
    values = features.to_numpy(dtype=float)
    rows = {symbol: [float(f"{v:.5g}") if np.isfinite(v) else None for v in row]
            for symbol, row in zip(features.index, values, strict=True)}
    return {"columns": [str(c) for c in features.columns], "rows": rows}


def feature_value(payload: dict[str, Any] | None, symbol: str, column: str) -> float | None:
    if not payload:
        return None
    try:
        return payload["rows"][symbol][payload["columns"].index(column)]
    except (KeyError, ValueError, IndexError):
        return None


def dumps(payload: Any) -> str:  # pragma: no cover - convenience for debugging
    return jsonutil.dumps(payload)
