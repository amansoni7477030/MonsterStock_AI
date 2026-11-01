"""Synthetic frozen records for the evaluation tests: snapshots with a probability table, regime meta and features, model and
baseline picks with range forecasts, the session's bars and attached outcomes. No scan, no model and no network: every
number is controlled by the test."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtesting import evaluate as ev
from database.store import Store

SECTORS = ("BANKS", "IT", "AUTO", "PHARMA")
MODEL = "m-test"


def trading_days(count: int, start: str = "2026-06-01") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=count)]


@dataclass
class Pick:
    symbol: str
    p: float = 0.5  # frozen P(range >= primary threshold)
    q10: float = 1.5
    q50: float = 3.0
    q90: float = 6.0
    up: float | None = None  # P(up close); None = no direction forecast (the heuristic baseline has none)
    tier: str = "WATCH"
    event: str | None = None  # event type of the main catalyst


def names(count: int = 12, prefix: str = "S") -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(count)]


def seed_universe(store: Store, symbols: list[str]) -> None:
    for i, symbol in enumerate(symbols):
        store.upsert_instrument(symbol, "equity", sector=SECTORS[i % len(SECTORS)], in_universe=1, name=f"{symbol} Ltd")


def bar(range_pct: float, direction: float = 0.0) -> dict[str, float]:
    """An OHLC bar opening at 100 whose (high - low) / open is exactly ``range_pct`` and whose close moved ``direction`` %."""
    direction = min(max(direction, -0.4 * range_pct), 0.6 * range_pct)
    return {"open": 100.0, "high": 100.0 + 0.6 * range_pct, "low": 100.0 - 0.4 * range_pct, "close": 100.0 + direction}


def freeze_day(
    store: Store, day: str, *, picks: list[Pick], table: dict[str, float], realised: dict[str, tuple[float, float]],
    stage: str = "final", baselines: dict[str, list[str]] | None = None, regime: dict[str, str] | None = None,
    market: dict[str, float] | None = None, turnover: dict[str, float] | None = None, tag: str = "",
) -> tuple[int, list[int]]:
    """Record the session's bars (``realised``: symbol -> (range %, close-vs-open %)) and freeze one snapshot with its
    predictions for ``day``. Different ``tag`` / picks give a different snapshot (a forced re-scan)."""
    symbols = list(table)
    features = ev.feature_payload(pd.DataFrame(
        {"t_tv_med20_cr": [(turnover or {}).get(s, 100.0 + i) for i, s in enumerate(symbols)], "t_range_mean20": [2.0] * len(symbols)},
        index=symbols))
    payload = {"report": {"regime": regime or {}}, "table": table, "features": features, "baselines": baselines or {},
               "excluded": {}, "open_source": {}, "meta": {"market": market or {}, "tag": tag}}
    rows: list[dict[str, Any]] = []
    for rank, pick in enumerate(picks, 1):
        probabilities = {"hit_3": pick.p, "unusual": 0.3}
        if pick.up is not None:
            probabilities.update(up=pick.up, down=round(1.0 - pick.up, 6))
        catalysts = [{"headline": "headline", "event_type": pick.event}] if pick.event else []
        rows.append({"symbol": pick.symbol, "rank": rank, "tier": pick.tier, "model_version": MODEL, "model_status": "shadow",
                     "p": probabilities, "range": {"q10": pick.q10, "q50": pick.q50, "q90": pick.q90}, "uncertainty": 0.3,
                     "explanation": {"catalysts": catalysts}, "liquidity": {"turnover_cr": 100.0}})
    for name, symbols_ in (baselines or {}).items():
        rows += [{"symbol": symbol, "rank": position, "tier": "BASELINE", "model_version": MODEL, "model_status": "baseline",
                  "is_baseline": True, "baseline_name": name} for position, symbol in enumerate(symbols_, 1)]
    with store.tx():
        for symbol, (range_pct, direction) in realised.items():
            store.upsert_bars(store.instrument_id(symbol), [{"trade_date": day, **bar(range_pct, direction)}],
                              source="nse_bhavcopy", source_rank=2, quality="official")
        snapshot_id = store.insert_snapshot(trade_date=day, stage=stage, cutoff_ts=f"{day}T03:40:00Z", feature_version="fv-test",
                                            model_version=MODEL, code_version="test", payload=payload, data_quality={})
        return snapshot_id, store.insert_predictions(snapshot_id, day, stage, rows)


def attach_outcomes(store: Store, day: str, realised: dict[str, tuple[float, float]]) -> None:
    """Attach the official outcome of every effective prediction of ``day`` whose symbol has a realised session (what
    ``collect_outcomes`` does from the stored bars, without the per-prediction bar queries)."""
    with store.tx():
        for p in store.predictions_for(day, baselines=None):
            if p["symbol"] in realised:
                session = bar(*realised[p["symbol"]])
                store.upsert_outcome(p["prediction_id"], day, p["symbol"], source="nse_bhavcopy", validated=True, ohlcv=session,
                                     metrics=ev.outcome_metrics(session["open"], session["high"], session["low"], session["close"]))


def seed_days(store: Store, count: int, plan: Callable[[int, str], dict[str, Any]], *, start: str = "2026-06-01") -> list[str]:
    """Freeze ``count`` consecutive weekdays (``plan(i, day)`` returns the keyword arguments of ``freeze_day``) with their
    realised outcomes attached; returns the days."""
    days = trading_days(count, start)
    for i, day in enumerate(days):
        kwargs = plan(i, day)
        freeze_day(store, day, **kwargs)
        attach_outcomes(store, day, kwargs["realised"])
    return days
