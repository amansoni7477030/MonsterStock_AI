"""Feature construction entry points used by jobs: live (one date) and training (many dates) share one builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from app.clock import iso
from app.config import Settings
from app.resources import release_memory
from automation.calendar import ExchangeCalendar
from database.store import Store
from features.builder import FeatureSet, build_feature_set
from features.loader import load_panel

LIVE_HISTORY_DAYS = 560  # calendar days: covers 252-session windows plus similar-day matching warm-up


def cutoff_hhmm(settings: Settings, stage: str) -> str:
    return (settings.final_scan_time if stage == "final" else settings.prelim_scan_time)[:5]


@dataclass
class LiveFeatures:
    X: pd.DataFrame  # index = symbol
    stage: str
    trade_date: str
    cutoff: datetime
    open_source: dict[str, str] = field(default_factory=dict)  # symbol -> final | indicative | none
    X_pre: pd.DataFrame | None = None  # same stocks without pre-open features (fallback for names lacking a gap)
    market: dict[str, Any] = field(default_factory=dict)
    last_bar: str | None = None


def build_live_features(
    store: Store, settings: Settings, calendar: ExchangeCalendar, *, trade_date: date, stage: str, cutoff: datetime,
    symbols: list[str] | None = None, with_fallback: bool = True,
) -> LiveFeatures:
    """Features for ``trade_date`` as known at ``cutoff``. For stage ``final`` the pre-open book received at or before
    the cutoff supplies the opening price; names without one are flagged and can use the ``pre`` feature set."""
    start = (trade_date - timedelta(days=LIVE_HISTORY_DAYS)).isoformat()
    end = (trade_date - timedelta(days=1)).isoformat()
    live_open: dict[str, float] = {}
    source: dict[str, str] = {}
    if stage == "final":
        book = store.latest_preopen(trade_date.isoformat(), before=iso(cutoff))
        for symbol, quote in book.items():
            if quote.get("final_price"):
                live_open[symbol], source[symbol] = float(quote["final_price"]), "final"
            elif quote.get("iep"):
                live_open[symbol], source[symbol] = float(quote["iep"]), "indicative"
    panel = load_panel(store, start=start, end=end, symbols=symbols, today=trade_date, live_open=live_open, calendar=calendar)
    if not len(panel.dates) or panel.dates[-1].date() != trade_date:
        raise ValueError("panel has no row for the trade date")
    last_row = slice(len(panel.dates) - 1, len(panel.dates))
    fs = build_feature_set(panel, stage=stage, thresholds=settings.range_thresholds, unusual_k=settings.unusual_k,
                           cutoff_hhmm=cutoff_hhmm(settings, stage), with_labels=False, row_slice=last_row)
    x = fs.X.droplevel("date")
    x_pre = None
    if stage == "final" and with_fallback:
        fs_pre = build_feature_set(panel, stage="pre", thresholds=settings.range_thresholds, unusual_k=settings.unusual_k,
                                   cutoff_hhmm=cutoff_hhmm(settings, "pre"), with_labels=False, row_slice=last_row)
        x_pre = fs_pre.X.droplevel("date")
    market = {c: float(fs.market[c].iloc[-1]) for c in fs.market.columns if pd.notna(fs.market[c].iloc[-1])}
    prior = panel.dates[-2].date().isoformat() if len(panel.dates) > 1 else None
    return LiveFeatures(x, stage, trade_date.isoformat(), cutoff, source, x_pre, market, prior)


def build_training_features(
    store: Store, settings: Settings, calendar: ExchangeCalendar, *, stage: str, years: int, end: date,
) -> FeatureSet:
    """Features + labels for the last ``years`` years ending at ``end`` (inclusive of the final labelled session).

    The market-data panel is released before returning: only the feature matrix is needed afterwards, and on the 1 GB
    server both cannot be resident while the models train."""
    start = (end - timedelta(days=int(years * 365.25) + 420)).isoformat()
    panel = load_panel(store, start=start, end=end.isoformat(), calendar=calendar,
                       fno_start=(end - timedelta(days=int(years * 365.25) + 420)).isoformat())
    keep_from = max(0, len(panel.dates) - int(years * 252))
    fs = build_feature_set(panel, stage=stage, thresholds=settings.range_thresholds, unusual_k=settings.unusual_k,
                           cutoff_hhmm=cutoff_hhmm(settings, stage), with_labels=True, row_slice=slice(keep_from, len(panel.dates)))
    del panel
    release_memory()
    return fs
