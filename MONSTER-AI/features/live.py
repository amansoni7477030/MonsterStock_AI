"""Feature construction entry points used by jobs: live (one date) and training (many dates) share one builder."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from app.clock import iso
from app.config import Settings
from app.resources import release_memory
from automation.calendar import ExchangeCalendar
from database.store import Store
from features import similar_day
from features.builder import FeatureSet, Panel, build_feature_set
from features.calendar_features import RESULTS_HORIZON, next_trading_days
from features.loader import load_panel

log = logging.getLogger(__name__)

# NSE holds ~248 sessions in 365 days (1.47 calendar days each); the margin covers holiday clusters and the sessions the loader drops
CALENDAR_DAYS_PER_SESSION = 1.6
# Calendar days a live scan loads: enough sessions for every date's similar-day features to equal the ones training computes
# from the full history (see similar_day.HISTORY_SESSIONS), which also covers every 252-session rolling window.
LIVE_HISTORY_DAYS = math.ceil(similar_day.HISTORY_SESSIONS * CALENDAR_DAYS_PER_SESSION)


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


def _with_results_horizon(panel: Panel, calendar: ExchangeCalendar) -> Panel:
    """Make the panel carry ``RESULTS_HORIZON`` upcoming sessions: ``c_days_to_results`` looks that far ahead, in training and
    in live scans alike (the loader supplies only a few)."""
    if len(panel.dates) and len(panel.future_dates) < RESULTS_HORIZON:
        panel.future_dates = next_trading_days(panel.dates[-1].date(), RESULTS_HORIZON, calendar.is_trading_day)
        panel.next_after_last = panel.future_dates[0]
    return panel


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
    if len(panel.dates) - 1 < similar_day.HISTORY_SESSIONS:
        log.warning("only %d sessions of history before %s (similar-day features need %d to match training's): "
                    "they will differ from the ones the model was trained on", len(panel.dates) - 1, trade_date,
                    similar_day.HISTORY_SESSIONS)
    _with_results_horizon(panel, calendar)
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
    # 420 days (~290 sessions) of history behind the first emitted row: every 252-session window is full. The similar-day features of
    # the first ~240 rows then see a shallower pool and shorter statistics than ``similar_day.HISTORY_SESSIONS`` (a live scan
    # always has the full amount; later rows are identical). Loading the rest costs ~35 MB of peak memory at 5 years x 220 stocks,
    # which the 1 GB server cannot spare, so that small warm-up difference is accepted.
    start = (end - timedelta(days=int(years * 365.25) + 420)).isoformat()
    panel = load_panel(store, start=start, end=end.isoformat(), calendar=calendar, fno_start=start)
    _with_results_horizon(panel, calendar)
    keep_from = max(0, len(panel.dates) - int(years * 252))
    fs = build_feature_set(panel, stage=stage, thresholds=settings.range_thresholds, unusual_k=settings.unusual_k,
                           cutoff_hhmm=cutoff_hhmm(settings, stage), with_labels=True, row_slice=slice(keep_from, len(panel.dates)))
    del panel
    release_memory()
    return fs
