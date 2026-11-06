"""The closing-hour trade-idea scan (``closing_scan``) and the evaluation of its ideas (``evaluate_ideas``).

``closing_scan`` runs at 14:30 IST. NSE's live quotes (one call for the whole F&O segment) are laid over the stored daily history as
the running session of today; ``features.swing`` turns that panel into the features of the closing hour, ``ranking.ideas`` into
BTST, swing, bearish and next-day intraday ideas, the news pipeline adds catalysts to the shortlist, and the report is FROZEN
(written once per trading day and re-sent unchanged after a failed delivery, exactly like the morning snapshots) and sent to Telegram.
Once a report is frozen, what was said is the record: a retry, a late run and a forced run all send that report (``--force`` only
lifts the report deadline, and the copy that goes out says it is late); a new scan can only be previewed (``--dry-run``). Nothing is
ever traded: shadow mode.

``evaluate_ideas`` runs in the evening after the official end-of-day data is in. It judges every frozen idea whose next sessions are
stored and keeps the outcomes APART from the frozen reports (``ideas:out:<date>``): what was said is never edited, what happened is
attached. The live record the next report shows is built from those outcomes.

Both persist in ``kv`` under the ``ideas:`` prefix, which the nightly housekeeping prunes with the other records.
"""

from __future__ import annotations

import logging
import sqlite3
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app import jsonutil
from app.clock import at_ist, iso, parse_ts
from app.logging_setup import redact
from app.release import release_id
from automation.context import JobContext, JobResult
from automation.scan import monotonic_at
from backtesting.ideas import GROUPS, build_record, idea_outcome
from database.store import Store
from features.loader import load_panel
from features.swing import last_row, swing_frames, volume_fraction
from features.technical import ema
from ingestion.freshness import ist_label
from ingestion.livefeed import merge_live
from ingestion.nse import LIVE_LISTS
from ingestion.pipeline import (
    collect_kotak_indices,
    collect_kotak_quotes,
    collect_live_equities,
    collect_live_indices,
)
from ranking.ideas import IdeaConfig, eligibility, market_regime, overnight_block, rank_ideas
from ranking.types import IdeasReport, TradeIdea

log = logging.getLogger(__name__)

RUN_PREFIX = "ideas:run:"  # the frozen report of a trading day
OUT_PREFIX = "ideas:out:"  # what happened to its ideas, attached later
MIN_LIVE_COVERAGE = 0.60  # share of the universe that needs a live row of today for the scan to be worth sending
MAX_UNSEEN_SHARE = 0.40  # more live names than this without a stored previous close = the history is incomplete: no scan
HISTORY_DAYS = 430  # calendar days of daily history: a 252-session window plus the 200-day average
BASIS_TOLERANCE = 0.02  # the exchange's previous close may differ this much from the stored one before the name is left out
NO_PREVIOUS_CLOSE = "no stored previous close"
BASIS_MISMATCH = "price basis differs from the stored history (an ex-date today?)"
EVENT_LOOKAHEAD_DAYS = 9
SWING_EVENT_SESSIONS = 5  # a results date this many sessions ahead or less is inside a swing's horizon
DEADLINE_MARGIN_SEC = 20.0  # live requests must be over this long before the report deadline
LATE_FETCH_SEC = 90.0  # a run that starts when the report deadline is (nearly) over may spend this long on NSE
NSE_GRACE_SEC = 45.0  # once the broker has answered, NSE gets this much more to answer (the broker alone can carry the scan)
EVALUATE_DAYS = 14  # calendar days back whose ideas are judged (again) each night
RECORD_DAYS = 120  # the live record in a report covers the runs of this many calendar days
INDEX_NAMES = {"nifty": "NIFTY 50", "vix": "INDIA VIX"}


@dataclass
class LiveDay:
    """The usable live rows of today and what was left out of the answer."""

    rows: dict[str, dict[str, Any]]
    stale: int = 0
    incomplete: int = 0
    source_ts: str | None = None
    indices: dict[str, dict[str, Any]] = field(default_factory=dict)
    lists: tuple[str, ...] = ()
    feeds: dict[str, int] = field(default_factory=dict)  # rows taken from each live feed: {"nse": n, "kotak": n}
    cross_checked: int = 0  # names both feeds had and agreed on
    broker: str = "off"  # the broker feed: off | failed | lagging | unstamped | used
    notes: list[str] = field(default_factory=list)  # what a report must say about the feeds

    def source_label(self, universe: int) -> str:
        """Where the live prices of a report came from, and how many names they covered."""
        count = f"{len(self.rows)} of {universe} names with a complete live row"
        if self.broker == "used" and not self.feeds.get("nse"):
            return f"Kotak broker feed (NSE did not answer): {count}"
        if self.broker == "used":
            return f"NSE ({' + '.join(self.lists)}) + Kotak broker feed: {count}; {self.cross_checked} confirmed by both"
        return f"NSE ({' + '.join(self.lists)}): {count}"


def run_key(day: date | str) -> str:
    return f"{RUN_PREFIX}{day}"


def outcome_key(day: date | str) -> str:
    return f"{OUT_PREFIX}{day}"


# ------------------------------------------------------------------------------------------------ live
def fetch_bound(ctx: JobContext, deadline_at: datetime) -> float:
    """The ``clock.monotonic()`` instant by which the live requests must be over. A scan that is on time must not run past the
    report deadline, or its report would be withheld. One that starts after it (a forced, a manual or a dry run) has no deadline to
    protect, and a bound in the past would make the HTTP client refuse every attempt: it gets a budget of its own instead."""
    margin = deadline_at - timedelta(seconds=DEADLINE_MARGIN_SEC)
    if ctx.clock.now() < margin:
        return monotonic_at(ctx, margin)
    return ctx.budget(LATE_FETCH_SEC)


def live_feed_name(settings: Any) -> str:
    """What the operator calls the live feed in a failure notice."""
    return "NSE and Kotak live quotes" if settings.kotak_configured else "NSE live quotes"


def _trading_now(ctx: JobContext, day: date) -> bool:
    """Is the regular session of ``day`` under way right now? (A live feed is only judged for age while it is.)"""
    session = ctx.calendar.session(day)
    now = ctx.clock.now()
    return session.open is not None and session.close is not None and session.open <= now <= session.close


def fetch_live(ctx: JobContext, symbols: list[str], deadline: float | None, errors: list[str] | None = None) -> LiveDay | None:
    """The live rows of the universe from NSE and, when ``KOTAK_CONSUMER_KEY`` is set, from the broker too, merged by ``ingestion.livefeed``
    (each feed's rows must be stamped on today's date and complete before the feeds are compared; a name both feeds have must agree; the
    fresher row wins; a lagging broker feed is not used). Either feed alone carries the scan when the other fails; ``None`` when neither
    answered, ``errors`` then holds why.

    The broker is asked first: it answers in a second or two, where a blocked NSE can use up its whole retry budget, and a call that starts
    after the deadline gets nothing. Once the broker has answered NSE gets ``NSE_GRACE_SEC`` more at most."""
    s, day = ctx.settings, ctx.run_date
    lists = LIVE_LISTS.get(s.universe_mode, LIVE_LISTS["fno_nifty100"])
    nse = ctx.nse_api()
    quotes = collect_kotak_quotes(ctx.store, ctx.kotak(), symbols, day, deadline=deadline, errors=errors) if s.kotak_configured else None
    nse_deadline = deadline
    if quotes is not None:
        grace = ctx.budget(NSE_GRACE_SEC)
        nse_deadline = grace if deadline is None else min(deadline, grace)
    snapshot = collect_live_equities(ctx.store, nse, lists, symbols, deadline=nse_deadline, errors=errors)
    if snapshot is None and quotes is None:
        return None
    merged = merge_live(snapshot, quotes, configured=s.kotak_configured, day=day, lag_limit_sec=s.kotak_max_lag_sec,
                        judge_lag=_trading_now(ctx, day))
    live = LiveDay({row["symbol"]: row for row in merged.rows}, stale=merged.stale, incomplete=merged.incomplete, source_ts=merged.source_ts,
                   lists=tuple(lists), feeds=merged.feeds, cross_checked=merged.cross_checked, broker=merged.broker, notes=merged.notes)
    if snapshot is not None:
        live.indices = collect_live_indices(ctx.store, nse, deadline=nse_deadline)
    if INDEX_NAMES["nifty"] not in live.indices and merged.broker == "used":  # NSE's index call gave nothing: the broker has them too
        for name, row in collect_kotak_indices(ctx.store, ctx.kotak(), deadline=deadline).items():
            live.indices.setdefault(name, row)
    log.info("live feeds: %d NSE + %d Kotak rows (%d confirmed by both, %d in conflict), broker feed %s", merged.feeds["nse"],
             merged.feeds["kotak"], merged.cross_checked, len(merged.conflicts), merged.broker)
    return live


def session_fraction(ctx: JobContext, day: date, moment: datetime) -> float:
    """The share of the session's volume normally traded by ``moment`` (1.0 once the session is over)."""
    session = ctx.calendar.session(day)
    if session.open is None or session.close is None or moment >= session.close:
        return 1.0
    return volume_fraction((moment - session.open).total_seconds() / 60.0)


# ----------------------------------------------------------------------------------------------- table
def stored_panel(ctx: JobContext, history_days: int = HISTORY_DAYS) -> tuple[Any, date]:
    """The stored daily history up to the previous session, plus an empty row for today, and the date of that previous session.
    Raises ``ValueError`` when the history is not complete up to the previous REGULAR session (a special session such as Muhurat
    trading is not in the panel by design): figures computed over a gap would read a multi-day move as today's."""
    day = ctx.run_date
    panel = load_panel(ctx.store, start=(day - timedelta(days=history_days)).isoformat(), end=(day - timedelta(days=1)).isoformat(),
                       today=day, calendar=ctx.calendar)
    if len(panel.dates) < 2 or panel.dates[-1].date() != day:
        raise ValueError("the stored history has no previous session")
    last_stored = panel.dates[-2].date()
    missing = [d for d in ctx.calendar.trading_days(last_stored + timedelta(days=1), day - timedelta(days=1))
               if ctx.calendar.session(d).kind == "regular"]
    if missing:
        raise ValueError(f"the stored history ends on {last_stored} but {missing[-1]} was a regular session: run collect_india")
    return panel, last_stored


def basis_exclusions(symbols: list[str], rows: dict[str, dict[str, Any]], stored_prev: pd.Series, last_stored: date) -> dict[str, str]:
    """The names whose exchange previous close cannot be matched with the stored one (none stored, or more than ``BASIS_TOLERANCE``
    apart: an ex-date today). Raises ``ValueError`` when so many are affected that the stored history itself is off."""
    excluded: dict[str, str] = {}
    for symbol in symbols:
        stored = stored_prev.get(symbol)
        if stored is None or not np.isfinite(stored) or stored <= 0:
            excluded[symbol] = NO_PREVIOUS_CLOSE
        elif abs(float(rows[symbol]["prev_close"]) / float(stored) - 1.0) > BASIS_TOLERANCE:
            excluded[symbol] = BASIS_MISMATCH
    unseen = sum(1 for why in excluded.values() if why == NO_PREVIOUS_CLOSE)
    if symbols and unseen / len(symbols) > MAX_UNSEEN_SHARE:
        raise ValueError(f"{unseen} of {len(symbols)} names have no stored bar of {last_stored}: run collect_india")
    shifted = sum(1 for why in excluded.values() if why == BASIS_MISMATCH)
    if symbols and shifted / len(symbols) > MAX_UNSEEN_SHARE:  # an ex-date moves a few names; this many means the stored closes are off
        raise ValueError(f"the stored close of {last_stored} differs from the exchange's previous close for {shifted} of "
                         f"{len(symbols)} names: run collect_india")
    return excluded


def build_table(ctx: JobContext, live: LiveDay, instruments: dict[str, dict[str, Any]], fraction: float) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """The one-row-per-stock table of the closing hour, the Nifty close series with today's live value, and the names that were
    left out. Raises ``ValueError`` when the stored history is not complete up to the previous REGULAR session (see ``stored_panel``)."""
    panel, last_stored = stored_panel(ctx)
    stamp = panel.dates[-1]
    stored_prev = panel.c.iloc[-2]
    symbols = [s for s in panel.symbols if s in live.rows]
    for symbol in symbols:
        row = live.rows[symbol]
        for frame, key in ((panel.o, "open"), (panel.h, "high"), (panel.low, "low"), (panel.c, "last"), (panel.v, "volume")):
            frame.loc[stamp, symbol] = float(row[key])
    excluded = basis_exclusions(symbols, live.rows, stored_prev, last_stored)
    nifty_close = panel.nifty["close"].astype(float).copy()
    live_nifty = live.indices.get(INDEX_NAMES["nifty"])
    if live_nifty and len(nifty_close):
        nifty_close.iloc[-1] = live_nifty["last"]
    frames = swing_frames(panel.o, panel.h, panel.low, panel.c, panel.v, nifty_close, live_fraction=fraction)
    # A name left out has meaningless figures (a split makes its move -50 %): it must not take part in anything that looks at the
    # day's universe either (the sector peers' move, the breadth, the percentile ranks), so it is not in the table at all.
    usable = [s for s in symbols if s not in excluded]
    table = last_row(frames).loc[usable]
    table["is_fno"] = [bool(instruments.get(s, {}).get("is_fno")) for s in table.index]
    table["sector_chg"] = _peer_move(table, instruments)
    table["event_days"] = pd.Series(_event_sessions(ctx, set(usable)), dtype=float).reindex(table.index)
    info = {"excluded": excluded, "panel_symbols": len(panel.symbols)}
    return table, nifty_close, info


def _peer_move(table: pd.DataFrame, instruments: dict[str, dict[str, Any]]) -> pd.Series:
    """Per stock, the average move today of the other stocks of its sector (at least three of them), else NaN."""
    sectors = pd.Series({s: instruments.get(s, {}).get("sector") for s in table.index})
    move = table["ret1"]
    group = move.groupby(sectors)
    total, count = group.transform("sum"), group.transform("count")
    return ((total - move) / (count - 1)).where(count - 1 >= 3)


def _event_sessions(ctx: JobContext, symbols: set[str]) -> dict[str, float]:
    """Sessions from today to the results announcement of each stock that has one within a swing's horizon. 0 is today, which NSE
    companies mostly announce after the close; a date that is no session (a Saturday meeting, a holiday) is read the same way as
    the session before it, because its news lands before the next open: the overnight gap a BTST would carry."""
    day = ctx.run_date
    end = day + timedelta(days=EVENT_LOOKAHEAD_DAYS)
    sessions = [d.isoformat() for d in ctx.calendar.trading_days(day, end)]
    out: dict[str, float] = {}
    for row in ctx.store.board_meetings_between(day.isoformat(), end.isoformat()):
        n = bisect_right(sessions, row["meeting_date"]) - 1  # the last session on or before the meeting
        if 0 <= n <= SWING_EVENT_SESSIONS and row["is_results"] and row["symbol"] in symbols:
            out[row["symbol"]] = float(min(n, out.get(row["symbol"], n)))
    return out


def market_context(table: pd.DataFrame, nifty_close: pd.Series, live: LiveDay) -> dict[str, Any]:
    """The market as the closing hour sees it: Nifty and VIX today, breadth among the universe, Nifty's trend and the regime."""
    market: dict[str, Any] = {"median_chg": _f(table["ret1"].median()), "advancers": int((table["ret1"] > 0).sum()),
                              "decliners": int((table["ret1"] < 0).sum())}
    nifty, vix = live.indices.get(INDEX_NAMES["nifty"]), live.indices.get(INDEX_NAMES["vix"])
    market["nifty_chg"] = _f(nifty["change_pct"]) if nifty else None
    market["vix"] = _f(vix["last"]) if vix else None
    market["vix_chg"] = _f(vix["change_pct"]) if vix else None
    closes = nifty_close.dropna()
    if len(closes) >= 60 and nifty is not None:
        fast, slow = ema(closes, 20).iloc[-1], ema(closes, 50).iloc[-1]
        market["nifty_above_ema20"] = bool(closes.iloc[-1] > fast)
        market["nifty_ema20_above_ema50"] = bool(fast > slow)
    market["regime"] = market_regime(market)
    return market


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, 3) if np.isfinite(out) else None


# ---------------------------------------------------------------------------------------------- record
def load_runs(store: Store, start: str, end: str) -> dict[str, dict[str, Any]]:
    """The frozen reports of the runs dated in ``[start, end)``, by trade date."""
    rows = store.conn.execute("SELECT key, value FROM kv WHERE key >= ? AND key < ? ORDER BY key",
                              (run_key(start), run_key(end))).fetchall()
    runs: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = jsonutil.loads(row["value"])
        if isinstance(payload, dict) and isinstance(payload.get("report"), dict):
            runs[row["key"][len(RUN_PREFIX):]] = payload["report"]
    return runs


def _final(outcome: dict[str, Any]) -> dict[str, Any]:
    """An outcome judged on bars that were not the official end-of-day file is provisional: for the record it still waits (as
    pending) for them, and the next evaluation replaces it."""
    return outcome if outcome.get("official") else {**outcome, "complete": False}


def live_record(store: Store, day: date) -> dict[str, dict[str, Any]]:
    """What the ideas of the earlier runs did, per list: from the frozen reports and the outcomes attached to them. A report that
    was built after the report deadline is left out: too little of the entry window was left to act on its levels."""
    runs = load_runs(store, (day - timedelta(days=RECORD_DAYS)).isoformat(), day.isoformat())
    runs = {d: report for d, report in runs.items() if not report.get("late")}
    ideas = {d: list(report.get("ideas") or []) for d, report in runs.items()}
    outcomes: dict[str, dict[str, Any]] = {}
    for d in runs:
        stored = store.kv_get_json(outcome_key(d))
        if isinstance(stored, dict) and isinstance(stored.get("ideas"), dict):
            outcomes[d] = {key: _final(o) for key, o in stored["ideas"].items() if isinstance(o, dict)}
    return build_record(outcomes, ideas)


# -------------------------------------------------------------------------------------------------- scan
def _fail(ctx: JobContext, reason: str) -> JobResult:
    ctx.store.add_alert("error", "ideas_scan_failed", reason, dedupe_key=f"ideas-failed:{ctx.run_key}", dedupe_minutes=30)
    if ctx.last_attempt and not ctx.dry_run:
        ctx.notifier().send_failure_notice(ctx.run_key, "closing", reason)
    return JobResult("failed", reason)


def frozen_report(store: Store, day: date | str) -> IdeasReport | None:
    payload = store.kv_get_json(run_key(day))
    if isinstance(payload, dict) and isinstance(payload.get("report"), dict):
        return IdeasReport.from_dict(payload["report"])
    return None


def deliver(ctx: JobContext, report: IdeasReport) -> JobResult:
    """Send (or re-send, idempotently) a frozen report and translate the outcome into a job result. A report that goes out after the
    report deadline (only a forced send can) says so in the copy that is sent; the frozen report itself is never edited."""
    if ctx.dry_run:
        return JobResult("success", "dry run: report not sent")
    if ctx.force and not report.late and ctx.clock.now() > at_ist(ctx.run_date, ctx.settings.ideas_report_deadline_time):
        report = replace(report, late=True)
    outcome = ctx.notifier().send_ideas(report, force=ctx.force)
    if outcome.ok:
        return JobResult("success", f"report {outcome.status}", {"message_ids": outcome.message_ids})
    if outcome.status == "withheld":
        return JobResult("success", f"ideas frozen; report withheld ({outcome.reason})")
    if ctx.last_attempt:
        ctx.notifier().send_failure_notice(ctx.run_key, "closing", "report could not be delivered", outcome.reason)
    return JobResult("failed", f"report delivery failed: {outcome.reason}")


def run_closing_scan(ctx: JobContext) -> JobResult:
    s = ctx.settings
    if not s.ideas_enabled:
        return JobResult("skipped", "IDEAS_ENABLED=false")
    ctx.ensure_calendar()
    day = ctx.run_date
    if not ctx.calendar.is_trading_day(day):
        return JobResult("skipped", f"exchange closed on {day} ({ctx.calendar.holiday_name(day)})")
    kept = None if ctx.dry_run else frozen_report(ctx.store, day)
    if kept is not None:  # frozen once: a retry, a late run or --force sends what was frozen, never a second scan
        return deliver(ctx, kept)
    try:
        return _scan(ctx)
    except Exception as exc:  # noqa: BLE001 - one bad scan must be reported, not crash the daemon
        log.exception("closing scan failed")
        return _fail(ctx, f"{type(exc).__name__}: {redact(str(exc))[:200]}")


def _eligible(table: pd.DataFrame, settings: Any) -> int:
    return int(eligibility(table, IdeaConfig.from_settings(settings)).sum())


def _scan(ctx: JobContext) -> JobResult:
    s, store, clock, day = ctx.settings, ctx.store, ctx.clock, ctx.run_date
    universe = store.list_instruments(kind="equity", in_universe=True, status="active")
    symbols = [r["symbol"] for r in universe]
    if not symbols:
        return _fail(ctx, "no instruments in the universe (run bootstrap)")
    instruments = {r["symbol"]: {"name": r["name"], "sector": r["sector"], "is_fno": bool(r["is_fno"])} for r in universe}
    why: list[str] = []
    live = fetch_live(ctx, symbols, fetch_bound(ctx, at_ist(day, s.ideas_report_deadline_time)), why)
    if live is None:
        reason = "; ".join(dict.fromkeys(why))[:200] or "blocked or offline"
        return _fail(ctx, f"{live_feed_name(s)} unavailable ({reason}); nothing is guessed from older prices")
    coverage = len(live.rows) / len(symbols)
    if coverage < MIN_LIVE_COVERAGE:
        detail = f"{len(live.rows)} of {len(symbols)} names have a complete live row of {day}"
        if live.stale:
            detail += f"; {live.stale} are stamped on an earlier session (the market is not trading today yet or any more?)"
        return _fail(ctx, f"too few live quotes: {detail}")

    now = clock.now()
    stamp = parse_ts(live.source_ts)
    cutoff = stamp if stamp is not None and stamp <= now else now
    fraction = session_fraction(ctx, day, cutoff)
    # What the report must not leave unsaid comes first: only the first few notes fit into a message.
    notes: list[str] = list(live.notes)
    if INDEX_NAMES["nifty"] not in live.indices:
        notes.append("NSE index quotes unavailable: Nifty and VIX not shown, relative strength to Nifty not used")
    table, nifty_close, info = build_table(ctx, live, instruments, fraction)
    market = market_context(table, nifty_close, live)
    cfg = IdeaConfig.from_settings(s)
    pool = rank_ideas(table, market, cfg, instruments=instruments, pool=True)
    catalysts: dict[str, Any] = {}
    if pool:
        intel = ctx.intel()
        catalysts = intel.catalysts(sorted({i.symbol for i in pool}), now)
        if intel.coverage_note():
            notes.append(f"news coverage reduced: {intel.coverage_note()}")
    ideas = rank_ideas(table, market, cfg, instruments=instruments, catalysts=catalysts)
    report = _build_report(ctx, ideas, market, live, table, info, len(symbols), fraction, notes, cutoff, now)
    if ctx.dry_run:
        return JobResult("success", "dry run: nothing frozen or sent", {"report": report.to_dict()})
    shares = ", ".join(f"{g} {sum(1 for i in ideas if _group(i) == g)}" for g in GROUPS)
    store.kv_set_json(run_key(day), {"report": report.to_dict(), "frozen_at": iso(now), "code": release_id(), "cutoff": iso(cutoff)})
    log.info("closing scan frozen: %d ideas (%s), regime %s, volume fraction %.2f", len(ideas), shares, market.get("regime"), fraction)
    outcome = deliver(ctx, report)
    outcome.metrics.update(ideas=len(ideas), universe=len(symbols), live_rows=len(live.rows), late=report.late)
    return outcome


def _group(idea: TradeIdea) -> str:
    return "SHORT" if idea.side == "SHORT" and idea.horizon in ("BTST", "SWING") else idea.horizon


def _build_report(ctx: JobContext, ideas: list[TradeIdea], market: dict[str, Any], live: LiveDay, table: pd.DataFrame,
                  info: dict[str, Any], n_universe: int, fraction: float, notes: list[str], cutoff: datetime, now: datetime
                  ) -> IdeasReport:
    s, day = ctx.settings, ctx.run_date
    excluded = info["excluded"]
    source = live.source_label(n_universe)
    block = overnight_block(market)[1]
    if block:
        notes.append(block)
    if not any(i.horizon in ("BTST", "SWING") for i in ideas):
        notes.append("no BTST or swing setup passed the gates today: nothing is forced")
    age = (now - cutoff).total_seconds() / 60.0
    if age > 15 and fraction < 1.0:
        notes.append(f"the live prices are {age:.0f} minutes old")
    if fraction < 1.0:
        notes.append(f"volume is projected to a full day: {fraction:.0%} of a session's volume is normally traded by now (an estimate)")
    if excluded:
        notes.append(f"{len(excluded)} name(s) left out: " + "; ".join(sorted(set(excluded.values()))))
    if live.stale or live.incomplete:
        notes.append(f"{live.stale} stale and {live.incomplete} incomplete live row(s) were not used")
    late = now > at_ist(day, s.ideas_report_deadline_time)
    return IdeasReport(
        trade_date=day.isoformat(), snapshot_ist=ist_label(cutoff), entry_by=s.ideas_entry_by_time[:5], source=source, ideas=ideas,
        market={k: v for k, v in market.items() if v is not None}, record=live_record(ctx.store, day), notes=notes[:6], mode=s.mode,
        universe_size=n_universe, n_eligible=_eligible(table, s), late=late,
    )


# ----------------------------------------------------------------------------------------- evaluation
def _bars_after(store: Store, start: str, end: str, symbols: set[str]) -> dict[str, pd.DataFrame]:
    """The daily bars of ``symbols`` for the sessions after ``start`` up to ``end``: open / high / low / close and the source rank
    (2 = the official end-of-day file), by symbol."""
    try:
        rows = store.conn.execute(
            "SELECT i.symbol AS symbol, b.trade_date AS trade_date, b.open AS open, b.high AS high, b.low AS low, b.close AS close, "
            "b.source_rank AS source_rank FROM bars_daily b JOIN instruments i USING (instrument_id) "
            "WHERE i.kind = 'equity' AND b.trade_date > ? AND b.trade_date <= ? ORDER BY i.symbol, b.trade_date", (start, end)).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"bars unreadable: {type(exc).__name__}") from exc
    frame = pd.DataFrame([dict(r) for r in rows if r["symbol"] in symbols])
    if frame.empty:
        return {}
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return {symbol: group.set_index("trade_date").drop(columns="symbol") for symbol, group in frame.groupby("symbol")}


def evaluate_ideas(ctx: JobContext) -> JobResult:
    """Judge the frozen ideas of the last ``EVALUATE_DAYS`` days whose next sessions are stored, and attach the outcomes. Every night
    the whole window is judged again: an outcome that waited for its horizon or for the official bars gets completed, and one
    that was judged on provisional bars is replaced as soon as the official end-of-day file is in."""
    store, day = ctx.store, ctx.run_date
    runs = load_runs(store, (day - timedelta(days=EVALUATE_DAYS)).isoformat(), day.isoformat())
    if not runs:
        return JobResult("skipped", "no frozen trade ideas to evaluate")
    judged = pending = 0
    for trade_day, report in sorted(runs.items()):
        ideas = [i for i in report.get("ideas") or [] if isinstance(i, dict)]
        bars = _bars_after(store, trade_day, day.isoformat(), {str(i.get("symbol")) for i in ideas})
        found: dict[str, Any] = {}
        for idea in ideas:
            frame = bars.get(str(idea.get("symbol")))
            outcome = idea_outcome(idea, frame) if frame is not None else None
            if outcome is None:
                pending += 1
                continue
            used = frame.sort_index().iloc[: max(int(outcome.get("sessions") or 1), 1)]
            outcome["official"] = bool((used["source_rank"].fillna(0) >= 2).all())
            found[f"{idea.get('horizon')}:{idea.get('side')}:{idea.get('symbol')}"] = outcome
            if _final(outcome).get("complete"):
                judged += 1
            else:
                pending += 1
        store.kv_set_json(outcome_key(trade_day), {"evaluated_at": iso(ctx.clock.now()), "ideas": found})
    record = live_record(store, day + timedelta(days=1))
    parts = [f"{g} {record[g]['n']}" for g in GROUPS if g in record]
    return JobResult("success", f"{judged} idea(s) settled over {len(runs)} run(s), {pending} still waiting"
                     + (f"; live record (settled): {', '.join(parts)}" if parts else ""), {"judged": judged, "pending": pending})
