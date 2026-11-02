"""The closing-hour trade-idea scan (``closing_scan``) and the evaluation of its ideas (``evaluate_ideas``).

``closing_scan`` runs at 14:30 IST. NSE's live quotes (one call for the whole F&O segment) are laid over the stored daily history as
the running session of today; ``features.swing`` turns that panel into the features of the closing hour, ``ranking.ideas`` into
BTST, swing, bearish and next-day intraday ideas, the news pipeline adds catalysts to the shortlist, and the report is FROZEN
(written once per trading day and re-sent unchanged after a failed delivery, exactly like the morning snapshots) and sent to Telegram.
A forced run scans again only while the frozen report was never delivered; once it was sent, what was said is the record and a new
scan can only be previewed (``--dry-run``). Nothing is ever traded: shadow mode.

``evaluate_ideas`` runs in the evening after the official end-of-day data is in. It judges every frozen idea whose next sessions are
stored and keeps the outcomes APART from the frozen reports (``ideas:out:<date>``): what was said is never edited, what happened is
attached. The live record the next report shows is built from those outcomes.

Both persist in ``kv`` under the ``ideas:`` prefix, which the nightly housekeeping prunes with the other records.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app import jsonutil
from app.clock import at_ist, iso, ist_date, parse_ts
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
from ingestion.nse import LIVE_LISTS
from ingestion.pipeline import collect_live_equities, collect_live_indices
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


def run_key(day: date | str) -> str:
    return f"{RUN_PREFIX}{day}"


def outcome_key(day: date | str) -> str:
    return f"{OUT_PREFIX}{day}"


# ------------------------------------------------------------------------------------------------ live
def fetch_live(ctx: JobContext, symbols: list[str], deadline: float | None) -> LiveDay | None:
    """NSE's live rows of the universe, keeping only complete rows that are stamped on today's date (a row from an earlier session
    is not today's trading and is counted, never used). ``None`` when NSE did not answer at all."""
    s, day = ctx.settings, ctx.run_date
    lists = LIVE_LISTS.get(s.universe_mode, LIVE_LISTS["fno_nifty100"])
    nse = ctx.nse_api()
    snapshot = collect_live_equities(ctx.store, nse, lists, symbols, deadline=deadline)
    if snapshot is None:
        return None
    live = LiveDay({}, source_ts=snapshot.source_ts, lists=tuple(lists))
    for row in snapshot.rows:
        stamp = parse_ts(row.get("row_ts"))
        if stamp is None or ist_date(stamp) != day:
            live.stale += 1
        elif any(row.get(key) is None for key in ("open", "high", "low", "last", "prev_close", "volume")):
            live.incomplete += 1
        else:
            live.rows[row["symbol"]] = row
    live.indices = collect_live_indices(ctx.store, nse, deadline=deadline)
    return live


def session_fraction(ctx: JobContext, day: date, moment: datetime) -> float:
    """The share of the session's volume normally traded by ``moment`` (1.0 once the session is over)."""
    session = ctx.calendar.session(day)
    if session.open is None or session.close is None or moment >= session.close:
        return 1.0
    return volume_fraction((moment - session.open).total_seconds() / 60.0)


# ----------------------------------------------------------------------------------------------- table
def build_table(ctx: JobContext, live: LiveDay, instruments: dict[str, dict[str, Any]], fraction: float) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """The one-row-per-stock table of the closing hour, the Nifty close series with today's live value, and the names that were
    left out. Raises ``ValueError`` when the stored history is not complete up to the previous REGULAR session (a special session
    such as Muhurat trading is not in the panel by design): ideas computed over a gap would read a multi-day move as today's."""
    store, day = ctx.store, ctx.run_date
    panel = load_panel(store, start=(day - timedelta(days=HISTORY_DAYS)).isoformat(), end=(day - timedelta(days=1)).isoformat(),
                       today=day, calendar=ctx.calendar)
    if len(panel.dates) < 2 or panel.dates[-1].date() != day:
        raise ValueError("the stored history has no previous session")
    last_stored = panel.dates[-2].date()
    missing = [d for d in ctx.calendar.trading_days(last_stored + timedelta(days=1), day - timedelta(days=1))
               if ctx.calendar.session(d).kind == "regular"]
    if missing:
        raise ValueError(f"the stored history ends on {last_stored} but {missing[-1]} was a regular session: run collect_india")
    stamp = panel.dates[-1]
    stored_prev = panel.c.iloc[-2]
    symbols = [s for s in panel.symbols if s in live.rows]
    excluded: dict[str, str] = {}
    for symbol in symbols:
        row = live.rows[symbol]
        for frame, key in ((panel.o, "open"), (panel.h, "high"), (panel.low, "low"), (panel.c, "last"), (panel.v, "volume")):
            frame.loc[stamp, symbol] = float(row[key])
        stored = stored_prev.get(symbol)
        if stored is None or not np.isfinite(stored) or stored <= 0:
            excluded[symbol] = NO_PREVIOUS_CLOSE
        elif abs(float(row["prev_close"]) / float(stored) - 1.0) > BASIS_TOLERANCE:
            excluded[symbol] = BASIS_MISMATCH
    unseen = sum(1 for why in excluded.values() if why == NO_PREVIOUS_CLOSE)
    if symbols and unseen / len(symbols) > MAX_UNSEEN_SHARE:
        raise ValueError(f"{unseen} of {len(symbols)} names have no stored bar of {last_stored}: run collect_india")
    nifty_close = panel.nifty["close"].astype(float).copy()
    live_nifty = live.indices.get(INDEX_NAMES["nifty"])
    if live_nifty and len(nifty_close):
        nifty_close.iloc[-1] = live_nifty["last"]
    frames = swing_frames(panel.o, panel.h, panel.low, panel.c, panel.v, nifty_close, live_fraction=fraction)
    table = last_row(frames).loc[symbols]
    table["excluded"] = pd.Series(excluded, dtype=object).reindex(table.index).fillna("")
    table["is_fno"] = [bool(instruments.get(s, {}).get("is_fno")) for s in table.index]
    table["sector_chg"] = _peer_move(table, instruments)
    table["event_days"] = pd.Series(_event_sessions(ctx, set(symbols)), dtype=float).reindex(table.index)
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
    """Sessions from today to the results announcement of each stock that has one within a swing's horizon (0 = today, which
    NSE companies mostly announce after the close)."""
    day = ctx.run_date
    end = day + timedelta(days=EVENT_LOOKAHEAD_DAYS)
    index = {d.isoformat(): n for n, d in enumerate(ctx.calendar.trading_days(day, end))}
    out: dict[str, float] = {}
    for row in ctx.store.board_meetings_between(day.isoformat(), end.isoformat()):
        n = index.get(row["meeting_date"])
        if n is not None and n <= SWING_EVENT_SESSIONS and row["is_results"] and row["symbol"] in symbols:
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
    """Send (or re-send, idempotently) a frozen report and translate the outcome into a job result."""
    if ctx.dry_run:
        return JobResult("success", "dry run: report not sent")
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
    if kept is not None:
        if not ctx.force:
            return deliver(ctx, kept)
        if ctx.notifier().ideas_delivered(day.isoformat()):  # what was said is the record: a forced scan must not replace it
            return JobResult("skipped", f"the report of {day} was delivered and stays the record (--dry-run previews a new scan)")
        log.warning("forced closing scan replaces the frozen report of %s, which was never delivered", day)
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
    deadline_at = at_ist(day, s.ideas_report_deadline_time)
    live = fetch_live(ctx, symbols, monotonic_at(ctx, deadline_at - timedelta(seconds=DEADLINE_MARGIN_SEC)))
    if live is None:
        return _fail(ctx, "NSE live quotes unavailable (blocked or offline); nothing is guessed from older prices")
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
    notes: list[str] = []
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
    source = f"NSE ({' + '.join(live.lists)}): {len(live.rows)} of {n_universe} names with a complete live row"
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
