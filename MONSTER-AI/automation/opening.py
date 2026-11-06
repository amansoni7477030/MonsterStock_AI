"""The 09:30 opening read (``opening_scan``) and the evaluation of its reports (``evaluate_opening``).

``opening_scan`` starts at 09:30:15 IST, just after the third five-minute candle of the session has closed. NSE's live quotes (one call for
the whole F&O segment) hold each stock's open, high, low and last price: at that moment the day's high and low ARE the first fifteen
minutes' range. ``ranking.opening`` reads that range against each stock's own normal (the stored daily history), estimates how WIDE
the day will be (validated out of sample, see its docstring) and builds the report: the names of the 09:12 report read again, the new
opening movers, the market at the open and reference levels around the opening range. A scan that has to retry a few minutes late
is normalised back to the fifteen-minute basis; one later than 30 minutes after the open is not an opening read and does not run.

The report is FROZEN, exactly like the closing-hour one: written once per trading day and re-sent unchanged after a failed delivery; a
retry, a late run and ``--force`` send what was frozen (``--force`` only lifts the report deadline, and the copy says it is late); a new
scan can only be previewed (``--dry-run``). The scored table of ALL eligible names is frozen beside it (``opening:state:<date>``) so the
live record judges the model on every name it scored, and a later refit has point-in-time rows. Nothing is ever traded: shadow mode.

``evaluate_opening`` runs in the evening after the official end-of-day bars are in. It judges every frozen report of the last two
weeks on the day's own bar (and, for the listed names, on Yahoo's five-minute bars while they reach back that far), keeps the outcomes
APART from the frozen reports (``opening:out:<date>``) and builds the live record the next report shows.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Any

from app import jsonutil
from app.clock import IST, at_ist, iso, parse_ts
from app.errors import DataUnavailable
from app.logging_setup import redact
from app.release import release_id
from automation.calendar import OPEN_TIME
from automation.context import JobContext, JobResult
from automation.ideas import (
    INDEX_NAMES,
    MIN_LIVE_COVERAGE,
    basis_exclusions,
    fetch_bound,
    fetch_live,
    live_feed_name,
    stored_panel,
)
from backtesting.opening import ahead_outcome, build_record, day_outcome, plan_outcome, universe_outcome
from database.store import Store
from ingestion.freshness import ist_label
from ingestion.pipeline import kotak_tokens
from ranking.opening import (
    MAX_ELAPSED_MIN,
    MIN_ELAPSED_MIN,
    MODEL_INFO,
    OpeningConfig,
    build_ideas,
    eligibility,
    history_stats,
    market_pulse,
    opening_table,
    range_growth,
    score,
    state_rows,
)
from ranking.types import Candidate, OpeningReport, ReportContext

log = logging.getLogger(__name__)

RUN_PREFIX = "opening:run:"  # the frozen report of a trading day
STATE_PREFIX = "opening:state:"  # the scored table of all eligible names, frozen beside it
OUT_PREFIX = "opening:out:"  # what happened, attached later
HISTORY_DAYS = 150  # calendar days of stored history (about 100 sessions): the 20-session range and the 14-day ATR need far less
EVALUATE_DAYS = 14  # calendar days back whose reports are judged (again) each night
RECORD_DAYS = 120  # the live record in a report covers the runs of this many calendar days
INTRADAY_REACH_DAYS = 6  # Yahoo's five-minute feed reaches back five sessions: older reports are not judged intraday on it
INTRADAY_BUDGET_SEC = 120.0  # what one evening run spends in all on five-minute bars (about a second per name), however many reports wait
# The 72 five-minute bars of a day, 09:15 .. 15:10: what the broker serves of every session (the last 15 minutes are not in its archive, or
# only as a flat artefact after the day). A day is judged only on all of them, and on nothing else, so that the same day reads the same
# whether it is judged the evening it traded or a week later.
SESSION_HHMM = tuple(f"{minute // 60:02d}:{minute % 60:02d}" for minute in range(9 * 60 + 15, 15 * 60 + 15, 5))
MIN_STATE_NAMES = 20  # fewer judged names than this make no calibration statement


def run_key(day: date | str) -> str:
    return f"{RUN_PREFIX}{day}"


def state_key(day: date | str) -> str:
    return f"{STATE_PREFIX}{day}"


def outcome_key(day: date | str) -> str:
    return f"{OUT_PREFIX}{day}"


# ------------------------------------------------------------------------------------------------ stored
def frozen_report(store: Store, day: date | str) -> OpeningReport | None:
    payload = store.kv_get_json(run_key(day))
    if isinstance(payload, dict) and isinstance(payload.get("report"), dict):
        return OpeningReport.from_dict(payload["report"])
    return None


def load_runs(store: Store, start: str, end: str) -> dict[str, dict[str, Any]]:
    """The frozen runs (report, cutoff, ...) dated in ``[start, end)``, by trade date."""
    rows = store.conn.execute("SELECT key, value FROM kv WHERE key >= ? AND key < ? ORDER BY key", (run_key(start), run_key(end))).fetchall()
    runs: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = jsonutil.loads(row["value"])
        if isinstance(payload, dict) and isinstance(payload.get("report"), dict):
            runs[row["key"][len(RUN_PREFIX):]] = payload
    return runs


def live_record(store: Store, day: date) -> dict[str, Any]:
    """What the earlier reports came to: the model over every name it scored, and the lists. A report that was built after the report
    deadline is left out (its range no longer was an opening range)."""
    runs = {d: p for d, p in load_runs(store, (day - timedelta(days=RECORD_DAYS)).isoformat(), day.isoformat()).items()
            if not p["report"].get("late")}
    ideas = {d: [i for kind in ("picks", "movers") for i in p["report"].get(kind) or [] if isinstance(i, dict)] for d, p in runs.items()}
    outcomes: dict[str, dict[str, Any]] = {}
    for d in runs:
        stored = store.kv_get_json(outcome_key(d))
        if isinstance(stored, dict):
            outcomes[d] = stored
    return build_record(outcomes, ideas)


def morning_candidates(store: Store, day: date) -> tuple[list[Candidate], str]:
    """The candidates of the morning report to read again and which snapshot they came from: the final one (09:12) when it was
    frozen, else the silent 08:45 one; ``([], "")`` when neither was. A final scan that found no setup means no picks, not a reason
    to fall back to the superseded preliminary names."""
    for stage in ("final", "pre"):
        snapshot = store.latest_snapshot(day.isoformat(), stage)
        if snapshot is None:
            continue
        payload = snapshot.get("payload")
        report = payload.get("report") if isinstance(payload, dict) else None
        if isinstance(report, dict):
            try:
                return list(ReportContext.from_dict(report).candidates), stage
            except (TypeError, ValueError, KeyError):
                log.warning("the %s snapshot of %s has an unreadable report", stage, day)
    return [], ""


# ------------------------------------------------------------------------------------------------ scan
def _fail(ctx: JobContext, reason: str) -> JobResult:
    ctx.store.add_alert("error", "opening_scan_failed", reason, dedupe_key=f"opening-failed:{ctx.run_key}", dedupe_minutes=30)
    if ctx.last_attempt and not ctx.dry_run:
        ctx.notifier().send_failure_notice(ctx.run_key, "opening", reason)
    return JobResult("failed", reason)


def deliver(ctx: JobContext, report: OpeningReport) -> JobResult:
    """Send (or re-send, idempotently) a frozen report and translate the outcome into a job result. A report that goes out after the
    report deadline (only a forced send can) says so in the copy that is sent; the frozen report itself is never edited."""
    if ctx.dry_run:
        return JobResult("success", "dry run: report not sent")
    if ctx.force and not report.late and ctx.clock.now() > at_ist(ctx.run_date, ctx.settings.opening_report_deadline_time):
        report = replace(report, late=True)
    outcome = ctx.notifier().send_opening(report, force=ctx.force)
    if outcome.ok:
        return JobResult("success", f"report {outcome.status}", {"message_ids": outcome.message_ids})
    if outcome.status == "withheld":
        return JobResult("success", f"opening read frozen; report withheld ({outcome.reason})")
    if ctx.last_attempt:
        ctx.notifier().send_failure_notice(ctx.run_key, "opening", "report could not be delivered", outcome.reason)
    return JobResult("failed", f"report delivery failed: {outcome.reason}")


def run_opening_scan(ctx: JobContext) -> JobResult:
    s = ctx.settings
    if not s.opening_enabled:
        return JobResult("skipped", "OPENING_ENABLED=false")
    ctx.ensure_calendar()
    day = ctx.run_date
    if not ctx.calendar.is_trading_day(day):
        return JobResult("skipped", f"exchange closed on {day} ({ctx.calendar.holiday_name(day)})")
    session = ctx.calendar.session(day)
    if session.kind != "regular" or session.open is None or session.open != at_ist(day, OPEN_TIME):
        return JobResult("skipped", f"{day} is a {session.kind} session: the opening read is built for the regular {OPEN_TIME[:5]} open")
    kept = None if ctx.dry_run else frozen_report(ctx.store, day)
    if kept is not None:  # frozen once: a retry, a late run or --force sends what was frozen, never a second scan
        return deliver(ctx, kept)
    try:
        return _scan(ctx, session.open)
    except Exception as exc:  # noqa: BLE001 - one bad scan must be reported, not crash the daemon
        log.exception("opening scan failed")
        return _fail(ctx, f"{type(exc).__name__}: {redact(str(exc))[:200]}")


def _nifty_range20(panel: Any) -> float | None:
    """Nifty's mean daily range (% of the open) over the 20 sessions before today."""
    nifty = panel.nifty
    if not len(nifty):
        return None
    daily = ((nifty["high"] - nifty["low"]) / nifty["open"].where(nifty["open"] > 0) * 100.0).iloc[-21:-1]
    return float(daily.mean()) if int(daily.count()) >= 15 else None


def _scan(ctx: JobContext, opened: datetime) -> JobResult:
    s, store, clock, day = ctx.settings, ctx.store, ctx.clock, ctx.run_date
    universe = store.list_instruments(kind="equity", in_universe=True, status="active")
    symbols = [r["symbol"] for r in universe]
    if not symbols:
        return _fail(ctx, "no instruments in the universe (run bootstrap)")
    instruments = {r["symbol"]: {"name": r["name"], "sector": r["sector"], "is_fno": bool(r["is_fno"])} for r in universe}
    deadline_at = at_ist(day, s.opening_report_deadline_time)
    why: list[str] = []
    live = fetch_live(ctx, symbols, fetch_bound(ctx, deadline_at), why)
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
    elapsed = (cutoff - opened).total_seconds() / 60.0
    notes: list[str] = list(live.notes)
    if elapsed < MIN_ELAPSED_MIN or elapsed > MAX_ELAPSED_MIN:
        if not ctx.dry_run:
            if elapsed < MIN_ELAPSED_MIN:
                return _fail(ctx, f"the opening range is not formed yet: the live quotes are stamped {elapsed:.0f} minutes after the open "
                                  "(three five-minute candles are needed)")
            return _fail(ctx, f"too late for an opening read: the live quotes are stamped {elapsed:.0f} minutes after the open "
                              f"(at most {MAX_ELAPSED_MIN:.0f})")
        notes.append(f"PREVIEW: the live quotes are {elapsed:.0f} minutes after the open, so the figures cover that whole stretch and "
                     "not the opening range: run it at 09:30 for a real read")
    elif elapsed > 15.5:
        notes.append(f"the live range covers the first {elapsed:.0f} minutes and is scaled back to fifteen (an average growth factor of "
                     f"{range_growth(elapsed):.2f}): single names can differ")
    if INDEX_NAMES["nifty"] not in live.indices:
        notes.append("NSE index quotes unavailable: Nifty and VIX not shown")

    panel, last_stored = stored_panel(ctx, HISTORY_DAYS)
    stored_prev = panel.c.iloc[-2]
    on_panel = [sym for sym in panel.symbols if sym in live.rows]
    excluded = basis_exclusions(on_panel, live.rows, stored_prev, last_stored)
    hist = history_stats(panel.o.iloc[:-1], panel.h.iloc[:-1], panel.low.iloc[:-1], panel.c.iloc[:-1], panel.v.iloc[:-1])
    table, rejected = opening_table({sym: live.rows[sym] for sym in on_panel if sym not in excluded}, hist, elapsed)
    if table.empty:
        return _fail(ctx, "no stock has both a usable live row and stored history")
    scored = score(table)
    cfg = OpeningConfig.from_settings(s)
    ok = eligibility(scored, cfg)

    candidates, stage = morning_candidates(store, day)

    def news(names: list[str]) -> dict[str, Any]:
        try:
            intel = ctx.intel()
            found = intel.catalysts(names, now)
            if intel.coverage_note():
                notes.append(f"news coverage reduced: {intel.coverage_note()}")
            return found
        except Exception as exc:  # noqa: BLE001 - news is an extra: the read goes out without it
            log.warning("news for the opening movers unavailable: %s", redact(str(exc)))
            notes.append("news unavailable for the new movers")
            return {}

    picks, movers, missing = build_ideas(scored, cfg, instruments=instruments, candidates=candidates, stage=stage, news=news)
    if not stage:
        notes.append("no 09:12 report was stored for today: only the new movers are shown")
    elif not candidates:
        notes.append(f"the {stage} scan listed no candidates today")
    if missing:
        notes.append("no usable live quote for the morning pick(s): " + ", ".join(missing[:6]))
    if not movers:
        notes.append(f"no new name opened at {cfg.min_x:.2f}x or more of its normal daily range: nothing is forced")
    left_out = {**excluded, **rejected}
    if left_out:
        notes.append(f"{len(left_out)} name(s) left out: " + "; ".join(sorted(set(left_out.values()))))
    if live.stale or live.incomplete:
        notes.append(f"{live.stale} stale and {live.incomplete} incomplete live row(s) were not used")

    market = market_pulse(scored[ok] if len(scored) else scored, live.indices, instruments=instruments, nifty_range20=_nifty_range20(panel),
                          growth=range_growth(elapsed))
    report = OpeningReport(
        trade_date=day.isoformat(), snapshot_ist=ist_label(cutoff), elapsed_min=round(elapsed, 1),
        source=live.source_label(len(symbols)),
        picks=picks, movers=movers, pre_stage=stage, market=market, record=live_record(store, day), model=dict(MODEL_INFO),
        notes=notes[:6], mode=s.mode, universe_size=len(symbols), n_eligible=int(ok.sum()) if len(ok) else 0, late=now > deadline_at,
    )
    if ctx.dry_run:
        return JobResult("success", "dry run: nothing frozen or sent", {"report": report.to_dict()})
    store.kv_set_json(state_key(day), state_rows(scored, ok))
    store.kv_set_json(run_key(day), {"report": report.to_dict(), "frozen_at": iso(now), "code": release_id(), "cutoff": iso(cutoff)})
    log.info("opening scan frozen: %d picks, %d movers, %d eligible, %.0f min after the open", len(picks), len(movers), report.n_eligible, elapsed)
    outcome = deliver(ctx, report)
    outcome.metrics.update(picks=len(picks), movers=len(movers), eligible=report.n_eligible, live_rows=len(live.rows), late=report.late)
    return outcome


# ------------------------------------------------------------------------------------------------ evaluation
def _day_bars(store: Store, trade_day: str) -> dict[str, dict[str, Any]]:
    """The daily bar of every equity on ``trade_day``: open / high / low / close and the source rank (2 = the official file)."""
    rows = store.conn.execute(
        "SELECT i.symbol AS symbol, b.open AS open, b.high AS high, b.low AS low, b.close AS close, b.source_rank AS source_rank "
        "FROM bars_daily b JOIN instruments i USING (instrument_id) WHERE i.kind = 'equity' AND b.trade_date = ?", (trade_day,)).fetchall()
    return {r["symbol"]: dict(r) for r in rows}


def _from_hhmm(payload: dict[str, Any]) -> str:
    """The five-minute bar the scan's cutoff falls in (IST), as HH:MM: the first bar the reference levels can trigger in."""
    stamp = parse_ts(payload.get("cutoff"))
    if stamp is None:
        return "09:30"
    local = stamp.astimezone(IST)
    return f"{local.hour:02d}:{local.minute - local.minute % 5:02d}"


def _broker_failed(store: Store, exc: Exception, what: str) -> None:
    """Record a failure of the broker's candles and say it. The broker is an extra (Yahoo is asked for whatever it does not give), so an
    error that is not a ``DataUnavailable`` costs only this name or this evening's broker use, never the judging: it is logged with its
    trace as the bug it is."""
    if isinstance(exc, DataUnavailable):
        reason = redact(str(exc))
        log.info("%s: %s", what, reason[:100])
    else:
        reason = f"unexpected {type(exc).__name__}: {redact(str(exc))[:120]}"
        log.warning("%s: %s", what, reason, exc_info=exc)
    store.record_source_result("kotak_history", False, error=reason)


def _broker_bars(ctx: JobContext, symbols: list[str], trade_day: str, until: float) -> dict[str, list[dict[str, Any]]]:
    """The broker's five-minute candles of ``trade_day``, one request a name (exchange-grade: built from the exchange's own trades), cut to the
    72 bars of ``SESSION_HHMM``. A day that lacks any of them (unfinished, partly served, a bar dropped by the parser for the vendor's
    negative-volume artefact) is left out whole, so that a missing stretch is never judged as a quiet one; whatever is missing here is
    asked of Yahoo."""
    store, day = ctx.store, date.fromisoformat(trade_day)
    try:
        api = ctx.kotak()
        tokens = kotak_tokens(store, api, symbols, ctx.run_date, deadline=until)
    except Exception as exc:  # noqa: BLE001 - see _broker_failed
        _broker_failed(store, exc, "broker candles unavailable")
        return {}
    found: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        if symbol not in tokens or ctx.clock.monotonic() >= until:
            continue
        try:
            got = {b["hhmm"]: b for b in api.candles(tokens[symbol], "5min", day, day, deadline=until) if b.get("trade_date") == trade_day}
        except Exception as exc:  # noqa: BLE001 - see _broker_failed
            _broker_failed(store, exc, f"no broker candles for {symbol}")
            continue
        missing = sum(1 for hhmm in SESSION_HHMM if hhmm not in got)
        if missing:
            log.info("broker candles of %s on %s lack %d of the %d bars (09:15-15:10): Yahoo is asked", symbol, trade_day, missing,
                     len(SESSION_HHMM))
            continue
        bars = [got[hhmm] for hhmm in SESSION_HHMM]
        store.record_source_result("kotak_history", True, data_ts=bars[-1]["ts"])
        found[symbol] = bars
    return found


def _intraday(ctx: JobContext, wanted: dict[str, str], trade_day: str, until: float) -> dict[str, list[dict[str, Any]]]:
    """The five-minute bars of ``trade_day`` for the names in ``wanted`` (symbol -> Yahoo symbol), best effort until the ``until`` instant
    (``clock.monotonic()``; the evening shares one with every report it judges): the broker's candles first when ``KOTAK_CONSUMER_KEY`` is
    set, Yahoo's for every name the broker did not give."""
    out = _broker_bars(ctx, list(wanted), trade_day, until) if ctx.settings.kotak_configured else {}
    yahoo = None
    for symbol, yahoo_symbol in wanted.items():
        if symbol in out:
            continue
        if ctx.clock.monotonic() >= until:
            log.info("five-minute bars: budget used up after %d of %d names", len(out), len(wanted))
            break
        yahoo = yahoo or ctx.yahoo()
        try:
            bars = yahoo.fetch_intraday(yahoo_symbol)
        except Exception as exc:  # noqa: BLE001 - five-minute bars are an extra: a name without them is judged on the daily bar only
            log.info("no five-minute bars for %s: %s", symbol, redact(str(exc))[:80])
            continue
        out[symbol] = [b for b in bars if b.get("trade_date") == trade_day]
    return out


INTRADAY_KEYS = ("plan", "ahead_pct", "ahead_bars", "ahead_in_band")


def _judge_ideas(ctx: JobContext, trade_day: str, payload: dict[str, Any], bars: dict[str, dict[str, Any]],
                 stored: dict[str, Any], yahoo_symbols: dict[str, str], until: float) -> dict[str, dict[str, Any]]:
    """The outcome of every listed name of one run: from the day's bar, plus what the five-minute bars say about the range ahead and
    the reference levels (fetched until the ``until`` instant, which every report of the night shares). Intraday figures judged on an
    earlier night are kept (the feed reaches back only a few sessions)."""
    ideas = [i for kind in ("picks", "movers") for i in payload["report"].get(kind) or [] if isinstance(i, dict)]
    earlier = stored.get("ideas") or {}
    found: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for idea in ideas:
        key = f"{idea.get('kind')}:{idea.get('symbol')}"
        outcome = day_outcome(idea, bars.get(str(idea.get("symbol"))))
        if outcome is None:
            continue
        found[key] = outcome
        prior = earlier.get(key)
        if isinstance(prior, dict) and any(k in prior for k in ("plan", "ahead_pct")):
            outcome.update({k: prior[k] for k in INTRADAY_KEYS if k in prior})
        elif outcome.get("status") == "judged":
            pending.append(idea)
    reach = EVALUATE_DAYS if ctx.settings.kotak_configured else INTRADAY_REACH_DAYS  # the broker's archive reaches back years
    if not pending or (ctx.run_date - date.fromisoformat(trade_day)).days > reach:
        return found
    five = _intraday(ctx, {str(i["symbol"]): yahoo_symbols.get(str(i["symbol"])) or f"{i['symbol']}.NS" for i in pending}, trade_day, until)
    start = _from_hhmm(payload)
    for idea in pending:
        series = five.get(str(idea.get("symbol")))
        if not series:
            continue
        outcome = found[f"{idea.get('kind')}:{idea.get('symbol')}"]
        outcome.update(ahead_outcome(idea, series, start) or {})
        plan = plan_outcome(idea, series, start)
        if plan is not None:
            outcome["plan"] = plan
    return found


def evaluate_opening(ctx: JobContext) -> JobResult:
    """Judge the frozen opening reports of the last ``EVALUATE_DAYS`` days whose day bar is stored, and attach the outcomes. Every night
    the whole window is judged again: an outcome judged on a provisional bar is replaced as soon as the official file is in."""
    store, day = ctx.store, ctx.run_date
    runs = load_runs(store, (day - timedelta(days=EVALUATE_DAYS)).isoformat(), (day + timedelta(days=1)).isoformat())
    if not runs:
        return JobResult("skipped", "no frozen opening reports to evaluate")
    yahoo_symbols = {r["symbol"]: r["yahoo_symbol"] for r in store.list_instruments(kind="equity") if r["yahoo_symbol"]}
    until = ctx.budget(INTRADAY_BUDGET_SEC)  # one budget for all the reports of the night: a backlog must not multiply it
    judged = waiting = 0
    for trade_day, payload in sorted(runs.items()):
        bars = _day_bars(store, trade_day)
        if not bars:
            waiting += 1
            continue
        stored = store.kv_get_json(outcome_key(trade_day))
        stored = stored if isinstance(stored, dict) else {}
        official = all((b.get("source_rank") or 0) >= 2 for b in bars.values())
        outcome: dict[str, Any] = {"evaluated_at": iso(ctx.clock.now()), "official": official,
                                   "ideas": _judge_ideas(ctx, trade_day, payload, bars, stored, yahoo_symbols, until)}
        state = store.kv_get_json(state_key(trade_day))
        if isinstance(state, dict) and state.get("cols") and not payload["report"].get("late"):
            rows = [dict(zip(state["cols"], row, strict=False)) for row in state.get("rows") or []]
            universe = universe_outcome(rows, bars)
            if universe is not None:
                outcome["universe"] = universe
        store.kv_set_json(outcome_key(trade_day), outcome)
        judged += 1 if official else 0
        waiting += 0 if official else 1
    record = live_record(store, day + timedelta(days=1))
    model = record.get("model") or {}
    tail = ""
    if model.get("days"):
        tail = (f"; model record: {model['days']} session(s), {model['n']} names, stated {model.get('mean_p3', 0):.0%} vs "
                f"happened {model.get('hit3', 0):.0%} (>3% range)")
    return JobResult("success", f"{judged} run(s) settled, {waiting} waiting for the official bars{tail}",
                     {"judged": judged, "waiting": waiting})
