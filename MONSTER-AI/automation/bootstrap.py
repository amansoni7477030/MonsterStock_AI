"""First-run setup (blueprint section 17): everything the server needs before the daily timetable can start, resumable and
self-healing. The scheduler re-launches it until ``bootstrap.state`` is ``complete``; every step records its own status, so
a crash, reboot or out-of-memory kill resumes where it stopped instead of starting over."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from app.clock import IST, iso
from app.errors import DataUnavailable
from app.logging_setup import redact
from automation import jobs, scan
from automation.calendar import infer_holidays_from_bars, refresh_calendar, seed_calendar, trading_days_before
from automation.context import JobContext
from ingestion.pipeline import backfill_fno, backfill_history, ensure_series_instruments, ingest_bhavcopy
from ingestion.universe import build_universe

log = logging.getLogger(__name__)

STATE_KEY = "bootstrap.state"
STATE_RUNNING, STATE_DONE, STATE_FAILED = "running", "done", "failed"  # documented in deployment/systemd/*bootstrap*
STEPS_KEY = "bootstrap.steps"
OFFICIAL_RECENT_SESSIONS = 12
MAX_CRASHES_PER_STEP = 2
REQUIRED_STEPS = ("init", "universe", "history")
ProgressFn = Callable[[str, str], None]


@dataclass
class BootstrapOutcome:
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    complete: bool = False

    def summary(self) -> str:
        done = sum(1 for s in self.steps.values() if s.get("status") == "done")
        problems = [f"{k}: {v.get('detail', v.get('status'))}" for k, v in self.steps.items()
                    if v.get("status") in {"failed", "skipped"}]
        text = f"{done}/{len(self.steps)} steps done" + ("; complete" if self.complete else "; incomplete")
        return text + (" | " + "; ".join(problems[:4]) if problems else "")


class _Steps:
    def __init__(self, ctx: JobContext, resume: bool, progress: ProgressFn | None) -> None:
        self.ctx = ctx
        self.progress = progress or (lambda step, message: log.info("bootstrap %s: %s", step, message))
        stored = ctx.store.kv_get_json(STEPS_KEY, {}) if resume else {}
        self.state: dict[str, dict[str, Any]] = stored or {}

    def save(self) -> None:
        self.ctx.store.kv_set_json(STEPS_KEY, self.state)

    def run(self, name: str, fn: Callable[[], tuple[str, str]], *, required: bool = False) -> None:
        prior = self.state.get(name, {})
        if prior.get("status") == "done":
            self.progress(name, "already done")
            return
        crashes = int(prior.get("crashes", 0)) + (1 if prior.get("status") == "running" else 0)
        if crashes >= MAX_CRASHES_PER_STEP and not required:
            self.state[name] = {"status": "skipped", "crashes": crashes,
                                "detail": "earlier attempts died mid-step (out of memory?); skipped"}
            self.save()
            self.progress(name, self.state[name]["detail"])
            return
        self.state[name] = {"status": "running", "crashes": crashes, "started": iso(self.ctx.clock.now())}
        self.save()
        self.progress(name, "running")
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a failed step must not stop the rest of the bootstrap
            log.exception("bootstrap step %s failed", name)
            status, detail = "failed", f"{type(exc).__name__}: {redact(str(exc))[:200]}"
        self.state[name] = {"status": status, "detail": detail, "crashes": crashes, "finished": iso(self.ctx.clock.now())}
        self.save()
        self.progress(name, f"{status}: {detail}")


def run_bootstrap(ctx: JobContext, *, resume: bool = True, skip_train: bool = False, data_only: bool = False,
                  progress: ProgressFn | None = None) -> BootstrapOutcome:
    """``data_only`` collects the market data and nothing else (no training, smoke scan or Telegram test): used by the
    external training worker, which only needs history to train on."""
    s, store = ctx.settings, ctx.store
    steps = _Steps(ctx, resume, progress)
    store.kv_set(STATE_KEY, STATE_RUNNING)

    def init() -> tuple[str, str]:
        written = seed_calendar(store, s)
        ctx.calendar.reload()
        ensure_series_instruments(store)
        return "done", f"{written} calendar rows seeded"

    def universe() -> tuple[str, str]:
        result = build_universe(store, s, ctx.nse_archive() if s.nse_enabled else None, ctx.clock.now().date())
        graph = ctx.intel().reseed_graph()  # sector memberships follow the instrument master
        note = f"{len(result.symbols)} symbols from {result.source}"
        if graph:
            note += f"; event graph: {graph.get('nodes_created', 0)} nodes, {graph.get('edges_created', 0)} edges seeded"
        return "done", note + (f" ({'; '.join(result.notes)})" if result.notes else "")

    def history() -> tuple[str, str]:
        started = time.monotonic()
        failed = ok = 0
        while True:
            report = backfill_history(store, ctx.yahoo(), now=ctx.clock.now(), years=s.history_years,
                                      deadline=ctx.budget(30 * 60), clock=ctx.clock)
            ok, failed = ok + report.ok, failed + report.failed
            steps.progress("history", f"{ok} instruments loaded, {failed} failed")
            if not report.partial or ctx.expired() or time.monotonic() - started > 4 * 3600:
                break
        remaining = _history_remaining(ctx)
        if remaining > max(5, int(0.1 * _history_total(ctx))):
            return "failed", f"{remaining} instruments still without history (provider blocked or rate limited)"
        return "done", f"{ok} instruments loaded, {failed} failed ({remaining} remaining, retried weekly)"

    def calendar_official() -> tuple[str, str]:
        days = refresh_calendar(store, ctx.nse_api()) if s.nse_enabled else 0
        ctx.calendar.reload()
        return "done", f"{days} official holiday rows"

    def calendar_inferred() -> tuple[str, str]:
        inferred = infer_holidays_from_bars(store)
        ctx.calendar.reload()
        return "done", f"{inferred} closed days inferred from missing index bars"

    def official_recent() -> tuple[str, str]:
        if not s.nse_enabled:
            return "skipped", "NSE_ENABLED=false"
        last = ctx.calendar.last_completed_session(ctx.clock.now())
        loaded = 0
        for day in reversed(trading_days_before(ctx.calendar, last, OFFICIAL_RECENT_SESSIONS)):
            try:
                written, _ = ingest_bhavcopy(store, ctx.nse_archive(), day)
            except DataUnavailable as exc:
                return "failed", redact(str(exc))[:160]
            loaded += int(bool(written))
        return "done", f"official bhavcopy for {loaded} of the last {OFFICIAL_RECENT_SESSIONS} sessions"

    def fno() -> tuple[str, str]:
        if not s.nse_enabled:
            return "skipped", "NSE_ENABLED=false"
        end = ctx.calendar.last_completed_session(ctx.clock.now())
        start = max(jobs.FNO_HISTORY_START, end - timedelta(days=int(s.history_years * 365.25)))
        days = ctx.calendar.trading_days(start, end)
        report = backfill_fno(store, ctx.nse_archive(), days, deadline=ctx.budget(60 * 60), clock=ctx.clock,
                              on_day=lambda d, n: steps.progress("fno_history", f"{d} ({n} rows)"))
        if report.partial:
            return "failed", f"{report.ok} days loaded; deadline reached (will resume)"
        return "done", f"{report.ok} days loaded, {report.skipped} already present or unpublished"

    def board_meetings() -> tuple[str, str]:
        if not s.nse_enabled:
            return "skipped", "NSE_ENABLED=false"
        today = ctx.clock.now().date()
        symbols = set(jobs.universe_symbols(ctx))
        total = 0
        start = today - timedelta(days=3 * 365)
        cursor = start
        while cursor < today + timedelta(days=jobs.BOARD_MEETING_LOOKAHEAD_DAYS) and not ctx.expired():
            end = min(cursor + timedelta(days=90), today + timedelta(days=jobs.BOARD_MEETING_LOOKAHEAD_DAYS))
            total += ctx.news().collect_board_meetings(ctx.nse_api(), cursor, end, symbols)
            cursor = end + timedelta(days=1)
        return "done", f"{total} board-meeting rows (results-calendar features need them)"

    def news() -> tuple[str, str]:
        result = jobs.collect_news(ctx)
        return ("done" if result.status == "success" else "failed"), result.message

    def train() -> tuple[str, str]:
        if skip_train:
            return "skipped", "--skip-train"
        # a step that already died once (out of memory?) is retried with a leaner profile
        ctx.options = {**ctx.options, "lean": str(steps.state.get("train", {}).get("crashes", 0))}
        result = jobs.train(ctx) if s.training_mode != "external" else jobs.model_sync(ctx)
        status = {"success": "done", "skipped": "skipped"}.get(result.status, "failed")
        return status, result.message

    def smoke_scan() -> tuple[str, str]:
        # a dry run of the next regular morning scan (today when the exchange is open, otherwise the next trading day)
        today = ctx.clock.now().astimezone(IST).date()
        target = today if ctx.calendar.is_trading_day(today) else ctx.calendar.next_trading_day(today)
        original_key = ctx.run_key
        ctx.run_key, ctx.dry_run = target.isoformat(), True
        try:
            result = scan.run_scan(ctx, "pre")
        finally:
            ctx.run_key, ctx.dry_run = original_key, False
        if result.status == "success":
            report = result.metrics.get("report") or {}
            picks = ", ".join(c["symbol"] for c in report.get("candidates", [])[:5]) or "no candidates"
            return "done", f"end-to-end dry run OK ({report.get('model_status', '?')} model); top: {picks}"
        return "failed", result.message

    def telegram() -> tuple[str, str]:
        if not s.telegram_configured:
            return "skipped", "Telegram not configured (run: python -m app.cli telegram-setup)"
        outcome = ctx.notifier().send_test()
        return ("done" if outcome.ok else "failed"), outcome.reason or "test message sent"

    steps.run("init", init, required=True)
    steps.run("universe", universe, required=True)
    steps.run("calendar_official", calendar_official)
    steps.run("history", history, required=True)
    steps.run("calendar_inferred", calendar_inferred)
    steps.run("official_recent", official_recent)
    steps.run("fno_history", fno)
    steps.run("board_meetings", board_meetings)
    steps.run("news", news)
    if not data_only:
        steps.run("train", train)
        steps.run("smoke_scan", smoke_scan)
        steps.run("telegram", telegram)

    complete = all(steps.state.get(name, {}).get("status") == "done" for name in REQUIRED_STEPS)
    store.kv_set(STATE_KEY, STATE_DONE if complete else STATE_FAILED)
    if complete:
        store.kv_set("bootstrap.completed_at", iso(ctx.clock.now()) or "")
        store.add_alert("info", "bootstrap_complete", "Bootstrap finished; the daily timetable is now active.",
                        dedupe_key="bootstrap-complete", dedupe_minutes=10 ** 6)
    return BootstrapOutcome(steps.state, complete)


def _history_total(ctx: JobContext) -> int:
    return len(ctx.store.list_instruments(kind="equity", in_universe=True, status="active")) + len(
        [r for r in ctx.store.list_instruments(status="active") if r["kind"] != "equity"])


def _history_remaining(ctx: JobContext) -> int:
    store = ctx.store
    rows = [r for r in store.list_instruments(status="active") if r["kind"] != "equity"]
    rows += store.list_instruments(kind="equity", in_universe=True, status="active")
    return sum(1 for r in rows if (store.get_backfill(f"yahoo_history:{r['symbol']}") or {}).get("status") != "done")
