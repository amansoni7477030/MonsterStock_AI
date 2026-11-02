"""Scheduled job implementations (blueprint sections 5, 15-19).

Every job is idempotent and safe to re-run: it reads the database, does bounded work, and returns a ``JobResult``.
Failures are expected (providers go down); a job reports them precisely instead of raising, and the scheduler retries
inside the job's window. Jobs never place orders, and none of them is allowed to change frozen predictions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from app.clock import at_ist
from app.errors import DataUnavailable
from app.logging_setup import redact
from app.release import release_id
from app.resources import check_resources, disk_info, emergency_cleanup, memory_info, release_memory
from automation import ideas, scan
from automation.calendar import infer_holidays_from_bars, refresh_calendar, trading_days_before
from automation.context import JobContext, JobResult
from automation.health import CALENDAR_REFRESH_KEY
from ingestion.pipeline import (
    backfill_history,
    collect_gift,
    collect_live_context,
    collect_preopen,
    ensure_series_instruments,
    ingest_bhavcopy,
    ingest_fno_day,
    refresh_daily_bars,
)
from ingestion.series import INDIA_MARKET

log = logging.getLogger(__name__)

MIN_OFFICIAL_COVERAGE = 0.90
FNO_HISTORY_START = date(2024, 7, 8)  # first UDiFF-format F&O bhavcopy; older archives use a different layout
BOARD_MEETING_LOOKAHEAD_DAYS = 45
FILINGS_EXTRA_DAYS = 1  # collected from this many days before the previous trading session
PREOPEN_COLLECT_UNTIL = "09:08:15"
PREOPEN_COLLECT_EVERY_SEC = 45


# ---------------------------------------------------------------------------------------------------- helpers
def universe_symbols(ctx: JobContext) -> list[str]:
    return [r["symbol"] for r in ctx.store.list_instruments(kind="equity", in_universe=True, status="active")]


def _guarded(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run an optional step; failure is logged (redacted) and never aborts the job."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s failed: %s", label, redact(str(exc)))
        return None


def _non_trading(ctx: JobContext) -> JobResult | None:
    ctx.ensure_calendar()
    day = ctx.run_date
    if ctx.calendar.is_trading_day(day) or ctx.force:
        return None
    return JobResult("skipped", f"exchange closed on {day} ({ctx.calendar.holiday_name(day)})")


def session_coverage(ctx: JobContext, session: date) -> tuple[float, list[str]]:
    """Share of the universe with the ``session`` bar, and the symbols still missing it."""
    universe = universe_symbols(ctx)
    have = ctx.store.symbols_with_bar(session.isoformat(), universe_only=True)
    missing = [s for s in universe if s not in have]
    return (1 - len(missing) / len(universe)) if universe else 0.0, missing


def provisional_symbols(ctx: JobContext, session: date) -> list[str]:
    """Universe names whose bar of ``session`` is still the vendor's provisional one (the official file has not replaced it)."""
    rows = ctx.store.conn.execute(
        "SELECT i.symbol AS symbol FROM bars_daily b JOIN instruments i USING (instrument_id) WHERE b.trade_date = ? "
        "AND i.kind = 'equity' AND i.status = 'active' AND i.in_universe = 1 AND b.source_rank < 2 ORDER BY i.symbol",
        (session.isoformat(),)).fetchall()
    return [r["symbol"] for r in rows]


def ingest_official(ctx: JobContext, session: date) -> dict[str, Any]:
    """The official end-of-day file of ``session`` (one request); a provider quirk degrades to vendor data, never a crash."""
    try:
        written, _matched = ingest_bhavcopy(ctx.store, ctx.nse_archive(), session)
    except Exception as exc:  # noqa: BLE001
        log.warning("official bhavcopy for %s unavailable: %s", session, redact(str(exc)))
        return {"official_rows": 0, "official_error": redact(str(exc))[:120]}
    return {"official_rows": written}


def repair_basis_breaks(ctx: JobContext, session: date) -> dict[str, Any]:
    """A split or bonus shows in the official file - the exchange's previous close no longer matches the close we stored - before
    the vendor history reflects it. Ask the vendor again for exactly those stocks, every session, until it reports the split
    (which re-adjusts the stored history and records the event); after ``GIVE_UP_DAYS`` the operator is told. See
    ``ingestion.basis``."""
    from ingestion import basis

    store = ctx.store
    found = basis.official_breaks(store, session)
    basis.mark_pending(store, found)
    waiting = basis.pending(store)
    if not waiting:
        return {}
    refresh_daily_bars(store, ctx.yahoo(), now=ctx.clock.now(), years=ctx.settings.history_years, symbols=set(waiting),
                       include_series=False, probe=False, deadline=ctx.budget(180), clock=ctx.clock)
    repaired, gave_up = [], []
    for symbol, since in waiting.items():
        if basis.is_resolved(store, symbol, since):
            basis.clear(store, symbol)
            repaired.append(symbol)
            log.info("%s: the vendor reported the split; history re-based (break seen on %s)", symbol, since)
        elif basis.overdue(since, session):
            basis.clear(store, symbol)
            gave_up.append(symbol)
    if gave_up:
        log.error("price basis breaks the vendor never confirmed: %s", ", ".join(gave_up))
        store.add_alert("warning", "price_basis_unresolved",
                        f"the exchange's previous close differs from the stored one for {', '.join(gave_up)} but the vendor reports "
                        "no split: their history may be on the wrong price basis (check the corporate action)",
                        dedupe_key="basis:" + ",".join(sorted(gave_up)), dedupe_minutes=24 * 60)
    return {"basis_breaks": len(found), "basis_repaired": len(repaired), "basis_given_up": len(gave_up),
            "basis_waiting": len(waiting) - len(repaired) - len(gave_up)}


def ensure_session_bars(ctx: JobContext, session: date, *, budget_sec: float = 600.0) -> dict[str, Any]:
    """Make sure the universe has the bar of ``session``: the official file first (one request) - also to replace the vendor's
    provisional bars once it is published - and vendor data for what is still missing."""
    store, s = ctx.store, ctx.settings
    info: dict[str, Any] = {"official_rows": 0, "vendor_ok": 0, "vendor_failed": 0}
    coverage, missing = session_coverage(ctx, session)
    if s.nse_enabled and (missing or provisional_symbols(ctx, session)):
        info.update(ingest_official(ctx, session))
        coverage, missing = session_coverage(ctx, session)
    if missing:
        report = refresh_daily_bars(store, ctx.yahoo(), now=ctx.clock.now(), years=s.history_years, symbols=missing,
                                    include_series=False, deadline=ctx.budget(budget_sec), clock=ctx.clock)
        info["vendor_ok"], info["vendor_failed"] = report.ok, report.failed
        coverage, missing = session_coverage(ctx, session)
    info["coverage"] = round(coverage, 4)
    info["missing"] = len(missing)
    if s.nse_enabled:
        info.update(_guarded("price basis", repair_basis_breaks, ctx, session) or {})
    return info


def refresh_india_context(ctx: JobContext, session: date) -> dict[str, Any]:
    """Index and VIX bars plus the F&O end-of-day file for ``session`` (best effort)."""
    store, s = ctx.store, ctx.settings
    out: dict[str, Any] = {}
    report = refresh_daily_bars(store, ctx.yahoo(), now=ctx.clock.now(), years=s.history_years,
                                symbols={INDIA_MARKET, "^NSEBANK", "^INDIAVIX"}, include_equities=False,
                                deadline=ctx.budget(120), clock=ctx.clock)
    out["index_ok"] = report.ok
    have = store.latest_fno_date()
    if s.nse_enabled and (have is None or have < session.isoformat()):
        out["fno_rows"] = _guarded("F&O bhavcopy", ingest_fno_day, store, ctx.nse_archive(), session) or 0
    return out


# ----------------------------------------------------------------------------------------------------- jobs
def collect_global(ctx: JobContext) -> JobResult:
    """Overnight global closes (US/Europe/Asia, FX, commodities, rates, VIX), live Asia/futures context, GIFT Nifty."""
    s, store = ctx.settings, ctx.store
    ctx.ensure_calendar()
    ensure_series_instruments(store)
    deadline = ctx.budget(20 * 60)
    report = refresh_daily_bars(store, ctx.yahoo(), now=ctx.clock.now(), years=s.history_years, include_equities=False,
                                deadline=deadline, clock=ctx.clock)
    metrics: dict[str, Any] = {"series_ok": report.ok, "series_failed": report.failed, "series_unchanged": report.unchanged,
                               "partial": report.partial}
    metrics["live_context"] = _guarded("live context", collect_live_context, store, ctx.yahoo(), deadline=deadline, clock=ctx.clock)
    if s.nse_enabled:
        _guarded("GIFT Nifty", collect_gift, store, ctx.nse_api())
    ok = report.success_ratio >= 0.8 and not report.partial
    message = f"{report.ok} series ok, {report.failed} failed" + ("; deadline reached" if report.partial else "")
    if report.unchanged:
        message += f"; {report.unchanged} confirmed unchanged by the bulk probe"
    if report.notes:
        message += " (" + "; ".join(report.notes[:3]) + ")"
    return JobResult("success" if ok else "failed", message, metrics)


def collect_news(ctx: JobContext) -> JobResult:
    """RSS + GDELT discovery + NSE filings, then entity/event extraction. Optional enrichment: partial failure is tolerated."""
    from ingestion.news_sources import load_feed_registry

    s = ctx.settings
    ctx.ensure_calendar()
    collector = ctx.news()
    metrics: dict[str, Any] = {}
    deadline = ctx.budget(15 * 60)
    try:
        feeds, queries = load_feed_registry(s.news_feeds_file)
    except DataUnavailable as exc:
        return JobResult("failed", redact(str(exc)))
    rss = collector.collect_rss(feeds, deadline=deadline)
    metrics.update(rss_ok=rss.ok, rss_failed=rss.failed, rss_new=rss.new_items, rss_not_modified=rss.not_modified)
    if s.gdelt_enabled:
        gdelt = collector.collect_gdelt(queries, deadline=deadline)
        metrics.update(gdelt_ok=gdelt.ok, gdelt_new=gdelt.new_items)
    if s.nse_enabled:
        start, end = filings_window(ctx)
        metrics["filings_new"] = collector.collect_filings(ctx.nse_api(), start, end, set(universe_symbols(ctx)))
        metrics["filings_window"] = f"{start}..{end}"
    metrics["processed"] = ctx.intel().process()
    healthy = rss.success_ratio >= 0.5
    message = f"{rss.ok}/{rss.attempted} feeds ok, {rss.new_items} new items" + ai_note(metrics["processed"])
    return JobResult("success" if healthy else "failed", message, metrics)


def ai_note(processed: dict[str, Any]) -> str:
    """``; AI: llm 12 used, 3 cached; web 2 used; jev 40 used, 1 failed; 2 withheld (instruction-like text)`` for the optional AI
    services that did something in this news job (empty when none did), so ``monster-ai jobs`` and the log show each morning what
    they contributed and which items were kept from them."""
    parts = []
    for name in ("llm", "web", "jev"):
        used, cached, failed = (int(processed.get(f"{name}_{key}") or 0) for key in ("used", "cached", "failed"))
        if used or cached or failed:
            parts.append(f"{name} {used} used" + (f", {cached} cached" if cached else "") + (f", {failed} failed" if failed else ""))
    withheld = int(processed.get("ai_withheld") or 0)
    if withheld:
        parts.append(f"{withheld} withheld (instruction-like text)")
    return "; AI: " + "; ".join(parts) if parts else ""


def filings_window(ctx: JobContext) -> tuple[date, date]:
    """Filings are collected from one day before the previous *trading* session up to today. A fixed "today minus two days"
    window misses Friday's after-hours filings (results, orders, fund-raising) on Monday morning and everything filed before a
    long weekend or holiday - exactly the news a pre-open scan needs; the extra day absorbs late dissemination."""
    day = ctx.run_date
    previous = trading_days_before(ctx.calendar, day - timedelta(days=1), 1)
    anchor = previous[0] if previous else day - timedelta(days=4)
    return anchor - timedelta(days=FILINGS_EXTRA_DAYS), day


def collect_india(ctx: JobContext) -> JobResult:
    """Make sure the previous session is complete in the database (official file preferred) and refresh the event calendar."""
    skipped = _non_trading(ctx)
    if skipped:
        return skipped
    session = ctx.calendar.last_completed_session(ctx.clock.now())
    info = ensure_session_bars(ctx, session)
    info.update(refresh_india_context(ctx, session))
    if ctx.settings.nse_enabled:
        day = ctx.run_date
        info["board_meetings"] = _guarded("board meetings", ctx.news().collect_board_meetings, ctx.nse_api(), day,
                                          day + timedelta(days=BOARD_MEETING_LOOKAHEAD_DAYS), set(universe_symbols(ctx)))
    ok = info.get("coverage", 0.0) >= MIN_OFFICIAL_COVERAGE
    return JobResult("success" if ok else "failed", f"session {session}: {info.get('coverage', 0):.0%} of the universe has its bar", info)


def prelim_scan(ctx: JobContext) -> JobResult:
    skipped = _non_trading(ctx)
    if skipped:
        return skipped
    # collect_india retries inside its own window; when the previous session is still incomplete at scan time, repair it once
    # more with a small budget (a complete session costs one query). A failed repair never stops the scan: the quality gate decides.
    session = ctx.calendar.last_completed_session(ctx.clock.now())
    coverage, missing = session_coverage(ctx, session)
    if coverage < MIN_OFFICIAL_COVERAGE:
        log.info("session %s is %.0f%% complete (%d names missing): repairing before the preliminary scan", session,
                 coverage * 100, len(missing))
        _guarded("previous-session repair", ensure_session_bars, ctx, session, budget_sec=120)
    return scan.run_scan(ctx, "pre")


def final_scan(ctx: JobContext) -> JobResult:
    skipped = _non_trading(ctx)
    return skipped or scan.run_scan(ctx, "final")


def closing_scan(ctx: JobContext) -> JobResult:
    """14:30 IST: BTST, swing, bearish and next-day intraday ideas from NSE's live quotes and the stored history."""
    skipped = _non_trading(ctx)
    return skipped or ideas.run_closing_scan(ctx)


def evaluate_ideas(ctx: JobContext) -> JobResult:
    """Evening: attach the outcomes of the earlier trade ideas (kept apart from the frozen reports) for the live record."""
    skipped = _non_trading(ctx)
    return skipped or ideas.evaluate_ideas(ctx)


def refresh_preopen(ctx: JobContext) -> JobResult:
    """Collect the NSE pre-open book during order entry so the final scan always has the freshest equilibrium prices."""
    skipped = _non_trading(ctx)
    if skipped:
        return skipped
    s, day = ctx.settings, ctx.run_date
    if not s.nse_enabled:
        return JobResult("skipped", "NSE_ENABLED=false")
    symbols = universe_symbols(ctx)
    end = at_ist(day, PREOPEN_COLLECT_UNTIL)
    snapshots = 0
    final = 0.0
    while ctx.clock.now() < end:
        snap = collect_preopen(ctx.store, ctx.nse_api(), day.isoformat(), symbols, deadline=scan.monotonic_at(ctx, end))
        snapshots += int(bool(snap and snap.rows))
        final = scan.preopen_coverage(ctx.store, day.isoformat(), symbols, ctx.clock.now())[1]
        if final >= scan.PREOPEN_TARGET_COVERAGE:
            break
        ctx.clock.sleep(min(PREOPEN_COLLECT_EVERY_SEC, max((end - ctx.clock.now()).total_seconds(), 0.0)))
    _guarded("GIFT Nifty", collect_gift, ctx.store, ctx.nse_api(), deadline=scan.monotonic_at(ctx, end + timedelta(seconds=60)))
    if not snapshots:
        return JobResult("failed", "NSE pre-open book unavailable (the final scan will use the preliminary model)")
    return JobResult("success", f"{snapshots} pre-open snapshots, {final:.0%} final prices", {"snapshots": snapshots})


def ingest_eod(ctx: JobContext) -> JobResult:
    """After the close: official bhavcopy (OHLC + delivery), vendor top-up, F&O end-of-day file."""
    skipped = _non_trading(ctx)
    if skipped:
        return skipped
    day = ctx.run_date
    if ctx.clock.now() < ctx.calendar.session(day).close + timedelta(minutes=15):
        return JobResult("failed", "session has not closed yet")
    info = ensure_session_bars(ctx, day)
    info.update(refresh_india_context(ctx, day))
    refresh_daily_bars(ctx.store, ctx.yahoo(), now=ctx.clock.now(), years=ctx.settings.history_years, include_equities=False,
                       deadline=ctx.budget(300), clock=ctx.clock)
    coverage = info.get("coverage", 0.0)
    official = info.get("official_rows", 0)
    if coverage < MIN_OFFICIAL_COVERAGE:
        return JobResult("failed", f"only {coverage:.0%} of the universe has the {day} bar", info)
    note = "official bhavcopy ingested" if official else "official file unavailable; vendor data used (provisional)"
    return JobResult("success", f"{coverage:.0%} coverage; {note}", info)


def evaluate(ctx: JobContext) -> JobResult:
    """Attach realised outcomes to the frozen predictions, score them against baselines and send the evening report."""
    from backtesting.evaluate import collect_outcomes, evaluate_day, persist_metrics

    skipped = _non_trading(ctx)
    if skipped:
        return skipped
    day = ctx.run_date.isoformat()
    if not ctx.store.predictions_for(day, stage=None, baselines=None):
        return JobResult("skipped", f"no predictions were frozen for {day}")
    counts = collect_outcomes(ctx.store, ctx.settings, day)
    summary = evaluate_day(ctx.store, ctx.settings, day)
    if summary is None:
        return JobResult("failed", f"no outcomes available for {day} yet", counts)
    persist_metrics(ctx.store, summary)
    outcome = ctx.notifier().send_evening(summary)
    detail = f"{summary.n} picks scored, hit rate {summary.model_hit_rate if summary.model_hit_rate is not None else 'n/a'}"
    if not outcome.ok and outcome.status != "withheld":
        return JobResult("failed", f"{detail}; evening report not delivered ({outcome.reason})", counts)
    return JobResult("success", detail + (" (provisional)" if summary.provisional else ""), counts)


def backup(ctx: JobContext) -> JobResult:
    from database.backup import backup_database, verify_backup

    s = ctx.settings
    result = backup_database(s.db_path, s.backup_dir, keep_daily=s.backup_keep_daily, keep_weekly=s.backup_keep_weekly,
                             passphrase=s.backup_passphrase, clock=ctx.clock, max_total_mb=s.backup_max_mb)
    if not verify_backup(result.path, s.backup_passphrase):
        return JobResult("failed", f"backup {result.path.name} failed verification")
    info = {"file": result.path.name, "bytes": result.size_bytes, "encrypted": result.encrypted, "pruned": len(result.pruned)}
    if s.backup_passphrase and not result.encrypted:
        # Encryption was requested but gpg failed: the local copy stays (a restore point beats none) but it never leaves this server.
        ctx.store.add_alert("error", "backup_encryption_failed",
                            "BACKUP_PASSPHRASE is set but gpg could not encrypt the backup: the local copy is UNENCRYPTED and was NOT "
                            "uploaded off-box. Check that gnupg is installed (apt install gnupg) and run: monster-ai backup",
                            dedupe_key="backup-encryption", dedupe_minutes=720)
        log.error("backup %s is unencrypted although BACKUP_PASSPHRASE is set; not uploading it off-box", result.path.name)
        info["upload_skipped"] = "unencrypted"
        remote = None
    else:
        remote = _upload_backup(ctx, result.path)
    if remote:
        info["remote"] = remote
    ctx.store.kv_set_json("backup.last", {**info, "at": ctx.store.now()})
    return JobResult("success", f"{result.path.name} ({result.size_bytes / 1048576:.1f} MB)", info)


def _upload_backup(ctx: JobContext, path: Any) -> str | None:
    """Off-box copy to S3-compatible storage when configured; a failed upload is an alert, never a failed backup."""
    s = ctx.settings
    if not s.s3_configured:
        return None
    try:
        from database.object_store import build_object_store
        from database.remote_backup import prune_remote_backups

        store = build_object_store(s)
        key = f"backups/{path.name}"  # the store prepends S3_PREFIX itself
        store.put_file(key, path)
    except Exception as exc:  # noqa: BLE001
        ctx.store.add_alert("warning", "backup_upload_failed", f"off-box backup upload failed: {redact(str(exc))[:160]}",
                            dedupe_key="backup-upload", dedupe_minutes=720)
        return None
    try:
        prune_remote_backups(store, keep_daily=s.backup_keep_daily, keep_weekly=s.backup_keep_weekly)
    except Exception as exc:  # noqa: BLE001 - retention is housekeeping and must not fail an upload that succeeded
        log.warning("off-box backup retention failed: %s", redact(str(exc))[:160])
    return key


def housekeeping(ctx: JobContext) -> JobResult:
    """Retention of every table / folder / cache (``automation.retention``), disk guard, DB maintenance and outcome
    upgrades (provisional -> official). Nothing in the system grows without a bound."""
    from automation.retention import run_retention
    from backtesting.evaluate import collect_outcomes

    s, store = ctx.settings, ctx.store
    actions: dict[str, Any] = {}
    status = check_resources(s.data_dir, warn_disk_mb=s.min_free_disk_mb_warn, crit_disk_mb=s.min_free_disk_mb_crit,
                             min_ram_mb=s.min_available_ram_mb)
    if status.level == "crit":
        actions["emergency_cleanup"] = emergency_cleanup(s.cache_dir, s.tmp_dir, s.log_dir)
    if status.level in {"warn", "crit"}:
        store.add_alert("warning" if status.level == "warn" else "error", "resources", "; ".join(status.messages),
                        dedupe_key=f"resources:{status.level}", dedupe_minutes=360)
    registry = ctx.registry()
    actions["models"] = sum(registry.prune(stage) for stage in ("pre", "final"))
    # the pipeline's retention detaches event links in bulk (fast); the plain delete is only the fallback without it
    actions["news"] = ctx.intel().prune() or _guarded("news retention", ctx.news().prune_raw)
    upgraded = 0
    official = 0
    for day in trading_days_before(ctx.calendar, ctx.run_date, 3):
        # the vendor's provisional bars of the last sessions are replaced by the official file once it is published (it often
        # comes out after the evening job ran); the exchange's previous closes in it also reveal splits to repair
        if s.nse_enabled and provisional_symbols(ctx, day):
            official += (_guarded("official upgrade", ingest_official, ctx, day) or {}).get("official_rows", 0)
        if s.nse_enabled:
            _guarded("price basis", repair_basis_breaks, ctx, day)
    actions["official_rows"] = official
    for day in trading_days_before(ctx.calendar, ctx.run_date, 5):
        if store.predictions_without_validated_outcome(day.isoformat()):
            counts = _guarded("outcome upgrade", collect_outcomes, store, s, day.isoformat()) or {}
            upgraded += counts.get("validated", 0)
    actions["outcomes_validated"] = upgraded
    report = run_retention(store, s, now=ctx.clock.now(), clock=ctx.clock, free_disk_mb=status.free_disk_mb,
                           run_date=ctx.run_date)
    actions.update({"rows_removed": report.total_rows, "files_removed": sum(report.files.values()),
                    "removed": report.removed(), "maintenance": report.maintenance, "data_mb": report.data_mb,
                    "db_mb": report.db_mb, "vacuum": "vacuum_s" in report.maintenance})
    if report.notes:
        store.add_alert("warning", "retention", "; ".join(report.notes)[:300], dedupe_key="retention", dedupe_minutes=720)
    free = status.free_disk_mb or 0
    return JobResult("success", f"disk {free:.0f} MB free; {status.level}; data {report.data_mb:.0f} MB; "
                                f"retention removed {report.total_rows} rows, {sum(report.files.values())} files", actions)


def train(ctx: JobContext) -> JobResult:
    """Walk-forward retraining of both stages with fixed gates. Candidates never replace a healthy champion unless they
    match or beat it on the same gate metrics (see ``models.registry``)."""
    from features.live import build_training_features
    from models.training import TrainConfig, TrainingError, train_stage

    s, store = ctx.settings, ctx.store
    if s.training_mode == "external":
        return JobResult("skipped", "TRAINING_MODE=external: models are trained in CI and synced (see model_sync)")
    if s.training_mode == "disabled":
        return JobResult("skipped", "TRAINING_MODE=disabled: the heuristic baseline stays in use")
    ctx.ensure_calendar()
    free = disk_info(s.data_dir)["free_mb"]
    if free < s.min_free_disk_mb_crit:  # training writes datasets, boosters and bundles: never into the last megabytes
        return JobResult("failed", f"only {free:.0f} MB of disk is free (critical below {s.min_free_disk_mb_crit} MB); "
                                   "training waits for the cleanup to make room")
    end = ctx.calendar.last_completed_session(ctx.clock.now())
    lean = lean_level(ctx)
    years = train_years_for(s, len(universe_symbols(ctx)), lean)
    fit_rows = fit_row_cap(s, lean)
    if lean:
        log.warning("training with the lean profile (level %d): %d years of history, %d rows per fit", lean, years, fit_rows)
    registry = ctx.registry()
    stages = tuple(x for x in ctx.options.get("stages", "pre,final").split(",") if x in {"pre", "final"}) or ("pre", "final")
    outcomes: dict[str, Any] = {}
    failures = []
    for stage in stages:
        try:
            fs = build_training_features(store, s, ctx.calendar, stage=stage, years=years, end=end)
            cfg = TrainConfig(
                stage=stage, thresholds=s.range_thresholds, primary_threshold=s.primary_threshold, unusual_k=s.unusual_k,
                top_n=s.top_n, gate_min_days=s.gate_min_days, gate_max_ece=s.gate_max_ece,
                gate_min_brier_skill=s.gate_min_brier_skill, max_rows_per_fit=fit_rows,
            )
            result = train_stage(fs, cfg, s.models_dir, now=ctx.clock.now(), code_commit=release_id(), stop=ctx.expired)
            del fs
            decision = registry.register(result)
            outcomes[stage] = {"version": result.version, "status": result.status, "activated": decision["activate"],
                               "reason": decision["reason"], "seconds": round(result.seconds)}
        except TrainingError as exc:
            failures.append(f"{stage}: {exc}")
            outcomes[stage] = {"error": str(exc)}
        finally:
            release_memory()
    if failures:
        store.add_alert("warning", "training_failed", "; ".join(failures)[:400], dedupe_key=f"train:{ctx.run_key}",
                        dedupe_minutes=720)
    summary = ", ".join(f"{k}: {v.get('status', 'error')}" for k, v in outcomes.items())
    if lean:
        summary += f" (lean profile, level {lean})"
    return JobResult("failed" if len(failures) == len(stages) else "success", summary, outcomes)


MIN_TRAIN_YEARS = 3  # with fewer years the walk-forward has too few folds to validate anything
MIN_FIT_ROWS = 15_000


def train_years_for(settings: Any, n_universe: int, lean: int = 0) -> int:
    """History used for training: as many years as fit the row budget (RAM-bound on the 1 GB server), at least 3.
    ``lean`` drops one more year per level (see ``lean_level``)."""
    fit = int(settings.train_max_rows / max(n_universe * 252, 1))
    return max(MIN_TRAIN_YEARS, min(settings.train_years, fit) - max(lean, 0))


LIGHT_RAM_MB = 1600  # below this total RAM the lighter fitting profile is used
LIGHT_FIT_ROWS, FULL_FIT_ROWS = 60_000, 120_000


def fit_row_cap(settings: Any, lean: int = 0) -> int:
    """Rows sampled per booster fit: small servers sample less; every ``lean`` level halves it."""
    total = memory_info()["total_mb"]
    cap = LIGHT_FIT_ROWS if total is not None and total < LIGHT_RAM_MB else FULL_FIT_ROWS
    return max(MIN_FIT_ROWS, min(cap, settings.train_max_rows) >> max(lean, 0))


def lean_level(ctx: JobContext) -> int:
    """0 = normal profile. A training attempt that died (memory watchdog, OOM kill) is retried one level leaner per failed
    attempt, so a server that is too small for the full profile still ends up with a model instead of failing the same way
    every week. The bootstrap passes the level explicitly through the ``lean`` option."""
    try:
        forced = int(ctx.options.get("lean", "0"))
    except ValueError:
        forced = 0
    return max(ctx.attempt - 1, forced, 0)


def model_sync(ctx: JobContext) -> JobResult:
    """External training mode: fetch the latest signed artifact bundle, verify it, register it."""
    s = ctx.settings
    if s.training_mode != "external":
        return JobResult("skipped", "TRAINING_MODE is not external")
    if not (s.model_artifact_url or s.s3_configured):
        return JobResult("skipped", "no MODEL_ARTIFACT_URL or S3 storage configured")
    from models.external import sync_models

    outcome = sync_models(s, ctx.store, ctx.registry(), clock=ctx.clock)
    ok = outcome["synced"] or outcome["current"] or not outcome["failed"]
    return JobResult("success" if ok else "failed", outcome["message"], outcome)


def refresh_calendar_job(ctx: JobContext) -> JobResult:
    ctx.ensure_calendar()
    metrics: dict[str, Any] = {}
    if ctx.settings.nse_enabled:
        metrics["nse_days"] = _guarded("NSE holiday list", refresh_calendar, ctx.store, ctx.nse_api())
    metrics["inferred"] = _guarded("holiday inference", infer_holidays_from_bars, ctx.store)
    ctx.calendar.reload()
    ok = metrics.get("nse_days") or not ctx.settings.nse_enabled
    if ok:
        ctx.store.kv_set(CALENDAR_REFRESH_KEY, ctx.store.now())
    return JobResult("success" if ok else "failed", f"{metrics.get('nse_days') or 0} official holiday rows", metrics)


def refresh_universe(ctx: JobContext) -> JobResult:
    """Weekly: index membership / F&O eligibility changes, history for newly added names, upcoming results calendar."""
    from ingestion.universe import build_universe

    s, store = ctx.settings, ctx.store
    ctx.ensure_calendar()
    result = build_universe(store, s, ctx.nse_archive() if s.nse_enabled else None, ctx.run_date)
    ensure_series_instruments(store)
    report = backfill_history(store, ctx.yahoo(), now=ctx.clock.now(), years=s.history_years, deadline=ctx.budget(25 * 60),
                              clock=ctx.clock)
    metrics: dict[str, Any] = {"universe": len(result.symbols), "added": result.added, "removed": result.removed,
                               "history_ok": report.ok, "history_failed": report.failed, "source": result.source}
    metrics["graph"] = ctx.intel().reseed_graph()  # sector memberships follow the instrument master
    metrics["graph_learning"] = ctx.intel().learn_graph(ctx.run_date)  # measured sensitivities replace priors where the data allows
    if s.nse_enabled:
        day = ctx.run_date
        metrics["board_meetings"] = _guarded("board meetings", ctx.news().collect_board_meetings, ctx.nse_api(), day,
                                             day + timedelta(days=BOARD_MEETING_LOOKAHEAD_DAYS), set(result.symbols))
    return JobResult("success", f"{len(result.symbols)} symbols ({result.source}), +{result.added}/-{result.removed}", metrics)


def health_check(ctx: JobContext) -> JobResult:
    from automation.health import evaluate_health, raise_alerts

    report = evaluate_health(ctx.store, ctx.settings, ctx.clock, ctx.calendar)
    raise_alerts(ctx.store, report)
    return JobResult("success", f"health {report.level}: {len(report.problems)} problem(s)", {"level": report.level})


def bootstrap_job(ctx: JobContext) -> JobResult:
    from automation.bootstrap import run_bootstrap

    outcome = run_bootstrap(ctx, resume=not ctx.force, skip_train=ctx.options.get("skip_train") == "1",
                            data_only=ctx.options.get("data_only") == "1")
    return JobResult("success" if outcome.complete else "failed", outcome.summary(), {"steps": outcome.steps})


# -------------------------------------------------------------------------------------------------- registry
FUNCTIONS: dict[str, Callable[[JobContext], JobResult]] = {
    "collect_global": collect_global,
    "collect_news": collect_news,
    "collect_india": collect_india,
    "prelim_scan": prelim_scan,
    "refresh_preopen": refresh_preopen,
    "final_scan": final_scan,
    "closing_scan": closing_scan,
    "ingest_eod": ingest_eod,
    "evaluate": evaluate,
    "evaluate_ideas": evaluate_ideas,
    "backup": backup,
    "housekeeping": housekeeping,
    "train": train,
    "model_sync": model_sync,
    "refresh_calendar": refresh_calendar_job,
    "refresh_universe": refresh_universe,
    "health_check": health_check,
    "bootstrap": bootstrap_job,
}
