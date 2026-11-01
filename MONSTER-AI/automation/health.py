"""Operational health (blueprint section 18): one place that decides whether the system is trustworthy right now.

``evaluate_health`` is pure inspection (no network, no writes). ``raise_alerts`` turns problems into deduplicated alerts that
the notification service delivers to the operator chat. Exit codes for the CLI: 0 ok, 1 warnings, 2 critical.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.clock import Clock, at_ist, get_clock, iso, ist_date, parse_ts
from app.config import Settings
from app.locks import lock_dir
from app.resources import check_resources, dir_size_mb
from automation.calendar import ExchangeCalendar
from automation.retention import BUDGET_WARN_SHARE, RETENTION_KEY
from database.migrate import check_integrity
from database.store import Store

log = logging.getLogger(__name__)

HEARTBEAT_KEY = "daemon.heartbeat"
HEARTBEAT_FILE = "daemon.heartbeat"
HEARTBEAT_MAX_AGE_SEC = 180
BACKUP_WARN_HOURS = 36
BACKUP_CRIT_HOURS = 72
TRAIN_WARN_DAYS = 14
MODEL_STALE_DAYS = 120
SOURCE_FAIL_WARN = 5
CALENDAR_REFRESH_KEY = "calendar.last_refresh"  # written by the refresh_calendar job
CALENDAR_STALE_DAYS = 10
CALENDAR_HORIZON_WARN_DAYS = 21  # warn when the official holiday list ends within this many days
RETENTION_STALE_DAYS = 3  # the nightly housekeeping (retention) has not completed for this long
WAL_WARN_MB = 256  # a WAL this large means checkpoints are starved (a reader never lets go)
EOD_BARS_EXPECTED_AFTER = timedelta(hours=3)  # end-of-day bars are expected this long after the close (ingest_eod runs ~17:45)
ECE_WARN = 0.10
SKILL_WARN = -0.02
CRITICAL_JOBS = ("prelim_scan", "final_scan")


@dataclass
class Problem:
    level: str  # warn | crit
    code: str
    message: str


@dataclass
class HealthReport:
    generated_at: str
    level: str = "ok"
    problems: list[Problem] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str) -> None:
        self.problems.append(Problem(level, code, message))
        if level == "crit" or self.level == "ok":
            self.level = "crit" if level == "crit" else "warn"

    @property
    def exit_code(self) -> int:
        return {"ok": 0, "warn": 1, "crit": 2}[self.level]

    def to_dict(self) -> dict[str, Any]:
        return {"generated_at": self.generated_at, "level": self.level, "problems": [asdict(p) for p in self.problems],
                "checks": self.checks}


def _age_hours(stamp: str | None, now: datetime) -> float | None:
    moment = parse_ts(stamp)
    return None if moment is None else (now - moment).total_seconds() / 3600.0


# ------------------------------------------------------------------------------------------- heartbeat
def heartbeat_path(data_dir: Path) -> Path:
    return lock_dir(data_dir) / HEARTBEAT_FILE


def write_heartbeat_file(data_dir: Path, now: datetime) -> bool:
    """The scheduler's liveness proof that does not depend on the database: a VACUUM or a bulk delete can hold the SQLite
    writer lock for a while, but must never make a healthy daemon look dead. Atomic (temp file + replace)."""
    path = heartbeat_path(data_dir)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps({"at": iso(now), "pid": os.getpid()}), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        log.warning("heartbeat file not written: %s", exc)
        with contextlib.suppress(OSError):
            tmp.unlink()
        return False


def read_heartbeat_file(data_dir: Path) -> datetime | None:
    try:
        return parse_ts(json.loads(heartbeat_path(data_dir).read_text(encoding="utf-8")).get("at"))
    except (OSError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------------------------------- checks
def _check_database(report: HealthReport, store: Store) -> None:
    try:
        ok = check_integrity(store.conn, quick=True)
    except Exception as exc:  # noqa: BLE001
        report.add("crit", "db_unreadable", f"database check failed: {type(exc).__name__}")
        return
    report.checks["db_integrity"] = ok
    if not ok:
        report.add("crit", "db_corrupt", "database integrity check FAILED - restore from backup (monster backup --restore-latest)")


def _check_resources(report: HealthReport, settings: Settings) -> None:
    status = check_resources(settings.data_dir, warn_disk_mb=settings.min_free_disk_mb_warn,
                             crit_disk_mb=settings.min_free_disk_mb_crit, min_ram_mb=settings.min_available_ram_mb)
    report.checks["free_disk_mb"] = round(status.free_disk_mb or 0)
    report.checks["available_ram_mb"] = None if status.available_ram_mb is None else round(status.available_ram_mb)
    if status.level in {"warn", "crit"}:
        report.add(status.level, "resources", "; ".join(status.messages))


def _check_heartbeat(report: HealthReport, store: Store, settings: Settings, now: datetime) -> None:
    """The scheduler is alive if EITHER proof is fresh: the database row (what the dashboard shows) or the heartbeat file
    (written independently of the SQLite writer lock, so a long VACUUM never makes a healthy daemon look dead)."""
    db_beat = parse_ts(store.kv_get(HEARTBEAT_KEY))
    file_beat = read_heartbeat_file(settings.data_dir)
    beats = [b for b in (db_beat, file_beat) if b is not None]
    if not beats:
        if store.recent_jobs(1):
            report.add("crit", "scheduler_silent", "scheduler has never reported a heartbeat although jobs exist")
        return
    age = (now - max(beats)).total_seconds()
    report.checks["heartbeat_age_sec"] = round(age)
    if db_beat is not None:
        report.checks["heartbeat_db_age_sec"] = round((now - db_beat).total_seconds())
    if age > HEARTBEAT_MAX_AGE_SEC:
        report.add("crit", "scheduler_stalled", f"scheduler heartbeat is {age / 60:.0f} min old (service stopped or hung?)")
    elif db_beat is not None and (now - db_beat).total_seconds() > HEARTBEAT_MAX_AGE_SEC:
        report.add("warn", "heartbeat_db_stale", "the scheduler is alive (heartbeat file) but cannot write to the database: "
                   "a long maintenance run or a locked database")


def _check_jobs(report: HealthReport, store: Store, calendar: ExchangeCalendar, now: datetime) -> None:
    """Any (job, run) of the last 36 h that ended without success is a problem; critical scans are checked by deadline."""
    since = iso(now - timedelta(hours=36))
    rows = store.conn.execute(
        "SELECT job_name, run_key, status, message, run_id FROM job_runs WHERE started_at >= ? ORDER BY run_id", (since,)
    ).fetchall()
    final_status: dict[tuple[str, str], tuple[str, str | None]] = {}
    for r in rows:
        key = (r["job_name"], r["run_key"])
        if final_status.get(key, ("", None))[0] in {"success", "skipped"}:
            continue
        final_status[key] = (r["status"], r["message"])
    for (job, run_key), (status, message) in final_status.items():
        if status in {"failed", "missed", "timeout"}:
            level = "crit" if job in CRITICAL_JOBS else "warn"
            report.add(level, f"job_{job}", f"{job} [{run_key}] {status}: {message or 'no detail'}")
    today = ist_date(now)
    first_start = parse_ts(store.kv_get("daemon.first_start"))
    installed_before_scan = first_start is not None and first_start < at_ist(today, "09:08")
    if calendar.is_trading_day(today) and installed_before_scan:
        deadline = at_ist(today, "09:20")
        if now > deadline and not store.job_settled("final_scan", today.isoformat()):
            report.add("crit", "final_scan_missing", f"no final scan recorded for {today} by 09:20 IST")


def _check_data(report: HealthReport, store: Store, calendar: ExchangeCalendar, now: datetime) -> None:
    universe = store.list_instruments(kind="equity", in_universe=True, status="active")
    report.checks["universe"] = len(universe)
    if not universe:
        report.add("warn", "no_universe", "no instruments in the universe yet (bootstrap not finished?)")
        return
    # The official end-of-day file is ingested ~17:45 IST: from the close (15:30) until then the newest session legitimately
    # has no bars yet, so judge the universe against the session that SHOULD be complete by now (a daily false alarm before).
    session = calendar.last_completed_session(now - EOD_BARS_EXPECTED_AFTER).isoformat()
    have_bar = store.symbols_with_bar(session, universe_only=True)
    coverage = sum(1 for r in universe if r["symbol"] in have_bar) / len(universe)
    report.checks.update(last_session=session, coverage=round(coverage, 3))
    if coverage < 0.90:
        report.add("warn", "stale_bars", f"only {coverage:.0%} of the universe has the {session} bar")


def _check_models(report: HealthReport, store: Store, settings: Settings, now: datetime) -> None:
    for stage in ("pre", "final"):
        active = store.active_model(stage)
        report.checks[f"model_{stage}"] = None if active is None else {"version": active["model_version"], "status": active["status"]}
        if active is None:
            report.add("warn", f"no_model_{stage}", f"no trained '{stage}' model is active; the heuristic baseline is in use")
            continue
        age = _age_hours(active.get("created_at"), now)
        if age is not None and age > MODEL_STALE_DAYS * 24:
            report.add("warn", f"model_stale_{stage}", f"active {stage} model is {age / 24:.0f} days old")
    if settings.training_mode == "local-light":
        last = store.last_job_success("train")
        age = _age_hours(last["finished_at"], now) if last else None
        if store.active_model("pre") is not None and (age is None or age > TRAIN_WARN_DAYS * 24):
            report.add("warn", "training_overdue", f"last successful training was {'never' if age is None else f'{age / 24:.0f} days ago'}")


def _check_sources(report: HealthReport, store: Store) -> None:
    failing = [h for h in store.source_health() if h["enabled"] and h["consecutive_failures"] >= SOURCE_FAIL_WARN]
    for h in failing:
        report.add("warn", f"source_{h['source_id']}",
                   f"source {h['source_id']} failed {h['consecutive_failures']} times in a row ({(h['last_error'] or '')[:80]})")
    report.checks["failing_sources"] = [h["source_id"] for h in failing]


def _check_backup(report: HealthReport, store: Store, now: datetime) -> None:
    last = store.kv_get_json("backup.last")
    age = _age_hours(last.get("at"), now) if last else None
    report.checks["backup_age_h"] = None if age is None else round(age, 1)
    if age is None:
        if store.recent_jobs(1):
            report.add("warn", "no_backup", "no database backup has completed yet")
    elif age > BACKUP_CRIT_HOURS:
        report.add("crit", "backup_old", f"last backup is {age:.0f} h old")
    elif age > BACKUP_WARN_HOURS:
        report.add("warn", "backup_old", f"last backup is {age:.0f} h old")


def _check_notifications(report: HealthReport, store: Store, settings: Settings) -> None:
    report.checks["telegram_configured"] = settings.telegram_configured
    if not settings.telegram_configured:
        report.add("warn", "telegram_not_configured",
                   "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set: reports are generated but cannot be delivered "
                   "(run: python -m app.cli telegram-setup)")
        return
    row = store.conn.execute(
        "SELECT status, error, trade_date FROM deliveries WHERE kind = 'final_report' ORDER BY delivery_id DESC LIMIT 1").fetchone()
    if row and row["status"] in {"failed", "withheld"}:
        report.add("warn", "report_not_delivered", f"final report for {row['trade_date']} was {row['status']}: {row['error']}")


def _check_performance(report: HealthReport, store: Store) -> None:
    rolling = {m["name"]: m["value"] for m in store.latest_metrics("rolling") if m.get("slice", "all") == "all"}
    n_days = rolling.get("n_days_90") or 0
    report.checks["evaluated_days_90"] = n_days
    if n_days < 20:
        return
    ece, skill = rolling.get("ece_90"), rolling.get("brier_skill_90")
    if ece is not None and ece > ECE_WARN:
        report.add("warn", "calibration_drift", f"live calibration error {ece:.3f} over {int(n_days)} days (> {ECE_WARN})")
    if skill is not None and skill < SKILL_WARN:
        report.add("warn", "skill_negative", f"live Brier skill {skill:+.3f} is below the base rate; consider rollback "
                                              "(python -m app.cli models rollback)")


def _check_calendar(report: HealthReport, store: Store, settings: Settings, now: datetime,
                    calendar: ExchangeCalendar | None = None) -> None:
    """Holidays must be right for ANY date. NSE publishes next year's list in December; until then (and if the NSE refresh keeps
    failing) rule-based holidays and the final scan's live check cover the gap, and this check says so early."""
    cal = calendar or ExchangeCalendar(store)
    cal.reload()
    status = cal.status(ist_date(now))
    report.checks["calendar_official_through"] = status.official_through.isoformat() if status.official_through else None
    report.checks["calendar_days_left"] = status.days_left
    if status.days_left is None:
        report.add("warn", "calendar_unofficial", "no official NSE holiday list is loaded; holidays come from rules only "
                                                  f"(fixed-date holidays and Good Friday){_refresh_hint(settings)}")
    elif status.days_left < 0:
        report.add("warn", "calendar_unofficial", f"the official holiday list ended on {status.official_through}; holidays "
                    f"since then come from rules only{_refresh_hint(settings)}")
    elif status.days_left <= CALENDAR_HORIZON_WARN_DAYS:
        report.add("warn", "calendar_horizon", f"the official holiday list ends in {status.days_left} days "
                    f"({status.official_through}); NSE publishes the next year's list in December and the daily refresh "
                    "picks it up")
    if not settings.nse_enabled:
        return
    age = _age_hours(store.kv_get(CALENDAR_REFRESH_KEY), now)
    report.checks["calendar_refresh_age_h"] = None if age is None else round(age, 1)
    first = parse_ts(store.kv_get("daemon.first_start"))
    running_days = (now - first).total_seconds() / 86400 if first else 0
    if age is None:
        if running_days > 3:
            report.add("warn", "calendar_never_refreshed", "the NSE holiday calendar has never been refreshed (API blocked?); "
                                                          "holidays beyond the shipped list are unknown")
    elif age > CALENDAR_STALE_DAYS * 24:
        report.add("warn", "calendar_stale", f"the holiday calendar was last refreshed {age / 24:.0f} days ago")


def _check_storage(report: HealthReport, store: Store, settings: Settings, now: datetime) -> None:
    """Nothing may grow without bound: the data directory stays within its budget and the nightly retention keeps running."""
    data_mb = dir_size_mb(settings.data_dir) if settings.data_dir.exists() else 0.0
    budget = settings.data_max_mb
    report.checks["data_mb"] = round(data_mb)
    report.checks["data_budget_mb"] = budget
    if budget and data_mb >= budget:
        report.add("warn", "storage_over_budget", f"the data directory uses {data_mb:.0f} MB, above its {budget} MB budget "
                   "(retention trims optional data harder; inspect with: python -m app.cli storage)")
    elif budget and data_mb >= BUDGET_WARN_SHARE * budget:
        report.add("warn", "storage_budget", f"the data directory uses {data_mb:.0f} MB of its {budget} MB budget")
    wal = settings.db_path.with_name(settings.db_path.name + "-wal")
    wal_mb = wal.stat().st_size / 1048576.0 if wal.exists() else 0.0
    report.checks["wal_mb"] = round(wal_mb, 1)
    if wal_mb > WAL_WARN_MB:
        report.add("warn", "wal_large", f"the database WAL file is {wal_mb:.0f} MB: checkpoints are blocked by a long-running "
                   "reader (restart the dashboard / daemon)")
    last = store.kv_get_json(RETENTION_KEY)
    age = _age_hours(last.get("at"), now) if isinstance(last, dict) else None
    report.checks["retention_age_h"] = None if age is None else round(age, 1)
    first = parse_ts(store.kv_get("daemon.first_start"))
    running_days = (now - first).total_seconds() / 86400 if first else 0
    if age is None:
        if running_days > RETENTION_STALE_DAYS:
            report.add("warn", "retention_never_ran", "the nightly retention (housekeeping job) has never completed although "
                        "the scheduler has been running for days")
    elif age > RETENTION_STALE_DAYS * 24:
        report.add("warn", "retention_stale", f"the last retention run was {age / 24:.0f} days ago (housekeeping job failing?)")
    elif last.get("notes"):
        report.add("warn", "retention_errors", "retention rules failed: " + "; ".join(last["notes"])[:200])


def _refresh_hint(settings: Settings) -> str:
    return "; the daily refresh from NSE fixes this as soon as the list is published" if settings.nse_enabled else \
        " (NSE_ENABLED=false: add the dates to calendar_overrides.json)"


def evaluate_health(store: Store, settings: Settings, clock: Clock | None = None,
                    calendar: ExchangeCalendar | None = None) -> HealthReport:
    clk = clock or get_clock()
    now = clk.now()
    cal = calendar or ExchangeCalendar(store)
    report = HealthReport(generated_at=iso(now) or "")
    for name, check in (
        ("database", lambda: _check_database(report, store)),
        ("resources", lambda: _check_resources(report, settings)),
        ("heartbeat", lambda: _check_heartbeat(report, store, settings, now)),
        ("jobs", lambda: _check_jobs(report, store, cal, now)),
        ("data", lambda: _check_data(report, store, cal, now)),
        ("calendar", lambda: _check_calendar(report, store, settings, now, cal)),
        ("storage", lambda: _check_storage(report, store, settings, now)),
        ("models", lambda: _check_models(report, store, settings, now)),
        ("sources", lambda: _check_sources(report, store)),
        ("backup", lambda: _check_backup(report, store, now)),
        ("notifications", lambda: _check_notifications(report, store, settings)),
        ("performance", lambda: _check_performance(report, store)),
    ):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - a broken check is itself a finding, never a crash
            log.exception("health check %s failed", name)
            report.add("warn", f"check_{name}", f"health check '{name}' could not run: {type(exc).__name__}")
    return report


def raise_alerts(store: Store, report: HealthReport) -> int:
    """Queue one deduplicated alert per problem (critical repeats every 2 h, warnings every 12 h)."""
    queued = 0
    for p in report.problems:
        fresh = store.add_alert("error" if p.level == "crit" else "warning", f"health_{p.code}", p.message,
                                dedupe_key=f"health:{p.code}", dedupe_minutes=120 if p.level == "crit" else 720)
        queued += int(fresh)
    return queued
