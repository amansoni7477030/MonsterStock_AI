"""Bounded storage: every table, folder and cache has an explicit retention rule, applied by the nightly housekeeping job.

Nothing in MONSTER AI may grow forever. The rules (all configurable; ``0`` = keep forever where that makes sense):

==========================  ============================================================  ===========================
what                        rule                                                          setting
==========================  ============================================================  ===========================
raw news / filing text      deleted after N days, structured events kept                  NEWS_RETENTION_DAYS (45)
structured events           deleted N days after their last update                        EVENT_RETENTION_DAYS (1100)
pre-open polls              thinned to the last poll per symbol and day after 14 days     (constant)
pre-open / live quotes      deleted after N days                                          QUOTE_RETENTION_DAYS (400)
daily bars, F&O rows        kept HISTORY_YEARS + 1 years (the oldest data ever read)      HISTORY_YEARS (10)
snapshots, predictions,     deleted after N days (immutable: pruning is explicit)         RECORD_RETENTION_DAYS (1825)
outcomes, metrics
graph edge history          superseded versions older than N days (every edge keeps       RECORD_RETENTION_DAYS (1825)
                            the version current when the window starts)
job runs, deliveries,       deleted after N days (delivered alerts: half)                 OPS_RETENTION_DAYS (180)
alerts
dated kv keys               evaluation summaries follow the records; fetch markers 30 d   (constant)
trade-idea records          ``ideas:run:`` (frozen) and ``ideas:out:`` (outcomes)         RECORD_RETENTION_DAYS (1825)
backups                     daily / weekly policy + a total size budget                   BACKUP_KEEP_*, BACKUP_MAX_MB
models                      last versions per stage                                       (models.registry)
cache / tmp                 age and size caps                                             (constants below)
logs                        age and size budget, today's files are never removed          LOG_RETENTION_DAYS, LOG_MAX_MB
whole data directory        budget; above it the optional data is trimmed harder          DATA_MAX_MB (4000)
==========================  ============================================================  ===========================

Deletes run in small transactions with a pause in between, so the scheduler heartbeat and the scans are never starved of
the single SQLite writer, and they are foreign-key safe (a prediction is only removed once no outcome points to it).
Under disk pressure (free space below the warning threshold) the optional windows shrink to half, below the critical
threshold to a quarter; the immutable records (snapshots, predictions, outcomes, metrics) and the price history are never
shortened by pressure, only by their own window.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.clock import Clock, get_clock, iso, ist_date, parse_ts
from app.config import Settings
from app.logging_setup import prune_logs
from app.resources import dir_size_mb, disk_info, prune_directory
from database.backup import enforce_budget
from database.store import Store

log = logging.getLogger(__name__)

RETENTION_KEY = "retention.last"
PRUNE_FLAG_KEY = "maintenance.allow_prune"
MB = 1048576.0
CHUNK_ROWS = 5000
CHUNK_PAUSE_SEC = 0.02
MAX_CHUNKS_PER_RULE = 4000          # ~20 M rows; a bigger backlog is finished by the next night
PREOPEN_DETAIL_DAYS = 14            # every poll is kept this long, then only the last one per symbol and day
FETCH_MARKER_DAYS = 30              # "yahoo:fetched:<symbol>" freshness markers of symbols that vanished
CACHE_MAX_MB = 200
CACHE_MAX_AGE_DAYS = 14
TMP_MAX_AGE_DAYS = 1
STALE_TEMP_DAYS = 1                 # leftovers of a killed backup / verification / restore
QUARANTINE_DAYS = 14                # quarantined databases of a restore (*.corrupt-<ts>)
DIAG_KEEP = 5                       # diagnostic bundles kept
MIN_WINDOW_DAYS = 7                 # pressure never shortens an optional window below this
STALE_FLAG_MINUTES = 30             # a prune guard left open this long is the leftover of a crashed run
BUDGET_WARN_SHARE = 0.80
VACUUM_FREE_FACTOR = 3.0            # VACUUM rewrites the file through the WAL: it needs ~3x the database size free
VACUUM_MIN_RECLAIM_MB = 32.0
VACUUM_MIN_FREE_SHARE = 0.15
PRESSURE_FACTOR = {"ok": 1.0, "unknown": 1.0, "warn": 0.5, "crit": 0.25}

NO_OUTCOME = "NOT EXISTS (SELECT 1 FROM outcomes o WHERE o.prediction_id = predictions.prediction_id)"
NO_PREDICTION = "NOT EXISTS (SELECT 1 FROM predictions p WHERE p.snapshot_id = snapshots.snapshot_id)"
# The edge history answers "which edges were valid at time T"; a row is only disposable once a newer row of the same edge was
# already valid at the cutoff, so every edge keeps the version that was current when the retained window starts.
SUPERSEDED_EDGE = (
    "EXISTS (SELECT 1 FROM graph_edge_history n WHERE n.src = graph_edge_history.src AND n.dst = graph_edge_history.dst "
    "AND n.relation = graph_edge_history.relation AND n.valid_from > graph_edge_history.valid_from AND n.valid_from <= ?1)")


@dataclass(frozen=True)
class TablePolicy:
    name: str
    table: str
    column: str                         # age expression: ISO dates ('YYYY-MM-DD', IST) or UTC stamps ('...Z') sort as text
    unit: str                           # "date" or "ts"
    days: Callable[[Settings], int]     # 0 = keep forever
    key: tuple[str, ...] = ("rowid",)   # row identity (the primary key of WITHOUT ROWID tables)
    where: str = ""                     # extra predicate restricting what may be deleted
    guarded: bool = False               # immutable table: pruning needs Store.allow_prune()
    shrink: bool = True                 # window shortened under disk pressure


@dataclass(frozen=True)
class KvPolicy:
    name: str
    prefix: str
    days: Callable[[Settings], int]


def _half(days: int) -> int:
    return max(days // 2, 1) if days > 0 else 0


def _history_days(settings: Settings) -> int:
    """Price history is read back at most HISTORY_YEARS (training window + feature look-back); one more year is the margin."""
    return (settings.history_years + 1) * 366


# Order matters: children before parents (outcomes -> predictions -> snapshots).
POLICIES: tuple[TablePolicy, ...] = (
    TablePolicy("live_quotes", "live_quotes", "received_at", "ts", lambda s: s.quote_retention_days),
    TablePolicy("preopen_quotes", "preopen_quotes", "trade_date", "date", lambda s: s.quote_retention_days),
    TablePolicy("events", "events", "last_updated_at", "ts", lambda s: s.event_retention_days),
    TablePolicy("alerts_delivered", "alerts", "created_at", "ts", lambda s: _half(s.ops_retention_days),
                where="delivered = 1"),
    TablePolicy("alerts", "alerts", "created_at", "ts", lambda s: s.ops_retention_days),
    TablePolicy("deliveries", "deliveries", "created_at", "ts", lambda s: s.ops_retention_days),
    TablePolicy("job_runs", "job_runs", "COALESCE(started_at, scheduled_for, '')", "ts", lambda s: s.ops_retention_days),
    TablePolicy("eval_metrics", "eval_metrics", "as_of_date", "date", lambda s: s.record_retention_days, shrink=False),
    TablePolicy("outcomes", "outcomes", "trade_date", "date", lambda s: s.record_retention_days, shrink=False),
    TablePolicy("predictions", "predictions", "trade_date", "date", lambda s: s.record_retention_days,
                where=NO_OUTCOME, guarded=True, shrink=False),
    TablePolicy("snapshots", "snapshots", "trade_date", "date", lambda s: s.record_retention_days,
                where=NO_PREDICTION, guarded=True, shrink=False),
    TablePolicy("graph_edge_history", "graph_edge_history", "valid_from", "ts", lambda s: s.record_retention_days,
                where=SUPERSEDED_EDGE, guarded=True, shrink=False),
    TablePolicy("fno_daily", "fno_daily", "trade_date", "date", _history_days, key=("symbol", "trade_date"), shrink=False),
    TablePolicy("bars_daily", "bars_daily", "trade_date", "date", _history_days, key=("instrument_id", "trade_date"),
                shrink=False),
)

KV_POLICIES: tuple[KvPolicy, ...] = (
    KvPolicy("kv_evaluation", "evaluation:", lambda s: s.record_retention_days),
    KvPolicy("kv_ideas", "ideas:", lambda s: s.record_retention_days),
    KvPolicy("kv_fetch_markers", "yahoo:fetched:", lambda s: FETCH_MARKER_DAYS),
)


def disk_pressure(free_mb: float | None, settings: Settings) -> str:
    """ok | warn | crit | unknown - by free disk only (RAM pressure is not a reason to delete data)."""
    if free_mb is None:
        return "unknown"
    if free_mb < settings.min_free_disk_mb_crit:
        return "crit"
    if free_mb < settings.min_free_disk_mb_warn:
        return "warn"
    return "ok"


@dataclass
class RetentionReport:
    at: str = ""
    pressure: str = "ok"
    factor: float = 1.0
    dry_run: bool = False
    rows: dict[str, int] = field(default_factory=dict)
    files: dict[str, int] = field(default_factory=dict)
    maintenance: dict[str, Any] = field(default_factory=dict)
    data_mb: float = 0.0
    db_mb: float = 0.0
    free_disk_mb: float | None = None
    over_budget: bool = False
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.rows.values())

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def removed(self) -> dict[str, int]:
        return {k: v for k, v in {**self.rows, **self.files}.items() if v}

    def summary(self) -> str:
        parts = [f"{k} -{v}" for k, v in self.removed().items()]
        return (f"retention {'(dry run) ' if self.dry_run else ''}{', '.join(parts) or 'nothing to remove'}; "
                f"data {self.data_mb:.0f} MB (db {self.db_mb:.0f} MB), disk pressure {self.pressure}")


# --------------------------------------------------------------------------------------------- windows
def window_days(policy: TablePolicy, settings: Settings, factor: float) -> int:
    """The retention window in days under the current pressure factor (0 = keep forever)."""
    days = int(policy.days(settings))
    if days <= 0:
        return 0
    if policy.shrink and factor < 1.0:
        days = max(int(days * factor), min(days, MIN_WINDOW_DAYS))
    return days


def cutoff_for(unit: str, now: datetime, days: int) -> str:
    """Rows with ``column < cutoff`` are older than the window. Trade dates are exchange-local (IST) calendar dates, so the
    cutoff never depends on the server's time zone."""
    if unit == "date":
        return (ist_date(now) - timedelta(days=days)).isoformat()
    return iso(now - timedelta(days=days)) or ""


# ---------------------------------------------------------------------------------------------- deletes
def _where(policy: TablePolicy) -> str:
    """``?1`` is the cutoff (a policy's own ``where`` may refer to it again), ``?2`` the chunk size of a delete."""
    return f"{policy.column} < ?1" + (f" AND ({policy.where})" if policy.where else "")


def _count_older(store: Store, policy: TablePolicy, cutoff: str) -> int:
    row = store.conn.execute(f"SELECT COUNT(*) AS n FROM {policy.table} WHERE {_where(policy)}", (cutoff,)).fetchone()
    return int(row["n"])


def _delete_sql(policy: TablePolicy) -> str:
    keys = ", ".join(policy.key)
    lhs = keys if len(policy.key) == 1 else f"({keys})"
    return f"DELETE FROM {policy.table} WHERE {lhs} IN (SELECT {keys} FROM {policy.table} WHERE {_where(policy)} LIMIT ?2)"


def _delete_chunked(store: Store, sql: str, params: tuple[Any, ...], clock: Clock, *, guarded: bool = False) -> int:
    """Run a ``... LIMIT ?`` delete repeatedly in short transactions, pausing between chunks (single-writer friendliness)."""
    total = 0
    with store.allow_prune() if guarded else contextlib.nullcontext():
        for _ in range(MAX_CHUNKS_PER_RULE):
            with store.tx() as conn:
                deleted = max(conn.execute(sql, (*params, CHUNK_ROWS)).rowcount, 0)
            total += deleted
            if deleted < CHUNK_ROWS:
                break
            clock.sleep(CHUNK_PAUSE_SEC)
    return total


def apply_policy(store: Store, policy: TablePolicy, settings: Settings, now: datetime, clock: Clock, factor: float,
                 *, dry_run: bool = False) -> int:
    days = window_days(policy, settings, factor)
    if days <= 0:
        return 0
    cutoff = cutoff_for(policy.unit, now, days)
    if dry_run:
        return _count_older(store, policy, cutoff)
    return _delete_chunked(store, _delete_sql(policy), (cutoff,), clock, guarded=policy.guarded)


def apply_kv_policy(store: Store, policy: KvPolicy, settings: Settings, now: datetime, *, dry_run: bool = False) -> int:
    days = int(policy.days(settings))
    if days <= 0:
        return 0
    cutoff = iso(now - timedelta(days=days)) or ""
    args = (len(policy.prefix), policy.prefix, cutoff)
    where = "substr(key, 1, ?) = ? AND updated_at < ?"
    if dry_run:
        return int(store.conn.execute(f"SELECT COUNT(*) FROM kv WHERE {where}", args).fetchone()[0])
    with store.tx() as conn:
        return max(conn.execute(f"DELETE FROM kv WHERE {where}", args).rowcount, 0)


def thin_preopen(store: Store, now: datetime, clock: Clock, factor: float, *, dry_run: bool = False) -> int:
    """Pre-open polls repeat every ~45 s during order entry. After ``PREOPEN_DETAIL_DAYS`` only the last poll per symbol and
    day is worth keeping (it is the equilibrium price); the rest is deleted."""
    detail_days = max(int(PREOPEN_DETAIL_DAYS * factor), 3)
    cutoff = cutoff_for("date", now, detail_days)
    conn = store.conn
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _keep_preopen (id INTEGER PRIMARY KEY)")
    try:
        conn.execute("DELETE FROM _keep_preopen")
        conn.execute("INSERT INTO _keep_preopen SELECT MAX(id) FROM preopen_quotes WHERE trade_date < ? "
                     "GROUP BY trade_date, symbol", (cutoff,))
        if dry_run:
            row = conn.execute("SELECT COUNT(*) FROM preopen_quotes WHERE trade_date < ? AND id NOT IN "
                               "(SELECT id FROM _keep_preopen)", (cutoff,)).fetchone()
            return int(row[0])
        sql = ("DELETE FROM preopen_quotes WHERE id IN (SELECT id FROM preopen_quotes WHERE trade_date < ? "
               "AND id NOT IN (SELECT id FROM _keep_preopen) LIMIT ?)")
        return _delete_chunked(store, sql, (cutoff,), clock)
    finally:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("DROP TABLE IF EXISTS _keep_preopen")


def reset_stale_prune_flag(store: Store, now: datetime) -> bool:
    """``Store.allow_prune`` lifts the immutability guard of snapshots / predictions while a prune runs. If the process died
    inside it the guard would stay open for ever, so a flag left on for a long time is closed again."""
    row = store.conn.execute("SELECT value, updated_at FROM kv WHERE key = ?", (PRUNE_FLAG_KEY,)).fetchone()
    if row is None or row["value"] != "1":
        return False
    since = parse_ts(row["updated_at"])
    if since is not None and now - since < timedelta(minutes=STALE_FLAG_MINUTES):
        return False
    store.kv_set(PRUNE_FLAG_KEY, "0")
    log.warning("immutability guard of snapshots/predictions was left open by a crashed prune; closed it again")
    return True


# ----------------------------------------------------------------------------------------------- files
def _stale(path: Path, days: float, now: float) -> bool:
    try:
        return now - path.stat().st_mtime >= days * 86400
    except OSError:
        return False


def _remove(path: Path) -> bool:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True
    except OSError:
        return False


def sweep_leftovers(directory: Path, names: Callable[[str], float | None], *, now: float | None = None,
                    dry_run: bool = False) -> int:
    """Remove entries of ``directory`` whose name ``names`` maps to a maximum age in days (``None`` = not a leftover)."""
    if not directory.exists():
        return 0
    moment = now if now is not None else time.time()
    removed = 0
    for path in directory.iterdir():
        limit = names(path.name)
        if limit is None or not _stale(path, limit, moment):
            continue
        if dry_run or _remove(path):
            removed += 1
    return removed


def backup_leftover_age(name: str, quarantine_days: float = QUARANTINE_DAYS) -> float | None:
    """What a killed backup / verification / restore leaves behind; real backups (``monster-*.db.gz``) are never matched."""
    if name.startswith((".tmp-backup-", ".verify-")) or name.endswith((".restoring", ".gz.tmp")):
        return STALE_TEMP_DAYS
    if ".corrupt-" in name:
        return quarantine_days
    return None


def sweep_backup_dir(backup_dir: Path, *, quarantine_days: float = QUARANTINE_DAYS, now: float | None = None,
                     dry_run: bool = False) -> int:
    return sweep_leftovers(backup_dir, lambda n: backup_leftover_age(n, quarantine_days), now=now, dry_run=dry_run)


def sweep_database_leftovers(db_path: Path, *, quarantine_days: float = QUARANTINE_DAYS, now: float | None = None,
                             dry_run: bool = False) -> int:
    """Quarantined copies of the live database (``monster.db.corrupt-<ts>`` and their -wal / -shm) and the staging files of
    an interrupted restore, next to the database."""
    stem = db_path.name

    def age(name: str) -> float | None:
        if not name.startswith(stem):
            return None
        return backup_leftover_age(name, quarantine_days)

    return sweep_leftovers(db_path.parent, age, now=now, dry_run=dry_run)


def prune_diag_bundles(folder: Path, keep: int, *, dry_run: bool = False) -> int:
    if not folder.exists():
        return 0
    bundles = sorted((p for p in folder.glob("diag-*") if p.is_file()), key=lambda p: p.stat().st_mtime)
    victims = bundles[:-keep] if keep > 0 else bundles
    return sum(1 for path in victims if dry_run or _remove(path))


def prune_files(settings: Settings, factor: float, *, dry_run: bool = False) -> dict[str, int]:
    """Caches, temporary files, logs, diagnostic bundles, leftovers of killed backups and the backup size budget."""
    scale = factor if factor < 1.0 else 1.0
    quarantine = max(QUARANTINE_DAYS * scale, 2.0)
    return {
        "cache": prune_directory(settings.cache_dir, max_mb=CACHE_MAX_MB * scale, max_age_days=CACHE_MAX_AGE_DAYS,
                                 dry_run=dry_run),
        "tmp": prune_directory(settings.tmp_dir, max_age_days=TMP_MAX_AGE_DAYS, dry_run=dry_run),
        "logs": prune_logs(settings.log_dir, settings.log_retention_days, settings.log_max_mb * scale, dry_run=dry_run),
        "backup_leftovers": sweep_backup_dir(settings.backup_dir, quarantine_days=quarantine, dry_run=dry_run),
        "db_leftovers": sweep_database_leftovers(settings.db_path, quarantine_days=quarantine, dry_run=dry_run),
        "diag": prune_diag_bundles(settings.data_dir / "diag", DIAG_KEEP, dry_run=dry_run),
        "backups_over_budget": len(enforce_budget(settings.backup_dir, settings.backup_max_mb * scale, dry_run=dry_run)),
    }


# --------------------------------------------------------------------------------------- database upkeep
def database_size(conn: sqlite3.Connection) -> tuple[float, float]:
    """(file size in MB, MB inside it that are free pages) - what VACUUM could give back."""
    page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
    pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
    free = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
    return pages * page_size / MB, free * page_size / MB


def maintain_database(store: Store, settings: Settings, now: datetime, *, free_disk_mb: float | None = None,
                      pressure: str = "ok", force_vacuum: bool = False, run_date: date | None = None) -> dict[str, Any]:
    """Checkpoint the WAL (so it cannot grow without bound), refresh planner statistics and - on Sundays (IST), under disk
    pressure or when forced - VACUUM so deleted pages go back to the file system. VACUUM is skipped when it would not
    reclaim anything worthwhile or when the disk has no room for the copy it makes."""
    actions: dict[str, Any] = {}
    conn = store.conn
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA optimize")
        db_mb, reclaimable_mb = database_size(conn)
        actions.update(db_mb=round(db_mb, 1), reclaimable_mb=round(reclaimable_mb, 1))
        sunday = (run_date or ist_date(now)).weekday() == 6
        due = force_vacuum or sunday or pressure in {"warn", "crit"}
        worth = force_vacuum or reclaimable_mb >= VACUUM_MIN_RECLAIM_MB or (
            db_mb > 0 and reclaimable_mb / db_mb >= VACUUM_MIN_FREE_SHARE)
        free = free_disk_mb if free_disk_mb is not None else disk_info(settings.data_dir)["free_mb"]
        if not (due and worth):
            return actions
        if free < VACUUM_FREE_FACTOR * db_mb:
            actions["vacuum_skipped"] = f"free disk {free:.0f} MB < {VACUUM_FREE_FACTOR:.0f} x database {db_mb:.0f} MB"
            log.warning("VACUUM skipped: %s", actions["vacuum_skipped"])
            return actions
        started = time.monotonic()
        conn.execute("VACUUM")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        actions["vacuum_s"] = round(time.monotonic() - started, 1)
        actions["db_mb_after"] = round(database_size(conn)[0], 1)
    except sqlite3.Error as exc:
        actions["error"] = f"{type(exc).__name__}: {exc}"[:200]
        log.warning("database maintenance failed: %s", exc)
    return actions


# ------------------------------------------------------------------------------------------- the run
def _file_mb(path: Path) -> float:
    return path.stat().st_size / MB if path.exists() else 0.0


def run_retention(store: Store, settings: Settings, *, now: datetime | None = None, clock: Clock | None = None,
                  free_disk_mb: float | None = None, dry_run: bool = False, maintenance: bool = True,
                  force_vacuum: bool = False, run_date: date | None = None) -> RetentionReport:
    """Apply every retention rule. ``dry_run`` only counts what would go. A failing rule is reported and never stops the
    others (or the housekeeping job that calls this)."""
    clk = clock or get_clock()
    moment = now or clk.now()
    started = time.monotonic()
    free = free_disk_mb if free_disk_mb is not None else disk_info(settings.data_dir)["free_mb"]
    pressure = disk_pressure(free, settings)
    factor = PRESSURE_FACTOR.get(pressure, 1.0)
    report = RetentionReport(at=iso(moment) or "", pressure=pressure, factor=factor, dry_run=dry_run, free_disk_mb=free)
    data_mb = dir_size_mb(settings.data_dir) if settings.data_dir.exists() else 0.0
    if settings.data_max_mb and data_mb > settings.data_max_mb:
        factor = min(factor, 0.5)  # over budget: trim the optional data as under disk pressure
        report.factor, report.over_budget = factor, True
        report.notes.append(f"data directory {data_mb:.0f} MB exceeds its {settings.data_max_mb} MB budget: "
                            "optional data trimmed harder")
    if not dry_run:
        with contextlib.suppress(sqlite3.Error):
            reset_stale_prune_flag(store, moment)

    steps: list[tuple[str, Callable[[], int]]] = [
        ("preopen_thinned", lambda: thin_preopen(store, moment, clk, factor, dry_run=dry_run)),
        *((p.name, lambda p=p: apply_policy(store, p, settings, moment, clk, factor, dry_run=dry_run)) for p in POLICIES),
        *((k.name, lambda k=k: apply_kv_policy(store, k, settings, moment, dry_run=dry_run)) for k in KV_POLICIES),
    ]
    for name, step in steps:
        try:
            report.rows[name] = step()
        except Exception as exc:  # noqa: BLE001 - one failing rule must never stop the others
            report.rows[name] = 0
            report.notes.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
            log.exception("retention rule %s failed", name)
    try:
        report.files = prune_files(settings, factor, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"files: {type(exc).__name__}: {str(exc)[:120]}")
        log.exception("retention of files failed")
    if maintenance and not dry_run:
        report.maintenance = maintain_database(store, settings, moment, free_disk_mb=free, pressure=pressure,
                                               force_vacuum=force_vacuum, run_date=run_date)
    report.data_mb = round(dir_size_mb(settings.data_dir) if settings.data_dir.exists() else 0.0, 1)
    report.db_mb = round(_file_mb(settings.db_path), 1)
    report.duration_s = round(time.monotonic() - started, 2)
    if not dry_run:
        with contextlib.suppress(sqlite3.Error):
            store.kv_set_json(RETENTION_KEY, report.to_dict())
    log.info("%s (%.1fs)", report.summary(), report.duration_s)
    for note in report.notes:
        log.warning("retention: %s", note)
    return report


# ----------------------------------------------------------------------------------------------- report
GROWING_TABLES: tuple[tuple[str, str | None], ...] = (
    ("bars_daily", "trade_date"), ("fno_daily", "trade_date"), ("preopen_quotes", "trade_date"),
    ("live_quotes", "received_at"), ("news_raw", "first_seen_at"), ("filings", "first_seen_at"),
    ("events", "last_updated_at"), ("snapshots", "trade_date"), ("predictions", "trade_date"),
    ("outcomes", "trade_date"), ("eval_metrics", "as_of_date"), ("job_runs", "started_at"),
    ("alerts", "created_at"), ("deliveries", "created_at"), ("graph_edge_history", "valid_from"), ("kv", None),
)


def table_sizes_mb(conn: sqlite3.Connection) -> dict[str, float]:
    """Per-table size from the ``dbstat`` virtual table; empty when this SQLite build does not have it."""
    try:
        rows = conn.execute("SELECT name, SUM(pgsize) AS bytes FROM dbstat GROUP BY name ORDER BY bytes DESC").fetchall()
    except sqlite3.Error:
        return {}
    return {r["name"]: round(r["bytes"] / MB, 2) for r in rows if r["bytes"]}


def storage_report(store: Store, settings: Settings, *, sizes: bool = True) -> dict[str, Any]:
    """Sizes per component, row counts / date ranges of the tables that grow, the policies and the last retention run."""
    def mb(path: Path) -> float:
        if path.is_file():
            return path.stat().st_size / MB
        return dir_size_mb(path) if path.exists() else 0.0

    components = {
        "database": mb(settings.db_path), "database_wal": mb(settings.db_path.with_name(settings.db_path.name + "-wal")),
        "models": mb(settings.models_dir), "backups": mb(settings.backup_dir), "cache": mb(settings.cache_dir),
        "tmp": mb(settings.tmp_dir), "logs": mb(settings.log_dir), "diagnostics": mb(settings.data_dir / "diag"),
    }
    tables: dict[str, dict[str, Any]] = {}
    for table, column in GROWING_TABLES:
        try:
            if column is None:
                tables[table] = {"rows": int(store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])}
                continue
            row = store.conn.execute(f"SELECT COUNT(*) AS n, MIN({column}) AS lo, MAX({column}) AS hi FROM {table}").fetchone()
            tables[table] = {"rows": int(row["n"]), "oldest": row["lo"], "newest": row["hi"]}
        except sqlite3.Error:
            continue
    policies = {p.name: {"days": int(p.days(settings)), "table": p.table} for p in POLICIES}
    policies["preopen_quotes"]["thinned_after_days"] = PREOPEN_DETAIL_DAYS
    for k in KV_POLICIES:
        policies[k.name] = {"days": int(k.days(settings)), "table": "kv"}
    db_mb, reclaimable = database_size(store.conn)
    disk = disk_info(settings.data_dir)
    return {
        "components_mb": {k: round(v, 1) for k, v in components.items()},
        "data_dir_mb": round(mb(settings.data_dir), 1), "data_budget_mb": settings.data_max_mb,
        "database_reclaimable_mb": round(reclaimable, 1), "database_pages_mb": round(db_mb, 1),
        "free_disk_mb": round(disk["free_mb"]), "disk_total_mb": round(disk["total_mb"]),
        "disk_pressure": disk_pressure(disk["free_mb"], settings),
        "tables": tables, "table_sizes_mb": table_sizes_mb(store.conn) if sizes else {}, "policies": policies,
        "last_run": store.kv_get_json(RETENTION_KEY),
    }


def storage_text(report: dict[str, Any]) -> str:
    lines = ["Storage", "-------"]
    for name, size in report["components_mb"].items():
        lines.append(f"  {name:<16}{size:>10.1f} MB")
    budget = report["data_budget_mb"]
    lines.append(f"  {'data directory':<16}{report['data_dir_mb']:>10.1f} MB"
                 + (f" of a {budget} MB budget" if budget else " (no budget)"))
    lines.append(f"  {'free disk':<16}{report['free_disk_mb']:>10,} MB of {report['disk_total_mb']:,} MB "
                 f"(pressure: {report['disk_pressure']})")
    lines.append(f"  {'reclaimable':<16}{report['database_reclaimable_mb']:>10.1f} MB of free pages inside the database")
    lines += ["", "Tables (rows, oldest .. newest)", "-------------------------------"]
    sizes = report.get("table_sizes_mb") or {}
    for name, info in report["tables"].items():
        span = f"  {info['oldest']} .. {info['newest']}" if info.get("oldest") else ""
        size = f"  {sizes[name]:>8.1f} MB" if name in sizes else ""
        lines.append(f"  {name:<20}{info['rows']:>12,}{size}{span}")
    lines += ["", "Retention windows (days; 0 = keep forever)", "------------------------------------------"]
    for name, info in report["policies"].items():
        extra = f", thinned to one poll a day after {info['thinned_after_days']} d" if "thinned_after_days" in info else ""
        lines.append(f"  {name:<20}{info['days']:>6}{extra}")
    last = report.get("last_run")
    if last:
        removed = {k: v for k, v in {**last["rows"], **last["files"]}.items() if v}
        lines += ["", f"Last retention run: {last['at']} (pressure {last['pressure']}, {last.get('duration_s', 0)} s) - "
                      + (", ".join(f"{k} -{v}" for k, v in removed.items()) or "nothing to remove")]
    else:
        lines += ["", "Last retention run: never"]
    return "\n".join(lines)


def storage_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)
