"""Run one job in the current process with locking, a memory watchdog and a complete ledger entry.

The scheduler starts every job as ``python -m app.cli run-job <name> --key <run key>``; operators can run the same command by
hand. Whatever happens (exception, SIGTERM, timeout, OOM guard), the ledger row is closed with the true outcome.
"""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Mapping
from typing import Any

from app.clock import Clock, get_clock
from app.config import Settings
from app.locks import heavy_lock, job_lock
from app.logging_setup import job_context, redact
from app.resources import RssWatchdog, peak_rss_mb
from automation.context import JobContext, JobResult
from automation.jobspec import JOBS
from automation.schedule import PREEMPTED
from database.store import Store

log = logging.getLogger(__name__)

TERMINATE_REASON_KEY = "daemon.terminate_reason"
HEAVY_LOCK_WAIT_SEC = 300
OK_STATUSES = {"success", "skipped"}


class Terminated(BaseException):
    """Raised inside the job process when the scheduler asks it to stop (SIGTERM)."""


def _install_sigterm() -> None:
    def handler(_signum: int, _frame: Any) -> None:
        raise Terminated()

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # pragma: no cover - not in the main thread / unsupported platform
        pass


def _terminated_outcome(store: Store, timeout_min: int) -> tuple[str, str]:
    reason = store.kv_get(TERMINATE_REASON_KEY, "") or "stop requested"
    if reason == "timeout":
        return "timeout", f"timed out after {timeout_min} min"
    return "failed", f"{PREEMPTED}: {reason}"


def execute_job(
    settings: Settings, name: str, run_key: str, *, attempt: int = 1, max_attempts: int = 1, last_attempt: bool = True,
    force: bool = False, dry_run: bool = False, timeout_min: int | None = None, clock: Clock | None = None,
    store: Store | None = None, install_signal: bool = False, options: Mapping[str, str] | None = None,
) -> JobResult:
    spec = JOBS.get(name)
    if spec is None:
        raise KeyError(f"unknown job {name!r}; known: {', '.join(sorted(JOBS))}")
    clk = clock or get_clock()
    minutes = timeout_min or spec.timeout_min
    own_store = store is None
    st = store or Store.open(settings.db_path, clock=clk)
    lock = job_lock(settings.data_dir, name)
    heavy = heavy_lock(settings.data_dir) if spec.heavy else None
    if install_signal:
        _install_sigterm()
    try:
        if not lock.acquire(blocking=False):
            return JobResult("skipped", f"{name} is already running (pid {lock.holder_pid()})")
        if heavy is not None and not heavy.acquire(blocking=True, timeout=HEAVY_LOCK_WAIT_SEC):
            st.job_record(name, run_key, "failed", message=f"{PREEMPTED}: another heavy job holds the lock", attempt=attempt)
            return JobResult("failed", "another heavy job is running")
        # imported late: it pulls in the data stack, which only job processes need
        from automation.jobs import FUNCTIONS

        return _run(settings, spec.name, FUNCTIONS[name], st, clk, run_key, attempt, max_attempts, last_attempt, force, dry_run,
                    minutes, dict(options or {}))
    finally:
        if heavy is not None:
            heavy.release()
        lock.release()
        if own_store:
            st.close()


def _run(settings, name, fn, store, clock, run_key, attempt, max_attempts, last_attempt, force, dry_run, minutes,
         options) -> JobResult:
    run_id = store.job_start(name, run_key, attempt=attempt, pid=os.getpid())
    # every line logged from here on reads "[final_scan 2026-11-02#1 r1234]": job, run key, attempt and ledger row
    with job_context(name, f"{run_key}#{attempt} r{run_id}"):
        return _execute(settings, name, fn, store, clock, run_key, run_id, attempt, max_attempts, last_attempt, force, dry_run,
                        minutes, options)


def _execute(settings, name, fn, store, clock, run_key, run_id, attempt, max_attempts, last_attempt, force, dry_run, minutes,
             options) -> JobResult:
    from ingestion.http import (  # job processes only: keeps the scheduler daemon lean
        http_summary,
        reset_http_stats,
    )

    log.info("job %s [%s] attempt %d/%d starting: pid %d, timeout %d min, force=%s, dry_run=%s%s", name, run_key, attempt,
             max_attempts, os.getpid(), minutes, force, dry_run,
             f", options {sorted(options)}" if options else "")
    reset_http_stats()
    started = clock.monotonic()
    watchdog = RssWatchdog(float(settings.job_max_rss_mb)).start()
    ctx = JobContext.create(settings, store, run_key=run_key, clock=clock, attempt=attempt, max_attempts=max_attempts,
                            last_attempt=last_attempt, dry_run=dry_run, force=force, options=options,
                            deadline=clock.monotonic() + minutes * 60)
    result = JobResult("failed", "job did not complete")
    try:
        result = fn(ctx)
        if result.status not in {"success", "skipped", "failed"}:
            result = JobResult("failed", f"job returned invalid status {result.status!r}")
    except Terminated:
        status, message = _terminated_outcome(store, minutes)
        result = JobResult(status, message)
    except Exception as exc:  # noqa: BLE001 - the ledger must record every failure
        log.exception("job %s [%s] crashed", name, run_key)
        result = JobResult("failed", f"{type(exc).__name__}: {redact(str(exc))[:300]}")
    finally:
        watchdog.stop()
        store.job_finish(run_id, result.status, message=result.message, exit_code=0 if result.status in OK_STATUSES else 1,
                         peak_rss_mb=peak_rss_mb())
    level = log.info if result.status in OK_STATUSES else log.warning
    peak = peak_rss_mb()
    level("job %s [%s] attempt %d/%d -> %s in %.1f s (peak memory %s): %s", name, run_key, attempt, max_attempts, result.status,
          clock.monotonic() - started, f"{peak:.0f} MB" if peak else "n/a", result.message)
    traffic = http_summary()
    if traffic:
        log.info("job %s network: %s", name, traffic)
    if result.status in {"failed", "timeout"} and last_attempt and not result.message.startswith(PREEMPTED):
        store.add_alert("error", f"job_failed_{name}", f"{name} [{run_key}] failed: {result.message}",
                        dedupe_key=f"job-failed:{name}:{run_key}", dedupe_minutes=24 * 60)
    return result
