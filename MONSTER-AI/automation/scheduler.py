"""The scheduler daemon (systemd ``Type=notify``): one small, long-lived process that never imports the data stack.

It consults the ledger-based engine every couple of seconds, runs one job at a time as a child process (memory isolation
on a 1 GB server), lets time-critical jobs preempt long ones, enforces timeouts, recovers from crashed children, keeps a
heartbeat for the watchdog and flushes operator alerts. A failure of any single job or tick never stops the loop.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import socket
import sqlite3
import subprocess  # noqa: S404 - children are our own CLI, launched with a fixed argument vector (no shell)
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.clock import Clock, get_clock, iso, parse_ts
from app.config import Settings
from app.locks import FileLock, lock_dir
from app.resources import check_resources, disk_info, emergency_cleanup, memory_info
from automation.calendar import ExchangeCalendar, seed_calendar
from automation.health import HEARTBEAT_KEY, read_heartbeat_file, write_heartbeat_file
from automation.jobspec import JOBS
from automation.runner import TERMINATE_REASON_KEY
from automation.schedule import PREEMPTED, Due, ScheduleEngine
from database.store import Store
from ingestion.registry import register_all

log = logging.getLogger(__name__)

BOOTSTRAP_STATE_KEY = "bootstrap.state"
RUN_STATE_KEY = "daemon.run_state"  # {"state": "running" | "stopped", ...}: "running" at the next start = an unclean stop
BOOTSTRAP_RETRY_SEC = 1800
BOOTSTRAP_MAX_ATTEMPTS_PER_DAY = 6
HEARTBEAT_EVERY_SEC = 15.0
HEARTBEAT_DB_WAIT_MS = 2000  # the heartbeat never waits the connection's full busy timeout for a locked database
ALERT_FLUSH_EVERY_SEC = 60.0
RESOURCE_CHECK_EVERY_SEC = 300.0
TERMINATE_GRACE_SEC = 20.0
CRITICAL_QUIET_SEC = 120  # no alert flushing this close to a time-critical job


def sd_notify(message: str) -> bool:
    """Minimal systemd notification (READY=1, WATCHDOG=1, STATUS=...); a no-op outside systemd."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode())
        return True
    except OSError:
        return False


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":  # pragma: no cover - dev machines only; os.kill(pid, 0) would TERMINATE on Windows
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, 0, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------------------------- launching
class SubprocessLauncher:
    """Starts ``python -m app.cli run-job ...`` children (no shell, fixed argv)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def start(self, due: Due) -> subprocess.Popen:
        cmd = [sys.executable, "-m", "app.cli", "run-job", due.entry.job, "--key", due.run_key, "--attempt", str(due.attempt),
               "--max-attempts", str(due.entry.max_attempts), "--timeout-min", str(due.timeout_min)]
        if due.last_attempt:
            cmd.append("--last-attempt")
        return subprocess.Popen(cmd, cwd=str(self.settings.home), stdin=subprocess.DEVNULL)  # noqa: S603

    def start_named(self, args: list[str]) -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-m", "app.cli", *args], cwd=str(self.settings.home),  # noqa: S603
                                stdin=subprocess.DEVNULL)

    @staticmethod
    def poll(handle: subprocess.Popen) -> int | None:
        return handle.poll()

    @staticmethod
    def pid(handle: subprocess.Popen) -> int | None:
        return handle.pid

    @staticmethod
    def terminate(handle: subprocess.Popen) -> None:
        with contextlib.suppress(OSError):
            handle.terminate()

    @staticmethod
    def kill(handle: subprocess.Popen) -> None:
        with contextlib.suppress(OSError):
            handle.kill()


@dataclass
class Running:
    label: str
    job: str
    run_key: str
    attempt: int
    critical: bool
    handle: Any
    started: float
    timeout_sec: float
    reason: str | None = None
    kill_at: float | None = None


class Scheduler:
    def __init__(
        self, settings: Settings, store: Store, engine: ScheduleEngine, launcher: Any, *, clock: Clock | None = None,
        notifier_factory: Callable[[], Any] | None = None, notify: Callable[[str], bool] = sd_notify, tick_sec: float = 2.0,
    ) -> None:
        self.settings, self.store, self.engine, self.launcher = settings, store, engine, launcher
        self._clock = clock
        self._notifier_factory = notifier_factory
        self._notifier: Any = None
        self._notify = notify
        self.tick_sec = tick_sec
        self.running: Running | None = None
        self.stop_requested = False
        self._last = {"heartbeat": -1e9, "flush": -1e9, "resources": -1e9, "bootstrap": -1e9, "ram_warn": -1e9,
                      "disk_warn": -1e9, "db_beat_warn": -1e9}

    @property
    def clock(self) -> Clock:
        return self._clock or get_clock()

    # --------------------------------------------------------------------------------------------- loop
    def run_forever(self) -> None:
        recovered = self.engine.recover_stale(alive=pid_alive)
        if recovered:
            log.warning("recovered %d ledger rows left running by a previous process", recovered)
        self._register_start(self.clock.now())
        self._notify("READY=1")
        log.info("scheduler started (tick %.1fs, pid %d)", self.tick_sec, os.getpid())
        while not self.stop_requested:
            self.tick_safely()
            self.clock.sleep(self.tick_sec)
        self.shutdown()

    def tick_safely(self) -> None:
        try:
            self.tick()
        except Exception:  # noqa: BLE001 - the daemon must outlive any single failure
            log.exception("scheduler tick failed")
            self.clock.sleep(5.0)

    def tick(self) -> None:
        now = self.clock.now()
        mono = self.clock.monotonic()
        self._heartbeat(now, mono)
        self._reap(mono)
        # the timetable only starts once the bootstrap is done, so its windows never "miss" while the first setup runs
        due = self.engine.due(now) if self._bootstrapped() else []
        self._dispatch(due, now, mono)
        self._maintain(now, mono, due)

    # ----------------------------------------------------------------------------------------- heartbeat
    def _heartbeat(self, now: datetime, mono: float) -> None:
        if mono - self._last["heartbeat"] < HEARTBEAT_EVERY_SEC:
            return
        self._last["heartbeat"] = mono
        status = f"running {self.running.label}" if self.running else "idle"
        # systemd first, then the file, the database last: a long VACUUM or bulk delete holds the SQLite writer lock, and a
        # healthy daemon must neither be killed by the watchdog nor look dead to the health check because of it
        self._notify(f"WATCHDOG=1\nSTATUS={status}")
        write_heartbeat_file(self.settings.data_dir, now)
        self._heartbeat_db(now, mono)

    def _heartbeat_db(self, now: datetime, mono: float) -> None:
        conn = self.store.conn
        try:
            wait = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.execute(f"PRAGMA busy_timeout = {HEARTBEAT_DB_WAIT_MS}")
        except sqlite3.Error:
            wait = None
        try:
            self.store.kv_set(HEARTBEAT_KEY, iso(now) or "")
        except sqlite3.OperationalError as exc:  # locked by a long write: the file heartbeat still proves liveness
            if mono - self._last["db_beat_warn"] > 300:
                self._last["db_beat_warn"] = mono
                log.warning("heartbeat not stored in the database (%s); the heartbeat file is still written", exc)
        finally:
            if wait is not None:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute(f"PRAGMA busy_timeout = {int(wait)}")

    def _register_start(self, now: datetime) -> None:
        """A daemon that finds the previous run still marked "running" was killed (out of memory, power loss, watchdog): systemd
        restarts it silently, so say so - once - instead of leaving the operator to discover gaps later."""
        previous = self.store.kv_get_json(RUN_STATE_KEY)
        self.store.kv_set_json(RUN_STATE_KEY, {"state": "running", "pid": os.getpid(), "since": iso(now)})
        if not (isinstance(previous, dict) and previous.get("state") == "running"):
            return
        last = max((b for b in (read_heartbeat_file(self.settings.data_dir), parse_ts(self.store.kv_get(HEARTBEAT_KEY)))
                    if b is not None), default=None)
        count = int(self.store.kv_get("daemon.unclean_restarts", "0") or 0) + 1
        self.store.kv_set("daemon.unclean_restarts", str(count))
        message = (f"the scheduler restarted after an unclean stop (process {previous.get('pid')} had been running since "
                   f"{previous.get('since')}; last heartbeat {iso(last) or 'unknown'}; {count} unclean restarts so far)")
        log.warning("%s", message)
        self.store.add_alert("warning", "daemon_restarted", message + ". See: python -m app.cli logs --role daemon",
                             dedupe_key="daemon-restart", dedupe_minutes=30)

    # ------------------------------------------------------------------------------------------- reaping
    def _reap(self, mono: float) -> None:
        run = self.running
        if run is None:
            return
        code = self.launcher.poll(run.handle)
        if code is None:
            if run.reason is None and mono - run.started > run.timeout_sec:
                self._terminate(run, "timeout", mono)
            elif run.kill_at is not None and mono >= run.kill_at:
                log.error("job %s ignored SIGTERM; killing it", run.label)
                self.launcher.kill(run.handle)
                run.kill_at = None
            return
        self._after_exit(run, code)
        self.running = None

    def _terminate(self, run: Running, reason: str, mono: float) -> None:
        self.store.kv_set(TERMINATE_REASON_KEY, reason)
        run.reason = reason
        run.kill_at = mono + TERMINATE_GRACE_SEC
        log.warning("terminating %s (%s)", run.label, reason)
        self.launcher.terminate(run.handle)

    def _after_exit(self, run: Running, code: int) -> None:
        pid = self.launcher.pid(run.handle)
        rows = self.store.job_runs_for(run.job, run.run_key)
        orphaned = [r for r in rows if r["status"] == "running" and (pid is None or r["pid"] in (None, pid))]
        if run.reason == "timeout":
            status, message = "timeout", f"timed out after {run.timeout_sec / 60:.0f} min"
        elif run.reason:
            status, message = "failed", f"{PREEMPTED}: {run.reason}"
        else:
            hint = " (killed: out of memory?)" if code in (137, -9) else ""
            status, message = "failed", f"process exited with code {code} before reporting a result{hint}"
        for row in orphaned:
            self.store.job_finish(int(row["run_id"]), status, message=message, exit_code=code)
        if not rows and code != 0:  # the child never got far enough to write its own row
            self.store.job_record(run.job, run.run_key, "failed", message=f"could not run (exit {code})", attempt=run.attempt)
        if orphaned and run.reason is None:
            self.store.add_alert("error", f"job_crashed_{run.job}", f"{run.label} [{run.run_key}] {message}",
                                 dedupe_key=f"crash:{run.job}:{run.run_key}", dedupe_minutes=24 * 60)
        log.info("job %s finished with exit code %s after %.0f s", run.label, code, self.clock.monotonic() - run.started)

    # ---------------------------------------------------------------------------------------- dispatching
    def _dispatch(self, due: list[Due], now: datetime, mono: float) -> None:
        if self.stop_requested:
            return
        if not self._bootstrapped():
            self._maybe_bootstrap(now, mono)
            return
        if not due:
            return
        head = due[0]
        if self.running is not None:
            if head.critical and not self.running.critical and self.running.reason is None:
                self._terminate(self.running, f"preempted by {head.entry.label}", mono)
            return
        for candidate in due:
            spec = JOBS[candidate.entry.job]
            # a heavy job that has to wait for RAM or disk must not hold back the light jobs behind it
            if spec.heavy and not candidate.critical and not (self._ram_ok(mono) and self._disk_ok(mono)):
                continue
            self._launch(candidate, mono)
            return

    def _launch(self, due: Due, mono: float) -> None:
        log.info("starting %s [%s] attempt %d/%d%s", due.entry.label, due.run_key, due.attempt, due.entry.max_attempts,
                 " (last)" if due.last_attempt else "")
        try:
            handle = self.launcher.start(due)
        except OSError as exc:
            log.error("cannot start %s: %s", due.entry.label, exc)
            self.store.job_record(due.entry.job, due.run_key, "failed", message=f"could not start: {exc}", attempt=due.attempt)
            return
        self.store.kv_delete(TERMINATE_REASON_KEY)
        self.running = Running(due.entry.label, due.entry.job, due.run_key, due.attempt, due.critical, handle, mono,
                               due.timeout_min * 60.0)

    def _disk_ok(self, mono: float) -> bool:
        """Heavy jobs (training, backfills) write a lot: with the disk at its critical level they wait for the emergency cleanup
        and the retention to make room instead of filling the last megabytes."""
        free = disk_info(self.settings.data_dir)["free_mb"]
        if free >= self.settings.min_free_disk_mb_crit:
            return True
        if mono - self._last["disk_warn"] > 600:
            self._last["disk_warn"] = mono
            log.warning("deferring heavy job: only %.0f MB of disk is free (critical below %d MB)", free,
                        self.settings.min_free_disk_mb_crit)
        return False

    def _ram_ok(self, mono: float) -> bool:
        available = memory_info()["available_mb"]
        if available is None or available >= self.settings.min_available_ram_mb:
            return True
        if mono - self._last["ram_warn"] > 600:
            self._last["ram_warn"] = mono
            log.warning("deferring heavy job: only %.0f MB RAM available", available)
        return False

    # --------------------------------------------------------------------------------------------- bootstrap
    def _bootstrapped(self) -> bool:
        return self.store.kv_get(BOOTSTRAP_STATE_KEY) == "done"

    def _maybe_bootstrap(self, now: datetime, mono: float) -> None:
        """First-run self-healing: (re)start the resumable bootstrap until it reports complete."""
        if self.running is not None or mono - self._last["bootstrap"] < BOOTSTRAP_RETRY_SEC:
            return
        self._last["bootstrap"] = mono
        attempts_today = sum(1 for r in self.store.recent_jobs(50)
                             if r["job_name"] == "bootstrap" and (r["started_at"] or "")[:10] == (iso(now) or "")[:10])
        if attempts_today >= BOOTSTRAP_MAX_ATTEMPTS_PER_DAY:
            return
        log.info("bootstrap incomplete; starting it (resumable)")
        key = (iso(now) or "")[:10]
        try:
            handle = self.launcher.start_named(["run-job", "bootstrap", "--key", key, "--attempt", str(attempts_today + 1),
                                                "--max-attempts", str(BOOTSTRAP_MAX_ATTEMPTS_PER_DAY), "--timeout-min", "480"])
        except OSError as exc:
            log.error("cannot start bootstrap: %s", exc)
            return
        self.running = Running("bootstrap", "bootstrap", key, attempts_today + 1, False, handle, mono, 480 * 60.0)

    # ------------------------------------------------------------------------------------- housekeeping
    def _maintain(self, now: datetime, mono: float, due: list[Due]) -> None:
        if mono - self._last["resources"] >= RESOURCE_CHECK_EVERY_SEC:
            self._last["resources"] = mono
            s = self.settings
            status = check_resources(s.data_dir, warn_disk_mb=s.min_free_disk_mb_warn, crit_disk_mb=s.min_free_disk_mb_crit,
                                     min_ram_mb=s.min_available_ram_mb)
            if status.level == "crit":
                emergency_cleanup(s.cache_dir, s.tmp_dir, s.log_dir)
                self.store.add_alert("error", "resources", "; ".join(status.messages), dedupe_key="resources:crit",
                                     dedupe_minutes=120)
        if mono - self._last["flush"] >= ALERT_FLUSH_EVERY_SEC and not self._critical_imminent(now, due):
            self._last["flush"] = mono
            self._flush_alerts()

    def _critical_imminent(self, now: datetime, due: list[Due]) -> bool:
        if any(d.critical for d in due):
            return True
        from app.clock import at_ist, ist_date

        today = ist_date(now)
        for entry in self.engine.entries:
            if JOBS[entry.job].critical:
                start = at_ist(today, entry.at)
                if -CRITICAL_QUIET_SEC <= (start - now).total_seconds() <= CRITICAL_QUIET_SEC:
                    return True
        return False

    def _flush_alerts(self) -> None:
        if not self.store.pending_alerts(1):
            return
        try:
            if self._notifier is None and self._notifier_factory is not None:
                self._notifier = self._notifier_factory()
            if self._notifier is not None:
                self._notifier.flush_alerts()
        except Exception:  # noqa: BLE001 - alert delivery must never disturb scheduling
            log.exception("alert flush failed")

    # ------------------------------------------------------------------------------------------ shutdown
    def request_stop(self) -> None:
        self.stop_requested = True

    def shutdown(self) -> None:
        run = self.running
        if run is not None:
            self._terminate(run, "service stopping", self.clock.monotonic())
            for _ in range(int(TERMINATE_GRACE_SEC)):
                if self.launcher.poll(run.handle) is not None:
                    break
                self.clock.sleep(1.0)
            else:
                self.launcher.kill(run.handle)
            self._after_exit(run, self.launcher.poll(run.handle) or 0)
            self.running = None
        self._notify("STOPPING=1")
        self.store.kv_set_json(RUN_STATE_KEY, {"state": "stopped", "pid": os.getpid(), "at": iso(self.clock.now())})
        log.info("scheduler stopped")


# ----------------------------------------------------------------------------------------------------- entry
def run_daemon(settings: Settings) -> int:
    """Run the scheduler until SIGTERM/SIGINT. Exactly one daemon per data directory."""
    lock = FileLock(lock_dir(settings.data_dir) / "daemon.lock")
    if not lock.acquire(blocking=False):
        log.error("another scheduler is already running (pid %s)", lock.holder_pid())
        return 3
    store = Store.open(settings.db_path)
    try:
        seed_calendar(store, settings)
        register_all(store)
        engine = ScheduleEngine(store, ExchangeCalendar(store))

        def make_notifier() -> Any:
            from notifications.service import NotificationService

            return NotificationService(settings, store)

        scheduler = Scheduler(settings, store, engine, SubprocessLauncher(settings), notifier_factory=make_notifier)

        def stop(_signum: int, _frame: Any) -> None:
            scheduler.request_stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, stop)
        scheduler.run_forever()
        return 0
    finally:
        store.close()
        lock.release()
