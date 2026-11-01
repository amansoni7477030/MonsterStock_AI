"""Scheduler details the main tests skip: systemd notification, POSIX pid checks, the real launcher's command line, RAM deferral,
start failures, bootstrap attempt caps, resource emergencies, failing alert flushes and the ``run_daemon`` entry point."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.resources import ResourceStatus
from automation import scheduler as sched
from automation.schedule import Due, ScheduleEntry
from tests.scheduler_support import FakeLauncher, Handle, ist, make_env


@pytest.fixture()
def env(store, settings, fake_clock):
    return make_env(store, settings, fake_clock)


# --------------------------------------------------------------------------------------- systemd notify
class FakeSocket:
    instances: list[FakeSocket] = []
    fail_connect = False

    def __init__(self, *args):
        self.args, self.sent, self.address = args, [], None
        FakeSocket.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def connect(self, address):
        if FakeSocket.fail_connect:
            raise OSError("no such socket")
        self.address = address

    def sendall(self, data):
        self.sent.append(data)


@pytest.fixture()
def fake_socket(monkeypatch):
    FakeSocket.instances, FakeSocket.fail_connect = [], False
    # a stand-in module: some platforms (e.g. a Windows development machine) have no AF_UNIX at all
    monkeypatch.setattr(sched, "socket", SimpleNamespace(AF_UNIX=1, SOCK_DGRAM=2, socket=FakeSocket))
    return FakeSocket


def test_sd_notify_sends_to_the_unix_socket_and_understands_abstract_addresses(monkeypatch, fake_socket):
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    assert sched.sd_notify("READY=1") is True
    assert fake_socket.instances[0].address == "/run/systemd/notify" and fake_socket.instances[0].sent == [b"READY=1"]
    monkeypatch.setenv("NOTIFY_SOCKET", "@abstract-name")
    assert sched.sd_notify("WATCHDOG=1") is True and fake_socket.instances[1].address == "\0abstract-name"


def test_sd_notify_never_raises_when_the_socket_is_gone(monkeypatch, fake_socket):
    monkeypatch.setenv("NOTIFY_SOCKET", "/run/systemd/notify")
    fake_socket.fail_connect = True
    assert sched.sd_notify("READY=1") is False


# ------------------------------------------------------------------------------------------ pid_alive
def test_the_posix_pid_check_distinguishes_dead_foreign_and_live_processes(monkeypatch):
    outcomes = {}

    def kill(pid, sig):
        result = outcomes[pid]
        if isinstance(result, BaseException):
            raise result

    monkeypatch.setattr(sched, "os", SimpleNamespace(name="posix", kill=kill, environ={}))
    outcomes.update({10: ProcessLookupError(), 11: PermissionError(), 12: None})
    assert sched.pid_alive(10) is False  # gone
    assert sched.pid_alive(11) is True  # exists but belongs to another user
    assert sched.pid_alive(12) is True and sched.pid_alive("12") is True
    assert sched.pid_alive(0) is False and sched.pid_alive(None) is False


# ------------------------------------------------------------------------------------------- launcher
class FakePopen:
    calls: list[tuple[list[str], dict]] = []

    def __init__(self, cmd, **kwargs):
        FakePopen.calls.append((cmd, kwargs))
        self.pid, self.code, self.terminated, self.killed = 4242, None, False, False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_the_launcher_builds_the_exact_command_line_the_systemd_units_rely_on(settings, monkeypatch):
    FakePopen.calls = []
    monkeypatch.setattr(sched.subprocess, "Popen", FakePopen)
    launcher = sched.SubprocessLauncher(settings)
    entry = ScheduleEntry("final_scan", "09:08:45", "09:13:30", retry_sec=45, max_attempts=3)
    due = Due(entry=entry, run_key="2026-11-02", day=ist(2026, 11, 2, 9, 9).date(), attempt=2, last_attempt=False)
    handle = launcher.start(due)
    cmd, kwargs = FakePopen.calls[0]
    assert cmd[:6] == [sys.executable, "-m", "app.cli", "run-job", "final_scan", "--key"] and "--last-attempt" not in cmd
    assert cmd[cmd.index("--attempt") + 1] == "2" and cmd[cmd.index("--max-attempts") + 1] == "3"
    assert cmd[cmd.index("--timeout-min") + 1] == str(due.timeout_min)
    assert kwargs["cwd"] == str(settings.home) and kwargs["stdin"] == subprocess.DEVNULL and "shell" not in kwargs

    last = Due(entry=entry, run_key="2026-11-02", day=due.day, attempt=3, last_attempt=True)
    launcher.start(last)
    assert FakePopen.calls[1][0][-1] == "--last-attempt"

    launcher.start_named(["run-job", "bootstrap", "--key", "2026-11-02"])
    assert FakePopen.calls[2][0] == [sys.executable, "-m", "app.cli", "run-job", "bootstrap", "--key", "2026-11-02"]
    assert launcher.poll(handle) is None and launcher.pid(handle) == 4242
    launcher.terminate(handle)
    launcher.kill(handle)
    assert handle.terminated and handle.killed


def test_terminating_an_already_dead_child_is_harmless(settings):
    class Dead:
        def terminate(self):
            raise OSError("no such process")

        kill = terminate

    sched.SubprocessLauncher(settings).terminate(Dead())
    sched.SubprocessLauncher(settings).kill(Dead())


# --------------------------------------------------------------------------------------------- dispatch
def test_heavy_jobs_wait_for_memory_but_critical_ones_never_do(env, monkeypatch):
    monkeypatch.setattr(sched, "memory_info", lambda: {"available_mb": 10.0})
    env.clock.set(ist(2026, 11, 1, 4, 5))  # Sunday: refresh_universe (heavy) is the only job due at this time of day
    env.s.tick()
    assert env.launcher.started == [] and env.s._last["ram_warn"] > -1e9  # deferred, and said so once
    warned_at = env.s._last["ram_warn"]
    env.clock.advance(5)
    env.s.tick()
    assert env.s._last["ram_warn"] == warned_at  # the warning is rate limited

    monkeypatch.setattr(sched, "memory_info", lambda: {"available_mb": 4000.0})
    env.clock.advance(5)
    env.s.tick()
    assert [h.due.entry.job for h in env.launcher.started] == ["train"]  # Saturday's retraining, caught up on Sunday
    assert sched.JOBS["train"].heavy


def test_unknown_available_memory_does_not_block_jobs(env, monkeypatch):
    monkeypatch.setattr(sched, "memory_info", lambda: {"available_mb": None})
    assert env.s._ram_ok(0.0) is True


def test_heavy_jobs_wait_for_disk_space_but_critical_ones_never_do(env, monkeypatch):
    monkeypatch.setattr(sched, "memory_info", lambda: {"available_mb": 4000.0})
    monkeypatch.setattr(sched, "disk_info", lambda path: {"free_mb": 100.0, "total_mb": 8000.0, "used_mb": 7900.0})
    env.clock.set(ist(2026, 11, 1, 4, 5))  # Sunday: the caught-up weekly training (heavy) is the only job due
    env.s.tick()
    assert env.launcher.started == [] and env.s._last["disk_warn"] > -1e9  # deferred, and said so
    warned_at = env.s._last["disk_warn"]
    env.clock.advance(5)
    env.s.tick()
    assert env.s._last["disk_warn"] == warned_at  # the warning is rate limited
    monkeypatch.setattr(sched, "disk_info", lambda path: {"free_mb": 5000.0, "total_mb": 8000.0, "used_mb": 3000.0})
    env.clock.advance(5)
    env.s.tick()
    assert [h.due.entry.job for h in env.launcher.started] == ["train"]


# ----------------------------------------------------------------------- heartbeat vs. a busy database
def test_the_watchdog_and_the_heartbeat_file_survive_a_locked_database(env, monkeypatch, caplog):
    import logging
    import sqlite3

    from automation.health import HEARTBEAT_KEY, read_heartbeat_file

    real = env.store.kv_set

    def locked(key, value):
        if key == HEARTBEAT_KEY:
            raise sqlite3.OperationalError("database is locked")  # a VACUUM holds the writer lock
        return real(key, value)

    monkeypatch.setattr(env.store, "kv_set", locked)
    with caplog.at_level(logging.WARNING, logger="automation.scheduler"):
        env.s.tick()  # must not raise
        env.clock.advance(sched.HEARTBEAT_EVERY_SEC + 1)
        env.s.tick()
    assert sum(note.startswith("WATCHDOG=1") for note in env.notes) == 2  # systemd heard the daemon both times
    assert read_heartbeat_file(env.s.settings.data_dir) == env.clock.now()
    assert sum("heartbeat not stored in the database" in r.message for r in caplog.records) == 1  # rate limited
    assert env.store.kv_get(HEARTBEAT_KEY) is None


def test_the_database_heartbeat_never_waits_the_full_busy_timeout(env, monkeypatch):
    from automation.health import HEARTBEAT_KEY

    seen = {}
    real = env.store.kv_set

    def spy(key, value):
        if key == HEARTBEAT_KEY:
            seen["during"] = env.store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        return real(key, value)

    before = env.store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    monkeypatch.setattr(env.store, "kv_set", spy)
    env.s.tick()
    assert seen["during"] == sched.HEARTBEAT_DB_WAIT_MS < before
    assert env.store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == before  # restored for everything else
    assert env.store.kv_get(HEARTBEAT_KEY) == "2026-11-01T23:30:00Z"  # Monday 05:00 IST


# ------------------------------------------------------------------------------------ unclean restarts
def test_an_unclean_stop_is_reported_once_and_a_clean_one_is_not(env):
    s, store = env.s, env.store
    s._register_start(env.clock.now())
    assert store.pending_alerts(10) == [] and store.kv_get_json(sched.RUN_STATE_KEY)["state"] == "running"  # first ever start

    env.clock.advance(3600)
    s._register_start(env.clock.now())  # the previous process never marked itself stopped
    alerts = [a for a in store.pending_alerts(10) if a["code"] == "daemon_restarted"]
    assert len(alerts) == 1 and "unclean stop" in alerts[0]["message"] and "1 unclean restarts" in alerts[0]["message"]
    assert store.kv_get("daemon.unclean_restarts") == "1"

    env.clock.advance(60)
    s._register_start(env.clock.now())  # a crash loop: counted, but the alert is deduplicated
    assert store.kv_get("daemon.unclean_restarts") == "2"
    assert len([a for a in store.pending_alerts(10) if a["code"] == "daemon_restarted"]) == 1

    s.stop_requested = True
    s.run_forever()  # starts (running -> alert path), loops zero times, shuts down cleanly
    assert store.kv_get_json(sched.RUN_STATE_KEY)["state"] == "stopped"
    store.conn.execute("DELETE FROM alerts")
    s._register_start(env.clock.now())  # after a clean stop nothing is reported
    assert store.pending_alerts(10) == []
    assert "READY=1" in env.notes and "STOPPING=1" in env.notes


def test_a_job_that_cannot_be_started_is_recorded_as_failed(env):
    class Broken(FakeLauncher):
        def start(self, due):
            raise OSError("Cannot allocate memory")

    env.s.launcher = Broken(env.store)
    env.clock.set(ist(2026, 11, 2, 6, 0, 3))
    env.s.tick()
    assert env.s.running is None
    row = env.store.recent_jobs(1)[0]
    assert row["status"] == "failed" and "could not start: Cannot allocate memory" in row["message"]


def test_bootstrap_is_not_restarted_forever_and_a_start_failure_is_tolerated(env):
    env.store.kv_set("bootstrap.state", "failed")
    env.clock.set(ist(2026, 11, 2, 6, 0, 0))
    today = env.clock.now().strftime("%Y-%m-%d")
    for attempt in range(sched.BOOTSTRAP_MAX_ATTEMPTS_PER_DAY):
        env.store.job_record("bootstrap", today, "failed", message="x", attempt=attempt + 1)
    env.s.tick()
    assert env.launcher.named == [] and env.s.running is None  # six failures today: wait for tomorrow

    class Broken(FakeLauncher):
        def start_named(self, args):
            raise OSError("fork failed")

    env.store.conn.execute("DELETE FROM job_runs")
    env.s.launcher = Broken(env.store)
    env.clock.advance(sched.BOOTSTRAP_RETRY_SEC + 1)
    env.s.tick()
    assert env.s.running is None


# ----------------------------------------------------------------------------------------- maintenance
def test_a_critical_disk_triggers_an_emergency_cleanup_and_a_deduplicated_alert(env, monkeypatch):
    cleaned = []
    status = ResourceStatus(level="crit", messages=["only 200 MB free"], free_disk_mb=200.0, available_ram_mb=500.0)
    monkeypatch.setattr(sched, "check_resources", lambda *a, **k: status)
    monkeypatch.setattr(sched, "emergency_cleanup", lambda *a: cleaned.append(a) or {"removed": 1})
    env.clock.set(ist(2026, 11, 2, 12, 0))
    env.s.tick()
    env.clock.advance(sched.RESOURCE_CHECK_EVERY_SEC + 1)
    env.s.tick()
    assert len(cleaned) == 2
    alerts = [a for a in env.store.pending_alerts(10) if a["code"] == "resources"]
    assert len(alerts) == 1 and "only 200 MB free" in alerts[0]["message"]  # same alert inside the dedupe window


def test_a_failing_notifier_never_disturbs_scheduling(env):
    class Exploding:
        def flush_alerts(self):
            raise RuntimeError("telegram is down")

    env.s._notifier_factory = Exploding
    env.store.add_alert("warning", "x", "pending")
    env.clock.set(ist(2026, 11, 2, 12, 0))
    env.s.tick()  # must not raise
    env.clock.advance(sched.ALERT_FLUSH_EVERY_SEC + 1)
    env.s.tick()
    assert env.store.pending_alerts(5)  # still queued for the next attempt


def test_nothing_is_flushed_when_there_is_nothing_to_send(env):
    class Notifier:
        def flush_alerts(self):
            pytest.fail("no pending alerts")

    env.s._notifier_factory = Notifier
    env.clock.set(ist(2026, 11, 2, 5, 0, 10))  # right after the daemon started: no window has been missed, so no alert exists
    env.s.tick()


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace_period(env):
    handle = Handle(None, 91)
    env.s.running = sched.Running("collect_global", "collect_global", "2026-11-02", 1, False, handle, 0.0, 3600.0,
                                  reason="preempted by prelim_scan", kill_at=env.clock.monotonic() + 5.0)
    env.s.tick()
    assert not handle.killed  # inside the grace period
    env.clock.advance(6)
    env.s.tick()
    assert handle.killed


def test_a_stuck_child_at_shutdown_is_killed(env):
    handle = Handle(None, 92)
    env.s.running = sched.Running("collect_global", "collect_global", "2026-11-02", 1, False, handle, 0.0, 3600.0)
    env.store.job_start("collect_global", "2026-11-02", pid=92)
    env.s.shutdown()  # poll() stays None for the whole grace period
    assert handle.terminated and handle.killed and env.s.running is None


def test_the_run_loop_recovers_stale_rows_announces_readiness_and_stops_on_request(env):
    env.store.job_start("collect_global", "2026-11-02", pid=424242)  # left "running" by a previous daemon that died
    ticks = []

    def sleep(seconds):
        ticks.append(seconds)
        if len(ticks) == 2:
            env.s.request_stop()

    env.clock.sleep = sleep
    env.s.run_forever()
    assert "READY=1" in env.notes and "STOPPING=1" in env.notes and len(ticks) == 2
    assert env.store.job_runs_for("collect_global", "2026-11-02")[0]["status"] != "running"


# -------------------------------------------------------------------------------------------- run_daemon
def test_run_daemon_refuses_a_second_instance(settings):
    from app.locks import FileLock, lock_dir

    held = FileLock(lock_dir(settings.data_dir) / "daemon.lock")
    assert held.acquire(blocking=False)
    try:
        assert sched.run_daemon(settings) == 3
    finally:
        held.release()


def test_run_daemon_wires_everything_installs_signal_handlers_and_releases_the_lock(settings, monkeypatch):
    seen = {}

    class FakeScheduler:
        def __init__(self, cfg, store, engine, launcher, notifier_factory=None, **kw):
            seen.update(store=store, launcher=type(launcher).__name__, notifier=notifier_factory())
            self.stopped = False

        def request_stop(self):
            self.stopped = True

        def run_forever(self):
            seen["handlers"] = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
            seen["handlers"][signal.SIGTERM](signal.SIGTERM, None)
            seen["stopped"] = self.stopped

    monkeypatch.setattr(sched, "Scheduler", FakeScheduler)
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert sched.run_daemon(settings) == 0
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    assert seen["launcher"] == "SubprocessLauncher" and seen["stopped"] is True
    assert type(seen["notifier"]).__name__ == "NotificationService"
    assert Path(settings.db_path).exists()
    from app.locks import FileLock, lock_dir

    again = FileLock(lock_dir(settings.data_dir) / "daemon.lock")
    assert again.acquire(blocking=False)  # the daemon released it on exit
    again.release()
