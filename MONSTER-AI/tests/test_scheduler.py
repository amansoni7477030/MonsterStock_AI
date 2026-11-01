"""Scheduler daemon loop with a fake launcher: dispatch, preemption, timeouts, crash recovery, bootstrap gating, shutdown."""

from __future__ import annotations

import os

import pytest

from automation import scheduler as sched
from automation.health import HEARTBEAT_KEY
from tests.scheduler_support import Handle, ist, make_env


@pytest.fixture()
def env(store, settings, fake_clock):
    return make_env(store, settings, fake_clock)


def test_tick_starts_the_due_job_and_sends_heartbeat(env):
    env.clock.set(ist(2026, 11, 2, 6, 0, 3))
    env.s.tick()
    assert [h.due.entry.job for h in env.launcher.started][0] in {"model_sync", "refresh_calendar"}  # earliest window first
    assert len(env.launcher.started) == 1
    assert env.store.kv_get(HEARTBEAT_KEY)
    assert any(n.startswith("WATCHDOG=1") for n in env.notes)


def test_only_one_job_runs_at_a_time(env):
    env.clock.set(ist(2026, 11, 2, 6, 0, 3))
    env.s.tick()
    first = env.launcher.started[0]
    env.s.tick()
    assert len(env.launcher.started) == 1  # first still running
    env.launcher.finish_ok(first)
    env.clock.advance(2)
    env.s.tick()  # reaps and starts the next
    assert len(env.launcher.started) == 2
    assert env.launcher.started[1].due.entry.job != first.due.entry.job


def test_critical_job_preempts_a_running_noncritical_job(env):
    env.clock.set(ist(2026, 11, 2, 8, 44, 50))
    # a long collection job is running when the preliminary scan becomes due
    env.store.job_record("collect_global", "2026-11-02", "success")
    env.store.job_record("collect_india", "2026-11-02", "success")
    env.store.job_record("preflight", "2026-11-02", "success")
    for slot in ("early", "late"):
        env.store.job_record("collect_news", f"2026-11-02:{slot}", "success")
    env.store.job_record("health_check", "2026-11-02:morning", "success")
    env.store.job_record("model_sync", "2026-11-02", "success")
    env.store.job_record("refresh_calendar", "2026-11-02", "success")
    env.store.job_record("housekeeping", "2026-11-01", "success")
    env.store.job_record("backup", "2026-11-01", "success")
    env.store.kv_set("daemon.first_start", "2026-10-01T00:00:00Z")
    env.store.job_record("train", "2026-W44", "success")
    env.store.job_record("refresh_universe", "2026-W44", "success")
    env.s.running = sched.Running("collect_global", "collect_global", "2026-11-02", 1, False, Handle(None, 55), 0.0, 3600.0)
    env.clock.set(ist(2026, 11, 2, 8, 45, 5))
    env.s.tick()
    run = env.s.running
    assert run is not None and run.reason and run.reason.startswith("preempted by prelim_scan")
    assert run.handle.terminated
    run.handle.code = -15  # the child exits after SIGTERM
    env.store.job_start("collect_global", "2026-11-02", pid=55)  # row it left behind
    env.clock.advance(2)
    env.s.tick()
    rows = env.store.job_runs_for("collect_global", "2026-11-02")
    assert rows[-1]["status"] == "failed" and rows[-1]["message"].startswith("preempted")
    assert env.launcher.started and env.launcher.started[-1].due.entry.job == "prelim_scan"


def test_timeout_terminates_then_kills_and_records_timeout(env):
    env.clock.set(ist(2026, 11, 2, 6, 0, 3))
    env.s.running = sched.Running("collect_global", "collect_global", "2026-11-02", 1, False, Handle(None, 77),
                                  env.clock.monotonic(), 60.0)
    env.store.job_start("collect_global", "2026-11-02", pid=77)
    env.clock.advance(61)
    env.s.tick()
    assert env.s.running.reason == "timeout" and env.s.running.handle.terminated
    env.clock.advance(sched.TERMINATE_GRACE_SEC + 1)
    env.s.tick()
    assert env.s.running.handle.killed
    env.s.tick()  # reaps the killed process
    row = env.store.job_runs_for("collect_global", "2026-11-02")[-1]
    assert row["status"] == "timeout" and "timed out" in row["message"]


def test_crashed_child_is_recorded_with_oom_hint_and_alert(env):
    handle = Handle(None, 88)
    env.s.running = sched.Running("collect_global", "collect_global", "2026-11-02", 1, False, handle, 0.0, 3600.0)
    env.store.job_start("collect_global", "2026-11-02", pid=88)
    handle.code = 137
    env.s.tick()
    row = env.store.job_runs_for("collect_global", "2026-11-02")[-1]
    assert row["status"] == "failed" and "out of memory" in row["message"] and row["exit_code"] == 137
    assert any(a["code"] == "job_crashed_collect_global" for a in env.store.pending_alerts(10))


def test_child_that_never_started_is_recorded(env):
    handle = Handle(None, 89)
    env.s.running = sched.Running("collect_global", "collect_global", "2026-11-02", 2, False, handle, 0.0, 3600.0)
    handle.code = 1
    env.s.tick()
    rows = env.store.job_runs_for("collect_global", "2026-11-02")
    assert rows and rows[-1]["status"] == "failed" and "could not run" in rows[-1]["message"]


def test_bootstrap_gating_and_self_launch(env):
    env.store.kv_set("bootstrap.state", "failed")
    env.clock.set(ist(2026, 11, 2, 6, 0, 3))
    env.s.tick()
    assert env.launcher.started == []  # the timetable is held back
    assert env.launcher.named and env.launcher.named[0][:2] == ["run-job", "bootstrap"]
    handle_args = env.launcher.named[0]
    assert "--timeout-min" in handle_args
    env.s.running.handle.code = 1
    env.clock.advance(60)
    env.s.tick()
    assert len(env.launcher.named) == 1  # retried only after the back-off
    env.clock.advance(sched.BOOTSTRAP_RETRY_SEC)
    env.s.tick()
    assert len(env.launcher.named) == 2


def test_tick_survives_internal_errors(env, monkeypatch):
    monkeypatch.setattr(env.engine, "due", lambda now=None: (_ for _ in ()).throw(RuntimeError("db locked")))
    env.s.tick_safely()  # must not raise
    monkeypatch.undo()


def test_shutdown_terminates_running_child(env):
    handle = Handle(None, 99)
    env.s.running = sched.Running("collect_global", "collect_global", "2026-11-02", 1, False, handle, 0.0, 3600.0)
    env.store.job_start("collect_global", "2026-11-02", pid=99)
    handle.code = None
    env.s.request_stop()
    original_poll = env.launcher.poll
    env.launcher.poll = lambda h: (setattr(h, "code", 0) or 0) if h.terminated else original_poll(h)
    env.s.shutdown()
    assert handle.terminated and env.s.running is None
    assert "STOPPING=1" in env.notes
    assert env.store.job_runs_for("collect_global", "2026-11-02")[-1]["message"].startswith("preempted: service stopping")


def test_alerts_are_flushed_through_the_notifier(env):
    class Notifier:
        calls = 0

        def flush_alerts(self):
            Notifier.calls += 1

    env.s._notifier_factory = Notifier
    env.store.add_alert("warning", "x", "something happened")
    env.clock.set(ist(2026, 11, 2, 12, 0))
    env.s.tick()
    assert Notifier.calls == 1


def test_alert_flush_is_postponed_near_a_critical_job(env):
    class Notifier:
        calls = 0

        def flush_alerts(self):
            Notifier.calls += 1

    env.s._notifier_factory = Notifier
    env.store.add_alert("warning", "x", "something happened")
    env.clock.set(ist(2026, 11, 2, 9, 8, 0))  # 45 s before the final scan window opens
    env.s.tick()
    assert Notifier.calls == 0


def test_sd_notify_is_a_noop_without_a_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert sched.sd_notify("READY=1") is False


def test_pid_alive_detects_this_process_and_a_missing_one():
    assert sched.pid_alive(os.getpid()) is True
    assert sched.pid_alive(None) is False
    assert sched.pid_alive(0) is False
