"""Scheduler test doubles shared by the scheduler test modules: a fake launcher/handle and a ready-to-tick scheduler environment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from automation import scheduler as sched
from automation.calendar import ExchangeCalendar, seed_calendar
from automation.schedule import ScheduleEngine

UTC = timezone.utc


def ist(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=UTC) - timedelta(hours=5, minutes=30)


class Handle:
    def __init__(self, due, pid):
        self.due, self.pid, self.code, self.terminated, self.killed = due, pid, None, False, False


class FakeLauncher:
    def __init__(self, store):
        self.store = store
        self.started: list[Handle] = []
        self.named: list[list[str]] = []
        self._pid = 1000

    def start(self, due):
        self._pid += 1
        handle = Handle(due, self._pid)
        self.started.append(handle)
        return handle

    def start_named(self, args):
        self._pid += 1
        self.named.append(args)
        return Handle(None, self._pid)

    def poll(self, handle):
        return handle.code

    def pid(self, handle):
        return handle.pid

    def terminate(self, handle):
        handle.terminated = True

    def kill(self, handle):
        handle.killed = True
        handle.code = -9

    # helpers used by tests
    def finish_ok(self, handle, status="success"):
        """Behave like the child process: write the ledger row, then exit 0."""
        due = handle.due
        run_id = self.store.job_start(due.entry.job, due.run_key, attempt=due.attempt, pid=handle.pid)
        self.store.job_finish(run_id, status, message="done", exit_code=0)
        handle.code = 0


def make_env(store, settings, fake_clock):
    """A scheduler on a bootstrapped database at Monday 05:00 IST, driven by a fake launcher and a recording notifier."""
    seed_calendar(store, settings)
    store.kv_set("bootstrap.state", "done")
    fake_clock.set(ist(2026, 11, 2, 5, 0))
    engine = ScheduleEngine(store, ExchangeCalendar(store), fake_clock)
    engine.first_start()
    launcher = FakeLauncher(store)
    notes: list[str] = []
    s = sched.Scheduler(settings, store, engine, launcher, clock=fake_clock, notify=lambda m: notes.append(m) or True, tick_sec=2.0)
    return type("Env", (), {"s": s, "launcher": launcher, "store": store, "clock": fake_clock, "notes": notes, "engine": engine})
