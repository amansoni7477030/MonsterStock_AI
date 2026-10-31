"""Ledger-based schedule engine: windows, retries, dependencies, catch-up, missed jobs and weekly runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from automation.calendar import ExchangeCalendar, seed_calendar
from automation.jobspec import JOBS
from automation.schedule import SCHEDULE, ScheduleEngine, ScheduleEntry, run_key_for

UTC = timezone.utc
IST_OFFSET = timedelta(hours=5, minutes=30)


def ist(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=UTC) - IST_OFFSET


MON = (2026, 11, 2)  # regular trading Monday
SAT = (2026, 10, 31)


@pytest.fixture()
def engine(store, settings, fake_clock):
    seed_calendar(store, settings)
    fake_clock.set(ist(*MON, 5, 0))
    eng = ScheduleEngine(store, ExchangeCalendar(store), fake_clock)
    eng.first_start()  # the scheduler "was installed" at 05:00 on Monday
    return eng


def labels(due):
    return [d.entry.label for d in due]


def test_every_scheduled_job_has_metadata_and_function():
    from automation.jobs import FUNCTIONS

    assert {e.job for e in SCHEDULE} <= set(JOBS) == set(FUNCTIONS)
    assert all(e.at < e.until for e in SCHEDULE)


def test_nothing_before_windows_then_first_job_due(engine, store):
    assert "collect_global" not in labels(engine.due(ist(*MON, 5, 59)))  # only the early housekeeping-style jobs are open
    due = [d for d in engine.due(ist(*MON, 6, 0, 1)) if d.entry.job == "collect_global"]
    assert len(due) == 1
    assert due[0].run_key == "2026-11-02" and due[0].attempt == 1 and not due[0].last_attempt


def test_retry_spacing_then_give_up_alerts_and_marks_missed(engine, store):
    t0 = ist(*MON, 6, 0, 1)
    for attempt in range(1, 7):
        store.job_record("collect_global", "2026-11-02", "failed", message=f"boom {attempt}")
        # right after a failure the job waits for the retry delay
        assert "collect_global" not in labels(engine.due(engine.clock.now()))
    engine.clock.set(t0 + timedelta(hours=2))
    assert "collect_global" not in labels(engine.due(engine.clock.now()))
    missed = store.job_runs_for("collect_global", "2026-11-02")[-1]
    assert missed["status"] == "missed" and "gave up" in missed["message"]
    assert any(a["code"] == "job_missed_collect_global" for a in store.pending_alerts(20))


def test_retry_after_delay_increments_attempt(engine, store):
    engine.clock.set(ist(*MON, 6, 1))
    store.job_record("collect_global", "2026-11-02", "failed", message="provider down")
    engine.clock.set(ist(*MON, 6, 5))
    assert "collect_global" not in labels(engine.due())  # 600 s retry delay not elapsed
    engine.clock.set(ist(*MON, 6, 20))
    nxt = [d for d in engine.due() if d.entry.job == "collect_global"][0]
    assert nxt.attempt == 2


def test_success_settles_the_run(engine, store):
    engine.clock.set(ist(*MON, 6, 1))
    store.job_record("collect_global", "2026-11-02", "success")
    assert "collect_global" not in labels(engine.due(ist(*MON, 8, 20)))


def test_window_closed_without_success_is_missed_not_retried(engine, store):
    now = ist(*MON, 8, 40)  # collect_global window ended 08:30
    due = engine.due(now)
    assert "collect_global" not in labels(due)
    rows = store.job_runs_for("collect_global", "2026-11-02")
    assert rows and rows[-1]["status"] == "missed"
    alert = next(a for a in store.pending_alerts(30) if a["code"] == "job_missed_collect_global")
    assert alert["level"] == "warning"
    assert engine.due(now) == [d for d in engine.due(now)]  # idempotent: no duplicate missed rows
    assert len([r for r in store.job_runs_for("collect_global", "2026-11-02") if r["status"] == "missed"]) == 1


def test_missed_critical_job_alerts_as_error(engine, store):
    engine.due(ist(*MON, 9, 20))
    alert = next(a for a in store.pending_alerts(50) if a["code"] == "job_missed_final_scan")
    assert alert["level"] == "error"


def test_dependency_blocks_until_prerequisite_is_terminal(engine, store):
    now = ist(*MON, 8, 31)
    assert "preflight" not in labels(engine.due(now))  # collect_india has not run
    store.job_record("collect_india", "2026-11-02", "success")
    assert "preflight" in labels(engine.due(now))


def test_dependency_released_when_prerequisite_window_closes(engine, store):
    store.job_record("collect_india", "2026-11-02", "failed", message="x")
    assert "preflight" not in labels(engine.due(ist(*MON, 8, 31)))


def test_critical_jobs_sort_first(engine, store):
    store.job_record("collect_global", "2026-11-02", "success")
    store.job_record("collect_india", "2026-11-02", "success")
    store.job_record("preflight", "2026-11-02", "success")
    for slot in ("early", "late"):
        store.job_record("collect_news", f"2026-11-02:{slot}", "success")
    store.job_record("health_check", "2026-11-02:morning", "success")
    due = engine.due(ist(*MON, 8, 45, 5))
    assert labels(due)[0] == "prelim_scan" and due[0].critical


def test_slots_get_independent_run_keys(engine):
    early = next(e for e in SCHEDULE if e.label == "collect_news#early")
    late = next(e for e in SCHEDULE if e.label == "collect_news#late")
    assert run_key_for(early, ist(*MON, 6, 0).date()) == "2026-11-02:early"
    assert run_key_for(late, ist(*MON, 6, 0).date()) == "2026-11-02:late"


def test_no_trading_jobs_on_weekend_but_backup_still_runs(engine, store):
    engine.clock.set(ist(*SAT, 6, 0))
    pipeline = {"collect_global", "collect_news", "collect_india", "preflight", "prelim_scan", "final_scan", "ingest_eod",
                "evaluate"}
    assert not pipeline & {d.entry.job for d in engine.due(ist(*SAT, 9, 10))}
    assert "backup" in labels(engine.due(ist(*SAT, 23, 31)))


def test_catch_up_runs_yesterdays_eod_chain_in_order(store, settings, fake_clock):
    seed_calendar(store, settings)
    fake_clock.set(ist(2026, 11, 2, 5, 0))
    eng = ScheduleEngine(store, ExchangeCalendar(store), fake_clock)
    store.kv_set("daemon.first_start", "2026-10-25T00:00:00Z")  # scheduler has existed for a while
    due = eng.due(ist(2026, 11, 3, 7, 0))  # Tuesday morning; Monday evening jobs never ran
    pairs = [(d.entry.job, d.run_key) for d in due]
    assert ("ingest_eod", "2026-11-02") in pairs
    assert ("evaluate", "2026-11-02") not in pairs  # waits for ingest_eod
    store.job_record("ingest_eod", "2026-11-02", "success")
    pairs = [(d.entry.job, d.run_key) for d in eng.due(ist(2026, 11, 3, 7, 1))]
    assert ("evaluate", "2026-11-02") in pairs


def test_weekly_job_runs_once_per_iso_week_with_catch_up(store, settings, fake_clock):
    seed_calendar(store, settings)
    eng = ScheduleEngine(store, ExchangeCalendar(store), fake_clock)
    store.kv_set("daemon.first_start", "2026-10-01T00:00:00Z")
    friday = eng.due(ist(2026, 10, 30, 3, 0))
    assert "train" not in labels(friday)
    saturday = [d for d in eng.due(ist(2026, 10, 31, 2, 5)) if d.entry.job == "train"]
    assert saturday and saturday[0].run_key == "2026-W44"
    sunday = [d for d in eng.due(ist(2026, 11, 1, 15, 0)) if d.entry.job == "train"]  # missed Saturday -> still due
    assert sunday and sunday[0].run_key == "2026-W44"
    store.job_record("train", "2026-W44", "success")
    assert "train" not in labels(eng.due(ist(2026, 11, 1, 16, 0)))
    assert "train" not in labels(eng.due(ist(2026, 11, 2, 3, 0)))  # Monday starts a new week but train is Saturday's job


def test_installed_after_the_window_is_skipped_silently(store, settings, fake_clock):
    seed_calendar(store, settings)
    fake_clock.set(ist(*MON, 11, 0))
    eng = ScheduleEngine(store, ExchangeCalendar(store), fake_clock)
    eng.first_start()  # first ever start at 11:00 on Monday
    assert "prelim_scan" not in labels(eng.due(ist(*MON, 11, 1)))
    rows = store.job_runs_for("prelim_scan", "2026-11-02")
    assert rows and rows[0]["status"] == "skipped"
    assert not any(a["code"].startswith("job_missed") for a in store.pending_alerts(50))


def test_preempted_attempts_do_not_count(engine, store):
    for _ in range(4):
        store.job_record("collect_global", "2026-11-02", "failed", message="preempted: preempted by prelim_scan")
    engine.clock.set(ist(*MON, 7, 30))
    nxt = [d for d in engine.due() if d.entry.job == "collect_global"]
    assert nxt and nxt[0].attempt == 1


def test_last_attempt_flag(engine, store):
    for minute in (45, 48, 51, 54):  # four failed attempts, each at its own time
        engine.clock.set(ist(*MON, 8, minute))
        store.job_record("prelim_scan", "2026-11-02", "failed", message="x")
    engine.clock.set(ist(*MON, 8, 58))
    due = [d for d in engine.due() if d.entry.job == "prelim_scan"]
    assert due and due[0].attempt == 5 and due[0].last_attempt
    store.job_record("prelim_scan", "2026-11-02", "failed", message="x")
    engine.clock.set(ist(*MON, 9, 1))
    assert not [d for d in engine.due() if d.entry.job == "prelim_scan"]  # exhausted -> missed
    assert store.job_runs_for("prelim_scan", "2026-11-02")[-1]["status"] == "missed"


def test_recover_stale_marks_dead_runner_failed(engine, store):
    run_id = store.job_start("collect_global", "2026-11-02", pid=999999)
    assert engine.recover_stale(alive=lambda pid: False) == 1
    row = store.job_runs_for("collect_global", "2026-11-02")[0]
    assert row["run_id"] == run_id and row["status"] == "failed" and "recovered" in row["message"]
    run2 = store.job_start("collect_india", "2026-11-02", pid=4242)
    assert engine.recover_stale(alive=lambda pid: True) == 0
    assert store.job_runs_for("collect_india", "2026-11-02")[0]["run_id"] == run2


def test_running_job_is_not_started_twice(engine, store):
    store.job_start("collect_global", "2026-11-02", pid=1)
    assert "collect_global" not in labels(engine.due(ist(*MON, 6, 30)))


def test_custom_schedule_entries_are_supported(store, settings, fake_clock):
    seed_calendar(store, settings)
    entry = ScheduleEntry("backup", "10:00", "10:30", days="any", max_attempts=2)
    eng = ScheduleEngine(store, ExchangeCalendar(store), fake_clock, entries=(entry,))
    store.kv_set("daemon.first_start", "2026-10-01T00:00:00Z")
    assert labels(eng.due(ist(*MON, 10, 5))) == ["backup"]
