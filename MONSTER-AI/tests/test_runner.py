"""Job runner: ledger entries, locks, termination handling and failure alerts (no real jobs are executed)."""

from __future__ import annotations

import pytest

from app.locks import heavy_lock, job_lock
from automation import jobs as jobs_module
from automation import runner
from automation.context import JobResult


@pytest.fixture()
def fake_jobs(monkeypatch):
    table = {}
    monkeypatch.setattr(jobs_module, "FUNCTIONS", table)
    return table


def run(settings, store, fake_clock, name="backup", **kw):
    return runner.execute_job(settings, name, "2026-11-02", store=store, clock=fake_clock, **kw)


def test_success_is_recorded_with_message(settings, store, fake_clock, fake_jobs):
    fake_jobs["backup"] = lambda ctx: JobResult("success", "ok: 3 rows", {"rows": 3})
    result = run(settings, store, fake_clock)
    assert result.status == "success"
    row = store.job_runs_for("backup", "2026-11-02")[0]
    assert row["status"] == "success" and row["exit_code"] == 0 and row["message"] == "ok: 3 rows"
    assert row["finished_at"] and row["pid"]


def test_exception_is_recorded_redacted_and_alerts_on_last_attempt(settings, store, fake_clock, fake_jobs):
    def boom(ctx):
        raise RuntimeError("failed calling https://x.example/?api_key=SUPERSECRET123456 token=abcdef1234567890")

    fake_jobs["backup"] = boom
    result = run(settings, store, fake_clock, attempt=2, max_attempts=2, last_attempt=True)
    assert result.status == "failed"
    row = store.job_runs_for("backup", "2026-11-02")[0]
    assert row["status"] == "failed" and row["exit_code"] == 1
    assert "SUPERSECRET123456" not in row["message"] and row["message"].startswith("RuntimeError")
    assert any(a["code"] == "job_failed_backup" for a in store.pending_alerts(10))


def test_failure_before_last_attempt_does_not_alert(settings, store, fake_clock, fake_jobs):
    fake_jobs["backup"] = lambda ctx: JobResult("failed", "provider down")
    run(settings, store, fake_clock, attempt=1, max_attempts=3, last_attempt=False)
    assert store.pending_alerts(10) == []


def test_invalid_status_is_converted_to_failure(settings, store, fake_clock, fake_jobs):
    fake_jobs["backup"] = lambda ctx: JobResult("great", "??")
    assert run(settings, store, fake_clock).status == "failed"


def test_sigterm_reason_decides_timeout_or_preempted(settings, store, fake_clock, fake_jobs):
    def terminated(ctx):
        raise runner.Terminated()

    fake_jobs["backup"] = terminated
    store.kv_set(runner.TERMINATE_REASON_KEY, "timeout")
    assert run(settings, store, fake_clock, timeout_min=7).status == "timeout"
    assert "7 min" in store.job_runs_for("backup", "2026-11-02")[0]["message"]

    store.kv_set(runner.TERMINATE_REASON_KEY, "preempted by final_scan")
    assert run(settings, store, fake_clock).status == "failed"
    last = store.job_runs_for("backup", "2026-11-02")[-1]
    assert last["message"].startswith("preempted")


def test_second_instance_is_skipped_without_ledger_row(settings, store, fake_clock, fake_jobs):
    fake_jobs["backup"] = lambda ctx: JobResult("success", "ran")
    held = job_lock(settings.data_dir, "backup")
    assert held.acquire()
    try:
        result = run(settings, store, fake_clock)
    finally:
        held.release()
    assert result.status == "skipped" and "already running" in result.message
    assert store.job_runs_for("backup", "2026-11-02") == []


def test_heavy_job_waits_for_the_heavy_lock_then_gives_up(settings, store, fake_clock, fake_jobs, monkeypatch):
    fake_jobs["train"] = lambda ctx: JobResult("success", "trained")
    monkeypatch.setattr(runner, "HEAVY_LOCK_WAIT_SEC", 0.2)
    held = heavy_lock(settings.data_dir)
    assert held.acquire()
    try:
        result = run(settings, store, fake_clock, name="train")
    finally:
        held.release()
    assert result.status == "failed"
    row = store.job_runs_for("train", "2026-11-02")[0]
    assert row["status"] == "failed" and row["message"].startswith("preempted")  # does not count as a real attempt


def test_unknown_job_is_rejected(settings, store, fake_clock):
    with pytest.raises(KeyError):
        runner.execute_job(settings, "nope", "2026-11-02", store=store, clock=fake_clock)


def test_context_is_wired_for_the_job(settings, store, fake_clock, fake_jobs):
    seen = {}

    def inspect(ctx):
        seen.update(key=ctx.run_key, date=str(ctx.run_date), attempt=ctx.attempt, last=ctx.last_attempt, dry=ctx.dry_run,
                    remaining=ctx.remaining())
        return JobResult("success", "ok")

    fake_jobs["backup"] = inspect
    run(settings, store, fake_clock, attempt=3, max_attempts=4, last_attempt=False, dry_run=True, timeout_min=10)
    assert seen["key"] == "2026-11-02" and seen["date"] == "2026-11-02" and seen["attempt"] == 3
    assert seen["last"] is False and seen["dry"] is True and 0 < seen["remaining"] <= 600
