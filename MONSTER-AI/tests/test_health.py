"""Operational health evaluation and alerting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from automation.calendar import ExchangeCalendar, seed_calendar
from automation.health import HEARTBEAT_KEY, evaluate_health, raise_alerts
from tests.seed_data import panel_ending, populate_store

UTC = timezone.utc


def ist(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=UTC) - timedelta(hours=5, minutes=30)


@pytest.fixture()
def env(store, settings, fake_clock):
    seed_calendar(store, settings)
    fake_clock.set(ist(2026, 11, 2, 10, 0))
    return type("Env", (), {"store": store, "settings": settings, "clock": fake_clock})


def codes(report):
    return {p.code for p in report.problems}


def test_fresh_install_reports_expected_warnings_not_critical(env):
    report = evaluate_health(env.store, env.settings, env.clock)
    assert report.level == "warn" and report.exit_code == 1
    assert {"no_universe", "no_model_pre", "no_model_final", "telegram_not_configured"} <= codes(report)
    assert report.checks["db_integrity"] is True


def test_stale_heartbeat_is_critical(env):
    env.store.kv_set(HEARTBEAT_KEY, "2026-11-02T04:00:00Z")  # 09:30 IST; now is 10:00 IST
    report = evaluate_health(env.store, env.settings, env.clock)
    assert "scheduler_stalled" in codes(report) and report.level == "crit" and report.exit_code == 2


def test_fresh_heartbeat_passes(env):
    env.store.kv_set(HEARTBEAT_KEY, "2026-11-02T04:29:50Z")
    assert "scheduler_stalled" not in codes(evaluate_health(env.store, env.settings, env.clock))


def test_a_fresh_heartbeat_file_proves_liveness_while_the_database_is_busy(env):
    from automation.health import write_heartbeat_file

    env.store.kv_set(HEARTBEAT_KEY, "2026-11-02T04:00:00Z")  # 30 min old: a long VACUUM blocked the database write
    write_heartbeat_file(env.settings.data_dir, env.clock.now() - timedelta(seconds=20))
    report = evaluate_health(env.store, env.settings, env.clock)
    assert "scheduler_stalled" not in codes(report) and "heartbeat_db_stale" in codes(report)
    assert report.checks["heartbeat_age_sec"] == 20 and report.checks["heartbeat_db_age_sec"] == 1800
    write_heartbeat_file(env.settings.data_dir, env.clock.now() - timedelta(minutes=10))  # now both proofs are old
    assert "scheduler_stalled" in codes(evaluate_health(env.store, env.settings, env.clock))


def test_the_heartbeat_file_alone_is_enough_and_garbage_is_ignored(env):
    from automation.health import heartbeat_path, read_heartbeat_file, write_heartbeat_file

    assert read_heartbeat_file(env.settings.data_dir) is None
    heartbeat_path(env.settings.data_dir).write_text("not json")
    assert read_heartbeat_file(env.settings.data_dir) is None
    heartbeat_path(env.settings.data_dir).write_text('["list"]')
    assert read_heartbeat_file(env.settings.data_dir) is None
    assert write_heartbeat_file(env.settings.data_dir, env.clock.now()) is True
    assert read_heartbeat_file(env.settings.data_dir) == env.clock.now()
    assert not list(heartbeat_path(env.settings.data_dir).parent.glob("*.tmp"))  # atomic: no temp file left behind
    assert "scheduler_stalled" not in codes(evaluate_health(env.store, env.settings, env.clock))


def test_failed_critical_job_and_missing_final_scan_are_critical(env):
    env.store.kv_set("daemon.first_start", "2026-10-01T00:00:00Z")
    env.store.job_record("prelim_scan", "2026-11-02", "failed", message="no data")
    report = evaluate_health(env.store, env.settings, env.clock)
    assert "job_prelim_scan" in codes(report) and "final_scan_missing" in codes(report)
    assert report.level == "crit"


def test_missing_final_scan_is_not_flagged_before_the_scheduler_existed(env):
    assert "final_scan_missing" not in codes(evaluate_health(env.store, env.settings, env.clock))


def test_success_after_failure_clears_the_job_problem(env):
    env.store.job_record("collect_global", "2026-11-02", "failed", message="x")
    env.store.job_record("collect_global", "2026-11-02", "success", message="ok")
    assert "job_collect_global" not in codes(evaluate_health(env.store, env.settings, env.clock))


def test_stale_bars_and_models_and_backup(env):
    panel = panel_ending(n_dates=60, n_symbols=12, seed=3, last="2026-10-28")  # two sessions behind
    populate_store(env.store, panel)
    report = evaluate_health(env.store, env.settings, env.clock)
    assert "stale_bars" in codes(report)
    env.store.kv_set_json("backup.last", {"at": "2026-10-28T00:00:00Z"})
    assert "backup_old" in codes(evaluate_health(env.store, env.settings, env.clock))
    env.store.kv_set_json("backup.last", {"at": "2026-10-20T00:00:00Z"})
    crit = [p for p in evaluate_health(env.store, env.settings, env.clock).problems if p.code == "backup_old"]
    assert crit and crit[0].level == "crit"


def test_an_unencrypted_backup_is_flagged_only_when_encryption_was_requested(env):
    env.store.kv_set_json("backup.last", {"at": "2026-11-02T04:00:00Z", "encrypted": False})
    assert "backup_unencrypted" not in codes(evaluate_health(env.store, env.settings, env.clock))  # no passphrase: plain is intended
    settings = env.settings.__class__(**{**env.settings.__dict__, "backup_passphrase": "secret-pass"})
    report = evaluate_health(env.store, settings, env.clock)
    problem = next(p for p in report.problems if p.code == "backup_unencrypted")
    assert problem.level == "warn" and "secret-pass" not in problem.message
    env.store.kv_set_json("backup.last", {"at": "2026-11-02T04:00:00Z", "encrypted": True})
    assert "backup_unencrypted" not in codes(evaluate_health(env.store, settings, env.clock))


def test_failing_source_and_calibration_drift(env):
    from ingestion.registry import register_all

    register_all(env.store)
    for _ in range(6):
        env.store.record_source_result("yahoo_daily", False, error="HTTP 429")
    env.store.insert_metrics("2026-11-02", "rolling", [
        {"name": "n_days_90", "value": 60}, {"name": "ece_90", "value": 0.15}, {"name": "brier_skill_90", "value": -0.05}])
    report = evaluate_health(env.store, env.settings, env.clock)
    assert "source_yahoo_daily" in codes(report) and "calibration_drift" in codes(report) and "skill_negative" in codes(report)


def test_alerts_are_deduplicated(env):
    report = evaluate_health(env.store, env.settings, env.clock)
    first = raise_alerts(env.store, report)
    assert first == len(report.problems) > 0
    assert raise_alerts(env.store, report) == 0
    env.clock.advance(13 * 3600)
    assert raise_alerts(env.store, report) > 0  # warnings repeat after 12 h


def test_stale_holiday_calendar_is_flagged_once_the_system_has_been_running(env):
    env.store.kv_set("daemon.first_start", "2026-10-01T00:00:00Z")
    report = evaluate_health(env.store, env.settings, env.clock)
    assert "calendar_never_refreshed" in codes(report)
    env.store.kv_set("calendar.last_refresh", "2026-10-31T00:00:00Z")
    assert not {"calendar_never_refreshed", "calendar_stale"} & codes(evaluate_health(env.store, env.settings, env.clock))
    env.store.kv_set("calendar.last_refresh", "2026-10-10T00:00:00Z")
    assert "calendar_stale" in codes(evaluate_health(env.store, env.settings, env.clock))


def test_a_broken_check_is_reported_not_raised(env, monkeypatch):
    from automation import health

    monkeypatch.setattr(health, "_check_backup", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    report = evaluate_health(env.store, env.settings, env.clock)
    assert "check_backup" in codes(report)


def test_calendar_is_used_for_trading_day_logic(env):
    cal = ExchangeCalendar(env.store)
    saturday = datetime(2026, 10, 31, 5, 0, tzinfo=UTC)  # 10:30 IST Saturday: no final scan expected
    env.clock.set(saturday)
    assert "final_scan_missing" not in codes(evaluate_health(env.store, env.settings, env.clock, cal))


# ------------------------------------------------------------------------------------------- storage
def test_data_directory_budget_warnings(env):
    import dataclasses

    from app.resources import dir_size_mb

    with open(env.settings.cache_dir / "blob.bin", "wb") as handle:
        handle.truncate(50 * 1024 * 1024)
    used = dir_size_mb(env.settings.data_dir)
    budget_codes = {"storage_budget", "storage_over_budget"}

    def check(limit):
        return evaluate_health(env.store, dataclasses.replace(env.settings, data_max_mb=limit), env.clock)

    assert not budget_codes & codes(check(1000))  # far below the budget
    near = codes(check(int(used / 0.9) + 1))  # between 80 % and 100 % of the budget
    assert "storage_budget" in near and "storage_over_budget" not in near
    over = check(int(used) - 5)
    assert "storage_over_budget" in codes(over) and over.checks["data_budget_mb"] == int(used) - 5
    assert not budget_codes & codes(check(0))  # 0 = no budget


def test_retention_that_stopped_running_is_flagged(env):
    from automation.retention import RETENTION_KEY

    env.store.kv_set("daemon.first_start", "2026-10-01T00:00:00Z")
    assert "retention_never_ran" in codes(evaluate_health(env.store, env.settings, env.clock))
    env.store.kv_set_json(RETENTION_KEY, {"at": "2026-11-02T01:00:00Z", "notes": []})
    assert not {"retention_never_ran", "retention_stale", "retention_errors"} & codes(
        evaluate_health(env.store, env.settings, env.clock))
    env.store.kv_set_json(RETENTION_KEY, {"at": "2026-10-20T01:00:00Z", "notes": []})
    assert "retention_stale" in codes(evaluate_health(env.store, env.settings, env.clock))
    env.store.kv_set_json(RETENTION_KEY, {"at": "2026-11-02T01:00:00Z", "notes": ["events: OperationalError: locked"]})
    failing = evaluate_health(env.store, env.settings, env.clock)
    assert "events: OperationalError" in next(p.message for p in failing.problems if p.code == "retention_errors")


def test_a_new_install_is_not_blamed_for_missing_retention(env):
    assert "retention_never_ran" not in codes(evaluate_health(env.store, env.settings, env.clock))


def test_an_oversized_wal_is_reported(env, monkeypatch, tmp_path):
    import dataclasses

    from automation import health

    elsewhere = dataclasses.replace(env.settings, db_path=tmp_path / "other.db")  # never touch the live WAL
    elsewhere.db_path.with_name("other.db-wal").write_bytes(b"x" * 3_000_000)
    monkeypatch.setattr(health, "WAL_WARN_MB", 1)
    report = evaluate_health(env.store, elsewhere, env.clock)
    assert "wal_large" in codes(report) and report.checks["wal_mb"] >= 2.8
    monkeypatch.setattr(health, "WAL_WARN_MB", 256)
    assert "wal_large" not in codes(evaluate_health(env.store, elsewhere, env.clock))
