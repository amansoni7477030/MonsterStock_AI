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
