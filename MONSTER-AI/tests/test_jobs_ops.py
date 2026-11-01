"""Operational jobs: backup, housekeeping, calendar and universe refresh, model sync gating."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from automation import jobs
from automation.scan import run_scan
from backtesting import evaluate as ev
from tests.seed_data import FakeNseApi, make_job_context, panel_ending, populate_store

UTC = timezone.utc


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=300, n_symbols=24, seed=17)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    populate_store(store, panel)
    return type("Env", (), {"settings": settings, "store": store, "clock": fake_clock, "panel": panel})


# ------------------------------------------------------------------------------------------------ backup
def test_backup_job_writes_a_verified_copy_and_records_it(env):
    env.store.kv_set("anything", "worth backing up")
    result = jobs.backup(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "success", result.message
    files = list(Path(env.settings.backup_dir).glob("monster-*.db.gz"))
    assert len(files) == 1 and files[0].with_name(files[0].name + ".sha256").exists()
    info = env.store.kv_get_json("backup.last")
    assert info["file"] == files[0].name and info["bytes"] > 0 and info["encrypted"] is False


def test_backup_failure_to_verify_is_a_failed_job(env, monkeypatch):
    from database import backup as backup_module

    monkeypatch.setattr(backup_module, "verify_backup", lambda *a, **k: False)
    assert jobs.backup(make_job_context(env.settings, env.store, env.clock)).status == "failed"


def test_an_unencrypted_backup_is_never_uploaded_when_encryption_was_requested(env, monkeypatch):
    from database import backup as backup_module

    def no_gpg(*_a, **_k):
        raise OSError("gpg missing")

    monkeypatch.setattr(backup_module.subprocess, "run", no_gpg)
    uploads = []
    monkeypatch.setattr(jobs, "_upload_backup", lambda ctx, path: uploads.append(path) or "backups/x")
    settings = env.settings.__class__(**{**env.settings.__dict__, "backup_passphrase": "secret-pass"})
    result = jobs.backup(make_job_context(settings, env.store, env.clock))
    assert result.status == "success" and result.metrics["encrypted"] is False and result.metrics["upload_skipped"] == "unencrypted"
    assert uploads == [] and "remote" not in result.metrics  # the plaintext copy stays on the server
    alert = next(a for a in env.store.pending_alerts(10) if a["code"] == "backup_encryption_failed")
    assert alert["level"] == "error" and "UNENCRYPTED" in alert["message"] and "secret-pass" not in alert["message"]
    assert list(Path(settings.backup_dir).glob("monster-*.db.gz"))  # a restore point exists


def test_a_plain_backup_without_a_passphrase_is_uploaded_as_before(env, monkeypatch):
    uploads = []
    monkeypatch.setattr(jobs, "_upload_backup", lambda ctx, path: uploads.append(path) or "backups/x")
    result = jobs.backup(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "success" and len(uploads) == 1 and result.metrics["remote"] == "backups/x"
    assert not [a for a in env.store.pending_alerts(10) if a["code"] == "backup_encryption_failed"]


# --------------------------------------------------------------------------------------------- housekeeping
def test_housekeeping_cleans_up_and_reports_disk_state(env):
    env.store.add_alert("warning", "old", "stale alert")
    env.store.conn.execute("UPDATE alerts SET delivered = 1, created_at = '2025-01-01T00:00:00Z'")
    (env.settings.cache_dir / "junk.bin").write_bytes(b"x" * 1024)
    result = jobs.housekeeping(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "success" and "MB free" in result.message
    assert env.store.conn.execute("SELECT COUNT(*) FROM alerts WHERE code = 'old'").fetchone()[0] == 0
    assert result.metrics["news"] == 0 and "models" in result.metrics


def test_housekeeping_runs_emergency_cleanup_when_the_disk_is_critical(env, monkeypatch):
    from app.resources import ResourceStatus

    called = {}
    monkeypatch.setattr(jobs, "check_resources", lambda *a, **k: ResourceStatus("crit", 100.0, 500.0, ["free disk 100 MB"]))
    monkeypatch.setattr(jobs, "emergency_cleanup", lambda *a, **k: called.setdefault("done", {"cache": 3}))
    result = jobs.housekeeping(make_job_context(env.settings, env.store, env.clock))
    assert called and result.metrics["emergency_cleanup"] == {"cache": 3}
    assert any(a["code"] == "resources" and a["level"] == "error" for a in env.store.pending_alerts(10))


def _leave_free_pages(store, rows: int = 3000) -> None:
    """Delete a few MB of rows so the database file holds pages a VACUUM can give back."""
    store.conn.execute("BEGIN")
    store.conn.executemany("INSERT INTO kv (key, value, updated_at) VALUES (?, ?, '2026-11-08T00:00:00Z')",
                           [(f"junk:{i}", "x" * 2000) for i in range(rows)])
    store.conn.execute("COMMIT")
    store.conn.execute("DELETE FROM kv WHERE key LIKE 'junk:%'")


def test_housekeeping_upgrades_provisional_outcomes_and_vacuums_on_sunday(env):
    env.clock.set(datetime(2026, 11, 2, 3, 15, tzinfo=UTC))
    ctx = make_job_context(env.settings, env.store, env.clock)
    assert run_scan(ctx, "pre").status == "success"
    last = env.panel.c.iloc[-1]
    for symbol in env.panel.symbols:
        o = float(last[symbol])
        env.store.upsert_bars(env.store.instrument_id(symbol), [{"trade_date": "2026-11-02", "open": o, "high": o * 1.03,
                                                                "low": o * 0.99, "close": o * 1.01, "volume": 1e6}], source="yahoo")
    assert ev.collect_outcomes(env.store, env.settings, "2026-11-02")["validated"] == 0
    for symbol in env.panel.symbols:  # official file arrives later
        o = float(last[symbol])
        env.store.upsert_bars(env.store.instrument_id(symbol), [{"trade_date": "2026-11-02", "open": o, "high": o * 1.03,
                                                                "low": o * 0.99, "close": o * 1.01, "volume": 1e6}],
                              source="nse_bhavcopy", source_rank=2, quality="official")
    env.clock.set(datetime(2026, 11, 8, 18, 30, tzinfo=UTC))  # Sunday (Muhurat) evening: no trading-day requirement
    _leave_free_pages(env.store)
    result = jobs.housekeeping(make_job_context(env.settings, env.store, env.clock, run_key="2026-11-08"))
    assert result.metrics["outcomes_validated"] > 0 and result.metrics.get("vacuum") is True
    assert result.metrics["maintenance"]["db_mb_after"] < result.metrics["maintenance"]["db_mb"]


def test_housekeeping_records_what_retention_removed_and_skips_a_pointless_vacuum(env):
    env.clock.set(datetime(2026, 11, 8, 18, 30, tzinfo=UTC))  # a Sunday run
    env.store.conn.execute("INSERT INTO live_quotes (key, received_at, price) VALUES ('old', '2024-01-01T00:00:00Z', 1)")
    result = jobs.housekeeping(make_job_context(env.settings, env.store, env.clock, run_key="2026-11-08"))
    assert result.metrics["removed"]["live_quotes"] == 1 and result.metrics["rows_removed"] >= 1
    assert result.metrics["vacuum"] is False  # nothing worth reclaiming: no heavy rewrite of the database
    assert env.store.kv_get_json("retention.last")["rows"]["live_quotes"] == 1
    assert "retention removed" in result.message


def test_housekeeping_raises_an_alert_when_a_retention_rule_fails(env, monkeypatch):
    from automation import retention

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(retention, "apply_kv_policy", broken)
    result = jobs.housekeeping(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "success"  # retention trouble never fails the nightly job
    assert any(a["code"] == "retention" for a in env.store.pending_alerts(10))


# ------------------------------------------------------------------------------- calendar / universe / sync
def test_refresh_calendar_uses_the_official_list_and_reports_failures(env):
    ctx = make_job_context(env.settings, env.store, env.clock)
    ctx._cache["nse_api"] = FakeNseApi(holidays=[{"trade_date": "2026-12-25", "description": "Christmas"}])
    ok = jobs.refresh_calendar_job(ctx)
    assert ok.status == "success" and ok.metrics["nse_days"] == 1
    assert not ctx.calendar.is_trading_day(datetime(2026, 12, 25).date())
    assert env.store.kv_get("calendar.last_refresh")
    ctx._cache["nse_api"] = FakeNseApi(fail=True)
    assert jobs.refresh_calendar_job(ctx).status == "failed"


def test_refresh_universe_updates_membership_and_loads_history(env, monkeypatch):
    from ingestion import universe as universe_module
    from ingestion.pipeline import IngestReport

    class Result:
        symbols = ["S00", "S01", "S02"]
        added, removed, source, notes = 1, 0, "seed", []

    monkeypatch.setattr(universe_module, "build_universe", lambda *a, **k: Result())
    monkeypatch.setattr(jobs, "backfill_history", lambda *a, **k: IngestReport(ok=2, failed=1))
    ctx = make_job_context(env.settings, env.store, env.clock)
    result = jobs.refresh_universe(ctx)
    assert result.status == "success" and result.metrics["universe"] == 3 and result.metrics["history_failed"] == 1
    assert result.metrics["board_meetings"] == 2
    assert ctx.intel().calls == ["reseed_graph", "learn_graph"]  # memberships follow the new universe, then the edges are re-measured


def test_model_sync_and_training_are_gated_by_mode(env):
    local = make_job_context(env.settings, env.store, env.clock)
    assert jobs.model_sync(local).status == "skipped" and "not external" in jobs.model_sync(local).message
    external = env.settings.__class__(**{**env.settings.__dict__, "training_mode": "external", "model_artifact_url": None,
                                         "model_signing_key": "k" * 32})
    assert jobs.train(make_job_context(external, env.store, env.clock)).status == "skipped"
    assert jobs.model_sync(make_job_context(external, env.store, env.clock)).status == "skipped"


def test_training_waits_when_the_disk_is_at_its_critical_level(env, monkeypatch):
    monkeypatch.setattr(jobs, "disk_info", lambda path: {"free_mb": 120.0, "total_mb": 8000.0, "used_mb": 7880.0})
    result = jobs.train(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "failed" and "120 MB of disk is free" in result.message  # retried inside its window, alerted at the end
    assert env.store.conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0] == 0


def test_train_years_fit_the_row_budget():
    from app.config import Settings

    s = Settings(train_years=6, train_max_rows=320000)
    assert jobs.train_years_for(s, 221) == 5  # 320000 / (221 * 252)
    assert jobs.train_years_for(s, 400) == 3  # never below three years
    assert jobs.train_years_for(Settings(train_years=4, train_max_rows=10**9), 221) == 4


def test_each_lean_level_drops_a_year_and_halves_the_fit_sample(monkeypatch):
    from app.config import Settings

    s = Settings(train_years=6, train_max_rows=320000)
    assert [jobs.train_years_for(s, 221, lean) for lean in (0, 1, 2, 3, 9)] == [5, 4, 3, 3, 3]
    assert jobs.train_years_for(s, 221, -4) == 5  # nonsense levels are ignored

    monkeypatch.setattr(jobs, "memory_info", lambda: {"total_mb": 1000.0})  # the 1 GB server: light profile
    assert [jobs.fit_row_cap(s, lean) for lean in (0, 1, 2, 9)] == [60_000, 30_000, 15_000, jobs.MIN_FIT_ROWS]
    monkeypatch.setattr(jobs, "memory_info", lambda: {"total_mb": 16_000.0})
    assert jobs.fit_row_cap(s) == 120_000 and jobs.fit_row_cap(Settings(train_max_rows=50_000)) == 50_000
    monkeypatch.setattr(jobs, "memory_info", lambda: {"total_mb": None})  # unknown RAM: full profile
    assert jobs.fit_row_cap(s) == 120_000


@pytest.mark.parametrize("attempt, options, expected", [
    (1, {}, 0), (2, {}, 1), (3, {}, 2), (1, {"lean": "1"}, 1), (3, {"lean": "1"}, 2), (1, {"lean": "x"}, 0), (1, {"lean": "-3"}, 0),
])
def test_the_lean_level_follows_failed_attempts_or_the_explicit_option(env, attempt, options, expected):
    ctx = make_job_context(env.settings, env.store, env.clock, attempt=attempt)
    ctx.options = options
    assert jobs.lean_level(ctx) == expected


def test_a_retried_training_job_uses_the_leaner_profile(env, monkeypatch):
    from features import live
    from models import training

    seen = {}
    monkeypatch.setattr(live, "build_training_features", lambda *a, **k: seen.update(years=k["years"]) or object())
    monkeypatch.setattr(training, "train_stage", lambda fs, cfg, *a, **k: seen.update(rows=cfg.max_rows_per_fit) or (_ for _ in ()).throw(
        training.TrainingError("stop here")))
    monkeypatch.setattr(jobs, "memory_info", lambda: {"total_mb": 1000.0})
    ctx = make_job_context(env.settings, env.store, env.clock, attempt=2)
    ctx.options = {"stages": "pre"}
    result = jobs.train(ctx)
    assert seen == {"years": 5, "rows": 30_000} and "lean profile, level 1" in result.message
    ctx = make_job_context(env.settings, env.store, env.clock, attempt=1)
    ctx.options = {"stages": "pre"}
    assert "lean" not in jobs.train(ctx).message and seen == {"years": 6, "rows": 60_000}


def test_health_check_job_raises_alerts(env):
    result = jobs.health_check(make_job_context(env.settings, env.store, env.clock))
    assert result.status == "success" and result.metrics["level"] in {"warn", "crit"}
    assert env.store.pending_alerts(20)
