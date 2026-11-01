"""Retention: every table / folder has a bound, deletes are foreign-key safe and chunked, pressure shrinks only optional data."""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone

import pytest

from app.clock import FakeClock, iso
from automation import retention as rt
from database import backup as bk
from database.store import Store

NOW = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)  # Monday 08:30 IST
TODAY = date(2026, 11, 2)


def d(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


def ts(days_ago: float) -> str:
    return iso(NOW - timedelta(days=days_ago)) or ""


def count(store: Store, table: str, where: str = "1=1") -> int:
    return int(store.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0])


def run(store, settings, clock, **kwargs) -> rt.RetentionReport:
    kwargs.setdefault("free_disk_mb", 100_000)  # plenty: pressure tests pass their own value
    kwargs.setdefault("maintenance", False)
    return rt.run_retention(store, settings, now=NOW, clock=clock, **kwargs)


def snapshot(store: Store, trade_date: str, tag: str) -> int:
    cur = store.conn.execute(
        "INSERT INTO snapshots (trade_date, stage, cutoff_ts, created_at, feature_version, payload, data_quality_json, "
        "content_hash) VALUES (?, 'final', ?, ?, 'v1', x'00', '{}', ?)", (trade_date, ts(0), ts(0), tag))
    return int(cur.lastrowid)


def prediction(store: Store, snapshot_id: int, trade_date: str, symbol: str) -> int:
    cur = store.conn.execute(
        "INSERT INTO predictions (snapshot_id, trade_date, stage, symbol, rank, tier, model_version, model_status, p_json, "
        "range_json, explanation_json, data_quality, created_at) VALUES (?, ?, 'final', ?, 1, 'watch', 'm1', 'shadow', "
        "'{}', '{}', '{}', 'ok', ?)", (snapshot_id, trade_date, symbol, ts(0)))
    return int(cur.lastrowid)


def outcome(store: Store, prediction_id: int, trade_date: str, symbol: str) -> None:
    store.conn.execute("INSERT INTO outcomes (prediction_id, trade_date, symbol, source, recorded_at) VALUES (?, ?, ?, 'x', ?)",
                       (prediction_id, trade_date, symbol, ts(0)))


@pytest.fixture()
def clock(fake_clock: FakeClock) -> FakeClock:
    fake_clock.set(NOW)
    return fake_clock


@pytest.fixture()
def seeded(store: Store) -> Store:
    c = store.conn
    for key, received in (("old", ts(500)), ("new", ts(10))):
        c.execute("INSERT INTO live_quotes (key, received_at, price) VALUES (?, ?, 1)", (key, received))
    for day in (d(500), d(30), d(3)):
        for symbol in ("AAA", "BBB"):
            for poll in range(4):
                c.execute("INSERT INTO preopen_quotes (trade_date, symbol, received_at, iep) VALUES (?, ?, ?, ?)",
                          (day, symbol, f"{day}T03:{30 + poll:02d}:00Z", 100.0 + poll))
    for key, updated in (("old", ts(1200)), ("new", ts(100))):
        cur = c.execute("INSERT INTO events (cluster_key, event_type, headline, first_seen_at, last_updated_at) "
                        "VALUES (?, 'x', 'h', ?, ?)", (key, updated, updated))
        c.execute("INSERT INTO event_articles (event_id, added_at) VALUES (?, ?)", (cur.lastrowid, updated))
    for age, delivered in ((100, 1), (100, 0), (200, 0), (10, 1)):
        c.execute("INSERT INTO alerts (created_at, level, code, message, delivered) VALUES (?, 'info', 'c', 'm', ?)",
                  (ts(age), delivered))
    for key, age in (("a", 200), ("b", 5)):
        c.execute("INSERT INTO deliveries (channel, kind, dedupe_key, status, created_at, updated_at) "
                  "VALUES ('t', 'k', ?, 'sent', ?, ?)", (key, ts(age), ts(age)))
    c.execute("INSERT INTO job_runs (job_name, run_key, started_at, status) VALUES ('j', 'old', ?, 'success')", (ts(200),))
    c.execute("INSERT INTO job_runs (job_name, run_key, scheduled_for, status) VALUES ('j', 'missed', ?, 'missed')", (ts(200),))
    c.execute("INSERT INTO job_runs (job_name, run_key, started_at, status) VALUES ('j', 'new', ?, 'success')", (ts(2),))
    for age in (2000, 10):
        c.execute("INSERT INTO eval_metrics (as_of_date, computed_at, scope, name, value) VALUES (?, ?, 's', 'n', 1)",
                  (d(age), ts(0)))
    instrument = store.upsert_instrument("AAA")
    for age in (5000, 3000, 1):
        c.execute("INSERT INTO bars_daily (instrument_id, trade_date, close, source, received_at) VALUES (?, ?, 1, 'x', ?)",
                  (instrument, d(age), ts(0)))
        c.execute("INSERT INTO fno_daily (symbol, trade_date, received_at) VALUES ('AAA', ?, ?)", (d(age), ts(0)))
    for age in (5000, 100):
        c.execute("INSERT INTO macro_observations (series_id, observation_date, realtime_start, realtime_end, first_seen_at) "
                  "VALUES ('S', ?, ?, ?, ?)", (d(age), d(age), d(age), ts(0)))
    for key, scheduled in (("old", ts(500)), ("future", ts(-3))):
        c.execute("INSERT INTO econ_calendar (event_key, title, scheduled_at, source, first_seen_at, updated_at) "
                  "VALUES (?, 't', ?, 's', ?, ?)", (key, scheduled, ts(0), ts(0)))
    for key, updated in (("evaluation:2019-01-01", ts(2000)), ("evaluation:2026-10-30", ts(3)),
                         ("yahoo:fetched:GONE", ts(60)), ("yahoo:fetched:AAA", ts(1)), ("bootstrap.state", ts(900))):
        c.execute("INSERT INTO kv (key, value, updated_at) VALUES (?, 'v', ?)", (key, updated))
    return store


# ------------------------------------------------------------------------------------------- tables
def test_expired_rows_go_and_recent_rows_stay(seeded, settings, clock):
    report = run(seeded, settings, clock)
    assert count(seeded, "live_quotes") == 1 and count(seeded, "live_quotes", "key = 'new'") == 1
    assert count(seeded, "preopen_quotes", f"trade_date = '{d(3)}'") == 8  # recent days keep every poll
    assert count(seeded, "preopen_quotes", f"trade_date = '{d(30)}'") == 2  # thinned to the last poll per symbol
    assert count(seeded, "preopen_quotes", f"trade_date = '{d(500)}'") == 0  # past the quote window
    last = seeded.conn.execute("SELECT iep FROM preopen_quotes WHERE trade_date = ? AND symbol = 'AAA'", (d(30),)).fetchone()
    assert last["iep"] == 103.0
    assert [r["cluster_key"] for r in seeded.conn.execute("SELECT cluster_key FROM events")] == ["new"]
    assert count(seeded, "event_articles") == 1  # cascaded with its event
    assert count(seeded, "alerts") == 2  # delivered@100d goes (half window), undelivered@200d goes
    assert count(seeded, "deliveries") == 1
    assert [r["run_key"] for r in seeded.conn.execute("SELECT run_key FROM job_runs")] == ["new"]  # incl. a NULL started_at
    assert [r["as_of_date"] for r in seeded.conn.execute("SELECT as_of_date FROM eval_metrics")] == [d(10)]
    assert [r["trade_date"] for r in seeded.conn.execute("SELECT trade_date FROM bars_daily ORDER BY 1")] == [d(3000), d(1)]
    assert [r["trade_date"] for r in seeded.conn.execute("SELECT trade_date FROM fno_daily ORDER BY 1")] == [d(3000), d(1)]
    assert [r["observation_date"] for r in seeded.conn.execute("SELECT observation_date FROM macro_observations")] == [d(100)]
    assert [r["event_key"] for r in seeded.conn.execute("SELECT event_key FROM econ_calendar")] == ["future"]
    keys = {r["key"] for r in seeded.conn.execute("SELECT key FROM kv")}
    assert "evaluation:2019-01-01" not in keys and "yahoo:fetched:GONE" not in keys
    assert {"evaluation:2026-10-30", "yahoo:fetched:AAA", "bootstrap.state"} <= keys  # state is never pruned by age
    assert report.total_rows > 0 and not report.notes


def test_history_window_follows_history_years(seeded, settings, clock):
    run(seeded, dataclasses.replace(settings, history_years=2), clock)  # keeps 3 years of bars
    assert [r["trade_date"] for r in seeded.conn.execute("SELECT trade_date FROM bars_daily")] == [d(1)]


def test_zero_means_keep_forever(seeded, settings, clock):
    forever = dataclasses.replace(settings, record_retention_days=0, quote_retention_days=0, event_retention_days=0,
                                  ops_retention_days=0)
    before = {t: count(seeded, t) for t in ("live_quotes", "events", "alerts", "deliveries", "job_runs", "eval_metrics",
                                            "econ_calendar", "preopen_quotes")}
    run(seeded, forever, clock)
    after = {t: count(seeded, t) for t in before}
    assert after.pop("preopen_quotes") < before.pop("preopen_quotes")  # thinning is not a window
    assert after == before


def test_immutable_records_are_pruned_foreign_key_safe_and_the_guard_closes(store, settings, clock):
    old, recent = d(2000), d(10)
    s1 = snapshot(store, old, "s1")  # fully expired chain -> removed
    outcome(store, prediction(store, s1, old, "AAA"), old, "AAA")
    s2 = snapshot(store, old, "s2")  # an outcome with a recent date still points at the prediction -> stays
    p2 = prediction(store, s2, old, "BBB")
    outcome(store, p2, recent, "BBB")
    s3 = snapshot(store, old, "s3")  # prediction without outcome -> removed
    prediction(store, s3, old, "CCC")
    s4 = snapshot(store, recent, "s4")
    prediction(store, s4, recent, "DDD")

    report = run(store, settings, clock)

    assert not report.notes
    assert {r["snapshot_id"] for r in store.conn.execute("SELECT snapshot_id FROM snapshots")} == {s2, s4}
    assert {r["symbol"] for r in store.conn.execute("SELECT symbol FROM predictions")} == {"BBB", "DDD"}
    assert count(store, "outcomes") == 1
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):  # the delete guard is closed again
        store.conn.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (s4,))


def test_edge_history_keeps_the_version_that_was_current_when_the_window_starts(store, settings, clock):
    def version(src: str, dst: str, when: str, change: str, direction: int = 1) -> None:
        store.conn.execute(
            "INSERT INTO graph_edge_history (src, dst, relation, version, change, direction, valid_from) "
            "VALUES (?, ?, 'drives', 1, ?, ?, ?)", (src, dst, change, direction, when))

    version("oil", "ONGC", ts(3000), "created")  # superseded before the window -> removed
    version("oil", "ONGC", ts(2500), "updated")  # the version current at the cutoff (5 years back) -> kept
    version("oil", "ONGC", ts(100), "updated")  # inside the window -> kept
    version("gold", "TITAN", ts(3000), "created")  # never changed: its only row is still the current one -> kept
    version("fx", "TCS", ts(3000), "created")
    version("fx", "TCS", ts(2200), "removed", 0)  # the edge was removed before the window: its last row stays as the answer

    report = run(store, settings, clock)

    kept = [(r["src"], r["valid_from"]) for r in store.conn.execute("SELECT src, valid_from FROM graph_edge_history ORDER BY 1, 2")]
    assert report.rows["graph_edge_history"] == 2 and not report.notes  # oil@3000d, fx@3000d
    assert kept == [("fx", ts(2200)), ("gold", ts(3000)), ("oil", ts(2500)), ("oil", ts(100))]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):  # the guard is closed again
        store.conn.execute("DELETE FROM graph_edge_history")


def test_dry_run_counts_without_deleting_or_recording(seeded, settings, clock):
    before = {t: count(seeded, t) for t in ("live_quotes", "events", "alerts", "bars_daily", "preopen_quotes", "kv")}
    report = run(seeded, settings, clock, dry_run=True)
    assert report.dry_run and report.rows["live_quotes"] == 1 and report.rows["events"] == 1 and report.rows["bars_daily"] == 1
    assert report.rows["preopen_thinned"] == 12 and report.rows["kv_evaluation"] == 1
    assert {t: count(seeded, t) for t in before} == before
    assert seeded.kv_get(rt.RETENTION_KEY) is None


def test_the_run_is_recorded_and_is_idempotent(seeded, settings, clock):
    first = run(seeded, settings, clock)
    assert seeded.kv_get_json(rt.RETENTION_KEY)["at"] == first.at
    second = run(seeded, settings, clock)
    assert second.total_rows == 0 and not any(second.files.values())


# ------------------------------------------------------------------------------------------- pressure
def test_disk_pressure_levels(settings):
    assert rt.disk_pressure(None, settings) == "unknown"
    assert rt.disk_pressure(settings.min_free_disk_mb_warn + 1, settings) == "ok"
    assert rt.disk_pressure(settings.min_free_disk_mb_crit + 1, settings) == "warn"
    assert rt.disk_pressure(settings.min_free_disk_mb_crit - 1, settings) == "crit"


def test_pressure_shrinks_optional_windows_but_never_the_records(store, settings, clock):
    store.conn.execute("INSERT INTO live_quotes (key, received_at, price) VALUES ('q150', ?, 1)", (ts(150),))
    store.conn.execute("INSERT INTO live_quotes (key, received_at, price) VALUES ('q250', ?, 1)", (ts(250),))
    store.conn.execute("INSERT INTO eval_metrics (as_of_date, computed_at, scope, name, value) VALUES (?, ?, 's', 'n', 1)",
                       (d(1500), ts(0)))

    report = run(store, settings, clock, free_disk_mb=1000)  # warn: windows halve (400 -> 200)
    assert report.pressure == "warn" and report.factor == 0.5
    assert [r["key"] for r in store.conn.execute("SELECT key FROM live_quotes")] == ["q150"]

    report = run(store, settings, clock, free_disk_mb=300)  # crit: windows quarter (400 -> 100)
    assert report.pressure == "crit" and report.factor == 0.25
    assert count(store, "live_quotes") == 0
    assert count(store, "eval_metrics") == 1  # records only follow their own window


def test_window_never_collapses_below_a_week(settings):
    policy = rt.TablePolicy("x", "x", "c", "ts", lambda s: 10)
    assert rt.window_days(policy, settings, 0.25) == 7
    assert rt.window_days(rt.TablePolicy("x", "x", "c", "ts", lambda s: 3), settings, 0.25) == 3
    assert rt.window_days(rt.TablePolicy("x", "x", "c", "ts", lambda s: 0), settings, 0.25) == 0


def test_exceeding_the_data_budget_trims_harder(store, settings, clock):
    store.conn.execute("INSERT INTO live_quotes (key, received_at, price) VALUES ('q250', ?, 1)", (ts(250),))
    (settings.cache_dir / "blob.bin").write_bytes(b"x" * 2_000_000)
    report = run(store, dataclasses.replace(settings, data_max_mb=1), clock)
    assert report.over_budget and report.factor == 0.5 and "budget" in report.notes[0]
    assert count(store, "live_quotes") == 0


# ----------------------------------------------------------------------------------- robustness
def test_deletes_are_chunked_with_a_pause_between_chunks(store, settings, clock, monkeypatch):
    monkeypatch.setattr(rt, "CHUNK_ROWS", 3)
    for i in range(10):
        store.conn.execute("INSERT INTO live_quotes (key, received_at, price) VALUES (?, ?, 1)", (f"k{i}", ts(500)))
    run(store, settings, clock)
    assert count(store, "live_quotes") == 0
    assert clock.sleeps.count(rt.CHUNK_PAUSE_SEC) == 3  # 3 + 3 + 3 + 1


def test_a_failing_rule_is_reported_and_the_others_still_run(seeded, settings, clock, monkeypatch):
    original = rt.apply_policy

    def flaky(store, policy, *args, **kwargs):
        if policy.name == "events":
            raise sqlite3.OperationalError("database is locked")
        return original(store, policy, *args, **kwargs)

    monkeypatch.setattr(rt, "apply_policy", flaky)
    report = run(seeded, settings, clock)
    assert any("events" in note and "locked" in note for note in report.notes)
    assert count(seeded, "events") == 2 and count(seeded, "live_quotes") == 1


def test_a_prune_guard_left_open_by_a_crash_is_closed(store, clock):
    store.conn.execute("INSERT INTO kv (key, value, updated_at) VALUES (?, '1', ?)", (rt.PRUNE_FLAG_KEY, ts(0.5)))
    assert rt.reset_stale_prune_flag(store, NOW) is True
    assert store.kv_get(rt.PRUNE_FLAG_KEY) == "0"
    store.kv_set(rt.PRUNE_FLAG_KEY, "1")  # written "now" by a prune that is still running
    assert rt.reset_stale_prune_flag(store, clock.now()) is False
    assert store.kv_get(rt.PRUNE_FLAG_KEY) == "1"


def test_cutoffs_use_exchange_dates_not_the_server_zone():
    late_utc = datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc)  # already 01:30 on 2 November in India
    assert rt.cutoff_for("date", late_utc, 1) == "2026-11-01"
    assert rt.cutoff_for("ts", late_utc, 1) == "2026-10-31T20:00:00Z"


# ------------------------------------------------------------------------------------------- files
def age(path, days: float) -> None:
    moment = time.time() - days * 86400
    os.utime(path, (moment, moment))


def test_cache_tmp_and_leftovers_are_swept(settings, store, clock):
    old_cache, fresh_cache = settings.cache_dir / "old.bin", settings.cache_dir / "fresh.bin"
    old_tmp = settings.tmp_dir / "t.tmp"
    for path in (old_cache, fresh_cache, old_tmp):
        path.write_bytes(b"x")
    age(old_cache, 20)
    age(old_tmp, 3)
    backups = settings.backup_dir
    killed, fresh_run, real = backups / ".tmp-backup-abc", backups / ".verify-xyz", backups / "monster-20260101T000000Z.db.gz"
    for folder in (killed, fresh_run):
        folder.mkdir()
        (folder / "copy.db").write_bytes(b"x")
    real.write_bytes(b"x")
    age(killed, 3)
    age(real, 400)  # a real backup is never a "leftover", however old
    db = settings.db_path
    quarantined, recent, staging = (db.with_name(db.name + suffix) for suffix in
                                    (".corrupt-20261010T000000Z", ".corrupt-20261030T000000Z", ".restoring"))
    for path in (quarantined, recent, staging):
        path.write_bytes(b"x")
    age(quarantined, 20)
    age(recent, 4)
    age(staging, 3)

    report = run(store, settings, clock)

    assert not old_cache.exists() and fresh_cache.exists() and not old_tmp.exists()
    assert not killed.exists() and fresh_run.exists() and real.exists()
    assert not quarantined.exists() and recent.exists() and not staging.exists()
    assert report.files["cache"] == 1 and report.files["tmp"] == 1
    assert report.files["backup_leftovers"] == 1 and report.files["db_leftovers"] == 2


def test_logs_follow_age_and_size_budget_but_live_files_stay(settings, store, clock):
    logs = settings.log_dir
    names = {"ancient": "cli-2026-09-01.log", "mid_a": "jobs-2026-10-20.log", "mid_b": "jobs-2026-10-21.log",
             "today": "jobs-2026-11-02.log"}
    for name in names.values():
        (logs / name).write_bytes(b"x" * 700_000)
    age(logs / names["ancient"], 40)
    age(logs / names["mid_a"], 12)
    age(logs / names["mid_b"], 11)
    report = run(store, dataclasses.replace(settings, log_retention_days=30, log_max_mb=1), clock)
    assert not (logs / names["ancient"]).exists()  # older than the retention window
    assert not (logs / names["mid_a"]).exists() and not (logs / names["mid_b"]).exists()  # oldest first until it fits
    assert (logs / names["today"]).exists()  # a live file is never removed, even though it counts towards the budget
    assert report.files["logs"] == 3


def test_backup_budget_drops_the_oldest_but_always_keeps_the_newest(tmp_path):
    folder = tmp_path / "bk"
    folder.mkdir()
    names = [f"monster-2026110{i}T000000Z.db.gz" for i in range(1, 5)]
    for name in names:
        (folder / name).write_bytes(b"x" * 600_000)
        (folder / f"{name}.sha256").write_text("0" * 64 + f"  {name}\n")
    assert bk.enforce_budget(folder, 0) == []  # 0 = no budget
    assert bk.enforce_budget(folder, 100, dry_run=True) == []  # already within
    assert bk.enforce_budget(folder, 1.5, dry_run=True) == names[:2]
    assert (folder / names[0]).exists()  # dry run
    assert bk.enforce_budget(folder, 1.5) == names[:2]
    assert not (folder / names[0]).exists() and not (folder / f"{names[0]}.sha256").exists()
    assert bk.enforce_budget(folder, 0.001) == [names[2]]  # the newest one stays even when it alone is too big
    assert [p.name for p in bk.list_backups(folder)] == [names[3]]


def test_backup_job_applies_the_budget(tmp_path):
    db = tmp_path / "live.db"
    store = Store.open(db)
    store.kv_set("k", "v")
    store.close()
    for day in range(1, 4):
        bk.backup_database(db, tmp_path / "bk", keep_daily=10, clock=FakeClock(datetime(2026, 11, day, 1, tzinfo=timezone.utc)))
    assert len(bk.list_backups(tmp_path / "bk")) == 3
    result = bk.backup_database(db, tmp_path / "bk", keep_daily=10, max_total_mb=0.0001,
                                clock=FakeClock(datetime(2026, 11, 4, 1, tzinfo=timezone.utc)))
    assert [p.name for p in bk.list_backups(tmp_path / "bk")] == [result.path.name]
    assert len(result.pruned) == 3


# ------------------------------------------------------------------------------- database upkeep
def make_garbage(store: Store) -> None:
    store.conn.execute("BEGIN")
    store.conn.executemany("INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?)",
                           [(f"junk:{i}", "x" * 2000, ts(0)) for i in range(3000)])
    store.conn.execute("COMMIT")
    store.conn.execute("DELETE FROM kv WHERE key LIKE 'junk:%'")


def test_vacuum_returns_deleted_pages_to_the_file_system(store, settings):
    make_garbage(store)
    _before, reclaimable = rt.database_size(store.conn)
    assert reclaimable > 3
    actions = rt.maintain_database(store, settings, NOW, free_disk_mb=100_000, force_vacuum=True)
    assert "vacuum_s" in actions and actions["db_mb_after"] < actions["db_mb"]
    assert rt.database_size(store.conn)[1] < 0.5


def test_vacuum_waits_for_sunday_and_is_skipped_without_room(store, settings):
    make_garbage(store)
    monday = rt.maintain_database(store, settings, NOW, free_disk_mb=100_000)
    assert "vacuum_s" not in monday and "vacuum_skipped" not in monday
    sunday = datetime(2026, 11, 1, 3, 0, tzinfo=timezone.utc)
    cramped = rt.maintain_database(store, settings, sunday, free_disk_mb=1)
    assert "vacuum_s" not in cramped and "free disk" in cramped["vacuum_skipped"]
    sunday_run = rt.maintain_database(store, settings, sunday, free_disk_mb=100_000)
    assert "vacuum_s" in sunday_run
    again = rt.maintain_database(store, settings, sunday, free_disk_mb=100_000)  # nothing left to reclaim -> no rewrite
    assert "vacuum_s" not in again


def test_pressure_triggers_a_vacuum_on_any_weekday(store, settings):
    make_garbage(store)
    actions = rt.maintain_database(store, settings, NOW, free_disk_mb=100_000, pressure="warn")
    assert "vacuum_s" in actions


# ---------------------------------------------------------------------------------------- reporting
def test_storage_report_and_text(seeded, settings, clock):
    run(seeded, settings, clock)
    report = rt.storage_report(seeded, settings)
    assert report["tables"]["bars_daily"]["rows"] == 2 and report["tables"]["bars_daily"]["oldest"] == d(3000)
    assert report["policies"]["bars_daily"]["days"] == (settings.history_years + 1) * 366
    assert report["policies"]["preopen_quotes"]["thinned_after_days"] == rt.PREOPEN_DETAIL_DAYS
    assert report["last_run"]["pressure"] == "ok" and set(report["components_mb"]) >= {"database", "backups", "logs"}
    text = rt.storage_text(report)
    assert "Retention windows" in text and "bars_daily" in text and "Last retention run: 2026-11-02T03:00:00Z" in text
    assert '"data_dir_mb"' in rt.storage_json(report)


def test_storage_text_before_any_run(store, settings):
    assert rt.storage_text(rt.storage_report(store, settings, sizes=False)).endswith("Last retention run: never")


def test_every_policy_targets_a_real_table_and_column(store):
    for policy in rt.POLICIES:
        store.conn.execute(f"SELECT COUNT(*) FROM {policy.table} WHERE {rt._where(policy)}", ("0",)).fetchone()
        store.conn.execute(rt._delete_sql(policy), ("0", 1))  # nothing matches; proves the statement is valid SQL
