from __future__ import annotations

import sqlite3

import pytest

from database.connection import connect, transaction
from database.migrate import (
    MigrationError,
    apply_migrations,
    check_integrity,
    current_version,
    discover,
    iter_statements,
)
from database.store import Store
from features.schema import FEATURE_VERSION


# ------------------------------------------------------------------------------------------ migrations
def test_migrations_apply_once_and_are_idempotent(tmp_path):
    conn = connect(tmp_path / "m.db")
    versions = [m.version for m in discover()]
    assert versions == list(range(1, len(versions) + 1))  # contiguous, no gaps
    first = apply_migrations(conn)
    assert first == versions and current_version(conn) == versions[-1]
    assert apply_migrations(conn) == []
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"bars_daily", "snapshots", "predictions", "outcomes", "model_registry", "job_runs", "deliveries",
            "macro_observations", "econ_calendar"} <= tables
    assert check_integrity(conn)


def test_modified_migration_is_refused(tmp_path):
    folder = tmp_path / "migrations"
    folder.mkdir()
    (folder / "0001_a.sql").write_text("CREATE TABLE a (id INTEGER);\n")
    conn = connect(tmp_path / "m.db")
    apply_migrations(conn, folder)
    (folder / "0001_a.sql").write_text("CREATE TABLE a (id INTEGER, extra TEXT);\n")
    with pytest.raises(MigrationError):
        apply_migrations(conn, folder)


def test_failed_migration_rolls_back_completely(tmp_path):
    folder = tmp_path / "migrations"
    folder.mkdir()
    (folder / "0001_bad.sql").write_text("CREATE TABLE ok_table (id INTEGER);\nCREATE TABLE ok_table (id INTEGER);\n")
    conn = connect(tmp_path / "m.db")
    with pytest.raises(sqlite3.OperationalError):
        apply_migrations(conn, folder)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "ok_table" not in names and current_version(conn) == 0


def test_bad_names_and_duplicates_rejected(tmp_path):
    (tmp_path / "x.sql").write_text("SELECT 1;")
    with pytest.raises(MigrationError):
        discover(tmp_path)


def test_statement_splitter_keeps_trigger_bodies_whole():
    script = (
        "-- header; with semicolon\nCREATE TABLE t (a);\n"
        "CREATE TRIGGER g BEFORE UPDATE ON t BEGIN SELECT RAISE(ABORT, 'no'); END;\nINSERT INTO t VALUES (1);\n"
    )
    statements = list(iter_statements(script))
    assert len(statements) == 3 and statements[1].startswith("CREATE TRIGGER") and statements[1].endswith("END;")
    with pytest.raises(MigrationError):
        list(iter_statements("CREATE TABLE broken (a"))


# -------------------------------------------------------------------------------------- transactions
def test_transaction_commit_rollback_and_savepoints(tmp_path):
    conn = connect(tmp_path / "t.db")
    conn.execute("CREATE TABLE t (a INTEGER)")
    with transaction(conn):
        conn.execute("INSERT INTO t VALUES (1)")
        with pytest.raises(ValueError):
            with transaction(conn):  # nested -> savepoint
                conn.execute("INSERT INTO t VALUES (2)")
                raise ValueError("inner")
    assert [r[0] for r in conn.execute("SELECT a FROM t")] == [1]
    with pytest.raises(RuntimeError):
        with transaction(conn):
            conn.execute("INSERT INTO t VALUES (3)")
            raise RuntimeError("outer")
    assert [r[0] for r in conn.execute("SELECT a FROM t")] == [1]
    assert not conn.in_transaction


def test_readonly_connection_cannot_write(tmp_path):
    path = tmp_path / "r.db"
    conn = connect(path)
    conn.execute("CREATE TABLE t (a)")
    ro = connect(path, readonly=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("INSERT INTO t VALUES (1)")


# ------------------------------------------------------------------------------------------ store: kv
def test_kv_roundtrip(store):
    assert store.kv_get("a") is None and store.kv_get("a", "d") == "d"
    store.kv_set("a", "1")
    store.kv_set("a", "2")
    assert store.kv_get("a") == "2"
    store.kv_set_json("j", {"x": [1, 2], "n": float("nan")})
    assert store.kv_get_json("j") == {"x": [1, 2], "n": None}
    store.kv_delete("a")
    assert store.kv_get("a") is None


# ------------------------------------------------------------------------------------- instruments
def test_instrument_upsert_preserves_existing_values(store):
    iid = store.upsert_instrument("RELIANCE", "equity", sector="Energy", is_fno=1, lot_size=500)
    assert store.upsert_instrument("RELIANCE", "equity", sector=None, industry="Refining") == iid
    row = store.get_instrument("RELIANCE")
    assert row["sector"] == "Energy" and row["industry"] == "Refining" and row["lot_size"] == 500
    with pytest.raises(ValueError):
        store.upsert_instrument("X", "equity", bogus=1)
    assert store.instrument_id("NOPE") is None


def test_universe_membership_and_point_in_time_history(store):
    ids = {s: store.upsert_instrument(s, "equity", sector="S", is_fno=1) for s in ("AAA", "BBB", "CCC")}
    result = store.set_universe(["AAA", "BBB"], as_of="2026-01-01")
    assert result == {"added": 2, "removed": 0, "total": 2}
    assert [r["symbol"] for r in store.list_instruments(kind="equity", in_universe=True)] == ["AAA", "BBB"]
    result = store.set_universe(["AAA", "CCC"], as_of="2026-06-01")
    assert result["added"] == 1 and result["removed"] == 1
    assert store.instrument_attrs_asof(ids["BBB"], "2026-03-01")["in_universe"] == 1
    assert store.instrument_attrs_asof(ids["AAA"], "2025-12-31") is None  # before first record -> unknown
    assert store.instrument_attrs_asof(ids["CCC"], "2026-07-01")["in_universe"] == 1
    assert store.record_instrument_history(ids["AAA"], "2026-06-01", {"in_universe": 1, "sector": "S", "industry": None,
                                                                         "is_fno": 1, "lot_size": None, "status": "active"}) is False


def test_status_and_filters(store):
    store.upsert_instrument("AAA", "equity", is_fno=1)
    store.upsert_instrument("BBB", "equity", is_fno=0)
    store.set_instrument_status("BBB", "equity", "quarantined", "no data")
    assert [r["symbol"] for r in store.list_instruments(kind="equity")] == ["AAA"]
    assert len(store.list_instruments(kind="equity", status=None)) == 2
    assert [r["symbol"] for r in store.list_instruments(fno=True)] == ["AAA"]


# --------------------------------------------------------------------------------------------- bars
def _bar(day, o=100.0, h=103.0, l=99.0, c=101.0, v=1000.0, **extra):
    return {"trade_date": day, "open": o, "high": h, "low": l, "close": c, "volume": v, **extra}


def test_official_bars_win_over_vendor_but_keep_vendor_extras(store):
    iid = store.upsert_instrument("AAA", "equity")
    store.upsert_bars(iid, [_bar("2026-10-01", adj_close=99.5)], source="yahoo", source_rank=1)
    store.upsert_bars(iid, [_bar("2026-10-01", o=100.5, h=104, l=98.5, c=102, turnover=5e8, delivery_pct=55)],
                      source="nse_bhavcopy", source_rank=2)
    bars = store.get_bars(iid)
    row = bars.iloc[0]
    assert row["high"] == 104 and row["source"] == "nse_bhavcopy" and row["adj_close"] == 99.5
    assert row["delivery_pct"] == 55
    # a later vendor refresh must not overwrite the official OHLC
    store.upsert_bars(iid, [_bar("2026-10-01", o=1, h=2, l=0.5, c=1.5)], source="yahoo", source_rank=1)
    assert store.get_bars(iid).iloc[0]["high"] == 104
    assert store.latest_bar_date(iid) == "2026-10-01"


def test_bars_sanitise_nan_and_blank(store):
    iid = store.upsert_instrument("AAA", "equity")
    store.upsert_bars(iid, [{"trade_date": "2026-10-02", "open": float("nan"), "high": "", "low": None, "close": "101.5",
                             "volume": float("inf")}], source="yahoo")
    row = store.get_bars(iid).iloc[0]
    assert row["close"] == 101.5 and row["open"] != row["open"] and row["volume"] != row["volume"]
    assert store.get_bars(iid)["open"].dtype == "float64"  # NULLs surface as NaN, not object/None


def test_bars_long_and_latest_dates_and_vendor_delete(store):
    a, b = store.upsert_instrument("AAA", "equity"), store.upsert_instrument("BBB", "equity")
    g = store.upsert_instrument("^GSPC", "global")
    store.upsert_bars(a, [_bar("2026-10-01"), _bar("2026-10-02")], source="yahoo")
    store.upsert_bars(b, [_bar("2026-10-01")], source="yahoo")
    store.upsert_bars(g, [_bar("2026-10-01")], source="yahoo")
    long = store.bars_long(kind="equity", start="2026-10-02")
    assert list(long["symbol"]) == ["AAA"]
    assert set(store.bars_long(kind=["equity", "global"])["symbol"]) == {"AAA", "BBB", "^GSPC"}
    assert store.bars_long(kind="equity", symbols=[]).empty
    assert store.latest_bar_dates("equity") == {"AAA": "2026-10-02", "BBB": "2026-10-01"}
    store.upsert_bars(b, [_bar("2026-10-03")], source="nse_bhavcopy", source_rank=2)
    assert store.delete_vendor_bars(b) == 1  # only the yahoo row
    assert store.latest_bar_date(b) == "2026-10-03"
    assert store.get_bars(store.upsert_instrument("ZZZ", "equity")).empty


# ----------------------------------------------------------------------------------- F&O / live / calendar
def test_fno_preopen_live_calendar_and_meetings(store, fake_clock):
    store.upsert_fno([{"symbol": "AAA", "trade_date": "2026-10-30", "fut_oi": 100, "pcr_oi": 0.8, "iv_atm": float("nan")}])
    store.upsert_fno([{"symbol": "AAA", "trade_date": "2026-10-30", "fut_oi": 120}])
    frame = store.fno_frame(symbols=["AAA"])
    assert frame.iloc[0]["fut_oi"] == 120 and store.latest_fno_date() == "2026-10-30"
    assert store.fno_frame(symbols=[]).empty

    store.add_preopen_quotes("2026-11-02", [{"symbol": "AAA", "iep": 100}], "2026-11-02T03:37:00Z")  # received 03:00:00
    fake_clock.advance(60)
    store.add_preopen_quotes("2026-11-02", [{"symbol": "AAA", "iep": 101, "final_price": 101.5}], None)  # received 03:01:00
    latest = store.latest_preopen("2026-11-02")
    assert latest["AAA"]["final_price"] == 101.5
    early = store.latest_preopen("2026-11-02", before="2026-11-02T03:00:30Z")
    assert early["AAA"]["iep"] == 100  # point-in-time view excludes the later observation

    store.add_live_quote("GIFT", price=23242.5, prev_close=23086, change_pct=0.68, extra={"x": 1})
    assert store.latest_live_quote("GIFT")["price"] == 23242.5 and store.latest_live_quote("NOPE") is None

    store.upsert_calendar_days([{"trade_date": "2026-11-10", "is_trading_day": False, "session_kind": "holiday",
                                 "description": "Diwali"}], source="test")
    assert store.get_calendar_day("2026-11-10")["is_trading_day"] == 0
    assert len(store.calendar_between("2026-11-01", "2026-11-30")) == 1

    store.upsert_board_meetings([{"symbol": "AAA", "meeting_date": "2026-11-05", "purpose": "Financial Results", "is_results": 1}])
    assert len(store.board_meetings_between("2026-11-01", "2026-11-30", ["AAA"])) == 1
    assert store.board_meetings_between("2026-11-01", "2026-11-30", ["ZZZ"]) == []


def test_backfill_progress(store):
    assert store.get_backfill("yahoo:AAA") is None
    store.set_backfill("yahoo:AAA", "running")
    store.set_backfill("yahoo:AAA", "failed", error="timeout")
    store.set_backfill("yahoo:AAA", "done", last_date="2026-10-30")
    progress = store.get_backfill("yahoo:AAA")
    assert progress["status"] == "done" and progress["attempts"] == 3 and progress["last_date"] == "2026-10-30"


# ---------------------------------------------------------------------- snapshots / predictions / outcomes
def _snapshot(store, **overrides):
    args = dict(trade_date="2026-11-02", stage="final", cutoff_ts="2026-11-02T03:42:00Z", feature_version=FEATURE_VERSION,
                model_version="m1", code_version="c1", payload={"AAA": {"x": 1.5, "n": float("nan")}}, data_quality={"ok": True})
    args.update(overrides)
    return store.insert_snapshot(**args)


def test_snapshots_are_immutable_and_idempotent(store):
    sid = _snapshot(store)
    assert _snapshot(store) == sid  # identical content -> same snapshot
    assert _snapshot(store, payload={"AAA": {"x": 2.0}}) != sid  # different content -> new frozen record
    snap = store.get_snapshot(sid)
    assert snap["payload"] == {"AAA": {"x": 1.5, "n": None}} and snap["data_quality"] == {"ok": True}
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE snapshots SET model_version = 'hack' WHERE snapshot_id = ?", (sid,))
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (sid,))
    assert store.latest_snapshot("2026-11-02", "final")["snapshot_id"] > sid - 1
    assert store.latest_snapshot("2026-11-02")["trade_date"] == "2026-11-02" and store.latest_snapshot("2020-01-01") is None
    assert store.snapshot_days("final") == 1


def _pred(symbol="AAA", rank=1, **extra):
    base = {"symbol": symbol, "rank": rank, "tier": "WATCH", "model_version": "m1", "model_status": "shadow",
            "p": {"hit_3": 0.31}, "range": {"q50": 2.4}, "explanation": {"catalyst": "x"}, "uncertainty": 0.2,
            "data_quality": "ok", "liquidity": {"turnover_cr": 100.0}}
    base.update(extra)
    return base


def test_predictions_frozen_outcomes_attached_separately(store):
    sid = _snapshot(store)
    ids = store.insert_predictions(sid, "2026-11-02", "final", [_pred("AAA", 1), _pred("BBB", 2),
                                                                _pred("CCC", 1, is_baseline=True, baseline_name="atr")])
    assert len(ids) == 3
    assert store.insert_predictions(sid, "2026-11-02", "final", [_pred("AAA", 1)]) == []  # re-insert is a no-op
    got = store.predictions_for("2026-11-02")
    assert [p["symbol"] for p in got] == ["AAA", "BBB"] and got[0]["p"] == {"hit_3": 0.31}
    assert len(store.predictions_for("2026-11-02", baselines=None)) == 3
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE predictions SET tier = 'STRONG'")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("DELETE FROM predictions")

    assert len(store.predictions_without_validated_outcome("2026-11-02")) == 3  # includes the baseline pick
    ohlc = {"open": 100, "high": 104, "low": 99, "close": 103}
    store.upsert_outcome(ids[0], "2026-11-02", "AAA", source="yahoo", validated=False, ohlcv=ohlc,
                         metrics={"range_pct": 5.0, "exc_pct": 4.0, "dir_ret_pct": 3.0}, hits={"hit_3": 1})
    store.upsert_outcome(ids[0], "2026-11-02", "AAA", source="nse_bhavcopy", validated=True, ohlcv=ohlc,
                         metrics={"range_pct": 5.05}, hits={"hit_3": 1})
    assert [p["symbol"] for p in store.predictions_without_validated_outcome("2026-11-02")] == ["BBB", "CCC"]
    rows = store.evaluated_rows(include_baselines=False)
    assert len(rows) == 1 and rows[0]["o_validated"] == 1 and rows[0]["range_pct"] == 5.05  # validated beats provisional
    assert store.prediction_dates() == ["2026-11-02"]
    assert store.prediction_dates(since="2026-11-03") == []


def test_allow_prune_lifts_delete_guard_only_inside_context(store):
    sid = _snapshot(store)
    store.insert_predictions(sid, "2026-11-02", "final", [_pred("AAA", 1)])
    with store.allow_prune():
        store.conn.execute("DELETE FROM predictions")
        store.conn.execute("DELETE FROM snapshots")
    assert store.kv_get("maintenance.allow_prune") == "0"
    sid2 = _snapshot(store, payload={"k": 1})
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("DELETE FROM snapshots WHERE snapshot_id = ?", (sid2,))


def test_metrics_roundtrip(store):
    assert store.latest_metrics("daily") == []
    store.insert_metrics("2026-11-02", "daily", [{"name": "top5_hit_rate", "value": 0.4, "n": 5, "ci_low": 0.1, "ci_high": 0.8}])
    store.insert_metrics("2026-11-03", "daily", [{"name": "top5_hit_rate", "value": 0.6, "n": 5}, {"name": "brier", "value": 0.2}])
    assert {m["name"] for m in store.latest_metrics("daily")} == {"top5_hit_rate", "brier"}
    assert len(store.latest_metrics("daily", name="brier")) == 1
    assert store.insert_metrics("2026-11-04", "daily", []) == 0


# ------------------------------------------------------------------------------------ model registry
def test_only_one_active_model_per_stage_and_rollback(store):
    for version in ("m1", "m2"):
        store.register_model(model_version=version, stage="final", metrics={"brier": 0.1})
    store.register_model(model_version="p1", stage="pre")
    store.set_model_status("m1", "validated", notes="gates passed")
    store.activate_model("m1")
    store.activate_model("p1")
    store.set_model_status("m2", "shadow")
    store.activate_model("m2")
    assert store.active_model("final")["model_version"] == "m2"
    assert store.get_model("m1")["is_active"] == 0 and store.get_model("m1")["approved_at"]
    assert store.active_model("pre")["model_version"] == "p1"  # other stage untouched
    store.activate_model("m1")  # rollback
    assert store.active_model("final")["model_version"] == "m1"
    assert [m["model_version"] for m in store.list_models("final")] == ["m2", "m1"] or len(store.list_models("final")) == 2
    assert len(store.list_models(status="validated")) == 1
    with pytest.raises(KeyError):
        store.activate_model("missing")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE model_registry SET is_active = 1 WHERE model_version = 'm2'")  # second active


# ----------------------------------------------------------------------------------- ledger / alerts
def test_job_ledger_settlement_rules(store):
    rid = store.job_start("final_scan", "2026-11-02", scheduled_for="2026-11-02T03:38:45Z")
    assert store.running_jobs()[0]["run_id"] == rid and not store.job_settled("final_scan", "2026-11-02")
    store.job_finish(rid, "failed", message="boom", exit_code=1, peak_rss_mb=312.5)
    assert not store.job_settled("final_scan", "2026-11-02")  # failures stay retryable
    rid2 = store.job_start("final_scan", "2026-11-02", attempt=2)
    store.job_finish(rid2, "success")
    assert store.job_settled("final_scan", "2026-11-02")
    assert len(store.job_runs_for("final_scan", "2026-11-02")) == 2
    store.job_record("train", "2026-11-02", "skipped", message="holiday")
    assert store.job_settled("train", "2026-11-02") and store.running_jobs() == []
    assert store.last_job_success("final_scan")["attempt"] == 2 and store.last_job_success("nope") is None
    assert len(store.recent_jobs(5)) == 3


def test_alert_dedupe_window(store, fake_clock):
    assert store.add_alert("error", "feed_down", "yahoo down", dedupe_key="yahoo", dedupe_minutes=60)
    assert not store.add_alert("error", "feed_down", "yahoo down", dedupe_key="yahoo", dedupe_minutes=60)
    fake_clock.advance(61 * 60)
    assert store.add_alert("error", "feed_down", "yahoo down", dedupe_key="yahoo", dedupe_minutes=60)
    assert store.add_alert("warn", "other", "x")  # no dedupe key -> always queued
    pending = store.pending_alerts()
    assert len(pending) == 3
    store.mark_alert_delivered(pending[0]["alert_id"])
    assert len(store.pending_alerts()) == 2


def test_delivery_ledger_is_idempotent(store):
    assert store.get_delivery("telegram", "final:2026-11-02") is None
    store.save_delivery("telegram", "final_report", "final:2026-11-02", "failed", attempts=2, error="429")
    store.save_delivery("telegram", "final_report", "final:2026-11-02", "sent", attempts=1, message_ids=[11, 12], body_hash="h")
    delivery = store.get_delivery("telegram", "final:2026-11-02")
    assert delivery["status"] == "sent" and delivery["attempts"] == 3 and delivery["sent_at"]
    assert delivery["message_ids"] == "[11,12]"


def test_sources_and_health(store):
    store.register_source(source_id="yahoo_daily", provider="Yahoo", endpoint="x", coverage="c", timestamp_meaning="t",
                          update_frequency="d", history_depth="10y", rate_limits="1/s", license_notes="research",
                          failure_behavior="retry", backup_source="nse_bhavcopy", criticality="critical")
    store.record_source_result("yahoo_daily", False, error="HTTP 429")
    store.record_source_result("yahoo_daily", False, error="HTTP 429")
    health = store.source_health()[0]
    assert health["consecutive_failures"] == 2 and health["last_error"] == "HTTP 429"
    store.record_source_result("yahoo_daily", True, latency_ms=120.0, data_ts="2026-10-30")
    health = store.source_health()[0]
    assert health["consecutive_failures"] == 0 and health["total_success"] == 1 and health["last_data_ts"] == "2026-10-30"
    store.set_source_enabled("yahoo_daily", False)
    assert store.source_health()[0]["enabled"] == 0


def test_corporate_actions_and_counts(store):
    iid = store.upsert_instrument("AAA", "equity")
    assert store.add_corporate_action(iid, "split", "2026-09-01", ratio=2.0, source="yahoo") is True
    assert store.add_corporate_action(iid, "split", "2026-09-01", ratio=2.0) is False
    assert len(store.corporate_actions(iid)) == 1 and store.corporate_actions(iid, "bonus") == []
    assert store.table_counts()["corporate_actions"] == 1


def test_store_open_readonly_and_close(tmp_path):
    path = tmp_path / "s.db"
    Store.open(path).close()
    ro = Store.open(path, readonly=True)
    assert ro.kv_get("x") is None
    ro.close()
