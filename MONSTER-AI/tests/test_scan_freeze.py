"""Freezing a scan: the snapshot and its predictions are written atomically, a snapshot that lost its predictions (an old interrupted
freeze) is repaired on retry, the stored features can be re-verified from their hash and the inputs the scan saw are recorded."""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from automation import scan
from automation.scan import run_scan
from backtesting import evaluate as ev
from database.store import Store
from tests.seed_data import make_job_context, panel_ending, populate_store

UTC = timezone.utc
PRELIM = datetime(2026, 11, 2, 3, 15, tzinfo=UTC)  # 08:45 IST
DAY = "2026-11-02"
COMPARED = ("rank", "tier", "model_version", "model_status", "p", "range", "uncertainty", "data_quality", "liquidity",
            "explanation", "features_hash")


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=300, n_symbols=24, seed=11)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42",
                                "send_prelim_report": True})
    populate_store(store, panel)
    fake_clock.set(PRELIM)
    return SimpleNamespace(settings=cfg, store=store, clock=fake_clock, panel=panel)


def context(env, **kwargs):
    return make_job_context(env.settings, env.store, env.clock, **kwargs)


def pre_predictions(env, **kwargs):
    return env.store.predictions_for(DAY, stage="pre", baselines=None, **kwargs)


def keyed(rows):
    return {(r["symbol"], r["is_baseline"], r["baseline_name"]): r for r in rows}


# ----------------------------------------------------------------------------------------- atomic freeze
def test_a_failure_between_the_snapshot_and_its_predictions_leaves_nothing_frozen(env, monkeypatch):
    def failing(self, *args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Store, "insert_predictions", failing)
    ctx = context(env, attempt=1, max_attempts=3, last_attempt=False)
    result = run_scan(ctx, "pre")
    assert result.status == "failed" and "disk full" in result.message
    assert env.store.conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()["n"] == 0  # no snapshot without predictions
    assert env.store.latest_snapshot(DAY, "pre") is None and ctx.telegram.sent == []  # and nothing was reported

    monkeypatch.undo()
    retry = context(env, attempt=2, max_attempts=3, last_attempt=False)
    assert run_scan(retry, "pre").status == "success"
    snapshot = env.store.latest_snapshot(DAY, "pre")
    assert snapshot is not None and len(env.store.predictions_for(DAY, stage="pre")) == 5 and retry.telegram.sent
    assert env.store.conn.execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()["n"] == 1


def test_a_successful_freeze_commits_the_snapshot_and_every_prediction_together(env):
    result = run_scan(context(env), "pre")
    assert result.status == "success"
    snapshot = env.store.latest_snapshot(DAY, "pre")
    assert result.metrics["snapshot_id"] == snapshot["snapshot_id"]
    assert result.metrics["predictions"] == len(pre_predictions(env, all_snapshots=True)) > 5  # picks and baselines


def test_the_frozen_snapshot_names_the_exact_code_that_made_it(env):
    from app.release import release_id

    assert run_scan(context(env), "pre").status == "success"
    row = env.store.conn.execute("SELECT code_version FROM snapshots WHERE trade_date = ? AND stage = 'pre'", (DAY,)).fetchone()
    assert row["code_version"] == release_id() and "+" in row["code_version"]
    assert env.store.latest_snapshot(DAY, "pre")["payload"]["meta"]["code"] == release_id()


# ------------------------------------------------------------------------------------- repair on retry
def lose_predictions(env):
    """The state an old interruption between the two writes left behind: a frozen snapshot with no predictions."""
    with env.store.allow_prune():
        env.store.conn.execute("DELETE FROM predictions")


def test_the_retry_restores_the_predictions_a_legacy_freeze_lost_from_its_own_payload(env, caplog):
    assert run_scan(context(env), "pre").status == "success"
    original = env.store.predictions_for(DAY, stage="pre", baselines=None)
    lose_predictions(env)
    assert pre_predictions(env) == []

    with caplog.at_level(logging.WARNING, logger="automation.scan"):
        result = run_scan(context(env, attempt=2, max_attempts=3, last_attempt=False), "pre")
    assert result.status == "success"
    restored = keyed(pre_predictions(env))
    wanted = [r for r in original if r["tier"] != "NONE"]  # the shadow-only extras were never part of the payload
    assert wanted and set(restored) == set(keyed(wanted))
    for key, row in keyed(wanted).items():
        assert {name: restored[key][name] for name in COMPARED} == {name: row[name] for name in COMPARED}, key
    assert {r["symbol"] for r in original if r["is_baseline"]} <= {r["symbol"] for r in restored.values()}
    assert any("restored" in rec.getMessage() and rec.levelno == logging.WARNING for rec in caplog.records)
    assert any(a["code"] == "snapshot_repaired" for a in env.store.pending_alerts(50))

    again = run_scan(context(env, attempt=3, max_attempts=3, last_attempt=True), "pre")
    assert again.status == "success" and len(pre_predictions(env)) == len(restored)  # idempotent: nothing is duplicated


def test_a_healthy_snapshot_is_left_alone_by_the_retry(env):
    assert run_scan(context(env), "pre").status == "success"
    before = [r["prediction_id"] for r in pre_predictions(env, all_snapshots=True)]
    assert run_scan(context(env, attempt=2, max_attempts=3, last_attempt=False), "pre").status == "success"
    assert [r["prediction_id"] for r in pre_predictions(env, all_snapshots=True)] == before
    assert not [a for a in env.store.pending_alerts(50) if a["code"] == "snapshot_repaired"]


def test_a_failed_repair_never_stops_the_frozen_report_from_being_resent(env, monkeypatch, caplog):
    assert run_scan(context(env), "pre").status == "success"
    lose_predictions(env)
    monkeypatch.setattr(scan, "restore_predictions", lambda *a, **k: (_ for _ in ()).throw(ValueError("corrupt payload")))
    with caplog.at_level(logging.WARNING, logger="automation.scan"):
        result = run_scan(context(env, attempt=2, max_attempts=3, last_attempt=False), "pre")
    assert result.status == "success"  # the frozen report is still delivered (here: recognised as already sent)
    assert any("could not be restored" in rec.getMessage() for rec in caplog.records)
    assert any(a["code"] == "snapshot_repair_failed" for a in env.store.pending_alerts(50))
    assert pre_predictions(env) == []


# ------------------------------------------------------------------------------------ reproducibility
def test_the_features_hash_can_be_re_verified_from_what_the_snapshot_stores(env):
    assert run_scan(context(env), "pre").status == "success"
    stored = env.store.latest_snapshot(DAY, "pre")["payload"]["features"]
    rows = env.store.predictions_for(DAY, stage="pre")  # the model's picks
    assert len(rows) == 5 and all(r["features_hash"] for r in rows)
    for row in rows:
        assert ev.verify_features_hash(stored, row["symbol"], row["features_hash"]), row["symbol"]
    tampered = copy.deepcopy(stored)
    tampered["rows"][rows[0]["symbol"]][0] = (tampered["rows"][rows[0]["symbol"]][0] or 0.0) + 1.0
    assert not ev.verify_features_hash(tampered, rows[0]["symbol"], rows[0]["features_hash"])
    baselines = env.store.predictions_for(DAY, stage="pre", baselines=True)
    assert baselines and all(r["features_hash"] is None for r in baselines)


def test_features_hash_hashes_exactly_the_stored_rounded_values():
    stored = {"columns": ["a", "b"], "rows": {"X": [1.2346, None], "Y": [2.0, 3.0]}}
    assert ev.features_hash(stored, "X") == ev.features_hash({"columns": ["b", "a"], "rows": {"X": [None, 1.2346]}}, "X")
    assert ev.features_hash(stored, "X") != ev.features_hash({"columns": ["a", "b"], "rows": {"X": [1.2347, None]}}, "X")
    assert ev.features_hash(stored, "X") != ev.features_hash(stored, "Y")
    for payload, symbol in ((stored, "Z"), (None, "X"), ({}, "X"), ({"columns": ["a"], "rows": {"X": [1.0, 2.0]}}, "X")):
        assert ev.features_hash(payload, symbol) is None
    assert ev.verify_features_hash(stored, "X", ev.features_hash(stored, "X")) and not ev.verify_features_hash(stored, "X", None)


# ----------------------------------------------------------------------------------- freshness inputs
def test_the_snapshot_records_the_latest_timestamp_and_arrival_of_every_input(env):
    env.store.add_live_quote("GIFT_NIFTY", price=24100.0, source_ts="02-Nov-2026 08:40:00")
    assert run_scan(context(env), "pre").status == "success"
    meta = env.store.latest_snapshot(DAY, "pre")["payload"]["meta"]
    fresh = meta["freshness"]
    assert meta["expected_session"] == "2026-10-30" and set(fresh) == {
        "market_bars", "global_closes", "fno", "preopen", "gift", "news", "filings"}
    assert fresh["market_bars"] == {"latest": "2026-10-30", "received_at": "2026-11-02T03:00:00Z", "n": len(env.panel.symbols)}
    assert fresh["global_closes"]["latest"] is not None and fresh["global_closes"]["n"] > 0
    assert fresh["fno"]["latest"] is not None and fresh["fno"]["n"] > 0
    assert fresh["gift"] == {"latest": "02-Nov-2026 08:40:00", "received_at": "2026-11-02T03:15:00Z", "n": 1}
    assert fresh["preopen"] == {"latest": None, "received_at": None, "n": 0}  # the preliminary scan runs before the book exists


def test_a_failure_reading_the_freshness_inputs_never_aborts_the_scan(env, monkeypatch, caplog):
    def failing(self, **kwargs):
        raise RuntimeError("locked")

    monkeypatch.setattr(Store, "source_freshness", failing)
    with caplog.at_level(logging.WARNING, logger="automation.scan"):
        assert run_scan(context(env), "pre").status == "success"
    meta = env.store.latest_snapshot(DAY, "pre")["payload"]["meta"]
    assert meta["freshness"] == {} and meta["expected_session"] == "2026-10-30"
    assert any("freshness" in rec.getMessage() and rec.levelno == logging.WARNING for rec in caplog.records)
