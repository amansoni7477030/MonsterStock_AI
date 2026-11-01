"""First-run bootstrap: step bookkeeping, resumability, crash accounting and the final state flag."""

from __future__ import annotations

import pytest

from app.errors import DataUnavailable
from automation import bootstrap as bs
from automation import jobs
from automation.context import JobResult
from ingestion.pipeline import IngestReport
from tests.seed_data import make_job_context, panel_ending, populate_store


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=120, n_symbols=10, seed=3)


@pytest.fixture()
def env(settings, store, fake_clock, panel, monkeypatch):
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42"})
    populate_store(store, panel)
    calls: dict[str, int] = {}

    def counted(name, result):
        def fn(*a, **k):
            calls[name] = calls.get(name, 0) + 1
            if isinstance(result, Exception):
                raise result
            return result(*a, **k) if callable(result) else result
        return fn

    class Universe:
        symbols = list(panel.symbols)
        source, notes = "seed", []

    def history(store_, yahoo, **kw):
        for row in store_.list_instruments(status="active"):
            if row["kind"] != "equity" or row["in_universe"]:
                store_.set_backfill(f"yahoo_history:{row['symbol']}", "done")
        return IngestReport(ok=40)

    monkeypatch.setattr(bs, "build_universe", counted("universe", Universe()))
    monkeypatch.setattr(bs, "backfill_history", counted("history", history))
    monkeypatch.setattr(bs, "refresh_calendar", counted("calendar", 5))
    monkeypatch.setattr(bs, "ingest_bhavcopy", counted("bhavcopy", (100, 100)))
    monkeypatch.setattr(bs, "backfill_fno", counted("fno", IngestReport(ok=30)))
    monkeypatch.setattr(jobs, "collect_news", counted("news", JobResult("success", "5/5 feeds ok")))
    monkeypatch.setattr(jobs, "train", counted("train", JobResult("success", "pre: shadow, final: shadow")))
    monkeypatch.setattr(bs.scan, "run_scan", counted("scan", JobResult("success", "ok", {"report": {
        "model_status": "baseline", "candidates": [{"symbol": "S01"}, {"symbol": "S02"}]}})))
    return type("Env", (), {"settings": cfg, "store": store, "clock": fake_clock, "calls": calls, "counted": staticmethod(counted)})


def run(env, **kw):
    ctx = make_job_context(env.settings, env.store, env.clock)
    ctx.options = kw.pop("options", {})
    return ctx, bs.run_bootstrap(ctx, **kw)


def test_full_bootstrap_completes_and_announces_itself(env):
    ctx, outcome = run(env)
    assert outcome.complete, outcome.summary()
    assert env.store.kv_get("bootstrap.state") == "done" and env.store.kv_get("bootstrap.completed_at")
    statuses = {k: v["status"] for k, v in outcome.steps.items()}
    assert statuses["init"] == statuses["universe"] == statuses["history"] == "done"
    assert statuses["train"] == statuses["smoke_scan"] == statuses["telegram"] == "done"
    assert "top: S01, S02" in outcome.steps["smoke_scan"]["detail"]
    assert ctx.telegram.sent, "the Telegram test message proves delivery works"
    assert any(a["code"] == "bootstrap_complete" for a in env.store.pending_alerts(20))


def test_the_universe_step_reports_the_size_of_the_event_graph(env):
    ctx = make_job_context(env.settings, env.store, env.clock)
    ctx.intel().reseed_graph = lambda: {"nodes": 140, "edges": 260, "nodes_created": 0, "edges_created": 0}
    outcome = bs.run_bootstrap(ctx)
    assert "event graph ready (140 nodes, 260 edges)" in outcome.steps["universe"]["detail"]
    assert "seeded" not in outcome.steps["universe"]["detail"]  # "0 seeded" would be misleading: it was seeded on first use


def test_resume_skips_finished_steps_and_retries_the_failed_one(env, monkeypatch):
    monkeypatch.setattr(bs, "backfill_history", env.counted("history", RuntimeError("yahoo blocked")))
    _ctx, first = run(env)
    assert not first.complete and env.store.kv_get("bootstrap.state") == "failed"
    assert first.steps["history"]["status"] == "failed" and "RuntimeError" in first.steps["history"]["detail"]
    universe_calls = env.calls["universe"]

    monkeypatch.setattr(bs, "backfill_history", env.counted("history", lambda s, y, **kw: (
        [s.set_backfill(f"yahoo_history:{r['symbol']}", "done") for r in s.list_instruments(status="active")
         if r["kind"] != "equity" or r["in_universe"]], IngestReport(ok=40))[1]))
    _ctx2, second = run(env, resume=True)
    assert second.complete
    assert env.calls["universe"] == universe_calls  # not repeated
    assert env.store.kv_get("bootstrap.state") == "done"


def test_force_restarts_from_scratch(env):
    run(env)
    first_calls = env.calls["universe"]
    run(env, resume=False)
    assert env.calls["universe"] == first_calls + 1


def test_incomplete_history_is_a_failed_required_step(env, monkeypatch):
    monkeypatch.setattr(bs, "backfill_history", env.counted("history", IngestReport(ok=2, failed=30)))
    _ctx, outcome = run(env)
    assert outcome.steps["history"]["status"] == "failed" and "without history" in outcome.steps["history"]["detail"]
    assert not outcome.complete


def test_optional_step_failures_do_not_block_completion(env, monkeypatch):
    monkeypatch.setattr(bs, "ingest_bhavcopy", env.counted("bhavcopy", DataUnavailable("nse", "blocked")))
    monkeypatch.setattr(bs, "backfill_fno", env.counted("fno", RuntimeError("zip bomb")))
    monkeypatch.setattr(bs.scan, "run_scan", env.counted("scan", JobResult("failed", "quality gate blocked")))
    _ctx, outcome = run(env)
    assert outcome.complete
    assert outcome.steps["official_recent"]["status"] == "failed"
    assert outcome.steps["fno_history"]["status"] == "failed"
    assert outcome.steps["smoke_scan"]["status"] == "failed"


def test_steps_that_keep_crashing_are_skipped_not_retried_forever(env):
    env.store.kv_set_json(bs.STEPS_KEY, {"train": {"status": "running", "crashes": 1}})  # died mid-step once before
    _ctx, outcome = run(env)
    assert outcome.steps["train"]["status"] == "skipped" and "died mid-step" in outcome.steps["train"]["detail"]
    assert env.calls.get("train", 0) == 0 and outcome.complete


def test_training_after_one_crash_uses_the_lean_profile(env, monkeypatch):
    seen = []
    monkeypatch.setattr(jobs, "train", lambda ctx: seen.append(dict(ctx.options)) or JobResult("success", "trained"))
    env.store.kv_set_json(bs.STEPS_KEY, {"train": {"status": "running", "crashes": 0}})  # the first attempt died mid-step
    run(env)
    assert seen == [{"lean": "1"}]

    seen.clear()
    run(env, resume=False)  # a clean first run starts at the normal profile
    assert seen == [{"lean": "0"}]


def test_required_steps_are_always_retried(env):
    env.store.kv_set_json(bs.STEPS_KEY, {"history": {"status": "running", "crashes": 5}})
    _ctx, outcome = run(env)
    assert outcome.steps["history"]["status"] == "done"


def test_skip_train_and_unconfigured_telegram(env, settings):
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": None, "telegram_chat_id": None})
    ctx = make_job_context(cfg, env.store, env.clock)
    outcome = bs.run_bootstrap(ctx, skip_train=True)
    assert outcome.steps["train"]["status"] == "skipped" and outcome.steps["telegram"]["status"] == "skipped"
    assert "telegram-setup" in outcome.steps["telegram"]["detail"] and outcome.complete


def test_external_training_mode_syncs_instead_of_training(env, monkeypatch):
    external = env.settings.__class__(**{**env.settings.__dict__, "training_mode": "external", "model_artifact_url":
                                         "https://example.com/models", "model_signing_key": "k" * 32})
    synced = []
    monkeypatch.setattr(jobs, "model_sync", lambda ctx: synced.append(1) or JobResult("success", "synced 2 models"))
    outcome = bs.run_bootstrap(make_job_context(external, env.store, env.clock))
    assert synced and outcome.steps["train"]["detail"] == "synced 2 models"


def test_summary_text_lists_problems(env, monkeypatch):
    monkeypatch.setattr(bs, "backfill_fno", env.counted("fno", RuntimeError("boom")))
    _ctx, outcome = run(env)
    text = outcome.summary()
    assert "steps done" in text and "fno_history" in text and "complete" in text
