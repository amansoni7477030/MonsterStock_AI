"""Shared plumbing of the optional AI services: parsing helpers, the HTTP sender, health bookkeeping and the cache prune."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from ingestion.registry import register_all
from news_intelligence.jev import JevJudge
from news_intelligence.llm import LLMExtractor
from news_intelligence.remote import (
    RemoteJudge,
    clamp,
    describe_status,
    error_reason,
    prune_prefix,
    retry_after_seconds,
)
from tests.test_news_support import FakeResponse, FakeSession, jev_response, ok

TITLE = "Reliance bags Rs 5,000 crore NTPC order"
TEXT = "Reliance Industries has bagged a Rs 5,000 crore order from NTPC for supply of solar modules over three years."


# ------------------------------------------------------------------------------------------- helpers
@pytest.mark.parametrize(("value", "expected"), [
    (0.5, 0.5), ("0.7", 0.7), (5, 1.0), (-3, 0.0), (None, None), ("x", None), (float("nan"), None), (float("inf"), None),
    (float("-inf"), None), (True, 1.0),
])
def test_clamp_limits_a_number_and_rejects_everything_else(value, expected):
    assert clamp(value, 0.0, 1.0) == expected


def test_clamp_returns_the_default_for_what_is_not_a_finite_number():
    assert clamp("junk", 0.0, 1.0, default=0.4) == 0.4 and clamp(float("nan"), 0, 1, default=0.0) == 0.0


@pytest.mark.parametrize(("payload", "text", "reason"), [
    ({"error": {"message": "Rate limit reached for model `x`", "type": "tokens"}}, "", "Rate limit reached for model `x`"),
    ({"detail": {"error_type": "api_usage_error", "message": "Unknown model: jev-9"}}, "", "Unknown model: jev-9"),
    ({"detail": {"error_type": "api_usage_error"}}, "", "api_usage_error"),
    ({"detail": "plain detail"}, "", "plain detail"),
    ({"error": "just a string"}, "", "just a string"),
    ({"error": {"message": "  spaced \n out  "}}, "", "spaced out"),
    (None, "gateway timeout", "gateway timeout"),
    (None, "<html><body>502 Bad Gateway</body></html>", ""),  # an error page is not an explanation
    ({"unrelated": 1}, "", ""), ({"error": {}}, "", ""), ([], "", ""), (None, "", ""),
])
def test_error_reason_extracts_the_providers_own_short_explanation(payload, text, reason):
    assert error_reason(payload, text) == reason


def test_error_reason_is_capped_and_redacted():
    from app.logging_setup import register_secret

    register_secret("gsk_leaky_secret_value_0123456789")
    long = error_reason({"error": {"message": "x" * 1000}})
    assert len(long) == 140
    assert "gsk_leaky_secret_value_0123456789" not in error_reason({"error": {"message": "key gsk_leaky_secret_value_0123456789 bad"}})
    assert len(error_reason({"error": {"message": "y" * 1000}}, limit=30)) == 30


def test_describe_status_adds_the_reason_only_when_there_is_one():
    assert describe_status(400) == "http 400" and describe_status(400, "") == "http 400"
    assert describe_status(401, "Invalid API Key") == "http 401: Invalid API Key"


@pytest.mark.parametrize(("header", "expected"), [
    ("12", 12.0), ("0", 0.0), ("1.5", 1.5), ("-5", 0.0), ("1m30s", 90.0), ("250ms", 0.25), ("2h", 7200.0), ("1h2m3.5s", 3723.5),
    ("  45S ", 45.0), ("soon", None), ("", None), ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("99999999999", 7 * 86400.0),
])
def test_retry_after_accepts_seconds_and_durations_and_ignores_the_rest(header, expected):
    assert retry_after_seconds({"Retry-After": header}) == expected


def test_retry_after_is_found_in_either_spelling_or_not_at_all():
    assert retry_after_seconds({"retry-after": "7"}) == 7.0
    assert retry_after_seconds({}) is None and retry_after_seconds(None) is None


def test_prune_prefix_removes_only_old_entries_of_that_exact_prefix(store, fake_clock):
    for key in ("jev:old1", "jev:old2", "jevx:other", "llm:old", "jev;edge"):
        store.kv_set(key, "{}")
    fake_clock.advance(10 * 86400)
    store.kv_set("jev:fresh", "{}")
    assert prune_prefix(store, fake_clock.now(), "jev:", 5) == 2
    assert all(store.kv_get(key) for key in ("jev:fresh", "jevx:other", "llm:old", "jev;edge"))
    assert store.kv_get("jev:old1") is None and store.kv_get("jev:old2") is None
    assert prune_prefix(store, fake_clock.now(), "jev:", 5) == 0  # nothing left to remove


# ------------------------------------------------------------------------------------------ sending
def test_send_authenticates_identifies_itself_and_returns_status_json_text_and_headers(settings, fake_clock):
    base = RemoteJudge(clock=fake_clock, session=FakeSession(FakeResponse(200, {"a": 1}, headers={"x": "y"}, text="t" * 900),
                                                             gets=[FakeResponse(200, {"models": []})]))
    status, payload, text, headers = base.send("POST", "https://x.example/p", "KEY123", 9.0, {"q": 1})
    assert (status, payload, headers) == (200, {"a": 1}, {"x": "y"}) and len(text) == 500
    call = base.session.calls[0]
    assert call["headers"]["Authorization"] == "Bearer KEY123" and call["headers"]["User-Agent"].startswith("monster-ai/")
    assert call["json"] == {"q": 1} and call["timeout"] == 9.0
    status, payload, _, _ = base.send("GET", "https://x.example/models", None, 5.0)
    assert status == 200 and payload == {"models": []}
    assert "Authorization" not in base.session.calls[1]["headers"] and base.session.calls[1]["method"] == "GET"


def test_send_tolerates_a_body_that_is_not_json(fake_clock):
    base = RemoteJudge(clock=fake_clock, session=FakeSession(FakeResponse(502, json_error=True, text="<html>bad gateway</html>")))
    assert base.send("POST", "https://x.example", "k", 1.0, {}) == (502, None, "<html>bad gateway</html>", {})


# ---------------------------------------------------------------------------------- source health
def health(store, source_id):
    return next(row for row in store.source_health() if row["source_id"] == source_id)


def test_every_outcome_is_written_to_source_health_so_repeated_failures_raise_an_alert(settings, store, fake_clock):
    register_all(store)
    cfg = dataclasses.replace(settings, typesafe_api_key="ts_test_typesafe_key_1234567890")
    judge = JevJudge(cfg, session=FakeSession(jev_response(), FakeResponse(422), FakeResponse(422)), clock=fake_clock, store=store)
    assert judge.judge("a1", TEXT) is not None
    row = health(store, "jev_judgment")
    assert row["total_success"] == 1 and row["consecutive_failures"] == 0 and row["last_latency_ms"] is not None
    assert judge.judge("a2", TEXT) is None and judge.judge("a3", TEXT) is None
    row = health(store, "jev_judgment")
    assert row["consecutive_failures"] == 2 and row["total_failure"] == 2 and "422" in row["last_error"]
    assert judge.last_error is not None and "422" in judge.last_error


def test_the_llm_reports_to_its_own_source_row(settings, store, fake_clock):
    register_all(store)
    cfg = dataclasses.replace(settings, llm_api_base="https://llm.example/v1", llm_model="m", llm_api_key="sk-x")
    extractor = LLMExtractor(cfg, session=FakeSession(FakeResponse(401)), clock=fake_clock, store=store)
    assert extractor.extract(TITLE, TEXT) is None
    row = health(store, "llm_extraction")
    assert row["consecutive_failures"] == 1 and "401" in row["last_error"]
    assert health(store, "jev_judgment")["consecutive_failures"] == 0  # the other service is untouched


def test_a_time_budget_stop_is_not_a_source_failure(settings, store, fake_clock):
    register_all(store)
    cfg = dataclasses.replace(settings, typesafe_api_key="ts_test_typesafe_key_1234567890")
    judge = JevJudge(cfg, session=FakeSession(FakeResponse(503), jev_response()), clock=fake_clock, store=store,
                     backoff_base=500.0, backoff_cap=500.0)
    judge.start_run()
    assert judge.judge(TITLE, TEXT) is None
    assert health(store, "jev_judgment")["consecutive_failures"] == 0 and judge.last_error is None


def test_health_bookkeeping_can_never_lose_an_item(settings, store, fake_clock, monkeypatch, caplog):
    cfg = dataclasses.replace(settings, typesafe_api_key="ts_test_typesafe_key_1234567890")
    judge = JevJudge(cfg, session=FakeSession(jev_response()), clock=fake_clock, store=store)

    def broken(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(store, "record_source_result", broken)
    with caplog.at_level(logging.DEBUG):
        assert judge.judge(TITLE, TEXT) is not None  # the answer is still delivered
    assert "could not record the health of jev_judgment" in caplog.text


def test_an_unregistered_source_is_skipped_without_error(settings, store, fake_clock):
    cfg = dataclasses.replace(settings, llm_api_base="https://llm.example/v1", llm_model="m", llm_api_key="sk-x")
    extractor = LLMExtractor(cfg, session=FakeSession(ok("not json")), clock=fake_clock, store=store)
    assert extractor.extract(TITLE, TEXT) is None and extractor.run_failures == 1  # failed on content, not on bookkeeping


def test_a_service_without_a_store_records_nothing_and_still_works(settings, fake_clock):
    cfg = dataclasses.replace(settings, typesafe_api_key="ts_test_typesafe_key_1234567890")
    judge = JevJudge(cfg, session=FakeSession(jev_response()), clock=fake_clock)
    assert judge.judge(TITLE, TEXT) is not None and judge.store is None


def test_the_failure_reason_never_contains_the_key(settings, store, fake_clock):
    key = "ts_test_typesafe_key_1234567890"
    cfg = dataclasses.replace(settings, typesafe_api_key=key)
    judge = JevJudge(cfg, session=FakeSession(RuntimeError(f"boom with {key} inside")), clock=fake_clock, store=store)
    assert judge.judge(TITLE, TEXT) is None
    assert judge.last_error and key not in judge.last_error
