"""Groq behaviour of the LLM extractor: defaults, strict JSON schema, reasoning models, model fallback, quota cooldown,
learned capabilities and the wall-clock budget. The generic OpenAI-compatible path is covered by ``test_news_llm.py``."""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta

import pytest

from news_intelligence import llm as llm_mod
from news_intelligence.entities import EntityResolver
from news_intelligence.events import EVENT_TYPES, HORIZONS
from news_intelligence.llm import LLMExtractor, extraction_schema, is_reasoning_model
from tests.test_news_support import INSTRUMENTS, FakeResponse, FakeSession, add_instruments, ok

GROQ_KEY = "gsk_test_groq_key_1234567890"
TITLE = "Reliance bags Rs 5,000 crore NTPC order"
TEXT = (
    "Reliance Industries has bagged a Rs 5,000 crore order from NTPC for supply of solar modules over three years, "
    "the company said in a stock exchange filing on Monday."
)
QUOTE = "bagged a Rs 5,000 crore order from NTPC"
PRIMARY, FALLBACK = "openai/gpt-oss-120b", "openai/gpt-oss-20b"


def reply(**over) -> str:
    data = {
        "event_type": "order_win", "entities": [{"id": "RELIANCE", "name": None, "type": "company", "polarity": 0.8}],
        "severity": 0.6, "horizon": "days", "event_time": None, "confidence": 0.85, "evidence_quote": QUOTE,
    }
    data.update(over)
    return json.dumps(data)


def with_usage(response: FakeResponse, tokens_in: int = 410, tokens_out: int = 150) -> FakeResponse:
    response._payload = {**response._payload, "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out}}
    return response


@pytest.fixture()
def cfg(settings):
    return dataclasses.replace(settings, groq_api_key=GROQ_KEY, llm_max_output_tokens=900)


@pytest.fixture()
def resolver(store):
    add_instruments(store, INSTRUMENTS)
    return EntityResolver(store)


def make(cfg, store, resolver, clock, *responses, **kwargs):
    session = FakeSession(*(responses or [ok(reply())]))
    return LLMExtractor(cfg, session=session, clock=clock, store=store, resolver=resolver, **kwargs), session


# ------------------------------------------------------------------------------------------ defaults
def test_a_groq_key_alone_configures_the_extractor_with_the_best_production_model(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock)
    assert extractor.configured and extractor.models == [PRIMARY, FALLBACK]
    result = extractor.extract(TITLE, TEXT)
    assert result is not None and result.event_type == "order_win" and result.quote_verified
    call = session.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer " + GROQ_KEY and call["headers"]["User-Agent"].startswith("monster-ai/")
    body = call["json"]
    assert body["model"] == PRIMARY and body["temperature"] == 0 and body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 900 and "max_tokens" not in body
    assert GROQ_KEY not in json.dumps(body)


def test_the_request_asks_for_a_strict_json_schema(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock)
    extractor.extract(TITLE, TEXT)
    response_format = session.calls[0]["json"]["response_format"]
    assert response_format["type"] == "json_schema" and response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == extraction_schema()


def test_the_schema_is_strict_compatible_and_matches_the_taxonomy():
    """Constrained decoding needs every property required and no extras, at every level."""
    schema = extraction_schema()

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False and node["required"] == list(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    props = schema["properties"]
    assert props["event_type"]["enum"] == list(EVENT_TYPES) and props["horizon"]["enum"] == list(HORIZONS)
    assert set(props) == {"event_type", "entities", "severity", "horizon", "event_time", "confidence", "evidence_quote"}
    assert props["event_time"]["type"] == ["string", "null"] and props["evidence_quote"]["type"] == ["string", "null"]
    json.dumps(schema)  # serialisable as sent


def test_a_reply_in_the_schemas_exact_shape_validates_including_null_ids_and_names(cfg, store, resolver, fake_clock):
    entities = [{"id": None, "name": "Tata Consultancy Services", "type": "company", "polarity": -0.4},
                {"id": "RELIANCE", "name": None, "type": "company", "polarity": 0.5}]
    extractor, _ = make(cfg, store, resolver, fake_clock, ok(reply(entities=entities, event_time=None)))
    result = extractor.extract(TITLE, TEXT)
    assert [(e.id, e.polarity) for e in result.entities] == [("TCS", -0.4), ("RELIANCE", 0.5)] and result.event_time is None


def test_reasoning_effort_goes_only_to_reasoning_models_and_only_when_set(cfg, store, resolver, fake_clock):
    assert is_reasoning_model("openai/gpt-oss-20b") and not is_reasoning_model("meta-llama/llama-4-scout") and not is_reasoning_model(None)
    other = dataclasses.replace(cfg, llm_model="qwen/qwen3.6-27b")
    extractor, session = make(other, store, resolver, fake_clock)
    extractor.extract(TITLE, TEXT)
    assert "reasoning_effort" not in session.calls[0]["json"] and session.calls[0]["json"]["model"] == "qwen/qwen3.6-27b"
    silent = dataclasses.replace(cfg, llm_reasoning_effort="")
    extractor, session = make(silent, store, resolver, fake_clock)
    extractor.extract(TITLE + " x", TEXT)
    assert "reasoning_effort" not in session.calls[0]["json"]


def test_a_model_override_replaces_the_primary_and_the_fallback_never_equals_it(cfg):
    assert dataclasses.replace(cfg, llm_model=FALLBACK).llm_fallback_name is None
    assert dataclasses.replace(cfg, llm_fallback_model="x/y").llm_fallback_name == "x/y"


# ------------------------------------------------------------------------------------- model fallback
def test_a_rate_limited_model_hands_over_to_the_other_at_once_without_sleeping(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock, FakeResponse(429, headers={"retry-after": "7"}), ok(reply()))
    assert extractor.extract(TITLE, TEXT) is not None
    assert [c["json"]["model"] for c in session.calls] == [PRIMARY, FALLBACK] and fake_clock.sleeps == []
    assert extractor.run_failures == 0
    assert extractor.extract(TITLE + " 2", TEXT) is not None and session.calls[-1]["json"]["model"] == FALLBACK
    extractor.start_run()
    assert extractor._active_model() == PRIMARY  # every run starts on the best model again


def test_when_both_models_are_limited_a_short_wait_is_slept_and_a_long_one_is_a_quota(cfg, store, resolver, fake_clock):
    short, session = make(cfg, store, resolver, fake_clock, FakeResponse(429, headers={"retry-after": "4"}),
                          FakeResponse(429, headers={"retry-after": "4"}), ok(reply()))
    assert short.extract(TITLE, TEXT) is not None and fake_clock.sleeps == [4.0] and len(session.calls) == 3
    long, session = make(cfg, store, resolver, fake_clock, FakeResponse(429, headers={"retry-after": "3600"}),
                         FakeResponse(429, headers={"retry-after": "3600"}), ok(reply()))
    assert long.extract(TITLE + " b", TEXT) is None and long.tripped and len(session.calls) == 2
    assert fake_clock.sleeps == [4.0]  # nothing slept for the quota


def test_a_quota_starts_a_cooldown_that_the_next_run_respects_until_it_ends(cfg, store, resolver, fake_clock):
    extractor, _ = make(cfg, store, resolver, fake_clock, FakeResponse(429, headers={"retry-after": "7200"}),
                        FakeResponse(429, headers={"retry-after": "7200"}))
    assert extractor.extract(TITLE, TEXT) is None and store.kv_get("llm.cooldown_until")
    later, session = make(cfg, store, resolver, fake_clock, ok(reply()))
    assert later.extract(TITLE + " b", TEXT) is None and session.calls == [] and later.tripped  # no request at all
    fake_clock.advance(7200 + 5)
    again, session = make(cfg, store, resolver, fake_clock, ok(reply()))
    assert again.extract(TITLE + " b", TEXT) is not None and len(session.calls) == 1


def test_a_cooldown_never_lasts_longer_than_six_hours(cfg, store, resolver, fake_clock):
    extractor, _ = make(cfg, store, resolver, fake_clock, FakeResponse(429, headers={"retry-after": str(20 * 3600)}),
                        FakeResponse(429, headers={"retry-after": str(20 * 3600)}))
    extractor.extract(TITLE, TEXT)
    fake_clock.advance(6 * 3600 + 5)
    assert extractor.cooldown_active() is False  # one request re-probes instead of waiting a whole day


def test_cached_results_are_served_during_a_cooldown(cfg, store, resolver, fake_clock):
    first, _ = make(cfg, store, resolver, fake_clock, ok(reply()))
    assert first.extract(TITLE, TEXT) is not None
    first.start_cooldown(3600)
    again, session = make(cfg, store, resolver, fake_clock, ok(reply()))
    assert again.extract(TITLE, TEXT).cached is True and again.extract(TITLE + " new", TEXT) is None and session.calls == []


# --------------------------------------------------------------------- retired or unknown models
@pytest.mark.parametrize("gone", [FakeResponse(404, {"error": {"message": "model not found"}}),
                                  FakeResponse(400, {"error": {"message": "The model has been decommissioned", "code": "model_decommissioned"}})])
def test_a_retired_primary_model_falls_back_for_the_rest_of_the_run(cfg, store, resolver, fake_clock, gone):
    extractor, session = make(cfg, store, resolver, fake_clock, gone, ok(reply()))
    assert extractor.extract(TITLE, TEXT) is not None
    assert [c["json"]["model"] for c in session.calls] == [PRIMARY, FALLBACK] and extractor.tripped is False
    assert extractor.extract(TITLE + " b", TEXT) is not None and session.calls[-1]["json"]["model"] == FALLBACK


def test_when_every_model_is_gone_the_circuit_opens(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock, FakeResponse(404), FakeResponse(404))
    assert extractor.extract(TITLE, TEXT) is None and extractor.tripped and len(session.calls) == 2


# --------------------------------------------------------------------------- learned capabilities
def test_a_rejected_schema_steps_down_to_json_object_and_is_remembered_between_runs(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock,
                              FakeResponse(400, {"error": {"message": "response_format json_schema is not supported by this model"}}),
                              ok(reply()))
    assert extractor.extract(TITLE, TEXT) is not None
    formats = [c["json"].get("response_format", {}).get("type") for c in session.calls]
    assert formats == ["json_schema", "json_object"] and fake_clock.sleeps == []
    later, session = make(cfg, store, resolver, fake_clock, ok(reply()))  # a new process
    later.extract(TITLE + " b", TEXT)
    assert session.calls[0]["json"]["response_format"] == {"type": "json_object"}  # no failed request to rediscover it
    fake_clock.advance(llm_mod.CAPS_TTL_DAYS * 86400 + 60)
    expired, session = make(cfg, store, resolver, fake_clock, ok(reply()))
    expired.extract(TITLE + " c", TEXT)
    assert session.calls[0]["json"]["response_format"]["type"] == "json_schema"  # a week later: ask for the best mode again


def test_the_ladder_ends_without_a_response_format_and_then_gives_up(cfg, store, resolver, fake_clock):
    complaint = {"error": {"message": "response_format is not supported"}}
    extractor, session = make(cfg, store, resolver, fake_clock, FakeResponse(400, complaint), FakeResponse(400, complaint),
                              FakeResponse(400, complaint))
    assert extractor.extract(TITLE, TEXT) is None
    assert ["response_format" in c["json"] for c in session.calls] == [True, True, False]


def test_a_rejected_reasoning_effort_is_dropped_and_remembered(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock, FakeResponse(400, {"error": {"message": "unknown field reasoning_effort"}}),
                              ok(reply()))
    assert extractor.extract(TITLE, TEXT) is not None
    assert "reasoning_effort" in session.calls[0]["json"] and "reasoning_effort" not in session.calls[1]["json"]
    later, session = make(cfg, store, resolver, fake_clock, ok(reply()))
    later.extract(TITLE + " b", TEXT)
    assert "reasoning_effort" not in session.calls[0]["json"]


def test_the_token_limit_parameter_swaps_when_the_endpoint_wants_the_other_one(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock,
                              FakeResponse(400, {"error": {"message": "max_completion_tokens is not supported, use max_tokens"}}), ok(reply()))
    assert extractor.extract(TITLE, TEXT) is not None
    assert "max_completion_tokens" in session.calls[0]["json"] and session.calls[1]["json"]["max_tokens"] == 900
    assert "max_completion_tokens" not in session.calls[1]["json"]


def test_capabilities_are_kept_per_model(cfg, store, resolver, fake_clock):
    extractor, session = make(cfg, store, resolver, fake_clock,
                              FakeResponse(400, {"error": {"message": "response_format json_schema unsupported"}}), ok(reply()))
    extractor.extract(TITLE, TEXT)
    assert extractor._caps(PRIMARY)["fmt"] == "json_object" and extractor._caps(FALLBACK)["fmt"] == "json_schema"


# ----------------------------------------------------------------------------- budgets and counters
def test_the_wall_clock_budget_stops_a_run_without_counting_a_failure(cfg, store, resolver, fake_clock):
    tight = dataclasses.replace(cfg, llm_max_seconds_per_run=10)
    extractor, session = make(tight, store, resolver, fake_clock, FakeResponse(503), ok(reply()), backoff_base=20.0, backoff_cap=20.0)
    extractor.start_run()
    assert extractor.extract(TITLE, TEXT) is None
    assert extractor.run_failures == 0 and extractor.run_calls == 0 and extractor.run_budget_skipped == 1 and extractor.tripped
    assert fake_clock.sleeps == [] and len(session.calls) == 1  # it did not sleep past its budget


def test_an_exhausted_time_budget_skips_new_items(cfg, store, resolver, fake_clock):
    tight = dataclasses.replace(cfg, llm_max_seconds_per_run=30)
    extractor, session = make(tight, store, resolver, fake_clock)
    extractor.start_run()
    fake_clock.advance(31)
    assert extractor.out_of_time and extractor.extract(TITLE, TEXT) is None
    assert extractor.run_budget_skipped == 1 and session.calls == []


def test_without_a_run_there_is_no_wall_clock_budget(cfg, store, resolver, fake_clock):
    extractor, _ = make(dataclasses.replace(cfg, llm_max_seconds_per_run=10), store, resolver, fake_clock)
    fake_clock.advance(10_000)
    assert extractor.out_of_time is False and extractor.elapsed() == 0.0


def test_token_usage_is_counted_per_run_and_in_the_database(cfg, store, resolver, fake_clock):
    extractor, _ = make(cfg, store, resolver, fake_clock, with_usage(ok(reply())), with_usage(ok(reply()), 100, 50))
    extractor.extract(TITLE, TEXT)
    extractor.extract(TITLE + " b", TEXT)
    assert (extractor.run_tokens_in, extractor.run_tokens_out) == (510, 200)
    assert store.kv_get("llm.tokens_in") == "510" and store.kv_get("llm.tokens_out") == "200"
    extractor.start_run()
    assert (extractor.run_tokens_in, extractor.run_tokens_out) == (0, 0)


def test_a_response_without_usage_counts_nothing(cfg, store, resolver, fake_clock):
    extractor, _ = make(cfg, store, resolver, fake_clock)
    extractor.extract(TITLE, TEXT)
    assert store.kv_get("llm.tokens_in") is None


def test_the_groq_key_is_registered_for_redaction_and_never_logged(cfg, store, resolver, fake_clock, caplog):
    from app.logging_setup import redact

    leaky = FakeResponse(500, {"error": f"bad key {GROQ_KEY}"}, text=f"bad key {GROQ_KEY}")
    extractor, _ = make(cfg, store, resolver, fake_clock, leaky)
    with caplog.at_level("DEBUG"):
        extractor.extract(TITLE, TEXT)
    assert GROQ_KEY not in caplog.text and GROQ_KEY not in redact(f"key={GROQ_KEY}")


def test_the_caps_ttl_is_a_week_and_the_cache_ttl_two_weeks():
    assert llm_mod.CAPS_TTL_DAYS == 7 and timedelta(days=llm_mod.CACHE_TTL_DAYS) == timedelta(days=14)
