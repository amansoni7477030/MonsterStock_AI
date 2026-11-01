"""``ai-test`` self-check: what it sends, how it reads the answer, and the remedy it names for each failure. No real network."""

from __future__ import annotations

import dataclasses
import json

import pytest
import requests

from news_intelligence import ai_check
from news_intelligence.ai_check import SAMPLE_TEXT, SAMPLE_TITLE, check_jev, check_llm, run_checks
from news_intelligence.entities import EntityResolver
from tests.test_news_support import INSTRUMENTS, FakeResponse, FakeSession, add_instruments, jev_response, ok

GROQ_KEY = "gsk_test_groq_key_1234567890"
JEV_KEY = "ts_test_typesafe_key_1234567890"
PRIMARY, FALLBACK = "openai/gpt-oss-120b", "openai/gpt-oss-20b"
QUOTE = "received a Rs 5,000 crore order from NTPC"


def reply(**over) -> str:
    data = {"event_type": "order_win", "entities": [{"id": "RELIANCE", "name": None, "type": "company", "polarity": 0.8}],
            "severity": 0.6, "horizon": "days", "event_time": None, "confidence": 0.85, "evidence_quote": QUOTE}
    data.update(over)
    return json.dumps(data)


def groq_models(*names):
    return FakeResponse(200, {"object": "list", "data": [{"id": name} for name in (names or (PRIMARY, FALLBACK))]})


def jev_models(*names):
    return FakeResponse(200, {"models": [{"name": name, "description": "x", "release_date": "2026-09-15"}
                                        for name in (names or ("jev-1.14.0", "jev-latest"))]})


@pytest.fixture()
def groq(settings):
    return dataclasses.replace(settings, groq_api_key=GROQ_KEY)


@pytest.fixture()
def jev(settings):
    return dataclasses.replace(settings, typesafe_api_key=JEV_KEY)


@pytest.fixture()
def resolver(store):
    add_instruments(store, INSTRUMENTS)
    return EntityResolver(store)


# --------------------------------------------------------------------------------------- not configured
def test_unconfigured_services_are_off_and_say_which_keys_switch_them_on(settings):
    llm, jev_check = run_checks(settings)
    assert (llm.status, jev_check.status) == ("off", "off") and not llm.usable and not jev_check.usable
    assert "GROQ_API_KEY" in llm.detail and "TYPESAFE_API_KEY" in jev_check.detail


def test_only_the_wanted_service_is_checked_and_the_order_is_fixed(settings):
    assert [c.service for c in run_checks(settings, only=["jev"])] == ["jev"]
    assert [c.service for c in run_checks(settings)] == ["llm", "jev"]


# ----------------------------------------------------------------------------------------------- Groq
def test_a_healthy_groq_check_lists_the_models_then_sends_the_sample(groq, resolver, fake_clock):
    session = FakeSession(ok(reply()), gets=[groq_models()])
    check = check_llm(groq, session=session, clock=fake_clock, resolver=resolver)
    assert check.status == "ok" and check.usable and check.label == "Groq LLM"
    assert f"model {PRIMARY}: order_win" in check.detail and "RELIANCE +0.8" in check.detail
    listing, request = session.calls
    assert listing["method"] == "GET" and listing["url"] == "https://api.groq.com/openai/v1/models"
    assert listing["headers"]["Authorization"] == "Bearer " + GROQ_KEY
    assert request["url"].endswith("/chat/completions") and SAMPLE_TITLE in json.dumps(request["json"])
    assert GROQ_KEY not in json.dumps(check.to_dict())


def test_nothing_is_cached_counted_or_written_by_a_check(groq, store, resolver, fake_clock):
    check_llm(groq, session=FakeSession(ok(reply()), gets=[groq_models()]), clock=fake_clock, resolver=resolver)
    assert store.conn.execute("SELECT COUNT(*) FROM kv WHERE key LIKE 'llm%'").fetchone()[0] == 0


def test_the_model_that_actually_answered_is_named_when_the_fallback_took_over(groq, fake_clock):
    session = FakeSession(FakeResponse(429, headers={"retry-after": "1"}), ok(reply()), gets=[groq_models()])
    check = check_llm(groq, session=session, clock=fake_clock)
    assert check.status == "ok" and f"model {FALLBACK}:" in check.detail
    assert "none resolved" in check.detail  # without a database the entities cannot be resolved: not an error


def test_a_rejected_key_fails_at_the_listing_before_any_tokens_are_spent(groq, fake_clock):
    session = FakeSession(ok(reply()), gets=[FakeResponse(401)])
    check = check_llm(groq, session=session, clock=fake_clock)
    assert check.status == "fail" and "rejected" in check.detail and "GROQ_API_KEY" in check.detail
    assert len(session.calls) == 1  # only the free listing was called


def test_an_unreachable_service_fails_with_the_exception_type_only(groq, fake_clock):
    session = FakeSession(ok(reply()), gets=[requests.ConnectionError("dns failure for api.groq.com?key=" + GROQ_KEY)])
    check = check_llm(groq, session=session, clock=fake_clock)
    assert check.status == "fail" and check.detail == "cannot reach the service (ConnectionError)"


def test_a_configured_model_missing_from_the_listing_is_a_warning_not_a_failure(groq, fake_clock):
    check = check_llm(groq, session=FakeSession(ok(reply()), gets=[groq_models(FALLBACK)]), clock=fake_clock)
    assert check.status == "warn" and check.usable and f"not listed for this key: {PRIMARY}" in check.detail


def test_a_provider_without_a_model_list_is_still_checked_with_a_real_request(groq, fake_clock):
    check = check_llm(groq, session=FakeSession(ok(reply()), gets=[FakeResponse(404)]), clock=fake_clock)
    assert check.status == "warn" and "model list unavailable (HTTP 404)" in check.detail and "order_win" in check.detail


def test_list_only_spends_no_tokens(groq, fake_clock):
    session = FakeSession(ok(reply()), gets=[groq_models()])
    check = check_llm(groq, deep=False, session=session, clock=fake_clock)
    assert check.status == "ok" and check.detail == "key accepted" and len(session.calls) == 1


@pytest.mark.parametrize(("response", "remedy"), [
    (FakeResponse(401), "the key was rejected, check GROQ_API_KEY"),
    (FakeResponse(404), "check the model name"),
    (FakeResponse(429, headers={"retry-after": "3600"}), "quota is used up"),
])
def test_a_failed_request_names_the_remedy(groq, fake_clock, response, remedy):
    check = check_llm(groq, session=FakeSession(response, gets=[groq_models()]), clock=fake_clock)
    assert check.status == "fail" and remedy in check.detail


def test_an_answer_the_pipeline_would_ignore_is_a_warning(groq, fake_clock):
    session = FakeSession(ok(reply(evidence_quote="a sentence that is not in the text")), gets=[groq_models()])
    check = check_llm(groq, session=session, clock=fake_clock)
    assert check.status == "warn" and "evidence quote was not found" in check.detail
    wrong = check_llm(groq, session=FakeSession(ok(reply(event_type="litigation_regulatory")), gets=[groq_models()]), clock=fake_clock)
    assert wrong.status == "warn" and "expected order_win" in wrong.detail


def test_another_openai_compatible_provider_is_checked_against_its_own_endpoint(settings, fake_clock):
    cfg = dataclasses.replace(settings, llm_api_base="https://llm.example/v1/chat/completions", llm_model="m", llm_api_key="sk-custom-1234567890")
    session = FakeSession(ok(reply()), gets=[FakeResponse(200, {"data": [{"id": "m"}]})])
    check = check_llm(cfg, session=session, clock=fake_clock)
    assert check.status == "ok" and check.label == "LLM (custom provider)"
    assert session.calls[0]["url"] == "https://llm.example/v1/models" and "LLM_API_KEY" not in check.detail


# ----------------------------------------------------------------------------------------------- Jev
def test_a_healthy_jev_check_lists_the_models_then_asks_the_five_questions(jev, fake_clock):
    session = FakeSession(jev_response(), gets=[jev_models()])
    check = check_jev(jev, session=session, clock=fake_clock)
    assert check.status == "ok" and check.label == "Jev (TypeSafe AI)"
    assert "model jev-1.14.0: order_win (p 0.82" in check.detail and "direction +0.60" in check.detail
    listing, request = session.calls
    assert listing["url"] == "https://api.typesafe.ai/v1/models" and listing["headers"]["Authorization"] == "Bearer " + JEV_KEY
    assert request["url"] == "https://api.typesafe.ai/v1/systemone"
    assert request["json"]["state"]["company_in_focus"] == "Reliance Industries Limited"
    assert set(request["json"]["questions"]) == {"event_type", "rumour", "denial", "severity", "polarity"}
    assert JEV_KEY not in json.dumps(check.to_dict())


def test_a_pinned_jev_model_the_account_does_not_list_is_a_warning(jev, fake_clock):
    check = check_jev(jev, session=FakeSession(jev_response(), gets=[jev_models("jev-latest")]), clock=fake_clock)
    assert check.status == "warn" and "not listed for this key: jev-1.14.0" in check.detail


def test_a_rejected_jev_key_fails_at_the_listing(jev, fake_clock):
    session = FakeSession(jev_response(), gets=[FakeResponse(401)])
    check = check_jev(jev, session=session, clock=fake_clock)
    assert check.status == "fail" and "TYPESAFE_API_KEY" in check.detail and len(session.calls) == 1


def test_a_jev_request_failure_and_an_unexpected_answer_are_reported(jev, fake_clock):
    refused = check_jev(jev, session=FakeSession(FakeResponse(422, {"detail": "bad"}), gets=[jev_models()]), clock=fake_clock)
    assert refused.status == "fail" and "422" in refused.detail
    odd = check_jev(jev, session=FakeSession(jev_response(event="litigation_regulatory"), gets=[jev_models()]), clock=fake_clock)
    assert odd.status == "warn" and "expected order_win" in odd.detail


def test_jev_list_only_spends_no_tokens(jev, fake_clock):
    session = FakeSession(jev_response(), gets=[jev_models()])
    check = check_jev(jev, deep=False, session=session, clock=fake_clock)
    assert check.status == "ok" and check.detail == "key accepted" and len(session.calls) == 1


# ------------------------------------------------------------------------------------------- the rest
def test_the_sample_is_an_unambiguous_order_win_that_names_a_company():
    assert "Reliance" in SAMPLE_TITLE and QUOTE in SAMPLE_TEXT and ai_check.SAMPLE_EVENT == "order_win"


def test_to_dict_rounds_the_seconds_and_keeps_every_field():
    check = ai_check.AiCheck("llm", "Groq LLM", "ok", "fine", 1.23456)
    assert check.to_dict() == {"service": "llm", "label": "Groq LLM", "status": "ok", "detail": "fine", "seconds": 1.23}
    assert ai_check.AiCheck("jev", "Jev", "off", "x").to_dict()["seconds"] is None
