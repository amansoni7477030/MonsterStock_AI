"""Jev judge: request shape, strict response validation, conservative merge into the rule draft, and the service behaviour."""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from news_intelligence import jev as jev_mod
from news_intelligence.events import EVENT_TYPES, EventDraft
from news_intelligence.jev import (
    EVENT_DESCRIPTIONS,
    POLARITY_CRITERIA,
    SEVERITY_LEVELS,
    JevJudge,
    JevJudgment,
    build_request,
    merge_jev,
    parse_response,
    prune_jev_cache,
)
from tests.test_news_support import FakeResponse, FakeSession
from tests.test_news_support import jev_answers as answers
from tests.test_news_support import jev_response as response

KEY = "ts_test_typesafe_key_1234567890"
TITLE = "Reliance bags Rs 5,000 crore NTPC order"
TEXT = "Reliance Industries has bagged a Rs 5,000 crore order from NTPC for supply of solar modules over three years."


@pytest.fixture()
def cfg(settings):
    return dataclasses.replace(settings, typesafe_api_key=KEY)


def make(cfg, store, clock, *responses, **kwargs):
    session = FakeSession(*(responses or [response()]))
    return JevJudge(cfg, session=session, clock=clock, store=store, **kwargs), session


# ----------------------------------------------------------------------------------- the request
def test_every_event_type_is_described_and_the_rubrics_fit_the_api_limits():
    assert list(EVENT_DESCRIPTIONS) == list(EVENT_TYPES)  # the taxonomy and what Jev is told can never drift apart
    assert all(text.strip() for text in EVENT_DESCRIPTIONS.values())
    assert len(EVENT_DESCRIPTIONS) <= 255 and len(SEVERITY_LEVELS) <= 10  # the API's limits for choices and score levels
    assert set(POLARITY_CRITERIA) == {"positive", "negative", "neutral"}


def test_the_request_follows_the_systemone_contract(cfg, store, fake_clock):
    judge, session = make(cfg, store, fake_clock)
    assert judge.judge(TITLE, TEXT, company="Reliance Industries Limited", source_kind="media") is not None
    call = session.calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone" and call["timeout"] == 20.0
    assert call["headers"]["Authorization"] == "Bearer " + KEY and call["headers"]["Content-Type"] == "application/json"
    body = call["json"]
    assert body["model"] == "jev-1.14.0"
    assert body["state"] == {"headline": TITLE, "article": TEXT, "source_type": "news media",
                             "company_in_focus": "Reliance Industries Limited"}
    questions = body["questions"]
    assert list(questions) == ["event_type", "rumour", "denial", "severity", "polarity"]
    assert questions["event_type"]["type"] == "choice" and questions["event_type"]["criteria"] == EVENT_DESCRIPTIONS
    assert questions["rumour"]["type"] == "noul" and questions["denial"]["type"] == "noul"
    assert questions["severity"] == {"type": "score", "instructions": jev_mod.SEVERITY_QUESTION, "criteria": list(SEVERITY_LEVELS)}
    assert questions["polarity"]["criteria"] == POLARITY_CRITERIA
    assert KEY not in json.dumps(body)


def test_without_a_company_there_is_no_polarity_question_and_no_company_in_the_state():
    body = build_request("jev-latest", "t", "x" * 9000)
    assert "polarity" not in body["questions"] and "company_in_focus" not in body["state"]
    assert len(body["state"]["article"]) == jev_mod.MAX_ARTICLE_CHARS and "source_type" not in body["state"]
    assert len(build_request("m", "h" * 1000, "")["state"]["headline"]) == jev_mod.MAX_HEADLINE_CHARS
    assert "source_type" not in build_request("m", "t", "x", source_kind="unknown")["state"]


def test_the_model_id_is_pinned_by_default_and_can_be_the_alias(cfg, store, fake_clock):
    assert cfg.jev_model == "jev-1.14.0"
    judge, session = make(dataclasses.replace(cfg, jev_model="jev-latest"), store, fake_clock)
    judge.judge(TITLE, TEXT)
    assert session.calls[0]["json"]["model"] == "jev-latest"


# ---------------------------------------------------------------------------------- the response
def test_a_valid_response_becomes_a_judgment_with_scaled_severity_and_net_polarity():
    judgment = parse_response({"model": "jev-1.14.0", "answers": answers()}, company_in_focus=True)
    assert (judgment.event_type, judgment.event_prob, judgment.event_confidence) == ("order_win", 0.82, 0.78)
    assert (judgment.rumour, judgment.denial) == (0.05, 0.02)
    assert judgment.severity == 0.8 and judgment.severity_confidence == 0.7  # level 3.2 of 0..4
    assert judgment.polarity == 0.6 and judgment.polarity_confidence == 0.6  # P(good) - P(bad)
    assert judgment.model == "jev-1.14.0" and judgment.cached is False


def test_without_a_company_the_polarity_answer_is_not_required():
    judgment = parse_response({"answers": answers(company=False)}, company_in_focus=False)
    assert judgment.polarity is None and judgment.polarity_confidence is None and judgment.model == ""


@pytest.mark.parametrize("break_it", [
    lambda a: a.pop("event_type"),
    lambda a: a.pop("rumour"),
    lambda a: a.pop("severity"),
    lambda a: a.pop("polarity"),
    lambda a: a["event_type"].update(type="noul"),
    lambda a: a["event_type"].update(choice="not_a_type"),
    lambda a: a["event_type"].update(choice=7),
    lambda a: a["event_type"].update(probabilities="nope"),
    lambda a: a["event_type"]["probabilities"].update(order_win="high"),
    lambda a: a["event_type"].update(confidence=float("nan")),
    lambda a: a["rumour"].update(noul=None),
    lambda a: a["denial"].update(noul="0.2x"),
    lambda a: a["severity"].update(score=float("inf")),
    lambda a: a["severity"].update(confidence=None),
    lambda a: a["polarity"].update(choice="sideways"),
    lambda a: a["polarity"]["probabilities"].pop("negative"),
], ids=range(16))
def test_an_answer_that_is_not_exactly_what_was_asked_is_rejected(break_it):
    broken = answers()
    break_it(broken)
    with pytest.raises(ValueError):
        parse_response({"answers": broken}, company_in_focus=True)


@pytest.mark.parametrize("payload", [None, [], "x", {}, {"answers": None}, {"answers": []}, {"nope": 1}])
def test_a_response_without_answers_is_rejected(payload):
    with pytest.raises(ValueError):
        parse_response(payload, company_in_focus=False)


def test_out_of_range_numbers_are_clamped_not_trusted():
    judgment = parse_response({"answers": answers(rumour=1.7, denial=-0.3, level=9, polarity=(1.4, -0.2, 0))}, company_in_focus=True)
    assert judgment.rumour == 1.0 and judgment.denial == 0.0 and judgment.severity == 1.0 and judgment.polarity == 1.0


def test_a_judgment_round_trips_through_the_cache_and_malformed_entries_are_ignored():
    original = parse_response({"model": "jev-1.14.0", "answers": answers()}, company_in_focus=True)
    assert JevJudgment.from_dict(original.to_dict(), cached=True) == dataclasses.replace(original, cached=True)
    for bad in (None, [], {}, {"event_type": "bogus"}, {**original.to_dict(), "rumour": "x"}, {**original.to_dict(), "severity": None}):
        assert JevJudgment.from_dict(bad) is None
    no_company = parse_response({"answers": answers(company=False)}, company_in_focus=False)
    assert JevJudgment.from_dict(no_company.to_dict()).polarity is None


# ------------------------------------------------------------------------------------------- merge
def draft(**over) -> EventDraft:
    base = dict(event_type="other", severity=0.15, confidence=0.3, horizon="days", polarity_hint=0.4, scope="company")
    base.update(over)
    return EventDraft(**base)


def judgment(**over) -> JevJudgment:
    base = dict(event_type="order_win", event_prob=0.82, event_confidence=0.78, rumour=0.05, denial=0.02, severity=0.8,
                severity_confidence=0.7, polarity=0.6, polarity_confidence=0.6, model="jev-1.14.0")
    base.update(over)
    return JevJudgment(**base)


def merged(d, j, *, min_confidence=0.5, official=False):
    return merge_jev(d, j, min_confidence=min_confidence, official=official)


def test_a_confident_type_replaces_an_unclassified_draft_and_resets_what_depended_on_the_old_type():
    d = draft()
    effect = merged(d, judgment())
    assert effect.retyped and d.event_type == "order_win" and d.scope == EVENT_TYPES["order_win"].scope
    assert d.polarity_hint == 0.0 and d.confidence == 0.82


def test_agreement_lifts_the_rules_confidence_but_never_lowers_it():
    d = draft(event_type="order_win", confidence=0.5)
    assert merged(d, judgment()).retyped is False and d.confidence == 0.82
    d = draft(event_type="order_win", confidence=0.9)
    merged(d, judgment())
    assert d.confidence == 0.9


@pytest.mark.parametrize(("rule_conf", "jev_p", "expected"), [
    (0.5, 0.82, "order_win"),   # the rules are unsure and Jev is surer
    (0.5, 0.55, "earnings_result"),  # Jev below the 0.6 floor: the rules stay
    (0.8, 0.95, "earnings_result"),  # the rules are sure (e.g. an exchange category): they keep their type
    (0.7, 0.65, "earnings_result"),  # Jev is not surer than the rules
])
def test_a_different_type_wins_only_when_jev_is_confident_and_the_rules_are_not(rule_conf, jev_p, expected):
    d = draft(event_type="earnings_result", confidence=rule_conf)
    merged(d, judgment(event_prob=jev_p))
    assert d.event_type == expected


def test_jev_saying_other_or_low_confidence_never_changes_the_type():
    d = draft()
    assert merged(d, judgment(event_type="other")).retyped is False and d.event_type == "other"
    assert merged(d, judgment(event_confidence=0.49)).retyped is False and d.event_type == "other"


def test_a_rumour_is_flagged_when_jev_is_sure_never_for_official_sources_and_never_cleared():
    d = draft(event_type="order_win", confidence=0.8, polarity_hint=0.6)
    effect = merged(d, judgment(rumour=0.8))
    assert effect.rumour_flagged and d.is_rumour and d.confidence == 0.656 and d.polarity_hint == 0.3  # 0.82 (agreement) x 0.8
    official = draft()
    assert merged(official, judgment(rumour=0.99), official=True).rumour_flagged is False and not official.is_rumour
    already = draft(is_rumour=True)
    merged(already, judgment(rumour=0.0))
    assert already.is_rumour  # Jev's "not a rumour" never clears what the rules saw
    unsure = draft()
    assert merged(unsure, judgment(rumour=0.7)).rumour_flagged is False  # needs p >= 0.75 at the default confidence of 0.5


def test_a_denial_is_flagged_and_replaces_the_rumour_flag():
    d = draft(event_type="order_win", confidence=0.8, is_rumour=True, polarity_hint=0.3)  # the rules already halved a rumour's
    effect = merged(d, judgment(denial=0.9, rumour=0.9))
    assert effect.denial_flagged and d.is_denial and not d.is_rumour and d.polarity_hint == 0.3  # never halved twice
    fresh = draft(event_type="order_win", confidence=0.8, polarity_hint=0.6)
    merged(fresh, judgment(denial=0.9))
    assert fresh.is_denial and fresh.polarity_hint == 0.3
    assert merged(draft(), judgment(denial=0.9), official=True).denial_flagged is False
    both = draft()
    merged(both, judgment(rumour=0.9, denial=0.9))
    assert both.is_denial and not both.is_rumour  # a denial outranks the rumour it denies


def test_a_flagged_rumour_or_denial_gets_the_rules_own_severity_discount():
    quiet = {"severity_confidence": 0.0}  # keeps the severity blend out of the arithmetic
    rumour = draft(severity=0.4)
    merged(rumour, judgment(rumour=0.9, **quiet))
    assert rumour.severity == 0.34  # x 0.85, as events._severity does
    denial = draft(severity=0.4)
    merged(denial, judgment(denial=0.9, **quiet))
    assert denial.severity == 0.2  # x 0.5
    was_rumour = draft(severity=0.4, is_rumour=True)  # already trimmed by 0.85: only the difference is applied
    merged(was_rumour, judgment(denial=0.9, **quiet))
    assert was_rumour.severity == 0.235


def test_a_direction_for_a_rumour_or_a_denial_is_halved():
    assert merged(draft(), judgment(polarity=0.8)).polarity == 0.8
    assert merged(draft(is_rumour=True), judgment(polarity=0.8)).polarity == 0.4  # the rules saw the rumour
    assert merged(draft(), judgment(polarity=0.8, rumour=0.9)).polarity == 0.4  # Jev saw it
    assert merged(draft(), judgment(polarity=-0.8, denial=0.9)).polarity == -0.4


def test_the_yes_threshold_follows_the_confidence_setting():
    """A Noul has no confidence of its own: |2p - 1| >= JEV_MIN_CONFIDENCE, i.e. p >= 0.5 + setting / 2."""
    assert merged(draft(), judgment(rumour=0.62), min_confidence=0.2).rumour_flagged is True
    assert merged(draft(), judgment(rumour=0.62), min_confidence=0.5).rumour_flagged is False
    assert merged(draft(), judgment(rumour=0.95), min_confidence=0.95).rumour_flagged is False


def test_severity_is_averaged_with_the_rules_only_when_jev_is_confident():
    d = draft(severity=0.4)
    assert merged(d, judgment(severity=0.8)).severity_blended and d.severity == 0.6
    d = draft(severity=0.4)
    assert merged(d, judgment(severity=0.8, severity_confidence=0.3)).severity_blended is False and d.severity == 0.4


@pytest.mark.parametrize(("polarity", "conf", "expected"), [
    (0.6, 0.6, 0.6), (-0.7, 0.9, -0.7), (0.1, 0.9, None), (-0.15, 0.9, None), (0.8, 0.3, None), (None, None, None),
])
def test_a_direction_is_offered_only_when_clear_and_confident(polarity, conf, expected):
    assert merged(draft(), judgment(polarity=polarity, polarity_confidence=conf)).polarity == expected


# -------------------------------------------------------------------------------------- the service
def test_a_judgment_is_cached_by_content_and_served_without_a_request(cfg, store, fake_clock):
    judge, session = make(cfg, store, fake_clock)
    first = judge.judge(TITLE, TEXT, company="Reliance")
    again = judge.judge("  " + TITLE.upper() + " ", TEXT.lower(), company="Reliance")
    assert first.cached is False and again.cached is True and again.event_type == first.event_type
    assert len(session.calls) == 1 and judge.run_cache_hits == 1 and store.kv_get("jev.calls") == "1"
    other_company, _ = judge.judge(TITLE, TEXT, company="NTPC"), None
    assert other_company.cached is False and len(session.calls) == 2  # the company in focus is part of the question
    assert judge.judge(TITLE, TEXT, company="Reliance", source_kind="media").cached is False  # so is the source type


def test_the_cache_expires_after_two_weeks_and_prunes_only_its_own_keys(cfg, store, fake_clock):
    judge, session = make(cfg, store, fake_clock)
    judge.judge(TITLE, TEXT)
    store.kv_set("jev:stale", "{}")
    store.kv_set("jevx:other", "keep")
    store.kv_set("llm:other", "keep")
    fake_clock.advance(jev_mod.CACHE_TTL_DAYS * 86400 + 60)
    assert judge.judge(TITLE, TEXT).cached is False and len(session.calls) == 2  # expired: asked again
    store.kv_set("jev:fresh", "{}")
    assert prune_jev_cache(store, fake_clock.now()) == 1  # only jev:stale: the refreshed entry and jev:fresh are young
    assert store.kv_get("jev:stale") is None and store.kv_get("jev:fresh") == "{}"
    assert store.kv_get("jevx:other") == "keep" and store.kv_get("llm:other") == "keep"


def test_the_per_run_budget_caps_requests_and_cache_hits_are_free(cfg, store, fake_clock):
    judge, session = make(dataclasses.replace(cfg, jev_max_items_per_run=2), store, fake_clock)
    results = [judge.judge(f"title {i}", TEXT) for i in range(4)]
    assert [r is not None for r in results] == [True, True, False, False]
    assert len(session.calls) == 2 and judge.run_budget_skipped == 2
    assert judge.judge("title 0", TEXT).cached is True
    judge.start_run()
    assert judge.judge("title 2", TEXT) is not None and judge.run_calls == 1


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_bad_key_or_endpoint_opens_the_circuit_at_once(cfg, store, fake_clock, status):
    judge, session = make(cfg, store, fake_clock, FakeResponse(status))
    assert judge.judge(TITLE, TEXT) is None and judge.tripped
    assert judge.judge(TITLE + " 2", TEXT) is None and len(session.calls) == 1 and fake_clock.sleeps == []
    assert store.kv_get("jev.failures") == "1"


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_retryable_statuses_back_off_and_then_succeed(cfg, store, fake_clock, status):
    judge, session = make(cfg, store, fake_clock, FakeResponse(status), response())
    assert judge.judge(TITLE, TEXT) is not None and len(session.calls) == 2 and fake_clock.sleeps == [1.0]


def test_retry_after_is_honoured_up_to_thirty_seconds(cfg, store, fake_clock):
    judge, _ = make(cfg, store, fake_clock, FakeResponse(429, headers={"Retry-After": "12"}), response())
    assert judge.judge(TITLE, TEXT) is not None and fake_clock.sleeps == [12.0]


def test_a_long_retry_after_is_a_quota_that_starts_a_cooldown_for_the_next_run(cfg, store, fake_clock):
    judge, session = make(cfg, store, fake_clock, FakeResponse(429, headers={"retry-after": "5400"}), response())
    assert judge.judge(TITLE, TEXT) is None and judge.tripped and fake_clock.sleeps == [] and len(session.calls) == 1
    assert store.kv_get("jev.cooldown_until")
    nxt, session = make(cfg, store, fake_clock, response())
    assert nxt.judge(TITLE + " b", TEXT) is None and session.calls == []
    fake_clock.advance(5400 + 5)
    after, session = make(cfg, store, fake_clock, response())
    assert after.judge(TITLE + " b", TEXT) is not None and len(session.calls) == 1


def test_a_client_error_fails_the_item_without_retrying_and_five_failures_open_the_circuit(cfg, store, fake_clock):
    judge, session = make(cfg, store, fake_clock, FakeResponse(422, {"detail": "bad question"}), max_consecutive_failures=2)
    assert judge.judge("a1", TEXT) is None and judge.tripped is False
    assert judge.judge("a2", TEXT) is None and judge.tripped is True
    assert judge.judge("a3", TEXT) is None and len(session.calls) == 2 and fake_clock.sleeps == []


def test_a_success_resets_the_consecutive_failure_count(cfg, store, fake_clock):
    judge, _ = make(cfg, store, fake_clock, FakeResponse(422), response(), FakeResponse(422), max_consecutive_failures=2)
    assert judge.judge("b1", TEXT) is None and judge.judge("b2", TEXT) is not None and judge.consecutive_failures == 0
    assert judge.judge("b3", TEXT) is None and judge.tripped is False


@pytest.mark.parametrize("bad", [FakeResponse(200, None, json_error=True), FakeResponse(200, {"answers": {}}),
                                 FakeResponse(200, {"model": "x", "answers": {"event_type": {"type": "choice", "choice": "zzz"}}}),
                                 FakeResponse(200, "text")])
def test_a_malformed_response_is_a_failure_that_falls_back_to_the_rules(cfg, store, fake_clock, bad):
    judge, _ = make(cfg, store, fake_clock, bad)
    assert judge.judge(TITLE, TEXT) is None and store.kv_get("jev.failures") == "1" and judge.run_failures == 1


def test_transport_errors_are_retried_and_giving_up_returns_none(cfg, store, fake_clock, caplog):
    import requests

    judge, session = make(cfg, store, fake_clock, requests.Timeout("slow"), requests.ConnectionError("down"), response())
    assert judge.judge(TITLE, TEXT) is not None and len(session.calls) == 3 and len(fake_clock.sleeps) == 2
    dead, session = make(cfg, store, fake_clock, requests.Timeout("slow"))
    with caplog.at_level(logging.WARNING):
        assert dead.judge(TITLE + " dead", TEXT) is None
    assert len(session.calls) == 3 and "gave up" in caplog.text


def test_the_wall_clock_budget_ends_a_run_without_counting_a_failure(cfg, store, fake_clock):
    judge, session = make(cfg, store, fake_clock, FakeResponse(503), response(), backoff_base=500.0, backoff_cap=500.0)
    judge.start_run()
    assert judge.judge(TITLE, TEXT) is None
    assert judge.run_failures == 0 and judge.run_calls == 0 and judge.run_budget_skipped == 1 and judge.tripped
    assert fake_clock.sleeps == []


def test_token_usage_is_counted(cfg, store, fake_clock):
    judge, _ = make(cfg, store, fake_clock, response(usage=(300, 20)), response(usage=(100, 10)))
    judge.judge("t1", TEXT)
    judge.judge("t2", TEXT)
    assert (judge.run_tokens_in, judge.run_tokens_out) == (400, 30)
    assert store.kv_get("jev.tokens_in") == "400" and store.kv_get("jev.tokens_out") == "30"


def test_unconfigured_or_empty_requests_do_nothing(settings, cfg, store, fake_clock):
    session = FakeSession(response())
    off = JevJudge(settings, session=session, clock=fake_clock, store=store)
    assert off.configured is False and off.judge(TITLE, TEXT) is None and session.calls == []
    on = JevJudge(cfg, session=session, clock=fake_clock, store=store)
    assert on.judge("", "") is None and on.judge(None, None) is None and session.calls == []
    assert on.judge("", TEXT) is not None  # text alone is enough


def test_the_api_key_never_appears_in_logs_the_body_or_a_reason(cfg, store, fake_clock, caplog):
    leaky = FakeResponse(500, {"detail": f"invalid key {KEY}"}, text=f"bad key {KEY}")
    judge, session = make(cfg, store, fake_clock, leaky)
    with caplog.at_level(logging.DEBUG):
        judge.judge(TITLE, TEXT)
    assert KEY not in caplog.text and all(KEY not in json.dumps(c["json"]) for c in session.calls)


def test_the_session_is_created_lazily(cfg, store, fake_clock, monkeypatch):
    created = []

    class Lazy(FakeSession):
        def __init__(self):
            super().__init__(response())
            created.append(self)

    monkeypatch.setattr(jev_mod.requests, "Session", Lazy)
    judge = JevJudge(cfg, clock=fake_clock, store=store)
    assert judge.judge(TITLE, TEXT) is not None and len(created) == 1


def test_bind_attaches_the_store_once(cfg, store, fake_clock):
    judge = JevJudge(cfg, session=FakeSession(response()), clock=fake_clock)
    assert judge.judge(TITLE, TEXT) is not None  # works without a store: no cache, no counters
    judge.bind(store=store, resolver=object())
    assert judge.store is store
    other = object()
    judge.bind(store=None, resolver=other)
    assert judge.resolver is not other  # never replaced once set
