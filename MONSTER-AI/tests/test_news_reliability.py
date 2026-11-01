"""Reliability of the news pipeline: failed items are retried (boundedly) instead of silently lost, a known market reaction is never
erased, unverified LLM claims leave no trace, and the event metadata the blueprint asks for (novelty class, rumour flag, source
class, independent sources, conflicting reports) is persisted and point-in-time."""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3

import pandas as pd
import pytest

from news_intelligence import features as feat
from news_intelligence import pipeline as pl
from news_intelligence.pipeline import NewsPipeline
from news_intelligence.source_quality import source_class_of
from ranking.types import Catalyst
from tests.test_news_features import _seed_bars, add, ent
from tests.test_news_pipeline import BODY, QUOTE, TITLE, count, llm_pipe, llm_reply, story
from tests.test_news_support import add_article, add_filing, add_instruments, events, ok

ORDER_FILING = {"category": "Award of Order / Receipt of Order", "subject": "TCS - Award of order",
                "text": "TCS has received an order worth Rs 900 crore."}
RUMOUR = ("Reliance in talks to acquire stake in Wipro, sources said",
          "Reliance Industries is in talks to acquire a stake in Wipro, sources said. The talks are at an early stage.")


@pytest.fixture()
def pipe(store, settings, fake_clock):
    add_instruments(store)
    return NewsPipeline(store, settings, clock=fake_clock)


@pytest.fixture()
def llm_settings(settings):
    return dataclasses.replace(settings, llm_api_base="https://llm.example/v1", llm_model="m", llm_api_key="sk-x")


def explode_on(pipe, monkeypatch, *ids, attr="_prepare_article", key="article_id"):
    """Make the preparation of the given rows raise (a poison item); returns the list of ids that were attempted."""
    original = getattr(pipe, attr)
    attempted: list[int] = []

    def maybe(row, stats):
        if row[key] in ids:
            attempted.append(row[key])
            raise RuntimeError("boom")
        return original(row, stats)

    monkeypatch.setattr(pipe, attr, maybe)
    return attempted


def pending(store, table="news_raw", where="processed = 0"):
    return count(store, table, where)


# ------------------------------------------------------------------------------- N1: bounded retry, never silent loss
def test_a_failing_item_stays_unprocessed_and_is_retried_until_the_attempt_limit(store, pipe, settings, fake_clock, monkeypatch,
                                                                               caplog):
    story(store, 0, 3)  # articles 1..3: syndicated copies of one story
    add_article(store, 50, "TCS bags Rs 900 crore deal from European bank", "Multi-year deal.", source="Mint", feed="mint",
                url="https://www.livemint.com/tcs", seen="2026-11-01T11:00:00Z")  # article 4
    explode_on(pipe, monkeypatch, 2)
    with caplog.at_level(logging.ERROR, logger="news_intelligence.pipeline"):
        first = pipe.process_pending()
    assert first.errors == 1 and first.gave_up == 0 and len(events(store)) == 2  # the others went through
    assert [r[0] for r in store.conn.execute("SELECT article_id FROM news_raw WHERE processed = 0")] == [2]
    logged = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(logged) == 1 and "article 2" in logged[0].getMessage() and logged[0].exc_info  # the item id and the traceback
    assert store.kv_get_json(pl.RETRY_KEY) == {"a:2": 1}

    # a restart (a new pipeline object) keeps counting: the attempts live in the database
    again = NewsPipeline(store, settings, clock=fake_clock)
    explode_on(again, monkeypatch, 2)
    assert again.process_pending().errors == 1 and pending(store) == 1 and store.kv_get_json(pl.RETRY_KEY) == {"a:2": 2}

    last = NewsPipeline(store, settings, clock=fake_clock)
    explode_on(last, monkeypatch, 2)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="news_intelligence.pipeline"):
        final = last.process_pending()
    assert final.errors == 1 and final.gave_up == 1 and pending(store) == 0
    assert store.conn.execute("SELECT processed FROM news_raw WHERE article_id = 2").fetchone()[0] == pl.GAVE_UP == 2
    assert any("giving up" in r.getMessage() and "article 2" in r.getMessage() for r in caplog.records)
    assert store.kv_get(pl.RETRY_KEY) is None and store.kv_get_json("news.last_run")["gave_up"] == 1
    quiet = NewsPipeline(store, settings, clock=fake_clock).process_pending()
    assert quiet.errors == 0 and quiet.articles_in == 0  # no infinite loop: the abandoned item is not fetched again


def test_a_transient_failure_recovers_on_the_next_run(store, pipe, monkeypatch):
    story(store, 0, 1)
    explode_on(pipe, monkeypatch, 1)
    assert pipe.process_pending().errors == 1 and events(store) == [] and pending(store) == 1
    monkeypatch.undo()
    healthy = pipe.process_pending()
    assert healthy.errors == 0 and healthy.gave_up == 0 and len(events(store)) == 1 and pending(store) == 0
    assert store.kv_get(pl.RETRY_KEY) is None  # the retry record is cleared once the item went through


def test_a_failing_item_is_attempted_once_per_run_not_once_per_batch(store, pipe, monkeypatch):
    monkeypatch.setattr(pl, "BATCH_SIZE", 3)
    story(store, 0, 8)
    attempted = explode_on(pipe, monkeypatch, 1)
    stats = pipe.process_pending()
    assert attempted == [1] and stats.errors == 1 and stats.articles_in == 8 and pending(store) == 1
    assert store.kv_get_json(pl.RETRY_KEY) == {"a:1": 1}


def test_filings_get_the_same_bounded_retry(store, pipe, monkeypatch):
    add_filing(store, "F1", symbol="TCS", **ORDER_FILING)
    attempted = explode_on(pipe, monkeypatch, 1, attr="_prepare_filing", key="filing_id")
    for _ in (1, 2):
        assert pipe.process_pending().errors == 1 and pending(store, "filings") == 1
    final = pipe.process_pending()
    assert final.gave_up == 1 and pending(store, "filings") == 0 and pending(store, "filings", "processed = 2") == 1
    assert attempted == [1, 1, 1] and pipe.process_pending().filings_in == 0


def test_a_busy_database_does_not_use_up_an_items_attempts(store, pipe, monkeypatch):
    story(store, 0, 2)

    def busy(kind, row, clusterer, stats, linked=None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(pipe, "_process_row", busy)
    assert pipe.process_pending().errors == 1 and pending(store) == 2
    assert store.kv_get(pl.RETRY_KEY) is None


def test_prune_removes_abandoned_items_and_forgets_retry_records_of_rows_that_are_gone(store, pipe, monkeypatch):
    add_filing(store, "F1", symbol="TCS", seen="2026-10-01T10:05:00Z", disseminated="2026-10-01T10:00:00Z", **ORDER_FILING)
    explode_on(pipe, monkeypatch, 1, attr="_prepare_filing", key="filing_id")
    for _ in range(pl.MAX_ATTEMPTS):
        pipe.process_pending()
    assert pending(store, "filings", "processed = 2") == 1
    story(store, 0, 1, seen="2026-11-02T01:00:00Z")  # a recent article that is still failing
    explode_on(pipe, monkeypatch, 1, key="article_id")
    pipe.process_pending()
    store.kv_set_json(pl.RETRY_KEY, {**store.kv_get_json(pl.RETRY_KEY), "a:999": 2, "f:77": 1, "a:abc": 1})  # rows that are gone
    out = pipe.prune(retention_days=1)
    assert out["filings"] == 1 and count(store, "filings") == 0
    assert store.kv_get_json(pl.RETRY_KEY) == {"a:1": 1}


def test_a_corrupt_retry_record_is_ignored_and_a_failure_to_save_it_is_logged_not_raised(store, pipe, monkeypatch, caplog):
    for garbage in ("not json", "[1, 2]", '{"a:1": "x", "a:2": true, "a:3": 2}'):
        store.kv_set(pl.RETRY_KEY, garbage)
        assert pipe._load_retries() == ({"a:3": 2} if garbage.startswith("{") else {})
    store.kv_delete(pl.RETRY_KEY)

    story(store, 0, 1)
    explode_on(pipe, monkeypatch, 1)
    real = store.kv_set_json

    def locked(key, value):
        if key == pl.RETRY_KEY:
            raise sqlite3.OperationalError("database is locked")
        real(key, value)

    monkeypatch.setattr(store, "kv_set_json", locked)
    with caplog.at_level(logging.ERROR, logger="news_intelligence.pipeline"):
        stats = pipe.process_pending()
    assert stats.errors == 1 and pending(store) == 1  # the item is still pending: nothing was lost
    assert any("could not save the retry counters" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------- N2: a known reaction is never erased
def known_reaction(store, flag, *, as_of="2026-11-03T03:40:00Z", note="pre-open indicative price", **fields):
    return add(store, "tcs", [ent("TCS", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:30:00Z", market_reacted=flag,
               reaction_json=json.dumps({"reacted": bool(flag), "as_of": as_of, "symbol": "TCS", "note": note}), **fields)


def no_bars(symbol, start, end):
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.mark.parametrize("flag", [1, 0])
def test_a_known_reaction_survives_a_refresh_whose_bars_are_missing(store, pipe, flag):
    known_reaction(store, flag)
    before = events(store)[0]
    assert pipe.refresh_reactions("2026-11-03T04:30:00Z", no_bars) == 0  # 10:00 IST: no post-event price yet: unknown now
    after = events(store)[0]
    assert after["market_reacted"] == flag and after["reaction_json"] == before["reaction_json"]  # untouched, incl. its as_of
    # and the earlier finding stays visible to point-in-time readers from the moment it was made
    assert pipe.recent_events("2026-11-03T05:00:00Z", 36)[0]["market_reacted"] is bool(flag)


def test_an_unknown_reaction_is_still_recorded_as_unknown_and_a_newer_known_one_replaces_an_older(store, pipe):
    add(store, "fresh", [ent("INFY", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:31:00Z")
    assert pipe.refresh_reactions("2026-11-03T04:30:00Z", no_bars) == 1
    row = events(store)[0]
    assert row["market_reacted"] is None and json.loads(row["reaction_json"])["as_of"] == "2026-11-03T04:30:00Z"

    _seed_bars(store, "TCS", "2026-11-02", gap=0.06)
    known_reaction(store, 0, as_of="2026-11-02T12:00:00Z", note="no trading session since the event")
    assert pipe.refresh_reactions("2026-11-03T12:00:00Z") == 2  # the gap session's bar is known now: INFY (unknown) and TCS
    tcs = next(e for e in events(store) if e["headline"] == "tcs")
    assert tcs["market_reacted"] == 1 and json.loads(tcs["reaction_json"])["as_of"] == "2026-11-03T12:00:00Z"


def test_unavailable_bars_are_logged_as_a_warning_naming_the_symbols(store, pipe, caplog):
    add(store, "tcs", [ent("TCS", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:30:00Z")
    add(store, "infy", [ent("INFY", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:31:00Z")

    def provider(symbol, start, end):
        if symbol in ("INFY", "TCS"):
            raise RuntimeError("vendor down")
        return no_bars(symbol, start, end)

    with caplog.at_level(logging.WARNING, logger="news_intelligence.pipeline"):
        assert pipe.refresh_reactions("2026-11-03T12:00:00Z", provider) == 0
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1 and "INFY" in warned[0].getMessage() and "TCS" in warned[0].getMessage()
    assert "RuntimeError" in warned[0].getMessage()


# ------------------------------------------------------ N3: unverified LLM claims leave no trace, rules get a time bound
def test_an_llm_event_time_is_dropped_when_its_supporting_quote_failed_verification(store, llm_settings, fake_clock):
    pipe, _session, _ = llm_pipe(store, llm_settings, fake_clock, ok(llm_reply(
        evidence_quote="Reliance will buy Disney for ten billion dollars", event_time="2026-10-30T08:00:00Z")))
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/u",
                published="2026-11-01T09:30:00Z", seen="2026-11-01T10:00:00Z")
    pipe.process_pending()
    (event,) = events(store)
    assert event["extraction_method"] == "rules+llm" and event["event_time"] is None
    evidence = json.loads(event["evidence_json"])[0]
    assert "quote" not in evidence and evidence["event_time_bound"] == "2026-11-01T09:30:00Z"  # an upper bound, not a claim


def test_a_verified_llm_event_time_is_kept_and_needs_no_bound(store, llm_settings, fake_clock):
    pipe, _session, _ = llm_pipe(store, llm_settings, fake_clock, ok(llm_reply(event_time="2026-11-01T08:00:00Z")))
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/v",
                published="2026-11-01T09:30:00Z", seen="2026-11-01T10:00:00Z")
    pipe.process_pending()
    (event,) = events(store)
    assert QUOTE in BODY and event["event_time"] == "2026-11-01T08:00:00Z"
    assert "event_time_bound" not in json.loads(event["evidence_json"])[0]


def test_a_filing_falls_back_to_its_dissemination_time_when_the_llm_time_is_unverified(store, llm_settings, fake_clock):
    pipe, _session, _ = llm_pipe(store, llm_settings, fake_clock, ok(llm_reply(
        entities=[{"id": "TCS", "polarity": 0.9}], event_time="2026-10-30T08:00:00Z", evidence_quote="TCS bought a skyscraper")))
    add_filing(store, "F1", symbol="TCS", seen="2026-11-01T09:31:00Z", disseminated="2026-11-01T09:30:00Z", **ORDER_FILING)
    pipe.process_pending()
    (event,) = events(store)
    assert event["extraction_method"] == "rules+llm" and event["event_time"] == "2026-11-01T09:30:00Z"
    assert "event_time_bound" not in json.loads(event["evidence_json"])[0]


def test_rules_only_articles_keep_a_null_event_time_but_expose_the_publication_time_as_an_upper_bound(store, pipe):
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/p",
                published="2026-11-01T09:30:00Z", seen="2026-11-01T10:00:00Z")
    add_article(store, 1, "TCS bags Rs 900 crore deal from European bank", "Multi-year deal.", source="Mint", feed="mint",
                url="https://www.livemint.com/q", seen="2026-11-01T10:05:00Z")  # no publication time at all
    pipe.process_pending()
    reliance, tcs = events(store)
    assert reliance["event_time"] is None and tcs["event_time"] is None
    assert json.loads(reliance["evidence_json"])[0]["event_time_bound"] == "2026-11-01T09:30:00Z"
    assert "event_time_bound" not in json.loads(tcs["evidence_json"])[0]
    views = {v["headline"][:8]: v for v in pipe.recent_events("2026-11-01T11:00:00Z", 24)}
    assert views["Reliance"]["event_time"] is None and views["Reliance"]["event_time_bound"] == "2026-11-01T09:30:00Z"
    assert views["TCS bags"]["event_time_bound"] is None


def test_the_time_bound_is_the_earliest_visible_one_and_legacy_events_fall_back_to_their_publication_time(store, pipe):
    from tests.test_news_features import evidence

    first = {**evidence("blog", 10), "event_time_bound": "2026-11-01T08:30:00Z"}
    later = {**evidence("reuters", 14, q=0.9), "event_time_bound": "2026-11-01T07:00:00Z"}
    add(store, "story", [ent("TCS", polarity=0.7)], severity=0.6, published_at="2026-11-01T07:00:00Z",
        evidence_json=json.dumps([first, later]), last_updated_at="2026-11-01T14:00:00Z")
    add(store, "legacy", [ent("INFY", polarity=0.7)], severity=0.6, published_at="2026-11-01T06:00:00Z", hour=11)
    by_headline = lambda hour: {v["headline"]: v for v in pipe.recent_events(f"2026-11-01T{hour}:00:00Z", 36)}  # noqa: E731
    assert by_headline(12)["blog story"]["event_time_bound"] == "2026-11-01T08:30:00Z"
    assert by_headline(15)["reuters story"]["event_time_bound"] == "2026-11-01T07:00:00Z"  # the later origin tightens it
    assert by_headline(12)["legacy"]["event_time_bound"] == "2026-11-01T06:00:00Z"
    add(store, "dated", [ent("WIPRO", polarity=0.7)], severity=0.6, event_time="2026-11-01T05:00:00Z", published_at="2026-11-01T06:00:00Z")
    assert by_headline(12)["dated"]["event_time"] == "2026-11-01T05:00:00Z" and by_headline(12)["dated"]["event_time_bound"] is None


# ------------------------------------------------ event metadata: novelty class, rumour, source class, conflicts
def test_source_classes_follow_the_blueprint_vocabulary():
    assert {k: source_class_of(k) for k in ("exchange_filing", "official", "company", "wire", "business_media", "aggregator",
                                            "social", "unknown")} == {
        "exchange_filing": "exchange_filing", "official": "official", "company": "company", "wire": "media",
        "business_media": "media", "aggregator": "media", "social": "social", "unknown": "unknown"}
    assert source_class_of(None) == "unknown" and source_class_of("brand-new-tier") == "unknown"


def test_a_better_copy_of_an_existing_origin_upgrades_its_source_class(store, pipe):
    add_article(store, 0, TITLE, BODY, source="Obscure Blog", feed="blog", url="https://blog.example.net/a",
                seen="2026-11-01T10:00:00Z")
    add_article(store, 1, TITLE, BODY, source="Reuters", feed="reuters", url="https://www.reuters.com/a", seen="2026-11-01T10:05:00Z")
    pipe.process_pending()
    (event,) = events(store)
    (origin,) = json.loads(event["evidence_json"])  # the same text: one origin, now backed by a wire service
    assert origin["n"] == 2 and origin["sc"] == "media" and event["source_class"] == "media"
    assert [v["source_class"] for v in pipe.recent_events("2026-11-01T11:00:00Z", 24)] == ["media"]


def test_the_novelty_class_of_a_new_follow_up_resurfaced_and_rumour_event_is_stored(store, pipe):
    for i, day in enumerate(("2026-10-10", "2026-10-14", "2026-10-30")):
        add_article(store, i, TITLE, BODY, source="Mint", feed="mint", url=f"https://www.livemint.com/{i}", seen=f"{day}T10:00:00Z")
    add_article(store, 9, *RUMOUR, source="Mint", feed="mint", url="https://www.livemint.com/rumour", seen="2026-10-31T10:00:00Z")
    pipe.process_pending()
    rows = events(store)
    assert [r["novelty_class"] for r in rows] == ["new", "follow_up", "resurfaced", "rumour"]
    assert [r["rumour"] for r in rows] == [0, 0, 0, 1]
    assert [r["novelty"] for r in rows[:3]] == [1.0, 0.35, 0.1]


def test_source_class_independent_sources_and_conflicts_are_persisted_with_the_event(store, pipe):
    add_article(store, 0, "TCS Q2 profit jumps 12%, beats estimates", "Net profit rose to Rs 12,000 crore, ahead of analyst estimates.",
                source="Mint", feed="mint", url="https://www.livemint.com/t1", seen="2026-11-01T10:00:00Z")
    add_article(store, 1, "TCS Q2 profit falls 8%, misses estimates", "Net profit fell to Rs 10,000 crore, below analyst estimates.",
                source="Reuters", feed="reuters", url="https://www.reuters.com/t2", seen="2026-11-01T12:00:00Z")
    pipe.process_pending()
    (row,) = events(store)
    assert row["source_class"] == "media" and row["n_independent_sources"] == 2 and row["n_conflicts"] == 1
    assert row["novelty_class"] == "new" and row["rumour"] == 0
    evidence = json.loads(row["evidence_json"])
    assert [e["sc"] for e in evidence] == ["media", "media"] and not any(e.get("rum") for e in evidence)


def test_a_conflict_recorded_on_another_event_is_counted_there_too(store, pipe):
    from news_intelligence.dedup import ClusterEvent, Contradiction

    event_id = add(store, "story", [ent("TCS", polarity=0.7)], severity=0.6, evidence_json=json.dumps([
        {"kind": "article", "ref": "a:1", "t": "2026-11-01T10:00:00Z", "owner": "mint", "q": 0.6, "conf": 0.7, "pol": 0.7}]))
    cluster = ClusterEvent(key="story", event_id=event_id, event_type="order_win", first_seen=0.0, last_seen=0.0, primary="TCS",
                           polarity=0.7)
    with pipe.store.tx() as conn:
        pipe._annotate(conn, cluster, Contradiction("polarity", "a:2", 1_000.0, -0.6, 0.7, "reuters"))
        pipe._annotate(conn, cluster, Contradiction("denial", "a:3", 2_000.0, -0.6, 0.7, "pti"))
    row = events(store)[0]
    assert row["n_conflicts"] == 2 and len(json.loads(row["contradictions_json"])) == 2


def confirm_a_rumour(store, pipe):
    add_article(store, 1, *RUMOUR, source="Mint", feed="mint", url="https://www.livemint.com/r", seen="2026-11-01T10:00:00Z")
    pipe.process_pending()
    add_filing(store, "F1", symbol="RELIANCE", category="Acquisition", company="Reliance Industries Limited",
               subject="Reliance Industries Limited - Acquisition of stake in Wipro",
               text="Reliance Industries has announced the acquisition of a stake in Wipro. The board approved the acquisition of a "
                    "stake in Wipro.", seen="2026-11-01T11:01:00Z", disseminated="2026-11-01T11:00:00Z")
    add_article(store, 2, "Reliance to acquire stake in Wipro, deal confirmed",
                "Reliance Industries will acquire a stake in Wipro, the company confirmed in a statement.", source="Reuters",
                feed="reuters", url="https://www.reuters.com/r2", seen="2026-11-01T12:00:00Z")
    stats = pipe.process_pending()
    assert stats.events_created == 0 and stats.events_updated == 2


def test_a_rumour_stops_being_one_when_a_better_report_joins_and_the_best_source_class_follows(store, pipe):
    confirm_a_rumour(store, pipe)
    (row,) = events(store)
    assert row["novelty_class"] == "rumour"  # how the story began is history
    assert row["rumour"] == 0 and row["source_class"] == "exchange_filing" and row["n_independent_sources"] == 3
    assert [e.get("rum") for e in json.loads(row["evidence_json"])] == [True, None, None]


def test_a_denial_does_not_confirm_a_rumour(store, pipe):
    add_article(store, 1, *RUMOUR, source="Mint", feed="mint", url="https://www.livemint.com/r", seen="2026-11-01T10:00:00Z")
    pipe.process_pending()
    add_article(store, 2, "Reliance denies it is in talks to acquire stake in Wipro",
                "Reliance Industries denied reports that it is in talks to acquire a stake in Wipro.", source="Reuters",
                feed="reuters", url="https://www.reuters.com/denial", seen="2026-11-01T11:00:00Z")
    pipe.process_pending()
    (row,) = events(store)
    assert row["rumour"] == 1 and row["n_conflicts"] >= 1
    view = pipe.recent_events("2026-11-01T12:00:00Z", 24)[0]
    assert view["rumour"] is True and view["n_conflicts"] >= 1


def test_event_views_expose_the_metadata_as_known_at_the_time(store, pipe):
    confirm_a_rumour(store, pipe)

    def view(hour):
        return pipe.recent_events(f"2026-11-01T{hour}:00:00Z", 24)[0]

    early, mid, late = view("10:30"), view("11:30"), view("13:00")
    assert (early["rumour"], early["source_class"], early["n_independent_sources"]) == (True, "media", 1)
    assert (mid["rumour"], mid["source_class"], mid["n_independent_sources"]) == (False, "exchange_filing", 2)
    assert (late["rumour"], late["source_class"], late["n_independent_sources"]) == (False, "exchange_filing", 3)
    assert {v["novelty_class"] for v in (early, mid, late)} == {"rumour"} and early["n_conflicts"] == 0


def test_legacy_events_report_unknown_metadata_instead_of_guessing(store, pipe):
    add(store, "plain", [ent("TCS", polarity=0.7)], severity=0.6, n_independent_sources=2)
    legacy = pipe.recent_events("2026-11-01T11:00:00Z", 24)[0]
    assert (legacy["novelty_class"], legacy["rumour"], legacy["source_class"]) == (None, None, None)
    assert legacy["n_conflicts"] == 0 and legacy["n_independent_sources"] == 2
    old_evidence = [{"kind": "article", "ref": "a:1", "t": "2026-11-01T10:00:00Z", "owner": "mint", "q": 0.6, "conf": 0.7,
                     "pol": 0.5, "title": "old format"}]  # written before the metadata existed: no per-origin class
    add(store, "oldfmt", [ent("INFY", polarity=0.7)], severity=0.6, evidence_json=json.dumps(old_evidence),
        novelty_class="new", rumour=1, source_class="exchange_filing", hour=10)
    view = {v["headline"]: v for v in pipe.recent_events("2026-11-01T11:00:00Z", 24)}["old format"]
    assert view["novelty_class"] == "new" and view["rumour"] is None and view["source_class"] is None  # never from a later column


class Recording:
    """Stands in for ranking.types.Catalyst: keeps exactly what the pipeline hands to ``from_dict``."""

    @staticmethod
    def from_dict(data):
        return dict(data)


def test_catalysts_are_built_from_the_event_metadata(store, pipe, monkeypatch):
    confirm_a_rumour(store, pipe)
    monkeypatch.setattr(pl, "Catalyst", Recording)
    (early,) = pipe.catalysts_for("RELIANCE", "2026-11-01T10:30:00Z")
    (late,) = pipe.catalysts_for("RELIANCE", "2026-11-01T13:00:00Z")
    wanted = ("novelty_class", "rumour", "source_class", "n_independent", "n_conflicts")
    assert {k: early[k] for k in wanted} == {"novelty_class": "rumour", "rumour": True, "source_class": "media", "n_independent": 1,
                                             "n_conflicts": 0}
    assert {k: late[k] for k in wanted} == {"novelty_class": "rumour", "rumour": False, "source_class": "exchange_filing",
                                            "n_independent": 3, "n_conflicts": 0}
    assert early["event_id"] == late["event_id"] == events(store)[0]["event_id"]
    assert early["headline"] and early["event_type"] == "merger_acquisition" and early["polarity"] is not None


def test_real_catalysts_carry_the_metadata_once_the_fields_exist_and_never_fail_before(store, pipe):
    confirm_a_rumour(store, pipe)
    (catalyst,) = pipe.catalysts_for("RELIANCE", "2026-11-01T13:00:00Z")
    assert isinstance(catalyst, Catalyst) and catalyst.event_type == "merger_acquisition"  # unknown keys are ignored, never an error
    names = {f.name for f in dataclasses.fields(Catalyst)}
    if {"event_id", "novelty_class", "rumour", "source_class", "n_independent", "n_conflicts"} <= names:
        assert (catalyst.novelty_class, catalyst.rumour, catalyst.source_class, catalyst.n_independent, catalyst.n_conflicts) == (
            "rumour", False, "exchange_filing", 3, 0)
        assert catalyst.event_id == events(store)[0]["event_id"]


# ----------------------------------------------------------------------- a stock is only an event's subject, not a mention
def test_a_macro_event_that_reaches_a_stock_only_through_broken_links_is_neither_its_catalyst_nor_a_feature(store, settings,
                                                                                                        fake_clock):
    from event_graph.repository import GraphRepository
    from event_graph.seed import seed_graph
    from tests.test_news_features import H

    pipe = NewsPipeline(store, settings, clock=fake_clock)
    add_instruments(store)
    seed_graph(store)
    add(store, "brent", [ent("macro:brent", type="commodity", polarity=0.7, name="Brent crude", move_pct=3.1)], scope="macro",
        event_type="commodity_supply", severity=0.8, confidence=0.9, source_quality=0.75)
    assert pipe.event_features(["HINDPETRO"], H.format(12))["HINDPETRO"]["ev_global_macro_severity"] > 0.0
    assert [c.headline for c in pipe.catalysts_for("HINDPETRO", H.format(12))] == ["brent"]

    repo = GraphRepository(store)
    for edge in repo.get_edges(src="macro:brent"):
        repo.upsert_edge(edge.src, edge.dst, edge.relation, direction=0, sensitivity=None, confidence=0.05, rationale=edge.rationale,
                         source="relationship_break", evidence_count=300)
    fresh = NewsPipeline(store, settings, clock=fake_clock)  # impacts are cached per pipeline object, edges change in another job
    assert fresh.event_features(["HINDPETRO"], H.format(12))["HINDPETRO"]["ev_global_macro_severity"] == 0.0
    assert fresh.catalysts_for("HINDPETRO", H.format(12)) == []


def test_subject_companies_are_signed_or_named_in_the_headline_never_passing_mentions():
    rows = json.dumps([ent("TCS", polarity=0.6), ent("INFY", polarity=0.0, in_title=True), ent("WIPRO", polarity=0.0, in_title=False),
                       ent("SBIN", polarity=0.5, confidence=0.3), ent("global:NVDA", polarity=0.9),
                       ent("sector:pharma", type="sector", polarity=0.5), ent("macro:brent", type="commodity", polarity=0.7)])
    assert feat.subject_companies(rows) == {"TCS", "INFY", "global:NVDA"}
    assert feat.subject_companies(None) == set() and feat.subject_companies("not json") == set()
    late = json.dumps([ent("TCS", polarity=0.6), ent("INFY", polarity=0.6, t="2026-11-01T12:00:00Z")])
    assert feat.subject_companies(late, "2026-11-01T11:00:00Z") == {"TCS"}  # an entity attached after as_of is not known yet
