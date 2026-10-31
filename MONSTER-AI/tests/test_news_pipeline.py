from __future__ import annotations

import dataclasses
import json
import sqlite3

import pytest

from news_intelligence import pipeline as pl
from news_intelligence.dedup import event_confidence
from news_intelligence.llm import LLMExtractor
from news_intelligence.pipeline import NewsPipeline, ProcessStats
from tests.test_news_support import (
    FakeResponse,
    FakeSession,
    add_article,
    add_filing,
    add_instruments,
    events,
    ok,
)

TITLE = "Reliance Industries bags Rs 5,000 crore order from NTPC for solar modules"
BODY = "Reliance Industries has received a Rs 5,000 crore order from NTPC for supply of solar modules over three years."
QUOTE = "received a Rs 5,000 crore order from NTPC"


@pytest.fixture()
def pipe(store, settings, fake_clock):
    add_instruments(store)
    return NewsPipeline(store, settings, clock=fake_clock)


@pytest.fixture()
def llm_settings(settings):
    return dataclasses.replace(settings, llm_api_base="https://llm.example/v1", llm_model="m", llm_api_key="sk-x")


def llm_pipe(store, llm_settings, clock, *responses, **kwargs):
    add_instruments(store)
    session = FakeSession(*responses)
    extractor = LLMExtractor(llm_settings, session=session, clock=clock, **kwargs)
    return NewsPipeline(store, llm_settings, clock=clock, llm=extractor), session, extractor


def llm_reply(**over):
    data = {"event_type": "order_win", "entities": [{"id": "RELIANCE", "polarity": 0.9}], "severity": 0.9, "horizon": "weeks",
            "event_time": "2026-11-01T08:00:00Z", "confidence": 0.9, "evidence_quote": QUOTE}
    data.update(over)
    return json.dumps(data)


def count(store, table, where="1=1"):
    return store.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


def story(store, start=0, n=5, *, seen="2026-11-01T10:{:02d}:00Z"):
    for i in range(n):
        add_article(store, start + i, TITLE, BODY, source=f"Publisher {i}", feed=f"feed{i}",
                    url=f"https://news{i}.example.com/a/{start + i}", seen=seen.format(i))


# ------------------------------------------------------------------------------------------ clustering
def test_a_syndicated_story_is_one_event_across_filing_wire_rewrites_and_copies(store, pipe):
    story(store, 0, 50)
    add_filing(store, "NSE-1", symbol="RELIANCE", category="Award of Order / Receipt of Order",
               subject="Reliance Industries Limited - Award of Order", company="Reliance Industries Limited",
               text="Reliance Industries has received an order worth Rs 5,000 crore from NTPC for supply of solar modules.",
               seen="2026-11-01T09:31:00Z", disseminated="2026-11-01T09:30:00Z")
    add_article(store, 900, "NTPC awards Reliance Rs 5,000 crore solar module contract - Reuters",
                "NTPC placed an order worth Rs 5,000 crore with Reliance for solar modules.", source="Google News",
                feed="gn_markets", url="https://news.google.com/rss/articles/abc", seen="2026-11-01T11:00:00Z")
    add_article(store, 901, "Reliance wins NTPC solar order", "Order value Rs 5,000 crore over three years.", source="Mint",
                feed="mint_markets", url="https://www.livemint.com/x", seen="2026-11-01T12:00:00Z")
    stats = pipe.process_pending()
    rows = events(store)
    assert len(rows) == 1 and stats.events_created == 1 and stats.errors == 0
    event = rows[0]
    assert event["cluster_key"] == "filing:RELIANCE:order_win:2026-11-01" and event["event_type"] == "order_win"
    assert event["scope"] == "company" and event["n_articles"] == 53 and 2 <= event["n_independent_sources"] <= 5
    assert event["first_seen_at"] == "2026-11-01T09:31:00Z" and event["last_updated_at"] == "2026-11-01T12:00:00Z"
    assert event["source_quality"] == 1.0 and event["extraction_method"] == "rules"
    assert stats.articles_in == 52 and stats.filings_in == 1 and stats.events_updated == 52 and stats.duplicates == 49
    assert count(store, "event_articles") == 53 and count(store, "news_raw", "processed = 0") == 0
    assert count(store, "filings", "processed = 0") == 0
    evidence = json.loads(event["evidence_json"])
    assert sum(e["n"] for e in evidence) == 53 and {e["owner"] for e in evidence} >= {"nse", "reuters", "livemint.com"}
    assert all(len(e["domains"]) <= 8 for e in evidence)
    entities = {e["id"]: e for e in json.loads(event["entities_json"])}
    assert entities["RELIANCE"]["polarity"] > 0 and entities["NTPC"]["polarity"] == 0.0


def test_articles_alone_use_news_cluster_keys_and_syndication_counts_one_origin(store, pipe):
    story(store, 0, 50)
    stats = pipe.process_pending()
    (event,) = events(store)
    assert event["cluster_key"].startswith("news:") and len(event["cluster_key"]) == len("news:") + 24
    assert event["n_articles"] == 50 and event["n_independent_sources"] == 1
    assert stats.events_created == 1 and stats.events_updated == 49 and stats.duplicates == 49
    assert count(store, "event_articles") == 50
    added = [r[0] for r in store.conn.execute("SELECT added_at FROM event_articles ORDER BY added_at")]
    assert added[0] == "2026-11-01T10:00:00Z" and added[-1] == "2026-11-01T10:49:00Z"  # first-seen, not processing time


def test_distinct_stories_stay_distinct_with_scope_and_entities(store, pipe):
    add_article(store, 1, "TCS bags Rs 900 crore deal from European bank", "Multi-year deal to modernise core systems.")
    add_article(store, 2, "RBI keeps repo rate unchanged at 6.5%", "The monetary policy committee voted unanimously.",
                source="Reuters", feed="rbi", url="https://www.rbi.org.in/pr/1")
    add_article(store, 3, "Brent crude jumps 3.1% as Red Sea attack escalates", "Oil prices rose after the attack on a tanker.",
                source="Reuters", feed="gdelt:oil", url="https://www.reuters.com/oil/1", region="global")
    add_article(store, 4, "Weather: heavy rain expected in Mumbai on Tuesday", "The IMD issued a yellow alert.")
    stats = pipe.process_pending()
    by_type = {e["event_type"]: e for e in events(store)}
    assert set(by_type) >= {"order_win", "central_bank"} and len(events(store)) == 3
    assert by_type["order_win"]["scope"] == "company" and by_type["central_bank"]["scope"] == "macro"
    macro = next(e for e in events(store) if e["event_type"] not in ("order_win", "central_bank"))
    assert macro["scope"] in ("macro", "global") and macro["region"] == "global"
    ids = {e["id"] for e in json.loads(macro["entities_json"])}
    assert "macro:brent" in ids
    assert stats.skipped_irrelevant == 1 and stats.events_created == 3


def test_items_without_an_event_are_skipped_but_marked_processed(store, pipe):
    add_article(store, 1, "रिलायंस इंडस्ट्रीज़ के शेयर में तेज़ी", "कंपनी को बड़ा ऑर्डर मिला है और शेयर बढ़े।", language="hi")
    add_article(store, 2, "रिलायंस को नया ऑर्डर मिला", "यह खबर हिंदी में लिखी गई है और इसमें कोई अंग्रेज़ी शब्द नहीं हैं।")
    add_article(store, 3, "", "")
    add_article(store, 4, "Weather: heavy rain expected", "Nothing about markets.")
    add_filing(store, "NSE-T", symbol="TCS", category="Trading Window", subject="Intimation of closure of trading window")
    add_filing(store, "NSE-U", symbol="TCS", category="Updates", subject="", text="")
    add_filing(store, "NSE-E", symbol="TCS", category="", subject="", text="")
    stats = pipe.process_pending()
    assert events(store) == []
    assert stats.skipped_non_english == 2 and stats.skipped_empty == 2 and stats.skipped_irrelevant == 2
    assert stats.skipped_routine == 1 and stats.articles_in == 4 and stats.filings_in == 3
    assert count(store, "news_raw", "processed = 0") == 0 and count(store, "filings", "processed = 0") == 0


def test_reprocessing_is_idempotent(store, pipe, settings, fake_clock):
    story(store, 0, 6)
    add_filing(store, "NSE-1", symbol="TCS", category="Financial Results", subject="Financial Results for Q2 FY26",
               text="TCS reported net profit up 12% for the quarter, beating estimates.")
    pipe.process_pending()
    first_events, first_links = events(store), count(store, "event_articles")
    again = pipe.process_pending()
    assert again.articles_in == 0 and again.filings_in == 0 and events(store) == first_events
    # even if every row were flagged unprocessed again (e.g. a crash after commit), nothing is duplicated
    store.conn.execute("UPDATE news_raw SET processed = 0")
    store.conn.execute("UPDATE filings SET processed = 0")
    fresh = NewsPipeline(store, settings, clock=fake_clock)
    redo = fresh.process_pending()
    assert redo.skipped_already_linked == 7 and redo.events_created == 0 and redo.events_updated == 0
    assert events(store) == first_events and count(store, "event_articles") == first_links
    assert count(store, "news_raw", "processed = 0") == 0


def test_a_second_run_extends_events_created_by_the_first(store, pipe, settings, fake_clock):
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint_markets", url="https://www.livemint.com/a", seen="2026-11-01T10:00:00Z")
    pipe.process_pending()
    add_article(store, 1, TITLE, BODY, source="Business Standard", feed="bs", url="https://www.business-standard.com/b",
                seen="2026-11-01T11:00:00Z")
    add_article(store, 2, "NTPC awards Reliance Rs 5,000 crore solar module contract",
                "NTPC placed an order worth Rs 5,000 crore with Reliance for the supply of solar modules.",
                source="Reuters", feed="reuters", url="https://www.reuters.com/c", seen="2026-11-01T12:00:00Z")
    second = NewsPipeline(store, settings, clock=fake_clock).process_pending()
    (event,) = events(store)
    assert second.events_created == 0 and second.events_updated == 2 and second.duplicates == 1
    assert event["n_articles"] == 3 and event["n_independent_sources"] == 2 and event["last_updated_at"] == "2026-11-01T12:00:00Z"


def test_filing_events_are_reloaded_by_key_even_without_signatures(store, pipe, settings, fake_clock):
    add_filing(store, "A", symbol="TCS", category="Award of Order / Receipt of Order", subject="TCS - Award of order",
               text="TCS has received an order worth Rs 900 crore.", seen="2026-11-01T09:31:00Z", disseminated="2026-11-01T09:30:00Z")
    pipe.process_pending()
    store.conn.execute("DELETE FROM kv WHERE key LIKE 'news.sig:%'")
    add_filing(store, "B", symbol="TCS", category="Award of Order / Receipt of Order", subject="TCS - Award of order (update)",
               text="Further details on the order worth Rs 900 crore.", seen="2026-11-01T10:31:00Z", disseminated="2026-11-01T10:30:00Z")
    stats = NewsPipeline(store, settings, clock=fake_clock).process_pending()
    assert len(events(store)) == 1 and stats.events_updated == 1 and events(store)[0]["n_articles"] == 2


def test_resurfacing_and_followups_get_low_novelty_and_unique_keys(store, pipe):
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/0", seen="2026-10-10T10:00:00Z")
    add_article(store, 1, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/1", seen="2026-10-14T10:00:00Z")
    add_article(store, 2, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/2", seen="2026-10-30T10:00:00Z")
    stats = pipe.process_pending()
    first, followup, resurfaced = events(store)
    assert stats.events_created == 3
    assert (first["novelty"], followup["novelty"], resurfaced["novelty"]) == (1.0, 0.35, 0.1)
    keys = [first["cluster_key"], followup["cluster_key"], resurfaced["cluster_key"]]
    assert len(set(keys)) == 3 and keys[1].startswith(keys[0]) and keys[2].startswith(keys[0])


def test_rumours_lower_novelty_and_confidence(store, pipe):
    add_article(store, 0, "Reliance in talks to acquire stake in Wipro, sources said", "The talks are at an early stage and may not conclude.",
                source="Mint", feed="mint", url="https://www.livemint.com/r")
    add_article(store, 1, "Reliance to acquire stake in Wipro for Rs 3,000 crore", "The company said in a filing that the board approved the deal.",
                source="Reuters", feed="reuters", url="https://www.reuters.com/r", seen="2026-11-01T14:00:00Z")
    pipe.process_pending()
    rumour, official = events(store)
    assert rumour["event_type"] == official["event_type"] == "merger_acquisition"
    assert rumour["novelty"] == pytest.approx(0.7) and official["novelty"] in (1.0, 0.5)
    assert rumour["confidence"] < official["confidence"]


def test_contradicting_reports_are_preserved_and_lower_confidence(store, pipe):
    add_article(store, 0, "TCS Q2 profit jumps 12%, beats estimates", "Net profit rose to Rs 12,000 crore, ahead of analyst estimates.",
                source="Mint", feed="mint", url="https://www.livemint.com/t1", seen="2026-11-01T10:00:00Z")
    add_article(store, 1, "TCS Q2 profit falls 8%, misses estimates", "Net profit fell to Rs 10,000 crore, below analyst estimates.",
                source="Reuters", feed="reuters", url="https://www.reuters.com/t2", seen="2026-11-01T12:00:00Z")
    pipe.process_pending()
    (row,) = events(store)  # one story, two opposite readings: kept together, never silently resolved
    notes = json.loads(row["contradictions_json"])
    assert [n["kind"] for n in notes] == ["polarity"] and notes[0]["t"] == "2026-11-01T12:00:00Z"
    evidence = json.loads(row["evidence_json"])
    assert [e["pol"] > 0 for e in evidence] == [True, False]
    assert row["confidence"] < event_confidence(evidence[0]["conf"], max(e["q"] for e in evidence), 2, 0)
    assert row["n_independent_sources"] == 2
    before = pipe.recent_events("2026-11-01T11:00:00Z", 24)[0]["entities"][0]["polarity"]
    after = pipe.recent_events("2026-11-01T13:00:00Z", 24)[0]["entities"][0]["polarity"]
    assert before > 0.4 and abs(after) < 0.2  # the disagreement shrinks the stated direction once it is known


def test_time_fields_are_stored_separately(store, pipe):
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/p",
                published="2026-11-01T08:30:00Z", seen="2026-11-01T10:00:00Z")
    add_article(store, 1, "TCS bags Rs 900 crore deal from European bank", "Multi-year deal.", source="Mint", feed="mint",
                url="https://www.livemint.com/q", published="2026-11-01T23:00:00Z", seen="2026-11-01T10:05:00Z")
    add_filing(store, "F1", symbol="INFY", category="Award of Order / Receipt of Order", subject="Infosys - Award of order",
               text="Infosys has received an order worth Rs 700 crore.", seen="2026-11-01T11:10:00Z", disseminated="2026-11-01T11:00:00Z")
    pipe.process_pending()
    by_key = {e["headline"][:12]: e for e in events(store)}
    reliance = next(e for e in events(store) if e["headline"].startswith("Reliance"))
    assert reliance["published_at"] == "2026-11-01T08:30:00Z" and reliance["first_seen_at"] == "2026-11-01T10:00:00Z"
    tcs = next(e for e in events(store) if e["headline"].startswith("TCS"))
    assert tcs["published_at"] is None  # a publication time after we saw the item is not credible
    infy = next(e for e in events(store) if e["cluster_key"].startswith("filing:INFY"))
    assert infy["published_at"] == "2026-11-01T11:00:00Z" and infy["event_time"] == "2026-11-01T11:00:00Z"
    assert infy["first_seen_at"] == "2026-11-01T11:10:00Z" and len(by_key) >= 2


def test_aggregator_items_credit_the_original_publisher(store, pipe):
    add_article(store, 0, "Reliance bags Rs 5,000 crore order from NTPC - Reuters", BODY, source="Google News", feed="gn_markets",
                url="https://news.google.com/rss/articles/xyz")
    pipe.process_pending()
    (event,) = events(store)
    evidence = json.loads(event["evidence_json"])
    assert evidence[0]["owner"] == "reuters" and not evidence[0]["title"].endswith("Reuters")
    assert evidence[0]["q"] <= 0.6 and event["headline"].startswith("Reliance bags")


def test_html_and_oversized_text_are_neutralised_and_bounded(store, pipe):
    nasty = "<script>alert('x')</script><p>Reliance Industries <b>bags</b> Rs 5,000 crore order from NTPC</p>" + "<div>" * 3000
    add_article(store, 0, "<b>Reliance</b> wins order", nasty, source="Mint", feed="mint", url="https://www.livemint.com/h",
                text="word " * 40_000 + " Reliance Industries order from NTPC")
    pipe.process_pending()
    (event,) = events(store)
    assert "<" not in event["headline"] and "<" not in (event["summary"] or "") and "alert" not in (event["summary"] or "")
    assert len(event["headline"]) <= 300 and len(event["summary"] or "") <= 500
    assert len(event["evidence_json"]) < 6000


def test_late_independent_reports_add_entities_with_their_own_timestamps(store, pipe):
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/a", seen="2026-11-01T10:00:00Z")
    add_article(store, 1, "NTPC awards Reliance Rs 5,000 crore solar module contract",
                "NTPC awarded the Rs 5,000 crore solar module order to Reliance Industries; Infosys shares were flat.",
                source="Reuters", feed="reuters", url="https://www.reuters.com/b", seen="2026-11-01T12:00:00Z")
    pipe.process_pending()
    (event,) = events(store)
    entities = {e["id"]: e for e in json.loads(event["entities_json"])}
    assert entities["INFY"]["t"] == "2026-11-01T12:00:00Z" and "t" not in entities["RELIANCE"]


# ---------------------------------------------------------------------------------- crash safety / limits
def test_a_poison_item_is_isolated_and_the_rest_still_processed(store, pipe, monkeypatch):
    story(store, 0, 3)
    add_article(store, 50, "TCS bags Rs 900 crore deal from European bank", "Multi-year deal.", source="Mint", feed="mint",
                url="https://www.livemint.com/tcs", seen="2026-11-01T11:00:00Z")
    original = pipe._prepare_article

    def explode(row, stats):
        if row["article_id"] == 2:
            raise RuntimeError("boom")
        return original(row, stats)

    monkeypatch.setattr(pipe, "_prepare_article", explode)
    stats = pipe.process_pending()
    assert stats.errors == 1 and count(store, "news_raw", "processed = 0") == 0
    assert {e["event_type"] for e in events(store)} == {"order_win"} and len(events(store)) == 2
    assert count(store, "event_articles") == 3


def test_a_failed_write_rolls_back_the_whole_item(store, pipe, monkeypatch):
    story(store, 0, 1)
    real = store.kv_set

    def failing(key, value):
        if key.startswith(pl.SIG_PREFIX):
            raise RuntimeError("disk error")
        real(key, value)

    monkeypatch.setattr(store, "kv_set", failing)
    stats = pipe.process_pending()
    assert stats.errors == 1
    assert events(store) == [] and count(store, "event_articles") == 0  # no half-written event


def test_database_busy_keeps_rows_pending_for_the_next_run(store, pipe, settings, fake_clock, monkeypatch):
    story(store, 0, 3)

    def busy(kind, row, clusterer, stats, linked=None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(pipe, "_process_row", busy)
    stats = pipe.process_pending()
    assert stats.errors == 1 and count(store, "news_raw", "processed = 0") == 3
    monkeypatch.undo()
    healthy = NewsPipeline(store, settings, clock=fake_clock).process_pending()
    assert healthy.errors == 0 and len(events(store)) == 1 and count(store, "news_raw", "processed = 0") == 0


def test_max_items_bounds_a_run_and_filings_come_first(store, pipe):
    story(store, 0, 3)
    add_filing(store, "F1", symbol="TCS", category="Award of Order / Receipt of Order", subject="TCS - Award of order",
               text="TCS has received an order worth Rs 900 crore.", seen="2026-11-01T11:00:00Z")
    add_filing(store, "F2", symbol="INFY", category="Award of Order / Receipt of Order", subject="Infosys - Award of order",
               text="Infosys has received an order worth Rs 700 crore.", seen="2026-11-01T11:01:00Z")
    first = pipe.process_pending(max_items=3)
    assert first.filings_in == 2 and first.articles_in == 1
    assert count(store, "news_raw", "processed = 0") == 2
    assert pipe.process_pending(max_items=0).articles_in == 0
    assert pipe.process_pending().articles_in == 2 and count(store, "news_raw", "processed = 0") == 0


def test_bad_timestamps_are_a_poison_item_not_a_crash(store, pipe):
    add_article(store, 0, TITLE, BODY, seen="not-a-time")
    add_filing(store, "F", symbol="TCS", category="Award of Order / Receipt of Order", subject="TCS - Award of order",
               text="TCS has received an order worth Rs 900 crore.", seen="garbage")
    stats = pipe.process_pending()
    assert stats.errors == 2 and events(store) == []
    assert count(store, "news_raw", "processed = 0") == 0 and count(store, "filings", "processed = 0") == 0


def test_last_run_stats_are_persisted_and_usable_as_a_dict(store, pipe, fake_clock):
    story(store, 0, 2)
    stats = pipe.process_pending()
    assert isinstance(stats, ProcessStats) and stats.__dict__ == stats.as_dict() and stats.elapsed_sec >= 0
    saved = store.kv_get_json("news.last_run")
    assert saved["articles_in"] == 2 and saved["events_created"] == 1 and saved["at"] == "2026-11-02T03:00:00Z"
    assert store.kv_get("news.reduced_coverage") is None  # no LLM configured: nothing to report


def test_filings_without_a_symbol_still_resolve_by_company_name(store, pipe):
    add_filing(store, "F", symbol=None, company="Infosys Limited", category="Award of Order / Receipt of Order",
               subject="Infosys Limited - Award of order", text="Infosys has received an order worth Rs 700 crore.")
    pipe.process_pending()
    (event,) = events(store)
    assert event["cluster_key"].startswith("filing:Infosys Limited:order_win:")
    assert "INFY" in {e["id"] for e in json.loads(event["entities_json"])}


# --------------------------------------------------------------------------------------------- the LLM
def test_pipeline_builds_its_own_extractor_only_when_configured(store, settings, llm_settings, fake_clock):
    assert NewsPipeline(store, settings, clock=fake_clock).llm is None
    built = NewsPipeline(store, llm_settings, clock=fake_clock)
    assert isinstance(built.llm, LLMExtractor) and built.llm.store is store and built.llm.resolver is built.resolver


def test_validated_llm_output_enriches_but_never_overrides_evidence(store, llm_settings, fake_clock):
    pipe, session, _ = llm_pipe(store, llm_settings, fake_clock, ok(llm_reply(
        entities=[{"id": "RELIANCE", "polarity": 0.9}, {"id": "TCS", "polarity": 0.5}, {"name": "NTPC", "polarity": 0.0}],
    )))
    add_article(store, 0, "Reliance bags Rs 5,000 crore NTPC order", "Reliance Industries has " + QUOTE + " for supply of solar modules.",
                source="Mint", feed="mint", url="https://www.livemint.com/l")
    stats = pipe.process_pending()
    (event,) = events(store)
    assert event["extraction_method"] == "rules+llm" and stats.llm_used == 1 and stats.llm_failed == 0
    assert event["event_time"] == "2026-11-01T08:00:00Z" and event["horizon"] == "weeks"
    entities = {e["id"]: e for e in json.loads(event["entities_json"])}
    assert entities["RELIANCE"]["polarity"] == 0.9 and "TCS" not in entities  # never mentioned: dropped
    assert json.loads(event["evidence_json"])[0]["quote"] == QUOTE
    assert len(session.calls) == 1 and store.kv_get("news.reduced_coverage") == "0"


def test_unverified_llm_claims_do_not_change_the_event(store, llm_settings, fake_clock):
    pipe, _session, _ = llm_pipe(store, llm_settings, fake_clock, ok(llm_reply(
        event_type="merger_acquisition", evidence_quote="Reliance will buy Disney for ten billion dollars", severity=0.99,
        entities=[{"id": "RELIANCE", "polarity": -1}],
    )))
    add_article(store, 0, TITLE, BODY, source="Mint", feed="mint", url="https://www.livemint.com/u")
    pipe.process_pending()
    (event,) = events(store)
    assert event["event_type"] == "order_win" and event["severity"] < 0.9
    entities = {e["id"]: e for e in json.loads(event["entities_json"])}
    assert entities["RELIANCE"]["polarity"] > 0  # the unverified -1 is ignored
    assert "quote" not in json.loads(event["evidence_json"])[0]


def test_verified_llm_can_classify_what_rules_call_other(store, llm_settings, fake_clock):
    quote = "Reliance Industries has appointed a new chief financial officer"
    pipe, _session, _ = llm_pipe(store, llm_settings, fake_clock, ok(llm_reply(
        event_type="management_change", evidence_quote=quote, entities=[{"id": "RELIANCE", "polarity": 0.0}],
    )))
    add_article(store, 0, "Reliance Industries news", quote + " with immediate effect after a board discussion.",
                source="Mint", feed="mint", url="https://www.livemint.com/m")
    pipe.process_pending()
    (event,) = events(store)
    assert event["event_type"] == "management_change" and event["extraction_method"] == "rules+llm"


def test_a_failing_llm_falls_back_to_rules_and_raises_the_reduced_coverage_flag(store, llm_settings, fake_clock):
    pipe, session, extractor = llm_pipe(store, llm_settings, fake_clock, FakeResponse(500))
    story(store, 0, 1)
    add_article(store, 10, "TCS bags Rs 900 crore deal from European bank", "Multi-year deal.", source="Mint", feed="mint",
                url="https://www.livemint.com/tcs")
    stats = pipe.process_pending()
    assert stats.llm_failed == 2 and stats.llm_used == 0 and len(events(store)) == 2
    assert {e["extraction_method"] for e in events(store)} == {"rules"}
    assert store.kv_get("news.reduced_coverage") == "1" and store.kv_get("llm.failures") == "2"
    assert fake_clock.sleeps  # backoff used the injected clock, never real time
    extractor.max_attempts = 1
    session.responses = [ok(llm_reply())]
    add_article(store, 20, "Reliance Industries news", "Reliance Industries has " + QUOTE + " for solar modules.", source="Mint",
                feed="mint", url="https://www.livemint.com/ok")
    stats = pipe.process_pending()
    assert stats.llm_failed == 0 and stats.llm_used == 1 and store.kv_get("news.reduced_coverage") == "0"


def test_llm_budget_and_cache_show_up_in_the_run_stats(store, llm_settings, fake_clock):
    limited = dataclasses.replace(llm_settings, llm_max_articles_per_run=2)
    pipe, session, _ = llm_pipe(store, limited, fake_clock, ok(llm_reply()))
    for i, name in enumerate(("TCS", "Infosys", "Wipro", "Sun Pharma")):
        add_article(store, i, f"{name} bags Rs {900 + i} crore deal from a European bank", "Multi-year deal.", source="Mint",
                    feed="mint", url=f"https://www.livemint.com/{i}", seen=f"2026-11-01T10:{i:02d}:00Z")
    stats = pipe.process_pending()
    assert stats.llm_used == 2 and stats.llm_skipped_budget == 2 and len(session.calls) == 2
    for table in ("event_articles", "events"):
        store.conn.execute(f"DELETE FROM {table}")
    store.conn.execute("DELETE FROM kv WHERE key LIKE 'news.sig:%'")
    store.conn.execute("UPDATE news_raw SET processed = 0")
    stats = pipe.process_pending()
    assert stats.llm_cached == 2 and stats.llm_used == 2 and stats.llm_skipped_budget == 0  # cache hits cost no budget
    assert len(session.calls) == 4 and len(events(store)) == 4
