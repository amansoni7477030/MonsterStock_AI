"""News, filings and calendar collection: RSS parsing and de-duplication, per-feed outage isolation, GDELT, NSE filings and board
meetings, and raw-news retention (all against scripted HTTP, no network)."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone

import pytest

from app.errors import DataUnavailable
from ingestion import news_sources as ns
from ingestion.http import HttpClient
from ingestion.news_sources import FeedSpec, NewsCollector
from tests.helpers import FakeResponse, FakeSession

NOW = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)
FEED = FeedSpec("et_markets", "https://feeds.example.com/markets.xml", "Economic Times", "india", "markets")


def rfc822(moment: datetime) -> str:
    return moment.strftime("%a, %d %b %Y %H:%M:%S +0000")


def rss(*items: tuple[str, str, datetime | None, str]) -> bytes:
    body = "".join(
        f"<item><title>{title}</title><link>{link}</link>" + (f"<pubDate>{rfc822(when)}</pubDate>" if when else "")
        + f"<description>{summary}</description></item>" for title, link, when, summary in items)
    return f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>{body}</channel></rss>'.encode()


GOOD = rss(
    ("Reliance wins NTPC order", "https://news.example.com/a/1?utm_source=x&id=7", NOW - timedelta(hours=2), "A <b>big</b> order"),
    ("Sensex rallies", "https://news.example.com/a/2", NOW - timedelta(hours=5), "Markets up"),
    ("Ancient story", "https://news.example.com/a/3", NOW - timedelta(days=30), "too old"),
    ("From the future", "https://news.example.com/a/4", NOW + timedelta(days=3), "bad clock"),
    ("", "https://news.example.com/a/5", NOW, "no title"),
    ("Odd link", "ftp://news.example.com/a/6", NOW, "not http"),
    ("No timestamp at all", "https://news.example.com/a/7", None, "kept, published unknown"),
)


@pytest.fixture()
def collector(store, settings, fake_clock):
    fake_clock.set(NOW)
    news = NewsCollector(store, settings, clock=fake_clock)
    # hermetic: a test that needs the feed client scripts its own (with_session); an unscripted request fails instead of going out
    news.http = HttpClient("guard", session=FakeSession([]), clock=fake_clock, max_attempts=1)
    return news


def with_session(collector, fake_clock, script):
    collector.http = HttpClient("unit", session=FakeSession(script), clock=fake_clock, max_attempts=1)
    return collector


def stored(store):
    return [dict(r) for r in store.conn.execute("SELECT * FROM news_raw ORDER BY article_id").fetchall()]


def source_row(store, source_id):
    return dict(store.conn.execute("SELECT * FROM source_health WHERE source_id = ?", (source_id,)).fetchone())


# ------------------------------------------------------------------------------------------ registry / urls
def test_the_feed_registry_keeps_only_https_feeds_with_an_id(tmp_path):
    path = tmp_path / "feeds.json"
    path.write_text(json.dumps({
        "feeds": [{"id": "a", "url": "https://x.example/rss", "publisher": "A", "region": "india", "category": "markets"},
                  {"id": "b", "url": "http://insecure.example/rss"}, {"url": "https://noid.example/rss"},
                  {"id": "c", "url": "https://y.example/rss"}],
        "gdelt_queries": [{"id": "q1", "query": "india stocks", "category": "markets"}]}), encoding="utf-8")
    feeds, queries = ns.load_feed_registry(path)
    assert [f.id for f in feeds] == ["a", "c"] and feeds[1].region == "global" and feeds[1].kind == "media"
    assert queries == [{"id": "q1", "query": "india stocks", "category": "markets"}]


def test_an_unreadable_registry_is_a_non_retryable_outage(tmp_path):
    for content in (None, "{not json"):
        path = tmp_path / "feeds.json"
        if content:
            path.write_text(content, encoding="utf-8")
        with pytest.raises(DataUnavailable) as info:
            ns.load_feed_registry(path)
        assert info.value.retryable is False


def test_canonical_urls_ignore_tracking_noise_case_fragments_and_param_order():
    base = ns.canonical_url("https://News.Example.com/a/b/?id=7&z=1")
    assert ns.canonical_url("https://news.example.com/a/b?z=1&id=7&utm_campaign=x&fbclid=abc#top") == base
    assert ns.canonical_url("  HTTPS://NEWS.EXAMPLE.COM/a/b/?id=7&z=1 ") == base
    assert ns.canonical_url("https://news.example.com") == "https://news.example.com/"


# ----------------------------------------------------------------------------------------------------- RSS
def test_rss_items_are_stored_cleaned_and_filtered(collector, store, fake_clock):
    with_session(collector, fake_clock, [FakeResponse(200, GOOD)])
    report = collector.collect_rss([FEED])
    assert report.attempted == report.ok == 1 and report.failed == 0 and report.new_items == 4
    rows = stored(store)
    assert [r["title"] for r in rows] == ["Reliance wins NTPC order", "Sensex rallies", "From the future", "No timestamp at all"]
    first = rows[0]
    assert first["source"] == "Economic Times" and first["feed"] == "et_markets" and first["region"] == "india"
    assert first["summary"] == "A big order" and first["first_seen_at"].startswith("2026-11-02T03:00:00")
    assert first["published_at"].startswith("2026-11-02T01:00:00")
    assert rows[2]["published_at"] is None and rows[3]["published_at"] is None  # an implausible or missing time is "unknown"
    assert source_row(store, "rss_news")["consecutive_failures"] == 0


def test_the_same_story_is_never_stored_twice_even_with_a_different_tracking_tail(collector, store, fake_clock):
    with_session(collector, fake_clock, [FakeResponse(200, GOOD)] * 2)
    collector.collect_rss([FEED])
    again = collector.collect_rss([FEED])
    assert again.ok == 1 and again.new_items == 0 and len(stored(store)) == 4


def test_a_dead_feed_is_counted_noted_and_does_not_stop_the_others(collector, store, fake_clock):
    second = FeedSpec("mint", "https://feeds.example.com/mint.xml", "Mint", "india", "markets")
    third = FeedSpec("gn", "https://feeds.example.com/gn.xml", "Google News", "global", "world")
    script = [FakeResponse(500, b"oops"), FakeResponse(200, rss(("Mint story", "https://mint.example.com/1", NOW, "s"))),
              RuntimeError("socket exploded")]
    with_session(collector, fake_clock, script)
    report = collector.collect_rss([FEED, second, third])
    assert (report.attempted, report.ok, report.failed, report.new_items) == (3, 1, 2, 1)
    assert any(n.startswith("et_markets:") for n in report.notes) and "gn: RuntimeError" in report.notes
    assert report.success_ratio == pytest.approx(1 / 3) and source_row(store, "rss_news")["consecutive_failures"] == 1


def test_an_unparseable_page_without_entries_is_a_failure(collector, fake_clock):
    with_session(collector, fake_clock, [FakeResponse(200, b"<html><body>Please enable cookies</body>")])
    report = collector.collect_rss([FEED])
    assert report.failed == 1 and "unparseable feed" in report.notes[0]


def test_a_feed_that_keeps_failing_is_parked_for_a_day_then_retried(collector, store, fake_clock):
    for _ in range(ns.FEED_DISABLE_AFTER):
        with_session(collector, fake_clock, [FakeResponse(503, b"down")])
        collector.collect_rss([FEED])
    skipped = collector.collect_rss([FEED])
    assert skipped.skipped == 1 and skipped.attempted == 0  # parked: no request is even made
    fake_clock.advance(ns.FEED_DISABLE_HOURS * 3600 + 60)
    with_session(collector, fake_clock, [FakeResponse(200, GOOD)])
    assert collector.collect_rss([FEED]).ok == 1
    assert store.kv_get(f"feed:{FEED.id}:fails") == "0"


def test_the_deadline_and_region_filters_limit_what_is_fetched(collector, fake_clock):
    other = FeedSpec("bbc", "https://feeds.example.com/bbc.xml", "BBC", "global", "world")
    with_session(collector, fake_clock, [FakeResponse(200, GOOD)])
    only_india = collector.collect_rss([other, FEED], region="india")
    assert only_india.attempted == 1 and only_india.ok == 1

    late = collector.collect_rss([FEED, other], deadline=fake_clock.monotonic() - 1)
    assert late.attempted == 0 and late.notes == ["deadline reached; remaining feeds deferred"]


def test_a_mostly_failing_run_marks_the_source_unhealthy(collector, store, fake_clock):
    with_session(collector, fake_clock, [FakeResponse(500, b"x")] * 2)
    collector.collect_rss([FEED, FeedSpec("b", "https://b.example.com/rss", "B", "india", "m")])
    row = source_row(store, "rss_news")
    assert row["consecutive_failures"] == 1 and "2/2 feeds failed" in row["last_error"]


# --------------------------------------------------------------------------------------------------- GDELT
def patch_http(monkeypatch, fake_clock, handler):
    """The collectors build their own HttpClient for GDELT: hand them a scripted one instead."""
    sessions = []

    def factory(source, settings, **kwargs):
        session = FakeSession(handler)
        sessions.append(session)
        return HttpClient(source, session=session, clock=fake_clock, max_attempts=1)

    monkeypatch.setattr(ns.HttpClient, "from_settings", staticmethod(factory))
    return sessions


GDELT = {"articles": [
    {"url": "https://www.reuters.com/world/x?utm_source=feed", "title": "Oil jumps as supply tightens", "domain": "reuters.com",
     "language": "English", "seendate": "20261102T020000Z"},
    {"url": "https://www.lemonde.fr/y", "title": "Le pétrole monte", "domain": "lemonde.fr", "language": "French",
     "seendate": "20261102T020000Z"},
    {"url": "ftp://bad.example/z", "title": "Bad link", "domain": "bad.example", "language": "English"},
    {"url": "https://www.ft.com/w", "title": "", "domain": "ft.com", "language": "English"},
]}


def test_gdelt_keeps_english_http_articles_and_deduplicates(collector, store, fake_clock, monkeypatch):
    sessions = patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, GDELT))
    queries = [{"id": "oil", "query": "crude oil", "category": "commodities"}]
    first = collector.collect_gdelt(queries)
    assert first.attempted == first.ok == 1 and first.new_items == 1
    row = stored(store)[0]
    assert row["source"] == "reuters.com" and row["feed"] == "gdelt:oil" and row["region"] == "global"
    assert row["category"] == "commodities" and row["published_at"].startswith("2026-11-02T02:00:00")
    params = sessions[0].calls[0][2]["params"]
    assert params["query"] == "crude oil" and params["mode"] == "artlist" and params["timespan"].endswith("h")
    assert collector.collect_gdelt(queries).new_items == 0


def test_gdelt_outages_are_reported_and_a_disabled_source_is_skipped(collector, store, fake_clock, monkeypatch):
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(429, b"slow down"))
    sources_only(collector, gdelt_fallback=False)  # GDELT alone: the fallback has its own tests below
    report = collector.collect_gdelt([{"id": "oil", "query": "crude"}, {"id": "fx", "query": "rupee"}])
    assert report.failed == 2 and report.ok == 0 and source_row(store, "gdelt")["consecutive_failures"] == 1

    late = collector.collect_gdelt([{"id": "oil", "query": "crude"}], deadline=fake_clock.monotonic() - 1)
    assert late.attempted == 0 and late.notes == ["deadline reached"]

    import dataclasses
    off = NewsCollector(store, dataclasses.replace(collector.settings, gdelt_enabled=False), clock=fake_clock)
    assert off.collect_gdelt([{"id": "oil", "query": "crude"}]).attempted == 0


def test_a_gdelt_reply_that_is_not_a_mapping_stores_nothing(collector, fake_clock, monkeypatch):
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, [1, 2, 3]))
    sources_only(collector, gdelt_fallback=False)
    report = collector.collect_gdelt([{"id": "oil", "query": "crude"}])
    assert report.ok == 1 and report.new_items == 0


# ------------------------------------------------------------------- Google News search as the GDELT fallback
@pytest.mark.parametrize(("query", "hours", "expected"), [
    ("(oil OR crude OR OPEC OR Brent) sourcelang:eng", 36, "(oil OR crude OR OPEC OR Brent) when:2d"),
    ('("Middle East" OR Iran) (oil OR shipping) sourcelang:eng', 24, '("Middle East" OR Iran) (oil OR shipping) when:1d'),
    ("sourcelang:eng Nifty sourcecountry:india tone<-5 toneabs>3 theme:ECON_STOCKMARKET", 1, "Nifty when:1d"),
    ("( RBI OR \"India GDP\" ) sourcelang:eng", 48, '(RBI OR "India GDP") when:2d'),
    ("domain:reuters.com domainis:ft.com oil", 49, "oil when:3d"),
    ("a theme park OR a tone poem", 24, "a theme park OR a tone poem when:1d"),  # a word that only looks like an operator is kept
])
def test_a_gdelt_query_becomes_the_google_news_search_that_asks_the_same(query, hours, expected):
    assert ns.gnews_search(query, hours) == expected


@pytest.mark.parametrize("query", ["", "   ", None, "sourcelang:eng", "() sourcelang:eng", "tone<-5 theme:X"])
def test_a_query_with_nothing_left_to_search_for_gives_no_search(query):
    assert ns.gnews_search(query, 24) == ""


@pytest.mark.parametrize(("hours", "days"), [(1, 1), (24, 1), (25, 2), (36, 2), (48, 2), (49, 3), (0, 1)])
def test_the_age_limit_is_in_whole_days_and_never_below_one(hours, days):
    assert ns.gnews_search("oil", hours) == f"oil when:{days}d"


def test_the_edition_is_the_english_language_edition_of_a_country_and_falls_back_to_the_us():
    assert ns.gnews_params("oil when:1d") == {"q": "oil when:1d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    assert ns.gnews_params("rbi", "IN") == ns.gnews_params("rbi", "in") == {"q": "rbi", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    for bad in ("", None, "india", "U1", "  ", "IN&gl=XX"):
        assert ns.gnews_params("oil", bad)["gl"] == "US", bad


def google_items(search: str, n: int = 2, *, age_h: int = 2) -> bytes:
    """A Google News answer whose items depend on the search, so two topics never share a link."""
    slug = re.sub(r"\W+", "-", search).strip("-")[:30]
    return rss(*((f"Headline {i} of {slug} - Reuters", f"https://news.google.com/rss/articles/{slug}-{i}?oc=5",
                  NOW - timedelta(hours=age_h + i), "") for i in range(n)))


TOPICS = [{"id": "oil", "query": "(oil OR crude) sourcelang:eng", "category": "energy"},
          {"id": "india_macro", "query": '(RBI OR "India GDP") sourcelang:eng', "category": "india_macro", "edition": "IN"},
          {"id": "semis", "query": "(semiconductor OR TSMC) sourcelang:eng", "category": "technology"}]
GDELT_DOWN = FakeResponse(429, b"Please limit requests to one every 5 seconds")


def route(collector, fake_clock, monkeypatch, *, gdelt, google=lambda url, kw: FakeResponse(200, google_items(kw["params"]["q"]))):
    """GDELT answers as ``gdelt`` says and Google News as ``google`` says; returns (GDELT sessions, the Google session)."""
    sessions = patch_http(monkeypatch, fake_clock, gdelt)
    with_session(collector, fake_clock, google)
    return sessions, collector.http.session


def sources_only(collector, **over):
    import dataclasses

    collector.settings = dataclasses.replace(collector.settings, **over)
    return collector


def test_when_gdelt_answers_google_is_never_asked(collector, store, fake_clock, monkeypatch):
    gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: FakeResponse(200, GDELT))
    report = collector.collect_gdelt(TOPICS)
    assert (report.attempted, report.ok, report.failed, report.fallback) == (3, 3, 0, 0) and report.new_items == 1
    assert len(gdelt[0].calls) == 3 and google.calls == []
    assert {row["feed"] for row in stored(store)} == {"gdelt:oil"} and source_row(store, "gdelt")["consecutive_failures"] == 0


def test_after_two_failed_queries_in_a_row_the_rest_of_the_run_goes_to_google(collector, store, fake_clock, monkeypatch):
    gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN)
    report = collector.collect_gdelt(TOPICS)
    assert (report.attempted, report.ok, report.failed, report.fallback) == (3, 3, 0, 3) and report.new_items == 6
    assert len(gdelt[0].calls) == 2  # the third topic never waits for a GDELT that is not answering
    assert [kw["params"]["q"] for _, _, kw in google.calls] == ["(oil OR crude) when:2d", '(RBI OR "India GDP") when:2d',
                                                               "(semiconductor OR TSMC) when:2d"]
    assert any("remaining topics go to Google News search" in note for note in report.notes)
    row = stored(store)[0]
    assert (row["feed"], row["source"], row["region"], row["category"]) == ("gn_topic:oil", "Google News", "global", "energy")
    assert row["url"].startswith("https://news.google.com/rss/articles/") and row["title"].endswith(" - Reuters")
    assert row["published_at"].startswith("2026-11-02T01:00")
    assert source_row(store, "gdelt")["consecutive_failures"] == 0 and source_row(store, "gnews_search")["consecutive_failures"] == 0


def test_the_request_to_google_names_the_edition_and_the_age_and_has_no_gdelt_operator(collector, fake_clock, monkeypatch):
    _gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN)
    collector.collect_gdelt(TOPICS)
    assert {url for _, url, _ in google.calls} == {ns.GNEWS_URL}
    assert [kw["params"] for _, _, kw in google.calls] == [
        {"q": "(oil OR crude) when:2d", "hl": "en-US", "gl": "US", "ceid": "US:en"},
        {"q": '(RBI OR "India GDP") when:2d', "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},  # the India topic asks the Indian edition
        {"q": "(semiconductor OR TSMC) when:2d", "hl": "en-US", "gl": "US", "ceid": "US:en"}]
    assert "sourcelang" not in " ".join(kw["params"]["q"] for _, _, kw in google.calls)


def test_one_failed_query_does_not_end_gdelt_for_the_run_and_a_topic_gdelt_answers_with_nothing_is_read_from_google(
        collector, store, fake_clock, monkeypatch):
    answers = iter([GDELT_DOWN, FakeResponse(200, {"articles": []}), GDELT_DOWN])
    gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: next(answers))
    report = collector.collect_gdelt(TOPICS)
    assert len(gdelt[0].calls) == 3  # never two failures in a row: GDELT is tried for every topic
    assert len(google.calls) == 3 and (report.ok, report.failed, report.fallback) == (3, 0, 3)
    assert not any("remaining topics" in note for note in report.notes)


def test_a_topic_gdelt_answered_is_not_asked_again_and_the_failure_count_starts_over(collector, store, fake_clock, monkeypatch):
    answers = iter([GDELT_DOWN, FakeResponse(200, GDELT), GDELT_DOWN])
    gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: next(answers))
    report = collector.collect_gdelt(TOPICS)
    assert len(gdelt[0].calls) == 3 and len(google.calls) == 2 and report.fallback == 2
    assert {row["feed"] for row in stored(store)} == {"gn_topic:oil", "gdelt:india_macro", "gn_topic:semis"}


def test_with_gdelt_switched_off_google_search_does_the_discovery_alone(collector, store, fake_clock, monkeypatch):
    gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: pytest.fail("GDELT must not be asked"))
    report = sources_only(collector, gdelt_enabled=False).collect_gdelt(TOPICS)
    assert gdelt == [] and len(google.calls) == 3  # not even a GDELT client was built
    assert (report.attempted, report.ok, report.failed, report.fallback) == (3, 3, 0, 3)
    assert source_row(store, "gdelt")["consecutive_failures"] == 0


def test_with_the_fallback_switched_off_gdelt_behaves_as_it_always_did(collector, store, fake_clock, monkeypatch):
    gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN)
    report = sources_only(collector, gdelt_fallback=False).collect_gdelt(TOPICS)
    assert (report.attempted, report.ok, report.failed, report.fallback) == (3, 0, 3, 0) and report.new_items == 0
    assert len(gdelt[0].calls) == 3 and google.calls == []  # no give-up either: every topic gets its try
    row = source_row(store, "gdelt")
    assert row["consecutive_failures"] == 1 and row["last_error"] == "queries failing (rate limit?)"
    assert store.conn.execute("SELECT COUNT(*) FROM source_health WHERE source_id = 'gnews_search'").fetchone()[0] == 0  # never used


def test_with_both_switched_off_nothing_is_asked(collector, fake_clock, monkeypatch):
    gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: pytest.fail("no request expected"),
                          google=lambda url, kw: pytest.fail("no request expected"))
    report = sources_only(collector, gdelt_enabled=False, gdelt_fallback=False).collect_gdelt(TOPICS)
    assert report.attempted == 0 and gdelt == [] and google.calls == []


def test_a_topic_that_neither_source_answers_is_a_coverage_gap_and_the_reason_is_kept(collector, store, fake_clock, monkeypatch):
    _gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN,
                           google=lambda url, kw: FakeResponse(500, b"oops"))
    report = collector.collect_gdelt(TOPICS)
    assert (report.attempted, report.ok, report.failed, report.fallback) == (3, 0, 3, 0)
    assert sum("Google News search:" in note for note in report.notes) == 3
    gdelt_row, google_row = source_row(store, "gdelt"), source_row(store, "gnews_search")
    assert gdelt_row["consecutive_failures"] == 1 and gdelt_row["last_error"] == "GDELT and Google News search are not answering"
    assert google_row["consecutive_failures"] == 1 and google_row["last_error"] == "Google News search is not answering"
    assert len(google.calls) == 3


def test_a_page_that_is_not_a_feed_and_an_empty_feed_are_failures_not_successes(collector, store, fake_clock, monkeypatch):
    for answer in (b"<html><body>Before you continue to Google</body></html>", rss(), b"\x00\x01 not xml at all <<<", b""):
        route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: pytest.fail("off"), google=lambda url, kw, a=answer: FakeResponse(200, a))
        report = sources_only(collector, gdelt_enabled=False).collect_gdelt(TOPICS[:1])
        assert (report.ok, report.failed) == (0, 1), answer
        assert "Google News search:" in report.notes[0] and ("no items" in report.notes[0] or "not a readable feed" in report.notes[0]), report.notes
    assert stored(store) == []


def test_google_answers_one_topic_and_fails_the_next_without_ending_the_run(collector, store, fake_clock, monkeypatch):
    def google(url, kw):
        return FakeResponse(500, b"x") if "RBI" in kw["params"]["q"] else FakeResponse(200, google_items(kw["params"]["q"]))

    route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN, google=google)
    report = collector.collect_gdelt(TOPICS)
    assert (report.ok, report.failed, report.fallback) == (2, 1, 2) and source_row(store, "gdelt")["consecutive_failures"] == 0
    assert {row["feed"] for row in stored(store)} == {"gn_topic:oil", "gn_topic:semis"}


def test_an_unexpected_error_in_one_google_topic_is_contained(collector, store, fake_clock, monkeypatch, caplog):
    calls = []

    def google(url, kw):
        calls.append(kw["params"]["q"])
        if len(calls) == 1:
            raise ValueError("boom")
        return FakeResponse(200, google_items(kw["params"]["q"]))

    route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN, google=google)
    with caplog.at_level(logging.WARNING):
        report = collector.collect_gdelt(TOPICS)
    assert len(calls) == 3 and (report.ok, report.failed) == (2, 1)
    assert "ValueError" in " ".join(report.notes) and "unexpected ValueError" in caplog.text


def test_google_items_older_than_the_lookback_are_dropped_and_a_second_run_adds_nothing(collector, store, fake_clock, monkeypatch):
    def google(url, kw):
        return FakeResponse(200, rss(("Fresh - Reuters", "https://news.google.com/rss/articles/f1?oc=5", NOW - timedelta(hours=2), ""),
                                      ("Stale - Reuters", "https://news.google.com/rss/articles/s1?oc=5", NOW - timedelta(days=5), "")))

    route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN, google=google)
    first = collector.collect_gdelt(TOPICS[:1], lookback_hours=12)
    assert first.new_items == 1 and [row["title"] for row in stored(store)] == ["Fresh - Reuters"]
    assert collector.collect_gdelt(TOPICS[:1], lookback_hours=12).new_items == 0
    assert collector.http.session.calls[0][2]["params"]["q"] == "(oil OR crude) when:1d"  # 12 hours is asked as one day


def test_a_deadline_ends_the_fallback_too(collector, fake_clock, monkeypatch):
    _gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: GDELT_DOWN)
    late = collector.collect_gdelt(TOPICS, deadline=fake_clock.monotonic() - 1)
    assert late.attempted == 0 and late.notes == ["deadline reached"] and google.calls == []


def test_a_topic_whose_query_is_only_gdelt_operators_cannot_be_searched_and_is_reported(collector, fake_clock, monkeypatch):
    _gdelt, google = route(collector, fake_clock, monkeypatch, gdelt=lambda url, kw: pytest.fail("off"))
    report = sources_only(collector, gdelt_enabled=False).collect_gdelt([{"id": "bare", "query": "sourcelang:eng"}])
    assert (report.ok, report.failed) == (0, 1) and "nothing left to search for" in report.notes[0] and google.calls == []


def test_the_fallback_items_are_aggregator_grade_like_the_google_news_feeds():
    from news_intelligence import source_quality as sq

    assert sq.quality_for("Google News", "gn_topic:oil", None) == (0.4, "aggregator")
    link = "https://news.google.com/rss/articles/a1?oc=5"
    assert sq.quality_for("Reuters", "gn_topic:oil", link) == (0.6, "business_media")  # a publisher behind the redirect is capped
    assert sq.quality_for("reuters.com", "gdelt:oil", "https://www.reuters.com/x")[1] == "wire"  # GDELT links the publisher itself


# ------------------------------------------------------------------------------------- filings, meetings
class FakeNse:
    def __init__(self, filings=None, meetings=None, fail=False):
        self.filings, self.meetings, self.fail = filings or [], meetings or [], fail

    def announcements(self, start, end, symbols):
        if self.fail:
            raise DataUnavailable("nse", "blocked")
        return self.filings

    def event_calendar(self, start, end, symbols):
        if self.fail:
            raise DataUnavailable("nse", "blocked")
        return self.meetings


FILING = {"exchange": "NSE", "symbol": "RELIANCE", "isin": "INE002A01018", "company": "Reliance Industries Limited",
          "category": "Award of Order / Receipt of Order", "subject": "Award of order", "text": "Received an order",
          "attachment_url": None, "disseminated_at": "2026-11-02T01:00:00Z", "source_ref": "NSE-1"}


def test_filings_are_stored_once_and_an_nse_block_is_recorded(collector, store):
    nse = FakeNse(filings=[FILING, {**FILING, "source_ref": "NSE-2", "symbol": "TCS"}])
    assert collector.collect_filings(nse, date(2026, 11, 1), date(2026, 11, 2), {"RELIANCE", "TCS"}) == 2
    assert collector.collect_filings(nse, date(2026, 11, 1), date(2026, 11, 2), {"RELIANCE"}) == 0  # same refs: no duplicates
    assert store.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0] == 2
    assert source_row(store, "nse_announcements")["consecutive_failures"] == 0

    assert collector.collect_filings(FakeNse(fail=True), date(2026, 11, 1), date(2026, 11, 2), set()) == 0
    assert source_row(store, "nse_announcements")["consecutive_failures"] == 1


def test_board_meetings_track_how_far_back_the_calendar_is_known(collector, store):
    meetings = [{"symbol": "TCS", "meeting_date": "2026-11-10", "purpose": "Financial Results", "is_results": True},
                {"symbol": "INFY", "meeting_date": "2026-11-05", "purpose": "Dividend", "is_results": False}]
    assert collector.collect_board_meetings(FakeNse(meetings=meetings), date(2026, 11, 1), date(2026, 11, 30), {"TCS", "INFY"}) == 2
    assert store.kv_get("board_meetings.coverage_start") == "2026-11-05"
    later = [{"symbol": "SBIN", "meeting_date": "2026-12-01", "purpose": "Results", "is_results": True}]
    collector.collect_board_meetings(FakeNse(meetings=later), date(2026, 12, 1), date(2026, 12, 31), {"SBIN"})
    assert store.kv_get("board_meetings.coverage_start") == "2026-11-05"  # only ever moves earlier
    earlier = [{"symbol": "LT", "meeting_date": "2026-10-20", "purpose": "Results", "is_results": True}]
    collector.collect_board_meetings(FakeNse(meetings=earlier), date(2026, 10, 1), date(2026, 10, 31), {"LT"})
    assert store.kv_get("board_meetings.coverage_start") == "2026-10-20"
    assert collector.collect_board_meetings(FakeNse(fail=True), date(2026, 11, 1), date(2026, 11, 2), set()) == 0


# ------------------------------------------------------------------------------------------- retention
def test_swallowed_collection_failures_are_logged_as_warnings_without_urls(collector, caplog):
    with caplog.at_level(logging.WARNING, logger="ingestion.news_sources"):
        assert collector.collect_board_meetings(FakeNse(fail=True), date(2026, 11, 1), date(2026, 11, 2), set()) == 0
        assert collector.collect_filings(FakeNse(fail=True), date(2026, 11, 1), date(2026, 11, 2), set()) == 0
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == "ingestion.news_sources"]
    assert len(warned) == 2
    assert any("board-meeting" in m for m in warned) and any("filings" in m for m in warned)
    assert not any("?" in m or "://" in m for m in warned)  # never a URL, let alone its query string


def test_raw_news_retention_only_removes_old_processed_rows(collector, store, fake_clock):
    with_session(collector, fake_clock, [FakeResponse(200, GOOD)])
    collector.collect_rss([FEED])
    old = (NOW - timedelta(days=collector.settings.news_retention_days + 5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store.conn.execute("UPDATE news_raw SET first_seen_at = ?, processed = 1 WHERE article_id = 1", (old,))
    store.conn.execute("UPDATE news_raw SET first_seen_at = ?, processed = 0 WHERE article_id = 2", (old,))
    assert collector.prune_raw() == 1  # the unprocessed old row stays until the pipeline has seen it
    assert [r["article_id"] for r in stored(store)] == [2, 3, 4]
    assert collector.prune_raw(retention_days=0) == 0  # 0 falls back to the configured retention
    store.conn.execute("UPDATE news_raw SET processed = 2 WHERE article_id = 2")  # the pipeline gave up on it: no reason to keep it
    assert collector.prune_raw() == 1 and [r["article_id"] for r in stored(store)] == [3, 4]
