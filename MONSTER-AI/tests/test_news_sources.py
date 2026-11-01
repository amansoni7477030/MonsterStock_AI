"""News, filings and calendar collection: RSS parsing and de-duplication, per-feed outage isolation, GDELT, NSE filings and board
meetings, the economic calendar and raw-news retention (all against scripted HTTP, no network)."""

from __future__ import annotations

import json
import logging
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
    return NewsCollector(store, settings, clock=fake_clock)


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
    """The collectors build their own HttpClient for GDELT / the calendar: hand them a scripted one instead."""
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
    report = collector.collect_gdelt([{"id": "oil", "query": "crude"}, {"id": "fx", "query": "rupee"}])
    assert report.failed == 2 and report.ok == 0 and source_row(store, "gdelt")["consecutive_failures"] == 1

    late = collector.collect_gdelt([{"id": "oil", "query": "crude"}], deadline=fake_clock.monotonic() - 1)
    assert late.attempted == 0 and late.notes == ["deadline reached"]

    import dataclasses
    off = NewsCollector(store, dataclasses.replace(collector.settings, gdelt_enabled=False), clock=fake_clock)
    assert off.collect_gdelt([{"id": "oil", "query": "crude"}]).attempted == 0


def test_a_gdelt_reply_that_is_not_a_mapping_stores_nothing(collector, fake_clock, monkeypatch):
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, [1, 2, 3]))
    report = collector.collect_gdelt([{"id": "oil", "query": "crude"}])
    assert report.ok == 1 and report.new_items == 0


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


# ------------------------------------------------------------------------------- calendar and retention
def test_the_economic_calendar_is_upserted_by_event_and_bad_rows_are_skipped(collector, store, fake_clock, monkeypatch):
    payload = [
        {"date": "2026-11-04T14:00:00-05:00", "country": "USD", "title": "FOMC Statement", "impact": "High", "forecast": "", "previous": "4.00%"},
        {"date": "garbage", "country": "EUR", "title": "Broken date", "impact": "Low"},
        {"date": "2026-11-05T08:30:00-05:00", "country": "USD", "title": "", "impact": "Low"},
    ]
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, payload))
    assert collector.collect_econ_calendar() == 1
    payload[0]["forecast"] = "4.25%"
    assert collector.collect_econ_calendar() == 1
    rows = store.conn.execute("SELECT * FROM econ_calendar").fetchall()
    assert len(rows) == 1 and rows[0]["title"] == "FOMC Statement" and rows[0]["forecast"] == "4.25%"
    assert rows[0]["scheduled_at"].startswith("2026-11-04T19:00:00")


def test_an_economic_calendar_outage_is_recorded_not_raised(collector, store, fake_clock, monkeypatch):
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(503, b"down"))
    assert collector.collect_econ_calendar() == 0
    assert source_row(store, "econ_calendar")["consecutive_failures"] == 1


def calendar_item(date_text="2026-11-04T14:00:00-05:00", **over):
    return {"date": date_text, "country": "USD", "title": "FOMC Statement", "impact": "High", "forecast": "", "previous": "4.00%",
            **over}


def calendar_rows(store):
    return [dict(r) for r in store.conn.execute("SELECT * FROM econ_calendar ORDER BY scheduled_at, title").fetchall()]


def test_a_rescheduled_release_updates_its_row_instead_of_adding_a_duplicate(collector, store, fake_clock, monkeypatch):
    payload = [calendar_item()]
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, payload))
    assert collector.collect_econ_calendar() == 1
    (first,) = calendar_rows(store)
    fake_clock.advance(3600)
    payload[0].update(date="2026-11-04T14:30:00-05:00", impact="Medium", forecast="4.25%")  # the same release, half an hour later
    assert collector.collect_econ_calendar() == 1
    (row,) = calendar_rows(store)
    assert row["event_key"] == first["event_key"] and row["scheduled_at"].startswith("2026-11-04T19:30:00")
    assert row["impact"] == "Medium" and row["forecast"] == "4.25%"
    assert row["first_seen_at"] == first["first_seen_at"] and row["updated_at"] > first["updated_at"]


def test_the_release_date_is_the_feeds_own_date_so_an_evening_release_moved_by_an_hour_stays_one_row(collector, store, fake_clock,
                                                                                                  monkeypatch):
    payload = [calendar_item("2026-11-04T19:30:00-05:00", title="Fed Chair Speaks")]  # 00:30 UTC on the 5th
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, payload))
    collector.collect_econ_calendar()
    payload[0]["date"] = "2026-11-04T18:30:00-05:00"  # 23:30 UTC on the 4th: the UTC date moved, the release date did not
    collector.collect_econ_calendar()
    (row,) = calendar_rows(store)
    assert row["scheduled_at"].startswith("2026-11-04T23:30:00")


def test_the_same_title_on_another_date_or_in_another_country_is_a_different_release(collector, store, fake_clock, monkeypatch):
    payload = [calendar_item("2026-11-05T08:30:00-05:00", title="Unemployment Claims", impact="Low"),
               calendar_item("2026-11-12T08:30:00-05:00", title="Unemployment Claims", impact="Low"),
               calendar_item("2026-11-05T08:30:00-05:00", title="Unemployment Claims", impact="Low", country="CAD")]
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, payload))
    assert collector.collect_econ_calendar() == 3
    assert collector.collect_econ_calendar() == 3
    assert len(calendar_rows(store)) == 3


def test_two_same_titled_releases_on_one_date_keep_the_earlier_time_whatever_the_feed_order(collector, store, fake_clock, monkeypatch):
    later, earlier = (calendar_item("2026-11-04T15:00:00-05:00", title="FOMC Member Speaks"),
                      calendar_item("2026-11-04T09:00:00-05:00", title="FOMC Member Speaks"))
    payload = [later, earlier]
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, payload))
    assert collector.collect_econ_calendar() == 1
    payload[:] = [earlier, later]
    assert collector.collect_econ_calendar() == 1
    (row,) = calendar_rows(store)
    assert row["scheduled_at"].startswith("2026-11-04T14:00:00")  # 09:00 ET, deterministic


def test_rows_written_with_the_old_time_based_key_are_converted_not_duplicated(collector, store, fake_clock, monkeypatch):
    when = datetime(2026, 11, 4, 19, 0, tzinfo=timezone.utc)
    old_key = ns._hash(f"{ns.iso(when)}|USD|FOMC Statement")  # what a release of this time was keyed on before
    store.conn.execute(
        "INSERT INTO econ_calendar (event_key, country, title, impact, scheduled_at, forecast, previous, source, first_seen_at, "
        "updated_at) VALUES (?, 'USD', 'FOMC Statement', 'High', '2026-11-04T19:00:00Z', '', '4.00%', 'faireconomy', "
        "'2026-10-30T00:00:00Z', '2026-10-30T00:00:00Z')", (old_key,))
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, [calendar_item()]))
    assert collector.collect_econ_calendar() == 1
    (row,) = calendar_rows(store)
    assert row["event_key"] != old_key and row["first_seen_at"] == "2026-10-30T00:00:00Z"  # history kept, key migrated
    assert collector.collect_econ_calendar() == 1 and len(calendar_rows(store)) == 1


def test_a_calendar_item_whose_date_text_is_not_iso_falls_back_to_the_utc_date(collector, store, fake_clock, monkeypatch):
    payload = [calendar_item("04-11-2026 19:00:00")]  # parsed by parse_ts (as UTC) but not an ISO date: the UTC date is the key's
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, payload))
    assert collector.collect_econ_calendar() == 1
    payload[0]["date"] = "04-11-2026 20:00:00"  # rescheduled the same day
    assert collector.collect_econ_calendar() == 1
    (row,) = calendar_rows(store)
    assert row["scheduled_at"] == "2026-11-04T20:00:00Z"


def test_a_calendar_payload_that_cannot_be_used_is_a_failure_not_a_quiet_week(collector, store, fake_clock, monkeypatch, caplog):
    payload: list | dict = {"error": "rate limited"}  # an error object answered with HTTP 200
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(200, payload))
    with caplog.at_level(logging.WARNING, logger="ingestion.news_sources"):
        assert collector.collect_econ_calendar() == 0
        assert source_row(store, "econ_calendar")["consecutive_failures"] == 1
        payload = [{"date": "garbage", "title": "Broken date"}, "not a mapping", {"date": "2026-11-04T14:00:00-05:00", "title": ""}]
        assert collector.collect_econ_calendar() == 0  # something was sent, none of it readable: the format must have changed
        assert source_row(store, "econ_calendar")["consecutive_failures"] == 2
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == "ingestion.news_sources"]
    assert any("unexpected payload (dict)" in m for m in warned) and any("none of the 3 items" in m for m in warned)
    payload = []  # a genuinely empty calendar is fine
    assert collector.collect_econ_calendar() == 0 and source_row(store, "econ_calendar")["consecutive_failures"] == 0
    assert calendar_rows(store) == []


def test_swallowed_collection_failures_are_logged_as_warnings_without_urls(collector, store, fake_clock, monkeypatch, caplog):
    patch_http(monkeypatch, fake_clock, lambda url, kw: FakeResponse(503, b"down"))
    with caplog.at_level(logging.WARNING, logger="ingestion.news_sources"):
        assert collector.collect_econ_calendar() == 0
        assert collector.collect_board_meetings(FakeNse(fail=True), date(2026, 11, 1), date(2026, 11, 2), set()) == 0
        assert collector.collect_filings(FakeNse(fail=True), date(2026, 11, 1), date(2026, 11, 2), set()) == 0
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name == "ingestion.news_sources"]
    assert len(warned) == 3
    assert any("economic calendar" in m for m in warned) and any("board-meeting" in m for m in warned)
    assert any("filings" in m for m in warned)
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
