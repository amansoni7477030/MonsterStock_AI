"""Bulk fetching: Yahoo's multi-symbol spark probe (skip only when certain) and conditional GETs for RSS feeds."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from app.clock import iso
from app.errors import DataUnavailable
from ingestion import news_sources as ns
from ingestion import pipeline as pl
from ingestion.http import HttpClient
from ingestion.news_sources import FeedSpec, NewsCollector
from ingestion.series import SERIES_BY_KEY
from ingestion.yahoo import SPARK_MAX_SYMBOLS, SparkPoint, YahooClient, parse_spark
from tests.helpers import FakeResponse, FakeSession
from tests.test_ingestion_pipeline import FakeYahoo, bars, series

UTC = timezone.utc
NOW = datetime(2026, 11, 2, 3, 30, tzinfo=UTC)
ET_CLOSE_OCT30 = datetime(2026, 10, 30, 13, 30, tzinfo=UTC)  # 09:30 New York: the start of the 30 Oct daily bar


# ------------------------------------------------------------------------------------- spark parsing
SPARK = {
    "^GSPC": {"symbol": "^GSPC", "timestamp": [1793021400, 1793107800, 1793194200, 1793280600, 1793367000],
              "close": [7676.77, 7722.6, 7743.98, 7720.39, 7580.07], "chartPreviousClose": 7720.39},
    "CL=F": {"symbol": "CL=F", "timestamp": [1793021400, 1793107800, 1793194200], "close": [64.2, 63.9, None]},
    "USDINR=X": {"symbol": "USDINR=X", "timestamp": [1793021400], "close": [None]},
    "EMPTY": {"timestamp": [], "close": []},
    "SHORT": {"timestamp": [1793021400, 1793107800], "close": [10.0]},
    "BADTS": {"timestamp": ["soon"], "close": [1.0]},
    "NOTADICT": "oops",
}


def test_spark_keeps_the_newest_usable_point_of_each_symbol():
    points = parse_spark(SPARK)
    assert points["^GSPC"] == SparkPoint(datetime.fromtimestamp(1793367000, UTC), 7580.07)
    assert points["CL=F"].close == 63.9 and points["CL=F"].ts == datetime.fromtimestamp(1793107800, UTC)  # newest NON-null close
    assert points["SHORT"].close == 10.0
    assert set(points) == {"^GSPC", "CL=F", "SHORT"}  # nothing usable -> omitted, so the caller fetches it normally


@pytest.mark.parametrize("payload", [None, [], "x", {"spark": {"result": None, "error": {"code": "Bad Request"}}}, {}])
def test_unexpected_spark_payloads_yield_no_points(payload):
    assert parse_spark(payload) == {}


def yahoo_with(handler, fake_clock, settings):
    return YahooClient(settings, http=HttpClient("yahoo", session=FakeSession(handler), clock=fake_clock, max_attempts=1), clock=fake_clock)


def test_spark_requests_are_chunked_to_twenty_symbols_and_deduplicated(fake_clock, settings):
    seen: list[list[str]] = []

    def handler(url, kwargs):
        symbols = kwargs["params"]["symbols"].split(",")
        seen.append(symbols)
        return FakeResponse(200, {s: {"timestamp": [1793367000], "close": [1.0]} for s in symbols})

    client = yahoo_with(handler, fake_clock, settings)
    wanted = [f"S{i}" for i in range(45)] + ["S3", "S7"]
    points = client.spark_latest(wanted)
    assert [len(chunk) for chunk in seen] == [20, 20, 5] and max(len(c) for c in seen) == SPARK_MAX_SYMBOLS
    assert len(points) == 45 and {"range": "5d", "interval": "1d"}.items() <= client.http.session.calls[0][2]["params"].items()
    assert client.spark_latest([]) == {}


def test_spark_falls_back_to_the_second_yahoo_host_and_gives_up_only_when_both_fail(fake_clock, settings):
    hosts: list[str] = []

    def first_host_down(url, kwargs):
        hosts.append(url.split("/")[2])
        return FakeResponse(503) if "query1" in url else FakeResponse(200, {"A": {"timestamp": [1793367000], "close": [2.0]}})

    assert yahoo_with(first_host_down, fake_clock, settings).spark_latest(["A"])["A"].close == 2.0
    assert hosts == ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
    with pytest.raises(DataUnavailable):
        yahoo_with(lambda url, kwargs: FakeResponse(503), fake_clock, settings).spark_latest(["A"])


# ---------------------------------------------------------------------------------- the safe skip rule
GLOBAL_SERIES = ("^GSPC", "^IXIC", "^DJI", "^RUT", "CL=F", "GC=F")


class ProbeYahoo(FakeYahoo):
    """FakeYahoo plus the bulk endpoint: ``spark`` maps yahoo symbol -> SparkPoint (missing = not returned)."""

    def __init__(self, script, spark, error=None):
        super().__init__(script)
        self.spark, self.error, self.spark_calls = spark, error, []

    def spark_latest(self, symbols):
        self.spark_calls.append(list(symbols))
        if self.error:
            raise self.error
        return {s: p for s, p in self.spark.items() if s in symbols}


def iid_of(store, key):
    return store.instrument_id(key, SERIES_BY_KEY[key].kind)


@pytest.fixture()
def seeded(store):
    """Six global series with a stored 30 Oct bar and a recent full fetch."""
    pl.ensure_series_instruments(store)
    for key in GLOBAL_SERIES:
        store.upsert_bars(iid_of(store, key), bars({"2026-10-29": 100.0, "2026-10-30": 101.0}), source="yahoo", source_rank=1)
        store.kv_set(f"yahoo:fetched:{key}", iso(NOW - timedelta(hours=3)))
    return store


def same_as_stored() -> dict[str, SparkPoint]:
    return {key: SparkPoint(ET_CLOSE_OCT30, 101.0) for key in GLOBAL_SERIES}


def refresh(store, yahoo, **kw):
    return pl.refresh_daily_bars(store, yahoo, now=NOW, years=10, symbols=set(GLOBAL_SERIES), include_equities=False, **kw)


def script_for(*keys):
    return {SERIES_BY_KEY[k].yahoo: [series({"2026-10-30": 101.0})] for k in keys}


def test_series_confirmed_unchanged_by_the_bulk_probe_cost_no_request_at_all(seeded):
    yahoo = ProbeYahoo(script_for(*GLOBAL_SERIES), same_as_stored())
    report = refresh(seeded, yahoo)
    assert len(yahoo.spark_calls) == 1 and sorted(yahoo.spark_calls[0]) == sorted(GLOBAL_SERIES)  # ONE bulk request
    assert yahoo.calls == [] and report.ok == report.unchanged == 6 and report.failed == 0 and report.success_ratio == 1.0


@pytest.mark.parametrize("change, expect_fetch", [
    ("newer_date", True), ("different_close", True), ("missing_from_spark", True), ("older_date_same_close", False),
    ("tiny_float_noise", False),
])
def test_any_doubt_means_the_normal_fetch(seeded, change, expect_fetch):
    spark = same_as_stored()
    spark["CL=F"] = {
        "newer_date": SparkPoint(ET_CLOSE_OCT30 + timedelta(days=1), 101.0), "different_close": SparkPoint(ET_CLOSE_OCT30, 101.5),
        "missing_from_spark": None, "older_date_same_close": SparkPoint(ET_CLOSE_OCT30 - timedelta(days=2), 101.0),
        "tiny_float_noise": SparkPoint(ET_CLOSE_OCT30, 101.0 + 1e-6),
    }[change]
    if spark["CL=F"] is None:
        del spark["CL=F"]
    yahoo = ProbeYahoo(script_for(*GLOBAL_SERIES), spark)
    report = refresh(seeded, yahoo)
    assert [c[0] for c in yahoo.calls] == (["CL=F"] if expect_fetch else [])
    assert report.ok == 6 and report.unchanged == (5 if expect_fetch else 6)


def test_a_series_not_fetched_in_full_for_a_day_is_always_fetched(seeded):
    seeded.kv_set("yahoo:fetched:^GSPC", iso(NOW - timedelta(hours=25)))
    seeded.kv_delete("yahoo:fetched:^IXIC")  # never fetched by this code: also not eligible
    yahoo = ProbeYahoo(script_for(*GLOBAL_SERIES), same_as_stored())
    report = refresh(seeded, yahoo)
    assert sorted(c[0] for c in yahoo.calls) == ["^GSPC", "^IXIC"] and "^GSPC" not in yahoo.spark_calls[0]
    assert report.unchanged == 4
    assert seeded.kv_get("yahoo:fetched:^GSPC") == iso(NOW) and seeded.kv_get("yahoo:fetched:^IXIC") == iso(NOW)  # marker renewed


def test_a_failing_probe_costs_requests_never_data(seeded, caplog):
    yahoo = ProbeYahoo(script_for(*GLOBAL_SERIES), {}, error=DataUnavailable("yahoo", "HTTP 429"))
    with caplog.at_level(logging.WARNING, logger="ingestion.pipeline"):
        report = refresh(seeded, yahoo)
    assert len(yahoo.calls) == 6 and report.ok == 6 and report.unchanged == 0
    assert "bulk freshness probe failed" in caplog.text and "HTTP 429" in caplog.text


def test_a_surprising_probe_error_also_fails_open(seeded):
    yahoo = ProbeYahoo(script_for(*GLOBAL_SERIES), {}, error=KeyError("shape changed"))
    assert refresh(seeded, yahoo).ok == 6 and len(yahoo.calls) == 6


def test_a_fake_without_the_bulk_endpoint_is_simply_not_probed(seeded):
    yahoo = FakeYahoo(script_for(*GLOBAL_SERIES))  # no spark_latest at all
    assert refresh(seeded, yahoo).ok == 6 and len(yahoo.calls) == 6


def test_probing_can_be_switched_off_and_is_not_worth_it_for_a_handful_of_series(seeded):
    yahoo = ProbeYahoo(script_for(*GLOBAL_SERIES), same_as_stored())
    refresh(seeded, yahoo, probe=False)
    assert yahoo.spark_calls == [] and len(yahoo.calls) == 6
    few = ProbeYahoo(script_for("^GSPC", "^IXIC"), same_as_stored())
    pl.refresh_daily_bars(seeded, few, now=NOW, years=10, symbols={"^GSPC", "^IXIC"}, include_equities=False)
    assert few.spark_calls == [] and len(few.calls) == 2


def test_equities_are_never_probed(seeded):
    for symbol in "AAA BBB CCC DDD EEE".split():
        iid = seeded.upsert_instrument(symbol, "equity", yahoo_symbol=f"{symbol}.NS", in_universe=1)
        seeded.upsert_bars(iid, bars({"2026-10-30": 101.0}), source="nse", source_rank=0)
        seeded.kv_set(f"yahoo:fetched:{symbol}", iso(NOW - timedelta(hours=1)))
    script = {f"{s}.NS": [series({"2026-10-30": 101.0})] for s in "AAA BBB CCC DDD EEE".split()}
    yahoo = ProbeYahoo(script, {f"{s}.NS": SparkPoint(ET_CLOSE_OCT30, 101.0) for s in "AAA BBB CCC DDD EEE".split()})
    report = pl.refresh_daily_bars(seeded, yahoo, now=NOW, years=10, include_series=False)
    assert yahoo.spark_calls == [] and len(yahoo.calls) == 5 and report.unchanged == 0


def test_a_probed_series_whose_market_dates_the_bar_in_its_own_time_zone(seeded):
    """Tokyo's 31 Oct bar starts 31 Oct 00:00 JST = 30 Oct 15:00 UTC: it is the NEXT session, not a stale point."""
    tokyo = SERIES_BY_KEY["^N225"]
    seeded.upsert_bars(iid_of(seeded, "^N225"), bars({"2026-10-29": 100.0, "2026-10-30": 101.0}), source="yahoo", source_rank=1)
    seeded.kv_set("yahoo:fetched:^N225", iso(NOW - timedelta(hours=1)))
    same_day = SparkPoint(datetime(2026, 10, 29, 15, 0, tzinfo=UTC), 101.0)  # 30 Oct 00:00 JST
    next_day = SparkPoint(datetime(2026, 10, 30, 15, 0, tzinfo=UTC), 101.0)  # 31 Oct 00:00 JST
    keys = (*GLOBAL_SERIES[:3], "^N225")
    for point, skipped in ((same_day, True), (next_day, False)):
        spark = {k: SparkPoint(ET_CLOSE_OCT30, 101.0) for k in GLOBAL_SERIES[:3]} | {tokyo.yahoo: point}
        yahoo = ProbeYahoo(script_for(*keys), spark)
        refresh_keys = pl.refresh_daily_bars(seeded, yahoo, now=NOW, years=10, symbols=set(keys), include_equities=False)
        assert (tokyo.yahoo not in [c[0] for c in yahoo.calls]) is skipped and refresh_keys.ok == 4
        seeded.kv_set("yahoo:fetched:^N225", iso(NOW - timedelta(hours=1)))


# ------------------------------------------------------------------------------- RSS conditional GET
FEED = FeedSpec("et_markets", "https://feeds.example.com/markets.xml", "Economic Times", "india", "markets")
RSS = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title><item><title>Sensex rallies</title>'
       b'<link>https://news.example.com/a/2</link></item></channel></rss>')


@pytest.fixture()
def collector(store, settings, fake_clock):
    fake_clock.set(NOW)
    return NewsCollector(store, settings, clock=fake_clock)


def script(collector, fake_clock, items):
    session = FakeSession(items)
    collector.http = HttpClient("unit", session=session, clock=fake_clock, max_attempts=1)
    return session


def sent_headers(session, index):
    return session.calls[index][2]["headers"] or {}


def test_an_unchanged_feed_costs_one_304_and_is_not_downloaded_or_parsed_again(collector, store, fake_clock):
    session = script(collector, fake_clock, [
        FakeResponse(200, RSS, headers={"ETag": '"v1"', "Last-Modified": "Sun, 01 Nov 2026 20:00:00 GMT"}), FakeResponse(304)])
    first = collector.collect_rss([FEED])
    assert first.ok == 1 and first.not_modified == 0 and first.new_items == 1 and sent_headers(session, 0) == {}
    fake_clock.advance(1800)
    second = collector.collect_rss([FEED])
    assert second.ok == second.attempted == 1 and second.not_modified == 1 and second.new_items == 0 and second.failed == 0
    assert sent_headers(session, 1) == {"If-None-Match": '"v1"', "If-Modified-Since": "Sun, 01 Nov 2026 20:00:00 GMT"}
    assert store.kv_get("feed:et_markets:fails") == "0"
    assert store.conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0] == 1


def test_a_changed_feed_is_downloaded_and_its_validators_are_replaced(collector, store, fake_clock):
    session = script(collector, fake_clock, [
        FakeResponse(200, RSS, headers={"ETag": '"v1"'}), FakeResponse(200, RSS, headers={"ETag": '"v2"'}), FakeResponse(304)])
    collector.collect_rss([FEED])
    fake_clock.advance(600)
    assert collector.collect_rss([FEED]).not_modified == 0  # the server ignored the validator and sent the body again
    fake_clock.advance(600)
    collector.collect_rss([FEED])
    assert sent_headers(session, 2) == {"If-None-Match": '"v2"'}  # only the ETag exists: no If-Modified-Since


def test_a_publisher_that_always_says_304_is_still_fetched_in_full_every_six_hours(collector, store, fake_clock):
    session = script(collector, fake_clock, [FakeResponse(200, RSS, headers={"ETag": '"v1"'}), FakeResponse(304),
                                             FakeResponse(200, RSS, headers={"ETag": '"v1"'})])
    collector.collect_rss([FEED])
    fake_clock.advance(ns.FEED_FULL_REFRESH_HOURS * 3600 - 60)
    collector.collect_rss([FEED])
    fake_clock.advance(120)  # now older than the refresh interval
    collector.collect_rss([FEED])
    assert sent_headers(session, 1) != {} and sent_headers(session, 2) == {}


def test_feeds_without_validators_are_fetched_plainly(collector, fake_clock):
    session = script(collector, fake_clock, [FakeResponse(200, RSS), FakeResponse(200, RSS)])
    collector.collect_rss([FEED])
    fake_clock.advance(300)
    collector.collect_rss([FEED])
    assert sent_headers(session, 1) == {}


def test_a_failed_feed_is_still_counted_and_does_not_touch_the_validators(collector, store, fake_clock):
    script(collector, fake_clock, [FakeResponse(200, RSS, headers={"ETag": '"v1"'}), FakeResponse(500)])
    collector.collect_rss([FEED])
    fake_clock.advance(300)
    report = collector.collect_rss([FEED])
    assert report.failed == 1 and report.ok == 0 and store.kv_get("feed:et_markets:etag") == '"v1"'
