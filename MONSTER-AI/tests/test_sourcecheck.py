"""``sources-check``: every endpoint family is read through the production clients; what does not answer is named, never hidden.

No network: each client is replaced by a fake that answers what ``World`` says (a value, or an exception to raise)."""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from types import SimpleNamespace

import pytest

from app import sourcecheck
from app.doctor import render
from app.errors import DataUnavailable
from app.logging_setup import register_secret
from app.sourcecheck import SECTIONS, run_checks

TODAY = date(2026, 11, 2)  # a Monday: the weekdays looked at are 2 Nov, 30 Oct, 29 Oct and 28 Oct
FEED = (b'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
        b"<item><title>a</title><link>https://example.com/a</link></item></channel></rss>")
SECRET = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"


class World:
    """What the fake providers answer. A value that is an exception is raised instead of returned."""

    def __init__(self) -> None:
        self.answers: dict[str, object] = {}
        self.feeds: dict[str, object] = {}  # url -> bytes or an exception
        self.calls: list[tuple[str, object]] = []
        self.closed: list[str] = []

    def answer(self, key: str, default: object) -> object:
        value = self.answers.get(key, default)
        if isinstance(value, Exception):
            raise value
        return value

    def closer(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(close=lambda: self.closed.append(name))


@pytest.fixture()
def world(monkeypatch, settings, tmp_path):
    import ingestion.news_sources as news
    import ingestion.nse as nse
    import ingestion.yahoo as yahoo

    w = World()
    feeds = tmp_path / "feeds.json"
    feeds.write_text(json.dumps({"feeds": [{"id": f"f{n}", "url": f"https://feeds.example/{n}.xml"} for n in range(4)]}), encoding="utf-8")
    w.cfg = dataclasses.replace(settings, news_feeds_file=feeds, gdelt_enabled=True, gdelt_fallback=True, nse_enabled=True)
    for n in range(4):
        w.feeds[f"https://feeds.example/{n}.xml"] = FEED

    class Nse:
        def __init__(self, _settings):
            self.http = w.closer("nse")

        def market_status(self):
            return w.answer("market_status", {"giftnifty": {}, "marketState": []})

        def gift_nifty(self):
            return w.answer("gift_nifty", {"price": 23424.0})

        def preopen(self, key="ALL", symbols=None, **_):
            return w.answer("preopen", SimpleNamespace(rows=[1, 2, 3], source_ts="t"))

        def all_indices(self, **_):
            return w.answer("all_indices", {"NIFTY 50": {}, "INDIA VIX": {}})

        def live_equities(self, name, symbols=None, **_):
            w.calls.append(("live", name))
            return w.answer(f"live:{name}", SimpleNamespace(rows=[1] * 200, source_ts="2026-11-02T10:10:12Z"))

        def announcements(self, start, end, symbols=None):
            w.calls.append(("announcements", (start, end)))
            return w.answer("announcements", [1, 2])

        def event_calendar(self, start, end, symbols=None):
            w.calls.append(("event_calendar", (start, end)))
            return w.answer("event_calendar", [1])

        def holidays(self):
            return w.answer("holidays", [1] * 20)

    class Archive:
        def __init__(self, _settings):
            self.http = w.closer("archive")

        def bhavcopy(self, day):
            w.calls.append(("bhavcopy", day))
            return w.answer(f"bhavcopy:{day}", [1] * 3000)

        def fo_bhavcopy(self, day):
            w.calls.append(("fo_bhavcopy", day))
            return w.answer(f"fo_bhavcopy:{day}", [1] * 30000)

        def index_constituents(self, key):
            return w.answer(f"index:{key}", [1] * 50)

        def fo_lot_sizes(self):
            return w.answer("lots", {"RELIANCE": 500})

    class Yahoo:
        def __init__(self, _settings):
            self.http = w.closer("yahoo")

        def spark_latest(self, keys):
            return w.answer("spark", {key: object() for key in keys})

        def fetch_daily(self, symbol, range_="1mo"):
            return w.answer(f"chart:{symbol}", SimpleNamespace(bars=[{"trade_date": "2026-11-02"}]))

        def fetch_quote(self, symbol):
            return w.answer(f"quote:{symbol}", {"price": 10.0})

    class Http:
        def get(self, url, params=None, **_):
            if url == news.GNEWS_URL:  # the Google News search that stands in for GDELT
                return SimpleNamespace(content=w.answer("gnews", FEED))
            value = w.feeds[url]
            if isinstance(value, Exception):
                raise value
            return SimpleNamespace(content=value)

        def close(self):
            w.closed.append("rss")

    class Collector:
        def __init__(self, store, _settings):
            self.http = Http()

    class Gdelt:
        def get(self, url, params=None, **_):
            value = w.answer("gdelt", {"articles": [{}, {}]})
            return SimpleNamespace(json=lambda: value)

        def close(self):
            w.closed.append("gdelt")

    monkeypatch.setattr(nse, "NseApi", Nse)
    monkeypatch.setattr(nse, "NseArchive", Archive)
    monkeypatch.setattr(yahoo, "YahooClient", Yahoo)
    monkeypatch.setattr(news, "NewsCollector", Collector)
    monkeypatch.setattr(news, "gdelt_client", lambda _settings: Gdelt())
    return w


def rows(report):
    return {check.name: (check.status, check.detail) for check in report.checks}


# ------------------------------------------------------------------------------------------- healthy
def test_every_endpoint_that_answers_is_ok_and_every_client_is_closed(world):
    report = run_checks(world.cfg, today=TODAY)
    assert not report.failed and {check.status for check in report.checks} == {"ok"}
    got = rows(report)
    for name in ("NSE market status", "NSE GIFT Nifty", "NSE pre-open book", "NSE all indices", "NSE live quotes (SECURITIES IN F&O)",
                 "NSE announcements", "NSE board meetings", "NSE trading holidays", "NSE bhavcopy (cash market)",
                 "NSE bhavcopy (F&O, UDiFF)", "NSE index list nifty500", "NSE F&O lot sizes", "Yahoo spark (bulk probe)",
                 "Yahoo chart ^NSEI", "Yahoo chart RELIANCE.NS", "Yahoo live-context quotes", "RSS feeds", "GDELT DOC API (optional)",
                 "Google News search (GDELT fallback)"):
        assert name in got and got[name][0] == "ok", name
    assert got["RSS feeds"][1] == "4 of 4 answer" and got["NSE live quotes (SECURITIES IN F&O)"][1].startswith("200 stocks, answer stamped")
    assert got["Google News search (GDELT fallback)"][1] == "1 items"
    assert sorted(world.closed) == ["archive", "gdelt", "nse", "rss", "rss", "yahoo"]  # the feeds and the Google search each close theirs


def test_the_windows_and_lists_are_the_ones_the_scans_use(world):
    run_checks(dataclasses.replace(world.cfg, universe_mode="nifty200"), today=TODAY)
    assert [name for kind, name in world.calls if kind == "live"] == ["SECURITIES IN F&O", "NIFTY 200"]
    assert ("announcements", (date(2026, 10, 30), TODAY)) in world.calls and ("event_calendar", (TODAY, date(2026, 11, 16))) in world.calls
    assert sorted({day for kind, day in world.calls if kind == "bhavcopy"}, reverse=True)[:1] == [TODAY]


# ---------------------------------------------------------------------------------------------- NSE
def test_a_retired_endpoint_is_named_with_its_reason_and_the_others_are_still_read(world):
    world.answers["live:SECURITIES IN F&O"] = DataUnavailable("nse_api", "no live-quote endpoint knows SECURITIES IN F&O: did NSE change its API?")
    report = run_checks(world.cfg, ["nse"], today=TODAY)
    got = rows(report)
    assert report.failed and got["NSE live quotes (SECURITIES IN F&O)"][0] == "fail"
    assert "did NSE change its API" in got["NSE live quotes (SECURITIES IN F&O)"][1]
    assert {status for name, (status, _) in got.items() if name != "NSE live quotes (SECURITIES IN F&O)"} == {"ok"}


def test_an_answer_without_data_is_judged_not_trusted(world):
    world.answers.update({"preopen": SimpleNamespace(rows=[], source_ts=None), "all_indices": {"NIFTY 50": {}}, "gift_nifty": None,
                          "announcements": [], "event_calendar": [], "holidays": [], "market_status": {}})
    got = rows(run_checks(world.cfg, ["nse"], today=TODAY))
    assert got["NSE pre-open book"] == ("fail", "answered with no stocks")
    assert got["NSE all indices"][0] == "fail" and "INDIA VIX is missing" in got["NSE all indices"][1]
    assert got["NSE GIFT Nifty"][0] == "warn" and got["NSE announcements"][0] == "warn" and got["NSE board meetings"][0] == "warn"
    assert got["NSE trading holidays"][0] == "fail" and got["NSE market status"][0] == "fail"


def test_an_answer_of_the_wrong_shape_is_a_failure_not_a_crash(world):
    world.answers["all_indices"] = None
    got = rows(run_checks(world.cfg, ["nse"], today=TODAY))
    assert got["NSE all indices"] == ("fail", "unexpected answer (TypeError)")


def test_nse_can_be_switched_off(world):
    got = rows(run_checks(dataclasses.replace(world.cfg, nse_enabled=False), ["nse"], today=TODAY))
    assert got == {"NSE API": ("ok", "off (NSE_ENABLED=false)")}


# ------------------------------------------------------------------------------------------- archive
def test_a_file_that_is_not_published_yet_passes_when_an_earlier_day_is_there(world):
    world.answers[f"bhavcopy:{TODAY}"] = None
    world.answers["fo_bhavcopy:2026-11-02"] = None
    world.answers["fo_bhavcopy:2026-10-30"] = None
    got = rows(run_checks(world.cfg, ["archive"], today=TODAY))
    assert got["NSE bhavcopy (cash market)"] == ("ok", "3000 rows for 2026-10-30 (the file of 2026-11-02 is not published yet)")
    assert got["NSE bhavcopy (F&O, UDiFF)"][0] == "ok" and "2026-10-29" in got["NSE bhavcopy (F&O, UDiFF)"][1]


def test_no_file_for_the_last_four_weekdays_fails_and_a_broken_one_says_why(world):
    for day in ("2026-11-02", "2026-10-30", "2026-10-29", "2026-10-28"):
        world.answers[f"bhavcopy:{day}"] = None
        world.answers[f"fo_bhavcopy:{day}"] = DataUnavailable("nse_archive", "HTTP 503")
    world.answers["index:nifty100"] = DataUnavailable("nse_lists", "ind_nifty100list.csv unavailable (HTTP 404)")
    got = rows(run_checks(world.cfg, ["archive"], today=TODAY))
    assert got["NSE bhavcopy (cash market)"] == ("fail", "no file for the last 4 weekdays")
    assert got["NSE bhavcopy (F&O, UDiFF)"][0] == "fail" and "HTTP 503" in got["NSE bhavcopy (F&O, UDiFF)"][1]
    assert got["NSE index list nifty100"][0] == "fail" and "HTTP 404" in got["NSE index list nifty100"][1]
    assert got["NSE index list nifty50"][0] == "ok"


# --------------------------------------------------------------------------------------------- Yahoo
def test_yahoo_names_the_series_that_do_not_answer(world):
    world.answers["spark"] = {key: object() for key in ("^GSPC", "^NSEI")}
    world.answers["chart:^NSEI"] = SimpleNamespace(bars=[])
    world.answers["quote:ES=F"] = DataUnavailable("yahoo", "symbol not found: ES=F")
    world.answers["quote:NQ=F"] = {"price": None}
    got = rows(run_checks(world.cfg, ["yahoo"], today=TODAY))
    status, detail = got["Yahoo spark (bulk probe)"]
    assert status == "warn" and detail.startswith("2 of 33 series answered; missing ")
    assert got["Yahoo chart ^NSEI"] == ("fail", "no bars") and got["Yahoo chart RELIANCE.NS"][0] == "ok"
    assert got["Yahoo live-context quotes"] == ("warn", "no quote for ES=F, NQ=F")
    world.answers["spark"] = {}
    assert rows(run_checks(world.cfg, ["yahoo"], today=TODAY))["Yahoo spark (bulk probe)"][0] == "fail"


# ---------------------------------------------------------------------------------------- RSS, GDELT
def test_a_feed_that_fails_is_a_warning_until_most_of_them_fail(world):
    world.feeds["https://feeds.example/1.xml"] = DataUnavailable("rss_news", "HTTP 404")
    world.feeds["https://feeds.example/2.xml"] = b"<html>not a feed</html>"
    report = run_checks(world.cfg, ["rss"], today=TODAY)
    got = rows(report)
    assert got["RSS feeds"] == ("warn", "2 of 4 answer") and got["RSS f1"] == ("warn", "[rss_news] HTTP 404")
    assert got["RSS f2"] == ("warn", "the feed has no items") and not report.failed
    world.feeds["https://feeds.example/3.xml"] = DataUnavailable("rss_news", "HTTP 500")
    report = run_checks(world.cfg, ["rss"], today=TODAY)
    assert rows(report)["RSS feeds"] == ("fail", "1 of 4 answer") and report.failed


def test_a_registry_without_feeds_or_that_cannot_be_read_is_reported(world, tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text('{"feeds": []}', encoding="utf-8")
    assert rows(run_checks(dataclasses.replace(world.cfg, news_feeds_file=empty), ["rss"], today=TODAY)) == {
        "RSS feeds": ("warn", "the registry lists no feeds")}
    gone = rows(run_checks(dataclasses.replace(world.cfg, news_feeds_file=tmp_path / "missing.json"), ["rss"], today=TODAY))
    assert gone["RSS feed registry"][0] == "fail" and "unreadable" in gone["RSS feed registry"][1]


def test_gdelt_is_optional_a_failure_is_only_a_warning_and_it_can_be_off(world):
    world.answers["gdelt"] = DataUnavailable("gdelt", "HTTP 429: one request every 5 seconds")
    report = run_checks(world.cfg, ["gdelt"], today=TODAY)
    assert rows(report) == {"GDELT DOC API (optional)": ("warn", "[gdelt] HTTP 429: one request every 5 seconds"),
                            "Google News search (GDELT fallback)": ("ok", "1 items")} and not report.failed
    world.answers["gdelt"] = {}
    assert rows(run_checks(world.cfg, ["gdelt"], today=TODAY))["GDELT DOC API (optional)"] == ("warn", "answered with no articles")
    assert rows(run_checks(dataclasses.replace(world.cfg, gdelt_enabled=False), ["gdelt"], today=TODAY)) == {
        "GDELT DOC API": ("ok", "off (GDELT_ENABLED=false)"), "Google News search (GDELT fallback)": ("ok", "1 items")}


def test_the_fallback_is_probed_too_and_a_discovery_without_either_source_is_shown_as_such(world):
    world.answers["gdelt"] = DataUnavailable("gdelt", "Timeout after 2 attempts")
    world.answers["gnews"] = DataUnavailable("gnews", "HTTP 503")
    report = run_checks(world.cfg, ["gdelt"], today=TODAY)
    assert rows(report) == {"GDELT DOC API (optional)": ("warn", "[gdelt] Timeout after 2 attempts"),
                            "Google News search (GDELT fallback)": ("warn", "[gnews] HTTP 503")}
    assert not report.failed  # both are optional: the feeds carry on
    for page, why in ((b"<html><body>consent</body></html>", "the search returned no items"), (b"", "the search returned no items")):
        world.answers["gnews"] = page
        assert rows(run_checks(world.cfg, ["gdelt"], today=TODAY))["Google News search (GDELT fallback)"] == (
            "warn", f"[gnews] {why}"), page


def test_the_fallback_can_be_switched_off_and_is_then_not_asked(world):
    world.answers["gnews"] = DataUnavailable("gnews", "must not be asked")
    got = rows(run_checks(dataclasses.replace(world.cfg, gdelt_fallback=False), ["gdelt"], today=TODAY))
    assert got == {"GDELT DOC API (optional)": ("ok", "2 articles"), "Google News search (GDELT fallback)": ("ok", "off (GDELT_FALLBACK=false)")}
    assert "rss" not in world.closed  # no Google client was even opened


# ----------------------------------------------------------------------------------------------- Kotak
class FakeBroker:
    """Stands in for ``KotakApi``: ``script`` maps a call name to a value (or an exception to raise); closes are recorded on the class."""

    script: dict[str, object] = {}
    closed = 0

    def __init__(self, settings):
        FakeBroker.closed = 0

    def _get(self, name, default):
        value = FakeBroker.script.get(name, default)
        if isinstance(value, Exception):
            raise value
        return value

    def token_map(self, **_):
        return self._get("master", {f"S{i}": str(i) for i in range(600)} | {"RELIANCE": "2885", "HDFCBANK": "1333", "TCS": "11536",
                                                                             "INFY": "1594", "SBIN": "3045"})

    def indices(self, **_):
        return self._get("indices", {"NIFTY 50": {"last": 25030.35}, "INDIA VIX": {"last": 17.47}})

    def quotes(self, tokens, **_):
        from datetime import datetime, timezone

        from ingestion.kotak import KotakQuotes

        got = self._get("quotes", None)
        if got is not None:
            return got
        stamp = datetime(2026, 11, 2, 4, 0, tzinfo=timezone.utc)
        return KotakQuotes([{"symbol": s} for s in tokens], stamp, len(tokens), stamp)

    def candles(self, token, interval, start, end, **_):
        return self._get("candles", [{"trade_date": "2026-10-30", "hhmm": "15:10"}] * 72)

    def close(self):
        FakeBroker.closed += 1


@pytest.fixture()
def broker(monkeypatch, world):
    import ingestion.kotak as kotak

    FakeBroker.script = {}
    monkeypatch.setattr(kotak, "KotakApi", FakeBroker)
    return dataclasses.replace(world.cfg, kotak_consumer_key="kotak-test-token-0123456789abcdef0123")


def test_the_broker_section_is_ok_and_says_so_when_there_is_no_key(world):
    assert rows(run_checks(world.cfg, ["kotak"], today=TODAY)) == {
        "Kotak broker feed": ("ok", "off (KOTAK_CONSUMER_KEY is not set, or KOTAK_ENABLED=false)")}


def test_a_broker_that_answers_is_ok_through_every_step_and_its_client_is_closed(broker):
    report = run_checks(broker, ["kotak"], today=TODAY)
    assert rows(report) == {
        "Kotak instrument master": ("ok", "605 stocks"), "Kotak index quotes": ("ok", "NIFTY 50 25030.3, INDIA VIX 17.47"),
        "Kotak live quotes": ("ok", "5 stocks, newest trade 2026-11-02 04:00:00 UTC"),
        "Kotak 5-minute candles": ("ok", "72 bars of RELIANCE, the last on 2026-10-30 starting 15:10 IST")} and not report.failed
    assert FakeBroker.closed == 1


def test_the_broker_is_optional_so_every_failure_is_a_warning_and_never_a_failed_check(broker):
    FakeBroker.script = {"master": DataUnavailable("kotak_quotes", "HTTP 503")}
    report = run_checks(broker, ["kotak"], today=TODAY)
    assert rows(report)["Kotak instrument master"] == ("warn", "[kotak_quotes] HTTP 503") and not report.failed
    assert "Kotak live quotes" not in rows(report)  # nothing to quote without the master
    FakeBroker.script = {"master": {"S1": "1"}, "indices": {}, "candles": []}
    got = rows(run_checks(broker, ["kotak"], today=TODAY))
    assert got["Kotak instrument master"] == ("warn", "only 1 stocks") and got["Kotak index quotes"] == ("warn", "no index quote")


def test_partial_quotes_and_missing_candles_are_warnings_with_what_was_found(broker):
    from datetime import datetime, timezone

    from ingestion.kotak import KotakQuotes

    stamp = datetime(2026, 11, 2, 4, 0, tzinfo=timezone.utc)
    FakeBroker.script = {"quotes": KotakQuotes([{"symbol": "RELIANCE"}], stamp, 5, stamp), "candles": []}
    report = run_checks(broker, ["kotak"], today=TODAY)
    got = rows(report)
    assert got["Kotak live quotes"] == ("warn", "1 of 5 stocks answered") and got["Kotak 5-minute candles"] == ("warn", "no five-minute bars")
    assert not report.failed


# ---------------------------------------------------------------------------------- selection and output
def test_only_the_chosen_sections_run_in_the_fixed_order(world):
    got = list(rows(run_checks(world.cfg, ["rss", "nse"], today=TODAY)))
    assert got[0] == "NSE market status" and got[-1] == "RSS feeds" and not any(name.startswith(("Yahoo", "GDELT")) for name in got)
    assert SECTIONS == ("nse", "archive", "yahoo", "kotak", "rss", "gdelt")


def test_a_secret_in_an_error_never_reaches_the_report(world):
    register_secret(SECRET)
    world.answers["holidays"] = RuntimeError(f"HTTPSConnectionPool(host='x'): Max retries exceeded with url: /bot{SECRET}/getMe")
    text = render(run_checks(world.cfg, ["nse"], today=TODAY))
    assert "NSE trading holidays" in text and SECRET not in text


def test_the_report_reads_like_the_doctor_with_its_own_verdict(world):
    world.answers["holidays"] = DataUnavailable("nse_api", "HTTP 404")
    text = sourcecheck.render(run_checks(world.cfg, ["nse"], today=TODAY))
    assert "[FAIL] NSE trading holidays" in text and "[ ok ] NSE market status" in text
    assert text.rstrip().endswith("1 endpoint(s) FAIL: see above")
    assert sourcecheck.render(run_checks(world.cfg, ["gdelt"], today=TODAY)).rstrip().endswith(
        "every endpoint answers (warnings, if any, are optional or temporary)")
