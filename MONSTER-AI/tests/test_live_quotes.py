"""NSE live quotes: the ``equity-stockIndices`` / ``allIndices`` parsers, the two client calls and the collection helpers."""

from __future__ import annotations

import pytest

from app.errors import DataUnavailable
from ingestion import nse
from ingestion import pipeline as pl
from ingestion.http import HttpClient
from ingestion.nse import LiveEquities
from ingestion.registry import SOURCES
from tests.helpers import FakeResponse, FakeSession, fixture_json


def row(symbol: str, **over):
    base = {"symbol": symbol, "open": 100.0, "high": 103.0, "low": 99.0, "last": 102.0, "prev_close": 100.0, "change_pct": 2.0,
            "volume": 1000.0, "turnover": 1.0e8, "year_high": None, "year_low": None, "row_ts": "2026-11-02T09:00:00Z"}
    base.update(over)
    return base


# ------------------------------------------------------------------------------------------ parsers
def test_live_equities_are_normalised_and_the_lists_own_index_row_is_kept_apart():
    snapshot = nse.parse_live_equities(fixture_json("nse_live_equities_sample.json"))
    assert [r["symbol"] for r in snapshot.rows] == ["RELIANCE", "TCS", "M&M", "INFY"]  # the index row, junk and unusable rows are out
    reliance = snapshot.rows[0]
    assert (reliance["open"], reliance["high"], reliance["low"], reliance["last"], reliance["prev_close"]) == (1251.0, 1286.5, 1249.0, 1284.2, 1248.6)
    assert reliance["volume"] == 11250000 and reliance["turnover"] == 1.4e10 and reliance["year_high"] == 1608.95
    assert reliance["change_pct"] == pytest.approx((1284.2 / 1248.6 - 1) * 100)
    assert reliance["row_ts"] == "2026-11-02T09:00:58Z"  # 14:30:58 IST
    assert snapshot.index == {"name": "SECURITIES IN F&O", "last": 24380.4, "prev_close": 24290.0,
                              "change_pct": pytest.approx((24380.4 / 24290.0 - 1) * 100), "row_ts": "2026-11-02T09:00:59Z"}
    assert snapshot.source_ts == "2026-11-02T09:01:00Z"  # the newest stamp of the answer (14:31:00 IST)


def test_numbers_written_as_text_with_thousands_separators_dashes_and_zeros_are_read_defensively():
    tcs = next(r for r in nse.parse_live_equities(fixture_json("nse_live_equities_sample.json")).rows if r["symbol"] == "TCS")
    assert tcs["last"] == 4102.5 and tcs["prev_close"] == 4090.0 and tcs["volume"] == 1250000
    assert tcs["open"] is None  # 0 is the state before the session opens, not a price
    assert tcs["high"] is None and tcs["low"] is None  # "-" and null


def test_a_stock_needs_a_last_price_and_a_previous_close_and_the_wanted_symbols_filter():
    snapshot = nse.parse_live_equities(fixture_json("nse_live_equities_sample.json"), {"INFY", "NOQUOTE", "NOPREV", "OTHER"})
    assert [r["symbol"] for r in snapshot.rows] == ["INFY"]
    assert snapshot.rows[0]["year_high"] is None and snapshot.rows[0]["row_ts"] == "2026-11-02T09:01:00Z"  # no own stamp: the answer's


def test_a_stock_symbol_with_an_ampersand_is_a_stock_but_a_name_with_a_space_is_an_index():
    data = {"name": "NIFTY 500", "timestamp": "02-Nov-2026 14:31:00", "data": [
        {"symbol": "NIFTY 500", "lastPrice": 22000, "previousClose": 21900},
        {"symbol": "NIFTY NEXT 50", "lastPrice": 70000, "previousClose": 69900},
        {"symbol": "M&M", "lastPrice": 3000, "previousClose": 2990},
        {"symbol": "BAJAJ-AUTO", "lastPrice": 9000, "previousClose": 8990},
    ]}
    snapshot = nse.parse_live_equities(data)
    assert [r["symbol"] for r in snapshot.rows] == ["M&M", "BAJAJ-AUTO"] and snapshot.index["name"] == "NIFTY 500"


@pytest.mark.parametrize("garbage", [None, [], "text", 42, {}, {"data": None}, {"data": "x"}])
def test_garbage_answers_give_an_empty_snapshot(garbage):
    snapshot = nse.parse_live_equities(garbage)
    assert snapshot.rows == [] and snapshot.index is None and snapshot.source_ts is None


def test_all_indices_are_keyed_by_upper_case_name_and_unusable_rows_are_dropped():
    indices = nse.parse_all_indices(fixture_json("nse_all_indices_sample.json"))
    assert list(indices) == ["NIFTY 50", "NIFTY BANK", "INDIA VIX"]
    nifty = indices["NIFTY 50"]
    assert (nifty["last"], nifty["prev_close"], nifty["open"], nifty["high"], nifty["low"]) == (24380.4, 24290.0, 24310.5, 24398.2, 24270.1)
    assert nifty["change_pct"] == pytest.approx((24380.4 / 24290.0 - 1) * 100) and nifty["row_ts"] == "2026-11-02T09:01:00Z"
    assert indices["INDIA VIX"]["change_pct"] == pytest.approx((13.2 / 13.6 - 1) * 100)
    assert nse.parse_all_indices([]) == {} and nse.parse_all_indices({"data": None}) == {}


# ------------------------------------------------------------------------------------------- client
def api_for(settings, fake_clock, calls):
    def handler(url, kwargs):
        calls.append((url, kwargs.get("params")))
        if url == nse.HOME:
            return FakeResponse(200, b"<html/>")
        if "equity-stockIndices" in url:
            return FakeResponse(200, fixture_json("nse_live_equities_sample.json"))
        if "allIndices" in url:
            return FakeResponse(200, fixture_json("nse_all_indices_sample.json"))
        raise AssertionError(url)

    return nse.NseApi(settings, http=HttpClient("nse", session=FakeSession(handler), clock=fake_clock))


def test_the_client_asks_for_one_list_and_one_index_call(settings, fake_clock):
    calls: list = []
    api = api_for(settings, fake_clock, calls)
    snapshot = api.live_equities("SECURITIES IN F&O", ["RELIANCE", "INFY"])
    assert [r["symbol"] for r in snapshot.rows] == ["RELIANCE", "INFY"]
    assert api.all_indices()["INDIA VIX"]["last"] == 13.2
    assert [(url.rsplit("/", 1)[-1], params) for url, params in calls if url != nse.HOME] == [
        ("equity-stockIndices", {"index": "SECURITIES IN F&O"}), ("allIndices", None)]


def test_the_lists_follow_the_universe_policy():
    assert nse.LIVE_LISTS["fno_nifty100"] == ("SECURITIES IN F&O",)
    assert all(lists[0] == "SECURITIES IN F&O" for lists in nse.LIVE_LISTS.values())
    assert set(nse.LIVE_LISTS) == {"fno_nifty100", "nifty200", "nifty500", "custom"}


# ---------------------------------------------------------------------------------- collection helpers
class FakeNse:
    def __init__(self, answers=None, *, indices=None, error=None):
        self.answers = dict(answers or {})
        self.indices = indices
        self.error = error
        self.requested: list[str] = []

    def live_equities(self, index, symbols=None, *, deadline=None):
        self.requested.append(index)
        answer = self.answers.get(index)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def all_indices(self, *, deadline=None):
        if self.error:
            raise self.error
        return self.indices or {}


def health(store, source_id="nse_live_quotes"):
    return next((h for h in store.source_health() if h["source_id"] == source_id), None)


def test_the_first_list_wins_for_a_symbol_and_the_newest_stamp_and_the_index_row_are_kept(store):
    from ingestion.registry import register_all

    register_all(store)
    api = FakeNse({
        "SECURITIES IN F&O": LiveEquities("2026-11-02T09:00:00Z", [row("AAA", last=111.0), row("BBB")], {"name": "SECURITIES IN F&O", "last": 1.0}),
        "NIFTY 200": LiveEquities("2026-11-02T09:01:00Z", [row("AAA", last=999.0), row("CCC")], {"name": "NIFTY 200", "last": 2.0}),
    })
    merged = pl.collect_live_equities(store, api, ("SECURITIES IN F&O", "NIFTY 200"), ["AAA", "BBB", "CCC"])
    assert [r["symbol"] for r in merged.rows] == ["AAA", "BBB", "CCC"] and merged.rows[0]["last"] == 111.0
    assert merged.source_ts == "2026-11-02T09:01:00Z" and merged.index["name"] == "SECURITIES IN F&O"
    assert api.requested == ["SECURITIES IN F&O", "NIFTY 200"]
    assert health(store)["total_success"] == 1 and health(store)["consecutive_failures"] == 0


def test_a_list_that_fails_is_skipped_and_the_one_that_answers_still_counts(store):
    from ingestion.registry import register_all

    register_all(store)
    api = FakeNse({"SECURITIES IN F&O": DataUnavailable("nse_api", "HTTP 403"),
                   "NIFTY 200": LiveEquities("2026-11-02T09:01:00Z", [row("CCC")])})
    merged = pl.collect_live_equities(store, api, ("SECURITIES IN F&O", "NIFTY 200"), ["CCC"])
    assert [r["symbol"] for r in merged.rows] == ["CCC"] and health(store)["total_failure"] == 1 and health(store)["total_success"] == 1


def test_when_no_list_answers_the_result_is_none_and_the_failure_is_on_record(store):
    from ingestion.registry import register_all

    register_all(store)
    api = FakeNse({"SECURITIES IN F&O": DataUnavailable("nse_api", "HTTP 403: https://x?token=SECRET")})
    assert pl.collect_live_equities(store, api, ("SECURITIES IN F&O",), ["AAA"]) is None
    state = health(store)
    assert state["consecutive_failures"] == 1 and "SECRET" not in (state["last_error"] or "x")


def test_an_answer_without_a_single_wanted_stock_is_recorded_as_a_failure_but_still_returned(store):
    from ingestion.registry import register_all

    register_all(store)
    api = FakeNse({"SECURITIES IN F&O": LiveEquities("2026-11-02T09:01:00Z", [])})
    merged = pl.collect_live_equities(store, api, ("SECURITIES IN F&O",), ["AAA"])
    assert merged is not None and merged.rows == [] and health(store)["consecutive_failures"] == 1


def test_live_indices_come_back_or_an_empty_dict(store):
    from ingestion.registry import register_all

    register_all(store)
    indices = {"NIFTY 50": {"last": 24380.4}}
    assert pl.collect_live_indices(store, FakeNse(indices=indices)) == indices
    assert pl.collect_live_indices(store, FakeNse(error=DataUnavailable("nse_api", "HTTP 401"))) == {}
    assert health(store)["total_failure"] == 1


def test_the_source_is_in_the_catalogue_as_an_optional_one():
    entry = next(s for s in SOURCES if s["source_id"] == "nse_live_quotes")
    assert entry["criticality"] == "optional" and "equity-stockIndices" in entry["endpoint"] and "allIndices" in entry["endpoint"]
