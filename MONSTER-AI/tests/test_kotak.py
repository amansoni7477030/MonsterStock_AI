"""The Kotak market-data client: parsers (instrument master, quotes, indices, candles), the request rails (header, host, batches, bisecting a
refused batch, partial failures, window guards) and what must never leak. The fixtures keep the shape of the broker's answers as read from
the live API on 2026-11-06 (RELIANCE's row and candles, the instrument master's columns), with the figures kept as they were."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.errors import DataUnavailable
from ingestion import kotak as kt
from ingestion.http import HttpClient
from ingestion.kotak import KotakApi
from tests.helpers import FakeResponse, FakeSession

UTC = timezone.utc
KEY = "kotak-test-token-0123456789abcdef0123"
EPOCH = 1793960999  # 2026-11-06 10:29:59 UTC = 15:59:59 IST, the post-close session
STAMP = "2026-11-06T10:29:59Z"
MASTER = (
    "pSymbol,pGroup,pExchSeg,pInstType,pSymbolName,pTrdSymbol,pOptionType,lLotSize,dStrikePrice;\n"
    "2885,EQ,nse_cm,,RELIANCE,RELIANCE-EQ,,1,-1\n"
    "22,BE,nse_cm,,RELIANCE,RELIANCE-BE,,1,-1\n"
    "2031,EQ,nse_cm,,M&M,M&M-EQ,,1,-1\n"
    "16669,EQ,nse_cm,,BAJAJ-AUTO,BAJAJ-AUTO-EQ,,1,-1\n"
    "11536,  EQ , nse_cm ,,TCS, TCS-EQ ,,1,-1\n"
    "99999,EQ,bse_cm,,ONLYBSE,ONLYBSE-EQ,,1,-1\n"
    "26000,,nse_cm,,NIFTY,NIFTY,,1,-1\n"
    "abc,EQ,nse_cm,,BROKEN,BROKEN-EQ,,1,-1\n"
)
RELIANCE_ROW = {
    "exchange_token": "2885", "display_symbol": "RELIANCE-EQ", "exchange": "nse_cm", "lstup_time": str(EPOCH), "ltp": "1320.0000",
    "last_traded_quantity": "3", "total_buy": "0", "total_sell": "20629", "last_volume": "22768301", "avg_cost": "1311.7000",
    "open_int": "0", "change": "6.6000", "per_change": "0.5000", "year_high": "1611.8", "year_low": "1152.2",
    "ohlc": {"open": "1295.0000", "high": "1320.0000", "low": "1295.0000", "close": "1313.4000"},
}
CANDLES = {"status": "success", "message": "Data fetched successfully", "data": {"candles": [
    ["2026-11-06T09:15:00+0530", 1295, 1306.4, 1295, 1301.1, 1158306],
    ["2026-11-06T09:20:00+0530", 1301.1, 1304, 1299.5, 1303.2, 410221],
    ["2026-11-06T15:10:00+0530", 1316.9, 1317.1, 1315.2, 1316, 765449],
]}}


def quote_item(token: str, last: float = 100.0, prev: float = 99.0, **over):
    row = {"exchange_token": token, "ltp": str(last), "last_volume": "5000", "avg_cost": "99.5", "lstup_time": str(EPOCH),
           "ohlc": {"open": "99.5", "high": "101", "low": "98.5", "close": str(prev)}}
    row.update(over)
    return row


# ------------------------------------------------------------------------------------------ parsers
def test_the_instrument_master_maps_cash_equities_only_and_survives_its_odd_header_and_padding():
    tokens = kt.parse_token_map(MASTER)
    # the stray ';' on the last header cell and the padded cells are cleaned; BE series, another exchange, an index and a bad token are out
    assert tokens == {"RELIANCE": "2885", "M&M": "2031", "BAJAJ-AUTO": "16669", "TCS": "11536"}


def test_the_first_row_of_a_symbol_wins_and_the_trading_symbol_names_it_too():
    text = "pSymbol,pGroup,pExchSeg,pSymbolName,pTrdSymbol\n1,EQ,nse_cm,,ABC-EQ\n2,EQ,nse_cm,ABC,ABC-EQ\n"
    assert kt.parse_token_map(text) == {"ABC": "1"}  # the empty name falls back to the trading symbol; the later row never replaces it


def test_a_quote_row_becomes_an_nse_shaped_row_with_the_stocks_own_update_time():
    rows = kt.parse_quotes([RELIANCE_ROW], {"2885": "RELIANCE"})
    assert rows == [{
        "symbol": "RELIANCE", "open": 1295.0, "high": 1320.0, "low": 1295.0, "last": 1320.0, "prev_close": 1313.4,  # ohlc.close IS the previous close
        "change_pct": pytest.approx((1320.0 / 1313.4 - 1) * 100), "volume": 22768301.0, "turnover": pytest.approx(22768301.0 * 1311.7),
        "vwap": 1311.7, "year_high": 1611.8, "year_low": 1152.2, "row_ts": STAMP,
    }]


def test_a_row_without_a_last_price_or_a_previous_close_or_a_known_token_is_skipped():
    items = [
        quote_item("1"),
        quote_item("2", ltp="0"), quote_item("3", ltp="-"), quote_item("4", ohlc={"open": "1", "high": "2", "low": "1", "close": "0"}),
        quote_item("5", ohlc="x"), quote_item("999"), "junk", None, {"exchange_token": None},
    ]
    assert [r["symbol"] for r in kt.parse_quotes(items, {str(i): f"S{i}" for i in range(1, 6)})] == ["S1"]


def test_figures_the_broker_leaves_out_or_zeroes_before_the_open_stay_none_and_a_bad_stamp_is_no_stamp():
    item = quote_item("1", lstup_time="0", last_volume="n/a", avg_cost="0", ohlc={"open": "0.0000", "high": "0", "low": "-", "close": "99"})
    (row,) = kt.parse_quotes([item], {"1": "AAA"})
    assert (row["open"], row["high"], row["low"]) == (None, None, None)
    assert row["volume"] is None and row["turnover"] is None and row["vwap"] is None and row["row_ts"] is None
    ms = quote_item("1", lstup_time=str(EPOCH * 1000))  # milliseconds are recognised
    assert kt.parse_quotes([ms], {"1": "AAA"})[0]["row_ts"] == STAMP
    assert kt.parse_quotes([quote_item("1", lstup_time="12345")], {"1": "AAA"})[0]["row_ts"] is None  # 1970 is not a trade time


@pytest.mark.parametrize("garbage", [None, {}, "text", 42, {"fault": {}}, [[]]])
def test_garbage_quote_answers_give_no_rows(garbage):
    assert kt.parse_quotes(garbage, {"1": "AAA"}) == []


def test_index_quotes_are_keyed_by_the_names_the_scans_use():
    nifty = {"exchange_token": "Nifty 50", "ltp": "25030.3500", "lstup_time": str(EPOCH),
             "ohlc": {"open": "24538.2", "high": "25030.35", "low": "24388.05", "close": "24536.6"}}
    vix = {"exchange_token": "INDIA VIX", "ltp": "17.47", "lstup_time": str(EPOCH), "ohlc": {"open": "15.58", "high": "17.77", "low": "14.46", "close": "15.58"}}
    found = kt.parse_indices([nifty, vix, {"exchange_token": "Nifty Bank", "ltp": "1"}, {"exchange_token": "INDIA VIX", "ltp": "0"}])
    assert set(found) == {"NIFTY 50", "INDIA VIX"}
    assert found["NIFTY 50"] == {"name": "NIFTY 50", "last": 25030.35, "prev_close": 24536.6, "open": 24538.2, "high": 25030.35, "low": 24388.05,
                                 "change_pct": pytest.approx((25030.35 / 24536.6 - 1) * 100), "row_ts": STAMP}


def test_candles_are_bars_in_the_shape_yahoos_have_dated_and_timed_in_ist():
    bars = kt.parse_candles(CANDLES)
    assert [(b["trade_date"], b["hhmm"]) for b in bars] == [("2026-11-06", "09:15"), ("2026-11-06", "09:20"), ("2026-11-06", "15:10")]
    assert bars[0] == {"ts": "2026-11-06T03:45:00Z", "trade_date": "2026-11-06", "hhmm": "09:15", "open": 1295.0, "high": 1306.4, "low": 1295.0,
                       "close": 1301.1, "volume": 1158306.0}  # the first bar carries the opening auction's volume


def test_rows_that_are_not_a_bar_of_the_regular_session_are_dropped():
    good = ["2026-11-06T10:00:00+0530", 100, 102, 99, 101, 10]
    rows = [
        good,
        ["2026-11-06T09:10:00+0530", 100, 102, 99, 101, 10],  # before the open
        ["2026-11-06T15:30:00+0530", 100, 102, 99, 101, 10],  # after the close
        ["2026-11-06T10:05:00+0530", 100, 99, 99, 101, 10],  # the high is below the close
        ["2026-11-06T10:10:00+0530", 100, 102, 100.5, 101, 10],  # the low is above the open
        ["2026-11-06T10:15:00+0530", 100, 102, 99, 101, -5],  # a negative volume is an artefact of the vendor's archive
        ["2026-11-06T10:20:00+0530", 0, 102, 99, 101, 10],  # no price
        ["not a time", 100, 102, 99, 101, 10], ["2026-11-06T10:25:00+0530", 100, 102], "junk", None,
    ]
    assert [b["hhmm"] for b in kt.parse_candles({"data": {"candles": rows}})] == ["10:00"]


def test_a_timestamp_that_appears_twice_keeps_its_last_row_and_bars_come_out_sorted():
    rows = [["2026-11-06T10:05:00+0530", 100, 102, 99, 101, 10], ["2026-11-06T10:00:00+0530", 100, 102, 99, 101, 10],
            ["2026-11-06T10:05:00+0530", 100, 103, 99, 102, 77]]
    bars = kt.parse_candles({"data": {"candles": rows}})
    assert [(b["hhmm"], b["volume"], b["high"]) for b in bars] == [("10:00", 10.0, 102.0), ("10:05", 77.0, 103.0)]


def test_a_candle_without_a_volume_is_kept_with_none_and_naive_or_epoch_stamps_are_read():
    rows = [["2026-11-06 10:00:00", 100, 102, 99, 101], [1793939400, 100, 102, 99, 101, 5]]  # IST wall clock; 10:00 IST as an epoch
    bars = kt.parse_candles({"data": {"candles": rows}})
    assert [b["hhmm"] for b in bars] == ["10:00"] and bars[0]["volume"] == 5.0  # both are the same minute: the last row wins


@pytest.mark.parametrize("garbage", [None, {}, "x", 7, {"data": None}, {"data": {}}, {"data": {"candles": "x"}}, {"data": {"candles": None}}, []])
def test_garbage_candle_answers_give_no_bars(garbage):
    assert kt.parse_candles(garbage) == []


# ------------------------------------------------------------------------------------------ the client
def make_api(settings, fake_clock, handler, history=None, **over):
    cfg = settings.__class__(**{**settings.__dict__, "kotak_consumer_key": KEY, **over})
    quotes_session, history_session = FakeSession(handler), FakeSession(history or handler)
    api = KotakApi(cfg, http=HttpClient("kotak_quotes", session=quotes_session, clock=fake_clock),
                   history_http=HttpClient("kotak_history", session=history_session, clock=fake_clock), clock=fake_clock)
    return api, quotes_session, history_session


def test_the_client_needs_a_key_and_refuses_a_host_the_key_may_not_go_to(settings, fake_clock):
    with pytest.raises(DataUnavailable, match="KOTAK_CONSUMER_KEY is not set"):
        KotakApi(settings, clock=fake_clock)
    evil = settings.__class__(**{**settings.__dict__, "kotak_consumer_key": KEY, "kotak_base_url": "https://mis.kotaksecurities.com.evil.example"})
    with pytest.raises(DataUnavailable, match="the key is not sent"):
        KotakApi(evil, clock=fake_clock)


def test_the_token_travels_in_the_header_and_never_in_the_url(settings, fake_clock):
    def handler(url, kwargs):
        return FakeResponse(200, [quote_item("2885")]) if "/quotes/" in url else FakeResponse(200, CANDLES)

    api, quotes, history = make_api(settings, fake_clock, handler)
    api.quotes({"RELIANCE": "2885"})
    api.candles("2885", "5min", date(2026, 11, 6), date(2026, 11, 6))
    for method, url, kwargs in [*quotes.calls, *history.calls]:
        assert kwargs["headers"]["Authorization"] == KEY and KEY not in url and KEY not in str(kwargs.get("params"))
        assert url.startswith("https://mis.kotaksecurities.com/")
    assert quotes.calls[0][1] == "https://mis.kotaksecurities.com/script-details/1.0/quotes/neosymbol/nse_cm|2885/all"
    assert history.calls[0][2]["params"] == {"neosymbol": "nse_cm|2885", "interval": "5min", "fromdate": "2026-11-06", "todate": "2026-11-06"}


def test_the_instrument_master_is_found_through_the_file_list_and_fetched_without_the_token(settings, fake_clock):
    files = {"data": {"baseFolder": "https://lapi.kotaksecurities.com/x", "filesPaths": [
        "https://lapi.kotaksecurities.com/x/transformed/nse_com.csv", "https://lapi.kotaksecurities.com/x/transformed/nse_fo.csv",
        "https://lapi.kotaksecurities.com/x/transformed-v1/nse_cm-v1.csv"]}}
    big = MASTER + "".join(f"{1000 + i},EQ,nse_cm,,S{i},S{i}-EQ,,1,-1\n" for i in range(600))

    def handler(url, kwargs):
        return FakeResponse(200, files) if url.endswith("file-paths") else FakeResponse(200, big)

    api, session, _ = make_api(settings, fake_clock, handler)
    tokens = api.token_map()
    assert tokens["RELIANCE"] == "2885" and len(tokens) == 604
    listing, csv_call = session.calls
    assert listing[2]["headers"]["Authorization"] == KEY
    assert csv_call[1] == "https://lapi.kotaksecurities.com/x/transformed-v1/nse_cm-v1.csv"  # the nse_cm file, not nse_com
    assert "Authorization" not in csv_call[2]["headers"]  # the token goes to the API, never to the file host


def test_an_instrument_master_on_a_foreign_host_or_that_is_too_short_is_not_used(settings, fake_clock):
    foreign = {"data": {"filesPaths": ["https://evil.example/nse_cm-v1.csv"]}}
    api, session, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, foreign))
    with pytest.raises(DataUnavailable, match="outside kotaksecurities.com"):
        api.token_map()
    assert len(session.calls) == 1  # the foreign URL was never requested
    short = {"data": {"filesPaths": ["https://lapi.kotaksecurities.com/x/nse_cm-v1.csv"]}}
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, short if url.endswith("file-paths") else MASTER))
    with pytest.raises(DataUnavailable, match="not the real file"):
        api.token_map()
    nothing = {"data": {"filesPaths": ["https://lapi.kotaksecurities.com/x/nse_fo.csv"]}}
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, nothing))
    with pytest.raises(DataUnavailable, match="no nse_cm file"):
        api.token_map()


def test_quotes_go_out_forty_stocks_to_a_request_with_the_exchange_tokens(settings, fake_clock):
    tokens = {f"S{i:03d}": str(1000 + i) for i in range(95)}

    def handler(url, kwargs):
        asked = [pair.split("|")[1] for pair in url.split("/neosymbol/")[1].rsplit("/", 1)[0].split(",")]
        return FakeResponse(200, [quote_item(t) for t in asked])

    api, session, _ = make_api(settings, fake_clock, handler)
    result = api.quotes(tokens)
    sizes = [len(url.split("/neosymbol/")[1].rsplit("/", 1)[0].split(",")) for _, url, _ in session.calls]
    assert sizes == [40, 40, 15] and kt.QUOTE_BATCH == 40  # the live API refuses 50 although its documentation allows it
    assert len(result.rows) == 95 and result.asked == 95 and result.failed == []
    assert {r["symbol"] for r in result.rows} == set(tokens)


def test_the_age_of_the_feed_is_that_of_its_freshest_trade_and_a_stamp_from_the_future_is_not_believed(settings, fake_clock):
    now = int(fake_clock.now().timestamp())
    items = [quote_item("1", lstup_time=str(now - 400)), quote_item("2", lstup_time=str(now - 30)), quote_item("3", lstup_time=str(now - 900))]
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, items))
    result = api.quotes({"A": "1", "B": "2", "C": "3"})
    assert result.lag_sec == pytest.approx(30, abs=1) and result.received == fake_clock.now()
    skewed = [quote_item("1", lstup_time=str(now + 20))]  # a few seconds of clock skew is capped at the moment of the answer
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, skewed))
    capped = api.quotes({"A": "1"})
    assert capped.asof == capped.received and capped.lag_sec == 0
    future = [quote_item("1", lstup_time=str(now + 3600))]  # an hour ahead is a fault, not freshness: the lag gate must see no stamp
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, future))
    assert api.quotes({"A": "1"}).lag_sec is None
    mixed = [quote_item("1", lstup_time=str(now + 3600)), quote_item("2", lstup_time=str(now - 45))]  # the believable stamp still counts
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, mixed))
    assert api.quotes({"A": "1", "B": "2"}).lag_sec == pytest.approx(45, abs=1)
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, [quote_item("1", lstup_time="0")]))
    assert api.quotes({"A": "1"}).lag_sec is None  # no stamp: the age cannot be judged


def test_a_stock_the_api_leaves_out_of_an_answer_is_recorded_as_not_read(settings, fake_clock):
    tokens = {"A": "1", "B": "2", "C": "3"}

    def handler(url, kwargs):
        asked = [pair.split("|")[1] for pair in url.split("/neosymbol/")[1].rsplit("/", 1)[0].split(",")]
        return FakeResponse(200, [quote_item(t) for t in asked if t != "2"])  # the API drops a token it does not know without any error

    api, _, _ = make_api(settings, fake_clock, handler)
    result = api.quotes(tokens)
    assert result.failed == ["B"] and {r["symbol"] for r in result.rows} == {"A", "C"} and result.asked == 3


def test_a_refused_batch_is_halved_until_the_one_stock_the_api_cannot_quote_is_found(settings, fake_clock):
    tokens = {f"S{i}": str(100 + i) for i in range(9)}
    bad = "104"
    fault = {"fault": {"code": "400", "description": "Please pass valid neosymbol values for getQuote", "message": "Invalid neosymbol values"}}

    def handler(url, kwargs):
        asked = [pair.split("|")[1] for pair in url.split("/neosymbol/")[1].rsplit("/", 1)[0].split(",")]
        return FakeResponse(400, fault) if bad in asked else FakeResponse(200, [quote_item(t) for t in asked])

    api, session, _ = make_api(settings, fake_clock, handler)
    result = api.quotes(tokens)
    assert result.failed == ["S4"] and {r["symbol"] for r in result.rows} == set(tokens) - {"S4"}
    assert len(session.calls) < 2 * len(tokens)  # bisecting, not one request per stock


def test_a_batch_that_cannot_be_read_costs_only_its_own_stocks(settings, fake_clock):
    tokens = {f"S{i:03d}": str(1000 + i) for i in range(100)}  # batches of 40, 40, 20
    seen = []

    def handler(url, kwargs):
        asked = [pair.split("|")[1] for pair in url.split("/neosymbol/")[1].rsplit("/", 1)[0].split(",")]
        seen.append(asked[0])
        if asked[0] == "1040":  # the second batch: the server errors on every attempt
            return FakeResponse(503, b"unavailable")
        return FakeResponse(200, [quote_item(t) for t in asked])

    api, _, _ = make_api(settings, fake_clock, handler)
    result = api.quotes(tokens)
    assert len(result.rows) == 60 and len(result.failed) == 40 and result.failed[0] == "S040"


def test_when_nothing_can_be_read_the_failure_is_raised_without_the_key_in_it(settings, fake_clock):
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(503, f"down {KEY}".encode()))
    with pytest.raises(DataUnavailable) as err:
        api.quotes({"A": "1"})
    assert KEY not in str(err.value)
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, []))
    with pytest.raises(DataUnavailable):  # an empty answer for something asked is a failure, not a quiet market
        api.quotes({"A": "1"})


@pytest.mark.parametrize("code", [401, 403])
def test_a_refused_token_says_what_to_do_and_stops_at_once(settings, fake_clock, code):
    api, session, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(code, b"nope"))
    with pytest.raises(DataUnavailable, match="refused the application token") as err:
        api.quotes({f"S{i:03d}": str(i) for i in range(100)})
    assert KEY not in str(err.value) and "KOTAK_CONSUMER_KEY" in str(err.value) and not err.value.retryable
    assert len(session.calls) == 1  # the other two batches would only be refused again


def test_a_successful_answer_with_a_message_is_not_a_fault_but_an_error_status_is(settings, fake_clock):
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, CANDLES))
    assert len(api.candles("2885", "5min", date(2026, 11, 6), date(2026, 11, 6))) == 3  # "Data fetched successfully" is information
    err = {"status": "ERROR", "message": "Invalid interval value"}
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, err))
    with pytest.raises(DataUnavailable, match="Invalid interval value"):
        api.candles("2885", "5min", date(2026, 11, 6), date(2026, 11, 6))


def test_a_candle_window_the_broker_refuses_is_reported_and_the_guards_hold(settings, fake_clock):
    fault = {"status": "ERROR", "fault": {"code": 400, "message": "Invalid interval value"}}
    api, session, history = make_api(settings, fake_clock, lambda url, kw: FakeResponse(400, fault))
    with pytest.raises(DataUnavailable, match="refused the window: Invalid interval value") as err:
        api.candles("2885", "5min", date(2026, 11, 6), date(2026, 11, 6))
    assert not err.value.retryable and len(history.calls) == 1  # a refusal is not retried
    with pytest.raises(ValueError, match="unsupported candle interval"):
        api.candles("2885", "2min", date(2026, 11, 6), date(2026, 11, 6))
    with pytest.raises(ValueError, match="1 to 30 days"):
        api.candles("2885", "5min", date(2026, 10, 1), date(2026, 11, 6))
    with pytest.raises(ValueError, match="1 to 30 days"):
        api.candles("2885", "5min", date(2026, 11, 6), date(2026, 11, 5))
    assert len(history.calls) == 1  # none of the guarded calls reached the network


def test_the_index_quotes_are_asked_by_name(settings, fake_clock):
    nifty = {"exchange_token": "Nifty 50", "ltp": "25030.35", "lstup_time": str(EPOCH), "ohlc": {"open": "1", "high": "2", "low": "1", "close": "24536.6"}}
    api, session, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(200, [nifty]))
    assert set(api.indices()) == {"NIFTY 50"}
    assert session.calls[0][1].endswith("/neosymbol/nse_cm|Nifty%2050,nse_cm|INDIA%20VIX/all")
    api, _, _ = make_api(settings, fake_clock, lambda url, kw: FakeResponse(400, {"fault": {"description": "Invalid neosymbol values"}}))
    with pytest.raises(DataUnavailable, match="refused the index quote"):
        api.indices()


def test_the_response_stamp_arithmetic_uses_utc(settings, fake_clock):
    assert kt._epoch(EPOCH) == datetime(2026, 11, 6, 10, 29, 59, tzinfo=UTC)
    assert kt._bar_stamp("2026-11-06T09:15:00+0530") == datetime(2026, 11, 6, 3, 45, tzinfo=UTC)
    assert kt._bar_stamp("2026-11-06 09:15:00") == datetime(2026, 11, 6, 3, 45, tzinfo=UTC)  # naive text is IST
