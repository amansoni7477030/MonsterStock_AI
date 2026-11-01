from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timezone

import numpy as np
import pytest

from app.config import load_settings
from app.errors import DataUnavailable
from ingestion import nse
from ingestion.fno import aggregate_stock_fno, bs_price, implied_vol
from ingestion.http import HttpClient
from ingestion.series import SERIES, SERIES_BY_KEY, latest_available_date
from ingestion.yahoo import SymbolNotFound, YahooClient
from tests.helpers import FakeResponse, FakeSession, fixture_bytes, fixture_json

DAY = date(2026, 10, 30)


@pytest.fixture()
def cfg(tmp_path):
    return load_settings({"MONSTER_ENV": "test"}, env_file=tmp_path / "none.env")


def http_with(session, fake_clock):
    return HttpClient("unit", session=session, clock=fake_clock, max_attempts=2)


# ------------------------------------------------------------------------------------------------ yahoo
@pytest.mark.parametrize("fixture, tz, last_date", [
    ("yahoo_reliance_1mo.json", "Asia/Kolkata", "2026-10-30"),
    ("yahoo_gspc_1mo.json", "America/New_York", "2026-10-30"),
    ("yahoo_clf_1mo.json", "America/New_York", "2026-10-30"),
])
def test_yahoo_daily_parsing_uses_exchange_local_dates(cfg, fake_clock, fixture, tz, last_date):
    client = YahooClient(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes(fixture))]), fake_clock))
    series = client.fetch_daily("X")
    assert series.tz_name == tz and series.bars[-1]["trade_date"] == last_date
    bar = series.bars[-1]
    assert bar["high"] >= max(bar["open"], bar["close"]) and bar["low"] <= min(bar["open"], bar["close"])


def test_yahoo_fx_bars_are_london_dated_and_include_partial_bar(cfg, fake_clock):
    client = YahooClient(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes("yahoo_usdinr_1mo.json"))]), fake_clock))
    series = client.fetch_daily("USDINR=X")
    assert series.tz_name == "Europe/London"
    assert series.bars[-1]["trade_date"] == "2026-10-31"  # live partial bar: the PIT rule (not the parser) excludes it
    spec = SERIES_BY_KEY["USDINR=X"]
    assert spec.available_at(date(2026, 10, 31)) > datetime(2026, 10, 31, 17, 0, tzinfo=timezone.utc)


def test_yahoo_events_and_symbol_not_found(cfg, fake_clock):
    client = YahooClient(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes("yahoo_events_sample.json"))]), fake_clock))
    series = client.fetch_daily("INFY.NS", range_="max")
    assert series.splits[0]["ex_date"] == "1997-08-19" and series.splits[0]["numerator"] == 2.0
    assert series.dividends and series.dividends[-1]["amount"] > 0

    missing = FakeResponse(404, fixture_bytes("yahoo_notfound.json"))
    client = YahooClient(cfg, http=http_with(FakeSession([missing]), fake_clock))
    with pytest.raises(SymbolNotFound):
        client.fetch_daily("NOSUCH.NS")


def test_yahoo_falls_back_to_second_host_and_handles_garbage(cfg, fake_clock):
    session = FakeSession([FakeResponse(200, {"unexpected": True}), FakeResponse(200, fixture_bytes("yahoo_nsei_1mo.json"))])
    client = YahooClient(cfg, http=http_with(session, fake_clock))
    assert client.fetch_daily("^NSEI").bars
    assert "query1" in session.calls[0][1] and "query2" in session.calls[1][1]
    assert "%5ENSEI" in session.calls[0][1]  # symbol is URL-quoted

    session = FakeSession([FakeResponse(200, {"chart": {"result": None, "error": {"code": "Internal", "description": "x"}}})] * 2)
    with pytest.raises(DataUnavailable):
        YahooClient(cfg, http=http_with(session, fake_clock)).fetch_daily("^NSEI")


def test_yahoo_quote_comes_from_the_chart_meta(cfg, fake_clock):
    result = fixture_json("yahoo_nsei_1mo.json")["chart"]["result"][0]
    session = FakeSession([FakeResponse(200, {"chart": {"result": [result], "error": None}})])
    client = YahooClient(cfg, http=http_with(session, fake_clock))
    quote = client.fetch_quote("^NSEI")
    assert quote["price"] and quote["prev_close"] and quote["source_ts"]


# -------------------------------------------------------------------------------------------------- NSE
def test_bhavcopy_parsing_and_edge_cases(cfg, fake_clock):
    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes("bhavcopy_full_sample.csv"))]), fake_clock))
    rows = archive.bhavcopy(DAY)
    reliance = next(r for r in rows if r["symbol"] == "RELIANCE")
    assert reliance["open"] == 1170.4 and reliance["delivery_pct"] == 54.73
    assert reliance["turnover"] == pytest.approx(171191.72 * 1e5) and reliance["vwap"] == 1160.33
    assert next(r for r in rows if r["symbol"] == "M&M")["series"] == "EQ"

    # wrong date inside the file, HTML error page served with 200, and 404 all mean "not available"
    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes("bhavcopy_full_sample.csv"))]), fake_clock))
    assert archive.bhavcopy(date(2026, 10, 29)) is None
    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(200, b"<html>blocked</html>")]), fake_clock))
    assert archive.bhavcopy(DAY) is None
    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(404)]), fake_clock))
    assert archive.bhavcopy(DAY) is None


def test_fo_zip_parsing_and_safety():
    rows = nse.parse_fo_zip(fixture_bytes("fo_udiff_sample.zip"), DAY)
    assert {r["instrument"] for r in rows} >= {"STF", "STO", "IDF"}
    assert any(r["option_type"] == "CE" and r["strike"] for r in rows)
    assert nse.parse_fo_zip(fixture_bytes("fo_udiff_sample.zip"), date(2026, 10, 29)) is None  # wrong trade date
    assert nse.parse_fo_zip(b"not a zip", DAY) is None
    bomb = io.BytesIO()
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.csv", b"x" * 1000)
        z.writestr("b.csv", b"x" * 1000)
        z.writestr("c.csv", b"x" * 1000)
        z.writestr("d.csv", b"x" * 1000)
    assert nse.parse_fo_zip(bomb.getvalue(), DAY) is None  # too many members
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as z:
        z.writestr("readme.txt", "no csv here")
    assert nse.parse_fo_zip(empty.getvalue(), DAY) is None


def test_fo_bhavcopy_before_udiff_and_lists(cfg, fake_clock):
    archive = nse.NseArchive(cfg, http=http_with(FakeSession(), fake_clock))
    assert archive.fo_bhavcopy(date(2024, 1, 2)) is None  # legacy format is deliberately unsupported
    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes("fo_udiff_sample.zip"))]), fake_clock))
    assert archive.fo_bhavcopy(DAY)

    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes("nifty500_sample.csv"))]), fake_clock))
    members = archive.index_constituents("nifty500")
    assert {m["symbol"] for m in members} >= {"RELIANCE", "M&M", "BAJAJ-AUTO"}
    assert next(m for m in members if m["symbol"] == "RELIANCE")["industry"] == "Oil Gas & Consumable Fuels"
    with pytest.raises(ValueError):
        archive.index_constituents("nifty42")
    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(404)]), fake_clock))
    with pytest.raises(DataUnavailable):
        archive.index_constituents("nifty100")

    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(200, fixture_bytes("fo_mktlots_sample.csv"))]), fake_clock))
    lots = archive.fo_lot_sizes()
    assert lots["RELIANCE"] == 500 and lots["TCS"] == 225
    archive = nse.NseArchive(cfg, http=http_with(FakeSession([FakeResponse(200, b"")]), fake_clock))
    with pytest.raises(DataUnavailable):
        archive.index_constituents("nifty500")


def test_api_parsers_on_real_fixtures():
    gift = nse.parse_gift_nifty(fixture_json("nse_marketstatus.json"))
    assert gift["price"] == 23242.5 and gift["prev_close"] == pytest.approx(23086.0) and gift["source_ts"] == "2026-10-30T20:59:00Z"
    assert nse.parse_gift_nifty({}) is None and nse.parse_gift_nifty({"giftnifty": {"LASTPRICE": "-"}}) is None

    holidays = nse.parse_holidays(fixture_json("nse_holiday_master.json"))
    assert any(h["trade_date"] == "2026-10-20" and "Dussehra" in h["description"] for h in holidays)
    assert any(h["special"] for h in holidays)  # Diwali Laxmi Pujan* (Muhurat) is flagged
    assert nse.parse_holidays(None) == []

    filings = nse.parse_announcements(fixture_json("nse_announcements_sample.json"))
    assert filings and filings[0]["disseminated_at"].endswith("Z") and filings[0]["source_ref"]
    assert all(f["attachment_url"] is None or f["attachment_url"].startswith("https://") for f in filings)
    assert nse.parse_announcements(fixture_json("nse_announcements_sample.json"), symbols={"NOPE"}) == []
    assert nse.parse_announcements([{"symbol": "X", "an_dt": "garbage"}, "junk", {"symbol": ""}]) == []

    meetings = nse.parse_event_calendar(fixture_json("nse_event_calendar_sample.json"))
    assert meetings and meetings[0]["is_results"] is True and meetings[0]["meeting_date"] == "2026-11-01"
    assert nse.parse_event_calendar("nope") == []


def test_preopen_parser_handles_partial_books():
    payload = {
        "timestamp": "02-Nov-2026 09:08:12",
        "data": [
            {"metadata": {"symbol": "RELIANCE", "previousClose": 1164.9, "pChange": 0.8, "lastPrice": 1174.2, "finalQuantity": 5000},
             "detail": {"preOpenMarket": {"IEP": 1174.2, "finalPrice": 1174.25, "totalBuyQuantity": 90000,
                                           "totalSellQuantity": 60000, "lastUpdateTime": "02-Nov-2026 09:08:30"}}},
            {"metadata": {"symbol": "TCS", "previousClose": 2500, "lastPrice": 2490}, "detail": {}},
            {"metadata": {"symbol": ""}}, "junk",
        ],
    }
    snapshot = nse.parse_preopen(payload)
    assert [r["symbol"] for r in snapshot.rows] == ["RELIANCE", "TCS"]
    reliance, tcs = snapshot.rows
    assert reliance["final_price"] == 1174.25 and reliance["buy_qty"] == 90000 and reliance["iep"] == 1174.2
    assert tcs["final_price"] is None and tcs["iep"] == 2490  # IEP falls back to metadata.lastPrice
    assert snapshot.source_ts == "2026-11-02T03:38:30Z"
    assert [r["symbol"] for r in nse.parse_preopen(payload, {"TCS"}).rows] == ["TCS"]
    assert nse.parse_preopen([]).rows == [] and nse.parse_preopen({"data": []}).source_ts is None


def test_nse_api_warms_session_and_rewarms_on_403(cfg, fake_clock):
    calls = []

    def handler(url, kwargs):
        calls.append(url)
        if url == nse.HOME:
            return FakeResponse(403, b"denied")
        if url.endswith("marketStatus") and sum(u.endswith("marketStatus") for u in calls) == 1:
            return FakeResponse(403, b"denied")
        return FakeResponse(200, fixture_bytes("nse_marketstatus.json"))

    api = nse.NseApi(cfg, http=HttpClient("nse", session=FakeSession(handler), clock=fake_clock, max_attempts=3))
    assert api.gift_nifty()["price"] == 23242.5
    assert calls.count(nse.HOME) == 2  # initial handshake + automatic re-warm after the 403
    with pytest.raises(DataUnavailable):
        nse.NseApi(cfg, http=HttpClient("nse", session=FakeSession(lambda u, k: FakeResponse(200, b"[]")), clock=fake_clock)).market_status()


def test_nse_api_lists(cfg, fake_clock):
    def handler(url, kwargs):
        if url == nse.HOME:
            return FakeResponse(200, b"<html/>")
        if "holiday-master" in url:
            return FakeResponse(200, fixture_bytes("nse_holiday_master.json"))
        if "corporate-announcements" in url:
            return FakeResponse(200, fixture_bytes("nse_announcements_sample.json"))
        if "event-calendar" in url:
            return FakeResponse(200, fixture_bytes("nse_event_calendar_sample.json"))
        if "market-data-pre-open" in url:
            return FakeResponse(200, {"data": []})
        raise AssertionError(url)

    api = nse.NseApi(cfg, http=HttpClient("nse", session=FakeSession(handler), clock=fake_clock))
    assert api.holidays() and api.announcements(DAY, DAY) and api.event_calendar(DAY, DAY) and api.preopen("FO").rows == []


# ------------------------------------------------------------------------------------------- F&O maths
def test_black_scholes_roundtrip_and_invalid_prices():
    spot, strike = np.array([100.0, 100.0]), np.array([100.0, 110.0])
    is_call = np.array([True, False])
    for sigma in (0.15, 0.3, 0.8):
        price = bs_price(spot, strike, 30 / 365, np.full(2, sigma), is_call)
        recovered = implied_vol(price, spot, strike, 30 / 365, is_call)
        assert recovered == pytest.approx([sigma, sigma], abs=1e-4)
    below_intrinsic = implied_vol(np.array([0.5, np.nan, -1.0, 150.0]), np.full(4, 120.0), np.full(4, 100.0), 0.1, np.full(4, True))
    assert np.isnan(below_intrinsic).all()


def test_stock_fno_aggregation_matches_exchange_data():
    rows = nse.parse_fo_zip(fixture_bytes("fo_udiff_sample.zip"), DAY)
    out = {r["symbol"]: r for r in aggregate_stock_fno(rows, DAY, {"RELIANCE", "TCS"})}
    rel = out["RELIANCE"]
    assert rel["dte"] == 24 and rel["lot_size"] == 500
    assert rel["fut_oi"] == 149028500 and rel["fut_oi_chg"] == 2054500 and rel["next_fut_oi"] == 13739500
    assert rel["basis_pct"] == pytest.approx((1170.7 / 1164.88 - 1) * 100, rel=1e-3)
    assert 0 < rel["rollover_pct"] < 20 and rel["pcr_oi"] == pytest.approx(rel["put_oi"] / rel["call_oi"])
    assert rel["atm_strike"] == 1160.0 and rel["max_call_oi_strike"] == 1200.0 and rel["max_put_oi_strike"] == 1100.0
    # NSE's own chain showed ~25.8% ATM IV for the same day; the settlement-price inversion must be close
    assert 22.0 < rel["iv_atm"] < 30.0
    assert rel["expected_move_pct"] == pytest.approx(rel["atm_straddle"] / 1164.88 * 100)
    assert "iv_skew" in rel and abs(rel["iv_skew"]) < 15
    assert set(out) == {"RELIANCE", "TCS"}
    assert aggregate_stock_fno(rows, DAY, {"NOPE"}) == [] and aggregate_stock_fno([], DAY) == []


def test_aggregation_on_expiry_day_uses_half_day_time_floor():
    rows = [
        {"instrument": "STF", "symbol": "AAA", "expiry": DAY.isoformat(), "strike": None, "option_type": None, "close": 101.0,
         "settle": 101.0, "underlying": 100.0, "oi": 10.0, "oi_chg": -5.0, "volume": 3.0, "lot_size": 100.0},
        {"instrument": "STO", "symbol": "AAA", "expiry": DAY.isoformat(), "strike": 100.0, "option_type": "CE", "close": 2.0,
         "settle": 2.0, "underlying": 100.0, "oi": 50.0, "oi_chg": 1.0, "volume": 9.0},
        {"instrument": "STO", "symbol": "AAA", "expiry": DAY.isoformat(), "strike": 100.0, "option_type": "PE", "close": 1.5,
         "settle": 1.5, "underlying": 100.0, "oi": 70.0, "oi_chg": 2.0, "volume": 4.0},
    ]
    row = aggregate_stock_fno(rows, DAY)[0]
    assert row["dte"] == 0 and row["pcr_oi"] == pytest.approx(1.4) and row["atm_straddle"] == 3.5
    assert row["iv_atm"] > 0 and "next_fut_oi" not in row


# ----------------------------------------------------------------------------- point-in-time availability
def test_availability_respects_daylight_saving_shifts():
    spx = SERIES_BY_KEY["^GSPC"]
    before_dst = spx.available_at(date(2026, 3, 6))  # EST: 16:00 = 21:00 UTC (+30 min lag)
    after_dst = spx.available_at(date(2026, 3, 9))  # EDT: 16:00 = 20:00 UTC
    assert (before_dst.hour, before_dst.minute) == (21, 30) and (after_dst.hour, after_dst.minute) == (20, 30)
    nikkei = SERIES_BY_KEY["^N225"].available_at(date(2026, 11, 2))
    assert (nikkei.hour, nikkei.minute) == (7, 0)  # 15:30 JST = 06:30 UTC, + 30 min lag


def test_final_scan_cutoff_sees_previous_us_session_not_the_current_one():
    cutoff = datetime(2026, 11, 2, 3, 42, tzinfo=timezone.utc)  # 09:12 IST Monday
    spx = SERIES_BY_KEY["^GSPC"]
    dates = [date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)]
    assert latest_available_date(spx, dates, cutoff) == date(2026, 10, 30)  # Friday close; Monday's session has not happened
    fx = SERIES_BY_KEY["USDINR=X"]
    assert latest_available_date(fx, [date(2026, 10, 30), date(2026, 10, 31), date(2026, 11, 2)], cutoff) == date(2026, 10, 31)
    assert latest_available_date(spx, [date(2026, 11, 3)], cutoff) is None


def test_series_registry_is_consistent():
    keys = [s.key for s in SERIES]
    assert len(keys) == len(set(keys)) and "^NSEI" in keys and "^GSPC" in keys
    assert all(s.tz and s.close_hhmm and s.kind for s in SERIES)
