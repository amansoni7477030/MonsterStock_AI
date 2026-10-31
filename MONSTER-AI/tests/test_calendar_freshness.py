from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from automation.calendar import (
    ExchangeCalendar,
    holidays_between,
    infer_holidays_from_bars,
    refresh_calendar,
    seed_calendar,
    trading_days_before,
)
from ingestion import pipeline as pl
from ingestion.freshness import assess_data_quality, expected_bar_date
from ingestion.nse import parse_holidays
from ingestion.registry import register_all
from ingestion.series import SERIES
from tests.helpers import fixture_json


@pytest.fixture()
def calendar(store, settings):
    seed_calendar(store, settings)
    return ExchangeCalendar(store)


def ist(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone(timedelta(hours=5, minutes=30))).astimezone(timezone.utc)


# ------------------------------------------------------------------------------------------ calendar
def test_trading_days_and_official_holidays(calendar):
    assert calendar.is_trading_day(date(2026, 11, 2))  # Monday
    assert not calendar.is_trading_day(date(2026, 10, 31)) and not calendar.is_trading_day(date(2026, 11, 1))  # weekend
    assert not calendar.is_trading_day(date(2026, 10, 20))  # Dussehra, taken from NSE's official list
    assert not calendar.is_trading_day(date(2026, 11, 10)) and not calendar.is_trading_day(date(2026, 11, 24))
    assert "Dussehra" in calendar.holiday_name(date(2026, 10, 20)) and calendar.holiday_name(date(2026, 10, 31)) == "weekend"
    assert calendar.holiday_name(date(2026, 11, 2)) is None
    assert "Muhurat" in calendar.holiday_name(date(2026, 11, 8))  # Diwali evening session only: no regular session


def test_previous_next_and_ranges(calendar):
    assert calendar.previous_trading_day(date(2026, 11, 2)) == date(2026, 10, 30)  # skips the weekend
    assert calendar.previous_trading_day(date(2026, 10, 21)) == date(2026, 10, 19)  # skips the Dussehra holiday
    assert calendar.next_trading_day(date(2026, 11, 9)) == date(2026, 11, 11)  # skips Diwali Balipratipada
    assert calendar.trading_days(date(2026, 11, 6), date(2026, 11, 11)) == [date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 11)]
    assert trading_days_before(calendar, date(2026, 11, 2), 3) == [date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)]
    holidays = list(holidays_between(calendar, date(2026, 10, 15), date(2026, 10, 25)))
    assert holidays == [(date(2026, 10, 20), "Dussehra")]


def test_session_times_are_ist_aware(calendar):
    session = calendar.session(date(2026, 11, 2))
    assert session.is_trading and session.kind == "regular"
    assert session.preopen_start == ist(2026, 11, 2, 9, 0) and session.open == ist(2026, 11, 2, 9, 15)
    assert session.close == ist(2026, 11, 2, 15, 30)
    closed = calendar.session(date(2026, 10, 20))
    assert not closed.is_trading and closed.kind == "holiday" and closed.open is None
    assert calendar.session(date(2026, 11, 1)).kind == "weekend"


def test_last_completed_session_boundaries(calendar):
    assert calendar.last_completed_session(ist(2026, 11, 2, 8, 30)) == date(2026, 10, 30)  # Monday pre-open -> Friday
    assert calendar.last_completed_session(ist(2026, 11, 2, 15, 35)) == date(2026, 10, 30)  # inside the publication lag
    assert calendar.last_completed_session(ist(2026, 11, 2, 15, 41)) == date(2026, 11, 2)
    assert calendar.last_completed_session(ist(2026, 11, 10, 12, 0)) == date(2026, 11, 9)  # holiday Tuesday
    assert calendar.session_for_scan(ist(2026, 11, 2, 9, 10)) == date(2026, 11, 2)
    assert calendar.session_for_scan(ist(2026, 11, 10, 9, 10)) is None


def test_special_session_override_reopens_a_saturday(store, calendar):
    store.upsert_calendar_days([{"trade_date": "2026-11-07", "is_trading_day": True, "session_kind": "special",
                                 "open_time": "09:15:00", "close_time": "11:00:00", "description": "special live session"}],
                               source="override")
    calendar.reload()
    session = calendar.session(date(2026, 11, 7))
    assert session.is_trading and session.kind == "special" and session.close == ist(2026, 11, 7, 11, 0)
    assert calendar.next_trading_day(date(2026, 11, 6)) == date(2026, 11, 7)


def test_refresh_from_nse_keeps_overrides_and_validates(store, calendar):
    class Nse:
        def holidays(self):
            return parse_holidays(fixture_json("nse_holiday_master.json"))

    store.upsert_calendar_days([{"trade_date": "2026-11-24", "is_trading_day": True, "session_kind": "special",
                                 "description": "manual override"}], source="override")
    assert refresh_calendar(store, Nse()) >= 19
    calendar.reload()
    assert calendar.is_trading_day(date(2026, 11, 24))  # operator override wins over the API
    assert store.get_calendar_day("2026-10-20")["source"] == "nse_api"

    class Empty:
        def holidays(self):
            return []

    from app.errors import DataUnavailable

    with pytest.raises(DataUnavailable):
        refresh_calendar(store, Empty())


def test_infer_holidays_from_missing_index_bars(store):
    iid = store.upsert_instrument("^NSEI", "index")
    day, rows = date(2026, 9, 1), []
    while day <= date(2026, 10, 9):
        if day.weekday() < 5 and day != date(2026, 9, 17):  # 2026-09-17 has no bar -> inferred closed
            rows.append({"trade_date": day.isoformat(), "open": 1, "high": 2, "low": 1, "close": 1.5})
        day += timedelta(days=1)
    store.upsert_bars(iid, rows, source="yahoo")
    assert infer_holidays_from_bars(store) == 1
    cal = ExchangeCalendar(store)
    assert not cal.is_trading_day(date(2026, 9, 17)) and cal.is_trading_day(date(2026, 9, 18))
    assert infer_holidays_from_bars(store) == 0  # idempotent
    assert infer_holidays_from_bars(ExchangeCalendar(store).store.__class__(store.conn)) == 0


def test_calendar_without_store_falls_back_to_weekdays():
    cal = ExchangeCalendar(None)
    assert cal.is_trading_day(date(2026, 11, 2)) and not cal.is_trading_day(date(2026, 11, 1))


# ------------------------------------------------------------------------------------------ freshness
CUTOFF_FINAL = ist(2026, 11, 2, 9, 12)


def seed_market(store, calendar, *, skip: set[str] = frozenset(), india_missing: int = 0, stale: set[str] = frozenset()):
    """A healthy universe of 20 stocks (10 F&O) + all global series with bars final at the cutoff."""
    pl.ensure_series_instruments(store)
    for i in range(20):
        iid = store.upsert_instrument(f"S{i:02d}", "equity", is_fno=1 if i < 10 else 0, in_universe=1)
        if i >= 20 - india_missing:
            continue
        store.upsert_bars(iid, [{"trade_date": "2026-10-30", "open": 100, "high": 102, "low": 99, "close": 101}], source="yahoo")
    for spec in SERIES:
        if spec.key in skip:
            continue
        iid = store.instrument_id(spec.key, spec.kind)
        d = expected_bar_date(spec, CUTOFF_FINAL)
        if spec.category == "india_index" or spec.key == "^INDIAVIX":
            d = date(2026, 10, 30)
        if spec.key in stale:
            d = d - timedelta(days=9)
        store.upsert_bars(iid, [{"trade_date": d.isoformat(), "open": 1, "high": 2, "low": 1, "close": 1.5}], source="yahoo")
    store.upsert_fno([{"symbol": f"S{i:02d}", "trade_date": "2026-10-30", "fut_oi": 1} for i in range(10)])
    register_all(store)  # production registers the source catalogue at init; health queries are driven by it
    store.record_source_result("rss_news", True)
    store.record_source_result("nse_announcements", True)


def test_healthy_data_is_ok(store, calendar, fake_clock):
    seed_market(store, calendar)
    store.add_live_quote("GIFT_NIFTY", price=23000, source_ts="2026-11-02T03:30:00Z")
    store.add_preopen_quotes("2026-11-02", [{"symbol": f"S{i:02d}", "iep": 100, "final_price": 101} for i in range(20)], None)
    fake_clock.set(CUTOFF_FINAL)
    quality = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="final", trade_date=date(2026, 11, 2))
    assert quality.status == "ok", quality.messages
    assert quality.expected_session == "2026-10-30" and quality.india_coverage == 1.0 and quality.fno_coverage == 1.0
    assert quality.preopen_final_coverage == 1.0 and quality.global_fresh == quality.global_total
    assert quality.missing_feeds() == [] and "India daily bars" in quality.freshness()


def test_missing_india_bars_block_the_report(store, calendar):
    seed_market(store, calendar, india_missing=4)  # 80% coverage
    quality = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert quality.blocked and quality.feeds["India daily bars"].level == "stale"
    assert any("India daily bars" in m for m in quality.missing_feeds())


def test_partial_india_coverage_only_warns(store, calendar):
    seed_market(store, calendar, india_missing=1)  # 95%
    quality = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert quality.status == "degraded" and quality.feeds["India daily bars"].level == "warn"


def test_stale_critical_global_series_degrade_until_the_outage_is_systemic(store, calendar):
    seed_market(store, calendar, stale={"^GSPC"})
    one = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert not one.blocked and one.feeds["Global closes"].level == "warn"
    store.conn.execute("DELETE FROM bars_daily")
    seed_market(store, calendar, stale={"^GSPC", "^IXIC", "^VIX"})
    three = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert not three.blocked and three.status == "degraded" and "^GSPC" in three.feeds["Global closes"].detail
    store.conn.execute("DELETE FROM bars_daily")
    seed_market(store, calendar, stale={"^GSPC", "^IXIC", "^VIX", "USDINR=X", "DX-Y.NYB", "^TNX"})
    systemic = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert systemic.blocked and "^GSPC" in systemic.feeds["Global closes"].detail


def test_missing_nifty_index_always_blocks(store, calendar):
    seed_market(store, calendar, skip={"^NSEI"})
    quality = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert quality.blocked


def test_missing_preopen_degrades_final_but_not_prelim(store, calendar):
    seed_market(store, calendar)
    prelim = assess_data_quality(store, calendar, cutoff=ist(2026, 11, 2, 8, 45), stage="pre", trade_date=date(2026, 11, 2))
    assert "Pre-open book" not in prelim.feeds
    final = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="final", trade_date=date(2026, 11, 2))
    assert final.status == "degraded" and final.feeds["Pre-open book"].level == "missing"
    store.add_preopen_quotes("2026-11-02", [{"symbol": f"S{i:02d}", "iep": 100} for i in range(20)], None)
    partial = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="final", trade_date=date(2026, 11, 2))
    assert partial.feeds["Pre-open book"].level == "warn"  # indicative prices only, no final equilibrium price


def test_gift_news_and_fno_states(store, calendar):
    seed_market(store, calendar)
    store.conn.execute("DELETE FROM fno_daily")
    store.add_live_quote("GIFT_NIFTY", price=23000, source_ts="2026-11-02T01:00:00Z")  # 2.7 h old at the cutoff
    store.conn.execute("DELETE FROM source_health WHERE source_id = 'rss_news'")
    quality = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert quality.feeds["GIFT Nifty"].level == "stale" and quality.feeds["News feeds"].level == "missing"
    assert quality.feeds["F&O end-of-day"].level == "missing" and quality.status == "degraded"
    assert not quality.blocked


def test_no_universe_is_blocked(store, calendar):
    quality = assess_data_quality(store, calendar, cutoff=CUTOFF_FINAL, stage="pre", trade_date=date(2026, 11, 2))
    assert quality.blocked and quality.feeds["India daily bars"].detail == "no universe loaded"
    assert quality.to_dict()["status"] == "blocked"
