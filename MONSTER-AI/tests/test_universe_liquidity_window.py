"""The universe cap ranks by traded value over a window anchored on the date it is given (an IST trading date), never on the
SQL clock: that ignored the injected clock and the server timezone, so replaying or testing a date gave a different universe."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from ingestion.universe import LIQUIDITY_WINDOW_DAYS, _cap_by_liquidity, build_universe


class Lists:
    """NSE index lists: 80 names, all in the Nifty 100, none in F&O."""

    def __init__(self, n=80):
        self.rows = [{"symbol": f"S{i:03d}", "name": "n", "industry": "x", "isin": "i", "series": "EQ"} for i in range(n)]

    def index_constituents(self, index):
        return {"nifty500": self.rows, "nifty100": self.rows, "nifty200": []}[index]

    def fo_lot_sizes(self):
        return {}


def traded(store, symbol, day, turnover):
    store.upsert_instrument(symbol, "equity")
    store.upsert_bars(store.instrument_id(symbol), [{"trade_date": day, "open": 1, "high": 2, "low": 1, "close": 1.5, "turnover": turnover}],
                      source="nse_bhavcopy", source_rank=2)


def test_the_liquidity_window_ends_on_the_date_given_not_on_the_sql_clock(store, settings):
    as_of = date(2031, 6, 2)  # years away from the real clock: a clock-based window could not contain these bars
    traded(store, "S079", "2031-05-20", 9e12)  # 13 days before as_of: inside the window
    traded(store, "S078", "2031-03-01", 9e13)  # 93 days before as_of: far outside it, however large
    result = build_universe(store, replace(settings, universe_max_size=50), Lists(), as_of)
    assert len(result.symbols) == 50 and "S079" in result.symbols and "S078" not in result.symbols
    assert any("capped" in n for n in result.notes)


def test_the_window_edge_is_inclusive_and_exactly_the_documented_length(store, settings):
    as_of = date(2031, 6, 2)
    traded(store, "S077", (as_of - timedelta(days=LIQUIDITY_WINDOW_DAYS)).isoformat(), 9e12)  # exactly on the edge: counted
    traded(store, "S076", (as_of - timedelta(days=LIQUIDITY_WINDOW_DAYS + 1)).isoformat(), 9e13)  # one day too old
    result = build_universe(store, replace(settings, universe_max_size=50), Lists(), as_of)
    assert "S077" in result.symbols and "S076" not in result.symbols


def test_a_past_date_ranks_by_its_own_window_not_by_todays(store, settings):
    """Replaying a refresh for a date in the past must see the traded value of THAT window (the SQL clock saw today's)."""
    traded(store, "S079", "2020-02-14", 9e12)
    result = build_universe(store, replace(settings, universe_max_size=50), Lists(), date(2020, 3, 2))
    assert "S079" in result.symbols


def test_without_a_date_the_injected_clock_decides_in_ist(store, fake_clock):
    """FakeClock: 2026-11-02 03:00 UTC = 08:30 IST. Late evening UTC is already the next IST day, so the IST date is used."""
    rows = [{"symbol": f"S{i:03d}", "nifty100": True, "is_fno": False} for i in range(80)]
    traded(store, "S079", "2026-09-20", 9e12)  # 43 days before 2026-11-02: inside the window
    traded(store, "S078", "2026-09-17", 9e13)  # 46 days before: outside
    symbols = [r["symbol"] for r in rows]
    assert "S079" in _cap_by_liquidity(store, symbols, 50, rows) and "S078" not in _cap_by_liquidity(store, symbols, 50, rows)
    fake_clock.set(datetime(2026, 11, 2, 19, 0, tzinfo=timezone.utc))  # 00:30 IST on 2026-11-03: the window moves a day
    traded(store, "S077", "2026-09-19", 9e14)  # 45 days before 2026-11-03: the last day inside; 46 days before 2026-11-04
    assert "S077" in _cap_by_liquidity(store, symbols, 50, rows)
    fake_clock.set(datetime(2026, 11, 3, 19, 0, tzinfo=timezone.utc))  # 00:30 IST on 2026-11-04: now 46 days old
    assert "S077" not in _cap_by_liquidity(store, symbols, 50, rows)
