"""Panel loading: wide-matrix assembly, live row, special sessions (Muhurat) and data-quality coverage by exact date."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from automation.calendar import ExchangeCalendar, seed_calendar
from features.loader import _special_sessions, _wide_fields, load_panel
from ingestion.freshness import assess_data_quality
from tests.seed_data import panel_ending, populate_store

UTC = timezone.utc


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=200, n_symbols=12, seed=31)


@pytest.fixture()
def calendar(store, settings):
    seed_calendar(store, settings)
    return ExchangeCalendar(store)


def test_wide_fields_places_values_and_ignores_unknown_rows():
    dates = pd.DatetimeIndex(["2026-10-28", "2026-10-29", "2026-10-30"])
    long = pd.DataFrame({
        "symbol": ["AAA", "AAA", "BBB", "ZZZ", "BBB"],
        "trade_date": pd.to_datetime(["2026-10-28", "2026-10-30", "2026-10-29", "2026-10-29", "2026-11-05"]),
        "close": [1.0, 3.0, 20.0, 99.0, 55.0],
    })
    wide = _wide_fields(long, ("close",), dates, ["AAA", "BBB"])["close"]
    assert wide.loc["2026-10-28", "AAA"] == 1.0 and wide.loc["2026-10-30", "AAA"] == 3.0 and wide.loc["2026-10-29", "BBB"] == 20.0
    assert np.isnan(wide.loc["2026-10-29", "AAA"]) and wide.shape == (3, 2)  # unknown symbol / date dropped
    empty = _wide_fields(pd.DataFrame(), ("close",), dates, ["AAA"])["close"]
    assert empty.isna().all().all() and empty.shape == (3, 1)


def test_special_session_detection_flags_muhurat_like_days_only():
    dates = pd.bdate_range("2026-01-01", periods=60)
    volume = pd.DataFrame(1e6, index=dates, columns=["A", "B", "C"])
    volume.iloc[40] = 4e4  # a 4% session
    volume.iloc[50] = 6e5  # a merely quiet day (60%)
    flagged = _special_sessions(volume)
    assert flagged[40] and not flagged[50] and flagged.sum() == 1
    # weekend sessions need only a moderately low volume (short special sessions are rarely tiny)
    sat = pd.DatetimeIndex(list(dates[:30]) + [pd.Timestamp("2026-02-14")])  # a Saturday
    v2 = pd.DataFrame(1e6, index=sat, columns=["A", "B"])
    v2.iloc[-1] = 5e5
    assert _special_sessions(v2)[-1]


def test_load_panel_excludes_a_muhurat_session_from_the_trading_dates(store, panel):
    populate_store(store, panel)
    muhurat = "2026-11-01"  # a Sunday after the last regular session in the panel
    for symbol in panel.symbols:
        last_close = float(panel.c.iloc[-1][symbol])
        store.upsert_bars(store.instrument_id(symbol), [{"trade_date": muhurat, "open": last_close, "high": last_close * 1.002,
                                                         "low": last_close * 0.999, "close": last_close * 1.001,
                                                         "volume": 3e4}], source="yahoo")
    loaded = load_panel(store, start="2026-01-01", end="2026-11-01")
    assert pd.Timestamp(muhurat) not in loaded.dates and loaded.dates[-1] == pd.Timestamp("2026-10-30")
    assert loaded.c.index.equals(loaded.dates) and len(loaded.o) == len(loaded.dates)


def test_live_row_is_appended_and_open_is_the_only_value(store, panel, calendar):
    populate_store(store, panel)
    book = {s: float(panel.c.iloc[-1][s]) * 1.01 for s in panel.symbols[:5]}
    loaded = load_panel(store, start="2026-01-01", end="2026-10-30", today=date(2026, 11, 2), live_open=book, calendar=calendar)
    today = pd.Timestamp("2026-11-02")
    assert loaded.dates[-1] == today
    row_o, row_h = loaded.o.loc[today], loaded.h.loc[today]
    assert row_o[panel.symbols[:5]].notna().all() and row_o[panel.symbols[5:]].isna().all()
    assert row_h.isna().all() and loaded.c.loc[today].isna().all() and loaded.v.loc[today].isna().all()
    assert loaded.future_dates and loaded.next_after_last == loaded.future_dates[0]


def test_coverage_is_judged_on_the_expected_session_even_when_a_later_bar_exists(store, panel, calendar):
    """A Muhurat bar on the Sunday after the session must not make a missing Friday bar look present (or absent)."""
    populate_store(store, panel, skip_last_for=6)  # 6 of 12 symbols lack Friday 2026-10-30
    for symbol in panel.symbols:
        store.upsert_bars(store.instrument_id(symbol), [{"trade_date": "2026-11-01", "open": 100, "high": 101, "low": 99,
                                                         "close": 100.5, "volume": 1e4}], source="yahoo")
    cutoff = datetime(2026, 11, 2, 3, 15, tzinfo=UTC)
    quality = assess_data_quality(store, calendar, cutoff=cutoff, stage="pre", trade_date=date(2026, 11, 2))
    assert quality.expected_session == "2026-10-30" and quality.india_coverage == pytest.approx(0.5)
    assert quality.blocked
    assert store.symbols_with_bar("2026-10-30", universe_only=True) == set(panel.symbols[6:])
