"""Scheduled-results horizon (training and live must know the same meetings), the sector dispersion that is leave-one-out as documented,
and the dead ``c_n`` feature."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.clock import at_ist
from automation.calendar import ExchangeCalendar, seed_calendar
from features import calendar_features as cal
from features.builder import build_feature_set
from features.calendar_features import RESULTS_HORIZON, next_trading_days
from features.live import _with_results_horizon, build_live_features
from features.sector import sector_features
from tests.seed_data import panel_ending, populate_store
from tests.synthetic import make_panel

DATES = pd.bdate_range("2026-01-05", periods=80)
SYMBOLS = ["AAA", "BBB"]


def meetings_on(*items: tuple[str, int]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "meeting_date": DATES[i].date().isoformat(), "purpose": "Results", "is_results": 1}
                         for s, i in items])


def upcoming(after: int, n: int = RESULTS_HORIZON) -> list[date]:
    return [d.date() for d in pd.bdate_range(DATES[after] + pd.Timedelta(days=1), periods=n)]


def results(meetings, dates=DATES, future=None, coverage=None):
    return cal.results_features(meetings, dates, upcoming(len(dates) - 1) if future is None else future, SYMBOLS,
                                coverage or DATES[0].date())


def value(frame: pd.DataFrame, symbol: str, row: int) -> float:
    return float(frame[symbol].iloc[row])


# ----------------------------------------------------------------------------------- results horizon
def test_days_to_results_counts_sessions_and_stops_at_the_horizon():
    out = results(meetings_on(("AAA", 40)))["c_days_to_results"]
    assert RESULTS_HORIZON == 10
    assert [value(out, "AAA", i) for i in (39, 35, 31, 30)] == [1.0, 5.0, 9.0, 10.0]
    assert np.isnan(value(out, "AAA", 29)) and np.isnan(value(out, "AAA", 10))  # 11 and 30 sessions ahead: it said 11 and 30
    assert np.isnan(value(out, "AAA", 40)) and np.isnan(out["BBB"]).all()  # nothing after the meeting itself, nothing for BBB


def test_a_meeting_beyond_the_horizon_is_unknown_not_far_away():
    out = results(meetings_on(("AAA", 75)))["c_days_to_results"]
    assert np.isnan(out["AAA"].iloc[:65]).all() and value(out, "AAA", 65) == 10.0 and value(out, "AAA", 74) == 1.0


@pytest.mark.parametrize("ahead", [1, 5, 10, 11, 20])
def test_a_live_style_panel_sees_exactly_what_the_full_history_sees(ahead):
    """Live knows the upcoming sessions from the calendar; training from the later rows of the history. Same answer."""
    row = 30
    full = results(meetings_on(("AAA", row + ahead)))["c_days_to_results"]["AAA"].iloc[row]
    live = results(meetings_on(("AAA", row + ahead)), dates=DATES[:row + 1], future=upcoming(row))["c_days_to_results"]["AAA"].iloc[-1]
    assert (np.isnan(full) and np.isnan(live)) or full == live == ahead
    assert np.isnan(full) == (ahead > RESULTS_HORIZON)


def test_without_enough_upcoming_sessions_the_far_end_of_the_horizon_is_missing():
    short = results(meetings_on(("AAA", 38)), dates=DATES[:31], future=upcoming(30, 5))["c_days_to_results"]["AAA"].iloc[-1]
    assert np.isnan(short)  # a meeting 8 sessions ahead needs 8 upcoming sessions in the panel
    full = results(meetings_on(("AAA", 38)), dates=DATES[:31], future=upcoming(30))["c_days_to_results"]["AAA"].iloc[-1]
    assert full == 8.0


def test_the_other_results_features_are_unchanged():
    out = results(meetings_on(("AAA", 40)))
    assert value(out["c_days_since_results"], "AAA", 45) == 5.0 and value(out["c_days_since_results"], "AAA", 79) == 39.0
    assert value(out["c_results_next2"], "AAA", 38) == 1.0 and value(out["c_results_next2"], "AAA", 37) == 0.0
    assert value(out["c_results_today"], "AAA", 40) == 1.0 and value(out["c_results_prev"], "AAA", 41) == 1.0


def test_dates_before_the_coverage_start_stay_unknown():
    out = results(meetings_on(("AAA", 40)), coverage=DATES[35].date())["c_days_to_results"]
    assert out["AAA"].iloc[:35].isna().all() and value(out, "AAA", 39) == 1.0


# --------------------------------------------------------- the panel carries the horizon (live and training)
class Calendar:
    def __init__(self, closed: set[date]) -> None:
        self.closed = closed

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self.closed


def test_the_panel_is_given_the_upcoming_sessions_the_horizon_needs():
    panel = make_panel(n_dates=30, n_symbols=4, with_fno=False, with_meetings=False)
    assert len(panel.future_dates) == 5  # what the loader supplies
    holiday = panel.future_dates[2]
    assert _with_results_horizon(panel, Calendar({holiday})) is panel
    assert len(panel.future_dates) == RESULTS_HORIZON and holiday not in panel.future_dates
    assert panel.future_dates == next_trading_days(panel.dates[-1].date(), RESULTS_HORIZON, Calendar({holiday}).is_trading_day)
    assert panel.next_after_last == panel.future_dates[0]


def test_a_panel_that_already_has_enough_sessions_is_left_alone():
    panel = make_panel(n_dates=30, n_symbols=4, with_fno=False, with_meetings=False)
    panel.future_dates = [panel.dates[-1].date() + timedelta(days=i) for i in range(1, 13)]
    before, nxt = list(panel.future_dates), panel.next_after_last
    _with_results_horizon(panel, Calendar(set()))
    assert panel.future_dates == before and panel.next_after_last == nxt


def test_a_live_scan_sees_a_meeting_eight_sessions_ahead(store, settings, fake_clock):
    trade_date = date(2026, 11, 2)
    panel = panel_ending("2026-10-30", n_dates=140, n_symbols=6, seed=3, with_fno=False, with_meetings=False)
    seed_calendar(store, settings)
    calendar = ExchangeCalendar(store)
    sessions = next_trading_days(trade_date, RESULTS_HORIZON + 1, calendar.is_trading_day)
    panel.meetings = pd.DataFrame([
        {"symbol": "S00", "meeting_date": sessions[7].isoformat(), "purpose": "Financial Results", "is_results": 1},  # 8th session ahead
        {"symbol": "S01", "meeting_date": sessions[10].isoformat(), "purpose": "Financial Results", "is_results": 1},  # 11th: too far
        {"symbol": "S02", "meeting_date": sessions[0].isoformat(), "purpose": "Financial Results", "is_results": 1},   # the next one
    ])
    panel.meetings_coverage_start = panel.dates[0].date()
    populate_store(store, panel)
    live = build_live_features(store, settings, calendar, trade_date=trade_date, stage="pre", cutoff=at_ist(trade_date, "08:45"),
                               with_fallback=False)
    days = live.X["c_days_to_results"]
    assert days["S00"] == 8.0 and days["S02"] == 1.0 and np.isnan(days["S01"]) and np.isnan(days["S03"])


# -------------------------------------------------------------------------------- sector dispersion
def sector_inputs(r1: pd.DataFrame):
    zeros = pd.DataFrame(0.0, index=r1.index, columns=r1.columns)
    ones = zeros + 1.0
    sectors = {"A1": "Fin", "A2": "Fin", "A3": "Fin", "A4": "Fin", "B1": "IT", "B2": "IT", "C1": None}
    return sector_features(r1, zeros, zeros, ones, ones, sectors, pd.Series(0.0, index=r1.index))


def returns() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(rng.normal(0, 1.5, (5, 7)), index=pd.bdate_range("2026-03-02", periods=5),
                        columns=["A1", "A2", "A3", "A4", "B1", "B2", "C1"])


def test_sector_dispersion_leaves_the_stocks_own_return_out():
    r1 = returns()
    out = sector_inputs(r1)["s_disp1"]
    for member in ("A1", "A2", "A3", "A4"):
        others = r1[[m for m in ("A1", "A2", "A3", "A4") if m != member]]
        np.testing.assert_allclose(out[member], others.std(axis=1, ddof=0), rtol=1e-9, err_msg=member)


def test_a_stocks_own_move_cannot_change_its_own_sector_dispersion_but_changes_its_peers():
    r1 = returns()
    base = sector_inputs(r1)["s_disp1"]
    moved = r1.copy()
    moved["A1"] = 40.0  # A1 is the outlier of the day
    out = sector_inputs(moved)["s_disp1"]
    np.testing.assert_allclose(out["A1"], base["A1"])  # it was included before: the outlier inflated its own context
    assert (out["A2"] > 5 * base["A2"]).all() and (out["A2"] > 5.0).all()


def test_dispersion_ignores_missing_returns_and_needs_two_other_stocks():
    r1 = returns()
    r1.loc[r1.index[0], "A2"] = np.nan  # A2 has no return that day
    r1.loc[r1.index[1], ["A2", "A3"]] = np.nan
    r1.loc[r1.index[2], ["A2", "A3", "A4"]] = np.nan  # only A1 has a return
    out = sector_inputs(r1)["s_disp1"]
    assert out["A1"].iloc[0] == pytest.approx(r1.loc[r1.index[0], ["A3", "A4"]].std(ddof=0))
    assert out["A2"].iloc[0] == pytest.approx(r1.loc[r1.index[0], ["A1", "A3", "A4"]].std(ddof=0))  # its own gap: all others count
    assert np.isnan(out["A1"].iloc[1]) and np.isnan(out["A4"].iloc[1])  # a single other stock has no dispersion
    assert np.isnan(out["A4"].iloc[2]) and np.isnan(out["A1"].iloc[2])


def test_small_sectors_have_no_dispersion():
    out = sector_inputs(returns())["s_disp1"]
    assert out[["B1", "B2", "C1"]].isna().all().all()  # one other stock, or none: nothing to measure


# ----------------------------------------------------------------------------------------- the dead c_n
def test_the_constant_panel_length_feature_is_gone():
    assert "c_n" not in cal.calendar_market_features(DATES, None)
    panel = make_panel(n_dates=150, n_symbols=6, with_fno=False, with_meetings=False)
    fs = build_feature_set(panel, stage="pre", thresholds=(1.0, 2.0, 3.0, 5.0), unusual_k=1.5, cutoff_hhmm="08:45",
                           with_labels=False, row_slice=slice(100, 150))
    assert "c_n" not in fs.X.columns and "c_dow" in fs.X.columns
