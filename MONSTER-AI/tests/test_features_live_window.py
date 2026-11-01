"""Train/serve parity of the feature builder: the window a live scan loads (LIVE_HISTORY_DAYS) must reproduce, for the scan date, EVERY
feature of the training row computed from the full history. The only exception is ``u_hist_days``, a cumulative count that no
model reads (it filters rows). Any rolling window deeper than the live window, or an expanding statistic, shows up here."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.clock import at_ist
from automation.calendar import ExchangeCalendar, seed_calendar
from features import similar_day as sd
from features.builder import Panel, build_feature_set
from features.calendar_features import RESULTS_HORIZON
from features.live import LIVE_HISTORY_DAYS, build_live_features
from tests.seed_data import panel_ending, populate_store
from tests.synthetic import make_panel

THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
NSE_SESSIONS_PER_YEAR = 245  # fewer than any recent year's regular sessions
LIVE_ROWS = int(LIVE_HISTORY_DAYS * 5 / 7)  # business days in a live window (the synthetic calendar has no holidays)
NOT_COMPARABLE = {"u_hist_days"}


def slice_panel(p: Panel, start: int, stop: int) -> Panel:
    """Rows [start, stop) of a panel as a loader would supply them for a scan on the session ``stop - 1``: the global series of
    that window, the F&O rows of its dates, and the upcoming sessions after its last date."""
    rows = slice(start, stop)
    first, last = p.dates[start], p.dates[stop - 1]
    closes = {k: s[(s.index >= first - pd.Timedelta(days=30)) & (s.index <= last + pd.Timedelta(days=1))] for k, s in p.closes.items()}
    fno = None if p.fno is None else p.fno[(p.fno["trade_date"] >= first) & (p.fno["trade_date"] <= last)]
    upcoming = [d.date() for d in pd.bdate_range(last + pd.Timedelta(days=1), periods=RESULTS_HORIZON)]
    return replace(p, dates=p.dates[rows], o=p.o.iloc[rows], h=p.h.iloc[rows], low=p.low.iloc[rows], c=p.c.iloc[rows],
                   v=p.v.iloc[rows], nifty=p.nifty.iloc[rows], vix=p.vix.iloc[rows], closes=closes, fno=fno,
                   next_after_last=upcoming[0], future_dates=upcoming)


def live_window(p: Panel, end: int) -> Panel:
    """What a live scan loads for the session at row ``end``: every session within LIVE_HISTORY_DAYS calendar days of it."""
    first = int(np.searchsorted(p.dates, p.dates[end] - pd.Timedelta(days=LIVE_HISTORY_DAYS), side="left"))
    return slice_panel(p, first, end + 1)


def one_row(p: Panel, stage: str, row: int | None = None) -> pd.DataFrame:
    row = len(p.dates) - 1 if row is None else row
    fs = build_feature_set(p, stage=stage, thresholds=THRESHOLDS, unusual_k=1.5, cutoff_hhmm="08:45", with_labels=False,
                           row_slice=slice(row, row + 1))
    return fs.X.droplevel("date")


@pytest.fixture(scope="module")
def long_panel() -> Panel:
    return make_panel(n_dates=LIVE_ROWS + 120, n_symbols=12, seed=5, with_fno=True, with_meetings=True)


def test_the_live_window_holds_the_history_the_engine_needs():
    assert LIVE_HISTORY_DAYS / 365.25 * NSE_SESSIONS_PER_YEAR >= sd.HISTORY_SESSIONS
    assert LIVE_ROWS > sd.HISTORY_SESSIONS


@pytest.mark.parametrize("stage", ["pre", "final"])
@pytest.mark.parametrize("back", [0, 9, 40])
def test_a_live_window_reproduces_every_feature_of_the_training_row(long_panel, stage, back):
    end = len(long_panel.dates) - 1 - back
    training = one_row(long_panel, stage, row=end)  # every earlier AND later session is available
    window = live_window(long_panel, end)
    assert len(window.dates) < end  # the live window really is shorter than the history
    live = one_row(window, stage)
    assert list(live.columns) == list(training.columns) and "c_n" not in live.columns
    differing = [c for c in training.columns
                 if not np.allclose(live[c].to_numpy(float), training[c].to_numpy(float), rtol=1e-5, atol=1e-6, equal_nan=True)]
    assert set(differing) <= NOT_COMPARABLE, differing
    sim = [c for c in training.columns if c.startswith("x_sim_")]
    assert len(sim) == 10 and training[sim].notna().all().all()  # not vacuous: the similar-day features exist for the date


def test_a_window_shorter_than_the_engine_needs_would_not_reproduce_the_row(long_panel):
    """The guard bites: with 300 sessions instead of the live window the similar-day features differ from the full history."""
    end = len(long_panel.dates) - 1
    training = one_row(long_panel, "pre", row=end)
    short = slice_panel(long_panel, end - 300, end + 1)
    live = one_row(short, "pre")
    cols = [c for c in training.columns if c.startswith("x_sim_")]
    assert not np.allclose(live[cols].to_numpy(float), training[cols].to_numpy(float), rtol=1e-5, atol=1e-6, equal_nan=True)


# ------------------------------------------------------------------------ the database path (what a scan runs)
def test_the_live_scan_features_come_out_the_same_as_the_training_row_for_that_date(store, settings, fake_clock, caplog):
    trade_date = date(2026, 11, 2)
    panel = panel_ending(trade_date.isoformat(), n_dates=LIVE_ROWS + 120, n_symbols=10, seed=5, with_fno=False,
                         with_meetings=False)
    populate_store(store, slice_panel(panel, 0, len(panel.dates) - 1))  # the database holds everything before the scan day
    seed_calendar(store, settings)
    with caplog.at_level(logging.WARNING, logger="features.live"):
        live = build_live_features(store, settings, ExchangeCalendar(store), trade_date=trade_date, stage="pre",
                                   cutoff=at_ist(trade_date, "08:45"), with_fallback=False)
    training = one_row(panel, "pre")
    cols = [c for c in training.columns if c.startswith("x_sim_")]
    np.testing.assert_allclose(live.X.loc[training.index, cols].to_numpy(dtype=float), training[cols].to_numpy(dtype=float),
                               rtol=1e-5, atol=1e-6)
    assert live.market["x_sim_n"] == sd.K
    assert "similar-day features need" not in caplog.text  # the window was long enough: nothing to warn about


def test_a_live_scan_with_too_little_history_says_so(store, settings, fake_clock, caplog):
    panel = panel_ending("2026-10-30", n_dates=200, n_symbols=6, seed=2, with_fno=False, with_meetings=False)
    populate_store(store, panel)
    seed_calendar(store, settings)
    trade_date = date(2026, 11, 2)
    with caplog.at_level(logging.WARNING, logger="features.live"):
        live = build_live_features(store, settings, ExchangeCalendar(store), trade_date=trade_date, stage="pre",
                                   cutoff=at_ist(trade_date, "08:45"), with_fallback=False)
    assert len(live.X) == 6  # the scan still runs
    warning = next(r.getMessage() for r in caplog.records if "similar-day" in r.getMessage())
    assert f"only 200 sessions of history before {trade_date}" in warning and f"need {sd.HISTORY_SESSIONS}" in warning
