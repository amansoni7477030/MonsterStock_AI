"""Point-in-time F&O membership (``u_is_fno``) and the survivorship-aware universe of training panels."""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.clock import FakeClock
from database.store import Store
from features.builder import build_feature_set
from features.loader import fno_first_dates, fno_membership, load_panel
from tests.bars_support import add_equity, bar_rows, economic_path, put_official, sessions
from tests.seed_data import panel_ending, populate_store
from tests.synthetic import make_panel

N = 30
DAYS = sessions(N)
THRESHOLDS = (1.0, 2.0, 3.0, 5.0)


def fno_rows(symbol, positions):
    return [{"symbol": symbol, "trade_date": DAYS[i]} for i in positions]


def load(store, **kw):
    return load_panel(store, start=DAYS[0], end=DAYS[-1], **kw)


# ============================================================================================= F&O membership
def test_membership_is_nan_before_any_fno_data_zero_before_the_first_row_and_one_after():
    dates = pd.DatetimeIndex(DAYS[:10])
    frame = fno_membership(dates, ["AAA", "BBB", "CCC"], DAYS[2], {"AAA": DAYS[5], "BBB": DAYS[2]})
    assert frame["AAA"].isna().tolist() == [True, True] + [False] * 8
    assert frame["AAA"].iloc[2:].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert frame["BBB"].iloc[2:].tolist() == [1.0] * 8
    assert frame["CCC"].iloc[2:].tolist() == [0.0] * 8  # F&O data exists on those dates, just not for CCC
    assert frame.iloc[:2].isna().all().all()


def test_membership_without_any_fno_data_is_unknown_not_zero():
    frame = fno_membership(pd.DatetimeIndex(DAYS[:3]), ["AAA"], None, {})
    assert frame.shape == (3, 1) and frame.isna().all().all()
    assert fno_membership(pd.DatetimeIndex([]), ["AAA"], DAYS[0], {}).shape == (0, 1)  # an empty panel is fine too


def test_first_fno_dates_cover_the_whole_database(store):
    assert fno_first_dates(store) == (None, {})
    store.upsert_fno(fno_rows("AAA", range(10, 20)) + fno_rows("BBB", range(5, 8)))
    assert fno_first_dates(store) == (DAYS[5], {"AAA": DAYS[10], "BBB": DAYS[5]})


def test_the_panel_carries_point_in_time_membership_not_todays_flag(store):
    for i, (symbol, flagged) in enumerate((("AAA", 1), ("BBB", 0), ("CCC", 0))):
        put_official(store, add_equity(store, symbol, is_fno=flagged), bar_rows(DAYS, economic_path(N, seed=i + 1), range(N)))
    store.upsert_fno(fno_rows("AAA", range(10, N)) + fno_rows("BBB", range(5, 16)) + fno_rows("ZZZ", range(2, N)))
    panel = load(store)
    member = panel.is_fno_pit
    assert panel.is_fno == {"AAA": 1, "BBB": 0, "CCC": 0}  # the static flag still describes today
    assert member.iloc[:2].isna().all().all()  # before the earliest F&O data of the database (ZZZ, day 2): no information
    assert member["AAA"].iloc[2:].tolist() == [0.0] * 8 + [1.0] * (N - 10)  # flagged today, but F&O only from day 10
    assert member["BBB"].iloc[2:].tolist() == [0.0] * 3 + [1.0] * (N - 5)  # left F&O after day 15; membership is "has F&O rows so far"
    assert member["CCC"].iloc[2:].tolist() == [0.0] * (N - 2)
    assert set(panel.fno["symbol"]) == {"AAA", "BBB"}  # a lapsed member keeps the F&O rows of its F&O years


def test_the_live_row_follows_the_same_definition(store):
    for i, symbol in enumerate(("AAA", "BBB")):
        put_official(store, add_equity(store, symbol), bar_rows(DAYS, economic_path(N, seed=i + 1), range(N)))
    store.upsert_fno(fno_rows("AAA", range(0, N)))
    panel = load(store, today=date(2026, 11, 2))
    assert panel.dates[-1] == pd.Timestamp("2026-11-02") and panel.is_fno_pit.iloc[-1].tolist() == [1.0, 0.0]


# ======================================================================== through the feature builder
@pytest.fixture(scope="module")
def pit_world(tmp_path_factory):
    clock = FakeClock(datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc))
    store = Store.open(tmp_path_factory.mktemp("pit") / "pit.db", clock=clock)
    panel = panel_ending(n_dates=100, n_symbols=6, seed=4, with_meetings=False)  # S00..S02 are F&O stocks with rows on every day
    populate_store(store, panel)
    first_known, joined = panel.dates[20].date().isoformat(), panel.dates[60].date().isoformat()
    store.conn.execute("DELETE FROM fno_daily WHERE trade_date < ?", (first_known,))  # the database's F&O history starts late
    store.conn.execute("DELETE FROM fno_daily WHERE symbol = 'S00' AND trade_date < ?", (joined,))  # S00 joined F&O later still
    start, end = panel.dates[0].date().isoformat(), panel.dates[-1].date().isoformat()

    def u_fno(loaded, **kw):
        fs = build_feature_set(loaded, stage="pre", thresholds=THRESHOLDS, unusual_k=1.5, cutoff_hhmm="08:45", with_labels=False, **kw)
        return fs.X["u_is_fno"].unstack("symbol")

    batch = u_fno(load_panel(store, start=start, end=end))
    live_panel = load_panel(store, start=start, end=end, today=date(2026, 11, 2))
    last = slice(len(live_panel.dates) - 1, len(live_panel.dates))
    yield {"batch": batch, "live_all": u_fno(live_panel), "live_row": u_fno(live_panel, row_slice=last),
           "first_known": pd.Timestamp(first_known), "joined": pd.Timestamp(joined)}
    store.close()


def test_u_is_fno_is_unknown_before_the_earliest_fno_data(pit_world):
    batch = pit_world["batch"]
    assert batch[batch.index < pit_world["first_known"]].isna().all().all() and len(batch[batch.index < pit_world["first_known"]]) == 20
    assert batch[batch.index >= pit_world["first_known"]].notna().all().all()


def test_u_is_fno_does_not_look_ahead_to_the_current_flag(pit_world):
    batch, first_known, joined = pit_world["batch"], pit_world["first_known"], pit_world["joined"]
    known = batch[batch.index >= first_known]
    assert (known["S01"] == 1.0).all() and (known["S02"] == 1.0).all()  # F&O for the whole known history
    assert (known["S03"] == 0.0).all() and (known["S05"] == 0.0).all()  # never F&O
    before, after = known[known.index < joined]["S00"], known[known.index >= joined]["S00"]
    assert (before == 0.0).all() and len(before) == 40  # S00 is an F&O stock TODAY but was not one then (the old look-ahead: 1)
    assert (after == 1.0).all()


def test_the_live_row_equals_the_last_training_row_and_a_single_row_build(pit_world):
    batch, live_all, live_row = pit_world["batch"], pit_world["live_all"], pit_world["live_row"]
    assert live_all.index[-1] == pd.Timestamp("2026-11-02") and len(live_row) == 1
    np.testing.assert_array_equal(live_all.iloc[-1].to_numpy(), batch.iloc[-1].to_numpy())
    np.testing.assert_array_equal(live_row.iloc[0].to_numpy(), batch.iloc[-1].to_numpy())
    assert live_all.iloc[-1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_a_hand_built_panel_without_membership_keeps_the_static_flag():
    panel = make_panel(n_dates=100, n_symbols=6, seed=2, with_meetings=False)
    fs = build_feature_set(panel, stage="pre", thresholds=THRESHOLDS, unusual_k=1.5, cutoff_hhmm="08:45", with_labels=False)
    flags = fs.X["u_is_fno"].unstack("symbol")
    assert (flags.iloc[-1] == pd.Series({s: float(v) for s, v in panel.is_fno.items()})).all()
    assert (flags.nunique() == 1).all()  # constant over time, as before


# ===================================================================================== survivorship
def seed_universe(store, status="quarantined", last_position=19):
    for i, symbol in enumerate(("AAA", "BBB")):
        put_official(store, add_equity(store, symbol), bar_rows(DAYS, economic_path(N, seed=i + 1), range(N)))
    gone = add_equity(store, "QQQ", in_universe=0, status=status)
    put_official(store, gone, bar_rows(DAYS, economic_path(N, seed=7), range(last_position + 1)))


@pytest.mark.parametrize("status", ["quarantined", "suspended", "delisted"])
def test_a_training_panel_keeps_names_that_stopped_trading_up_to_their_last_bar(store, status):
    seed_universe(store, status)
    panel = load(store)
    assert panel.symbols == ["AAA", "BBB", "QQQ"]
    assert panel.c["QQQ"].iloc[:20].notna().all() and panel.c["QQQ"].iloc[20:].isna().all()  # no rows after their last bar
    assert panel.c[["AAA", "BBB"]].notna().all().all() and len(panel.dates) == N
    assert panel.sectors.keys() == set(panel.symbols) and panel.is_fno["QQQ"] == 0


def test_a_live_panel_scores_the_active_universe_only(store):
    seed_universe(store)
    assert load(store, today=date(2026, 11, 2)).symbols == ["AAA", "BBB"]


def test_the_behaviour_can_be_forced_either_way(store):
    seed_universe(store)
    assert load(store, include_inactive=False).symbols == ["AAA", "BBB"]
    assert load(store, today=date(2026, 11, 2), include_inactive=True).symbols == ["AAA", "BBB", "QQQ"]


def test_a_name_without_bars_in_the_window_is_not_added(store):
    seed_universe(store, last_position=19)  # last bar DAYS[19]
    assert load_panel(store, start=DAYS[25], end=DAYS[-1]).symbols == ["AAA", "BBB"]
    assert load_panel(store, start=DAYS[0], end=DAYS[10]).symbols == ["AAA", "BBB", "QQQ"]


def test_explicit_symbols_filter_the_inactive_names_too(store):
    seed_universe(store)
    assert load(store, symbols=["AAA", "QQQ"]).symbols == ["AAA", "QQQ"]
    assert load(store, symbols=["AAA"]).symbols == ["AAA"]


def test_an_active_name_outside_the_universe_stays_out(store):
    seed_universe(store)
    put_official(store, add_equity(store, "ZZZ", in_universe=0), bar_rows(DAYS, economic_path(N, seed=3), range(N)))
    assert load(store).symbols == ["AAA", "BBB", "QQQ"]


def test_inactive_names_do_not_hide_the_trading_dates_of_the_universe(store):
    seed_universe(store)
    panel = load(store)
    assert list(panel.dates.strftime("%Y-%m-%d")) == DAYS
