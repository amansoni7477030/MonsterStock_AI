"""One price basis whatever the arrival order: official rows are stored raw, vendor rows are split-adjusted, and the panel handed
to feature building must show neither a fake crash nor a fake rally at a split (blueprint: raw prices and corporate-action
records stay in the database, the basis is applied when the data is read)."""

from __future__ import annotations

import itertools
import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from database.store import Store
from features.builder import build_feature_set
from features.loader import load_panel
from features.technical import true_range, wilder
from ingestion import pipeline as pl
from ingestion.adjust import adjust_for_splits, split_factors
from ingestion.yahoo import YahooSeries
from tests.bars_support import (
    add_equity,
    bar_rows,
    bars_snapshot,
    economic_path,
    old_scale,
    put_official,
    put_vendor,
    sessions,
)
from tests.seed_data import panel_ending, populate_store

N = 60
DAYS = sessions(N)
PATH = economic_path(N)
NOW_AFTER_CLOSE = datetime(2026, 11, 2, 11, 0, tzinfo=timezone.utc)  # 16:30 IST Monday: every session up to Monday is final


def load(store, **kw):
    return load_panel(store, start=DAYS[0], end=DAYS[-1], **kw)


def assert_on_the_economic_basis(panel, symbol="AAA", n=N):
    """The first ``n`` rows of the panel (a live panel has one more, empty, row) are the economic path."""
    for frame, key in ((panel.o, "open"), (panel.h, "high"), (panel.low, "low"), (panel.c, "close"), (panel.v, "volume")):
        np.testing.assert_allclose(frame[symbol].to_numpy()[:n], PATH[key][:n], rtol=1e-9, err_msg=key)


def assert_no_fake_move(panel, symbol="AAA"):
    """The signature of the defect: a +400 % / -80 % close-to-close return and a 400 % true range at the ex-date."""
    close = panel.c[symbol]
    assert close.pct_change().abs().max() < 0.05
    true_range_pct = true_range(panel.h[[symbol]], panel.low[[symbol]], panel.c[[symbol]])[symbol] / close
    assert true_range_pct.max() < 0.05
    assert (wilder(true_range(panel.h, panel.low, panel.c), 14) / panel.c * 100.0)[symbol].max() < 3.0


def record(store, iid, position, ratio, source="yahoo"):
    return store.add_corporate_action(iid, "split", DAYS[position], ratio=ratio, source=source)


# ------------------------------------------------------------------------------------------- the defect itself
def test_a_five_for_one_split_leaves_no_fake_crash_in_the_panel(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), old_scale(N, [(40, 5.0)])))
    raw = store.get_bars(iid)["close"]
    assert raw.pct_change().min() < -0.75  # what the stored (raw) rows alone show: the auditor's reproduction
    record(store, iid, 40, 5.0)
    panel = load(store)
    assert_on_the_economic_basis(panel)
    assert_no_fake_move(panel)


def test_a_one_for_two_reverse_split_has_no_fake_rally(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), old_scale(N, [(35, 0.5)])))
    assert store.get_bars(iid)["close"].pct_change().max() > 0.9  # raw rows alone: +100 % at the ex-date
    record(store, iid, 35, 0.5)
    panel = load(store)
    assert_on_the_economic_basis(panel)
    assert_no_fake_move(panel)


def test_two_splits_compose(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), old_scale(N, [(25, 5.0), (45, 2.0)])))
    record(store, iid, 25, 5.0)
    record(store, iid, 45, 2.0)
    panel = load(store)
    assert_on_the_economic_basis(panel)
    assert_no_fake_move(panel)


def test_a_bonus_issue_reported_as_a_split_is_a_ratio_of_numerator_to_denominator(store):
    """Yahoo reports a 1:1 bonus (one free share per share held) as numerator 2 / denominator 1."""
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), old_scale(N, [(40, 2.0)])))

    class Yahoo:
        def fetch_daily(self, symbol, range_="1mo"):
            return YahooSeries("AAA.NS", {}, bar_rows(DAYS, PATH, range(N)),
                               [{"ex_date": DAYS[40], "numerator": 2.0, "denominator": 1.0}], [])

    pl.ingest_symbol_daily(store, Yahoo(), store.get_instrument("AAA"), now=NOW_AFTER_CLOSE, years=1, force_full=True)
    assert [r["ratio"] for r in store.corporate_actions(iid, "split")] == [2.0]
    assert_on_the_economic_basis(load(store))


# ------------------------------------------------------------------ order of arrival never changes the basis
ORDERS = list(itertools.permutations(("official_old", "official_new", "event", "vendor")))


@pytest.mark.parametrize("order", ORDERS, ids=["-".join(step[-3:] for step in o) for o in ORDERS])
def test_the_basis_does_not_depend_on_the_order_official_vendor_and_split_events_arrive(store, order):
    """Official rows before or after the event, a late backfill of the old official days, a vendor reload at any point."""
    iid = add_equity(store)
    factor = old_scale(N, [(40, 5.0)])
    steps = {
        "official_old": lambda: put_official(store, iid, bar_rows(DAYS, PATH, range(40), factor)),
        "official_new": lambda: put_official(store, iid, bar_rows(DAYS, PATH, range(40, N), factor)),
        "event": lambda: record(store, iid, 40, 5.0),
        "vendor": lambda: put_vendor(store, iid, bar_rows(DAYS, PATH, range(N))),  # a reload: every day, already adjusted
    }
    for name in order:
        steps[name]()
    raw = bars_snapshot(store)
    panel = load(store)
    assert_on_the_economic_basis(panel)
    assert_no_fake_move(panel)
    assert bars_snapshot(store) == raw  # reading never rewrites the raw rows


MIXED_ORDERS = list(itertools.permutations(("official", "event", "vendor")))


@pytest.mark.parametrize("order", MIXED_ORDERS, ids=["-".join(o) for o in MIXED_ORDERS])
def test_vendor_history_older_than_the_official_file_is_not_adjusted_again(store, order):
    """Vendor rows (already on the new basis) for the first 40 sessions, raw official rows for the last 30, the split in between."""
    iid = add_equity(store)
    factor = old_scale(N, [(45, 4.0)])
    steps = {
        "official": lambda: put_official(store, iid, bar_rows(DAYS, PATH, range(30, N), factor)),
        "event": lambda: record(store, iid, 45, 4.0),
        "vendor": lambda: put_vendor(store, iid, bar_rows(DAYS, PATH, range(40))),
    }
    for name in order:
        steps[name]()
    panel = load(store)
    assert_on_the_economic_basis(panel)
    assert_no_fake_move(panel)


def test_a_split_older_than_the_official_file_changes_nothing(store):
    iid = add_equity(store)
    put_vendor(store, iid, bar_rows(DAYS, PATH, range(30)))
    put_official(store, iid, bar_rows(DAYS, PATH, range(30, N)))
    record(store, iid, 10, 3.0)  # long before the first official row
    assert_on_the_economic_basis(load(store))


def test_vendor_only_history_is_never_touched(store):
    iid = add_equity(store)
    put_vendor(store, iid, bar_rows(DAYS, PATH, range(N)))
    record(store, iid, 40, 5.0)
    assert_on_the_economic_basis(load(store))


def test_reading_is_idempotent_and_leaves_the_raw_rows_alone(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), old_scale(N, [(40, 5.0)])))
    record(store, iid, 40, 5.0)
    raw = bars_snapshot(store)
    first, second = load(store), load(store)
    pd.testing.assert_frame_equal(first.c, second.c)
    pd.testing.assert_frame_equal(first.v, second.v)
    assert_on_the_economic_basis(second)
    assert bars_snapshot(store) == raw


def test_each_ex_date_counts_once_even_when_two_sources_recorded_it(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), old_scale(N, [(40, 5.0)])))
    assert record(store, iid, 40, 5.0, source="yahoo") is True
    assert record(store, iid, 40, 5.0, source="nse_corporate_actions") is False
    assert len(store.corporate_actions(iid, "split")) == 1
    assert_on_the_economic_basis(load(store))


def test_only_splits_rebase_the_history_dividends_and_other_actions_do_not(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N)))
    store.add_corporate_action(iid, "dividend", DAYS[40], amount=12.0, source="yahoo")
    store.add_corporate_action(iid, "bonus", DAYS[41], ratio=2.0, source="manual")  # unknown to the loader: a split row is required
    assert_on_the_economic_basis(load(store))


def test_only_the_stock_that_split_is_rebased(store):
    a, b = add_equity(store, "AAA"), add_equity(store, "BBB")
    put_official(store, a, bar_rows(DAYS, PATH, range(N), old_scale(N, [(40, 5.0)])))
    other = economic_path(N, seed=9)
    put_official(store, b, bar_rows(DAYS, other, range(N)))
    record(store, a, 40, 5.0)
    panel = load(store)
    assert_on_the_economic_basis(panel, "AAA")
    np.testing.assert_allclose(panel.c["BBB"].to_numpy(), other["close"], rtol=1e-12)


def test_a_window_that_starts_after_the_split_is_untouched(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), old_scale(N, [(20, 5.0)])))
    record(store, iid, 20, 5.0)
    panel = load_panel(store, start=DAYS[25], end=DAYS[-1])
    np.testing.assert_allclose(panel.c["AAA"].to_numpy(), PATH["close"][25:], rtol=1e-12)


@pytest.mark.parametrize("ratio", [None, 0.0, -3.0, 1e9])
def test_an_unusable_ratio_is_ignored_and_reported(store, caplog, ratio):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N)))
    record(store, iid, 40, ratio)
    with caplog.at_level(logging.WARNING, logger="ingestion.adjust"):
        panel = load(store)
    assert_on_the_economic_basis(panel)
    assert "unusable ratio" in caplog.text and "AAA" in caplog.text


def test_a_ratio_of_one_is_a_no_op(store, caplog):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N)))
    record(store, iid, 40, 1.0)
    with caplog.at_level(logging.WARNING, logger="ingestion.adjust"):
        assert_on_the_economic_basis(load(store))
    assert not caplog.records


# --------------------------------------------------------------------------- announced and live ex-dates
def test_a_split_that_has_not_taken_effect_does_not_rescale_the_history(store):
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N)))
    store.add_corporate_action(iid, "split", "2026-12-15", ratio=5.0, source="announcement")
    assert_on_the_economic_basis(load(store))
    assert_on_the_economic_basis(load(store, today=date(2026, 11, 2)))  # not even live: the ex-date is still in the future


def test_a_window_that_reaches_the_ex_date_counts_the_split_even_without_a_bar_after_it(store):
    """The latest bar is the last pre-split session (the ex-date's file is not in yet): a window that already covers the
    ex-date sees the new scale, one that ends before it does not."""
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), np.full(N, 5.0)))
    store.add_corporate_action(iid, "split", "2026-11-02", ratio=5.0, source="yahoo")
    np.testing.assert_allclose(load_panel(store, start=DAYS[0], end="2026-11-02").c["AAA"].to_numpy(), PATH["close"], rtol=1e-9)
    np.testing.assert_allclose(load_panel(store, start=DAYS[0], end=DAYS[-1]).c["AAA"].to_numpy(), PATH["close"] * 5.0, rtol=1e-9)


def test_a_split_on_the_live_session_is_applied_so_the_opening_gap_is_not_a_fake_crash(store):
    """The scan runs before the open on the ex-date: history is old-scale, the pre-open price is already new-scale."""
    iid = add_equity(store)
    put_official(store, iid, bar_rows(DAYS, PATH, range(N), np.full(N, 5.0)))
    store.add_corporate_action(iid, "split", "2026-11-02", ratio=5.0, source="yahoo")
    new_scale_open = float(PATH["close"][-1]) * 1.01
    panel = load(store, today=date(2026, 11, 2), live_open={"AAA": new_scale_open})
    assert abs(panel.o["AAA"].iloc[-1] / panel.c["AAA"].iloc[-2] - 1.0 - 0.01) < 1e-9
    np.testing.assert_allclose(panel.c["AAA"].iloc[:-1].to_numpy(), PATH["close"], rtol=1e-9)


# ----------------------------------------------------------------------------- ingestion keeps the raw rows
class Archive:
    def __init__(self, rows_by_day):
        self.rows_by_day = rows_by_day

    def bhavcopy(self, day):
        return self.rows_by_day.get(day)


def official_file(position, factor):
    row = bar_rows(DAYS, PATH, [position], factor)[0]
    row.update(symbol="AAA", series="EQ", prev_close=None, turnover=None, delivery_pct=None, vwap=None, trades=None)
    return [row]


def test_bhavcopy_stores_raw_prices_and_a_vendor_history_with_the_split_event_lines_up(store):
    """The realistic sequence: a vendor backfill that already carries the split, then the official file of an old day."""
    iid = add_equity(store)
    factor = old_scale(N, [(40, 5.0)])

    class Yahoo:
        def fetch_daily(self, symbol, range_="1mo"):
            series = YahooSeries("AAA.NS", {}, bar_rows(DAYS, PATH, range(N)), [], [])
            series.splits = [{"ex_date": DAYS[40], "numerator": 5.0, "denominator": 1.0}]
            return series

    pl.ingest_symbol_daily(store, Yahoo(), store.get_instrument("AAA"), now=NOW_AFTER_CLOSE, years=1)
    archive = Archive({date.fromisoformat(DAYS[i]): official_file(i, factor) for i in range(N)})
    for i in (50, 20, 39, 40):  # new days, then a late backfill of old days around the ex-date
        pl.ingest_bhavcopy(store, archive, date.fromisoformat(DAYS[i]))
    raw = store.get_bars(iid)
    assert raw.loc[DAYS[39], "close"] == pytest.approx(PATH["close"][39] * 5.0)  # stored raw: old scale
    assert raw.loc[DAYS[39], "source"] == "nse_bhavcopy" and raw.loc[DAYS[10], "source"] == "yahoo"
    assert_on_the_economic_basis(load(store))


def test_a_vendor_reload_after_a_split_lines_up_with_the_raw_official_rows(store):
    """Production order: vendor history loaded long before the split (old scale), the official file ingested every day around
    it (raw), then the vendor refresh sees the split, reloads the history and records the event."""
    iid = add_equity(store)
    factor = old_scale(N, [(50, 5.0)])
    put_vendor(store, iid, bar_rows(DAYS, PATH, range(30), factor))  # old scale: loaded before the split
    put_official(store, iid, bar_rows(DAYS, PATH, range(30, N), factor))  # raw: old scale before day 50, new scale from it

    class Yahoo:
        def __init__(self):
            self.calls = []

        def fetch_daily(self, symbol, range_="1mo"):
            self.calls.append(range_)
            days = range(55, N) if range_ == "5d" else range(N)  # the incremental window, then the full reload: all adjusted
            return YahooSeries("AAA.NS", {}, bar_rows(DAYS, PATH, days),
                               [{"ex_date": DAYS[50], "numerator": 5.0, "denominator": 1.0}], [])

    yahoo = Yahoo()
    pl.ingest_symbol_daily(store, yahoo, store.get_instrument("AAA"), now=NOW_AFTER_CLOSE, years=1)
    assert yahoo.calls == ["5d", "1y"] and len(store.corporate_actions(iid, "split")) == 1
    assert_on_the_economic_basis(load(store))


# ----------------------------------------------------------------------------------------------- unit level
def test_split_factors_are_the_product_of_the_later_ratios():
    ex = np.array(["2026-03-02", "2026-06-01"], dtype="datetime64[D]")
    days = np.array(["2026-02-27", "2026-03-02", "2026-05-29", "2026-06-01", "2026-09-01"], dtype="datetime64[D]")
    np.testing.assert_allclose(split_factors(ex, np.array([5.0, 0.5]), days), [2.5, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(split_factors(ex[:0], np.array([]), days), 1.0)


def test_an_empty_frame_passes_through(store):
    empty = pd.DataFrame()
    assert adjust_for_splits(store, empty) is empty


def test_same_day_ranges_are_scale_free_so_evaluation_is_unaffected():
    from backtesting.evaluate import outcome_metrics

    assert outcome_metrics(100.0, 105.0, 98.0, 103.0) == pytest.approx(outcome_metrics(20.0, 21.0, 19.6, 20.6))


# -------------------------------------------------------------------- the same features with and without a split
def test_features_of_a_stock_that_split_equal_those_of_the_unsplit_history(tmp_path, fake_clock):
    """End to end through the builder: after a 5:1 split the whole feature matrix equals the one of the same economic history
    without a split (every feature is scale free, volume features need the volume re-based too)."""
    panel = panel_ending(n_dates=150, n_symbols=6, seed=3, with_meetings=False)
    base, split = Store.open(tmp_path / "base.db", clock=fake_clock), Store.open(tmp_path / "split.db", clock=fake_clock)
    try:
        populate_store(base, panel)
        populate_store(split, panel)
        ex = panel.dates[100].date().isoformat()
        iid = split.instrument_id("S00")
        split.conn.execute(
            "UPDATE bars_daily SET open = open / 5, high = high / 5, low = low / 5, close = close / 5, volume = volume * 5 "
            "WHERE instrument_id = ? AND trade_date >= ?", (iid, ex))
        start, end = panel.dates[0].date().isoformat(), panel.dates[-1].date().isoformat()

        def features(store):
            loaded = load_panel(store, start=start, end=end)
            return build_feature_set(loaded, stage="final", thresholds=(1.0, 2.0, 3.0, 5.0), unusual_k=1.5, cutoff_hhmm="08:45").X

        broken = features(split)  # the event is not recorded yet: the defect, visible in the features
        assert broken.xs("S00", level="symbol")["t_ret1"].min() < -70.0
        split.add_corporate_action(iid, "split", ex, ratio=5.0, source="yahoo")
        fixed, reference = features(split), features(base)
        assert fixed.xs("S00", level="symbol")["t_ret1"].abs().max() < 15.0
        np.testing.assert_allclose(fixed.to_numpy(dtype=float), reference.to_numpy(dtype=float), rtol=1e-3, atol=1e-3, equal_nan=True)
    finally:
        base.close()
        split.close()
