"""Per-symbol readers (news reactions, graph learning) see official bars on the post-split basis, like the feature loader."""

from __future__ import annotations

import pytest

from automation import graph_learning as gl
from ingestion.adjust import adjusted_bars


def bar(day: str, close: float, volume: float = 1_000.0) -> dict:
    return {"trade_date": day, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume}


@pytest.fixture()
def split_stock(store):
    """AAA traded at 1000 and split 5:1 on 6 Nov: the official rows before it are raw (old scale), the ones after are new."""
    iid = store.upsert_instrument("AAA", "equity", in_universe=1)
    days = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"]
    closes = [1000.0, 1010.0, 1005.0, 1012.0, 203.0, 205.0]  # 1012 / 5 = 202.4 -> a +0.3 % day, not a -80 % crash
    store.upsert_bars(iid, [bar(d, c) for d, c in zip(days, closes)], source="nse_bhavcopy", source_rank=2, quality="official")
    store.add_corporate_action(iid, "split", "2026-11-06", ratio=5.0, source="yahoo")
    return iid


def test_official_rows_before_the_split_are_rebased_and_the_rest_is_untouched(store, split_stock):
    frame = adjusted_bars(store, "AAA", "2026-11-02", "2026-11-09")
    assert list(frame["close"].round(2)) == [200.0, 202.0, 201.0, 202.4, 203.0, 205.0]
    assert list(frame["volume"]) == [5000.0] * 4 + [1000.0] * 2  # old-scale volume is in old shares: x ratio
    assert {"source", "source_rank", "quality"} <= set(frame.columns) and frame.index.name == "trade_date"
    raw = store.get_bars(split_stock, "2026-11-02", "2026-11-09")
    assert raw["close"].iloc[0] == 1000.0  # nothing is written back


def test_reading_twice_gives_the_same_bars_and_vendor_rows_are_never_touched(store, split_stock):
    iid = store.upsert_instrument("BBB", "equity", in_universe=1)
    store.upsert_bars(iid, [bar("2026-11-02", 200.0), bar("2026-11-09", 205.0)], source="yahoo")  # vendor: already adjusted
    store.add_corporate_action(iid, "split", "2026-11-06", ratio=5.0, source="yahoo")
    first, second = adjusted_bars(store, "AAA"), adjusted_bars(store, "AAA")
    assert first.equals(second)
    assert list(adjusted_bars(store, "BBB")["close"]) == [200.0, 205.0]


def test_series_unknown_symbols_and_empty_ranges_pass_through(store, split_stock):
    series = store.upsert_instrument("^GSPC", "global")
    store.upsert_bars(series, [bar("2026-11-02", 5000.0), bar("2026-11-09", 5100.0)], source="yahoo")
    assert list(adjusted_bars(store, "^GSPC", kind="global")["close"]) == [5000.0, 5100.0]
    assert adjusted_bars(store, "NOPE").empty and adjusted_bars(store, "AAA", "2030-01-01", "2030-02-01").empty


def test_a_split_that_has_not_taken_effect_yet_does_not_rescale_the_history(store):
    iid = store.upsert_instrument("CCC", "equity", in_universe=1)
    store.upsert_bars(iid, [bar("2026-11-02", 1000.0), bar("2026-11-03", 1010.0)], source="nse_bhavcopy", source_rank=2)
    store.add_corporate_action(iid, "split", "2026-11-20", ratio=5.0, source="yahoo")  # announced, still trading at the old price
    assert list(adjusted_bars(store, "CCC", "2026-11-02", "2026-11-03")["close"]) == [1000.0, 1010.0]


def test_graph_learning_keeps_the_split_day_instead_of_dropping_a_fake_crash(store, split_stock):
    returns = gl._close_returns(store, "AAA", "equity", "2026-11-01", "2026-11-10")
    assert len(returns) == 5 and returns.abs().max() < 0.05  # raw prices would show -80 % on the split day (dropped as a glitch)
    assert gl._close_returns(store, "GHOST", "equity", "2026-11-01", "2026-11-10") is None


def test_news_reactions_read_the_adjusted_bars(store, split_stock):
    from news_intelligence.pipeline import NewsPipeline

    class Probe(NewsPipeline):
        def __init__(self, store):  # only the bar reader is under test
            self.store = store

    frame = Probe(store)._default_bars("AAA", "2026-11-02", "2026-11-09")
    assert frame["close"].iloc[0] == pytest.approx(200.0) and frame["close"].pct_change().abs().max() < 0.05
    assert list(Probe(store)._default_bars("GHOST", None, None).columns) == ["open", "high", "low", "close", "volume"]
