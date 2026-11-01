"""Weekly graph learning: alignment of global drivers to Indian sessions, sector aggregation, and the end-to-end update of edges
from stored bars (synthetic data with a known relationship)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from automation import graph_learning as gl
from automation.calendar import ExchangeCalendar
from automation.intel import Intel
from event_graph.repository import GraphRepository
from tests.test_news_support import add_instruments

AS_OF = date(2026, 10, 30)  # a Friday
N_DAYS = 700


def idx(*days: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(days)))


# ------------------------------------------------------------------------------------------------ alignment
def test_a_global_bar_is_usable_from_the_next_indian_session_and_weekends_roll_to_monday():
    sessions = idx("2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-11-02")
    returns = pd.Series([0.01, 0.02, 0.03], index=idx("2026-10-29", "2026-10-30", "2026-11-01"))
    mapped = gl.known_before_session(returns, sessions)
    assert list(mapped.index) == list(idx("2026-10-30", "2026-11-02")) and mapped.iloc[0] == pytest.approx(0.01)
    # Friday's US bar (Oct 30) and Sunday's FX bar (Nov 1) both become known before Monday's session: compounded
    assert mapped.iloc[1] == pytest.approx(1.02 * 1.03 - 1.0)


def test_bars_after_the_last_session_are_dropped_and_empty_input_is_safe():
    sessions = idx("2026-10-29", "2026-10-30")
    assert gl.known_before_session(pd.Series([0.1], index=idx("2026-10-30")), sessions).empty
    assert gl.known_before_session(pd.Series(dtype=float), sessions).empty


def test_sector_returns_need_enough_reporting_members():
    frame = pd.DataFrame({"A": [0.01, 0.02, np.nan], "B": [0.03, np.nan, np.nan], "C": [0.02, 0.04, 0.01], "D": [0.0, 0.0, 0.0]},
                         index=idx("2026-10-28", "2026-10-29", "2026-10-30"))
    out = gl.sector_returns(frame, {"sector:x": ["A", "B", "C"], "sector:tiny": ["A", "B"], "sector:ghost": ["Z"]})
    assert list(out.columns) == ["sector:x"]  # fewer than three known members: no sector series
    assert out["sector:x"].iloc[0] == pytest.approx(0.02) and np.isnan(out["sector:x"].iloc[1]) and np.isnan(out["sector:x"].iloc[2])


# ------------------------------------------------------------------------------------------- end to end
@pytest.fixture()
def world(store, settings, fake_clock):
    """Six stocks of one sector. They follow Brent in the first session after the move (beta 0.8) and WTI one session later
    (beta 0.6); gold is unrelated noise."""
    rng = np.random.default_rng(11)
    calendar = ExchangeCalendar(store)
    sessions = [d for d in pd.bdate_range(end="2026-10-30", periods=N_DAYS) if calendar.is_trading_day(d.date())]
    stamp = pd.DatetimeIndex(sessions)

    def driver() -> pd.Series:  # bar-dated daily returns
        return pd.Series(rng.normal(0, 0.015, len(stamp)), index=stamp)

    brent_ret, wti_ret, gold_ret = driver(), driver(), driver()

    def close_rows(returns: pd.Series, base: float = 100.0):
        close = base * (1.0 + returns).cumprod()
        return [{"trade_date": d.date().isoformat(), "open": c, "high": c, "low": c, "close": c, "volume": 1000.0}
                for d, c in zip(close.index, close.to_numpy(), strict=True)]

    for symbol, series in (("BZ=F", brent_ret), ("CL=F", wti_ret), ("GC=F", gold_ret)):
        store.upsert_bars(store.upsert_instrument(symbol, "commodity"), close_rows(series), source="test", source_rank=1)
    # a driver bar dated D is known before the NEXT session: that is the estimator's lag 0
    brent_known = pd.Series(brent_ret.shift(1).to_numpy(), index=stamp).fillna(0.0)
    wti_known = pd.Series(wti_ret.shift(1).to_numpy(), index=stamp).fillna(0.0)
    members = []
    for i in range(6):
        symbol = f"STK{i}"
        members.append(symbol)
        own = 0.8 * brent_known + 0.6 * wti_known.shift(1).fillna(0.0) + pd.Series(rng.normal(0, 0.004, len(stamp)), index=stamp)
        store.upsert_bars(store.upsert_instrument(symbol, "equity", in_universe=1, status="active"), close_rows(own),
                          source="test", source_rank=1)
    repo = GraphRepository(store)
    repo.upsert_node("sector:test", "sector", "Test sector")
    for node in ("macro:brent", "macro:wti", "macro:gold"):
        repo.upsert_node(node, "commodity", node)
    for symbol in members:
        repo.upsert_node(symbol, "stock", symbol)
        repo.upsert_edge(symbol, "sector:test", "membership", direction=1, sensitivity=1.0, confidence=0.9, source="curated")
    for node, lag in (("macro:brent", 0.0), ("macro:wti", 1.0), ("macro:gold", 0.0)):  # the prior lag decides which lags are tested
        repo.upsert_edge(node, "sector:test", "exposure", direction=1, sensitivity=0.3, lag_days_mean=lag, lag_days_sd=1.0,
                         confidence=0.3, rationale=f"{node}: test prior")
    return type("World", (), {"store": store, "calendar": calendar, "repo": repo, "settings": settings, "clock": fake_clock})


def test_real_relationships_are_measured_at_their_lag_and_an_unrelated_one_decays(world):
    summary = gl.learn_graph(world.store, world.calendar, AS_OF)

    assert summary["tested"] == 3 and summary["drivers"] == 3 and summary["targets"] >= 7
    brent = world.repo.get_edge("macro:brent", "sector:test", "exposure")
    wti = world.repo.get_edge("macro:wti", "sector:test", "exposure")
    gold = world.repo.get_edge("macro:gold", "sector:test", "exposure")
    assert brent.source == "measured" and brent.evidence_count >= 250 and brent.lag_days_mean == 0.0
    assert 0.65 < brent.sensitivity < 0.95 and brent.confidence > 0.3  # moved from the 0.3 prior toward the true 0.8
    assert wti.source == "measured" and wti.lag_days_mean == 1.0 and 0.45 < wti.sensitivity < 0.75
    assert gold.source == "prior" and gold.sensitivity == pytest.approx(0.3) and gold.confidence < 0.3
    assert summary["passed"] == 2 and summary["updated"] == 2 and summary["failed_gates"] == 1
    assert world.store.kv_get(gl.LEARNED_KEY)


def test_learning_twice_on_the_same_day_changes_nothing_more(world):
    gl.learn_graph(world.store, world.calendar, AS_OF)
    once = {e.key: (e.sensitivity, e.confidence) for e in world.repo.get_edges()}
    gl.learn_graph(world.store, world.calendar, AS_OF)
    assert {e.key: (e.sensitivity, e.confidence) for e in world.repo.get_edges()} == once


def test_an_empty_or_unseeded_database_is_a_no_op(store):
    calendar = ExchangeCalendar(store)
    assert gl.learn_graph(store, calendar, AS_OF) == {"as_of": "2026-10-30", "tested": 0, "note": "the graph is not seeded"}
    repo = GraphRepository(store)
    repo.upsert_node("macro:brent", "commodity", "Brent")
    repo.upsert_node("sector:x", "sector", "X")
    repo.upsert_edge("macro:brent", "sector:x", "exposure", direction=1, sensitivity=0.3, confidence=0.3)
    result = gl.learn_graph(store, calendar, AS_OF)
    assert result["tested"] == 0 and "no driver or target series" in result["note"]


def test_nested_sectors_contain_the_stocks_of_their_children(world):
    repo = world.repo
    repo.upsert_node("sector:parent", "sector", "Parent")
    repo.upsert_edge("sector:test", "sector:parent", "membership", direction=1, sensitivity=1.0, confidence=0.9, source="curated")
    repo.upsert_edge("sector:parent", "sector:test", "membership", direction=1, sensitivity=1.0, confidence=0.9, source="curated")  # a cycle
    members = gl._sector_members(repo)
    assert members["sector:test"] == {f"STK{i}" for i in range(6)} == members["sector:parent"]


def test_glitchy_prices_are_not_returns(store):
    iid = store.upsert_instrument("GLITCH", "equity")
    rows = [{"trade_date": f"2026-10-{d:02d}", "open": p, "high": p, "low": p, "close": p, "volume": 1.0}
            for d, p in zip((26, 27, 28, 29, 30), (100.0, 101.0, 505.0, 102.0, 0.0), strict=True)]
    store.upsert_bars(iid, rows, source="test", source_rank=1)
    out = gl._close_returns(store, "GLITCH", "equity", "2026-10-01", "2026-10-31")
    assert out is not None and list(out.round(4)) == [0.01]  # the +400 % and -80 % prints and the zero close are dropped
    assert gl._close_returns(store, "MISSING", "equity", "2026-10-01", "2026-10-31") is None


# --------------------------------------------------------------------------------------- the adapter
def test_the_intel_adapter_runs_the_learning_and_never_raises(store, settings, fake_clock, monkeypatch):
    add_instruments(store)
    intel = Intel(store, settings, clock=fake_clock)
    assert intel.learn_graph(AS_OF)["tested"] >= 0

    monkeypatch.setattr(gl, "learn_graph", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert intel.learn_graph(AS_OF) == {} and "event graph learning failed (RuntimeError)" in intel.coverage_note()
    intel.graph = None
    assert intel.learn_graph(AS_OF) == {}
