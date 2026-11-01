"""Weekly learning step for the event graph: build return frames from the stored bars and let the graph re-estimate its
expert priors (``event_graph.sensitivity``: lagged regression, out-of-sample lag choice, sign stability, Benjamini-Hochberg).

Alignment follows the estimator's convention. Driver returns are indexed by the Indian session on which the driver value was
KNOWN before the open (lag 0 = known before the session): a global bar dated D is first usable in the first session strictly
after D, and when an Indian holiday makes several bars land on one session their returns are compounded. Target returns use
their own session date. Only edges whose driver has a price series are testable; policy, monsoon, tariff... drivers simply stay
labelled expert priors."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd

from automation.calendar import ExchangeCalendar
from database.store import Store
from event_graph.repository import GraphRepository
from event_graph.sensitivity import update_graph_from_returns
from ingestion.adjust import adjusted_bars
from ingestion.series import SERIES_BY_KEY

log = logging.getLogger(__name__)

LEARNED_KEY = "graph.learned_at"
HISTORY_YEARS = 6
MIN_SECTOR_MEMBERS = 3
MAX_ABS_RETURN = 0.4  # larger one-day moves are data glitches (unadjusted splits, bad prints), not information

# graph driver node -> stored series symbol (ingestion.series)
DRIVER_SERIES: dict[str, str] = {
    "macro:usdinr": "USDINR=X", "macro:dxy": "DX-Y.NYB", "macro:brent": "BZ=F", "macro:wti": "CL=F",
    "macro:natgas": "NG=F", "macro:gold": "GC=F", "macro:silver": "SI=F", "macro:copper": "HG=F",
    "macro:us10y": "^TNX", "macro:nasdaq": "^IXIC", "macro:sox": "^SOX", "macro:us_equities": "^GSPC",
    "macro:nikkei": "^N225", "macro:hsi": "^HSI", "macro:vix": "^VIX", "macro:india_vix": "^INDIAVIX",
}
# series measured in the Indian session itself, usable as targets (global series cannot be: they trade on another clock)
INDIA_TARGET_SERIES: dict[str, str] = {"macro:india_vix": "^INDIAVIX"}


def _close_returns(store: Store, symbol: str, kind: str, start: str, end: str) -> pd.Series | None:
    bars = adjusted_bars(store, symbol, start, end, kind=kind)  # stocks on the post-split basis, series unchanged
    if bars.empty or "close" not in bars:
        return None
    close = bars["close"].where(bars["close"] > 0).dropna()
    returns = close.pct_change(fill_method=None).dropna()
    returns = returns[returns.abs() <= MAX_ABS_RETURN]
    return returns if len(returns) else None


def known_before_session(returns: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Re-index returns from their bar date to the first Indian session strictly after it, compounding bars that share one."""
    if returns.empty:
        return pd.Series(dtype=float)
    position = sessions.searchsorted(returns.index, side="right")
    keep = position < len(sessions)
    if not keep.any():
        return pd.Series(dtype=float)
    mapped = pd.Series(returns.to_numpy()[keep], index=sessions[position[keep]])
    return (1.0 + mapped).groupby(level=0).prod() - 1.0


def sector_returns(stock_returns: pd.DataFrame, members: Mapping[str, Iterable[str]]) -> pd.DataFrame:
    """Equal-weight return of each sector's member stocks (needs at least ``MIN_SECTOR_MEMBERS`` reporting names per day)."""
    columns: dict[str, pd.Series] = {}
    for sector, names in members.items():
        present = [n for n in names if n in stock_returns.columns]
        if len(present) < MIN_SECTOR_MEMBERS:
            continue
        block = stock_returns[present]
        columns[sector] = block.mean(axis=1).where(block.notna().sum(axis=1) >= MIN_SECTOR_MEMBERS)
    return pd.DataFrame(columns, index=stock_returns.index)


def _sector_members(repo: GraphRepository) -> dict[str, set[str]]:
    """Stocks under each sector node, following sector-in-sector (industry within sector) memberships."""
    direct: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {}
    stock_ids = {n.node_id for n in repo.nodes("stock")}
    for edge in repo.get_edges(relation="membership"):
        if edge.src in stock_ids:
            direct.setdefault(edge.dst, set()).add(edge.src)
        else:
            children.setdefault(edge.dst, set()).add(edge.src)

    def closure(sector: str, seen: frozenset[str]) -> set[str]:
        out = set(direct.get(sector, ()))
        for child in children.get(sector, ()):
            if child not in seen:
                out |= closure(child, seen | {sector})
        return out

    return {sector: closure(sector, frozenset()) for sector in set(direct) | set(children)}


def learn_graph(store: Store, calendar: ExchangeCalendar, as_of: date, *, years: int = HISTORY_YEARS) -> dict[str, Any]:
    """Re-estimate every testable edge from returns up to ``as_of`` and write the outcome into the graph. Idempotent."""
    repo = GraphRepository(store)
    if not repo.get_edges():
        return {"as_of": as_of.isoformat(), "tested": 0, "note": "the graph is not seeded"}
    start = (as_of - timedelta(days=int(years * 366))).isoformat()
    end = as_of.isoformat()
    sessions = pd.DatetimeIndex([pd.Timestamp(d) for d in calendar.trading_days(as_of - timedelta(days=int(years * 366) + 7),
                                                                                  as_of + timedelta(days=10))])

    drivers: dict[str, pd.Series] = {}
    for node, symbol in DRIVER_SERIES.items():
        spec = SERIES_BY_KEY.get(symbol)
        returns = _close_returns(store, symbol, spec.kind if spec else "global", start, end)
        if returns is not None:
            mapped = known_before_session(returns, sessions)
            if len(mapped):
                drivers[node] = mapped

    stocks: dict[str, pd.Series] = {}
    for row in store.list_instruments(kind="equity", in_universe=True):
        returns = _close_returns(store, row["symbol"], "equity", start, end)
        if returns is not None:
            stocks[row["symbol"]] = returns
    stock_frame = pd.DataFrame(stocks).sort_index()
    targets: dict[str, pd.Series] = {name: stock_frame[name] for name in stock_frame.columns}
    for node, symbol in INDIA_TARGET_SERIES.items():
        spec = SERIES_BY_KEY.get(symbol)
        returns = _close_returns(store, symbol, spec.kind if spec else "volatility", start, end)
        if returns is not None:
            targets[node] = returns
    if len(stock_frame.columns):
        sectors = sector_returns(stock_frame, _sector_members(repo))
        targets.update({name: sectors[name] for name in sectors.columns})

    if not drivers or not targets:
        return {"as_of": as_of.isoformat(), "tested": 0, "drivers": len(drivers), "targets": len(targets),
                "note": "no driver or target series with history"}
    summary = update_graph_from_returns(store, pd.DataFrame(drivers), pd.DataFrame(targets), as_of=as_of)
    store.kv_set(LEARNED_KEY, store.now())
    compact = {key: (len(value) if isinstance(value, list) and key != "breaks" else value) for key, value in summary.items()}
    compact.update(drivers=len(drivers), targets=len(targets))
    log.info("graph learning: %s", compact)
    return compact
