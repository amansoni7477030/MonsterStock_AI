"""Split / bonus price basis: ONE consistent basis for the bars handed to feature building.

Official exchange rows (``source_rank >= 2``) are stored RAW, exactly as traded on their day. Vendor rows
(``source_rank <= 1``) are split-adjusted by the vendor when they are loaded (a split reported in a vendor response always
comes with a full vendor reload, see ``ingestion.pipeline``) and a vendor reload can never overwrite an official row. After a
split on ex-date E the official rows older than E are therefore on the OLD price scale next to NEW-scale rows: a fake +400 %
return, a 401 % true range and a year of inflated ATR / return / gap features for a 5:1 split.

``adjust_for_splits`` re-expresses exactly those official rows on the post-split basis AT READ TIME: price / ratio and
volume * ratio (turnover, delivery and trade counts are unit-free and stay). Vendor rows are already adjusted and are never
touched again, rows dated on or after the ex-date are untouched, and every (symbol, ex-date) split counts once however many
sources recorded it. Nothing is written back: the raw tradable prices and the corporate-action records stay in ``bars_daily``
and ``corporate_actions``, and the result is a pure function of the database state. It does not depend on the order in which
official rows, vendor rows and split events arrived (official rows before or after the event, a vendor reload, a late
backfill of old official days), and reading twice gives the same bars.

``ratio`` is new shares per old share (Yahoo's numerator / denominator, bonus issues included): 5:1 -> 5, a 1:2 reverse split ->
0.5. A split only counts once it has taken effect as far as the caller knows: the symbol has a bar on or after the ex-date, or
``through`` (the end of the requested window, or the live trading date) reaches it. An announced split of a stock that still
trades at the old price must not rescale its whole history.

Every reader of raw ``bars_daily`` prices that compares prices across days (returns, volatility, volume ratios) needs this basis;
``adjust_for_splits(store, store.bars_long(...))`` gives it for any symbols and date range."""

from __future__ import annotations

import logging
import math
from collections.abc import Collection

import numpy as np
import pandas as pd

from database.store import Store

log = logging.getLogger(__name__)

OFFICIAL_RANK = 2  # bars_daily.source_rank of exchange files (raw); vendor rows are 1 and below
PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "prev_close", "vwap")
VOLUME_FIELDS = ("volume",)  # turnover, delivery_pct and trades are unit-free: a split does not change them
RATIO_LIMITS = (1e-3, 1e3)  # outside this a "split" is a data error, not a corporate action
NO_DAYS = np.array([], dtype="datetime64[D]")

SplitSchedule = dict[str, tuple[np.ndarray, np.ndarray]]  # symbol -> (ex-dates ascending, ratios)


def split_factors(ex_dates: np.ndarray, ratios: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Per day, the product of the ratios of every split whose ex-date is later than it (1.0 on and after the last one).

    A raw price dated ``d`` is ``price / factor`` on the post-split basis and its volume ``volume * factor``."""
    suffix = np.append(np.cumprod(np.asarray(ratios, dtype=np.float64)[::-1])[::-1], 1.0)
    return suffix[np.searchsorted(np.asarray(ex_dates, dtype="datetime64[D]"), days.astype("datetime64[D]"), side="right")]


def _usable(ratio: object) -> bool:
    return isinstance(ratio, (int, float)) and math.isfinite(ratio) and RATIO_LIMITS[0] <= ratio <= RATIO_LIMITS[1]


def split_schedule(store: Store, symbols: Collection[str], *, after: str, through: str | None = None) -> SplitSchedule:
    """Usable splits of ``symbols`` with an ex-date later than ``after`` (ISO date): per symbol, ex-dates ascending and ratios."""
    wanted = set(symbols)
    found: dict[str, dict[str, float]] = {}
    for row in store.conn.execute(
        "SELECT i.symbol AS symbol, c.ex_date AS ex_date, c.ratio AS ratio, "
        "(SELECT MAX(b.trade_date) FROM bars_daily b WHERE b.instrument_id = c.instrument_id) AS last_bar "
        "FROM corporate_actions c JOIN instruments i ON i.instrument_id = c.instrument_id "
        "WHERE c.action_type = 'split' AND i.kind = 'equity' AND c.ex_date > ? ORDER BY i.symbol, c.ex_date, c.id", (after,),
    ).fetchall():
        symbol, ex_date = row["symbol"], str(row["ex_date"])[:10]
        if symbol not in wanted or ex_date in found.get(symbol, {}) or ex_date > max(row["last_bar"] or "", through or ""):
            continue  # not requested, already counted for this ex-date, or not yet effective
        if not _usable(row["ratio"]):
            log.warning("%s: split on %s ignored, unusable ratio %r", symbol, ex_date, row["ratio"])
            continue
        if row["ratio"] != 1.0:
            found.setdefault(symbol, {})[ex_date] = float(row["ratio"])
    return {s: (np.array(list(ev), dtype="datetime64[D]"), np.array(list(ev.values()), dtype=np.float64))
            for s, ev in found.items()}


def _official_days(store: Store, symbols: Collection[str], *, since: str, until: str) -> dict[str, np.ndarray]:
    """Days in ``[since, until)`` on which each symbol's stored bar is an official (raw) row."""
    marks = ", ".join("?" * len(symbols))
    days: dict[str, list[str]] = {}
    for row in store.conn.execute(
        "SELECT i.symbol AS symbol, b.trade_date AS trade_date FROM bars_daily b JOIN instruments i "
        f"ON i.instrument_id = b.instrument_id WHERE i.kind = 'equity' AND b.source_rank >= ? AND b.trade_date >= ? "
        f"AND b.trade_date < ? AND i.symbol IN ({marks})", (OFFICIAL_RANK, since, until, *symbols),
    ).fetchall():
        days.setdefault(row["symbol"], []).append(row["trade_date"])
    return {s: np.array(d, dtype="datetime64[D]") for s, d in days.items()}


def adjust_for_splits(store: Store, long: pd.DataFrame, *, through: str | None = None) -> pd.DataFrame:
    """Put the official rows of a ``Store.bars_long`` frame on the post-split basis. The frame is modified in place and
    returned (the caller owns it); only symbols with a usable split inside the frame's date range are looked at.

    ``through`` is the last session the caller treats as known (the end of its window, or the live scan's trading date): a
    split whose ex-date it reaches counts even though the database has no bar on or after that day yet."""
    if long.empty:
        return long
    days = long["trade_date"].to_numpy().astype("datetime64[D]")
    schedule = split_schedule(store, long["symbol"].unique(), after=str(days.min()), through=through)
    if not schedule:
        return long
    official = _official_days(store, schedule, since=str(days.min()), until=str(max(ex[-1] for ex, _ in schedule.values())))
    names = long["symbol"].to_numpy()
    candidates = np.flatnonzero(long["symbol"].isin(list(schedule)).to_numpy())
    hit_rows: list[np.ndarray] = []
    hit_factors: list[np.ndarray] = []
    for symbol, (ex_dates, ratios) in schedule.items():
        pos = candidates[names[candidates] == symbol]
        factor = split_factors(ex_dates, ratios, days[pos])
        hit = np.isin(days[pos], official.get(symbol, NO_DAYS)) & (factor != 1.0)
        hit_rows.append(pos[hit])
        hit_factors.append(factor[hit])
    rows, factors = np.concatenate(hit_rows), np.concatenate(hit_factors)
    if len(rows):  # one assignment per column for all symbols together: .iloc writes are the slow part
        for field in PRICE_FIELDS:
            if field in long.columns:
                j = long.columns.get_loc(field)
                long.iloc[rows, j] = long.iloc[rows, j].to_numpy() / factors
        for field in VOLUME_FIELDS:
            if field in long.columns:
                j = long.columns.get_loc(field)
                long.iloc[rows, j] = long.iloc[rows, j].to_numpy() * factors
    log.debug("split basis: %d official rows of %d symbols re-based", len(rows), len(schedule))
    return long


def adjusted_bars(store: Store, symbol: str, start: str | None = None, end: str | None = None, *, kind: str = "equity",
                  through: str | None = None) -> pd.DataFrame:
    """``Store.get_bars`` of one instrument with its official rows on the post-split basis: the form every per-symbol reader
    that compares prices across days (news reactions, graph learning) should use. Series that are not equities pass through."""
    instrument_id = store.instrument_id(symbol, kind)
    if instrument_id is None:
        return store.get_bars(-1, start, end)  # the empty frame, with the columns callers expect
    bars = store.get_bars(instrument_id, start, end)
    if bars.empty or kind != "equity":
        return bars
    long = bars.reset_index()
    long.insert(0, "symbol", symbol)
    adjust_for_splits(store, long, through=through or end)
    return long.drop(columns="symbol").set_index("trade_date")
