"""Load market data from the database into the wide ``Panel`` consumed by the feature builder.

Three properties are enforced here, once, for training and live alike: ONE price basis (official rows re-based on the
post-split basis, see ``ingestion.adjust``), POINT-IN-TIME F&O membership (``LoadedPanel.is_fno_pit``) and, for training
panels, a universe that also holds the names that stopped trading or were quarantined (no survivorship bias)."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from automation.calendar import ExchangeCalendar
from database.store import Store
from features.builder import Panel
from features.calendar_features import next_trading_days
from ingestion.adjust import adjust_for_splits
from ingestion.series import INDIA_MARKET, SERIES

SERIES_KINDS = ["global", "fx", "commodity", "rate", "volatility", "index"]
MASTER_MIN_COVERAGE = 0.30
# Market-wide volume relative to the trailing 20-day median below which a "session" is not a regular one.
SPECIAL_SESSION_VOLUME_RATIO = 0.30
SPECIAL_WEEKEND_VOLUME_RATIO = 0.60


@dataclass
class LoadedPanel(Panel):
    """A ``Panel`` read from the database. ``is_fno_pit`` (dates x symbols) is the point-in-time F&O membership behind the
    ``u_is_fno`` feature: 1 once the symbol has F&O rows, 0 while F&O data exists but not for the symbol, NaN on dates before
    the earliest F&O data in the database (no information). The static ``is_fno`` flag describes today, not history."""

    is_fno_pit: pd.DataFrame | None = None


def fno_first_dates(store: Store) -> tuple[str | None, dict[str, str]]:
    """The earliest ``fno_daily`` date of the whole database and the first F&O date of every symbol that has any."""
    first = {r["symbol"]: r["first_day"] for r in store.conn.execute(
        "SELECT symbol, MIN(trade_date) AS first_day FROM fno_daily GROUP BY symbol").fetchall()}
    return (min(first.values()) if first else None), first


def fno_membership(dates: pd.DatetimeIndex, symbols: Sequence[str], earliest: str | None, first: Mapping[str, str]) -> pd.DataFrame:
    """Point-in-time F&O membership: from a symbol's first F&O row on 1, before it 0, and NaN on every date before the earliest
    F&O data of the database (nothing is known then, which is not the same as "not in F&O"). Dates after the last F&O row of the
    database (the live session) follow the same rule, so training and live agree."""
    symbols = list(symbols)
    if earliest is None:
        return pd.DataFrame(np.nan, index=dates, columns=symbols)
    day = dates.to_numpy().astype("datetime64[D]")
    joined = np.array([first.get(s, "9999-12-31") for s in symbols], dtype="datetime64[D]")
    member = (day[:, None] >= joined[None, :]).astype(np.float64)
    member[day < np.datetime64(earliest, "D")] = np.nan
    return pd.DataFrame(member, index=dates, columns=symbols)


def _inactive_with_bars(store: Store, start: str, end: str) -> list[sqlite3.Row]:
    """Equities that are no longer active (quarantined, suspended, delisted) but have bars inside ``[start, end]``."""
    return store.conn.execute(
        "SELECT * FROM instruments i WHERE i.kind = 'equity' AND i.status <> 'active' AND EXISTS ("
        "SELECT 1 FROM bars_daily b WHERE b.instrument_id = i.instrument_id AND b.trade_date BETWEEN ? AND ?) "
        "ORDER BY i.symbol", (start, end),
    ).fetchall()


def _wide_fields(long: pd.DataFrame, fields: Sequence[str], dates: pd.DatetimeIndex, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Long (symbol, date, value) rows -> one dates x symbols frame per field, by direct positional indexing.

    ``pivot_table`` would copy the long frame several times per field; this touches it once. (symbol, date) is unique."""
    if long.empty:
        return {f: pd.DataFrame(float("nan"), index=dates, columns=symbols) for f in fields}
    di = dates.get_indexer(long["trade_date"])
    si = pd.Index(symbols).get_indexer(long["symbol"])
    keep = (di >= 0) & (si >= 0)
    di, si = di[keep], si[keep]
    out: dict[str, pd.DataFrame] = {}
    for field in fields:
        values = np.full((len(dates), len(symbols)), np.nan)
        values[di, si] = long[field].to_numpy(dtype=np.float64)[keep]
        out[field] = pd.DataFrame(values, index=dates, columns=symbols)
    return out


def _special_sessions(volume: pd.DataFrame) -> np.ndarray:
    """Dates whose market-wide volume is a small fraction of normal: Muhurat and other short special sessions (and stale
    holiday bars). Their ranges and returns are not comparable with regular sessions, so they are not trading dates here."""
    base = volume.rolling(20, min_periods=5).median().shift(1)
    ratio = (volume / base.where(base > 0)).median(axis=1).to_numpy()
    weekend = volume.index.dayofweek.to_numpy() >= 5
    with np.errstate(invalid="ignore"):
        return (ratio < SPECIAL_SESSION_VOLUME_RATIO) | (weekend & (ratio < SPECIAL_WEEKEND_VOLUME_RATIO))


def load_panel(
    store: Store, *, start: str, end: str, symbols: Sequence[str] | None = None, today: date | None = None,
    live_open: Mapping[str, float] | None = None, calendar: ExchangeCalendar | None = None, fno_start: str | None = None,
    include_inactive: bool | None = None,
) -> LoadedPanel:
    """Equities of the universe (or ``symbols``) on the master trading-date index.

    ``today`` appends a not-yet-traded row (all NaN except ``live_open`` = the pre-open equilibrium price), which is how
    the live scan reuses exactly the training code path.

    Price basis: official rows are stored raw, so those older than a split's ex-date are re-based on the post-split basis
    (``ingestion.adjust``) whatever the order in which rows and split events arrived; the stored rows are never changed.

    Survivorship: a TRAINING panel (``include_inactive`` left at None while there is no ``today``) also holds the quarantined,
    suspended or delisted equities that have bars in the window; they contribute rows up to their last bar and none after.
    A live panel (``today`` given) scores the active universe only; pass ``include_inactive`` to force either behaviour.
    Names this database never ingested (delisted before it started, or never in the instrument master) stay unknown: the panel
    cannot contain what was never loaded (documented limitation). Names that merely left the universe while still active are
    not restored either."""
    wanted = None if symbols is None else set(symbols)
    rows = store.list_instruments(kind="equity", in_universe=True, status="active")
    if wanted is not None:
        rows = [r for r in store.list_instruments(kind="equity", status="active") if r["symbol"] in wanted]
    inactive = (today is None) if include_inactive is None else include_inactive
    if inactive:
        have = {r["symbol"] for r in rows}
        rows += [r for r in _inactive_with_bars(store, start, end)
                 if r["symbol"] not in have and (wanted is None or r["symbol"] in wanted)]
    syms = sorted(r["symbol"] for r in rows)
    sectors = {r["symbol"]: r["sector"] for r in rows}
    is_fno = {r["symbol"]: int(r["is_fno"]) for r in rows}

    long = store.bars_long(kind="equity", symbols=syms, start=start, end=end)
    long = adjust_for_splits(store, long, through=end if today is None else today.isoformat())
    nifty_long = store.bars_long(kind=["index", "volatility"], symbols=[INDIA_MARKET, "^INDIAVIX"], start=start, end=end)
    counts = long.groupby("trade_date")["symbol"].nunique() if not long.empty else pd.Series(dtype=int)
    stock_dates = set(counts[counts >= max(1, MASTER_MIN_COVERAGE * max(len(syms), 1))].index)
    nifty_dates = set(nifty_long[nifty_long["symbol"] == INDIA_MARKET]["trade_date"]) if not nifty_long.empty else set()
    master = sorted(stock_dates | nifty_dates)
    if today is not None and (not master or pd.Timestamp(today) > master[-1]):
        master.append(pd.Timestamp(today))
    dates = pd.DatetimeIndex(master)

    wide = _wide_fields(long, ("open", "high", "low", "close", "volume"), dates, syms)
    del long
    o, h, low, c, v = (wide[f] for f in ("open", "high", "low", "close", "volume"))
    special = _special_sessions(v)
    if special.any():
        keep = ~special
        dates = dates[keep]
        o, h, low, c, v = (frame.loc[keep] for frame in (o, h, low, c, v))
    if today is not None and live_open:
        ts = pd.Timestamp(today)
        for sym, price in live_open.items():
            if sym in o.columns and price and price > 0 and ts == dates[-1]:
                o.loc[ts, sym] = float(price)
        for frame in (h, low, c, v):
            frame.loc[pd.Timestamp(today)] = float("nan")

    def index_series(symbol: str, field: str) -> pd.Series:
        sub = nifty_long[nifty_long["symbol"] == symbol] if not nifty_long.empty else nifty_long
        if sub.empty:
            return pd.Series(float("nan"), index=dates)
        return sub.set_index("trade_date")[field].astype(float).reindex(dates)

    nifty = pd.DataFrame({f: index_series(INDIA_MARKET, f) for f in ("open", "high", "low", "close")})
    vix = index_series("^INDIAVIX", "close")

    series_keys = [s.key for s in SERIES if s.key not in (INDIA_MARKET, "^NSEBANK", "^INDIAVIX")]
    glob = store.bars_long(kind=SERIES_KINDS, symbols=series_keys, start=(pd.Timestamp(start) - pd.Timedelta(days=30)).date().isoformat(),
                           end=None if today is None else (today + timedelta(days=1)).isoformat())
    closes: dict[str, pd.Series] = {}
    if not glob.empty:
        for key, group in glob.groupby("symbol"):
            closes[key] = group.set_index("trade_date")["close"].astype(float).sort_index()

    earliest_fno, first_fno = fno_first_dates(store)
    fno_syms = [s for s in syms if is_fno.get(s) or s in first_fno]  # names that left F&O keep the rows of their F&O years
    fno = store.fno_frame(symbols=fno_syms, start=fno_start or start, end=end) if fno_syms else None

    meetings = None
    coverage = None
    mrows = store.conn.execute("SELECT symbol, meeting_date, purpose, is_results FROM board_meetings").fetchall()
    if mrows:
        meetings = pd.DataFrame([dict(r) for r in mrows])
        cov = store.kv_get("board_meetings.coverage_start")
        coverage = date.fromisoformat(cov) if cov else date.fromisoformat(str(meetings["meeting_date"].min())[:10])

    last = dates[-1].date() if len(dates) else date.fromisoformat(end)
    next_after = future = None
    if calendar is not None:
        future = next_trading_days(last, 5, calendar.is_trading_day)
        next_after = future[0]
    return LoadedPanel(
        dates=dates, symbols=syms, o=o, h=h, low=low, c=c, v=v, nifty=nifty, vix=vix, closes=closes, sectors=sectors,
        is_fno=is_fno, fno=fno, meetings=meetings, meetings_coverage_start=coverage, next_after_last=next_after,
        future_dates=future or [], is_fno_pit=fno_membership(dates, syms, earliest_fno, first_fno),
    )
