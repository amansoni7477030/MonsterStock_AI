"""Load market data from the database into the wide ``Panel`` consumed by the feature builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from automation.calendar import ExchangeCalendar
from database.store import Store
from features.builder import Panel
from features.calendar_features import next_trading_days
from ingestion.series import INDIA_MARKET, SERIES

SERIES_KINDS = ["global", "fx", "commodity", "rate", "volatility", "index"]
MASTER_MIN_COVERAGE = 0.30
# Market-wide volume relative to the trailing 20-day median below which a "session" is not a regular one.
SPECIAL_SESSION_VOLUME_RATIO = 0.30
SPECIAL_WEEKEND_VOLUME_RATIO = 0.60


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
) -> Panel:
    """Equities of the universe (or ``symbols``) on the master trading-date index.

    ``today`` appends a not-yet-traded row (all NaN except ``live_open`` = the pre-open equilibrium price), which is how
    the live scan reuses exactly the training code path."""
    rows = store.list_instruments(kind="equity", in_universe=True, status="active")
    if symbols is not None:
        wanted = set(symbols)
        rows = [r for r in store.list_instruments(kind="equity", status="active") if r["symbol"] in wanted]
    syms = sorted(r["symbol"] for r in rows)
    sectors = {r["symbol"]: r["sector"] for r in rows}
    is_fno = {r["symbol"]: int(r["is_fno"]) for r in rows}

    long = store.bars_long(kind="equity", symbols=syms, start=start, end=end)
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

    fno_syms = [s for s in syms if is_fno.get(s)]
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
    return Panel(
        dates=dates, symbols=syms, o=o, h=h, low=low, c=c, v=v, nifty=nifty, vix=vix, closes=closes, sectors=sectors,
        is_fno=is_fno, fno=fno, meetings=meetings, meetings_coverage_start=coverage, next_after_last=next_after,
        future_dates=future or [],
    )
