"""Instrument universe: official NSE lists with an offline seed fallback, F&O flags and point-in-time history."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.clock import get_clock, ist_date
from app.config import Settings
from app.errors import DataUnavailable
from database.store import Store
from ingestion.nse import NseArchive
from ingestion.registry import tracked

log = logging.getLogger(__name__)

LIQUIDITY_WINDOW_DAYS = 45  # calendar days of traded value that rank names when the universe has to be capped


@dataclass
class UniverseResult:
    symbols: list[str]
    source: str  # nse | seed | db
    added: int = 0
    removed: int = 0
    notes: list[str] = field(default_factory=list)


def seed_path(settings: Settings) -> Path:
    return settings.home / "data" / "seeds" / "universe_seed.csv"


def load_seed(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "symbol": row["symbol"].strip(), "name": row.get("name", ""), "industry": row.get("industry", ""),
                    "isin": row.get("isin", ""), "is_fno": row.get("is_fno") == "1",
                    "lot_size": int(row["lot_size"]) if (row.get("lot_size") or "").isdigit() else None,
                    "nifty100": row.get("nifty100") == "1", "nifty200": row.get("nifty200") == "1",
                }
            )
    return rows


def _from_nse(archive: NseArchive, store: Store) -> list[dict[str, Any]]:
    with tracked(store, "nse_lists"):
        n500 = archive.index_constituents("nifty500")
        n100 = {r["symbol"] for r in archive.index_constituents("nifty100")}
        n200 = {r["symbol"] for r in archive.index_constituents("nifty200")}
        lots = archive.fo_lot_sizes()
    return [
        {"symbol": r["symbol"], "name": r["name"], "industry": r["industry"], "isin": r["isin"],
         "is_fno": r["symbol"] in lots, "lot_size": lots.get(r["symbol"]), "nifty100": r["symbol"] in n100,
         "nifty200": r["symbol"] in n200}
        for r in n500
    ]


def select_symbols(rows: list[dict[str, Any]], settings: Settings) -> list[str]:
    if settings.universe_mode == "custom":
        return list(settings.universe_symbols)
    if settings.universe_mode == "nifty500":
        return [r["symbol"] for r in rows]
    if settings.universe_mode == "nifty200":
        return [r["symbol"] for r in rows if r["nifty200"]]
    return [r["symbol"] for r in rows if r["is_fno"] or r["nifty100"]]


def build_universe(
    store: Store, settings: Settings, archive: NseArchive | None, as_of: date,
) -> UniverseResult:
    """Refresh instruments and the active universe. Source order: NSE official lists, shipped seed, existing DB.
    ``as_of`` is the IST date the refresh is for; membership history and the liquidity ranking are anchored on it."""
    notes: list[str] = []
    rows: list[dict[str, Any]] = []
    source = "nse"
    if archive is not None and settings.nse_enabled:
        try:
            rows = _from_nse(archive, store)
        except DataUnavailable as exc:
            notes.append(f"NSE lists unavailable ({exc}); using seed")
    if not rows:
        rows = load_seed(seed_path(settings))
        source = "seed"
    if not rows:
        existing = store.list_instruments(kind="equity", in_universe=True)
        if existing:
            notes.append("no list source available; keeping the current universe")
            return UniverseResult([r["symbol"] for r in existing], "db", 0, 0, notes)
        raise DataUnavailable("nse_lists", "no universe source available (NSE lists and seed both missing)", retryable=False)

    for r in rows:
        store.upsert_instrument(
            r["symbol"], "equity", isin=r.get("isin") or None, name=r.get("name") or None, sector=r.get("industry") or None,
            industry=r.get("industry") or None, yahoo_symbol=f"{r['symbol']}.NS", is_fno=1 if r["is_fno"] else 0,
            lot_size=r.get("lot_size"),
        )
        if not r["is_fno"]:  # F&O eligibility can lapse; upsert ignores None so clear it explicitly
            store.conn.execute("UPDATE instruments SET is_fno = 0, lot_size = NULL WHERE symbol = ? AND kind = 'equity'",
                               (r["symbol"],))

    selected = select_symbols(rows, settings)
    for symbol in selected:  # custom symbols may fall outside the Nifty 500
        if store.get_instrument(symbol, "equity") is None:
            store.upsert_instrument(symbol, "equity", name=symbol, yahoo_symbol=f"{symbol}.NS")

    active = {r["symbol"] for r in store.list_instruments(kind="equity", status="active")}
    selected = [s for s in selected if s in active]
    if len(selected) > settings.universe_max_size:
        selected = _cap_by_liquidity(store, selected, settings.universe_max_size, rows, as_of)
        notes.append(f"universe capped at {settings.universe_max_size}")
    summary = store.set_universe(selected, as_of=as_of.isoformat())
    log.info("universe: %d symbols (%s) added=%d removed=%d", len(selected), source, summary["added"], summary["removed"])
    return UniverseResult(sorted(selected), source, summary["added"], summary["removed"], notes)


def _cap_by_liquidity(
    store: Store, symbols: list[str], cap: int, rows: list[dict[str, Any]], as_of: date | None = None
) -> list[str]:
    """Keep the most liquid names (mean traded value over the ``LIQUIDITY_WINDOW_DAYS`` calendar days up to ``as_of``) and fall
    back to F&O / Nifty 100 membership. ``as_of`` is an IST trading date; without one the injected clock's IST date is used.
    The SQL clock is never consulted: it ignores both the injected clock and the server timezone."""
    flags = {r["symbol"]: (r["nifty100"], r["is_fno"]) for r in rows}
    turnover: dict[str, float] = {}
    since = ((as_of or ist_date(get_clock().now())) - timedelta(days=LIQUIDITY_WINDOW_DAYS)).isoformat()
    for row in store.conn.execute(
        "SELECT i.symbol AS symbol, AVG(b.turnover) AS t FROM bars_daily b JOIN instruments i USING (instrument_id) "
        "WHERE i.kind = 'equity' AND b.turnover IS NOT NULL AND b.trade_date >= ? GROUP BY i.symbol", (since,)
    ).fetchall():
        turnover[row["symbol"]] = float(row["t"] or 0.0)
    return sorted(symbols, key=lambda s: (-turnover.get(s, 0.0), -int(flags.get(s, (False, False))[0]),
                                          -int(flags.get(s, (False, False))[1]), s))[:cap]
