"""Spotting a split or bonus before the vendor history reflects it, and repairing the price basis.

The official end-of-day file states, per stock, the previous close *as the exchange sees it*: on the ex-date of a split or bonus
that is the adjusted base price, so it no longer equals the close we stored for the previous session. Whenever the two differ
by more than ``JUMP_TOL`` the stock has a corporate action the stored (vendor) history does not know about yet. Re-fetching exactly
those stocks from the vendor makes it re-adjust the history and report the split event, which ``ingest_symbol_daily`` records;
``ingestion.adjust`` then re-bases the official rows on the post-split basis. Until the vendor has caught up the stock stays
"pending" and is retried every session, for at most ``GIVE_UP_DAYS``, then reported to the operator.

Without this, a split of a stock that already has history would never be recorded: the daily flow takes the official file for
the session and does not ask the vendor about stocks it already holds.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from database.store import Store

log = logging.getLogger(__name__)

JUMP_TOL = 0.03          # |previous close (exchange) / stored previous close - 1| above this = a corporate action
GIVE_UP_DAYS = 6         # sessions the vendor gets to report the split before the operator is told
SPLIT_GRACE_DAYS = 3     # a recorded split counts as the explanation when its ex-date is this close to the detection
PENDING_PREFIX = "basis:pending:"


@dataclass(frozen=True)
class BasisBreak:
    symbol: str
    day: str              # the session whose row revealed it
    prev_close: float     # the exchange's (adjusted) previous close
    stored_close: float   # the close we hold for the previous session

    @property
    def factor(self) -> float:
        """``stored / adjusted``: about 2.0 for a 1:2 split, 1.5 for a 1:2 bonus, 0.5 for a 2:1 reverse split."""
        return self.stored_close / self.prev_close if self.prev_close else float("nan")


def official_breaks(store: Store, day: date | str, tol: float = JUMP_TOL) -> list[BasisBreak]:
    """Stocks whose official row of ``day`` disagrees with the stored close of the session before it and whose split (if any)
    the vendor has not reported: once it has, the break is explained and the stock is no longer listed."""
    when = day.isoformat() if isinstance(day, date) else str(day)[:10]
    rows = store.conn.execute(
        """
        SELECT i.symbol AS symbol, b.prev_close AS prev_close, p.close AS stored_close
        FROM bars_daily b
        JOIN instruments i ON i.instrument_id = b.instrument_id
        JOIN bars_daily p ON p.instrument_id = b.instrument_id
         AND p.trade_date = (SELECT MAX(q.trade_date) FROM bars_daily q
                             WHERE q.instrument_id = b.instrument_id AND q.trade_date < b.trade_date)
        WHERE b.trade_date = ? AND b.source_rank >= 2 AND i.kind = 'equity'
          AND b.prev_close > 0 AND p.close > 0 AND ABS(b.prev_close / p.close - 1.0) > ?
          AND NOT EXISTS (SELECT 1 FROM corporate_actions c WHERE c.instrument_id = b.instrument_id AND c.action_type = 'split'
                          AND c.ex_date >= date(b.trade_date, ?))
        ORDER BY i.symbol
        """, (when, tol, f"-{SPLIT_GRACE_DAYS} days")).fetchall()
    return [BasisBreak(r["symbol"], when, float(r["prev_close"]), float(r["stored_close"])) for r in rows]


def preopen_breaks(store: Store, trade_date: str, tol: float = JUMP_TOL) -> list[BasisBreak]:
    """The same test on the morning of ``trade_date``, from the pre-open book (whose previous close is also the exchange's):
    stocks with an ex-date today, whose stored history is still on the old basis."""
    rows = store.conn.execute(
        """
        SELECT q.symbol AS symbol, q.prev_close AS prev_close, p.close AS stored_close
        FROM preopen_quotes q
        JOIN instruments i ON i.symbol = q.symbol AND i.kind = 'equity'
        JOIN bars_daily p ON p.instrument_id = i.instrument_id
         AND p.trade_date = (SELECT MAX(x.trade_date) FROM bars_daily x
                             WHERE x.instrument_id = i.instrument_id AND x.trade_date < q.trade_date)
        WHERE q.trade_date = ? AND q.id = (SELECT MAX(l.id) FROM preopen_quotes l WHERE l.trade_date = q.trade_date
                                           AND l.symbol = q.symbol AND l.prev_close > 0)
          AND q.prev_close > 0 AND p.close > 0 AND ABS(q.prev_close / p.close - 1.0) > ?
          AND NOT EXISTS (SELECT 1 FROM corporate_actions c WHERE c.instrument_id = i.instrument_id AND c.action_type = 'split'
                          AND c.ex_date >= date(q.trade_date, ?))
        ORDER BY q.symbol
        """, (trade_date, tol, f"-{SPLIT_GRACE_DAYS} days")).fetchall()
    return [BasisBreak(r["symbol"], trade_date, float(r["prev_close"]), float(r["stored_close"])) for r in rows]


# ------------------------------------------------------------------------------------------- pending list
def pending(store: Store) -> dict[str, str]:
    """symbol -> the session on which the break was first seen, for every break not yet explained by a recorded split."""
    rows = store.conn.execute("SELECT key, value FROM kv WHERE substr(key, 1, ?) = ?",
                              (len(PENDING_PREFIX), PENDING_PREFIX)).fetchall()
    return {r["key"][len(PENDING_PREFIX):]: r["value"] for r in rows}


def mark_pending(store: Store, breaks: Iterable[BasisBreak]) -> list[str]:
    """Remember new breaks (the first sighting is kept); returns the symbols that were not pending before."""
    known = pending(store)
    fresh = []
    for item in breaks:
        if item.symbol in known:
            continue
        store.kv_set(PENDING_PREFIX + item.symbol, item.day)
        known[item.symbol] = item.day
        fresh.append(item.symbol)
        log.warning("%s: price basis break on %s (exchange previous close %.2f, stored %.2f, factor %.3f): a split or bonus the "
                    "history does not reflect yet; asking the vendor", item.symbol, item.day, item.prev_close,
                    item.stored_close, item.factor)
    return fresh


def clear(store: Store, symbol: str) -> None:
    store.kv_delete(PENDING_PREFIX + symbol)


def is_resolved(store: Store, symbol: str, since: str) -> bool:
    """True once a split is recorded whose ex-date is on or shortly before the day the break was first seen."""
    iid = store.instrument_id(symbol)
    if iid is None:
        return True  # the instrument is gone: nothing left to repair
    floor = (date.fromisoformat(since) - timedelta(days=SPLIT_GRACE_DAYS)).isoformat()
    return any(str(r["ex_date"]) >= floor for r in store.corporate_actions(iid, "split"))


def overdue(since: str, today: date) -> bool:
    return (today - date.fromisoformat(since)).days >= GIVE_UP_DAYS
