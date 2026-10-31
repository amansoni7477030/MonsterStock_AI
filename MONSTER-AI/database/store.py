"""Repository facade over SQLite. All domain modules go through ``Store`` (or its ``conn`` for module-owned
tables such as news/graph) so SQL stays in one place and every write path is transactional."""

from __future__ import annotations

import gzip
import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app import jsonutil
from app.clock import Clock, get_clock, iso
from database.connection import connect, transaction
from database.migrate import apply_migrations

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

BAR_FIELDS = ("open", "high", "low", "close", "adj_close", "volume", "prev_close", "turnover", "delivery_pct", "vwap", "trades")
INSTRUMENT_ATTRS = {
    "exchange", "isin", "name", "sector", "industry", "yahoo_symbol", "is_fno", "lot_size", "status",
    "status_reason", "listed_from", "delisted_on", "in_universe",
}
FNO_FIELDS = (
    "underlying", "lot_size", "fut_close", "fut_settle", "fut_oi", "fut_oi_chg", "fut_volume", "fut_oi_all",
    "fut_oi_chg_all", "next_fut_oi", "basis_pct", "rollover_pct", "call_oi", "put_oi", "call_oi_chg", "put_oi_chg",
    "call_vol", "put_vol", "pcr_oi", "pcr_vol", "atm_strike", "atm_straddle", "expected_move_pct", "iv_atm",
    "iv_skew", "max_call_oi_strike", "max_put_oi_strike", "dte",
)


def _num(value: Any) -> float | None:
    """Coerce to a finite float or ``None`` (NaN / inf / blanks never reach the database)."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


class Store:
    def __init__(self, conn: sqlite3.Connection, clock: Clock | None = None, path: str | Path | None = None) -> None:
        self.conn = conn
        self._clock = clock
        self.path = Path(path) if path and str(path) != ":memory:" else None

    # ----------------------------------------------------------------------------------- lifecycle
    @classmethod
    def open(
        cls, path: str | Path, *, migrate: bool = True, readonly: bool = False, clock: Clock | None = None
    ) -> Store:
        conn = connect(path, readonly=readonly)
        store = cls(conn, clock, path)
        if migrate and not readonly:
            apply_migrations(conn)
        return store

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass

    def now(self) -> str:
        return iso((self._clock or get_clock()).now()) or ""

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with transaction(self.conn) as conn:
            yield conn

    # ------------------------------------------------------------------------------------------- kv
    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, str(value), self.now()),
        )

    def kv_get_json(self, key: str, default: Any = None) -> Any:
        return jsonutil.loads(self.kv_get(key), default)

    def kv_set_json(self, key: str, value: Any) -> None:
        self.kv_set(key, jsonutil.dumps(value))

    def kv_delete(self, key: str) -> None:
        self.conn.execute("DELETE FROM kv WHERE key = ?", (key,))

    # ---------------------------------------------------------------------------------------- sources
    def register_source(self, **row: Any) -> None:
        cols = ("source_id", "provider", "endpoint", "coverage", "timestamp_meaning", "update_frequency",
                "history_depth", "rate_limits", "license_notes", "failure_behavior", "backup_source", "criticality")
        values = [row.get(c) for c in cols]
        self.conn.execute(
            f"INSERT INTO sources ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            "ON CONFLICT(source_id) DO UPDATE SET "
            + ", ".join(f"{c} = excluded.{c}" for c in cols[1:]),
            values,
        )

    def set_source_enabled(self, source_id: str, enabled: bool) -> None:
        self.conn.execute("UPDATE sources SET enabled = ? WHERE source_id = ?", (1 if enabled else 0, source_id))

    def record_source_result(
        self, source_id: str, ok: bool, *, latency_ms: float | None = None, error: str | None = None,
        data_ts: str | None = None,
    ) -> None:
        now = self.now()
        self.conn.execute(
            "INSERT INTO source_health (source_id) VALUES (?) ON CONFLICT(source_id) DO NOTHING", (source_id,)
        )
        if ok:
            self.conn.execute(
                "UPDATE source_health SET last_attempt_at = ?, last_success_at = ?, consecutive_failures = 0, "
                "total_success = total_success + 1, last_latency_ms = ?, last_data_ts = COALESCE(?, last_data_ts) "
                "WHERE source_id = ?",
                (now, now, latency_ms, data_ts, source_id),
            )
        else:
            self.conn.execute(
                "UPDATE source_health SET last_attempt_at = ?, last_error = ?, last_error_at = ?, "
                "consecutive_failures = consecutive_failures + 1, total_failure = total_failure + 1, "
                "last_latency_ms = COALESCE(?, last_latency_ms) WHERE source_id = ?",
                (now, (error or "")[:300], now, latency_ms, source_id),
            )

    def source_health(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT s.source_id, s.provider, s.criticality, s.enabled, h.last_attempt_at, h.last_success_at, "
            "h.last_error, h.last_error_at, COALESCE(h.consecutive_failures, 0) AS consecutive_failures, "
            "COALESCE(h.total_success, 0) AS total_success, COALESCE(h.total_failure, 0) AS total_failure, "
            "h.last_latency_ms, h.last_data_ts FROM sources s LEFT JOIN source_health h USING (source_id) "
            "ORDER BY s.source_id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------------------------ instruments
    def get_instrument(self, symbol: str, kind: str = "equity") -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM instruments WHERE symbol = ? AND kind = ?", (symbol, kind)
        ).fetchone()

    def instrument_id(self, symbol: str, kind: str = "equity") -> int | None:
        row = self.get_instrument(symbol, kind)
        return int(row["instrument_id"]) if row else None

    def upsert_instrument(self, symbol: str, kind: str = "equity", **attrs: Any) -> int:
        unknown = set(attrs) - INSTRUMENT_ATTRS
        if unknown:
            raise ValueError(f"unknown instrument attributes: {sorted(unknown)}")
        existing = self.get_instrument(symbol, kind)
        now = self.now()
        if existing is None:
            columns = ["symbol", "kind", "updated_at"] + list(attrs)
            values = [symbol, kind, now] + list(attrs.values())
            cursor = self.conn.execute(
                f"INSERT INTO instruments ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", values
            )
            return int(cursor.lastrowid or 0)
        changes = {k: v for k, v in attrs.items() if v is not None}
        if changes:
            assignments = ", ".join(f"{k} = ?" for k in changes)
            self.conn.execute(
                f"UPDATE instruments SET {assignments}, updated_at = ? WHERE instrument_id = ?",
                [*changes.values(), now, existing["instrument_id"]],
            )
        return int(existing["instrument_id"])

    def list_instruments(
        self, *, kind: str | None = None, in_universe: bool | None = None, status: str | None = "active",
        fno: bool | None = None,
    ) -> list[sqlite3.Row]:
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if in_universe is not None:
            clauses.append("in_universe = ?")
            params.append(1 if in_universe else 0)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if fno is not None:
            clauses.append("is_fno = ?")
            params.append(1 if fno else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.conn.execute(f"SELECT * FROM instruments {where} ORDER BY symbol", params).fetchall()

    def set_instrument_status(self, symbol: str, kind: str, status: str, reason: str | None = None) -> None:
        self.conn.execute(
            "UPDATE instruments SET status = ?, status_reason = ?, updated_at = ? WHERE symbol = ? AND kind = ?",
            (status, reason, self.now(), symbol, kind),
        )

    def record_instrument_history(self, instrument_id: int, as_of: str, attrs: Mapping[str, Any]) -> bool:
        """Append a point-in-time attribute version if it differs from the latest one."""
        payload = jsonutil.dumps(attrs)
        last = self.conn.execute(
            "SELECT id, effective_from, attrs_json FROM instrument_history WHERE instrument_id = ? "
            "ORDER BY effective_from DESC LIMIT 1", (instrument_id,),
        ).fetchone()
        if last is not None and last["attrs_json"] == payload:
            return False
        with self.tx():
            if last is not None:
                if last["effective_from"] >= as_of:
                    self.conn.execute("DELETE FROM instrument_history WHERE id = ?", (last["id"],))
                else:
                    self.conn.execute("UPDATE instrument_history SET effective_to = ? WHERE id = ?", (as_of, last["id"]))
            self.conn.execute(
                "INSERT OR REPLACE INTO instrument_history (instrument_id, effective_from, attrs_json) VALUES (?, ?, ?)",
                (instrument_id, as_of, payload),
            )
        return True

    def instrument_attrs_asof(self, instrument_id: int, as_of: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT attrs_json FROM instrument_history WHERE instrument_id = ? AND effective_from <= ? "
            "AND (effective_to IS NULL OR effective_to > ?) ORDER BY effective_from DESC LIMIT 1",
            (instrument_id, as_of, as_of),
        ).fetchone()
        return jsonutil.loads(row["attrs_json"]) if row else None

    def set_universe(self, symbols: Iterable[str], *, as_of: str, kind: str = "equity") -> dict[str, int]:
        """Mark exactly ``symbols`` as the active universe and record membership history."""
        wanted = {s for s in symbols}
        current = {r["symbol"]: r for r in self.list_instruments(kind=kind, status=None)}
        added = removed = 0
        with self.tx():
            for symbol, row in current.items():
                should = symbol in wanted
                if should and not row["in_universe"]:
                    added += 1
                elif not should and row["in_universe"]:
                    removed += 1
                else:
                    continue
                self.conn.execute(
                    "UPDATE instruments SET in_universe = ?, updated_at = ? WHERE instrument_id = ?",
                    (1 if should else 0, self.now(), row["instrument_id"]),
                )
        for symbol in wanted & set(current):
            row = self.get_instrument(symbol, kind)
            if row is not None:
                self.record_instrument_history(
                    int(row["instrument_id"]), as_of,
                    {"in_universe": 1, "sector": row["sector"], "industry": row["industry"],
                     "is_fno": row["is_fno"], "lot_size": row["lot_size"], "status": row["status"]},
                )
        return {"added": added, "removed": removed, "total": len(wanted & set(current))}

    def add_corporate_action(
        self, instrument_id: int, action_type: str, ex_date: str, *, ratio: float | None = None,
        amount: float | None = None, details: str | None = None, source: str | None = None,
    ) -> bool:
        cursor = self.conn.execute(
            "INSERT INTO corporate_actions (instrument_id, action_type, ex_date, ratio, amount, details, source, first_seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (instrument_id, action_type, ex_date, ratio, amount, details, source, self.now()),
        )
        return cursor.rowcount > 0

    def corporate_actions(self, instrument_id: int, action_type: str | None = None) -> list[sqlite3.Row]:
        if action_type:
            return self.conn.execute(
                "SELECT * FROM corporate_actions WHERE instrument_id = ? AND action_type = ? ORDER BY ex_date",
                (instrument_id, action_type),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM corporate_actions WHERE instrument_id = ? ORDER BY ex_date", (instrument_id,)
        ).fetchall()

    # --------------------------------------------------------------------------------------------- bars
    def upsert_bars(
        self, instrument_id: int, rows: Sequence[Mapping[str, Any]], *, source: str, source_rank: int = 1,
        quality: str = "ok",
    ) -> int:
        """Insert/update daily bars. A lower-ranked source never overwrites a higher-ranked one (an official
        exchange file always wins over a vendor row); optional extras are merged with COALESCE."""
        if not rows:
            return 0
        now = self.now()
        sql = (
            "INSERT INTO bars_daily (instrument_id, trade_date, open, high, low, close, adj_close, volume, prev_close, "
            "turnover, delivery_pct, vwap, trades, source, source_rank, quality, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(instrument_id, trade_date) DO UPDATE SET "
            "open = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN excluded.open ELSE bars_daily.open END, "
            "high = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN excluded.high ELSE bars_daily.high END, "
            "low = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN excluded.low ELSE bars_daily.low END, "
            "close = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN excluded.close ELSE bars_daily.close END, "
            "volume = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN COALESCE(excluded.volume, bars_daily.volume) ELSE bars_daily.volume END, "
            "source = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN excluded.source ELSE bars_daily.source END, "
            "quality = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN excluded.quality ELSE bars_daily.quality END, "
            "received_at = CASE WHEN excluded.source_rank >= bars_daily.source_rank THEN excluded.received_at ELSE bars_daily.received_at END, "
            "source_rank = MAX(excluded.source_rank, bars_daily.source_rank), "
            "adj_close = COALESCE(excluded.adj_close, bars_daily.adj_close), "
            "prev_close = COALESCE(excluded.prev_close, bars_daily.prev_close), "
            "turnover = COALESCE(excluded.turnover, bars_daily.turnover), "
            "delivery_pct = COALESCE(excluded.delivery_pct, bars_daily.delivery_pct), "
            "vwap = COALESCE(excluded.vwap, bars_daily.vwap), "
            "trades = COALESCE(excluded.trades, bars_daily.trades)"
        )
        payload = []
        for row in rows:
            payload.append(
                (
                    instrument_id, str(row["trade_date"])[:10],
                    *(_num(row.get(f)) for f in ("open", "high", "low", "close", "adj_close", "volume", "prev_close",
                                                  "turnover", "delivery_pct", "vwap", "trades")),
                    source, source_rank, row.get("quality", quality), now,
                )
            )
        with self.tx():
            self.conn.executemany(sql, payload)
        return len(payload)

    def delete_vendor_bars(self, instrument_id: int) -> int:
        """Drop vendor-sourced bars (rank 1) so a re-adjusted history can be reloaded. Official rows are kept."""
        cursor = self.conn.execute(
            "DELETE FROM bars_daily WHERE instrument_id = ? AND source_rank <= 1", (instrument_id,)
        )
        return cursor.rowcount

    def latest_bar_date(self, instrument_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT MAX(trade_date) AS d FROM bars_daily WHERE instrument_id = ?", (instrument_id,)
        ).fetchone()
        return row["d"] if row and row["d"] else None

    def latest_bar_dates(self, kind: str = "equity", *, universe_only: bool = False) -> dict[str, str]:
        sql = (
            "SELECT i.symbol AS symbol, MAX(b.trade_date) AS d FROM bars_daily b JOIN instruments i USING (instrument_id) "
            "WHERE i.kind = ?" + (" AND i.in_universe = 1" if universe_only else "") + " GROUP BY i.symbol"
        )
        return {r["symbol"]: r["d"] for r in self.conn.execute(sql, (kind,)).fetchall() if r["d"]}

    def symbols_with_bar(self, trade_date: str, kind: str = "equity", *, universe_only: bool = False) -> set[str]:
        """Symbols that have a bar exactly on ``trade_date`` (a later special-session bar, e.g. Muhurat, must not hide a gap)."""
        sql = (
            "SELECT i.symbol AS symbol FROM bars_daily b JOIN instruments i USING (instrument_id) "
            "WHERE b.trade_date = ? AND i.kind = ? AND i.status = 'active'" + (" AND i.in_universe = 1" if universe_only else "")
        )
        return {r["symbol"] for r in self.conn.execute(sql, (trade_date, kind)).fetchall()}

    def get_bars(self, instrument_id: int, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        import pandas as pd

        clauses, params = ["instrument_id = ?"], [instrument_id]
        if start:
            clauses.append("trade_date >= ?")
            params.append(start)
        if end:
            clauses.append("trade_date <= ?")
            params.append(end)
        rows = self.conn.execute(
            f"SELECT trade_date, {', '.join(BAR_FIELDS)}, source, source_rank, quality FROM bars_daily "
            f"WHERE {' AND '.join(clauses)} ORDER BY trade_date",
            params,
        ).fetchall()
        frame = pd.DataFrame([dict(r) for r in rows])
        if frame.empty:
            return pd.DataFrame(columns=["trade_date", *BAR_FIELDS, "source", "source_rank", "quality"]).set_index("trade_date")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame[list(BAR_FIELDS)] = frame[list(BAR_FIELDS)].astype("float64")  # NULL -> NaN, never object dtype
        return frame.set_index("trade_date")

    def bars_long(
        self, *, kind: str | Sequence[str] = "equity", symbols: Sequence[str] | None = None, start: str | None = None,
        end: str | None = None, universe_only: bool = False,
    ) -> pd.DataFrame:
        """Long-format bars for many instruments in one query (symbol, trade_date, OHLCV...)."""
        import pandas as pd

        kinds = [kind] if isinstance(kind, str) else list(kind)
        clauses = [f"i.kind IN ({', '.join('?' * len(kinds))})"]
        params: list[Any] = list(kinds)
        if universe_only:
            clauses.append("i.in_universe = 1")
        if symbols is not None:
            if not symbols:
                return pd.DataFrame()
            clauses.append(f"i.symbol IN ({', '.join('?' * len(symbols))})")
            params.extend(symbols)
        if start:
            clauses.append("b.trade_date >= ?")
            params.append(start)
        if end:
            clauses.append("b.trade_date <= ?")
            params.append(end)
        cols = ", ".join(f"b.{c}" for c in BAR_FIELDS)
        cursor = self.conn.execute(
            f"SELECT i.symbol AS symbol, b.trade_date AS trade_date, {cols}, b.quality AS quality FROM bars_daily b "
            f"JOIN instruments i USING (instrument_id) WHERE {' AND '.join(clauses)} ORDER BY i.symbol, b.trade_date",
            params,
        )
        names = ["symbol", "trade_date", *BAR_FIELDS, "quality"]
        parts = []
        while True:  # chunked: a list of 500k dict rows would cost ~400 MB on the 1 GB server
            chunk = cursor.fetchmany(100_000)
            if not chunk:
                break
            parts.append(pd.DataFrame.from_records([tuple(r) for r in chunk], columns=names))
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not frame.empty:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame[list(BAR_FIELDS)] = frame[list(BAR_FIELDS)].astype("float64")
        return frame

    # ---------------------------------------------------------------------------------------------- F&O
    def upsert_fno(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        cols = ("symbol", "trade_date", *FNO_FIELDS, "received_at")
        sql = (
            f"INSERT INTO fno_daily ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            "ON CONFLICT(symbol, trade_date) DO UPDATE SET "
            + ", ".join(f"{c} = excluded.{c}" for c in cols[2:])
        )
        now = self.now()
        payload = [
            (str(r["symbol"]), str(r["trade_date"])[:10], *(_num(r.get(f)) for f in FNO_FIELDS), now) for r in rows
        ]
        with self.tx():
            self.conn.executemany(sql, payload)
        return len(payload)

    def fno_frame(self, *, symbols: Sequence[str] | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        import pandas as pd

        clauses, params = [], []
        if symbols is not None:
            if not symbols:
                return pd.DataFrame()
            clauses.append(f"symbol IN ({', '.join('?' * len(symbols))})")
            params.extend(symbols)
        if start:
            clauses.append("trade_date >= ?")
            params.append(start)
        if end:
            clauses.append("trade_date <= ?")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM fno_daily {where} ORDER BY symbol, trade_date", params).fetchall()
        frame = pd.DataFrame([dict(r) for r in rows])
        if not frame.empty:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"])
            frame[list(FNO_FIELDS)] = frame[list(FNO_FIELDS)].astype("float64")
        return frame

    def latest_fno_date(self) -> str | None:
        row = self.conn.execute("SELECT MAX(trade_date) AS d FROM fno_daily").fetchone()
        return row["d"] if row and row["d"] else None

    # ----------------------------------------------------------------------------- live observations
    def add_preopen_quotes(self, trade_date: str, rows: Sequence[Mapping[str, Any]], source_ts: str | None) -> int:
        if not rows:
            return 0
        now = self.now()
        payload = [
            (trade_date, r["symbol"], source_ts, now, _num(r.get("iep")), _num(r.get("final_price")), _num(r.get("prev_close")),
             _num(r.get("final_qty")), _num(r.get("buy_qty")), _num(r.get("sell_qty")), _num(r.get("pchange")))
            for r in rows
        ]
        with self.tx():
            self.conn.executemany(
                "INSERT INTO preopen_quotes (trade_date, symbol, source_ts, received_at, iep, final_price, prev_close, "
                "final_qty, buy_qty, sell_qty, pchange) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", payload,
            )
        return len(payload)

    def latest_preopen(self, trade_date: str, *, before: str | None = None) -> dict[str, dict[str, Any]]:
        """Latest pre-open observation per symbol received at or before ``before`` (point-in-time)."""
        params: list[Any] = [trade_date]
        extra = ""
        if before:
            extra = " AND received_at <= ?"
            params.append(before)
        rows = self.conn.execute(
            "SELECT p.* FROM preopen_quotes p JOIN ("
            f"SELECT symbol, MAX(received_at) AS r FROM preopen_quotes WHERE trade_date = ?{extra} GROUP BY symbol"
            ") m ON p.symbol = m.symbol AND p.received_at = m.r WHERE p.trade_date = ?",
            [*params, trade_date],
        ).fetchall()
        return {r["symbol"]: dict(r) for r in rows}

    def add_live_quote(
        self, key: str, *, price: float | None, prev_close: float | None = None, change_pct: float | None = None,
        source_ts: str | None = None, extra: Mapping[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO live_quotes (key, source_ts, received_at, price, prev_close, change_pct, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, source_ts, self.now(), _num(price), _num(prev_close), _num(change_pct),
             jsonutil.dumps(extra) if extra else None),
        )

    def latest_live_quote(self, key: str, *, before: str | None = None) -> dict[str, Any] | None:
        params: list[Any] = [key]
        extra = ""
        if before:
            extra = " AND received_at <= ?"
            params.append(before)
        row = self.conn.execute(
            f"SELECT * FROM live_quotes WHERE key = ?{extra} ORDER BY received_at DESC, id DESC LIMIT 1", params
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------------------------------ calendar
    def upsert_calendar_days(self, rows: Sequence[Mapping[str, Any]], *, source: str) -> int:
        if not rows:
            return 0
        now = self.now()
        with self.tx():
            self.conn.executemany(
                "INSERT INTO calendar_days (trade_date, is_trading_day, session_kind, open_time, close_time, description, "
                "source, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(trade_date) DO UPDATE SET "
                "is_trading_day = excluded.is_trading_day, session_kind = excluded.session_kind, "
                "open_time = excluded.open_time, close_time = excluded.close_time, description = excluded.description, "
                "source = excluded.source, updated_at = excluded.updated_at",
                [
                    (str(r["trade_date"])[:10], 1 if r["is_trading_day"] else 0, r["session_kind"], r.get("open_time"),
                     r.get("close_time"), r.get("description"), source, now)
                    for r in rows
                ],
            )
        return len(rows)

    def get_calendar_day(self, trade_date: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM calendar_days WHERE trade_date = ?", (trade_date,)).fetchone()

    def calendar_between(self, start: str, end: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM calendar_days WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date", (start, end)
        ).fetchall()

    # ---------------------------------------------------------------------------------- board meetings
    def upsert_board_meetings(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        now = self.now()
        with self.tx():
            self.conn.executemany(
                "INSERT INTO board_meetings (symbol, meeting_date, purpose, description, is_results, first_seen_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(symbol, meeting_date, purpose) DO UPDATE SET "
                "description = excluded.description, is_results = excluded.is_results",
                [
                    (r["symbol"], str(r["meeting_date"])[:10], r.get("purpose") or "", r.get("description"),
                     1 if r.get("is_results") else 0, now)
                    for r in rows
                ],
            )
        return len(rows)

    def board_meetings_between(self, start: str, end: str, symbols: Sequence[str] | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM board_meetings WHERE meeting_date BETWEEN ? AND ?"
        params: list[Any] = [start, end]
        if symbols:
            sql += f" AND symbol IN ({', '.join('?' * len(symbols))})"
            params.extend(symbols)
        return self.conn.execute(sql + " ORDER BY meeting_date, symbol", params).fetchall()

    # --------------------------------------------------------------------------------- backfill progress
    def get_backfill(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM backfill_progress WHERE key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def set_backfill(self, key: str, status: str, *, last_date: str | None = None, error: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO backfill_progress (key, last_date, status, attempts, last_error, updated_at) "
            "VALUES (?, ?, ?, 1, ?, ?) ON CONFLICT(key) DO UPDATE SET "
            "last_date = COALESCE(excluded.last_date, backfill_progress.last_date), status = excluded.status, "
            "attempts = backfill_progress.attempts + 1, last_error = excluded.last_error, updated_at = excluded.updated_at",
            (key, last_date, status, (error or "")[:300] or None, self.now()),
        )

    # ------------------------------------------------------------------------- snapshots / predictions
    def insert_snapshot(
        self, *, trade_date: str, stage: str, cutoff_ts: str, feature_version: str, model_version: str | None,
        code_version: str | None, payload: Mapping[str, Any], data_quality: Mapping[str, Any],
    ) -> int:
        """Persist an immutable, gzip-compressed snapshot. Identical content is idempotent (same id returned)."""
        digest = jsonutil.content_hash(
            {"d": trade_date, "s": stage, "c": cutoff_ts, "f": feature_version, "m": model_version,
             "p": payload, "q": data_quality}
        )
        existing = self.conn.execute(
            "SELECT snapshot_id FROM snapshots WHERE trade_date = ? AND stage = ? AND content_hash = ?",
            (trade_date, stage, digest),
        ).fetchone()
        if existing:
            return int(existing["snapshot_id"])
        blob = gzip.compress(jsonutil.dumps(payload).encode("utf-8"), compresslevel=6)
        cursor = self.conn.execute(
            "INSERT INTO snapshots (trade_date, stage, cutoff_ts, created_at, feature_version, model_version, code_version, "
            "payload, data_quality_json, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (trade_date, stage, cutoff_ts, self.now(), feature_version, model_version, code_version, blob,
             jsonutil.dumps(data_quality), digest),
        )
        return int(cursor.lastrowid or 0)

    def get_snapshot(self, snapshot_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        return self._snapshot_dict(row) if row else None

    def latest_snapshot(self, trade_date: str, stage: str | None = None) -> dict[str, Any] | None:
        if stage:
            row = self.conn.execute(
                "SELECT * FROM snapshots WHERE trade_date = ? AND stage = ? ORDER BY snapshot_id DESC LIMIT 1",
                (trade_date, stage),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM snapshots WHERE trade_date = ? ORDER BY snapshot_id DESC LIMIT 1", (trade_date,)
            ).fetchone()
        return self._snapshot_dict(row) if row else None

    @staticmethod
    def _snapshot_dict(row: sqlite3.Row) -> dict[str, Any]:
        out = {k: row[k] for k in row.keys() if k != "payload"}
        out["payload"] = jsonutil.loads(gzip.decompress(row["payload"]))
        out["data_quality"] = jsonutil.loads(row["data_quality_json"], {})
        return out

    def snapshot_days(self, stage: str = "final") -> int:
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT trade_date) AS n FROM snapshots WHERE stage = ?", (stage,)
        ).fetchone()
        return int(row["n"]) if row else 0

    def insert_predictions(self, snapshot_id: int, trade_date: str, stage: str, preds: Sequence[Mapping[str, Any]]) -> list[int]:
        """Insert frozen predictions (immutable by trigger). Re-inserting the same snapshot is a no-op."""
        ids: list[int] = []
        now = self.now()
        with self.tx():
            for p in preds:
                cursor = self.conn.execute(
                    "INSERT INTO predictions (snapshot_id, trade_date, stage, symbol, rank, tier, model_version, model_status, "
                    "p_json, range_json, explanation_json, uncertainty, data_quality, liquidity_json, features_hash, "
                    "is_baseline, baseline_name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(snapshot_id, symbol, is_baseline, baseline_name) DO NOTHING",
                    (snapshot_id, trade_date, stage, p["symbol"], int(p["rank"]), p["tier"], p["model_version"],
                     p.get("model_status", "shadow"), jsonutil.dumps(p.get("p", {})), jsonutil.dumps(p.get("range", {})),
                     jsonutil.dumps(p.get("explanation", {})), _num(p.get("uncertainty")), p.get("data_quality", "ok"),
                     jsonutil.dumps(p.get("liquidity", {})), p.get("features_hash"), 1 if p.get("is_baseline") else 0,
                     p.get("baseline_name") or "", now),
                )
                if cursor.rowcount:
                    ids.append(int(cursor.lastrowid or 0))
        return ids

    def predictions_for(
        self, trade_date: str, *, stage: str | None = None, baselines: bool | None = False
    ) -> list[dict[str, Any]]:
        clauses, params = ["trade_date = ?"], [trade_date]
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if baselines is not None:
            clauses.append("is_baseline = ?")
            params.append(1 if baselines else 0)
        rows = self.conn.execute(
            f"SELECT * FROM predictions WHERE {' AND '.join(clauses)} ORDER BY stage, is_baseline, baseline_name, rank", params
        ).fetchall()
        return [self._prediction_dict(r) for r in rows]

    @staticmethod
    def _prediction_dict(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        for src, dst in (("p_json", "p"), ("range_json", "range"), ("explanation_json", "explanation"),
                         ("liquidity_json", "liquidity")):
            out[dst] = jsonutil.loads(out.pop(src), {})
        return out

    def prediction_dates(self, *, since: str | None = None, until: str | None = None) -> list[str]:
        clauses, params = [], []
        if since:
            clauses.append("trade_date >= ?")
            params.append(since)
        if until:
            clauses.append("trade_date <= ?")
            params.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT DISTINCT trade_date FROM predictions {where} ORDER BY trade_date", params
        ).fetchall()
        return [r["trade_date"] for r in rows]

    def predictions_without_validated_outcome(self, trade_date: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT p.* FROM predictions p WHERE p.trade_date = ? AND NOT EXISTS ("
            "SELECT 1 FROM outcomes o WHERE o.prediction_id = p.prediction_id AND o.validated = 1) "
            "ORDER BY p.stage, p.is_baseline, p.rank",
            (trade_date,),
        ).fetchall()
        return [self._prediction_dict(r) for r in rows]

    def upsert_outcome(self, prediction_id: int, trade_date: str, symbol: str, *, source: str, validated: bool,
                       ohlcv: Mapping[str, Any], metrics: Mapping[str, Any], hits: Mapping[str, Any] | None = None,
                       extra: Mapping[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO outcomes (prediction_id, trade_date, symbol, open, high, low, close, volume, range_pct, exc_pct, "
            "dir_ret_pct, mfe_pct, mae_pct, hit_json, extra_json, source, validated, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(prediction_id, source) DO UPDATE SET open = excluded.open, high = excluded.high, low = excluded.low, "
            "close = excluded.close, volume = excluded.volume, range_pct = excluded.range_pct, exc_pct = excluded.exc_pct, "
            "dir_ret_pct = excluded.dir_ret_pct, mfe_pct = excluded.mfe_pct, mae_pct = excluded.mae_pct, "
            "hit_json = excluded.hit_json, extra_json = excluded.extra_json, validated = excluded.validated, "
            "recorded_at = excluded.recorded_at",
            (prediction_id, trade_date, symbol, _num(ohlcv.get("open")), _num(ohlcv.get("high")), _num(ohlcv.get("low")),
             _num(ohlcv.get("close")), _num(ohlcv.get("volume")), _num(metrics.get("range_pct")),
             _num(metrics.get("exc_pct")), _num(metrics.get("dir_ret_pct")), _num(metrics.get("mfe_pct")),
             _num(metrics.get("mae_pct")), jsonutil.dumps(hits or {}), jsonutil.dumps(extra) if extra else None, source,
             1 if validated else 0, self.now()),
        )

    def evaluated_rows(self, *, since: str | None = None, until: str | None = None,
                       include_baselines: bool = True) -> list[dict[str, Any]]:
        """Predictions joined with their best outcome (validated beats provisional)."""
        clauses, params = [], []
        if since:
            clauses.append("p.trade_date >= ?")
            params.append(since)
        if until:
            clauses.append("p.trade_date <= ?")
            params.append(until)
        if not include_baselines:
            clauses.append("p.is_baseline = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            "SELECT p.*, o.range_pct, o.exc_pct, o.dir_ret_pct, o.mfe_pct, o.mae_pct, o.validated AS o_validated, "
            "o.source AS o_source, o.open AS o_open, o.high AS o_high, o.low AS o_low, o.close AS o_close "
            "FROM predictions p JOIN outcomes o ON o.prediction_id = p.prediction_id "
            "AND o.outcome_id = (SELECT o2.outcome_id FROM outcomes o2 WHERE o2.prediction_id = p.prediction_id "
            "ORDER BY o2.validated DESC, o2.outcome_id DESC LIMIT 1) "
            f"{where} ORDER BY p.trade_date, p.stage, p.is_baseline, p.rank",
            params,
        ).fetchall()
        return [self._prediction_dict(r) for r in rows]

    # ------------------------------------------------------------------------------------- eval metrics
    def insert_metrics(self, as_of_date: str, scope: str, metrics: Iterable[Mapping[str, Any]]) -> int:
        now = self.now()
        payload = [
            (as_of_date, now, scope, m.get("slice", "all"), m["name"], _num(m.get("value")), m.get("n"),
             _num(m.get("ci_low")), _num(m.get("ci_high")), jsonutil.dumps(m["extra"]) if m.get("extra") else None)
            for m in metrics
        ]
        if not payload:
            return 0
        with self.tx():
            self.conn.executemany(
                "INSERT INTO eval_metrics (as_of_date, computed_at, scope, slice, name, value, n, ci_low, ci_high, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", payload,
            )
        return len(payload)

    def latest_metrics(self, scope: str, *, name: str | None = None) -> list[dict[str, Any]]:
        row = self.conn.execute("SELECT MAX(as_of_date) AS d FROM eval_metrics WHERE scope = ?", (scope,)).fetchone()
        if not row or not row["d"]:
            return []
        params: list[Any] = [scope, row["d"]]
        extra = ""
        if name:
            extra = " AND name = ?"
            params.append(name)
        rows = self.conn.execute(
            f"SELECT * FROM eval_metrics WHERE scope = ? AND as_of_date = ?{extra} ORDER BY slice, name", params
        ).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------------------------- model registry
    def register_model(self, *, model_version: str, stage: str, status: str = "candidate", train_start: str | None = None,
                       train_end: str | None = None, feature_schema_hash: str | None = None,
                       feature_version: str | None = None, code_commit: str | None = None,
                       metrics: Mapping[str, Any] | None = None, artifact_path: str | None = None,
                       artifact_sha256: str | None = None, artifact_signature: str | None = None,
                       parent_version: str | None = None, notes: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO model_registry (model_version, stage, created_at, status, train_start, train_end, feature_schema_hash, "
            "feature_version, code_commit, metrics_json, artifact_path, artifact_sha256, artifact_signature, parent_version, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (model_version, stage, self.now(), status, train_start, train_end, feature_schema_hash, feature_version,
             code_commit, jsonutil.dumps(metrics or {}), artifact_path, artifact_sha256, artifact_signature,
             parent_version, notes),
        )

    def get_model(self, model_version: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM model_registry WHERE model_version = ?", (model_version,)).fetchone()
        return self._model_dict(row) if row else None

    @staticmethod
    def _model_dict(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["metrics"] = jsonutil.loads(out.pop("metrics_json"), {})
        return out

    def list_models(self, stage: str | None = None, *, status: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM model_registry {where} ORDER BY created_at DESC", params).fetchall()
        return [self._model_dict(r) for r in rows]

    def active_model(self, stage: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM model_registry WHERE stage = ? AND is_active = 1", (stage,)
        ).fetchone()
        return self._model_dict(row) if row else None

    def set_model_status(self, model_version: str, status: str, *, notes: str | None = None) -> None:
        approved = ", approved_at = ?" if status in {"shadow", "validated"} else ""
        params: list[Any] = [status, notes]
        if approved:
            params.append(self.now())
        params.append(model_version)
        self.conn.execute(
            f"UPDATE model_registry SET status = ?, notes = COALESCE(?, notes){approved} WHERE model_version = ?", params
        )

    def activate_model(self, model_version: str) -> None:
        """Make ``model_version`` the single active model of its stage (previous one is kept for rollback)."""
        model = self.get_model(model_version)
        if model is None:
            raise KeyError(model_version)
        with self.tx():
            self.conn.execute("UPDATE model_registry SET is_active = 0 WHERE stage = ? AND is_active = 1", (model["stage"],))
            self.conn.execute(
                "UPDATE model_registry SET is_active = 1, activated_at = ? WHERE model_version = ?",
                (self.now(), model_version),
            )

    # --------------------------------------------------------------------------------------- job ledger
    def job_start(self, job_name: str, run_key: str, *, scheduled_for: str | None = None, attempt: int = 1,
                  pid: int | None = None) -> int:
        cursor = self.conn.execute(
            "INSERT INTO job_runs (job_name, run_key, scheduled_for, started_at, status, attempt, pid) "
            "VALUES (?, ?, ?, ?, 'running', ?, ?)",
            (job_name, run_key, scheduled_for, self.now(), attempt, pid if pid is not None else os.getpid()),
        )
        return int(cursor.lastrowid or 0)

    def job_finish(self, run_id: int, status: str, *, message: str | None = None, exit_code: int | None = None,
                   peak_rss_mb: float | None = None) -> None:
        self.conn.execute(
            "UPDATE job_runs SET status = ?, finished_at = ?, message = ?, exit_code = ?, peak_rss_mb = ? WHERE run_id = ?",
            (status, self.now(), (message or "")[:500] or None, exit_code, peak_rss_mb, run_id),
        )

    def job_record(self, job_name: str, run_key: str, status: str, *, message: str | None = None,
                   scheduled_for: str | None = None, attempt: int = 1) -> int:
        """Write a terminal ledger row directly (e.g. ``skipped`` on a holiday, ``missed`` past the window)."""
        now = self.now()
        cursor = self.conn.execute(
            "INSERT INTO job_runs (job_name, run_key, scheduled_for, started_at, finished_at, status, attempt, message) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (job_name, run_key, scheduled_for, now, now, status, attempt, (message or "")[:500] or None),
        )
        return int(cursor.lastrowid or 0)

    def job_runs_for(self, job_name: str, run_key: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM job_runs WHERE job_name = ? AND run_key = ? ORDER BY run_id", (job_name, run_key)
        ).fetchall()
        return [dict(r) for r in rows]

    def job_settled(self, job_name: str, run_key: str) -> bool:
        """True when the job needs no further attempts (success, skipped or missed)."""
        row = self.conn.execute(
            "SELECT 1 FROM job_runs WHERE job_name = ? AND run_key = ? AND status IN ('success', 'skipped', 'missed') LIMIT 1",
            (job_name, run_key),
        ).fetchone()
        return row is not None

    def recent_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM job_runs ORDER BY run_id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def last_job_success(self, job_name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM job_runs WHERE job_name = ? AND status = 'success' ORDER BY run_id DESC LIMIT 1", (job_name,)
        ).fetchone()
        return dict(row) if row else None

    def running_jobs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM job_runs WHERE status = 'running'").fetchall()]

    # ------------------------------------------------------------------------------ alerts / deliveries
    def add_alert(self, level: str, code: str, message: str, *, dedupe_key: str | None = None,
                  dedupe_minutes: int = 0) -> bool:
        """Queue an operational alert. With ``dedupe_key`` the same alert is suppressed for ``dedupe_minutes``."""
        if dedupe_key and dedupe_minutes > 0:
            from datetime import timedelta

            from app.clock import parse_ts

            cutoff = iso((parse_ts(self.now()) or (self._clock or get_clock()).now()) - timedelta(minutes=dedupe_minutes))
            row = self.conn.execute(
                "SELECT 1 FROM alerts WHERE dedupe_key = ? AND created_at >= ? LIMIT 1", (dedupe_key, cutoff)
            ).fetchone()
            if row:
                return False
        self.conn.execute(
            "INSERT INTO alerts (created_at, level, code, message, dedupe_key) VALUES (?, ?, ?, ?, ?)",
            (self.now(), level, code, message[:1500], dedupe_key),
        )
        return True

    def pending_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM alerts WHERE delivered = 0 ORDER BY alert_id LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_alert_delivered(self, alert_id: int) -> None:
        self.conn.execute(
            "UPDATE alerts SET delivered = 1, delivered_at = ? WHERE alert_id = ?", (self.now(), alert_id)
        )

    def get_delivery(self, channel: str, dedupe_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM deliveries WHERE channel = ? AND dedupe_key = ?", (channel, dedupe_key)
        ).fetchone()
        return dict(row) if row else None

    def save_delivery(self, channel: str, kind: str, dedupe_key: str, status: str, *, trade_date: str | None = None,
                      attempts: int = 0, message_ids: Sequence[Any] | None = None, error: str | None = None,
                      body_hash: str | None = None) -> None:
        now = self.now()
        self.conn.execute(
            "INSERT INTO deliveries (channel, kind, trade_date, dedupe_key, status, attempts, message_ids, error, body_hash, "
            "created_at, updated_at, sent_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(channel, dedupe_key) DO UPDATE SET status = excluded.status, "
            "attempts = deliveries.attempts + excluded.attempts, "
            "message_ids = COALESCE(excluded.message_ids, deliveries.message_ids), error = excluded.error, "
            "body_hash = COALESCE(excluded.body_hash, deliveries.body_hash), updated_at = excluded.updated_at, "
            "sent_at = COALESCE(excluded.sent_at, deliveries.sent_at)",
            (channel, kind, trade_date, dedupe_key, status, attempts,
             jsonutil.dumps(list(message_ids)) if message_ids is not None else None, (error or "")[:300] or None,
             body_hash, now, now, now if status == "sent" else None),
        )

    # ---------------------------------------------------------------------------------------- maintenance
    @contextmanager
    def allow_prune(self) -> Iterator[None]:
        """Temporarily lift the immutability delete-guard for retention pruning only."""
        self.kv_set("maintenance.allow_prune", "1")
        try:
            yield
        finally:
            self.kv_set("maintenance.allow_prune", "0")

    def table_counts(self) -> dict[str, int]:
        names = [r["name"] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]
        return {n: int(self.conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]) for n in names}
