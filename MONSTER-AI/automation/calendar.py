"""NSE exchange calendar: trading days, holidays, special sessions and session times (all IST).

Sources, in increasing authority: weekends -> shipped seed (data/seeds/nse_holidays.json, taken from NSE) -> NSE's
live holiday API (refreshed weekly) -> days inferred as closed because the index has no bar. A special-session
override file (data/seeds/special_sessions.json) can re-open a weekend/holiday (e.g. a budget-day Saturday).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from app.clock import IST, UTC, at_ist
from app.config import Settings
from app.errors import DataUnavailable
from database.store import Store

log = logging.getLogger(__name__)

PREOPEN_START = "09:00:00"
PREOPEN_ORDER_END = "09:08:00"
PREOPEN_MATCH_END = "09:12:00"
OPEN_TIME = "09:15:00"
CLOSE_TIME = "15:30:00"
CLOSE_LAG_MIN = 10


@dataclass(frozen=True)
class SessionInfo:
    day: date
    is_trading: bool
    kind: str  # regular | special | holiday | weekend
    description: str | None = None
    preopen_start: datetime | None = None
    open: datetime | None = None
    close: datetime | None = None


class ExchangeCalendar:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store
        self._rows: dict[str, Any] = {}
        self._loaded = False

    # ----------------------------------------------------------------------------------------- data
    def reload(self) -> None:
        self._rows = {}
        if self.store is not None:
            for row in self.store.conn.execute("SELECT * FROM calendar_days").fetchall():
                self._rows[row["trade_date"]] = row
        self._loaded = True

    def _row(self, day: date) -> Any:
        if not self._loaded:
            self.reload()
        return self._rows.get(day.isoformat())

    # --------------------------------------------------------------------------------------- queries
    def is_trading_day(self, day: date) -> bool:
        row = self._row(day)
        if row is not None:
            return bool(row["is_trading_day"])
        return day.weekday() < 5

    def holiday_name(self, day: date) -> str | None:
        row = self._row(day)
        if row is not None and not row["is_trading_day"]:
            return row["description"] or row["session_kind"]
        if day.weekday() >= 5:
            return "weekend"
        return None

    def session(self, day: date) -> SessionInfo:
        row = self._row(day)
        trading = self.is_trading_day(day)
        if not trading:
            kind = "weekend" if day.weekday() >= 5 and row is None else (row["session_kind"] if row else "holiday")
            return SessionInfo(day, False, kind, self.holiday_name(day))
        open_hhmm = (row["open_time"] if row is not None and row["open_time"] else OPEN_TIME)
        close_hhmm = (row["close_time"] if row is not None and row["close_time"] else CLOSE_TIME)
        kind = row["session_kind"] if row is not None else "regular"
        return SessionInfo(day, True, kind, row["description"] if row is not None else None,
                           at_ist(day, PREOPEN_START), at_ist(day, open_hhmm), at_ist(day, close_hhmm))

    def previous_trading_day(self, day: date) -> date:
        cursor = day - timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(cursor):
                return cursor
            cursor -= timedelta(days=1)
        raise DataUnavailable("calendar", f"no trading day found before {day}", retryable=False)

    def next_trading_day(self, day: date) -> date:
        cursor = day + timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(cursor):
                return cursor
            cursor += timedelta(days=1)
        raise DataUnavailable("calendar", f"no trading day found after {day}", retryable=False)

    def trading_days(self, start: date, end: date) -> list[date]:
        out, cursor = [], start
        while cursor <= end:
            if self.is_trading_day(cursor):
                out.append(cursor)
            cursor += timedelta(days=1)
        return out

    def last_completed_session(self, now: datetime) -> date:
        """Most recent trading day whose close (+ publication lag) is at or before ``now``."""
        today = now.astimezone(IST).date()
        cursor = today
        for _ in range(40):
            if self.is_trading_day(cursor):
                close = self.session(cursor).close
                if close is not None and close + timedelta(minutes=CLOSE_LAG_MIN) <= now:
                    return cursor
            cursor -= timedelta(days=1)
        raise DataUnavailable("calendar", "no completed session found", retryable=False)

    def session_for_scan(self, now: datetime) -> date | None:
        """The trading day a scan at ``now`` is about, or ``None`` when the exchange is closed that day."""
        today = now.astimezone(IST).date()
        return today if self.is_trading_day(today) else None


# ---------------------------------------------------------------------------------------------- loaders
def _seed_dir(settings: Settings) -> Path:
    return settings.home / "data" / "seeds"


def seed_calendar(store: Store, settings: Settings) -> int:
    """Load the shipped holiday list and special-session overrides (idempotent, offline)."""
    rows: list[dict[str, Any]] = []
    holidays_file = _seed_dir(settings) / "nse_holidays.json"
    if holidays_file.exists():
        try:
            data = json.loads(holidays_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for item in data.get("holidays", []):
            note = " (Muhurat evening session only)" if item.get("special") else ""
            rows.append({"trade_date": item["trade_date"], "is_trading_day": False, "session_kind": "holiday",
                         "description": f"{item.get('description', 'Exchange holiday')}{note}"})
    written = 0
    existing = {r["trade_date"]: r["source"] for r in store.conn.execute("SELECT trade_date, source FROM calendar_days").fetchall()}
    fresh = [r for r in rows if existing.get(r["trade_date"]) in (None, "seed")]
    written += store.upsert_calendar_days(fresh, source="seed")

    special_file = _seed_dir(settings) / "special_sessions.json"
    if special_file.exists():
        try:
            specials = json.loads(special_file.read_text(encoding="utf-8")).get("sessions", [])
        except (OSError, ValueError):
            specials = []
        special_rows = [
            {"trade_date": s["date"], "is_trading_day": True, "session_kind": "special", "open_time": s.get("open"),
             "close_time": s.get("close"), "description": s.get("description", "Special session")}
            for s in specials if "date" in s
        ]
        written += store.upsert_calendar_days(special_rows, source="override")
    return written


def refresh_calendar(store: Store, nse: Any) -> int:
    """Refresh holidays from NSE's API; official data replaces seed/inferred rows (overrides are kept)."""
    holidays = nse.holidays()
    if not holidays:
        raise DataUnavailable("nse_api_market", "holiday list empty", retryable=True)
    existing = {r["trade_date"]: r["source"] for r in store.conn.execute("SELECT trade_date, source FROM calendar_days").fetchall()}
    rows = [
        {"trade_date": h["trade_date"], "is_trading_day": False, "session_kind": "holiday",
         "description": h["description"] + (" (Muhurat evening session only)" if h.get("special") else "")}
        for h in holidays if existing.get(h["trade_date"]) != "override"
    ]
    return store.upsert_calendar_days(rows, source="nse_api")


def infer_holidays_from_bars(store: Store, weekdays_back: int = 400) -> int:
    """Mark a weekday as closed when the Nifty index has no bar although earlier and later sessions exist."""
    iid = store.instrument_id("^NSEI", "index")
    if iid is None:
        return 0
    dates = [r["trade_date"] for r in store.conn.execute(
        "SELECT trade_date FROM bars_daily WHERE instrument_id = ? ORDER BY trade_date DESC LIMIT ?", (iid, weekdays_back)
    ).fetchall()]
    if len(dates) < 20:
        return 0
    have = set(dates)
    oldest, newest = date.fromisoformat(min(dates)), date.fromisoformat(max(dates))
    known = {r["trade_date"] for r in store.conn.execute("SELECT trade_date FROM calendar_days").fetchall()}
    rows, cursor = [], oldest
    while cursor <= newest:
        iso_day = cursor.isoformat()
        if cursor.weekday() < 5 and iso_day not in have and iso_day not in known:
            rows.append({"trade_date": iso_day, "is_trading_day": False, "session_kind": "holiday",
                         "description": "no index bar (inferred closed)"})
        cursor += timedelta(days=1)
    return store.upsert_calendar_days(rows, source="inferred")


def trading_days_before(calendar: ExchangeCalendar, end: date, count: int) -> list[date]:
    out: list[date] = []
    cursor = end
    while len(out) < count:
        if calendar.is_trading_day(cursor):
            out.append(cursor)
        cursor -= timedelta(days=1)
        if (end - cursor).days > count * 3 + 30:
            break
    return sorted(out)


def parse_hhmm(value: str) -> time:
    parts = [int(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return time(*parts[:3])


def utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC)


def holidays_between(calendar: ExchangeCalendar, start: date, end: date) -> Iterable[tuple[date, str]]:
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5 and not calendar.is_trading_day(cursor):
            yield cursor, calendar.holiday_name(cursor) or "holiday"
        cursor += timedelta(days=1)
