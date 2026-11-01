"""NSE exchange calendar: trading days, holidays, special sessions and session times (all IST).

The calendar is layered so it stays correct for ANY month and year, with or without network access:

1. weekends are always closed;
2. the shipped seed (data/seeds/nse_holidays.json, taken from NSE) and NSE's live holiday API (refreshed daily) are the
   OFFICIAL lists. A year counts as officially covered once it has at least ``MIN_OFFICIAL_HOLIDAYS`` entries;
3. for a year without an official list (NSE publishes next year's in December) the rule layer applies the holidays that can
   be derived with certainty - fixed-date national holidays and Good Friday - flagged ``source='rule'``, and the final scan
   confirms live that the exchange really is open (see ``automation.scan``). Rule rows disappear once the official list arrives;
4. days inferred as closed because the index has no bar, and an operator override file (data/seeds/special_sessions.json:
   ``sessions`` re-open a weekend / holiday, ``closures`` close a day at short notice) have the last word.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from app.clock import IST, UTC, at_ist, get_clock
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

OFFICIAL_SOURCES = ("seed", "nse_api")
RULE_SOURCE = "rule"
MIN_OFFICIAL_HOLIDAYS = 8  # NSE lists 14-20 trading holidays a year; fewer means a partial / unpublished list
RULE_YEARS_AHEAD = 2       # rule rows are kept for this many years beyond the current one
# Fixed-date holidays NSE observes every year when they fall on a weekday (month, day, name).
FIXED_HOLIDAYS = (
    (1, 26, "Republic Day"), (4, 14, "Dr. Baba Saheb Ambedkar Jayanti"), (5, 1, "Maharashtra Day"),
    (8, 15, "Independence Day"), (10, 2, "Mahatma Gandhi Jayanti"), (12, 25, "Christmas"),
)


def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm); Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def rule_holidays(year: int) -> list[dict[str, Any]]:
    """Weekday exchange holidays of ``year`` that need no published list: fixed-date national holidays and Good Friday.
    Lunar-calendar holidays (Holi, Diwali, Eid ...) cannot be derived and are deliberately NOT guessed."""
    found = {date(year, m, d): name for m, d, name in FIXED_HOLIDAYS}
    found[easter_sunday(year) - timedelta(days=2)] = "Good Friday"
    note = f" (rule-based: the official {year} list is not published yet)"
    return [{"trade_date": day.isoformat(), "description": name + note}
            for day, name in sorted(found.items()) if day.weekday() < 5]


@dataclass(frozen=True)
class CalendarStatus:
    official_years: tuple[int, ...]
    official_through: date | None  # last day covered by an official list (31 Dec of the newest official year)
    days_left: int | None          # days from today to ``official_through`` (negative once it lapsed)
    rule_years: tuple[int, ...]    # years that currently rely on rule-based holidays only


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
        self._official: set[int] = set()
        self._loaded = False

    # ----------------------------------------------------------------------------------------- data
    def reload(self) -> None:
        self._rows = {}
        counts: dict[int, int] = {}
        if self.store is not None:
            for row in self.store.conn.execute("SELECT * FROM calendar_days").fetchall():
                self._rows[row["trade_date"]] = row
                if row["source"] in OFFICIAL_SOURCES and not row["is_trading_day"]:
                    year = int(row["trade_date"][:4])
                    counts[year] = counts.get(year, 0) + 1
        self._official = {year for year, n in counts.items() if n >= MIN_OFFICIAL_HOLIDAYS}
        self._loaded = True

    def _row(self, day: date) -> Any:
        if not self._loaded:
            self.reload()
        return self._rows.get(day.isoformat())

    # --------------------------------------------------------------------------------------- queries
    def is_official(self, day: date) -> bool:
        """Whether an official NSE holiday list covers ``day``'s year (else holidays come from rules and live confirmation)."""
        self._row(day)  # make sure the rows are loaded
        return day.year in self._official

    def status(self, today: date) -> CalendarStatus:
        self._row(today)
        years = tuple(sorted(self._official))
        through = date(years[-1], 12, 31) if years else None
        rule_years = tuple(sorted({int(d[:4]) for d, r in self._rows.items() if r["source"] == RULE_SOURCE} - self._official))
        return CalendarStatus(years, through, (through - today).days if through else None, rule_years)

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


def seed_calendar(store: Store, settings: Settings, now: datetime | None = None) -> int:
    """Load the shipped holiday list, the operator's special sessions / closures and the rule layer (idempotent, offline).
    ``now`` (default: the clock) decides which years get rule-based holidays: this year and the next ``RULE_YEARS_AHEAD``."""
    rows: list[dict[str, Any]] = []
    holidays_file = _seed_dir(settings) / "nse_holidays.json"
    if holidays_file.exists():
        try:
            data = json.loads(holidays_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("holiday seed %s unreadable (%s: %s); relying on the NSE refresh and the rule layer", holidays_file,
                        type(exc).__name__, exc)
            data = {}
        for item in data.get("holidays", []):
            note = " (Muhurat evening session only)" if item.get("special") else ""
            rows.append({"trade_date": item["trade_date"], "is_trading_day": False, "session_kind": "holiday",
                         "description": f"{item.get('description', 'Exchange holiday')}{note}"})
    written = 0
    existing = {r["trade_date"]: r["source"] for r in store.conn.execute("SELECT trade_date, source FROM calendar_days").fetchall()}
    fresh = [r for r in rows if existing.get(r["trade_date"]) in (None, "seed", RULE_SOURCE)]
    written += store.upsert_calendar_days(fresh, source="seed")

    # operator overrides: the shipped template (replaced on upgrade) and the operator's own file in the persistent data dir
    for override_file in (_seed_dir(settings) / "special_sessions.json", settings.data_dir / "calendar_overrides.json"):
        if not override_file.exists():
            continue
        try:
            overrides = json.loads(override_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("operator calendar overrides %s unreadable (%s: %s)", override_file, type(exc).__name__, exc)
            continue
        special_rows = [
            {"trade_date": s["date"], "is_trading_day": True, "session_kind": "special", "open_time": s.get("open"),
             "close_time": s.get("close"), "description": s.get("description", "Special session")}
            for s in overrides.get("sessions", []) if "date" in s
        ]
        closure_rows = [
            {"trade_date": c["date"], "is_trading_day": False, "session_kind": "holiday",
             "description": c.get("description", "Exchange closed (operator override)")}
            for c in overrides.get("closures", []) if "date" in c
        ]
        written += store.upsert_calendar_days(special_rows + closure_rows, source="override")

    year = (now or get_clock().now()).astimezone(IST).year
    written += ensure_rule_holidays(store, range(year, year + RULE_YEARS_AHEAD + 1))
    return written


def official_holiday_count(store: Store, year: int) -> int:
    row = store.conn.execute(
        "SELECT COUNT(*) AS n FROM calendar_days WHERE substr(trade_date, 1, 4) = ? AND is_trading_day = 0 "
        "AND source IN ('seed', 'nse_api')", (str(year),)).fetchone()
    return int(row["n"])


def ensure_rule_holidays(store: Store, years: Iterable[int]) -> int:
    """Add rule-based holidays for every year in ``years`` that has no official list yet. Never overwrites an existing row
    (official, inferred or operator rows always win). Returns the rows written."""
    existing = {r["trade_date"] for r in store.conn.execute("SELECT trade_date FROM calendar_days").fetchall()}
    rows: list[dict[str, Any]] = []
    for year in years:
        if official_holiday_count(store, year) >= MIN_OFFICIAL_HOLIDAYS:
            continue
        rows += [{**r, "is_trading_day": False, "session_kind": "holiday"} for r in rule_holidays(year)
                 if r["trade_date"] not in existing]
    if rows:
        log.info("calendar: %d rule-based holidays added for %s (no official list yet)", len(rows),
                 sorted({r["trade_date"][:4] for r in rows}))
    return store.upsert_calendar_days(rows, source=RULE_SOURCE)


def refresh_calendar(store: Store, nse: Any) -> int:
    """Refresh holidays from NSE's API; official data replaces seed / rule / inferred rows (operator overrides are kept).
    Rule rows of a year that now has an official list are retired, and official rows of that year that NSE no longer lists
    (a cancelled holiday) are dropped."""
    holidays = nse.holidays()
    if not holidays:
        raise DataUnavailable("nse_api_market", "holiday list empty", retryable=True)
    existing = {r["trade_date"]: r["source"] for r in store.conn.execute("SELECT trade_date, source FROM calendar_days").fetchall()}
    rows = [
        {"trade_date": h["trade_date"], "is_trading_day": False, "session_kind": "holiday",
         "description": h["description"] + (" (Muhurat evening session only)" if h.get("special") else "")}
        for h in holidays if existing.get(h["trade_date"]) != "override"
    ]
    written = store.upsert_calendar_days(rows, source="nse_api")
    listed = {h["trade_date"] for h in holidays}
    by_year: dict[str, set[str]] = {}
    for day in listed:
        by_year.setdefault(day[:4], set()).add(day)
    today = store.now()[:10]
    with store.tx():
        for year, days in by_year.items():
            if len(days) < MIN_OFFICIAL_HOLIDAYS:
                continue  # a partial list: neither retire rule rows nor drop anything
            store.conn.execute("DELETE FROM calendar_days WHERE source = ? AND substr(trade_date, 1, 4) = ?", (RULE_SOURCE, year))
            # only FUTURE dates can be un-listed: past holidays must stay known whatever a later API answer contains
            stale = [d for d, src in existing.items()
                     if d[:4] == year and d >= today and src in ("seed", "nse_api") and d not in days]
            if stale:
                log.warning("calendar: NSE no longer lists %s as a holiday; the day is a trading day again", sorted(stale))
                store.conn.executemany("DELETE FROM calendar_days WHERE trade_date = ?", [(d,) for d in stale])
    return written


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


def utc(dt: datetime) -> datetime:
    return dt.astimezone(UTC)


def holidays_between(calendar: ExchangeCalendar, start: date, end: date) -> Iterable[tuple[date, str]]:
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5 and not calendar.is_trading_day(cursor):
            yield cursor, calendar.holiday_name(cursor) or "holiday"
        cursor += timedelta(days=1)
