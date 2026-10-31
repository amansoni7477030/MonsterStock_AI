"""Time helpers. All persisted timestamps are UTC ISO-8601 strings ending in ``Z``; exchange-local
calendar dates are plain ``YYYY-MM-DD`` strings. Code that needs "now" takes a ``Clock`` so tests are
deterministic and never sleep."""

from __future__ import annotations

import time as _time
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")

_ISO_Z = "%Y-%m-%dT%H:%M:%SZ"


class Clock:
    """Real clock. Subclassed by ``FakeClock`` in tests."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            _time.sleep(seconds)

    def monotonic(self) -> float:
        return _time.monotonic()


class FakeClock(Clock):
    """Deterministic clock: ``sleep`` advances time instead of blocking."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock requires an aware datetime")
        self._now = start.astimezone(UTC)
        self._mono = 0.0
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if seconds > 0:
            self.advance(seconds)

    def monotonic(self) -> float:
        return self._mono

    def set(self, value: datetime) -> None:
        self._now = value.astimezone(UTC)


_default_clock: Clock = Clock()


def get_clock() -> Clock:
    return _default_clock


def set_clock(clock: Clock) -> None:
    global _default_clock
    _default_clock = clock


def utcnow() -> datetime:
    return _default_clock.now()


def now_ist() -> datetime:
    return _default_clock.now().astimezone(IST)


def to_utc(value: datetime) -> datetime:
    """Convert to aware UTC; naive values are rejected because their meaning is ambiguous."""
    if value.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    return value.astimezone(UTC)


def iso(value: datetime | None) -> str | None:
    """Canonical persisted form: UTC, second resolution, trailing ``Z``."""
    if value is None:
        return None
    return to_utc(value).strftime(_ISO_Z)


def parse_ts(value: str | datetime | int | float | None, *, assume_tz=UTC) -> datetime | None:
    """Parse many timestamp spellings to aware UTC. Returns ``None`` for empty/unparseable input.

    ``assume_tz`` is applied to naive strings (e.g. NSE stamps are IST). Numbers are epoch seconds
    (values above 1e11 are treated as milliseconds).
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=assume_tz).astimezone(UTC)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1e11:
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=assume_tz).astimezone(UTC)
    except ValueError:
        pass
    for fmt in (
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
        "%d%m%Y%H%M%S",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y%m%dT%H%M%SZ",
        "%Y%m%d%H%M%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=assume_tz).astimezone(UTC)
    return None


def ist_date(value: datetime) -> date:
    return to_utc(value).astimezone(IST).date()


def date_str(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value[:10]
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def at_ist(day: date | str, hhmm: str | time) -> datetime:
    """Aware UTC datetime for an IST wall-clock time on ``day``."""
    d = parse_date(day)
    if isinstance(hhmm, time):
        t = hhmm
    else:
        parts = [int(p) for p in str(hhmm).split(":")]
        while len(parts) < 3:
            parts.append(0)
        t = time(parts[0], parts[1], parts[2])
    return datetime.combine(d, t, tzinfo=IST).astimezone(UTC)


def local_close_utc(day: date | str, tz_name: str, close_hhmm: str) -> datetime:
    """UTC instant of a market's close on its local ``day`` (DST handled by zoneinfo)."""
    d = parse_date(day)
    hh, mm = (int(p) for p in close_hhmm.split(":")[:2])
    return datetime.combine(d, time(hh, mm), tzinfo=ZoneInfo(tz_name)).astimezone(UTC)
