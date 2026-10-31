from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.clock import (
    IST,
    UTC,
    FakeClock,
    at_ist,
    date_str,
    iso,
    ist_date,
    local_close_utc,
    parse_date,
    parse_ts,
    to_utc,
)


def test_iso_is_canonical_utc():
    assert iso(datetime(2026, 11, 2, 9, 15, 30, 999, tzinfo=IST)) == "2026-11-02T03:45:30Z"
    assert iso(None) is None
    with pytest.raises(ValueError):
        to_utc(datetime(2026, 1, 1))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-11-02T03:45:30Z", datetime(2026, 11, 2, 3, 45, 30, tzinfo=UTC)),
        ("2026-11-02T09:15:30+05:30", datetime(2026, 11, 2, 3, 45, 30, tzinfo=UTC)),
        ("31-Oct-2026 02:29", datetime(2026, 10, 30, 20, 59, tzinfo=UTC)),  # naive NSE stamp is IST
        ("31102026230716", datetime(2026, 10, 31, 17, 37, 16, tzinfo=UTC)),
        ("Mon, 02 Nov 2026 03:45:00 +0000", datetime(2026, 11, 2, 3, 45, tzinfo=UTC)),
        ("20261102T034500Z", datetime(2026, 11, 2, 3, 45, tzinfo=UTC)),
        (1793591100, datetime(2026, 11, 2, 3, 45, tzinfo=UTC)),
        (1793591100000, datetime(2026, 11, 2, 3, 45, tzinfo=UTC)),
    ],
)
def test_parse_ts_variants(text, expected):
    assert parse_ts(text, assume_tz=IST) == expected


def test_parse_ts_garbage_returns_none():
    assert parse_ts("not a date") is None
    assert parse_ts("") is None and parse_ts(None) is None
    assert parse_ts(float("inf")) is None


def test_parse_ts_naive_uses_assumed_timezone():
    assert parse_ts("2026-11-02 09:15:00", assume_tz=IST) == datetime(2026, 11, 2, 3, 45, tzinfo=UTC)
    assert parse_ts("2026-11-02 09:15:00") == datetime(2026, 11, 2, 9, 15, tzinfo=UTC)


def test_at_ist_and_ist_date():
    assert at_ist("2026-11-02", "09:12") == datetime(2026, 11, 2, 3, 42, tzinfo=UTC)
    assert at_ist(date(2026, 11, 2), "09:08:45").second == 45
    assert ist_date(datetime(2026, 11, 1, 20, 0, tzinfo=UTC)) == date(2026, 11, 2)  # 01:30 IST next day


def test_foreign_close_respects_daylight_saving():
    summer = local_close_utc("2026-07-01", "America/New_York", "16:00")  # EDT = UTC-4
    winter = local_close_utc("2026-12-01", "America/New_York", "16:00")  # EST = UTC-5
    assert summer.hour == 20 and winter.hour == 21
    tokyo = local_close_utc("2026-07-01", "Asia/Tokyo", "15:30")
    assert tokyo.hour == 6 and tokyo.minute == 30


def test_date_helpers():
    assert date_str(datetime(2026, 11, 2, 5)) == "2026-11-02"
    assert date_str("2026-11-02T10:00:00Z") == "2026-11-02"
    assert parse_date("2026-11-02") == date(2026, 11, 2)
    assert parse_date(datetime(2026, 11, 2, 1)) == date(2026, 11, 2)


def test_fake_clock_sleep_advances_time():
    clock = FakeClock(datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc))
    clock.sleep(90)
    assert clock.now() == datetime(2026, 11, 2, 3, 1, 30, tzinfo=timezone.utc)
    assert clock.sleeps == [90] and clock.monotonic() == 90
    clock.set(datetime(2026, 11, 3, tzinfo=timezone.utc))
    assert clock.now().day == 3
    with pytest.raises(ValueError):
        FakeClock(datetime(2026, 1, 1))
