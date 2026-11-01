"""The same code gives the same answers on a server in any time zone.

``test_timezone_safety.py`` forbids the APIs that read the server's zone; this test proves the behaviour: a probe script is run
in fresh interpreters whose process-wide zone is UTC, US Pacific (DST), India, New Zealand (DST) and UTC+14, and every
time-dependent answer (exchange dates, session windows, the job timetable, holidays, retention cut-offs, log stamps) must be
identical, and equal to values worked out by hand.
"""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404 - runs this repo's own probe script with a fixed argument vector
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# POSIX TZ strings work for glibc and for the Windows C runtime alike
ZONES = ("UTC0", "PST8PDT", "IST-5:30", "NZST-12NZDT", "LINT-14")

PROBE = r'''
import json, logging, os, sys, tempfile, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from app.clock import IST, FakeClock, at_ist, iso, ist_date, parse_ts, set_clock
from app.config import load_settings
from app.logging_setup import RedactingFormatter, setup_logging, reset_logging
from automation import retention
from automation.calendar import ExchangeCalendar, seed_calendar
from automation.schedule import ScheduleEngine
from database.store import Store

UTC = timezone.utc
out = {"local_hour_of_epoch": time.localtime(0).tm_hour}
base = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
probes = [base + timedelta(minutes=m) for m in range(0, 24 * 60, 90)]
probes += [datetime(2026, 11, 1, 18, 29, 59, tzinfo=UTC), datetime(2026, 11, 1, 18, 30, tzinfo=UTC),
           datetime(2026, 3, 8, 7, 0, tzinfo=UTC), datetime(2026, 11, 1, 6, 30, tzinfo=UTC)]  # US / NZ DST edges
out["ist_dates"] = [ist_date(p).isoformat() for p in probes]
out["at_ist"] = [iso(at_ist("2026-11-02", t)) for t in ("08:30", "09:08:15", "09:15", "15:30", "23:59:59")]
out["parse"] = [iso(parse_ts("02-Nov-2026 09:08:00", assume_tz=IST)), iso(parse_ts("2026-11-02T09:15:00")),
                iso(parse_ts("2026-11-02T09:15:00+05:30")), iso(parse_ts(1762050000)), iso(parse_ts("20261102T031500Z"))]

tmp = Path(tempfile.mkdtemp())
env = {"MONSTER_ENV": "test", "MONSTER_HOME": os.getcwd(), "MONSTER_DATA_DIR": str(tmp / "data"), "MONSTER_LOG_DIR": str(tmp / "logs")}
settings = load_settings(env, env_file=tmp / "none.env")
settings.ensure_dirs()
clock = FakeClock(datetime(2026, 11, 2, 3, 0, tzinfo=UTC))
set_clock(clock)
store = Store.open(settings.db_path, clock=clock)
seed_calendar(store, settings, now=clock.now())
cal = ExchangeCalendar(store)
days = [date(2026, 11, 2), date(2026, 11, 1), date(2026, 12, 25), date(2026, 11, 3), date(2027, 1, 26), date(2027, 3, 26)]
out["trading"] = [cal.is_trading_day(d) for d in days]
out["last_completed"] = [str(cal.last_completed_session(p)) for p in probes]
engine = ScheduleEngine(store, cal, clock)
engine.first_start()
due = []
for minutes in range(0, 24 * 60, 45):
    moment = base + timedelta(minutes=minutes)
    clock.set(moment)
    due.append(sorted(f"{d.entry.label}:{d.run_key}" for d in engine.due(moment)))
out["due"] = due
out["cutoffs"] = [retention.cutoff_for(unit, p, 30) for p in probes[:4] for unit in ("date", "ts")]

record = logging.LogRecord("probe", logging.INFO, __file__, 1, "hello", (), None)
record.created = datetime(2026, 11, 2, 3, 0, 5, tzinfo=UTC).timestamp()
record.jobtag, record.role = "x", "probe"
formatter = setup_logging(None, "INFO", console=False, timezone_name=settings.timezone).handlers[0].formatter
out["log_line"] = formatter.format(record)
reset_logging()
print(json.dumps(out, sort_keys=True))
'''


def run_probe(zone: str, tmp_path: Path) -> dict:
    script = tmp_path / "probe.py"
    script.write_text(PROBE, encoding="utf-8")
    env = {**os.environ, "TZ": zone, "PYTHONPATH": str(ROOT), "MONSTER_ENV": "test", "PYTHONIOENCODING": "utf-8"}
    for name in ("MONSTER_TIMEZONE", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, capture_output=True, text=True,  # noqa: S603
                            timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def answers(tmp_path_factory):
    return {zone: run_probe(zone, tmp_path_factory.mktemp(zone.replace(":", "_"))) for zone in ZONES}


def test_every_zone_gives_identical_answers(answers):
    reference = {k: v for k, v in answers["UTC0"].items() if k != "local_hour_of_epoch"}
    for zone in ZONES[1:]:
        got = {k: v for k, v in answers[zone].items() if k != "local_hour_of_epoch"}
        for key in reference:
            assert got[key] == reference[key], f"{key} differs when the server runs in {zone}"


def test_the_probe_really_ran_in_different_zones(answers):
    hours = {a["local_hour_of_epoch"] for a in answers.values()}
    # POSIX honours every TZ string; the Windows C runtime understands only some of them (and no offsets beyond 12 h)
    assert len(hours) >= (4 if hasattr(time, "tzset") else 3), hours  # the harness changed the server zone


def test_answers_match_values_worked_out_by_hand(answers):
    a = answers["PST8PDT"]
    assert a["at_ist"] == ["2026-11-02T03:00:00Z", "2026-11-02T03:38:15Z", "2026-11-02T03:45:00Z", "2026-11-02T10:00:00Z",
                           "2026-11-02T18:29:59Z"]
    assert a["parse"][:3] == ["2026-11-02T03:38:00Z", "2026-11-02T09:15:00Z", "2026-11-02T03:45:00Z"]  # naive text = UTC / IST by rule
    assert a["trading"] == [True, False, False, True, False, False]  # Mon, Sun, Christmas, Tue, Republic Day 2027, Good Friday 2027
    assert a["ist_dates"][-4:-2] == ["2026-11-01", "2026-11-02"]  # 18:29:59 UTC is still 1 Nov in India; 18:30 is 2 Nov
    assert a["cutoffs"][0] == "2026-10-03" and a["cutoffs"][1].endswith("Z")


def test_log_lines_carry_an_explicit_zone_and_never_the_servers(answers):
    lines = {zone: a["log_line"] for zone, a in answers.items()}
    assert len(set(lines.values())) == 1, lines
    assert lines["PST8PDT"].startswith("2026-11-02T08:30:05.000+0530 INFO   ")  # 03:00:05 UTC shown in the display zone (IST)
