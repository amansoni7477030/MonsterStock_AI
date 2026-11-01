"""The daily timetable and the ledger-based "what is due now" engine (blueprint section 16).

Time is never trusted to a cron tick: every (job, run key) has rows in the ``job_runs`` ledger, and the engine derives from
the ledger alone what must run, what is waiting for a retry, and what was missed. A restart, reboot or clock step therefore
cannot skip a job silently or run it twice. All times are Asia/Kolkata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from app.clock import Clock, at_ist, get_clock, iso, ist_date, parse_ts
from automation.calendar import ExchangeCalendar
from automation.jobspec import JOBS
from database.store import Store

log = logging.getLogger(__name__)

FIRST_START_KEY = "daemon.first_start"
CATCH_UP_DAYS = 2
PREEMPTED = "preempted"


@dataclass(frozen=True)
class ScheduleEntry:
    job: str
    at: str  # earliest start, IST "HH:MM[:SS]"
    until: str  # latest start of a first attempt / retry, IST
    days: str = "trading"  # trading | any
    slot: str = ""  # distinguishes several runs of one job per day
    retry_sec: int = 300
    max_attempts: int = 3
    after: tuple[str, ...] = ()  # jobs of the same date that must have reached a terminal state first
    weekly_dow: int | None = None  # 0=Mon..6=Sun: run once per ISO week, eligible from that weekday on
    catch_up: bool = False  # may still run after its window (or on a later day) when it was missed

    @property
    def label(self) -> str:
        return f"{self.job}#{self.slot}" if self.slot else self.job


@dataclass(frozen=True)
class Due:
    entry: ScheduleEntry
    run_key: str
    day: date
    attempt: int
    last_attempt: bool

    @property
    def critical(self) -> bool:
        return JOBS[self.entry.job].critical

    @property
    def timeout_min(self) -> int:
        return JOBS[self.entry.job].timeout_min


E = ScheduleEntry
SCHEDULE: tuple[ScheduleEntry, ...] = (
    # ---- morning pipeline (trading days) -------------------------------------------------------------------
    E("collect_global", "06:00", "08:30", retry_sec=600, max_attempts=6),
    E("collect_news", "06:30", "08:35", slot="early", retry_sec=600, max_attempts=3),
    E("collect_news", "08:10", "08:38", slot="late", retry_sec=600, max_attempts=2),
    E("health_check", "07:50", "08:20", days="any", slot="morning", retry_sec=600, max_attempts=2),
    E("collect_india", "08:00", "08:35", retry_sec=300, max_attempts=6),
    E("prelim_scan", "08:45", "09:02", retry_sec=180, max_attempts=5),
    E("refresh_preopen", "09:00:30", "09:07:30", retry_sec=60, max_attempts=2),
    E("final_scan", "09:08:45", "09:13:30", retry_sec=45, max_attempts=3),
    # ---- after the close -------------------------------------------------------------------------------------
    E("ingest_eod", "17:45", "23:30", retry_sec=1200, max_attempts=12, catch_up=True),
    E("evaluate", "18:15", "23:45", retry_sec=1200, max_attempts=10, after=("ingest_eod",), catch_up=True),
    E("health_check", "20:00", "23:00", days="any", slot="evening", retry_sec=600, max_attempts=2),
    E("backup", "23:30", "23:59", days="any", retry_sec=900, max_attempts=3, catch_up=True),
    E("housekeeping", "23:50", "23:59", days="any", retry_sec=900, max_attempts=2, after=("backup",), catch_up=True),
    # ---- weekly ----------------------------------------------------------------------------------------------
    E("refresh_calendar", "05:30", "07:00", days="any", retry_sec=1800, max_attempts=3, catch_up=True),
    E("train", "02:00", "06:00", days="any", weekly_dow=5, retry_sec=900, max_attempts=3, catch_up=True),
    E("refresh_universe", "04:00", "12:00", days="any", weekly_dow=6, retry_sec=3600, max_attempts=2, catch_up=True),
    E("model_sync", "05:30", "07:00", days="any", retry_sec=1800, max_attempts=3, catch_up=True),
)


def run_key_for(entry: ScheduleEntry, day: date) -> str:
    if entry.weekly_dow is not None:
        year, week, _ = day.isocalendar()
        base = f"{year}-W{week:02d}"
    else:
        base = day.isoformat()
    return f"{base}:{entry.slot}" if entry.slot else base


class ScheduleEngine:
    def __init__(self, store: Store, calendar: ExchangeCalendar, clock: Clock | None = None,
                 entries: tuple[ScheduleEntry, ...] = SCHEDULE) -> None:
        self.store, self.calendar, self._clock = store, calendar, clock
        self.entries = entries

    @property
    def clock(self) -> Clock:
        return self._clock or get_clock()

    # ---------------------------------------------------------------------------------------- ledger state
    def _state(self, job: str, key: str) -> dict[str, Any]:
        rows = self.store.job_runs_for(job, key)
        settled = any(r["status"] in {"success", "skipped", "missed"} for r in rows)
        running = any(r["status"] == "running" for r in rows)
        failures = [r for r in rows if r["status"] in {"failed", "timeout"} and not (r["message"] or "").startswith(PREEMPTED)]
        last_end = max((parse_ts(r["finished_at"]) for r in rows if r["finished_at"]), default=None)
        return {"settled": settled, "running": running, "attempts": len(failures), "last_end": last_end}

    def first_start(self) -> datetime:
        stamp = self.store.kv_get(FIRST_START_KEY)
        if stamp is None:
            stamp = iso(self.clock.now()) or ""
            self.store.kv_set(FIRST_START_KEY, stamp)
        return parse_ts(stamp) or self.clock.now()

    # --------------------------------------------------------------------------------------- applicability
    def _eligible_day(self, entry: ScheduleEntry, day: date) -> bool:
        if entry.weekly_dow is not None:
            if day.weekday() < entry.weekly_dow:
                return False
        if entry.days == "trading":
            return self.calendar.is_trading_day(day)
        return True

    @staticmethod
    def _window(entry: ScheduleEntry, day: date) -> tuple[datetime, datetime]:
        start = at_ist(day, entry.at)
        end = at_ist(day, entry.until)
        if entry.weekly_dow is not None and day.weekday() > entry.weekly_dow:
            # catching up later in the week: no time-of-day restriction
            return at_ist(day, "00:00"), at_ist(day, "23:59:59")
        return start, end

    def _dependencies_done(self, entry: ScheduleEntry, day: date, now: datetime) -> bool:
        for dep in entry.after:
            dep_entries = [e for e in self.entries if e.job == dep and not e.slot]
            for d in dep_entries:
                if not self._eligible_day(d, day):
                    continue
                key = run_key_for(d, day)
                state = self._state(d.job, key)
                _start, end = self._window(d, day)
                window_over = now > end and not d.catch_up  # a catch-up prerequisite may still run after its window
                if not (state["settled"] or state["attempts"] >= d.max_attempts or window_over):
                    return False
        return True

    # ------------------------------------------------------------------------------------------------ due
    def due(self, now: datetime | None = None) -> list[Due]:
        now = now or self.clock.now()
        today = ist_date(now)
        first = self.first_start()
        found: list[Due] = []
        for back in range(CATCH_UP_DAYS, -1, -1):
            day = today - timedelta(days=back)
            for entry in self.entries:
                item = self._evaluate(entry, day, today, now, first)
                if item is not None:
                    found.append(item)
        found.sort(key=lambda d: (not d.critical, at_ist(d.day, d.entry.at), d.entry.label))
        return found

    def _evaluate(self, entry: ScheduleEntry, day: date, today: date, now: datetime, first: datetime) -> Due | None:
        if not self._eligible_day(entry, day):
            return None
        # weekly jobs are one logical run per ISO week: evaluate them only on today's date
        if entry.weekly_dow is not None and day != today:
            return None
        key = run_key_for(entry, day)
        start, end = self._window(entry, day)
        if now < start:
            return None
        state = self._state(entry.job, key)
        if state["settled"] or state["running"]:
            return None
        if day < today and end <= first:  # that window closed before the scheduler existed: nothing to catch up
            return None
        past_window = now > end if day == today else True
        if state["attempts"] >= entry.max_attempts:
            self._miss(entry, key, f"gave up after {state['attempts']} failed attempts")
            return None
        if past_window and not entry.catch_up:
            if end <= first:  # the scheduler did not exist yet when this window was open
                if day == today:
                    self.store.job_record(entry.job, key, "skipped", message="scheduler was not installed yet")
            else:
                self._miss(entry, key, f"window closed at {entry.until} IST without a successful run")
            return None
        if not self._dependencies_done(entry, day, now):
            return None
        last_end = state["last_end"]
        if last_end is not None and (now - last_end).total_seconds() < entry.retry_sec:
            return None
        attempt = state["attempts"] + 1
        retries_left = not past_window and (now + timedelta(seconds=entry.retry_sec)) <= end
        last = attempt >= entry.max_attempts or (not entry.catch_up and not retries_left)
        return Due(entry, key, day, attempt, last)

    def _miss(self, entry: ScheduleEntry, key: str, message: str) -> None:
        self.store.job_record(entry.job, key, "missed", message=message)
        critical = JOBS[entry.job].critical
        self.store.add_alert("error" if critical else "warning", f"job_missed_{entry.job}", f"{entry.label} [{key}] {message}",
                             dedupe_key=f"missed:{entry.label}:{key}", dedupe_minutes=24 * 60)
        log.warning("job %s [%s] missed: %s", entry.label, key, message)

    # ------------------------------------------------------------------------------------------- recovery
    def recover_stale(self, now: datetime | None = None, alive: Any = None) -> int:
        """Mark ledger rows left ``running`` by a crashed process as failed (so they can be retried)."""
        now = now or self.clock.now()
        recovered = 0
        for row in self.store.running_jobs():
            spec = JOBS.get(row["job_name"])
            started = parse_ts(row["started_at"]) or now
            limit = timedelta(minutes=(spec.timeout_min if spec else 60) + 5)
            dead = alive is not None and row["pid"] is not None and not alive(row["pid"])
            if dead or now - started > limit:
                self.store.job_finish(int(row["run_id"]), "failed",
                                      message="recovered: process ended without reporting a result")
                recovered += 1
        return recovered
