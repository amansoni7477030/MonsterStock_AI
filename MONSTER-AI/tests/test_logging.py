from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import logging_setup as ls
from app.clock import IST, UTC, FakeClock, format_stamp, resolve_zone
from app.logging_setup import (
    clear_secrets,
    job_context,
    redact,
    register_secret,
    reset_logging,
    setup_logging,
)

TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"
NOON_IST = datetime(2026, 11, 2, 6, 30, tzinfo=timezone.utc)  # 12:00 IST on a Monday


def record_at(moment: datetime, message: str = "hello", level: int = logging.INFO, name: str = "t") -> logging.LogRecord:
    rec = logging.LogRecord(name, level, __file__, 1, message, (), None)
    rec.created = moment.timestamp()
    rec.msecs = (rec.created % 1) * 1000
    return rec


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------ redaction
def test_telegram_token_in_url_is_redacted():
    text = f"POST https://api.telegram.org/bot{TOKEN}/sendMessage failed"
    assert TOKEN not in redact(text)
    assert "bot<redacted>" in redact(text)


def test_credential_shapes_are_redacted():
    assert "supersecret1" not in redact("api_key=supersecret1&x=1")
    assert "abcdef123456" not in redact('{"Authorization": "******"}')
    assert "AKIAABCDEFGHIJKLMNOP" not in redact("key AKIAABCDEFGHIJKLMNOP used")
    assert "deadbeef" not in redact("https://s3/x?X-Amz-Signature=deadbeef&a=1")
    assert redact("") == ""


def test_registered_secret_values_are_masked_everywhere():
    register_secret("my-very-private-value")
    assert redact("boom my-very-private-value boom") == "boom <redacted> boom"
    clear_secrets()
    assert redact("boom my-very-private-value") == "boom my-very-private-value"
    register_secret("abc")  # too short to be a meaningful secret; ignored
    assert redact("abc") == "abc"


# -------------------------------------------------------------------------------------- time stamps
def test_stamps_are_iso_with_milliseconds_and_an_explicit_offset_in_the_display_zone():
    moment = datetime(2026, 11, 2, 3, 0, 5, 123456, tzinfo=timezone.utc)
    assert format_stamp(moment.timestamp(), UTC) == "2026-11-02T03:00:05.123+0000"
    assert format_stamp(moment.timestamp(), IST) == "2026-11-02T08:30:05.123+0530"
    assert format_stamp(moment.timestamp(), resolve_zone("America/New_York")) == "2026-11-01T22:00:05.123-0500"  # DST ended on 1 Nov


def test_zone_resolution_never_raises():
    assert resolve_zone(None) is UTC and resolve_zone("UTC") is UTC and resolve_zone("Mars/Base") is UTC
    assert format_stamp(0.0, resolve_zone("Asia/Kolkata")).endswith("+0530")


def test_formatter_prints_the_display_zone_not_the_servers():
    formatter = ls.RedactingFormatter(ls.TEXT_FORMAT, zone=IST)
    rec = record_at(datetime(2026, 11, 2, 3, 0, 5, tzinfo=timezone.utc))
    rec.jobtag = "final_scan 2026-11-02#1 r7"
    assert formatter.format(rec) == "2026-11-02T08:30:05.000+0530 INFO    [final_scan 2026-11-02#1 r7] t: hello"


def test_stamp_of_the_fake_clock_in_ist(fake_clock: FakeClock):
    assert format_stamp(fake_clock.now().timestamp(), IST).startswith("2026-11-02T08:30:00")


# ------------------------------------------------------------------------------------------- setup
def test_setup_is_idempotent_and_writes_redacted_daily_files(tmp_path):
    for _ in range(3):
        setup_logging(tmp_path, "INFO", console=False, secrets=[TOKEN], filename="cli.log", timezone_name="Asia/Kolkata")
    managed = [h for h in logging.getLogger().handlers if getattr(h, "_monster_managed", False)]
    assert len(managed) == 2  # the role file and the shared errors file
    with job_context("unit", "2026-11-02#1 r5"):
        logging.getLogger("test").error("request to /bot%s/x failed: %s", TOKEN, TOKEN)
    for handler in managed:
        handler.flush()
    files = sorted(p.name for p in tmp_path.glob("*.log"))
    assert len(files) == 2 and files[0].startswith("cli-") and files[1].startswith("errors-")
    content = read(next(tmp_path.glob("cli-*.log")))
    assert "[unit 2026-11-02#1 r5]" in content and "failed" in content and "+0530" in content
    assert TOKEN not in content and TOKEN not in read(next(tmp_path.glob("errors-*.log")))


def test_errors_file_collects_warnings_with_role_and_pid_only(tmp_path):
    setup_logging(tmp_path, "INFO", console=False, filename="daemon.log")
    logging.getLogger("a").info("routine")
    logging.getLogger("a").warning("careful")
    logging.getLogger("a").error("broken", exc_info=ValueError("boom"))
    reset_logging()
    errors = read(next(tmp_path.glob("errors-*.log")))
    assert "routine" not in errors and "careful" in errors and "broken" in errors and "ValueError: boom" in errors
    assert "daemon[" in errors  # role and pid
    assert "routine" in read(next(tmp_path.glob("daemon-*.log")))


def test_unwritable_log_dir_falls_back_to_console(tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    setup_logging(blocker / "sub", "INFO")  # parent is a file -> mkdir fails
    root = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    assert not any(isinstance(h, ls.DailyFileHandler) for h in root.handlers)


def test_json_format_is_valid_and_redacted(tmp_path):
    setup_logging(tmp_path, "INFO", console=False, filename="jobs.log", fmt="json", secrets=[TOKEN])
    with job_context("evaluate", "2026-11-02#2 r9"):
        logging.getLogger("x").warning("token=%s", TOKEN, exc_info=ValueError("bad"))
    reset_logging()
    row = json.loads(read(next(tmp_path.glob("jobs-*.log"))).splitlines()[0])
    assert row["level"] == "WARNING" and row["job"] == "evaluate" and row["run"] == "2026-11-02#2 r9" and row["role"] == "jobs"
    assert "ValueError: bad" in row["exc"] and TOKEN not in json.dumps(row)


def test_noisy_libraries_are_kept_quiet_and_python_warnings_are_captured(tmp_path):
    setup_logging(tmp_path, "DEBUG", console=False)
    assert logging.getLogger("urllib3").level == logging.WARNING and logging.getLogger("curl_cffi").level == logging.WARNING
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        logging.captureWarnings(True)
        warnings.warn("something deprecated", DeprecationWarning, stacklevel=1)
    reset_logging()
    assert "something deprecated" in read(next(tmp_path.glob("errors-*.log")))


def test_exception_site_names_the_place_and_never_the_message():
    def inner():
        raise RuntimeError(f"POST https://api.telegram.org/bot{TOKEN}/sendMessage failed")

    def outer():
        inner()

    try:
        outer()
    except RuntimeError as exc:
        site = ls.exception_site(exc)
    assert site.startswith("test_logging.py:") and site.endswith("in inner") and TOKEN not in site and "telegram" not in site
    assert ls.exception_site(ValueError("never raised")) == "unknown location"


def test_settings_drive_the_configuration(tmp_path, settings):
    import dataclasses

    cfg = dataclasses.replace(settings, log_dir=tmp_path / "lg", log_format="json", timezone="Asia/Tokyo", log_level="WARNING")
    ls.configure_from_settings(cfg, "health.log", console=False, crash_hooks=False)
    logging.getLogger("x").warning("from settings")
    reset_logging()
    row = json.loads(read(next((tmp_path / "lg").glob("health-*.log"))).splitlines()[0])
    assert row["role"] == "health" and row["ts"].endswith("+0900")


# ----------------------------------------------------------------------------------- daily files
def handler(tmp_path, **kwargs) -> ls.DailyFileHandler:
    h = ls.DailyFileHandler(tmp_path, "jobs", IST, **kwargs)
    h.setFormatter(ls.RedactingFormatter("%(asctime)s %(message)s", zone=IST))
    return h


def test_a_new_day_opens_a_new_file_by_name_without_renaming(tmp_path):
    h = handler(tmp_path)
    h.handle(record_at(NOON_IST, "day one"))
    h.handle(record_at(NOON_IST + timedelta(days=1), "day two"))
    h.handle(record_at(NOON_IST, "late line of day one from another process"))  # appends to day one's file again
    h.close()
    assert read(tmp_path / "jobs-2026-11-02.log").count("\n") == 2
    assert "day two" in read(tmp_path / "jobs-2026-11-03.log")


def test_the_day_follows_the_display_zone_not_utc(tmp_path):
    h = handler(tmp_path)
    h.handle(record_at(datetime(2026, 11, 1, 18, 29, 59, tzinfo=timezone.utc)))  # 23:59:59 on 1 Nov in India
    h.handle(record_at(datetime(2026, 11, 1, 18, 30, 0, tzinfo=timezone.utc)))  # 00:00:00 on 2 Nov in India
    h.close()
    assert (tmp_path / "jobs-2026-11-01.log").exists() and (tmp_path / "jobs-2026-11-02.log").exists()


def test_full_files_roll_over_by_name_and_the_daily_cap_drops_the_rest(tmp_path):
    h = handler(tmp_path, max_file_bytes=1024, max_files_per_day=2)
    for i in range(100):
        h.handle(record_at(NOON_IST, f"line {i:03d} " + "x" * 80))
    assert sorted(p.name for p in tmp_path.glob("jobs-*.log")) == ["jobs-2026-11-02.1.log", "jobs-2026-11-02.log"]
    assert h._dropped > 50  # everything beyond 2 x 1 KB was dropped, not written
    h.handle(record_at(NOON_IST + timedelta(days=1), "next day"))
    h.close()
    marker = read(tmp_path / "jobs-2026-11-03.log")
    assert "were dropped after the daily cap" in marker and "next day" in marker
    assert sum(p.stat().st_size for p in tmp_path.glob("jobs-2026-11-02*.log")) < 3000


def test_the_drop_count_is_written_when_the_process_ends_on_a_capped_day(tmp_path):
    h = handler(tmp_path, max_file_bytes=1024, max_files_per_day=1)
    for i in range(40):
        h.handle(record_at(NOON_IST, f"line {i:03d} " + "x" * 80))
    h.close()
    assert "were dropped after the daily cap" in read(tmp_path / "jobs-2026-11-02.log").splitlines()[-1]


def test_another_process_that_filled_todays_first_file_is_respected(tmp_path):
    (tmp_path / "jobs-2026-11-02.log").write_text("x" * 2048)
    h = handler(tmp_path, max_file_bytes=1024, max_files_per_day=3)
    h.handle(record_at(NOON_IST, "mine"))
    h.close()
    assert "mine" in read(tmp_path / "jobs-2026-11-02.1.log")


def test_concurrent_writers_lose_no_lines(tmp_path):
    h = handler(tmp_path)

    def work(tag: str) -> None:
        for i in range(200):
            h.handle(record_at(NOON_IST, f"{tag}-{i}"))

    threads = [threading.Thread(target=work, args=(f"t{n}",)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    h.close()
    assert len(read(tmp_path / "jobs-2026-11-02.log").splitlines()) == 800


def test_two_handlers_sharing_a_file_both_land_in_it(tmp_path):
    a, b = handler(tmp_path), handler(tmp_path)  # stand-ins for two processes of the same role
    for i in range(50):
        a.handle(record_at(NOON_IST, f"a{i}"))
        b.handle(record_at(NOON_IST, f"b{i}"))
    a.close()
    b.close()
    lines = read(tmp_path / "jobs-2026-11-02.log").splitlines()
    assert len(lines) == 100 and sum(1 for line in lines if " a" in line) == 50


def test_files_are_private_on_posix(tmp_path):
    if sys.platform == "win32":
        pytest.skip("POSIX permissions")
    h = handler(tmp_path)
    h.handle(record_at(NOON_IST))
    h.close()
    assert (tmp_path / "jobs-2026-11-02.log").stat().st_mode & 0o777 == 0o640


# --------------------------------------------------------------------------------------- pruning
def test_prune_logs_by_age_then_budget_and_never_the_live_files(tmp_path):
    def make(name: str, age_days: float) -> Path:
        path = tmp_path / name
        path.write_bytes(b"x" * 400_000)
        moment = time.time() - age_days * 86400
        os.utime(path, (moment, moment))
        return path

    old, mid, newer, today = make("jobs-old.log", 45), make("jobs-mid.log", 10), make("jobs-newer.log", 3), make("jobs-today.log", 0)
    assert ls.prune_logs(tmp_path, 30, 0, dry_run=True) == 1 and old.exists()
    assert ls.prune_logs(tmp_path, 30, 0) == 1 and not old.exists() and mid.exists()
    assert ls.prune_logs(tmp_path, 30, 1.0) == 1  # 1.1 MB used, budget 1 MB: the oldest goes, the live file counts but stays
    assert not mid.exists() and newer.exists() and today.exists()
    assert ls.prune_logs(tmp_path, 30, 0.5) == 1 and not newer.exists() and today.exists()
    assert ls.prune_logs(tmp_path, 30, 0.0001) == 0 and today.exists()  # only the live file is left: nothing more to remove
    assert ls.prune_logs(tmp_path / "missing", 30, 1) == 0


def test_setup_prunes_old_files_at_every_start(tmp_path):
    stale = tmp_path / "cli-2026-01-01.log"
    stale.write_text("old")
    moment = time.time() - 90 * 86400
    os.utime(stale, (moment, moment))
    setup_logging(tmp_path, "INFO", console=False, retention_days=30)
    assert not stale.exists()


# ------------------------------------------------------------------------------------ crash hooks
def test_uncaught_exceptions_are_logged_and_the_hooks_are_restored(tmp_path):
    original = (sys.excepthook, threading.excepthook, sys.unraisablehook)
    setup_logging(tmp_path, "INFO", console=False, filename="daemon.log", crash_hooks=True)
    assert sys.excepthook is not original[0]
    try:
        raise RuntimeError("main thread blew up")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())

    def explode() -> None:
        raise KeyError("worker blew up")

    worker = threading.Thread(target=explode, name="worker-1")
    worker.start()
    worker.join()
    reset_logging()
    assert (sys.excepthook, threading.excepthook, sys.unraisablehook) == original
    errors = read(next(tmp_path.glob("errors-*.log")))
    assert "uncaught exception, the process is ending" in errors and "RuntimeError: main thread blew up" in errors
    assert "uncaught exception in thread worker-1" in errors and "KeyError" in errors
    assert (tmp_path / "fault-daemon.log").exists()  # native crashes dump their Python stacks here


def test_keyboard_interrupt_is_not_reported_as_a_crash(tmp_path, capsys):
    setup_logging(tmp_path, "INFO", console=False, crash_hooks=True)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    reset_logging()
    assert not list(tmp_path.glob("errors-*.log")) or "uncaught" not in read(next(tmp_path.glob("errors-*.log")))


def test_crash_hooks_are_off_unless_requested(tmp_path):
    before = sys.excepthook
    setup_logging(tmp_path, "INFO", console=False)
    assert sys.excepthook is before
    reset_logging()


# ------------------------------------------------------------------------------------ job context
def test_job_context_nests_and_restores():
    assert ls.current_job.get() == "-"
    with job_context("outer", "k#1"):
        with job_context("inner", "k#2"):
            assert (ls.current_job.get(), ls.current_run.get()) == ("inner", "k#2")
        assert ls.current_job.get() == "outer"
    assert (ls.current_job.get(), ls.current_run.get()) == ("-", "")


def test_context_filter_builds_the_tag():
    flt = ls.ContextFilter("daemon")
    rec = record_at(NOON_IST)
    flt.filter(rec)
    assert rec.jobtag == "-" and rec.role == "daemon"
    with job_context("train", "2026-W44#1 r3"):
        flt.filter(rec)
    assert rec.jobtag == "train 2026-W44#1 r3" and rec.job == "train"
