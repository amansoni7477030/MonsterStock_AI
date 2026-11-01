"""Logging that makes every problem traceable: redacted, time-zone explicit, one line per fact, bounded on disk.

* **Secrets never reach disk or the journal.** Redaction happens in the formatter, which sees the final text including
  tracebacks.
* **Every line says when, how bad, which job and which run.** Stamps are ISO-8601 with milliseconds and an explicit UTC offset
  in the display zone (``MONSTER_TIMEZONE``, default IST); they never depend on the server's own zone.
* **Daily files, no renames**: ``<role>-YYYY-MM-DD.log`` (role = daemon, jobs, cli, ...). A new day or a full file opens a new
  file *by name*, so the daemon and the job processes it launches can share a role without the rotation races of
  ``RotatingFileHandler``; a hard daily cap per role means a log storm cannot fill the disk.
* ``errors-YYYY-MM-DD.log`` collects WARNING and above from every process (with the process role and pid): the first place to
  look when something went wrong.
* Crashes are captured: uncaught exceptions (main thread, other threads, unraisable), Python warnings and - through
  ``faulthandler`` - native crashes of the libraries underneath.
* Old files are pruned by age and total size at every start and by the nightly housekeeping.
"""

from __future__ import annotations

import contextlib
import contextvars
import faulthandler
import json
import logging
import os
import re
import sys
import threading
import traceback
from collections.abc import Iterator
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, TextIO

from app.clock import UTC, format_stamp, resolve_zone

_SECRETS: set[str] = set()
_LOCK = threading.Lock()

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"bot\d{5,}:[A-Za-z0-9_-]{20,}"), "bot<redacted>"),
    (re.compile(r"\b\d{5,}:[A-Za-z0-9_-]{30,}\b"), "<redacted-token>"),
    # Authorization headers: keep the scheme word, mask the credential itself.
    (re.compile(r"(?i)\b(authorization\b\"?\s*[:=]\s*\"?)((?:bearer|basic|token)\s+)?[^\s\"',;]+"), r"\1\2<redacted>"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1<redacted>"),
    (
        re.compile(r"(?i)\b(api[_-]?key|apikey|token|secret|password|passphrase)(\"?\s*[:=]\s*\"?)([^\s\"&,;']{4,})"),
        r"\1\2<redacted>",
    ),
    (re.compile(r"(?i)(X-Amz-Signature=)[0-9a-f]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(X-Amz-Credential=)[^&\s]+"), r"\1<redacted>"),
    (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), "<redacted-aws-key>"),
)

current_job: contextvars.ContextVar[str] = contextvars.ContextVar("monster_job", default="-")
current_run: contextvars.ContextVar[str] = contextvars.ContextVar("monster_run", default="")

TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(jobtag)s] %(name)s: %(message)s"
ERROR_FORMAT = "%(asctime)s %(levelname)-7s %(role)s[%(process)d] [%(jobtag)s] %(name)s: %(message)s"
ERRORS_ROLE = "errors"
DEFAULT_MAX_FILE_MB = 25
MAX_FILES_PER_DAY = 4  # per role: 4 x 25 MB = 100 MB is the most one role can write in a day
ERRORS_MAX_FILE_BYTES = 10_000_000
ERRORS_MAX_FILES_PER_DAY = 3
SIZE_RESYNC_EVERY = 64  # records; other processes append to the same file, so the tracked size is re-read now and then
NOISY_LOGGERS = ("urllib3", "requests", "httpx", "httpcore", "curl_cffi", "matplotlib", "lightgbm", "PIL", "fontTools",
                 "charset_normalizer", "asyncio")


# ------------------------------------------------------------------------------------------- redaction
def register_secret(value: str | None) -> None:
    """Remember an exact secret value; any later occurrence in a log line is masked."""
    if value and len(value) >= 6:
        with _LOCK:
            _SECRETS.add(value)


def register_secrets(values: list[str]) -> None:
    for value in values:
        register_secret(value)


def clear_secrets() -> None:
    with _LOCK:
        _SECRETS.clear()


def redact(text: str) -> str:
    """Mask registered secrets and common credential shapes."""
    if not text:
        return text
    with _LOCK:
        secrets = sorted(_SECRETS, key=len, reverse=True)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, "<redacted>")
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def exception_site(exc: BaseException) -> str:
    """``telegram.py:482 in _post`` - where an exception was raised, WITHOUT its message. For code that must keep even
    unformatted log records free of whatever the message may embed (a request URL with a bot token)."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return "unknown location"
    frame = frames[-1]
    return f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}"


# -------------------------------------------------------------------------------------------- context
class ContextFilter(logging.Filter):
    """Adds the job / run of the current context and the process role to every record."""

    def __init__(self, role: str = "-") -> None:
        super().__init__()
        self.role = role

    def filter(self, record: logging.LogRecord) -> bool:
        job, run = current_job.get(), current_run.get()
        record.job = job
        record.run = run
        record.jobtag = (f"{job} {run}".strip() if job != "-" else run) or "-"
        record.role = self.role
        return True


JobContextFilter = ContextFilter  # the previous name


@contextlib.contextmanager
def job_context(name: str, run: str = "") -> Iterator[None]:
    """Everything logged inside carries ``[name run]``, e.g. ``[final_scan 2026-11-02#1 r1234]``."""
    job_token, run_token = current_job.set(name), current_run.set(run)
    try:
        yield
    finally:
        current_run.reset(run_token)
        current_job.reset(job_token)


# ------------------------------------------------------------------------------------------ formatters
class RedactingFormatter(logging.Formatter):
    """Text lines in the display zone, secrets masked in the final text (tracebacks included)."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, zone: tzinfo = UTC) -> None:
        super().__init__(fmt, datefmt)
        self.zone = zone

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802 - logging API
        return format_stamp(record.created, self.zone)

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class JsonFormatter(RedactingFormatter):
    """One JSON object per line, for log shippers (``LOG_FORMAT=json``)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record), "level": record.levelname, "role": getattr(record, "role", "-"),
            "pid": record.process, "job": getattr(record, "job", "-"), "run": getattr(record, "run", ""),
            "logger": record.name, "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return redact(json.dumps(payload, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------------------------- files
class DailyFileHandler(logging.Handler):
    """Appends to ``<directory>/<role>-<day>.log`` (the day in the display zone).

    Nothing is ever renamed: a new day or a full file opens the next name (``<role>-<day>.1.log`` ...), which is safe for several
    processes writing the same role. After ``max_files_per_day`` full files the rest of the day is dropped (and counted in a
    marker line the next day) - a log storm must never fill the disk."""

    def __init__(self, directory: Path, role: str, zone: tzinfo, *, max_file_bytes: int = DEFAULT_MAX_FILE_MB * 1_000_000,
                 max_files_per_day: int = MAX_FILES_PER_DAY, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.directory = Path(directory)
        self.role = role
        self.zone = zone
        self.max_file_bytes = max(int(max_file_bytes), 1024)
        self.max_files = max(int(max_files_per_day), 1)
        self._stream: TextIO | None = None
        self._day = ""
        self._index = 0
        self._size = 0
        self._since_sync = 0
        self._dropped = 0
        self._dropped_day = ""

    # ----------------------------------------------------------------------------------- file handling
    def path_for(self, day: str, index: int = 0) -> Path:
        return self.directory / f"{self.role}-{day}{f'.{index}' if index else ''}.log"

    def _first_index(self, day: str) -> int:
        """Another process may already have filled the first files of today."""
        index = 0
        while index + 1 < self.max_files:
            path = self.path_for(day, index)
            if not path.exists() or path.stat().st_size < self.max_file_bytes:
                break
            index += 1
        return index

    def _open(self, day: str, index: int) -> None:
        self._close_stream()
        path = self.path_for(day, index)
        self.directory.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        self._stream = open(path, "a", encoding="utf-8", errors="replace", newline="\n")  # noqa: SIM115 - kept open on purpose
        self._day, self._index, self._since_sync = day, index, 0
        self._size = path.stat().st_size
        if not existed:
            with contextlib.suppress(OSError):
                os.chmod(path, 0o640)

    def _close_stream(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(OSError, ValueError):
                self._stream.close()
            self._stream = None

    def _write(self, text: str) -> None:
        assert self._stream is not None
        self._stream.write(text)
        self._stream.flush()
        self._size += len(text.encode("utf-8", "replace"))
        self._since_sync += 1
        if self._since_sync >= SIZE_RESYNC_EVERY:
            self._since_sync = 0
            with contextlib.suppress(OSError, ValueError):
                self._size = os.fstat(self._stream.fileno()).st_size

    def _drop_marker(self, now: float) -> str:
        return (f"{format_stamp(now, self.zone)} WARNING [log] {self._dropped} log records of {self._dropped_day} were dropped "
                f"after the daily cap of {self.max_files} x {self.max_file_bytes // 1_000_000 or 1} MB was reached\n")

    def ensure_open(self, now: float | None = None) -> None:
        """Open today's file now (raises OSError when the directory is not writable)."""
        moment = now if now is not None else datetime.now(UTC).timestamp()
        day = datetime.fromtimestamp(moment, self.zone).strftime("%Y-%m-%d")
        self._open(day, self._first_index(day))

    # ----------------------------------------------------------------------------------------- handler
    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            day = datetime.fromtimestamp(record.created, self.zone).strftime("%Y-%m-%d")
            if self._stream is None or day != self._day:
                self._open(day, self._first_index(day))
                if self._dropped:
                    self._write(self._drop_marker(record.created))
                    self._dropped = 0
            if self._size >= self.max_file_bytes:
                if self._index + 1 >= self.max_files:
                    self._dropped += 1
                    self._dropped_day = day
                    return
                self._open(day, self._index + 1)
            self._write(line)
        except Exception:  # noqa: BLE001 - logging must never take the application down
            self.handleError(record)

    def flush(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            if self._stream is not None:
                self._stream.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._dropped and self._stream is not None:
                self._write(self._drop_marker(datetime.now(UTC).timestamp()))
        self._close_stream()
        super().close()


def prune_logs(log_dir: Path | str, retention_days: int, max_mb: float, *, dry_run: bool = False) -> int:
    """Delete log files older than ``retention_days``, then the oldest until the folder fits ``max_mb`` (0 = no size budget).
    Files written in the last day are the live logs: they are never removed, only counted towards the budget."""
    import time

    from app.resources import prune_directory

    base = Path(log_dir)
    if not base.exists():
        return 0
    now = time.time()
    live: set[str] = set()
    for path in base.rglob("*"):
        with contextlib.suppress(OSError):
            if path.is_file() and now - path.stat().st_mtime < 86400:
                live.add(path.name)
    return prune_directory(base, max_mb=max_mb if max_mb > 0 else None, max_age_days=max(retention_days, 1), keep=live,
                           dry_run=dry_run)


# ---------------------------------------------------------------------------------------------- crashes
_hooks: dict[str, Any] = {}
_fault_file: TextIO | None = None
_fault_was_enabled = False


def install_crash_hooks(role: str, log_dir: Path | str | None = None) -> None:
    """Log uncaught exceptions of the main thread and of other threads, unraisable exceptions, and dump the Python stacks of a
    native crash (segfault in a C extension) to ``fault-<role>.log``. Idempotent; ``reset_logging`` undoes it."""
    global _fault_file, _fault_was_enabled
    if _hooks:
        return
    crash = logging.getLogger("crash")
    _hooks.update(excepthook=sys.excepthook, threading=threading.excepthook, unraisable=sys.unraisablehook)

    def excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            _hooks["excepthook"](exc_type, exc, tb)
            return
        crash.critical("uncaught exception, the process is ending", exc_info=(exc_type, exc, tb))
        if not any(getattr(h, "_monster_managed", False) for h in logging.getLogger().handlers):
            _hooks["excepthook"](exc_type, exc, tb)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread is not None else "?"
        crash.critical("uncaught exception in thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    def unraisable(info: Any) -> None:
        crash.error("unraisable exception: %s", info.err_msg or "", exc_info=(info.exc_type, info.exc_value, info.exc_traceback))

    sys.excepthook, threading.excepthook, sys.unraisablehook = excepthook, thread_hook, unraisable
    _fault_was_enabled = faulthandler.is_enabled()
    if log_dir is not None:
        try:
            _fault_file = open(Path(log_dir) / f"fault-{role}.log", "a", encoding="utf-8")  # noqa: SIM115 - must outlive this call
            faulthandler.enable(file=_fault_file, all_threads=True)
        except (OSError, RuntimeError, ValueError):
            _fault_file = None


def _remove_crash_hooks() -> None:
    global _fault_file
    if not _hooks:
        return
    sys.excepthook, threading.excepthook, sys.unraisablehook = _hooks["excepthook"], _hooks["threading"], _hooks["unraisable"]
    _hooks.clear()
    if _fault_file is not None:
        with contextlib.suppress(Exception):
            faulthandler.disable()
            _fault_file.close()
        _fault_file = None
        if _fault_was_enabled:
            with contextlib.suppress(Exception):
                faulthandler.enable()


# ----------------------------------------------------------------------------------------------- setup
_MANAGED = "_monster_managed"


def reset_logging() -> None:
    """Remove handlers installed by ``setup_logging`` and the crash hooks (used by tests and on re-configuration)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MANAGED, False):
            root.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()
    logging.captureWarnings(False)
    _remove_crash_hooks()


def setup_logging(
    log_dir: Path | str | None,
    level: str = "INFO",
    *,
    console: bool = True,
    filename: str = "monster.log",
    secrets: list[str] | None = None,
    timezone_name: str = "UTC",
    fmt: str = "text",
    retention_days: int = 30,
    max_mb: float = 150,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
    crash_hooks: bool = False,
) -> logging.Logger:
    """Idempotent logging setup: today's ``<role>-<day>.log`` (role = the stem of ``filename``), the shared
    ``errors-<day>.log`` (WARNING+), and stderr (captured by journald).

    Falls back to console-only if the log directory is unwritable; logging must never crash the app."""
    register_secrets(secrets or [])
    reset_logging()
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    zone = resolve_zone(timezone_name)
    role = Path(filename).stem or "monster"
    formatter_class = JsonFormatter if str(fmt).lower() == "json" else RedactingFormatter
    formatter = formatter_class(TEXT_FORMAT, zone=zone)
    context_filter = ContextFilter(role)

    handlers: list[logging.Handler] = []
    if log_dir is not None:
        try:
            directory = Path(log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                prune_logs(directory, retention_days, max_mb)
            main = DailyFileHandler(directory, role, zone, max_file_bytes=int(max_file_mb * 1_000_000))
            main.ensure_open()
            errors = DailyFileHandler(directory, ERRORS_ROLE, zone, max_file_bytes=ERRORS_MAX_FILE_BYTES,
                                      max_files_per_day=ERRORS_MAX_FILES_PER_DAY, level=logging.WARNING)
            errors.setFormatter(formatter_class(ERROR_FORMAT, zone=zone))
            handlers += [main, errors]
        except OSError as exc:
            handlers = []
            sys.stderr.write(f"log directory unavailable ({exc}); continuing with console logging only\n")
    if console or not handlers:
        handlers.append(logging.StreamHandler(sys.stderr))

    for handler in handlers:
        if handler.formatter is None:
            handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        setattr(handler, _MANAGED, True)
        root.addHandler(handler)

    # Chatty libraries: keep them at WARNING so request URLs (which may embed keys) are not logged.
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.captureWarnings(True)
    if crash_hooks:
        install_crash_hooks(role, log_dir if handlers and log_dir is not None and Path(log_dir).is_dir() else None)
    return root


def configure_from_settings(settings: Any, filename: str, *, console: bool = True, crash_hooks: bool = True) -> logging.Logger:
    """``setup_logging`` with every knob taken from the settings (LOG_LEVEL, LOG_FORMAT, MONSTER_TIMEZONE, LOG_RETENTION_DAYS,
    LOG_MAX_MB)."""
    return setup_logging(
        settings.log_dir, settings.log_level, console=console, filename=filename, secrets=settings.secrets(),
        timezone_name=settings.timezone, fmt=settings.log_format, retention_days=settings.log_retention_days,
        max_mb=settings.log_max_mb, crash_hooks=crash_hooks)
