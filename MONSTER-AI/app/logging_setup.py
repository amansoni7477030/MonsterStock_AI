"""Logging with mandatory secret redaction and rotating files.

Secrets (Telegram tokens appear inside request URLs, API keys inside headers) must never reach disk or the
journal, so redaction happens in the *formatter*: it sees the final text including exception tracebacks.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import logging.handlers
import re
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

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


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class JobContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job = current_job.get()
        return True


@contextlib.contextmanager
def job_context(name: str) -> Iterator[None]:
    token = current_job.set(name)
    try:
        yield
    finally:
        current_job.reset(token)


_FORMAT = "%(asctime)s %(levelname)-7s [%(job)s] %(name)s: %(message)s"
_MANAGED = "_monster_managed"


def reset_logging() -> None:
    """Remove handlers installed by ``setup_logging`` (used by tests)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MANAGED, False):
            root.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()


def setup_logging(
    log_dir: Path | None,
    level: str = "INFO",
    *,
    console: bool = True,
    filename: str = "monster.log",
    secrets: list[str] | None = None,
) -> logging.Logger:
    """Idempotent logging setup: rotating file (5 x 5 MB) plus stderr (captured by journald).

    Falls back to console-only if the log directory is unwritable; logging must never crash the app.
    """
    register_secrets(secrets or [])
    reset_logging()
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = RedactingFormatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z")
    context_filter = JobContextFilter()

    handlers: list[logging.Handler] = []
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                Path(log_dir) / filename, maxBytes=5_000_000, backupCount=5, encoding="utf-8", delay=True
            )
            handlers.append(file_handler)
        except OSError as exc:
            sys.stderr.write(f"log directory unavailable ({exc}); continuing with console logging only\n")
    if console or not handlers:
        handlers.append(logging.StreamHandler(sys.stderr))

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        setattr(handler, _MANAGED, True)
        root.addHandler(handler)

    # Chatty libraries: keep them at WARNING so request URLs (which may embed keys) are not logged.
    for noisy in ("urllib3", "requests", "httpx", "httpcore", "matplotlib", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root
