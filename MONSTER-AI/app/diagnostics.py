"""One file for a bug report: ``python -m app.cli diag`` writes ``<data>/diag/diag-<UTC stamp>.tar.gz``.

It holds everything needed to understand a failure without access to the server - versions, the doctor and health reports,
storage, recent jobs / alerts / source health, the redacted configuration and the tail of the logs. Every text goes through the
log redaction, so the archive is safe to attach to a ticket (the database itself is never included). Old bundles are pruned by
the nightly retention (the newest five stay).
"""

from __future__ import annotations

import io
import json
import logging
import platform
import sqlite3
import sys
import tarfile
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from app import __version__, logview
from app.clock import get_clock, iso, resolve_zone
from app.config import Settings
from app.logging_setup import redact

log = logging.getLogger(__name__)

PACKAGES = ("numpy", "pandas", "scipy", "lightgbm", "requests", "curl_cffi", "feedparser", "tzdata", "fastapi", "uvicorn")
ERROR_TAIL_LINES = 400
ROLE_TAIL_LINES = 150
MAX_FILE_BYTES = 400_000
RECENT_ROWS = 40
# What the optional AI services keep in ``kv`` (counters, cooldowns, the coverage flags and the latest news run). None of it is secret.
AI_STATE_KEYS = (
    "llm.calls", "llm.failures", "llm.tokens_in", "llm.tokens_out", "llm.cooldown_until",
    "llmweb.calls", "llmweb.failures", "llmweb.tokens_in", "llmweb.tokens_out", "llmweb.cooldown_until", "llmweb.day",
    "jev.calls", "jev.failures", "jev.tokens_in", "jev.tokens_out", "jev.cooldown_until",
    "news.reduced_coverage", "news.coverage", "news.last_run",
)


def _section(title: str, body: str) -> str:
    return f"==== {title} ====\n{body.rstrip()}\n"


def versions() -> str:
    lines = [f"monster-ai {__version__}", f"python {platform.python_version()} ({platform.python_implementation()}) on "
             f"{platform.platform()}", f"executable {sys.executable}"]
    for name in PACKAGES:
        try:
            lines.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name} (not installed)")
    return "\n".join(lines)


def _rows(store: Any, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    try:
        return [dict(r) for r in store.conn.execute(sql, params).fetchall()]
    except sqlite3.Error as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _table(rows: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in rows) or "(none)"


def database_state(store: Any) -> str:
    parts = [
        _section("recent job runs", _table(_rows(
            store, "SELECT run_id, job_name, run_key, attempt, status, started_at, finished_at, peak_rss_mb, message "
                   "FROM job_runs ORDER BY run_id DESC LIMIT ?", (RECENT_ROWS,)))),
        _section("recent alerts", _table(_rows(
            store, "SELECT alert_id, created_at, level, code, delivered, message FROM alerts ORDER BY alert_id DESC LIMIT ?",
            (RECENT_ROWS,)))),
        _section("source health", _table(_rows(store, "SELECT * FROM source_health ORDER BY source_id"))),
        _section("ai services (counters, cooldowns and the latest news run)", _table(_rows(
            store, f"SELECT key, value, updated_at FROM kv WHERE key IN ({', '.join('?' * len(AI_STATE_KEYS))}) ORDER BY key",
            AI_STATE_KEYS))),
        _section("models", _table(_rows(
            store, "SELECT model_version, stage, status, is_active, created_at, train_end, notes FROM model_registry "
                   "ORDER BY created_at DESC LIMIT ?", (RECENT_ROWS,)))),
        _section("recent deliveries", _table(_rows(
            store, "SELECT kind, trade_date, status, attempts, error, updated_at FROM deliveries ORDER BY delivery_id DESC "
                   "LIMIT ?", (RECENT_ROWS,)))),
        _section("state keys (names and update times only)", _table(_rows(
            store, "SELECT key, updated_at FROM kv WHERE key NOT LIKE 'llm:%' AND key NOT LIKE 'llmweb:%' "
                   "AND key NOT LIKE 'jev:%' AND key NOT LIKE 'news.sig:%' ORDER BY key"))),
    ]
    return "\n".join(parts)


def log_excerpts(settings: Settings) -> dict[str, str]:
    """The tail of today's errors file and of the newest file of every role, redacted."""
    out: dict[str, str] = {}
    files = logview.list_log_files(settings.log_dir)
    roles = sorted({f.role for f in files})
    for role in roles:
        newest = max((f for f in files if f.role == role), key=lambda f: (f.day, f.index))
        count = ERROR_TAIL_LINES if role == logview.ERRORS_ROLE else ROLE_TAIL_LINES
        text = "\n".join(logview.tail_lines(newest.path, count))
        out[f"logs/{newest.path.name}"] = redact(text)
    return out


def build_files(settings: Settings, store: Any | None, *, now: datetime | None = None) -> dict[str, str]:
    """Name -> redacted text of every file in the bundle."""
    from app.doctor import render, run_doctor

    moment = now or get_clock().now()
    zone = resolve_zone(settings.timezone)
    header = (f"generated {iso(moment)} UTC = {moment.astimezone(zone):%Y-%m-%d %H:%M:%S %z} ({settings.timezone})\n"
              f"env={settings.env} mode={settings.mode}\n")
    files: dict[str, str] = {"summary.txt": header + "\n" + versions()}
    try:
        files["doctor.txt"] = render(run_doctor(offline=True))
    except Exception as exc:  # noqa: BLE001 - a broken probe is itself useful in a bug report
        files["doctor.txt"] = f"doctor failed: {type(exc).__name__}: {exc}"
    files["settings.json"] = json.dumps(settings.redacted(), indent=2, default=str)
    if store is not None:
        from automation.health import evaluate_health
        from automation.retention import storage_report, storage_text

        for name, produce in (
            ("health.json", lambda: json.dumps(evaluate_health(store, settings).to_dict(), indent=2, default=str)),
            ("storage.txt", lambda: storage_text(storage_report(store, settings, sizes=False))),
            ("database.txt", lambda: database_state(store)),
        ):
            try:
                files[name] = produce()
            except Exception as exc:  # noqa: BLE001
                files[name] = f"{name} failed: {type(exc).__name__}: {exc}"
    files.update(log_excerpts(settings))
    return {name: redact(text)[:MAX_FILE_BYTES] for name, text in files.items()}


def build_bundle(settings: Settings, store: Any | None = None, *, now: datetime | None = None) -> Path:
    """Write the archive and return its path."""
    moment = now or get_clock().now()
    folder = settings.data_dir / "diag"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (iso(moment) or "").replace("-", "").replace(":", "")
    path = folder / f"diag-{stamp}.tar.gz"
    files = build_files(settings, store, now=moment)
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            data = text.encode("utf-8", "replace")
            info = tarfile.TarInfo(name=f"diag-{stamp}/{name}")
            info.size = len(data)
            info.mtime = int(moment.timestamp())
            info.mode = 0o640
            archive.addfile(info, io.BytesIO(data))
    if sys.platform != "win32":
        path.chmod(0o640)
    log.info("diagnostics bundle written: %s (%d files, %.0f KB)", path, len(files), path.stat().st_size / 1024)
    return path
