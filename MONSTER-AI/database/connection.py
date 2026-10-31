"""SQLite connection factory tuned for a small server.

* WAL journal: readers (dashboard, health) never block the single writer.
* ``busy_timeout``: concurrent writers wait instead of failing with "database is locked".
* Explicit transactions (``isolation_level=None``): Python's implicit transaction handling is avoided, so
  every write path is a visible ``BEGIN IMMEDIATE`` ... ``COMMIT`` block.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

_SAVEPOINT_IDS = itertools.count(1)

PRAGMAS_RW = (
    "PRAGMA busy_timeout = 30000",
    "PRAGMA foreign_keys = ON",
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = FILE",
    "PRAGMA cache_size = -8192",
    "PRAGMA mmap_size = 0",
    "PRAGMA wal_autocheckpoint = 1000",
    "PRAGMA journal_size_limit = 33554432",
)
PRAGMAS_RO = (
    "PRAGMA busy_timeout = 30000",
    "PRAGMA foreign_keys = ON",
    "PRAGMA cache_size = -4096",
    "PRAGMA mmap_size = 0",
    "PRAGMA query_only = ON",
)


def connect(path: str | Path, *, readonly: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    """Open the database. ``path`` may be ``:memory:`` (tests)."""
    target = str(path)
    if target == ":memory:":
        conn = sqlite3.connect(target, timeout=timeout, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    file_path = Path(path)
    if readonly:
        conn = sqlite3.connect(
            f"{file_path.resolve().as_uri()}?mode=ro", uri=True, timeout=timeout, isolation_level=None,
            check_same_thread=False,
        )
        pragmas = PRAGMAS_RO
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not file_path.exists()
        conn = sqlite3.connect(str(file_path), timeout=timeout, isolation_level=None, check_same_thread=False)
        pragmas = PRAGMAS_RW
        if new_file:
            with contextlib.suppress(OSError):
                os.chmod(file_path, 0o600)
    conn.row_factory = sqlite3.Row
    for pragma in pragmas:
        conn.execute(pragma)
    return conn


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection, *, immediate: bool = True, retries: int = 6) -> Iterator[sqlite3.Connection]:
    """Atomic block. Nested use becomes a SAVEPOINT; ``database is locked`` is retried with backoff."""
    if conn.in_transaction:
        name = f"sp_{next(_SAVEPOINT_IDS)}"
        conn.execute(f"SAVEPOINT {name}")
        try:
            yield conn
        except BaseException:
            conn.execute(f"ROLLBACK TO {name}")
            conn.execute(f"RELEASE {name}")
            raise
        else:
            conn.execute(f"RELEASE {name}")
        return

    delay = 0.25
    for attempt in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 4.0)
    try:
        yield conn
    except BaseException:
        with contextlib.suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
