"""Forward-only SQL migrations with checksums.

Each ``NNNN_name.sql`` is applied once, statement by statement inside one transaction (so a failure leaves
no half-applied schema). A modified, already-applied migration is refused: schema history is append-only.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.clock import iso, utcnow
from app.errors import MonsterError
from database.connection import transaction

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_NAME = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


class MigrationError(MonsterError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


def discover(directory: Path | None = None) -> list[Migration]:
    directory = directory or MIGRATIONS_DIR
    found: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _NAME.match(path.name)
        if not match:
            raise MigrationError(f"bad migration file name: {path.name}")
        sql = path.read_text(encoding="utf-8")
        found.append(Migration(int(match.group(1)), match.group(2), sql, hashlib.sha256(sql.encode()).hexdigest()))
    versions = [m.version for m in found]
    if versions != sorted(set(versions)):
        raise MigrationError("duplicate or unordered migration versions")
    return found


def iter_statements(script: str) -> Iterator[str]:
    """Split a SQL script into complete statements (trigger bodies containing ';' are kept intact)."""
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            buffer = ""
            if statement:
                yield statement
    tail = "\n".join(
        ln for ln in buffer.splitlines() if ln.strip() and not ln.strip().startswith("--")
    ).strip()
    if tail:
        raise MigrationError(f"incomplete SQL statement at end of script: {tail[:80]!r}")


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply_migrations(conn: sqlite3.Connection, directory: Path | None = None) -> list[int]:
    """Apply pending migrations; returns the versions applied by this call."""
    migrations = discover(directory)
    applied_now: list[int] = []
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    for migration in migrations:
        with transaction(conn):
            row = conn.execute("SELECT checksum FROM schema_migrations WHERE version = ?", (migration.version,)).fetchone()
            if row is not None:
                if row["checksum"] != migration.checksum:
                    raise MigrationError(
                        f"migration {migration.version:04d}_{migration.name} was modified after being applied"
                    )
                continue
            for statement in iter_statements(migration.sql):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
                (migration.version, migration.name, migration.checksum, iso(utcnow())),
            )
            conn.execute(f"PRAGMA user_version = {int(migration.version)}")
            applied_now.append(migration.version)
    return applied_now


def check_integrity(conn: sqlite3.Connection, *, quick: bool = True) -> bool:
    """``PRAGMA quick_check`` / ``integrity_check``; True when the database is healthy."""
    pragma = "quick_check" if quick else "integrity_check"
    try:
        rows = [r[0] for r in conn.execute(f"PRAGMA {pragma}").fetchall()]
    except sqlite3.DatabaseError:
        return False
    return rows == ["ok"]
