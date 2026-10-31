from __future__ import annotations

import gzip
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app.clock import FakeClock
from app.errors import DatabaseCorrupt
from database import backup as bk
from database.store import Store


def _populate(path, rows=50):
    store = Store.open(path)
    for i in range(rows):
        store.kv_set(f"k{i}", "v" * 100)
    store.close()


def test_backup_roundtrip_with_checksum_and_restore(tmp_path):
    db = tmp_path / "live.db"
    _populate(db)
    result = bk.backup_database(db, tmp_path / "bk")
    assert result.path.exists() and result.path.name.endswith(".db.gz") and not result.encrypted
    assert (result.path.with_name(result.path.name + ".sha256")).exists()
    assert bk.verify_backup(result.path)

    db.write_bytes(b"this is not sqlite")  # catastrophic corruption
    quarantined = bk.restore_backup(result.path, db)
    assert quarantined is not None and quarantined.read_bytes() == b"this is not sqlite"
    store = Store.open(db)
    assert store.kv_get("k7") == "v" * 100
    store.close()


def test_tampered_backup_is_rejected(tmp_path):
    db = tmp_path / "live.db"
    _populate(db)
    result = bk.backup_database(db, tmp_path / "bk")
    with open(result.path, "ab") as handle:
        handle.write(b"tamper")
    assert not bk.verify_backup(result.path)
    with pytest.raises(DatabaseCorrupt):
        bk.restore_backup(result.path, db)
    assert not bk.verify_backup(tmp_path / "bk" / "missing.db.gz")


def test_non_database_payload_fails_integrity(tmp_path):
    bad = tmp_path / "monster-20261101T000000Z.db.gz"
    with gzip.open(bad, "wb") as handle:
        handle.write(b"garbage that is not a database file at all")
    assert not bk.verify_backup(bad)


def test_rotation_keeps_daily_and_weekly_generations(tmp_path):
    db = tmp_path / "live.db"
    _populate(db, rows=3)
    clock = FakeClock(datetime(2026, 9, 1, 2, 0, tzinfo=timezone.utc))
    for _ in range(40):  # 40 consecutive daily backups
        bk.backup_database(db, tmp_path / "bk", keep_daily=7, keep_weekly=4, clock=clock)
        clock.advance(86400)
    kept = bk.list_backups(tmp_path / "bk")
    names = [p.name for p in kept]
    assert 7 <= len(kept) <= 11  # 7 newest days + up to 4 older weekly generations
    assert names[-1].startswith("monster-20261010")  # newest retained
    assert len(list((tmp_path / "bk").glob("*.sha256"))) == len(kept)  # sidecars follow their backups


def test_restore_latest_valid_skips_corrupt_generations(tmp_path):
    db = tmp_path / "live.db"
    _populate(db, rows=5)
    clock = FakeClock(datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc))
    good = bk.backup_database(db, tmp_path / "bk", clock=clock)
    clock.advance(86400)
    newest = bk.backup_database(db, tmp_path / "bk", clock=clock)
    newest.path.write_bytes(b"corrupted")
    os.remove(db)
    used = bk.restore_latest_valid(db, tmp_path / "bk")
    assert used == good.path
    assert Store.open(db).kv_get("k1") is not None
    for p in (tmp_path / "bk").glob("monster-*"):
        p.write_bytes(b"x")
    with pytest.raises(DatabaseCorrupt):
        bk.restore_latest_valid(db, tmp_path / "bk")


def test_backup_of_corrupt_live_db_is_refused(tmp_path):
    db = tmp_path / "live.db"
    db.write_bytes(b"\x00" * 4096)
    with pytest.raises((DatabaseCorrupt, sqlite3.DatabaseError)):
        bk.backup_database(db, tmp_path / "bk")
    assert not list((tmp_path / "bk").glob("monster-*"))
    with pytest.raises(FileNotFoundError):
        bk.backup_database(tmp_path / "nope.db", tmp_path / "bk")


def test_encryption_failure_falls_back_to_plain_with_flag(tmp_path, monkeypatch):
    db = tmp_path / "live.db"
    _populate(db, rows=2)

    def boom(*_a, **_k):
        raise OSError("gpg missing")

    monkeypatch.setattr(bk.subprocess, "run", boom)
    result = bk.backup_database(db, tmp_path / "bk", passphrase="secret-pass")
    assert result.encrypted is False and bk.verify_backup(result.path)


def test_name_parsing_and_retention_helpers(tmp_path):
    assert bk._parse_name("monster-20261101T020000Z.db.gz") == datetime(2026, 11, 1, 2, 0, tzinfo=timezone.utc)
    assert bk._parse_name("monster-20261101T020000Z.db.gz.gpg") is not None
    assert bk._parse_name("other.txt") is None
    assert bk.list_backups(tmp_path / "missing") == []
    paths = [tmp_path / f"monster-202611{d:02d}T020000Z.db.gz" for d in range(1, 6)]
    keep = bk._retention(paths, keep_daily=2, keep_weekly=0)
    assert keep == set(paths[-2:])
    assert timedelta(0) == timedelta(0)
