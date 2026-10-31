"""Consistent, verified, rotated SQLite backups.

Uses SQLite's online backup API (safe while the writer is active), verifies the copy with a full
``integrity_check`` before accepting it, writes a SHA-256 sidecar, and rotates daily + weekly generations.
Restore verifies before touching the live database and keeps the bad file aside for forensics.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.clock import Clock, get_clock
from app.errors import DatabaseCorrupt, ResourceExhausted
from app.resources import disk_info

log = logging.getLogger(__name__)

_NAME = re.compile(r"^monster-(\d{8})T(\d{6})Z\.db\.gz(\.gpg)?$")
CHUNK = 1024 * 1024


@dataclass(frozen=True)
class BackupResult:
    path: Path
    size_bytes: int
    sha256: str
    encrypted: bool
    pruned: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_name(name: str) -> datetime | None:
    match = _NAME.match(name)
    if not match:
        return None
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def list_backups(backup_dir: str | Path) -> list[Path]:
    """Backups sorted oldest -> newest."""
    base = Path(backup_dir)
    if not base.exists():
        return []
    found = [(ts, p) for p in base.iterdir() if (ts := _parse_name(p.name)) is not None]
    return [p for _ts, p in sorted(found)]


def _integrity_ok(path: Path) -> bool:
    conn = None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        return [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()] == ["ok"]
    except sqlite3.DatabaseError:
        return False
    finally:
        if conn is not None:
            conn.close()


def _gpg(args: list[str], passphrase: str) -> None:
    command = ["gpg", "--batch", "--yes", "--quiet", "--pinentry-mode", "loopback", "--passphrase-fd", "0", *args]
    result = subprocess.run(command, input=passphrase.encode(), capture_output=True, timeout=300, check=False)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"gpg failed: {result.stderr.decode(errors='replace')[:200]}")


def _retention(paths: list[Path], keep_daily: int, keep_weekly: int) -> set[Path]:
    """Newest backup of each of the last ``keep_daily`` days + of each of the last ``keep_weekly`` ISO weeks."""
    by_day: dict[str, Path] = {}
    by_week: dict[str, Path] = {}
    for path in paths:  # ascending, later entries overwrite earlier ones
        ts = _parse_name(path.name)
        if ts is None:  # pragma: no cover - list_backups filters
            continue
        by_day[ts.strftime("%Y-%m-%d")] = path
        iso_year, iso_week, _ = ts.isocalendar()
        by_week[f"{iso_year}-W{iso_week:02d}"] = path
    keep = {by_day[k] for k in sorted(by_day)[-max(keep_daily, 1):]}
    keep |= {by_week[k] for k in sorted(by_week)[-max(keep_weekly, 0):]} if keep_weekly > 0 else set()
    return keep


def retained_names(names: list[str], keep_daily: int, keep_weekly: int) -> set[str]:
    """Which of these backup file names (any order) the retention policy keeps; unparseable names are ignored."""
    paths = sorted((Path(n) for n in names if _parse_name(n) is not None), key=lambda p: _parse_name(p.name))
    return {p.name for p in _retention(paths, keep_daily, keep_weekly)}


def backup_database(
    db_path: str | Path,
    backup_dir: str | Path,
    *,
    keep_daily: int = 7,
    keep_weekly: int = 4,
    passphrase: str | None = None,
    clock: Clock | None = None,
) -> BackupResult:
    source = Path(db_path)
    target_dir = Path(backup_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(target_dir, 0o700)
    if not source.exists():
        raise FileNotFoundError(str(source))

    needed_mb = (source.stat().st_size / 1048576.0) * 2.5 + 16
    if disk_info(target_dir)["free_mb"] < needed_mb:
        raise ResourceExhausted(f"not enough free disk for a backup (need ~{needed_mb:.0f} MB)")

    stamp = (clock or get_clock()).now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final = target_dir / f"monster-{stamp}.db.gz"
    with tempfile.TemporaryDirectory(dir=target_dir, prefix=".tmp-backup-") as tmp_name:
        tmp = Path(tmp_name)
        copy = tmp / "copy.db"
        src = sqlite3.connect(str(source), timeout=60)
        try:
            dst = sqlite3.connect(str(copy))
            try:
                src.backup(dst, pages=2000, sleep=0.02)
            finally:
                dst.close()
        finally:
            src.close()
        if not _integrity_ok(copy):
            raise DatabaseCorrupt("backup copy failed integrity_check; live database may be corrupt")

        gz = tmp / "copy.db.gz"
        with open(copy, "rb") as fin, gzip.open(gz, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout, CHUNK)
        copy.unlink()

        encrypted = False
        artifact = gz
        if passphrase:
            try:
                enc = tmp / "copy.db.gz.gpg"
                _gpg(["--symmetric", "--cipher-algo", "AES256", "-o", str(enc), str(gz)], passphrase)
                artifact, encrypted = enc, True
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                log.error("backup encryption unavailable (%s); storing UNENCRYPTED backup", exc)
        if encrypted:
            final = final.with_name(final.name + ".gpg")
        digest = _sha256(artifact)
        os.replace(artifact, final)
        final.with_name(final.name + ".sha256").write_text(f"{digest}  {final.name}\n")
        with contextlib.suppress(OSError):
            os.chmod(final, 0o600)

    keep = _retention(list_backups(target_dir), keep_daily, keep_weekly)
    pruned: list[str] = []
    for path in list_backups(target_dir):
        if path not in keep and path != final:
            for victim in (path, path.with_name(path.name + ".sha256")):
                with contextlib.suppress(OSError):
                    victim.unlink()
            pruned.append(path.name)
    log.info("backup written %s (%d bytes, encrypted=%s, pruned=%d)", final.name, final.stat().st_size, encrypted, len(pruned))
    return BackupResult(final, final.stat().st_size, digest, encrypted, pruned)


def _decode(path: Path, out: Path, passphrase: str | None) -> None:
    source = path
    tmp_plain: Path | None = None
    try:
        if path.name.endswith(".gpg"):
            if not passphrase:
                raise RuntimeError("backup is encrypted but no passphrase is configured")
            tmp_plain = out.with_suffix(".gz.tmp")
            _gpg(["--decrypt", "-o", str(tmp_plain), str(path)], passphrase)
            source = tmp_plain
        with gzip.open(source, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout, CHUNK)
    finally:
        if tmp_plain is not None:
            with contextlib.suppress(OSError):
                tmp_plain.unlink()


def verify_backup(path: str | Path, passphrase: str | None = None) -> bool:
    """Checksum sidecar (when present) + decompress + full integrity check."""
    target = Path(path)
    if not target.exists():
        return False
    sidecar = target.with_name(target.name + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().split()[0]
        if _sha256(target) != expected:
            log.error("backup checksum mismatch: %s", target.name)
            return False
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".verify-") as tmp:
        out = Path(tmp) / "restored.db"
        try:
            _decode(target, out, passphrase)
        except (OSError, EOFError, RuntimeError, gzip.BadGzipFile, subprocess.SubprocessError) as exc:
            log.error("backup %s unreadable: %s", target.name, exc)
            return False
        return _integrity_ok(out)


def restore_backup(backup: str | Path, db_path: str | Path, passphrase: str | None = None, clock: Clock | None = None) -> Path | None:
    """Replace ``db_path`` with the verified ``backup``. The previous file is kept as ``*.corrupt-<ts>``.
    Returns the quarantined path (or ``None`` when there was no previous database)."""
    backup_path = Path(backup)
    target = Path(db_path)
    if not verify_backup(backup_path, passphrase):
        raise DatabaseCorrupt(f"backup {backup_path.name} failed verification; refusing to restore")
    stamp = (clock or get_clock()).now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantined: Path | None = None
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".restoring")
    _decode(backup_path, staged, passphrase)
    if target.exists():
        quarantined = target.with_name(f"{target.name}.corrupt-{stamp}")
        os.replace(target, quarantined)
    for suffix in ("-wal", "-shm"):
        side = target.with_name(target.name + suffix)
        if side.exists():
            with contextlib.suppress(OSError):
                os.replace(side, side.with_name(side.name + f".corrupt-{stamp}"))
    os.replace(staged, target)
    with contextlib.suppress(OSError):
        os.chmod(target, 0o600)
    log.warning("database restored from %s (previous file kept: %s)", backup_path.name, quarantined)
    return quarantined


def restore_latest_valid(db_path: str | Path, backup_dir: str | Path, passphrase: str | None = None) -> Path | None:
    """Restore the newest backup that passes verification. Raises ``DatabaseCorrupt`` if none does."""
    for candidate in reversed(list_backups(backup_dir)):
        if verify_backup(candidate, passphrase):
            restore_backup(candidate, db_path, passphrase)
            return candidate
        log.error("skipping invalid backup %s", candidate.name)
    raise DatabaseCorrupt("no valid backup available")
