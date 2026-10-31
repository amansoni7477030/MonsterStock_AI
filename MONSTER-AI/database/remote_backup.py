"""Disaster recovery from the off-box copy: fetch the newest database backup from object storage and restore it."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.config import Settings
from app.errors import DataUnavailable
from database.backup import restore_backup, retained_names, verify_backup
from database.object_store import ObjectStore

log = logging.getLogger(__name__)

BACKUP_PREFIX = "backups/"
_NAME = re.compile(r"^monster-\d{8}T\d{6}Z\.db\.gz(\.gpg)?$")
MAX_BACKUP_BYTES = 2 * 1024 * 1024 * 1024
MAX_CANDIDATES = 3  # newest backups tried before giving up


def _remote_names(store: ObjectStore) -> dict[str, str]:
    """Our own backups in the store: file name -> object key (anything else under the prefix is ignored)."""
    names = {o.key.rsplit("/", 1)[-1]: o.key for o in store.list(BACKUP_PREFIX)}
    return {name: key for name, key in names.items() if _NAME.match(name)}


def fetch_latest_backup(store: ObjectStore, dest_dir: str | Path, passphrase: str | None = None) -> Path | None:
    """Download the newest off-box backup that verifies into ``dest_dir`` (an unreadable one falls back to the previous
    one); ``None`` when there is no usable backup."""
    remote = _remote_names(store)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name in sorted(remote, reverse=True)[:MAX_CANDIDATES]:  # names embed a UTC timestamp, so they sort chronologically
        target = dest / name
        try:
            store.get_to_file(remote[name], target, max_bytes=MAX_BACKUP_BYTES)
        except (DataUnavailable, OSError) as exc:
            log.warning("remote backup %s could not be downloaded: %s", name, exc)
            target.unlink(missing_ok=True)
            continue
        if verify_backup(target, passphrase):
            return target
        log.error("remote backup %s failed verification; trying the previous one", name)
        target.unlink(missing_ok=True)
    return None


def prune_remote_backups(store: ObjectStore, *, keep_daily: int, keep_weekly: int) -> list[str]:
    """Apply the local retention policy to the off-box copies so storage cannot grow without bound. Only objects named
    like our own backups are ever deleted; the newest backup is always kept."""
    remote = _remote_names(store)
    removed = []
    for name in sorted(set(remote) - retained_names(list(remote), keep_daily, keep_weekly)):
        store.delete(remote[name])
        removed.append(name)
    return removed


def restore_from_remote(settings: Settings, store: ObjectStore) -> Path | None:
    """Fetch the newest usable off-box backup and restore it into ``settings.db_path`` (the previous file is kept as
    ``*.corrupt-<ts>``); returns the backup path, or ``None`` when no usable backup exists."""
    path = fetch_latest_backup(store, settings.backup_dir, settings.backup_passphrase)
    if path is None:
        return None
    restore_backup(path, settings.db_path, settings.backup_passphrase)
    return path
