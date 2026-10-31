from __future__ import annotations

import os
import time

import pytest

from app.locks import FileLock, LockBusy, heavy_lock, job_lock
from app.resources import (
    RssWatchdog,
    check_resources,
    dir_size_mb,
    disk_info,
    emergency_cleanup,
    memory_info,
    peak_rss_mb,
    prune_directory,
    rss_mb,
)


def test_lock_is_exclusive_and_released(tmp_path):
    a, b = FileLock(tmp_path / "x.lock"), FileLock(tmp_path / "x.lock")
    assert a.acquire() and a.held
    assert not b.acquire()  # second holder refused
    with pytest.raises(LockBusy):
        with b:
            pass
    a.release()
    assert not a.held
    assert b.acquire()
    b.release()


def test_lock_blocking_with_timeout(tmp_path):
    a, b = FileLock(tmp_path / "y.lock"), FileLock(tmp_path / "y.lock")
    a.acquire()
    started = time.monotonic()
    assert not b.acquire(blocking=True, timeout=0.4, poll=0.05)
    assert time.monotonic() - started >= 0.35
    a.release()


def test_lock_context_manager_and_pid(tmp_path):
    lock = FileLock(tmp_path / "z.lock")
    with lock:
        assert lock.holder_pid() == os.getpid()
    assert lock.acquire() and lock.acquire()  # re-acquire while held is a no-op success
    lock.release()
    lock.release()  # double release is harmless


def test_named_locks_are_sanitised(tmp_path):
    assert job_lock(tmp_path, "final scan/../x").path.parent.name == "locks"
    assert heavy_lock(tmp_path).path.name == "heavy.lock"


def test_metrics_helpers_return_sane_values(tmp_path):
    mem = memory_info()
    assert set(mem) == {"total_mb", "available_mb", "swap_total_mb", "swap_free_mb"}
    assert disk_info(tmp_path / "does" / "not" / "exist")["free_mb"] > 0
    assert rss_mb() is None or rss_mb() > 0
    assert peak_rss_mb() is None or peak_rss_mb() > 0
    (tmp_path / "f.bin").write_bytes(b"x" * 2048)
    assert dir_size_mb(tmp_path) > 0


def test_check_resources_levels(tmp_path):
    free = disk_info(tmp_path)["free_mb"]
    assert check_resources(tmp_path, warn_disk_mb=1, crit_disk_mb=0.5, min_ram_mb=0).level in {"ok", "unknown"}
    warn = check_resources(tmp_path, warn_disk_mb=free + 10, crit_disk_mb=1, min_ram_mb=0)
    assert warn.level == "warn" and warn.messages
    crit = check_resources(tmp_path, warn_disk_mb=free + 20, crit_disk_mb=free + 10, min_ram_mb=0)
    assert crit.level == "crit" and not crit.ok


def test_prune_by_age_and_size(tmp_path):
    old = tmp_path / "old.bin"
    old.write_bytes(b"x" * 1000)
    os.utime(old, (time.time() - 10 * 86400,) * 2)
    for i in range(3):
        (tmp_path / f"new{i}.bin").write_bytes(b"y" * 1_000_000)
        os.utime(tmp_path / f"new{i}.bin", (time.time() - i,) * 2)
    assert prune_directory(tmp_path, max_age_days=5) == 1
    assert not old.exists()
    removed = prune_directory(tmp_path, max_mb=1.5)  # keep ~1 MB -> oldest two removed
    assert removed == 2 and len(list(tmp_path.iterdir())) == 1
    assert prune_directory(tmp_path / "missing", max_mb=0) == 0


def test_emergency_cleanup(tmp_path):
    cache, tmp, logs = tmp_path / "cache", tmp_path / "tmp", tmp_path / "logs"
    for d in (cache, tmp, logs):
        d.mkdir()
    (cache / "a").write_bytes(b"1" * 10)
    (logs / "monster.log").write_text("live")
    (logs / "monster.log.1").write_text("old")
    (logs / "monster.log.2").write_text("older")
    removed = emergency_cleanup(cache, tmp, logs)
    assert removed["cache"] == 1 and removed["logs"] >= 1
    assert (logs / "monster.log").exists()


def test_rss_watchdog_calls_back_when_limit_exceeded():
    hits = []
    dog = RssWatchdog(limit_mb=0.001, on_exceed=hits.append)
    if rss_mb() is None:
        pytest.skip("RSS not measurable on this platform")
    assert dog.check_once() is True and hits and dog.peak_mb > 0
    assert RssWatchdog(limit_mb=10**9, on_exceed=hits.append).check_once() is False
    dog.start()
    dog.stop()
