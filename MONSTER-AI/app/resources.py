"""RAM / disk guards for the 1 GB / 8 GB production target.

Everything here degrades gracefully: on platforms where a metric is unavailable the value is ``None`` and the
guard reports ``unknown`` instead of failing the job.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MB = 1024 * 1024


# --------------------------------------------------------------------------------------------- metrics
def memory_info() -> dict[str, float | None]:
    """System memory in MB: total / available / swap_total / swap_free (``None`` when unknown)."""
    info: dict[str, float | None] = {"total_mb": None, "available_mb": None, "swap_total_mb": None, "swap_free_mb": None}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values: dict[str, float] = {}
        try:
            for line in meminfo.read_text().splitlines():
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    values[key] = float(parts[0]) / 1024.0  # kB -> MB
        except (OSError, ValueError):  # pragma: no cover
            return info
        info["total_mb"] = values.get("MemTotal")
        available = values.get("MemAvailable")
        if available is None and "MemFree" in values:
            available = values["MemFree"] + values.get("Cached", 0.0) + values.get("Buffers", 0.0)
        info["available_mb"] = available
        info["swap_total_mb"] = values.get("SwapTotal")
        info["swap_free_mb"] = values.get("SwapFree")
        return info
    if sys.platform == "win32":  # pragma: no cover - dev machines only

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            info["total_mb"] = status.ullTotalPhys / MB
            info["available_mb"] = status.ullAvailPhys / MB
    return info


class _ProcessCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS. Defined once at module level: a ctypes type created per call is never reclaimed
    promptly, so sampling memory every few milliseconds used to leak tens of MB."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


@functools.lru_cache(maxsize=1)
def _windows_api() -> tuple[Any, Any]:  # pragma: no cover - dev machines only
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessCounters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    return kernel32, psapi


def _windows_counters() -> Any:  # pragma: no cover - dev machines only
    kernel32, psapi = _windows_api()
    counters = _ProcessCounters()
    counters.cb = ctypes.sizeof(_ProcessCounters)
    if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return counters
    return None


def rss_mb() -> float | None:
    """Resident set size of the current process in MB."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            pages = int(statm.read_text().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / MB
        except (OSError, ValueError, IndexError):  # pragma: no cover
            return None
    if sys.platform == "win32":  # pragma: no cover - dev machines only
        counters = _windows_counters()
        if counters is not None:
            return counters.WorkingSetSize / MB
    return None


def peak_rss_mb() -> float | None:
    """Peak resident set size (high-water mark) of the current process in MB."""
    status = Path("/proc/self/status")
    if status.exists():
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):  # pragma: no cover
            return None
    if sys.platform == "win32":  # pragma: no cover - dev machines only
        counters = _windows_counters()
        return None if counters is None else counters.PeakWorkingSetSize / MB
    try:  # pragma: no cover - macOS / fallback
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (MB if sys.platform == "darwin" else 1024.0)
    except (ImportError, OSError):  # pragma: no cover
        return rss_mb()


def release_memory() -> None:
    """Return freed memory to the OS now (matters on a 1 GB server): collect garbage and, on glibc, trim the heap."""
    import gc

    gc.collect()
    if sys.platform.startswith("linux"):  # pragma: no cover - exercised on the server only
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


def disk_info(path: str | Path) -> dict[str, float]:
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    return {"total_mb": usage.total / MB, "free_mb": usage.free / MB, "used_mb": usage.used / MB}


def dir_size_mb(path: str | Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total / MB


# --------------------------------------------------------------------------------------------- guards
@dataclass
class ResourceStatus:
    level: str = "ok"  # ok | warn | crit | unknown
    free_disk_mb: float | None = None
    available_ram_mb: float | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.level in {"ok", "unknown"}


_ORDER = {"unknown": 0, "ok": 0, "warn": 1, "crit": 2}


def check_resources(
    data_dir: str | Path,
    *,
    warn_disk_mb: float,
    crit_disk_mb: float,
    min_ram_mb: float,
) -> ResourceStatus:
    status = ResourceStatus()
    disk = disk_info(data_dir)
    status.free_disk_mb = disk["free_mb"]
    if disk["free_mb"] < crit_disk_mb:
        status.level = "crit"
        status.messages.append(f"free disk {disk['free_mb']:.0f} MB < critical {crit_disk_mb:.0f} MB")
    elif disk["free_mb"] < warn_disk_mb:
        status.level = "warn"
        status.messages.append(f"free disk {disk['free_mb']:.0f} MB < warning {warn_disk_mb:.0f} MB")
    mem = memory_info()
    status.available_ram_mb = mem["available_mb"]
    if mem["available_mb"] is None:
        if status.level == "ok":
            status.level = "unknown"
    elif mem["available_mb"] < min_ram_mb:
        status.messages.append(f"available RAM {mem['available_mb']:.0f} MB < minimum {min_ram_mb:.0f} MB")
        if _ORDER["warn"] > _ORDER[status.level]:
            status.level = "warn"
    return status


def prune_directory(path: str | Path, *, max_mb: float | None = None, max_age_days: float | None = None,
                    keep: Iterable[str] = (), dry_run: bool = False) -> int:
    """Delete old files (and oldest-first until under ``max_mb``). Returns the number of files removed
    (or that would be, with ``dry_run``). Files whose *name* is in ``keep`` are never removed, but they still count
    towards the ``max_mb`` budget."""
    base = Path(path)
    if not base.exists():
        return 0
    keep_set = set(keep)
    now = time.time()
    files: list[tuple[float, int, Path]] = []
    kept_bytes = 0
    for root, _dirs, names in os.walk(base):
        for name in names:
            candidate = Path(root) / name
            try:
                stat = candidate.stat()
            except OSError:
                continue
            if name in keep_set:
                kept_bytes += stat.st_size
                continue
            files.append((stat.st_mtime, stat.st_size, candidate))
    files.sort()
    removed = 0
    remaining: list[tuple[float, int, Path]] = []
    for mtime, size, candidate in files:
        if max_age_days is not None and (now - mtime) > max_age_days * 86400:
            try:
                if not dry_run:
                    candidate.unlink()
                removed += 1
                continue
            except OSError:
                pass
        remaining.append((mtime, size, candidate))
    if max_mb is not None:
        total = (sum(size for _m, size, _p in remaining) + kept_bytes) / MB
        for _mtime, size, candidate in remaining:
            if total <= max_mb:
                break
            try:
                if not dry_run:
                    candidate.unlink()
                removed += 1
                total -= size / MB
            except OSError:
                continue
    return removed


def emergency_cleanup(cache_dir: Path, tmp_dir: Path, log_dir: Path | None = None) -> dict[str, int]:
    """Free disk quickly: drop all cache, stale temp files and the oldest rotated logs."""
    removed = {
        "cache": prune_directory(cache_dir, max_mb=0),
        "tmp": prune_directory(tmp_dir, max_age_days=0.04),
        "logs": 0,
    }
    if log_dir is not None and Path(log_dir).exists():
        for rotated in sorted(Path(log_dir).glob("*.log.*"), key=lambda p: p.stat().st_mtime)[:-1]:
            try:
                rotated.unlink()
                removed["logs"] += 1
            except OSError:
                continue
    log.warning("emergency cleanup removed %s", removed)
    return removed


class RssWatchdog:
    """Aborts the process when RSS exceeds a ceiling, so one runaway job can never trigger the kernel OOM
    killer against the whole 1 GB server. The abort uses ``os._exit(137)``; SQLite WAL recovery makes this safe."""

    def __init__(self, limit_mb: float, *, interval: float = 2.0, on_exceed: Callable[[float], None] | None = None) -> None:
        self.limit_mb = limit_mb
        self.interval = interval
        self.on_exceed = on_exceed or self._die
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb = 0.0

    @staticmethod
    def _die(current: float) -> None:  # pragma: no cover - terminates the process
        log.critical("RSS %.0f MB exceeded ceiling; aborting job", current)
        logging_flush()
        os._exit(137)

    def check_once(self) -> bool:
        current = rss_mb()
        if current is None:
            return False
        self.peak_mb = max(self.peak_mb, current)
        if current > self.limit_mb:
            self.on_exceed(current)
            return True
        return False

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.check_once()

    def start(self) -> RssWatchdog:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="rss-watchdog", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


def logging_flush() -> None:  # pragma: no cover - best effort
    for handler in logging.getLogger().handlers:
        with contextlib.suppress(Exception):
            handler.flush()
