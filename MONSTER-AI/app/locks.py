"""Cross-platform advisory file locks.

The OS releases ``flock`` locks when a process dies, so a crashed job can never leave a stale lock behind
(unlike PID files). Used to guarantee that the same job - or any two heavy jobs - never overlap.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

try:  # POSIX
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
    import msvcrt  # type: ignore[import-not-found]


class LockBusy(RuntimeError):
    """Raised when a lock is held by another process."""


class FileLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, *, blocking: bool = False, timeout: float | None = None, poll: float = 0.2) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                self._try_lock(fd)
            except OSError:
                if not blocking or (deadline is not None and time.monotonic() >= deadline):
                    os.close(fd)
                    return False
                time.sleep(poll)
                continue
            break
        self._fd = fd
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:  # pragma: no cover - diagnostics only
            pass
        return True

    @staticmethod
    def _try_lock(fd: int) -> None:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:  # pragma: no cover - Windows
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:  # pragma: no cover - Windows
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:  # pragma: no cover
            pass
        finally:
            os.close(fd)

    def holder_pid(self) -> int | None:
        if self._fd is not None:
            return os.getpid()
        try:
            return int(self.path.read_text().strip() or 0) or None
        except (OSError, ValueError):
            return None

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise LockBusy(f"lock {self.path} is held by pid {self.holder_pid()}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def lock_dir(data_dir: Path) -> Path:
    path = Path(data_dir) / "locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_lock(data_dir: Path, name: str) -> FileLock:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return FileLock(lock_dir(data_dir) / f"{safe}.lock")


def heavy_lock(data_dir: Path) -> FileLock:
    """Global lock: at most one heavy job (backfill, training, evaluation) at a time."""
    return FileLock(lock_dir(data_dir) / "heavy.lock")
