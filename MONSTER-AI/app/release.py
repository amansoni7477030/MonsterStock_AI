"""Which code produced a result: ``1.0.0+3fa9c1d2`` = the release number plus a fingerprint of the code itself.

The server runs a rsync'ed release tree without ``.git``, so a commit id is not available there; a hash of the Python sources and
the seed data is, and it identifies the code exactly (two trees with the same id behave the same). It is stamped on every frozen
snapshot (``code_version``) and on every registered model (``code_commit``), so any prediction can be traced to the code that made
it. ``MONSTER_RELEASE`` overrides it (a CI build can pass the git commit)."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from app import __version__

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("app", "automation", "backtesting", "dashboard", "database", "event_graph", "features", "ingestion", "models",
             "news_intelligence", "notifications", "ranking")
DATA_DIRS = ("data/seeds",)
CODE_SUFFIXES = {".py", ".sql", ".json", ".txt", ".yaml", ".yml"}
SKIP_PARTS = {"__pycache__", ".git", ".venv", "venv", "node_modules"}


def code_fingerprint(root: Path = ROOT) -> str:
    """SHA-256 (first 8 hex digits) over the path and content of every source / migration / seed file under ``root``."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for name in (*CODE_DIRS, *DATA_DIRS):
        base = root / name
        if base.is_dir():
            files += [p for p in base.rglob("*") if p.is_file() and p.suffix in CODE_SUFFIXES
                      and not (set(p.relative_to(root).parts) & SKIP_PARTS)]
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:8]


@lru_cache(maxsize=1)
def release_id() -> str:
    """``<version>+<fingerprint>`` of the running code (computed once per process, about 20 ms)."""
    override = os.environ.get("MONSTER_RELEASE", "").strip()
    if override:
        return f"{__version__}+{override[:40]}"
    try:
        return f"{__version__}+{code_fingerprint()}"
    except OSError:  # an unreadable tree must never stop a scan from being frozen
        return f"{__version__}+unknown"
