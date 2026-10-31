"""Safe, atomic edits of a dotenv-style file (used by ``telegram-setup --write-env``)."""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

_KEY = re.compile(r"[A-Z][A-Z0-9_]*")


def upsert_env(path: str | Path, updates: Mapping[str, str]) -> None:
    """Set ``KEY=value`` lines, keeping every other line, comments and the file mode. Values may not contain newlines."""
    target = Path(path)
    for key, value in updates.items():
        if not _KEY.fullmatch(key):
            raise ValueError(f"invalid variable name {key!r}")
        if any(ch in value for ch in "\r\n\0"):
            raise ValueError(f"value of {key} contains a control character")
    lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        key = None
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.removeprefix("export ").split("=", 1)[0].strip()
        if key in updates:
            if key not in seen:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
            continue
        out.append(line)
    out += [f"{key}={value}" for key, value in updates.items() if key not in seen]
    mode = target.stat().st_mode & 0o777 if target.exists() else 0o600
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".env-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(out) + "\n")
        with contextlib.suppress(OSError):
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
