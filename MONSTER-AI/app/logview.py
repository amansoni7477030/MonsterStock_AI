"""Reading the daily log files: what ``python -m app.cli logs`` shows.

Files are ``<role>-YYYY-MM-DD[.N].log`` (see ``app.logging_setup``). A record is a header line
(``<stamp> LEVEL role[pid] [job run] logger: message``) plus any continuation lines (tracebacks); filters always keep a record
whole.
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from app.clock import UTC
from app.logging_setup import ERRORS_ROLE

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
FILE_NAME = re.compile(r"^(?P<role>[a-z][a-z0-9_]*)-(?P<day>\d{4}-\d{2}-\d{2})(?:\.(?P<index>\d+))?\.log$")
HEADER = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4}) (?P<level>[A-Z]+)\s+(?:(?P<role>\w+)\[(?P<pid>\d+)\] )?"
    r"\[(?P<job>[^\]]*)\] (?P<logger>[^:]+): (?P<msg>.*)$")
DURATION = re.compile(r"^(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>[smhdw])$")
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
TAIL_BLOCK = 64 * 1024


@dataclass(frozen=True)
class LogFile:
    path: Path
    role: str
    day: str
    index: int
    size: int


@dataclass(frozen=True)
class Record:
    lines: tuple[str, ...]
    level: str = ""
    job: str = ""
    stamp: datetime | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def list_log_files(log_dir: Path | str) -> list[LogFile]:
    """Every daily log file, oldest day first (then role, then index)."""
    base = Path(log_dir)
    if not base.is_dir():
        return []
    found = []
    for path in base.iterdir():
        match = FILE_NAME.match(path.name)
        if match and path.is_file():
            found.append(LogFile(path, match["role"], match["day"], int(match["index"] or 0), path.stat().st_size))
    return sorted(found, key=lambda f: (f.day, f.role, f.index))


def select_files(files: Iterable[LogFile], *, role: str | None = None, day: str | None = None) -> list[LogFile]:
    """``role`` None / "all" = every role; ``day`` None = the newest day that has any file for the role."""
    wanted = [f for f in files if role in (None, "all") or f.role == role]
    if day is None and wanted:
        day = max(f.day for f in wanted)
    return [f for f in wanted if f.day == day] if day else []


def parse_stamp(text: str) -> datetime | None:
    """``2026-11-02T08:30:05.123+0530`` -> aware datetime (the offset is part of the stamp)."""
    try:
        return datetime.fromisoformat(f"{text[:-5]}{text[-5:-2]}:{text[-2:]}")
    except ValueError:
        return None


def parse_since(value: str, now: datetime) -> datetime:
    """``90m`` / ``2h`` / ``1d`` before ``now``, or an ISO date / datetime (naive values are read as UTC)."""
    match = DURATION.match(value.strip().lower())
    if match:
        return now - timedelta(seconds=float(match["n"]) * UNITS[match["unit"]])
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def tail_lines(path: Path, count: int) -> list[str]:
    """The last ``count`` lines of a file, reading from the end (log files can be tens of MB)."""
    if count <= 0:
        return []
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        position, data = handle.tell(), b""
        while position > 0 and data.count(b"\n") <= count:
            step = min(TAIL_BLOCK, position)
            position -= step
            handle.seek(position)
            data = handle.read(step) + data
    return data.decode("utf-8", "replace").splitlines()[-count:]


def iter_lines(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def group_records(lines: Iterable[str]) -> Iterator[Record]:
    """Join continuation lines (tracebacks) to the header line above them."""
    current: list[str] = []
    meta: tuple[str, str, datetime | None] = ("", "", None)

    def flush() -> Record | None:
        return Record(tuple(current), *meta) if current else None

    for line in lines:
        match = HEADER.match(line)
        if match:
            done = flush()
            if done:
                yield done
            current = [line]
            meta = (match["level"], match["job"], parse_stamp(match["ts"]))
        else:
            current.append(line)
    done = flush()
    if done:
        yield done


def filter_records(records: Iterable[Record], *, min_level: str | None = None, job: str | None = None,
                   grep: str | None = None, since: datetime | None = None) -> Iterator[Record]:
    floor = LEVELS.index(min_level.upper()) if min_level else 0
    needle = grep.lower() if grep else None
    for record in records:
        if min_level and record.level not in LEVELS:
            continue  # continuation lines without their header (a cut-off tail)
        if record.level in LEVELS and LEVELS.index(record.level) < floor:
            continue
        if job and job not in record.job:
            continue
        if since is not None and record.stamp is not None and record.stamp < since:
            continue
        if needle and needle not in record.text.lower():
            continue
        yield record


def read_records(files: Iterable[LogFile], *, limit: int = 100, **filters: object) -> list[Record]:
    """The newest ``limit`` records of ``files`` that pass the filters, merged by time stamp. Files are streamed, so memory
    stays bounded by ``limit`` records per file. A record without a stamp (an orphaned traceback) sorts right after the record
    before it."""
    keyed: list[tuple[float, int, int, Record]] = []
    for number, file in enumerate(files):
        last = 0.0
        kept = deque(filter_records(group_records(iter_lines(file.path)), **filters), maxlen=max(limit, 1))  # type: ignore[arg-type]
        for seq, record in enumerate(kept):
            last = record.stamp.timestamp() if record.stamp else last
            keyed.append((last, number, seq, record))
    keyed.sort(key=lambda item: item[:3])
    return [item[3] for item in keyed][-max(limit, 1):]


def follow(files_for: Callable[[], list[LogFile]], emit: Callable[[str], None], *, sleep: Callable[[float], None] = time.sleep,
           poll: float = 1.0, max_polls: int | None = None) -> None:
    """Print lines appended after the call, until interrupted (``max_polls`` bounds it for tests). New files (a new day, a
    new role) are picked up from their start."""
    offsets: dict[Path, int] = {}
    first = True
    polls = 0
    while max_polls is None or polls < max_polls:
        for file in files_for():
            size = file.path.stat().st_size if file.path.exists() else 0
            if file.path not in offsets:
                offsets[file.path] = size if first else 0
            if size > offsets[file.path]:
                with open(file.path, "rb") as handle:
                    handle.seek(offsets[file.path])
                    chunk = handle.read(size - offsets[file.path])
                offsets[file.path] = size
                for line in chunk.decode("utf-8", "replace").splitlines():
                    emit(line)
        first = False
        polls += 1
        if max_polls is None or polls < max_polls:
            sleep(poll)


def summary(files: list[LogFile]) -> str:
    if not files:
        return "no log files yet"
    lines = [f"{'day':<12}{'role':<12}{'size':>10}  file"]
    for f in files:
        lines.append(f"{f.day:<12}{f.role:<12}{f.size / 1024:>8.1f} KB  {f.path.name}")
    return "\n".join(lines)


__all__ = ["ERRORS_ROLE", "LEVELS", "LogFile", "Record", "filter_records", "follow", "group_records", "list_log_files",
           "parse_since", "parse_stamp", "read_records", "select_files", "summary", "tail_lines"]
