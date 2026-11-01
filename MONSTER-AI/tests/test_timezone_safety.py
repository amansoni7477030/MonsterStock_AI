"""The code must behave identically on a server in any time zone.

Nothing may depend on the machine's local zone: instants are UTC (``app.clock``), exchange logic is IST, and display zones are
explicit. This test walks every production module and fails on the APIs that silently read the server's zone:

* ``datetime.now()`` / ``today()`` / ``utcnow()`` / ``fromtimestamp(x)`` without an explicit tz
* ``.astimezone()`` without an argument (converts to the *local* zone)
* ``time.localtime`` / ``mktime`` / ``ctime`` / ``asctime`` / ``strftime(fmt)`` without a time tuple
* ``datetime.strptime`` whose naive result is used as an instant
* SQLite ``date('now')`` / ``'localtime'`` and ``CURRENT_DATE`` style defaults that mix UTC dates into IST logic
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "venv", "tests", "__pycache__", "node_modules", ".git", "build", "dist", ".ruff_cache"}

DATETIME_RECEIVERS = {"datetime", "date", "Timestamp", "pd.Timestamp", "pandas.Timestamp", "datetime.datetime", "datetime.date",
                      "dt.datetime", "dt.date"}
NAIVE_WHEN_BARE = {"now", "today", "utcnow", "utcfromtimestamp"}
LOCAL_TIME_FUNCS = {"localtime", "mktime", "ctime", "asctime"}
TIME_MODULES = {"time", "_time"}

# (path relative to the repo, call text) -> why a naive reading is fine there. Every entry must still be needed (see the last test).
ALLOWED: dict[tuple[str, str], str] = {
    ("app/clock.py", "datetime.strptime()"): "parse_ts attaches the caller's assume_tz right after parsing (it is the zone-attaching parser)",
    ("app/doctor.py", "now.astimezone()"): "informational only: the doctor shows the server's own zone to prove it is irrelevant",
}

SQL_PATTERNS = (
    (re.compile(r"(?i)\b(?:date|datetime|time|julianday|strftime)\s*\([^)]*'now'"), "SQLite 'now' is UTC, not the exchange date"),
    (re.compile(r"(?i)'localtime'"), "SQLite 'localtime' reads the server zone"),
    (re.compile(r"(?i)\bDEFAULT\s+\(?\s*(?:CURRENT_DATE|CURRENT_TIME|CURRENT_TIMESTAMP)\b"),
     "CURRENT_* defaults write a space-separated UTC form that does not match the ISO 'Z' timestamps"),
    (re.compile(r"(?i)\bWHERE\b[^;]*\bCURRENT_(?:DATE|TIMESTAMP)\b"), "CURRENT_* in a filter mixes UTC into the logic"),
)


def source_files(suffixes: tuple[str, ...]) -> list[Path]:
    out = []
    for path in ROOT.rglob("*"):
        if path.suffix in suffixes and path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
            out.append(path)
    return sorted(out)


def dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def has_tz_argument(call: ast.Call, positional_index: int) -> bool:
    if len(call.args) > positional_index:
        return not (isinstance(call.args[positional_index], ast.Constant) and call.args[positional_index].value is None)
    for kw in call.keywords:
        if kw.arg in {"tz", "tzinfo"}:
            return not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
    return False


def findings_in(path: Path, root: Path = ROOT, *, use_allowlist: bool = True) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found: list[str] = []
    rel = path.relative_to(root).as_posix()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr, receiver = node.func.attr, dotted(node.func.value)
        text = f"{receiver}.{attr}()"
        where = f"{rel}:{node.lineno}: {text}"
        if use_allowlist and (rel, text) in ALLOWED:
            continue
        if receiver in DATETIME_RECEIVERS and attr in NAIVE_WHEN_BARE and not has_tz_argument(node, 0):
            found.append(f"{where} reads the server's local time; use app.clock (UTC) or pass a tz")
        elif receiver in DATETIME_RECEIVERS and attr == "fromtimestamp" and not has_tz_argument(node, 1):
            found.append(f"{where} converts an epoch to the server's local zone; pass tz=UTC")
        elif attr == "astimezone" and not node.args and not node.keywords:
            found.append(f"{where} converts to the server's local zone; name the zone")
        elif receiver in TIME_MODULES and (attr in LOCAL_TIME_FUNCS or (attr == "strftime" and len(node.args) < 2)):
            found.append(f"{where} depends on the server's local zone")
        elif receiver in DATETIME_RECEIVERS and attr == "strptime":
            parent = parents.get(node)
            grand = parents.get(parent) if parent is not None else None
            used_as = parent.attr if isinstance(parent, ast.Attribute) and isinstance(grand, ast.Call) else None
            if used_as not in {"replace", "date", "astimezone"}:
                found.append(f"{where} yields a naive datetime; attach a zone (.replace(tzinfo=...)) or take .date()")
    return found


def test_the_guard_itself_catches_what_it_is_meant_to(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import time\nfrom datetime import datetime, date, timezone\nimport pandas as pd\n"
        "a = datetime.now()\nb = date.today()\nc = datetime.utcnow()\nd = datetime.fromtimestamp(1)\n"
        "e = datetime.now().astimezone()\nf = time.localtime()\ng = datetime.strptime('2026', '%Y')\nh = pd.Timestamp.now()\n"
        "i = time.strftime('%Y')\n", encoding="utf-8")
    good = tmp_path / "good.py"
    good.write_text(
        "import time\nfrom datetime import datetime, timezone\nimport pandas as pd\nfrom zoneinfo import ZoneInfo\n"
        "a = datetime.now(timezone.utc)\nb = datetime.fromtimestamp(1, tz=timezone.utc)\nc = datetime.now(tz=ZoneInfo('UTC'))\n"
        "d = datetime.strptime('2026', '%Y').replace(tzinfo=timezone.utc)\ne = datetime.strptime('2026', '%Y').date()\n"
        "f = time.strftime('%Y', time.gmtime(0))\ng = pd.Timestamp.now(tz='UTC')\nh = clock.now()\n", encoding="utf-8")
    assert len(findings_in(bad, tmp_path)) == 10, findings_in(bad, tmp_path)
    assert findings_in(good, tmp_path) == [], findings_in(good, tmp_path)


@pytest.mark.parametrize("path", source_files((".py",)), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_no_module_reads_the_server_time_zone(path):
    assert findings_in(path) == []


def test_sql_does_not_mix_utc_dates_into_exchange_logic():
    problems = []
    for path in source_files((".py", ".sql")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern, why in SQL_PATTERNS:
                if pattern.search(line):
                    problems.append(f"{path.relative_to(ROOT).as_posix()}:{number}: {why}: {line.strip()[:100]}")
    assert problems == []


def test_every_allowlist_entry_is_still_needed():
    for (rel, text), reason in ALLOWED.items():
        raw = findings_in(ROOT / rel, use_allowlist=False)
        assert any(f": {text} " in line for line in raw), f"{rel} no longer needs the exemption for {text} ({reason})"
