"""The Markdown documentation stays navigable and its diagrams stay well-formed.

CI has no Mermaid renderer, so the diagrams get a structural check (known diagram type, balanced ``subgraph`` / block ``end``,
balanced brackets and quotes per line, no tabs) and every relative link - including ``#anchor`` fragments - must resolve.
The full syntax of the diagrams was additionally verified with the Mermaid parser when they were written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted({ROOT / "README.md", ROOT / "CHANGELOG.md", ROOT / "deployment" / "README.md", *(ROOT / "docs").glob("*.md")})
DIAGRAM_TYPES = ("flowchart", "graph", "sequenceDiagram", "stateDiagram-v2", "classDiagram", "erDiagram", "gantt", "pie", "timeline")
MIN_DIAGRAMS = {"README.md": 3, "docs/ARCHITECTURE.md": 15}
FENCE = re.compile(r"^(```|~~~)")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
BLOCK_OPEN = re.compile(r"^\s*(subgraph\b|alt\b|opt\b|loop\b|par\b|critical\b|break\b|rect\b)")
BLOCK_CLOSE = re.compile(r"^\s*end\s*$")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def diagrams(path: Path) -> list[tuple[int, str]]:
    """(first line number, body) of every ```mermaid block."""
    found, lines, start, body = [], read(path).split("\n"), 0, None
    for number, line in enumerate(lines, 1):
        if body is None and line.strip() == "```mermaid":
            start, body = number + 1, []
        elif body is not None and line.strip() == "```":
            found.append((start, "\n".join(body)))
            body = None
        elif body is not None:
            body.append(line)
    assert body is None, f"{rel(path)}: unterminated mermaid block starting at line {start}"
    return found


def prose_lines(text: str):
    """(number, line) outside fenced code blocks."""
    fenced = False
    for number, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line.strip()):
            fenced = not fenced
            continue
        if not fenced:
            yield number, line


def slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[*_]{2}|`", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for _number, line in prose_lines(read(path)):
        match = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if match:
            base = slug(match.group(2))
            count = seen.get(base, 0)
            seen[base] = count + 1
            out.add(base if count == 0 else f"{base}-{count}")
    return out


@pytest.mark.parametrize("path", DOCS, ids=rel)
def test_every_diagram_is_structurally_sound(path):
    for start, body in diagrams(path):
        where = f"{rel(path)}:{start}"
        lines = [line for line in body.split("\n") if line.strip()]
        assert lines, f"{where}: empty diagram"
        assert lines[0].split()[0] in DIAGRAM_TYPES, f"{where}: unknown diagram type {lines[0]!r}"
        assert "\t" not in body, f"{where}: tab characters break the parser"
        opened = sum(bool(BLOCK_OPEN.match(line)) for line in lines)
        closed = sum(bool(BLOCK_CLOSE.match(line)) for line in lines)
        assert opened == closed, f"{where}: {opened} block openers (subgraph, alt, loop ...) but {closed} 'end'"
        for line in lines:
            assert line.count('"') % 2 == 0, f"{where}: unbalanced quote in {line.strip()!r}"
            for left, right in ("[]", "()", "{}"):
                assert line.count(left) == line.count(right), f"{where}: unbalanced {left}{right} in {line.strip()!r}"


@pytest.mark.parametrize(("name", "minimum"), sorted(MIN_DIAGRAMS.items()))
def test_the_flow_diagrams_are_still_there(name, minimum):
    assert len(diagrams(ROOT / name)) >= minimum, f"{name} should keep at least {minimum} diagrams"


@pytest.mark.parametrize("path", DOCS, ids=rel)
def test_every_relative_link_and_anchor_resolves(path):
    problems = []
    own_anchors = anchors(path)
    for number, line in prose_lines(read(path)):
        for target in LINK.findall(re.sub(r"`[^`]*`", "", line)):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):  # http(s), mailto ...
                continue
            file_part, _, fragment = target.partition("#")
            destination = path if not file_part else (path.parent / file_part).resolve()
            if not destination.exists():
                problems.append(f"line {number}: {target} -> missing file")
            elif fragment and destination.suffix == ".md":
                known = own_anchors if destination == path else anchors(destination)
                if fragment not in known:
                    problems.append(f"line {number}: {target} -> no such heading")
    assert not problems, f"{rel(path)}:\n" + "\n".join(problems)


def test_the_slug_matches_githubs_anchors():
    assert slug("Daily timetable (Asia/Kolkata, trading days)") == "daily-timetable-asiakolkata-trading-days"
    assert slug("Network layer: curl_cffi, bulk fetching, rate limits") == "network-layer-curl_cffi-bulk-fetching-rate-limits"
    assert slug("`monster-ai` **health**") == "monster-ai-health"
