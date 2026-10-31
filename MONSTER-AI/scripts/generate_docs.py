"""Regenerate generated documentation: ``python scripts/generate_docs.py`` (run from the project root).

``tests/test_docs.py`` fails when the committed files differ from the generated ones, so docs cannot drift from the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import env_example_text  # noqa: E402
from ingestion.registry import render_markdown  # noqa: E402

TARGETS = {
    ROOT / ".env.example": env_example_text,
    ROOT / "docs" / "DATA_SOURCES.md": render_markdown,
}


def main() -> int:
    for path, render in TARGETS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(), encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
