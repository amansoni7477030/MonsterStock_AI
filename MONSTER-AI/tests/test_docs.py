"""Generated documentation must match the code (single source of truth)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import env_example_text
from ingestion.registry import SOURCES, render_markdown

ROOT = Path(__file__).resolve().parent.parent


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_data_sources_doc_is_in_sync():
    assert read(ROOT / "docs" / "DATA_SOURCES.md") == render_markdown(), "run: python scripts/generate_docs.py"


def test_env_example_is_in_sync():
    assert read(ROOT / ".env.example") == env_example_text(), "run: python scripts/generate_docs.py"


def test_every_source_has_licensing_and_failure_documentation():
    for source in SOURCES:
        for field in ("license_notes", "failure_behavior", "rate_limits", "timestamp_meaning", "history_depth"):
            assert source[field], f"{source['source_id']} lacks {field}"


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md", "docs/ARCHITECTURE.md", "docs/OPERATIONS.md", "docs/DATA_SOURCES.md"])
def test_required_documents_exist_and_are_not_empty(name):
    path = ROOT / name
    assert path.is_file() and len(path.read_text(encoding="utf-8")) > 500, f"{name} is missing or too short"
