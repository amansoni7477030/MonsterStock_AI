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


def test_every_source_documents_how_it_is_requested():
    from ingestion.registry import _fetch_strategy

    strategy = _fetch_strategy()
    assert set(strategy) == {source["source_id"] for source in SOURCES}
    assert all(text.strip() for text in strategy.values())
    document = read(ROOT / "docs" / "DATA_SOURCES.md")
    assert "## Transport and request strategy" in document and "curl_cffi" in document and document.count("**Transport and request strategy:**") == len(SOURCES)


def test_every_catalogue_entry_has_the_fields_the_database_register_needs():
    needed = {"source_id", "provider", "endpoint", "coverage", "timestamp_meaning", "update_frequency", "history_depth", "rate_limits",
              "license_notes", "failure_behavior", "backup_source", "criticality"}
    for source in SOURCES:
        assert set(source) == needed, f"{source['source_id']}: {sorted(set(source) ^ needed)}"


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md", "docs/ARCHITECTURE.md", "docs/OPERATIONS.md", "docs/DATA_SOURCES.md"])
def test_required_documents_exist_and_are_not_empty(name):
    path = ROOT / name
    assert path.is_file() and len(path.read_text(encoding="utf-8")) > 500, f"{name} is missing or too short"
