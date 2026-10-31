"""Shared fixtures. Tests are hermetic: temp directories, an injected clock, no network."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MONSTER_ENV", "test")

from app.clock import FakeClock, set_clock  # noqa: E402
from app.config import Settings, load_settings  # noqa: E402
from app.logging_setup import clear_secrets, reset_logging  # noqa: E402
from database.store import Store  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MONSTER_RUN_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test (set MONSTER_RUN_NETWORK_TESTS=1)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolate_globals():
    yield
    reset_logging()
    clear_secrets()
    set_clock(__import__("app.clock", fromlist=["Clock"]).Clock())


@pytest.fixture()
def fake_clock() -> FakeClock:
    # Monday 2026-11-02 03:00 UTC = 08:30 IST (a regular NSE trading day)
    clock = FakeClock(datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc))
    set_clock(clock)
    return clock


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    env = {
        "MONSTER_ENV": "test",
        "MONSTER_HOME": str(ROOT),
        "MONSTER_DATA_DIR": str(tmp_path / "data"),
        "MONSTER_LOG_DIR": str(tmp_path / "logs"),
    }
    cfg = load_settings(env, env_file=tmp_path / "missing.env")
    cfg.ensure_dirs()
    return cfg


@pytest.fixture()
def store(settings: Settings, fake_clock: FakeClock) -> Store:
    st = Store.open(settings.db_path, clock=fake_clock)
    yield st
    st.close()
