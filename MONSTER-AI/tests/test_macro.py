"""FRED client (optional macro series): inert without a key, first-release observations only, resilient to missing values and outages."""

from __future__ import annotations

import dataclasses

import pytest

from ingestion.http import HttpClient
from ingestion.macro import FredClient
from tests.helpers import FakeResponse, FakeSession

OBSERVATIONS = {"observations": [
    {"date": "2026-09-01", "realtime_start": "2026-10-10", "realtime_end": "9999-12-31", "value": "3.1"},
    {"date": "2026-08-01", "realtime_start": "2026-09-10", "realtime_end": "2026-10-09", "value": "3.0"},
    {"date": "2026-10-01", "realtime_start": "2026-11-01", "realtime_end": "9999-12-31", "value": "."},  # FRED's "missing"
    {"date": "2026-07-01", "realtime_start": "2026-08-10", "realtime_end": "9999-12-31"},  # no value at all
]}


def client(store, settings, fake_clock, handler, key="abc123"):
    cfg = dataclasses.replace(settings, fred_api_key=key)
    session = FakeSession(handler)
    http = HttpClient("fred", session=session, clock=fake_clock, max_attempts=1)
    return FredClient(cfg, store, http=http, clock=fake_clock), session


def rows(store):
    return [dict(r) for r in store.conn.execute("SELECT * FROM macro_observations ORDER BY observation_date, realtime_start")]


def test_without_an_api_key_nothing_is_requested(store, settings, fake_clock):
    fred, session = client(store, settings, fake_clock, lambda url, kw: pytest.fail("no request expected"), key=None)
    assert fred.enabled is False and fred.collect() == 0 and session.calls == []


def test_first_release_observations_are_stored_with_the_time_they_became_public(store, settings, fake_clock):
    fred, session = client(store, settings, fake_clock, lambda url, kw: FakeResponse(200, OBSERVATIONS))
    assert fred.collect(("CPIAUCSL",)) == 2
    stored = rows(store)
    assert [(r["observation_date"], r["realtime_start"], r["value"]) for r in stored] == [
        ("2026-08-01", "2026-09-10", 3.0), ("2026-09-01", "2026-10-10", 3.1)]
    params = session.calls[0][2]["params"]
    assert params["output_type"] == "4" and params["series_id"] == "CPIAUCSL" and params["realtime_start"] == "1776-07-04"
    assert fred.collect(("CPIAUCSL",)) == 2 and len(rows(store)) == 2  # re-running never duplicates or rewrites history
    health = dict(store.conn.execute("SELECT * FROM source_health WHERE source_id = 'fred'").fetchone())
    assert health["consecutive_failures"] == 0 and health["total_success"] == 2


def test_one_failing_series_does_not_stop_the_others_and_is_recorded_without_the_key(store, settings, fake_clock):
    def handler(url, kw):
        return FakeResponse(500, b"boom") if kw["params"]["series_id"] == "UNRATE" else FakeResponse(200, OBSERVATIONS)

    fred, _ = client(store, settings, fake_clock, handler)
    assert fred.collect(("CPIAUCSL", "UNRATE", "PAYEMS")) == 4  # two good series, two rows each
    health = dict(store.conn.execute("SELECT * FROM source_health WHERE source_id = 'fred'").fetchone())
    assert health["total_failure"] == 1 and health["total_success"] == 2 and "abc123" not in (health["last_error"] or "")


def test_a_malformed_reply_stores_nothing(store, settings, fake_clock):
    fred, _ = client(store, settings, fake_clock, lambda url, kw: FakeResponse(200, ["not", "a", "mapping"]))
    assert fred.collect(("CPIAUCSL",)) == 0 and rows(store) == []
