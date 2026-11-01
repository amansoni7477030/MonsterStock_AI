"""Catalyst provenance fields (optional, backward compatible) and the evening summary's live record."""

from __future__ import annotations

import json

from app import jsonutil
from ranking.types import Candidate, Catalyst, EveningSummary, ReportContext

NEW_FIELDS = ("event_id", "novelty_class", "rumour", "source_class", "n_independent", "n_conflicts")


def full_catalyst() -> Catalyst:
    return Catalyst(headline="Plant outage", event_type="operations", source="Wire", novelty=0.9, event_id=7,
                    novelty_class="new", rumour=False, source_class="primary", n_independent=3, n_conflicts=1)


def test_new_catalyst_fields_default_to_none_so_existing_callers_are_unchanged():
    plain = Catalyst(headline="h")
    assert all(getattr(plain, name) is None for name in NEW_FIELDS)
    assert Catalyst(headline="h", confidence=0.4).confidence == 0.4


def test_catalyst_round_trips_through_a_dict_and_json_keeping_falsy_values():
    original = full_catalyst()
    data = original.to_dict()
    assert {name: data[name] for name in NEW_FIELDS} == {
        "event_id": 7, "novelty_class": "new", "rumour": False, "source_class": "primary", "n_independent": 3,
        "n_conflicts": 1}
    assert Catalyst.from_dict(json.loads(jsonutil.dumps(data))) == original  # a False rumour flag is not lost as None
    zero = Catalyst(headline="h", n_independent=0, n_conflicts=0, event_id=0)
    assert Catalyst.from_dict(zero.to_dict()) == zero


def test_catalyst_from_dict_ignores_unknown_keys_and_old_payloads_without_the_new_ones():
    future = Catalyst.from_dict({"headline": "h", "event_id": 3, "added_next_year": [1, 2], "url": None})
    assert future.event_id == 3 and future.headline == "h"
    old = Catalyst.from_dict({"headline": "h", "event_type": "results", "source": "NSE", "polarity": -0.4})
    assert old.event_type == "results" and all(getattr(old, name) is None for name in NEW_FIELDS)


def test_candidates_and_reports_carry_the_new_fields_through_a_frozen_round_trip():
    candidate = Candidate(symbol="AAA", rank=1, tier="WATCH", p={"hit_3": 0.4}, primary_threshold=3.0, range={"q50": 3.0},
                          catalysts=[full_catalyst(), Catalyst(headline="second")])
    context = ReportContext(trade_date="2026-11-02", stage="final", snapshot_ist="x", model_version="m", model_status="shadow",
                            candidates=[candidate])
    frozen = json.loads(jsonutil.dumps(context.to_dict()))
    restored = ReportContext.from_dict(frozen)
    assert restored.candidates[0].catalysts[0] == full_catalyst()
    assert restored.candidates[0].catalysts[1].event_id is None
    assert restored == context
    legacy = json.loads(jsonutil.dumps(context.to_dict()))
    for item in legacy["candidates"][0]["catalysts"]:
        for name in NEW_FIELDS:
            item.pop(name)  # a snapshot frozen before the fields existed
    assert ReportContext.from_dict(legacy).candidates[0].catalysts[0].novelty == 0.9


def test_evening_summary_has_an_empty_live_record_by_default_and_no_shared_state():
    first, second = EveningSummary("2026-11-02", "pre", 3.0, "m"), EveningSummary("2026-11-02", "pre", 3.0, "m")
    assert first.live_record == {} and first.to_dict()["live_record"] == {}
    first.live_record["20d"] = [{"name": "range_mae_q50", "slice": "all", "value": 1.0, "n": 5}]
    assert second.live_record == {}
