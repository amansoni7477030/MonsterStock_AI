"""The adapter between the scan/jobs and the news-intelligence and event-graph packages: seeding, reactions, retention and
failure isolation (news is enrichment - it may reduce coverage but must never block or fabricate)."""

from __future__ import annotations

import builtins
import json
import logging
import sqlite3
from datetime import timedelta
from types import SimpleNamespace

import pytest

from automation.intel import Intel
from event_graph import seed as graph_seed
from tests.test_news_support import add_article, add_instruments, insert_event

COMPANY = '[{"id":"RELIANCE","type":"company","polarity":0.8,"confidence":0.9,"in_title":true}]'


@pytest.fixture()
def intel(store, settings, fake_clock):
    add_instruments(store)
    return Intel(store, settings, clock=fake_clock)


class Boom:
    """Stands in for a pipeline / graph whose every call fails."""

    def __getattr__(self, name):
        def fail(*_args, **_kwargs):
            raise RuntimeError("boom")

        return fail


# ----------------------------------------------------------------------------------------------- seeding
def test_the_graph_is_seeded_once_and_only_again_when_the_seed_changes(store, settings, fake_clock, monkeypatch):
    add_instruments(store)
    calls = []
    real = graph_seed.seed_graph
    monkeypatch.setattr(graph_seed, "seed_graph", lambda *a, **k: calls.append(1) or real(*a, **k))

    Intel(store, settings, clock=fake_clock)
    assert calls == [1] and store.kv_get(graph_seed.SEED_KEY) == graph_seed.seed_fingerprint(graph_seed.load_seed())
    assert store.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] > 200

    Intel(store, settings, clock=fake_clock)  # same seed: a cheap fingerprint check, no work
    assert calls == [1]

    store.kv_set(graph_seed.SEED_KEY, "0:outdated")  # a release that ships a new seed
    Intel(store, settings, clock=fake_clock)
    assert calls == [1, 1]


def test_reseeding_reports_the_graph_size_without_the_long_unmapped_list_and_is_idempotent(intel):
    first = intel.reseed_graph()
    assert isinstance(first["unmapped"], int) and first["edges_created"] == 0  # the adapter already seeded it on first use
    assert first["nodes"] > 100 and first["edges"] > 200  # the size is reported whoever did the seeding
    second = intel.reseed_graph()
    assert second["nodes_created"] == 0 and second["edges_created"] == 0 and second["edges_updated"] == 0
    assert (second["nodes"], second["edges"]) == (first["nodes"], first["edges"])


def test_a_broken_seed_is_reported_as_reduced_coverage_not_raised(store, settings, fake_clock, monkeypatch):
    monkeypatch.setattr(graph_seed, "load_seed", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad seed")))
    reduced = Intel(store, settings, clock=fake_clock)
    assert reduced.graph is not None and "event graph seed unavailable (ValueError)" in reduced.coverage_note()

    def broken_seed(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.undo()
    monkeypatch.setattr(graph_seed, "seed_graph", broken_seed)
    other = Intel(store, settings, clock=fake_clock)
    assert other.reseed_graph() == {} and "event graph seeding failed (RuntimeError)" in other.coverage_note()


# ------------------------------------------------------------------------------------ processing / reactions
def test_processing_turns_articles_into_events_and_then_checks_market_reactions(intel, store, monkeypatch):
    add_article(store, 1, "Reliance Industries bags Rs 5,000 crore order from NTPC for solar modules",
                "Reliance Industries has received a Rs 5,000 crore order from NTPC for supply of solar modules.",
                seen="2026-11-02T01:00:00Z")
    seen = []
    real = intel.pipeline.refresh_reactions
    monkeypatch.setattr(intel.pipeline, "refresh_reactions", lambda as_of, *a, **k: seen.append(as_of) or real(as_of, *a, **k))

    stats = intel.process()

    assert stats["events_created"] == 1 and stats["articles_in"] == 1 and stats["errors"] == 0
    assert stats["reactions"] >= 0 and len(seen) == 1 and seen[0] == intel.clock.now()
    assert store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert intel.coverage_note() is None


class Batches:
    """A pipeline whose ``process_pending`` reports scripted batch sizes (a backlog being drained)."""

    def __init__(self, sizes, errors=0):
        self.sizes, self.calls, self.new_runs, self.errors = list(sizes), [], [], errors

    def process_pending(self, max_items=600, *, new_run=True):
        self.calls.append(max_items)
        self.new_runs.append(new_run)
        size = self.sizes.pop(0) if self.sizes else 0
        counts = {"articles_in": size, "filings_in": 0, "events_created": size // 2, "errors": self.errors}
        return SimpleNamespace(as_dict=lambda: dict(counts))

    def refresh_reactions(self, as_of, *a, **k):
        return 3


def test_a_backlog_is_drained_in_bounded_batches_and_the_counts_are_summed(intel):
    from automation import intel as intel_module

    intel.pipeline = Batches([600, 600, 250])
    totals = intel.process()
    assert intel.pipeline.calls == [intel_module.PROCESS_BATCH] * 3  # stops as soon as a batch is not full
    assert intel.pipeline.new_runs == [True, False, False]  # the AI budgets belong to the job, not to a batch
    assert totals["articles_in"] == 1450 and totals["events_created"] == 725 and totals["batches"] == 3 and totals["reactions"] == 3


def test_draining_is_capped_so_one_job_cannot_run_forever(intel):
    from automation import intel as intel_module

    intel.pipeline = Batches([600] * 50)
    totals = intel.process()
    assert totals["batches"] == intel_module.MAX_PROCESS_BATCHES == len(intel.pipeline.calls)
    assert totals["articles_in"] == 600 * intel_module.MAX_PROCESS_BATCHES


def test_draining_stops_at_the_first_batch_with_errors(intel):
    intel.pipeline = Batches([600, 600], errors=2)
    totals = intel.process()
    assert len(intel.pipeline.calls) == 1 and totals["errors"] == 2 and totals["batches"] == 1


def test_a_failing_pipeline_reduces_coverage_without_raising(intel):
    intel.pipeline = Boom()
    assert intel.process() == {"error": "RuntimeError"}
    assert intel.refresh_reactions(intel.clock.now()) == 0
    assert intel.prune() == {}
    note = intel.coverage_note()
    assert "news processing failed (RuntimeError)" in note and "market-reaction check unavailable (RuntimeError)" in note


def test_a_failing_reaction_check_keeps_the_processing_stats(intel, store, monkeypatch):
    add_article(store, 1, "Reliance Industries bags Rs 5,000 crore order from NTPC for solar modules",
                "Reliance Industries has received a Rs 5,000 crore order from NTPC.", seen="2026-11-02T01:00:00Z")
    monkeypatch.setattr(intel.pipeline, "refresh_reactions", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no bars")))
    stats = intel.process()
    assert stats["events_created"] == 1 and stats["reactions"] == 0
    assert "market-reaction check unavailable" in intel.coverage_note()


def test_without_the_packages_everything_degrades_to_empty_results(store, settings, fake_clock, monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith(("news_intelligence", "event_graph")):
            raise ImportError(f"{name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    bare = Intel(store, settings, clock=fake_clock)
    monkeypatch.undo()

    assert bare.pipeline is None and bare.graph is None
    assert bare.process() == {"error": "pipeline unavailable"}
    assert bare.refresh_reactions(fake_clock.now()) == 0 and bare.prune() == {} and bare.reseed_graph() == {}
    assert bare.catalysts(["RELIANCE"], fake_clock.now()) == {} and bare.chains(["RELIANCE"], fake_clock.now()) == {}
    assert bare.event_features(["RELIANCE"], fake_clock.now()) == {}
    assert "news pipeline unavailable" in bare.coverage_note() and "event graph unavailable" in bare.coverage_note()


# ------------------------------------------------------------------------------------------------ queries
def test_reacted_counts_events_that_already_moved_the_stock_inside_the_lookback(intel, store, fake_clock):
    now = fake_clock.now()
    stamp = lambda hours: (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    insert_event(store, "a", entities_json=COMPANY, market_reacted=1, first_seen_at=stamp(3))
    insert_event(store, "b", entities_json=COMPANY, market_reacted=0, first_seen_at=stamp(5))
    insert_event(store, "c", entities_json=COMPANY, market_reacted=None, first_seen_at=stamp(6))
    insert_event(store, "old", entities_json=COMPANY, market_reacted=1, first_seen_at=stamp(24 * 10))
    insert_event(store, "other", entities_json=COMPANY.replace("RELIANCE", "TCS"), market_reacted=1, first_seen_at=stamp(2))
    assert intel.reacted(["RELIANCE", "INFY"], now) == {"RELIANCE": (1, 3)}
    assert intel.reacted(["TCS"], now) == {"TCS": (1, 1)}


def test_catalysts_skip_symbols_that_fail_or_have_none(intel):
    def catalysts_for(symbol, as_of, limit=3):
        if symbol == "TCS":
            raise RuntimeError("bad row")
        return ["catalyst"] if symbol == "RELIANCE" else []

    intel.pipeline = SimpleNamespace(catalysts_for=catalysts_for)
    assert intel.catalysts(["RELIANCE", "TCS", "INFY"], intel.clock.now()) == {"RELIANCE": ["catalyst"]}


def test_chains_keep_only_the_requested_symbols_and_at_most_three_each(intel):
    impacts = {
        "RELIANCE": SimpleNamespace(chains=["c1", "c2", "c3", "c4"]),
        "TCS": SimpleNamespace(chains=["x"]),
        "INFY": SimpleNamespace(chains=[]),
    }
    intel.pipeline = SimpleNamespace(recent_events=lambda *a, **k: ["event"])
    intel.graph = SimpleNamespace(impacts=lambda events, as_of: impacts)
    assert intel.chains(["RELIANCE", "INFY"], intel.clock.now()) == {"RELIANCE": ["c1", "c2", "c3"]}
    intel.graph = Boom()
    assert intel.chains(["RELIANCE"], intel.clock.now()) == {}


def test_event_features_pass_through_and_fail_soft(intel):
    intel.pipeline = SimpleNamespace(event_features=lambda symbols, as_of, hours: {s: {"ev_count": 1.0} for s in symbols})
    assert intel.event_features(["RELIANCE"], intel.clock.now()) == {"RELIANCE": {"ev_count": 1.0}}
    intel.pipeline = Boom()
    assert intel.event_features(["RELIANCE"], intel.clock.now()) == {}


def test_prune_returns_the_retention_counts(intel, store):
    add_article(store, 1, "Old headline", seen="2026-01-01T00:00:00Z")
    store.conn.execute("UPDATE news_raw SET processed = 1")
    counts = intel.prune()
    assert counts["news_raw"] == 1 and store.conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0] == 0


def test_the_coverage_note_mentions_a_degraded_llm(intel, store):
    store.kv_set("news.reduced_coverage", "1")
    assert "LLM extraction degraded" in intel.coverage_note()


# --------------------------------------------------------------- a stock is only counted for events it is the subject of
def entity_json(symbol, **over):
    item = {"id": symbol, "type": "company", "name": symbol, "polarity": 0.0, "confidence": 0.9, **over}
    return json.dumps([item])


def test_reacted_does_not_count_a_passing_mention_as_an_event_for_the_stock(intel, store, fake_clock):
    now = fake_clock.now()
    stamp = lambda hours: (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    reliance_story = json.dumps([
        {"id": "RELIANCE", "type": "company", "polarity": 0.8, "confidence": 0.9, "in_title": True},
        {"id": "TCS", "type": "company", "polarity": 0.0, "confidence": 0.9},  # TCS is only named in the body
    ])
    insert_event(store, "story", entities_json=reliance_story, market_reacted=1, first_seen_at=stamp(2))
    insert_event(store, "real", entities_json=entity_json("TCS", polarity=-0.5), market_reacted=0, first_seen_at=stamp(3))
    assert intel.reacted(["RELIANCE", "TCS"], now) == {"RELIANCE": (1, 1), "TCS": (0, 1)}  # not (1, 2) for TCS


def test_reacted_matches_structured_entities_not_text(intel, store, fake_clock):
    now = fake_clock.now()
    stamp = lambda hours: (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    spaced = '[ { "id" : "TCS", "type": "company", "polarity": 0.6, "confidence": 0.9 } ]'  # any JSON layout will do
    insert_event(store, "spaced", entities_json=spaced, market_reacted=1, first_seen_at=stamp(2))
    insert_event(store, "headline", entities_json=entity_json("TCS", in_title=True), market_reacted=0, first_seen_at=stamp(3))
    insert_event(store, "unsure", entities_json=entity_json("TCS", polarity=0.7, confidence=0.3), first_seen_at=stamp(4))
    insert_event(store, "sector", entities_json=entity_json("TCS", type="sector", polarity=0.7), first_seen_at=stamp(5))
    insert_event(store, "broken", entities_json="{not json", first_seen_at=stamp(6))
    insert_event(store, "text-only", entities_json="[]", headline="TCS wins an order", first_seen_at=stamp(7))
    assert intel.reacted(["TCS"], now) == {"TCS": (1, 2)}


def test_reacted_only_knows_entities_attached_by_the_as_of_time(intel, store, fake_clock):
    now = fake_clock.now()
    seen = (now - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    late = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    story = json.dumps([{"id": "RELIANCE", "type": "company", "polarity": 0.8, "confidence": 0.9},
                        {"id": "TCS", "type": "company", "polarity": 0.4, "confidence": 0.9, "t": late}])
    insert_event(store, "story", entities_json=story, market_reacted=1, first_seen_at=seen)
    assert intel.reacted(["TCS"], now - timedelta(hours=2)) == {}
    assert intel.reacted(["TCS"], now) == {"TCS": (1, 1)}


# -------------------------------------------------------------- news failures reduce coverage out loud, never silently
def test_failing_catalyst_lookups_are_logged_and_reported_as_reduced_coverage(intel, caplog):
    def catalysts_for(symbol, as_of, limit=3):
        if symbol in ("TCS", "INFY"):
            raise RuntimeError("bad row")
        return ["catalyst"] if symbol == "RELIANCE" else []

    intel.pipeline = SimpleNamespace(catalysts_for=catalysts_for)
    with caplog.at_level(logging.WARNING, logger="automation.intel"):
        out = intel.catalysts(["RELIANCE", "TCS", "INFY", "SBIN"], intel.clock.now())
        intel.catalysts(["RELIANCE", "TCS", "INFY", "SBIN"], intel.clock.now())
    assert out == {"RELIANCE": ["catalyst"]}
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 4 and "TCS" in warned[0].getMessage() and "INFY" in warned[1].getMessage()
    assert all(r.exc_info for r in warned)  # the traceback is kept
    assert intel.coverage_note() == "catalysts unavailable for 2 of 4 stock(s) (RuntimeError)"  # once, however often it recurs


def test_a_failing_graph_propagation_is_logged_and_reported(intel, caplog):
    intel.pipeline = SimpleNamespace(recent_events=lambda *a, **k: ["event"])
    intel.graph = Boom()
    with caplog.at_level(logging.WARNING, logger="automation.intel"):
        assert intel.chains(["RELIANCE"], intel.clock.now()) == {}
        assert intel.chains(["TCS"], intel.clock.now()) == {}
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2
    assert intel.coverage_note() == "event chains unavailable (RuntimeError)"  # reported once however often it fails
    intel.graph = None
    assert intel.chains(["RELIANCE"], intel.clock.now()) == {}  # no graph at all: already reported at start-up


def test_failing_event_features_are_logged_and_reported(intel, caplog):
    intel.pipeline = Boom()
    with caplog.at_level(logging.WARNING, logger="automation.intel"):
        assert intel.event_features(["RELIANCE"], intel.clock.now()) == {}
    assert [r.levelno for r in caplog.records if "event features" in r.getMessage()] == [logging.WARNING]
    assert intel.coverage_note() == "event features unavailable (RuntimeError)"


def test_a_failing_priced_in_check_is_logged_and_reported(intel, caplog):
    class BrokenConnection:
        def execute(self, *_a, **_k):
            raise sqlite3.OperationalError("disk I/O error")

    intel.store = SimpleNamespace(conn=BrokenConnection(), kv_get=lambda *a, **k: None)
    with caplog.at_level(logging.WARNING, logger="automation.intel"):
        assert intel.reacted(["RELIANCE"], intel.clock.now()) == {}
    assert any(r.levelno == logging.WARNING and "priced-in" in r.getMessage() for r in caplog.records)
    assert intel.coverage_note() == "already-priced-in check unavailable (OperationalError)"


def test_healthy_queries_leave_the_coverage_note_empty(intel, store, fake_clock):
    insert_event(store, "a", entities_json=COMPANY, first_seen_at=(fake_clock.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    now = fake_clock.now()
    intel.catalysts(["RELIANCE", "TCS"], now)
    intel.chains(["RELIANCE"], now)
    intel.event_features(["RELIANCE"], now)
    intel.reacted(["RELIANCE"], now)
    assert intel.coverage_note() is None


# ---------------------------------------------------------------------- the report chains tell measured, prior and broken apart
def test_report_chains_label_each_link_and_call_a_broken_relationship_broken(intel, store, fake_clock):
    from event_graph.repository import GraphRepository

    now = fake_clock.now()
    brent = json.dumps([{"id": "macro:brent", "type": "commodity", "name": "Brent crude", "polarity": 0.7, "confidence": 0.9,
                         "move_pct": 3.1}])
    insert_event(store, "brent", entities_json=brent, scope="macro", event_type="commodity_supply", severity=0.8, confidence=0.9,
                 first_seen_at=(now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    chains = intel.chains(["HINDPETRO"], now)["HINDPETRO"]
    assert chains[0].startswith("Brent +3.1% -> ") and "[expert prior]" in chains[0] and "[broken]" not in chains[0]

    repo = GraphRepository(store)
    for edge in repo.get_edges(src="macro:brent"):  # the data says the crude relationships no longer hold
        repo.upsert_edge(edge.src, edge.dst, edge.relation, direction=0, sensitivity=None, lag_days_mean=edge.lag_days_mean,
                         lag_days_sd=edge.lag_days_sd, confidence=0.05, rationale=edge.rationale, source="relationship_break",
                         evidence_count=300)
    broken = intel.chains(["HINDPETRO"], now)["HINDPETRO"]
    assert broken and all("[broken]" in chain and "[expert prior]" not in chain for chain in broken)
    assert intel.coverage_note() is None  # a broken relationship is information, not a news outage
