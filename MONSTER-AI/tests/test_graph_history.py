"""Versioned relationship graph (blueprint 23.6): every real change of an edge is appended to ``graph_edge_history`` in the same
transaction, so the graph that was valid at any past instant can be rebuilt instead of replaying history with today's edges."""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import timedelta

import pytest

from app.clock import iso
from database.connection import connect
from database.migrate import MIGRATIONS_DIR, apply_migrations, current_version, discover
from event_graph.propagation import GraphImpactEngine
from event_graph.repository import Graph, GraphRepository
from event_graph.seed import MASTER_MEMBERSHIP, seed_graph
from tests.test_news_support import add_instruments, insert_event

KEY = "macro:brent|sector:oil_marketing|input_cost"


@pytest.fixture()
def repo(store):
    repo = GraphRepository(store)
    for node_id, kind in (("macro:brent", "commodity"), ("sector:oil_marketing", "sector"), ("HINDPETRO", "stock"),
                          ("sector:aviation", "sector"), ("INDIGO", "stock")):
        repo.upsert_node(node_id, kind, node_id.split(":")[-1])
    return repo


def put(repo, src="macro:brent", dst="sector:oil_marketing", relation="input_cost", **kw):
    kw.setdefault("direction", -1)
    kw.setdefault("sensitivity", -0.4)
    kw.setdefault("confidence", 0.3)
    kw.setdefault("rationale", "crude input costs: refiners buy crude")
    return repo.upsert_edge(src, dst, relation, **kw)


def history(store, where="1=1"):
    return [dict(r) for r in store.conn.execute(
        f"SELECT * FROM graph_edge_history WHERE {where} ORDER BY history_id").fetchall()]


# ------------------------------------------------------------------------------------------------ migration
def migrate_in_two_steps(tmp_path):
    older = tmp_path / "older"
    older.mkdir()
    for migration in discover():
        if migration.version < 4:
            shutil.copy(MIGRATIONS_DIR / f"{migration.version:04d}_{migration.name}.sql", older)
    return older


def test_the_history_migration_is_the_next_contiguous_version():
    versions = {m.version: m.name for m in discover()}
    assert versions[4] == "graph_edge_history" and sorted(versions) == list(range(1, max(versions) + 1))


def test_the_migration_is_additive_keeps_existing_rows_and_baselines_the_current_edges(tmp_path):
    older = migrate_in_two_steps(tmp_path)
    conn = connect(tmp_path / "m.db")
    assert apply_migrations(conn, older) == [1, 2, 3] and current_version(conn) == 3
    conn.execute("INSERT INTO graph_nodes (node_id, node_type, name) VALUES ('macro:brent', 'commodity', 'Brent'), "
                 "('sector:x', 'sector', 'X')")
    conn.execute(
        "INSERT INTO graph_edges (src, dst, relation, direction, sensitivity, lag_days_mean, lag_days_sd, confidence, "
        "regime_dependence, rationale, source, evidence_count, version, last_updated) VALUES ('macro:brent', 'sector:x', "
        "'exposure', 1, 0.3, 0.0, 0.5, 0.6, 'policy', 'crude: realisations', 'measured', 400, 3, '2026-10-20T04:00:00Z')")
    conn.execute("INSERT INTO events (cluster_key, event_type, headline, first_seen_at, last_updated_at, n_independent_sources) "
                 "VALUES ('k1', 'order_win', 'old event', '2026-11-01T10:00:00Z', '2026-11-01T10:00:00Z', 3)")

    full = tmp_path / "full"
    shutil.copytree(MIGRATIONS_DIR, full)
    assert apply_migrations(conn, full) == [4] and current_version(conn) == 4

    edge = dict(conn.execute("SELECT * FROM graph_edges").fetchone())
    assert edge["version"] == 3 and edge["source"] == "measured"  # the live table is untouched
    (base,) = [dict(r) for r in conn.execute("SELECT * FROM graph_edge_history").fetchall()]
    assert base["change"] == "baseline" and base["valid_from"] == "2026-10-20T04:00:00Z" and base["version"] == 3
    for column in ("src", "dst", "relation", "direction", "sensitivity", "lag_days_mean", "lag_days_sd", "confidence",
                   "regime_dependence", "rationale", "source", "evidence_count"):
        assert base[column] == edge[column], column
    old = dict(conn.execute("SELECT * FROM events").fetchone())
    assert old["headline"] == "old event" and old["n_independent_sources"] == 3
    assert (old["novelty_class"], old["rumour"], old["source_class"], old["n_conflicts"]) == (None, None, None, None)

    assert apply_migrations(conn, full) == []  # re-running changes nothing, the baseline is not repeated
    assert conn.execute("SELECT COUNT(*) FROM graph_edge_history").fetchone()[0] == 1


def test_a_fresh_database_starts_with_an_empty_history_and_the_new_event_columns(store):
    assert store.conn.execute("SELECT COUNT(*) FROM graph_edge_history").fetchone()[0] == 0
    columns = {r["name"] for r in store.conn.execute("PRAGMA table_info(events)")}
    assert {"novelty_class", "rumour", "source_class", "n_conflicts", "n_independent_sources"} <= columns
    indexed = {r["name"] for r in store.conn.execute("PRAGMA index_list(graph_edge_history)")}
    assert "idx_graph_edge_history_key" in indexed


def test_the_history_is_append_only_except_for_explicit_maintenance(store, repo):
    put(repo)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("UPDATE graph_edge_history SET confidence = 0.9")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("DELETE FROM graph_edge_history")
    assert len(history(store)) == 1
    with store.allow_prune():
        store.conn.execute("DELETE FROM graph_edge_history")
    assert history(store) == []


# ------------------------------------------------------------------------------------------ writing history
def test_history_is_written_once_per_real_change_and_never_for_a_no_op(store, repo, fake_clock):
    created, result = put(repo, confidence=0.3, lag_days_mean=1.0, regime_dependence="policy")
    assert result == "created"
    (first,) = history(store)
    assert first["change"] == "created" and first["version"] == 1 and first["valid_from"] == store.now()
    assert (first["src"], first["dst"], first["relation"]) == ("macro:brent", "sector:oil_marketing", "input_cost")
    assert (first["direction"], first["sensitivity"], first["confidence"], first["lag_days_mean"]) == (-1, -0.4, 0.3, 1.0)
    assert first["regime_dependence"] == "policy" and first["source"] == "prior" and first["rationale"].startswith("crude")

    for _ in range(3):  # a re-seed or a repeated estimation that changes nothing writes nothing
        assert put(repo, confidence=0.3, lag_days_mean=1.0, regime_dependence="policy")[1] == "unchanged"
    assert put(repo, confidence=0.3 + 1e-12, lag_days_mean=1.0, regime_dependence="policy")[1] == "unchanged"  # float noise
    assert len(history(store)) == 1

    fake_clock.advance(600)
    _, result = put(repo, confidence=0.45, lag_days_mean=1.0, regime_dependence="policy")  # a tracked field changed
    assert result == "updated"
    _, result = put(repo, confidence=0.5, lag_days_mean=0.0, source="measured", evidence_count=400, sensitivity=-0.37,
                    regime_dependence="policy", rationale="crude input costs: refiners buy crude | measured beta -0.37")
    assert result == "updated"
    rows = history(store)
    assert [(r["change"], r["version"]) for r in rows] == [("created", 1), ("updated", 2), ("updated", 3)]
    last = rows[-1]
    assert last["source"] == "measured" and last["evidence_count"] == 400 and last["sensitivity"] == -0.37
    assert last["valid_from"] == iso(fake_clock.now()) and rows[1]["valid_from"] == last["valid_from"]
    live = repo.get_edge("macro:brent", "sector:oil_marketing", "input_cost")
    assert live.version == 3 and (live.confidence, live.source) == (last["confidence"], last["source"])


def test_each_edge_keeps_its_own_history(store, repo):
    put(repo)
    put(repo, dst="sector:aviation", relation="exposure", direction=1)
    put(repo, confidence=0.4)
    assert len(history(store, "relation = 'input_cost'")) == 2 and len(history(store, "relation = 'exposure'")) == 1
    assert [r["version"] for r in repo.edge_history("macro:brent", "sector:oil_marketing", "input_cost")] == [1, 2]
    assert repo.edge_history("macro:brent", "sector:aviation", "input_cost") == []


def test_the_edge_and_its_history_row_are_written_atomically(store, repo, monkeypatch):
    put(repo)

    def fail(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(repo, "_record", fail)
    with pytest.raises(RuntimeError):
        put(repo, confidence=0.9)
    with pytest.raises(RuntimeError):
        put(repo, dst="sector:aviation", relation="exposure")
    monkeypatch.undo()
    assert repo.get_edge("macro:brent", "sector:oil_marketing", "input_cost").confidence == 0.3  # rolled back
    assert repo.get_edge("macro:brent", "sector:aviation", "exposure") is None
    assert len(history(store)) == 1 and not store.conn.in_transaction


def test_changes_made_inside_a_bulk_import_are_recorded_too(store, repo):
    dump = {"nodes": [], "edges": [{"src": "macro:brent", "dst": "sector:aviation", "relation": "exposure", "direction": -1,
                                    "confidence": 0.3}]}
    repo.import_json(dump)
    repo.import_json(dump)  # unchanged
    dump["edges"][0]["confidence"] = 0.5
    repo.import_json(dump)
    assert [(r["change"], r["version"]) for r in history(store)] == [("created", 1), ("updated", 2)]


def test_removing_an_edge_is_recorded_and_a_recreated_edge_is_a_new_life(store, repo, fake_clock):
    put(repo)
    assert repo.delete_edge("macro:brent", "sector:oil_marketing", "input_cost") is True
    assert repo.delete_edge("macro:brent", "sector:oil_marketing", "input_cost") is False  # nothing left to remove
    assert repo.get_edge("macro:brent", "sector:oil_marketing", "input_cost") is None
    removed = history(store)[-1]
    assert removed["change"] == "removed" and removed["version"] == 1 and removed["confidence"] == 0.3  # its last known values
    fake_clock.advance(60)
    again, result = put(repo, confidence=0.35)
    assert result == "created" and again.version == 1
    assert [r["change"] for r in history(store)] == ["created", "removed", "created"]


def test_the_cache_stamp_notices_an_edge_replaced_within_the_same_second(store, repo):
    put(repo)
    cached = repo.load()
    repo.delete_edge("macro:brent", "sector:oil_marketing", "input_cost")
    put(repo, dst="sector:aviation", relation="exposure")  # same count, same version, same second
    fresh = repo.load()
    assert fresh is not cached and [e.dst for e in fresh.edges] == ["sector:aviation"]


# -------------------------------------------------------------------------------------------- as-of lookup
def test_edges_as_of_returns_the_edges_valid_at_that_instant(store, repo, fake_clock):
    t0 = fake_clock.now()
    put(repo, confidence=0.3)
    fake_clock.advance(3600)
    t1 = fake_clock.now()
    put(repo, confidence=0.5, source="measured", evidence_count=300)
    fake_clock.advance(3600)
    t2 = fake_clock.now()
    put(repo, dst="sector:aviation", relation="exposure", direction=1, sensitivity=0.2, confidence=0.35)
    fake_clock.advance(3600)
    t3 = fake_clock.now()
    repo.delete_edge("macro:brent", "sector:oil_marketing", "input_cost")
    second = "macro:brent|sector:aviation|exposure"

    def at(moment):
        return {e.key: e for e in repo.edges_as_of(moment)}

    one_second = timedelta(seconds=1)
    assert at(t0 - one_second) == {}  # before the edge existed
    assert at(t0)[KEY].version == 1 and at(t0)[KEY].confidence == 0.3 and at(t0)[KEY].source == "prior"  # inclusive
    assert at(t1 - one_second)[KEY].version == 1 and at(t1 - one_second)[KEY].confidence == 0.3
    assert at(t1)[KEY].version == 2 and at(t1)[KEY].source == "measured" and at(t1)[KEY].evidence_count == 300
    assert set(at(t2 - one_second)) == {KEY} and set(at(t2)) == {KEY, second}
    assert set(at(t3 - one_second)) == {KEY, second} and set(at(t3)) == {second}  # a removed edge is gone from then on
    assert set(at(t3 + timedelta(days=400))) == {second}
    assert at(t1)[KEY].last_updated == iso(t1) and at(t1)[KEY].edge_id is None
    # the same instant given as an ISO string, and the live table is never touched by a look-up
    assert [e.key for e in repo.edges_as_of(iso(t2))] == sorted([KEY, second]) and len(repo.get_edges()) == 1


def test_changes_within_one_second_resolve_to_the_latest_one(store, repo):
    for confidence in (0.3, 0.4, 0.5):
        put(repo, confidence=confidence)
    (edge,) = repo.edges_as_of(store.now())
    assert edge.version == 3 and edge.confidence == 0.5


def test_edges_as_of_orders_edges_and_copes_with_unparseable_times(store, repo, fake_clock):
    put(repo, dst="sector:aviation", relation="exposure")
    put(repo)
    assert [e.key for e in repo.edges_as_of(fake_clock.now())] == sorted(e.key for e in repo.get_edges())
    for bad in ("not a time", None, ""):
        with pytest.raises(ValueError, match="as-of"):
            repo.edges_as_of(bad)


def test_graph_as_of_is_a_graph_of_the_edges_of_that_time(store, repo, fake_clock):
    t0 = fake_clock.now()
    put(repo)
    repo.upsert_edge("HINDPETRO", "sector:oil_marketing", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    fake_clock.advance(86_400)
    put(repo, confidence=0.1)
    past = repo.graph_as_of(t0 + timedelta(hours=1))
    assert isinstance(past, Graph) and set(past.nodes) == {n.node_id for n in repo.nodes()}
    assert [e.confidence for e in past.out["macro:brent"]] == [0.3]
    assert [e.confidence for e in repo.load().out["macro:brent"]] == [0.1]
    assert sorted(e.relation for e in past.inc["sector:oil_marketing"]) == ["input_cost", "membership"]


def test_seed_replacements_of_a_master_membership_are_recorded_as_removals(store, fake_clock):
    add_instruments(store)
    seed_graph(store)
    repo = GraphRepository(store)
    store.upsert_instrument("WIPRO", "equity", industry="Textiles")
    fake_clock.advance(3600)
    seed_graph(store)
    moved = history(store, "src = 'WIPRO' AND relation = 'membership' AND source = '%s'" % MASTER_MEMBERSHIP)
    assert [r["change"] for r in moved] == ["created"] and moved[0]["dst"] == "sector:textiles"
    store.upsert_instrument("WIPRO", "equity", industry="Realty")
    fake_clock.advance(3600)
    seed_graph(store)
    rows = history(store, "src = 'WIPRO' AND relation = 'membership' AND source = '%s'" % MASTER_MEMBERSHIP)
    assert [(r["dst"], r["change"]) for r in rows] == [("sector:textiles", "created"), ("sector:textiles", "removed"),
                                                      ("sector:realty", "created")]
    now_edges = {e.dst for e in repo.edges_as_of(store.now()) if e.src == "WIPRO" and e.source == MASTER_MEMBERSHIP}
    assert now_edges == {"sector:realty"} == {e.dst for e in repo.get_edges(src="WIPRO", source=MASTER_MEMBERSHIP)}


def test_reseeding_an_unchanged_seed_writes_no_history(store):
    add_instruments(store)
    seed_graph(store)
    rows = store.conn.execute("SELECT COUNT(*) FROM graph_edge_history").fetchone()[0]
    assert rows >= store.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] > 200
    seed_graph(store)
    assert store.conn.execute("SELECT COUNT(*) FROM graph_edge_history").fetchone()[0] == rows


# --------------------------------------------------------------------------------------- replays with the engine
def impact_event(**over):
    base = {"event_id": 1, "event_type": "commodity_supply", "headline": "Brent jumps", "severity": 0.8, "confidence": 0.9,
            "novelty": 1.0, "first_seen_at": "2026-11-02T01:00:00Z",
            "entities": [{"id": "macro:brent", "type": "commodity", "name": "Brent", "polarity": 0.7, "confidence": 0.9}]}
    return {**base, **over}


def test_a_replay_with_point_in_time_edges_uses_the_edges_of_its_own_time(store, repo, fake_clock):
    repo.upsert_edge("HINDPETRO", "sector:oil_marketing", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    put(repo, confidence=0.4)
    fake_clock.advance(86_400)
    put(repo, confidence=0.2, source="measured", evidence_count=300)  # the next day the link is re-estimated, much weaker
    past, future = "2026-11-02T12:00:00Z", "2026-11-04T12:00:00Z"

    today = GraphImpactEngine(store)
    replay = GraphImpactEngine(store, point_in_time_edges=True)
    live = today.impacts([impact_event()], past)["HINDPETRO"]
    then = replay.impacts([impact_event()], past)["HINDPETRO"]
    later = replay.impacts([impact_event()], future)["HINDPETRO"]
    assert then.event_effects[1] == pytest.approx(2.0 * later.event_effects[1], rel=1e-3)  # 0.4 vs 0.2 edge confidence
    assert live.event_effects[1] == pytest.approx(later.event_effects[1], rel=1e-3)  # the default engine only knows today's edges
    assert then.basis == "prior" and later.basis in ("measured", "mixed")
    before_the_graph_existed = replay.impacts([impact_event(first_seen_at="2026-11-01T01:00:00Z")], "2026-11-01T12:00:00Z")
    assert before_the_graph_existed == {}


def test_the_news_pipeline_can_replay_over_the_edges_of_the_time(store, repo, settings, fake_clock):
    from news_intelligence.pipeline import NewsPipeline

    repo.upsert_edge("HINDPETRO", "sector:oil_marketing", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    put(repo)
    fake_clock.advance(86_400)
    repo.delete_edge("macro:brent", "sector:oil_marketing", "input_cost")  # the link is gone by the next day
    brent = [{"type": "commodity", "id": "macro:brent", "name": "Brent", "count": 1, "confidence": 0.9, "polarity": 0.7}]
    insert_event(store, "brent", entities_json=json.dumps(brent), scope="macro", event_type="commodity_supply", severity=0.8,
                 confidence=0.9, first_seen_at="2026-11-02T04:00:00Z", last_updated_at="2026-11-02T04:00:00Z")

    def macro_severity(**kwargs):
        pipe = NewsPipeline(store, settings, clock=fake_clock, **kwargs)
        return pipe.event_features(["HINDPETRO"], "2026-11-02T06:00:00Z")["HINDPETRO"]["ev_global_macro_severity"]

    assert macro_severity() == 0.0  # today's graph: no route from crude to the stock any more
    assert macro_severity(point_in_time_edges=True) == pytest.approx(0.72)  # at 06:00 the route still existed (0.8 x 0.9)
