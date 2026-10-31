from __future__ import annotations

import pytest

from event_graph.repository import RELATIONS, Edge, Graph, GraphRepository, Node


@pytest.fixture()
def repo(store):
    repo = GraphRepository(store)
    for node_id, kind in (("macro:brent", "commodity"), ("sector:oil_marketing", "sector"), ("HINDPETRO", "stock"),
                          ("BPCL", "stock"), ("sector:aviation", "sector"), ("INDIGO", "stock")):
        repo.upsert_node(node_id, kind, node_id.split(":")[-1])
    return repo


def edge(repo, src="macro:brent", dst="sector:oil_marketing", relation="input_cost", **kw):
    kw.setdefault("direction", -1)
    kw.setdefault("sensitivity", -0.4)
    kw.setdefault("rationale", "crude input costs: refiners buy crude")
    return repo.upsert_edge(src, dst, relation, **kw)


def test_nodes_are_created_updated_and_validated(repo):
    assert repo.upsert_node("macro:gold", "commodity", "Gold") == "created"
    assert repo.upsert_node("macro:gold", "commodity", "Gold") == "unchanged"
    assert repo.upsert_node("macro:gold", "commodity", "Gold bullion", {"unit": "USD/oz"}) == "updated"
    node = repo.get_node("macro:gold")
    assert node == Node("macro:gold", "commodity", "Gold bullion", {"unit": "USD/oz"}) and node.attrs == {"unit": "USD/oz"}
    assert repo.get_node("nope") is None
    for bad in (("", "stock"), ("X", "")):
        with pytest.raises(ValueError):
            repo.upsert_node(*bad)
    assert [n.node_id for n in repo.nodes("stock")] == ["BPCL", "HINDPETRO", "INDIGO"]
    assert len(repo.nodes()) == 7


def test_edges_are_validated(repo):
    with pytest.raises(ValueError, match="relation"):
        edge(repo, relation="causes")
    with pytest.raises(ValueError, match="direction"):
        edge(repo, direction=2)
    with pytest.raises(ValueError, match="unknown node"):
        edge(repo, dst="sector:missing")
    with pytest.raises(ValueError, match="unknown node"):
        edge(repo, src="macro:missing")
    assert set(RELATIONS) == {"exposure", "membership", "input_cost", "export_demand", "fx_translation", "rate_sensitivity",
                              "co_movement", "supply_chain"}
    assert repo.get_edges() == []


def test_version_is_bumped_only_when_the_edge_changes(repo, store):
    created, result = edge(repo, confidence=0.3, lag_days_mean=1.0, lag_days_sd=0.5, regime_dependence="all regimes")
    assert result == "created" and created.version == 1 and created.source == "prior" and created.evidence_count == 0
    assert created.key == "macro:brent|sector:oil_marketing|input_cost" and created.last_updated
    same, result = edge(repo, confidence=0.3, lag_days_mean=1.0, lag_days_sd=0.5, regime_dependence="all regimes")
    assert result == "unchanged" and same.version == 1 and same.last_updated == created.last_updated
    nearly, result = edge(repo, confidence=0.3 + 1e-12, lag_days_mean=1.0, lag_days_sd=0.5, regime_dependence="all regimes")
    assert result == "unchanged" and nearly.version == 1  # float noise is not a change
    changed, result = edge(repo, confidence=0.35, lag_days_mean=1.0, lag_days_sd=0.5, regime_dependence="all regimes")
    assert result == "updated" and changed.version == 2 and changed.confidence == 0.35
    measured, _ = edge(repo, confidence=0.5, lag_days_mean=1.0, lag_days_sd=0.5, regime_dependence="all regimes",
                       source="measured", evidence_count=400)
    assert measured.version == 3 and measured.source == "measured" and measured.evidence_count == 400
    assert store.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0] == 1


def test_confidence_is_clamped_and_sensitivity_may_be_unknown(repo):
    high, _ = edge(repo, confidence=7.0, sensitivity=None)
    low, _ = edge(repo, dst="sector:aviation", confidence=-2.0)
    assert high.confidence == 1.0 and high.sensitivity is None and low.confidence == 0.0


def test_get_edges_filters_and_ordering(repo):
    edge(repo)
    edge(repo, dst="sector:aviation", relation="exposure", source="measured")
    repo.upsert_edge("HINDPETRO", "sector:oil_marketing", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    assert len(repo.get_edges()) == 3
    assert [e.dst for e in repo.get_edges(src="macro:brent")] == ["sector:aviation", "sector:oil_marketing"]
    assert [e.src for e in repo.get_edges(dst="sector:oil_marketing")] == ["HINDPETRO", "macro:brent"]
    assert len(repo.get_edges(relation="membership")) == 1 and len(repo.get_edges(source="measured")) == 1
    assert repo.get_edges(src="macro:brent", dst="sector:aviation", relation="exposure", source="measured")[0].direction == -1
    assert repo.get_edge("macro:brent", "sector:aviation", "input_cost") is None
    assert isinstance(repo.get_edge("macro:brent", "sector:aviation", "exposure"), Edge)


def test_load_builds_adjacency_and_is_cached_by_stamp(repo):
    edge(repo)
    repo.upsert_edge("HINDPETRO", "sector:oil_marketing", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    repo.upsert_edge("BPCL", "sector:oil_marketing", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    graph = repo.load()
    assert isinstance(graph, Graph) and repo.load() is graph
    assert set(graph.nodes) >= {"macro:brent", "HINDPETRO"} and len(graph.edges) == 3
    assert [e.dst for e in graph.out["macro:brent"]] == ["sector:oil_marketing"]
    assert sorted(e.src for e in graph.inc["sector:oil_marketing"]) == ["BPCL", "HINDPETRO", "macro:brent"]
    edge(repo, confidence=0.4)  # a version bump alone must invalidate the cache
    assert repo.load() is not graph
    graph = repo.load()
    repo.upsert_node("macro:gold", "commodity", "Gold")
    assert repo.load() is not graph
    graph = repo.load()
    repo.upsert_node("macro:gold", "commodity", "Gold", {"unit": "oz"})
    assert repo.load() is not graph


def test_membership_flows_downwards_only(repo):
    edge(repo)
    repo.upsert_edge("HINDPETRO", "sector:oil_marketing", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    graph = repo.load()
    assert [(e.relation, t) for e, t in graph.impact_edges("sector:oil_marketing")] == [("membership", "HINDPETRO")]
    assert [(e.relation, t) for e, t in graph.impact_edges("macro:brent")] == [("input_cost", "sector:oil_marketing")]
    assert graph.impact_edges("HINDPETRO") == []  # a stock does not push shocks up to its sector
    assert graph.reachable_from(["macro:brent"]) == {"macro:brent", "sector:oil_marketing", "HINDPETRO"}
    assert graph.reachable_from(["HINDPETRO"]) == {"HINDPETRO"} and graph.reachable_from(["nope"]) == set()
    assert graph.containers_of("HINDPETRO") == {"sector:oil_marketing"} and graph.containers_of("macro:brent") == set()
    repo.upsert_edge("sector:oil_marketing", "sector:aviation", "membership", direction=1, sensitivity=1.0, confidence=0.9)
    assert repo.load().containers_of("HINDPETRO") == {"sector:oil_marketing", "sector:aviation"}
    assert repo.load().containers_of("HINDPETRO", max_depth=1) == {"sector:oil_marketing"}


def test_counts_summarise_the_graph(repo):
    edge(repo)
    edge(repo, dst="sector:aviation", relation="exposure", source="measured")
    counts = repo.counts()
    assert counts["nodes"] == 6 and counts["edges"] == 2
    assert counts["nodes_by_type"] == {"commodity": 1, "sector": 2, "stock": 3}
    assert counts["edges_by_source"] == {"prior": 1, "measured": 1}
    assert counts["edges_by_relation"] == {"input_cost": 1, "exposure": 1}


def test_export_and_import_round_trip(repo, store, settings, fake_clock):
    from database.store import Store

    edge(repo, confidence=0.35, lag_days_mean=1.0, lag_days_sd=0.5, regime_dependence="x", source="measured", evidence_count=300)
    repo.upsert_node("macro:gold", "commodity", "Gold", {"unit": "oz"})
    dump = repo.export_json()
    assert dump["version"] == 1 and "edge_id" not in dump["edges"][0]
    other = Store.open(settings.data_dir / "other.db", clock=fake_clock)
    try:
        target = GraphRepository(other)
        counts = target.import_json(dump)
        assert counts["nodes_created"] == 7 and counts["edges_created"] == 1
        assert target.export_json() == dump
        assert target.import_json(dump) == {"nodes_created": 0, "nodes_updated": 0, "edges_created": 0, "edges_updated": 0,
                                            "edges_unchanged": 1}
        relabelled = GraphRepository(other)
        relabelled.import_json(dump, source="imported")
        assert relabelled.get_edges()[0].source == "imported"
    finally:
        other.close()


def test_import_is_atomic(repo, store):
    bad = {"nodes": [{"id": "macro:new", "type": "macro", "name": "New"}],
           "edges": [{"src": "macro:new", "dst": "missing-node", "relation": "exposure", "direction": 1}]}
    with pytest.raises(ValueError):
        repo.import_json(bad)
    assert repo.get_node("macro:new") is None  # the whole import rolled back
