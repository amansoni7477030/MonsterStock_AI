from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from event_graph import seed as seed_mod
from event_graph.repository import RELATIONS, GraphRepository
from event_graph.sectors import (
    INDUSTRY_NODE_IDS,
    NSE_INDUSTRIES,
    canonical_industry,
    industry_node_id,
    slugify,
)
from event_graph.seed import (
    CURATED_MEMBERSHIP,
    MASTER_MEMBERSHIP,
    PRIOR_SOURCE,
    load_seed,
    seed_fingerprint,
    seed_graph,
)
from tests.test_news_support import INSTRUMENTS, add_instruments

REQUIRED_DRIVERS = {
    "macro:brent", "macro:wti", "macro:natgas", "macro:gold", "macro:silver", "macro:copper", "macro:aluminium", "macro:usdinr",
    "macro:dxy", "macro:us10y", "macro:fed", "macro:rbi", "macro:china_demand", "macro:us_equities", "macro:nasdaq", "macro:sox",
    "macro:nikkei", "macro:hsi", "macro:vix", "macro:geopolitics_mideast", "macro:tariffs",
}
REQUIRED_FINE_SECTORS = {
    "sector:oil_marketing", "sector:upstream_oil", "sector:aviation", "sector:paints", "sector:private_banks", "sector:psu_banks",
    "sector:nbfc", "sector:it_services", "sector:pharma", "sector:metals_steel", "sector:metals_nonferrous", "sector:cement",
    "sector:autos", "sector:fmcg", "sector:capital_goods", "sector:power_utilities", "sector:telecom", "sector:realty",
    "sector:chemicals", "sector:textiles",
}
DRIVER_TYPES = ("macro", "commodity", "fx", "rate", "index", "country", "institution", "global_company")


@pytest.fixture()
def seed_data():
    return load_seed()


def drivers_of(graph):
    return {n for n, node in graph.nodes.items() if node.node_type in DRIVER_TYPES}


# ------------------------------------------------------------------------------------------ the data file
def test_seed_file_is_internally_consistent(seed_data):
    node_ids = [n["id"] for n in seed_data["nodes"]]
    stock_ids = [s["symbol"] for s in seed_data["stocks"]]
    assert len(node_ids) == len(set(node_ids)) and len(stock_ids) == len(set(stock_ids))
    assert not set(node_ids) & set(stock_ids)
    known = set(node_ids) | set(stock_ids)
    assert REQUIRED_DRIVERS <= set(node_ids) and REQUIRED_FINE_SECTORS <= set(node_ids)
    assert set(INDUSTRY_NODE_IDS) <= set(node_ids)
    assert len(stock_ids) >= 80
    sector_ids = {n["id"] for n in seed_data["nodes"] if n["type"] == "sector"}
    for stock in seed_data["stocks"]:
        assert canonical_industry(stock["industry"]) in NSE_INDUSTRIES, stock
        assert set(stock.get("sectors", [])) <= sector_ids, stock
    seen_edges = set()
    for e in seed_data["edges"]:
        assert e["src"] in known and e["dst"] in known, e
        assert e["rel"] in RELATIONS and e["rel"] != "membership", e
        key = (e["src"], e["dst"], e["rel"])
        assert key not in seen_edges, key
        seen_edges.add(key)
        assert e["dir"] in (-1, 0, 1) and 0.25 <= e["conf"] <= 0.4, e
        assert isinstance(e["why"], str) and ": " in e["why"] and len(e["why"]) >= 20, e
        assert e["src"] != e["dst"]
        if e.get("sens") is not None and e["dir"] != 0:
            assert math.copysign(1, e["sens"]) == e["dir"], e
        if e.get("lag"):
            mean, sd = e["lag"]
            assert mean >= 0 and sd >= 0, e
    for node in seed_data["nodes"]:
        parent = (node.get("attrs") or {}).get("parent")
        assert parent is None or parent in sector_ids, node


def test_every_industry_node_has_an_incoming_driver(seed_data):
    incoming = {e["dst"] for e in seed_data["edges"]}
    parents = {(n.get("attrs") or {}).get("parent") for n in seed_data["nodes"]}
    missing = [i for i in INDUSTRY_NODE_IDS if i not in incoming and i not in parents]
    assert missing == []


def test_seed_fingerprint_tracks_content(seed_data):
    first = seed_fingerprint(seed_data)
    assert first.startswith(f"{seed_data['version']}:") and seed_fingerprint(json.loads(json.dumps(seed_data))) == first
    changed = json.loads(json.dumps(seed_data))
    changed["edges"][0]["conf"] = 0.31
    assert seed_fingerprint(changed) != first


def test_load_seed_rejects_non_objects(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_seed(path)


# --------------------------------------------------------------------------------------------- seeding
def test_seeding_is_idempotent(store, seed_data):
    add_instruments(store)
    first = seed_graph(store)
    repo = GraphRepository(store)
    assert first["nodes_created"] > 100 and first["edges_created"] == len(seed_data["edges"])
    snapshot = repo.export_json()
    versions = {e.key: e.version for e in repo.get_edges()}
    second = seed_graph(store)
    assert second["nodes_created"] == second["nodes_updated"] == 0
    assert second["edges_created"] == second["edges_updated"] == 0 and second["edges_unchanged"] == len(seed_data["edges"])
    assert second["membership_curated"] == second["membership_master"] == 0
    assert repo.export_json() == snapshot and {e.key: e.version for e in repo.get_edges()} == versions
    assert store.kv_get(seed_mod.SEED_KEY) == seed_fingerprint(seed_data)


def test_priors_are_stored_as_priors_never_as_measurements(store):
    add_instruments(store)
    seed_graph(store)
    repo = GraphRepository(store)
    for edge in repo.get_edges():
        if edge.relation == "membership":
            assert edge.source in (CURATED_MEMBERSHIP, MASTER_MEMBERSHIP) and edge.confidence == 0.9 and edge.direction == 1
            assert edge.evidence_count == 0 and "member of" in edge.rationale
        else:
            assert edge.source == PRIOR_SOURCE and edge.evidence_count == 0 and 0.25 <= edge.confidence <= 0.4, edge
            assert edge.rationale and edge.version == 1
    counts = repo.counts()
    assert counts["edges_by_source"][PRIOR_SOURCE] == 239 and "measured" not in counts["edges_by_source"]
    assert counts["nodes_by_type"]["stock"] >= 136


def test_edges_are_expert_priors_with_signs_that_depend_on_the_channel(store):
    seed_graph(store)
    repo = GraphRepository(store)
    crude = {e.dst: e for e in repo.get_edges(src="macro:brent")}
    assert crude["sector:oil_marketing"].direction == -1 and crude["sector:upstream_oil"].direction == 1
    usd = {e.dst: e for e in repo.get_edges(src="macro:usdinr")}
    assert usd["sector:it_services"].direction == 1
    assert all(e.regime_dependence for e in (crude["sector:oil_marketing"], usd["sector:it_services"]))


def test_every_universe_stock_is_reachable_from_a_driver(store):
    extras = [
        ("ZZTEXTILE", "Zz Textile Mills Limited", "Textiles", 0),
        ("ZZSOFT", "Zz Software Limited", "Software", 0),  # vendor spelling folded onto Information Technology
        ("ZZSPACE", "Zz Space Mining Limited", "Space Mining", 0),  # unknown label still gets a sector node
    ]
    add_instruments(store, INSTRUMENTS + extras)
    store.upsert_instrument("ZZNOIND", "equity", name="Zz Unclassified Limited", is_fno=0, in_universe=1)
    summary = seed_graph(store)
    graph = GraphRepository(store).load()
    reachable = graph.reachable_from(drivers_of(graph))
    universe = [r["symbol"] for r in store.list_instruments(kind="equity", status=None)]
    unreachable = [s for s in universe if s not in reachable and s not in ("ZZNOIND", "ZZSPACE")]
    assert unreachable == []
    assert summary["unmapped"] == ["ZZNOIND"] and "ZZNOIND" not in graph.nodes
    assert "sector:space_mining" in graph.nodes and graph.nodes["sector:space_mining"].attrs["auto"] is True
    assert graph.nodes["sector:space_mining"].node_type == "sector"
    assert graph.containers_of("ZZSPACE") == {"sector:space_mining"}  # no prior links it to a driver: honest, not guessed
    assert "sector:information_technology" in graph.containers_of("ZZSOFT")
    assert "sector:textiles" in graph.containers_of("ZZTEXTILE")


def test_every_curated_stock_is_reachable_without_an_instrument_master(store, seed_data):
    seed_graph(store)
    graph = GraphRepository(store).load()
    reachable = graph.reachable_from(drivers_of(graph))
    assert [s["symbol"] for s in seed_data["stocks"] if s["symbol"] not in reachable] == []


def test_membership_follows_the_instrument_master(store):
    add_instruments(store)
    seed_graph(store)
    repo = GraphRepository(store)
    curated = {e.dst for e in repo.get_edges(src="WIPRO", relation="membership", source=CURATED_MEMBERSHIP)}
    assert "sector:information_technology" in curated  # the curated edge already covers what the master says
    assert repo.get_edges(src="WIPRO", relation="membership", source=MASTER_MEMBERSHIP) == []
    store.upsert_instrument("WIPRO", "equity", industry="Textiles")
    summary = seed_graph(store)
    assert summary["membership_master"] == 1
    moved = {e.dst for e in repo.get_edges(src="WIPRO", relation="membership", source=MASTER_MEMBERSHIP)}
    assert moved == {"sector:textiles"}
    store.upsert_instrument("WIPRO", "equity", industry="Realty")
    seed_graph(store)  # the previous master edge is replaced, not accumulated
    assert {e.dst for e in repo.get_edges(src="WIPRO", relation="membership", source=MASTER_MEMBERSHIP)} == {"sector:realty"}
    assert "sector:it_services" in {e.dst for e in repo.get_edges(src="WIPRO", relation="membership", source=CURATED_MEMBERSHIP)}


def test_remeasured_edges_survive_reseeding_unless_forced(store, seed_data):
    seed_graph(store)
    repo = GraphRepository(store)
    target = next(e for e in seed_data["edges"] if e["rel"] == "input_cost")
    prior_edge = repo.get_edge(target["src"], target["dst"], target["rel"])
    repo.upsert_edge(
        target["src"], target["dst"], target["rel"], direction=prior_edge.direction, sensitivity=-0.77, confidence=0.6,
        rationale=prior_edge.rationale + " | measured beta -0.77", source="measured", evidence_count=500,
    )
    kept = seed_graph(store)
    assert kept["edges_kept_measured"] == 1 and kept["edges_updated"] == 0
    assert repo.get_edge(target["src"], target["dst"], target["rel"]).source == "measured"
    forced = seed_graph(store, force=True)
    assert forced["edges_updated"] == 1 and forced["edges_kept_measured"] == 0
    restored = repo.get_edge(target["src"], target["dst"], target["rel"])
    assert restored.source == PRIOR_SOURCE and restored.evidence_count == 0 and restored.confidence == target["conf"]
    assert restored.version == 3  # created, measured, restored: the history is visible in the version


def test_a_changed_seed_file_updates_edges_and_bumps_versions(store, tmp_path):
    seed = {
        "version": 1,
        "nodes": [{"id": "macro:brent", "type": "commodity", "name": "Brent"}, {"id": "sector:oil_marketing", "type": "sector", "name": "OMC"}],
        "stocks": [{"symbol": "HINDPETRO", "name": "HPCL", "industry": "Oil Gas & Consumable Fuels", "sectors": ["sector:oil_marketing"]}],
        "edges": [{"src": "macro:brent", "dst": "sector:oil_marketing", "rel": "input_cost", "dir": -1, "sens": -0.4, "lag": [0.0, 1.0],
                   "conf": 0.3, "why": "crude input costs: refiners buy crude", "regime": "policy dependent"}],
    }
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    first = seed_graph(store, path=path)
    assert first["edges_created"] == 1 and first["membership_curated"] == 1
    seed["edges"][0]["conf"] = 0.35
    path.write_text(json.dumps(seed), encoding="utf-8")
    second = seed_graph(store, path=path)
    assert second["edges_updated"] == 1 and second["edges_created"] == 0
    assert GraphRepository(store).get_edge("macro:brent", "sector:oil_marketing", "input_cost").version == 2
    assert store.kv_get(seed_mod.SEED_KEY).startswith("1:")


# ------------------------------------------------------------------------------------- sector helpers
def test_industry_helpers():
    assert len(NSE_INDUSTRIES) == 20 and len(INDUSTRY_NODE_IDS) == 20 and len(set(INDUSTRY_NODE_IDS)) == 20
    assert slugify("Oil Gas & Consumable Fuels") == "oil_gas_and_consumable_fuels" and slugify("  --Metals & Mining!! ") == "metals_and_mining"
    assert industry_node_id("Metals & Mining") == "sector:metals_and_mining"
    assert industry_node_id("metals and mining") == "sector:metals_and_mining"
    assert industry_node_id("Software") == "sector:information_technology" and industry_node_id("FMCG") == "sector:fast_moving_consumer_goods"
    assert industry_node_id("Space Mining") == "sector:space_mining"
    assert industry_node_id("") is None and industry_node_id(None) is None and industry_node_id("   ") is None
    assert canonical_industry("Space Mining") is None and canonical_industry(None) is None
    assert canonical_industry("Oil, Gas & Consumable Fuels") == "Oil Gas & Consumable Fuels"
    assert all(canonical_industry(name) == name for name in NSE_INDUSTRIES)


def test_seed_path_default_points_at_the_shipped_file():
    assert Path(seed_mod.SEED_PATH).name == "graph_seed.json" and Path(seed_mod.SEED_PATH).exists()
