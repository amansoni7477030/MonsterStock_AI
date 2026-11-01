"""Idempotent loading of the expert-prior graph seed and instrument-master membership edges.

Priors are loaded with ``source='prior_expert'``, low confidence and zero evidence, so they are never presented as
measured. Re-seeding changes nothing unless the seed file changed; edges that have since been re-estimated from data
(``measured`` / ``relationship_break``) are kept unless ``force=True``.

Seed edge keys: ``src, dst, rel, dir, sens, lag [mean, sd], conf, why, regime`` (see ``data/seeds/graph_seed.json``).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from database.store import Store
from event_graph.repository import GraphRepository
from event_graph.sectors import canonical_industry, industry_node_id

log = logging.getLogger(__name__)

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "seeds" / "graph_seed.json"
PRIOR_SOURCE = "prior_expert"
CURATED_MEMBERSHIP = "membership_curated"
MASTER_MEMBERSHIP = "instrument_master"
MEMBERSHIP_CONFIDENCE = 0.9
SEED_KEY = "graph.seed_version"


def load_seed(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path else SEED_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("graph seed must be a JSON object")
    return data


def seed_fingerprint(data: dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{data.get('version', 0)}:{hashlib.sha256(blob).hexdigest()[:16]}"


def _membership(repo: GraphRepository, src: str, dst: str, source: str, name: str) -> str:
    _, result = repo.upsert_edge(
        src, dst, "membership", direction=1, sensitivity=1.0, lag_days_mean=0.0, lag_days_sd=0.0,
        confidence=MEMBERSHIP_CONFIDENCE, rationale=f"member of {name}: sector classification ({source})", source=source,
    )
    return result


def seed_graph(store: Store, *, force: bool = False, path: Path | str | None = None) -> dict[str, Any]:
    """Load priors + memberships; returns counts (``edges_created`` is 0 on an unchanged re-run)."""
    data = load_seed(path)
    repo = GraphRepository(store)
    summary: dict[str, Any] = {
        "nodes_created": 0, "nodes_updated": 0, "edges_created": 0, "edges_updated": 0, "edges_unchanged": 0,
        "edges_kept_measured": 0, "membership_curated": 0, "membership_master": 0, "unmapped": [],
    }

    def bump_node(result: str) -> None:
        if result != "unchanged":
            summary[f"nodes_{result}"] += 1

    with store.tx():
        sector_names: dict[str, str] = {}
        for node in data.get("nodes", []):
            bump_node(repo.upsert_node(node["id"], node["type"], node.get("name"), node.get("attrs") or {}))
            if node["type"] == "sector":
                sector_names[node["id"]] = node.get("name") or node["id"]
        stock_industry: dict[str, str] = {}
        for stock in data.get("stocks", []):
            symbol = stock["symbol"]
            attrs = {"industry": stock.get("industry"), "sectors": stock.get("sectors", [])}
            bump_node(repo.upsert_node(symbol, "stock", stock.get("name"), attrs))
            industry = industry_node_id(stock.get("industry"))
            if industry:
                stock_industry[symbol] = industry

        for edge in data.get("edges", []):
            relation = edge["rel"]
            lag = edge.get("lag") or [None, None]
            existing = repo.get_edge(edge["src"], edge["dst"], relation)
            if existing is not None and existing.source != PRIOR_SOURCE and not force:
                summary["edges_kept_measured"] += 1
                continue
            _, result = repo.upsert_edge(
                edge["src"], edge["dst"], relation, direction=int(edge.get("dir", 0)), sensitivity=edge.get("sens"),
                lag_days_mean=lag[0], lag_days_sd=lag[1], confidence=float(edge.get("conf", 0.3)),
                regime_dependence=edge.get("regime"), rationale=str(edge.get("why") or ""), source=PRIOR_SOURCE,
                evidence_count=0,
            )
            summary[f"edges_{result}"] += 1

        for node in data.get("nodes", []):
            parent = (node.get("attrs") or {}).get("parent")
            if parent and parent in sector_names:
                if _membership(repo, node["id"], parent, CURATED_MEMBERSHIP, sector_names[parent]) != "unchanged":
                    summary["membership_curated"] += 1
        for stock in data.get("stocks", []):
            targets = list(stock.get("sectors", []))
            if stock_industry.get(stock["symbol"]):
                targets.append(stock_industry[stock["symbol"]])
            for target in targets:
                if target in sector_names and _membership(
                    repo, stock["symbol"], target, CURATED_MEMBERSHIP, sector_names[target]
                ) != "unchanged":
                    summary["membership_curated"] += 1

        known_stocks = {s["symbol"] for s in data.get("stocks", [])}
        for row in store.list_instruments(kind="equity", status=None):
            symbol = str(row["symbol"])
            label = row["industry"] or row["sector"]
            target = industry_node_id(label)
            if target is None:
                if symbol not in known_stocks:
                    summary["unmapped"].append(symbol)
                continue
            if target not in sector_names:
                canon = canonical_industry(label)
                bump_node(repo.upsert_node(target, "sector", canon or str(label), {"level": "industry", "auto": True}))
                sector_names[target] = canon or str(label)
            if symbol not in known_stocks and repo.get_node(symbol) is None:
                bump_node(repo.upsert_node(symbol, "stock", row["name"] or symbol, {"industry": label, "sectors": []}))
            stale = [
                e for e in repo.get_edges(src=symbol, relation="membership", source=MASTER_MEMBERSHIP) if e.dst != target
            ]
            for edge_row in stale:
                repo.delete_edge(edge_row.src, edge_row.dst, edge_row.relation)
            current = repo.get_edge(symbol, target, "membership")
            if current is not None and current.source == CURATED_MEMBERSHIP:
                continue  # the curated edge already says the same; never flip provenance back and forth
            if _membership(repo, symbol, target, MASTER_MEMBERSHIP, sector_names[target]) != "unchanged":
                summary["membership_master"] += 1
        store.kv_set(SEED_KEY, seed_fingerprint(data))
    return summary
