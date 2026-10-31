"""Relational storage for the relationship graph (``graph_nodes`` / ``graph_edges``) plus an in-memory adjacency view.

Edges carry their provenance (source, rationale, evidence count, version), so every relationship is auditable and a
prior is never presented as a measurement. ``load()`` returns a plain-dict :class:`Graph` cached by a version stamp.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from app import jsonutil
from database.store import Store

RELATIONS = (
    "exposure", "membership", "input_cost", "export_demand", "fx_translation", "rate_sensitivity", "co_movement",
    "supply_chain",
)
_TRACKED = ("direction", "sensitivity", "lag_days_mean", "lag_days_sd", "confidence", "regime_dependence", "rationale",
            "source", "evidence_count")


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    direction: int = 0
    sensitivity: float | None = None
    lag_days_mean: float | None = None
    lag_days_sd: float | None = None
    confidence: float = 0.3
    regime_dependence: str | None = None
    rationale: str = ""
    source: str = "prior"
    evidence_count: int = 0
    version: int = 1
    last_updated: str = ""
    edge_id: int | None = None

    @property
    def key(self) -> str:
        return f"{self.src}|{self.dst}|{self.relation}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Graph:
    """Plain adjacency dicts over nodes and edges (no graph library)."""

    def __init__(self, nodes: dict[str, Node], edges: Iterable[Edge], stamp: tuple = ()) -> None:
        self.nodes = nodes
        self.stamp = stamp
        self.edges: list[Edge] = list(edges)
        self.out: dict[str, list[Edge]] = defaultdict(list)
        self.inc: dict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            self.out[edge.src].append(edge)
            self.inc[edge.dst].append(edge)

    def impact_edges(self, node_id: str) -> list[tuple[Edge, str]]:
        """Edges along which a shock at ``node_id`` propagates, as ``(edge, target node)``.

        Causal edges flow ``src -> dst``; a ``membership`` edge (member -> container) flows from the container to its
        members, never upwards.
        """
        out = [(e, e.dst) for e in self.out.get(node_id, ()) if e.relation != "membership"]
        out += [(e, e.src) for e in self.inc.get(node_id, ()) if e.relation == "membership"]
        return sorted(out, key=lambda pair: (pair[1], pair[0].relation))

    def reachable_from(self, sources: Iterable[str], max_depth: int = 8) -> set[str]:
        seen = {s for s in sources if s in self.nodes}
        frontier = set(seen)
        for _ in range(max_depth):
            nxt: set[str] = set()
            for node_id in frontier:
                for _edge, target in self.impact_edges(node_id):
                    if target not in seen:
                        seen.add(target)
                        nxt.add(target)
            if not nxt:
                break
            frontier = nxt
        return seen

    def containers_of(self, node_id: str, max_depth: int = 3) -> set[str]:
        """Sector nodes a stock/sector belongs to, following membership edges upwards."""
        found: set[str] = set()
        frontier = {node_id}
        for _ in range(max_depth):
            nxt = {e.dst for n in frontier for e in self.out.get(n, ()) if e.relation == "membership"} - found
            if not nxt:
                break
            found |= nxt
            frontier = nxt
        return found


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return abs(float(a) - float(b)) < 1e-9
    return a == b


class GraphRepository:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._cached: Graph | None = None

    # ------------------------------------------------------------------------------------------ nodes
    def upsert_node(self, node_id: str, node_type: str, name: str | None = None, attrs: dict[str, Any] | None = None) -> str:
        """Insert or update a node; returns ``created`` / ``updated`` / ``unchanged``."""
        if not node_id or not node_type:
            raise ValueError("node_id and node_type are required")
        payload = jsonutil.dumps(attrs or {})
        row = self.store.conn.execute("SELECT node_type, name, attrs_json FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            self.store.conn.execute(
                "INSERT INTO graph_nodes (node_id, node_type, name, attrs_json) VALUES (?, ?, ?, ?)",
                (node_id, node_type, name, payload),
            )
            return "created"
        if row["node_type"] == node_type and row["name"] == name and row["attrs_json"] == payload:
            return "unchanged"
        self.store.conn.execute(
            "UPDATE graph_nodes SET node_type = ?, name = ?, attrs_json = ? WHERE node_id = ?", (node_type, name, payload, node_id)
        )
        return "updated"

    def get_node(self, node_id: str) -> Node | None:
        row = self.store.conn.execute("SELECT * FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone()
        return self._node(row) if row else None

    def nodes(self, node_type: str | None = None) -> list[Node]:
        if node_type:
            rows = self.store.conn.execute("SELECT * FROM graph_nodes WHERE node_type = ? ORDER BY node_id", (node_type,)).fetchall()
        else:
            rows = self.store.conn.execute("SELECT * FROM graph_nodes ORDER BY node_id").fetchall()
        return [self._node(r) for r in rows]

    @staticmethod
    def _node(row: Any) -> Node:
        return Node(row["node_id"], row["node_type"], row["name"], jsonutil.loads(row["attrs_json"], {}) or {})

    # ------------------------------------------------------------------------------------------ edges
    @staticmethod
    def _edge(row: Any) -> Edge:
        return Edge(
            src=row["src"], dst=row["dst"], relation=row["relation"], direction=int(row["direction"]),
            sensitivity=row["sensitivity"], lag_days_mean=row["lag_days_mean"], lag_days_sd=row["lag_days_sd"],
            confidence=float(row["confidence"]), regime_dependence=row["regime_dependence"], rationale=row["rationale"] or "",
            source=row["source"], evidence_count=int(row["evidence_count"]), version=int(row["version"]),
            last_updated=row["last_updated"], edge_id=int(row["edge_id"]),
        )

    def get_edge(self, src: str, dst: str, relation: str) -> Edge | None:
        row = self.store.conn.execute(
            "SELECT * FROM graph_edges WHERE src = ? AND dst = ? AND relation = ?", (src, dst, relation)
        ).fetchone()
        return self._edge(row) if row else None

    def get_edges(
        self, src: str | None = None, dst: str | None = None, relation: str | None = None, source: str | None = None
    ) -> list[Edge]:
        clauses, params = [], []
        for column, value in (("src", src), ("dst", dst), ("relation", relation), ("source", source)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.store.conn.execute(f"SELECT * FROM graph_edges {where} ORDER BY src, dst, relation", params).fetchall()
        return [self._edge(r) for r in rows]

    def upsert_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        *,
        direction: int = 0,
        sensitivity: float | None = None,
        lag_days_mean: float | None = None,
        lag_days_sd: float | None = None,
        confidence: float = 0.3,
        regime_dependence: str | None = None,
        rationale: str = "",
        source: str = "prior",
        evidence_count: int = 0,
    ) -> tuple[Edge, str]:
        """Insert or update an edge. The version is bumped only when a tracked field actually changes."""
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation {relation!r}")
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or +1")
        for node_id in (src, dst):
            if self.get_node(node_id) is None:
                raise ValueError(f"unknown node {node_id!r}")
        values = {
            "direction": int(direction), "sensitivity": sensitivity, "lag_days_mean": lag_days_mean, "lag_days_sd": lag_days_sd,
            "confidence": max(0.0, min(1.0, float(confidence))), "regime_dependence": regime_dependence,
            "rationale": rationale, "source": source, "evidence_count": int(evidence_count),
        }
        now = self.store.now()
        existing = self.get_edge(src, dst, relation)
        if existing is None:
            self.store.conn.execute(
                "INSERT INTO graph_edges (src, dst, relation, direction, sensitivity, lag_days_mean, lag_days_sd, confidence, "
                "regime_dependence, rationale, source, evidence_count, version, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (src, dst, relation, *(values[k] for k in _TRACKED), now),
            )
            return self._require_edge(src, dst, relation), "created"
        if all(_same(getattr(existing, k), values[k]) for k in _TRACKED):
            return existing, "unchanged"
        assignments = ", ".join(f"{k} = ?" for k in _TRACKED)
        self.store.conn.execute(
            f"UPDATE graph_edges SET {assignments}, version = version + 1, last_updated = ? WHERE edge_id = ?",
            (*(values[k] for k in _TRACKED), now, existing.edge_id),
        )
        return self._require_edge(src, dst, relation), "updated"

    def _require_edge(self, src: str, dst: str, relation: str) -> Edge:
        edge = self.get_edge(src, dst, relation)
        if edge is None:  # pragma: no cover - the row was written a statement ago
            raise RuntimeError(f"edge {src}->{dst} ({relation}) vanished")
        return edge

    # --------------------------------------------------------------------------------------- views
    def _stamp(self) -> tuple:
        edges = self.store.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(version), 0) AS v, COALESCE(MAX(last_updated), '') AS t FROM graph_edges"
        ).fetchone()
        nodes = self.store.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(attrs_json)), 0) AS a FROM graph_nodes"
        ).fetchone()
        return int(edges["n"]), int(edges["v"]), edges["t"], int(nodes["n"]), int(nodes["a"])

    def load(self) -> Graph:
        """In-memory adjacency view, rebuilt only when the stored graph changed."""
        stamp = self._stamp()
        if self._cached is not None and self._cached.stamp == stamp:
            return self._cached
        nodes = {n.node_id: n for n in self.nodes()}
        rows = self.store.conn.execute("SELECT * FROM graph_edges ORDER BY edge_id").fetchall()
        self._cached = Graph(nodes, (self._edge(r) for r in rows), stamp)
        return self._cached

    def counts(self) -> dict[str, Any]:
        by_type = {
            r["node_type"]: int(r["n"])
            for r in self.store.conn.execute("SELECT node_type, COUNT(*) AS n FROM graph_nodes GROUP BY node_type").fetchall()
        }
        by_source = {
            r["source"]: int(r["n"])
            for r in self.store.conn.execute("SELECT source, COUNT(*) AS n FROM graph_edges GROUP BY source").fetchall()
        }
        by_relation = {
            r["relation"]: int(r["n"])
            for r in self.store.conn.execute("SELECT relation, COUNT(*) AS n FROM graph_edges GROUP BY relation").fetchall()
        }
        return {
            "nodes": sum(by_type.values()), "edges": sum(by_source.values()), "nodes_by_type": by_type,
            "edges_by_source": by_source, "edges_by_relation": by_relation,
        }

    # ------------------------------------------------------------------------------------- import/export
    def export_json(self) -> dict[str, Any]:
        nodes = [{"id": n.node_id, "type": n.node_type, "name": n.name, "attrs": n.attrs} for n in self.nodes()]
        edges = []
        for e in self.get_edges():
            item = e.to_dict()
            item.pop("edge_id", None)
            edges.append(item)
        return {"version": 1, "nodes": nodes, "edges": edges}

    def import_json(self, data: dict[str, Any], *, source: str | None = None) -> dict[str, int]:
        """Load nodes and edges from an export (or seed) document; ``source`` overrides the edges' provenance."""
        counts = {"nodes_created": 0, "nodes_updated": 0, "edges_created": 0, "edges_updated": 0, "edges_unchanged": 0}
        with self.store.tx():
            for node in data.get("nodes", []):
                result = self.upsert_node(str(node["id"]), str(node["type"]), node.get("name"), node.get("attrs") or {})
                if result != "unchanged":
                    counts[f"nodes_{result}"] += 1
            for edge in data.get("edges", []):
                _, result = self.upsert_edge(
                    str(edge["src"]), str(edge["dst"]), str(edge["relation"]),
                    direction=int(edge.get("direction", 0)), sensitivity=edge.get("sensitivity"),
                    lag_days_mean=edge.get("lag_days_mean"), lag_days_sd=edge.get("lag_days_sd"),
                    confidence=float(edge.get("confidence", 0.3)), regime_dependence=edge.get("regime_dependence"),
                    rationale=str(edge.get("rationale") or ""), source=source or str(edge.get("source") or "prior"),
                    evidence_count=int(edge.get("evidence_count", 0)),
                )
                counts[f"edges_{result}"] += 1
        return counts
