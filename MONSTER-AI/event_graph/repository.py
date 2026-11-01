"""Relational storage for the relationship graph (``graph_nodes`` / ``graph_edges``) plus an in-memory adjacency view.

Edges carry their provenance (source, rationale, evidence count, version), so every relationship is auditable and a
prior is never presented as a measurement. ``load()`` returns a plain-dict :class:`Graph` cached by a version stamp.

The graph is versioned (blueprint 23.6): every real change of an edge (created, updated, removed) appends the new state to
the append-only ``graph_edge_history`` table in the same transaction, so :meth:`GraphRepository.edges_as_of` can rebuild the
edges that were valid at any past instant and a replay does not have to use today's edges.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from app import jsonutil
from app.clock import iso, parse_ts
from database.store import Store

RELATIONS = (
    "exposure", "membership", "input_cost", "export_demand", "fx_translation", "rate_sensitivity", "co_movement",
    "supply_chain",
)
MEASURED_SOURCE = "measured"  # re-estimated from returns
BROKEN_SOURCE = "relationship_break"  # a significant measurement contradicts the prior: the edge asserts no direction
_TRACKED = ("direction", "sensitivity", "lag_days_mean", "lag_days_sd", "confidence", "regime_dependence", "rationale",
            "source", "evidence_count")
_HISTORY_COLUMNS = ("src", "dst", "relation", "version", "change", *_TRACKED, "valid_from")


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
            last_updated=row["last_updated"], edge_id=None if row["edge_id"] is None else int(row["edge_id"]),
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
        """Insert or update an edge. The version is bumped, and the new state appended to the history, only when a tracked
        field actually changes."""
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
        if existing is not None and all(_same(getattr(existing, k), values[k]) for k in _TRACKED):
            return existing, "unchanged"
        with self.store.tx():  # the edge and its history row are written together or not at all
            if existing is None:
                self.store.conn.execute(
                    "INSERT INTO graph_edges (src, dst, relation, direction, sensitivity, lag_days_mean, lag_days_sd, "
                    "confidence, regime_dependence, rationale, source, evidence_count, version, last_updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                    (src, dst, relation, *(values[k] for k in _TRACKED), now),
                )
                change = "created"
            else:
                assignments = ", ".join(f"{k} = ?" for k in _TRACKED)
                self.store.conn.execute(
                    f"UPDATE graph_edges SET {assignments}, version = version + 1, last_updated = ? WHERE edge_id = ?",
                    (*(values[k] for k in _TRACKED), now, existing.edge_id),
                )
                change = "updated"
            edge = self._require_edge(src, dst, relation)
            self._record(edge, change, now)
        return edge, change

    def delete_edge(self, src: str, dst: str, relation: str) -> bool:
        """Remove an edge (kept in the history as ``removed``); ``False`` when there was no such edge."""
        existing = self.get_edge(src, dst, relation)
        if existing is None:
            return False
        with self.store.tx():
            self.store.conn.execute("DELETE FROM graph_edges WHERE edge_id = ?", (existing.edge_id,))
            self._record(existing, "removed", self.store.now())
        return True

    def _record(self, edge: Edge, change: str, valid_from: str) -> None:
        """Append the state ``edge`` has from ``valid_from`` on (always inside the transaction that made the change)."""
        state = {**asdict(edge), "change": change, "valid_from": valid_from}
        self.store.conn.execute(
            f"INSERT INTO graph_edge_history ({', '.join(_HISTORY_COLUMNS)}) VALUES ({', '.join('?' * len(_HISTORY_COLUMNS))})",
            [state[column] for column in _HISTORY_COLUMNS],
        )

    def _require_edge(self, src: str, dst: str, relation: str) -> Edge:
        edge = self.get_edge(src, dst, relation)
        if edge is None:  # pragma: no cover - the row was written a statement ago
            raise RuntimeError(f"edge {src}->{dst} ({relation}) vanished")
        return edge

    # ---------------------------------------------------------------------------------------- history
    def edge_history(self, src: str, dst: str, relation: str) -> list[dict[str, Any]]:
        """Every recorded state of one edge, oldest first (``change``: baseline / created / updated / removed)."""
        rows = self.store.conn.execute(
            "SELECT * FROM graph_edge_history WHERE src = ? AND dst = ? AND relation = ? ORDER BY valid_from, history_id",
            (src, dst, relation),
        ).fetchall()
        return [dict(r) for r in rows]

    def edges_as_of(self, when: Any) -> list[Edge]:
        """The edges that were valid at ``when``: the latest recorded state of each edge at or before that instant. An edge
        that did not exist yet, or had been removed by then, is absent. ``last_updated`` is the time the state began."""
        stamp = parse_ts(when)
        if stamp is None:
            raise ValueError(f"unparseable as-of time {when!r}")
        rows = self.store.conn.execute(
            "SELECT h.*, h.valid_from AS last_updated, NULL AS edge_id FROM graph_edge_history h WHERE h.history_id = ("
            "SELECT h2.history_id FROM graph_edge_history h2 WHERE h2.src = h.src AND h2.dst = h.dst "
            "AND h2.relation = h.relation AND h2.valid_from <= ? ORDER BY h2.valid_from DESC, h2.history_id DESC LIMIT 1) "
            "AND h.change <> 'removed' ORDER BY h.src, h.dst, h.relation",
            (iso(stamp),),
        ).fetchall()
        return [self._edge(r) for r in rows]

    def graph_as_of(self, when: Any) -> Graph:
        """The graph as it was at ``when`` (a replay): the nodes as they are now, the edges as they were then."""
        edges = self.edges_as_of(when)
        return Graph({n.node_id: n for n in self.nodes()}, edges, ("as_of", iso(parse_ts(when))))

    # --------------------------------------------------------------------------------------- views
    def _stamp(self) -> tuple:
        edges = self.store.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(version), 0) AS v, COALESCE(MAX(last_updated), '') AS t FROM graph_edges"
        ).fetchone()
        nodes = self.store.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(attrs_json)), 0) AS a FROM graph_nodes"
        ).fetchone()
        # every change of an edge appends a history row, so this tells apart states a count / version / second cannot
        latest = self.store.conn.execute("SELECT COALESCE(MAX(history_id), 0) FROM graph_edge_history").fetchone()[0]
        return int(edges["n"]), int(edges["v"]), edges["t"], int(nodes["n"]), int(nodes["a"]), int(latest)

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
