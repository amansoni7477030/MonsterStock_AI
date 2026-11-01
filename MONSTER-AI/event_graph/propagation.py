"""Propagation of events through the relationship graph to per-stock exposure estimates.

An impact is a signed *exposure strength*, never a trade call. Path weight =
event severity x confidence x novelty x (1 - priced-in penalty) x product of edge confidences x decay per causal hop
after the first; paths below ``min_weight`` are pruned, nodes are never revisited on a path (cycle-safe) and all
iteration orders are sorted so results are deterministic and point-in-time safe (events first seen after ``as_of`` are
ignored).

Every causal step of a chain says whether its edge is a ``[measured]`` estimate, an ``[expert prior]`` or ``[broken]``. A
relationship that stopped working (source ``relationship_break``) asserts no direction: a shock that meets one keeps flowing
down the graph with weight 0, so the chain is still reported - labelled ``[broken]`` - but adds nothing to a score, an
event effect or a confidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.clock import iso, parse_ts
from database.store import Store
from event_graph.repository import BROKEN_SOURCE, MEASURED_SOURCE, Edge, Graph, GraphRepository

PRICED_IN_PENALTY = 0.5
MAX_ENTITIES_PER_EVENT = 12
MAX_CHAINS = 3
_BASIS_TAGS = {MEASURED_SOURCE: "measured", BROKEN_SOURCE: "broken"}  # every other provenance is an expert prior


@dataclass
class StockImpact:
    symbol: str
    score: float
    direction: int
    confidence: float
    chains: list[str] = field(default_factory=list)
    top_event_ids: list[int] = field(default_factory=list)
    basis: str = "prior"
    n_events: int = 0
    event_effects: dict[int, float] = field(default_factory=dict)  # signed weight contributed by each event


@dataclass
class _Path:
    weight: float  # signed
    confidence: float
    steps: list[tuple[Edge, str]]
    sources: set[str]


@dataclass
class _Contribution:
    event_id: int
    weight: float
    confidence: float
    chain: str
    sources: set[str]


_SHORT_NAMES = {
    "brent": "Brent", "wti": "WTI", "natgas": "Natural gas", "usdinr": "USD/INR", "dxy": "Dollar index", "us10y": "US 10y yield",
    "in10y": "India 10y yield", "fed": "Fed", "rbi": "RBI", "ecb": "ECB", "boj": "BoJ", "opec": "OPEC", "sebi": "SEBI",
    "us_equities": "US equities", "sox": "SOX", "vix": "VIX", "india_vix": "India VIX", "hsi": "Hang Seng",
}


def _label(edge: Edge) -> str:
    text = (edge.rationale or "").split(":", 1)[0].strip()
    return (text if ":" in (edge.rationale or "") and text else edge.relation.replace("_", " "))[:60]


def _basis_tag(edge: Edge) -> str:
    return _BASIS_TAGS.get(edge.source, "expert prior")


def _short(entity_id: str, name: str) -> str:
    """Compact driver name for chains: 'macro:brent' -> 'Brent'."""
    if entity_id.startswith("macro:"):
        key = entity_id.split(":", 1)[1]
        return _SHORT_NAMES.get(key, key.replace("_", " ").capitalize())
    return name.split("(")[0].strip() or name


def _start_label(entity: Mapping[str, Any], node_name: str | None) -> str:
    name = _short(str(entity.get("id")), str(entity.get("name") or node_name or entity.get("id")))
    move = entity.get("move_pct")
    if isinstance(move, (int, float)) and math.isfinite(float(move)):
        return f"{name} {float(move):+.1f}%"
    polarity = float(entity.get("polarity") or 0.0)
    return f"{name} {'up' if polarity > 0 else 'down'}"


def _event_weight(event: Mapping[str, Any]) -> float:
    def number(key: str, default: float) -> float:
        try:
            value = float(event.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    reacted = event.get("market_reacted")
    if reacted is None and isinstance(event.get("reaction"), Mapping):
        reacted = event["reaction"].get("reacted")
    penalty = PRICED_IN_PENALTY if reacted in (True, 1) else 0.0
    weight = number("severity", 0.0) * number("confidence", 0.0) * number("novelty", 1.0) * (1.0 - penalty)
    return max(0.0, min(1.0, weight))


class GraphImpactEngine:
    def __init__(self, store: Store, *, point_in_time_edges: bool = False) -> None:
        """``point_in_time_edges``: for replays, propagate over the edges that were valid at each ``as_of`` (from the edge
        history) instead of today's edges. Live runs keep the default: the current graph."""
        self.store = store
        self.repo = GraphRepository(store)
        self.point_in_time_edges = point_in_time_edges

    # ------------------------------------------------------------------------------------------ paths
    def _paths_from(
        self, graph: Graph, start: str, strength: float, *, max_depth: int, decay: float, min_weight: float
    ) -> dict[str, _Path]:
        """Strongest signed path from ``start`` to every reachable stock node. A shock that meets a broken relationship keeps
        flowing with weight 0 ("carried"): the chain is reported, but nothing downstream of it is signed."""
        best: dict[str, _Path] = {}
        stack: list[tuple[str, float, float, int, int, list[str], list[tuple[Edge, str]], bool]] = [
            (start, strength, 1.0, 0, 0, [start], [], False)
        ]
        while stack:
            node, weight, path_conf, depth, hops, visited, steps, carried = stack.pop()
            if depth >= max_depth:
                continue
            for edge, target in graph.impact_edges(node):
                broken = edge.source == BROKEN_SOURCE
                if target in visited or (edge.direction == 0 and not broken):
                    continue
                causal = edge.relation != "membership"
                if broken or carried:
                    if not carried and abs(weight) < min_weight:
                        continue  # the shock was already too weak to be worth a note about a broken link
                    new_weight, new_carried = 0.0, True
                else:
                    factor = edge.confidence * (decay**hops if causal else 1.0)
                    new_weight, new_carried = weight * edge.direction * factor, False
                    if abs(new_weight) < min_weight:
                        continue
                new_steps = [*steps, (edge, target)]
                new_conf = path_conf * edge.confidence
                target_node = graph.nodes.get(target)
                if target_node is not None and target_node.node_type == "stock":
                    current = best.get(target)
                    if current is None or abs(new_weight) > abs(current.weight):
                        best[target] = _Path(new_weight, new_conf, new_steps, {e.source for e, _t in new_steps})
                stack.append(
                    (target, new_weight, new_conf, depth + 1, hops + (1 if causal else 0), [*visited, target], new_steps, new_carried)
                )
        return best

    @staticmethod
    def _chain(start: str, steps: Iterable[tuple[Edge, str]]) -> str:
        parts = [start]
        for edge, target in steps:
            if edge.relation != "membership":
                parts.append(f"{_label(edge)} [{_basis_tag(edge)}]")
            parts.append(target)
        return " -> ".join(parts)

    # ---------------------------------------------------------------------------------------- public
    def impacts(
        self,
        events: Iterable[Mapping[str, Any]],
        as_of: Any,
        *,
        max_depth: int = 4,
        decay: float = 0.6,
        min_weight: float = 0.05,
    ) -> dict[str, StockImpact]:
        """Per-symbol impacts of ``events`` (dicts as returned by ``NewsPipeline.recent_events``) known at ``as_of``."""
        as_of_ts = parse_ts(as_of)
        cutoff = iso(as_of_ts) if as_of_ts is not None else None
        graph = self.repo.graph_as_of(as_of_ts) if self.point_in_time_edges and as_of_ts is not None else self.repo.load()
        ordered = sorted(
            (e for e in events if isinstance(e, Mapping)),
            key=lambda e: (str(e.get("first_seen_at") or ""), int(e.get("event_id") or 0)),
        )
        contributions: dict[str, dict[tuple[int, str], _Contribution]] = {}

        def add(symbol: str, key: tuple[int, str], item: _Contribution) -> None:
            slot = contributions.setdefault(symbol, {})
            current = slot.get(key)
            if current is None or abs(item.weight) > abs(current.weight):
                slot[key] = item

        for event in ordered:
            seen = event.get("first_seen_at")
            if cutoff is not None and seen and str(seen) > cutoff:
                continue
            w0 = _event_weight(event)
            if w0 <= 0.0:
                continue
            event_id = int(event.get("event_id") or 0)
            headline = str(event.get("headline") or "")[:60]
            for entity in list(event.get("entities") or [])[:MAX_ENTITIES_PER_EVENT]:
                if not isinstance(entity, Mapping) or not entity.get("id"):
                    continue
                try:
                    polarity = float(entity.get("polarity") or 0.0)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(polarity) or abs(polarity) < 1e-9:
                    continue
                ident = str(entity["id"])
                strength = w0 * (0.5 + 0.5 * min(1.0, abs(polarity))) * (1 if polarity > 0 else -1)
                node = graph.nodes.get(ident)
                if entity.get("type") == "company" and not ident.startswith("global:"):
                    chain = f"{event.get('event_type', 'event')}: {headline} -> {ident}"
                    add(ident, (event_id, ident), _Contribution(event_id, strength, float(event.get("confidence") or 0.0), chain, {"direct"}))
                if node is None:
                    continue
                start = _start_label(entity, node.name)
                for symbol, path in sorted(
                    self._paths_from(graph, ident, strength, max_depth=max_depth, decay=decay, min_weight=min_weight).items()
                ):
                    add(
                        symbol, (event_id, ident),
                        _Contribution(event_id, path.weight, path.confidence * float(event.get("confidence") or 0.0),
                                      self._chain(start, path.steps), path.sources),
                    )
        return self._aggregate(contributions)

    @staticmethod
    def _aggregate(contributions: dict[str, dict[tuple[int, str], _Contribution]]) -> dict[str, StockImpact]:
        out: dict[str, StockImpact] = {}
        for symbol in sorted(contributions):
            items = sorted(contributions[symbol].values(), key=lambda c: (-abs(c.weight), c.event_id, c.chain))
            total = sum(c.weight for c in items)
            score = math.tanh(total)
            chains: list[str] = []
            for c in items:
                if c.chain not in chains:
                    chains.append(c.chain)
                if len(chains) >= MAX_CHAINS:
                    break
            event_ids: list[int] = []
            effective = [c for c in items if c.weight != 0.0]  # a broken link shows in the chains but is no effect of the event
            for c in effective:
                if c.event_id not in event_ids:
                    event_ids.append(c.event_id)
            sources = set().union(*(c.sources for c in items[:MAX_CHAINS]))
            if sources == {"direct"}:
                basis = "direct"
            elif "prior_expert" in sources:
                basis = "mixed" if sources & {"measured", "relationship_break"} else "prior"
            elif "measured" in sources:
                basis = "measured"
            else:
                basis = "relationship_break" if "relationship_break" in sources else "prior"
            out[symbol] = StockImpact(
                symbol=symbol, score=round(score, 4), direction=(score > 1e-9) - (score < -1e-9),
                confidence=round(min(0.99, max(c.confidence for c in items)), 4), chains=chains,
                top_event_ids=event_ids[:MAX_CHAINS], basis=basis, n_events=len(event_ids),
                event_effects={eid: round(sum(c.weight for c in effective if c.event_id == eid), 5) for eid in event_ids},
            )
        return out
