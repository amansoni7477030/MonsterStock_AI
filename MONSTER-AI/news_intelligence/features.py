"""Point-in-time readers over the ``events`` table: feature vectors, recent events and catalyst selection.

Every read takes ``as_of`` and only uses information that was known then: events are filtered on ``first_seen_at``,
article counts on ``event_articles.added_at``, and origins / contradictions / entities on their own timestamps. Values
that later articles changed (corroboration, confidence, novelty) are recomputed from the visible evidence, never read
from the mutable columns, whenever the evidence list is present.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app import jsonutil
from app.clock import parse_ts
from database.store import Store
from news_intelligence.dedup import event_confidence, novelty_at
from news_intelligence.entities import entity_from_dict

FEATURE_KEYS: tuple[str, ...] = (
    "ev_n_24h", "ev_n_36h", "ev_max_severity", "ev_conf_weighted_severity", "ev_mean_polarity", "ev_abs_polarity",
    "ev_max_novelty", "ev_max_independent_sources", "ev_has_results", "ev_has_order", "ev_has_ma", "ev_has_regulatory",
    "ev_has_rating", "ev_has_mgmt", "ev_market_reacted_share", "ev_sector_severity", "ev_global_macro_severity",
    "ev_contradiction_flag", "ev_source_quality_max", "ev_hours_since_latest",
)
GLOBAL_KEYS: tuple[str, ...] = (
    "gl_n_events", "gl_max_severity", "gl_conf_weighted_severity", "gl_max_novelty", "gl_stress_severity",
    "gl_geopolitical_severity", "gl_central_bank_severity", "gl_commodity_severity", "gl_macro_release_severity",
    "gl_trade_policy_severity", "gl_politics_severity", "gl_global_company_severity", "gl_contradiction_flag",
    "gl_source_quality_max", "gl_hours_since_latest",
)
RESULT_TYPES = frozenset({"earnings_result", "earnings_guidance"})
ORDER_TYPES = frozenset({"order_win", "order_loss", "government_contract"})
REGULATORY_TYPES = frozenset({"litigation_regulatory"})
BROAD_TYPES = frozenset({"market_stress", "central_bank", "geopolitical_conflict", "sanctions_tariffs", "macro_release",
                         "elections_politics", "trade_policy"})
MIN_ENTITY_CONFIDENCE = 0.5
MAX_VIEWS = 6000
_COLUMNS = (
    "event_id, cluster_key, event_type, headline, summary, region, scope, entities_json, source_quality, confidence, novelty, "
    "severity, horizon, event_time, published_at, first_seen_at, last_updated_at, n_articles, n_independent_sources, "
    "market_reacted, reaction_json, evidence_json, contradictions_json, extraction_method"
)


@dataclass
class EventView:
    """An event as it looked at ``as_of`` (counts, corroboration, confidence, novelty and entities all point-in-time)."""

    event_id: int
    cluster_key: str
    event_type: str
    scope: str
    headline: str
    summary: str | None
    region: str | None
    entities: list[dict[str, Any]]
    severity: float
    confidence: float
    novelty: float
    horizon: str | None
    source_quality: float
    n_articles: int
    n_independent: int
    first_seen_at: str
    published_at: str | None
    event_time: str | None
    market_reacted: bool | None
    reaction: dict[str, Any] | None
    contradictions: list[dict[str, Any]]
    extraction_method: str
    hours_since: float
    source: str = ""
    url: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "cluster_key": self.cluster_key, "event_type": self.event_type, "scope": self.scope,
            "headline": self.headline, "summary": self.summary, "region": self.region, "entities": self.entities,
            "severity": self.severity, "confidence": self.confidence, "novelty": self.novelty, "horizon": self.horizon,
            "source_quality": self.source_quality, "n_articles": self.n_articles,
            "n_independent_sources": self.n_independent, "first_seen_at": self.first_seen_at,
            "published_at": self.published_at, "event_time": self.event_time, "market_reacted": self.market_reacted,
            "reaction": self.reaction, "contradictions": self.contradictions, "extraction_method": self.extraction_method,
            "hours_since": round(self.hours_since, 2), "source": self.source, "url": self.url, "evidence": self.evidence[:4],
        }

    def entity(self, entity_id: str) -> dict[str, Any] | None:
        for item in self.entities:
            if item.get("id") == entity_id:
                return item
        return None

    @property
    def impact(self) -> float:
        """Severity x confidence (the base importance used for ranking and for sector / macro features)."""
        return self.severity * self.confidence


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def json_list(text: str | None) -> list[dict[str, Any]]:
    data = jsonutil.loads(text, [])
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def is_subject(entity: dict[str, Any]) -> bool:
    """A company the event is actually about: signed by the event or named in the headline (not a passing mention).
    Views fix this flag before any disagreement shrinks the polarity, so a contested event still concerns its subject."""
    if "subject" in entity:
        return bool(entity["subject"])
    return float(entity.get("polarity") or 0.0) != 0.0 or bool(entity.get("in_title"))


def _lead_origin(evidence: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The origin shown for an event at ``as_of``: the best visible source, earliest on ties (later, better sources
    never leak into an earlier view because only visible evidence is considered)."""
    if not evidence:
        return {}
    best = max(range(len(evidence)), key=lambda i: (_num(evidence[i].get("q")), -i))
    return evidence[best]


def _agreement_scale(evidence: Sequence[dict[str, Any]], contradictions: Sequence[dict[str, Any]]) -> float | None:
    """Factor applied to signed entities when origins disagree: the quality-weighted mean origin polarity relative to the
    first origin's (which set the stored polarities). 0 when evenly split, negative when the weight favours the opposite
    sign. ``None`` when no polarity contradiction is visible."""
    if not evidence or not any(c.get("kind") == "polarity" for c in contradictions):
        return None
    ref = _num(evidence[0].get("pol"))
    if ref == 0.0:
        return None
    total = mean = 0.0
    for entry in evidence:
        weight = max(_num(entry.get("q")), 0.05) * max(_num(entry.get("conf")), 0.05)
        total += weight
        mean += weight * _num(entry.get("pol"))
    return max(-1.0, min(1.0, (mean / total) / ref))


def _entities(text: str | None, as_of_iso: str, first_seen: str) -> list[dict[str, Any]]:
    """Stored entities that were already attached at ``as_of`` (late additions carry their own ``t``)."""
    data = jsonutil.loads(text, [])
    out: list[dict[str, Any]] = []
    for raw in data if isinstance(data, list) else []:
        entity = entity_from_dict(raw)
        if entity is None:
            continue
        added = raw.get("t") if isinstance(raw, dict) else None
        if isinstance(added, str) and added > as_of_iso:
            continue
        item = entity.to_dict()
        item["in_title"] = bool(raw.get("in_title")) if isinstance(raw, dict) else False
        item["subject"] = is_subject(item)
        out.append(item)
    return out


def _visible_counts(store: Store, event_ids: Sequence[int], as_of_iso: str) -> dict[int, tuple[int, int]]:
    """(articles added by as_of, all articles) per event."""
    out: dict[int, tuple[int, int]] = {}
    for start in range(0, len(event_ids), 400):
        chunk = list(event_ids[start : start + 400])
        rows = store.conn.execute(
            "SELECT event_id, COUNT(*) AS total, SUM(CASE WHEN added_at <= ? THEN 1 ELSE 0 END) AS visible "
            f"FROM event_articles WHERE event_id IN ({', '.join('?' * len(chunk))}) GROUP BY event_id",
            [as_of_iso, *chunk],
        ).fetchall()
        for row in rows:
            out[int(row["event_id"])] = (int(row["visible"] or 0), int(row["total"]))
    return out


def build_views(
    store: Store, as_of_iso: str, since_iso: str, *, scope: str | None = None, limit: int = MAX_VIEWS
) -> list[EventView]:
    """Point-in-time views of events first seen in ``(since, as_of]``, newest first."""
    as_of_ts = parse_ts(as_of_iso)
    if as_of_ts is None:
        raise ValueError(f"unparseable as_of {as_of_iso!r}")
    params: list[Any] = [since_iso, as_of_iso]
    scope_sql = ""
    if scope:
        scope_sql = " AND scope = ?"
        params.append(scope)
    rows = store.conn.execute(
        f"SELECT {_COLUMNS} FROM events WHERE first_seen_at > ? AND first_seen_at <= ?{scope_sql} "
        "ORDER BY first_seen_at DESC, event_id DESC LIMIT ?",
        [*params, int(limit)],
    ).fetchall()
    counts = _visible_counts(store, [int(r["event_id"]) for r in rows], as_of_iso)
    views = []
    for row in rows:
        first_seen = row["first_seen_at"]
        first_ts = parse_ts(first_seen)
        evidence = [e for e in json_list(row["evidence_json"]) if str(e.get("t") or first_seen) <= as_of_iso]
        contradictions = [c for c in json_list(row["contradictions_json"]) if str(c.get("t") or first_seen) <= as_of_iso]
        visible, total = counts.get(int(row["event_id"]), (0, 0))
        n_articles = max(visible, 1) if total else max(int(row["n_articles"] or 1), 1)
        if evidence:
            n_independent = len({str(e.get("owner") or e.get("ref")) for e in evidence}) or 1
            best_conf = max(_num(e.get("conf"), 0.0) for e in evidence)
            best_quality = max(_num(e.get("q"), 0.0) for e in evidence)
            quality = best_quality
            confidence = event_confidence(best_conf, best_quality, n_independent, len(contradictions))
        else:
            n_independent = max(int(row["n_independent_sources"] or 1), 1)
            quality = _num(row["source_quality"], 0.0)
            confidence = _num(row["confidence"], 0.0)
        n_independent = min(n_independent, n_articles) if total else n_independent
        reaction = jsonutil.loads(row["reaction_json"], None)
        reacted: bool | None = None
        if isinstance(reaction, dict) and row["market_reacted"] is not None and str(reaction.get("as_of") or "") <= as_of_iso:
            reacted = bool(row["market_reacted"])
        else:
            reaction = None
        base_novelty = _num(row["novelty"], 1.0)
        novelty = novelty_at(base_novelty, first_ts.timestamp() if first_ts else as_of_ts.timestamp(), as_of_ts.timestamp(), n_independent)
        origin = _lead_origin(evidence)
        headline = str(origin.get("title") or row["headline"]) if evidence else row["headline"]
        published = min((str(e["published_at"]) for e in evidence if e.get("published_at")), default=None) if evidence else row["published_at"]
        entities = _entities(row["entities_json"], as_of_iso, first_seen)
        scale = _agreement_scale(evidence, contradictions)
        if scale is not None:
            for item in entities:
                if item["polarity"]:
                    item["polarity"] = round(item["polarity"] * scale, 3)
        views.append(
            EventView(
                event_id=int(row["event_id"]), cluster_key=row["cluster_key"], event_type=row["event_type"],
                scope=row["scope"] or "company", headline=headline, summary=row["summary"], region=row["region"],
                entities=entities, severity=_num(row["severity"]),
                confidence=confidence, novelty=novelty, horizon=row["horizon"], source_quality=quality,
                n_articles=n_articles, n_independent=n_independent, first_seen_at=first_seen,
                published_at=published, event_time=row["event_time"], market_reacted=reacted, reaction=reaction,
                contradictions=contradictions, extraction_method=row["extraction_method"],
                hours_since=max(0.0, (as_of_ts.timestamp() - (first_ts.timestamp() if first_ts else as_of_ts.timestamp())) / 3600.0),
                source=str(origin.get("owner") or ""), url=origin.get("url"), evidence=evidence,
            )
        )
    return views


# ------------------------------------------------------------------------------------------- features
def _prob_or(values: Iterable[float]) -> float:
    miss = 1.0
    for v in values:
        miss *= 1.0 - min(1.0, max(0.0, v))
    return 1.0 - miss


def symbol_vector(
    pairs: Sequence[tuple[EventView, float]], sector_severity: float, macro_severity: float
) -> dict[str, float]:
    """Feature vector for one symbol from its relevant events and the sector / macro severities."""
    out = dict.fromkeys(FEATURE_KEYS, 0.0)
    out["ev_sector_severity"] = round(sector_severity, 4)
    out["ev_global_macro_severity"] = round(macro_severity, 4)
    if not pairs:
        return out
    views = [v for v, _p in pairs]
    weights = [max(v.impact, 1e-9) for v in views]
    total_w = sum(weights)
    out["ev_n_24h"] = float(sum(1 for v in views if v.hours_since <= 24.0))
    out["ev_n_36h"] = float(len(views))
    out["ev_max_severity"] = max(v.severity for v in views)
    out["ev_conf_weighted_severity"] = round(_prob_or(v.impact for v in views), 4)
    out["ev_mean_polarity"] = round(sum(w * p for w, (_v, p) in zip(weights, pairs, strict=True)) / total_w, 4)
    out["ev_abs_polarity"] = round(sum(w * abs(p) for w, (_v, p) in zip(weights, pairs, strict=True)) / total_w, 4)
    out["ev_max_novelty"] = max(v.novelty for v in views)
    out["ev_max_independent_sources"] = float(max(v.n_independent for v in views))
    types = {v.event_type for v in views}
    out["ev_has_results"] = float(bool(types & RESULT_TYPES))
    out["ev_has_order"] = float(bool(types & ORDER_TYPES))
    out["ev_has_ma"] = float("merger_acquisition" in types)
    out["ev_has_regulatory"] = float(bool(types & REGULATORY_TYPES))
    out["ev_has_rating"] = float("rating_action" in types)
    out["ev_has_mgmt"] = float("management_change" in types)
    known = [v.market_reacted for v in views if v.market_reacted is not None]
    out["ev_market_reacted_share"] = round(sum(1 for r in known if r) / len(known), 4) if known else 0.0
    out["ev_contradiction_flag"] = float(any(v.contradictions for v in views))
    out["ev_source_quality_max"] = max(v.source_quality for v in views)
    out["ev_hours_since_latest"] = round(min(v.hours_since for v in views), 3)
    return out


def global_features(views: Sequence[EventView]) -> dict[str, float]:
    """Market-wide feature vector from macro / global scope events."""
    out = dict.fromkeys(GLOBAL_KEYS, 0.0)
    pool = [v for v in views if v.scope in ("macro", "global")]
    if not pool:
        return out

    def peak(types: Iterable[str]) -> float:
        wanted = set(types)
        return round(max((v.impact for v in pool if v.event_type in wanted), default=0.0), 4)

    out["gl_n_events"] = float(len(pool))
    out["gl_max_severity"] = max(v.severity for v in pool)
    out["gl_conf_weighted_severity"] = round(_prob_or(v.impact for v in pool), 4)
    out["gl_max_novelty"] = max(v.novelty for v in pool)
    out["gl_stress_severity"] = peak(["market_stress"])
    out["gl_geopolitical_severity"] = peak(["geopolitical_conflict", "sanctions_tariffs"])
    out["gl_central_bank_severity"] = peak(["central_bank"])
    out["gl_commodity_severity"] = peak(["commodity_supply"])
    out["gl_macro_release_severity"] = peak(["macro_release"])
    out["gl_trade_policy_severity"] = peak(["trade_policy"])
    out["gl_politics_severity"] = peak(["elections_politics"])
    out["gl_global_company_severity"] = peak(["global_company"])
    out["gl_contradiction_flag"] = float(any(v.contradictions for v in pool))
    out["gl_source_quality_max"] = max(v.source_quality for v in pool)
    out["gl_hours_since_latest"] = round(min(v.hours_since for v in pool), 3)
    return out
