"""Raw items -> structured, deduplicated, point-in-time-queryable events.

``process_pending`` turns unprocessed ``news_raw`` rows and ``filings`` into ``events`` / ``event_articles`` (rules first,
optional validated LLM enrichment), marking each row processed in the same transaction as its event write, so it is
idempotent and crash-safe. The read side (``event_features``, ``catalysts_for`` ...) only uses data known at ``as_of``.

kv keys used: ``news.sig:<event_id>`` (clustering signatures, pruned with the lookup window), ``news.last_run`` (stats),
``news.reduced_coverage`` ('1' while the configured LLM is failing), ``llm:<hash>`` / ``llm.calls`` / ``llm.failures``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from app import jsonutil
from app.clock import IST, UTC, Clock, get_clock, iso, parse_ts
from app.config import Settings
from database.store import Store
from event_graph.propagation import GraphImpactEngine, StockImpact
from event_graph.repository import GraphRepository
from news_intelligence import features as feat
from news_intelligence.dedup import (
    MAX_ORIGINS,
    Clusterer,
    ClusterEvent,
    Contradiction,
    Decision,
    Item,
    Origin,
    Signature,
    event_confidence,
)
from news_intelligence.entities import Entity, EntityResolver
from news_intelligence.events import EVENT_TYPES, EventDraft, assign_polarity, classify_rules, extract_moves
from news_intelligence.llm import LLMExtractor, LLMResult, prune_llm_cache
from news_intelligence.priced_in import assess_reaction
from news_intelligence.source_quality import get_registry, split_publisher_suffix
from news_intelligence.text import clean_text, domain_of, is_probably_english, lead, url_hash
from ranking.types import Catalyst

log = logging.getLogger(__name__)

BATCH_SIZE = 100
LOOKUP_DAYS = 14
REACTION_DAYS = 5
SIG_PREFIX = "news.sig:"
MAX_ENTITIES = 24
MAX_CONTRADICTIONS = 8
ENGLISH_CODES = frozenset({"", "en", "eng", "english"})
BarsProvider = Callable[[str, str | None, str | None], pd.DataFrame]


@dataclass
class ProcessStats:
    articles_in: int = 0
    filings_in: int = 0
    events_created: int = 0
    events_updated: int = 0
    duplicates: int = 0
    llm_used: int = 0
    llm_cached: int = 0
    llm_failed: int = 0
    llm_skipped_budget: int = 0
    skipped_non_english: int = 0
    skipped_empty: int = 0
    skipped_routine: int = 0
    skipped_irrelevant: int = 0
    skipped_already_linked: int = 0
    errors: int = 0
    elapsed_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class _Prepared:
    kind: str
    row_id: int
    item: Item
    title: str
    body: str
    url: str | None
    seen_iso: str
    published_iso: str | None
    event_time_iso: str | None
    region: str | None
    draft: EventDraft
    entities: list[Entity]
    scope: str
    method: str
    quote: str | None
    quality: float
    source_kind: str
    cluster_base: str = ""
    symbol: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _primary_id(entities: Sequence[Entity]) -> str | None:
    """The event's subject for clustering: the company the event is signed for, else the most prominent mention."""
    pool = [e for e in entities if e.type == "company"] or [e for e in entities if e.polarity != 0.0] or list(entities)
    if not pool:
        return None
    return max(pool, key=lambda e: (abs(e.polarity), e.in_title, e.confidence, e.count, e.id)).id


def _entity_json(entity: Entity) -> dict[str, Any]:
    data = entity.to_dict()
    if entity.in_title:
        data["in_title"] = True
    return data


class NewsPipeline:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        *,
        clock: Clock | None = None,
        llm: LLMExtractor | None = None,
        resolver: EntityResolver | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.clock = clock or get_clock()
        self.resolver = resolver or EntityResolver(store)
        self.llm = llm if llm is not None else (LLMExtractor(settings, clock=self.clock) if settings.llm_configured else None)
        if self.llm is not None:
            self.llm.bind(store=store, resolver=self.resolver)
        self._engine: GraphImpactEngine | None = None
        self._graph_repo = GraphRepository(store)
        self._cache: dict[tuple, Any] = {}

    # ================================================================================== processing
    def process_pending(self, max_items: int = 600) -> ProcessStats:
        """Cluster unprocessed filings and articles into events; returns counts. Never duplicates events on re-run."""
        started = self.clock.monotonic()
        stats = ProcessStats()
        if self.llm is not None:
            self.llm.start_run()
        now = self.clock.now()
        clusterer = self._load_clusterer(now)
        remaining = max(0, int(max_items))
        for kind in ("filing", "article"):
            while remaining > 0:
                rows = self._fetch_batch(kind, min(BATCH_SIZE, remaining))
                if not rows:
                    break
                linked = self._linked_ids(kind, [int(r[0]) for r in rows])
                for row in rows:
                    remaining -= 1
                    if kind == "filing":
                        stats.filings_in += 1
                    else:
                        stats.articles_in += 1
                    try:
                        self._process_row(kind, row, clusterer, stats, linked)
                    except sqlite3.OperationalError:
                        stats.errors += 1
                        log.exception("database busy while processing %s %s; will retry next run", kind, row[0])
                        return self._finish(stats, started, now)
                    except Exception:
                        stats.errors += 1
                        log.exception("failed to process %s %s; marking it processed to avoid a retry loop", kind, row[0])
                        self._mark_processed(kind, int(row[0]))
                        clusterer = self._load_clusterer(now)
        return self._finish(stats, started, now)

    def _finish(self, stats: ProcessStats, started: float, now: datetime) -> ProcessStats:
        if self.llm is not None:
            stats.llm_used = max(0, self.llm.run_calls - self.llm.run_failures)
            stats.llm_failed = self.llm.run_failures
            stats.llm_cached = self.llm.run_cache_hits
            stats.llm_skipped_budget = self.llm.run_budget_skipped
            if self.llm.configured and (self.llm.run_calls or self.llm.tripped):
                failing = self.llm.run_failures > 0 and (
                    self.llm.tripped or self.llm.run_failures / max(self.llm.run_calls, 1) >= 0.2
                )
                self.store.kv_set("news.reduced_coverage", "1" if failing else "0")
        stats.elapsed_sec = round(self.clock.monotonic() - started, 3)
        self.store.kv_set_json("news.last_run", {"at": iso(now), **stats.as_dict()})
        return stats

    # ------------------------------------------------------------------------------------ fetching
    def _fetch_batch(self, kind: str, limit: int) -> list[sqlite3.Row]:
        if kind == "filing":
            sql = (
                "SELECT filing_id, exchange, symbol, isin, company, category, subject, substr(text, 1, 2000) AS text, "
                "attachment_url, disseminated_at, first_seen_at, source_ref FROM filings WHERE processed = 0 "
                "ORDER BY first_seen_at, filing_id LIMIT ?"
            )
        else:
            sql = (
                "SELECT article_id, source, feed, url, title, summary, substr(text, 1, 12000) AS text, language, published_at, "
                "first_seen_at, region, category FROM news_raw WHERE processed = 0 ORDER BY first_seen_at, article_id LIMIT ?"
            )
        return self.store.conn.execute(sql, (limit,)).fetchall()

    def _mark_processed(self, kind: str, row_id: int) -> None:
        table, column = ("filings", "filing_id") if kind == "filing" else ("news_raw", "article_id")
        try:
            self.store.conn.execute(f"UPDATE {table} SET processed = 1 WHERE {column} = ?", (row_id,))
        except sqlite3.Error:
            log.exception("could not mark %s %s processed", kind, row_id)

    def _linked_ids(self, kind: str, row_ids: Sequence[int]) -> set[int]:
        """Ids among ``row_ids`` that already belong to an event (one query per batch; the child columns are unindexed)."""
        column = "filing_id" if kind == "filing" else "article_id"
        marks = ", ".join("?" * len(row_ids))
        rows = self.store.conn.execute(
            f"SELECT DISTINCT {column} FROM event_articles WHERE {column} IN ({marks})", list(row_ids)
        ).fetchall()
        return {int(r[0]) for r in rows}

    # ------------------------------------------------------------------------------ clusterer state
    def _load_clusterer(self, now: datetime) -> Clusterer:
        clusterer = Clusterer()
        since = iso(now - timedelta(days=LOOKUP_DAYS))
        rows = self.store.conn.execute(
            "SELECT event_id, cluster_key, event_type, first_seen_at, last_updated_at, n_articles FROM events "
            "WHERE last_updated_at >= ? ORDER BY event_id",
            (since,),
        ).fetchall()
        if not rows:
            return clusterer
        blobs: dict[int, str] = {}
        ids = [int(r["event_id"]) for r in rows]
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            keys = [SIG_PREFIX + str(i) for i in chunk]
            found = self.store.conn.execute(
                f"SELECT key, value FROM kv WHERE key IN ({', '.join('?' * len(keys))})", keys
            ).fetchall()
            for kv in found:
                blobs[int(kv["key"][len(SIG_PREFIX) :])] = kv["value"]
        for row in rows:
            first, last = parse_ts(row["first_seen_at"]), parse_ts(row["last_updated_at"])
            if first is None or last is None:
                continue
            blob = jsonutil.loads(blobs.get(int(row["event_id"])), None)
            key = row["cluster_key"]
            if isinstance(blob, dict):
                event = ClusterEvent(
                    key=key, event_id=int(row["event_id"]), event_type=row["event_type"], first_seen=first.timestamp(),
                    last_seen=last.timestamp(), primary=blob.get("p"), polarity=float(blob.get("pol") or 0.0),
                    is_rumour=bool(blob.get("rum")), is_denial=bool(blob.get("den")), n_articles=int(row["n_articles"] or 1),
                    origins=[Origin.from_json(o) for o in blob.get("o", []) if isinstance(o, dict)],
                    filing_keys={str(f) for f in blob.get("fk", [])},
                )
            elif key.startswith("filing:") and key.count(":") >= 3:
                event = ClusterEvent(
                    key=key, event_id=int(row["event_id"]), event_type=row["event_type"], first_seen=first.timestamp(),
                    last_seen=last.timestamp(), primary=key.split(":")[1], polarity=0.0, n_articles=int(row["n_articles"] or 1),
                    filing_keys={key[len("filing:") :]},
                )
            else:
                continue
            clusterer.add_event(event)
        return clusterer

    # --------------------------------------------------------------------------------- one item
    def _process_row(
        self, kind: str, row: sqlite3.Row, clusterer: Clusterer, stats: ProcessStats, linked: set[int] | None = None
    ) -> None:
        row_id = int(row[0])
        if linked is not None and row_id in linked:
            stats.skipped_already_linked += 1
            self._mark_processed(kind, row_id)
            return
        prepared = self._prepare_filing(row, stats) if kind == "filing" else self._prepare_article(row, stats)
        if prepared is None:
            self._mark_processed(kind, row_id)
            return
        decision = clusterer.decide(prepared.item)
        key = self._unique_key(prepared.cluster_base, prepared) if decision.action == "new" else ""
        event = clusterer.commit(prepared.item, decision, key=key, event_id=None)
        with self.store.tx() as conn:
            if decision.action == "new":
                event.event_id = self._insert_event(conn, prepared, decision, key)
                stats.events_created += 1
            else:
                self._update_event(conn, event, prepared, decision)
                stats.events_updated += 1
                if decision.origin is not None:
                    stats.duplicates += 1
            if event.event_id is None:
                raise RuntimeError("event was not persisted")
            conn.execute(
                "INSERT INTO event_articles (event_id, article_id, filing_id, added_at) VALUES (?, ?, ?, ?)",
                (event.event_id, row_id if kind == "article" else None, row_id if kind == "filing" else None, prepared.seen_iso),
            )
            for other, record in decision.other_contradictions:
                if other.event_id is not None:
                    self._annotate(conn, other, record)
            self.store.kv_set(SIG_PREFIX + str(event.event_id), jsonutil.dumps(event.to_blob()))
            table, column = ("filings", "filing_id") if kind == "filing" else ("news_raw", "article_id")
            conn.execute(f"UPDATE {table} SET processed = 1 WHERE {column} = ?", (row_id,))

    def _unique_key(self, base: str, prepared: _Prepared) -> str:
        key = base
        for suffix in (prepared.seen_iso[:10].replace("-", ""), prepared.item.ref):
            if self.store.conn.execute("SELECT 1 FROM events WHERE cluster_key = ?", (key,)).fetchone() is None:
                return key
            key = f"{base}:{suffix}"
        return key

    # ----------------------------------------------------------------------------------- analysis
    def _analyse(
        self, title: str, body: str, *, category: str | None, source_kind: str, symbol: str | None, company: str | None,
        stats: ProcessStats,
    ) -> tuple[EventDraft, list[Entity], str, str, str | None, str | None] | None:
        """(draft, entities, scanned text, method, quote, event_time) or ``None`` when the item is not an event."""
        scan = f"{title}. {body}" if title else body
        draft = classify_rules(title, body, category=category, source_kind=source_kind)
        if draft.routine:
            stats.skipped_routine += 1
            return None
        entities = self.resolver.extract(body, title=title)
        if symbol:
            found = next((e for e in entities if e.id == symbol), None)
            if found is None:
                found = Entity("company", symbol, company or self.resolver.company_name(symbol) or symbol)
                entities.insert(0, found)
            found.confidence, found.in_title = 1.0, True
        if draft.event_type == "other" and source_kind == "exchange_filing":
            stats.skipped_irrelevant += 1  # an unclassifiable filing has no structure worth an event or an LLM call
            return None
        method, quote, event_time = "rules", None, None
        llm_polarity: dict[str, float] = {}
        result = self.llm.extract(title, body) if self.llm is not None and (draft.event_type != "other" or entities) else None
        if result is not None:
            method = "rules+llm"
            quote, event_time = self._merge_llm(draft, entities, result, scan, llm_polarity)
        pol = assign_polarity(draft, entities, scan)
        for entity in entities:
            entity.polarity = llm_polarity.get(entity.id, pol.get(entity.id, 0.0))
        for entity_id, move in extract_moves(scan, entities).items():
            next(e for e in entities if e.id == entity_id).move_pct = move
        if draft.event_type == "other" and not entities:
            stats.skipped_irrelevant += 1
            return None
        return draft, entities[:MAX_ENTITIES], scan, method, quote, event_time

    @staticmethod
    def _merge_llm(
        draft: EventDraft, entities: list[Entity], result: LLMResult, scan: str, llm_polarity: dict[str, float]
    ) -> tuple[str | None, str | None]:
        """Fold a validated LLM result into the rule draft; returns ``(verified quote, event time)``."""
        verified = result.quote_verified
        if (
            result.event_type != "other"
            and result.event_type != draft.event_type
            and verified
            and (draft.event_type == "other" or result.confidence >= draft.confidence)
        ):
            draft.event_type, draft.scope, draft.polarity_hint = result.event_type, EVENT_TYPES[result.event_type].scope, 0.0
        if verified:
            if result.severity is not None:
                draft.severity = round(0.5 * (draft.severity + result.severity), 3)
            if result.horizon:
                draft.horizon = result.horizon
            draft.confidence = max(draft.confidence, result.confidence)
        present = {e.id for e in entities}
        lowered = scan.lower()
        for entity in result.entities:
            if entity.id not in present:
                if entity.id.lower() not in lowered and entity.name.lower() not in lowered:
                    continue  # resolvable but never mentioned: treat as fabricated
                entity.in_title, entity.confidence = False, min(entity.confidence, 0.7)
                entities.append(entity)
                present.add(entity.id)
            if verified and entity.polarity:
                llm_polarity[entity.id] = entity.polarity
        return (result.evidence_quote if verified else None), result.event_time

    @staticmethod
    def _scope_for(draft: EventDraft, entities: Sequence[Entity]) -> str:
        scope = draft.scope
        has_company = any(e.type == "company" and not e.id.startswith("global:") for e in entities)
        if scope == "company" and not has_company:
            if any(e.type == "sector" for e in entities):
                return "sector"
            if entities:
                return "macro"
        return scope

    @staticmethod
    def _sane_published(published: str | None, seen_iso: str) -> str | None:
        stamp, seen = parse_ts(published), parse_ts(seen_iso)
        if stamp is None or seen is None or stamp > seen + timedelta(minutes=5):
            return None
        return iso(stamp)

    def _prepare_article(self, row: sqlite3.Row, stats: ProcessStats) -> _Prepared | None:
        title = clean_text(row["title"], 300)
        body = clean_text(row["text"] or row["summary"] or "", 6000)
        if not title and not body:
            stats.skipped_empty += 1
            return None
        language = str(row["language"] or "").strip().lower()
        if language not in ENGLISH_CODES or not is_probably_english(f"{title} {body[:300]}"):
            stats.skipped_non_english += 1
            return None
        url = row["url"]
        registry = get_registry()
        source_name = row["source"]
        host = domain_of(url)
        matched = registry.domain_match(host) if host else None
        if matched is not None and matched[1] == "aggregator":
            title, publisher = split_publisher_suffix(title)
            source_name = publisher or source_name
        quality, source_kind = registry.resolve(source_name, row["feed"], url, row["category"])
        owner = registry.publisher_key(source_name, row["feed"], url)
        analysed = self._analyse(
            title, body, category=None, source_kind=source_kind, symbol=None, company=None, stats=stats
        )
        if analysed is None:
            return None
        draft, entities, _scan, method, quote, event_time = analysed
        seen = parse_ts(row["first_seen_at"])
        if seen is None:
            raise ValueError("first_seen_at is not a timestamp")
        sig = Signature.build(title, body)
        primary = _primary_id(entities)
        polarity = next((e.polarity for e in entities if e.id == primary), 0.0)
        item = Item(
            ref=f"a:{row['article_id']}", kind="article", seen=seen.timestamp(), sig=sig, owner=owner, quality=quality,
            confidence=draft.confidence, event_type=draft.event_type, polarity=polarity, primary=primary,
            is_rumour=draft.is_rumour, is_denial=draft.is_denial, uhash=url_hash(url) if url else "", domain=host or owner,
        )
        return _Prepared(
            kind="article", row_id=int(row["article_id"]), item=item, title=title, body=body, url=url, seen_iso=iso(seen) or "",
            published_iso=self._sane_published(row["published_at"], iso(seen) or ""), event_time_iso=event_time,
            region=row["region"], draft=draft, entities=entities, scope=self._scope_for(draft, entities), method=method,
            quote=quote, quality=quality, source_kind=source_kind, cluster_base="news:" + sig.chash[:24],
        )

    def _prepare_filing(self, row: sqlite3.Row, stats: ProcessStats) -> _Prepared | None:
        title = clean_text(row["subject"] or row["category"] or "", 300)
        body = clean_text(row["text"] or "", 2000)
        if not title and not body:
            stats.skipped_empty += 1
            return None
        symbol = (row["symbol"] or "").strip() or None
        quality, source_kind = get_registry().resolve(row["exchange"], None, row["attachment_url"], "filing")
        analysed = self._analyse(
            title, body, category=row["category"], source_kind=source_kind, symbol=symbol, company=row["company"], stats=stats
        )
        if analysed is None:
            return None
        draft, entities, _scan, method, quote, event_time = analysed
        seen = parse_ts(row["first_seen_at"])
        disseminated = parse_ts(row["disseminated_at"])
        if seen is None:
            raise ValueError("first_seen_at is not a timestamp")
        day = (disseminated or seen).astimezone(IST).date().isoformat()
        filing_key = f"{symbol or row['company'] or 'NA'}:{draft.event_type}:{day}"
        sig = Signature.build(title, body)
        primary = symbol or _primary_id(entities)
        polarity = next((e.polarity for e in entities if e.id == primary), 0.0)
        owner = str(row["exchange"] or "exchange").lower()
        item = Item(
            ref=f"f:{row['filing_id']}", kind="filing", seen=seen.timestamp(), sig=sig, owner=owner, quality=quality,
            confidence=draft.confidence, event_type=draft.event_type, polarity=polarity, primary=primary,
            is_rumour=False, is_denial=False, uhash="", domain=owner, filing_key=filing_key,
        )
        disseminated_iso = iso(disseminated)
        return _Prepared(
            kind="filing", row_id=int(row["filing_id"]), item=item, title=title, body=body, url=row["attachment_url"],
            seen_iso=iso(seen) or "", published_iso=disseminated_iso, event_time_iso=event_time or disseminated_iso,
            region="india", draft=draft, entities=entities, scope=self._scope_for(draft, entities), method=method, quote=quote,
            quality=quality, source_kind=source_kind, cluster_base=f"filing:{filing_key}", symbol=symbol,
        )

    # ------------------------------------------------------------------------------------ writing
    @staticmethod
    def _evidence_entry(p: _Prepared) -> dict[str, Any]:
        entry = {
            "kind": p.kind, "ref": p.item.ref, "t": p.seen_iso, "owner": p.item.owner, "domains": [p.item.domain], "n": 1,
            "q": round(p.quality, 3), "conf": round(p.draft.confidence, 3), "pol": round(p.item.polarity, 3),
            "title": p.title[:200], "url": p.url, "published_at": p.published_iso, "quote": p.quote,
            "fkey": p.item.filing_key,
        }
        return {k: v for k, v in entry.items() if v is not None}

    @staticmethod
    def _contradiction_json(record: Contradiction) -> dict[str, Any]:
        return {
            "kind": record.kind, "t": iso(datetime.fromtimestamp(record.seen, tz=UTC)), "ref": record.ref,
            "source": record.source, "pol": round(record.polarity, 3), "against": round(record.against, 3),
        }

    def _insert_event(self, conn: sqlite3.Connection, p: _Prepared, decision: Decision, key: str) -> int:
        contradictions = [self._contradiction_json(c) for c in decision.own_contradictions][:MAX_CONTRADICTIONS]
        confidence = event_confidence(p.draft.confidence, p.quality, 1, len(contradictions))
        cursor = conn.execute(
            "INSERT INTO events (cluster_key, event_type, headline, summary, region, scope, entities_json, source_quality, "
            "confidence, novelty, severity, horizon, event_time, published_at, first_seen_at, last_updated_at, n_articles, "
            "n_independent_sources, evidence_json, contradictions_json, extraction_method) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?)",
            (
                key, p.draft.event_type, (p.title or p.body)[:300], lead(p.body, 500) or None, p.region, p.scope,
                jsonutil.dumps([_entity_json(e) for e in p.entities]), round(p.quality, 3), confidence,
                round(decision.novelty, 4), p.draft.severity, p.draft.horizon, p.event_time_iso, p.published_iso, p.seen_iso,
                p.seen_iso, jsonutil.dumps([self._evidence_entry(p)]), jsonutil.dumps(contradictions), p.method,
            ),
        )
        return int(cursor.lastrowid or 0)

    def _update_event(self, conn: sqlite3.Connection, event: ClusterEvent, p: _Prepared, decision: Decision) -> None:
        row = conn.execute(
            "SELECT headline, entities_json, source_quality, published_at, first_seen_at, last_updated_at, evidence_json, "
            "contradictions_json FROM events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"event {event.event_id} missing from the database")
        evidence = feat.json_list(row["evidence_json"])
        contradictions = feat.json_list(row["contradictions_json"])
        if decision.origin is not None:
            index = next((i for i, o in enumerate(event.origins) if o is decision.origin), None)
            if index is not None and index < len(evidence):
                entry = evidence[index]
                entry["n"] = decision.origin.n
                entry["q"] = round(max(float(entry.get("q", 0.0)), p.quality), 3)
                domains = list(entry.get("domains") or [])
                if p.item.domain and p.item.domain not in domains and len(domains) < 8:
                    domains.append(p.item.domain)
                entry["domains"] = domains
        elif len(evidence) < MAX_ORIGINS:
            evidence.append(self._evidence_entry(p))
        contradictions = (contradictions + [self._contradiction_json(c) for c in decision.own_contradictions])[:MAX_CONTRADICTIONS]
        entities = feat.json_list(row["entities_json"])
        known = {e.get("id") for e in entities}
        for entity in p.entities:
            if entity.id not in known and len(entities) < MAX_ENTITIES:
                data = _entity_json(entity)
                data["t"] = p.seen_iso
                entities.append(data)
        quality = max(float(row["source_quality"] or 0.0), p.quality)
        headline = row["headline"]
        if p.quality >= quality and p.quality - float(row["source_quality"] or 0.0) >= 0.3 and p.title:
            headline = p.title[:300]
        published = min((x for x in (row["published_at"], p.published_iso) if x), default=None)
        confidence = self._confidence_from(evidence, event.n_independent, len(contradictions))
        conn.execute(
            "UPDATE events SET headline = ?, entities_json = ?, source_quality = ?, confidence = ?, published_at = ?, "
            "first_seen_at = ?, last_updated_at = ?, n_articles = ?, n_independent_sources = ?, evidence_json = ?, "
            "contradictions_json = ? WHERE event_id = ?",
            (
                headline, jsonutil.dumps(entities), round(quality, 3), confidence, published,
                min(row["first_seen_at"], p.seen_iso), max(row["last_updated_at"], p.seen_iso), event.n_articles,
                event.n_independent, jsonutil.dumps(evidence), jsonutil.dumps(contradictions), event.event_id,
            ),
        )

    @staticmethod
    def _confidence_from(evidence: Sequence[Mapping[str, Any]], n_independent: int, n_contradictions: int) -> float:
        best_conf = max((float(e.get("conf", 0.0)) for e in evidence), default=0.3)
        best_quality = max((float(e.get("q", 0.0)) for e in evidence), default=0.2)
        return event_confidence(best_conf, best_quality, n_independent, n_contradictions)

    def _annotate(self, conn: sqlite3.Connection, event: ClusterEvent, record: Contradiction) -> None:
        row = conn.execute(
            "SELECT last_updated_at, evidence_json, contradictions_json FROM events WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        if row is None:
            return
        contradictions = feat.json_list(row["contradictions_json"])
        if len(contradictions) >= MAX_CONTRADICTIONS:
            return
        contradictions.append(self._contradiction_json(record))
        evidence = feat.json_list(row["evidence_json"])
        confidence = self._confidence_from(evidence, event.n_independent, len(contradictions))
        stamp = iso(datetime.fromtimestamp(record.seen, tz=UTC)) or row["last_updated_at"]
        conn.execute(
            "UPDATE events SET contradictions_json = ?, confidence = ?, last_updated_at = ? WHERE event_id = ?",
            (jsonutil.dumps(contradictions), confidence, max(row["last_updated_at"], stamp), event.event_id),
        )

    # ============================================================================= market reaction
    def refresh_reactions(self, as_of: Any, bars_provider: BarsProvider | None = None) -> int:
        """Fill ``market_reacted`` / ``reaction_json`` for recent company events from daily bars (point-in-time)."""
        as_of_dt, as_of_iso = self._as_of(as_of)
        since = iso(as_of_dt - timedelta(days=REACTION_DAYS))
        rows = self.store.conn.execute(
            "SELECT event_id, event_type, first_seen_at, event_time, published_at, entities_json, reaction_json FROM events "
            "WHERE first_seen_at > ? AND first_seen_at <= ? AND event_type <> 'other' AND scope IN ('company', 'global') "
            "ORDER BY event_id",
            (since, as_of_iso),
        ).fetchall()
        provider = bars_provider or self._default_bars
        preopen: dict[str, dict[str, Any]] = {}
        ist = as_of_dt.astimezone(IST)
        if ist.hour >= 9:
            preopen = self.store.latest_preopen(ist.date().isoformat(), before=as_of_iso)
        by_symbol: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            previous = jsonutil.loads(row["reaction_json"], None)
            if isinstance(previous, dict) and previous.get("as_of") == as_of_iso:
                continue
            symbol = self._event_symbol(row["entities_json"])
            if symbol:
                by_symbol.setdefault(symbol, []).append(row)
        updated = 0
        for symbol in sorted(by_symbol):
            earliest = min(r["first_seen_at"] for r in by_symbol[symbol])
            start = (parse_ts(earliest) - timedelta(days=90)).date().isoformat() if parse_ts(earliest) else None
            try:
                bars = provider(symbol, start, as_of_dt.astimezone(IST).date().isoformat())
            except Exception:
                log.debug("bars unavailable for %s", symbol, exc_info=True)
                continue
            for row in by_symbol[symbol]:
                published = row["published_at"] if row["published_at"] and row["published_at"] <= row["first_seen_at"] else None
                reaction = assess_reaction(
                    row["first_seen_at"], row["event_time"] or published, bars, as_of_dt, preopen.get(symbol), None
                )
                payload = {**reaction.to_dict(), "as_of": as_of_iso, "symbol": symbol}
                flag = None if reaction.reacted is None else int(reaction.reacted)
                with self.store.tx():
                    self.store.conn.execute(
                        "UPDATE events SET market_reacted = ?, reaction_json = ? WHERE event_id = ?",
                        (flag, jsonutil.dumps(payload), row["event_id"]),
                    )
                updated += 1
        return updated

    @staticmethod
    def _event_symbol(entities_json: str | None) -> str | None:
        best: tuple[float, str] | None = None
        for raw in feat.json_list(entities_json):
            if raw.get("type") == "company" and str(raw.get("id", "")).strip() and not str(raw["id"]).startswith("global:"):
                score = float(raw.get("confidence") or 0.0) + (0.5 if raw.get("in_title") else 0.0)
                score += 0.5 if float(raw.get("polarity") or 0.0) != 0.0 else 0.0
                if best is None or score > best[0]:
                    best = (score, str(raw["id"]))
        return best[1] if best else None

    def _default_bars(self, symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
        instrument_id = self.store.instrument_id(symbol)
        if instrument_id is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return self.store.get_bars(instrument_id, start, end)

    # ===================================================================================== pruning
    @contextmanager
    def _bulk_delete(self) -> Iterator[None]:
        """Skip per-row foreign-key lookups during a bulk delete whose child rows were detached beforehand: the child
        columns are unindexed, so each enforced lookup would scan the whole ``event_articles`` table."""
        conn = self.store.conn
        enabled = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        if not enabled or conn.in_transaction:  # the pragma is a no-op inside a transaction
            yield
            return
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            yield
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

    def prune(self, retention_days: int | None = None) -> dict[str, int]:
        """Delete raw rows older than the retention window (events are kept) and expired caches / signatures."""
        days = int(retention_days if retention_days is not None else self.settings.news_retention_days)
        now = self.clock.now()
        cutoff = iso(now - timedelta(days=days))
        out = {"news_raw": 0, "filings": 0, "event_articles_detached": 0, "signatures": 0, "llm_cache": 0}
        for table, column, extra in (("news_raw", "article_id", ""), ("filings", "filing_id", " AND processed = 1")):
            with self._bulk_delete():
                while True:
                    with self.store.tx() as conn:
                        ids = [
                            int(r[0])
                            for r in conn.execute(
                                f"SELECT {column} FROM {table} WHERE first_seen_at < ?{extra} ORDER BY {column} LIMIT 2000",
                                (cutoff,),
                            ).fetchall()
                        ]
                        if not ids:
                            break
                        marks = ", ".join("?" * len(ids))
                        detached = conn.execute(f"UPDATE event_articles SET {column} = NULL WHERE {column} IN ({marks})", ids)
                        out["event_articles_detached"] += max(detached.rowcount, 0)
                        deleted = conn.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", ids)
                        out[table] += max(deleted.rowcount, 0)
        window = iso(now - timedelta(days=LOOKUP_DAYS))
        with self.store.tx() as conn:
            removed = conn.execute(
                "DELETE FROM kv WHERE key >= ? AND key < ? AND CAST(substr(key, ?) AS INTEGER) NOT IN "
                "(SELECT event_id FROM events WHERE last_updated_at >= ?)",
                (SIG_PREFIX, SIG_PREFIX[:-1] + ";", len(SIG_PREFIX) + 1, window),
            )
            out["signatures"] = max(removed.rowcount, 0)
        out["llm_cache"] = self.llm.prune_cache() if self.llm is not None else prune_llm_cache(self.store, now)
        return out

    # ===================================================================================== queries
    def _as_of(self, as_of: Any) -> tuple[datetime, str]:
        stamp = parse_ts(as_of)
        if stamp is None:
            raise ValueError(f"unparseable as_of {as_of!r}")
        return stamp, iso(stamp) or ""

    def _events_stamp(self) -> tuple[int, str]:
        row = self.store.conn.execute("SELECT COUNT(*) AS n, COALESCE(MAX(last_updated_at), '') AS t FROM events").fetchone()
        return int(row["n"]), str(row["t"])

    def _remember(self, key: tuple, build: Callable[[], Any]) -> Any:
        full = (*key, self._events_stamp())
        if full not in self._cache:
            if len(self._cache) >= 6:
                self._cache.pop(next(iter(self._cache)))
            self._cache[full] = build()
        return self._cache[full]

    def _views(self, as_of_dt: datetime, as_of_iso: str, lookback_hours: float, scope: str | None = None) -> list[feat.EventView]:
        since = iso(as_of_dt - timedelta(hours=float(lookback_hours))) or ""
        return self._remember(
            ("views", as_of_iso, float(lookback_hours), scope),
            lambda: feat.build_views(self.store, as_of_iso, since, scope=scope),
        )

    def _engine_impacts(self, views: Sequence[feat.EventView], as_of_iso: str, tag: str) -> dict[str, StockImpact]:
        if self._engine is None:
            self._engine = GraphImpactEngine(self.store)
        engine = self._engine
        return self._remember(
            ("impacts", as_of_iso, tag, tuple(v.event_id for v in views)),
            lambda: engine.impacts([v.to_dict() for v in views], as_of_iso),
        )

    def _sector_ids(self, symbol: str) -> set[str]:
        graph = self._graph_repo.load()  # cached by a version stamp, so this is cheap per symbol
        ids = set(graph.containers_of(symbol)) if symbol in graph.nodes else set()
        industry = self.resolver.industry_node(symbol)
        if industry:
            ids.add(industry)
        return ids

    def event_features(self, symbols: Iterable[str], as_of: Any, lookback_hours: float = 36) -> dict[str, dict[str, float]]:
        """Point-in-time numeric event features per symbol (all keys always present, 0.0 when there are no events)."""
        as_of_dt, as_of_iso = self._as_of(as_of)
        wanted = list(dict.fromkeys(symbols))
        views = self._views(as_of_dt, as_of_iso, lookback_hours)
        pairs: dict[str, list[tuple[feat.EventView, float]]] = {s: [] for s in wanted}
        sector_peak: dict[str, float] = {}
        macro_views = [v for v in views if v.scope in ("macro", "global")]
        for view in views:
            if view.scope == "sector":
                for entity in view.entities:
                    if entity["type"] == "sector" and entity["confidence"] >= feat.MIN_ENTITY_CONFIDENCE:
                        sector_peak[entity["id"]] = max(sector_peak.get(entity["id"], 0.0), view.impact)
            if view.event_type == "other":
                continue
            for entity in view.entities:
                if (
                    entity["type"] == "company" and entity["id"] in pairs
                    and entity["confidence"] >= feat.MIN_ENTITY_CONFIDENCE and feat.is_subject(entity)
                ):
                    pairs[entity["id"]].append((view, float(entity["polarity"])))
        broad = max((v.impact for v in macro_views if v.event_type in feat.BROAD_TYPES), default=0.0)
        reach = self._engine_impacts(macro_views, as_of_iso, "macro") if macro_views else {}
        by_id = {v.event_id: v for v in macro_views}
        out: dict[str, dict[str, float]] = {}
        for symbol in wanted:
            sector = max((sector_peak.get(i, 0.0) for i in self._sector_ids(symbol)), default=0.0)
            impact = reach.get(symbol)
            graph_part = max((by_id[i].impact for i in impact.top_event_ids if i in by_id), default=0.0) if impact else 0.0
            out[symbol] = feat.symbol_vector(pairs[symbol], sector, max(broad, graph_part))
        return out

    def global_event_features(self, as_of: Any, lookback_hours: float = 36) -> dict[str, float]:
        as_of_dt, as_of_iso = self._as_of(as_of)
        return feat.global_features(self._views(as_of_dt, as_of_iso, lookback_hours))

    def recent_events(
        self, as_of: Any, lookback_hours: float, scope: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Most important recent events (severity x confidence x novelty), newest first on ties."""
        as_of_dt, as_of_iso = self._as_of(as_of)
        views = self._views(as_of_dt, as_of_iso, lookback_hours, scope)
        ranked = sorted(views, key=lambda v: (-(v.impact * v.novelty), v.hours_since, -v.event_id))
        return [v.to_dict() for v in ranked[: max(0, int(limit))]]

    def catalysts_for(self, symbol: str, as_of: Any, limit: int = 3) -> list[Catalyst]:
        """Best catalysts for a stock: its own events plus sector / macro events that reach it. Polarity is the sign of
        the event for THIS stock (never a trading signal)."""
        as_of_dt, as_of_iso = self._as_of(as_of)
        views = self._views(as_of_dt, as_of_iso, float(self.settings.news_lookback_hours))
        sectors = self._sector_ids(symbol)
        impacts = self._engine_impacts([v for v in views if v.event_type != "other"], as_of_iso, "all")
        reach = impacts.get(symbol)
        candidates: list[tuple[float, int, feat.EventView, float]] = []
        for view in views:
            if view.event_type == "other" and view.severity < 0.2:
                continue
            damp = 0.6 if view.market_reacted else 1.0
            own = next(
                (
                    e for e in view.entities
                    if e["type"] == "company" and e["id"] == symbol and e["confidence"] >= feat.MIN_ENTITY_CONFIDENCE
                    and feat.is_subject(e)
                ),
                None,
            )
            if own is not None:
                candidates.append((view.impact * view.novelty * damp, view.event_id, view, float(own["polarity"])))
                continue
            sector_hit = next((e for e in view.entities if e["type"] == "sector" and e["id"] in sectors), None)
            if view.scope == "sector" and sector_hit is not None:
                candidates.append((view.impact * view.novelty * damp * 0.8, view.event_id, view, float(sector_hit["polarity"])))
            elif reach is not None and view.event_id in reach.event_effects and view.scope != "company":
                effect = reach.event_effects.get(view.event_id, 0.0)
                candidates.append(
                    (view.impact * view.novelty * damp * 0.7, view.event_id, view, float((effect > 0) - (effect < 0)))
                )
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        return [
            Catalyst(
                headline=view.headline, event_type=view.event_type, source=view.source, url=view.url,
                published_at=view.published_at or view.first_seen_at, novelty=round(view.novelty, 3),
                confidence=round(view.confidence, 3), severity=round(view.severity, 3), polarity=round(pol, 3),
            )
            for _score, _eid, view, pol in candidates[: max(0, int(limit))]
        ]
