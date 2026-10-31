"""Event clustering: syndicated copies collapse into ONE event; corroboration, novelty and contradictions are tracked.

Matching order for each incoming item:
  (c) company filings keyed by symbol + event type + exchange day;
  (a) exact content-hash / canonical-URL match;
  (b) near duplicate: simhash Hamming distance <= 3 or hashed TF-IDF cosine >= 0.85;
  (d) same story from an independent writer: same primary entity + event type within 24 h and similar wording.
Copies of the same text from different domains are ONE origin; origins differ when similarity < 0.85, and
``n_independent_sources`` counts distinct origin owners. Everything here is in-memory and deterministic; the
pipeline persists the results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from news_intelligence.text import DocFreq, content_hash, cosine, hamming, term_vector, text_simhash, tokenize

SIMHASH_MAX_DISTANCE = 3
NEAR_DUP_COSINE = 0.85
STORY_COSINE = 0.30
STORY_OVERLAP = 0.5
STORY_OVERLAP_FILING = 0.2
MERGE_WINDOW_H = 72.0
STORY_WINDOW_H = 24.0
CONTRADICTION_WINDOW_H = 24.0
RESURFACE_DAYS = 7.0
MAX_ORIGINS = 16
MAX_DOMAINS = 8
VEC_TERMS = 40
POSTING_TERMS = 12
NOVELTY_FOLLOWUP = 0.35
NOVELTY_RESURFACED = 0.10
RUMOUR_NOVELTY_FACTOR = 0.7


# ----------------------------------------------------------------------------------------------- model
@dataclass
class Signature:
    """Compact fingerprint of an article's headline + lead (what near-duplicate matching compares)."""

    simhash: int
    vec: dict[int, float]
    chash: str

    @classmethod
    def build(cls, title: str, body: str = "") -> Signature:
        tokens = tokenize(f"{title} {(body or '')[:600]}", max_tokens=250)
        return cls(text_simhash(tokens), term_vector(tokens, top_k=VEC_TERMS), content_hash(title, body))

    def terms(self) -> frozenset[int]:
        return frozenset(self.vec)

    def to_json(self) -> dict[str, Any]:
        return {"h": f"{self.simhash:016x}", "v": [[b, round(w, 3)] for b, w in sorted(self.vec.items())], "c": self.chash[:20]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Signature:
        vec = {int(b): float(w) for b, w in data.get("v", [])}
        return cls(int(data.get("h", "0"), 16), vec, str(data.get("c", "")))


@dataclass
class Origin:
    """One distinct piece of text; later copies (any domain) only bump ``n`` and add domains."""

    sig: Signature
    owner: str
    t: float
    quality: float
    confidence: float = 0.5
    polarity: float = 0.0
    n: int = 1
    domains: list[str] = field(default_factory=list)
    uhash: str = ""
    kind: str = "article"

    def to_json(self) -> dict[str, Any]:
        return {
            "s": self.sig.to_json(), "w": self.owner, "t": int(self.t), "q": round(self.quality, 3),
            "k": round(self.confidence, 3), "p": round(self.polarity, 3), "n": self.n, "d": self.domains[:MAX_DOMAINS],
            "u": self.uhash, "x": self.kind,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Origin:
        return cls(
            Signature.from_json(data.get("s", {})), str(data.get("w", "")), float(data.get("t", 0)), float(data.get("q", 0.2)),
            float(data.get("k", 0.5)), float(data.get("p", 0.0)), int(data.get("n", 1)), list(data.get("d", [])),
            str(data.get("u", "")), str(data.get("x", "article")),
        )


@dataclass
class ClusterEvent:
    key: str
    event_id: int | None
    event_type: str
    first_seen: float
    last_seen: float
    primary: str | None
    polarity: float
    is_rumour: bool = False
    is_denial: bool = False
    n_articles: int = 1
    origins: list[Origin] = field(default_factory=list)
    filing_keys: set[str] = field(default_factory=set)
    contradictions: int = 0

    @property
    def n_independent(self) -> int:
        return len({o.owner for o in self.origins}) or 1

    def to_blob(self) -> dict[str, Any]:
        return {
            "p": self.primary, "pol": round(self.polarity, 3), "rum": int(self.is_rumour), "den": int(self.is_denial),
            "fk": sorted(self.filing_keys), "o": [o.to_json() for o in self.origins[:MAX_ORIGINS]],
        }


@dataclass
class Item:
    """One incoming article or filing, prepared for clustering."""

    ref: str
    kind: str  # "article" | "filing"
    seen: float
    sig: Signature
    owner: str
    quality: float
    confidence: float
    event_type: str
    polarity: float = 0.0
    primary: str | None = None
    is_rumour: bool = False
    is_denial: bool = False
    uhash: str = ""
    domain: str = ""
    filing_key: str | None = None


@dataclass
class Contradiction:
    kind: str  # "polarity" | "denial"
    ref: str  # what conflicts (an item reference or an event key)
    seen: float
    polarity: float
    against: float
    source: str


@dataclass
class Decision:
    action: str  # "new" | "attach"
    match: str  # new | filing_key | exact_content | exact_url | near_simhash | near_cosine | story_link
    event: ClusterEvent | None = None
    similarity: float = 0.0
    origin: Origin | None = None  # the origin a copy merges into (None -> a new origin)
    novelty: float = 1.0
    followup_of: ClusterEvent | None = None
    resurfaced_of: ClusterEvent | None = None
    own_contradictions: list[Contradiction] = field(default_factory=list)
    other_contradictions: list[tuple[ClusterEvent, Contradiction]] = field(default_factory=list)


# -------------------------------------------------------------------------------------------- scoring
def base_novelty(*, rumour: bool = False, followup: bool = False, resurfaced: bool = False) -> float:
    """Novelty of a newly created event: 1 for a first sighting, low for old stories, penalised for rumours."""
    value = NOVELTY_RESURFACED if resurfaced else NOVELTY_FOLLOWUP if followup else 1.0
    return value * (RUMOUR_NOVELTY_FACTOR if rumour else 1.0)


def novelty_at(base: float, first_seen: float, as_of: float, n_independent: int) -> float:
    """Novelty decays with age (48 h half-life) and with independent coverage (the story becomes known)."""
    age_hours = max(0.0, (as_of - first_seen) / 3600.0)
    coverage = 1.0 / (1.0 + 0.12 * max(0, n_independent - 1))
    return round(min(1.0, max(0.0, base * 0.5 ** (age_hours / 48.0) * coverage)), 4)


def event_confidence(best_confidence: float, best_quality: float, n_independent: int, n_contradictions: int) -> float:
    """Confidence in an event: classifier confidence x source quality, boosted by independent corroboration and
    reduced (never hidden) by contradictions."""
    corroboration = 1.0 + 0.12 * min(max(n_independent - 1, 0), 4)
    value = best_confidence * (0.55 + 0.45 * best_quality) * corroboration
    value *= max(0.45, 0.75 ** min(n_contradictions, 3))
    return round(min(0.97, max(0.0, value)), 3)


def _blocks(simhash: int) -> tuple[int, int, int, int]:
    """Four 16-bit blocks: two hashes within Hamming distance 3 share at least one identical block."""
    return simhash & 0xFFFF, (simhash >> 16) & 0xFFFF, (simhash >> 32) & 0xFFFF, (simhash >> 48) & 0xFFFF


def _heaviest(vec: dict[int, float], n: int) -> list[int]:
    return [b for b, _ in sorted(vec.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


# ------------------------------------------------------------------------------------------ clusterer
class Clusterer:
    """In-memory index of recent events used to assign each new item to an event (or start a new one)."""

    def __init__(
        self,
        *,
        merge_window_h: float = MERGE_WINDOW_H,
        story_window_h: float = STORY_WINDOW_H,
        resurface_days: float = RESURFACE_DAYS,
    ) -> None:
        self.merge_window_h = merge_window_h
        self.story_window_h = story_window_h
        self.resurface_days = resurface_days
        self.events: dict[str, ClusterEvent] = {}
        self.df = DocFreq()
        self._by_chash: dict[str, list[str]] = {}
        self._by_uhash: dict[str, list[str]] = {}
        self._blocks: list[dict[int, set[str]]] = [{}, {}, {}, {}]
        self._postings: dict[int, set[str]] = {}
        self._by_primary: dict[str, list[str]] = {}
        self._filing_keys: dict[str, str] = {}

    # -------------------------------------------------------------------------------------- indexing
    def _index_origin(self, key: str, origin: Origin) -> None:
        sig = origin.sig
        if sig.chash:
            self._by_chash.setdefault(sig.chash[:20], []).append(key)
        if origin.uhash:
            self._by_uhash.setdefault(origin.uhash, []).append(key)
        for table, block in zip(self._blocks, _blocks(sig.simhash), strict=True):
            table.setdefault(block, set()).add(key)
        for bucket in _heaviest(sig.vec, POSTING_TERMS):
            self._postings.setdefault(bucket, set()).add(key)
        self.df.add(sig.vec.keys())

    def add_event(self, event: ClusterEvent) -> None:
        """Index an event (also used when loading persisted events)."""
        self.events[event.key] = event
        if event.primary:
            self._by_primary.setdefault(event.primary, []).append(event.key)
        for fkey in event.filing_keys:
            self._filing_keys[fkey] = event.key
        for origin in event.origins:
            self._index_origin(event.key, origin)

    def __len__(self) -> int:
        return len(self.events)

    # ----------------------------------------------------------------------------------------- query
    def _best_origin(self, sig: Signature, event: ClusterEvent) -> tuple[Origin | None, float, int]:
        best: Origin | None = None
        best_cos, best_ham = -1.0, 64
        for origin in event.origins:
            cos = 1.0 if origin.sig.chash and origin.sig.chash[:20] == sig.chash[:20] else cosine(sig.vec, origin.sig.vec, self.df)
            ham = hamming(sig.simhash, origin.sig.simhash)
            if cos > best_cos or (cos == best_cos and ham < best_ham):
                best, best_cos, best_ham = origin, cos, ham
        return best, best_cos, best_ham

    def _exact(self, item: Item) -> tuple[ClusterEvent, str] | None:
        found: list[tuple[float, str, str]] = []
        for key in self._by_chash.get(item.sig.chash[:20], []):
            event = self.events.get(key)
            if event is not None:
                found.append((event.last_seen, key, "exact_content"))
        if item.uhash:
            for key in self._by_uhash.get(item.uhash, []):
                event = self.events.get(key)
                if event is not None:
                    found.append((event.last_seen, key, "exact_url"))
        if not found:
            return None
        _, key, how = max(found)
        return self.events[key], how

    def _near(self, item: Item) -> tuple[ClusterEvent, str, float] | None:
        keys: set[str] = set()
        for table, block in zip(self._blocks, _blocks(item.sig.simhash), strict=True):
            keys |= table.get(block, set())
        shared: Counter[str] = Counter()
        for bucket in _heaviest(item.sig.vec, POSTING_TERMS):
            for key in self._postings.get(bucket, ()):
                shared[key] += 1
        keys |= {k for k, c in shared.items() if c >= 3}
        best: tuple[float, int, ClusterEvent, str] | None = None
        for key in sorted(keys):
            event = self.events.get(key)
            if event is None:
                continue
            origin, cos, ham = self._best_origin(item.sig, event)
            if origin is None or not (ham <= SIMHASH_MAX_DISTANCE or cos >= NEAR_DUP_COSINE):
                continue
            how = "near_simhash" if ham <= SIMHASH_MAX_DISTANCE else "near_cosine"
            candidate = (max(cos, 1.0 - ham / 64.0), -ham, event, how)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        return (best[2], best[3], best[0]) if best else None

    def _story_link(self, item: Item) -> tuple[ClusterEvent, float] | None:
        if not item.primary or item.event_type == "other":
            return None
        item_terms = item.sig.terms()
        best: tuple[float, ClusterEvent] | None = None
        for key in self._by_primary.get(item.primary, []):
            event = self.events.get(key)
            if event is None or event.event_type != item.event_type:
                continue
            if item.kind == "filing" and event.filing_keys:
                continue  # a filing of another day is a distinct disclosure; same-day ones match on the key
            if abs(item.seen - event.first_seen) > self.story_window_h * 3600:
                continue
            for origin in event.origins:
                cos = cosine(item.sig.vec, origin.sig.vec, self.df)
                terms = origin.sig.terms()
                shared = len(item_terms & terms)
                overlap = shared / min(len(item_terms), len(terms)) if item_terms and terms else 0.0
                relaxed = item.kind == "filing" or origin.kind == "filing"  # filings are short summaries of the same facts
                lexical = shared >= (3 if relaxed else 4) and overlap >= (STORY_OVERLAP_FILING if relaxed else STORY_OVERLAP)
                if cos >= STORY_COSINE or lexical:
                    score = max(cos, overlap * 0.6 if lexical else 0.0)
                    if best is None or score > best[0]:
                        best = (score, event)
        return (best[1], best[0]) if best else None

    # ------------------------------------------------------------------------------------- decision
    def decide(self, item: Item) -> Decision:
        """Assign ``item`` to an existing event or start a new one (does not modify the index)."""
        fkey = self._filing_keys.get(item.filing_key) if item.filing_key else None
        if fkey is not None and fkey in self.events:
            return self._attach(item, self.events[fkey], "filing_key", 1.0)
        exact = self._exact(item)
        if exact is not None:
            return self._by_age(item, exact[0], exact[1], 1.0)
        near = self._near(item)
        if near is not None:
            return self._by_age(item, near[0], near[1], near[2])
        link = self._story_link(item)
        if link is not None:
            return self._attach(item, link[0], "story_link", link[1])
        return self._new(item)

    def _by_age(self, item: Item, event: ClusterEvent, match: str, similarity: float) -> Decision:
        gap_hours = (item.seen - event.last_seen) / 3600.0
        if gap_hours <= self.merge_window_h:
            return self._attach(item, event, match, similarity)
        if gap_hours / 24.0 <= self.resurface_days:
            decision = self._new(item, followup_of=event)
        else:
            decision = self._new(item, resurfaced_of=event)
        decision.match, decision.similarity = match, similarity
        return decision

    def _attach(self, item: Item, event: ClusterEvent, match: str, similarity: float) -> Decision:
        origin, cos, ham = self._best_origin(item.sig, event)
        same = origin if origin is not None and (cos >= NEAR_DUP_COSINE or ham <= SIMHASH_MAX_DISTANCE) else None
        age_hours = max(0.0, (item.seen - event.first_seen) / 3600.0)
        decision = Decision("attach", match, event, similarity, same, novelty=round(0.5 * 0.5 ** (age_hours / 24.0), 4))
        opposite = abs(event.polarity) >= 0.3 and abs(item.polarity) >= 0.3 and event.polarity * item.polarity < 0
        if same is None and opposite:
            decision.own_contradictions.append(
                Contradiction("polarity", event.key, item.seen, item.polarity, event.polarity, item.owner)
            )
        if item.is_denial and event.is_rumour:
            decision.own_contradictions.append(
                Contradiction("denial", event.key, item.seen, item.polarity, event.polarity, item.owner)
            )
        self._cross_contradictions(item, decision, skip=event)
        return decision

    def _new(self, item: Item, *, followup_of: ClusterEvent | None = None, resurfaced_of: ClusterEvent | None = None) -> Decision:
        novelty = base_novelty(rumour=item.is_rumour, followup=followup_of is not None, resurfaced=resurfaced_of is not None)
        decision = Decision("new", "new", None, 0.0, None, novelty, followup_of, resurfaced_of)
        self._cross_contradictions(item, decision, skip=None)
        return decision

    def _cross_contradictions(self, item: Item, decision: Decision, *, skip: ClusterEvent | None) -> None:
        """Opposite-polarity events about the same entity and type within 24 h (and denials of rumours)."""
        if not item.primary:
            return
        window = CONTRADICTION_WINDOW_H * 3600
        for key in self._by_primary.get(item.primary, []):
            other = self.events.get(key)
            if other is None or other is skip or other.event_type != item.event_type:
                continue
            recent = abs(item.seen - other.last_seen) <= window
            opposite = recent and abs(other.polarity) >= 0.3 and abs(item.polarity) >= 0.3 and other.polarity * item.polarity < 0
            denial = item.is_denial and other.is_rumour and abs(item.seen - other.last_seen) <= self.merge_window_h * 3600
            if not (opposite or denial):
                continue
            kind = "polarity" if opposite else "denial"
            source = other.origins[0].owner if other.origins else ""
            decision.own_contradictions.append(Contradiction(kind, other.key, item.seen, other.polarity, item.polarity, source))
            decision.other_contradictions.append(
                (other, Contradiction(kind, item.ref, item.seen, item.polarity, other.polarity, item.owner))
            )

    # --------------------------------------------------------------------------------------- commit
    def commit(self, item: Item, decision: Decision, *, key: str, event_id: int | None) -> ClusterEvent:
        """Apply a decision to the in-memory index and return the (new or updated) event."""
        if decision.action == "new":
            event = ClusterEvent(
                key=key, event_id=event_id, event_type=item.event_type, first_seen=item.seen, last_seen=item.seen,
                primary=item.primary, polarity=item.polarity, is_rumour=item.is_rumour, is_denial=item.is_denial,
                n_articles=1, origins=[self._origin_of(item)], filing_keys={item.filing_key} if item.filing_key else set(),
                contradictions=len(decision.own_contradictions),
            )
            self.add_event(event)
        else:
            event = decision.event
            if event is None:
                raise ValueError("attach decision without an event")
            event.n_articles += 1
            event.first_seen = min(event.first_seen, item.seen)
            event.last_seen = max(event.last_seen, item.seen)
            event.contradictions += len(decision.own_contradictions)
            if decision.origin is not None:
                decision.origin.n += 1
                decision.origin.quality = max(decision.origin.quality, item.quality)
                if item.domain and item.domain not in decision.origin.domains and len(decision.origin.domains) < MAX_DOMAINS:
                    decision.origin.domains.append(item.domain)
            elif len(event.origins) < MAX_ORIGINS:
                origin = self._origin_of(item)
                event.origins.append(origin)
                self._index_origin(event.key, origin)
            if item.filing_key and item.filing_key not in event.filing_keys:
                event.filing_keys.add(item.filing_key)
                self._filing_keys[item.filing_key] = event.key
            if item.uhash:
                self._by_uhash.setdefault(item.uhash, []).append(event.key)
        for other, _record in decision.other_contradictions:
            other.contradictions += 1
        return event

    @staticmethod
    def _origin_of(item: Item) -> Origin:
        return Origin(
            sig=item.sig, owner=item.owner, t=item.seen, quality=item.quality, confidence=item.confidence,
            polarity=item.polarity, n=1, domains=[item.domain] if item.domain else [], uhash=item.uhash, kind=item.kind,
        )
