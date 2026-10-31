"""Entity resolution: companies, sectors and macro drivers mentioned in article text.

One case-insensitive alternation regex (built from a trie, so longest alias wins and scanning stays fast) covers
company names, sector keywords and macro terms. Tickers are matched separately as exact upper-case tokens;
ambiguous tickers (LT, ITC, TCS, IOC, GAIL, BEL ...) never match in all-caps text or when the context shows another
meaning. Matching is purely lexical: an entity is a *mention*, not evidence that the text is about it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from database.store import Store
from event_graph.sectors import industry_node_id

log = logging.getLogger(__name__)

ALIASES_PATH = Path(__file__).resolve().parent.parent / "data" / "seeds" / "aliases.json"
ENTITY_TYPES = ("company", "sector", "macro", "commodity", "fx", "rate", "index", "country", "institution")
MAX_ENTITIES = 24

_LEGAL_SUFFIXES = frozenset(
    "limited ltd pvt private public plc inc corporation corp company co llp".split()
)
_GENERIC_TAIL = frozenset(
    """industries industry services finance financial enterprises holdings group technologies technology laboratories labs
    pharma pharmaceuticals chemicals motors power energy infra infrastructure projects developers international global
    india bank""".split()
)
_COMMON_SINGLE = frozenset(
    """reliance titan prime star sun apollo bharat india indian national general united global royal universal western
    eastern northern southern central century premier alpha beta delta omega sigma excel ideal golden silver diamond crown
    eagle phoenix vision power energy steel cement paints motors pharma textiles sugars foods oil gas coal gold bank
    finance capital media infra tech digital smart bright fortune genesis nexus orbit pioneer unique supreme apex mega
    super grand first union federal state city metro market trade trading general industries enterprises""".split()
)
_NEVER_SYMBOL = frozenset(
    """IT US UK EU UN CEO CFO COO CTO GDP CPI WPI IIP PMI RBI SEBI FII FPI DII IPO FPO QIP NSE MPC CRR SLR EPS PAT PE ROE AI
    EV LNG USD INR EUR GBP JPY YOY QOQ FY AGM EGM NCLT ED CBI ETF SIP MF NAV AUM GST TDS TCS-TAX OPEC WTO IMF FED ECB BOJ PBOC
    NYSE LSE MSCI FTSE BRICS NATO OTT API ESG ARR NIM NPA GNPA NNPA CASA CAGR EBITDA PAT PBT USA UAE PM CM MD FM""".split()
)
_SYMBOL_TOKEN = re.compile(r"(?<![A-Za-z0-9&.\-])[A-Z][A-Z0-9]*(?:[&-][A-Z0-9]+)*(?![A-Za-z0-9&])")
_ABBREVIATION = re.compile(r"\b(dr|mr|mt|st|co|inc|ltd|corp)\b(?!\.)")
_APOSTROPHE = "'"
_VETO_WINDOW = 60


@dataclass
class Entity:
    """A resolved mention. ``polarity`` is filled later (per entity, the event's sign for it); ``spans`` are
    ``(start, end, invert)`` offsets of the first mentions in the scanned text."""

    type: str
    id: str
    name: str
    count: int = 1
    confidence: float = 0.5
    polarity: float = 0.0
    move_pct: float | None = None
    in_title: bool = False
    spans: tuple[tuple[int, int, bool], ...] = field(default=(), repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "count": int(self.count),
            "confidence": round(float(self.confidence), 3),
            "polarity": round(float(self.polarity), 3),
        }
        if self.move_pct is not None:
            out["move_pct"] = round(float(self.move_pct), 2)
        return out


def entity_from_dict(item: Any) -> Entity | None:
    """Lenient reader for stored entities (dicts, or bare ids); ``None`` when unusable."""
    if isinstance(item, str):
        item = {"id": item}
    if not isinstance(item, dict):
        return None
    ident = item.get("id") or item.get("symbol") or item.get("entity_id")
    if not ident or not isinstance(ident, str):
        return None
    etype = item.get("type")
    if etype not in ENTITY_TYPES:
        etype = "company" if not (ident.startswith(("sector:", "macro:"))) else ("sector" if ident.startswith("sector:") else "macro")

    def num(key: str, default: float) -> float:
        try:
            value = float(item.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value == value else default

    move = item.get("move_pct")
    try:
        move_pct = float(move) if move is not None else None
    except (TypeError, ValueError):
        move_pct = None
    return Entity(
        type=etype,
        id=ident,
        name=str(item.get("name") or ident),
        count=max(1, int(num("count", 1))),
        confidence=min(1.0, max(0.0, num("confidence", 0.5))),
        polarity=min(1.0, max(-1.0, num("polarity", 0.0))),
        move_pct=move_pct,
    )


@dataclass(frozen=True)
class _Target:
    id: str
    type: str
    name: str
    confidence: float
    invert: bool = False


# ----------------------------------------------------------------------------------------- regex helpers
def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u2019", _APOSTROPHE).strip().lower())


def _variants(alias: str) -> set[str]:
    """Surface variants of an alias: with/without periods and apostrophes, '&' vs 'and', hyphen vs space."""
    out = {_fold(alias)}
    steps = (
        lambda s: s.replace(".", ""),
        lambda s: s.replace(_APOSTROPHE, ""),
        lambda s: s.replace(" & ", " and ") if " & " in s else s.replace(" and ", " & "),
        lambda s: s.replace("-", " "),
        lambda s: _ABBREVIATION.sub(r"\1.", s),
    )
    for step in steps:
        out |= {_fold(step(v)) for v in out}
    return {v for v in out if v}


def _emit(node: dict) -> str:
    """Regex for a trie node: optional groups are greedy, so the longest alias matches first."""
    ends = "" in node
    branches = []
    for ch in sorted(k for k in node if k != ""):
        piece = "['\u2019]" if ch == _APOSTROPHE else re.escape(ch)
        branches.append(piece + _emit(node[ch]))
    if not branches:
        return ""
    if len(branches) == 1 and not ends:
        return branches[0]
    body = "|".join(branches)
    return f"(?:{body})?" if ends else f"(?:{body})"


def _trie_regex(words: Iterable[str], *, flags: int = 0) -> re.Pattern[str] | None:
    root: dict = {}
    count = 0
    for word in words:
        node = root
        for ch in word:
            node = node.setdefault(ch, {})
        node[""] = True
        count += 1
    if not count:
        return None
    return re.compile(r"(?<![\w&])(?:" + _emit(root) + r")(?![\w&])", flags)


def _clean_company_name(name: str) -> list[str]:
    """Distinctive name variants of an instrument name ('Reliance Industries Limited' -> 'reliance industries')."""
    tokens = [t for t in re.sub(r"[.,()]", " ", _fold(name)).split() if t]
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    out: list[str] = []
    if tokens:
        out.append(" ".join(tokens))
    short = list(tokens)
    while len(short) > 1 and short[-1] in _GENERIC_TAIL:
        short.pop()
        out.append(" ".join(short))
    return out


def _distinctive(alias: str) -> bool:
    tokens = alias.split()
    if len(tokens) >= 2:
        return len(alias) >= 7
    return len(alias) >= 5 and alias not in _COMMON_SINGLE and alias.isalpha()


def _upper_ratio(text: str, limit: int = 400) -> tuple[float, int]:
    letters = [c for c in text[:limit] if c.isalpha()]
    if not letters:
        return 0.0, 0
    return sum(c.isupper() for c in letters) / len(letters), len(letters)


@dataclass
class _Hit:
    target: _Target
    count: int = 0
    confidence: float = 0.0
    in_title: bool = False
    spans: list[tuple[int, int, bool]] = field(default_factory=list)


class EntityResolver:
    """Alias dictionaries over the instrument master plus curated seeds; rebuilt when the instrument count changes."""

    def __init__(self, store: Store, aliases_path: Path | str | None = None) -> None:
        self.store = store
        self.aliases_path = Path(aliases_path) if aliases_path else ALIASES_PATH
        self._stamp: tuple[int, float] | None = None
        self._alias_map: dict[str, _Target] = {}
        self._alias_rx: re.Pattern[str] | None = None
        self._cs_map: dict[str, _Target] = {}
        self._cs_rx: re.Pattern[str] | None = None
        self._patterns: list[tuple[re.Pattern[str], _Target]] = []
        self._symbols: dict[str, _Target] = {}
        self._by_id: dict[str, _Target] = {}
        self._ambiguous: set[str] = set()
        self._wordlike: set[str] = set()
        self._vetoes: dict[str, tuple[str, ...]] = {}
        self._industry: dict[str, str | None] = {}
        self.builds = 0

    # ------------------------------------------------------------------------------------------ build
    def _current_stamp(self) -> tuple[int, float]:
        row = self.store.conn.execute("SELECT COUNT(*) AS n FROM instruments WHERE kind = 'equity'").fetchone()
        try:
            mtime = self.aliases_path.stat().st_mtime
        except OSError:
            mtime = -1.0
        return int(row["n"]), mtime

    def refresh(self, force: bool = False) -> bool:
        stamp = self._current_stamp()
        if not force and stamp == self._stamp:
            return False
        self._build()
        self._stamp = stamp
        return True

    def _load_seed(self) -> dict[str, Any]:
        try:
            data = json.loads(self.aliases_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("aliases seed unreadable (%s); only instrument names are used", type(exc).__name__)
            return {}
        return data if isinstance(data, dict) else {}

    def _build(self) -> None:
        seed = self._load_seed()
        alias_map: dict[str, _Target] = {}
        cs_map: dict[str, _Target] = {}
        symbols: dict[str, _Target] = {}
        by_id: dict[str, _Target] = {}
        industry: dict[str, str | None] = {}
        self._ambiguous = {str(s) for s in seed.get("ambiguous_symbols", [])}
        self._wordlike = {str(s) for s in seed.get("wordlike_symbols", [])}
        self._vetoes = {str(k): tuple(_fold(p) for p in v) for k, v in (seed.get("vetoes") or {}).items()}

        def add_alias(alias: str, target: _Target, *, override: bool = False) -> None:
            for variant in _variants(alias):
                if override or variant not in alias_map:
                    alias_map[variant] = target

        # instrument master (lowest priority for conflicting aliases)
        for row in self.store.list_instruments(kind="equity", status=None):
            symbol = str(row["symbol"])
            name = str(row["name"] or symbol)
            target = _Target(symbol, "company", name, 0.85)
            symbols[symbol] = target
            by_id[symbol] = target
            industry[symbol] = industry_node_id(row["industry"] or row["sector"])
            for alias in _clean_company_name(name):
                if _distinctive(alias):
                    conf = 0.85 if len(alias.split()) >= 2 else 0.75
                    add_alias(alias, _Target(symbol, "company", name, conf))
        # curated companies
        for item in seed.get("companies", []):
            symbol = str(item.get("symbol") or "").strip()
            if not symbol:
                continue
            name = str(item.get("name") or symbol)
            known = by_id.get(symbol)
            display = known.name if known and known.name != symbol else name
            symbols[symbol] = _Target(symbol, "company", display, 0.9)
            by_id[symbol] = symbols[symbol]
            for alias in item.get("aliases", []):
                conf = 0.92 if len(str(alias).split()) >= 2 else 0.85
                add_alias(str(alias), _Target(symbol, "company", display, conf), override=True)
        for item in seed.get("global_companies", []):
            ident = str(item.get("id") or "")
            if not ident:
                continue
            target = _Target(ident, "company", str(item.get("name") or ident), 0.85)
            by_id[ident] = target
            for alias in item.get("aliases", []):
                add_alias(str(alias), target, override=True)
        for item in seed.get("sectors", []):
            ident = str(item.get("id") or "")
            if not ident:
                continue
            target = _Target(ident, "sector", str(item.get("name") or ident), 0.6)
            by_id[ident] = target
            for alias in item.get("keywords", []):
                add_alias(str(alias), target, override=True)
        for item in seed.get("macros", []):
            ident = str(item.get("id") or "")
            if not ident:
                continue
            etype = item.get("type") if item.get("type") in ENTITY_TYPES else "macro"
            target = _Target(ident, str(etype), str(item.get("name") or ident), 0.72, bool(item.get("invert")))
            by_id.setdefault(ident, _Target(ident, str(etype), str(item.get("name") or ident), 0.72))
            for alias in item.get("keywords", []):
                add_alias(str(alias), target, override=True)
            for alias in item.get("cs_keywords", []):
                cs_map[str(alias)] = _Target(ident, str(etype), str(item.get("name") or ident), 0.75, bool(item.get("invert")))
        patterns: list[tuple[re.Pattern[str], _Target]] = []
        for item in seed.get("patterns", []):
            ident = str(item.get("id") or "")
            known = by_id.get(ident)
            try:
                rx = re.compile(str(item.get("regex")), 0 if item.get("flags") == "cs" else re.IGNORECASE)
            except re.error:
                log.warning("bad alias pattern for %s ignored", ident)
                continue
            if known is not None:
                patterns.append((rx, _Target(known.id, known.type, known.name, float(item.get("confidence", 0.7)))))

        self._alias_map = alias_map
        self._alias_rx = _trie_regex(alias_map, flags=re.IGNORECASE)
        self._cs_map = cs_map
        self._cs_rx = (
            re.compile(
                r"(?<![\w.&-])(?:" + "|".join(re.escape(k) for k in sorted(cs_map, key=len, reverse=True)) + r")(?![\w&])"
            )
            if cs_map
            else None
        )
        self._patterns = patterns
        self._symbols = symbols
        self._by_id = by_id
        self._industry = industry
        self.builds += 1

    # ---------------------------------------------------------------------------------------- queries
    def known_ids(self) -> set[str]:
        self.refresh()
        return set(self._by_id)

    def industry_node(self, symbol: str) -> str | None:
        """Sector node id of a symbol's NSE industry (from the instrument master)."""
        self.refresh()
        return self._industry.get(symbol)

    def company_name(self, symbol: str) -> str | None:
        self.refresh()
        target = self._by_id.get(symbol)
        return target.name if target else None

    def resolve(self, id_or_name: str | None) -> Entity | None:
        """Resolve an id ('RELIANCE', 'sector:pharma', 'macro:brent') or a free-text name; ``None`` if unknown."""
        if not id_or_name or not isinstance(id_or_name, str):
            return None
        self.refresh()
        raw = id_or_name.strip()
        target = self._by_id.get(raw) or self._by_id.get(raw.upper()) or self._by_id.get(raw.lower())
        if target is None:
            target = self._alias_map.get(_fold(raw))
        if target is None and raw.upper() in self._symbols:
            target = self._symbols[raw.upper()]
        if target is None:
            return None
        return Entity(target.type, target.id, target.name, 1, target.confidence)

    # ---------------------------------------------------------------------------------------- extract
    def extract(self, text: str | None, *, title: str | None = None, max_entities: int = MAX_ENTITIES) -> list[Entity]:
        """Entities mentioned in ``title`` + ``text`` (spans refer to ``title + '. ' + text`` when a title is given)."""
        if not text and not title:
            return []
        self.refresh()
        title_len = len(title) if title else 0
        scan = f"{title}. {text or ''}" if title else (text or "")
        lowered = scan.lower()
        hits: dict[str, _Hit] = {}
        covered = bytearray(len(scan))

        def record(target: _Target, start: int, end: int, confidence: float, invert: bool = False) -> None:
            if any(covered[start:end]):  # already explained by a longer / earlier match
                return
            covered[start:end] = b"\x01" * (end - start)
            hit = hits.get(target.id)
            if hit is None:
                hit = hits[target.id] = _Hit(target)
            hit.count += 1
            hit.confidence = max(hit.confidence, confidence)
            if start < title_len:
                hit.in_title = True
            if len(hit.spans) < 3:
                hit.spans.append((start, end, invert))

        if self._alias_rx is not None:
            for match in self._alias_rx.finditer(scan):
                target = self._alias_map.get(_fold(match.group(0)))
                if target is not None:
                    record(target, match.start(), match.end(), target.confidence, target.invert)
        if self._cs_rx is not None:
            for match in self._cs_rx.finditer(scan):
                target = self._cs_map.get(match.group(0))
                if target is not None:
                    record(target, match.start(), match.end(), target.confidence, target.invert)
        for rx, target in self._patterns:
            for match in rx.finditer(scan):
                record(target, match.start(), match.end(), target.confidence)
        self._scan_symbols(scan, lowered, record)

        entities: list[Entity] = []
        for hit in hits.values():
            count = min(hit.count, 20)
            confidence = hit.confidence + 0.04 * min(count - 1, 3) + (0.05 if hit.in_title else 0.0)
            entities.append(
                Entity(
                    type=hit.target.type,
                    id=hit.target.id,
                    name=hit.target.name,
                    count=count,
                    confidence=round(min(0.98, confidence), 3),
                    in_title=hit.in_title,
                    spans=tuple(hit.spans),
                )
            )
        entities.sort(key=lambda e: (-e.confidence, -e.count, e.id))
        return entities[:max_entities]

    def _scan_symbols(self, scan: str, lowered: str, record) -> None:
        ratio, n_letters = _upper_ratio(scan)
        all_caps = n_letters >= 20 and ratio >= 0.75
        for match in _SYMBOL_TOKEN.finditer(scan):
            token = match.group(0)
            target = self._symbols.get(token)
            if target is None or token in _NEVER_SYMBOL:
                continue
            if token in self._ambiguous:
                if all_caps and token in self._wordlike:
                    continue
                around = lowered[max(0, match.start() - _VETO_WINDOW) : match.end() + _VETO_WINDOW]
                if any(phrase in around for phrase in self._vetoes.get(token, ())):
                    continue
                record(target, match.start(), match.end(), 0.7)
            else:
                record(target, match.start(), match.end(), 0.9)
