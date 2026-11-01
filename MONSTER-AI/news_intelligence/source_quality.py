"""Source quality tiers: how much weight a publisher's report deserves as evidence.

Tiers (default scores): exchange filing 1.0, official 0.95, company statement 0.85, reputable wire/major media 0.75,
business media 0.6, aggregator 0.4, social/unknown 0.2. The mapping lives in ``data/seeds/source_quality.json``.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from pathlib import Path

from news_intelligence.text import domain_of

log = logging.getLogger(__name__)

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "seeds" / "source_quality.json"
AGGREGATOR_CAP = 0.6  # a known publisher relayed through an aggregator is never rated above business media


class SourceQuality(str, Enum):
    EXCHANGE_FILING = "exchange_filing"
    OFFICIAL = "official"
    COMPANY = "company"
    WIRE = "wire"
    BUSINESS_MEDIA = "business_media"
    AGGREGATOR = "aggregator"
    SOCIAL = "social"
    UNKNOWN = "unknown"


DEFAULT_SCORES: dict[str, float] = {
    "exchange_filing": 1.0,
    "official": 0.95,
    "company": 0.85,
    "wire": 0.75,
    "business_media": 0.6,
    "aggregator": 0.4,
    "social": 0.2,
    "unknown": 0.2,
}
# The blueprint's coarser source classes (exchange filing / official / company / media / social); a tier we cannot place stays unknown.
SOURCE_CLASS: dict[str, str] = {
    "exchange_filing": "exchange_filing",
    "official": "official",
    "company": "company",
    "wire": "media",
    "business_media": "media",
    "aggregator": "media",
    "social": "social",
    "unknown": "unknown",
}


def source_class_of(kind: str | None) -> str:
    """Blueprint source class of a quality tier: exchange_filing | official | company | media | social | unknown."""
    return SOURCE_CLASS.get(str(kind or ""), "unknown")

_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "gov", "ac", "edu", "nic"})
_SUFFIX_PUBLISHER = re.compile(r"^(.{12,}?)\s+[-|\u2013\u2014]\s+([^-|\u2013\u2014]{2,40})$")


def _norm(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


class SourceRegistry:
    """Loaded tier mapping with domain / suffix / name resolution."""

    def __init__(self, data: dict | None = None) -> None:
        data = data or {}
        scores = dict(DEFAULT_SCORES)
        for kind, score in (data.get("tiers") or {}).items():
            if kind in scores and isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0:
                scores[kind] = float(score)
        self.scores = scores
        self.categories = self._valid(data.get("categories"))
        self.domains = self._valid(data.get("domains"))
        self.suffixes = self._valid(data.get("suffixes"))
        self.names = self._valid(data.get("source_names"))
        self.feed_prefixes = self._valid(data.get("feed_prefixes"))
        keys = sorted(self.names, key=len, reverse=True)
        self._name_rx = (
            re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![a-z0-9])") if keys else None
        )

    def _valid(self, mapping: object) -> dict[str, str]:
        out: dict[str, str] = {}
        if isinstance(mapping, dict):
            for key, kind in mapping.items():
                if isinstance(key, str) and kind in self.scores:
                    out[_norm(key)] = str(kind)
        return out

    @classmethod
    def load(cls, path: Path | str | None = None) -> SourceRegistry:
        target = Path(path) if path else SEED_PATH
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("source quality file unreadable (%s); using unknown tier for everything", type(exc).__name__)
            data = {}
        return cls(data if isinstance(data, dict) else {})

    # ---------------------------------------------------------------------------------------- lookups
    def domain_match(self, host: str) -> tuple[str, str] | None:
        """(matched registry key, kind) for a host, most specific first; suffix rules last."""
        labels = [p for p in host.lower().split(".") if p]
        for i in range(max(len(labels) - 1, 0)):
            candidate = ".".join(labels[i:])
            if candidate in self.domains:
                return candidate, self.domains[candidate]
        for suffix, kind in self.suffixes.items():
            if host == suffix or host.endswith("." + suffix):
                return suffix, kind
        return None

    def name_kind(self, name: str | None) -> str | None:
        text = _norm(name)
        if not text or self._name_rx is None:
            return None
        match = self._name_rx.search(text)
        return self.names[match.group(0)] if match else None

    def feed_kind(self, feed: str | None) -> str | None:
        text = _norm(feed)
        for prefix, kind in self.feed_prefixes.items():
            if text.startswith(prefix):
                return kind
        return None

    def resolve(self, source: str | None, feed: str | None, url: str | None, category: str | None = None) -> tuple[float, str]:
        kind = self._kind(source, feed, url, category)
        return self.scores[kind], kind

    def _kind(self, source: str | None, feed: str | None, url: str | None, category: str | None) -> str:
        by_category = self.categories.get(_norm(category))
        if by_category:
            return by_category
        host = domain_of(url)
        matched = self.domain_match(host) if host else None
        by_name = self.name_kind(source)
        if matched is not None and matched[1] != "aggregator":
            return matched[1]
        if matched is not None:  # aggregator link: trust a recognised publisher only up to business-media level
            if not by_name or by_name == "aggregator":
                return "aggregator"
            return by_name if self.scores[by_name] <= AGGREGATOR_CAP else "business_media"
        return by_name or self.feed_kind(feed) or "unknown"

    def publisher_key(self, source: str | None, feed: str | None, url: str | None) -> str:
        """Identity used to count independent sources: registry domain, else registrable domain, else publisher name."""
        host = domain_of(url)
        if host:
            matched = self.domain_match(host)
            if matched is None or matched[1] != "aggregator":
                if matched is not None and "." in matched[0]:
                    return matched[0]
                labels = host.split(".")
                keep = 3 if len(labels) >= 3 and labels[-2] in _SECOND_LEVEL else 2
                return ".".join(labels[-keep:])
        name = _norm(source) or _norm(feed)
        return name or host or "unknown"


_registries: dict[str, tuple[float, SourceRegistry]] = {}


def get_registry(path: Path | str | None = None) -> SourceRegistry:
    """Cached registry (reloaded when the JSON file changes)."""
    target = Path(path) if path else SEED_PATH
    try:
        stamp = target.stat().st_mtime
    except OSError:
        stamp = -1.0
    cached = _registries.get(str(target))
    if cached is not None and cached[0] == stamp:
        return cached[1]
    registry = SourceRegistry.load(target)
    _registries[str(target)] = (stamp, registry)
    return registry


def quality_for(
    source: str | None, feed: str | None = None, url: str | None = None, category: str | None = None
) -> tuple[float, str]:
    """``(score, kind)`` for a news item, resolved by explicit category, URL domain, publisher name, then feed id."""
    return get_registry().resolve(source, feed, url, category)


def publisher_key(source: str | None, feed: str | None = None, url: str | None = None) -> str:
    return get_registry().publisher_key(source, feed, url)


# Tiers whose reports can settle a question of fact: an exchange, an official body, the company itself, a wire service and a
# business newspaper. An aggregator, social media and an unknown site cannot (anyone can write a page that looks like news).
REPUTABLE_TIERS = frozenset({"exchange_filing", "official", "company", "wire", "business_media"})


def is_reputable_host(host: str | None) -> bool:
    """True when the source registry knows ``host`` (or a parent domain of it) as a publisher in a reputable tier."""
    if not host:
        return False
    matched = get_registry().domain_match(str(host).lower())
    return matched is not None and matched[1] in REPUTABLE_TIERS


def split_publisher_suffix(title: str) -> tuple[str, str | None]:
    """Aggregators append `` - Publisher`` to headlines: returns (headline, publisher) or (title, None)."""
    match = _SUFFIX_PUBLISHER.match(title.strip())
    if not match:
        return title, None
    publisher = match.group(2).strip()
    if len(publisher.split()) > 5:
        return title, None
    return match.group(1).strip(), publisher
