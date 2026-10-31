"""NSE industry classification -> graph sector node ids (shared by the entity resolver and the graph seed)."""

from __future__ import annotations

import re

NSE_INDUSTRIES: tuple[str, ...] = (
    "Financial Services",
    "Capital Goods",
    "Healthcare",
    "Automobile and Auto Components",
    "Fast Moving Consumer Goods",
    "Chemicals",
    "Information Technology",
    "Consumer Services",
    "Metals & Mining",
    "Power",
    "Oil Gas & Consumable Fuels",
    "Consumer Durables",
    "Services",
    "Realty",
    "Construction",
    "Construction Materials",
    "Telecommunication",
    "Textiles",
    "Media Entertainment & Publication",
    "Diversified",
)

# Looser spellings (other data vendors, abbreviations) folded onto the NSE industry names.
_SYNONYMS = {
    "it": "Information Technology",
    "technology": "Information Technology",
    "software": "Information Technology",
    "fmcg": "Fast Moving Consumer Goods",
    "consumer defensive": "Fast Moving Consumer Goods",
    "auto": "Automobile and Auto Components",
    "automobile": "Automobile and Auto Components",
    "automobiles": "Automobile and Auto Components",
    "pharma": "Healthcare",
    "pharmaceuticals": "Healthcare",
    "oil and gas": "Oil Gas & Consumable Fuels",
    "energy": "Oil Gas & Consumable Fuels",
    "metals": "Metals & Mining",
    "basic materials": "Metals & Mining",
    "banks": "Financial Services",
    "banking": "Financial Services",
    "financial": "Financial Services",
    "industrials": "Capital Goods",
    "utilities": "Power",
    "real estate": "Realty",
    "infrastructure": "Construction",
    "telecom": "Telecommunication",
    "communication services": "Telecommunication",
    "media": "Media Entertainment & Publication",
    "consumer cyclical": "Consumer Services",
}


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower().replace("&", " and ")).strip("_")
    return text


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\band\b|&", "", value.lower()))


_CANONICAL = {_key(name): name for name in NSE_INDUSTRIES}
_SYNONYM_KEYS = {_key(alias): _CANONICAL[_key(target)] for alias, target in _SYNONYMS.items()}


def canonical_industry(value: str | None) -> str | None:
    """Map an industry/sector label to one of the 20 NSE industries (``None`` when it is not recognised)."""
    if not value or not value.strip():
        return None
    key = _key(value)
    return _CANONICAL.get(key) or _SYNONYM_KEYS.get(key)


def industry_node_id(value: str | None) -> str | None:
    """Graph node id for an industry label; unknown labels still get a slug id so nothing is left unreachable."""
    if not value or not value.strip():
        return None
    canon = canonical_industry(value)
    return f"sector:{slugify(canon or value)}"


INDUSTRY_NODE_IDS: tuple[str, ...] = tuple(f"sector:{slugify(name)}" for name in NSE_INDUSTRIES)
