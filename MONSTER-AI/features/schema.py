"""Feature schema: versioning, groups and the compatibility hash stored with every model artifact."""

from __future__ import annotations

import hashlib

FEATURE_VERSION = "fv1"

GROUPS: dict[str, str] = {
    "t_": "technical",
    "s_": "sector",
    "m_": "regime",
    "g_": "global",
    "f_": "fno",
    "c_": "calendar",
    "p_": "premarket",
    "x_": "similar",
    "u_": "static",
}

# Features that exist only once the pre-open equilibrium price is known (stage "final").
FINAL_ONLY_PREFIXES = ("p_",)

# Stable integer codes for NSE industries (never derive codes from data order: a model must see the same code forever).
SECTOR_CODES: dict[str, int] = {
    name: i for i, name in enumerate(
        (
            "Financial Services", "Capital Goods", "Healthcare", "Automobile and Auto Components",
            "Fast Moving Consumer Goods", "Chemicals", "Information Technology", "Consumer Services", "Metals & Mining",
            "Power", "Oil Gas & Consumable Fuels", "Consumer Durables", "Services", "Realty", "Construction",
            "Construction Materials", "Telecommunication", "Textiles", "Media Entertainment & Publication", "Diversified",
        )
    )
}


def sector_code(name: str | None) -> float:
    return float(SECTOR_CODES.get(name or "", 99))


def group_of(name: str) -> str:
    return GROUPS.get(name[:2], "other")


def columns_for_stage(columns: list[str], stage: str) -> list[str]:
    if stage == "final":
        return list(columns)
    return [c for c in columns if not c.startswith(FINAL_ONLY_PREFIXES)]


def schema_hash(columns: list[str]) -> str:
    """Order-sensitive hash of (name, group); a changed feature set yields a different hash."""
    payload = "|".join(f"{c}:{group_of(c)}" for c in columns) + f"#{FEATURE_VERSION}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
