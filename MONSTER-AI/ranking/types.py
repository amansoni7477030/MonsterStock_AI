"""Shared data contracts between ranking, notifications, evaluation and the dashboard.

Everything here is plain data (JSON round-trippable) so predictions can be frozen in SQLite and re-rendered
later without importing model code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

TIERS = ("STRONG", "MODERATE", "WATCH")
TIER_ORDER = {"STRONG": 3, "MODERATE": 2, "WATCH": 1}


def _filter_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass
class Catalyst:
    headline: str
    event_type: str = "other"
    source: str = ""
    url: str | None = None
    published_at: str | None = None  # UTC ISO
    novelty: float | None = None  # 0..1
    confidence: float | None = None  # 0..1
    severity: float | None = None  # 0..1
    polarity: float | None = None  # -1..1 for this stock (NOT a trading signal)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Catalyst:
        return cls(**_filter_fields(cls, data))


@dataclass
class Candidate:
    symbol: str
    rank: int
    tier: str  # STRONG | MODERATE | WATCH
    p: dict[str, float]  # hit_1, hit_2, hit_3, hit_5, unusual, up, down (calibrated probabilities, 0..1)
    primary_threshold: float
    range: dict[str, float]  # q10, q50, q90, exc_q50, exc_q90 (percent of open)
    name: str | None = None
    sector: str | None = None
    catalysts: list[Catalyst] = field(default_factory=list)
    chain: list[str] = field(default_factory=list)  # "event -> sector -> stock" strings
    confirmations: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    priced_in: str = "not assessed"
    liquidity_ok: bool = True
    liquidity_note: str = ""
    turnover_cr: float | None = None
    est_cost_pct: float | None = None
    data_quality: str = "ok"  # ok | warning
    uncertainty: float | None = None  # 0..1 (higher = less reliable)
    drivers: list[str] = field(default_factory=list)  # diagnostic only, not causal
    is_fno: bool = False
    gap_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        payload = _filter_fields(cls, data)
        payload["catalysts"] = [Catalyst.from_dict(c) if isinstance(c, dict) else c for c in payload.get("catalysts", [])]
        return cls(**payload)


@dataclass
class ReportContext:
    trade_date: str
    stage: str  # pre | final
    snapshot_ist: str  # display string, e.g. "2026-11-02 09:12:00 IST"
    model_version: str
    model_status: str  # shadow | validated | baseline
    mode: str = "shadow"
    candidates: list[Candidate] = field(default_factory=list)
    no_setup: bool = False
    no_setup_reason: str | None = None
    regime: dict[str, str] = field(default_factory=dict)
    freshness: dict[str, str] = field(default_factory=dict)  # feed -> "ok (3 min old)" / "STALE (2 d)"
    missing_feeds: list[str] = field(default_factory=list)
    uncertainty_note: str = ""
    notes: list[str] = field(default_factory=list)
    universe_size: int = 0
    n_eligible: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportContext:
        payload = _filter_fields(cls, data)
        payload["candidates"] = [Candidate.from_dict(c) if isinstance(c, dict) else c for c in payload.get("candidates", [])]
        return cls(**payload)


@dataclass
class EveningRow:
    symbol: str
    rank: int
    tier: str
    p_primary: float | None
    range_pct: float | None
    exc_pct: float | None
    dir_ret_pct: float | None
    hit_primary: bool | None
    source: str = ""
    validated: bool = False


@dataclass
class EveningSummary:
    trade_date: str
    stage: str
    primary_threshold: float
    model_version: str
    rows: list[EveningRow] = field(default_factory=list)
    n: int = 0
    model_hit_rate: float | None = None
    model_hit_ci: tuple[float, float] | None = None
    baseline_hit_rates: dict[str, float] = field(default_factory=dict)  # e.g. {"atr": 0.4, "volume": 0.2, "random": 0.2}
    mean_range_pct: float | None = None
    baseline_mean_range_pct: dict[str, float] = field(default_factory=dict)
    rolling: dict[str, float | None] = field(default_factory=dict)  # e.g. hit_rate_30d, brier_30d, n_days
    calibration_note: str = ""
    data_quality_notes: list[str] = field(default_factory=list)
    provisional: bool = True  # True until the official bhavcopy confirms the OHLC
    missing_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
