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
    # Provenance, all optional (None = not assessed) so older snapshots and callers stay valid.
    event_id: int | None = None  # the structured event this catalyst was drawn from
    novelty_class: str | None = None  # how new the information is: new | follow_up | resurfaced | rumour
    rumour: bool | None = None  # True = unconfirmed / hearsay (clears once a non-rumour report joins; a denial does not clear it)
    source_class: str | None = None  # best source so far: exchange_filing | official | company | media | social | unknown
    n_independent: int | None = None  # independent sources reporting it
    n_conflicts: int | None = None  # reports that contradict it

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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


# ------------------------------------------------------------------------------------------ trade ideas
HORIZONS = ("BTST", "SWING", "WATCH")  # buy before the close and sell next session | hold a few sessions | levels for tomorrow's intraday
SIDES = ("LONG", "SHORT", "BOTH")
GRADES = ("A", "B", "C")


@dataclass
class TradeIdea:
    """One closing-hour trade idea: a setup, its mechanical ATR levels and the figures it was built on. ``score`` is a ranking aid
    of the setup (0-100), never a probability."""

    symbol: str
    rank: int
    horizon: str  # BTST | SWING | WATCH
    side: str  # LONG | SHORT | BOTH (a WATCH idea has a trigger on each side)
    setup: str  # the pattern in words, e.g. "strong close on volume"
    score: float
    grade: str  # A | B | C: bands of the score
    ltp: float
    name: str | None = None
    sector: str | None = None
    is_fno: bool = False
    change_pct: float | None = None  # today's move so far, %
    entry: float | None = None  # reference entry: the last price at the scan
    max_entry: float | None = None  # do not chase beyond this (above for a long, below for a short)
    stop: float | None = None
    target1: float | None = None
    target2: float | None = None
    trigger_long: float | None = None  # WATCH: buy above this
    trigger_short: float | None = None  # WATCH: sell below this
    risk_pct: float | None = None  # entry to stop, % of the entry
    reward1_pct: float | None = None
    reward2_pct: float | None = None
    rr1: float | None = None
    rr2: float | None = None
    hold: str = ""
    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    catalysts: list[Catalyst] = field(default_factory=list)
    turnover_cr: float | None = None
    est_cost_pct: float | None = None  # all-in round trip, delivery-style, % of the position
    metrics: dict[str, float] = field(default_factory=dict)  # the figures behind the idea, frozen with it

    @property
    def key(self) -> str:
        """Identity inside one run: the same stock can appear under two horizons or sides."""
        return f"{self.horizon}:{self.side}:{self.symbol}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TradeIdea:
        payload = _filter_fields(cls, data)
        payload["catalysts"] = [Catalyst.from_dict(c) if isinstance(c, dict) else c for c in payload.get("catalysts", [])]
        return cls(**payload)


@dataclass
class IdeasReport:
    """The frozen closing-hour report: what was seen at the scan, the ideas, and the live record that stands behind them."""

    trade_date: str
    snapshot_ist: str  # display string, e.g. "2026-11-02 14:31:00 IST"
    entry_by: str  # IST wall clock after which the report says the entry window is over, e.g. "15:15"
    source: str = ""  # where the live prices came from, and how many names they covered
    ideas: list[TradeIdea] = field(default_factory=list)
    market: dict[str, Any] = field(default_factory=dict)  # nifty_chg, vix, vix_chg, advancers, decliners, regime
    record: dict[str, dict[str, Any]] = field(default_factory=dict)  # horizon -> live-record summary (see backtesting.ideas)
    notes: list[str] = field(default_factory=list)
    mode: str = "shadow"
    universe_size: int = 0
    n_eligible: int = 0
    late: bool = False  # built after the report deadline (a very late retry, a manual run): kept out of the live record; a frozen report that is only SENT late is marked in the copy that goes out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdeasReport:
        payload = _filter_fields(cls, data)
        payload["ideas"] = [TradeIdea.from_dict(i) if isinstance(i, dict) else i for i in payload.get("ideas", [])]
        return cls(**payload)


# ------------------------------------------------------------------------------------------ the 09:30 opening read
OPENING_KINDS = ("PICK", "MOVER")  # a name of the 09:12 report | a new name whose opening range stands out
STRENGTHS = ("QUIET", "NORMAL", "HOT", "EXPLOSIVE")  # how big the first fifteen minutes were against the stock's usual day


@dataclass
class OpeningIdea:
    """One stock at the 09:30 opening read: what its first fifteen minutes showed, what the opening-range model makes of it, and the
    reference levels around the opening range. Nothing here is a call to buy or sell: the model estimates how WIDE the day will be
    (validated), not which way it will go (no edge found)."""

    symbol: str
    rank: int
    kind: str  # PICK | MOVER
    ltp: float
    open: float
    prev_close: float
    or_high: float  # the day's high at the scan = the opening range's high
    or_low: float
    gap_pct: float  # the open against the previous close, %
    or_pct: float  # the opening range, % of the open, normalised to the first fifteen minutes
    move_pct: float  # the last price against the open, %
    x: float  # the opening range as a multiple of the stock's mean daily range of the last 20 sessions
    strength: str  # QUIET | NORMAL | HOT | EXPLOSIVE (bands of ``x``)
    p3: float  # estimated chance that the whole day's range exceeds 3 % of the open
    p5: float
    day_range: list[float] = field(default_factory=list)  # q10, q50, q90 of the day's range, % of the open
    ahead: list[float] = field(default_factory=list)  # q10, q50, q90 of the range still to come after the scan, % of the last price
    name: str | None = None
    sector: str | None = None
    is_fno: bool = False
    atr_pct: float | None = None
    turnover_cr: float | None = None
    est_cost_pct: float | None = None  # all-in intraday round trip, % of the position
    levels: dict[str, float] = field(default_factory=dict)  # trigger_long / long_stop / long_target1 / long_target2 and the short side
    pre: dict[str, Any] = field(default_factory=dict)  # a pick's 09:12 view: rank, tier, p (its primary-threshold probability), threshold
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.symbol}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpeningIdea:
        return cls(**_filter_fields(cls, data))


@dataclass
class OpeningReport:
    """The frozen 09:30 report: the first fifteen minutes of the session read against the 09:12 picks, the opening movers and the
    live record that stands behind the model."""

    trade_date: str
    snapshot_ist: str  # display string, e.g. "2026-11-02 09:30:42 IST"
    elapsed_min: float = 15.0  # minutes of trading the live range covers (15 = exactly the opening range)
    source: str = ""
    picks: list[OpeningIdea] = field(default_factory=list)  # the names of the 09:12 report, read again
    movers: list[OpeningIdea] = field(default_factory=list)  # new names whose opening range stands out
    pre_stage: str = ""  # which morning snapshot the picks were taken from: final | pre (empty = none was stored)
    market: dict[str, Any] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)  # model id and the evidence it stands on
    notes: list[str] = field(default_factory=list)
    mode: str = "shadow"
    universe_size: int = 0
    n_eligible: int = 0
    late: bool = False  # built after the report deadline: kept out of the live record

    @property
    def ideas(self) -> list[OpeningIdea]:
        return [*self.picks, *self.movers]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpeningReport:
        payload = _filter_fields(cls, data)
        for name in ("picks", "movers"):
            payload[name] = [OpeningIdea.from_dict(i) if isinstance(i, dict) else i for i in payload.get(name, [])]
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
    # Wilson interval of this ONE day's picks. The picks of a day share one market, so it is optimistic: the rolling
    # windows in ``live_record`` use a day-block bootstrap and are the intervals to rely on.
    model_hit_ci: tuple[float, float] | None = None
    baseline_hit_rates: dict[str, float] = field(default_factory=dict)  # e.g. {"atr": 0.4, "volume": 0.2, "random": 0.2}
    mean_range_pct: float | None = None
    baseline_mean_range_pct: dict[str, float] = field(default_factory=dict)
    rolling: dict[str, float | None] = field(default_factory=dict)  # e.g. hit_rate_30d, brier_30d, n_days
    calibration_note: str = ""
    data_quality_notes: list[str] = field(default_factory=list)
    provisional: bool = True  # True until the official bhavcopy confirms the OHLC
    missing_symbols: list[str] = field(default_factory=list)
    # Rolling live record of the frozen picks: window label ("20d", "60d", "all") -> metric rows, each
    # {name, slice, value, n, ci_low, ci_high, extra}; slice "all" = every pick, other slices are persisted only.
    live_record: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
