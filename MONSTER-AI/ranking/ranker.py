"""Meta-ranking, gating, tiers and abstention (blueprint section 14).

The ranking criterion is ONE calibrated probability - P(range > primary threshold) - with deterministic tie-breaks.
There is no hand-weighted "score" presented as a probability. Conviction tiers combine that probability with
uncertainty, contradictions, data quality and the validation status of the model; the system is allowed (and expected)
to say "no high-conviction setup" instead of forcing five names.
"""

from __future__ import annotations

import logging
import math
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from ranking.costs import estimate_round_trip_cost_pct
from ranking.signals import confirmations, contradictions, data_quality_notes, priced_in
from ranking.types import TIER_ORDER, Candidate, Catalyst

log = logging.getLogger(__name__)

TIER_BY_RANK = {v: k for k, v in TIER_ORDER.items()}


@dataclass
class RankInput:
    stage: str
    trade_date: str
    model_version: str
    model_status: str  # validated | shadow | baseline
    primary_threshold: float
    features: pd.DataFrame  # index = symbol
    preds: pd.DataFrame  # index = symbol (ModelBundle.predict output)
    instruments: Mapping[str, Mapping[str, Any]]
    last_close: Mapping[str, float]
    catalysts: Mapping[str, list[Catalyst]] = field(default_factory=dict)
    chains: Mapping[str, list[str]] = field(default_factory=dict)
    reacted: Mapping[str, tuple[int, int]] = field(default_factory=dict)  # symbol -> (events with reaction, events)
    quality_status: str = "ok"  # ok | degraded
    quality_notes: list[str] = field(default_factory=list)
    explainer: Any = None  # callable(symbol) -> list[str]
    # symbol -> (STRONG, MODERATE) uncertainty ceilings learned by the model that scored it; absent = configured constants
    uncertainty_ceilings: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    # symbol -> the specialist-disagreement limit of the model that scored it (models.inference.disagreement_limit);
    # absent = the fixed default of ranking.signals
    spec_std_limits: Mapping[str, float] = field(default_factory=dict)


@dataclass
class RankResult:
    candidates: list[Candidate]
    no_setup: bool
    no_setup_reason: str | None
    n_universe: int
    n_eligible: int
    excluded: dict[str, int]
    baselines: dict[str, list[str]]
    table: pd.DataFrame  # every eligible symbol with probabilities and tier (frozen into the snapshot)


def eligibility(features: pd.DataFrame, last_close: Mapping[str, float], settings: Settings) -> tuple[pd.Series, pd.Series]:
    """Boolean eligibility per symbol and the first failing reason (empty string when eligible)."""
    reason = pd.Series("", index=features.index, dtype=object)

    def mark(mask: pd.Series, text: str) -> None:
        todo = mask & (reason == "")
        reason[todo] = text

    close = pd.Series({s: last_close.get(s, np.nan) for s in features.index})
    mark(features["t_atr14_pct"].isna() | features["t_range_mean20"].isna(), "insufficient price history")
    mark(features.get("u_hist_days", pd.Series(np.nan, index=features.index)) < 120, "short history (recent listing)")
    mark(close.isna() | (close < settings.min_price), f"price below Rs {settings.min_price:.0f}")
    tv = features.get("t_tv_med20_cr", pd.Series(np.nan, index=features.index))
    mark(tv.isna() | (tv < settings.min_median_turnover_cr), f"median turnover below Rs {settings.min_median_turnover_cr:.0f} cr")
    return reason == "", reason


def _round(value: Any, digits: int) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, digits) if math.isfinite(out) else None


def p_and_range(pr: pd.Series) -> tuple[dict[str, float], dict[str, float]]:
    """Calibrated probabilities and forecast range quantiles of one prediction row, rounded and without gaps."""
    p: dict[str, float | None] = {c[2:]: _round(pr[c], 4) for c in pr.index if c.startswith("p_hit_")}
    p.update(unusual=_round(pr.get("p_unusual"), 4), up=_round(pr.get("p_up"), 4), down=_round(pr.get("p_down"), 4))
    rng = {"q10": _round(pr.get("rng_q10"), 2), "q50": _round(pr.get("rng_q50"), 2), "q90": _round(pr.get("rng_q90"), 2),
           "exc_q50": _round(pr.get("exc_q50"), 2), "exc_q90": _round(pr.get("exc_q90"), 2)}
    return {k: v for k, v in p.items() if v is not None}, {k: v for k, v in rng.items() if v is not None}


def _base_level(p: float, uncertainty: float, cfg: Settings, ceilings: tuple[float, float] | None) -> int:
    """Tier level (1 WATCH .. 3 STRONG) from probability and uncertainty alone.

    ``ceilings`` = (STRONG, MODERATE) uncertainty limits learned from the scoring model's own out-of-sample top picks;
    without them (older models, the baseline) the configured constants apply."""
    strong_cap, moderate_cap = ceilings or (cfg.max_uncertainty_strong, cfg.max_uncertainty_moderate)
    if p >= cfg.tier_strong_p and uncertainty <= strong_cap:
        return 3
    if p >= cfg.tier_moderate_p and uncertainty <= moderate_cap:
        return 2
    return 1


def decide_tier(
    p: float, uncertainty: float, n_contradictions: int, dq_warning: bool, model_status: str, cfg: Settings,
    ceilings: tuple[float, float] | None = None,
) -> str | None:
    """Conviction tier from calibrated probability, then downgrade rules. ``None`` = below the listing threshold."""
    if not p or p != p or p < cfg.tier_watch_p:
        return None
    level = _base_level(p, uncertainty, cfg, ceilings)
    if n_contradictions >= 2:
        level -= 1
    if dq_warning:
        level = min(level, 2)
    if model_status != "validated":
        level = 1  # shadow / baseline: never presented as a strong signal
    return TIER_BY_RANK[max(level, 1)]


BLOCKER_TEXT = {
    "probability": "probability below the {moderate_p:.0%} MODERATE threshold",
    "uncertainty": "model uncertainty above the usual range for names of similar probability",
    "contradictions": "conflicting evidence",
}


def why_below_moderate(
    p: float, uncertainty: float, n_contradictions: int, cfg: Settings, ceilings: tuple[float, float] | None,
) -> str:
    """The first reason a listed name of a validated model stayed at WATCH (a key of ``BLOCKER_TEXT``)."""
    if p < cfg.tier_moderate_p:
        return "probability"
    if _base_level(p, uncertainty, cfg, ceilings) < 2:
        return "uncertainty"
    return "contradictions"


def describe_blockers(blockers: Mapping[str, int], total: int, cfg: Settings) -> str:
    """'cause (n of total)' for each cause that held names back, most common first."""
    ordered = sorted(blockers.items(), key=lambda kv: (-kv[1], list(BLOCKER_TEXT).index(kv[0])))
    return "; ".join(f"{BLOCKER_TEXT[key].format(moderate_p=cfg.tier_moderate_p)} ({n} of {total})" for key, n in ordered)


def _candidate_pool(feats: pd.DataFrame, preds: pd.DataFrame, last_close: Mapping[str, float], settings: Settings,
                    ) -> tuple[pd.DataFrame, pd.Series, dict[str, int], pd.DataFrame, pd.Series]:
    """The names a report may list, best first: eligible (liquid, priced, enough history), cheap enough to trade relative to their
    expected range, with a probability. Returns (table, eligibility, exclusion counts, ordered pool, turnover)."""
    elig, reasons = eligibility(feats, last_close, settings)
    excluded: dict[str, int] = reasons[~elig].value_counts().to_dict()
    table = pd.DataFrame(index=feats.index)
    table["p_primary"] = preds["p_primary"]
    table["rng_q50"] = preds["rng_q50"]
    table["uncertainty"] = preds["uncertainty"]
    table["eligible"] = elig
    table["tier"] = None

    turnover = feats.get("t_tv_med20_cr", pd.Series(np.nan, index=feats.index))
    table["cost_pct"] = [estimate_round_trip_cost_pct(None if pd.isna(t) else float(t)) for t in turnover]
    table["cost_ratio"] = table["cost_pct"] / table["rng_q50"].clip(lower=1e-6)
    viable = table["cost_ratio"] <= settings.max_cost_to_range_ratio
    n_costly = int((elig & ~viable).sum())
    if n_costly:
        excluded["round-trip cost too high versus expected range"] = n_costly
    in_pool = elig & viable & table["p_primary"].notna()
    table["in_pool"] = in_pool
    pool = table[in_pool].copy()
    pool["raw"] = preds.get("p_raw_primary", preds["p_primary"]).reindex(pool.index)
    pool = pool.sort_values(["p_primary", "raw", "rng_q50", "uncertainty"], ascending=[False, False, False, True], kind="mergesort")
    return table, elig, excluded, pool, turnover


def enrichment_pool(features: pd.DataFrame, preds: pd.DataFrame, last_close: Mapping[str, float], settings: Settings,
                    n: int) -> list[str]:
    """The ``n`` best names the ranking can actually list - the ones worth the (slow) news / event-graph enrichment. Taken AFTER
    the eligibility and cost filters: picking the best-scored names first would spend the enrichment on illiquid stocks the
    ranking then drops, and leave the final top names without catalysts."""
    pool = _candidate_pool(features, preds.reindex(features.index), last_close, settings)[3]
    return list(pool.index[:n])


def rank(inp: RankInput, settings: Settings, *, top_n: int | None = None) -> RankResult:
    top_n = top_n or settings.top_n
    feats, preds = inp.features, inp.preds.reindex(inp.features.index)
    table, elig, excluded, pool, turnover = _candidate_pool(feats, preds, inp.last_close, settings)

    candidates: list[Candidate] = []
    blockers: Counter[str] = Counter()
    for symbol in pool.index:
        if len(candidates) >= top_n:
            break
        row = feats.loc[symbol].to_dict()
        pr = preds.loc[symbol]
        events = inp.catalysts.get(symbol, [])
        spec_std = _round(preds.loc[symbol].get("sp_std_unusual"), 4) if "sp_std_unusual" in preds else None
        contra = contradictions(row, events, _round(pr["uncertainty"], 3), spec_std,
                                spec_std_limit=inp.spec_std_limits.get(symbol))
        notes = data_quality_notes(row, bool(inp.instruments.get(symbol, {}).get("is_fno")), inp.stage)
        dq_warning = bool(notes) or inp.quality_status != "ok"
        p_symbol, unc_symbol, ceilings = float(pr["p_primary"]), float(pr["uncertainty"]), inp.uncertainty_ceilings.get(symbol)
        tier = decide_tier(p_symbol, unc_symbol, contra.count, dq_warning, inp.model_status, settings, ceilings)
        table.at[symbol, "tier"] = tier
        if tier is None:
            continue
        if tier == "WATCH" and inp.model_status == "validated":
            blockers[why_below_moderate(p_symbol, unc_symbol, contra.count, settings, ceilings)] += 1
        reacted, total = inp.reacted.get(symbol, (0, len(events)))
        priced_text, _score = priced_in(row, reacted, total)
        info = inp.instruments.get(symbol, {})
        turnover_cr = _round(turnover.get(symbol), 1)
        cost = _round(table.at[symbol, "cost_pct"], 2)
        liquidity_note = f"median traded value Rs {turnover_cr:.0f} cr/day; est. round-trip cost {cost:.2f}%" if turnover_cr else "turnover n/a"
        if notes:
            liquidity_note += "; data: " + ", ".join(notes)
        drivers: list[str] = []
        if inp.explainer is not None:
            try:
                drivers = list(inp.explainer(symbol))
            except Exception as exc:  # noqa: BLE001 - explanations are diagnostic; never block a report
                log.warning("explanation unavailable for %s (%s: %s); the report lists it without drivers", symbol,
                            type(exc).__name__, exc)
                drivers = []
        p_dict, range_dict = p_and_range(pr)
        candidates.append(
            Candidate(
                symbol=symbol, rank=len(candidates) + 1, tier=tier, p=p_dict,
                primary_threshold=inp.primary_threshold, name=info.get("name"), sector=info.get("sector"),
                range=range_dict,
                catalysts=events[:3], chain=list(inp.chains.get(symbol, []))[:3], confirmations=confirmations(row),
                contradictions=contra.items, priced_in=priced_text, liquidity_ok=True, liquidity_note=liquidity_note,
                turnover_cr=turnover_cr, est_cost_pct=cost, data_quality="warning" if dq_warning else "ok",
                uncertainty=_round(pr["uncertainty"], 3), drivers=drivers[:4], is_fno=bool(info.get("is_fno")),
                gap_pct=_round(row.get("p_gap_pct"), 2),
            )
        )

    best_p = float(pool["p_primary"].iloc[0]) if len(pool) else float("nan")
    strong_enough = [c for c in candidates if TIER_ORDER[c.tier] >= TIER_ORDER["MODERATE"]]
    no_setup = not strong_enough
    reason = None
    if no_setup:
        if inp.model_status != "validated":
            reason = (f"model is {inp.model_status.upper()} (no demonstrated edge over simple baselines): names are listed as "
                      f"watchlist only")
        elif not candidates:
            reason = f"no eligible name reached the minimum probability {settings.tier_watch_p:.0%} (best {best_p:.0%})"
        else:
            reason = (f"no candidate reached MODERATE conviction (best P = {best_p:.0%}): "
                      f"{describe_blockers(blockers, len(candidates), settings)}")
    return RankResult(
        candidates=candidates, no_setup=no_setup, no_setup_reason=reason, n_universe=len(feats), n_eligible=int(elig.sum()),
        excluded=excluded, baselines=baseline_picks(feats, elig, inp.stage, inp.trade_date, top_n), table=table,
    )


def baseline_picks(features: pd.DataFrame, eligible: pd.Series, stage: str, trade_date: str, top_n: int) -> dict[str, list[str]]:
    """Simple benchmark selections stored next to the model's picks (shadow comparison on identical days)."""
    pool = features[eligible]
    out: dict[str, list[str]] = {}
    scorers = {"atr": "t_atr14_pct", "range20": "t_range_mean20", "volume": "t_rvol1", "recent_range": "t_range_prev"}
    if stage == "final" and "p_gap_abs" in features:  # the opening gap is only known once the pre-open price is
        scorers["gap"] = "p_gap_abs"
    for name, col in scorers.items():
        if col in pool and pool[col].notna().any():
            out[name] = list(pool[col].sort_values(ascending=False, kind="mergesort").dropna().index[:top_n])
    seed = zlib.crc32(f"{trade_date}:{stage}".encode())
    rng = np.random.default_rng(seed)
    symbols = sorted(pool.index)
    if symbols:
        out["random"] = list(rng.choice(symbols, size=min(top_n, len(symbols)), replace=False))
    return out
