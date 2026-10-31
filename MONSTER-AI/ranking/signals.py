"""Evidence helpers for the ranker: confirmations, contradictions, priced-in (crowding) and data-quality notes.

All statements are derived from point-in-time features; wording avoids causal claims (features, not facts)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ranking.types import Catalyst

OI_PRICE = {1: "long build-up", 2: "short build-up", 3: "short covering", 4: "long unwinding"}


def _v(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def confirmations(row: Mapping[str, Any]) -> list[str]:
    """Features indicating elevated activity (supporting evidence for a large move, never for a direction)."""
    out: list[str] = []
    atr5, atr14 = _v(row, "t_atr5_pct"), _v(row, "t_atr14_pct")
    ratio = _v(row, "t_atr_ratio")
    if ratio is not None and ratio >= 1.15 and atr5 is not None:
        out.append(f"technical: volatility expanding (5d ATR {atr5:.1f}% = {ratio:.2f}x the 14d ATR)")
    exp = _v(row, "t_range_exp")
    if exp is not None and exp >= 1.6:
        out.append(f"technical: yesterday's range was {exp:.1f}x its 20-day average")
    nr7 = _v(row, "t_nr7")
    if nr7 == 1.0:
        out.append("technical: volatility compression (narrowest range in 7 sessions) - often precedes expansion")
    rvol = _v(row, "t_rvol1")
    if rvol is not None and rvol >= 1.5:
        out.append(f"volume: {rvol:.1f}x the 20-day average yesterday")
    if _v(row, "t_breakout_up") == 1.0 or _v(row, "t_breakdown") == 1.0:
        out.append("technical: closed beyond the prior 20-day range")
    sector = _v(row, "s_ret1")
    breadth = _v(row, "s_breadth1")
    if sector is not None and abs(sector) >= 1.0:
        text = f"sector: peers moved {sector:+.1f}% yesterday"
        if breadth is not None:
            text += f" (breadth {breadth:.0%})"
        out.append(text)
    iv_rank, exp_move = _v(row, "f_iv_rank120"), _v(row, "f_exp_move_1d")
    if iv_rank is not None and iv_rank >= 0.8:
        out.append(f"F&O: implied volatility at the {iv_rank:.0%} percentile of its 120-day range")
    elif exp_move is not None and atr14 and exp_move >= 0.9 * atr14:
        out.append(f"F&O: options imply a ~{exp_move:.1f}% one-day move (ATR {atr14:.1f}%)")
    code = _v(row, "f_oi_price_code")
    oi_chg = _v(row, "f_fut_oi_chg_pct")
    if code is not None and oi_chg is not None and abs(oi_chg) >= 4.0:
        out.append(f"F&O: {OI_PRICE.get(int(code), 'positioning change')} (futures OI {oi_chg:+.1f}%) - heuristic, not proof of intent")
    if _v(row, "c_expiry_today") == 1.0:
        out.append("calendar: F&O expiry day")
    if _v(row, "c_results_today") == 1.0 or _v(row, "c_results_prev") == 1.0:
        out.append("calendar: results / board-meeting catalyst today or yesterday")
    gap_atr = _v(row, "p_gap_atr")
    gap = _v(row, "p_gap_pct")
    if gap is not None and gap_atr is not None and abs(gap_atr) >= 0.8:
        out.append(f"pre-open: indicative gap {gap:+.1f}% ({abs(gap_atr):.1f}x ATR)")
    return out[:6]


@dataclass
class Contradictions:
    items: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)


def contradictions(row: Mapping[str, Any], catalysts: list[Catalyst], uncertainty: float | None, spec_std: float | None) -> Contradictions:
    """Conflicts that should lower confidence instead of being averaged away."""
    found: list[str] = []
    polarity = [c.polarity for c in catalysts if c.polarity not in (None, 0.0)]
    net = float(np.mean(polarity)) if polarity else 0.0
    sector, gap, ret1 = _v(row, "s_ret1"), _v(row, "p_gap_pct"), _v(row, "t_ret1")
    us, oil = _v(row, "g_us_ret"), _v(row, "g_oil_ret")
    if net > 0.2 and sector is not None and sector < -1.0:
        found.append(f"positive company news but the sector fell {sector:+.1f}% yesterday")
    if net < -0.2 and sector is not None and sector > 1.0:
        found.append(f"negative company news but the sector rose {sector:+.1f}% yesterday")
    if net > 0.2 and gap is not None and gap < -0.8:
        found.append(f"positive news but the stock is indicated {gap:+.1f}% lower at the open")
    if net < -0.2 and gap is not None and gap > 0.8:
        found.append(f"negative news but the stock is indicated {gap:+.1f}% higher at the open")
    code = _v(row, "f_oi_price_code")
    oi_chg = _v(row, "f_fut_oi_chg_pct")
    if code in (2.0, 4.0) and ret1 is not None and ret1 > 1.5 and oi_chg is not None and oi_chg > 3.0:
        found.append("price rose yesterday but futures OI build-up looks bearish (short build-up)")
    if code == 1.0 and ret1 is not None and ret1 < -1.5 and oi_chg is not None and oi_chg > 3.0:
        found.append("price fell yesterday while futures OI build-up looks bullish (long build-up)")
    if us is not None and us < -1.0 and net > 0.2:
        found.append(f"positive stock-specific news against a weak global tone (US {us:+.1f}%)")
    if oil is not None and abs(oil) > 3.0 and _v(row, "g_exp_oil") is not None and abs(_v(row, "g_exp_oil") or 0.0) < 0.1:
        found.append("large oil move but this stock shows little historical oil sensitivity")
    if spec_std is not None and spec_std > 0.12:
        found.append("specialist models disagree materially (low signal agreement)")
    if any(c.confidence is not None and c.confidence < 0.4 for c in catalysts):
        found.append("catalyst evidence is low-confidence (possible rumour / single source)")
    return Contradictions(found[:4])


def priced_in(row: Mapping[str, Any], reacted_events: int, n_events: int) -> tuple[str, float]:
    """Has the expected reaction already happened before the snapshot? Returns (text, score 0..1; 1 = fully priced in)."""
    gap_atr = _v(row, "p_gap_atr")
    gap = _v(row, "p_gap_pct")
    exp = _v(row, "t_range_exp")
    score = 0.0
    parts: list[str] = []
    if gap is not None and gap_atr is not None:
        if abs(gap_atr) >= 2.0:
            score += 0.5
            parts.append(f"large indicated gap {gap:+.1f}% ({abs(gap_atr):.1f}x ATR): part of the move may already be in the opening price")
        elif abs(gap_atr) >= 1.0:
            score += 0.25
            parts.append(f"moderate gap {gap:+.1f}% ({abs(gap_atr):.1f}x ATR)")
        else:
            parts.append(f"small gap {gap:+.1f}%")
    elif gap is None:
        parts.append("pre-open price not yet available")
    if exp is not None and exp >= 2.0:
        score += 0.2
        parts.append(f"yesterday's range already {exp:.1f}x normal")
    if n_events and reacted_events:
        share = reacted_events / n_events
        score += 0.3 * share
        parts.append(f"{reacted_events}/{n_events} catalyst(s) already show a market reaction")
    elif n_events:
        parts.append("no reaction yet to the catalyst(s)")
    return ("; ".join(parts) or "not assessed"), float(min(score, 1.0))


def data_quality_notes(row: Mapping[str, Any], is_fno: bool, stage: str) -> list[str]:
    notes: list[str] = []
    if _v(row, "t_atr14_pct") is None or _v(row, "t_range_mean20") is None:
        notes.append("insufficient price history")
    if is_fno and _v(row, "f_iv_atm") is None:
        notes.append("F&O data missing for this stock")
    if stage == "final" and _v(row, "p_gap_pct") is None:
        notes.append("no pre-open price: preliminary model used")
    if _v(row, "g_cover") is not None and (_v(row, "g_cover") or 0.0) < 0.7:
        notes.append("global overnight coverage below 70%")
    return notes
