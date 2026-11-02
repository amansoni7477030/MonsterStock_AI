"""Closing-hour trade ideas: setups, scores, mechanical levels and the market regime (pure: no I/O, no clock).

Three kinds of idea come out of the table of ONE session (``features.swing.last_row`` plus a few context columns the scan adds):

* ``BTST`` - buy before the close and sell in the next session: names closing strong on volume (long), or weak on volume (short,
  F&O names only: the cash market cannot be held short overnight);
* ``SWING`` - hold for a few sessions: a trend with a pattern (a pullback to the 20-day average, a base breakout, a tight range under
  the 20-day high) or its mirror image;
* ``WATCH`` - levels for tomorrow's intraday trading: names whose recent range is coiled at a key level, with a buy trigger above
  today's high and a sell trigger below today's low.

Honest by construction. The score of a setup is a weighted mean of fixed, documented components (percentile ranks inside the day's
universe, or clipped linear scales); it is a ranking aid, NOT a probability, nothing is fitted, and every figure that moved it is in the
idea's ``metrics`` and reasons. Stops and targets are mechanical multiples of the previous ATR, not an optimised strategy. Whether the
setups work is for the live record (``backtesting.ideas``) to say, and until it says so the report labels them heuristic."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ranking.costs import estimate_delivery_round_trip_cost_pct
from ranking.types import Catalyst, TradeIdea

GRADE_A, GRADE_B = 78.0, 68.0  # score bands; below ``IdeaConfig.min_score`` an idea is not listed at all
MIN_HISTORY_DAYS = 120
MIN_KNOWN_WEIGHT = 0.6  # a score needs components worth at least this share of its weights to be known
MIN_WATCH_ATR_PCT = 1.2  # a stock that moves less than this a day is no intraday watch
TICK = 0.05  # NSE price tick of the equities this system lists (price floor Rs 50)
CRASH_PCT = 2.0  # a Nifty move this large against a side withholds that side's overnight (BTST) ideas
NEWS_MIN_SEVERITY, NEWS_MIN_CONFIDENCE = 0.4, 0.5
OVERBOUGHT_RSI, STRETCH_ATR, LATE_CHASE_PCT = 70.0, 2.5, 5.0

BTST_WEIGHTS = {"strength": 0.25, "momentum": 0.10, "volume": 0.20, "trend": 0.15, "structure": 0.15, "relstrength": 0.08,
                "tailwind": 0.07}
SWING_WEIGHTS = {"trend": 0.25, "pattern": 0.25, "relstrength": 0.20, "volume": 0.10, "strength": 0.10, "tailwind": 0.10}
WATCH_WEIGHTS = {"expansion": 0.35, "coil": 0.25, "proximity": 0.25, "volume": 0.15}


@dataclass(frozen=True)
class IdeaConfig:
    top_n: int = 5
    min_score: float = 60.0
    max_risk_pct: float = 4.0  # entry to stop, % of the entry: a wider stop than this is not an idea worth listing
    include_shorts: bool = True
    min_price: float = 50.0
    min_turnover_cr: float = 25.0
    max_cost_ratio: float = 0.25  # delivery round-trip cost may be at most this share of the first target's reward
    min_rr: float = 1.5  # reward to the second target over the risk

    @classmethod
    def from_settings(cls, settings: Any) -> IdeaConfig:
        return cls(top_n=settings.ideas_top_n, min_score=settings.ideas_min_score, max_risk_pct=settings.ideas_max_risk_pct,
                   include_shorts=settings.ideas_include_shorts, min_price=settings.min_price,
                   min_turnover_cr=settings.min_median_turnover_cr, max_cost_ratio=settings.max_cost_to_range_ratio)


@dataclass(frozen=True)
class Plan:
    """Stop and targets as multiples of the previous 14-day ATR, and how long the idea is meant to be held. ``hold`` states the exit
    rule exactly as ``backtesting.ideas`` judges it (the first of the stop, the first target and the end of the horizon), so the record
    measures the plan the report shows."""

    stop_atr: float
    t1_atr: float
    t2_atr: float
    chase_atr: float  # how far beyond the reference entry the price may run before the idea is not worth chasing
    min_stop_atr: float
    buffer_atr: float  # beyond the structural level (the day's or the week's extreme) the stop is placed
    hold: str


BTST_PLAN = Plan(1.0, 1.0, 1.8, 0.15, 0.6, 0.1,
                 "sell in the next session: at the first target or at the stop, otherwise at the close; T2 only for a part you keep")
SWING_PLAN = Plan(1.8, 2.0, 3.6, 0.25, 1.0, 0.2,
                  "hold up to 5 sessions: sell at the first target or at the stop, otherwise at the 5th close; T2 only for a part you keep")
WATCH_PLAN = Plan(0.5, 0.6, 1.1, 0.0, 0.5, 0.05, "intraday only: act on a break of the level with volume, square off by 15:15 IST")


# ------------------------------------------------------------------------------------------ small helpers
def _col(t: pd.DataFrame, name: str) -> pd.Series:
    """The column ``name``, or an all-NaN series: a figure that could not be computed reads as unknown, never as zero."""
    return t[name].astype(float) if name in t else pd.Series(np.nan, index=t.index)


def _flag(condition: pd.Series, *needed: pd.Series) -> pd.Series:
    """1.0 / 0.0 for a condition, NaN where a figure it needs is missing."""
    known = pd.concat(list(needed), axis=1).notna().all(axis=1)
    return condition.astype(float).where(known)


def _mean_of(*parts: pd.Series) -> pd.Series:
    return pd.concat(list(parts), axis=1).mean(axis=1, skipna=True)


def _scale(x: pd.Series, low: float, high: float) -> pd.Series:
    return ((x - low) / (high - low)).clip(0.0, 1.0)


def _tick(price: float) -> float:
    return round(round(price / TICK) * TICK, 2)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _truthy(value: Any) -> bool:
    """True for a real ``True`` / 1; an unknown value (``None``, ``NaN``) is not."""
    return bool(value) and value == value


def _is_fno(row: Mapping[str, Any], info: Mapping[str, Any]) -> bool:
    """The F&O flag of the table (the one the shorting gates read), else of the instrument record."""
    return _truthy(row.get("is_fno", info.get("is_fno")))


def _r(value: Any, digits: int = 2) -> float | None:
    out = _finite(value)
    return None if out is None else round(out, digits)


# ---------------------------------------------------------------------------------------------- market
def market_regime(market: Mapping[str, Any]) -> str:
    """``bullish`` / ``neutral`` / ``bearish`` from Nifty's move today and trend and from how the stocks themselves are moving.
    Missing figures never push a verdict: with nothing known the regime is neutral."""
    nifty, median = _finite(market.get("nifty_chg")), _finite(market.get("median_chg"))
    above20, stack = market.get("nifty_above_ema20"), market.get("nifty_ema20_above_ema50")
    day = nifty if nifty is not None else median
    if (day is not None and day <= -1.0) or (median is not None and median <= -0.8) or (above20 is False and stack is False):
        return "bearish"
    if day is not None and day >= 0.3 and above20 is not False:
        return "bullish"
    return "neutral"


def overnight_block(market: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The side whose overnight (BTST) ideas are withheld today because the whole market moved sharply against it, and why."""
    day = _finite(market.get("nifty_chg"))
    if day is None:
        day = _finite(market.get("median_chg"))
    if day is not None and day <= -CRASH_PCT:
        return "LONG", f"the market is down {abs(day):.1f}% today: overnight long ideas (BTST) are withheld"
    if day is not None and day >= CRASH_PCT:
        return "SHORT", f"the market is up {day:.1f}% today: overnight short ideas are withheld"
    return None, None


# -------------------------------------------------------------------------------------------- components
def components(t: pd.DataFrame, market: Mapping[str, Any], side: str) -> tuple[pd.DataFrame, pd.Series]:
    """The 0-1 components of a setup for one side, and the name of the swing pattern each stock shows (empty = none)."""
    long = side == "LONG"
    sign = 1.0 if long else -1.0
    ltp, ema20, ema50, sma200 = (_col(t, n) for n in ("ltp", "ema20", "ema50", "sma200"))
    clv, ret1, rvol, slope = _col(t, "clv"), _col(t, "ret1"), _col(t, "rvol"), _col(t, "ema20_slope")
    rsi, dist_ema, comp = _col(t, "rsi14"), _col(t, "dist_ema20_atr"), _col(t, "range_comp")
    out = pd.DataFrame(index=t.index)
    out["strength"] = clv if long else 1.0 - clv
    out["momentum"] = _scale(sign * ret1, 0.0, 3.0)
    out["volume"] = _scale(rvol, 0.8, 2.5)
    if long:
        out["trend"] = _mean_of(_flag(ltp > ema20, ltp, ema20), _flag(ema20 > ema50, ema20, ema50), _flag(slope > 0, slope),
                                _flag(ltp > sma200, ltp, sma200))
        gap20, gap252 = _col(t, "dist_hi20"), _col(t, "dist_hi252")  # how far below the prior 20-day / 52-week extreme (+ = beyond it)
        broke = _col(t, "breakout20") == 1.0
        trending = (ltp > ema50) & (ema20 > ema50)
        pullback = trending & dist_ema.between(-0.5, 1.0) & rsi.between(40.0, 62.0) & (clv >= 0.5)
        tight = trending & (comp <= 0.85) & gap20.between(-3.0, 0.0)
        names = ("base breakout on volume", "pullback to the 20-day average in an uptrend", "tight range under the 20-day high")
    else:
        out["trend"] = _mean_of(_flag(ltp < ema20, ltp, ema20), _flag(ema20 < ema50, ema20, ema50), _flag(slope < 0, slope),
                                _flag(ltp < sma200, ltp, sma200))
        gap20, gap252 = -_col(t, "dist_lo20"), -_col(t, "dist_lo252")  # the same, mirrored: + = beyond the extreme
        broke = _col(t, "breakdown20") == 1.0
        trending = (ltp < ema50) & (ema20 < ema50)
        pullback = trending & dist_ema.between(-1.0, 0.5) & rsi.between(38.0, 60.0) & (clv <= 0.5)
        tight = trending & (comp <= 0.85) & gap20.between(-3.0, 0.0)
        names = ("breakdown on volume", "rally into the 20-day average in a downtrend", "tight range over the 20-day low")
    structure = pd.Series(0.0, index=t.index).mask(gap20 >= -3.0, 0.3).mask(gap20 >= -1.0, 0.6).mask(broke, 1.0)
    out["structure"] = (structure + 0.2 * (gap252 >= -3.0)).clip(0.0, 1.0)
    out["relstrength"] = (_col(t, "rs5") + _col(t, "rs20")).rank(pct=True).pipe(lambda r: r if long else 1.0 - r)
    sector = _scale(sign * _col(t, "sector_chg"), -1.5, 1.5).fillna(0.5)
    nifty = _finite(market.get("nifty_chg"))
    nifty = _finite(market.get("median_chg")) if nifty is None else nifty
    wide = 0.5 if nifty is None else float(np.clip(0.5 + sign * nifty / 3.0, 0.0, 1.0))
    out["tailwind"] = 0.7 * sector + 0.3 * wide
    base = broke & (rvol >= 1.2)
    out["pattern"] = pd.Series(0.0, index=t.index).mask(tight, 0.7).mask(pullback | base, 1.0)
    pattern = pd.Series("", index=t.index, dtype=object).mask(tight, names[2]).mask(pullback, names[1]).mask(base, names[0])
    return out, pattern


def watch_components(t: pd.DataFrame) -> pd.DataFrame:
    """Components of the next-session intraday watchlist: room to move, a coiled recent range, a key level close by, volume."""
    out = pd.DataFrame(index=t.index)
    out["expansion"] = _col(t, "atr_pct").rank(pct=True)
    out["coil"] = _scale(1.0 - _col(t, "range_comp"), 0.0, 0.4)
    near_high = _scale(3.0 - _col(t, "dist_hi20").abs(), 0.0, 3.0)
    near_low = _scale(3.0 - _col(t, "dist_lo20").abs(), 0.0, 3.0)
    out["proximity"] = pd.concat([near_high, near_low], axis=1).max(axis=1)
    out["volume"] = _scale(_col(t, "rvol"), 0.8, 2.5)
    return out


def weighted_score(comp: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """0-100: the weighted mean of the components that are known, ``NaN`` unless they carry ``MIN_KNOWN_WEIGHT`` of the weights."""
    names = [n for n in weights if n in comp]
    w = pd.Series({n: float(weights[n]) for n in names})
    values = comp[names]
    known = values.notna().mul(w, axis=1).sum(axis=1)
    total = (values * w).sum(axis=1, min_count=1)
    return (100.0 * total / known.where(known > 0)).where(known >= MIN_KNOWN_WEIGHT * float(sum(weights.values())))


# ------------------------------------------------------------------------------------------------ gates
def eligibility(t: pd.DataFrame, cfg: IdeaConfig) -> pd.Series:
    """Liquid, priced, long enough a history, with an ATR to size the levels, and not excluded by the scan (a stale row, a price basis
    that does not match the history)."""
    excluded = t["excluded"].fillna("").astype(str) != "" if "excluded" in t else pd.Series(False, index=t.index)
    ok = (_col(t, "ltp") >= cfg.min_price) & (_col(t, "tv_med20_cr") >= cfg.min_turnover_cr)
    ok &= (_col(t, "hist_days") >= MIN_HISTORY_DAYS) & (_col(t, "atr_abs") > 0) & ~excluded
    return ok.fillna(False)


def _fno(t: pd.DataFrame) -> pd.Series:
    if "is_fno" not in t:
        return pd.Series(False, index=t.index)
    return pd.Series([_truthy(v) for v in t["is_fno"]], index=t.index)


def _gate_btst_long(t: pd.DataFrame, _comp: pd.DataFrame) -> pd.Series:
    ok = _col(t, "ret1").between(0.3, 7.5) & (_col(t, "clv") >= 0.70) & (_col(t, "rvol") >= 1.0)
    ok &= (_col(t, "ltp") >= _col(t, "ema20")) | (_col(t, "breakout20") == 1.0)
    return ok & (_col(t, "rsi14") <= 80.0) & (_col(t, "dist_ema20_atr") <= 3.5)


def _gate_btst_short(t: pd.DataFrame, _comp: pd.DataFrame) -> pd.Series:
    ok = _col(t, "ret1").between(-7.5, -0.3) & (_col(t, "clv") <= 0.30) & (_col(t, "rvol") >= 1.0) & _fno(t)
    ok &= (_col(t, "ltp") <= _col(t, "ema20")) | (_col(t, "breakdown20") == 1.0)
    return ok & (_col(t, "rsi14") >= 20.0) & (_col(t, "dist_ema20_atr") >= -3.5)


def _gate_swing_long(t: pd.DataFrame, comp: pd.DataFrame) -> pd.Series:
    ok = (_col(t, "ltp") > _col(t, "ema50")) & (_col(t, "ema20") >= _col(t, "ema50") * 0.995)
    ok &= _col(t, "rsi14").between(40.0, 74.0) & (_col(t, "ret1") >= -1.0) & (_col(t, "clv") >= 0.5) & (_col(t, "adx14") >= 15.0)
    return ok & _col(t, "dist_ema20_atr").between(-1.0, 3.0) & (comp["pattern"] > 0)


def _gate_swing_short(t: pd.DataFrame, comp: pd.DataFrame) -> pd.Series:
    ok = (_col(t, "ltp") < _col(t, "ema50")) & (_col(t, "ema20") <= _col(t, "ema50") * 1.005) & _fno(t)
    ok &= _col(t, "rsi14").between(26.0, 60.0) & (_col(t, "ret1") <= 1.0) & (_col(t, "clv") <= 0.5) & (_col(t, "adx14") >= 15.0)
    return ok & _col(t, "dist_ema20_atr").between(-3.0, 1.0) & (comp["pattern"] > 0)


@dataclass(frozen=True)
class SetupSpec:
    horizon: str
    side: str
    weights: Mapping[str, float]
    plan: Plan
    gate: Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


SPECS = (
    SetupSpec("BTST", "LONG", BTST_WEIGHTS, BTST_PLAN, _gate_btst_long),
    SetupSpec("SWING", "LONG", SWING_WEIGHTS, SWING_PLAN, _gate_swing_long),
    SetupSpec("BTST", "SHORT", BTST_WEIGHTS, BTST_PLAN, _gate_btst_short),
    SetupSpec("SWING", "SHORT", SWING_WEIGHTS, SWING_PLAN, _gate_swing_short),
)


# ------------------------------------------------------------------------------------------------ levels
def plan_levels(side: str, plan: Plan, row: Mapping[str, Any], *, horizon: str) -> dict[str, float] | None:
    """Reference entry, the level beyond which the price is not worth chasing, stop and two targets of one idea, or ``None`` when
    the figures to place them are missing. The stop is the tighter of an ATR multiple and the nearest structural extreme (the day's
    for an overnight trade, the week's for a swing), never closer than ``min_stop_atr``."""
    entry, atr = _finite(row.get("ltp")), _finite(row.get("atr_abs"))
    if entry is None or atr is None or entry <= 0 or atr <= 0:
        return None
    long = side == "LONG"
    sign = 1.0 if long else -1.0
    if horizon == "BTST":
        extreme = _finite(row.get("day_low" if long else "day_high"))
    else:
        extreme = _finite(row.get("low5" if long else "high5"))
    stop = entry - sign * plan.stop_atr * atr
    if extreme is not None:
        structural = extreme - sign * plan.buffer_atr * atr
        stop = max(stop, structural) if long else min(stop, structural)
    stop = min(stop, entry - plan.min_stop_atr * atr) if long else max(stop, entry + plan.min_stop_atr * atr)
    out = {"entry": _tick(entry), "max_entry": _tick(entry + sign * plan.chase_atr * atr), "stop": _tick(stop),
           "target1": _tick(entry + sign * plan.t1_atr * atr), "target2": _tick(entry + sign * plan.t2_atr * atr)}
    risk = abs(out["entry"] - out["stop"])
    if risk <= 0:
        return None
    out["risk_pct"] = risk / out["entry"] * 100.0
    out["reward1_pct"] = abs(out["target1"] - out["entry"]) / out["entry"] * 100.0
    out["reward2_pct"] = abs(out["target2"] - out["entry"]) / out["entry"] * 100.0
    out["rr1"] = out["reward1_pct"] / out["risk_pct"]
    out["rr2"] = out["reward2_pct"] / out["risk_pct"]
    return out


def watch_levels(row: Mapping[str, Any]) -> dict[str, float] | None:
    """Both triggers of a watchlist name (above today's high, below today's low) with a stop and two targets each."""
    atr, high, low = _finite(row.get("atr_abs")), _finite(row.get("day_high")), _finite(row.get("day_low"))
    if atr is None or high is None or low is None or atr <= 0:
        return None
    p = WATCH_PLAN
    up, down = _tick(high + p.buffer_atr * atr), _tick(low - p.buffer_atr * atr)
    return {
        "trigger_long": up, "long_stop": _tick(up - p.stop_atr * atr), "long_target1": _tick(up + p.t1_atr * atr),
        "long_target2": _tick(up + p.t2_atr * atr),
        "trigger_short": down, "short_stop": _tick(down + p.stop_atr * atr), "short_target1": _tick(down - p.t1_atr * atr),
        "short_target2": _tick(down - p.t2_atr * atr),
    }


# --------------------------------------------------------------------------------------- words and news
def grade_for(score: float) -> str:
    return "A" if score >= GRADE_A else "B" if score >= GRADE_B else "C"


def _metrics(row: Mapping[str, Any]) -> dict[str, float]:
    names = ("clv", "rvol", "ret1", "ret5", "ret20", "atr_pct", "rsi14", "adx14", "dist_ema20_atr", "dist_hi20", "dist_lo20",
             "dist_hi252", "dist_lo252", "rs1", "rs20", "sector_chg", "range_comp", "ema20", "ema50", "atr_abs", "day_high", "day_low")
    return {name: out for name in names if (out := _r(row.get(name), 3)) is not None}


def reasons_for(row: Mapping[str, Any], side: str, pattern: str = "") -> list[str]:
    """What the setup shows, in words with the figures (at most five)."""
    long = side == "LONG"
    out: list[str] = []
    clv, rvol = _finite(row.get("clv")), _finite(row.get("rvol"))
    if clv is not None and (clv >= 0.85 if long else clv <= 0.15):
        out.append(f"closing near the day's {'high' if long else 'low'} (position in the day's range {clv:.2f})")
    if rvol is not None and rvol >= 1.3:
        out.append(f"volume {rvol:.1f}x its 20-day average (projected for the full day)")
    if pattern:
        out.append(pattern)
    gap = _finite(row.get("dist_hi20" if long else "dist_lo20"))
    if row.get("breakout20" if long else "breakdown20") == 1.0:
        out.append(f"trading beyond its prior 20-day {'high' if long else 'low'}")
    elif gap is not None and (gap >= -2.0 if long else gap <= 2.0):
        out.append(f"{abs(gap):.1f}% from its 20-day {'high' if long else 'low'}")
    far = _finite(row.get("dist_hi252" if long else "dist_lo252"))
    if far is not None and abs(far) <= 3.0:
        out.append(f"within {abs(far):.1f}% of its 52-week {'high' if long else 'low'}")
    ltp, ema20, ema50 = (_finite(row.get(n)) for n in ("ltp", "ema20", "ema50"))
    if ltp is not None and ema20 is not None and ema50 is not None and (ltp > ema20 > ema50 if long else ltp < ema20 < ema50):
        out.append(f"trend aligned: {'above' if long else 'below'} its 20 and 50-day averages")
    rs1, rs20 = _finite(row.get("rs1")), _finite(row.get("rs20"))
    if rs1 is not None and (rs1 >= 1.0 if long else rs1 <= -1.0):
        out.append(f"{'out' if long else 'under'}performing Nifty by {abs(rs1):.1f}% today")
    elif rs20 is not None and (rs20 >= 3.0 if long else rs20 <= -3.0):
        out.append(f"{rs20:+.1f}% against Nifty over 20 sessions")
    sector = _finite(row.get("sector_chg"))
    if sector is not None and (sector >= 0.5 if long else sector <= -0.5):
        out.append(f"sector peers {sector:+.1f}% today")
    return out[:5]


def cautions_for(row: Mapping[str, Any], side: str, regime: str, cost_pct: float | None) -> list[str]:
    """What argues against the idea: a stretched stock, a coming event, a market against the side, a high cost."""
    long = side == "LONG"
    out: list[str] = []
    rsi, stretch, move = _finite(row.get("rsi14")), _finite(row.get("dist_ema20_atr")), _finite(row.get("ret1"))
    if rsi is not None and (rsi >= OVERBOUGHT_RSI if long else rsi <= 100.0 - OVERBOUGHT_RSI):
        out.append(f"RSI {rsi:.0f}: already stretched")
    if stretch is not None and (stretch >= STRETCH_ATR if long else stretch <= -STRETCH_ATR):
        out.append(f"{abs(stretch):.1f} ATR {'above' if long else 'below'} its 20-day average: late-entry risk")
    if move is not None and (move >= LATE_CHASE_PCT if long else move <= -LATE_CHASE_PCT):
        out.append(f"already {move:+.1f}% today")
    days = _finite(row.get("event_days"))
    if days is not None:
        when = "today after the close" if days <= 0 else "next session" if days == 1 else f"in {int(days)} sessions"
        out.append(f"results / board meeting {when}: gap risk")
    if (regime == "bearish" and long) or (regime == "bullish" and not long):
        out.append(f"market regime is {regime}: this side is discounted")
    if cost_pct is not None and cost_pct >= 0.4:
        out.append(f"round-trip cost about {cost_pct:.2f}% of the position")
    return out


def news_view(catalysts: Sequence[Catalyst], side: str) -> tuple[float, float, list[str], list[str], bool]:
    """(multiplier, bonus points, reasons, cautions, veto) from the catalysts of a stock for one side. Only a first-hand or confirmed
    report counts; a rumour is only mentioned. ``polarity`` is the likely effect on the share price, never a trading signal."""
    sign = 1.0 if side == "LONG" else -1.0
    multiplier, bonus, veto = 1.0, 0.0, False
    reasons: list[str] = []
    cautions: list[str] = []
    for item in catalysts[:3]:
        headline = (item.headline or "")[:90]
        if item.rumour:
            cautions.append(f"unconfirmed report: {headline}")
            continue
        polarity, severity = _finite(item.polarity), _finite(item.severity)
        confidence = _finite(item.confidence)
        if polarity is None or severity is None or severity < NEWS_MIN_SEVERITY:
            continue
        if confidence is not None and confidence < NEWS_MIN_CONFIDENCE:
            continue
        signed = sign * polarity
        if signed <= -0.6 and severity >= 0.6:
            veto = True
            cautions.append(f"news against this side: {headline}")
        elif signed <= -0.3:
            multiplier = min(multiplier, 0.8)
            cautions.append(f"news against this side: {headline}")
        elif signed >= 0.3:
            bonus = max(bonus, 4.0)
            reasons.append(f"news in its favour: {headline}")
    return multiplier, bonus, reasons, cautions, veto


# ----------------------------------------------------------------------------------------------- ranking
def _adjustments(t: pd.DataFrame, spec: SetupSpec, regime: str) -> pd.Series:
    """Multipliers of the score for a stretched stock, a coming event (a swing runs through it) and a market against the side."""
    long = spec.side == "LONG"
    rsi, dist = _col(t, "rsi14"), _col(t, "dist_ema20_atr")
    if long:
        stretched = (rsi >= OVERBOUGHT_RSI) | (dist >= STRETCH_ATR)
    else:
        stretched = (rsi <= 100.0 - OVERBOUGHT_RSI) | (dist <= -STRETCH_ATR)
    factor = pd.Series(1.0, index=t.index).mask(stretched, 0.92)
    if spec.horizon == "SWING":
        factor = factor.mask(_col(t, "event_days").notna(), factor * 0.85)
    if (regime == "bearish" and long) or (regime == "bullish" and not long):
        factor = factor * 0.9
    return factor


def _make_idea(symbol: str, row: pd.Series, spec: SetupSpec, score: float, pattern: str, info: Mapping[str, Any],
               regime: str, cfg: IdeaConfig, news: Sequence[Catalyst]) -> TradeIdea | None:
    levels = plan_levels(spec.side, spec.plan, row, horizon=spec.horizon)
    if levels is None or levels["risk_pct"] > cfg.max_risk_pct or levels["rr2"] < cfg.min_rr:
        return None
    turnover = _finite(row.get("tv_med20_cr"))
    cost = estimate_delivery_round_trip_cost_pct(turnover)
    if cost > cfg.max_cost_ratio * levels["reward1_pct"]:
        return None
    multiplier, bonus, news_reasons, news_cautions, veto = news_view(news, spec.side)
    final = min(100.0, score * multiplier + bonus)
    if veto or final < cfg.min_score:
        return None
    long = spec.side == "LONG"
    broke = row.get("breakout20" if long else "breakdown20") == 1.0
    if spec.horizon == "SWING":
        setup = pattern
    elif long:
        setup = "closing strength with a breakout" if broke else "strong close on volume"
    else:
        setup = "closing weakness with a breakdown" if broke else "weak close on volume"
    reasons = [*news_reasons, *reasons_for(row, spec.side, pattern if spec.horizon == "SWING" else "")]
    return TradeIdea(
        symbol=symbol, rank=0, horizon=spec.horizon, side=spec.side, setup=setup, score=round(final, 1), grade=grade_for(final),
        ltp=round(float(row["ltp"]), 2), name=info.get("name"), sector=info.get("sector"), is_fno=_is_fno(row, info),
        change_pct=_r(row.get("ret1")), entry=levels["entry"], max_entry=levels["max_entry"], stop=levels["stop"],
        target1=levels["target1"], target2=levels["target2"], risk_pct=round(levels["risk_pct"], 2),
        reward1_pct=round(levels["reward1_pct"], 2), reward2_pct=round(levels["reward2_pct"], 2), rr1=round(levels["rr1"], 2),
        rr2=round(levels["rr2"], 2), hold=spec.plan.hold, reasons=reasons[:5],
        cautions=[*cautions_for(row, spec.side, regime, cost), *news_cautions][:5], catalysts=list(news[:2]),
        turnover_cr=_r(turnover, 1), est_cost_pct=round(cost, 2), metrics=_metrics(row),
    )


def _select(t: pd.DataFrame, market: Mapping[str, Any], cfg: IdeaConfig, spec: SetupSpec, regime: str, taken: set[str],
            instruments: Mapping[str, Mapping[str, Any]], catalysts: Mapping[str, Sequence[Catalyst]], limit: int) -> list[TradeIdea]:
    comp, pattern = components(t, market, spec.side)
    score = weighted_score(comp, spec.weights) * _adjustments(t, spec, regime)
    ok = spec.gate(t, comp) & eligibility(t, cfg)
    if spec.horizon == "BTST":
        ok &= ~(_col(t, "event_days") <= 1.0)  # an overnight trade into a results announcement is a coin flip on the gap
        if overnight_block(market)[0] == spec.side:
            ok &= False
    out: list[TradeIdea] = []
    for symbol in score.where(ok).dropna().sort_values(ascending=False, kind="mergesort").index:
        if len(out) >= limit:
            break
        if symbol in taken:
            continue
        idea = _make_idea(symbol, t.loc[symbol], spec, float(score[symbol]), str(pattern[symbol]), instruments.get(symbol, {}),
                          regime, cfg, catalysts.get(symbol, ()))
        if idea is not None:
            out.append(idea)
    return out


def _watch_idea(symbol: str, row: pd.Series, score: float, leaning: str, info: Mapping[str, Any]) -> TradeIdea | None:
    levels = watch_levels(row)
    if levels is None:
        return None
    comp_ratio, hi_gap, lo_gap = (_finite(row.get(n)) for n in ("range_comp", "dist_hi20", "dist_lo20"))
    near_hi, near_lo = hi_gap is not None and abs(hi_gap) <= 3.0, lo_gap is not None and abs(lo_gap) <= 3.0
    coiled = comp_ratio is not None and comp_ratio <= 0.85
    if coiled and near_hi:
        setup = "tight range below the 20-day high"
    elif coiled and near_lo:
        setup = "tight range above the 20-day low"
    else:
        setup = "volatility squeeze" if coiled else "wide-range stock at a key level"
    reasons = [f"typical daily range {_finite(row.get('atr_pct')) or 0.0:.1f}% (ATR): room to move"]
    if coiled:
        reasons.append(f"last 5 sessions' range {comp_ratio:.2f}x the 20-day average: coiled")
    if near_hi:
        reasons.append(f"{abs(hi_gap):.1f}% from its 20-day high")
    if near_lo:
        reasons.append(f"{abs(lo_gap):.1f}% from its 20-day low")
    rvol = _finite(row.get("rvol"))
    if rvol is not None and rvol >= 1.3:
        reasons.append(f"volume {rvol:.1f}x its average today")
    chosen = {"LONG": "long_", "SHORT": "short_"}.get(leaning)  # a one-sided lean carries its stop and targets in the idea itself
    cautions = cautions_for(row, "SHORT" if leaning == "SHORT" else "LONG", "neutral", None)[:3]
    return TradeIdea(
        symbol=symbol, rank=0, horizon="WATCH", side=leaning, setup=setup, score=round(score, 1), grade=grade_for(score),
        ltp=round(float(row["ltp"]), 2), name=info.get("name"), sector=info.get("sector"), is_fno=_is_fno(row, info),
        change_pct=_r(row.get("ret1")),
        trigger_long=levels["trigger_long"] if leaning != "SHORT" else None,
        trigger_short=levels["trigger_short"] if leaning != "LONG" else None,
        stop=levels[chosen + "stop"] if chosen else None, target1=levels[chosen + "target1"] if chosen else None,
        target2=levels[chosen + "target2"] if chosen else None, hold=WATCH_PLAN.hold, reasons=reasons[:5],
        cautions=[*cautions, "a gap at the open can jump past the level: do not chase it"][:4],
        turnover_cr=_r(row.get("tv_med20_cr"), 1), metrics={**_metrics(row), **levels},
    )


def _watch(t: pd.DataFrame, cfg: IdeaConfig, taken: set[str], instruments: Mapping[str, Mapping[str, Any]],
           limit: int) -> list[TradeIdea]:
    score = weighted_score(watch_components(t), WATCH_WEIGHTS)
    ok = (_col(t, "atr_pct") >= MIN_WATCH_ATR_PCT) & eligibility(t, cfg) & (score >= cfg.min_score)
    ltp, ema20, ema50 = _col(t, "ltp"), _col(t, "ema20"), _col(t, "ema50")
    up = _mean_of(_flag(ltp > ema20, ltp, ema20), _flag(ema20 > ema50, ema20, ema50))  # 1 = both averages say up, 0 = both say down
    clv = _col(t, "clv")
    out: list[TradeIdea] = []
    for symbol in score.where(ok).dropna().sort_values(ascending=False, kind="mergesort").index:
        if len(out) >= limit:
            break
        if symbol in taken:
            continue
        if not cfg.include_shorts or (up[symbol] >= 1.0 and clv[symbol] >= 0.5):
            leaning = "LONG"
        elif up[symbol] <= 0.0 and clv[symbol] <= 0.5:
            leaning = "SHORT"
        else:
            leaning = "BOTH"
        idea = _watch_idea(symbol, t.loc[symbol], float(score[symbol]), leaning, instruments.get(symbol, {}))
        if idea is not None:
            out.append(idea)
    return out


def rank_ideas(
    table: pd.DataFrame, market: Mapping[str, Any], cfg: IdeaConfig, *, instruments: Mapping[str, Mapping[str, Any]] | None = None,
    catalysts: Mapping[str, Sequence[Catalyst]] | None = None, pool: bool = False,
) -> list[TradeIdea]:
    """The ideas of one session, in report order: BTST longs, swing longs, the bearish ideas (BTST and swing shorts together), then
    the next-session intraday watchlist. ``pool=True`` returns three times as many per list: the shortlist that gets the (slow)
    news enrichment before the final call with ``catalysts``. A symbol appears in one list only; ranks restart in each list."""
    instruments = instruments or {}
    catalysts = catalysts or {}
    regime = market.get("regime") or market_regime(market)
    limit = cfg.top_n * (3 if pool else 1)
    taken: set[str] = set()
    lists: dict[str, list[TradeIdea]] = {}
    for spec in SPECS:
        if spec.side == "SHORT" and not cfg.include_shorts:
            continue
        found = _select(table, market, cfg, spec, regime, taken, instruments, catalysts, limit)
        taken.update(i.symbol for i in found)
        lists.setdefault("SHORT" if spec.side == "SHORT" else spec.horizon, []).extend(found)
    if "SHORT" in lists:
        lists["SHORT"] = sorted(lists["SHORT"], key=lambda i: -i.score)
    lists["WATCH"] = _watch(table, cfg, taken, instruments, limit)
    ideas: list[TradeIdea] = []
    for group in ("BTST", "SWING", "SHORT", "WATCH"):
        for rank, idea in enumerate(lists.get(group, [])[:limit], 1):
            idea.rank = rank
            ideas.append(idea)
    return ideas
