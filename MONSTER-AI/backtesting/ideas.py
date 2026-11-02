"""Outcomes and the live record of the closing-hour trade ideas (pure functions over stored daily bars).

An idea is judged on the daily bars of the sessions after it was frozen, with the rules the report states for it:

* ``BTST`` and ``SWING``: the position is entered at the reference price, and left at the FIRST of its stop, its first target or the
  end of its horizon (1 session for BTST, 5 for a swing), at the close of the last session. A session that touches both the stop
  and the target counts as stopped (a daily bar cannot say which came first: the judgement is the cautious one), a gap through the
  stop exits at the open. The result is also given net of the estimated delivery round-trip cost the report showed.
* ``WATCH``: whether the next session broke the trigger level(s) and where it closed relative to them.

The record per horizon pools these outcomes over days with a day-block bootstrap (the ideas of a day share one market), and a horizon
is only called ``validated`` when its interval excludes "no edge" at a sample large enough to mean something. Nothing here reads the
database: ``automation.ideas`` supplies the bars and stores the results apart from the frozen reports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from backtesting.metrics import MIN_BOOT_DAYS, day_block_bootstrap

HORIZON_SESSIONS = {"BTST": 1, "SWING": 5, "WATCH": 1}
MAX_GAP_PCT = 25.0  # a first open this far from the reference price is a price-basis change (split, bonus), not an outcome
VALIDATE_MIN_IDEAS = 60
VALIDATE_MIN_DAYS = 20
GROUPS = ("BTST", "SWING", "SHORT", "WATCH")


def group_of(idea: Mapping[str, Any]) -> str:
    """The list an idea was reported in: the bearish ideas (BTST and swing shorts) share one list."""
    horizon, side = str(idea.get("horizon") or ""), str(idea.get("side") or "")
    return "SHORT" if side == "SHORT" and horizon in ("BTST", "SWING") else horizon


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _sessions(bars: pd.DataFrame) -> list[tuple[float, float, float, float]]:
    """(open, high, low, close) of every session of ``bars`` that has all four, oldest first."""
    out = []
    for row in bars.sort_index().itertuples():
        values = tuple(_num(getattr(row, name)) for name in ("open", "high", "low", "close"))
        if None in values:
            break  # a session without a full bar ends what can be judged: never skip over it
        out.append(values)
    return out


def bracket_outcome(
    side: str, entry: float, stop: float, target1: float, target2: float | None,
    sessions: Sequence[tuple[float, float, float, float]], horizon_sessions: int,
) -> dict[str, Any]:
    """Walk the sessions after the idea: leave at the stop, the first target or the end of the horizon, whichever comes first. See
    the module docstring for the rules."""
    long = side == "LONG"
    sign = 1.0 if long else -1.0
    window = list(sessions[:horizon_sessions])
    status, exit_price = "time", window[-1][3]
    t1_hit = t2_hit = False
    for number, (o, h, low, _c) in enumerate(window, 1):
        gapped = o <= stop if long else o >= stop
        stopped = low <= stop if long else h >= stop
        reached1 = h >= target1 if long else low <= target1
        if target2 is not None and (h >= target2 if long else low <= target2):
            t2_hit = t2_hit or not (gapped or stopped)
        if gapped:
            status, exit_price = "stop", o
        elif stopped:
            status, exit_price = "stop", stop
        elif reached1:
            status, exit_price, t1_hit = "target1", (o if (o >= target1 if long else o <= target1) else target1), True
        if status != "time":
            window = window[:number]
            break
    first_open, first_close = sessions[0][0], sessions[0][3]
    highest, lowest = max(h for _o, h, _l, _c in window), min(low for _o, _h, low, _c in window)
    mfe = (highest / entry - 1.0) if long else (1.0 - lowest / entry)
    mae = (lowest / entry - 1.0) if long else (1.0 - highest / entry)
    return {
        "status": status, "exit_price": round(exit_price, 2), "ret_pct": round(sign * (exit_price / entry - 1.0) * 100.0, 3),
        "gap_pct": round(sign * (first_open / entry - 1.0) * 100.0, 3), "close_pct": round(sign * (first_close / entry - 1.0) * 100.0, 3),
        "mfe_pct": round(mfe * 100.0, 3), "mae_pct": round(mae * 100.0, 3),
        "t1_hit": bool(t1_hit or status == "target1"), "t2_hit": bool(t2_hit), "sessions": len(window),
    }


def watch_outcome(idea: Mapping[str, Any], session: tuple[float, float, float, float]) -> dict[str, Any]:
    """The next session against the trigger levels of a watchlist name."""
    o, h, low, c = session
    long_level, short_level = _num(idea.get("trigger_long")), _num(idea.get("trigger_short"))
    metrics = idea.get("metrics") or {}
    long_hit = long_level is not None and h >= long_level
    short_hit = short_level is not None and low <= short_level
    out: dict[str, Any] = {
        "status": "both" if long_hit and short_hit else "long" if long_hit else "short" if short_hit else "none",
        "range_pct": round((h - low) / o * 100.0, 3) if o > 0 else None, "long_triggered": bool(long_hit),
        "short_triggered": bool(short_hit), "sessions": 1,
    }
    if long_hit:
        out["long_close_pct"] = round((c / long_level - 1.0) * 100.0, 3)
        target = _num(metrics.get("long_target1"))
        out["long_t1_hit"] = bool(target is not None and h >= target)
    if short_hit:
        out["short_close_pct"] = round((short_level / c - 1.0) * 100.0, 3)
        target = _num(metrics.get("short_target1"))
        out["short_t1_hit"] = bool(target is not None and low <= target)
    return out


def idea_outcome(idea: Mapping[str, Any], bars: pd.DataFrame) -> dict[str, Any] | None:
    """The outcome of one frozen idea from the sessions AFTER its date (``bars``: open / high / low / close indexed by date), or
    ``None`` while there is no complete session to judge it on. ``complete`` says the whole horizon has been seen."""
    sessions = _sessions(bars)
    horizon = str(idea.get("horizon") or "")
    wanted = HORIZON_SESSIONS.get(horizon)
    if not sessions or wanted is None:
        return None
    if horizon == "WATCH":
        return {**watch_outcome(idea, sessions[0]), "complete": True}
    entry, stop, target1 = _num(idea.get("entry")), _num(idea.get("stop")), _num(idea.get("target1"))
    if entry is None or stop is None or target1 is None or entry <= 0:
        return None
    if abs(sessions[0][0] / entry - 1.0) * 100.0 > MAX_GAP_PCT:
        return {"status": "invalid", "reason": "price basis changed (split or bonus?)", "complete": True, "sessions": 0}
    out = bracket_outcome(str(idea.get("side")), entry, stop, target1, _num(idea.get("target2")), sessions, wanted)
    cost = _num(idea.get("est_cost_pct")) or 0.0
    out["net_pct"] = round(out["ret_pct"] - cost, 3)
    out["complete"] = out["status"] != "time" or len(sessions) >= wanted
    return out


# --------------------------------------------------------------------------------------------- record
def _by_day(values: Mapping[str, list[float]]) -> np.ndarray:
    """A (days x ideas) matrix, oldest day first, NaN where a day has fewer ideas: the shape ``day_block_bootstrap`` takes."""
    days = sorted(values)
    width = max((len(v) for v in values.values()), default=0)
    matrix = np.full((len(days), max(width, 1)), np.nan)
    for row, day in enumerate(days):
        matrix[row, : len(values[day])] = values[day]
    return matrix


def _interval(values: Mapping[str, list[float]]) -> list[float] | None:
    low, high = day_block_bootstrap(_by_day(values))
    return [round(low, 4), round(high, 4)] if math.isfinite(low) and math.isfinite(high) else None


def summarize_group(group: str, items: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """The live record of one list: ``items`` are (trade date, outcome) of every idea that was reported in it. Only ideas whose
    whole horizon has been seen count; the rest are in ``pending``."""
    judged = [(day, o) for day, o in items if o.get("complete") and o.get("status") != "invalid"]
    pending = sum(1 for _day, o in items if not o.get("complete"))
    days = sorted({day for day, _o in judged})
    out: dict[str, Any] = {"group": group, "n": len(judged), "days": len(days), "pending": pending, "validated": False}
    if not judged:
        return out
    if group == "WATCH":
        out["triggered_rate"] = round(sum(1 for _d, o in judged if o["status"] != "none") / len(judged), 4)
        ranges = [o["range_pct"] for _d, o in judged if o.get("range_pct") is not None]
        out["mean_range_pct"] = round(float(np.mean(ranges)), 3) if ranges else None
        closes = [v for _d, o in judged for v in (o.get("long_close_pct"), o.get("short_close_pct")) if v is not None]
        out["followed_through_rate"] = round(sum(1 for v in closes if v > 0) / len(closes), 4) if closes else None
        out["n_triggers"] = len(closes)
        return out
    net: dict[str, list[float]] = {}
    wins: dict[str, list[float]] = {}
    for day, o in judged:
        net.setdefault(day, []).append(float(o["net_pct"]))
        wins.setdefault(day, []).append(1.0 if o["net_pct"] > 0 else 0.0)
    out.update(
        win_rate=round(float(np.mean([v for vs in wins.values() for v in vs])), 4), win_ci=_interval(wins),
        mean_net_pct=round(float(np.mean([v for vs in net.values() for v in vs])), 3), mean_net_ci=_interval(net),
        t1_rate=round(sum(1 for _d, o in judged if o["t1_hit"]) / len(judged), 4),
        stop_rate=round(sum(1 for _d, o in judged if o["status"] == "stop") / len(judged), 4),
        mean_gap_pct=round(float(np.mean([o["gap_pct"] for _d, o in judged])), 3),
    )
    win_ci, net_ci = out["win_ci"], out["mean_net_ci"]
    out["validated"] = bool(
        len(judged) >= VALIDATE_MIN_IDEAS and len(days) >= VALIDATE_MIN_DAYS and win_ci and net_ci
        and win_ci[0] > 0.5 and net_ci[0] > 0.0
    )
    return out


def validation_hint(summary: Mapping[str, Any]) -> str:
    """What a list still needs before its record can pass the gate (empty once it has)."""
    if summary.get("validated"):
        return ""
    n, days = int(summary.get("n") or 0), int(summary.get("days") or 0)
    if n < VALIDATE_MIN_IDEAS or days < VALIDATE_MIN_DAYS:
        return f"needs {VALIDATE_MIN_IDEAS} judged ideas over {VALIDATE_MIN_DAYS} days (has {n} over {days})"
    if days < MIN_BOOT_DAYS or not summary.get("win_ci"):
        return "too few days for an interval"
    return "the interval still includes no edge"


def build_record(outcomes_by_day: Mapping[str, Mapping[str, Any]], ideas_by_day: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """The record of every list from the stored outcomes: ``outcomes_by_day[date][idea key]`` against the frozen ideas of each date."""
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = {g: [] for g in GROUPS}
    for day, ideas in ideas_by_day.items():
        found = outcomes_by_day.get(day, {})
        for idea in ideas:
            key = f"{idea.get('horizon')}:{idea.get('side')}:{idea.get('symbol')}"
            outcome = found.get(key)
            group = group_of(idea)
            if outcome is not None and group in grouped:
                grouped[group].append((day, outcome))
    return {group: summarize_group(group, items) for group, items in grouped.items() if items}
