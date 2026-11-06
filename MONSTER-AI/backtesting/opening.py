"""Outcomes of the 09:30 opening read and the live record that stands behind its model (pure: no I/O, no clock).

What is judged, and against what:

* Every name the model scored is judged on the day's own bar (the official end-of-day bar; an outcome judged on a provisional bar
  waits for the official one): did the day's range exceed 3 % / 5 % of the open, and did it fall inside the stated 80 % band. This is the
  model's own claim, so the record judges ALL the eligible names it scored (about 200 a day, frozen beside the report), not only the
  ones that were listed: calibration (what the stated chance came to), the model's top-six hit rate against the base rate. A name whose
  range so far had already passed 3 % at the scan has a stated chance of exactly 1: it is counted apart and is not a test (in the first,
  Yahoo-based replay 59 % of the six highest-rated names of a day had already passed it), so the chance of a range above 3 % is judged on the others.
* The listed names (the 09:12 picks read again and the new movers) are judged the same way, plus what the day did after the scan: the
  range still ahead, whether the opening range broke, and, from five-minute bars when they can be had, the reference levels as a plan
  (the first side to trigger, then the stop, the first target or the 15:15 square-off, costs included).

No edge is claimed anywhere: the plan lines exist so the levels' own record is shown, whatever it is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from backtesting.metrics import day_block_bootstrap

VALIDATE_MIN_DAYS = 20
CALIBRATION_TOLERANCE = 0.05  # the pooled stated chance may be this far (absolute) from what happened and the record still passes
TOP_N = 6  # the size of "the model's top names" in the record: the research's yardstick
MAX_BASIS_GAP_PCT = 8.0  # the day's open this far from the open the scan saw is a price-basis change (split, bonus), not an outcome
BINS = 10
SQUARE_OFF_LAST_BAR = "15:10"  # the last five-minute bar that starts before the 15:15 square-off


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _interval(values: Mapping[str, Sequence[float]]) -> list[float] | None:
    """Day-block bootstrap interval of the mean of per-day lists (see ``backtesting.metrics.day_block_bootstrap``)."""
    days = sorted(values)
    width = max((len(values[d]) for d in days), default=0)
    matrix = np.full((len(days), max(width, 1)), np.nan)
    for row, day in enumerate(days):
        matrix[row, : len(values[day])] = list(values[day])
    low, high = day_block_bootstrap(matrix)
    return [round(low, 4), round(high, 4)] if math.isfinite(low) and math.isfinite(high) else None


def _floats(mapping: Mapping[str, Any] | None, *names: str) -> list[float] | None:
    """The named values of ``mapping`` as floats, or ``None`` when any of them is missing or not finite."""
    values = [_num((mapping or {}).get(name)) for name in names]
    return None if any(v is None for v in values) else [float(v) for v in values if v is not None]


def _band(value: Any) -> tuple[float, float] | None:
    """(q10, q90) of a stored (q10, q50, q90) triple."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        low, high = _num(value[0]), _num(value[2])
        if low is not None and high is not None:
            return low, high
    return None


# ------------------------------------------------------------------------------------------------ one name, one day
def day_outcome(idea: Mapping[str, Any], bar: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What the day did to one frozen name, from the day's own bar (open / high / low / close); ``None`` without a usable bar."""
    day, seen = _floats(bar, "open", "high", "low", "close"), _floats(idea, "open", "or_high", "or_low", "ltp")
    if day is None or seen is None:
        return None
    o, h, low, c = day
    seen_open, scan_high, scan_low, last = seen
    if o <= 0 or h < low or last <= 0 or seen_open <= 0:
        return None
    if abs(o / seen_open - 1.0) * 100.0 > MAX_BASIS_GAP_PCT:
        return {"status": "invalid", "reason": "price basis changed (split or bonus?)", "complete": True}
    range_pct = (h - low) / o * 100.0
    so_far = _num((idea.get("metrics") or {}).get("or_raw_pct")) if isinstance(idea.get("metrics"), Mapping) else None
    so_far = so_far if so_far is not None else _num(idea.get("or_pct"))
    out: dict[str, Any] = {
        "status": "judged", "range_pct": round(range_pct, 3), "hit3": range_pct > 3.0, "hit5": range_pct > 5.0,
        # the range had already passed 3 % when the scan was read: its chance was 1 by construction, so it is not a test of the model
        "decided3": so_far is not None and so_far > 3.0,
        "close_pct": round((c / o - 1.0) * 100.0, 3), "after_scan_pct": round((c / last - 1.0) * 100.0, 3),
        "broke_up": h > scan_high, "broke_down": low < scan_low,
        "ext_up_pct": round(max(0.0, (h - scan_high) / last * 100.0), 3),
        "ext_down_pct": round(max(0.0, (scan_low - low) / last * 100.0), 3), "complete": True,
    }
    band = _band(idea.get("day_range"))
    if band is not None:
        out["in_band"] = band[0] <= range_pct <= band[1]
    return out


def _after(bars: Sequence[Mapping[str, Any]], from_hhmm: str) -> list[Mapping[str, Any]]:
    return [b for b in bars if from_hhmm <= str(b.get("hhmm") or "") <= SQUARE_OFF_LAST_BAR
            and _floats(b, "open", "high", "low", "close") is not None]


def ahead_outcome(idea: Mapping[str, Any], bars: Sequence[Mapping[str, Any]], from_hhmm: str) -> dict[str, Any] | None:
    """The range the price actually made after the scan (% of the price at the scan) against the stated band, from five-minute bars."""
    window = _after(bars, from_hhmm)
    last = _num(idea.get("ltp"))
    if not window or not last or last <= 0:
        return None
    pct = (max(float(b["high"]) for b in window) - min(float(b["low"]) for b in window)) / last * 100.0
    out: dict[str, Any] = {"ahead_pct": round(pct, 3), "ahead_bars": len(window)}
    band = _band(idea.get("ahead"))
    if band is not None:
        out["ahead_in_band"] = band[0] <= pct <= band[1]
    return out


def plan_outcome(idea: Mapping[str, Any], bars: Sequence[Mapping[str, Any]], from_hhmm: str) -> dict[str, Any] | None:
    """The reference levels as a plan, judged on five-minute bars from ``from_hhmm`` (the scan's time, floored to five minutes): the
    first side whose trigger the price reaches is entered (a gap through the trigger fills at the open), the other side is cancelled;
    the exit is the stop, the first target or the last bar before the 15:15 square-off; a bar that reaches both the stop and the target
    counts as stopped (the order inside a bar is unknown), and a bar that reaches both triggers is left out as unknowable. Costs are
    the stored all-in intraday estimate. ``None`` without levels or bars."""
    levels = idea.get("levels") or {}
    window = _after(bars, from_hhmm)
    if not window or not levels:
        return None
    long_t, short_t = _num(levels.get("trigger_long")), _num(levels.get("trigger_short"))
    if long_t is None or short_t is None:
        return None
    first_long = next((i for i, b in enumerate(window) if float(b["high"]) >= long_t), None)
    first_short = next((i for i, b in enumerate(window) if float(b["low"]) <= short_t), None)
    if first_long is None and first_short is None:
        return {"status": "none", "complete": True}
    if first_long is not None and first_long == first_short:
        return {"status": "both", "complete": True}
    side = 1 if first_short is None or (first_long is not None and first_long < first_short) else -1
    start = first_long if side > 0 else first_short
    prefix = "long" if side > 0 else "short"
    stop, target = _num(levels.get(f"{prefix}_stop")), _num(levels.get(f"{prefix}_target1"))
    trigger = long_t if side > 0 else short_t
    if stop is None or target is None or start is None:
        return None
    entry = max(trigger, float(window[start]["open"])) if side > 0 else min(trigger, float(window[start]["open"]))
    status, exit_price = "time", float(window[-1]["close"])
    for index in range(start, len(window)):
        bar = window[index]
        o, h, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
        later = index > start
        if side > 0:
            if later and o <= stop:
                status, exit_price = "stop", o
            elif low <= stop:
                status, exit_price = "stop", stop
            elif h >= target:
                status, exit_price = "target1", max(target, o) if later else target
        else:
            if later and o >= stop:
                status, exit_price = "stop", o
            elif h >= stop:
                status, exit_price = "stop", stop
            elif low <= target:
                status, exit_price = "target1", min(target, o) if later else target
        if status != "time":
            break
    gross = side * (exit_price / entry - 1.0) * 100.0
    cost = _num(idea.get("est_cost_pct")) or 0.0
    return {"status": status, "side": "LONG" if side > 0 else "SHORT", "entry": round(entry, 2), "exit": round(exit_price, 2),
            "ret_pct": round(gross, 3), "net_pct": round(gross - cost, 3), "t1_hit": status == "target1", "complete": True}


# ------------------------------------------------------------------------------------------------ the whole table, one day
def universe_outcome(rows: Sequence[Mapping[str, Any]], bars: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """The model judged on every eligible name it scored that day: ``rows`` are the frozen state rows (a dict per symbol with the
    columns of ``ranking.opening.state_rows``), ``bars`` the day's bars by symbol.

    The chance of a range above 3 % is judged on the names whose range so far had NOT yet passed 3 % at the scan: for the others it is 1
    by construction and a hit proves nothing (in the first, Yahoo-based replay 59 % of the six highest-rated names of a day were already past
    it). The stated 80 % band
    is judged on every name. ``None`` when fewer than 20 undecided names can be judged."""
    judged: list[tuple[float, float, float, bool]] = []  # undecided names: p3, p5, range, hit3
    inside: list[bool] = []  # every name: inside the stated band
    decided = 0
    for row in rows:
        day = _floats(bars.get(str(row.get("symbol"))), "open", "high", "low")
        seen = _floats(row, "open", "p3", "p5")
        if day is None or seen is None:
            continue
        o, h, low = day
        seen_open, p3, p5 = seen
        if o <= 0 or h < low or seen_open <= 0 or abs(o / seen_open - 1.0) * 100.0 > MAX_BASIS_GAP_PCT:
            continue
        rng = (h - low) / o * 100.0
        band = _band([row.get("day_q10"), None, row.get("day_q90")])
        if band is not None:
            inside.append(band[0] <= rng <= band[1])
        so_far = _num(row.get("or_raw_pct"))
        so_far = so_far if so_far is not None else _num(row.get("or_pct"))
        if so_far is not None and so_far > 3.0:
            decided += 1
            continue
        judged.append((p3, p5, rng, rng > 3.0))
    if len(judged) < 20:
        return None
    p3 = np.array([j[0] for j in judged])
    p5 = np.array([j[1] for j in judged])
    rng = np.array([j[2] for j in judged])
    hit3 = np.array([float(j[3]) for j in judged])
    top = np.argsort(-p3, kind="stable")[:TOP_N]
    bins = np.minimum((p3 * BINS).astype(int), BINS - 1)
    calib = [[int((bins == b).sum()), round(float(p3[bins == b].sum()), 4), int(hit3[bins == b].sum())] for b in range(BINS)]
    return {
        "n": len(judged), "n_decided": decided, "hit3": round(float(hit3.mean()), 4), "mean_p3": round(float(p3.mean()), 4),
        "hit5": round(float((rng > 5.0).mean()), 4), "mean_p5": round(float(p5.mean()), 4),
        "top_n": len(top), "top_hit3": round(float(hit3[top].mean()), 4), "top_mean_range": round(float(rng[top].mean()), 3),
        "mean_range": round(float(rng.mean()), 3), "band_cover": round(float(np.mean(inside)), 4) if inside else None, "band_n": len(inside),
        "calib": calib,
    }


# ------------------------------------------------------------------------------------------------ the record
def _rate(items: Sequence[bool]) -> float | None:
    return round(sum(1 for x in items if x) / len(items), 4) if items else None


def summarize_universe(days: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The pooled record of the model over every judged name: calibration of its stated chance of a range above 3 % (names not already
    past it at the scan), its top names against the base rate of those names, and the coverage of its stated band (every name).
    ``days`` is ``{trade date: universe outcome}`` of the days that are final."""
    ordered = {d: u for d, u in sorted(days.items()) if u}
    out: dict[str, Any] = {"days": len(ordered), "n": int(sum(u["n"] for u in ordered.values())), "validated": False}
    if not ordered:
        return out
    n = out["n"]
    out["n_decided"] = int(sum(u.get("n_decided") or 0 for u in ordered.values()))
    out["hit3"] = round(sum(u["hit3"] * u["n"] for u in ordered.values()) / n, 4)
    out["mean_p3"] = round(sum(u["mean_p3"] * u["n"] for u in ordered.values()) / n, 4)
    out["gap"] = round(out["hit3"] - out["mean_p3"], 4)
    out["gap_ci"] = _interval({d: [u["hit3"] - u["mean_p3"]] for d, u in ordered.items()})
    out["top_n"] = TOP_N
    out["top_hit"] = round(float(np.mean([u["top_hit3"] for u in ordered.values()])), 4)
    out["top_ci"] = _interval({d: [u["top_hit3"]] for d, u in ordered.items()})
    out["top_mean_range"] = round(float(np.mean([u["top_mean_range"] for u in ordered.values()])), 3)
    out["mean_range"] = round(float(np.mean([u["mean_range"] for u in ordered.values()])), 3)
    covers = [(u["band_cover"], u.get("band_n") or u["n"]) for u in ordered.values() if u.get("band_cover") is not None]
    out["band_cover"] = round(sum(c * k for c, k in covers) / sum(k for _c, k in covers), 4) if covers else None
    out["validated"] = bool(
        len(ordered) >= VALIDATE_MIN_DAYS and abs(out["gap"]) <= CALIBRATION_TOLERANCE and out["top_ci"] and out["top_ci"][0] > out["hit3"]
    )
    return out


def validation_hint(summary: Mapping[str, Any]) -> str:
    """What the model's record still needs before it passes its gate (empty once it has)."""
    if summary.get("validated"):
        return ""
    days = int(summary.get("days") or 0)
    if days < VALIDATE_MIN_DAYS:
        return f"needs {VALIDATE_MIN_DAYS} judged sessions (has {days})"
    gap = _num(summary.get("gap"))
    if gap is not None and abs(gap) > CALIBRATION_TOLERANCE:
        return f"the stated chance is off by {abs(gap) * 100:.0f} points"
    return "its top names do not yet clearly beat the day's base rate"


def summarize_listed(kind: str, items: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    """The record of one list (the 09:12 picks read again, the new movers): ``items`` are (trade date, frozen name, outcome). The chance
    of a range above 3 % (``hit3`` against ``mean_p3``) is judged on the names that had not already passed 3 % at the scan (``n_open``);
    the others are counted in ``n_decided`` and in the mean range only."""
    judged = [(d, i, o) for d, i, o in items if o.get("complete") and o.get("status") == "judged"]
    pending = sum(1 for _d, _i, o in items if not o.get("complete"))
    open3 = [(d, i, o) for d, i, o in judged if not o.get("decided3")]
    out: dict[str, Any] = {"kind": kind, "n": len(judged), "days": len({d for d, _i, _o in judged}), "pending": pending,
                           "n_open": len(open3), "n_decided": len(judged) - len(open3)}
    if not judged:
        return out
    out["hit3"] = _rate([bool(o["hit3"]) for _d, _i, o in open3])
    out["mean_p3"] = round(float(np.mean([float(i["p3"]) for _d, i, _o in open3])), 4) if open3 else None
    out["mean_range_pct"] = round(float(np.mean([o["range_pct"] for _d, _i, o in judged])), 3)
    in_band = [o["in_band"] for _d, _i, o in judged if "in_band" in o]
    out["band_cover"] = _rate(in_band) if in_band else None
    ahead = [o for _d, _i, o in judged if o.get("ahead_pct") is not None]
    out["n_ahead"] = len(ahead)
    out["ahead_cover"] = _rate([bool(o["ahead_in_band"]) for o in ahead if "ahead_in_band" in o]) if ahead else None
    out["broke_up_rate"] = _rate([bool(o["broke_up"]) for _d, _i, o in judged])
    out["broke_down_rate"] = _rate([bool(o["broke_down"]) for _d, _i, o in judged])
    return out


def summarize_plan(items: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """The reference levels judged as a plan, over every listed name whose five-minute bars could be had: (trade date, plan outcome)."""
    judged = [(d, p) for d, p in items if p.get("complete")]
    traded = [(d, p) for d, p in judged if p.get("status") in ("stop", "target1", "time")]
    out: dict[str, Any] = {"judged": len(judged), "n": len(traded), "days": len({d for d, _p in traded})}
    if not traded:
        return out
    net: dict[str, list[float]] = {}
    for day, plan in traded:
        net.setdefault(day, []).append(float(plan["net_pct"]))
    out.update(
        triggered_rate=round(len(traded) / len(judged), 4), t1_rate=_rate([p["t1_hit"] for _d, p in traded]),
        stop_rate=_rate([p["status"] == "stop" for _d, p in traded]),
        mean_net_pct=round(float(np.mean([v for vs in net.values() for v in vs])), 3), net_ci=_interval(net),
    )
    return out


def build_record(
    outcomes_by_day: Mapping[str, Mapping[str, Any]], ideas_by_day: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """The record from the stored outcomes (``{"universe": ..., "ideas": {key: outcome}}`` per trade date) against the frozen ideas of
    each date. A day that is not final (judged on provisional bars) is left to wait."""
    universes = {d: o["universe"] for d, o in outcomes_by_day.items() if o.get("official") and o.get("universe")}
    record: dict[str, Any] = {"model": summarize_universe(universes)}
    lists: dict[str, list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]] = {"PICK": [], "MOVER": []}
    plans: list[tuple[str, Mapping[str, Any]]] = []
    for day, ideas in ideas_by_day.items():
        found = (outcomes_by_day.get(day) or {}).get("ideas") or {}
        official = bool((outcomes_by_day.get(day) or {}).get("official"))
        for idea in ideas:
            outcome = found.get(f"{idea.get('kind')}:{idea.get('symbol')}")
            kind = str(idea.get("kind"))
            if outcome is None or kind not in lists:
                continue
            final = outcome if official else {**outcome, "complete": False}
            lists[kind].append((day, idea, final))
            plan = outcome.get("plan")
            if official and isinstance(plan, dict):
                plans.append((day, plan))
    for kind, items in lists.items():
        if items:
            record[kind] = summarize_listed(kind, items)
    if plans:
        record["plan"] = summarize_plan(plans)
    return record
