"""The 09:30 opening read: what the first fifteen minutes of a session add to the pre-open view (pure: no I/O, no clock, no network).

WHAT WAS MEASURED (138 sessions, 2026-04-13 .. 2026-10-30, 28,982 stock-days of the F&O stocks, the exchange's own five-minute candles from the
broker laid over the stored daily history; every figure below is OUT OF SAMPLE: an expanding window fitted on earlier sessions only, judged
from the 21st session on with one session's embargo, so 118 sessions and 25,578 stock-days are judged). The strength bands, the 3 % level and
the hot-opening line come from an earlier 60-session study on Yahoo's bars, which overlaps the last 59 of these sessions; the coefficients
and every figure below were refitted and re-measured on the exchange data, because the live inputs are exchange-grade (the live quotes of NSE
and the broker) while Yahoo's first-fifteen-minute range is about a tenth narrower:

* The SIZE of the day is predictable, mostly from each stock's own volatility (known before the open) and a little more from the opening
  range. The part of the day's range that is ALREADY on the board makes the day-range figures look better than they are (the earlier study
  found that 59 % of its six highest-rated names a day had already printed a range above 3 % in their first fifteen minutes). So the
  day-range claim is judged on the names that had NOT yet crossed the line (24,988 of 25,578 stock-days): AUC for "the day's range exceeds
  3 % of the open" 0.695 on the 09:12 information (history and the gap) and 0.763 with the opening range; of the six highest-rated of those
  names a day, 80 % made a range above 3 % against 63 % on the 09:12 information and 28 % for six random names (gain over the 09:12
  information +17 points, 95 % day-block interval +13 to +20); the stated chances were calibrated over all names (slope 0.96; 26.1 % stated
  against 25.9 % real) and the 80 % range band held 79 % of the time (85 % for hot openings).
* For the range STILL TO COME after 09:30 the opening adds nothing measurable. Among the hot openings (range at least 0.6 times the stock's
  normal day, about 47 names a day) the six with the largest expected remaining range made at least 2 % more on 83 % of the cases against
  54 % for six random hot openings (gain +25 to +33 points), and also 83 % when ranked on the 09:12 information alone (the opening's own
  gain over that: -2 to +2 points). The extension beyond the opening range is not predictable from it at all. A quiet opening in a volatile
  stock does not shrink the range ahead by much: the report shows "so far" and "still ahead" apart.
* DIRECTION is not predictable from the opening range. In the earlier study the move from 09:30 to the close, signed by the first fifteen
  minutes' direction, was statistically zero in every strength band, also market-neutral (not repeated on the exchange data).
* A breakout plan on the opening range (buy above its high or sell below its low, stop 0.5 ATR, first target 0.6 ATR, costs included)
  returned -0.20 % per trade over the 138 sessions (20,115 trades, interval -0.20 to -0.18; -0.02 % before costs): the levels are
  references and a way to size risk, not an edge.

The model is deliberately small and transparent: four inputs (the 20-session mean daily range, the 14-day ATR, the gap and the opening
range, all against the stock's own normal) into logistic and log-linear fits. It predicts how WIDE the day will be; it never says which
way. A chance of a range above a level is exactly 1 once the range so far has passed it, and the day's range band never starts below
the range so far (both cannot be otherwise). Every figure it produces is an estimate from 138 sessions of one market regime (a fit made on
the first half stated 27 % for the second half, which really ran 26 %) and the live record (``backtesting.opening``) keeps checking it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from features.labels import threshold_name
from ranking.costs import estimate_round_trip_cost_pct
from ranking.ideas import watch_levels
from ranking.types import Candidate, Catalyst, OpeningIdea

MODEL_ID = "opening-m3-20261107"
OPEN_MINUTES = 15.0  # three five-minute candles: the opening range
MIN_ELAPSED_MIN = 14.0  # a live feed that lags a little still shows the formed range
MAX_ELAPSED_MIN = 30.0  # later than this the day's range no longer is an opening range
MIN_HISTORY_DAYS = 60  # sessions of stored history a name needs (a recent listing has no "normal" yet)
MAX_GAP_PCT = 20.0  # the research's gate: a bigger gap is a corporate action or a halt, not an opening
PRICE_TOLERANCE = 0.002  # a live row's last / open may sit this far outside its day's high-low before the range is widened to cover them
PRICE_REJECT = 0.02  # ... and this far before the row is thrown out as inconsistent

# mean ratio of the range of the first N minutes to the range of the first fifteen, measured on 28,982 stock-days
GROWTH_MINUTES = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
GROWTH_RATIO = (1.0, 1.069, 1.131, 1.183, 1.232, 1.273)

# fitted on the 138 sessions above: [constant, ln(20-session mean range %), ln(ATR %), |gap| / mean range, ln(opening range / mean range)]
COEF: dict[str, tuple[float, ...]] = {
    "p2": (-2.0656, 3.6921, 1.1057, 0.6346, 2.2003),  # logit of P(day range > 2 %)
    "p3": (-3.3510, 3.6031, 0.4656, 0.4846, 2.0949),
    "p5": (-5.8794, 4.0009, 0.3670, 0.6271, 2.5651),
    "day": (0.3671, 0.7576, 0.1340, 0.1112, 0.4400),  # ln of the day's range, % of the open
    "ahead": (0.0041, 0.5547, 0.3066, 0.0928, 0.2440),  # ln of the range after the scan, % of the price at the scan
}
RESIDUAL_Q = {"day": (-0.4200, -0.0300, 0.4544), "ahead": (-0.4898, -0.0190, 0.5128)}  # 10th, 50th, 90th percentile of the fit's residuals
RANGE_CLIP, ATR_CLIP, X_CLIP, GAPX_CLIP = (0.3, 15.0), (0.2, 15.0), (0.02, 5.0), (0.0, 3.0)
QUANTILE_CLIP = (0.05, 60.0)

# the opening range as a multiple of the stock's mean daily range: bands, and what 138 sessions made of each (stock-days, share of the
# days that went on to a range above 3 % (partly already on the board), median range after 09:30 in % of the price)
STRENGTH_BANDS = ((0.85, "EXPLOSIVE"), (0.60, "HOT"), (0.30, "NORMAL"), (0.0, "QUIET"))
STRENGTH_HISTORY = {"QUIET": (6206, 0.156, 1.71), "NORMAL": (16710, 0.252, 1.85), "HOT": (4263, 0.422, 2.02), "EXPLOSIVE": (1803, 0.727, 2.45)}
DAY_TYPE_EDGES = (0.378, 0.442)  # terciles of the day's median opening-range multiple over the 138 sessions: calm | average | wide
THRESHOLDS = {"p2": 2.0, "p3": 3.0, "p5": 5.0}  # the day-range levels the logistic fits are for, % of the open
PASSED_EPS = 1e-6  # a level counts as passed only when the range so far exceeds it by more than rounding noise (a range of exactly 2 % has not passed 2 %)

MODEL_INFO = {
    "id": MODEL_ID, "sessions": 138, "window": "2026-04-13..2026-10-30", "stock_days": 28982,
    # a range above 3 % among names whose range so far had not crossed it: AUC, and the six highest-rated names of a day
    "auc_with_opening": 0.763, "auc_history_only": 0.695, "top6_hit": 0.801, "top6_hit_history_only": 0.634, "top6_hit_random": 0.281,
    # at least 2 % more range after 09:30 among hot openings: the six with the largest expected remaining range
    "ahead_top6_hit": 0.828, "ahead_top6_hit_history_only": 0.826, "ahead_top6_hit_random": 0.538,
    "band_coverage": 0.793, "calibration_slope": 0.96, "direction_edge": "none found", "breakout_net_pct": -0.196,
}

NO_HISTORY = "no stored history"
INCOMPLETE = "incomplete live row"
INCONSISTENT = "live row inconsistent (last or open far outside the day's range)"
HISTORY_COLUMNS = ("prev_close", "rm20", "atr_abs", "atr_pct", "tv_med20_cr", "hist_days")


# ------------------------------------------------------------------------------------------------ small helpers
def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _r(value: Any, digits: int = 2) -> float | None:
    out = _finite(value)
    return None if out is None else round(out, digits)


def range_growth(elapsed_min: float) -> float:
    """How much wider the range of the first ``elapsed_min`` minutes is, on average, than that of the first fifteen: a scan that runs a
    little late is normalised back to the fifteen-minute basis the model was fitted on (1.0 up to fifteen minutes)."""
    value = _finite(elapsed_min)
    return 1.0 if value is None else float(np.interp(value, GROWTH_MINUTES, GROWTH_RATIO))


def strength_of(x: Any) -> str:
    """The band of an opening-range multiple (``""`` when it is unknown: a missing figure is never read as quiet)."""
    value = _finite(x)
    if value is None:
        return ""
    return next(name for floor, name in STRENGTH_BANDS if value >= floor)


def day_type(median_x: Any) -> str:
    """calm | average | wide: how big the stocks' opening ranges are on the whole today, against the 138 sessions measured."""
    value = _finite(median_x)
    if value is None:
        return ""
    return "calm" if value < DAY_TYPE_EDGES[0] else "wide" if value > DAY_TYPE_EDGES[1] else "average"


# ------------------------------------------------------------------------------------------------ history
def history_stats(o: pd.DataFrame, h: pd.DataFrame, low: pd.DataFrame, c: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """What is known about each stock before today's open, from sessions x symbols frames whose LAST row is the previous session:
    its close, the mean daily range of the last 20 sessions (% of the open; at least 15 of them), the 14-day Wilder ATR (price and %
    of the close), the median traded value of the last 20 sessions (INR crore) and how many sessions of history there are."""
    if len(c) == 0:
        return pd.DataFrame(columns=list(HISTORY_COLUMNS))
    daily_range = (h - low) / o.where(o > 0) * 100.0
    previous = c.shift(1)
    true_range = np.maximum(h - low, np.maximum((h - previous).abs(), (low - previous).abs()))
    atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    window = daily_range.iloc[-20:]
    value = (c * v).iloc[-20:]
    last_close, atr_abs = c.iloc[-1], atr.iloc[-1]
    return pd.DataFrame({
        "prev_close": last_close, "rm20": window.mean().where(window.count() >= 15), "atr_abs": atr_abs,
        "atr_pct": atr_abs / last_close.where(last_close > 0) * 100.0,
        "tv_med20_cr": (value.median() / 1e7).where(value.count() >= 15), "hist_days": c.notna().sum().astype(float),
    })


# ------------------------------------------------------------------------------------------------ the table
def opening_table(live: Mapping[str, Mapping[str, Any]], hist: pd.DataFrame, elapsed_min: float) -> tuple[pd.DataFrame, dict[str, str]]:
    """One row per stock with a usable live row and stored history, and why each other name was left out. ``elapsed_min`` is how long the
    session had been trading when the live rows were stamped: the range is normalised to the first fifteen minutes."""
    growth = range_growth(elapsed_min)
    rows: dict[str, dict[str, float]] = {}
    rejected: dict[str, str] = {}
    for symbol, row in live.items():
        if symbol not in hist.index:
            rejected[symbol] = NO_HISTORY
            continue
        values = [_finite(row.get(k)) for k in ("open", "high", "low", "last", "prev_close")]
        if any(v is None or v <= 0 for v in values):
            rejected[symbol] = INCOMPLETE
            continue
        opened, high, low, last, prev = (float(v) for v in values if v is not None)
        if high < low:
            rejected[symbol] = INCOMPLETE
            continue
        if any(p > high * (1.0 + PRICE_REJECT) or p < low * (1.0 - PRICE_REJECT) for p in (opened, last)):
            rejected[symbol] = INCONSISTENT
            continue
        high, low = max(high, opened, last), min(low, opened, last)  # the day's range cannot be narrower than the prices it contains
        raw = (high - low) / opened * 100.0
        rows[symbol] = {
            "open": opened, "or_high": high, "or_low": low, "ltp": last, "prev_close": prev, "gap_pct": (opened / prev - 1.0) * 100.0,
            "or_raw_pct": raw, "or_pct": raw / growth, "move_pct": (last / opened - 1.0) * 100.0,
            "clv": (last - low) / (high - low) if high > low else 0.5,
        }
    table = pd.DataFrame.from_dict(rows, orient="index", dtype=float)
    if table.empty:
        return pd.DataFrame(columns=["open", "or_high", "or_low", "ltp", "prev_close", "gap_pct", "or_raw_pct", "or_pct", "move_pct", "clv",
                                     "rm20", "atr_pct", "atr_abs", "tv_med20_cr", "hist_days", "x", "gapx"]), rejected
    stored = hist.reindex(table.index)
    for name in ("rm20", "atr_pct", "atr_abs", "tv_med20_cr", "hist_days"):
        table[name] = stored[name].astype(float)
    table["x"] = table["or_pct"] / table["rm20"].where(table["rm20"] > 0)
    table["gapx"] = (table["gap_pct"].abs() / table["rm20"].where(table["rm20"] > 0)).clip(*GAPX_CLIP)
    return table, rejected


# ------------------------------------------------------------------------------------------------ the model
def _features(t: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(t)), np.log(t["rm20"].clip(*RANGE_CLIP)), np.log(t["atr_pct"].clip(*ATR_CLIP)), t["gapx"].clip(*GAPX_CLIP),
        np.log(t["x"].clip(*X_CLIP)),
    ])


def score(table: pd.DataFrame) -> pd.DataFrame:
    """The table with the model's estimates: ``p2`` / ``p3`` / ``p5`` (chance the day's range exceeds 2 / 3 / 5 % of the open; exactly 1 once
    the range so far has passed the level), the day's range quantiles ``day_q10`` / ``day_q50`` / ``day_q90`` (% of the open, never below
    the range so far), the range still to come ``ahead_q10`` / ``_q50`` / ``_q90`` (% of the price now) and the opening ``strength``.
    A row with a missing input gets NaN estimates, never a default."""
    out = table.copy()
    if out.empty:
        for name in ("p2", "p3", "p5", "day_q10", "day_q50", "day_q90", "ahead_q10", "ahead_q50", "ahead_q90"):
            out[name] = pd.Series(dtype=float)
        out["strength"] = pd.Series(dtype=object)
        return out
    features = _features(out)
    for name, threshold in THRESHOLDS.items():
        chance = 1.0 / (1.0 + np.exp(-np.clip(features @ np.array(COEF[name]), -30.0, 30.0)))
        out[name] = np.where(out["or_raw_pct"] > threshold + PASSED_EPS, 1.0, chance)  # a level the range so far has passed is passed: nothing to estimate
    for name in ("day", "ahead"):
        centre = features @ np.array(COEF[name])
        for label, residual in zip(("q10", "q50", "q90"), RESIDUAL_Q[name], strict=True):
            quantile = np.exp(centre + residual).clip(*QUANTILE_CLIP)
            # the day's range can never be narrower than what has printed; the range after the scan has no such bound
            out[f"{name}_{label}"] = np.maximum(quantile, out["or_raw_pct"]) if name == "day" else quantile
    out["strength"] = [strength_of(x) for x in out["x"]]
    return out


# ------------------------------------------------------------------------------------------------ choosing
@dataclass(frozen=True)
class OpeningConfig:
    top_n: int = 8  # new movers listed
    min_x: float = 0.60  # a mover's opening range must be at least this multiple of the stock's mean daily range (0.60 = HOT)
    min_turnover_cr: float = 25.0
    min_price: float = 50.0
    max_cost_ratio: float = 0.25  # the round-trip cost may be at most this share of the median range still ahead
    max_picks: int = 10  # names of the 09:12 report that are read again

    @classmethod
    def from_settings(cls, settings: Any) -> OpeningConfig:
        return cls(top_n=int(settings.opening_top_n), min_x=float(settings.opening_min_x),
                   min_turnover_cr=float(settings.min_median_turnover_cr), min_price=float(settings.min_price),
                   max_cost_ratio=float(settings.max_cost_to_range_ratio))


def est_costs(scored: pd.DataFrame) -> pd.Series:
    """The all-in intraday round-trip cost (% of the position) of each stock from its median traded value."""
    return pd.Series([estimate_round_trip_cost_pct(_finite(t)) for t in scored["tv_med20_cr"]], index=scored.index, dtype=float)


def eligibility(scored: pd.DataFrame, cfg: OpeningConfig) -> pd.Series:
    """The names the research's gates let through: liquid, priced, with enough history, a sane gap, an opening range that exists and a
    round-trip cost that leaves room in the range ahead."""
    if scored.empty:
        return pd.Series(dtype=bool)
    needed = scored[["p3", "day_q50", "ahead_q50", "x", "tv_med20_cr", "atr_abs"]].notna().all(axis=1)
    gates = ((scored["tv_med20_cr"] >= cfg.min_turnover_cr) & (scored["ltp"] >= cfg.min_price) & (scored["hist_days"] >= MIN_HISTORY_DAYS)
             & (scored["gap_pct"].abs() < MAX_GAP_PCT) & (scored["or_pct"] > 0) & (scored["atr_abs"] > 0))
    ratio = est_costs(scored) / scored["ahead_q50"]
    return (needed & gates & (ratio <= cfg.max_cost_ratio)).fillna(False)


def levels_for(row: Mapping[str, Any]) -> dict[str, float]:
    """The reference levels around the opening range: a trigger just beyond its high and its low, each with a stop and two targets, in
    the same ATR multiples as the next-day watchlist (``ranking.ideas.WATCH_PLAN``). References, not an edge: see the module docstring."""
    out = watch_levels({"atr_abs": row.get("atr_abs"), "day_high": row.get("or_high"), "day_low": row.get("or_low")})
    return out or {}


def _gap_note(row: Mapping[str, Any]) -> str | None:
    gap = _finite(row.get("gap_pct"))
    if gap is None or abs(gap) < 0.5:
        return None
    prev, high, low = _finite(row.get("prev_close")), _finite(row.get("or_high")), _finite(row.get("or_low"))
    if prev is None or high is None or low is None:
        return None
    filled = low <= prev if gap > 0 else high >= prev
    return f"gap {'up' if gap > 0 else 'down'} {gap:+.1f}%: " + ("filled inside the opening range" if filled else "still holding")


def _position_note(row: Mapping[str, Any]) -> str | None:
    clv, move = _finite(row.get("clv")), _finite(row.get("move_pct"))
    if clv is None or move is None:
        return None
    if clv >= 0.85 and move > 0:
        return "trading at the top of its opening range"
    if clv <= 0.15 and move < 0:
        return "trading at the bottom of its opening range"
    return None


def _idea(symbol: str, row: Mapping[str, Any], kind: str, rank: int, info: Mapping[str, Any], cost: float | None,
          pre: Mapping[str, Any], catalysts: Sequence[Catalyst]) -> OpeningIdea:
    notes = [n for n in (_gap_note(row), _position_note(row)) if n]
    ahead_median = _finite(row.get("ahead_q50"))
    if cost is not None and ahead_median and cost / ahead_median >= 0.15:
        notes.append(f"round-trip cost about {cost:.2f}% is {cost / ahead_median:.0%} of the median range still ahead")
    for catalyst in list(catalysts)[:1]:
        headline = " ".join(str(catalyst.headline or "").split())[:140]
        if headline:
            notes.append("news: " + headline + (f" ({catalyst.source})" if catalyst.source else ""))
    metrics = {k: v for k, v in ((k, _r(row.get(k), 3)) for k in ("clv", "rm20", "or_raw_pct", "gapx")) if v is not None}
    return OpeningIdea(
        symbol=symbol, rank=rank, kind=kind, ltp=float(row["ltp"]), open=float(row["open"]), prev_close=float(row["prev_close"]),
        or_high=float(row["or_high"]), or_low=float(row["or_low"]), gap_pct=round(float(row["gap_pct"]), 2),
        or_pct=round(float(row["or_pct"]), 2), move_pct=round(float(row["move_pct"]), 2), x=round(float(row["x"]), 2),
        strength=str(row["strength"]), p3=round(float(row["p3"]), 3), p5=round(float(row["p5"]), 3),
        day_range=[round(float(row[f"day_{q}"]), 2) for q in ("q10", "q50", "q90")],
        ahead=[round(float(row[f"ahead_{q}"]), 2) for q in ("q10", "q50", "q90")],
        name=info.get("name"), sector=info.get("sector"), is_fno=bool(info.get("is_fno")), atr_pct=_r(row.get("atr_pct")),
        turnover_cr=_r(row.get("tv_med20_cr"), 1), est_cost_pct=_r(cost), levels={k: float(v) for k, v in levels_for(row).items()},
        pre=dict(pre), notes=notes, metrics=metrics,
    )


def pre_view(candidate: Candidate, stage: str) -> dict[str, Any]:
    """What the morning report said about a name: its place, tier and the chance it gave for its own primary threshold."""
    threshold = _finite(candidate.primary_threshold)
    p = candidate.p.get(f"hit_{threshold_name(threshold)}") if threshold is not None else None
    return {"rank": int(candidate.rank), "tier": str(candidate.tier), "p": _r(p, 3), "threshold": threshold, "stage": stage}


def build_ideas(
    scored: pd.DataFrame, cfg: OpeningConfig, *, instruments: Mapping[str, Mapping[str, Any]], candidates: Sequence[Candidate] = (),
    stage: str = "", news: Callable[[list[str]], Mapping[str, Sequence[Catalyst]]] | None = None,
) -> tuple[list[OpeningIdea], list[OpeningIdea], list[str]]:
    """The names of the morning report read again (in their morning order, with the news the morning report found), the new movers
    (hot openings, those with the largest expected range still to come first; ``news`` is asked once for exactly their symbols) and the
    symbols of the morning report that have no usable row now. A name is never in both lists."""
    costs = est_costs(scored) if not scored.empty else pd.Series(dtype=float)
    ok = eligibility(scored, cfg)
    picks: list[OpeningIdea] = []
    missing: list[str] = []
    for candidate in sorted(candidates, key=lambda c: c.rank)[: cfg.max_picks]:
        symbol = candidate.symbol
        if symbol not in scored.index or not np.isfinite(scored.at[symbol, "p3"]):
            missing.append(symbol)
            continue
        info = instruments.get(symbol, {})
        picks.append(_idea(symbol, scored.loc[symbol], "PICK", len(picks) + 1, {"name": info.get("name") or candidate.name,
                                                                              "sector": info.get("sector") or candidate.sector,
                                                                              "is_fno": info.get("is_fno", candidate.is_fno)},
                           _finite(costs.get(symbol)), pre_view(candidate, stage), candidate.catalysts))
    taken = {i.symbol for i in picks}
    # A new mover is a name whose opening stands out; it is ranked by the range still to come, the part of the day that is not yet on the
    # board (ranking by the chance of a range above 3 % would put the names that have already passed it first, and say nothing new).
    pool = scored[ok & (scored["x"] >= cfg.min_x) & ~scored.index.isin(taken)]
    pool = pool.sort_values(["ahead_q50", "x"], ascending=False).head(cfg.top_n) if len(pool) else pool
    found = (news([str(s) for s in pool.index]) or {}) if news is not None and len(pool) else {}
    movers = [_idea(symbol, row, "MOVER", n, instruments.get(symbol, {}), _finite(costs.get(symbol)), {}, found.get(symbol) or [])
              for n, (symbol, row) in enumerate(pool.iterrows(), 1)]
    return picks, movers, missing


# ------------------------------------------------------------------------------------------------ the market
def market_pulse(scored: pd.DataFrame, indices: Mapping[str, Mapping[str, Any]], *, instruments: Mapping[str, Mapping[str, Any]],
                 nifty_range20: float | None = None, growth: float = 1.0) -> dict[str, Any]:
    """The market at 09:30 in a few figures: Nifty and VIX, breadth, how wide the stocks' opening ranges are on the whole (against the
    138 sessions measured) and the sectors that lead and lag."""
    market: dict[str, Any] = {}
    nifty, vix = indices.get("NIFTY 50"), indices.get("INDIA VIX")
    if nifty:
        market["nifty_chg"] = _r(nifty.get("change_pct"), 2)
        high, low, opened = _finite(nifty.get("high")), _finite(nifty.get("low")), _finite(nifty.get("open"))
        if high is not None and low is not None and opened and opened > 0 and high >= low:
            market["nifty_or_pct"] = round((high - low) / opened * 100.0 / growth, 2)
            normal = _finite(nifty_range20)
            if normal and normal > 0:
                market["nifty_x"] = round(market["nifty_or_pct"] / normal, 2)
    if vix:
        market["vix"], market["vix_chg"] = _r(vix.get("last"), 2), _r(vix.get("change_pct"), 1)
    if scored.empty:
        return {k: v for k, v in market.items() if v is not None}
    move = (scored["ltp"] / scored["prev_close"] - 1.0) * 100.0
    market["advancers"], market["decliners"] = int((move > 0).sum()), int((move < 0).sum())
    market["median_chg"] = _r(move.median(), 2)
    median_x = _finite(scored["x"].median())
    market["median_x"], market["day_type"] = _r(median_x, 3), day_type(median_x)
    market["hot_share"] = _r(float(scored["strength"].isin(["HOT", "EXPLOSIVE"]).mean()), 3)
    sectors = pd.Series({s: instruments.get(s, {}).get("sector") for s in scored.index})
    frame = pd.DataFrame({"sector": sectors, "move": move}).dropna()
    grouped = frame.groupby("sector")["move"].agg(["median", "count"])
    grouped = grouped[grouped["count"] >= 3].sort_values("median", ascending=False)
    if len(grouped) >= 4:
        market["sectors_up"] = [[str(s), round(float(m), 2)] for s, m in grouped["median"].head(3).items()]
        market["sectors_down"] = [[str(s), round(float(m), 2)] for s, m in grouped["median"].tail(3).iloc[::-1].items()]
    return {k: v for k, v in market.items() if v is not None}


def state_rows(scored: pd.DataFrame, ok: pd.Series) -> dict[str, Any]:
    """The eligible rows of the whole table in a compact form, frozen beside the report: the live record judges the model on EVERY
    name it scored (not only those it listed), and a later refit can use these point-in-time rows."""
    columns = ["ltp", "open", "or_high", "or_low", "or_pct", "or_raw_pct", "gap_pct", "x", "rm20", "atr_pct", "p3", "p5", "day_q10", "day_q50",
               "day_q90", "ahead_q50"]
    subset = scored[ok.reindex(scored.index).fillna(False)] if len(scored) else scored
    return {"cols": ["symbol", *columns],
            "rows": [[symbol, *[round(float(row[c]), 4) for c in columns]] for symbol, row in subset[columns].iterrows()]}
