"""Has the market already reacted to an event? A pure, point-in-time assessment over daily bars.

The answer is ``None`` whenever the data cannot support one (missing bars, too little history, no pre-open quote when
one is needed) - the function never guesses. Only bars whose session had closed before ``as_of`` are used.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd

from app.clock import IST, parse_ts

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
PREOPEN_START = time(9, 0)
BAR_KNOWN_AFTER = time(16, 0)  # a session's bar is treated as known only after this IST time


@dataclass(frozen=True)
class Reaction:
    reacted: bool | None
    gap_pct: float | None
    move_since_event_pct: float | None
    rel_volume: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unknown(note: str) -> Reaction:
    return Reaction(None, None, None, None, note)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def _visible_bars(bars: pd.DataFrame | None, as_of: datetime) -> pd.DataFrame:
    """Bars indexed by calendar date, sorted, with a positive close, restricted to sessions known at ``as_of``."""
    if bars is None or len(bars) == 0 or "close" not in bars.columns:
        return pd.DataFrame(columns=["open", "close", "volume"], index=pd.DatetimeIndex([]))
    frame = bars.copy()
    index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for col in ("open", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce") if col in frame.columns else float("nan")
    frame = frame[frame["close"] > 0]
    as_of_ist = as_of.astimezone(IST)
    last_known = as_of_ist.date() if as_of_ist.time() >= BAR_KNOWN_AFTER else as_of_ist.date() - timedelta(days=1)
    return frame[frame.index <= pd.Timestamp(last_known)]


def _first_session(ref_ist: datetime) -> tuple[date, bool]:
    """(date of the first session that can react, whether the event happened inside that session)."""
    t = ref_ist.time()
    if t < SESSION_OPEN:
        return ref_ist.date(), False
    if t <= SESSION_CLOSE:
        return ref_ist.date(), True
    return ref_ist.date() + timedelta(days=1), False


def assess_reaction(
    first_seen_at: str | datetime | None,
    event_time: str | datetime | None,
    stock_bars: pd.DataFrame | None,
    as_of: str | datetime | None,
    preopen: dict[str, Any] | None = None,
    sector_return: float | None = None,
    *,
    min_move_pct: float = 1.0,
    vol_mult: float = 1.5,
    rel_vol_hi: float = 2.5,
    min_history: int = 15,
    vol_window: int = 40,
) -> Reaction:
    """Did the stock already move (price or volume) because of an event, relative to its normal volatility?

    ``stock_bars``: daily open/close/volume indexed by trade date. ``preopen``: optional pre-open quote for the
    ``as_of`` session (``iep``/``final_price``). ``sector_return``: the sector's percent move over the same window;
    when given, the *excess* move is judged so sector-wide days are not mistaken for stock-specific reactions.
    """
    seen, event, asof = parse_ts(first_seen_at), parse_ts(event_time), parse_ts(as_of)
    if seen is None or asof is None:
        return _unknown("timestamps missing")
    ref = event if event is not None and event <= seen else seen
    if ref > asof:
        return _unknown("event is after the as-of time")
    bars = _visible_bars(stock_bars, asof)
    ref_ist = ref.astimezone(IST)
    session_date, intraday = _first_session(ref_ist)
    session_ts = pd.Timestamp(session_date)
    pre = bars[bars.index < session_ts]
    post = bars[bars.index >= session_ts]
    if len(pre) < min_history:
        return _unknown(f"only {len(pre)} daily bars before the event" if len(bars) else "no price bars")

    prev_close = float(pre["close"].iloc[-1])
    returns = pre["close"].pct_change().dropna().iloc[-vol_window:]
    vol_pct = float(returns.std(ddof=1) * 100.0) if len(returns) >= 10 else float("nan")
    if not math.isfinite(vol_pct):
        return _unknown("insufficient history to measure normal volatility")
    threshold = max(min_move_pct, vol_mult * vol_pct)
    base_volume = pre["volume"].iloc[-20:].dropna()
    base_volume_med = float(base_volume.median()) if len(base_volume) >= 5 else float("nan")

    asof_ist = asof.astimezone(IST)
    asof_day = asof_ist.date()
    price_now: float | None = None
    source = "daily bars"
    if len(post):
        if (post.index[0] - session_ts).days > 7:
            return _unknown("post-event bars start too late to be attributed to the event")
        price_now = float(post["close"].iloc[-1])
        if preopen and asof_day > post.index[-1].date() and asof_ist.time() >= PREOPEN_START:
            quote = _num(preopen.get("iep")) or _num(preopen.get("final_price"))
            if quote is not None:
                price_now, source = quote, "pre-open indicative price"
    else:
        if session_date > asof_day:
            return Reaction(False, None, None, None, "no trading session since the event")
        if session_date < asof_day:
            return _unknown("post-event bars are missing")
        quote = _num(preopen.get("iep")) or _num(preopen.get("final_price")) if preopen else None
        if asof_ist.time() < PREOPEN_START:
            return Reaction(False, None, None, None, "no price discovery yet today")
        if quote is None or asof_ist.time() >= SESSION_OPEN:
            return _unknown("no post-event price available yet")
        price_now, source = quote, "pre-open indicative price"

    gap_pct: float | None = None
    baseline = prev_close
    if len(post) and not intraday:
        first_open = _num(post["open"].iloc[0])
        if first_open is not None:
            gap_pct = (first_open / prev_close - 1.0) * 100.0
    elif not len(post):
        gap_pct = (price_now / prev_close - 1.0) * 100.0
    if intraday and len(post):
        first_open = _num(post["open"].iloc[0])
        baseline = first_open if first_open is not None else prev_close
    move = (price_now / baseline - 1.0) * 100.0

    rel_volume: float | None = None
    if len(post) and math.isfinite(base_volume_med) and base_volume_med > 0:
        recent = post["volume"].iloc[:3].dropna()
        if len(recent):
            rel_volume = float(recent.mean() / base_volume_med)

    adjust = sector_return if sector_return is not None and math.isfinite(sector_return) else 0.0
    excess = move - adjust
    price_hit = abs(excess) >= threshold
    gap_hit = gap_pct is not None and abs(gap_pct - adjust) >= threshold
    volume_hit = rel_volume is not None and rel_volume >= rel_vol_hi and abs(excess) >= 0.5 * threshold
    reacted = bool(price_hit or gap_hit or volume_hit)
    note = f"move {move:+.1f}% vs threshold {threshold:.1f}% (daily vol {vol_pct:.1f}%) from {source}"
    if rel_volume is not None:
        note += f", volume {rel_volume:.1f}x normal"
    if sector_return is not None:
        note += f", sector {sector_return:+.1f}%"
    if intraday:
        note += "; intraday event measured from the session open"
    return Reaction(
        reacted,
        None if gap_pct is None else round(gap_pct, 2),
        round(move, 2),
        None if rel_volume is None else round(rel_volume, 2),
        note,
    )
