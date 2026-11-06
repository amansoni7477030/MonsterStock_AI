"""Telegram-HTML renderer of the 09:30 opening read (pure: no I/O, no clock, no network).

The same guarantees as ``notifications.report`` and ``notifications.ideas``, all covered by tests: every dynamic value is escaped, missing
or non-finite numbers render as ``n/a`` and are never invented, every message is at most 3900 characters with balanced tags, and the
words next to an emoji carry every fact. In addition: what the first fifteen minutes DO and do NOT tell is stated in the head of every
report (the size of the day: validated out of sample; its direction: no edge found), "so far" and "still ahead" are always shown apart,
a probability is always labelled with the threshold it is for, and the live record that stands behind the model is shown with its
sample size and interval, or as the plain statement that there is none yet."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from backtesting.opening import validation_hint
from notifications import style
from notifications.ideas import _pct_text, _price, _signed, _table_price
from notifications.report import (
    NA,
    _clean,
    _clip,
    _date_pretty,
    _esc,
    _get,
    _items,
    _map,
    _num,
    _pack,
    _seq,
    _time_label,
    _txt,
)
from ranking.opening import MODEL_INFO, STRENGTH_HISTORY


def _share(value: Any) -> str:
    return f"{float(value):.0%}"


# every figure below is read from ranking.opening.MODEL_INFO / STRENGTH_HISTORY, so a refit of the model changes the report's words with it
DISCLAIMER = (
    "Research output; not investment advice and not a guarantee. The opening model estimates how WIDE today's range will be (tested out of "
    f"sample on {MODEL_INFO['sessions']} sessions); which way the stock goes was not predictable from the opening, and breakout levels showed no "
    "edge after costs. Stops and targets are mechanical multiples of the previous ATR and costs are estimates. Position size and every order are yours."
)
EVIDENCE = (
    f"📐 <b>What the opening tells ({MODEL_INFO['sessions']} sessions, out of sample):</b> how WIDE the day will be, mostly from each stock's own "
    "volatility plus the opening range. Among names not already past a 3% range, the six rated highest each day went on to a range above 3% in "
    f"{_share(MODEL_INFO['top6_hit'])} of cases ({_share(MODEL_INFO['top6_hit_history_only'])} on the 09:12 information alone, "
    f"{_share(MODEL_INFO['top6_hit_random'])} for six random names). Of the range still ahead after 09:30 the opening adds little. "
    "Direction: no edge found. Breakout levels: no edge after costs. \"So far\" and \"still ahead\" are shown apart."
)
LATE_BANNER = "sent after the report deadline, too late to act on: kept for the record only"
SHADOW_BANNER = "SHADOW MODE: a read of the opening, never an order. It estimates how WIDE the day will be, not which way."

_BADGE = {"QUIET": "😴", "NORMAL": "🙂", "HOT": "🔥", "EXPLOSIVE": "💥"}
_TIER_BADGE = {"STRONG": "🔥", "MODERATE": "⚡", "WATCH": "👀"}
_SYMBOL, _NAME, _SECTOR, _LINE = 24, 60, 40, 140
_GLANCE_ROWS = 20
_STAGE_WORD = {"final": "09:12 report", "pre": "08:45 scan"}


def _pct(value: Any, digits: int = 1) -> str:
    x = _num(value)
    return NA if x is None else f"{x:.{digits}f}%"


def _range(triple: Any, digits: int = 1, so_far: float | None = None) -> str:
    """``median (80%: low-high)`` of a stored (q10, q50, q90) triple, in percent. A low end that is the range already printed (``so_far``)
    reads "at least": the day's range can never be narrower than what the stock has done."""
    items = [_num(v) for v in _seq(triple)]
    if len(items) != 3 or any(v is None for v in items):
        return NA
    low, mid, high = (float(v) for v in items if v is not None)
    if so_far is not None and low <= so_far + 0.005:
        return f"{mid:.{digits}f}% (80%: at least {low:.{digits}f}%, up to {high:.{digits}f}%)"
    return f"{mid:.{digits}f}% (80%: {low:.{digits}f}-{high:.{digits}f}%)"


def _so_far(idea: Any) -> float | None:
    """The range the stock has printed since the open (% of the open), as the report froze it."""
    return _num(_map(_get(idea, "metrics")).get("or_raw_pct"))


def _chance_line(idea: Any) -> str:
    """The chance of a day range above 3 % / 5 %: once the range so far has passed a level there is nothing to estimate, and the line
    says so instead of printing a trivial 100 %."""
    so_far, p3, p5 = _so_far(idea), _num(_get(idea, "p3")), _num(_get(idea, "p5"))
    if so_far is not None and so_far > 5.0:
        return f"📊 Range so far {so_far:.1f}% · <b>already above 3% and 5%</b>"
    if so_far is not None and so_far > 3.0:
        return f"📊 Range so far {so_far:.1f}% · <b>already above 3%</b> · chance of above 5%: {_pct_text(p5)}"
    return f"📊 Chance of a range above 3%: <b>{_pct_text(p3)}</b> · above 5%: {_pct_text(p5)}"


def _by_rank(ideas: Sequence[Any]) -> list[Any]:
    return sorted(ideas, key=lambda i: _num(_get(i, "rank")) or math.inf)


# ------------------------------------------------------------------------------------------------ one name
def _levels(idea: Any) -> list[str]:
    levels = _map(_get(idea, "levels"))
    if _num(levels.get("trigger_long")) is None or _num(levels.get("trigger_short")) is None:
        return [f"🔼 Reference levels n/a ({NA})"]
    return [
        f"🔼 Above {_price(levels.get('trigger_long'))} · stop {_price(levels.get('long_stop'))} · T1 {_price(levels.get('long_target1'))} "
        f"· T2 {_price(levels.get('long_target2'))}",
        f"🔽 Below {_price(levels.get('trigger_short'))} · stop {_price(levels.get('short_stop'))} · T1 {_price(levels.get('short_target1'))} "
        f"· T2 {_price(levels.get('short_target2'))}",
    ]


def _pre_line(idea: Any) -> str | None:
    pre = _map(_get(idea, "pre"))
    if not pre:
        return None
    tier = _clean(pre.get("tier")).upper()
    stage = _STAGE_WORD.get(_clean(pre.get("stage")).lower(), "morning report")
    parts = [f"{_TIER_BADGE.get(tier, style.UNKNOWN_TIER)} <b>{_esc(tier or NA)}</b> in the {_esc(stage)}"]
    rank = _num(pre.get("rank"))
    if rank is not None:
        parts.append(f"rank #{int(rank)}")
    p, threshold = _num(pre.get("p")), _num(pre.get("threshold"))
    if p is not None and 0.0 <= p <= 1.0:
        parts.append(f"it gave {_pct_text(p)} for a range above {threshold:g}%" if threshold is not None else f"it gave {_pct_text(p)}")
    return "🗂 " + " · ".join(parts)


def _read_line(strength: str, ahead_median: float | None) -> str | None:
    if strength == "EXPLOSIVE":
        return "⚠️ big opening: a wide day is likely; which way it goes is not known"
    if strength == "HOT":
        return "⚠️ hot opening: a wider-than-usual day is more likely; which way it goes is not known"
    if strength == "QUIET":
        extra = f" (about {ahead_median:.1f}% is still expected)" if ahead_median is not None else ""
        return f"⏳ quiet opening: it lowers the expected range ahead only modestly{extra}"
    return None


def _idea_block(idea: Any, position: int) -> str:
    rank = _num(_get(idea, "rank"))
    number = int(rank) if rank is not None and rank >= 1 else position
    symbol = _esc(_clip(_get(idea, "symbol"), _SYMBOL) or "UNKNOWN")
    strength = _clean(_get(idea, "strength")).upper()
    badge = _BADGE.get(strength, style.UNKNOWN_TIER)
    lines = [f"{style.rank_badge(number)} <b>#{number} {symbol}</b> · {badge} <b>{_esc(strength or NA)}</b> opening"]
    meta = [_txt(_get(idea, "name"), _NAME, ""), _txt(_get(idea, "sector"), _SECTOR, "")]
    if _get(idea, "is_fno"):
        meta.append(_esc("F&O"))
    if any(meta):
        lines.append("<i>" + " · ".join(part for part in meta if part) + "</i>")
    pre = _pre_line(idea)
    if pre:
        lines.append(pre)
    x = _num(_get(idea, "x"))
    lines.append(f"💹 {_price(_get(idea, 'ltp'))} ({_signed(_get(idea, 'move_pct'), 1)} since the open) · gap {_signed(_get(idea, 'gap_pct'), 1)} "
                 f"· opening range {_pct(_get(idea, 'or_pct'), 2)}" + (f" = {x:.2f}× a normal day" if x is not None else ""))
    ahead = [_num(v) for v in _seq(_get(idea, "ahead"))]
    lines.append(f"🎯 Day range expected {_range(_get(idea, 'day_range'), so_far=_so_far(idea))} · still ahead {_range(_get(idea, 'ahead'))}")
    lines.append(_chance_line(idea))
    lines += _levels(idea)
    read = _read_line(strength, ahead[1] if len(ahead) == 3 else None)
    if read:
        lines.append(read)
    notes = _items(_get(idea, "notes"), 3, _LINE)
    if notes:
        lines.append("📝 " + "; ".join(notes))
    cost = _num(_get(idea, "est_cost_pct"))
    if cost is not None:
        lines.append(f"💧 est. round-trip cost {cost:.2f}% (intraday) · ATR {_pct(_get(idea, 'atr_pct'), 1)} a day")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the whole report
def _market_lines(market: Any) -> list[str]:
    m = _map(market)
    nifty = f"Nifty {_signed(m.get('nifty_chg'))}" if _num(m.get("nifty_chg")) is not None else "Nifty n/a"
    if _num(m.get("nifty_or_pct")) is not None:
        nifty += f" (opening range {_pct(m.get('nifty_or_pct'), 2)}" + (f" = {_num(m.get('nifty_x')):.2f}× normal)" if _num(m.get("nifty_x")) is not None else ")")
    parts = [nifty]
    vix, vix_chg = _num(m.get("vix")), _num(m.get("vix_chg"))
    if vix is not None:
        parts.append(f"VIX {vix:.1f}" + (f" ({_signed(vix_chg, 1)})" if vix_chg is not None else ""))
    up, down = _num(m.get("advancers")), _num(m.get("decliners"))
    if up is not None and down is not None:
        parts.append(f"breadth {int(up)}↑ / {int(down)}↓")
    lines = ["🌍 <b>Market:</b> " + " · ".join(_esc(p) if "<" in p or "&" in p else p for p in parts)]
    kind = _clean(m.get("day_type")).lower()
    median_x, hot = _num(m.get("median_x")), _num(m.get("hot_share"))
    if kind in ("calm", "average", "wide") and median_x is not None:
        icon = {"calm": "😌", "average": "🙂", "wide": "🌪"}[kind]
        lines.append(f"{icon} <b>Opening intensity: {kind}</b> · the median stock's opening range is {median_x:.2f}× its normal day"
                     + (f" · {_pct_text(hot)} of names are hot or explosive" if hot is not None and 0.0 <= hot <= 1.0 else ""))
    for key, icon, label in (("sectors_up", "📈", "Leading"), ("sectors_down", "📉", "Lagging")):
        pairs = [(_clean(_seq(p)[0]), _num(_seq(p)[1])) for p in _seq(m.get(key)) if len(_seq(p)) == 2]
        pairs = [(name, value) for name, value in pairs if name and value is not None]
        if pairs:
            lines.append(f"{icon} {label}: " + ", ".join(f"{_esc(_clip(name, 28))} {_signed(value, 1)}" for name, value in pairs[:3]))
    return lines


def _glance_blocks(picks: Sequence[Any], movers: Sequence[Any]) -> list[str]:
    ideas = [("P", i) for i in _by_rank(picks)] + [("M", i) for i in _by_rank(movers)]
    if not ideas:
        return []
    strength = [f"{'':2} {'SYMBOL':<10} {'OR%':>5} {'×NORM':>5} {'>3%':>4}"]
    levels = [f"{'':2} {'SYMBOL':<10} {'ABOVE':>8} {'BELOW':>8}"]
    for tag, idea in ideas[:_GLANCE_ROWS]:
        symbol = _clip(_get(idea, "symbol"), 10) or "UNKNOWN"
        x, p3, so_far = _num(_get(idea, "x")), _num(_get(idea, "p3")), _so_far(idea)
        chance = "yes" if so_far is not None and so_far > 3.0 else (_pct_text(p3) if p3 is not None else NA)
        strength.append(f"{tag:<2} {symbol:<10} {_table_price(_get(idea, 'or_pct')):>5} {(f'{x:.2f}' if x is not None else NA):>5} {chance:>4}")
        lv = _map(_get(idea, "levels"))
        levels.append(f"{tag:<2} {symbol:<10} {_table_price(lv.get('trigger_long')):>8} {_table_price(lv.get('trigger_short')):>8}")
    return [
        "📋 <b>At a glance</b> (OR = opening range % · ×NORM = vs a normal day · &gt;3% = chance of a range above 3%, yes = already past it)\n<pre>"
        + "\n".join(_esc(line) for line in strength) + "</pre>\n<i>P = a 09:12 pick read again · M = a new opening mover</i>",
        "📐 <b>Reference levels</b> (a break of the opening range's high / low; not a call to enter)\n<pre>"
        + "\n".join(_esc(line) for line in levels) + "</pre>",
    ]


def _legend() -> str:
    parts = []
    for name, floor in (("QUIET", "under 0.30"), ("NORMAL", "0.30-0.60"), ("HOT", "0.60-0.85"), ("EXPLOSIVE", "0.85 and above")):
        _n, rate, ahead = STRENGTH_HISTORY[name]
        parts.append(f"{_BADGE[name]} {name.title()} ({floor}): median {ahead:.1f}% range still ahead · {_pct_text(rate)} ended with a range above 3%")
    return (f"🧭 <b>How to read the opening strength</b> (opening range ÷ the stock's mean daily range; {MODEL_INFO['stock_days']:,} stock-days over "
            f"{MODEL_INFO['sessions']} sessions; the share above 3% is partly already on the board at 09:30)\n" + "\n".join(parts))


def _record_lines(record: Mapping[str, Any]) -> list[str]:
    model, lines = _map(record.get("model")), []
    days, n = _num(model.get("days")), _num(model.get("n"))
    if not model or not days or not n:
        lines.append(f"📚 <b>Live record:</b> none judged yet. The figures come from {MODEL_INFO['sessions']} sessions of research; this report starts its own record")
    else:
        decided = _num(model.get("n_decided"))
        aside = f" (+{int(decided):,} already past 3%: not a test)" if decided else ""
        head = (f"📚 <b>Live record:</b> {int(days)} session(s), {int(n):,} names judged{aside} · stated "
                f"{_pct_text(_num(model.get('mean_p3')))} for a range above 3%, happened {_pct_text(_num(model.get('hit3')))}")
        ci = _seq(model.get("top_ci"))
        top = f" · its top {int(_num(model.get('top_n')) or 6)} made one on {_pct_text(_num(model.get('top_hit')))}"
        if len(ci) == 2 and _num(ci[0]) is not None and _num(ci[1]) is not None:
            top += f" [{_pct_text(_num(ci[0]))}-{_pct_text(_num(ci[1]))}]"
        verdict = "✅ record passes the gate" if model.get("validated") else f"🧪 not validated: {_esc(validation_hint(model))}"
        lines.append(head + top + f" · {verdict}")
    for kind, label in (("PICK", "09:12 picks read again"), ("MOVER", "new movers")):
        s = _map(record.get(kind))
        if s and _num(s.get("n")):
            n_open = _num(s.get("n_open"))
            if n_open is None or n_open > 0:
                chance = (f" · {int(n_open)} not already past 3%: " if n_open is not None else " · ") + (
                    f"range above 3% on {_pct_text(_num(s.get('hit3')))} (stated {_pct_text(_num(s.get('mean_p3')))})")
            else:
                chance = " · all already past 3% at the scan"
            lines.append(f"   ▫️ {label}: {int(_num(s.get('n')) or 0)} judged{chance} · mean range {_pct(s.get('mean_range_pct'), 2)}")
    plan = _map(record.get("plan"))
    if plan and _num(plan.get("n")):
        ci = _seq(plan.get("net_ci"))
        spread = f" [{_signed(ci[0], 2)} to {_signed(ci[1], 2)}]" if len(ci) == 2 and _num(ci[0]) is not None and _num(ci[1]) is not None else ""
        lines.append(f"   ▫️ levels as a plan: {int(_num(plan.get('n')) or 0)} trades · first target {_pct_text(_num(plan.get('t1_rate')))} · stopped "
                     f"{_pct_text(_num(plan.get('stop_rate')))} · mean {_signed(plan.get('mean_net_pct'), 2)} net{spread}")
    return lines


def render_opening_report(report: Any) -> list[str]:
    """The 09:30 report as Telegram messages. Raises ``ValueError`` without a trade date (a report is never anonymous)."""
    if not _clip(_get(report, "trade_date"), 10):
        raise ValueError("trade_date is required")
    elapsed = _num(_get(report, "elapsed_min"))
    span = f" · first {elapsed:.0f} min of trading" if elapsed is not None else ""
    head = [
        "🌅 <b>MONSTER AI</b> · 09:30 Opening read",
        f"{_time_label(_get(report, 'snapshot_ist'))} · 📅 {_date_pretty(_get(report, 'trade_date'))}{span}",
        f"🧪 <b>{SHADOW_BANNER}</b>",
    ]
    if _get(report, "late"):
        head.append(f"⏰ <b>{LATE_BANNER}</b>")
    head += [style.RULE, *_market_lines(_get(report, "market"))]
    source = _clip(_get(report, "source"), 160)
    if source:
        head.append(f"📡 <b>Live prices:</b> {_esc(source)}")
    head += ["", EVIDENCE]
    blocks = ["\n".join(head)]

    picks, movers = _seq(_get(report, "picks")), _seq(_get(report, "movers"))
    blocks += _glance_blocks(picks, movers)
    stage = _STAGE_WORD.get(_clean(_get(report, "pre_stage")).lower(), "")
    if picks:
        title = f"🔎 <b>Your {_esc(stage or 'morning')} picks, read again</b> (what their first 15 minutes showed)"
        ordered = [_idea_block(idea, position) for position, idea in enumerate(_by_rank(picks), 1)]
        blocks += [f"{title}\n\n{ordered[0]}", *ordered[1:]]  # a section's title stays with its first name when the messages are split
    title = "🚀 <b>New opening movers</b> (names the morning report did not list whose opening range stands out; ranked by the range still expected)"
    if movers:
        ordered = [_idea_block(idea, position) for position, idea in enumerate(_by_rank(movers), 1)]
        blocks += [f"{title}\n\n{ordered[0]}", *ordered[1:]]
    else:
        blocks.append(f"{title}\n🛑 No new name passed the gates today: nothing is forced.")
    blocks.append("\n".join(_record_lines(_map(_get(report, "record")))))
    blocks.append(_legend())

    foot = [style.RULE]
    universe, eligible = _num(_get(report, "universe_size")), _num(_get(report, "n_eligible"))
    if universe is not None and universe > 0 and eligible is not None and eligible >= 0:
        foot.append(f"🔬 <b>Universe:</b> {int(eligible)} of {int(universe)} passed the liquidity and data gates")
    notes = _items(_get(report, "notes"), 6, 220)
    if notes:
        foot.append("📝 <b>Notes:</b> " + "; ".join(notes))
    foot += ["", f"{style.WARN} <i>{_esc(DISCLAIMER)}</i>"]
    blocks.append("\n".join(foot))
    return _pack(blocks)
