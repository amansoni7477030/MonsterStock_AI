"""Telegram-HTML renderer of the closing-hour trade-idea report (pure: no I/O, no clock, no network).

The same guarantees as ``notifications.report``, all covered by tests: every dynamic value is escaped, missing or non-finite numbers
render as ``n/a`` and are never invented, every message is at most 3900 characters with balanced tags, and the words next to an emoji
carry every fact. In addition: a score is always called a score and never drawn as a probability, every list shows the live record
that stands behind it (with its sample size and interval, or the plain statement that there is none yet), and a report built after the
report deadline says so at the top."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from backtesting.ideas import GROUPS, validation_hint
from notifications import style
from notifications.report import (
    NA,
    _catalyst_lines,
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
from ranking.types import IdeasReport

DISCLAIMER = (
    "Research output; not investment advice and not a guarantee. Stops and targets are mechanical multiples of the previous "
    "ATR, costs are estimates, and a bearish idea needs F&O (the cash market cannot hold a short overnight)."
)
HEURISTIC_BANNER = (
    "HEURISTIC SETUPS: the score ranks setups, it is NOT a probability, and no live record proves an edge yet."
)
# once a list's live record has passed the gate the sentence above would be untrue for it: point at the records instead
RECORD_BANNER = "HEURISTIC SETUPS: the score ranks setups, it is NOT a probability; each list shows its own live record."
LATE_BANNER = "built after the report deadline, too late to act on: kept for the record only"

_SECTIONS = {
    "BTST": ("🌙", "BTST - buy before the close, sell in the next session"),
    "SWING": ("📈", "Swing - hold for 2-5 sessions"),
    "SHORT": ("🔻", "Bearish setups - F&O only (short futures, or buy puts)"),
    "WATCH": ("⚡", "Tomorrow's intraday watchlist - levels to react to, not a call to enter today"),
}
_SYMBOL, _NAME, _SECTOR, _SETUP, _LINE = 24, 60, 40, 80, 140
_GRADE_BADGE = {"A": "🔥", "B": "⚡", "C": "👀"}
_HORIZON_WORD = {"BTST": "BTST", "SWING": "swing"}


def _price(value: Any) -> str:
    x = _num(value)
    return NA if x is None else f"₹{x:,.2f}"


def _signed(value: Any, digits: int = 2) -> str:
    x = _num(value)
    if x is None:
        return NA
    if round(x, digits) == 0:
        x = 0.0
    return f"{x:+.{digits}f}%"


def _unsigned(value: Any, digits: int = 2) -> str:
    x = _num(value)
    return NA if x is None else f"{abs(x):.{digits}f}%"


def _ratio(value: Any) -> str:
    x = _num(value)
    return NA if x is None else f"{x:.1f}"


def _interval(pair: Any, *, scale: float = 1.0, digits: int = 0, signed: bool = False) -> str:
    items = _seq(pair)
    low, high = (_num(items[0]), _num(items[1])) if len(items) == 2 else (None, None)
    if low is None or high is None:
        return ""
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return f" [{fmt.format(low * scale)} to {fmt.format(high * scale)}]"


# ------------------------------------------------------------------------------------------ one idea
def _head(idea: Any, position: int) -> list[str]:
    rank = _num(_get(idea, "rank"))
    number = int(rank) if rank is not None and rank >= 1 else position
    symbol = _esc(_clip(_get(idea, "symbol"), _SYMBOL) or "UNKNOWN")
    grade = _clean(_get(idea, "grade")).upper()[:1] or "C"
    horizon = _clean(_get(idea, "horizon")).upper()
    tag = _HORIZON_WORD.get(horizon, "")
    side = _clean(_get(idea, "side")).upper()
    lines = [f"{style.rank_badge(number)} <b>#{number} {symbol}</b> · {_GRADE_BADGE.get(grade, style.UNKNOWN_TIER)} "
             f"<b>Grade {_esc(grade)}</b> · score {_ratio(_get(idea, 'score'))}" + (f" · {tag} {side.lower()}" if tag and side == "SHORT" else "")]
    meta = [_txt(_get(idea, "name"), _NAME, ""), _txt(_get(idea, "sector"), _SECTOR, "")]
    if _get(idea, "is_fno"):
        meta.append(_esc("F&O"))
    if any(meta):
        lines.append("<i>" + " · ".join(part for part in meta if part) + "</i>")
    lines.append(f"💹 {_price(_get(idea, 'ltp'))} ({_signed(_get(idea, 'change_pct'), 1)} today) · setup: {_txt(_get(idea, 'setup'), _SETUP)}")
    return lines


def _levels(idea: Any) -> list[str]:
    return [
        f"🎯 Entry about {_price(_get(idea, 'entry'))} (do not chase beyond {_price(_get(idea, 'max_entry'))}) · "
        f"🛑 Stop {_price(_get(idea, 'stop'))} (risk {_unsigned(_get(idea, 'risk_pct'))})",
        f"🏁 T1 {_price(_get(idea, 'target1'))} ({_signed(_get(idea, 'reward1_pct'))}) · T2 {_price(_get(idea, 'target2'))} "
        f"({_signed(_get(idea, 'reward2_pct'))}) · reward to risk {_ratio(_get(idea, 'rr1'))} / {_ratio(_get(idea, 'rr2'))}",
    ]


def _watch_levels(idea: Any) -> list[str]:
    metrics = _map(_get(idea, "metrics"))
    side = _clean(_get(idea, "side")).upper()
    lines = []
    if _get(idea, "trigger_long") is not None and side in ("LONG", "BOTH"):
        lines.append(f"🔼 Buy above {_price(_get(idea, 'trigger_long'))} · stop {_price(metrics.get('long_stop'))} · "
                     f"T1 {_price(metrics.get('long_target1'))} · T2 {_price(metrics.get('long_target2'))}")
    if _get(idea, "trigger_short") is not None and side in ("SHORT", "BOTH"):
        lines.append(f"🔽 Sell below {_price(_get(idea, 'trigger_short'))} · stop {_price(metrics.get('short_stop'))} · "
                     f"T1 {_price(metrics.get('short_target1'))} · T2 {_price(metrics.get('short_target2'))}")
    return lines or [f"🔼 Levels n/a ({NA})"]


def _idea_block(idea: Any, position: int) -> str:
    watch = _clean(_get(idea, "horizon")).upper() == "WATCH"
    lines = _head(idea, position)
    lines += _watch_levels(idea) if watch else _levels(idea)
    hold = _clip(_get(idea, "hold"), _LINE)
    if hold:
        lines.append(f"⏱ {_esc(hold)}")
    reasons = _items(_get(idea, "reasons"), 4, _LINE)
    lines.append("✅ <b>Why:</b> " + ("; ".join(reasons) if reasons else NA))
    cautions = _items(_get(idea, "cautions"), 3, _LINE)
    if cautions:
        lines.append(f"{style.WARN} <b>Watch out:</b> " + "; ".join(cautions))
    catalysts = [c for c in _seq(_get(idea, "catalysts")) if _clip(_get(c, "headline"), 1)][:1]
    if catalysts:
        lines += _catalyst_lines("Catalyst", catalysts[0])
    cost, turnover = _num(_get(idea, "est_cost_pct")), _num(_get(idea, "turnover_cr"))
    if not watch and (cost is not None or turnover is not None):
        lines.append("💧 " + " · ".join([f"est. round-trip cost {cost:.2f}% (delivery-style)" if cost is not None else "cost n/a",
                                          f"turnover ₹{turnover:,.0f} cr a day" if turnover is not None else "turnover n/a"]))
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------- the record
def _record_line(group: str, record: Mapping[str, Any]) -> str:
    """One line on what the earlier ideas of this list did, always with its sample size and interval."""
    summary = _map(record.get(group))
    n, days = _num(summary.get("n")), _num(summary.get("days"))
    if not summary or not n:
        pending = _num(summary.get("pending")) or 0
        extra = f" ({int(pending)} still inside their horizon)" if pending else ""
        return f"📚 Live record: none judged yet{extra}: heuristic, treat as an experiment"
    head = f"📚 Live record: {int(n)} judged over {int(days or 0)} days"
    if group == "WATCH":
        trig, follow = _num(summary.get("triggered_rate")), _num(summary.get("followed_through_rate"))
        parts = [head, f"a level broke on {_pct_text(trig)} of them", f"closed beyond it on {_pct_text(follow)} of the breaks"]
        return " · ".join(parts)
    parts = [head, f"positive after costs {_pct_text(_num(summary.get('win_rate')))}{_interval(summary.get('win_ci'), scale=100)}",
             f"mean {_signed(summary.get('mean_net_pct'))} net{_interval(summary.get('mean_net_ci'), digits=2, signed=True)}",
             f"first target {_pct_text(_num(summary.get('t1_rate')))} · stopped {_pct_text(_num(summary.get('stop_rate')))}"]
    verdict = "✅ record passes the gate" if summary.get("validated") else f"🧪 not validated: {_esc(validation_hint(summary))}"
    return " · ".join(parts) + f" · {verdict}"


def _pct_text(value: float | None) -> str:
    return NA if value is None or not 0.0 <= value <= 1.0 else f"{int(math.floor(value * 100 + 0.5))}%"


# ------------------------------------------------------------------------------------------ the report
def _market_line(market: Any) -> str:
    m = _map(market)
    parts = [f"Nifty {_signed(m.get('nifty_chg'))}" if _num(m.get("nifty_chg")) is not None else "Nifty n/a"]
    vix, vix_chg = _num(m.get("vix")), _num(m.get("vix_chg"))
    if vix is not None:
        parts.append(f"VIX {vix:.1f}" + (f" ({_signed(vix_chg, 1)})" if vix_chg is not None else ""))
    up, down = _num(m.get("advancers")), _num(m.get("decliners"))
    if up is not None and down is not None:
        parts.append(f"breadth {int(up)}↑ / {int(down)}↓")
    regime = _clean(m.get("regime")).lower()
    if regime in ("bullish", "neutral", "bearish"):
        parts.append(f"regime {regime}")
    return "🌍 <b>Market:</b> " + " · ".join(_esc(p) if "<" in p or "&" in p else p for p in parts)


def render_ideas_report(report: IdeasReport) -> list[str]:
    """The closing-hour report as Telegram messages. Raises ``ValueError`` without a trade date (a report is never anonymous)."""
    if not _clip(_get(report, "trade_date"), 10):
        raise ValueError("trade_date is required")
    entry_by = _clip(_get(report, "entry_by"), 8) or "15:15"
    record = _map(_get(report, "record"))
    validated = any(_map(record.get(group)).get("validated") for group in GROUPS)
    head = [
        "🌙 <b>MONSTER AI</b> · Closing-hour trade ideas",
        f"{_time_label(_get(report, 'snapshot_ist'))} · 📅 {_date_pretty(_get(report, 'trade_date'))} · ⏳ enter before {_esc(entry_by)} IST",
        f"🧪 <b>{RECORD_BANNER if validated else HEURISTIC_BANNER}</b>",
    ]
    if _get(report, "late"):
        head.append(f"⏰ <b>{LATE_BANNER}</b>")
    head += [style.RULE, _market_line(_get(report, "market"))]
    source = _clip(_get(report, "source"), 160)
    if source:
        head.append(f"📡 <b>Live prices:</b> {_esc(source)}")
    blocks = ["\n".join(head)]

    ideas = _seq(_get(report, "ideas"))
    by_group: dict[str, list[Any]] = {group: [] for group in GROUPS}
    for idea in ideas:
        horizon, side = _clean(_get(idea, "horizon")).upper(), _clean(_get(idea, "side")).upper()
        group = "SHORT" if side == "SHORT" and horizon in ("BTST", "SWING") else horizon
        if group in by_group:
            by_group[group].append(idea)
    for group in GROUPS:
        icon, title = _SECTIONS[group]
        members = sorted(by_group[group], key=lambda i: _num(_get(i, "rank")) or math.inf)
        lines = [f"{icon} <b>{_esc(title)}</b>", _record_line(group, record)]
        if not members:
            lines.append("🛑 Nothing passed the gates for this list today.")
        blocks.append("\n".join(lines))
        blocks += [_idea_block(idea, position) for position, idea in enumerate(members, 1)]

    foot = [style.RULE]
    universe, eligible = _num(_get(report, "universe_size")), _num(_get(report, "n_eligible"))
    if universe is not None and universe > 0 and eligible is not None and eligible >= 0:
        foot.append(f"🔬 <b>Universe:</b> {int(eligible)} of {int(universe)} passed the liquidity and data gates")
    notes = _items(_get(report, "notes"), 6, 200)
    if notes:
        foot.append("📝 <b>Notes:</b> " + "; ".join(notes))
    foot += ["", f"{style.WARN} <i>{_esc(DISCLAIMER)}</i>"]
    blocks.append("\n".join(foot))
    return _pack(blocks)
