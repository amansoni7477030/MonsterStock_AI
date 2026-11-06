"""Pure Telegram-HTML renderers for MONSTER AI messages (no I/O, no clock, no network).

Guarantees, all covered by tests: every dynamic value is HTML-escaped; links must be http(s); missing or
non-finite numbers render as ``n/a`` and are never invented; only calibrated probabilities appear, as rounded
percentages (and only a calibrated one gets a probability bar); every returned message is at most 3900 characters
with balanced tags. Emoji only decorate: the words next to them carry every fact (see ``notifications.style``).
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from itertools import accumulate
from typing import Any
from urllib.parse import urlsplit

from app.clock import IST, parse_ts
from notifications import style
from notifications.telegram import MAX_MESSAGE_CHARS, split_message
from ranking.types import EveningSummary, ReportContext

NA = "n/a"
DISCLAIMER = (
    "Probabilistic research output; not investment advice and not a guarantee. "
    "A wide intraday range does not imply a profitable trade or a directional close."
)
SHADOW_BANNER = "SHADOW MODE: no demonstrated edge over simple baselines yet - tiers are capped at WATCH."
PRELIM_NOTICE = (
    "PRELIMINARY: based on the 08:45 scan, before pre-open price discovery. This list may change materially and must not be "
    "treated as confirmed."
)
NO_SETUP_TITLE = "NO HIGH-CONVICTION SETUP today"
WEAK_LABEL = "WEAK / WATCHLIST ONLY"
HEURISTIC_LABEL = "heuristic, NOT calibrated"
EVENING_NOTE = "A single day is noise; judge the model only on the rolling out-of-sample record."
PROVISIONAL_LABEL = "PROVISIONAL: vendor data, official file pending"
VALIDATED_LABEL = "VALIDATED: official exchange data"

_MODEL_BADGES = {"validated": ("✅", "VALIDATED"), "baseline": ("📏", "BASELINE (no trained model)")}
_SHADOW_BADGE = ("🧪", "SHADOW - unvalidated")
_STAGE_LABELS = {"pre": "PRELIMINARY", "final": "FINAL", "closing": "CLOSING-HOUR IDEAS", "opening": "09:30 OPENING READ"}
# what a failure notice says was not published, and what its absence must not be read as, per stage
_NOT_PUBLISHED = {"closing": ("No trade ideas were published for this date.", "there is nothing to trade"),
                  "opening": ("No opening read was published for this date.", "nothing stands out at the open")}
_NOT_PUBLISHED_DEFAULT = ("No candidate list was published for this date.", "there are no large-move setups")
_CAPPED_TIERS = frozenset({"STRONG", "MODERATE"})

_HIT_KEY = re.compile(r"^hit_(\d+)(?:[._](\d+))?$")
_CONTROL = re.compile("[\x00-\x08\x0e-\x1b\x7f-\x84\x86-\x9f\u202a-\u202e\u2066-\u2069\ud800-\udfff]")
_URL_BAD = re.compile("[\\s\x00-\x1f\x7f<>\"'`\\\\]")
_IST_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*IST\b")
_UNIT_NAME = re.compile(r"[A-Za-z0-9@:._-]{1,80}")

_SYMBOL, _NAME, _SECTOR = 24, 60, 40
_HEADLINE, _SOURCE, _CHAIN, _ITEM, _NOTE = 140, 40, 200, 120, 200
_MAX_CATALYSTS, _MAX_ITEMS, _MAX_PAIRS, _MAX_ROWS = 2, 3, 10, 20
_URL_MAX = 300
_PART_RESERVE = 30


# ------------------------------------------------------------------------------------------ primitives


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=False)


def _clean(value: Any) -> str:
    """One line of text: invisible/bidi/control characters dropped, whitespace runs collapsed."""
    if value is None:
        return ""
    return " ".join(_CONTROL.sub("", str(value)).split())


def _clip(value: Any, limit: int) -> str:
    text = _clean(value)
    return text if len(text) <= limit else text[: max(1, limit - 3)].rstrip() + "..."


def _txt(value: Any, limit: int, empty: str = NA) -> str:
    text = _clip(value, limit)
    return _esc(text) if text else empty


def _fit(value: Any, limit: int) -> str:
    """Escaped text whose *escaped* length never exceeds ``limit`` (never cut inside an entity)."""
    raw = _clean(value)
    escaped = _esc(raw)
    if len(escaped) <= limit:
        return escaped
    sizes = accumulate(len(_esc(char)) for char in raw)
    keep = next(index for index, size in enumerate(sizes) if size > limit - 3)
    return _esc(raw[:keep]).rstrip() + "..."


def _fit_tail(value: Any, limit: int) -> str:
    """A multi-line log tail, escaped and capped at ``limit`` characters; when it is too long the *newest* lines are kept."""
    raw = _CONTROL.sub("", "" if value is None else str(value)).replace("\r\n", "\n").strip("\n")
    escaped = _esc(raw)
    if len(escaped) <= limit:
        return escaped
    total, start = 0, len(raw)
    for index in range(len(raw) - 1, -1, -1):
        total += len(_esc(raw[index]))
        if total > limit - 3:
            break
        start = index
    return "..." + _esc(raw[start:]).lstrip("\n")


def _pct(value: Any) -> str:
    """Calibrated probability 0..1 as a rounded percentage; anything outside [0, 1] is not a probability."""
    x = _num(value)
    if x is None or x < -1e-9 or x > 1 + 1e-9:
        return NA
    return f"{int(math.floor(min(max(x, 0.0), 1.0) * 100 + 0.5))}%"


def _n1(x: float) -> str:
    return f"{0.0 if round(x, 1) == 0 else x:.1f}"


def _pp(value: Any, *, signed: bool = False, digits: int = 1) -> str:
    x = _num(value)
    if x is None:
        return NA
    if round(x, digits) == 0:
        x = 0.0
    return f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%"


def _thr(value: Any) -> str:
    x = _num(value)
    return NA if x is None else f"{x:g}"


def _safe_url(value: Any) -> str | None:
    """Only plain http(s) links with a host survive; javascript:, data:, relative URLs etc. are dropped."""
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url or len(url) > _URL_MAX or _URL_BAD.search(url):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        return None
    return url


def _date_pretty(value: Any) -> str:
    """``Mon 02 Nov 2026`` from an ISO date; text that is not a date is shown as it is (escaped, at most 10 characters)."""
    text = _clean(value)[:10]
    try:
        day = date.fromisoformat(text)
    except ValueError:
        return _esc(text) or NA
    return style.pretty_date(day)


def _time_label(value: Any) -> str:
    """``🕘 09:12 IST`` from a display string such as ``2026-11-02 09:12:00 IST``; any other text is shown as it is."""
    match = _IST_TIME.search(_clean(value))
    if match and int(match.group(1)) < 24:
        hour = int(match.group(1))
        return f"{style.clock(hour)} {hour:02d}:{match.group(2)} IST"
    return f"{style.NO_CLOCK} {_txt(value, 40)}"


def _ist_moment(value: Any) -> datetime | None:
    moment = parse_ts(value) if value else None
    return moment.astimezone(IST) if moment else None


def _when(value: Any) -> str:
    """``🕗 2026-11-02 08:30 IST`` (the clock face follows the hour), or an empty string when the time is unknown."""
    moment = _ist_moment(value)
    return f"{style.clock(moment.hour)} {moment:%Y-%m-%d %H:%M} IST" if moment else ""


def _seq(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _clean_items(value: Any, limit: int = _ITEM) -> list[str]:
    if value is None:
        return []
    sequence = list(value) if isinstance(value, (list, tuple)) else [value]
    return [_esc(text) for text in (_clip(item, limit) for item in sequence) if text]


def _items(value: Any, max_items: int = _MAX_ITEMS, limit: int = _ITEM) -> list[str]:
    return _clean_items(value, limit)[:max_items]


def _joined(value: Any, sep: str = "; ", empty: str = "none", max_items: int = _MAX_ITEMS) -> str:
    items = _items(value, max_items)
    return sep.join(items) if items else empty


def _stage_label(stage: Any) -> str:
    return _STAGE_LABELS.get(_clean(stage).lower()) or _esc(_clip(stage, 20).upper()) or NA


def _arrows(escaped: str) -> str:
    """``a -> b`` (already escaped, so ``-&gt;``) drawn as ``a → b``."""
    return escaped.replace("-&gt;", "→")


def _pack(blocks: Sequence[str]) -> list[str]:
    """Join blocks with blank lines into messages of at most ``MAX_MESSAGE_CHARS`` characters."""
    limit = MAX_MESSAGE_CHARS - _PART_RESERVE
    pieces: list[str] = []
    for block in blocks:
        pieces.extend(split_message(block, limit) if len(block) > limit else [block])
    messages: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > limit:
            messages.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    messages.append(current)
    if len(messages) > 1:
        total = len(messages)
        messages = [f"{text}\n<i>📄 part {index}/{total}</i>" for index, text in enumerate(messages, 1)]
    return messages


# ------------------------------------------------------------------------------------ report pieces


def _hit_items(p: Mapping[str, Any]) -> list[tuple[float, Any]]:
    items: list[tuple[float, Any]] = []
    for key, value in p.items():
        match = _HIT_KEY.match(str(key))
        if match:
            whole, fraction = match.groups()
            items.append((float(f"{whole}.{fraction}") if fraction else float(whole), value))
    return sorted(items, key=lambda item: item[0])


_SOURCE_CLASS_LABELS = {
    "exchange_filing": "exchange filing", "official": "official source", "company": "company statement",
    "media": "media report", "social": "social media", "unknown": "unverified source",
}


def _provenance(catalyst: Any) -> str:
    """How far a catalyst can be trusted, as one line (``🚩 RUMOUR - unconfirmed · social media · 2 independent sources``).
    Empty when the snapshot carries no provenance (older snapshots), so their messages do not grow."""
    parts = []
    rumour = _get(catalyst, "rumour") is True
    if rumour:
        parts.append("<b>RUMOUR - unconfirmed</b>")
    label = _SOURCE_CLASS_LABELS.get(_clip(_get(catalyst, "source_class"), 20))
    if label:
        parts.append(label)
    independent = _num(_get(catalyst, "n_independent"))
    if independent is not None and independent >= 2:
        parts.append(f"{int(independent)} independent sources")
    conflicts = _num(_get(catalyst, "n_conflicts"))
    if conflicts is not None and conflicts >= 1:
        parts.append(f"{int(conflicts)} conflicting")
    return f"{'🚩' if rumour else '🔎'} " + " · ".join(parts) if parts else ""


def _catalyst_lines(label: str, catalyst: Any) -> list[str]:
    line = f"📰 <b>{label}:</b> {_txt(_get(catalyst, 'headline'), _HEADLINE)}"
    source = _clip(_get(catalyst, "source"), _SOURCE)
    if source:
        line += f" - {_esc(source)}"
    url = _safe_url(_get(catalyst, "url"))
    if url:
        line += f' (<a href="{html.escape(url, quote=True)}">link</a>)'
    trust = _provenance(catalyst)
    return [line, trust] if trust else [line]


def _tier_name(tier: Any, *, weak: bool, capped: bool) -> str:
    if weak:
        return WEAK_LABEL
    label = _clip(tier, 16).upper() or NA
    return "WATCH" if capped and label in _CAPPED_TIERS else label


def _candidate_block(cand: Any, position: int, *, weak: bool, capped: bool, heuristic: bool = False) -> str:
    p = _map(_get(cand, "p"))
    span = _map(_get(cand, "range"))
    rank = _num(_get(cand, "rank"))
    number = int(rank) if rank is not None and rank >= 1 else position
    symbol = _esc(_clip(_get(cand, "symbol"), _SYMBOL) or "UNKNOWN")
    tier = _tier_name(_get(cand, "tier"), weak=weak, capped=capped)
    lead = style.WEAK_RANK if weak else style.rank_badge(number)
    badge = style.WEAK_BADGE if weak else style.tier_badge(tier)
    lines = [f"{lead} <b>#{number} {symbol}</b> · {badge} <b>{_esc(tier)}</b>"]
    meta = [_txt(_get(cand, "name"), _NAME, ""), _txt(_get(cand, "sector"), _SECTOR, "")]
    if _get(cand, "is_fno"):
        meta.append(_esc("F&O"))
    if any(meta):
        lines.append("<i>" + " · ".join(part for part in meta if part) + "</i>")

    primary = _num(_get(cand, "primary_threshold"))
    hits = _hit_items(p)
    primary_p = next((v for t, v in hits if primary is not None and abs(t - primary) < 1e-9), None)
    threshold = f"{_thr(primary)}%" if primary is not None else NA
    basis = HEURISTIC_LABEL if heuristic else "calibrated"
    lines.append(f"📊 P(range &gt; {threshold}): <b>{_pct(primary_p)}</b> ({basis})")
    drawn = "" if heuristic else style.bar(_num(primary_p))  # a bar only ever shows a calibrated probability
    if drawn:
        lines.append(f"<code>{drawn}</code>")
    if hits:
        lines.append("🔢 " + " · ".join(f"{_thr(t)}%: {_pct(v)}" for t, v in hits))

    q10, q90 = _num(span.get("q10")), _num(span.get("q90"))
    interval = f"{_n1(q10)}-{_n1(q90)}%" if q10 is not None and q90 is not None else NA
    lines += [
        f"📏 Expected intraday range: <b>{_pp(span.get('q50'))}</b> · 80% interval {interval}",
        f"📐 Max excursion from open (q50 / q90): {_pp(span.get('exc_q50'))} / {_pp(span.get('exc_q90'))}",
        f"✨ P(unusual for this stock): {_pct(p.get('unusual'))}",
        f"🧭 Direction (secondary, not a trade signal): P(up close) {_pct(p.get('up'))} · "
        f"P(down close) {_pct(p.get('down'))}",
        "",
    ]

    catalysts = [c for c in _seq(_get(cand, "catalysts")) if _clip(_get(c, "headline"), 1)][:_MAX_CATALYSTS]
    if catalysts:
        lines += _catalyst_lines("Main catalyst", catalysts[0])
        if len(catalysts) > 1:
            lines += _catalyst_lines("Also", catalysts[1])
    else:
        lines.append("📰 <b>Main catalyst:</b> none identified")
    chain = _items(_get(cand, "chain"), 1, _CHAIN)
    lines += [
        f"🔗 <b>Global-to-India chain:</b> {_arrows(chain[0]) if chain else NA}",
        f"✅ <b>Confirmations:</b> {_joined(_get(cand, 'confirmations'))}",
        f"{style.WARN} <b>Contradictions:</b> {_joined(_get(cand, 'contradictions'))}",
        f"💸 <b>Priced-in assessment:</b> {_txt(_get(cand, 'priced_in'), _NOTE, 'not assessed')}",
        _liquidity_line(cand),
        f"🧩 <b>Model drivers</b> (diagnostic, not causal): {_joined(_get(cand, 'drivers'), empty=NA)}",
    ]
    return "\n".join(lines)


def _liquidity_line(cand: Any) -> str:
    quality = _clean(_get(cand, "data_quality", "ok")).lower()
    clean = bool(_get(cand, "liquidity_ok", True)) and quality == "ok"
    note = _clip(_get(cand, "liquidity_note"), _ITEM)
    turnover = _num(_get(cand, "turnover_cr"))
    cost = _num(_get(cand, "est_cost_pct"))
    status = ("✅ pass" if clean else f"{style.WARN} warning") + (f" - {_esc(note)}" if note else "")
    parts = [
        status,
        f"turnover ₹{turnover:,.1f} cr" if turnover is not None else "turnover n/a",
        f"est. round-trip cost {cost:.2f}%" if cost is not None else "est. round-trip cost n/a",
    ]
    return "💧 <b>Liquidity and data quality:</b> " + " · ".join(parts)


def _regime_line(value: Any) -> str:
    mapping = _map(value)
    shown = []
    for key, val in list(mapping.items())[:_MAX_PAIRS]:
        icon, label = style.REGIME.get(_clean(key), (style.BULLET, _clip(key, 24)))
        shown.append(f"{icon} {_esc(label)}: {_esc(_clip(val, 48))}")
    if len(mapping) > _MAX_PAIRS:
        shown.append(f"+{len(mapping) - _MAX_PAIRS} more")
    return "🌍 <b>Market regime:</b> " + (" · ".join(shown) if shown else NA)


def _freshness_lines(value: Any) -> list[str]:
    mapping = _map(value)
    if not mapping:
        return [f"📡 <b>Data freshness:</b> {NA}"]
    rows = [
        f"{style.status_dot(val)} {_esc(_clip(key, 24))}: {_esc(_clip(val, 48))}"
        for key, val in list(mapping.items())[:_MAX_PAIRS]
    ]
    if len(mapping) > _MAX_PAIRS:
        rows.append(f"+{len(mapping) - _MAX_PAIRS} more")
    return ["📡 <b>Data freshness</b>", *rows]


def _footer(ctx: Any, candidates: Sequence[Any]) -> str:
    note = _clip(_get(ctx, "uncertainty_note"), _NOTE)
    values = [v for v in (_num(_get(c, "uncertainty")) for c in candidates) if v is not None]
    pieces = [_esc(note)] if note else []
    if values:
        pieces.append(f"max candidate uncertainty {max(values):.2f} (0-1, higher = less reliable)")
    eligible, universe = _num(_get(ctx, "n_eligible")), _num(_get(ctx, "universe_size"))
    if universe is not None and universe > 0 and eligible is not None and eligible >= 0:
        share = int(math.floor(eligible / universe * 100 + 0.5))
        coverage = f"🔬 <b>Universe:</b> {int(eligible)} of {int(universe)} passed liquidity/data gates ({share}%)"
    else:
        coverage = "🔬 <b>Universe:</b> n/a of n/a passed liquidity/data gates"
    missing = _joined(_get(ctx, "missing_feeds"), sep=", ", max_items=_MAX_PAIRS)
    lines = [
        style.RULE,
        _regime_line(_get(ctx, "regime")),
        *_freshness_lines(_get(ctx, "freshness")),
        f"{'✅' if missing == 'none' else '🚧'} <b>Missing feeds:</b> {missing}",
        f"❓ <b>Model uncertainty:</b> {' · '.join(pieces) if pieces else NA}",
    ]
    notes = _items(_get(ctx, "notes"), 5, _NOTE)
    if notes:
        lines.append(f"📝 <b>Notes:</b> {'; '.join(notes)}")
    lines += [coverage, "", f"{style.WARN} <i>{DISCLAIMER}</i>"]
    return "\n".join(lines)


def _glance_table(ranked: Sequence[Any], *, capped: bool, heuristic: bool) -> str:
    """The candidates at a glance as a monospace table, at most 39 characters wide so a phone needs no sideways scrolling. A
    probability appears only for a model that is not the heuristic baseline."""
    primary = next((t for t in (_num(_get(c, "primary_threshold")) for c in ranked) if t is not None), None)
    head = f"P(>{_thr(primary)}%)" if primary is not None else "P"
    lines = [f"{'#':>2} {'SYMBOL':<10} {'TIER':<8} {head:>7} {'RANGE %':>8}"]
    for index, cand in enumerate(ranked[:_MAX_ROWS], 1):
        rank = _num(_get(cand, "rank"))
        threshold = _num(_get(cand, "primary_threshold"))
        p = _map(_get(cand, "p"))
        primary_p = next((v for t, v in _hit_items(p) if threshold is not None and abs(t - threshold) < 1e-9), None)
        span = _map(_get(cand, "range"))
        q10, q90 = _num(span.get("q10")), _num(span.get("q90"))
        interval = f"{_n1(q10)}-{_n1(q90)}" if q10 is not None and q90 is not None else NA
        lines.append(
            f"{int(rank) if rank is not None and rank >= 1 else index:>2} {_clip(_get(cand, 'symbol'), 10) or 'UNKNOWN':<10} "
            f"{_tier_name(_get(cand, 'tier'), weak=False, capped=capped)[:8]:<8} {NA if heuristic else _pct(primary_p):>7} {interval:>8}"
        )
    title = "📋 <b>At a glance</b> (RANGE: 80% interval of the expected high-low range, % of the open"
    title += "; heuristic ranking, no probability)" if heuristic else ")"
    return title + "\n<pre>" + "\n".join(_esc(line) for line in lines) + "</pre>"


def _render_report(ctx: ReportContext, *, preliminary: bool) -> list[str]:
    version = _clip(_get(ctx, "model_version"), 60)
    if not version:
        raise ValueError("model_version is required: illustrative output must never be sent as live")
    status = _clean(_get(ctx, "model_status")).lower()
    validated = status == "validated"
    badge, status_label = _MODEL_BADGES.get(status, _SHADOW_BADGE)
    stage = "⏳ <b>PRELIMINARY</b> (08:45 scan)" if preliminary else "🎯 <b>FINAL</b>"
    head = [
        "🚀 <b>MONSTER AI</b> · NSE large-move scan",
        f"{stage} · 📅 {_date_pretty(_get(ctx, 'trade_date'))} · {_time_label(_get(ctx, 'snapshot_ist'))}",
        f"🧠 Model <code>{_esc(version)}</code> · {badge} <b>{status_label}</b>",
    ]
    if preliminary:
        head.append(f"⏳ <b>{PRELIM_NOTICE}</b>")
    if not validated:
        head.append(f"🧪 <b>{SHADOW_BANNER}</b>")

    ranked = sorted(_seq(_get(ctx, "candidates")), key=lambda c: _num(_get(c, "rank")) or math.inf)
    weak = bool(_get(ctx, "no_setup")) or not ranked
    head.append(style.RULE)
    if not weak:
        primary = _num(_get(ranked[0], "primary_threshold"))
        target = f"exceeds {_thr(primary)}% of the open" if primary is not None else "is unusually large"
        head.append(f"🎯 <b>Top {len(ranked)}</b> · chance that the day's range (high - low) {target}")
    blocks = ["\n".join(head)]
    if weak:
        stated = "not stated by the ranking stage"
        fallback = stated if _get(ctx, "no_setup") else "no candidate cleared the ranking gates"
        text = f"🛑 <b>{NO_SETUP_TITLE}</b>\n📝 <b>Reason:</b> {_txt(_get(ctx, 'no_setup_reason'), 300, fallback)}"
        if ranked:
            text += f"\n🔸 <b>{WEAK_LABEL}</b> - shown for transparency only; these are not signals."
        blocks.append(text)
    elif ranked:
        blocks.append(_glance_table(ranked, capped=not validated, heuristic=status == "baseline"))
    blocks += [
        _candidate_block(cand, index, weak=weak, capped=not validated, heuristic=status == "baseline")
        for index, cand in enumerate(ranked, 1)
    ]
    blocks.append(_footer(ctx, ranked))
    return _pack(blocks)


def render_final_report(ctx: ReportContext) -> list[str]:
    """FINAL report. Raises ``ValueError`` when ``ctx.model_version`` is empty."""
    return _render_report(ctx, preliminary=False)


def render_prelim_report(ctx: ReportContext) -> list[str]:
    """Same layout as the final report, titled PRELIMINARY and making no claim of finality."""
    return _render_report(ctx, preliminary=True)


# ------------------------------------------------------------------------------------ evening report


def _rolling_value(name: str, value: Any) -> str:
    lowered = name.lower()
    if lowered in ("n", "count") or lowered.startswith("n_") or lowered.endswith("_days"):
        x = _num(value)
        return NA if x is None else str(int(x))
    if "hit" in lowered or "rate" in lowered:
        return _pct(value)
    x = _num(value)
    return NA if x is None else f"{x:.3f}"


def _table_row(rank: Any, symbol: Any, tier: Any, p: Any, rng: Any, ret: Any, hit: Any, ph: str) -> str:
    return f"{rank:>2} {symbol:<10} {tier:<8} {p:>{max(5, len(ph))}} {rng:>6} {ret:>7} {hit}"


def _hit_mark(hit: Any) -> str:
    return "➖" if hit is None else ("✅" if hit else "❌")


_RECORD_TITLE = "Live record: frozen picks against realised outcomes (95% day-block intervals)"
_RECORD_EMPTY = "Live record: n/a (no evaluated picks yet)"
_WINDOW = 12
_MAX_BASELINES = 8


def _dec2(value: Any) -> str:
    x = _num(value)
    return NA if x is None else f"{x:.2f}"


def _signed2(value: Any) -> str:
    x = _num(value)
    return NA if x is None else f"{x:+.2f}"


_RECORD_SERIES: tuple[tuple[str, str, Callable[[Any], str]], ...] = (
    ("Hit rate", "topn_hit_rate", _pct),
    ("Range MAE of the median forecast (pp)", "range_mae_q50", _dec2),
    ("80% interval coverage (nominal 80%)", "range_coverage_80", _pct),
    ("Direction accuracy (non-flat closes)", "direction_accuracy", _pct),
)


def _window_order(label: str) -> tuple[int, int, str]:
    days = label[:-1] if label.endswith("d") else ""
    return (0, int(days), label) if days.isdigit() else (1, 0, label)


def _record_windows(record: Any) -> list[tuple[str, dict[str, Any]]]:
    """(window label, metric name -> its all-picks row) of a live record that may be malformed; shortest window first.
    Slice rows are for the database and the dashboard, not for the report."""
    windows = []
    for label, rows in _map(record).items():
        named: dict[str, Any] = {}
        for item in _seq(rows):
            name = _clean(_get(item, "name"))
            if name and _clean(_get(item, "slice", "all")) == "all":
                named[name] = item
        if named:
            windows.append((_clip(label, _WINDOW), named))
    return sorted(windows, key=lambda pair: _window_order(pair[0]))


def _stat(row: Any, fmt: Callable[[Any], str], unit: str = "") -> str:
    """``value [low to high] n=N``: the number, its interval and its sample size (each only when known)."""
    value, low, high, n = (_num(_get(row, key)) for key in ("value", "ci_low", "ci_high", "n"))
    text = NA if value is None else fmt(value)
    if low is not None and high is not None:
        text += f" [{fmt(low)} to {fmt(high)}]"
    if n is not None and n >= 0:
        count = int(n)
        text += f" n={count}{unit.removesuffix('s') if count == 1 else unit}"  # "n=1 day", "n=5 days"
    return text


def _latest_window(windows: list[tuple[str, dict[str, Any]]], *names: str) -> tuple[str, dict[str, Any]] | None:
    return next(((label, rows) for label, rows in reversed(windows) if any(name in rows for name in names)), None)


def _realised_line(windows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    found = _latest_window(windows, "pick_range_mean")
    if found is None:
        return []
    label, rows = found
    parts = [f"picks mean {_pp(_get(rows['pick_range_mean'], 'value'))} / median {_pp(_get(rows.get('pick_range_median'), 'value'))}"]
    prefix, suffix = "baseline_", "_range_mean"
    names = sorted(n[len(prefix):-len(suffix)] for n in rows if n.startswith(prefix) and n.endswith(suffix))
    for name in names[:_MAX_BASELINES]:
        mean = _pp(_get(rows[f"{prefix}{name}{suffix}"], "value"))
        median = _pp(_get(rows.get(f"{prefix}{name}_range_median"), "value"))
        parts.append(f"{_esc(_clip(name, 24))} {mean} / {median}")
    return [f"{style.BULLET} Realised range, picks vs baselines ({_esc(label)}): " + " | ".join(parts)]


def _ranking_line(windows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    found = _latest_window(windows, "ndcg_at_5", "rank_ic")
    if found is None:
        return []
    label, rows = found
    parts = []
    if "ndcg_at_5" in rows:
        text = f"NDCG@5 {_stat(rows['ndcg_at_5'], _dec2, ' days')}"
        random = _map(_get(rows["ndcg_at_5"], "extra")).get("random")
        parts.append(text if _num(random) is None else f"{text} vs random {_dec2(random)}")
    if "rank_ic" in rows:
        parts.append(f"rank IC {_stat(rows['rank_ic'], _signed2, ' days')}")
    return [f"{style.BULLET} Ranking quality ({_esc(label)}): " + " | ".join(parts)]


def _record_lines(record: Any) -> list[str]:
    """The rolling live record: range-forecast error, interval coverage, direction accuracy, realised range versus the
    baselines and ranking quality, every number with its sample size and interval (blueprint acceptance 23.4)."""
    windows = _record_windows(record)
    if not windows:
        return [f"📚 {_RECORD_EMPTY}"]
    out = [f"📚 <b>{_RECORD_TITLE}</b>"]
    for title, name, fmt in _RECORD_SERIES:
        pieces = [f"{_esc(label)} {_stat(rows[name], fmt)}" for label, rows in windows if name in rows]
        if pieces:
            out.append(f"{style.BULLET} {title}: " + " | ".join(pieces))
    return out + _realised_line(windows) + _ranking_line(windows)


def render_evening_report(summary: EveningSummary) -> list[str]:
    """Post-session evaluation. Raises ``ValueError`` when ``summary.model_version`` is empty."""
    version = _clip(_get(summary, "model_version"), 60)
    if not version:
        raise ValueError("model_version is required: illustrative output must never be sent as live")
    threshold = _thr(_get(summary, "primary_threshold"))
    provisional = bool(_get(summary, "provisional", True))
    rows = sorted(_seq(_get(summary, "rows")), key=lambda r: _num(_get(r, "rank")) or math.inf)
    head = [
        "🌆 <b>MONSTER AI</b> · Evening evaluation",
        f"🎯 <b>{_stage_label(_get(summary, 'stage'))}</b> · 📅 {_date_pretty(_get(summary, 'trade_date'))}",
        f"🧠 Model <code>{_esc(version)}</code>",
        f"📏 Target: intraday range &gt; {threshold}% of the open",
        f"{'🟡' if provisional else '🟢'} <b>{PROVISIONAL_LABEL if provisional else VALIDATED_LABEL}</b>",
        style.RULE,
    ]
    blocks = ["\n".join(head)]

    if rows:
        ph = f"P({threshold}%)"
        lines = [_table_row("#", "SYMBOL", "TIER", ph, "RANGE", "DIR", "HIT", ph)]
        for index, row in enumerate(rows[:_MAX_ROWS], 1):
            rank = _num(_get(row, "rank"))
            lines.append(
                _table_row(
                    int(rank) if rank is not None else index,
                    _clip(_get(row, "symbol"), 10) or "UNKNOWN",
                    _clip(_get(row, "tier"), 8).upper() or NA,
                    _pct(_get(row, "p_primary")),
                    _pp(_get(row, "range_pct")),
                    _pp(_get(row, "dir_ret_pct"), signed=True),
                    _hit_mark(_get(row, "hit_primary")),
                    ph,
                )
            )
        if len(rows) > _MAX_ROWS:
            lines.append(f"... +{len(rows) - _MAX_ROWS} more")
        blocks.append("📋 <b>Picks against what happened</b> (HIT: ✅ range above the threshold, ❌ not, ➖ unknown)\n"
                      "<pre>" + "\n".join(_esc(line) for line in lines) + "</pre>")
    else:
        blocks.append("📋 No evaluated picks for this date.")

    total = int(_num(_get(summary, "n")) or 0)
    rate = _num(_get(summary, "model_hit_rate"))
    if rate is not None and not 0.0 <= rate <= 1.0:
        rate = None
    size = len(rows) or total
    label = f"Top-{size} hit rate" if size else "Top-N hit rate"
    if rate is None or total <= 0:
        hit_text = f"{NA} (no evaluated picks)"
    else:
        interval = _get(summary, "model_hit_ci")
        pair = isinstance(interval, (list, tuple)) and len(interval) == 2
        low, high = (_num(interval[0]), _num(interval[1])) if pair else (None, None)
        ci = f" (CI {_pct(low)}-{_pct(high)})" if low is not None and high is not None else ""
        hit_text = f"{int(math.floor(rate * total + 0.5))}/{total}{ci}"
    stats = [f"🏁 <b>{label}:</b> {hit_text}"]
    baselines = sorted(_map(_get(summary, "baseline_hit_rates")).items(), key=lambda kv: str(kv[0]))
    stats.append(
        "🆚 <b>Baseline hit rates:</b> " + (", ".join(f"{_esc(_clip(k, 24))} {_pct(v)}" for k, v in baselines) or NA)
    )
    base_ranges = sorted(_map(_get(summary, "baseline_mean_range_pct")).items(), key=lambda kv: str(kv[0]))
    mean_range = f"📐 <b>Mean actual range:</b> model picks {_pp(_get(summary, 'mean_range_pct'))}"
    if base_ranges:
        mean_range += " | baselines: " + ", ".join(f"{_esc(_clip(k, 24))} {_pp(v)}" for k, v in base_ranges)
    stats.append(mean_range)
    rolling = list(_map(_get(summary, "rolling")).items())[:_MAX_PAIRS]
    stats.append(
        "📈 <b>Rolling record (out-of-sample):</b> "
        + (", ".join(f"{_esc(_clip(k, 24))} {_rolling_value(str(k), v)}" for k, v in rolling) or NA)
    )
    calibration = _clip(_get(summary, "calibration_note"), _NOTE)
    if calibration:
        stats.append(f"🧮 <b>Calibration:</b> {_esc(calibration)}")
    stats.extend(_record_lines(_get(summary, "live_record")))
    quality = _items(_get(summary, "data_quality_notes"), 5, _NOTE)
    if quality:
        stats.append(f"🧹 <b>Data-quality notes:</b> {'; '.join(quality)}")
    missing_all = _clean_items(_get(summary, "missing_symbols"), 20)
    if missing_all:
        extra = len(missing_all) - 15
        more = f" (+{extra} more)" if extra > 0 else ""
        stats.append(f"🚫 <b>Missing symbols:</b> {', '.join(missing_all[:15])}{more}")
    stats += ["", f"💡 <i>{EVENING_NOTE}</i>"]
    blocks.append("\n".join(stats))
    return _pack(blocks)


# ---------------------------------------------------------------------------- notices and alerts


def _details_text(details: Any) -> str:
    if details is None:
        return ""
    if isinstance(details, Mapping):
        return "; ".join(f"{_clean(k)}={_clean(v)}" for k, v in details.items())
    if isinstance(details, (list, tuple)):
        return "; ".join(_clean(item) for item in details if _clean(item))
    return _clean(details)


def render_failure_notice(trade_date: Any, stage: Any, reason: Any, details: Any = None) -> list[str]:
    """Explicit 'REPORT WITHHELD' notice so a missed report is never silent."""
    lines = [
        "🛑 <b>MONSTER AI</b> · REPORT WITHHELD",
        f"🎯 <b>{_stage_label(stage)}</b> · 📅 {_date_pretty(trade_date)}",
        f"❗ <b>Reason:</b> {_fit(reason, 300) or NA}",
    ]
    detail_text = _details_text(details)
    if detail_text:
        lines.append(f"🔎 <b>Details:</b> {_fit(detail_text, 700)}")
    published, nothing = _NOT_PUBLISHED.get(_clean(stage).lower(), _NOT_PUBLISHED_DEFAULT)
    lines += [
        style.RULE,
        f"💡 <b>{published}</b> This is a health notice, not a market signal: "
        f"it does not mean {nothing}, only that the scan could not be completed safely "
        "and on time.",
        f"🛠 <b>Next step:</b> <code>{style.scan_hint(stage)}</code>",
    ]
    return _pack(["\n".join(lines)])


def _level(value: Any) -> str:
    return _esc(_clip(value, 12).upper() or "INFO")


def render_alert(level: str, code: str, message: str, *, created_at: Any = None) -> str:
    """One operational alert as a single message (always within the Telegram limit)."""
    lines = [
        f"{style.severity_icon(level, code)} <b>{_level(level)}</b> · <code>{_txt(code, 60)}</code>",
        _fit(message, 3000) or NA,
    ]
    hint = style.alert_hint(code)
    if hint and "monster-ai" not in str(message) and "app.cli" not in str(message):
        lines.append(f"🛠 <b>Next step:</b> <code>{_esc(hint)}</code>")
    when = _when(created_at)
    if when:
        lines.append(f"<i>{when}</i>")
    return "\n".join(lines)


def render_alert_digest(alerts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Several queued alerts batched into one message (split only if the batch is unusually large)."""
    lines = [f"🔔 <b>MONSTER AI</b> · {len(alerts)} alerts", style.RULE]
    for alert in alerts:
        when = _when(alert.get("created_at"))
        stamp = f" · {when}" if when else ""
        icon = style.severity_icon(alert.get("level"), alert.get("code"))
        lines.append(f"{icon} <b>{_level(alert.get('level'))}</b> · <code>{_txt(alert.get('code'), 60)}</code>{stamp}")
        lines.append(_fit(alert.get("message"), 300) or NA)
    lines.append("🛠 <b>Next step:</b> <code>sudo monster-ai health</code> · <code>sudo monster-ai logs</code>")
    return _pack(["\n".join(lines)])


def render_unit_failure(unit: Any, tail: Any = "", when: Any = None) -> str:
    """The alert for a systemd unit that failed: its name, when, and the last journal lines (newest kept when long)."""
    lines = [f"🚨 <b>Service failed</b> · <code>{_txt(unit, 80)}</code>"]
    stamp = _when(when)
    if stamp:
        lines.append(f"<i>{stamp}</i>")
    lines += ["📜 <b>Last journal lines</b>", f"<pre>{_fit_tail(tail, 2400) or 'no log lines'}</pre>"]
    name = _clean(unit)
    if _UNIT_NAME.fullmatch(name):
        lines.append(f"🛠 <b>Next step:</b> <code>journalctl -u {_esc(name)} -n 100 --no-pager</code>")
    return "\n".join(lines)


def render_test_message(settings: Any) -> str:
    """Connectivity test: mode plus a configuration summary that never contains secrets."""
    redacted = settings.redacted() if hasattr(settings, "redacted") else {}

    def cfg(name: str, default: Any = NA) -> Any:
        value = redacted.get(name) if redacted else getattr(settings, name, None)
        return default if value is None or value == "" else value

    def flag(value: Any) -> str:
        return "✅ on" if value else "⚪ off"

    def state(value: Any) -> str:
        return "✅ configured" if value else "⚪ off"

    thresholds = [f"{x:g}" for x in _seq(cfg("range_thresholds", ())) if _num(x) is not None]
    alert_chat = cfg("telegram_alert_chat_id", "")
    same_chat = not alert_chat or alert_chat == cfg("telegram_chat_id", "")
    ready = bool(cfg("telegram_bot_token", "")) and bool(cfg("telegram_chat_id", ""))
    lines = [
        "🤖 <b>MONSTER AI online</b> ✅",
        "Your Telegram connection works: reports and alerts will arrive in this chat.",
        "",
        f"🧭 <b>Mode:</b> {_txt(cfg('mode'), 20)} - research output only; the system never places orders",
        f"🌏 <b>Environment:</b> {_txt(cfg('env'), 20)} · <b>Timezone:</b> {_txt(cfg('timezone'), 40)}",
        style.RULE,
        "📬 <b>Delivery</b>",
        f"{style.BULLET} Report chat: {'✅ configured' if ready else '❌ NOT configured'}",
        f"{style.BULLET} Alerts go to: {'the same chat' if same_chat else 'a separate chat'}",
        f"{style.BULLET} Preliminary report: {flag(cfg('send_prelim_report', False))} · "
        f"evening report: {flag(cfg('send_evening_report', False))}",
        f"🎯 <b>Targets:</b> primary range threshold {_thr(cfg('primary_threshold', None))}% · "
        f"thresholds {_esc(', '.join(thresholds) or NA)}% · top-N {_txt(cfg('top_n'), 6)} · "
        f"universe {_txt(cfg('universe_mode'), 24)}",
        "📆 <b>Daily schedule (IST)</b>",
        f"{style.BULLET} {_txt(cfg('prelim_scan_time'), 10)} preliminary scan",
        f"{style.BULLET} {_txt(cfg('final_scan_time'), 10)} final scan",
        f"{style.BULLET} {_txt(cfg('report_cutoff_time'), 10)} report cutoff · "
        f"{_txt(cfg('report_deadline_time'), 10)} delivery deadline",
        f"🔌 <b>Integrations:</b> LLM {state(getattr(settings, 'llm_configured', False))} · "
        f"web check {state(getattr(settings, 'llm_web_configured', False))} · "
        f"Jev {state(getattr(settings, 'jev_configured', False))} · "
        f"off-box backup {state(cfg('s3_bucket', ''))} · dashboard {flag(cfg('dashboard_enabled', False))} · "
        f"training {_txt(cfg('training_mode'), 20)}",
        "",
        "💡 <i>Connectivity test only; no market data or candidates are included.</i>",
    ]
    return "\n".join(lines)


def render_subscriber_welcome(settings: Any) -> str:
    """The one note an extra recipient gets when the operator checks the delivery (``telegram-test --all``): what will arrive here
    and that it is research, never advice or orders. Nothing about the system's configuration, which is the operator's."""
    def on(name: str, default: bool = True) -> bool:
        return bool(getattr(settings, name, default))

    items = []
    if on("send_prelim_report", False):
        items.append("the preliminary scan (about 08:45 IST)")
    items.append("the morning large-move scan (about 09:12 IST)")
    if on("ideas_enabled") and on("send_ideas_report"):
        items.append("the closing-hour trade ideas (before 15:05 IST)")
    if on("send_evening_report"):
        items.append("the evening review")
    lines = [
        "✅ <b>MONSTER AI</b> · you are subscribed",
        "You will receive the same reports as the owner of this bot, here in this chat:",
        *[f"{style.BULLET} {item}" for item in items],
        "",
        "💡 <i>Research output only: not investment advice, and the system never places orders.</i>",
    ]
    return "\n".join(lines)
