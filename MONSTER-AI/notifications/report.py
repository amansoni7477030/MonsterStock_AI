"""Pure Telegram-HTML renderers for MONSTER AI messages (no I/O, no clock, no network).

Guarantees, all covered by tests: every dynamic value is HTML-escaped; links must be http(s); missing or
non-finite numbers render as ``n/a`` and are never invented; only calibrated probabilities appear, as rounded
percentages; every returned message is at most 3900 characters with balanced tags.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from itertools import accumulate
from typing import Any
from urllib.parse import urlsplit

from app.clock import IST, parse_ts
from notifications.telegram import MAX_MESSAGE_CHARS, split_message
from ranking.types import EveningSummary, ReportContext

NA = "n/a"
DISCLAIMER = (
    "Probabilistic research output; not investment advice and not a guarantee. "
    "A wide intraday range does not imply a profitable trade or a directional close."
)
SHADOW_BANNER = "SHADOW MODE: no demonstrated edge over simple baselines yet - tiers are capped at WATCH."
NO_SETUP_TITLE = "NO HIGH-CONVICTION SETUP today"
WEAK_LABEL = "WEAK / WATCHLIST ONLY"
HEURISTIC_LABEL = "heuristic, NOT calibrated"
EVENING_NOTE = "A single day is noise; judge the model only on the rolling out-of-sample record."
PROVISIONAL_LABEL = "PROVISIONAL: vendor data, official file pending"
VALIDATED_LABEL = "VALIDATED: official exchange data"

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MODEL_LABELS = {"validated": "VALIDATED", "baseline": "BASELINE (no trained model)"}
_SHADOW_LABEL = "SHADOW - unvalidated"
_STAGE_LABELS = {"pre": "PRELIMINARY", "final": "FINAL"}
_CAPPED_TIERS = frozenset({"STRONG", "MODERATE"})

_HIT_KEY = re.compile(r"^hit_(\d+)(?:[._](\d+))?$")
_CONTROL = re.compile("[\x00-\x08\x0e-\x1b\x7f-\x84\x86-\x9f\u202a-\u202e\u2066-\u2069\ud800-\udfff]")
_URL_BAD = re.compile("[\\s\x00-\x1f\x7f<>\"'`\\\\]")

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


def _date_label(value: Any) -> str:
    text = _clean(value)[:10]
    try:
        day = date.fromisoformat(text)
    except ValueError:
        return _esc(text) or NA
    return f"{day.isoformat()} ({_WEEKDAYS[day.weekday()]})"


def _ist_label(value: Any) -> str:
    moment = parse_ts(value) if value else None
    return moment.astimezone(IST).strftime("%Y-%m-%d %H:%M IST") if moment else ""


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


def _pairs(value: Any, *, label: str, sep: str = ", ") -> str:
    mapping = _map(value)
    shown = [f"{_clip(k, 24)}={_clip(v, 48)}" for k, v in list(mapping.items())[:_MAX_PAIRS]]
    if len(mapping) > _MAX_PAIRS:
        shown.append(f"+{len(mapping) - _MAX_PAIRS} more")
    return f"{label}: {_esc(sep.join(shown)) if shown else NA}"


def _stage_label(stage: Any) -> str:
    return _STAGE_LABELS.get(_clean(stage).lower()) or _esc(_clip(stage, 20).upper()) or NA


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
        messages = [f"{text}\n<i>(part {index}/{total})</i>" for index, text in enumerate(messages, 1)]
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


def _catalyst_line(label: str, catalyst: Any) -> str:
    line = f"{label}: {_txt(_get(catalyst, 'headline'), _HEADLINE)}"
    source = _clip(_get(catalyst, "source"), _SOURCE)
    if source:
        line += f" - {_esc(source)}"
    url = _safe_url(_get(catalyst, "url"))
    if url:
        line += f' (<a href="{html.escape(url, quote=True)}">link</a>)'
    return line


def _tier_label(tier: Any, *, weak: bool, capped: bool) -> str:
    if weak:
        return WEAK_LABEL
    label = _clip(tier, 16).upper() or NA
    return "WATCH" if capped and label in _CAPPED_TIERS else _esc(label)


def _candidate_block(cand: Any, position: int, *, weak: bool, capped: bool, heuristic: bool = False) -> str:
    p = _map(_get(cand, "p"))
    span = _map(_get(cand, "range"))
    rank = _num(_get(cand, "rank"))
    number = int(rank) if rank is not None and rank >= 1 else position
    symbol = _esc(_clip(_get(cand, "symbol"), _SYMBOL) or "UNKNOWN")
    meta = [_txt(_get(cand, "name"), _NAME, ""), _txt(_get(cand, "sector"), _SECTOR, "")]
    if _get(cand, "is_fno"):
        meta.append(_esc("F&O"))
    head = f"<b>#{number} {symbol} - {_tier_label(_get(cand, 'tier'), weak=weak, capped=capped)}</b>"
    if any(meta):
        head += "  " + " | ".join(part for part in meta if part)

    primary = _num(_get(cand, "primary_threshold"))
    hits = _hit_items(p)
    primary_p = next((v for t, v in hits if primary is not None and abs(t - primary) < 1e-9), None)
    threshold = f"{_thr(primary)}%" if primary is not None else NA
    basis = HEURISTIC_LABEL if heuristic else "calibrated"
    p_line = f"P(range &gt;= {threshold}): {_pct(primary_p)} ({basis})"
    if hits:
        p_line += "   [" + " | ".join(f"{_thr(t)}%: {_pct(v)}" for t, v in hits) + "]"

    q10, q90 = _num(span.get("q10")), _num(span.get("q90"))
    interval = f"{_n1(q10)}-{_n1(q90)}%" if q10 is not None and q90 is not None else NA
    lines = [
        head,
        p_line,
        f"P(unusual for this stock): {_pct(p.get('unusual'))}",
        f"Expected intraday range: {_pp(span.get('q50'))} (80% interval {interval})   "
        f"Max excursion from open (q50/q90): {_pp(span.get('exc_q50'))} / {_pp(span.get('exc_q90'))}",
        f"Direction (secondary, not a trade signal): P(up close) {_pct(p.get('up'))} / "
        f"P(down close) {_pct(p.get('down'))}",
    ]

    catalysts = [c for c in _seq(_get(cand, "catalysts")) if _clip(_get(c, "headline"), 1)][:_MAX_CATALYSTS]
    if catalysts:
        lines.append(_catalyst_line("Main catalyst", catalysts[0]))
        if len(catalysts) > 1:
            lines.append(_catalyst_line("Also", catalysts[1]))
    else:
        lines.append("Main catalyst: none identified")
    chain = _items(_get(cand, "chain"), 1, _CHAIN)
    lines += [
        f"Global-to-India chain: {chain[0] if chain else NA}",
        f"Confirmations: {_joined(_get(cand, 'confirmations'))}",
        f"Contradictions: {_joined(_get(cand, 'contradictions'))}",
        f"Priced-in assessment: {_txt(_get(cand, 'priced_in'), _NOTE, 'not assessed')}",
        _liquidity_line(cand),
        f"Model drivers (diagnostic, not causal): {_joined(_get(cand, 'drivers'), empty=NA)}",
    ]
    return "\n".join(lines)


def _liquidity_line(cand: Any) -> str:
    quality = _clean(_get(cand, "data_quality", "ok")).lower()
    clean = bool(_get(cand, "liquidity_ok", True)) and quality == "ok"
    note = _clip(_get(cand, "liquidity_note"), _ITEM)
    turnover = _num(_get(cand, "turnover_cr"))
    cost = _num(_get(cand, "est_cost_pct"))
    parts = [
        "pass" if clean else "warning",
        f"turnover Rs {turnover:,.1f} cr" if turnover is not None else "turnover n/a",
        f"est. round-trip cost {cost:.2f}%" if cost is not None else "est. round-trip cost n/a",
    ]
    status = parts[0] + (f" - {_esc(note)}" if note else "")
    return "Liquidity and data quality: " + " | ".join([status, *parts[1:]])


def _footer(ctx: Any, candidates: Sequence[Any]) -> str:
    note = _clip(_get(ctx, "uncertainty_note"), _NOTE)
    values = [v for v in (_num(_get(c, "uncertainty")) for c in candidates) if v is not None]
    pieces = [_esc(note)] if note else []
    if values:
        pieces.append(f"max candidate uncertainty {max(values):.2f} (0-1, higher = less reliable)")
    eligible, universe = _num(_get(ctx, "n_eligible")), _num(_get(ctx, "universe_size"))
    if universe is not None and universe > 0 and eligible is not None and eligible >= 0:
        coverage = f"Universe: {int(eligible)} of {int(universe)} passed liquidity/data gates."
    else:
        coverage = "Universe: n/a of n/a passed liquidity/data gates."
    lines = [
        _pairs(_get(ctx, "regime"), label="Market regime"),
        _pairs(_get(ctx, "freshness"), label="Data freshness"),
        f"Missing feeds: {_joined(_get(ctx, 'missing_feeds'), sep=', ', max_items=_MAX_PAIRS)}",
        f"Model uncertainty: {' | '.join(pieces) if pieces else NA}",
    ]
    notes = _items(_get(ctx, "notes"), 5, _NOTE)
    if notes:
        lines.append(f"Notes: {'; '.join(notes)}")
    lines += [coverage, f"<i>{DISCLAIMER}</i>"]
    return "\n".join(lines)


def _render_report(ctx: ReportContext, *, preliminary: bool) -> list[str]:
    version = _clip(_get(ctx, "model_version"), 60)
    if not version:
        raise ValueError("model_version is required: illustrative output must never be sent as live")
    status = _clean(_get(ctx, "model_status")).lower()
    validated = status == "validated"
    title = "MONSTER AI | NSE LARGE-MOVE SCAN" + (" - PRELIMINARY (08:45 scan)" if preliminary else "")
    stage = "PRELIMINARY" if preliminary else "FINAL"
    head = [
        f"<b>{title}</b>",
        f"Date: {_date_label(_get(ctx, 'trade_date'))} | Snapshot: {_txt(_get(ctx, 'snapshot_ist'), 40)} | "
        f"Stage: {stage} | Model: {_esc(version)} [{_MODEL_LABELS.get(status, _SHADOW_LABEL)}]",
    ]
    if preliminary:
        head.append(
            "<b>PRELIMINARY: based on the 08:45 scan, before pre-open price discovery. This list may change "
            "materially and must not be treated as confirmed.</b>"
        )
    if not validated:
        head.append(f"<b>{SHADOW_BANNER}</b>")

    ranked = sorted(_seq(_get(ctx, "candidates")), key=lambda c: _num(_get(c, "rank")) or math.inf)
    weak = bool(_get(ctx, "no_setup")) or not ranked
    blocks = ["\n".join(head)]
    if weak:
        stated = "not stated by the ranking stage"
        fallback = stated if _get(ctx, "no_setup") else "no candidate cleared the ranking gates"
        text = f"<b>{NO_SETUP_TITLE}</b>\nReason: {_txt(_get(ctx, 'no_setup_reason'), 300, fallback)}"
        if ranked:
            text += f"\n<b>{WEAK_LABEL}</b> - shown for transparency only; these are not signals."
        blocks.append(text)
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
    return f"{rank:>2} {symbol:<10} {tier:<8} {p:>{max(5, len(ph))}} {rng:>6} {ret:>7} {hit:>3}"


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
    return [f"Realised range, picks vs baselines ({_esc(label)}): " + " | ".join(parts)]


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
    return [f"Ranking quality ({_esc(label)}): " + " | ".join(parts)]


def _record_lines(record: Any) -> list[str]:
    """The rolling live record: range-forecast error, interval coverage, direction accuracy, realised range versus the
    baselines and ranking quality, every number with its sample size and interval (blueprint acceptance 23.4)."""
    windows = _record_windows(record)
    if not windows:
        return [_RECORD_EMPTY]
    out = [f"<b>{_RECORD_TITLE}</b>"]
    for title, name, fmt in _RECORD_SERIES:
        pieces = [f"{_esc(label)} {_stat(rows[name], fmt)}" for label, rows in windows if name in rows]
        if pieces:
            out.append(f"{title}: " + " | ".join(pieces))
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
        "<b>MONSTER AI | EVENING EVALUATION</b>",
        f"Date: {_date_label(_get(summary, 'trade_date'))} | Stage: {_stage_label(_get(summary, 'stage'))} | "
        f"Model: {_esc(version)} | Target: intraday range &gt;= {threshold}% of open",
        f"<b>{PROVISIONAL_LABEL if provisional else VALIDATED_LABEL}</b>",
    ]
    blocks = ["\n".join(head)]

    if rows:
        ph = f"P({threshold}%)"
        lines = [_table_row("#", "SYMBOL", "TIER", ph, "RANGE", "DIR", "HIT", ph)]
        for index, row in enumerate(rows[:_MAX_ROWS], 1):
            rank = _num(_get(row, "rank"))
            hit = _get(row, "hit_primary")
            lines.append(
                _table_row(
                    int(rank) if rank is not None else index,
                    _clip(_get(row, "symbol"), 10) or "UNKNOWN",
                    _clip(_get(row, "tier"), 8).upper() or NA,
                    _pct(_get(row, "p_primary")),
                    _pp(_get(row, "range_pct")),
                    _pp(_get(row, "dir_ret_pct"), signed=True),
                    NA if hit is None else ("yes" if hit else "no"),
                    ph,
                )
            )
        if len(rows) > _MAX_ROWS:
            lines.append(f"... +{len(rows) - _MAX_ROWS} more")
        blocks.append("<pre>" + "\n".join(_esc(line) for line in lines) + "</pre>")
    else:
        blocks.append("No evaluated picks for this date.")

    total = int(_num(_get(summary, "n")) or 0)
    rate = _num(_get(summary, "model_hit_rate"))
    if rate is not None and not 0.0 <= rate <= 1.0:
        rate = None
    size = len(rows) or total
    label = f"Top-{size} hit rate" if size else "Top-N hit rate"
    if rate is None or total <= 0:
        stats = [f"{label}: {NA} (no evaluated picks)"]
    else:
        interval = _get(summary, "model_hit_ci")
        pair = isinstance(interval, (list, tuple)) and len(interval) == 2
        low, high = (_num(interval[0]), _num(interval[1])) if pair else (None, None)
        ci = f" (CI {_pct(low)}-{_pct(high)})" if low is not None and high is not None else ""
        stats = [f"{label}: {int(math.floor(rate * total + 0.5))}/{total}{ci}"]
    baselines = sorted(_map(_get(summary, "baseline_hit_rates")).items(), key=lambda kv: str(kv[0]))
    stats.append(
        "Baseline hit rates: " + (", ".join(f"{_esc(_clip(k, 24))} {_pct(v)}" for k, v in baselines) or NA)
    )
    base_ranges = sorted(_map(_get(summary, "baseline_mean_range_pct")).items(), key=lambda kv: str(kv[0]))
    mean_range = f"Mean actual range: model picks {_pp(_get(summary, 'mean_range_pct'))}"
    if base_ranges:
        mean_range += " | baselines: " + ", ".join(f"{_esc(_clip(k, 24))} {_pp(v)}" for k, v in base_ranges)
    stats.append(mean_range)
    rolling = list(_map(_get(summary, "rolling")).items())[:_MAX_PAIRS]
    stats.append(
        "Rolling record (out-of-sample): "
        + (", ".join(f"{_esc(_clip(k, 24))} {_rolling_value(str(k), v)}" for k, v in rolling) or NA)
    )
    calibration = _clip(_get(summary, "calibration_note"), _NOTE)
    if calibration:
        stats.append(f"Calibration: {_esc(calibration)}")
    stats.extend(_record_lines(_get(summary, "live_record")))
    quality = _items(_get(summary, "data_quality_notes"), 5, _NOTE)
    if quality:
        stats.append(f"Data-quality notes: {'; '.join(quality)}")
    missing_all = _clean_items(_get(summary, "missing_symbols"), 20)
    if missing_all:
        extra = len(missing_all) - 15
        more = f" (+{extra} more)" if extra > 0 else ""
        stats.append(f"Missing symbols: {', '.join(missing_all[:15])}{more}")
    stats.append(f"<i>{EVENING_NOTE}</i>")
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
        "<b>MONSTER AI | REPORT WITHHELD</b>",
        f"Date: {_date_label(trade_date)} | Stage: {_stage_label(stage)}",
        f"Reason: {_fit(reason, 300) or NA}",
    ]
    detail_text = _details_text(details)
    if detail_text:
        lines.append(f"Details: {_fit(detail_text, 700)}")
    lines.append(
        "<b>No candidate list was published for this date.</b> This is a health notice, not a market signal: "
        "it does not mean there are no large-move setups, only that the scan could not be completed safely "
        "and on time."
    )
    return _pack(["\n".join(lines)])


def _level(value: Any) -> str:
    return _esc(_clip(value, 12).upper() or "INFO")


def render_alert(level: str, code: str, message: str, *, created_at: Any = None) -> str:
    """One operational alert as a single message (always within the Telegram limit)."""
    head = f"<b>[{_level(level)}] MONSTER AI alert</b> <code>{_txt(code, 60)}</code>"
    text = f"{head}\n{_fit(message, 3000) or NA}"
    when = _ist_label(created_at)
    return f"{text}\n<i>{when}</i>" if when else text


def render_alert_digest(alerts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Several queued alerts batched into one message (split only if the batch is unusually large)."""
    lines = [f"<b>MONSTER AI | {len(alerts)} ALERTS</b>"]
    for alert in alerts:
        when = _ist_label(alert.get("created_at"))
        stamp = f" ({when})" if when else ""
        code = _txt(alert.get("code"), 60)
        text = _fit(alert.get("message"), 300) or NA
        lines.append(f"[{_level(alert.get('level'))}] <code>{code}</code>{stamp}: {text}")
    return _pack(["\n".join(lines)])


def render_test_message(settings: Any) -> str:
    """Connectivity test: mode plus a configuration summary that never contains secrets."""
    redacted = settings.redacted() if hasattr(settings, "redacted") else {}

    def cfg(name: str, default: Any = NA) -> Any:
        value = redacted.get(name) if redacted else getattr(settings, name, None)
        return default if value is None or value == "" else value

    def flag(value: Any) -> str:
        return "on" if value else "off"

    thresholds = [f"{x:g}" for x in _seq(cfg("range_thresholds", ())) if _num(x) is not None]
    alert_chat = cfg("telegram_alert_chat_id", "")
    same_chat = not alert_chat or alert_chat == cfg("telegram_chat_id", "")
    ready = bool(cfg("telegram_bot_token", "")) and bool(cfg("telegram_chat_id", ""))
    lines = [
        "<b>MONSTER AI online</b>",
        f"Mode: {_txt(cfg('mode'), 20)} (research output only; the system never places orders)",
        f"Environment: {_txt(cfg('env'), 20)} | Timezone: {_txt(cfg('timezone'), 40)}",
        f"Telegram: report chat {'configured' if ready else 'NOT configured'} | "
        f"alerts to {'the same chat' if same_chat else 'a separate chat'} | "
        f"prelim report {flag(cfg('send_prelim_report', False))} | "
        f"evening report {flag(cfg('send_evening_report', False))}",
        f"Targets: primary range threshold {_thr(cfg('primary_threshold', None))}% | "
        f"thresholds {_esc(', '.join(thresholds) or NA)}% | top-N {_txt(cfg('top_n'), 6)} | "
        f"universe {_txt(cfg('universe_mode'), 24)}",
        f"Schedule (IST): prelim {_txt(cfg('prelim_scan_time'), 10)} | "
        f"final scan {_txt(cfg('final_scan_time'), 10)} | "
        f"cutoff {_txt(cfg('report_cutoff_time'), 10)} | deadline {_txt(cfg('report_deadline_time'), 10)}",
        f"Integrations: LLM {'configured' if cfg('llm_api_base', '') else 'off'} | "
        f"off-box backup {'configured' if cfg('s3_bucket', '') else 'off'} | "
        f"dashboard {flag(cfg('dashboard_enabled', False))} | training {_txt(cfg('training_mode'), 20)}",
        "<i>Connectivity test only; no market data or candidates are included.</i>",
    ]
    return "\n".join(lines)
