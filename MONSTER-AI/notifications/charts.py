"""The chart images that go with the Telegram reports (pure: no clock, no network, no disk).

They are built from the same report objects as the text, so a chart can never show anything the message does not say, and they keep
its honesty rules: a closing-hour score is called a score and never drawn as a probability; a probability is shown only for a model
that is not the heuristic baseline, with the same shadow-mode banner as the text; a report with no high-conviction setup gets no chart;
missing numbers are left out and never invented. A chart is an extra: every builder returns ``[]`` instead of raising, so the text
report is never held back by a missing library, a malformed value or a failed drawing.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.logging_setup import exception_site
from backtesting.ideas import GROUPS
from notifications import canvas as cv
from notifications import ideas as idea_text
from notifications import report as text
from notifications import style
from notifications.canvas import Canvas
from ranking.types import EveningSummary, IdeasReport, ReportContext

log = logging.getLogger(__name__)

CAPTION_LIMIT = 1024  # Telegram's limit of a photo caption
MAX_ROWS = 10
_X0, _X1 = cv.MARGIN, cv.WIDTH - cv.MARGIN
_CONTENT = _X1 - _X0
_SAMPLE_STRIP = 62
_SAMPLE_TEXT = "SAMPLE PREVIEW - made-up data, not a market signal"
_CLOCK = re.compile(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b")

_GROUP_STYLE = {
    "BTST": ("BTST", cv.BLUE), "SWING": ("Swing", cv.PURPLE), "SHORT": ("Bearish F&O", cv.RED), "WATCH": ("Watchlist", cv.AMBER),
}
_TIER_COLOR = {"STRONG": cv.GREEN, "MODERATE": cv.BLUE, "WATCH": cv.AMBER}
_SLICE_COLORS = (cv.BLUE, cv.PURPLE, cv.TEAL, cv.AMBER, cv.PINK, cv.GRID)
_BASELINE_NAMES = {"atr": "ATR", "volume": "Volume", "random": "Random"}


@dataclass(frozen=True)
class Chart:
    """One image: its PNG bytes, a Telegram-HTML caption and every text drawn on it (for the tests)."""

    name: str
    png: bytes
    caption: str
    labels: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return f"{self.name}.png"


@dataclass(frozen=True)
class _Block:
    height: float
    paint: Callable[[Canvas, float], None]  # (canvas, y of the block's top edge)


# --------------------------------------------------------------------------------------- composing
def _compose(blocks: Sequence[_Block], *, sample: bool) -> Canvas:
    top = _SAMPLE_STRIP if sample else 0
    canvas = Canvas(top + 28 + sum(block.height for block in blocks) + 24)
    if sample:
        canvas.rect(0, 0, canvas.width, _SAMPLE_STRIP, cv.SAMPLE_RED)
        canvas.text(canvas.width / 2, _SAMPLE_STRIP / 2, _SAMPLE_TEXT, 27, cv.WHITE, anchor="mm", bold=True)
    y = top + 28
    for block in blocks:
        block.paint(canvas, y)
        y += block.height
    return canvas


def _finish(name: str, canvas: Canvas, caption: str) -> Chart:
    return Chart(name, canvas.png(), caption[:CAPTION_LIMIT], tuple(canvas.labels))


def _title(kicker: str, title: str, lines: Sequence[str]) -> _Block:
    def paint(c: Canvas, y: float) -> None:
        c.text(_X0, y + 14, kicker, 26, cv.TEAL, bold=True)
        c.text(_X0, y + 66, title, 54, cv.TEXT, bold=True)
        for index, line in enumerate(lines):
            c.text(_X0, y + 120 + 36 * index, cv.fit(line, 25, _CONTENT), 25, cv.MUTED)

    return _Block(110 + 36 * len(lines) + 14, paint)


def _pills(items: Sequence[tuple[str, cv.Color]]) -> _Block:
    placed: list[list[tuple[str, cv.Color, float]]] = [[]]
    x = _X0
    for label, color in items:
        width = cv.text_width(label, 24) + 37
        if x + width > _X1 and placed[-1]:
            placed.append([])
            x = _X0
        placed[-1].append((label, color, x))
        x += width + 14

    def paint(c: Canvas, y: float) -> None:
        for row, entries in enumerate(placed):
            for label, color, left in entries:
                c.pill(left, y + 58 * row, label, 24, color)

    return _Block(58 * len(placed) + 14, paint)


def _stacked_bar(title: str, left: tuple[str, float, cv.Color], right: tuple[str, float, cv.Color]) -> _Block:
    total = left[1] + right[1]

    def paint(c: Canvas, y: float) -> None:
        c.text(_X0, y + 18, title, 28, cv.TEXT, bold=True)
        c.rect(_X0, y + 46, _X1, y + 76, cv.TRACK, radius=15)
        share = left[1] / total
        c.rect(_X0, y + 46, _X0 + _CONTENT * share, y + 76, left[2], radius=15)
        c.rect(_X0 + _CONTENT * share, y + 46, _X1, y + 76, right[2], radius=15)
        c.rect(_X0 + _CONTENT * share - 15, y + 46, _X0 + _CONTENT * share + 15, y + 76, left[2] if share >= 0.5 else right[2])
        c.text(_X0, y + 100, left[0], 24, left[2], bold=True)
        c.text(_X1, y + 100, right[0], 24, right[2], anchor="rm", bold=True)

    return _Block(132, paint)


def _donut(title: str, slices: Sequence[tuple[str, float, cv.Color]], centre: str, sub: str) -> _Block:
    legend = [entry for entry in slices if entry[1] > 0] or list(slices)

    def paint(c: Canvas, y: float) -> None:
        c.text(_X0, y + 18, title, 28, cv.TEXT, bold=True)
        cx, cy = _X0 + 125, y + 56 + 125
        c.donut(cx, cy, 115, [(value, color) for _, value, color in slices])
        c.text(cx, cy - 8, centre, 54, cv.TEXT, anchor="mm", bold=True)
        c.text(cx, cy + 36, sub, 22, cv.MUTED, anchor="mm")
        left, top = _X0 + 300, cy - 23 * len(legend)
        for index, (label, value, color) in enumerate(legend):
            mid = top + 46 * index + 23
            c.rect(left, mid - 12, left + 24, mid + 12, color, radius=7)
            c.text(left + 40, mid, cv.fit(label, 28, _X1 - left - 120), 28, cv.TEXT)
            c.text(_X1, mid, f"{value:g}", 28, cv.TEXT, anchor="rm", bold=True)

    return _Block(56 + 250 + 16, paint)


def _list(title: str, note: str | None, items: Sequence[Any], row: float, paint_row: Callable[[Canvas, float, Any], None]) -> _Block:
    note_lines = cv.wrap(note, 22, _CONTENT) if note else []
    head = 46 + 30 * len(note_lines) + (8 if note_lines else 0)

    def paint(c: Canvas, y: float) -> None:
        c.text(_X0, y + 18, title, 28, cv.TEXT, bold=True)
        for index, line in enumerate(note_lines):
            c.text(_X0, y + 52 + 30 * index, line, 22, cv.MUTED)
        for index, item in enumerate(items):
            top = y + head + row * index
            if index % 2 == 0:
                c.rect(_X0 - 14, top + 2, _X1 + 14, top + row - 2, cv.CARD, radius=14)
            paint_row(c, top, item)

    return _Block(head + row * len(items) + 12, paint)


def _notes(lines: Sequence[tuple[str, cv.Color]]) -> _Block:
    wrapped = [(piece, color) for line, color in lines for piece in cv.wrap(line, 21, _CONTENT)]

    def paint(c: Canvas, y: float) -> None:
        c.line(_X0, y + 6, _X1, y + 6, cv.GRID, 1)
        for index, (piece, color) in enumerate(wrapped):
            c.text(_X0, y + 32 + 29 * index, piece, 21, color)

    return _Block(24 + 29 * len(wrapped) + 12, paint)


# ----------------------------------------------------------------------------------------- helpers
def _guard(name: str, build: Callable[[], Chart | None]) -> Chart | None:
    try:
        return build()
    except Exception as exc:  # noqa: BLE001 - a chart is an extra: it must never cost the report
        log.warning("chart %s skipped: %s at %s", name, type(exc).__name__, exception_site(exc))
        return None


def _charts(*builders: tuple[str, Callable[[], Chart | None]]) -> list[Chart]:
    if not cv.available():
        log.info("charts skipped: Pillow is not installed (the reports go out as text)")
        return []
    return [chart for name, build in builders if (chart := _guard(name, build)) is not None]


def _day(value: Any) -> str:
    raw = text._clean(value)[:10]
    try:
        return style.pretty_date(date.fromisoformat(raw))
    except ValueError:
        return raw or text.NA


def _clock(value: Any) -> str:
    match = _CLOCK.search(text._clean(value))
    return f"{int(match.group(1)):02d}:{match.group(2)} IST" if match else ""


def _signed(value: float | None, digits: int = 2) -> str:
    return text.NA if value is None else f"{0.0 if round(value, digits) == 0 else value:+.{digits}f}%"


def _sign_color(value: float | None) -> cv.Color:
    return cv.TEXT if value is None or value == 0 else cv.GREEN if value > 0 else cv.RED


def _fraction(value: float, low: float, high: float) -> float:
    return 0.0 if high <= low else min(max((value - low) / (high - low), 0.0), 1.0)


def _tidy(counts: Counter[str]) -> list[tuple[str, float, cv.Color]]:
    """The four biggest groups of ``counts`` and one slice for everything else."""
    ranked = counts.most_common()
    head, rest = ranked[:4], sum(count for _, count in ranked[4:])
    slices = [(label, float(count), _SLICE_COLORS[index]) for index, (label, count) in enumerate(head)]
    if rest:
        slices.append(("Other", float(rest), _SLICE_COLORS[4]))
    return slices


def _tick_step(high: float) -> float:
    return next((step for step in (0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0) if high / step <= 6), 50.0)


# --------------------------------------------------------------------------- closing-hour ideas
def ideas_charts(report: IdeasReport, *, sample: bool = False) -> list[Chart]:
    """The closing-hour board and the trade plan; ``[]`` when there is nothing to draw."""
    return _charts(("ideas-board", lambda: _ideas_board(report, sample)), ("ideas-plan", lambda: _ideas_plan(report, sample)))


def _idea_rows(report: Any) -> list[Any]:
    return [idea for idea in text._seq(text._get(report, "ideas")) if text._num(text._get(idea, "score")) is not None]


def _ideas_banner(report: Any) -> str:
    record = text._map(text._get(report, "record"))
    validated = any(text._map(record.get(group)).get("validated") for group in GROUPS)
    return idea_text.RECORD_BANNER if validated else idea_text.HEURISTIC_BANNER


def _market_pills(market: Any) -> list[tuple[str, cv.Color]]:
    m = text._map(market)
    pills: list[tuple[str, cv.Color]] = []
    nifty = text._num(m.get("nifty_chg"))
    if nifty is not None:
        pills.append((f"Nifty {_signed(nifty)}", _sign_color(nifty)))
    vix, vix_chg = text._num(m.get("vix")), text._num(m.get("vix_chg"))
    if vix is not None:
        pills.append((f"VIX {vix:.1f}" + (f" ({_signed(vix_chg, 1)})" if vix_chg is not None else ""), cv.TEXT))
    regime = text._clean(m.get("regime")).lower()
    if regime in ("bullish", "neutral", "bearish"):
        pills.append((f"Regime {regime}", {"bullish": cv.GREEN, "bearish": cv.RED}.get(regime, cv.AMBER)))
    return pills


def _ideas_board(report: Any, sample: bool) -> Chart | None:
    ideas = _idea_rows(report)
    if not ideas:
        return None
    groups = Counter(idea_text.idea_group(idea) for idea in ideas)
    slices = [(_GROUP_STYLE[group][0], float(groups[group]), _GROUP_STYLE[group][1]) for group in GROUPS if groups[group]]
    ranked = sorted(ideas, key=lambda idea: -(text._num(text._get(idea, "score")) or 0.0))
    shown = ranked[:MAX_ROWS]
    when = _clock(text._get(report, "snapshot_ist"))
    entry_by = text._clip(text._get(report, "entry_by"), 8) or "15:15"
    subtitle = " · ".join(part for part in (_day(text._get(report, "trade_date")), f"snapshot {when}" if when else "",
                                            f"enter before {entry_by} IST") if part)
    blocks = [_title("MONSTER AI", "Closing-hour ideas", [subtitle])]
    pills = _market_pills(text._get(report, "market"))
    if pills:
        blocks.append(_pills(pills))
    market = text._map(text._get(report, "market"))
    up, down = text._num(market.get("advancers")), text._num(market.get("decliners"))
    if up is not None and down is not None and up >= 0 and down >= 0 and up + down > 0:
        blocks.append(_stacked_bar("Breadth: stocks up and down today", (f"{int(up)} up", up, cv.GREEN), (f"{int(down)} down", down, cv.RED)))
    blocks.append(_donut("Ideas by list", slices, str(len(ideas)), "ideas"))

    def score_row(c: Canvas, y: float, idea: Any) -> None:
        group = idea_text.idea_group(idea)
        name, color = _GROUP_STYLE.get(group, (group.title(), cv.GRID))
        side = text._clean(text._get(idea, "side")).lower()
        tag = f"{name}" + (f" {side}" if group in ("BTST", "SWING") else "") + f" · grade {text._clean(text._get(idea, 'grade')).upper()[:1] or '?'}"
        score = text._num(text._get(idea, "score")) or 0.0
        mid = y + 32
        c.text(_X0, y + 21, cv.fit(text._clip(text._get(idea, "symbol"), 24), 28, 270), 28, cv.TEXT, bold=True)
        c.text(_X0, y + 47, cv.fit(tag, 20, 270), 20, cv.MUTED)
        left, right = 340, 930
        c.rect(left, mid - 13, right, mid + 13, cv.TRACK, radius=13)
        c.rect(left, mid - 13, left + (right - left) * _fraction(score, 0, 100), mid + 13, color, radius=13)
        c.text(_X1, mid, f"{score:.0f}", 30, cv.TEXT, anchor="rm", bold=True)

    blocks.append(_list(f"Setup score (top {len(shown)} of {len(ranked)}): a ranking aid, NOT a probability", None, shown, 64, score_row))
    blocks.append(_notes([(_ideas_banner(report), cv.AMBER), (idea_text.DISCLAIMER, cv.MUTED)]))
    caption = (f"🌙 <b>Closing-hour board</b> · {len(ideas)} ideas in {len(slices)} list{'s' if len(slices) != 1 else ''}\n"
               "<i>Scores rank setups; they are not probabilities.</i>")
    return _finish("ideas-board", _compose(blocks, sample=sample), caption)


def _ideas_plan(report: Any, sample: bool) -> Chart | None:
    plans = []
    for idea in _idea_rows(report):
        group = idea_text.idea_group(idea)
        risk, reward1, reward2 = (text._num(text._get(idea, key)) for key in ("risk_pct", "reward1_pct", "reward2_pct"))
        if group not in ("BTST", "SWING", "SHORT") or risk is None or reward1 is None or reward2 is None or risk <= 0 or reward1 <= 0:
            continue
        plans.append((idea, group, risk, reward1, max(reward1, reward2)))
    if not plans:
        return None
    plans.sort(key=lambda plan: -(text._num(text._get(plan[0], "score")) or 0.0))
    plans = plans[:8]
    low, high = -max(plan[2] for plan in plans) * 1.1, max(plan[4] for plan in plans) * 1.1
    left, right = _X0 + 8, _X1 - 8

    def x_of(value: float) -> float:
        return left + (right - left) * _fraction(value, low, high)

    def plan_row(c: Canvas, y: float, plan: tuple[Any, str, float, float, float]) -> None:
        idea, group, risk, reward1, reward2 = plan
        name, color = _GROUP_STYLE[group]
        short = text._clean(text._get(idea, "side")).upper() == "SHORT"
        c.text(_X0, y + 24, cv.fit(text._clip(text._get(idea, "symbol"), 24), 30, 300), 30, cv.TEXT, bold=True)
        edge = c.pill(_X0 + 330, y + 4, "SHORT" if short else "LONG", 20, cv.RED if short else cv.GREEN, background=cv.TRACK, pad=14, height=40)
        c.text(edge + 14, y + 24, name, 22, color)
        rr1, rr2 = text._num(text._get(idea, "rr1")), text._num(text._get(idea, "rr2"))
        if rr1 is not None and rr2 is not None:
            c.text(_X1, y + 24, f"reward / risk {rr1:.1f} / {rr2:.1f}", 24, cv.TEXT, anchor="rm", bold=True)
        mid = y + 66
        c.rect(left, mid - 8, right, mid + 8, cv.TRACK, radius=8)
        c.rect(x_of(-risk), mid - 15, x_of(0), mid + 15, cv.RED)
        c.rect(x_of(0), mid - 15, x_of(reward1), mid + 15, cv.GREEN)
        c.rect(x_of(reward1), mid - 15, x_of(reward2), mid + 15, cv.LIGHT_GREEN)
        c.line(x_of(0), mid - 24, x_of(0), mid + 24, cv.WHITE, 3)
        cost = text._num(text._get(idea, "est_cost_pct"))
        parts = [f"stop -{risk:.2f}%", f"T1 +{reward1:.2f}%", f"T2 +{reward2:.2f}%"] + ([f"cost ~{cost:.2f}%"] if cost is not None else [])
        c.text(_X0, y + 104, " · ".join(parts), 22, cv.MUTED)

    note = "Red: loss if the stop is hit. Green: gain at target 1 and target 2. White tick: entry. % of the entry price, before costs."
    blocks = [
        _title("MONSTER AI", "Trade plan: risk and reward", [f"{_day(text._get(report, 'trade_date'))} · from the entry price"]),
        _list("Stop to targets", note, plans, 124, plan_row),
        _notes([(_ideas_banner(report), cv.AMBER), (idea_text.DISCLAIMER, cv.MUTED)]),
    ]
    caption = ("🎯 <b>Trade plan</b> · stop and targets as % from the entry\n"
               "<i>Mechanical ATR levels, not forecasts; before costs.</i>")
    return _finish("ideas-plan", _compose(blocks, sample=sample), caption)


# ---------------------------------------------------------------------- the morning (scan) report
def report_charts(ctx: ReportContext, *, sample: bool = False) -> list[Chart]:
    """The candidates chart of a preliminary or final report; ``[]`` for a day without a high-conviction setup."""
    return _charts(("candidates", lambda: _candidates(ctx, sample)))


def _candidates(ctx: Any, sample: bool) -> Chart | None:
    if text._get(ctx, "no_setup"):
        return None
    ranked = sorted(text._seq(text._get(ctx, "candidates")), key=lambda c: text._num(text._get(c, "rank")) or math.inf)[:MAX_ROWS]
    rows = []
    for cand in ranked:
        span = text._map(text._get(cand, "range"))
        q10, q50, q90 = (text._num(span.get(key)) for key in ("q10", "q50", "q90"))
        if q10 is None or q90 is None or q90 < q10 or q90 <= 0:
            continue
        rows.append((cand, q10, q50, q90))
    if not rows:
        return None
    status = text._clean(text._get(ctx, "model_status")).lower()
    validated, heuristic = status == "validated", status == "baseline"
    capped = not validated
    threshold = next((t for t in (text._num(text._get(cand, "primary_threshold")) for cand, *_ in rows) if t is not None), None)
    high = max(max(q90 for *_, q90 in rows), threshold or 0.0) * 1.12
    left, right = 330, 900

    def x_of(value: float) -> float:
        return left + (right - left) * _fraction(value, 0, high)

    def row(c: Canvas, y: float, entry: tuple[Any, float, float | None, float]) -> None:
        cand, q10, q50, q90 = entry
        number = text._num(text._get(cand, "rank"))
        tier = text._tier_name(text._get(cand, "tier"), weak=False, capped=capped)
        mid = y + 38
        c.text(_X0, y + 26, cv.fit(f"#{int(number) if number else '?'} {text._clip(text._get(cand, 'symbol'), 24)}", 28, 280), 28, cv.TEXT, bold=True)
        c.text(_X0, y + 56, tier, 20, _TIER_COLOR.get(tier, cv.MUTED))
        c.rect(left, mid - 4, right, mid + 4, cv.TRACK, radius=4)
        c.rect(x_of(q10), mid - 14, x_of(q90), mid + 14, _TIER_COLOR.get(tier, cv.BLUE), radius=14)
        if q50 is not None:
            c.line(x_of(q50), mid - 20, x_of(q50), mid + 20, cv.WHITE, 3)
        if threshold is not None:
            c.dashed_vline(x_of(threshold), y + 4, y + 72, cv.AMBER, 2)
        span_text = f"{q10:.1f}-{q90:.1f}%"
        c.text(right + 14, mid - 12, span_text, 22, cv.MUTED)
        primary = text._num(text._get(cand, "primary_threshold"))
        p = next((v for t, v in text._hit_items(text._map(text._get(cand, "p"))) if primary is not None and abs(t - primary) < 1e-9), None)
        probability = text._pct(p)
        if heuristic or probability == text.NA:
            c.text(_X1, mid + 14, "heuristic" if heuristic else "n/a", 22, cv.MUTED, anchor="rm")
        else:
            c.text(_X1, mid + 14, f"P {probability}", 26, cv.TEXT, anchor="rm", bold=True)

    lead = f"{len(rows)} candidate{'s' if len(rows) != 1 else ''}"
    note = ("Bar: 80% interval of the expected high-low range, % of the open. White tick: median. "
            + (f"Dashed line: the {threshold:g}% target. " if threshold is not None else "")
            + ("P: model probability that the range beats it." if not heuristic else "No trained model: ranking is heuristic, no probability is shown."))
    stage = "Preliminary" if text._clean(text._get(ctx, "stage")).lower() == "pre" else "Final"
    version = text._clip(text._get(ctx, "model_version"), 40)
    blocks = [_title("MONSTER AI", f"{stage} large-move scan", [f"{_day(text._get(ctx, 'trade_date'))} · {_clock(text._get(ctx, 'snapshot_ist'))} · model {version}"])]
    sectors = Counter(text._clip(text._get(cand, "sector"), 24) for cand, *_ in rows if text._clip(text._get(cand, "sector"), 24))
    if len(sectors) >= 2:
        blocks.append(_donut("Sector mix of the candidates", _tidy(sectors), str(sum(sectors.values())), "names"))
    blocks.append(_list(lead + ": expected intraday range", note, rows, 78, row))
    badge = text._MODEL_BADGES.get(status, text._SHADOW_BADGE)[1]
    notes = [(f"Model status: {badge}", cv.GREEN if validated else cv.AMBER)]
    if not validated:
        notes.append((text.SHADOW_BANNER, cv.AMBER))
    blocks.append(_notes([*notes, (text.DISCLAIMER, cv.MUTED)]))
    caption = (f"🚀 <b>{stage} scan</b> · {lead}\n<i>"
               + ("Range forecasts; tiers are capped at WATCH while the model is unvalidated." if not validated
                  else "Range forecasts from the validated model.") + "</i>")
    return _finish("candidates", _compose(blocks, sample=sample), caption)


# ------------------------------------------------------------------------------ the evening report
def evening_charts(summary: EveningSummary, *, sample: bool = False) -> list[Chart]:
    """What the day's picks did against the target and against the baselines; ``[]`` without picks."""
    return _charts(("evening", lambda: _evening(summary, sample)))


def _evening(summary: Any, sample: bool) -> Chart | None:
    rows = sorted(text._seq(text._get(summary, "rows")), key=lambda r: text._num(text._get(r, "rank")) or math.inf)[:MAX_ROWS]
    if not rows:
        return None
    threshold = text._num(text._get(summary, "primary_threshold"))
    hits = sum(1 for r in rows if text._get(r, "hit_primary") is True)
    misses = sum(1 for r in rows if text._get(r, "hit_primary") is False)
    pending = len(rows) - hits - misses
    ranges = [v for v in (text._num(text._get(r, "range_pct")) for r in rows) if v is not None]
    high = max([*ranges, threshold or 0.0, 1.0]) * 1.12
    left, right = 330, 900

    def x_of(value: float) -> float:
        return left + (right - left) * _fraction(value, 0, high)

    def pick(c: Canvas, y: float, row: Any) -> None:
        number = text._num(text._get(row, "rank"))
        mid = y + 32
        c.text(_X0, y + 21, cv.fit(f"#{int(number) if number else '?'} {text._clip(text._get(row, 'symbol'), 24)}", 28, 280), 28, cv.TEXT, bold=True)
        c.text(_X0, y + 47, text._clip(text._get(row, "tier"), 12).upper(), 20, cv.MUTED)
        outcome = text._get(row, "hit_primary")
        color = cv.GREEN if outcome is True else cv.RED if outcome is False else cv.GRID
        realised = text._num(text._get(row, "range_pct"))
        c.rect(left, mid - 4, right, mid + 4, cv.TRACK, radius=4)
        if realised is not None:
            c.rect(left, mid - 13, max(x_of(realised), left + 6), mid + 13, color, radius=13)
        if threshold is not None:
            c.dashed_vline(x_of(threshold), y + 4, y + 60, cv.AMBER, 2)
        c.text(_X1, mid, ("hit " if outcome is True else "miss " if outcome is False else "pending ") + (f"{realised:.1f}%" if realised is not None else text.NA), 24,
               cv.TEXT, anchor="rm", bold=True)

    blocks = [_title("MONSTER AI", "Evening evaluation", [f"{_day(text._get(summary, 'trade_date'))} · model {text._clip(text._get(summary, 'model_version'), 40)}"])]
    blocks.append(_donut("Picks against the target", [("Hit", float(hits), cv.GREEN), ("Missed", float(misses), cv.RED), ("Not judged", float(pending), cv.GRID)],
                         f"{hits}/{len(rows)}", "hit"))
    rates = [("Model", text._num(text._get(summary, "model_hit_rate")), cv.GREEN)]
    rates += [(_BASELINE_NAMES.get(str(name).lower(), text._clip(name, 14).title()), text._num(value), cv.GRID)
              for name, value in sorted(text._map(text._get(summary, "baseline_hit_rates")).items())[:4]]
    rates = [(label, value, color) for label, value, color in rates if value is not None and 0 <= value <= 1]
    if len(rates) >= 2:
        def rate_row(c: Canvas, y: float, item: tuple[str, float, cv.Color]) -> None:
            label, value, color = item
            mid = y + 24
            c.text(_X0, mid, label, 26, cv.TEXT, bold=True)
            c.rect(left, mid - 13, right, mid + 13, cv.TRACK, radius=13)
            c.rect(left, mid - 13, left + (right - left) * value, mid + 13, color, radius=13)
            c.text(_X1, mid, f"{value * 100:.0f}%", 28, cv.TEXT, anchor="rm", bold=True)

        blocks.append(_list("Hit rate of this day's picks: model against baselines", "One day is noise: judge the model on the rolling record.", rates, 52, rate_row))
    target = f"{threshold:g}% of the open" if threshold is not None else "the target"
    blocks.append(_list("Realised intraday range", f"Bar: the day's high-low range, % of the open. Dashed line: the target ({target}).", rows, 64, pick))
    provisional = bool(text._get(summary, "provisional", True))
    blocks.append(_notes([(text.PROVISIONAL_LABEL if provisional else text.VALIDATED_LABEL, cv.AMBER if provisional else cv.GREEN),
                          (text.EVENING_NOTE, cv.MUTED), (text.DISCLAIMER, cv.MUTED)]))
    caption = f"🌆 <b>Evening evaluation</b> · {hits} of {len(rows)} picks beat the target\n<i>{text.EVENING_NOTE}</i>"
    return _finish("evening", _compose(blocks, sample=sample), caption)
