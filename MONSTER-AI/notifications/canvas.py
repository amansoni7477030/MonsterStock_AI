"""Drawing primitives of the Telegram chart images (nothing but Pillow and the standard library: no clock, no network, no disk).

Everything is drawn at ``SCALE`` times its size and shrunk once at the end, so edges are smooth without a vector library. The font
that ships with Pillow has ASCII and the middle dot only: every label goes through ``ascii_label``, which swaps the few symbols the
reports use (``Rs`` for the rupee sign, ``->`` for an arrow ...) and drops what it cannot draw, so a chart never shows a missing-glyph
box. ``Canvas.labels`` lists every text a chart shows, which is what the tests read to check what a chart claims.

Pillow is an optional extra: when it is missing ``available()`` is false and the reports are sent as text only.
"""

from __future__ import annotations

import io
import math
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - a missing library is simulated in the tests
    Image = ImageDraw = ImageFont = None  # type: ignore[assignment]

WIDTH = 1080  # a phone shows a photo about this wide; Telegram shrinks anything larger
SCALE = 2
MARGIN = 44

Color = tuple[int, int, int]
BG: Color = (13, 20, 36)
CARD: Color = (23, 34, 58)
TRACK: Color = (36, 50, 80)
GRID: Color = (74, 92, 128)
TEXT: Color = (233, 239, 250)
MUTED: Color = (150, 166, 196)
GREEN: Color = (52, 211, 153)
LIGHT_GREEN: Color = (134, 239, 172)
RED: Color = (248, 113, 113)
BLUE: Color = (96, 165, 250)
AMBER: Color = (251, 191, 36)
PURPLE: Color = (167, 139, 250)
TEAL: Color = (45, 212, 191)
PINK: Color = (244, 114, 182)
SAMPLE_RED: Color = (185, 28, 28)
WHITE: Color = (255, 255, 255)

_SUBSTITUTES = {
    "\u20b9": "Rs ", "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u00d7": "x",
    "\u2022": "-", "\u2191": "up", "\u2193": "down", "\u2713": "ok", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2026": "...",
}


def available() -> bool:
    return Image is not None


def ascii_label(text: Any) -> str:
    """``text`` as one line of what the built-in font can draw (printable ASCII and the middle dot)."""
    out: list[str] = []
    for char in "" if text is None else str(text):
        if char.isspace():
            out.append(" ")
        elif char in _SUBSTITUTES:
            out.append(_SUBSTITUTES[char])
        elif char == "\u00b7" or " " <= char <= "~":
            out.append(char)
        else:  # an accented letter keeps its base letter; a symbol without one is dropped
            out.extend(c for c in unicodedata.normalize("NFKD", char) if " " <= c <= "~")
    return " ".join("".join(out).split())


@lru_cache(maxsize=48)
def _font(pixels: int) -> Any:
    try:
        return ImageFont.load_default(size=pixels)
    except (TypeError, OSError):  # Pillow before 10.1, or without FreeType: the small bitmap font
        return ImageFont.load_default()


def text_width(text: Any, size: float) -> float:
    """Width in logical pixels of ``text`` set at ``size``."""
    return _font(round(size * SCALE)).getlength(ascii_label(text)) / SCALE


def fit(text: Any, size: float, max_width: float) -> str:
    """``text`` cut with ``...`` so that it is at most ``max_width`` wide."""
    label = ascii_label(text)
    if text_width(label, size) <= max_width:
        return label
    while label and text_width(label + "...", size) > max_width:
        label = label[:-1]
    return label.rstrip() + "..."


def wrap(text: Any, size: float, max_width: float) -> list[str]:
    """``text`` broken into lines of at most ``max_width`` (a word that is too long is cut)."""
    lines: list[str] = []
    current = ""
    for word in ascii_label(text).split():
        candidate = f"{current} {word}".strip()
        if text_width(candidate, size) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = fit(word, size, max_width)
    if current:
        lines.append(current)
    return lines


class Canvas:
    """A drawing surface of ``WIDTH`` x ``height`` logical pixels."""

    def __init__(self, height: float, *, background: Color = BG) -> None:
        if Image is None:
            raise RuntimeError("Pillow is not installed")
        self.width, self.height = WIDTH, max(1, round(height))
        self.image = Image.new("RGB", (self.width * SCALE, self.height * SCALE), background)
        self.draw = ImageDraw.Draw(self.image)
        self.labels: list[str] = []

    # ---------------------------------------------------------------------------------- shapes
    def rect(self, x0: float, y0: float, x1: float, y1: float, fill: Color, radius: float = 0) -> None:
        if x1 - x0 < 0.5 or y1 - y0 < 0.5:
            return
        self.draw.rounded_rectangle([x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE], radius=min(radius, (y1 - y0) / 2) * SCALE,
                                    fill=fill)

    def line(self, x0: float, y0: float, x1: float, y1: float, fill: Color, width: float = 2) -> None:
        self.draw.line([x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE], fill=fill, width=max(1, round(width * SCALE)))

    def dashed_vline(self, x: float, y0: float, y1: float, fill: Color, width: float = 2, dash: float = 9, gap: float = 7) -> None:
        y = y0
        while y < y1:
            self.line(x, y, x, min(y + dash, y1), fill, width)
            y += dash + gap

    def triangle(self, cx: float, cy: float, size: float, fill: Color, *, up: bool = True) -> None:
        half = size / 2
        tip, base = (cy - half, cy + half) if up else (cy + half, cy - half)
        self.draw.polygon([(cx * SCALE, tip * SCALE), ((cx - half) * SCALE, base * SCALE), ((cx + half) * SCALE, base * SCALE)], fill=fill)

    def donut(self, cx: float, cy: float, radius: float, slices: Sequence[tuple[float, Color]], *, hole: float = 0.6) -> None:
        """A ring cut into slices in the order given, starting at 12 o'clock; an empty ring is drawn when nothing is positive."""
        box = [(cx - radius) * SCALE, (cy - radius) * SCALE, (cx + radius) * SCALE, (cy + radius) * SCALE]
        shown = [(value, color) for value, color in slices if value > 0]
        total = sum(value for value, _ in shown)
        if not total:
            self.draw.ellipse(box, fill=TRACK)
        elif len(shown) == 1:
            self.draw.ellipse(box, fill=shown[0][1])
        else:
            start = -90.0
            for value, color in shown:
                end = start + 360.0 * value / total
                self.draw.pieslice(box, start, end, fill=color)
                start = end
            start = -90.0
            for value, _ in shown:  # a background-coloured spoke between neighbours
                angle = math.radians(start)
                self.line(cx, cy, cx + radius * math.cos(angle), cy + radius * math.sin(angle), BG, 3)
                start += 360.0 * value / total
        inner = radius * hole
        self.draw.ellipse([(cx - inner) * SCALE, (cy - inner) * SCALE, (cx + inner) * SCALE, (cy + inner) * SCALE], fill=BG)

    # ----------------------------------------------------------------------------------- text
    def text(self, x: float, y: float, text: Any, size: float = 26, fill: Color = TEXT, *, anchor: str = "lm",
             bold: bool = False) -> float:
        """Draw ``text`` (anchored by its middle line unless ``anchor`` says otherwise); returns its width."""
        label = ascii_label(text)
        if not label:
            return 0.0
        self.labels.append(label)
        weight = (1 if size < 40 else 2) if bold else 0
        font = _font(round(size * SCALE))
        self.draw.text((x * SCALE, y * SCALE), label, font=font, fill=fill, anchor=anchor, stroke_width=weight, stroke_fill=fill)
        return font.getlength(label) / SCALE

    def paragraph(self, x: float, y: float, text: Any, size: float, fill: Color, max_width: float, *, gap: float = 1.4) -> float:
        """Wrapped text whose first line starts at ``y``; returns the y below its last line."""
        for line in wrap(text, size, max_width):
            self.text(x, y + size / 2, line, size, fill)
            y += size * gap
        return y

    def pill(self, x: float, y: float, text: Any, size: float = 24, fill: Color = TEXT, *, background: Color = CARD,
             pad: float = 18, height: float = 46, bold: bool = True) -> float:
        """A rounded label with its left edge at ``x`` and its top at ``y``; returns its right edge."""
        label = ascii_label(text)
        width = text_width(label, size) + 2 * pad + (1 if bold else 0)
        self.rect(x, y, x + width, y + height, background, radius=height / 2)
        self.text(x + pad, y + height / 2, label, size, fill, bold=bold)
        return x + width

    def png(self) -> bytes:
        final = self.image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        final.save(buffer, format="PNG")
        return buffer.getvalue()
