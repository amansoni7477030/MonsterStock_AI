"""Visual vocabulary of the Telegram messages: emoji, bars, clock faces and small formatters (pure: no I/O, no clock).

Rules that keep the messages honest and readable everywhere:

* Emoji only *decorate*. The words next to them carry every fact, so a reader (or a screen reader) that drops the emoji
  loses decoration, never information.
* Every icon is a single code point that renders as a colour emoji by itself (``WARN`` is the one deliberate exception and
  carries its variation selector), so a message looks the same on Android, iOS and desktop.
* A probability bar is drawn only for a calibrated probability; nothing is drawn for a missing or heuristic value.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

RULE = "\u2501" * 18  # a heavy horizontal line used between the head, body and footer of a message
WARN = "\u26a0\ufe0f"  # the warning sign needs its variation selector to render as an emoji

BAR_WIDTH = 10
_FILLED, _EMPTY = "\u2588", "\u2591"

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")  # never the locale's names

_CLOCKS = ("🕛", "🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚")
NO_CLOCK = "🕒"

_MEDALS = ("🥇", "🥈", "🥉")
OTHER_RANK = "🏅"
WEAK_RANK = "🔹"  # a list that is flagged as weak never gets medals

TIER_BADGES = {"STRONG": "🔥", "MODERATE": "⚡", "WATCH": "👀"}
UNKNOWN_TIER = "🔹"
WEAK_BADGE = "🔸"

REGIME = {  # key -> (icon, label); any other key is shown as it is with a neutral bullet
    "trend": ("📈", "Trend"),
    "volatility": ("🌊", "Volatility"),
    "risk": ("🚦", "Risk"),
    "event": ("📌", "Event"),
    "gift_nifty": ("🌅", "GIFT Nifty"),
}
BULLET = "🔹"

_SEVERITY = {"critical": "🔴", "error": "🟠", "warning": "🟡", "warn": "🟡", "info": "🔵"}
_GOOD_NEWS = {"health_recovered": "✅", "bootstrap_complete": "🎉"}  # not problems: no severity colour


def bar(p: Any, width: int = BAR_WIDTH) -> str:
    """``██████░░░░`` for a probability in [0, 1] (rounded to the nearest tenth); empty when ``p`` is not a probability."""
    if p is None or isinstance(p, bool):
        return ""
    try:
        x = float(p)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x) or x < -1e-9 or x > 1 + 1e-9:
        return ""
    filled = int(math.floor(min(max(x, 0.0), 1.0) * width + 0.5))
    return _FILLED * filled + _EMPTY * (width - filled)


def clock(hour: int | None) -> str:
    """The clock-face emoji for an hour of the day (12-hour faces); a neutral clock when the hour is unknown."""
    if hour is None or isinstance(hour, bool) or not 0 <= int(hour) <= 24:
        return NO_CLOCK
    return _CLOCKS[int(hour) % 12]


def rank_badge(rank: int) -> str:
    """Medals for the first three places, a plain medal after that."""
    return _MEDALS[rank - 1] if 1 <= rank <= len(_MEDALS) else OTHER_RANK


def tier_badge(tier: str) -> str:
    return TIER_BADGES.get(str(tier).strip().upper(), UNKNOWN_TIER)


def status_dot(text: Any) -> str:
    """Traffic-light dot for a data-freshness text such as ``ok (3 min old)`` or ``STALE - quote is 600 min old``."""
    head = str(text).strip().lower()
    for prefix, dot in (("ok", "🟢"), ("warn", "🟡"), ("degraded", "🟡"), ("stale", "🟠"), ("missing", "🔴"),
                        ("blocked", "🔴"), ("fail", "🔴")):
        if head.startswith(prefix):
            return dot
    return "⚪"


def severity_icon(level: Any, code: Any = "") -> str:
    """Severity colour of an alert (info blue, warning yellow, error orange, critical red); good-news codes get a tick."""
    good = _GOOD_NEWS.get(str(code).strip())
    if good:
        return good
    return _SEVERITY.get(str(level).strip().lower() or "info", "⚪")


def pretty_date(day: date) -> str:
    """``Mon 02 Nov 2026`` (English names, independent of the server's locale)."""
    return f"{WEEKDAYS[day.weekday()]} {day.day:02d} {MONTHS[day.month - 1]} {day.year}"


# ------------------------------------------------------------------------------------------ next steps

_SAFE_PART = re.compile(r"[A-Za-z0-9_.-]{1,40}")

# (code prefix, command). First match wins. Every command only reads or reports: an alert never suggests a change.
_HINTS: tuple[tuple[str, str], ...] = (
    ("scan_pre_failed", "sudo monster-ai logs --job prelim_scan"),
    ("scan_final_failed", "sudo monster-ai logs --job final_scan"),
    ("preflight_failed", "sudo monster-ai scan --stage final --dry-run"),
    ("snapshot_repair", "sudo monster-ai logs --job final_scan"),
    ("report_", "sudo monster-ai jobs"),
    ("job_failed_", "sudo monster-ai logs --job {rest}"),
    ("job_crashed_", "sudo monster-ai logs --job {rest}"),
    ("job_missed_", "sudo monster-ai jobs"),
    ("daemon_", "sudo monster-ai logs --role daemon"),
    ("health_storage", "sudo monster-ai storage"),
    ("health_wal", "sudo monster-ai storage"),
    ("health_retention", "sudo monster-ai logs --job housekeeping"),
    ("health_telegram", "sudo monster-ai telegram-test"),
    ("health_source_", "sudo monster-ai doctor"),
    ("health_", "sudo monster-ai health"),
    ("resources", "sudo monster-ai storage"),
    ("retention", "sudo monster-ai storage"),
    ("model_", "sudo monster-ai models list"),
    ("training_", "sudo monster-ai logs --role train"),
    ("backup_", "sudo monster-ai logs --job backup"),
    ("telegram_", "sudo monster-ai telegram-test"),
    ("price_basis", "sudo monster-ai logs --job collect_india"),
    ("symbol_", "sudo monster-ai logs"),
)


def alert_hint(code: Any) -> str | None:
    """The command to run first for an alert code, or ``None`` when there is nothing specific to suggest."""
    text = str(code or "").strip()
    for prefix, command in _HINTS:
        if text.startswith(prefix):
            rest = text[len(prefix):]
            if "{rest}" in command:
                if not _SAFE_PART.fullmatch(rest):
                    return "sudo monster-ai logs"
                return command.format(rest=rest)
            return command
    return None


def scan_hint(stage: Any) -> str:
    """The log command for a scan stage (``pre`` / ``final``); the job list for anything else."""
    return {"pre": "sudo monster-ai logs --job prelim_scan", "final": "sudo monster-ai logs --job final_scan"}.get(
        str(stage or "").strip().lower(), "sudo monster-ai jobs")
