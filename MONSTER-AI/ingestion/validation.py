"""Sanity validation for market data: reject impossible bars, repair rounding noise, detect re-adjusted history.

Two classes of bad-looking bars are kept apart. IMPOSSIBLE bars (missing or non-positive prices, high below open/close,
low above open/close, negative equity volume, placeholders) cannot be real and are always rejected. EXTREME bars are internally
consistent but have a huge intraday range: those are the days this system exists to find (limit-up listings, crashes), so
callers that pass ``keep_extreme`` keep them, flagged ``quality = "suspect"``, instead of dropping them."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

REPAIR_TOLERANCE = 0.0005  # 0.05%: rounding noise in vendor OHLC is repaired, larger violations are rejected
MAX_RANGE_FRACTION = {"equity": 0.6, "index": 0.35, "global": 0.35, "fx": 0.15, "commodity": 0.6, "rate": 0.5, "volatility": 3.0}
SUSPECT = "suspect"  # bars.quality of a consistent bar whose range exceeds the plausibility bound


@dataclass
class ValidationReport:
    accepted: int = 0
    repaired: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)  # impossible bars: (trade_date, reason)
    suspect: list[tuple[str, str]] = field(default_factory=list)  # extreme but consistent bars that were kept: (trade_date, reason)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def suspect_count(self) -> int:
        return len(self.suspect)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_bars(
    bars: Sequence[Mapping[str, Any]], *, kind: str = "equity", require_volume: bool = False, keep_extreme: bool = False
) -> tuple[list[dict[str, Any]], ValidationReport]:
    """Return clean ascending, de-duplicated bars plus a report.

    Impossible bars are rejected: OHLC finite and positive; ``high >= max(open, close)`` and ``low <= min(open, close)``
    (tiny violations are repaired); no negative volume (for the series other than equities, whose volume feeds nothing, only the
    junk volume is dropped); NSE equity bars with zero volume are vendor placeholders for non-trading days. A bar that is
    consistent but whose intraday range exceeds the plausibility bound of its ``kind`` is rejected as "implausible range", or,
    with ``keep_extreme``, kept with ``quality = "suspect"`` and listed in ``report.suspect`` (genuine extreme days are exactly
    what the models learn from)."""
    report = ValidationReport()
    by_date: dict[str, dict[str, Any]] = {}
    extreme: dict[str, str] = {}
    limit = MAX_RANGE_FRACTION.get(kind, 0.6)
    for bar in bars:
        date_key = str(bar.get("trade_date", ""))[:10]
        if len(date_key) != 10:
            report.rejected.append((date_key or "?", "bad date"))
            continue
        o, h, low, c = (_finite(bar.get(k)) for k in ("open", "high", "low", "close"))
        if None in (o, h, low, c):
            report.rejected.append((date_key, "missing OHLC"))
            continue
        assert o is not None and h is not None and low is not None and c is not None
        if min(o, h, low, c) <= 0:
            report.rejected.append((date_key, "non-positive price"))
            continue
        repaired = False
        top, bottom = max(o, c), min(o, c)
        if h < top:
            if (top - h) / top <= REPAIR_TOLERANCE:
                h, repaired = top, True
            else:
                report.rejected.append((date_key, "high below open/close"))
                continue
        if low > bottom:
            if (low - bottom) / bottom <= REPAIR_TOLERANCE:
                low, repaired = bottom, True
            else:
                report.rejected.append((date_key, "low above open/close"))
                continue
        is_extreme = (h - low) / low > limit
        if is_extreme and not keep_extreme:
            report.rejected.append((date_key, "implausible range"))
            continue
        volume = _finite(bar.get("volume"))
        if volume is not None and volume < 0:
            if kind == "equity":
                report.rejected.append((date_key, "negative volume"))
                continue
            volume, repaired = None, True  # volume feeds no feature of the other series: keep the good price, drop the junk volume
        if kind == "equity" and (volume is None or volume <= 0) and require_volume:
            report.rejected.append((date_key, "zero volume"))
            continue
        if kind == "equity" and volume == 0 and h == low:
            report.rejected.append((date_key, "flat zero-volume placeholder"))
            continue
        clean = dict(bar)
        clean.update(trade_date=date_key, open=o, high=h, low=low, close=c, volume=volume)
        if is_extreme:
            clean["quality"] = SUSPECT
            extreme[date_key] = "implausible range"
        else:
            extreme.pop(date_key, None)  # a later duplicate of the same day replaces an earlier extreme one
        by_date[date_key] = clean
        report.accepted += 1
        report.repaired += int(repaired)
    report.suspect = [(d, extreme[d]) for d in sorted(by_date) if d in extreme]
    return [by_date[d] for d in sorted(by_date)], report


def overlap_consistent(
    stored_closes: Mapping[str, float], new_bars: Sequence[Mapping[str, Any]], *, tolerance: float = 0.005,
    min_mismatches: int = 3,
) -> bool:
    """False when overlapping closes disagree on >= ``min_mismatches`` dates: the vendor re-adjusted history (split,
    bonus, demerger) so previously stored rows are stale and the symbol must be reloaded in full."""
    mismatches = 0
    compared = 0
    for bar in new_bars:
        key = str(bar.get("trade_date", ""))[:10]
        old = stored_closes.get(key)
        new = _finite(bar.get("close"))
        if old is None or new is None or old <= 0:
            continue
        compared += 1
        if abs(new / old - 1.0) > tolerance:
            mismatches += 1
    if compared == 0:
        return True
    return mismatches < min(min_mismatches, compared) if compared >= min_mismatches else mismatches == 0
