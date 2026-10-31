"""Data-quality / freshness assessment at a cutoff (blueprint sections 9, 17, 20).

A report is *blocked* when critical inputs are stale (Indian daily bars, the Nifty index, several critical global
series), *degraded* when optional feeds are missing, and *ok* otherwise. Nothing here ever "fixes" a gap by
reusing stale prices: gaps are reported and the downstream models see missing values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.clock import IST, iso, parse_ts
from automation.calendar import ExchangeCalendar
from database.store import Store
from ingestion.series import SERIES, SeriesSpec

INDIA_BLOCK_COVERAGE = 0.90
INDIA_WARN_COVERAGE = 0.97
GLOBAL_WARN_RATIO = 0.75
# A report with a few missing overseas series is still useful (flagged as degraded); a systemic outage is not.
GLOBAL_BLOCK_STALE_CRITICAL = 6
GIFT_MAX_AGE_MIN = 30
NEWS_MAX_AGE_HOURS = 8


@dataclass
class FeedState:
    name: str
    level: str  # ok | warn | stale | missing
    detail: str
    critical: bool = False


@dataclass
class DataQuality:
    status: str = "ok"  # ok | degraded | blocked
    expected_session: str = ""
    feeds: dict[str, FeedState] = field(default_factory=dict)
    india_coverage: float = 0.0
    fno_coverage: float | None = None
    preopen_coverage: float | None = None
    preopen_final_coverage: float | None = None
    global_fresh: int = 0
    global_total: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.status == "blocked"

    def freshness(self) -> dict[str, str]:
        return {name: f"{s.level.upper() if s.level != 'ok' else 'ok'} - {s.detail}" if s.level != "ok" else f"ok ({s.detail})"
                for name, s in self.feeds.items()}

    def missing_feeds(self) -> list[str]:
        return [f"{s.name}: {s.detail}" for s in self.feeds.values() if s.level in {"missing", "stale"}]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prev_weekday(day: date) -> date:
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def expected_bar_date(spec: SeriesSpec, cutoff: datetime) -> date:
    """Latest weekday whose bar is final at ``cutoff`` in the series' own timezone."""
    day = cutoff.astimezone(ZoneInfo(spec.tz)).date()
    for _ in range(10):
        if day.weekday() < 5 and spec.available_at(day) <= cutoff:
            return day
        day -= timedelta(days=1)
    return day


def assess_data_quality(
    store: Store, calendar: ExchangeCalendar, *, cutoff: datetime, stage: str, trade_date: date,
) -> DataQuality:
    quality = DataQuality()
    expected = calendar.last_completed_session(cutoff)
    quality.expected_session = expected.isoformat()
    blocking = False

    # --- Indian daily bars ---------------------------------------------------------------------------
    universe = store.list_instruments(kind="equity", in_universe=True, status="active")
    have_bar = store.symbols_with_bar(expected.isoformat(), universe_only=True)
    at_expected = sum(1 for r in universe if r["symbol"] in have_bar)
    quality.india_coverage = at_expected / len(universe) if universe else 0.0
    if not universe:
        quality.feeds["India daily bars"] = FeedState("India daily bars", "missing", "no universe loaded", True)
        blocking = True
    elif quality.india_coverage < INDIA_BLOCK_COVERAGE:
        quality.feeds["India daily bars"] = FeedState(
            "India daily bars", "stale", f"{quality.india_coverage:.0%} of universe has the {expected} bar", True)
        blocking = True
    elif quality.india_coverage < INDIA_WARN_COVERAGE:
        quality.feeds["India daily bars"] = FeedState(
            "India daily bars", "warn", f"{quality.india_coverage:.0%} of universe has the {expected} bar", True)
    else:
        quality.feeds["India daily bars"] = FeedState(
            "India daily bars", "ok", f"{quality.india_coverage:.0%} of universe at {expected}", True)

    # --- global / index series -----------------------------------------------------------------------
    stale_critical: list[str] = []
    fresh = 0
    for spec in SERIES:
        iid = store.instrument_id(spec.key, spec.kind)
        last = store.latest_bar_date(iid) if iid is not None else None
        if last is None:
            state = FeedState(spec.name, "missing", "no data", spec.critical)
        else:
            exp = expected_bar_date(spec, cutoff) if spec.category != "india_index" and spec.key != "^INDIAVIX" else expected
            if date.fromisoformat(last) < _prev_weekday(exp) and spec.category != "india_index":
                state = FeedState(spec.name, "stale", f"latest {last}, expected ~{exp}", spec.critical)
            elif spec.category == "india_index" or spec.key == "^INDIAVIX":
                state = (FeedState(spec.name, "ok", f"latest {last}", spec.critical) if last >= expected.isoformat()
                         else FeedState(spec.name, "stale", f"latest {last}, expected {expected}", spec.critical))
            else:
                state = FeedState(spec.name, "ok", f"latest {last}", spec.critical)
        if state.level == "ok":
            fresh += 1
        elif spec.critical:
            stale_critical.append(spec.key)
    quality.global_fresh, quality.global_total = fresh, len(SERIES)
    ratio = fresh / len(SERIES)
    nifty_bad = "^NSEI" in stale_critical
    if nifty_bad or len(stale_critical) >= GLOBAL_BLOCK_STALE_CRITICAL:
        quality.feeds["Global closes"] = FeedState(
            "Global closes", "stale", f"{fresh}/{len(SERIES)} fresh; critical stale: {', '.join(stale_critical)}", True)
        blocking = True
    elif stale_critical or ratio < GLOBAL_WARN_RATIO:
        quality.feeds["Global closes"] = FeedState(
            "Global closes", "warn", f"{fresh}/{len(SERIES)} fresh" + (f"; stale: {', '.join(stale_critical)}" if stale_critical else ""))
    else:
        quality.feeds["Global closes"] = FeedState("Global closes", "ok", f"{fresh}/{len(SERIES)} fresh")

    # --- F&O end-of-day --------------------------------------------------------------------------------
    fno_universe = [r["symbol"] for r in universe if r["is_fno"]]
    if fno_universe:
        count = store.conn.execute(
            f"SELECT COUNT(*) AS n FROM fno_daily WHERE trade_date = ? AND symbol IN ({', '.join('?' * len(fno_universe))})",
            [expected.isoformat(), *fno_universe],
        ).fetchone()["n"]
        quality.fno_coverage = count / len(fno_universe)
        if quality.fno_coverage < 0.5:
            quality.feeds["F&O end-of-day"] = FeedState("F&O end-of-day", "missing", f"{quality.fno_coverage:.0%} of F&O stocks at {expected}")
        elif quality.fno_coverage < 0.9:
            quality.feeds["F&O end-of-day"] = FeedState("F&O end-of-day", "warn", f"{quality.fno_coverage:.0%} of F&O stocks")
        else:
            quality.feeds["F&O end-of-day"] = FeedState("F&O end-of-day", "ok", f"{quality.fno_coverage:.0%} of F&O stocks at {expected}")

    # --- live: pre-open book and GIFT Nifty --------------------------------------------------------------
    cutoff_iso = iso(cutoff) or ""
    if stage == "final" and universe:
        book = store.latest_preopen(trade_date.isoformat(), before=cutoff_iso)
        symbols = {r["symbol"] for r in universe}
        usable = [q for s, q in book.items() if s in symbols and (q.get("final_price") or q.get("iep"))]
        final = [q for q in usable if q.get("final_price")]
        quality.preopen_coverage = len(usable) / len(symbols)
        quality.preopen_final_coverage = len(final) / len(symbols)
        if quality.preopen_coverage < 0.30:
            quality.feeds["Pre-open book"] = FeedState("Pre-open book", "missing", "pre-open prices unavailable; preliminary model used")
        elif quality.preopen_final_coverage < 0.60:
            quality.feeds["Pre-open book"] = FeedState(
                "Pre-open book", "warn", f"{quality.preopen_final_coverage:.0%} have the final equilibrium price")
        else:
            quality.feeds["Pre-open book"] = FeedState("Pre-open book", "ok", f"{quality.preopen_final_coverage:.0%} final prices")
    gift = store.latest_live_quote("GIFT_NIFTY", before=cutoff_iso)
    if gift is None:
        quality.feeds["GIFT Nifty"] = FeedState("GIFT Nifty", "missing", "no quote")
    else:
        stamp = parse_ts(gift.get("source_ts") or gift.get("received_at"))
        age_min = (cutoff - stamp).total_seconds() / 60.0 if stamp else 1e9
        quality.feeds["GIFT Nifty"] = (
            FeedState("GIFT Nifty", "ok", f"{age_min:.0f} min old") if age_min <= GIFT_MAX_AGE_MIN
            else FeedState("GIFT Nifty", "stale", f"quote is {min(age_min, 99999):.0f} min old"))

    # --- news / filings coverage ---------------------------------------------------------------------------
    health = {h["source_id"]: h for h in store.source_health()}
    for label, source in (("News feeds", "rss_news"), ("Exchange filings", "nse_announcements")):
        last_ok = parse_ts((health.get(source) or {}).get("last_success_at"))
        if last_ok is None:
            quality.feeds[label] = FeedState(label, "missing", "never collected")
        elif cutoff - last_ok > timedelta(hours=NEWS_MAX_AGE_HOURS):
            quality.feeds[label] = FeedState(label, "stale", f"last success {(cutoff - last_ok).total_seconds() / 3600:.0f} h ago")
        else:
            quality.feeds[label] = FeedState(label, "ok", f"collected {(cutoff - last_ok).total_seconds() / 60:.0f} min ago")

    quality.status = "blocked" if blocking else ("degraded" if any(s.level != "ok" for s in quality.feeds.values()) else "ok")
    quality.messages = [f"{s.name}: {s.detail}" for s in quality.feeds.values() if s.level != "ok"]
    return quality


def ist_label(moment: datetime) -> str:
    return moment.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
