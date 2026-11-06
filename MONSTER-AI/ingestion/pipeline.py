"""Ingestion orchestration: bars, official end-of-day files, F&O and live pre-open data.

Principles: (1) a bar is stored only once its session is final (point-in-time availability rule); (2) official
exchange files outrank vendor data and are stored RAW; (3) every loop is bounded by a deadline and resumable; (4) one bad
symbol never stops the run; (5) vendor re-adjustments (splits/bonus) are detected and the history reloaded, while the raw
official rows are re-based on the post-split basis only when the loader reads them (``ingestion.adjust``); (6) a bar that
validation rejects is logged and counted, never dropped silently.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.clock import IST, Clock, get_clock, iso, ist_date, parse_ts
from app.errors import CircuitOpen, DataUnavailable
from app.logging_setup import redact
from database.store import Store
from ingestion.fno import aggregate_stock_fno
from ingestion.kotak import KotakApi, KotakQuotes
from ingestion.nse import LiveEquities, NseApi, NseArchive, PreopenSnapshot
from ingestion.series import NSE_EQUITY_SPEC, SERIES, SERIES_BY_KEY, SeriesSpec
from ingestion.validation import overlap_consistent, validate_bars
from ingestion.yahoo import SymbolNotFound, YahooClient

log = logging.getLogger(__name__)

NOTFOUND_LIMIT = 3
PROBE_MAX_AGE = timedelta(hours=24)  # a series is always fetched in full at least this often, whatever the probe says
PROBE_MIN_SERIES = 4                 # below this a bulk probe saves nothing
PROBE_REL_TOL = 1e-4


@dataclass
class IngestReport:
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    unchanged: int = 0  # confirmed up to date by the bulk probe (counted in ``ok``; no per-symbol request was made)
    rows: int = 0
    partial: bool = False
    notes: list[str] = field(default_factory=list)

    def note(self, text: str, limit: int = 12) -> None:
        if len(self.notes) < limit:
            self.notes.append(text)

    @property
    def success_ratio(self) -> float:
        total = self.ok + self.failed
        return self.ok / total if total else 1.0


# ----------------------------------------------------------------------------------------- instruments
def ensure_series_instruments(store: Store) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in SERIES:
        exchange = "NSE" if spec.category == "india_index" or spec.key == "^INDIAVIX" else "GLOBAL"
        out[spec.key] = store.upsert_instrument(
            spec.key, spec.kind, exchange=exchange, name=spec.name, yahoo_symbol=spec.yahoo, in_universe=0
        )
    return out


def spec_for_instrument(symbol: str, kind: str) -> SeriesSpec:
    if kind == "equity":
        return NSE_EQUITY_SPEC
    return SERIES_BY_KEY.get(symbol) or NSE_EQUITY_SPEC


def pit_filter(bars: Sequence[dict[str, Any]], spec: SeriesSpec, now: datetime) -> list[dict[str, Any]]:
    """Keep only bars that are final at ``now`` (session close + publication lag already passed)."""
    return [b for b in bars if spec.available_at(date.fromisoformat(b["trade_date"])) <= now]


def range_for_years(years: int) -> str:
    if years <= 1:
        return "1y"
    if years <= 2:
        return "2y"
    if years <= 5:
        return "5y"
    return "10y" if years <= 10 else "max"


def choose_range(last_date: str | None, today: date, years: int) -> str:
    if last_date is None:
        return range_for_years(years)
    gap = (today - date.fromisoformat(last_date)).days
    for limit, label in ((4, "5d"), (25, "1mo"), (80, "3mo"), (170, "6mo"), (350, "1y"), (700, "2y"), (1800, "5y")):
        if gap <= limit:
            return label
    return range_for_years(years)


# ----------------------------------------------------------------------------------------- vendor bars
REJECT_SAMPLE = 3  # reasons quoted in a rejection log line


def _quote(entries: Sequence[tuple[str, str]]) -> str:
    """The first few (subject, reason) pairs of a validation report as one log fragment."""
    text = "; ".join(f"{subject}: {reason}" for subject, reason in entries[:REJECT_SAMPLE])
    return text + ("; ..." if len(entries) > REJECT_SAMPLE else "")


def _count_rejected(store: Store, symbol: str, n: int) -> None:
    """Running total of the bars validation rejected for ``symbol`` (kv ``bars:rejected:<symbol>``). A bar that stays inside
    the next refresh window is counted again: the number says "this feed has problems", not "this many distinct days"."""
    key = f"bars:rejected:{symbol}"
    try:
        total = int(store.kv_get(key, "0") or 0)
    except ValueError:
        log.warning("%s: unreadable rejection counter %r reset", symbol, store.kv_get(key))
        total = 0
    store.kv_set(key, str(total + n))


def _store_series(store: Store, row: Any, bars: list[dict[str, Any]], series: Any) -> int:
    written = store.upsert_bars(int(row["instrument_id"]), bars, source="yahoo", source_rank=1)
    for split in series.splits:
        ratio = split["numerator"] / split["denominator"] if split["denominator"] else None
        store.add_corporate_action(int(row["instrument_id"]), "split", split["ex_date"], ratio=ratio, source="yahoo")
    for dividend in series.dividends[-12:]:
        store.add_corporate_action(int(row["instrument_id"]), "dividend", dividend["ex_date"], amount=dividend["amount"],
                                   source="yahoo")
    return written


def _fetch_clean(
    store: Store, yahoo: YahooClient, row: Any, spec: SeriesSpec, rng: str, now: datetime
) -> tuple[Any, list[dict[str, Any]]]:
    """One vendor request: validated, PIT-filtered bars plus the raw series (its split events). Impossible bars are rejected
    LOUDLY (WARNING with the count and the first reasons, plus the per-symbol counter); extreme but consistent days are kept
    flagged ``suspect``: they are exactly the large moves this system exists to learn from."""
    symbol = row["symbol"]
    series = yahoo.fetch_daily(row["yahoo_symbol"] or symbol, range_=rng)
    bars, report = validate_bars(series.bars, kind=row["kind"], keep_extreme=True)
    if report.rejected:
        log.warning("%s: %d of %d bars rejected by validation (%s)", symbol, report.rejected_count, len(series.bars),
                    _quote(report.rejected))
        _count_rejected(store, symbol, report.rejected_count)
    if report.suspect:
        log.info("%s: %d extreme but consistent bars kept as suspect (%s)", symbol, report.suspect_count, _quote(report.suspect))
    return series, pit_filter(bars, spec, now)


def _vendor_history_changed(store: Store, iid: int, bars: list[dict[str, Any]], series: Any) -> bool:
    """True when the vendor re-adjusted history: a split inside the stored history that has not been recorded yet, or stored
    vendor closes that no longer match the fresh ones. Only reads: the split is recorded together with the reload."""
    stored = {
        r["trade_date"]: r["close"]
        for r in store.conn.execute(
            "SELECT trade_date, close FROM bars_daily WHERE instrument_id = ? AND source_rank <= 1 "
            "AND trade_date BETWEEN ? AND ?", (iid, bars[0]["trade_date"], bars[-1]["trade_date"]),
        ).fetchall()
        if r["close"]
    }
    known = {r["ex_date"] for r in store.corporate_actions(iid, "split")}
    earliest = _earliest(store, iid)
    new_split = any(s["ex_date"] not in known and s["ex_date"] > earliest for s in series.splits)
    return new_split or not overlap_consistent(stored, bars)


def ingest_symbol_daily(
    store: Store, yahoo: YahooClient, row: Any, *, now: datetime, years: int, force_full: bool = False
) -> int:
    """Fetch, validate, PIT-filter and store one instrument; reload full history if the vendor re-adjusted it.

    The reload fetches the full history FIRST and only then, in one transaction, drops the vendor rows, writes the fresh ones
    and records the split events: a failing request leaves the stored history and the events as they were, so the next run
    detects the re-adjustment again instead of finding a half-reloaded series. Official rows are never touched: they stay raw
    and ``features.loader`` re-bases them on the post-split basis at read time (``ingestion.adjust``)."""
    iid, symbol = int(row["instrument_id"]), row["symbol"]
    spec = spec_for_instrument(symbol, row["kind"])
    last = store.latest_bar_date(iid)
    full = force_full or last is None
    rng = range_for_years(years) if full else choose_range(last, ist_date(now), years)
    series, bars = _fetch_clean(store, yahoo, row, spec, rng, now)
    if not bars:
        return 0
    if not full and _vendor_history_changed(store, iid, bars, series):
        log.warning("%s: vendor history changed (split/re-adjustment); reloading full history", symbol)
        series, bars = _fetch_clean(store, yahoo, row, spec, range_for_years(years), now)
        if not bars:
            log.warning("%s: the full reload returned no usable bars; the stored history is kept", symbol)
            return 0
        with store.tx():
            store.delete_vendor_bars(iid)
            return _store_series(store, row, bars, series)
    return _store_series(store, row, bars, series)


def _earliest(store: Store, iid: int) -> str:
    row = store.conn.execute("SELECT MIN(trade_date) AS d FROM bars_daily WHERE instrument_id = ?", (iid,)).fetchone()
    return row["d"] or "9999-12-31"


def _order_key(row: Any) -> tuple[int, str]:
    spec = SERIES_BY_KEY.get(row["symbol"])
    if row["kind"] != "equity" and spec is not None:
        return (0 if spec.critical else 1, row["symbol"])
    return (2, row["symbol"])


def unchanged_by_probe(store: Store, yahoo: YahooClient, rows: Sequence[Any], now: datetime) -> set[int]:
    """Instrument ids whose newest stored bar already equals the vendor's newest daily point (nothing new to download).

    One spark request answers 20 symbols, so the ~33 global series cost 2 requests instead of 33 whenever their markets are
    closed and nothing changed. The probe can only ever SKIP a fetch: any doubt (a newer vendor date, a different close, a
    symbol the endpoint did not return, a failing probe, or a series not fetched in full for ``PROBE_MAX_AGE``) falls through
    to the normal per-symbol fetch, which also keeps split / re-adjustment detection working."""
    candidates: list[tuple[Any, str, str, float]] = []
    for row in rows:
        if row["kind"] == "equity":  # equities come from the official file; their vendor close need not match it
            continue
        fetched = parse_ts(store.kv_get(f"yahoo:fetched:{row['symbol']}"))
        if fetched is None or now - fetched > PROBE_MAX_AGE:
            continue
        last = store.conn.execute(
            "SELECT trade_date, close FROM bars_daily WHERE instrument_id = ? AND source_rank <= 1 "
            "ORDER BY trade_date DESC LIMIT 1", (row["instrument_id"],)).fetchone()
        if last is None or not last["close"]:
            continue
        candidates.append((row, row["yahoo_symbol"] or row["symbol"], last["trade_date"], float(last["close"])))
    if len(candidates) < PROBE_MIN_SERIES:
        return set()
    try:
        points = yahoo.spark_latest([ysym for _, ysym, _, _ in candidates])
    except Exception as exc:  # noqa: BLE001 - the probe is an optimisation: failing open costs requests, never data
        log.warning("bulk freshness probe failed (%s: %s); fetching the series one by one", type(exc).__name__, redact(str(exc)))
        return set()
    same: set[int] = set()
    for row, ysym, last_date, last_close in candidates:
        point = points.get(ysym)
        if point is None:
            continue
        try:
            zone: ZoneInfo | None = ZoneInfo(spec_for_instrument(row["symbol"], row["kind"]).tz)
        except Exception:  # noqa: BLE001 - unknown zone: be conservative
            zone = None
        if zone is None:
            continue
        point_date = point.ts.astimezone(zone).date().isoformat()
        if point_date <= last_date and math.isclose(point.close, last_close, rel_tol=PROBE_REL_TOL, abs_tol=1e-9):
            same.add(int(row["instrument_id"]))
    log.info("bulk freshness probe: %d of %d series unchanged (%d requests saved)", len(same), len(candidates), len(same))
    return same


def refresh_daily_bars(
    store: Store, yahoo: YahooClient, *, now: datetime, years: int, symbols: Iterable[str] | None = None,
    include_series: bool = True, include_equities: bool = True, deadline: float | None = None,
    clock: Clock | None = None, only_universe: bool = True, probe: bool = True,
) -> IngestReport:
    """Incrementally refresh bars for global series and the equity universe. ``deadline`` is a monotonic timestamp.
    ``probe`` first asks Yahoo's bulk endpoint which series have nothing new, and only fetches the others."""
    clk = clock or get_clock()
    wanted = set(symbols) if symbols is not None else None
    rows = []
    if include_series:
        rows += [r for r in store.list_instruments(status="active") if r["kind"] != "equity"]
    if include_equities:
        rows += store.list_instruments(kind="equity", in_universe=True if only_universe else None, status="active")
    rows = [r for r in sorted(rows, key=_order_key) if wanted is None or r["symbol"] in wanted]
    unchanged = unchanged_by_probe(store, yahoo, rows, now) if probe else set()
    report = IngestReport()
    for row in rows:
        if deadline is not None and clk.monotonic() > deadline:
            report.partial = True
            report.note("deadline reached; remaining symbols deferred")
            break
        if int(row["instrument_id"]) in unchanged:
            report.ok += 1
            report.unchanged += 1
            continue
        try:
            written = ingest_symbol_daily(store, yahoo, row, now=now, years=years)
        except SymbolNotFound:
            _handle_not_found(store, row, report)
            continue
        except CircuitOpen as exc:
            report.partial = True
            report.failed += 1
            report.note(f"circuit open: {exc}")
            break
        except DataUnavailable as exc:
            report.failed += 1
            report.note(f"{row['symbol']}: {redact(str(exc))}")
            continue
        except Exception as exc:  # noqa: BLE001 - one bad symbol must never stop the run
            report.failed += 1
            report.note(f"{row['symbol']}: {type(exc).__name__}")
            log.exception("unexpected error ingesting %s", row["symbol"])
            continue
        store.kv_delete(f"notfound:{row['symbol']}")
        store.kv_set(f"yahoo:fetched:{row['symbol']}", iso(now) or "")
        report.ok += 1
        report.rows += written
    store.record_source_result(
        "yahoo_daily", report.success_ratio >= 0.8 or report.ok == 0 and report.failed == 0,
        error=None if report.success_ratio >= 0.8 else f"{report.failed} of {report.ok + report.failed} symbols failed",
    )
    return report


def _handle_not_found(store: Store, row: Any, report: IngestReport) -> None:
    key = f"notfound:{row['symbol']}"
    count = int(store.kv_get(key, "0") or 0) + 1
    store.kv_set(key, str(count))
    report.failed += 1
    if count >= NOTFOUND_LIMIT and row["kind"] == "equity":
        store.set_instrument_status(row["symbol"], row["kind"], "quarantined", "unknown to Yahoo (renamed or delisted?)")
        store.conn.execute("UPDATE instruments SET in_universe = 0 WHERE instrument_id = ?", (row["instrument_id"],))
        report.note(f"{row['symbol']}: quarantined after {count} not-found responses")
        store.add_alert("warning", "symbol_quarantined", f"{row['symbol']} quarantined (not found on Yahoo)",
                        dedupe_key=f"quarantine:{row['symbol']}", dedupe_minutes=24 * 60)


def backfill_history(
    store: Store, yahoo: YahooClient, *, now: datetime, years: int, deadline: float | None = None,
    clock: Clock | None = None,
) -> IngestReport:
    """One-time deep history load per instrument; resumable (``backfill_progress``) and deadline bounded."""
    clk = clock or get_clock()
    report = IngestReport()
    rows = [r for r in store.list_instruments(status="active") if r["kind"] != "equity"]
    rows += store.list_instruments(kind="equity", in_universe=True, status="active")
    for row in sorted(rows, key=_order_key):
        key = f"yahoo_history:{row['symbol']}"
        progress = store.get_backfill(key)
        if progress and progress["status"] == "done":
            report.skipped += 1
            continue
        if deadline is not None and clk.monotonic() > deadline:
            report.partial = True
            report.note("deadline reached; backfill will resume")
            break
        try:
            store.set_backfill(key, "running")
            written = ingest_symbol_daily(store, yahoo, row, now=now, years=years, force_full=True)
        except SymbolNotFound:
            store.set_backfill(key, "failed", error="not found")
            _handle_not_found(store, row, report)
            continue
        except CircuitOpen as exc:
            store.set_backfill(key, "failed", error=redact(str(exc)))
            report.partial = True
            report.note(f"circuit open: {exc}")
            break
        except DataUnavailable as exc:
            store.set_backfill(key, "failed", error=redact(str(exc)))
            report.failed += 1
            report.note(f"{row['symbol']}: {redact(str(exc))}")
            continue
        store.set_backfill(key, "done", last_date=store.latest_bar_date(int(row["instrument_id"])))
        report.ok += 1
        report.rows += written
    return report


# ------------------------------------------------------------------------------------ official end-of-day
def ingest_bhavcopy(store: Store, archive: NseArchive, day: date) -> tuple[int, int]:
    """Write official OHLC/turnover/delivery for ``day``; returns (rows written, instruments matched).

    Official rows are stored RAW, exactly as traded that day (``features.loader`` re-bases the ones older than a split on the
    post-split basis when it reads them, see ``ingestion.adjust``). A row that validation rejects is logged and counted per
    symbol; an extreme but consistent one is kept with quality ``suspect``."""
    rows = archive.bhavcopy(day)
    if not rows:
        return 0, 0
    ids = {r["symbol"]: int(r["instrument_id"]) for r in store.list_instruments(kind="equity", status=None)}
    by_symbol: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["symbol"] not in ids:
            continue
        current = by_symbol.get(r["symbol"])
        if current is None or (r["series"] == "EQ" and current["series"] != "EQ"):
            by_symbol[r["symbol"]] = r
    written = 0
    rejected: list[tuple[str, str]] = []
    suspect: list[tuple[str, str]] = []
    for symbol, r in by_symbol.items():
        clean, report = validate_bars([r], kind="equity", keep_extreme=True)
        if not clean:
            rejected.append((symbol, report.rejected[0][1] if report.rejected else "rejected"))
            _count_rejected(store, symbol, 1)
            continue
        suspect += [(symbol, reason) for _, reason in report.suspect]
        bar = clean[0]
        bar.update({k: r.get(k) for k in ("prev_close", "turnover", "delivery_pct", "vwap", "trades")})
        written += store.upsert_bars(ids[symbol], [bar], source="nse_bhavcopy", source_rank=2, quality="official")
    if rejected:
        log.warning("bhavcopy %s: %d of %d rows rejected by validation (%s)", day, len(rejected), len(by_symbol), _quote(rejected))
    if suspect:
        log.info("bhavcopy %s: %d extreme but consistent rows kept as suspect (%s)", day, len(suspect), _quote(suspect))
    store.record_source_result("nse_bhavcopy", True, data_ts=day.isoformat())
    return written, len(by_symbol)


def fno_symbols(store: Store) -> set[str]:
    return {r["symbol"] for r in store.list_instruments(kind="equity", status=None, fno=True)}


def ingest_fno_day(store: Store, archive: NseArchive, day: date) -> int:
    rows = archive.fo_bhavcopy(day)
    if not rows:
        return 0
    aggregated = aggregate_stock_fno(rows, day, fno_symbols(store) or None)
    written = store.upsert_fno(aggregated)
    store.record_source_result("nse_fo_bhavcopy", True, data_ts=day.isoformat())
    return written


def backfill_fno(
    store: Store, archive: NseArchive, days: Sequence[date], *, deadline: float | None = None,
    clock: Clock | None = None, on_day: Callable[[date, int], None] | None = None,
) -> IngestReport:
    """Load missing F&O days (newest first so recent history is available early). Resumable and deadline bounded."""
    clk = clock or get_clock()
    report = IngestReport()
    have = {r["trade_date"] for r in store.conn.execute("SELECT DISTINCT trade_date FROM fno_daily").fetchall()}
    for day in sorted(days, reverse=True):
        iso_day = day.isoformat()
        if iso_day in have:
            report.skipped += 1
            continue
        key = f"fno:{iso_day}"
        progress = store.get_backfill(key)
        if progress and progress["status"] == "missing" and progress["attempts"] >= 3:
            report.skipped += 1
            continue
        if deadline is not None and clk.monotonic() > deadline:
            report.partial = True
            report.note("deadline reached; F&O backfill will resume")
            break
        try:
            written = ingest_fno_day(store, archive, day)
        except CircuitOpen as exc:
            report.partial = True
            report.note(f"circuit open: {exc}")
            break
        except DataUnavailable as exc:
            report.failed += 1
            report.note(f"{iso_day}: {redact(str(exc))}")
            continue
        if written:
            store.set_backfill(key, "done", last_date=iso_day)
            report.ok += 1
            report.rows += written
        else:
            store.set_backfill(key, "missing")
            report.skipped += 1
        if on_day:
            on_day(day, written)
    return report


# ---------------------------------------------------------------------------------------- live collection
def fresh_preopen_rows(snapshot: PreopenSnapshot, trade_date: str) -> list[dict[str, Any]]:
    """The snapshot's rows that are stamped on ``trade_date`` (IST). Outside the pre-open session, or from a stale cache, the NSE
    endpoint serves the PREVIOUS session's book; stored under today's date it would turn yesterday's equilibrium prices into a
    bogus opening gap. A row falls back to the snapshot's own timestamp; with no timestamp at all it cannot be judged and is kept."""
    day = date.fromisoformat(trade_date)
    snapshot_stamp = parse_ts(snapshot.source_ts, assume_tz=IST)
    fresh = []
    for row in snapshot.rows:
        stamp = parse_ts(row.get("row_ts"), assume_tz=IST) or snapshot_stamp
        if stamp is None or ist_date(stamp) == day:
            fresh.append(row)
    return fresh


def collect_preopen(store: Store, nse: NseApi, trade_date: str, symbols: Iterable[str], *,
                    deadline: float | None = None) -> PreopenSnapshot | None:
    """Fetch the NSE pre-open book once and append today's rows (with source and received timestamps) to ``preopen_quotes``.
    A book that is not from ``trade_date`` yet counts as "not published": the returned snapshot has no rows. ``deadline`` is a
    ``clock.monotonic()`` instant that bounds the HTTP attempts."""
    try:
        snapshot = nse.preopen("ALL", set(symbols), deadline=deadline)
    except DataUnavailable as exc:
        store.record_source_result("nse_api_market", False, error=redact(str(exc)))
        log.warning("pre-open collection failed: %s", redact(str(exc)))
        return None
    rows = fresh_preopen_rows(snapshot, trade_date)
    if len(rows) < len(snapshot.rows):
        log.info("ignoring %d pre-open rows that are not from %s (book stamped %s)", len(snapshot.rows) - len(rows), trade_date,
                 snapshot.source_ts)
    if rows:
        store.add_preopen_quotes(trade_date, rows, snapshot.source_ts)
        store.record_source_result("nse_api_market", True, data_ts=snapshot.source_ts)
    return PreopenSnapshot(snapshot.source_ts, rows)


def collect_gift(store: Store, nse: NseApi, *, deadline: float | None = None) -> dict[str, Any] | None:
    try:
        gift = nse.gift_nifty(deadline=deadline)
    except DataUnavailable as exc:
        store.record_source_result("nse_api_market", False, error=redact(str(exc)))
        return None
    if gift:
        store.add_live_quote("GIFT_NIFTY", price=gift["price"], prev_close=gift.get("prev_close"),
                             change_pct=gift.get("change_pct"), source_ts=gift.get("source_ts"),
                             extra={"expiry": gift.get("expiry")})
        store.record_source_result("nse_api_market", True, data_ts=gift.get("source_ts"))
    return gift


LIVE_CONTEXT = ("^N225", "^HSI", "^KS11", "000001.SS", "ES=F", "NQ=F", "CL=F", "BZ=F", "USDINR=X")


def collect_live_equities(store: Store, nse: NseApi, lists: Sequence[str], symbols: Iterable[str], *,
                          deadline: float | None = None, errors: list[str] | None = None) -> LiveEquities | None:
    """Today's running figures of ``symbols`` from the stock lists ``lists`` (one call each; the first list that has a symbol
    wins). ``None`` when no list answered. A list that fails is logged and skipped, so a broad second list never costs the
    answer of the first. The outcome is recorded in ``source_health`` as ``nse_live_quotes``; the reason of every failure is
    appended to ``errors`` (redacted), so the caller can tell the operator why there is no answer."""
    wanted = set(symbols)
    merged: dict[str, dict[str, Any]] = {}
    stamps: list[str] = []
    index: dict[str, Any] | None = None
    answered = False
    for name in lists:
        try:
            snapshot = nse.live_equities(name, wanted, deadline=deadline)
        except DataUnavailable as exc:
            reason = redact(str(exc))
            store.record_source_result("nse_live_quotes", False, error=reason)
            log.warning("live quotes of %s failed: %s", name, reason)
            if errors is not None:
                errors.append(reason)
            continue
        answered = True
        for row in snapshot.rows:
            merged.setdefault(row["symbol"], row)
        if snapshot.source_ts:
            stamps.append(snapshot.source_ts)
        index = index or snapshot.index
    if not answered:
        return None
    latest = max(stamps) if stamps else None
    store.record_source_result("nse_live_quotes", bool(merged), data_ts=latest, error=None if merged else "no stock rows")
    return LiveEquities(latest, list(merged.values()), index)


def collect_live_indices(store: Store, nse: NseApi, *, deadline: float | None = None) -> dict[str, dict[str, Any]]:
    """Nifty 50, India VIX and the other indices live; ``{}`` when NSE does not answer (the scan then reads the market from the
    stocks themselves)."""
    try:
        indices = nse.all_indices(deadline=deadline)
    except DataUnavailable as exc:
        store.record_source_result("nse_live_quotes", False, error=redact(str(exc)))
        log.warning("live indices failed: %s", redact(str(exc)))
        return {}
    return indices


TOKENS_KEY = "kotak:tokens"  # {"day", "tokens": {symbol: token}, "absent": [symbols the broker does not list]}


def _recorded_tokens(stored: Any) -> tuple[dict[str, str], set[str], str | None]:
    """The map recorded in the database as ``(tokens, absent, day)``. Anything that is not exactly the shape ``kotak_tokens`` writes (an older
    release, a hand edit, a damaged row) counts as no map at all, so it can neither crash a scan nor stand in for a master that is down."""
    if not isinstance(stored, dict):
        return {}, set(), None
    tokens, absent, day = stored.get("tokens"), stored.get("absent"), stored.get("day")
    if not (isinstance(tokens, dict) and isinstance(absent, list) and isinstance(day, str)):
        return {}, set(), None
    if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in tokens.items()) or not all(isinstance(s, str) for s in absent):
        return {}, set(), None
    return dict(tokens), set(absent), day


def kotak_tokens(store: Store, api: KotakApi, symbols: Iterable[str], today: date, *, deadline: float | None = None) -> dict[str, str]:
    """``{symbol: instrument token}`` of the ``symbols`` the broker lists. The map is kept in the database and re-read from the broker's
    instrument master (a file of about 3 MB) once a day, or at once when a wanted name is neither known nor recorded as not listed. A
    token does not change from one day to the next, so when the master cannot be read an older map still serves, but only when it holds a
    token of at least one wanted name: otherwise the master's own failure is raised (it is the real reason there is no answer)."""
    wanted = set(symbols)
    known, absent, day = _recorded_tokens(store.kv_get_json(TOKENS_KEY))
    if day != today.isoformat() or wanted - known.keys() - absent:
        try:
            master = api.token_map(deadline=deadline)
        except DataUnavailable as exc:
            if not wanted & known.keys():
                raise
            log.warning("kotak instrument master unavailable (%s): the map of %s serves", redact(str(exc)), day)
        else:
            known = {symbol: master.get(symbol, token) for symbol, token in known.items()}
            known.update({symbol: master[symbol] for symbol in wanted if symbol in master})
            store.kv_set_json(TOKENS_KEY, {"day": today.isoformat(), "tokens": known, "absent": sorted(wanted - master.keys())})
    return {symbol: known[symbol] for symbol in wanted if symbol in known}


def collect_kotak_quotes(store: Store, api: KotakApi, symbols: Iterable[str], today: date, *, deadline: float | None = None,
                         errors: list[str] | None = None) -> KotakQuotes | None:
    """Live rows of ``symbols`` from the broker (about 40 stocks per request). ``None`` when nothing could be read; the reason is logged,
    recorded in ``source_health`` as ``kotak_quotes`` and appended (redacted) to ``errors``. A partial answer is returned as it is: the
    names it lacks are simply names this feed does not have. The broker is an extra: an unexpected error in its handling is a failure of
    this feed only, never of the scan that has NSE's rows."""
    started = time.monotonic()
    try:
        tokens = kotak_tokens(store, api, symbols, today, deadline=deadline)
        if not tokens:
            raise DataUnavailable("kotak_quotes", "the broker lists none of the requested symbols", retryable=False)
        quotes = api.quotes(tokens, deadline=deadline)
    except DataUnavailable as exc:
        reason = redact(str(exc))
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.exception("broker live quotes failed unexpectedly")
        reason = f"unexpected {type(exc).__name__}: {redact(str(exc))[:120]}"
    else:
        store.record_source_result("kotak_quotes", bool(quotes.rows), latency_ms=(time.monotonic() - started) * 1000.0,
                                   data_ts=iso(quotes.asof), error=None if quotes.rows else "no stock answered")
        if quotes.failed:
            log.warning("broker live quotes: %d of %d stocks could not be read", len(quotes.failed), quotes.asked)
        return quotes
    store.record_source_result("kotak_quotes", False, error=reason)
    log.warning("broker live quotes failed: %s", reason)
    if errors is not None:
        errors.append(reason)
    return None


def collect_kotak_indices(store: Store, api: KotakApi, *, deadline: float | None = None) -> dict[str, dict[str, Any]]:
    """Nifty 50 and India VIX from the broker; ``{}`` when it does not answer. Used only when NSE's own index call gave nothing."""
    try:
        return api.indices(deadline=deadline)
    except DataUnavailable as exc:
        reason = redact(str(exc))
    except Exception as exc:  # noqa: BLE001 - the broker is an extra, as in collect_kotak_quotes
        log.exception("broker index quotes failed unexpectedly")
        reason = f"unexpected {type(exc).__name__}: {redact(str(exc))[:120]}"
    store.record_source_result("kotak_quotes", False, error=reason)
    log.warning("broker index quotes failed: %s", reason)
    return {}


def collect_live_context(store: Store, yahoo: YahooClient, keys: Sequence[str] = LIVE_CONTEXT,
                         deadline: float | None = None, clock: Clock | None = None) -> int:
    """Descriptive overnight context (Asia live moves, US futures). Never a model input, never blocks a report."""
    clk = clock or get_clock()
    stored = 0
    for key in keys:
        if deadline is not None and clk.monotonic() > deadline:
            break
        try:
            quote = yahoo.fetch_quote(key)
        except DataUnavailable:
            continue
        if quote.get("price"):
            store.add_live_quote(f"LIVE:{key}", price=quote["price"], prev_close=quote.get("prev_close"),
                                 change_pct=quote.get("change_pct"), source_ts=quote.get("source_ts"))
            stored += 1
    store.record_source_result("yahoo_quote", stored > 0 or not keys)
    return stored


def monotonic_deadline(seconds: float, clock: Clock | None = None) -> float:
    return (clock or get_clock()).monotonic() + seconds
