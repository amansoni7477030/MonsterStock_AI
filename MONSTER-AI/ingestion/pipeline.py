"""Ingestion orchestration: bars, official end-of-day files, F&O and live pre-open data.

Principles: (1) a bar is stored only once its session is final (point-in-time availability rule); (2) official
exchange files outrank vendor data; (3) every loop is bounded by a deadline and resumable; (4) one bad symbol never
stops the run; (5) vendor re-adjustments (splits/bonus) are detected and the history reloaded.
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
from ingestion.nse import NseApi, NseArchive, PreopenSnapshot
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
def _store_series(store: Store, row: Any, bars: list[dict[str, Any]], series: Any) -> int:
    written = store.upsert_bars(int(row["instrument_id"]), bars, source="yahoo", source_rank=1)
    for split in series.splits:
        ratio = split["numerator"] / split["denominator"] if split["denominator"] else None
        store.add_corporate_action(int(row["instrument_id"]), "split", split["ex_date"], ratio=ratio, source="yahoo")
    for dividend in series.dividends[-12:]:
        store.add_corporate_action(int(row["instrument_id"]), "dividend", dividend["ex_date"], amount=dividend["amount"],
                                   source="yahoo")
    return written


def ingest_symbol_daily(
    store: Store, yahoo: YahooClient, row: Any, *, now: datetime, years: int, force_full: bool = False
) -> int:
    """Fetch, validate, PIT-filter and store one instrument; reload full history if the vendor re-adjusted it."""
    iid, symbol, kind = int(row["instrument_id"]), row["symbol"], row["kind"]
    ysym = row["yahoo_symbol"] or symbol
    spec = spec_for_instrument(symbol, kind)
    last = store.latest_bar_date(iid)
    full = force_full or last is None
    rng = range_for_years(years) if full else choose_range(last, ist_date(now), years)
    series = yahoo.fetch_daily(ysym, range_=rng)
    bars, report = validate_bars(series.bars, kind=kind)
    bars = pit_filter(bars, spec, now)
    if report.rejected_count > max(3, len(series.bars) // 5):
        log.warning("%s: %d of %d bars rejected by validation", symbol, report.rejected_count, len(series.bars))
    if not bars:
        return 0

    if not full:
        stored = {
            r["trade_date"]: r["close"]
            for r in store.conn.execute(
                "SELECT trade_date, close FROM bars_daily WHERE instrument_id = ? AND source_rank <= 1 "
                "AND trade_date BETWEEN ? AND ?", (iid, bars[0]["trade_date"], bars[-1]["trade_date"]),
            ).fetchall()
            if r["close"]
        }
        earliest = _earliest(store, iid)
        new_split = False
        for split in series.splits:
            ratio = split["numerator"] / split["denominator"] if split["denominator"] else None
            is_new = store.add_corporate_action(iid, "split", split["ex_date"], ratio=ratio, source="yahoo")
            new_split = new_split or (is_new and split["ex_date"] > earliest)
        if new_split or not overlap_consistent(stored, bars):
            log.warning("%s: vendor history changed (split/re-adjustment); reloading full history", symbol)
            store.delete_vendor_bars(iid)
            return ingest_symbol_daily(store, yahoo, row, now=now, years=years, force_full=True)
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
    """Write official OHLC/turnover/delivery for ``day``; returns (rows written, instruments matched)."""
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
    for symbol, r in by_symbol.items():
        clean, _report = validate_bars([r], kind="equity")
        if not clean:
            continue
        bar = clean[0]
        bar.update({k: r.get(k) for k in ("prev_close", "turnover", "delivery_pct", "vwap", "trades")})
        written += store.upsert_bars(ids[symbol], [bar], source="nse_bhavcopy", source_rank=2, quality="official")
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


def collect_preopen(store: Store, nse: NseApi, trade_date: str, symbols: Iterable[str]) -> PreopenSnapshot | None:
    """Fetch the NSE pre-open book once and append today's rows (with source and received timestamps) to ``preopen_quotes``.
    A book that is not from ``trade_date`` yet counts as "not published": the returned snapshot has no rows."""
    try:
        snapshot = nse.preopen("ALL", set(symbols))
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


def collect_gift(store: Store, nse: NseApi) -> dict[str, Any] | None:
    try:
        gift = nse.gift_nifty()
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


def sleep_budget(seconds: float) -> None:  # pragma: no cover - trivial wrapper used by CLI progress loops
    time.sleep(seconds)
