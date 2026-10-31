"""Source register (blueprint section 3): for every source record provider, endpoint, coverage, timestamp meaning,
update frequency, history depth, rate limits, commercial-use terms, failure behaviour and backup source.

This is the single source of truth behind ``sources`` in the database and docs/DATA_SOURCES.md. Free endpoints are
NOT assumed to be stable, complete or licensed for production trading use; ``license_notes`` says so explicitly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.errors import DataUnavailable
from app.logging_setup import redact

log = logging.getLogger(__name__)

UNLICENSED = (
    "Unofficial / public endpoint: suitable for research and shadow operation; confirm the provider's terms before "
    "any commercial or redistributed use and prefer a licensed feed for production."
)
EXCHANGE_PUBLIC = (
    "Published by NSE for public download/consumption; automated bulk use is subject to NSE terms - confirm "
    "entitlement for production; a licensed NSE data feed or broker API is the supported route."
)

SOURCES: tuple[dict[str, Any], ...] = (
    dict(
        source_id="yahoo_daily", provider="Yahoo Finance (unofficial chart API)",
        endpoint="https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=..&interval=1d",
        coverage="NSE equities (.NS), Indian indices, global indices, FX, rates, commodity futures",
        timestamp_meaning="Bar start in exchange-local time; final only after the session close (partial current bar is dropped)",
        update_frequency="Daily bars final after each local close", history_depth="10+ years (split-adjusted)",
        rate_limits="Undocumented; ~1 request/second sustained; HTTP 429 on bursts", license_notes=UNLICENSED,
        failure_behavior="Bounded retries with backoff; circuit breaker; NSE bhavcopy overrides vendor OHLC; stale data blocks or downgrades the report",
        backup_source="nse_bhavcopy", criticality="critical",
    ),
    dict(
        source_id="yahoo_quote", provider="Yahoo Finance (unofficial chart API)",
        endpoint="https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=5m",
        coverage="Live/recent intraday bars and last price (Asia open moves, US futures)",
        timestamp_meaning="Last trade time reported by the exchange (UTC epoch)", update_frequency="Real-time to ~15 min delayed",
        history_depth="~60 days of 5m bars", rate_limits="Same as yahoo_daily", license_notes=UNLICENSED,
        failure_behavior="Descriptive context only; absence never blocks a report", backup_source=None, criticality="optional",
    ),
    dict(
        source_id="nse_bhavcopy", provider="NSE archives (sec_bhavdata_full)",
        endpoint="https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
        coverage="All NSE cash-market securities: official OHLC, VWAP, volume, turnover, trades, delivery %",
        timestamp_meaning="End-of-day official values for the trade date", update_frequency="Daily, published ~17:30-19:30 IST",
        history_depth="Years (one file per day)", rate_limits="No published limit; keep >= 1 s spacing",
        license_notes=EXCHANGE_PUBLIC,
        failure_behavior="404 = not yet published (retry later); outcomes fall back to provisional vendor data and are flagged unvalidated",
        backup_source="yahoo_daily", criticality="important",
    ),
    dict(
        source_id="nse_fo_bhavcopy", provider="NSE archives (UDiFF F&O bhavcopy)",
        endpoint="https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip",
        coverage="Stock/index futures and options: OHLC, settle, OI, change in OI, volume per contract",
        timestamp_meaning="End-of-day for the trade date", update_frequency="Daily after close",
        history_depth="From 2024-07-08 (UDiFF format)", rate_limits="Keep >= 1 s spacing; ~1 MB per file",
        license_notes=EXCHANGE_PUBLIC,
        failure_behavior="F&O features become missing (NaN) and the F&O specialist abstains; flagged in the report", backup_source=None,
        criticality="important",
    ),
    dict(
        source_id="nse_lists", provider="NSE archives (index constituents, F&O lot sizes)",
        endpoint="https://nsearchives.nseindia.com/content/indices/ind_nifty*list.csv ; /content/fo/fo_mktlots.csv",
        coverage="Nifty 100/200/500 constituents with industry + ISIN; F&O eligible underlyings and lot sizes",
        timestamp_meaning="Latest published constituent list", update_frequency="On index reconstitution (semi-annual) / expiry cycles",
        history_depth="Current list only (survivorship bias in history is documented)", rate_limits="Rarely called (weekly)",
        license_notes=EXCHANGE_PUBLIC, failure_behavior="Keep the previously stored universe; seed file used on first install",
        backup_source="seed_universe", criticality="important",
    ),
    dict(
        source_id="nse_api_market", provider="NSE public JSON API (marketStatus, pre-open, option chain, holidays)",
        endpoint="https://www.nseindia.com/api/*", coverage="GIFT Nifty, indicative/final pre-open prices and order imbalance, live option chain, trading holidays",
        timestamp_meaning="Server timestamps in IST (parsed with an explicit IST assumption)", update_frequency="Live during pre-open / session",
        history_depth="Live only (not point-in-time reconstructable)", rate_limits="Cookie handshake required; ~1 request/second; frequent 401/403 from datacentre IPs",
        license_notes=EXCHANGE_PUBLIC + " Endpoints are undocumented and may change without notice.",
        failure_behavior="Pre-open features missing -> falls back to the preliminary model with a data-quality warning; never silently reuses stale quotes",
        backup_source="yahoo_quote", criticality="important",
    ),
    dict(
        source_id="nse_announcements", provider="NSE corporate announcements API",
        endpoint="https://www.nseindia.com/api/corporate-announcements ; /api/event-calendar",
        coverage="Exchange filings (results, orders, rating actions, M&A, management changes) and board-meeting calendar",
        timestamp_meaning="exchdisstime = exchange dissemination time (IST) - treated as the public availability time",
        update_frequency="Continuous", history_depth="Weeks (live capture builds the point-in-time archive)",
        rate_limits="Same session limits as nse_api_market", license_notes=EXCHANGE_PUBLIC,
        failure_behavior="Coverage gap flagged; scheduled-results features default to 0", backup_source="rss_news", criticality="important",
    ),
    dict(
        source_id="rss_news", provider="Publisher RSS/Atom feeds (data/seeds/feeds.json)",
        endpoint="Per feed", coverage="Indian and global business, markets, geopolitics, central banks, regulators",
        timestamp_meaning="Publisher timestamp when present; first_seen_at is always recorded separately",
        update_frequency="Minutes to hours", history_depth="Feed window only (items older than the lookback are ignored)",
        rate_limits="Polite polling (<= 1 request per feed per run)", license_notes="Headlines/summaries only are stored for classification; respect each publisher's terms.",
        failure_behavior="Per-feed circuit breaker; coverage gap flagged; other feeds continue", backup_source="gdelt", criticality="important",
    ),
    dict(
        source_id="gdelt", provider="GDELT Project DOC 2.0 API", endpoint="https://api.gdeltproject.org/api/v2/doc/doc",
        coverage="Global news discovery (titles, domains, time, language)", timestamp_meaning="seendate (UTC)",
        update_frequency="15 minutes", history_depth="Rolling ~3 months via API", rate_limits="<= 1 request per 5 s; 429 on bursts",
        license_notes="Free research use; discovery only - verify facts against primary sources.",
        failure_behavior="Backoff on 429; skipped queries recorded as coverage gap", backup_source="rss_news", criticality="optional",
    ),
    dict(
        source_id="fred", provider="Federal Reserve Bank of St. Louis (FRED/ALFRED)",
        endpoint="https://api.stlouisfed.org/fred/series/observations", coverage="US rates and macro series with vintages when an API key is set",
        timestamp_meaning="observation_date + realtime_start (first-release timestamp)", update_frequency="Per release",
        history_depth="Decades", rate_limits="120 requests/minute with key", license_notes="Free with an API key; see FRED terms.",
        failure_behavior="Optional; market-based yields come from Yahoo; revised values are never used as point-in-time history",
        backup_source="yahoo_daily", criticality="optional",
    ),
    dict(
        source_id="econ_calendar", provider="Public economic-calendar mirror (faireconomy JSON)",
        endpoint="https://nfs.faireconomy.media/ff_calendar_thisweek.json", coverage="Scheduled macro releases with impact tags",
        timestamp_meaning="Scheduled release time with UTC offset", update_frequency="Daily", history_depth="Current week",
        rate_limits="Fetch at most a few times per day", license_notes=UNLICENSED,
        failure_behavior="Optional: event-day regime flags default to 0", backup_source=None, criticality="optional",
    ),
    dict(
        source_id="llm_extraction", provider="Configured OpenAI-compatible LLM API", endpoint="LLM_API_BASE/chat/completions",
        coverage="Structured event extraction from news text", timestamp_meaning="n/a", update_frequency="On demand (cost-capped)",
        history_depth="n/a", rate_limits="Provider specific; per-run article budget", license_notes="Governed by the chosen provider's terms; no secrets/PII are sent.",
        failure_behavior="Deterministic rule-based extraction; report states reduced coverage", backup_source="rules", criticality="optional",
    ),
)


def register_all(store: Any) -> int:
    for row in SOURCES:
        store.register_source(**row)
    return len(SOURCES)


def render_markdown() -> str:
    """docs/DATA_SOURCES.md, generated from ``SOURCES`` so the documentation can never drift from the code."""
    lines = [
        "# Data sources",
        "",
        "Generated from `ingestion/registry.py` (run `python scripts/generate_docs.py`). Every source records its provider, "
        "endpoint, coverage, timestamp meaning, update frequency, history depth, rate limits, licensing, failure behaviour "
        "and backup. Free endpoints are **not** assumed to be stable, complete or licensed for production trading use.",
        "",
        "| Source | Criticality | Provider | Backup |",
        "|---|---|---|---|",
    ]
    for s in SOURCES:
        lines.append(f"| `{s['source_id']}` | {s['criticality']} | {s['provider']} | {s['backup_source'] or '-'} |")
    for s in SOURCES:
        lines += [
            "", f"## `{s['source_id']}` - {s['provider']}", "",
            f"- **Endpoint:** `{s['endpoint']}`", f"- **Coverage:** {s['coverage']}",
            f"- **Timestamp meaning:** {s['timestamp_meaning']}", f"- **Update frequency:** {s['update_frequency']}",
            f"- **History depth:** {s['history_depth']}", f"- **Rate limits:** {s['rate_limits']}",
            f"- **Licensing:** {s['license_notes']}", f"- **Failure behaviour:** {s['failure_behavior']}",
            f"- **Backup source:** {s['backup_source'] or 'none'}",
        ]
    return "\n".join(lines) + "\n"


@contextmanager
def tracked(store: Any, source_id: str) -> Iterator[None]:
    """Record success/failure + latency of an external call in ``source_health`` (never masks the exception)."""
    started = time.monotonic()
    try:
        yield
    except DataUnavailable as exc:
        _safe(store, source_id, False, (time.monotonic() - started) * 1000.0, redact(str(exc)))
        raise
    except Exception as exc:
        _safe(store, source_id, False, (time.monotonic() - started) * 1000.0, redact(f"{type(exc).__name__}: {exc}"))
        raise
    else:
        _safe(store, source_id, True, (time.monotonic() - started) * 1000.0, None)


def _safe(store: Any, source_id: str, ok: bool, latency_ms: float, error: str | None) -> None:
    try:
        store.record_source_result(source_id, ok, latency_ms=latency_ms, error=error)
    except Exception:  # noqa: BLE001  # health bookkeeping must never break ingestion
        log.debug("could not record health for %s", source_id, exc_info=True)
