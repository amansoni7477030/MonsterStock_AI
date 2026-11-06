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
        coverage="Last price and previous close (Asia open moves, US futures); in the evening also the 5-minute bars of the few names listed in the 09:30 opening read, to judge its reference levels when the broker (`kotak_history`) has no complete day for them (never a model input)",
        timestamp_meaning="Last trade time reported by the exchange (UTC epoch); each 5-minute bar is dated and timed in exchange-local time", update_frequency="Real-time to ~15 min delayed",
        history_depth="Latest quote only (the endpoint serves ~60 days of 5m bars; the evening judging reads the last 5 sessions of the listed names and stores only the outcome; the feed omits the opening auction's volume and some closing bars)", rate_limits="Same as yahoo_daily", license_notes=UNLICENSED,
        failure_behavior="Descriptive context only; absence never blocks a report (a night without 5-minute bars judges the opening reads on the daily bars only)", backup_source=None, criticality="optional",
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
        source_id="nse_api_market", provider="NSE public JSON API (marketStatus, pre-open, holidays)",
        endpoint="https://www.nseindia.com/api/*", coverage="GIFT Nifty, indicative/final pre-open prices and order imbalance, trading holidays",
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
        source_id="nse_live_quotes", provider="NSE public JSON API (equity-stockIndex / equity-stock-indices, allIndices)",
        endpoint="https://www.nseindia.com/api/equity-stockIndex?index=SECURITIES%20IN%20F%26O ; /api/equity-stock-indices?index=NIFTY%20500 ; /api/allIndices",
        coverage="Every F&O stock (or a broader index list) in ONE call: today's open, high, low, last price, previous close, volume so far, plus the 52-week range; Nifty 50, India VIX and the sector indices live",
        timestamp_meaning="the answer's own `timestamp`, exchange time in IST (parsed with an explicit IST assumption) and the stamp of every row; the per-stock `lastUpdateTime` is a placeholder (31-DEC-2999) or the date of an old update and is NOT read",
        update_frequency="Live during the session (seconds to a minute behind); the final values after the close",
        history_depth="Live only (not point-in-time reconstructable): every trade idea and every opening read freezes the figures it was based on",
        rate_limits="Same session as nse_api_market: cookie handshake, ~1 request/second; the 09:30 opening read and the closing-hour scan each make two calls; frequent 401/403 from datacentre IPs",
        license_notes=EXCHANGE_PUBLIC + " Endpoints are undocumented and may change without notice (NSE retired `equity-stockIndices` in 2026).",
        failure_behavior="The 09:30 opening read and the closing-hour scan fail visibly and retry inside their windows (nothing is guessed from older data, and a snapshot that is not from today is refused) unless the Kotak broker feed (`kotak_quotes`) is on: it then carries the scan alone; the pre-open morning scans do not use it",
        backup_source="kotak_quotes", criticality="optional",
    ),
    dict(
        source_id="kotak_quotes", provider="Kotak Neo market-data API (broker): live quotes and the instrument master",
        endpoint="https://mis.kotaksecurities.com/script-details/1.0/quotes/neosymbol/nse_cm|{token},nse_cm|{token}/all ; /script-details/1.0/masterscrip/file-paths ; the nse_cm instrument master CSV it lists (header Authorization: the application token)",
        coverage="Every F&O stock live in a handful of requests: open, high, low, last price, previous close, the day's volume, VWAP and the 52-week range with each stock's own last-update time; Nifty 50 and India VIX by name. After the close all 218 F&O stocks matched NSE's own rows exactly on open, high, low, last, previous close and volume",
        timestamp_meaning="`lstup_time`: UTC epoch seconds of each stock's last update (NSE's answer has one stamp for every row); `ohlc.close` is the PREVIOUS close; the age of the freshest trade of an answer is the age of the feed",
        update_frequency="Live during the session (exchange-fed; its age is measured on every answer and held against it, `KOTAK_MAX_LAG_SEC`); the final figures after the close",
        history_depth="Live only (not point-in-time reconstructable): every trade idea and every opening read freezes the figures it was based on",
        rate_limits="Documented 25 requests a second and 50 stocks a request; measured live: a request for exactly 50 is refused with HTTP 400 (49 works), so 40 are asked, 0.2 s apart; HTTP 429 slows the client down",
        license_notes="Broker market data for the account holder's own use under the broker's API terms (confirm them before any wider use). Only the application token is used and sent: no login, no orders, no personal data.",
        failure_behavior="NSE alone carries the scan, exactly as before the broker existed; a name the two feeds disagree on is left out and counted; a broker feed whose freshest trade is older than `KOTAK_MAX_LAG_SEC` during the session is not used; with NSE down the broker alone carries the 09:30 and 14:30 scans",
        backup_source="nse_live_quotes", criticality="optional",
    ),
    dict(
        source_id="kotak_history", provider="Kotak Neo market-data API (broker): historical candles",
        endpoint="https://mis.kotaksecurities.com/market-data/1.0/historical/details?neosymbol=nse_cm|{token}&interval=5min&fromdate=YYYY-MM-DD&todate=YYYY-MM-DD (header Authorization: the application token)",
        coverage="1, 3, 5, 10, 15, 30 and 60-minute, daily and weekly candles of every NSE stock: open, high, low, close and volume (the first bar carries the opening auction's volume, which Yahoo's lacks). Built from the exchange's own trades: the five-minute candles are exactly the aggregate of the one-minute ones and the day's low matched NSE's official low for 60 of 60 stocks, where Yahoo's first-15-minute range is about a tenth narrower",
        timestamp_meaning="the bar's START, ISO 8601 with +0530; a session normally ends with the 15:10 five-minute bar (the last 15 minutes are missing, or appear only as a flat artefact after the day, and on some days the 15:10 bar carries a negative volume)",
        update_frequency="Complete for the day in the evening; used after the close only",
        history_depth="Years (a 2023 window answered); at most 30 days per request for 1, 3 and 5-minute candles",
        rate_limits="About one request a second (a second request within the second is refused with HTTP 429), so one request a name; the client spaces them `KOTAK_HISTORY_INTERVAL_SEC` apart and slows down on 429",
        license_notes="Broker market data for the account holder's own use under the broker's API terms (confirm them before any wider use). Only the application token is used and sent.",
        failure_behavior="A name whose broker day is missing or unfinished is judged on Yahoo's bars; a night with neither judges the opening reads on the daily bars only",
        backup_source="yahoo_quote", criticality="optional",
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
        failure_behavior="Backoff on 429; after two failed queries in a row the rest of the run goes to the Google News search fallback "
                         "(`gnews_search`), and a topic GDELT answers with nothing is read from there too; a topic neither answers is "
                         "recorded as a coverage gap", backup_source="gnews_search", criticality="optional",
    ),
    dict(
        source_id="gnews_search", provider="Google News search (RSS output)", endpoint="https://news.google.com/rss/search?q=<query> when:Nd&hl=en-US&gl=US&ceid=US:en",
        coverage="The same global topics as GDELT (oil, rates and inflation, geopolitics, China, India macro, the Middle East, semiconductors, markets): headlines, publisher and time; no article text",
        timestamp_meaning="the item's publication time as Google News reports it (UTC); first_seen_at is recorded separately",
        update_frequency="Minutes", history_depth="The `when:` window of the run (about 36 h); a search returns up to 100 items and the first 60 are kept",
        rate_limits="Undocumented; one request per topic and run (eight a run), well inside what a browser makes",
        license_notes="Headlines and links only, as for the other aggregator feeds; no key. Not an official API: Google may change the feed or ask for consent, in which case the answer is rejected as unreadable and the topic is a coverage gap.",
        failure_behavior="The topic is skipped and counted as a coverage gap; the news job and the RSS feeds go on", backup_source="gdelt", criticality="optional",
    ),
    dict(
        source_id="llm_extraction", provider="Groq (open-weight gpt-oss models) or any OpenAI-compatible LLM API",
        endpoint="https://api.groq.com/openai/v1/chat/completions (GROQ_API_KEY) or LLM_API_BASE/chat/completions",
        coverage="Structured event extraction from news text: event type, entities with their direction, severity, horizon, event time and a verbatim evidence quote",
        timestamp_meaning="n/a", update_frequency="On demand (cost-capped: per-run call and time budgets, 14-day result cache)",
        history_depth="n/a",
        rate_limits="Groq free tier (measured live): 8,000 tokens per minute, 200,000 tokens per day and 1,000 requests per day for each model, about 2,000 tokens per extraction, so roughly 4 extractions per minute and about 100 a day per model; every model of the chain (gpt-oss-120b, gpt-oss-20b, qwen3.8-27b by default, `LLM_FALLBACK_MODELS` adds more) has its own allowance, a rate-limited model hands over to the next one at once and is used again once its limit has reset, a short Retry-After is waited out when every model is limited, a long one starts a cooldown that survives between runs",
        license_notes="Governed by the chosen provider's terms; only public news text is sent - never keys, positions or other data.",
        failure_behavior="Deterministic rule-based extraction; report states reduced coverage; repeated failures appear in source health",
        backup_source="rules", criticality="optional",
    ),
    dict(
        source_id="llm_web", provider="Groq built-in browser_search tool (web search run on Groq's servers) on a gpt-oss model",
        endpoint="https://api.groq.com/openai/v1/chat/completions with tools=[{\"type\": \"browser_search\"}] (GROQ_API_KEY)",
        coverage="Web check of one unconfirmed-looking company news item: do independent outlets report the same facts, did anyone deny or correct it, is it old news",
        timestamp_meaning="n/a",
        update_frequency="On demand (cost-capped: LLM_WEB_MAX_PER_RUN per news job and LLM_WEB_MAX_PER_DAY per IST day, 30% of the LLM time box, 1-day result cache)",
        history_depth="Live web only: never used for backtests or replays",
        rate_limits="Works on Groq's free tier (verified live) but is its most expensive call: about 7,000 tokens per search (4,000-12,000; the pages the model reads count as input tokens), charged to both the per-minute and the per-day token limit, so about 28 searches use up a free-tier model's 200,000 tokens a day (hence the daily cap); a rate limit starts a cooldown that survives between runs, HTTP 413 or a rejected tool backs the check off for 6 h",
        license_notes="Governed by Groq's terms (built-in tools are not HIPAA covered and are not available on regional endpoints); only public news text is sent - never keys, positions or other data.",
        failure_behavior="The item is extracted and judged without the web check; repeated failures appear in source health",
        backup_source="llm_extraction", criticality="optional",
    ),
    dict(
        source_id="jev_judgment", provider="TypeSafe AI - Jev (System One decision model)",
        endpoint="https://api.typesafe.ai/v1/systemone",
        coverage="Calibrated judgments on one news item: event type, rumour, denial, severity and the direction for the company in focus",
        timestamp_meaning="n/a", update_frequency="On demand (cost-capped: per-run item budget, 14-day result cache)",
        history_depth="n/a", rate_limits="HTTP 429 / 529 with Retry-After; a wait above 30 s starts a cooldown that survives between runs",
        license_notes="Governed by TypeSafe AI's terms (customer data is not used for training; zero data retention is an enterprise option); only public news text is sent - never keys, positions or other data.",
        failure_behavior="The deterministic rules decide alone; report states reduced coverage; repeated failures appear in source health",
        backup_source="rules", criticality="optional",
    ),
)


def register_all(store: Any) -> int:
    for row in SOURCES:
        store.register_source(**row)
    return len(SOURCES)


def _fetch_strategy() -> dict[str, str]:
    """How each source is requested (transport, spacing, batching), phrased from the code's own constants (lazy imports keep this
    module light for the callers that only need ``tracked``)."""
    from app.config import GROQ_EXTRA_FALLBACK_MODELS, GROQ_FALLBACK_MODEL, GROQ_MODEL
    from ingestion.kotak import QUOTE_BATCH
    from ingestion.news_sources import (
        FEED_DISABLE_AFTER,
        FEED_DISABLE_HOURS,
        FEED_FULL_REFRESH_HOURS,
        GDELT_GIVE_UP_AFTER,
    )
    from ingestion.pipeline import PROBE_MAX_AGE
    from ingestion.yahoo import SPARK_MAX_SYMBOLS
    from news_intelligence.jev import JEV_BASE_URL, JEV_PATH
    from news_intelligence.remote import COOLDOWN_MAX_SEC, RETRY_AFTER_CAP_SEC

    probe_hours = int(PROBE_MAX_AGE.total_seconds() // 3600)
    nse_client = ("curl_cffi with browser headers, spacing `NSE_MIN_INTERVAL_SEC`; 403 / 404 are soft statuses meaning "
                  "\"not published yet\" and are retried by the job")
    return {
        "yahoo_daily": (
            "client `yahoo`: curl_cffi, spacing `YAHOO_MIN_INTERVAL_SEC`, query1 host with a query2 fallback. Incremental: only the "
            "range since the newest stored bar is requested. The global series (indices, FX, commodities, rates, VIX) are first "
            f"probed in bulk (`spark`, up to {SPARK_MAX_SYMBOLS} symbols per request) and skipped while their newest bar is "
            f"unchanged (every series is still fetched in full at least every {probe_hours} h); equities come from the official NSE "
            "file, Yahoo only repairs the names the file lacks and serves the resumable history backfill (one full download per symbol)."),
        "yahoo_quote": (
            "client `yahoo` (shared spacing): one 5-minute chart request per context symbol (Asia indices, US futures, USD/INR, "
            "crude) per collection; a failure never blocks a report. In the evening `evaluate_opening` asks for the last 5 sessions "
            "of 5-minute bars (`range=5d&interval=5m`) of each name the 09:30 opening read listed that the broker (`kotak_history`) "
            "could not give a complete day for, within a 120-second budget, "
            "and only for runs whose intraday figures are not yet stored; bars Yahoo leaves null are skipped."),
        "nse_bhavcopy": (
            f"client `nse_archive`: {nse_client}. ONE file per trading day covers every stock; a page served with status 200 but "
            "holding HTML is rejected."),
        "nse_fo_bhavcopy": (
            "client `nse_archive` (same policy): one zip per day for all contracts, parsed once; the bootstrap backfills day by "
            "day within a deadline and resumes where it stopped."),
        "nse_lists": "client `nse_archive` (same policy): one CSV per index list plus the lot-size file, weekly.",
        "nse_api_market": (
            "client `nse_api`: curl_cffi with browser headers; a cookie handshake on the NSE home page, repeated on 401 / 403 and "
            "every 20 minutes. One pre-open call returns the whole book (polled every 15 s until final prices are published or "
            "the report cutoff, each call bounded by that deadline); GIFT Nifty comes from the market-status call; the holiday "
            "list is one call a day."),
        "nse_live_quotes": (
            "client `nse_api` (same session and cookie handshake as the pre-open book): ONE `equity-stockIndex` call returns the "
            "running figures of the whole F&O segment (a larger universe adds one `equity-stock-indices` call per index list; the "
            "retired `equity-stockIndices` is tried last, and only a 404 moves on to the next name) and ONE `allIndices` call the "
            "indices; both are bounded by the report deadline of the scan that asks (09:45 for the opening read, 15:05 for the "
            "closing-hour scan). The answer's own timestamp is the stamp of "
            "every row, and one that is not today's date is not today's session and is refused. With `KOTAK_CONSUMER_KEY` set the "
            "same scans also read the broker (`kotak_quotes`) and merge the two (`ingestion.livefeed`)."),
        "kotak_quotes": (
            "client `kotak_quotes`: plain `requests` transport (the API needs no browser fingerprint, and curl_cffi's streamed responses "
            "drop the connection so that every call pays the host's 2 s handshake again), spacing `KOTAK_QUOTE_INTERVAL_SEC`. The "
            "application token travels in the `Authorization` header to https hosts on kotaksecurities.com only (`KOTAK_BASE_URL` is "
            "refused at start-up otherwise); the instrument master (`file-paths`, then the `nse_cm` CSV of about 3 MB, fetched WITHOUT the "
            f"token and only from a listed host on kotaksecurities.com) is read once a day into the database, {QUOTE_BATCH} stocks go into "
            "each quote request, a refused request is halved until the one stock the API cannot quote is found, and a stock the API "
            "merely omits is a name this feed does not have. `ingestion.livefeed` merges the rows with NSE's: a name both have must "
            "agree on previous close (0.3 %), open (0.5 %) and last price (3 %) or it is left out; of two agreeing rows the one with the "
            "larger day volume (the fresher one) is taken whole; the feed is not used when, during the session, even its freshest trade "
            "is older than `KOTAK_MAX_LAG_SEC`. Nifty 50 and India VIX are asked only when NSE's index call gave nothing."),
        "kotak_history": (
            "client `kotak_history` (same transport and token rules), spacing `KOTAK_HISTORY_INTERVAL_SEC`: in the evening "
            "`evaluate_opening` asks for one `5min` window of the trading day per listed name, inside the same 120-second budget as "
            "Yahoo; a day counts only when all 72 five-minute bars from 09:15 to 15:10 are there (the day is cut to those bars, and a bar "
            "the parser drops, such as the archive's end-of-day artefact with a negative volume, makes the day unfinished) and otherwise "
            "Yahoo is asked for that name instead. The candles are normalised to the shape Yahoo's bars have, a row that is not a bar of the regular session "
            "or whose high and low do not contain its open and close is dropped, and a timestamp that appears twice keeps its last row."),
        "nse_announcements": (
            "client `nse_api` (same session): one call per date window for all symbols (filtered locally); the board-meeting "
            "calendar likewise."),
        "rss_news": (
            "client `rss`: 0.4 s spacing, 2 attempts, 6 MB cap. Conditional GET (`ETag` / `Last-Modified`): an unchanged feed costs "
            f"one 304; one unconditional fetch every {FEED_FULL_REFRESH_HOURS} h; a feed that fails {FEED_DISABLE_AFTER} times in a "
            f"row is parked for {FEED_DISABLE_HOURS} h; collection is deadline-bounded."),
        "gdelt": ("client `gdelt`: one request per configured query, 6 s apart, back-off on 429; after "
                  f"{GDELT_GIVE_UP_AFTER} failed queries in a row the rest of the run is read from the fallback."),
        "gnews_search": ("client `rss` (browser-grade transport, 0.4 s spacing): one `news.google.com/rss/search` request per topic that "
                         "GDELT did not deliver, in the English edition of the topic's country (`US`; `IN` for the India topic). The "
                         "GDELT query is translated: boolean operators, groups and quoted phrases carry over, GDELT's own operators "
                         "(`sourcelang:eng`) are dropped and `when:Nd` limits the age. Items are stored like the Google News feeds "
                         "(`gn_topic:<id>`, an aggregator: the publisher is split off the title)."),
        "llm_extraction": (
            "not scraped: the provider's own API through a `requests` session. `GROQ_API_KEY` selects Groq "
            f"(`{GROQ_MODEL}`, then `{GROQ_FALLBACK_MODEL}`, then {', '.join(f'`{name}`' for name in GROQ_EXTRA_FALLBACK_MODELS)}; "
            "`LLM_FALLBACK_MODELS` adds or replaces the later ones: Groq limits tokens and requests per model, so each model of the "
            "chain brings its own allowance. A rate-limited model hands over to the next at once and is asked again once its limit has "
            "reset; a retired or not enabled model is skipped for the run; strict JSON-schema output with a per-model capability "
            "ladder); `LLM_API_BASE` + `LLM_MODEL` select any other OpenAI-compatible provider. Per-run call "
            f"and time budgets, a 14-day result cache keyed by the article text, bounded retries that honour `Retry-After` (up to "
            f"{int(RETRY_AFTER_CAP_SEC)} s; a longer wait is a quota and starts a cooldown of at most {int(COOLDOWN_MAX_SEC // 3600)} h "
            "that survives between runs)."),
        "llm_web": (
            "not scraped: the same Groq chat endpoint, with `tools=[{\"type\": \"browser_search\"}]` and `tool_choice=required` on a "
            "gpt-oss model (`reasoning_effort=low`, as Groq recommends for browser search; the tool is server-side, so there is "
            "nothing to orchestrate). Its own call budgets (`LLM_WEB_MAX_PER_RUN` per news job, `LLM_WEB_MAX_PER_DAY` per IST day), "
            "30% of the LLM time box, a 1-day result cache, circuit breaker and cooldown; HTTP 413 or a rejected tool backs it off for "
            "6 h. A model that delivers its answer as a call to a tool that does not exist (`tool_use_failed`, seen live on about one "
            "search in six) is no refusal of the tool: the brief is read from the error body when it has the shape of one, else the "
            "search is retried once. The reply is a small JSON brief that is "
            "validated field by field, and the sources it cites are checked against the pages the search tool really returned "
            "(`executed_tools`), before it reaches the extraction or Jev."),
        "jev_judgment": (
            f"not scraped: `POST {JEV_BASE_URL}{JEV_PATH}` with the pinned model `JEV_MODEL` (Bearer key `TYPESAFE_API_KEY`). The "
            "article is the typed `state` and the atomic questions (event type, rumour, denial, severity, direction) are answered "
            "in ONE request, so an item costs one call. Same budgets, cache, retries and cooldown as the LLM."),
    }


def _transport_section() -> list[str]:
    from ingestion.http import DEFAULT_IMPERSONATE, DEFAULT_RETRY_STATUSES

    statuses = ", ".join(str(code) for code in DEFAULT_RETRY_STATUSES)
    return [
        "", "## Transport and request strategy", "",
        "Every scraped or public source is reached through one client, `ingestion.http.HttpClient`:", "",
        f"- **Transport:** curl_cffi - libcurl with a real browser TLS / HTTP-2 fingerprint (profile `{DEFAULT_IMPERSONATE}` by "
        "default, `HTTP_IMPERSONATE`) - falling back to `requests` when curl_cffi cannot be used (`HTTP_TRANSPORT` forces one).",
        "- **Politeness:** a per-source minimum spacing with 20 % jitter; a 429 or `Retry-After` doubles that source's spacing (up to a "
        "cap) and every success eases it back; a circuit breaker opens after 6 consecutive failures and half-opens after 5 minutes.",
        f"- **Retries:** statuses {statuses} retry with exponential back-off and jitter (or `Retry-After`, at most 60 s), at most "
        "`HTTP_MAX_ATTEMPTS` attempts; no attempt starts after the caller's deadline.",
        "- **Safety:** HTTPS only; responses larger than `HTTP_MAX_RESPONSE_MB` are refused; logs and errors carry host and path only, "
        "never the query string.",
        "- **Traceability:** per-source counters (requests, retries, 429s, 304s, bytes) are logged as a `network:` line after every job.",
        "- Official, authenticated APIs (Telegram, S3-compatible storage, the optional Groq LLM and Jev) use their own `requests` "
        "sessions and retry rules.",
    ]


def render_markdown() -> str:
    """docs/DATA_SOURCES.md, generated from ``SOURCES`` so the documentation can never drift from the code."""
    strategy = _fetch_strategy()
    lines = [
        "# Data sources",
        "",
        "Generated from `ingestion/registry.py` (run `python scripts/generate_docs.py`). Every source records its provider, "
        "endpoint, coverage, timestamp meaning, update frequency, history depth, rate limits, request strategy, licensing, failure "
        "behaviour and backup. Free endpoints are **not** assumed to be stable, complete or licensed for production trading use.",
        "",
        "| Source | Criticality | Provider | Backup |",
        "|---|---|---|---|",
    ]
    for s in SOURCES:
        lines.append(f"| `{s['source_id']}` | {s['criticality']} | {s['provider']} | {s['backup_source'] or '-'} |")
    lines += _transport_section()
    for s in SOURCES:
        lines += [
            "", f"## `{s['source_id']}` - {s['provider']}", "",
            f"- **Endpoint:** `{s['endpoint']}`", f"- **Coverage:** {s['coverage']}",
            f"- **Timestamp meaning:** {s['timestamp_meaning']}", f"- **Update frequency:** {s['update_frequency']}",
            f"- **History depth:** {s['history_depth']}", f"- **Rate limits:** {s['rate_limits']}",
            f"- **Transport and request strategy:** {strategy[s['source_id']]}",
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
