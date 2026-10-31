# Data sources

Generated from `ingestion/registry.py` (run `python scripts/generate_docs.py`). Every source records its provider, endpoint, coverage, timestamp meaning, update frequency, history depth, rate limits, licensing, failure behaviour and backup. Free endpoints are **not** assumed to be stable, complete or licensed for production trading use.

| Source | Criticality | Provider | Backup |
|---|---|---|---|
| `yahoo_daily` | critical | Yahoo Finance (unofficial chart API) | nse_bhavcopy |
| `yahoo_quote` | optional | Yahoo Finance (unofficial chart API) | - |
| `nse_bhavcopy` | important | NSE archives (sec_bhavdata_full) | yahoo_daily |
| `nse_fo_bhavcopy` | important | NSE archives (UDiFF F&O bhavcopy) | - |
| `nse_lists` | important | NSE archives (index constituents, F&O lot sizes) | seed_universe |
| `nse_api_market` | important | NSE public JSON API (marketStatus, pre-open, option chain, holidays) | yahoo_quote |
| `nse_announcements` | important | NSE corporate announcements API | rss_news |
| `rss_news` | important | Publisher RSS/Atom feeds (data/seeds/feeds.json) | gdelt |
| `gdelt` | optional | GDELT Project DOC 2.0 API | rss_news |
| `fred` | optional | Federal Reserve Bank of St. Louis (FRED/ALFRED) | yahoo_daily |
| `econ_calendar` | optional | Public economic-calendar mirror (faireconomy JSON) | - |
| `llm_extraction` | optional | Configured OpenAI-compatible LLM API | rules |

## `yahoo_daily` - Yahoo Finance (unofficial chart API)

- **Endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=..&interval=1d`
- **Coverage:** NSE equities (.NS), Indian indices, global indices, FX, rates, commodity futures
- **Timestamp meaning:** Bar start in exchange-local time; final only after the session close (partial current bar is dropped)
- **Update frequency:** Daily bars final after each local close
- **History depth:** 10+ years (split-adjusted)
- **Rate limits:** Undocumented; ~1 request/second sustained; HTTP 429 on bursts
- **Licensing:** Unofficial / public endpoint: suitable for research and shadow operation; confirm the provider's terms before any commercial or redistributed use and prefer a licensed feed for production.
- **Failure behaviour:** Bounded retries with backoff; circuit breaker; NSE bhavcopy overrides vendor OHLC; stale data blocks or downgrades the report
- **Backup source:** nse_bhavcopy

## `yahoo_quote` - Yahoo Finance (unofficial chart API)

- **Endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=5m`
- **Coverage:** Live/recent intraday bars and last price (Asia open moves, US futures)
- **Timestamp meaning:** Last trade time reported by the exchange (UTC epoch)
- **Update frequency:** Real-time to ~15 min delayed
- **History depth:** ~60 days of 5m bars
- **Rate limits:** Same as yahoo_daily
- **Licensing:** Unofficial / public endpoint: suitable for research and shadow operation; confirm the provider's terms before any commercial or redistributed use and prefer a licensed feed for production.
- **Failure behaviour:** Descriptive context only; absence never blocks a report
- **Backup source:** none

## `nse_bhavcopy` - NSE archives (sec_bhavdata_full)

- **Endpoint:** `https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv`
- **Coverage:** All NSE cash-market securities: official OHLC, VWAP, volume, turnover, trades, delivery %
- **Timestamp meaning:** End-of-day official values for the trade date
- **Update frequency:** Daily, published ~17:30-19:30 IST
- **History depth:** Years (one file per day)
- **Rate limits:** No published limit; keep >= 1 s spacing
- **Licensing:** Published by NSE for public download/consumption; automated bulk use is subject to NSE terms - confirm entitlement for production; a licensed NSE data feed or broker API is the supported route.
- **Failure behaviour:** 404 = not yet published (retry later); outcomes fall back to provisional vendor data and are flagged unvalidated
- **Backup source:** yahoo_daily

## `nse_fo_bhavcopy` - NSE archives (UDiFF F&O bhavcopy)

- **Endpoint:** `https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyymmdd}_F_0000.csv.zip`
- **Coverage:** Stock/index futures and options: OHLC, settle, OI, change in OI, volume per contract
- **Timestamp meaning:** End-of-day for the trade date
- **Update frequency:** Daily after close
- **History depth:** From 2024-07-08 (UDiFF format)
- **Rate limits:** Keep >= 1 s spacing; ~1 MB per file
- **Licensing:** Published by NSE for public download/consumption; automated bulk use is subject to NSE terms - confirm entitlement for production; a licensed NSE data feed or broker API is the supported route.
- **Failure behaviour:** F&O features become missing (NaN) and the F&O specialist abstains; flagged in the report
- **Backup source:** none

## `nse_lists` - NSE archives (index constituents, F&O lot sizes)

- **Endpoint:** `https://nsearchives.nseindia.com/content/indices/ind_nifty*list.csv ; /content/fo/fo_mktlots.csv`
- **Coverage:** Nifty 100/200/500 constituents with industry + ISIN; F&O eligible underlyings and lot sizes
- **Timestamp meaning:** Latest published constituent list
- **Update frequency:** On index reconstitution (semi-annual) / expiry cycles
- **History depth:** Current list only (survivorship bias in history is documented)
- **Rate limits:** Rarely called (weekly)
- **Licensing:** Published by NSE for public download/consumption; automated bulk use is subject to NSE terms - confirm entitlement for production; a licensed NSE data feed or broker API is the supported route.
- **Failure behaviour:** Keep the previously stored universe; seed file used on first install
- **Backup source:** seed_universe

## `nse_api_market` - NSE public JSON API (marketStatus, pre-open, option chain, holidays)

- **Endpoint:** `https://www.nseindia.com/api/*`
- **Coverage:** GIFT Nifty, indicative/final pre-open prices and order imbalance, live option chain, trading holidays
- **Timestamp meaning:** Server timestamps in IST (parsed with an explicit IST assumption)
- **Update frequency:** Live during pre-open / session
- **History depth:** Live only (not point-in-time reconstructable)
- **Rate limits:** Cookie handshake required; ~1 request/second; frequent 401/403 from datacentre IPs
- **Licensing:** Published by NSE for public download/consumption; automated bulk use is subject to NSE terms - confirm entitlement for production; a licensed NSE data feed or broker API is the supported route. Endpoints are undocumented and may change without notice.
- **Failure behaviour:** Pre-open features missing -> falls back to the preliminary model with a data-quality warning; never silently reuses stale quotes
- **Backup source:** yahoo_quote

## `nse_announcements` - NSE corporate announcements API

- **Endpoint:** `https://www.nseindia.com/api/corporate-announcements ; /api/event-calendar`
- **Coverage:** Exchange filings (results, orders, rating actions, M&A, management changes) and board-meeting calendar
- **Timestamp meaning:** exchdisstime = exchange dissemination time (IST) - treated as the public availability time
- **Update frequency:** Continuous
- **History depth:** Weeks (live capture builds the point-in-time archive)
- **Rate limits:** Same session limits as nse_api_market
- **Licensing:** Published by NSE for public download/consumption; automated bulk use is subject to NSE terms - confirm entitlement for production; a licensed NSE data feed or broker API is the supported route.
- **Failure behaviour:** Coverage gap flagged; scheduled-results features default to 0
- **Backup source:** rss_news

## `rss_news` - Publisher RSS/Atom feeds (data/seeds/feeds.json)

- **Endpoint:** `Per feed`
- **Coverage:** Indian and global business, markets, geopolitics, central banks, regulators
- **Timestamp meaning:** Publisher timestamp when present; first_seen_at is always recorded separately
- **Update frequency:** Minutes to hours
- **History depth:** Feed window only (items older than the lookback are ignored)
- **Rate limits:** Polite polling (<= 1 request per feed per run)
- **Licensing:** Headlines/summaries only are stored for classification; respect each publisher's terms.
- **Failure behaviour:** Per-feed circuit breaker; coverage gap flagged; other feeds continue
- **Backup source:** gdelt

## `gdelt` - GDELT Project DOC 2.0 API

- **Endpoint:** `https://api.gdeltproject.org/api/v2/doc/doc`
- **Coverage:** Global news discovery (titles, domains, time, language)
- **Timestamp meaning:** seendate (UTC)
- **Update frequency:** 15 minutes
- **History depth:** Rolling ~3 months via API
- **Rate limits:** <= 1 request per 5 s; 429 on bursts
- **Licensing:** Free research use; discovery only - verify facts against primary sources.
- **Failure behaviour:** Backoff on 429; skipped queries recorded as coverage gap
- **Backup source:** rss_news

## `fred` - Federal Reserve Bank of St. Louis (FRED/ALFRED)

- **Endpoint:** `https://api.stlouisfed.org/fred/series/observations`
- **Coverage:** US rates and macro series with vintages when an API key is set
- **Timestamp meaning:** observation_date + realtime_start (first-release timestamp)
- **Update frequency:** Per release
- **History depth:** Decades
- **Rate limits:** 120 requests/minute with key
- **Licensing:** Free with an API key; see FRED terms.
- **Failure behaviour:** Optional; market-based yields come from Yahoo; revised values are never used as point-in-time history
- **Backup source:** yahoo_daily

## `econ_calendar` - Public economic-calendar mirror (faireconomy JSON)

- **Endpoint:** `https://nfs.faireconomy.media/ff_calendar_thisweek.json`
- **Coverage:** Scheduled macro releases with impact tags
- **Timestamp meaning:** Scheduled release time with UTC offset
- **Update frequency:** Daily
- **History depth:** Current week
- **Rate limits:** Fetch at most a few times per day
- **Licensing:** Unofficial / public endpoint: suitable for research and shadow operation; confirm the provider's terms before any commercial or redistributed use and prefer a licensed feed for production.
- **Failure behaviour:** Optional: event-day regime flags default to 0
- **Backup source:** none

## `llm_extraction` - Configured OpenAI-compatible LLM API

- **Endpoint:** `LLM_API_BASE/chat/completions`
- **Coverage:** Structured event extraction from news text
- **Timestamp meaning:** n/a
- **Update frequency:** On demand (cost-capped)
- **History depth:** n/a
- **Rate limits:** Provider specific; per-run article budget
- **Licensing:** Governed by the chosen provider's terms; no secrets/PII are sent.
- **Failure behaviour:** Deterministic rule-based extraction; report states reduced coverage
- **Backup source:** rules
