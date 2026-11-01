# Data sources

Generated from `ingestion/registry.py` (run `python scripts/generate_docs.py`). Every source records its provider, endpoint, coverage, timestamp meaning, update frequency, history depth, rate limits, request strategy, licensing, failure behaviour and backup. Free endpoints are **not** assumed to be stable, complete or licensed for production trading use.

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

## Transport and request strategy

Every scraped or public source is reached through one client, `ingestion.http.HttpClient`:

- **Transport:** curl_cffi - libcurl with a real browser TLS / HTTP-2 fingerprint (profile `chrome` by default, `HTTP_IMPERSONATE`) - falling back to `requests` when curl_cffi cannot be used (`HTTP_TRANSPORT` forces one).
- **Politeness:** a per-source minimum spacing with 20 % jitter; a 429 or `Retry-After` doubles that source's spacing (up to a cap) and every success eases it back; a circuit breaker opens after 6 consecutive failures and half-opens after 5 minutes.
- **Retries:** statuses 408, 425, 429, 500, 502, 503, 504 retry with exponential back-off and jitter (or `Retry-After`, at most 60 s), at most `HTTP_MAX_ATTEMPTS` attempts; no attempt starts after the caller's deadline.
- **Safety:** HTTPS only; responses larger than `HTTP_MAX_RESPONSE_MB` are refused; logs and errors carry host and path only, never the query string.
- **Traceability:** per-source counters (requests, retries, 429s, 304s, bytes) are logged as a `network:` line after every job.
- Official, authenticated APIs (Telegram, S3-compatible storage, the optional LLM) use their own `requests` sessions and retry rules.

## `yahoo_daily` - Yahoo Finance (unofficial chart API)

- **Endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=..&interval=1d`
- **Coverage:** NSE equities (.NS), Indian indices, global indices, FX, rates, commodity futures
- **Timestamp meaning:** Bar start in exchange-local time; final only after the session close (partial current bar is dropped)
- **Update frequency:** Daily bars final after each local close
- **History depth:** 10+ years (split-adjusted)
- **Rate limits:** Undocumented; ~1 request/second sustained; HTTP 429 on bursts
- **Transport and request strategy:** client `yahoo`: curl_cffi, spacing `YAHOO_MIN_INTERVAL_SEC`, query1 host with a query2 fallback. Incremental: only the range since the newest stored bar is requested. The global series (indices, FX, commodities, rates, VIX) are first probed in bulk (`spark`, up to 20 symbols per request) and skipped while their newest bar is unchanged (every series is still fetched in full at least every 24 h); equities come from the official NSE file, Yahoo only repairs the names the file lacks and serves the resumable history backfill (one full download per symbol).
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
- **Transport and request strategy:** client `yahoo` (shared spacing): one 5-minute chart request per context symbol (Asia indices, US futures, USD/INR, crude) per collection; a failure never blocks a report.
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
- **Transport and request strategy:** client `nse_archive`: curl_cffi with browser headers, spacing `NSE_MIN_INTERVAL_SEC`; 403 / 404 are soft statuses meaning "not published yet" and are retried by the job. ONE file per trading day covers every stock; a page served with status 200 but holding HTML is rejected.
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
- **Transport and request strategy:** client `nse_archive` (same policy): one zip per day for all contracts, parsed once; the bootstrap backfills day by day within a deadline and resumes where it stopped.
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
- **Transport and request strategy:** client `nse_archive` (same policy): one CSV per index list plus the lot-size file, weekly.
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
- **Transport and request strategy:** client `nse_api`: curl_cffi with browser headers; a cookie handshake on the NSE home page, repeated on 401 / 403 and every 20 minutes. One pre-open call returns the whole book (polled every 15 s until final prices are published or the report cutoff, each call bounded by that deadline); GIFT Nifty comes from the market-status call; the holiday list is one call a day.
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
- **Transport and request strategy:** client `nse_api` (same session): one call per date window for all symbols (filtered locally); the board-meeting calendar likewise.
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
- **Transport and request strategy:** client `rss`: 0.4 s spacing, 2 attempts, 6 MB cap. Conditional GET (`ETag` / `Last-Modified`): an unchanged feed costs one 304; one unconditional fetch every 6 h; a feed that fails 6 times in a row is parked for 24 h; collection is deadline-bounded.
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
- **Transport and request strategy:** client `gdelt`: one request per configured query, 6 s apart, back-off on 429.
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
- **Transport and request strategy:** client `fred`: curl_cffi without browser impersonation (an official API gets an honest User-Agent), 0.6 s spacing, 3 attempts; one request per series; a no-op without `FRED_API_KEY`.
- **Licensing:** Free with an API key; see FRED terms.
- **Failure behaviour:** Optional and informational: first-release values are stored (macro_observations) but no feature or report line reads them yet; market-based yields come from Yahoo; revised values are never used as point-in-time history
- **Backup source:** yahoo_daily

## `econ_calendar` - Public economic-calendar mirror (faireconomy JSON)

- **Endpoint:** `https://nfs.faireconomy.media/ff_calendar_thisweek.json`
- **Coverage:** Scheduled macro releases with impact tags
- **Timestamp meaning:** Scheduled release time with UTC offset
- **Update frequency:** Daily
- **History depth:** Current week
- **Rate limits:** Fetch at most a few times per day
- **Transport and request strategy:** client `econ_calendar`: one request per collection, 1 s spacing.
- **Licensing:** Unofficial / public endpoint: suitable for research and shadow operation; confirm the provider's terms before any commercial or redistributed use and prefer a licensed feed for production.
- **Failure behaviour:** Optional and informational: the schedule is stored (econ_calendar) but no feature or report line reads it yet, so a failure changes nothing
- **Backup source:** none

## `llm_extraction` - Configured OpenAI-compatible LLM API

- **Endpoint:** `LLM_API_BASE/chat/completions`
- **Coverage:** Structured event extraction from news text
- **Timestamp meaning:** n/a
- **Update frequency:** On demand (cost-capped)
- **History depth:** n/a
- **Rate limits:** Provider specific; per-run article budget
- **Transport and request strategy:** not scraped: the configured provider's own API through a `requests` session with its own timeout and retry rules and a per-run article budget.
- **Licensing:** Governed by the chosen provider's terms; no secrets/PII are sent.
- **Failure behaviour:** Deterministic rule-based extraction; report states reduced coverage
- **Backup source:** rules
