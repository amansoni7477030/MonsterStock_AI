# Data sources

Generated from `ingestion/registry.py` (run `python scripts/generate_docs.py`). Every source records its provider, endpoint, coverage, timestamp meaning, update frequency, history depth, rate limits, request strategy, licensing, failure behaviour and backup. Free endpoints are **not** assumed to be stable, complete or licensed for production trading use.

| Source | Criticality | Provider | Backup |
|---|---|---|---|
| `yahoo_daily` | critical | Yahoo Finance (unofficial chart API) | nse_bhavcopy |
| `yahoo_quote` | optional | Yahoo Finance (unofficial chart API) | - |
| `nse_bhavcopy` | important | NSE archives (sec_bhavdata_full) | yahoo_daily |
| `nse_fo_bhavcopy` | important | NSE archives (UDiFF F&O bhavcopy) | - |
| `nse_lists` | important | NSE archives (index constituents, F&O lot sizes) | seed_universe |
| `nse_api_market` | important | NSE public JSON API (marketStatus, pre-open, holidays) | yahoo_quote |
| `nse_announcements` | important | NSE corporate announcements API | rss_news |
| `nse_live_quotes` | optional | NSE public JSON API (equity-stockIndex / equity-stock-indices, allIndices) | - |
| `rss_news` | important | Publisher RSS/Atom feeds (data/seeds/feeds.json) | gdelt |
| `gdelt` | optional | GDELT Project DOC 2.0 API | rss_news |
| `llm_extraction` | optional | Groq (open-weight gpt-oss models) or any OpenAI-compatible LLM API | rules |
| `llm_web` | optional | Groq built-in browser_search tool (web search run on Groq's servers) on a gpt-oss model | llm_extraction |
| `jev_judgment` | optional | TypeSafe AI - Jev (System One decision model) | rules |

## Transport and request strategy

Every scraped or public source is reached through one client, `ingestion.http.HttpClient`:

- **Transport:** curl_cffi - libcurl with a real browser TLS / HTTP-2 fingerprint (profile `chrome` by default, `HTTP_IMPERSONATE`) - falling back to `requests` when curl_cffi cannot be used (`HTTP_TRANSPORT` forces one).
- **Politeness:** a per-source minimum spacing with 20 % jitter; a 429 or `Retry-After` doubles that source's spacing (up to a cap) and every success eases it back; a circuit breaker opens after 6 consecutive failures and half-opens after 5 minutes.
- **Retries:** statuses 408, 425, 429, 500, 502, 503, 504 retry with exponential back-off and jitter (or `Retry-After`, at most 60 s), at most `HTTP_MAX_ATTEMPTS` attempts; no attempt starts after the caller's deadline.
- **Safety:** HTTPS only; responses larger than `HTTP_MAX_RESPONSE_MB` are refused; logs and errors carry host and path only, never the query string.
- **Traceability:** per-source counters (requests, retries, 429s, 304s, bytes) are logged as a `network:` line after every job.
- Official, authenticated APIs (Telegram, S3-compatible storage, the optional Groq LLM and Jev) use their own `requests` sessions and retry rules.

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
- **Coverage:** Last price and previous close (Asia open moves, US futures); the 5-minute bars the endpoint also returns are not used
- **Timestamp meaning:** Last trade time reported by the exchange (UTC epoch)
- **Update frequency:** Real-time to ~15 min delayed
- **History depth:** Latest quote only (the endpoint serves ~60 days of 5m bars; none are stored)
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

## `nse_api_market` - NSE public JSON API (marketStatus, pre-open, holidays)

- **Endpoint:** `https://www.nseindia.com/api/*`
- **Coverage:** GIFT Nifty, indicative/final pre-open prices and order imbalance, trading holidays
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

## `nse_live_quotes` - NSE public JSON API (equity-stockIndex / equity-stock-indices, allIndices)

- **Endpoint:** `https://www.nseindia.com/api/equity-stockIndex?index=SECURITIES%20IN%20F%26O ; /api/equity-stock-indices?index=NIFTY%20500 ; /api/allIndices`
- **Coverage:** Every F&O stock (or a broader index list) in ONE call: today's open, high, low, last price, previous close, volume so far, plus the 52-week range; Nifty 50, India VIX and the sector indices live
- **Timestamp meaning:** the answer's own `timestamp`, exchange time in IST (parsed with an explicit IST assumption) and the stamp of every row; the per-stock `lastUpdateTime` is a placeholder (31-DEC-2999) or the date of an old update and is NOT read
- **Update frequency:** Live during the session (seconds to a minute behind); the final values after the close
- **History depth:** Live only (not point-in-time reconstructable): every trade idea freezes the figures it was based on
- **Rate limits:** Same session as nse_api_market: cookie handshake, ~1 request/second; the closing-hour scan makes two calls; frequent 401/403 from datacentre IPs
- **Transport and request strategy:** client `nse_api` (same session and cookie handshake as the pre-open book): ONE `equity-stockIndex` call returns the running figures of the whole F&O segment (a larger universe adds one `equity-stock-indices` call per index list; the retired `equity-stockIndices` is tried last, and only a 404 moves on to the next name) and ONE `allIndices` call the indices; both are bounded by the report deadline of the closing-hour scan. The answer's own timestamp is the stamp of every row, and one that is not today's date is not today's session and is refused.
- **Licensing:** Published by NSE for public download/consumption; automated bulk use is subject to NSE terms - confirm entitlement for production; a licensed NSE data feed or broker API is the supported route. Endpoints are undocumented and may change without notice (NSE retired `equity-stockIndices` in 2026).
- **Failure behaviour:** The closing-hour scan fails visibly and retries inside its window (nothing is guessed from older data, and a snapshot that is not from today is refused); the morning scans do not use it
- **Backup source:** none

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

## `llm_extraction` - Groq (open-weight gpt-oss models) or any OpenAI-compatible LLM API

- **Endpoint:** `https://api.groq.com/openai/v1/chat/completions (GROQ_API_KEY) or LLM_API_BASE/chat/completions`
- **Coverage:** Structured event extraction from news text: event type, entities with their direction, severity, horizon, event time and a verbatim evidence quote
- **Timestamp meaning:** n/a
- **Update frequency:** On demand (cost-capped: per-run call and time budgets, 14-day result cache)
- **History depth:** n/a
- **Rate limits:** Groq free tier (measured live): 8,000 tokens per minute, 200,000 tokens per day and 1,000 requests per day for each model, about 2,000 tokens per extraction, so roughly 4 extractions per minute and about 100 a day per model; every model of the chain (gpt-oss-120b, gpt-oss-20b, qwen3.8-27b by default, `LLM_FALLBACK_MODELS` adds more) has its own allowance, a rate-limited model hands over to the next one at once and is used again once its limit has reset, a short Retry-After is waited out when every model is limited, a long one starts a cooldown that survives between runs
- **Transport and request strategy:** not scraped: the provider's own API through a `requests` session. `GROQ_API_KEY` selects Groq (`openai/gpt-oss-120b`, then `openai/gpt-oss-20b`, then `qwen/qwen3.8-27b`; `LLM_FALLBACK_MODELS` adds or replaces the later ones: Groq limits tokens and requests per model, so each model of the chain brings its own allowance. A rate-limited model hands over to the next at once and is asked again once its limit has reset; a retired or not enabled model is skipped for the run; strict JSON-schema output with a per-model capability ladder); `LLM_API_BASE` + `LLM_MODEL` select any other OpenAI-compatible provider. Per-run call and time budgets, a 14-day result cache keyed by the article text, bounded retries that honour `Retry-After` (up to 30 s; a longer wait is a quota and starts a cooldown of at most 6 h that survives between runs).
- **Licensing:** Governed by the chosen provider's terms; only public news text is sent - never keys, positions or other data.
- **Failure behaviour:** Deterministic rule-based extraction; report states reduced coverage; repeated failures appear in source health
- **Backup source:** rules

## `llm_web` - Groq built-in browser_search tool (web search run on Groq's servers) on a gpt-oss model

- **Endpoint:** `https://api.groq.com/openai/v1/chat/completions with tools=[{"type": "browser_search"}] (GROQ_API_KEY)`
- **Coverage:** Web check of one unconfirmed-looking company news item: do independent outlets report the same facts, did anyone deny or correct it, is it old news
- **Timestamp meaning:** n/a
- **Update frequency:** On demand (cost-capped: LLM_WEB_MAX_PER_RUN per news job and LLM_WEB_MAX_PER_DAY per IST day, 30% of the LLM time box, 1-day result cache)
- **History depth:** Live web only: never used for backtests or replays
- **Rate limits:** Works on Groq's free tier (verified live) but is its most expensive call: about 7,000 tokens per search (4,000-12,000; the pages the model reads count as input tokens), charged to both the per-minute and the per-day token limit, so about 28 searches use up a free-tier model's 200,000 tokens a day (hence the daily cap); a rate limit starts a cooldown that survives between runs, HTTP 413 or a rejected tool backs the check off for 6 h
- **Transport and request strategy:** not scraped: the same Groq chat endpoint, with `tools=[{"type": "browser_search"}]` and `tool_choice=required` on a gpt-oss model (`reasoning_effort=low`, as Groq recommends for browser search; the tool is server-side, so there is nothing to orchestrate). Its own call budgets (`LLM_WEB_MAX_PER_RUN` per news job, `LLM_WEB_MAX_PER_DAY` per IST day), 30% of the LLM time box, a 1-day result cache, circuit breaker and cooldown; HTTP 413 or a rejected tool backs it off for 6 h. A model that delivers its answer as a call to a tool that does not exist (`tool_use_failed`, seen live on about one search in six) is no refusal of the tool: the brief is read from the error body when it has the shape of one, else the search is retried once. The reply is a small JSON brief that is validated field by field, and the sources it cites are checked against the pages the search tool really returned (`executed_tools`), before it reaches the extraction or Jev.
- **Licensing:** Governed by Groq's terms (built-in tools are not HIPAA covered and are not available on regional endpoints); only public news text is sent - never keys, positions or other data.
- **Failure behaviour:** The item is extracted and judged without the web check; repeated failures appear in source health
- **Backup source:** llm_extraction

## `jev_judgment` - TypeSafe AI - Jev (System One decision model)

- **Endpoint:** `https://api.typesafe.ai/v1/systemone`
- **Coverage:** Calibrated judgments on one news item: event type, rumour, denial, severity and the direction for the company in focus
- **Timestamp meaning:** n/a
- **Update frequency:** On demand (cost-capped: per-run item budget, 14-day result cache)
- **History depth:** n/a
- **Rate limits:** HTTP 429 / 529 with Retry-After; a wait above 30 s starts a cooldown that survives between runs
- **Transport and request strategy:** not scraped: `POST https://api.typesafe.ai/v1/systemone` with the pinned model `JEV_MODEL` (Bearer key `TYPESAFE_API_KEY`). The article is the typed `state` and the atomic questions (event type, rumour, denial, severity, direction) are answered in ONE request, so an item costs one call. Same budgets, cache, retries and cooldown as the LLM.
- **Licensing:** Governed by TypeSafe AI's terms (customer data is not used for training; zero data retention is an enterprise option); only public news text is sent - never keys, positions or other data.
- **Failure behaviour:** The deterministic rules decide alone; report states reduced coverage; repeated failures appear in source health
- **Backup source:** rules
