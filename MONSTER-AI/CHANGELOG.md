# Changelog

All notable changes to MONSTER AI. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/). Database migrations are forward-only (one of them drops two unused tables, see
*Removed*), so every release below can be installed over the previous one (`deployment/install.sh` is idempotent and rolls back
by itself).

## [Unreleased]

Closing-hour trade ideas (BTST, swing, bearish F&O and a next-day intraday watchlist), a longer Groq model chain, and tables and chart
images in the Telegram reports. No database migration; the new records live in the key-value store under `ideas:` and are pruned with
the other records.

### Upgrade notes

- **New dependency `Pillow==12.3.0`** (binary wheels for linux x86-64 and aarch64, CPython 3.10-3.13; the installer and the CI wheel
  guard cover it, because both read `requirements.txt`). It only draws the chart images: without it the reports go out as text.
  **New setting `SEND_CHARTS`** (true) switches the images off; the tables are part of the text and need nothing.
- **New setting `TELEGRAM_EXTRA_CHAT_IDS`** (empty): more Telegram chats that get the same reports and chart images as the main chat, never
  the alerts, failure notices or tests. Nothing changes until it is set. Each chat has its own delivery record, so the new rows in
  `deliveries` have the kinds `final_report_extra`, `prelim_report_extra`, `evening_report_extra` and `ideas_report_extra` (and
  `test_extra`); the installer appends the empty key to `monster.env` on the next run. See `docs/OPERATIONS.md`, *Adding or removing a
  reader*.
- **New setting `GDELT_FALLBACK`** (true): when GDELT does not deliver a topic, the news job reads it from Google News search instead
  (no key, no new dependency). `GDELT_ENABLED=false` now means "skip GDELT, use Google News search alone"; set both to `false` for no
  global news discovery at all. The 44 RSS feeds do not depend on either. A new source `gnews_search` is registered by the next job (or
  `monster-ai init`); it is listed in `docs/DATA_SOURCES.md` and, if it keeps failing, in `monster-ai health`.
- **New jobs and schedule entries** (picked up when the scheduler restarts): `collect_news` at 13:40 (slot `midday`), `closing_scan`
  at 14:30 (a try every minute for the whole window until 15:00) and `evaluate_ideas` after `ingest_eod`. Run `sudo monster-ai jobs`
  to see them.
- **New settings** (safe defaults, see `.env.example`): `IDEAS_ENABLED` (true), `SEND_IDEAS_REPORT` (true), `IDEAS_REPORT_DEADLINE_TIME`
  (15:05), `IDEAS_ENTRY_BY_TIME` (15:15), `IDEAS_TOP_N` (5), `IDEAS_MIN_SCORE` (60), `IDEAS_MAX_RISK_PCT` (4), `IDEAS_INCLUDE_SHORTS`
  (true) and `LLM_FALLBACK_MODELS`. The afternoon report goes to the same Telegram chat as the morning one; `SEND_IDEAS_REPORT=false`
  keeps the ideas computing, freezing and evaluating without sending, `IDEAS_ENABLED=false` switches the feature off.
- **With a Groq key the news layer now has three models**, not two: `qwen/qwen3.8-27b` follows `openai/gpt-oss-20b`. Set
  `LLM_FALLBACK_MODELS=none` to keep the old pair, or list your own further models. Run `sudo monster-ai ai-test --list-only`: it
  warns about a model of the chain that the key cannot see.

### Added

- **More than one reader for the Telegram reports** (`TELEGRAM_EXTRA_CHAT_IDS`; `NotificationService._deliver_extras`, the shared
  `_send_messages` loop, `send_test_extras`; `render_subscriber_welcome`). A friend sends the bot `/start`, `telegram-setup --list`
  shows every chat that wrote to it in the last 24 hours (newest first, with name, handle, time and whether it is already subscribed),
  and the id goes into the setting. The prelim, final, evening and closing-hour reports then reach the main chat first and, once that is
  recorded as sent, every extra chat with the same messages and deadline, followed by one drawing of the chart images for everybody
  (the text always reaches all chats before the first image). Alerts, failure notices and tests stay with the owner;
  `telegram-test --all` sends each extra chat a short "you are subscribed" note (nothing of the configuration) and names the ones that
  could not be reached; `doctor` says how many extra chats there are. Failure isolation is the point: every extra chat has its own ledger
  row, so a friend who blocked the bot never turns the owner's delivery, or the health check that reads the newest `final_report` row,
  into a failure, and costs one `telegram_extra_delivery_failed` warning per chat and day. Tests cover the fan-out, the ledger, the
  failure paths, the images and the commands.
- **Tables and chart images in the Telegram reports** (`notifications/canvas.py`, `notifications/charts.py`, `send_photo` in
  `notifications/telegram.py`, see README *What the Telegram messages look like*). The 09:12 report and the closing-hour report get an
  "at a glance" table in monospace (at most 39 / 37 characters wide, so a phone needs no sideways scrolling); the closing-hour report
  also gets a table of tomorrow's buy-above / sell-below levels. After the text, each report is followed by PNG images drawn with
  Pillow: the candidates' expected range against the target and the sector mix (morning), the **board** with a donut of the ideas by
  list, the breadth bar and the bars of the setup scores plus the **trade plan** with the loss at the stop and the gain at the two
  targets of each idea (afternoon), the hit / missed donut, the model against the baselines and the realised range of each pick
  (evening). The images keep the honesty rules of the text: a score is never drawn as a probability, a probability is shown only for a
  model that is not the heuristic baseline and under the shadow banner, a day without a setup and a late report get no image, a missing
  number is left out. A chart is best-effort and sent only after the text is delivered and recorded, within a 45-second budget (a slow
  Telegram cannot keep a job busy with pictures): no Pillow, `SEND_CHARTS=false`, a
  drawing error or a refused upload leaves the delivered report untouched (the failure is logged). `telegram-preview` sends the
  samples with a red SAMPLE strip on every image. The client's `send_photo` has the retries, rate-limit waits, chat migration and
  plain-caption fallback of `send_message`.
- **Google News search as the alternative to GDELT** (`ingestion/news_sources.py`: `collect_gdelt`, `gnews_search`, `read_gnews`;
  setting `GDELT_FALLBACK`). The global topic queries (oil, rates and inflation, geopolitics, China, India macro, the Middle East,
  semiconductors, markets) are asked of GDELT first; a topic it does not deliver, GDELT switched off, or the rest of a run once
  GDELT has failed two queries in a row (it was timing out, so the first run through all eight topics used to spend several minutes
  waiting) is read from Google News search with the same query: boolean operators, groups and quoted phrases carry over, GDELT's own
  operators (`sourcelang:eng`) are dropped and `when:Nd` limits the age. The India topic asks the Indian edition (`"edition": "IN"` in
  `data/seeds/feeds.json`), because a word such as RBI is a baseball statistic in the US edition. The answer is stored like the other
  Google News feeds (`gn_topic:<id>`, an aggregator: the publisher is split off each title and the source-quality cap of 0.6 applies,
  where a GDELT link to the publisher itself rates 0.75). An answer without a single item (a consent or block page parses as an empty
  feed) is an error, never a success. A topic counts as answered when either source answered it, so `source_health` for `gdelt` only
  warns when neither does, and `gnews_search` has a row of its own. Run live on 2026-11-02 with GDELT timing out: 8 of 8 topics
  covered in 51 s, 421 new items, and the news pipeline turned them into 301 events (geopolitical conflict, central bank, commodity
  supply, sanctions and tariffs, market stress ...) exactly as it does for GDELT items. `sources-check` probes both and `--only gdelt`
  shows which one is down.
- **`monster-ai sources-check [--only SECTION ...] [--json]`** (`app/sourcecheck.py`): the endpoint audit as a command. It reads every
  endpoint the system uses through the production clients (the NSE JSON API, the NSE archive files, Yahoo, all RSS feeds, GDELT and
  its Google News fallback) and
  prints ok / warn / FAIL with the reason of each one, so a retired or renamed endpoint (NSE did it once already in 2026) is found
  before a scan stumbles on it; exit code 1 when an endpoint the system needs does not answer, while GDELT, its fallback and a single
  feed are only warnings. It reads, never writes. Run against the live services on 2026-11-02 it reported every endpoint ok except
  GDELT (a timeout).
  The GDELT client now lives in one function (`ingestion.news_sources.gdelt_client`), used by the news job and by the check.
- **`doctor` checks the NSE JSON API** (the cookie handshake and the F&O live-quote list the closing scan reads, so a retired endpoint
  name shows up as a warning with its reason before 14:30) **and the chart library**.
- **Closing-hour trade ideas** (`automation/ideas.py`, `ranking/ideas.py`, `features/swing.py`, `backtesting/ideas.py`,
  `notifications/ideas.py`; CLI `monster-ai ideas [--dry-run] [--force]`): at 14:30 IST NSE's live quotes for the whole F&O segment (one
  request) are laid over the stored daily history as the session in progress, with volume projected to a full day by a typical
  intraday profile. Four lists come out, five names each by default: **BTST** (buy before the close, sell in the next session),
  **swing** (hold up to five sessions), **bearish** setups (BTST and swing mirrors, F&O names only) and a **watchlist** of buy / sell
  levels for the next day's intraday trading. Every idea has an entry reference with a "do not chase" price, a stop, two targets (all
  mechanical multiples of the previous ATR, the stop at the nearer of that multiple and the day's or week's extreme), the reward to
  risk, the estimated delivery cost, reasons and cautions. Gates: liquidity, price, history length, a stop within `IDEAS_MAX_RISK_PCT`,
  a reward of at least 1.5 times the risk, a cost of at most a quarter of the first target, no BTST into a results announcement due
  today or tomorrow, no overnight trade against a Nifty move of 2 % or more, and a veto for confirmed bad news (rumours are only
  mentioned). The news pipeline adds the catalysts of the shortlist (the new 13:40 news pass keeps them fresh).
- **Honest by construction.** A score is a weighted mean of fixed, documented components and is called a score, never a probability.
  Each list shows its own live record with sample sizes and 95 % intervals and is only called validated after 60 judged ideas over 20
  days with intervals that clear no-edge. The report is frozen once per day (`ideas:run:<date>`), re-sent unchanged after a failed
  delivery, and never replaced: a retry, a late run and `--force` all send the frozen report (`--force` only lifts the report deadline,
  and the copy that goes out says it is late; it scans first only when nothing is frozen). `evaluate_ideas` judges the ideas on the
  official bars (the first of the stop, the first target and the end of the horizon, a gap through the stop exits at the open, a stop
  touched on the day of a target wins) and keeps the outcomes apart (`ideas:out:<date>`); outcomes on provisional bars wait for the
  official file; a report built after the 15:05 deadline is withheld unless forced, marked *late* and left out of the record.
- **It refuses to guess.** No live quotes (the alert says why), quotes that are not from today, a gap in the stored history (special
  sessions such as Muhurat trading are tolerated), a history without the previous closes or with different ones for most names, or too
  few names with a complete row fail the scan with the alert `ideas_scan_failed` and, on the last attempt, a notice; a name whose
  price basis differs from the stored history is left out of everything, including the sector and market figures of the others.
- **`telegram-preview --only ideas`** sends a sample of the report with made-up data.
- **Groq model chain** (`news_intelligence/llm.py`, `app/config.py`): the extractor walks `openai/gpt-oss-120b` ->
  `openai/gpt-oss-20b` -> `qwen/qwen3.8-27b`; Groq's limits are per model, so the free tier's throughput roughly triples (about 12
  extractions a minute and 300 a day). A rate-limited model is parked until its own limit resets and the best model is used again as
  soon as it is free (before, the last model that answered 429 decided how long a job gave up); a model that is gone, retired or
  blocked for the account (404 / 403) is dropped for the run; only when all are gone is the service stopped. Qwen runs in its plain
  mode (`reasoning_effort` none) and `<think>` blocks are stripped from any reply. Per-model counters (`llm.model:<name>`) are in
  `monster-ai diag`, `doctor` and `ai-test` show the chain.

### Changed

- The failure notice names the closing-hour scan as such (`CLOSING-HOUR IDEAS`, "no trade ideas were published", its own log hint)
  instead of the large-move wording of the morning notices.
- The retention table lists the trade-idea records (`ideas:` prefix, `RECORD_RETENTION_DAYS`).

### Found against live services (2026-11-02) and fixed

- **NSE retired `equity-stockIndices`** (HTTP 404 for every list, also for Nifty 50). The F&O segment is now served by
  `equity-stockIndex` and the index lists by `equity-stock-indices`; the client tries the current names first, the retired one last,
  moves on only on a 404 and remembers the name that answered. `allIndices` was unchanged.
- **The per-stock `lastUpdateTime` of the F&O list is not a trade time** (144 of 218 rows carried the placeholder `31-DEC-2999`, the
  others the date of some old update) and its prices come as text; a first version that required today's date on every row would
  have refused the whole answer. Every row now carries the answer's own timestamp, and a stamp on a row is not trusted.
- `ai-test --list-only` confirmed that the project's Groq key lists all three models of the chain.
- **Full endpoint audit (2026-11-02 evening)**, every endpoint the code calls, through the project's own clients (curl_cffi, Chrome
  profile) and cross-checked with the plain `curl` CLI; nothing else was outdated:
  NSE API `marketStatus` (GIFT Nifty), `market-data-pre-open` (`ALL`, `FO`), `allIndices`, `equity-stockIndex`,
  `equity-stock-indices` (Nifty 200 / 500), `corporate-announcements`, `event-calendar` and `holiday-master` all answer with data;
  NSE archives (`sec_bhavdata_full`, the UDiFF F&O zip, the four index lists, `fo_mktlots.csv`) and Yahoo (`v8/finance/spark` for all
  33 series, the chart endpoint for every series and the live-context quotes) answer; 44 of 44 RSS feeds parse (newest item 6
  minutes to 88 hours old; the RBI, SEBI and Nikkei feeds carry dates feedparser cannot read, which the collector already tolerates);
  Telegram (`getMe`, real messages and photos), Groq (key, all three models and the web search) and Jev answer.
  **GDELT**, an optional discovery source, is up but was slow and throttling that day (HTTP 429 "one request every 5 seconds" on most
  tries, 14-36 s to first byte, once an empty answer, and later timeouts on every query); its URL is current and raising the client's
  timeouts cannot help a rate limit. Its failures were already isolated (counted, the 44 feeds unaffected), but every run waited
  minutes on it, so it now has the Google News search alternative described above.
  `NseApi.live_equities` also works with the client's default `Referer`.
- **curl on Windows**: `curl.exe` uses Schannel, which fails behind a network that blocks the certificate-revocation check
  (`CRYPT_E_NO_REVOCATION_CHECK`); add `--ssl-no-revoke` for diagnosis. The project's own client (BoringSSL via curl_cffi) is not
  affected. Plain curl gets a 403 on the NSE home page but still receives the cookies, and the API answers with them (see
  `docs/OPERATIONS.md`).

### Found by an independent review of this change, and fixed

- **A manual, forced or preview run after about 15:04:40 could never fetch quotes.** The live requests were bound by the report
  deadline, and once that is past the HTTP client starts no attempt at all, so `ideas --dry-run` or `--force` late in the afternoon
  failed with "quotes unavailable" without a single request (the test double ignored deadlines, which hid it, and now behaves like the
  client). A run that starts when the deadline is nearly over gets a 90-second budget of its own, and `--force` now sends the
  frozen report instead of scanning again.
- **`closing_scan` gave up after about five minutes** (5 tries a minute apart) although its window runs 30 minutes, so an NSE blip that
  cleared at 14:40 still lost the day's report. It now has a try for every minute of the window.
- **A split or bonus name polluted the others.** A name left out for a price-basis change kept its meaningless move (-50 % for a 1:2
  split) in the day's table, so every stock of its sector inherited a bogus "sector peers -9.9 %" and the breadth and percentile ranks
  were skewed. Left-out names are no longer in the table, and a stored history that disagrees with the exchange's previous close for
  more than 40 % of the names stops the scan.
- **A results meeting on a Saturday, Sunday or holiday was ignored**, so the BTST gate for an overnight trade into results missed it
  (a Friday BTST would carry a Saturday announcement across the weekend). Such a meeting now counts like one announced after the close
  of the session before it.
- **The cause of "no quotes" never reached the alert or the notice** (they always said "blocked or offline"), contrary to the runbook:
  the reason NSE's client gave, such as "no live-quote endpoint knows SECURITIES IN F&O ... did NSE change its API?", is now in both.

### Found by the first real deployment (Oracle Cloud, Ubuntu 24.04, 2026-11-02), and fixed

- **The installer waited five minutes for nothing and then failed on a stock Ubuntu server.** Its check for a busy package manager
  matched process names, and the resident `unattended-upgrade-shutdown` helper that every Ubuntu server runs idle has the process name
  `unattended-upgr`, so "another package manager is running" was always true while no apt or dpkg lock was held. `apt_busy`
  (`deployment/lib/system.sh`) now asks the lock files themselves (`fuser` on the dpkg and apt locks) and only without `psmisc`
  falls back to names, with the unattended upgrade matched by its command line and never the helper. A regression test covers
  the idle helper, a held lock and the fallback.
- **The preflight reported a blocked NSE that was not blocked.** The connectivity probe used curl's own User-Agent, and NSE's CDN
  drops such a request without any answer (HTTP 000, even for a real file), while it answers a browser's (200) at once, as the
  application's client, which impersonates a browser, always did. The probe now sends a browser's User-Agent
  (`PROBE_USER_AGENT` in `deployment/install.sh`); any HTTP answer still proves the route works. A test pins both.
- **A fresh install was warned "training overdue: last successful training was never" and the warning was sent to Telegram.** The
  check read the age of the last `train` job, and the first-time bootstrap trains through the job function without recording one,
  so two freshly validated models still counted as "never trained" until the first Saturday training (the alert repeats every
  12 hours). The age is now the younger of the last `train` job and the active model itself, whichever way it was trained; a
  model and a training that are both old still warn. Two tests cover it and fail without the change.

## [1.0.0] - 2026-11-01

First complete release: a shadow-mode NSE large-move intelligence engine that runs unattended on a 1 GB / 8 GB Ubuntu
server with native systemd (no containers).

### Upgrade notes

- **Feature version `fv2`.** The audit fixes below changed what some features mean, so `FEATURE_VERSION` is now `fv2`. Models trained
  with `fv1` are retired automatically and the scans use the labelled heuristic baseline until the next training run has produced
  an `fv2` model. A fresh install trains during the bootstrap; on an existing server run `sudo monster-ai train` (or wait for
  Saturday 02:00 IST).
- **New migrations** `0004_graph_edge_history.sql` (append-only event-graph edge history and event provenance columns) and
  `0005_drop_unused_macro_tables.sql` (drops the `macro_observations` and `econ_calendar` tables nothing read, and the registry rows of
  their two sources; it cannot be undone, see *Removed*); migrations run on start.
- **Settings that no longer exist:** `FRED_API_KEY`, `TRAINING_DOW` and `MIN_NEWS_MODEL_DAYS` are ignored if they are still in
  `monster.env`; delete them at your leisure.
- **New settings** (all with safe defaults, see `.env.example`): `HTTP_TRANSPORT`, `HTTP_IMPERSONATE`, `HTTP_MAX_RESPONSE_MB`,
  `DATA_MAX_MB`, `BACKUP_MAX_MB`, `EVENT_RETENTION_DAYS`, `QUOTE_RETENTION_DAYS`, `RECORD_RETENTION_DAYS`, `OPS_RETENTION_DAYS`,
  `LOG_FORMAT`, `LOG_RETENTION_DAYS`, `LOG_MAX_MB`. `MONSTER_TIMEZONE` now selects the zone of **log stamps only** (market logic is
  always IST).
- **Optional AI settings** (everything stays off until a key is set): `GROQ_API_KEY`, `LLM_FALLBACK_MODEL`, `LLM_REASONING_EFFORT`,
  `LLM_MAX_SECONDS_PER_RUN`, `LLM_MAX_OUTPUT_TOKENS`, `LLM_WEB_SEARCH`, `LLM_WEB_MAX_PER_RUN`, `LLM_WEB_MAX_PER_DAY`, `TYPESAFE_API_KEY`,
  `JEV_MODEL`, `JEV_TIMEOUT_SEC`, `JEV_MAX_ITEMS_PER_RUN`, `JEV_MIN_CONFIDENCE`. `LLM_API_BASE` + `LLM_MODEL` still select another
  OpenAI-compatible provider; `GROQ_API_KEY` is never sent to any endpoint but Groq's. The web check is **on by default as soon as
  a Groq key is set**. A check costs about 7,000 tokens (measured on a live key) and Groq's free tier gives a model 200,000 tokens a
  day, so it is capped at 8 per news job and 16 per IST day (`LLM_WEB_MAX_PER_RUN`, `LLM_WEB_MAX_PER_DAY`) to leave the extraction its
  share; `LLM_WEB_SEARCH=false` turns it off. After adding a key: `sudo systemctl restart monster-ai` and `sudo monster-ai ai-test`.
- **New dependency** `curl_cffi` (binary wheels for linux x86-64 and aarch64, Python 3.10-3.13; the installer and the CI wheel guard
  cover it).
- **logrotate** no longer touches the application's own log files (they are daily files pruned by the application); the installer
  replaces the policy.

### Added

- **Daily pipeline** (`automation/`): ledger-based scheduler daemon (sd_notify watchdog, one child process per job,
  preemption of non-critical jobs, catch-up, retries) and the full timetable - global and India data collection, news,
  08:45 preliminary scan, 09:00 pre-open book, 09:08:45 final scan with Telegram delivery by 09:12,
  end-of-day ingestion, evaluation, backup, housekeeping, weekly training / universe / calendar refresh, model sync.
- **Resumable 12-step bootstrap** that prepares everything on a new server (universe, 10 years of history, official
  bhavcopy, F&O history, board meetings, news, first models, smoke scan, Telegram test) and survives crashes, reboots
  and out-of-memory kills.
- **Browser-grade, rate-limit-friendly data collection** (`ingestion/http.py`): every scraped source (NSE, Yahoo, RSS, GDELT) goes
  through one **curl_cffi** client with a real browser TLS / HTTP-2 fingerprint, falling back to `requests`
  when curl_cffi is unusable. Per-source rate limiting adapts (a 429 doubles the spacing, successes ease it back) with jitter,
  `Retry-After`, exponential back-off, a circuit breaker, caller **deadlines** (the pre-open polls cannot push the report past its
  cutoff), HTTPS-only, a response-size cap, secret-free errors and per-source counters logged after every job. Requests are batched
  or conditional wherever the provider allows: one official NSE file per day for all stocks, one Yahoo `spark` request per 20 global
  series (skips the per-series fetch when nothing changed), conditional GET (`ETag` / 304) for RSS feeds, one pre-open call for the
  whole book.
- **Exchange calendar that works for any month and year** (`automation/calendar.py`): shipped official list + daily NSE refresh + a rule
  layer (fixed-date holidays and Good Friday from the Easter algorithm) for years NSE has not published yet + live confirmation
  from the pre-open book (`holiday_inferred`) + inference from missing index bars + operator overrides that survive upgrades; health
  warns 21 days before the official list ends; official data always replaces rules.
- **Bounded storage** (`automation/retention.py`, CLI `storage`): every table, folder, cache, log and backup has a retention rule applied
  nightly in small foreign-key-safe transactions, shorter windows under disk pressure, a `DATA_MAX_MB` budget, a backup size budget,
  a `VACUUM` policy (due, worthwhile and safe), a stored result per run and health checks (budget, WAL size, retention staleness).
- **Logging you can trace a bug with** (`app/logging_setup.py`, `app/logview.py`, `app/diagnostics.py`): daily files per process role
  (never renamed, safe for concurrent processes, size-capped), a shared `errors-<day>.log` of every WARNING and above, millisecond stamps
  with the zone offset, per-job tags (`[job key#attempt r<run id>]`), duration, peak-memory and network lines after every job, crash hooks
  (`excepthook`, threads, unraisable, `faulthandler`), secret redaction in the formatter, optional JSON lines, CLI `logs` (filter by
  level, job, text, time; follow) and `diag` (redacted bundle).
- **Server time zone independence**: UTC storage, explicit IST market logic, a process zone pinned to UTC, an IST fallback without tzdata,
  a display zone for logs; guarded by an AST scan for zone-reading APIs and a five-zone subprocess matrix test.
- **Operational hardening**: a heartbeat file next to the database heartbeat (a locked database no longer looks like a dead daemon), a
  `daemon_restarted` alert after an unclean stop, heavy jobs deferred and training refused on a critical disk, the `monster-ai version`
  command with the code fingerprint.
- **Price-basis repair** (`ingestion/basis.py`, `ingestion/adjust.py`): vendor bars are replaced by the official file in the morning job and
  again for the last three sessions in the nightly housekeeping; a split or bonus of a stock that already has history is detected from the
  exchange's own previous close (also from the pre-open book on the ex-date morning), the vendor is asked for exactly that stock until
  it reports the split, and the operator is told after six sessions (`price_basis_unresolved`); official rows are re-based at read time
  so the order of arrival never matters; on the ex-date morning the stock is scored without its meaningless opening gap.
- **Data layer** (`ingestion/`, `database/`): Yahoo, NSE archive and NSE API clients; validation; freshness / data-quality gate with an
  explicit blocked / degraded / ok verdict; source catalogue ([docs/DATA_SOURCES.md](docs/DATA_SOURCES.md), now with transport and fetch strategy per
  source); SQLite (WAL) with versioned migrations; verified, optionally gpg-encrypted online backups; Muhurat and short-session handling;
  a pre-open book stamped on an earlier session is ignored instead of being mistaken for today's.
- **One feature path for training and live scoring** (`features/`): technical, F&O, global, regime, calendar, sector and
  similar-day features, strictly point-in-time (price/volume shifted one session, global series by availability),
  guarded by leakage tests, with a float32 column-streamed builder sized for 1 GB.
- **Models** (`models/`): LightGBM specialists + meta model, isotonic calibration, quantile range heads, walk-forward
  training with embargo, fixed honesty gates (calibration, Brier skill, edge over simple baselines with a day-block bootstrap),
  artifact format without pickle, champion/challenger registry with integrity checks, rollback and automatic fallback to
  a labelled heuristic baseline; every model records the fingerprint of the code that trained it.
- **Ranking** (`ranking/`): eligibility, conviction tiers capped at WATCH unless the model is validated, contradiction and
  cost checks, explicit "no setup" when nothing qualifies, baseline pickers for comparison (ATR, 20-day range, volume, yesterday's range,
  opening gap, random). The uncertainty ceilings of the tiers are learned per model, as a function of probability, from the
  uncertainty of its own held-out top names (an absolute ceiling made STRONG unreachable on real data), and the "no setup" reason
  names the real cause with counts. The news / event-graph enrichment is applied to the names the report can actually list.
- **News and event graph** (`news_intelligence/`, `event_graph/`): deduplication, entity linking, rule-based event
  extraction with optional Groq / Jev enrichment (see the next entry), source quality, "already priced in" detection
  (market reactions are refreshed after each news run and again at the final scan), supply-chain and sector impact
  propagation with a seeded expert-prior graph (seeded automatically, re-seeded when the universe changes, and re-measured
  weekly from returns with FDR control; edges without a price series stay labelled priors).
- **Optional AI for the news layer** (`news_intelligence/llm.py`, `jev.py`, `remote.py`, `ai_check.py`): **Groq** (`openai/gpt-oss-120b`,
  falling back to `openai/gpt-oss-20b`; strict JSON-schema output with a per-model capability ladder; every claim needs a verbatim
  quote) and **Jev** (TypeSafe AI, `jev-1.14.0`: one request answers five typed questions - event type, rumour, denial, severity and
  direction - with calibrated probabilities; merged conservatively: it may add information, never clear a rule-set rumour flag or flag
  an official filing). Both share one set of rails: a 14-day result cache by text, per-run call and wall-clock budgets, `Retry-After`
  handling, model fallback, a rate-limit cooldown that survives between runs, a circuit breaker, per-service coverage flags and
  `source_health` rows (so repeated failures raise `health_source_<id>`). Items the rules already discard are never sent; only public
  news text leaves the server. Where both answer, a direction on which they disagree is left unknown. They improve the news evidence
  (catalysts, rumour flags, chains), **not** the calibrated forecast. `monster-ai ai-test` checks the keys with one made-up item
  (`--list-only` spends no tokens) and `doctor` probes them for free.
- **Complete context for both AI readers** (`news_intelligence/context.py`): besides the whole stored text (up to 6 000 characters; it was
  3 500 for the LLM and 2 500 for Jev), the LLM prompt and Jev's state carry when the item was published and when it is checked (IST),
  the source with its class and tier, the filing category, the issuer with its sector, the companies the entity matcher found and the
  web brief; the LLM prompt also defines every event type (the definitions moved to `events.EVENT_DESCRIPTIONS`, shared with Jev).
  Prices, returns and the rules' verdict are left out on purpose (an independent second reader). Free text is single-line, capped,
  stripped of control characters and of the prompt's block tags. Cache keys follow everything the prompt shows (the publication minute,
  the source, the matched companies, the web brief) but not the time of the check.
  Prompt versions `v4` / `j2` retire older cached answers. The LLM prompt anchors the direction scale (a bond issue is about 0, an
  order win +0.5 to +1, a fraud or a default -0.5 to -1): measured live, the size of an unanchored direction varied a lot between
  identical requests while its sign was stable.
- **Optional web check** (`news_intelligence/web.py`): Groq's server-side `browser_search` tool on the gpt-oss models, used before the
  extraction for company news that is not first-hand and is a rumour or could matter. A request of its own, because Groq documents that
  browser search cannot be combined with structured outputs; it returns a small brief validated field by field, which joins both
  readers' context. By itself it changes one thing: a *verified denial* (a page of a reputable publisher - an exchange, an official
  body, a wire, a business newspaper - that the search really returned and the reply cites contains the wording of a denial) turns the
  item into a refuted rumour (half the severity, the rumour horizon); a confirmation changes nothing. Own budgets
  (`LLM_WEB_MAX_PER_RUN`, 8 per job, and `LLM_WEB_MAX_PER_DAY`, 16 per IST day), 30 % of the LLM time box
  (a hard bound: every request of the three services is cut to the time left in the run and not started when almost none is
  left), 1-day cache, circuit breaker, cooldown and `source_health` row (`llm_web`); HTTP 413 (a request above the plan's per-minute
  token limit) or a model the provider says lacks the tool backs it off for 6 h; never used by a replay, even when a researcher is
  injected; shown by `ai-test`,
  `doctor`, the test message and the news job's AI note. It shares Groq's per-model quota with the extraction, whose fallback model
  absorbs the contention.
- **Text that tries to instruct the AI is withheld** (`news_intelligence/context.py`, `pipeline.py`): a news item whose text addresses
  the AI ("ignore all previous instructions", a `SYSTEM:` line, a chat-template token, "set severity to 1", a closing `</article>`
  tag) is read by the rules only and sent to no AI service; it is logged as a WARNING and counted in the news job's message.
  Deliberately narrow, so that ordinary regulatory wording is not flagged.
- **Verified against the live services (2026-11-02)** with the project's own keys: `ai-test` for all three services, a live end-to-end
  news run on a scratch database, a burst against Groq's free-tier limits (14 of 14 items extracted, 429s handed over to the fallback
  model), a prompt-injection test and dozens of live web searches. What it found, and what was done about it:
  - About one web search in six came back from Groq as HTTP 400 `tool_use_failed` (the model delivered its answer as a call to a tool
    named `json`). The first version took any 400 that mentioned a tool for a refusal of the tool and switched the web check off for
    six hours. It is now the model's slip: the brief is read from the error body when it has the shape of one (unverified, so it can
    inform but never discount an item), otherwise the search is retried once.
  - An earlier note, that Groq's free tier is too small for a web search (HTTP 413), was wrong: a search works there. It costs about
    7,000 tokens, charged to the per-minute **and** the per-day limit, so about 28 searches use up a free-tier model's 200,000 tokens
    a day. Hence the daily cap `LLM_WEB_MAX_PER_DAY` (16), kept in `kv` across runs and counted per IST day.
  - Jev lists only the aliases `jev-latest` and `jev-preview`; the pinned `jev-1.14.0` answers although it is not listed, and
    `ai-test` no longer warns about it.
  - The size of the direction the LLM returned varied between identical requests (its sign was stable): the scale is anchored now.
- **Found by a live news run on real feeds and filings, and fixed** (`news_intelligence/events.py`, `pipeline.py`, `automation/intel.py`):
  one `collect_news` job (44 of 44 feeds, 1 098 new items, 316 s, 144 MB peak) produced 855 events from 970 items (792 articles and
  178 filings), and reading them showed what the offline tests could not.
  - **Re-published filings are dropped.** 31 of the 178 filings that became events were newspaper copies of results, audio recordings
    and transcripts of the call: typed `earnings_result` at severity up to 0.6, "new", and 9 of Jev's 26 judged events (35 %). The
    routine-category rule only applied when the text rules found nothing, and the words of the results made them find something. A
    filing whose subject or category says it is a newspaper publication or advertisement, an audio / video recording or a transcript
    is now routine whatever it repeats (`_REPUBLICATION`); the first-hand filing remains the event.
  - **A type resting on passing body words cannot be a headline event.** About one article in seven (111 of 792) was typed by one or
    two rules that matched only in the body: "escalation" in a power-demand story became a geopolitical conflict at severity 0.95,
    "divest" in a story about FII selling an acquisition at 0.9, "attack" on a crypto exchange a conflict at 0.95. Such a type is kept
    (it may be right) but its severity is capped at 0.35 and the event's rules say `weak_evidence`; headline matches, filing
    categories, first-hand sources and body evidence of 3.0 rule weight or more are untouched. The average severity of the affected
    items fell from 0.63 to 0.35.
  - **The web check needs a listed company.** One of the two searches of that job went to a market-wide FII story typed as an
    acquisition; a story with no listed company has nothing to look up, whatever its type.
  - **A company-type event that names no listed company is a lead, not a headline** (`cap_unattributed`). "GST collections jump 22.9%"
    was typed `earnings_result` at severity 0.95 (two body rules, 3.3 of rule weight, so the cap above did not apply) and was the top
    catalyst the report would have listed for five of sixteen unrelated stocks. A company-type event whose text names no listed
    company keeps its type but gets the same 0.35 cap, recorded as `no_listed_company`; exchange filings, official statements and a
    company's own releases are exempt, and a verified AI answer may still raise it. Of the 855 events, 31 were of this kind (macro or
    sector scope, company type): average severity 0.61, 20 of them at 0.6 or more; with the cap the average is 0.36 and one is left
    at 0.6 or more. Together with the changes above, the events at severity 0.8 or more fell from 69 to 39 and those at 0.6 or more
    from 331 to 209 on the same data (rules only: the earlier figures include the AI-judged items).
  - **`news.last_run` describes the whole job.** A backlog is drained in batches under one AI budget, and the pipeline saved the
    counts of the last batch only (`llm_used` 0, `jev_skipped_budget` 241) beside a job message that said "llm 9 used, jev 30
    used". The job now saves its summed totals.
  - Not changed, on purpose: GDELT timed out on every query from this network (6 queries failed, then its circuit opened and the rest
    failed at once, about two minutes in all) and the job carried on with the feeds and filings, as designed; Groq's daily token
    quota ran out part-way (earlier tests of the same day had spent it) and the extraction fell back to the rules for the rest of
    the job with a 353 s cooldown.
  - **Verified on live NSE payloads, which had only been seen as recorded fixtures** (2026-11-02, 13:10 IST, market open): the
    pre-open endpoint (`market-data-pre-open?key=ALL`) still served the day's book, stamped 09:09:31 IST, and the parser read all
    2 411 rows with every field present, no non-positive price and the IST stamps converted to UTC; the `giftnifty` block of
    `marketStatus` parsed (price, change, previous close, timestamp). The book was read after the session, so the 09:00-09:12
    polling window itself, and the delivery latency before the cutoff, have still not run live.
- **Credential hygiene for the LLM endpoint** (`app/config.py`): `GROQ_API_KEY` is sent only to a URL that is exactly
  `https://api.groq.com...` (a look-alike such as `https://api.groq.com.evil.example` or a URL with credentials or a backslash is
  not Groq's: it is refused at load or treated as another provider that never gets the key), and plain `http` is accepted for
  loopback hosts only (`localhost`, `127.0.0.1`, `::1`), not for anything that merely starts with them.
- **Notifications** (`notifications/`): Telegram client, 4096-character-safe renderers (pre-open, final, evening, alerts,
  failure notices, failed-service alerts), idempotent delivery ledger, deadline guard, queued alerts flushed when Telegram returns; catalysts show their
  provenance (rumour flag, source class, independent sources, conflicts) and the regime line shows GIFT Nifty while its quote is fresh.
- **Messages made for a phone** (`notifications/style.py`, `report.py`): a short head, an icon before every label, medals and tier badges,
  a probability bar for calibrated probabilities only, traffic-light dots for data feeds, a severity colour on every alert with the
  first command to run (read-only commands only), clock faces that follow the hour, English month names whatever the server's locale.
  The icons only decorate: the words beside them carry every fact, and the honesty labels (SHADOW, PRELIMINARY, RUMOUR, "not
  calibrated", LATE) are unchanged. `monster-ai telegram-preview` (`notifications/samples.py`) sends one sample of every message type with
  made-up data, each marked *SAMPLE PREVIEW*.
- **Evaluation and live record** (`backtesting/`): outcomes are attached to frozen predictions without ever editing them; only the effective
  snapshot of a (date, stage) is scored and a day is scored once; the evening report prints a **live record** (hit rate, range-forecast
  error, 80 % interval coverage, direction accuracy, realised range against the baselines, NDCG@5 and rank IC) over 20 days, 60 days and
  all days, each with its sample size and a 95 % day-block interval, plus slices (rank, tier, regime, sector, liquidity, event) in
  `eval_metrics`.
- **Reproducible snapshots**: the snapshot and its predictions are frozen in one transaction (an interrupted legacy freeze is repaired
  on retry), the stored feature rows carry a re-verifiable hash, the freshness of every input is recorded, and `code_version` is the
  release plus a fingerprint of the code (`app/release.py`).
- **Health and self-healing** (`automation/health.py`): data freshness, missing sessions, model age and status, source
  failures, backup age, disk and RAM, storage budget and retention, calendar coverage, delivery failures and live calibration drift;
  de-duplicated alerts; RSS watchdog per job; a training attempt that dies from memory pressure is retried with a progressively
  leaner profile.
- **Deployment** (`deployment/`): one-command install from Windows (`deploy_remote.ps1`) or Linux/macOS
  (`deploy_remote.sh`); idempotent installer with swap, NTP, dedicated user, atomic release swap, automatic rollback,
  Telegram chat-id discovery and verification; six hardened systemd units; journald limits and a logrotate policy for what the
  application does not rotate itself; uninstaller.
- **External training (optional)** (`models/external.py`, `models/bundle.py`, `models/signing.py`,
  `database/object_store.py`): a GitHub Actions worker trains and publishes HMAC-signed bundles to S3-compatible storage;
  the server pulls, verifies (pointer, archive hash, manifest signature, every file) and registers them as ordinary
  challengers. Off-box database backups use the same store with retention; `backup --restore-latest --remote` recovers a
  lost server.
- **Dashboard (optional)** (`dashboard/`): read-only FastAPI view, loopback by default.
- **CLI** (`app/cli.py`): `init`, `doctor`, `bootstrap`, `daemon`, `run-job`, `jobs`, `scan`, `evaluate`, `train`, `backtest`, `models`,
  `backup`, `health`, `storage`, `logs`, `diag`, `telegram-setup`, `telegram-test`, `telegram-preview`, `ai-test`, `dashboard`, `env-example`, `version`.
- **Documentation**: [README](README.md), [architecture](docs/ARCHITECTURE.md) with 20 Mermaid diagrams (context, whole-code architecture, data
  flow, timetable, scan sequence, news and the optional AI services, training, scheduler / bootstrap state machines, HTTP stack, bulk fetching, calendar, time zones,
  retention, logging, deployment), [operations runbook](docs/OPERATIONS.md), [deployment guide](deployment/README.md), the generated
  [data-source catalogue](docs/DATA_SOURCES.md) and `.env.example` (both checked against the code by tests) and the
  [blueprint traceability matrix](docs/BLUEPRINT_TRACEABILITY.md); a test checks every diagram's structure and every link and anchor.
- **Tests**: hermetic suite (no network; temp directories; injected clock) with coverage gates in CI, shellcheck on every
  script, a wheel-availability check for the pinned dependencies, time-zone, retention, logging, calendar, bulk-fetch and documentation guards.

### Fixed (found by auditing the build against the blueprint)

- **Prices and universe.** Official rows are re-based on the post-split basis at read time (a vendor split reload no longer leaves raw
  official bars that look like +100 % / -50 % returns); F&O membership is point-in-time; training panels include delisted and
  quarantined names (survivorship); suspect bars are handled; the vendor reload is atomic; the universe liquidity window is computed in IST.
- **Features and signals.** The relative-volume baseline excludes the current session; the similar-day pool and window are identical in
  training and live (train/serve parity); stale global series are blanked after 4 days (7 for European and Asian equity series) instead
  of carrying a constant; the results-calendar horizon is aligned with training (`RESULTS_HORIZON = 10`); sector dispersion is
  leave-one-out; the OI / price contradiction rule is reachable; the heuristic baseline honours `UNUSUAL_K`; the model version stamp uses
  the injected clock; the similar-day validity count joins the meta-model context and each name is judged against the specialist-disagreement
  limit of the model that scored it.
- **News and event graph.** Append-only edge history (`graph_edge_history`) with point-in-time edges as an opt-in for replays; poison items
  are parked instead of blocking the queue (`processed = 2`, retry counter `news.retries`); market-reaction ("priced in") handling;
  measured-versus-prior edges are labelled `[expert prior]` / `[measured]` /
  `[broken]` in the chains; raw news text older than the retention is actually deleted.
- **Evaluation.** A forced re-scan was counted twice and pre/final snapshots were pooled: only the effective snapshot per (date, stage) is
  scored, once per day; `n_days_30/90` were never persisted, so the calibration-drift and negative-skill health checks could never fire;
  `eval_metrics` writes are idempotent; hit-rate intervals are day-block bootstraps (shown as n/a below five days); calibration reads
  bars in one query instead of one table scan per day; MFE / MAE are documented as hindsight excursions, not a strategy; the
  preliminary-stage "gap" baseline was really yesterday's range and is now `recent_range`.
- **Operations.** The filings window follows the previous trading day (it missed filings after a long weekend); the app's logs no longer
  clash with logrotate (`copytruncate` emptied daily files); concurrent job processes no longer share a rotating file handler; the job tag
  of log lines is set; the daily bar flow never replaced provisional vendor bars with the official file once vendor coverage was complete;
  the bootstrap used the UTC date where the exchange date is meant; a transient I/O error while reading the active model retired it for good
  (now the model is kept, the labelled baseline serves that scan and `model_load_failed` is raised; only definite damage retires a model);
  a heavy job waiting for RAM or disk held back every light job behind it in the queue; a failed gpg run with `BACKUP_PASSPHRASE` set left an
  unencrypted backup that would have been uploaded off-box (it now stays local, `backup_encryption_failed` and a health warning are raised);
  the evening summary's data-quality notes were never filled (now: preliminary-only scoring, provisional outcomes, degraded or late snapshots);
  the report and the dashboard said "range >=" where labels and evaluation use "range >";
  `telegram-setup` and `doctor` reported only the exception type ("rejected the token: DataUnavailable") for every Telegram failure, which
  hid whether the cause was a wrong token, a wrong chat or a blocked network (they now print Telegram's own reason, with secrets redacted).

### Removed

Work that cost requests, CPU or disk and fed nothing, and dead code, taken out completely (code, tests, documentation). The saving is
small in every case except the first; the gain is less to run, maintain and explain.

- **The 08:30 `preflight` job and its `preflight_failed` alert.** It repeated what `collect_global` (06:00) and `collect_india` (08:00)
  had already done (both retry until their windows close), then dry-ran the whole preliminary scan and threw the result away; the
  real `prelim_scan` ran the identical scan 15 minutes later. This is the only removal with a measurable saving: one dry-run scan (about
  11 seconds on the development PC) and a few provider requests every trading morning. Its one useful side effect stays: `prelim_scan` now
  repairs the previous session once more (a two-minute budget for the vendor fallback) when that session is below 90 % coverage, and
  costs one query when it is complete.
- **Economic-calendar and FRED collection** (`ingestion/macro.py`, `NewsCollector.collect_econ_calendar`, their retention rules, source
  entries and `FRED_API_KEY`). Nothing ever read the data: one small request a day (FRED did nothing without a key) and two tables.
  Migration `0005_drop_unused_macro_tables.sql` drops the `macro_observations` and `econ_calendar` tables and the registry rows of
  the two sources (a stale failure counter would otherwise keep raising a health warning). It is the first migration that removes
  data and it cannot be undone; the rows are in any earlier backup.
- **Readers nothing called:** the live option-chain reader (`NseApi.option_chain_summary`, its parser and fixtures),
  `YahooClient.fetch_intraday` and `HttpClient.get_bytes`.
- **Settings nothing read:** `TRAINING_DOW` (the weekly training day is fixed in the timetable, Saturday 02:00 IST, so changing the
  setting did nothing) and `MIN_NEWS_MODEL_DAYS`.
- **Dead code:** five unused exception types (`StaleDataError`, `ValidationFailed`, `LeakageError`, `CutoffMissed`, `JobTimeout`),
  `now_ist`, `set_oom_score_adjust`, `parse_hhmm`, `backfill_fno_history`, `require_integrity`, `finite_mask`, `utc_now_iso`,
  `sleep_budget`, `spec_for`, `bar_date_range` and `wait_for_preopen` (use `await_preopen(...).share`).

### Verified on real data

- Bootstrap against the live providers: 221-symbol universe, 254 instruments with 10 years of history, 12 official
  bhavcopy sessions, 573 F&O days, 3,249 board-meeting rows, 44 of 44 RSS feeds.
- A real dry-run scan completed in about 11 seconds on a development PC.
- A model trained on that data (221 stocks, five years, eight walk-forward folds) passed every gate: expected calibration
  error 0.014, Brier skill 0.17 against climatology, top-5 hit rate for a range of at least 3 % of 79 % against 73 % for the
  best simple baseline (20-day range) and 31 % for random picks. The advantage over the best baseline is real but modest
  (+6.7 points, 90 % interval +3 to +10 points); direction is not predictable (AUC 0.55) and the system says so. (Measured before the
  `fv2` feature fixes; the first training run on a new server produces the figures for the current schema.)
- Training memory on the real data was reduced to fit the 1 GB server's budget (smaller histogram sample, chunked
  statistics, float32 out-of-fold arrays, column-streamed feature matrix, per-fold memory release): measured peaks are
  about 640 MB resident for the feature build and 565 MB for the model training in the light profile. A training
  attempt that still dies from memory pressure is retried automatically with a leaner profile.
- The curl_cffi transport, the Yahoo `spark` probe and the NSE endpoints were exercised against the live providers from a development PC.
- The time-zone guarantees are tested: the five-zone subprocess matrix (UTC, US Pacific, India, New Zealand, UTC+14) passes, and the whole
  suite passes both in the development PC's own zone (India) and with the process zone forced to US Pacific (DST).

### Known limitations

- The Linux installer and systemd units were validated by static analysis, stubs and ShellCheck, not on a real host.
- The NSE pre-open JSON parser was verified against recorded fixtures and, later, against one live NSE book (2026-11-02, read after the
  session had closed: see "Found by a live news run"); the 09:00-09:12 polling window itself has not run live yet. NSE blocks some
  data-centre address ranges; the system degrades and reports it.
- No soak test was run on a real 1 GB server; the memory budget comes from measured peaks on a development PC.
- Edge over simple baselines is modest and was measured on one market regime; a live shadow record of weeks to months is
  the real test. Nothing in this project places orders.
- News features are not model inputs (they reach the report); VWAP is a daily proxy; pre-open order imbalance is parsed and stored but
  not used; survivorship bias is reduced, not eliminated (names the database never ingested stay unknown). See the deviations and the
  limits table in [docs/BLUEPRINT_TRACEABILITY.md](docs/BLUEPRINT_TRACEABILITY.md).
- Lunar-calendar exchange holidays of a year whose official list is not yet published are confirmed live on the day (no report is sent,
  the operator is told), not predicted.
- The Groq and Jev integrations, the web check included, were verified live on 2026-11-02 (see *Verified against the live services*)
  and tested against scripted responses; that is one day's behaviour of two third-party services. Groq retires models from time to
  time, so a model name may need updating (the fallback model and `ai-test` surface it), and the shape of a browser-search reply is
  read defensively. The size of the LLM's direction is noisy even with an anchored scale: the AI only enriches the news evidence and
  never changes the calibrated forecast. Jev judges news; it was not evaluated as a price forecaster. The web check cannot be
  evaluated historically (the web of the past is not available), so its value is judged by reading its evidence, not by a backtest.
