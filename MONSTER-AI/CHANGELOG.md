# Changelog

All notable changes to MONSTER AI. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/). Database migrations are forward-only and additive, so every
release below can be installed over the previous one (`deployment/install.sh` is idempotent and rolls back by itself).

## [1.0.0] - 2026-11-01

First complete release: a shadow-mode NSE large-move intelligence engine that runs unattended on a 1 GB / 8 GB Ubuntu
server with native systemd (no containers).

### Upgrade notes

- **Feature version `fv2`.** The audit fixes below changed what some features mean, so `FEATURE_VERSION` is now `fv2`. Models trained
  with `fv1` are retired automatically and the scans use the labelled heuristic baseline until the next training run has produced
  an `fv2` model. A fresh install trains during the bootstrap; on an existing server run `sudo monster-ai train` (or wait for
  Saturday 02:00 IST).
- **New migration** `0004_graph_edge_history.sql` (append-only event-graph edge history and event provenance columns); migrations
  run on start and are additive.
- **New settings** (all with safe defaults, see `.env.example`): `HTTP_TRANSPORT`, `HTTP_IMPERSONATE`, `HTTP_MAX_RESPONSE_MB`,
  `DATA_MAX_MB`, `BACKUP_MAX_MB`, `EVENT_RETENTION_DAYS`, `QUOTE_RETENTION_DAYS`, `RECORD_RETENTION_DAYS`, `OPS_RETENTION_DAYS`,
  `LOG_FORMAT`, `LOG_RETENTION_DAYS`, `LOG_MAX_MB`. `MONSTER_TIMEZONE` now selects the zone of **log stamps only** (market logic is
  always IST).
- **New dependency** `curl_cffi` (binary wheels for linux x86-64 and aarch64, Python 3.10-3.13; the installer and the CI wheel guard
  cover it).
- **logrotate** no longer touches the application's own log files (they are daily files pruned by the application); the installer
  replaces the policy.

### Added

- **Daily pipeline** (`automation/`): ledger-based scheduler daemon (sd_notify watchdog, one child process per job,
  preemption of non-critical jobs, catch-up, retries) and the full timetable - global and India data collection, news,
  preflight dry run, 08:45 preliminary scan, 09:00 pre-open book, 09:08:45 final scan with Telegram delivery by 09:12,
  end-of-day ingestion, evaluation, backup, housekeeping, weekly training / universe / calendar refresh, model sync.
- **Resumable 12-step bootstrap** that prepares everything on a new server (universe, 10 years of history, official
  bhavcopy, F&O history, board meetings, news, first models, smoke scan, Telegram test) and survives crashes, reboots
  and out-of-memory kills.
- **Browser-grade, rate-limit-friendly data collection** (`ingestion/http.py`): every scraped source (NSE, Yahoo, RSS, GDELT, FRED, the
  economic calendar) goes through one **curl_cffi** client with a real browser TLS / HTTP-2 fingerprint, falling back to `requests`
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
  extraction with an optional LLM (every claim needs a verifiable quote), source quality, "already priced in" detection
  (market reactions are refreshed after each news run and again at the final scan), supply-chain and sector impact
  propagation with a seeded expert-prior graph (seeded automatically, re-seeded when the universe changes, and re-measured
  weekly from returns with FDR control; edges without a price series stay labelled priors).
- **Notifications** (`notifications/`): Telegram client, 4096-character-safe renderers (pre-open, final, evening, alerts,
  failure notices), idempotent delivery ledger, deadline guard, queued alerts flushed when Telegram returns; catalysts show their
  provenance (rumour flag, source class, independent sources, conflicts) and the regime line shows GIFT Nifty while its quote is fresh.
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
  `backup`, `health`, `storage`, `logs`, `diag`, `telegram-setup`, `telegram-test`, `dashboard`, `env-example`, `version`.
- **Documentation**: [README](README.md), [architecture](docs/ARCHITECTURE.md) with 18 Mermaid diagrams (context, whole-code architecture, data
  flow, timetable, scan sequence, training, scheduler / bootstrap state machines, HTTP stack, bulk fetching, calendar, time zones,
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
  are parked instead of blocking the queue (`processed = 2`, retry counter `news.retries`); market-reaction ("priced in") handling; the
  economic calendar is keyed by country, title and date; measured-versus-prior edges are labelled `[expert prior]` / `[measured]` /
  `[broken]` in the chains; raw news text older than the retention is actually deleted.
- **Evaluation.** A forced re-scan was counted twice and pre/final snapshots were pooled: only the effective snapshot per (date, stage) is
  scored, once per day; `n_days_30/90` were never persisted, so the calibration-drift and negative-skill health checks could never fire;
  `eval_metrics` writes are idempotent; hit-rate intervals are day-block bootstraps (shown as n/a below five days); calibration reads
  bars in one query instead of one table scan per day; MFE / MAE are documented as hindsight excursions, not a strategy; the
  preliminary-stage "gap" baseline was really yesterday's range and is now `recent_range`.
- **Operations.** The filings window follows the previous trading day (it missed filings after a long weekend); the app's logs no longer
  clash with logrotate (`copytruncate` emptied daily files); concurrent job processes no longer share a rotating file handler; the job tag
  of log lines is set; the daily bar flow never replaced provisional vendor bars with the official file once vendor coverage was complete;
  the bootstrap used the UTC date where the exchange date is meant.

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
- The time-zone guarantees are tested: the five-zone subprocess matrix (UTC, US Pacific, India, New Zealand, UTC+14) passes, and a broad
  subset of the suite passes with the process zone set to US Pacific (DST).

### Known limitations

- The Linux installer and systemd units were validated by static analysis, stubs and ShellCheck, not on a real host.
- The NSE pre-open JSON parser is verified against recorded fixtures, not a live session (first chance: the next trading
  morning at 09:00 IST). NSE blocks some data-centre address ranges; the system degrades and reports it.
- No soak test was run on a real 1 GB server; the memory budget comes from measured peaks on a development PC.
- Edge over simple baselines is modest and was measured on one market regime; a live shadow record of weeks to months is
  the real test. Nothing in this project places orders.
- News features are not model inputs (they reach the report); VWAP is a daily proxy; pre-open order imbalance is parsed and stored but
  not used; survivorship bias is reduced, not eliminated (names the database never ingested stay unknown). See the deviations and the
  limits table in [docs/BLUEPRINT_TRACEABILITY.md](docs/BLUEPRINT_TRACEABILITY.md).
- Lunar-calendar exchange holidays of a year whose official list is not yet published are confirmed live on the day (no report is sent,
  the operator is told), not predicted.
