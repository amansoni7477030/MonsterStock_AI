# Changelog

All notable changes to MONSTER AI. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/). Database migrations are forward-only and additive, so every
release below can be installed over the previous one (`deployment/install.sh` is idempotent and rolls back by itself).

## [1.0.0] - 2026-11-01

First complete release: a shadow-mode NSE large-move intelligence engine that runs unattended on a 1 GB / 8 GB Ubuntu
server with native systemd (no containers).

### Added

- **Daily pipeline** (`automation/`): ledger-based scheduler daemon (sd_notify watchdog, one child process per job,
  preemption of non-critical jobs, catch-up, retries) and the full timetable - global and India data collection, news,
  preflight dry run, 08:45 preliminary scan, 09:00 pre-open book, 09:08:45 final scan with Telegram delivery by 09:12,
  end-of-day ingestion, evaluation, backup, housekeeping, weekly training / universe / calendar refresh, model sync.
- **Resumable 12-step bootstrap** that prepares everything on a new server (universe, 10 years of history, official
  bhavcopy, F&O history, board meetings, news, first models, smoke scan, Telegram test) and survives crashes, reboots
  and out-of-memory kills.
- **Data layer** (`ingestion/`, `database/`): Yahoo, NSE archive and NSE API clients with bounded retries, circuit
  breakers and rate limits; validation; freshness / data-quality gate with an explicit blocked / degraded / ok verdict;
  source catalogue ([docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)); SQLite (WAL) with versioned migrations; verified,
  optionally gpg-encrypted online backups; Muhurat and short-session handling; a pre-open book stamped on an earlier
  session is ignored instead of being mistaken for today's.
- **One feature path for training and live scoring** (`features/`): technical, F&O, global, regime, calendar, sector and
  similar-day features, strictly point-in-time (price/volume shifted one session, global series by availability),
  guarded by leakage tests, with a float32 column-streamed builder sized for 1 GB.
- **Models** (`models/`): LightGBM specialists + meta model, isotonic calibration, quantile range heads, walk-forward
  training with embargo, fixed honesty gates (calibration, Brier skill, edge over simple baselines with a bootstrap),
  artifact format without pickle, champion/challenger registry with integrity checks, rollback and automatic fallback to
  a labelled heuristic baseline.
- **Ranking** (`ranking/`): eligibility, conviction tiers capped at WATCH unless the model is validated, contradiction and
  cost checks, explicit "no setup" when nothing qualifies, baseline pickers for comparison.
- **News and event graph** (`news_intelligence/`, `event_graph/`): deduplication, entity linking, rule-based event
  extraction with an optional LLM (every claim needs a verifiable quote), source quality, "already priced in" detection
  (market reactions are refreshed after each news run and again at the final scan), supply-chain and sector impact
  propagation with a seeded expert-prior graph (seeded automatically, re-seeded when the universe changes, and re-measured
  weekly from returns with FDR control; edges without a price series stay labelled priors).
- **Notifications** (`notifications/`): Telegram client, 4096-character-safe renderers (pre-open, final, evening, alerts,
  failure notices), idempotent delivery ledger, deadline guard, queued alerts flushed when Telegram returns.
- **Health and self-healing** (`automation/health.py`): data freshness, missing sessions, model age and status, source
  failures, backup age, disk and RAM, delivery failures and live calibration drift; de-duplicated alerts; RSS watchdog per job;
  a training attempt that dies from memory pressure is retried with a progressively leaner profile.
- **Deployment** (`deployment/`): one-command install from Windows (`deploy_remote.ps1`) or Linux/macOS
  (`deploy_remote.sh`); idempotent installer with swap, NTP, dedicated user, atomic release swap, automatic rollback,
  Telegram chat-id discovery and verification; six hardened systemd units; journald and logrotate limits; uninstaller.
- **External training (optional)** (`models/external.py`, `models/bundle.py`, `models/signing.py`,
  `database/object_store.py`): a GitHub Actions worker trains and publishes HMAC-signed bundles to S3-compatible storage;
  the server pulls, verifies (pointer, archive hash, manifest signature, every file) and registers them as ordinary
  challengers. Off-box database backups use the same store with retention; `backup --restore-latest --remote` recovers a
  lost server.
- **Dashboard (optional)** (`dashboard/`): read-only FastAPI view, loopback by default.
- **CLI** (`app/cli.py`): `doctor`, `bootstrap`, `daemon`, `run-job`, `jobs`, `scan`, `evaluate`, `train`, `backtest`,
  `models`, `backup`, `health`, `telegram-setup`, `telegram-test`, `dashboard`.
- **Documentation**: [README](README.md), [architecture](docs/ARCHITECTURE.md), [operations runbook](docs/OPERATIONS.md),
  [deployment guide](deployment/README.md), generated [data-source catalogue](docs/DATA_SOURCES.md) and `.env.example`
  (both checked against the code by tests).
- **Tests**: hermetic suite (no network; temp directories; injected clock) with coverage gates in CI, shellcheck on every
  script and a wheel-availability check for the pinned dependencies.

### Verified on real data

- Bootstrap against the live providers: 221-symbol universe, 254 instruments with 10 years of history, 12 official
  bhavcopy sessions, 573 F&O days, 3,249 board-meeting rows, 44 of 44 RSS feeds.
- A real dry-run scan completed in about 11 seconds on a development PC.
- A model trained on that data (221 stocks, five years, eight walk-forward folds) passed every gate: expected calibration
  error 0.014, Brier skill 0.17 against climatology, top-5 hit rate for a range of at least 3 % of 79 % against 73 % for the
  best simple baseline (20-day range) and 31 % for random picks. The advantage over the best baseline is real but modest
  (+6.7 points, 90 % interval +3 to +10 points); direction is not predictable (AUC 0.55) and the system says so.
- Training memory on the real data was reduced to fit the 1 GB server's budget (smaller histogram sample, chunked
  statistics, float32 out-of-fold arrays, column-streamed feature matrix, per-fold memory release): measured peaks are
  about 640 MB resident for the feature build and 565 MB for the model training in the light profile. A training
  attempt that still dies from memory pressure is retried automatically with a leaner profile.

### Known limitations

- The Linux installer and systemd units were validated by static analysis, stubs and ShellCheck, not on a real host.
- The NSE pre-open JSON parser is verified against recorded fixtures, not a live session (first chance: the next trading
  morning at 09:00 IST). NSE blocks some data-centre address ranges; the system degrades and reports it.
- Edge over simple baselines is modest and was measured on one market regime; a live shadow record of weeks to months is
  the real test. Nothing in this project places orders.
