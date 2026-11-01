# MONSTER AI - architecture

Large-move intelligence for NSE equities. Every trading morning it ranks the liquid names by the **calibrated
probability of an unusually large intraday range**, sends a Telegram report before the open, freezes the
predictions, scores them after the close and retrains under fixed gates.

**Shadow mode only.** The system never places, modifies or cancels an order and holds no broker credentials. Its
output is research. See [honesty rules](#honesty-rules).

## Contents

1. [Data flow](#data-flow)
2. [Daily timeline](#daily-timeline)
3. [Packages](#packages)
4. [Point-in-time correctness](#point-in-time-correctness)
5. [Models and governance](#models-and-governance)
6. [Ranking and honesty rules](#honesty-rules)
7. [Scheduling and self-healing](#scheduling-and-self-healing)
8. [Degradation matrix](#degradation-matrix)
9. [Storage](#storage)
10. [External training and off-box storage](#external-training-and-off-box-storage)
11. [Security](#security)
12. [Resource budget](#resource-budget)
13. [Design decisions](#design-decisions)

## Data flow

```
 Yahoo Finance ─┐                                        ┌─> technical / F&O / global / regime / calendar features
 NSE archives  ─┤   ingestion/      SQLite (WAL)        │            (features/, one code path for training AND live)
 NSE API       ─┼─> validate ─────> database/store.py ───┤
 RSS / filings ─┤   freshness gate  (single source of    └─> news_intelligence/ + event_graph/ (catalysts, chains,
 macro / GIFT  ─┘                    truth, immutable                     "already priced in")
                                      snapshots)
                                          │
        data-quality gate (automation/scan.py) ──> models/inference ──> ranking/ranker ──> freeze snapshot + predictions
                                                                                │
                              Telegram report (notifications/) <───────────────┘   (idempotent delivery ledger)
                                                                                │
   after the close: official bhavcopy ──> backtesting/evaluate.py ──> outcomes ──> rolling metrics ──> evening report
                                                                                │
   weekly: walk-forward training (models/training.py) ──> gates ──> champion/challenger registry (models/registry.py)
```

Everything between "ingestion" and "report" reads and writes one SQLite file. There is no message bus, no
container and no second process holding state: a crash can only lose the work of the current job, and every job is
safe to run again.

## Daily timeline

All times are Asia/Kolkata. The authoritative table is `SCHEDULE` in [automation/schedule.py](../automation/schedule.py);
the daemon derives "what is due" from the `job_runs` ledger, never from a clock tick.

| Time | Job | Purpose |
|------|-----|---------|
| 06:00 | `collect_global` | global closes, live context, GIFT Nifty (retried until 08:30) |
| 06:30 / 08:10 | `collect_news` | RSS, filings, board meetings, event extraction |
| 07:50 | `health_check` | operational alerts (also 20:00) |
| 08:00 | `collect_india` | previous session completeness: official file first, Yahoo as repair |
| 08:30 | `preflight` | readiness check, repair, dry-run of the scan |
| 08:45 | `prelim_scan` | preliminary report (**critical**) |
| 09:00:30 | `refresh_preopen` | NSE pre-open book (**critical**) |
| 09:08:45 | `final_scan` | final report with pre-open gaps, delivered by 09:12 (**critical**, hard deadline 09:14:30) |
| 17:45 | `ingest_eod` | official end-of-day files |
| 18:15 | `evaluate` | outcomes, metrics, evening report |
| 23:30 / 23:50 | `backup`, `housekeeping` | verified online backup (+ off-box copy), retention, disk guard |
| Sat 02:00 | `train` | walk-forward retraining (skipped in external mode) |
| Sun 04:00 | `refresh_universe` | index / F&O membership, history for new names |
| daily 05:30 | `refresh_calendar`, `model_sync` | holiday calendar; pull signed models in external mode |

Critical jobs preempt non-critical ones. Non-trading days run only the "any day" jobs.

## Packages

| Package | Responsibility |
|---------|----------------|
| [app/](../app) | configuration (single typed catalogue in `config.py`, generates `.env.example`), injectable clock, structured logging with secret redaction, file locks, resource guards (RSS watchdog, disk, RAM), CLI, `doctor`, error hierarchy |
| [ingestion/](../ingestion) | Yahoo, NSE archive and API clients with bounded retries, circuit breakers and rate limits; validation; universe building; freshness / data-quality gate; source catalogue (`registry.py` -> [DATA_SOURCES.md](DATA_SOURCES.md)) |
| [database/](../database) | SQLite connection (WAL, busy timeout), versioned migrations, the `Store` repository, verified online backups (optionally gpg-encrypted), S3-compatible object store (SigV4, stdlib only), off-box backup helpers |
| [features/](../features) | the single feature builder used by training and live scoring: technical, F&O, global, regime, calendar, sector, similar-day; label construction; schema hash and `FEATURE_VERSION` |
| [news_intelligence/](../news_intelligence), [event_graph/](../event_graph) | deduplication, entity linking, event extraction (rules first, optional LLM), source quality, "priced in" detection; supply-chain / sector impact propagation |
| [models/](../models) | LightGBM specialists + meta model, isotonic calibration, walk-forward training, gates, artifact format (text models + JSON, **never pickle**), registry (champion/challenger, rollback), HMAC-signed bundles for external training |
| [ranking/](../ranking) | eligibility, conviction tiers, contradiction checks, cost estimates, abstention, baselines |
| [notifications/](../notifications) | Telegram client, report renderers (4096-char safe), idempotent delivery with a ledger and deadline guard |
| [automation/](../automation) | the scan, all jobs, the schedule engine, the daemon, the job runner, bootstrap, health, calendar |
| [backtesting/](../backtesting) | outcome evaluation (provisional -> official), metrics, walk-forward backtest |
| [dashboard/](../dashboard) | optional read-only FastAPI dashboard (loopback by default) |
| [deployment/](../deployment) | installer, systemd units, remote deploy scripts - see [deployment/README.md](../deployment/README.md) |

## Point-in-time correctness

A scan at time *T* may use only what was knowable at *T*.

- Price, volume and F&O features are **shifted by one session** (a morning scan never sees today's bar).
- Global series are used at each series' own availability: a US close is known in the morning, a live quote is
  taken at the scan cutoff, and a stale series is flagged rather than forward-filled silently.
- Calendar and results-calendar features are known in advance and are not shifted.
- Stage `final` adds `p_*` gap features computed from the pre-open price only; names without a pre-open price fall
  back to the `pre` model (reported as such).
- Muhurat and other special sessions are removed from the training panel (market-wide volume far below the
  trailing median) and never count as a "missing" session.
- `tests/test_features_leakage.py` guards this: future data cannot change past features, the current session's path
  never reaches the `pre` features, the `final` stage sees only the session open, global series follow availability,
  and a live row equals the batch row for the same day (train/serve parity). Walk-forward folds are separated by an
  embargo.

Snapshots (features, predictions, data quality, model and code version) are **immutable**: one is written per
(date, stage) in the normal flow, a retry re-sends the frozen report instead of recomputing it, and an operator's forced
re-scan adds a new snapshot without ever editing an old one.

## Models and governance

- **Specialists** (`models/specialists.py`): technical, F&O, global, regime, events, direction. Each is a small
  LightGBM model per target (range thresholds 1/2/3/5 %, "unusual" = range above 1.5x the trailing mean, up/down).
- **Meta model** on out-of-fold specialist outputs plus context; **isotonic calibration** per head; quantile heads
  for the expected range (q10/q50/q90).
- **Walk-forward training** with an embargo, 8 folds, a held-out final block; ranking edge is measured against
  simple baselines (20-day range, ATR, recent range, volume, random) with a bootstrap.
- **Gates** (fixed in config, not tuned per run): enough held-out days, ECE <= 0.06, positive Brier skill vs climatology
  and a one-sided 90 % edge over the best baseline. Status is `validated`, `shadow` (gates not met, still useful for
  research), `rejected` (never activated) or `baseline` (no trained model, heuristic only).
- **Registry**: every version is stored with metrics, gate outcome, feature-schema hash and a manifest checksum recorded
  at registration. A challenger replaces the champion only if it matches or beats it on the same gate metrics;
  a corrupt or stale champion is replaced or rolled back automatically, falling back to the heuristic baseline
  if nothing verifies. Daily outcomes never edit a model.

## Honesty rules

- Probabilities come from calibrated models. The heuristic baseline is always labelled "(heuristic, NOT calibrated)".
- Conviction tiers (STRONG >= 55 %, MODERATE >= 40 %, WATCH >= 25 % calibrated P, each with an uncertainty ceiling) are capped
  at **WATCH unless the model is `validated`**.
- When nothing qualifies the report says **"no setup"** and why; it never pads the list.
- A report that would be delivered after the 09:14:30 deadline is withheld (the alert is queued and sent later).
- Ranking uses the calibrated range probability only. The report states that a wide range does not imply a profitable
  trade or a directional close; the direction head is a diagnostic and is near chance in out-of-sample tests.

## Scheduling and self-healing

- **Ledger-based scheduler** (`automation/scheduler.py`): one child process at a time (`python -m app.cli run-job`), so
  a leak or crash is contained; sd_notify watchdog; timeouts; preemption of non-critical jobs by critical ones.
- **Idempotent jobs**: locks per job and a shared heavy lock; run keys (`2026-11-02`, `2026-W44`); retries with
  back-off; catch-up for after-close jobs; windows that closed before the first start are skipped silently.
- **Bootstrap** is a resumable 12-step state machine (`bootstrap.state` in the `kv` table). A crash, reboot or
  out-of-memory kill resumes where it stopped; repeated crashes in an optional step skip that step instead of looping.
- **Watchdogs**: per-job RSS ceiling (`JOB_MAX_RSS_MB`), systemd `MemoryHigh/MemoryMax`, service watchdog, a health
  timer that alerts independently of the daemon.
- **Health** (`automation/health.py`): data freshness, missing sessions, model age and status, source failures,
  backup age, disk and RAM, delivery failures, performance drift. Alerts are de-duplicated and queued until Telegram
  works again.

## Degradation matrix

| Failure | Behaviour |
|---------|-----------|
| NSE API blocked / pre-open empty | final scan uses the `pre` model for names without a pre-open price and says so; if the book never arrives the report is still sent, marked |
| Yahoo down or rate limited | official bhavcopy first; global block is withheld only when >= 6 critical series are stale, otherwise degraded with warnings |
| Previous session incomplete | repair from the official file; if coverage stays low the data-quality gate **blocks** the scan, alerts and (on the last attempt of the final scan) tells you why |
| RSS feeds failing | news catalysts reduce to "coverage reduced" in the report; scoring continues |
| LLM unavailable / unconfigured | rule-based extraction only |
| No active model, corrupt artifact, wrong feature version | automatic rollback, then heuristic baseline with the honest label |
| Telegram down | reports and alerts are queued in the ledger and flushed when it returns; late final reports are withheld |
| Process killed / reboot | systemd restarts the daemon; the ledger resumes; SQLite WAL recovery |
| Disk low | emergency cleanup of caches, logs and temp files; heavy jobs blocked; critical alert |
| Time glitch (NTP step) | windows are evaluated from the ledger and the IST clock; nothing runs twice |

## Storage

One SQLite database (`monster.db`, WAL) holds: `instruments`, `bars_daily`, `fno_daily`, `preopen_quotes`,
`live_quotes`, `macro_observations`, `calendar_days`, `news_raw`, `filings`, `board_meetings`, `events`,
`graph_nodes`/`graph_edges`, `snapshots`, `predictions`, `outcomes`, `eval_metrics`, `model_registry`, `job_runs`,
`deliveries`, `alerts`, `source_health`, `kv`. Migrations are versioned and run on open. Disposable caches live
under `data/runtime/cache` and are size-limited. Backups are written with SQLite's online backup API, compressed,
checksummed, verified by a test restore and retained 7 daily / 4 weekly.

## External training and off-box storage

Optional. A GitHub Actions worker ([.github/workflows/train.yml](../.github/workflows/train.yml)) restores the
server's newest off-box backup as its training data (or collects history itself), trains with a larger budget, and
publishes **HMAC-signed bundles** (`models/<stage>/<version>.zip` + a signed `latest.json` pointer). The server
(`TRAINING_MODE=external`) only pulls: it verifies the pointer, the archive hash and the manifest signature
*before* extracting anything, re-verifies every file, rejects a bundle that is not `shadow`/`validated`, built for another
feature version, or whose metadata disagrees with its manifest, and then registers it as a normal challenger.
Off-box backups use the same S3-compatible store and are pruned to the local retention.

## Security

- No orders, no broker keys. Secrets only in `/etc/monster-ai/monster.env` (`root:monster 0640`); logs redact them.
- Dedicated unprivileged `monster` user; every unit runs with `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`,
  `PrivateTmp`, an empty capability set and IP/Unix sockets only; the dashboard binds to loopback (a token is
  required otherwise).
- Model artifacts are LightGBM text + JSON; bundles are signed and extracted with zip-slip / zip-bomb protection.
- All inbound data is validated (schema, ranges, dates, size limits); SQL is parameterised; HTTP clients have
  timeouts, size limits and bounded retries.
- Outbound connections: the data providers, Telegram, and - only if configured - the object store and an LLM endpoint.

## Resource budget

Target: 1 GB RAM, 1 vCPU, 8 GB disk. The daemon is a light process (it never imports pandas, numpy or requests);
jobs run as short-lived children inside the service's cgroup (`MemoryHigh=750M`, `MemoryMax=880M`, job RSS ceiling
720 MB). A morning scan took about 11 s on a development PC. Training is the only heavy job: it runs weekly in a
lighter profile on small servers (smaller fit sample, fewer history years) or externally. The installer adds a
small swap file as a safety net; swap is not a substitute for RAM.

## Design decisions

| Decision | Why |
|----------|-----|
| Modular monolith, one SQLite file | one process owner of state, trivial backup/restore, enough for ~250 symbols |
| Native systemd, **no containers** | 1 GB budget; fewer moving parts; journald + systemd already give restart, limits and logs |
| Ledger instead of cron | a restart or clock step cannot skip or duplicate a job; missed work is visible |
| One feature path for training and live | removes train/serve skew, the usual source of silent leakage |
| Immutable snapshots | the evaluation can always be traced to what was known and sent |
| Calibrated probabilities + honesty gates | a ranking is only useful if "40 %" means 40 %; unvalidated models are visibly capped |
| Shadow-first | the edge must be demonstrated out of sample before anyone risks money |
