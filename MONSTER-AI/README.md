# MONSTER AI - NSE large-move intelligence engine

Every Indian trading morning at about 09:12 IST this system sends one Telegram report: the **top 5 liquid NSE stocks ranked by the
calibrated probability of an unusually large intraday range** (default: range > 3% of the open), with the evidence behind each name
(catalysts, regime, F&O positioning, liquidity and cost), explicit uncertainty, and an honest "no high-conviction setup" when nothing
qualifies. After the close it evaluates what it predicted against simple baselines and retrains weekly under fixed validation gates.

> **Status: shadow mode.** The system never places orders. A model is only labelled `VALIDATED` when its out-of-sample
> calibration, skill versus the base rate and ranking edge over simple baselines pass fixed gates; until then every list is
> labelled SHADOW/BASELINE and tiers are capped at WATCH. A wide intraday range is not a trade signal, and nothing here is
> investment advice. See [Limitations](#limitations).

## What you have to do

Only this (everything else is automated, including first-time setup, history backfill, training, scheduling and restarts):

1. Create a Telegram bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Run the one-command deployment (see [Server deployment](#server-deployment)). Optional keys (S3 backup, LLM extraction, FRED)
   can be added later to `/etc/monster-ai/monster.env`.

## Quick start (development machine)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt     # Windows: .venv\Scripts\pip ...
.venv/bin/python -m app.cli init            # database, calendar, source catalogue
.venv/bin/python -m app.cli doctor          # verifies Python, packages, paths, network, Telegram
.venv/bin/python -m pytest                  # hermetic unit/integration tests (no network)
.venv/bin/python -m app.cli bootstrap --skip-train   # real data: universe, 10y history, F&O (resumable)
.venv/bin/python -m app.cli scan --stage pre --dry-run   # prints the report; freezes and sends nothing
```

Python 3.10-3.13 on Linux x86-64/aarch64 (binary wheels only) and Windows for development.

## Server deployment

Target: Ubuntu 22.04/24.04 or Debian 12, 1 vCPU / **1 GB RAM** / 8 GB disk, **native systemd (no containers)**.

```bash
sudo bash deployment/install.sh --env-file ./monster.env.seed      # or from Windows: deployment\deploy_remote.ps1
```

The installer is idempotent: it installs OS packages, creates the unprivileged `monster` user, builds the virtualenv from wheels,
validates the configuration, discovers your Telegram chat id, initialises the database, installs and starts the systemd units,
runs the first-time bootstrap in the background and verifies the result; re-running it upgrades and rolls back automatically if
the new release does not come up. Details: [deployment/README.md](deployment/README.md) and
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## Daily timetable (Asia/Kolkata, trading days)

| Time | Job | Purpose |
|---|---|---|
| 06:00-08:30 | `collect_global` | US/Europe/Asia closes, FX, commodities, rates, VIX, GIFT Nifty (retries) |
| 06:30, 08:10 | `collect_news` | RSS + GDELT + NSE filings, entity/event extraction |
| 08:00 | `collect_india` | previous-session completeness (official bhavcopy first), results calendar |
| 08:30 | `preflight` | repair gaps, dry-run the scan so problems surface before 08:45 |
| 08:45 | `prelim_scan` | preliminary scan, frozen snapshot (Telegram only if `SEND_PRELIM_REPORT=true`) |
| 09:00:30 | `refresh_preopen` | NSE pre-open book (indicative/final equilibrium prices) |
| 09:08:45 | `final_scan` | final scan with the opening gap, frozen snapshot, **Telegram report by 09:12-09:14:30** |
| 17:45+ | `ingest_eod` | official end-of-day data (retries until 23:30) |
| 18:15+ | `evaluate` | outcomes vs baselines, evening report |
| 23:30 / 23:50 | `backup`, `housekeeping` | online DB backup (+ optional S3), retention |
| Sat 02:00 | `train` | walk-forward retraining, champion/challenger governance |

Weekly: `refresh_calendar` (Mon), `refresh_universe` (Sun). A ledger (`job_runs`) makes every job restart-safe: missed windows are
recorded and alerted, catch-up jobs run after a reboot, and nothing runs twice.

## Commands

`python -m app.cli <command>`: `init`, `doctor [--offline] [--json]`, `bootstrap [--resume] [--skip-train]`, `daemon`,
`jobs`, `scan --stage pre|final [--dry-run] [--date D]`, `evaluate [--date D]`, `train [--stage ..]`, `backtest`,
`models list|approve|activate|rollback`, `backup [--verify|--restore-latest]`, `health [--json] [--notify]`,
`telegram-setup`, `telegram-test`, `dashboard`, `env-example`, `version`.

## Configuration

All settings are environment variables (12-factor), validated at start-up. `.env.example` is generated from the single source
of truth in `app/config.py` (`python scripts/generate_docs.py`). Only `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` are required.
Key knobs: `MONSTER_MODE` (shadow), `UNIVERSE_MODE` (fno_nifty100), `PRIMARY_THRESHOLD` (3.0), `TOP_N` (5), tier probabilities,
scan times (`FINAL_SCAN_TIME`, `REPORT_CUTOFF_TIME`, `REPORT_DEADLINE_TIME`), data-quality and resource limits, training mode
(`local-light` or `external` signed models), S3 backup, optional LLM.

## How it works

- **Data**: Yahoo daily bars (unofficial) and NSE official files (bhavcopy, UDiFF F&O, index lists, pre-open API), news from
  RSS/GDELT/NSE filings. Every source is registered with timestamp meaning, rate limits, licence notes and failure behaviour
  ([docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)).
- **Point-in-time by construction**: one feature builder serves training and live scoring; price/volume/F&O features are shifted
  by one session, global series use each market's own close-availability time at the scan cutoff, and tests perturb the future
  and the current session to prove nothing leaks.
- **Models**: specialist LightGBM models (technical, global, regime, F&O, pre-market, events) feed a meta-model; probabilities are
  isotonic/Platt calibrated on earlier data than the holdout. Two stages: `pre` (08:45) and `final` (with the opening gap).
- **Governance**: walk-forward training with purge/embargo, fixed gates, champion/challenger, integrity-checked artifacts with
  rollback, heuristic baseline fallback, daily live evaluation with intervals and sample sizes.
- **Operations**: single-process scheduler daemon (systemd `Type=notify`, watchdog), one job at a time in memory-isolated child
  processes, per-job timeouts and RSS watchdog, health checks and Telegram alerts.

Architecture notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Operations runbook: [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Quality gates and CI/CD

```bash
python -m pytest --cov=. --cov-fail-under=80      # >= 80% coverage, hermetic (injected clock, fake HTTP, temp directories)
python -m ruff check .                            # lint (incl. security rules)
bash -n / shellcheck deployment/**/*.sh           # deployment scripts
```

`.github/workflows/ci.yml` runs lint, tests with the coverage gate and script checks on every push/PR;
`train.yml` can train and publish a signed model bundle (`TRAINING_MODE=external`) for servers too small to train locally.

## Project layout

```
app/            configuration, clock, logging (redaction), locks, resources, CLI, doctor
database/       SQLite (WAL) schema + migrations, repository (Store), backup/restore, object store
ingestion/      HTTP client (retries, breaker), Yahoo, NSE, F&O, universe, news, freshness/quality gate
features/       point-in-time feature engines + builder + loader
models/         training, calibration, inference, registry, artifacts, signing
ranking/        eligibility, costs, tiers, abstention, evidence
automation/     calendar, scan, jobs, scheduler, schedule engine, health, bootstrap
notifications/  Telegram client, report renderers, idempotent delivery
news_intelligence/, event_graph/   event extraction, entity resolution, causal-graph propagation
backtesting/    metrics, walk-forward report, live evaluation
dashboard/      optional read-only web UI (loopback)
deployment/     installer, systemd units, remote deploy
data/seeds/     universe, holidays, feeds, aliases, graph seed
tests/          hermetic tests built on real captured provider responses
```

## Limitations

- Free endpoints (Yahoo, NSE JSON, public RSS/GDELT) are unofficial: they can change, rate-limit or block datacentre IPs. The system
  degrades visibly (data-quality banner, preliminary model, abstention) instead of guessing. Confirm provider terms before any
  commercial use and prefer a licensed feed for production.
- The universe is today's F&O + Nifty 100 membership (survivorship bias in history is documented); vendor-adjusted prices precede
  the official-file period.
- Calibrated probabilities describe *range*, not direction or profit. Costs are modelled estimates.
- The live evidence base starts empty: statistics are shown with sample sizes and intervals and the "VALIDATED" label has to be earned.

## Release notes

See [CHANGELOG.md](CHANGELOG.md).
