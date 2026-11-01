# MONSTER AI - operations runbook

Everything here is for the **running** system. Installation, upgrade, rollback and uninstall are in
[deployment/README.md](../deployment/README.md); the design is in [ARCHITECTURE.md](ARCHITECTURE.md).

The system is unattended: after the one-time install nothing has to be done daily, weekly or monthly. This page is
for the moments you *want* to look, or when an alert asks you to.

## Contents

1. [What you receive](#what-you-receive)
2. [First day](#first-day)
3. [Everyday commands](#everyday-commands)
4. [Configuration](#configuration)
5. [Alerts and what to do](#alerts-and-what-to-do)
6. [Playbooks](#playbooks)
7. [Backups and recovery](#backups-and-recovery)
8. [Models](#models)
9. [External training](#external-training)
10. [Reading the reports honestly](#reading-the-reports-honestly)

## What you receive

All messages arrive in the Telegram chat that was linked at install time (alerts may use a separate chat:
`TELEGRAM_ALERT_CHAT_ID`).

| When (IST, trading days) | Message |
|--------------------------|---------|
| ~08:45 | preliminary report (only if `SEND_PRELIM_REPORT=true`; off by default) |
| **~09:09-09:12** | **final report**: the top-5 names, calibrated probabilities, expected range, catalysts, regime, data freshness |
| after 18:15 | evening summary: what happened to this morning's picks and the rolling record (`SEND_EVENING_REPORT`) |
| any time | alerts (de-duplicated; queued while Telegram is unreachable) |

A morning with nothing worth reporting produces a short **"no setup"** message that says why. A final report that
could not be delivered before **09:14:30** is withheld rather than sent late; you get an alert instead.

## First day

1. The installer ends with the bootstrap (10-60 minutes, in the background). You get *"installation complete"* on
   Telegram when it is done. Until then `health` may report CRITICAL: that is expected.
2. Check once: `sudo monster-ai health` (exit 0 = ok), `sudo monster-ai jobs` (timetable and recent runs),
   `sudo monster-ai models list` (the first models appear at the end of the bootstrap).
3. Optional dry run of the morning scan (prints the report, sends and freezes nothing):
   `sudo monster-ai scan --stage final --dry-run`.
4. The default mode is **shadow**. Nothing in this system places orders.

The very first model is usually `shadow` or `validated`; until one exists the report says *"heuristic baseline, NOT
calibrated"* and caps every name at WATCH.

## Everyday commands

`sudo monster-ai <command>` runs `python -m app.cli <command>` as the service user with the production environment.

| Command | Use |
|---------|-----|
| `health [--json]` | exit 0 ok / 1 warnings / 2 critical; every problem with its explanation |
| `jobs` | the timetable and the last runs of every job |
| `doctor` | full preflight: configuration, permissions, disk, RAM, clock, reachability of every data feed |
| `scan --stage pre\|final --dry-run [--date YYYY-MM-DD]` | print a report without freezing or sending |
| `scan --stage final --force` | recompute and re-send today's report (adds a new snapshot; the earlier one is kept) |
| `evaluate [--date D]` | evaluate a day's predictions now |
| `models list` / `rollback --stage S` / `approve V --status shadow\|validated` / `activate V` | model governance |
| `train [--stage pre\|final\|both]` | retrain now (heavy: 30-90 minutes on a 1 GB server) |
| `backtest --stage final --years 6` | walk-forward evaluation against the baselines (writes a report) |
| `backup`, `backup --verify`, `backup --restore-latest [--remote]` | backups, see [below](#backups-and-recovery) |
| `telegram-test`, `telegram-setup` | test message; discover the chat id |
| `bootstrap --resume` | continue a first-run setup that stopped |
| `dashboard` | optional read-only web view (loopback; use an SSH tunnel) |

Logs: `journalctl -u monster-ai -f` (daemon), `journalctl -u monster-ai-bootstrap`, `/var/log/monster-ai/` (job and CLI
logs, rotated). Secrets are redacted in all of them.

## Configuration

Everything is an environment variable in `/etc/monster-ai/monster.env` (`root:monster`, mode 0640). The full, documented
list is generated from the code into [.env.example](../.env.example). After a change:

```bash
sudo nano /etc/monster-ai/monster.env
sudo monster-ai doctor --offline        # validates the file and explains any mistake
sudo systemctl restart monster-ai
```

Safe to change at any time: `TOP_N`, tier thresholds (`TIER_*_P`), `SEND_*_REPORT`, `LOG_LEVEL`, resource limits, backup
retention. Changing `RANGE_THRESHOLDS`, `PRIMARY_THRESHOLD` or `UNUSUAL_K` changes what the models predict: retrain
(`monster-ai train`) afterwards, otherwise the registry refuses models with a different schema.

Optional capabilities, all off by default:

| Capability | Variables |
|------------|-----------|
| LLM-assisted event extraction | `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` (rules-only without them; every LLM claim needs a verifiable quote) |
| Off-box storage (backups, external models) | `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, optional `S3_ENDPOINT_URL`, `S3_REGION`, `S3_PREFIX` |
| Encrypted backups | `BACKUP_PASSPHRASE` (gpg AES256; keep a copy somewhere else, a backup without it cannot be read) |
| External training | `TRAINING_MODE=external`, `MODEL_SIGNING_KEY`, `MODEL_ARTIFACT_URL` or `S3_*` |
| Dashboard | `DASHBOARD_ENABLED=true` (+ `DASHBOARD_TOKEN` if bound beyond loopback) |
| Macro series | `FRED_API_KEY` |

## Alerts and what to do

Alerts are Telegram messages with a code. The same problem is not repeated within its de-duplication window, and a
`health_recovered` message follows when everything is fine again.

| Code | Meaning | What to do |
|------|---------|------------|
| `scan_final_failed` / `scan_pre_failed` | the scan stopped (usually the **data-quality gate**: previous session incomplete, global block stale, no universe) | read the message; `monster-ai scan --stage final --dry-run`; see [no report](#no-report-this-morning) |
| `preflight_failed` | the 08:30 dry run failed, so the 09:00 scan would have too | same as above; it is a warning *before* the report time |
| `job_failed_<job>` / `job_crashed_<job>` | a job exhausted its retries / its process died | `journalctl -u monster-ai --since today`; `/var/log/monster-ai/jobs.log`; the daemon tries the job again at its next window |
| `job_missed_<job>` | a window closed without a successful run (service was down, long outage) | check `health`; after-close jobs catch up on their own |
| `health_db_corrupt` / `health_db_unreadable` | SQLite integrity problem | [restore](#backups-and-recovery) immediately |
| `health_scheduler_stalled` / `health_scheduler_silent` | no heartbeat from the daemon | `systemctl status monster-ai`; systemd restarts it by itself (watchdog 180 s) |
| `health_final_scan_missing` | no final scan recorded by 09:20 on a trading day | `monster-ai jobs`, logs; run `monster-ai scan --stage final --force` (report only useful until the open) |
| `health_stale_bars` | less than 90 % of the universe has the last session's bar | usually a provider outage; `collect_india` / `ingest_eod` retry; check `doctor` |
| `health_no_model_<stage>` / `health_model_stale_<stage>` / `health_training_overdue` | no active model / model older than 120 days / no successful training for 14 days | `models list`; `train`; in external mode check the worker and `model_sync_failed` |
| `model_artifact_invalid` | the active model failed its integrity check; the system rolled back (or fell back to the baseline) | `models list`; retrain or `models activate` a good version; check disk health |
| `model_inputs_missing` / `model_predict_failed` | many model inputs were unavailable at scan time (scored with missing values, the report notes it) / the model crashed while scoring (the heuristic baseline is used for that report) | read the message; usually follows a data outage |
| `training_failed` | a stage could not be trained (not enough history, degenerate labels) | `journalctl`; the previous champion stays active |
| `model_sync_failed` | external mode: the signed bundle was missing, unsigned, tampered, for another feature version or failed its gates | check the worker run and that `MODEL_SIGNING_KEY` is identical on both sides |
| `backup_upload_failed` | local backup is fine, the off-box copy failed | check `S3_*` and connectivity; the next night retries |
| `health_backup_old` | last backup older than 36 h (warning) / 72 h (critical) | `monster-ai backup`; check disk space |
| `resources` | low disk or RAM | [disk / memory](#disk-or-memory-pressure) |
| `health_source_<id>` | a data source failed 5 times in a row | usually external; the system degrades and says so; `doctor` shows reachability |
| `health_telegram_not_configured` / `health_report_not_delivered` | no token/chat id, or the last final report could not be sent | `telegram-setup`; check the token |
| `health_calibration_drift` / `health_skill_negative` | live calibration worse than 0.10 ECE / live Brier skill below the base rate over the last 90 days | see [models](#models); consider `models rollback` |
| `health_calendar_stale` / `health_calendar_never_refreshed` | NSE holiday list could not be refreshed | the shipped holiday seed and inference from missing bars keep working; check NSE reachability |
| `symbol_quarantined` | a symbol no longer exists at the provider | informational; it leaves the universe |
| `bootstrap_complete` | first-run setup finished | none |

## Playbooks

### No report this morning

1. `sudo monster-ai jobs` - did `final_scan` run? Which status?
2. If it says **blocked by the data-quality gate**, the message lists what is missing. Typical causes:
   the previous session's bars are incomplete (provider outage), or the global-feed block is stale (>= 6 critical series).
   Wait for the retry window (until 09:13:30) or fix the cause; `monster-ai doctor` shows which feeds are unreachable.
3. If the Telegram delivery failed, the report is frozen; it is re-sent on the next attempt. A report that would be
   later than 09:14:30 is withheld on purpose (it would no longer be a pre-open report).
4. After 09:15 a manual `scan --stage final --force` is still useful for research, not for the open.

### Reports say "heuristic baseline"

No model is active: the first training has not run (bootstrap still running?), the artifact failed its integrity check,
or all candidates were rejected by the gates. `monster-ai models list`, then `train` (or wait for Saturday 02:00 IST).
This is the honest fallback, not a failure.

### NSE blocks the server (HTTP 403)

NSE blocks some data-centre ranges. The system runs degraded and says so: the official archive and Yahoo still feed the
price history; the pre-open book and option chains are missing, so the final report uses the preliminary model for
names without a pre-open price. If it is permanent: use a server in an Indian region, or set `NSE_ENABLED=false` to stop
the attempts and accept the reduced data.

### Telegram problems

`monster-ai telegram-test`. Wrong chat id: `monster-ai telegram-setup` (send any message to the bot while it waits). While
Telegram is down, reports and alerts queue in the database and are delivered when it returns (late final reports are withheld).

### Disk or memory pressure

`monster-ai health` shows free disk and RAM. Housekeeping runs nightly (cache, logs, temp, news retention, old model
versions, `VACUUM` on Sundays). Below `MIN_FREE_DISK_MB_CRIT` an emergency cleanup runs and heavy jobs are blocked. A job
that exceeds its RSS ceiling (`JOB_MAX_RSS_MB`, 720 MB) is aborted and retried; repeated aborts in one bootstrap step skip that
step instead of looping. If memory pressure is chronic, use `TRAINING_MODE=external` or a larger server.

### The daemon is not running

```bash
systemctl status monster-ai
journalctl -u monster-ai -n 100
sudo systemctl restart monster-ai
```

systemd restarts it automatically (`Restart=always`); the ledger resumes where it stopped and nothing runs twice.

### Wrong or missing holiday handling

`refresh_calendar` runs daily (official NSE list, then inference from days with no bars). If a special session (Muhurat)
or an unscheduled closure confuses the calendar: `monster-ai run-job refresh_calendar --force`.

## Backups and recovery

- **Nightly** (23:30 IST): an online SQLite backup, compressed, checksummed and **verified by a test restore**, kept 7 daily +
  4 weekly under `/var/lib/monster-ai/backups`. With `BACKUP_PASSPHRASE` it is gpg-encrypted.
- **Off-box** (when `S3_*` is set): the same file is uploaded to `backups/` in the bucket and pruned to the same retention.
- `monster-ai backup` runs one now; `monster-ai backup --verify` re-checks the newest.

**Restore on the same server** (corruption, bad upgrade):

```bash
sudo systemctl stop monster-ai
sudo monster-ai backup --restore-latest              # newest backup that verifies; the old file is kept as *.corrupt-<time>
sudo systemctl start monster-ai
```

**Restore on a new server** (the old one is gone): deploy as usual with the same env file (`S3_*` and `BACKUP_PASSPHRASE`
included) but **`--no-bootstrap`**, then:

```bash
sudo systemctl stop monster-ai
sudo monster-ai backup --restore-latest --remote     # downloads the newest verified off-box backup, then restores it
sudo systemctl start monster-ai
sudo monster-ai train                                # local training mode: model files are not part of the backup
```

The database holds all history, predictions, outcomes and configuration state. Model files are small and reproducible, so they are
not in the backup: in local mode retrain once (the system uses the labelled heuristic baseline until then); in external mode
`model_sync` fetches them again by itself at 05:30.

## Models

Statuses: `validated` (passed every gate), `shadow` (trained, gates not met: usable for research, capped at WATCH),
`rejected` (never activated), `retired` (replaced or rolled back), `baseline` (no model).

```bash
sudo monster-ai models list                       # version, stage, status, ACTIVE, created
sudo monster-ai models rollback --stage final     # re-activate the previous healthy version
sudo monster-ai models approve <version> --status shadow     # manual decision, overrides the gates: use deliberately
sudo monster-ai models activate <version>
```

A new model replaces the champion only if it matches or beats it on the same fixed gate metrics, so weekly retraining cannot
silently degrade the system. Daily outcomes never edit a model. The metrics of every version (ECE, Brier skill, edge over the
best simple baseline with its confidence interval) are in `models list` output and in each model's `meta.json`.

## External training

For full-size training off the small server, see "External training" in [deployment/README.md](../deployment/README.md). In short:
the GitHub worker restores the server's off-box backup, trains, and publishes **signed** bundles; the server pulls and verifies
them (`model_sync`, daily 05:30 IST) and registers them as ordinary challengers. Needed: the repository secrets
`MODEL_SIGNING_KEY`, `S3_*` (and `BACKUP_PASSPHRASE` if backups are encrypted), and on the server `TRAINING_MODE=external`
with the same key and bucket. Rotating the key: change both sides, then run the worker once.

## Reading the reports honestly

- A probability is a **calibrated frequency**: of all names rated 40 %, about 40 % should reach the threshold range. The evening
  summary and `health_calibration_drift` check exactly that on live data.
- Rank, not certainty: the useful claim is that the top picks reach large ranges *more often than simple rules* (recent range,
  ATR, volume), measured out of sample with a confidence interval. On the project's real-data training run the model beat the
  best simple baseline by a few percentage points (hit rate on the top-5 for range >= 3 %), not by a wide margin: volatility
  persistence explains most of it. Do not expect miracles.
- A wide range is **not** a direction and **not** a trade. Costs, slippage and gaps are not modelled as profit. There is no
  order logic in this system.
- `shadow` and `baseline` outputs are research aids. Only `validated` models may show STRONG/MODERATE tiers, and only a live
  shadow record of weeks to months should be allowed to inform any decision.
