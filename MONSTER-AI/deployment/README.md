# MONSTER AI - deployment

Native systemd deployment for a small Ubuntu 22.04 / 24.04 or Debian 12 server (x86_64 or aarch64, 1 GB RAM,
8 GB disk). **No containers of any kind.** One command from your PC installs, configures, starts and
verifies everything; afterwards the system heals itself (systemd restarts, watchdog, health timer, Telegram
alerts, nightly backups).

- [Quickstart](#quickstart)
- [What the installer does](#what-the-installer-does)
- [Upgrade, rollback, uninstall](#upgrade-rollback-uninstall)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)
- [Resource budget](#resource-budget)
- [External training](#external-training-trainingmodeexternal)
- [Reference](#reference)

## Quickstart

You need a server you can SSH into with a user that has `sudo` (or `root`), and - optionally - a Telegram bot
token from [@BotFather](https://t.me/BotFather). Put the token into a local `.env` file first (copy
`.env.example`, fill in `TELEGRAM_BOT_TOKEN=`), or add it to the server later.

**From Windows 10/11** (uses the built-in OpenSSH client and `tar`; run it in Windows Terminal / a PowerShell
window, not in the ISE):

```powershell
.\deployment\deploy_remote.ps1 -Target ubuntu@203.0.113.10
# optional: -Port 2222  -IdentityFile $HOME\.ssh\id_ed25519  -EnvFile C:\secrets\monster.env
#           -NoBootstrap  -EnableDashboard  -SkipSwap  -KeepRemoteTemp
```

**From Linux or macOS:**

```bash
deployment/deploy_remote.sh ubuntu@203.0.113.10
# optional: --port 2222 --identity ~/.ssh/id_ed25519 --env-file ~/secrets/monster.env
#           --no-bootstrap --enable-dashboard --skip-swap --keep-remote-temp
```

**Directly on the server** (after copying the project there):

```bash
sudo bash deployment/install.sh --yes
```

What happens: the project is packed (without `.git`, `.venv`, tests, runtime data and env files), uploaded with
your env file to a private temporary directory (`mktemp -d`, mode 700), installed with `sudo bash
deployment/install.sh --yes`, and the temporary directory is removed again - the env copy is **shredded** (even
when the installation fails). The exit code is that of the installer.

During the installation (about 5 minutes without the bootstrap):

1. If a bot token is present but no chat id, the installer asks you to **send any message to your bot** and waits
   up to 180 s; it then stores the chat id by itself.
2. The first-time **bootstrap** (universe, history backfill, F&O data, first models, smoke scan) runs in the
   background as `monster-ai-bootstrap.service`; progress is printed every 15 s. It takes 10-60 minutes and
   survives SSH disconnects. You get an "installation complete" message on Telegram when it is done.

Afterwards:

```bash
journalctl -u monster-ai -f                 # follow the daemon
sudo monster-ai health                      # exit 0 ok / 1 warning / 2 critical
sudo monster-ai doctor                      # full preflight incl. feed reachability
```

The default mode is **SHADOW**: research output only, the system never places orders.

## What the installer does

`deployment/install.sh` is idempotent: run it again at any time to upgrade or repair. Every step is a function
and prints `[n/14] ...`.

| # | Step | Notes |
|---|------|-------|
| 1 | Preflight | root (re-execs with `sudo`), OS (Ubuntu 22.04/24.04, Debian 12; other apt distros: warning only), arch (x86_64/aarch64), systemd is PID 1, Python 3.10-3.13, >= 3 GB free on `/` and the data path, RAM (warn < 900 MB), HTTPS reachability of PyPI (**required**), Telegram, Yahoo, NSE (optional, warning), waits for a running apt/dpkg |
| 2 | Packages | only what is missing: `python3 python3-venv python3-pip libgomp1 sqlite3 ca-certificates curl gnupg tzdata logrotate` and `chrony` (only if no NTP service is active; exactly one stays active), no recommends, `apt-get clean`; verifies `NTPSynchronized` |
| 3 | Swap | if total swap < 512 MB: `/swapfile` of 1 GB (512 MB when less than 3.5 GB would stay free), `fallocate` with `dd` fallback, `chmod 600`, one `/etc/fstab` line, `/etc/sysctl.d/99-monster.conf` (`vm.swappiness=10`, `vm.vfs_cache_pressure=50`, `vm.overcommit_memory=0`). Swap is a safety net, **not** a substitute for RAM. `--skip-swap` disables it |
| 4 | User and directories | system user/group `monster` (no login shell, no sudo); `/var/lib/monster-ai` and `/var/log/monster-ai` `monster:monster 0750`; `/etc/monster-ai` `root:monster 0750` |
| 5 | Release staging | copies the tree (no `.git .venv __pycache__ tests .github *.db *.log` runtime state, **never env files**) to `/opt/monster-ai.new`, stops the services gracefully (on weekdays it waits out 09:00-09:16 IST), swaps it in atomically (`/opt/monster-ai` -> `/opt/monster-ai.prev`) |
| 6 | Virtualenv | `/opt/monster-ai/.venv`; `pip install --no-cache-dir --only-binary=:all: --no-binary=sgmllib3k -r requirements.txt` with 3 tries and backoff (dashboard requirements only when enabled); reused (hard-linked) when the requirements hash is unchanged; `pip check`, smoke import, `compileall` as root |
| 7 | Configuration | `/etc/monster-ai/monster.env` (`root:monster 0640`) is created from `--env-file`, else the project `.env`, else `.env.example`. An existing file **never loses or changes a non-empty value**; missing keys are appended; with `--env-file`, keys that are non-empty there and empty/missing on the server are filled in. `MONSTER_ENV/HOME/DATA_DIR/LOG_DIR` are forced. Secrets are never printed |
| 8 | Validation | `doctor --offline` as the `monster` user; stops with the application's own message when the configuration is invalid |
| 9 | Telegram | token present, chat id missing -> waits for your message (180 s) and stores the id; no token -> loud warning with the exact file, key and command to re-run (never a failure) |
| 10 | Initialise | `init` as `monster` |
| 11 | systemd | installs the units, journald limits (100 MB / 50 MB / 2 weeks), logrotate, the `monster-ai` command, `enable --now` the daemon and the health timer; waits for `READY=1` and a stable process, **rolls back automatically** if the daemon does not come up |
| 12 | Bootstrap | `systemctl start --no-block monster-ai-bootstrap.service`, polled every 15 s for up to 120 min; on failure the last 40 log lines are shown (the daemon keeps retrying by itself) |
| 13 | Verification | daemon state, `health --json`, `telegram-test` (when configured), data-feed reachability from `doctor --json` |
| 14 | Summary | versions, paths, useful commands, warnings, reminder that the mode is SHADOW |

Robustness built in: a second installer run at the same time is refused (`flock`), low disk aborts before any
copy, an interrupted previous swap is repaired, an interrupted pip is retried (the venv is rebuilt when its
hash file is missing), an existing `monster` user or `monster.env` is reused, non-UTF-8 locales and needrestart
prompts are neutralised, a restrictive root umask (027/077) cannot hide the code from the service user, CRLF line
endings of a Windows checkout are repaired, a dropped SSH session does not interrupt apt/pip (SIGHUP is ignored)
and everything is also written to `/var/log/monster-ai-install.log`.

### Flags

| Flag | Meaning |
|------|---------|
| `--env-file PATH` | seed/merge the configuration (see step 7) |
| `--source DIR` | tree to install (default: the directory above `deployment/`) |
| `--no-bootstrap` | do not start the first-time bootstrap (the daemon starts it by itself later) |
| `--skip-swap` | no swap file and no sysctl tuning |
| `--enable-dashboard` | install the read-only dashboard (loopback only) |
| `--yes` | non-interactive |
| `--dry-run` | report only; read-only checks run, no root needed |
| `--no-start` | install and enable, start nothing |
| `--rollback` | manual rollback to `/opt/monster-ai.prev` |

Environment: `MONSTER_INSTALL_DEBUG=1` (shell trace; secrets stay masked), `MONSTER_INSTALL_IGNORE_WINDOW=1`
(do not wait for the 09:00-09:16 IST weekday report window).
Exit codes: `0` ok, `1` failure, `2` bad usage, `3` failed and rolled back, `4` installed but not running,
`75` another installer is running.

## Upgrade, rollback, uninstall

**Upgrade** - run the same command again (`deploy_remote.ps1` / `deploy_remote.sh`, or `sudo bash
/opt/monster-ai/deployment/install.sh --yes` on the server). Code and dependencies are updated, data and secrets
are kept, services are restarted. If the new release does not become active the previous one is restored
automatically (exit code 3); the failed tree stays in `/opt/monster-ai.failed` for inspection. Database
migrations are forward-only and additive; a rollback restores the code, not the schema - take a `sudo
monster-ai backup` before a major upgrade if you want a restore point.

**Rollback by hand** (the previous release is always kept):

```bash
sudo bash /opt/monster-ai/deployment/install.sh --rollback     # swaps /opt/monster-ai and /opt/monster-ai.prev
```

**Uninstall:**

```bash
sudo bash /opt/monster-ai/deployment/uninstall.sh              # keeps data, logs and configuration
sudo bash /opt/monster-ai/deployment/uninstall.sh --purge      # also deletes them (asks first; --yes skips)
sudo bash /opt/monster-ai/deployment/uninstall.sh --purge-swap # also removes /swapfile (never done otherwise)
```

## Troubleshooting

| Symptom | Cause | What to do |
|---------|-------|------------|
| `apt/dpkg is still busy after 300s` | `unattended-upgrades` or another package manager holds the lock (common on a fresh VM) | Wait a few minutes (or reboot) and run the command again - it is safe |
| `could not create a swap file` / `cannot hold a swap file` | file system without swap-file support (ZFS, overlay, tmpfs, some btrfs setups) | Use `--skip-swap`, add swap another way (partition, zram) or choose a larger server |
| `PyPI index ... is unreachable` | DNS, outbound firewall or proxy; the installer aborts before changing anything | `curl -I https://pypi.org/simple/pip/` on the server; allow outbound HTTPS to `pypi.org` and `files.pythonhosted.org`; behind a proxy log in as root and `export https_proxy=...` before running the installer |
| `NSE archives ... is unreachable` or HTTP 403 | NSE blocks some data-centre ranges | Not fatal: MONSTER AI runs in **degraded mode** (fewer sources, data gaps are flagged, reports say so). Try again later or use an Indian region; `NSE_ENABLED=false` stops the attempts |
| `Yahoo Finance ... is unreachable` | outbound block / rate limit | warning only; the bootstrap and the daemon retry with backoff |
| Chat id was not discovered | you did not message the bot within 180 s, a webhook is set on the bot, or another program consumes its updates | Send `/start` to the bot and run the installer again, or set `TELEGRAM_CHAT_ID=<number>` in `/etc/monster-ai/monster.env` |
| No Telegram token | `TELEGRAM_BOT_TOKEN` empty | Create a bot with @BotFather, `sudo nano /etc/monster-ai/monster.env`, set `TELEGRAM_BOT_TOKEN=...`, then `sudo bash /opt/monster-ai/deployment/install.sh --yes` |
| `the clock is not NTP-synchronised` | blocked UDP 123 or a slow first sync; the 09:12 IST cut-off is time critical | `timedatectl`, `chronyc tracking`, allow outbound UDP 123; exactly one NTP service must be active |
| `another MONSTER AI install is already running` (exit 75) | a second installer holds `/var/lock/monster-ai-install.lock` | wait for it |
| Exit 3: `rolled back` | the new release did not stay active | `/var/log/monster-ai-install.log`, `journalctl -u monster-ai -n 100`, `/opt/monster-ai.failed`; fix and run again |
| `health` is CRITICAL in the first hour | the bootstrap has not finished (no data/models yet) | expected; watch `journalctl -u monster-ai-bootstrap -f` |
| SSH session dropped during the installation | network blip | the installer keeps running; reconnect and `tail -f /var/log/monster-ai-install.log` |
| `sudo: a terminal is required` / password prompt hangs | no TTY | run from a real terminal (the deploy scripts use `ssh -tt`) or log in as root |

## Security notes

- **Secrets** live only in `/etc/monster-ai/monster.env` (`root:monster`, `0640`) and reach the services through
  systemd `EnvironmentFile=`. They are never written to unit files, process arguments, the install log or any
  other world-readable file; the installer and the application mask them in all output.
- **SSH**: use key authentication. The deploy scripts use `StrictHostKeyChecking=accept-new` (the first
  connection records the host key, later changes are refused), `ServerAliveInterval=30`, `ConnectTimeout=20`.
  The env file travels over SSH only and is shredded on the server afterwards.
- **No firewall or SSH configuration is changed.** MONSTER AI only makes outbound connections and listens on
  nothing by default. Hardening the host (key-only SSH, `ufw allow OpenSSH`, unattended security updates) is your
  decision.
- **The service user** `monster` has no login shell and no sudo. It is a member of `systemd-journal` so that the
  failure alert can quote the last log lines (read-only access to the journal).
- **Sandbox**: the units run with `NoNewPrivileges`, `ProtectSystem=strict` (only `/var/lib/monster-ai` and
  `/var/log/monster-ai` are writable), `ProtectHome`, `PrivateTmp`, `PrivateDevices`, no capabilities, restricted
  address families and more (see `deployment/systemd/`). `SystemCallFilter` and `MemoryDenyWriteExecute` are
  deliberately not used: numpy/LightGBM break under them.
- **Dashboard** (optional): stays on `127.0.0.1`; the application refuses another bind address unless
  `DASHBOARD_TOKEN` (>= 16 characters) is set. Reach it from your PC with an SSH tunnel:

  ```bash
  ssh -L 8080:127.0.0.1:8080 ubuntu@203.0.113.10      # then open http://127.0.0.1:8080
  ```

## Resource budget

| Resource | Budget |
|----------|--------|
| RAM | daemon `MemoryHigh=750M` / `MemoryMax=880M` (one heavy job at a time, job RSS watchdog at 720 MB), bootstrap the same ceiling with `Nice=10` and idle I/O, health check <= 450 MB, failure alert <= 450 MB, dashboard <= 250 MB; BLAS/OpenMP pinned to 1 thread; `OOMScoreAdjust=-300` for the daemon |
| Swap | 1 GB (512 MB on tight disks) as a safety net, `vm.swappiness=10` |
| Disk | needs >= 3 GB free to install; code + virtualenv about 0.5 GB, the previous release shares the venv through hard links when the requirements did not change; journal <= 100 MB; app logs 5 x 5 MB per file; database and backups are bounded by the application's retention (`BACKUP_KEEP_*`, cache limits, emergency cleanup at `MIN_FREE_DISK_MB_CRIT`) |
| CPU | one vCPU is enough; background work runs at `Nice=10` |
| Network | outbound HTTPS only (PyPI at install time, Yahoo, NSE, GDELT/RSS, Telegram, optional S3), UDP 123 for NTP |
| Time | install about 5 minutes; bootstrap 10-60 minutes |

## External training (`TRAINING_MODE=external`)

A 1 GB server can only train small, capped models. For full-size training the repository ships a GitHub Actions
worker (`.github/workflows/train.yml`) that trains elsewhere and **publishes signed artifacts that the server
pulls and verifies**:

1. **Worker** - weekly (Friday 19:30 UTC = Saturday 01:00 IST) and on demand (*Actions -> Train models (external
   worker) -> Run workflow*). It runs `python -m app.cli init` and `python -m app.cli train --stage both --publish`.
   `--publish` trains, packs `models/<stage>/<version>.zip`, signs the manifest and the `latest.json` pointer with
   HMAC-SHA256 (`MODEL_SIGNING_KEY`) and uploads them to the S3-compatible bucket (AWS, R2, B2, MinIO, Spaces). The
   training log and report files are attached to the run as the artifact `training-report-<run id>`.
   Repository secrets: `MODEL_SIGNING_KEY`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` (optional:
   `S3_ENDPOINT_URL`, `S3_REGION`, `S3_PREFIX`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).
2. **Server** - in `/etc/monster-ai/monster.env` set `TRAINING_MODE=external`, the **same** `MODEL_SIGNING_KEY`
   and either the same `S3_*` values (private bucket) or `MODEL_ARTIFACT_URL=https://...` (public HTTPS URL of the
   published `manifest.json`/pointer); then `sudo systemctl restart monster-ai` (or run the installer again). The
   daemon's `model_sync` job (daily 05:30-07:00 IST) downloads the pointer, verifies the signature, extracts the
   bundle with zip-slip/zip-bomb protection, re-verifies every file and registers the model. Unsigned, tampered or
   truncated artifacts are rejected; a candidate never replaces a healthy champion unless it passes the gates.
3. **Key rotation** - change `MODEL_SIGNING_KEY` in the repository secret and in `monster.env`, then trigger the
   worker once.

## Reference

| Path | Purpose |
|------|---------|
| `/opt/monster-ai` | code (root-owned, read-only to the service), `.venv`, `.requirements.sha256`, `.committed` (the release started successfully; only a committed release may replace `.prev`, so a killed installer never loses the last good release) |
| `/opt/monster-ai.prev` / `.failed` | previous release (rollback) / last failed release (diagnostics) |
| `/var/lib/monster-ai` | state: `monster.db`, `models/`, `cache/`, `backups/`, `tmp/`, `locks/` (`monster:monster 0750`) |
| `/var/log/monster-ai` | application logs (`monster:monster 0750`); `/var/log/monster-ai-install.log` is the installer transcript |
| `/etc/monster-ai/monster.env` | configuration and secrets (`root:monster 0640`) |
| `/etc/systemd/system/monster-ai*.service`, `*.timer` | the units |
| `/usr/local/bin/monster-ai` | wrapper: `sudo monster-ai <command>` runs `python -m app.cli` as `monster` with the production environment |

Units: `monster-ai.service` (Type=notify daemon, `Restart=always`, watchdog 180 s), `monster-ai-bootstrap.service`
(one-shot, retried 6 times per 6 h), `monster-ai-health.service` + `.timer` (every 5 minutes, alerts only on state
change), `monster-ai-alert@.service` (`OnFailure=` Telegram alert with the last journal lines),
`monster-ai-dashboard.service` (optional).

Notes for developers: the shell code is split into `deployment/lib/` (`common.sh`, `envfile.sh`, `decisions.sh`,
`system.sh`) so that the tricky parts - injection-free `set_env_var`, swap sizing, requirements hash and venv
reuse, staging excludes, the swap/rollback/exit-trap logic, the bootstrap polling and the verification verdicts -
are unit-tested (`tests/test_deployment_files.py` runs them under bash against stubs; no root, no network). On
Windows (Git Bash forks slowly) the full file takes a few minutes: `pytest -m "not slow"` skips the heavy ones.
`MONSTER_INSTALL_ROOT` and `MONSTER_INSTALL_SOURCE_ONLY=1` are test hooks only. CI lints everything with
ShellCheck and verifies that `requirements.txt` installs from binary wheels on linux x86_64/aarch64. The only
source build is `sgmllib3k`, a pure-Python package without wheels that `feedparser` needs.
