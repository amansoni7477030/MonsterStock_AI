#!/usr/bin/env bash
# MONSTER AI deployment - common helpers: output, retries, dry-run wrapper, redaction.
#
# Sourced by install.sh and uninstall.sh. Loading this file has no side effects (functions and plain
# variables only), so tests can source it safely. Bash >= 4 (Ubuntu 22.04+, Debian 12, Git Bash).
set -Eeuo pipefail

if [[ -n ${_MONSTER_COMMON_LOADED:-} ]]; then return 0; fi
_MONSTER_COMMON_LOADED=1

: "${DRY_RUN:=0}"
: "${ASSUME_YES:=0}"
: "${TOTAL_STEPS:=14}"
: "${DIE_CALLED:=0}"

# ------------------------------------------------------------------------------------------ conventions
# Paths and names used by every deployment script (the production layout is fixed on purpose).
# MONSTER_INSTALL_ROOT is a TEST HOOK (like DESTDIR): it relocates every path into a sandbox directory so
# the file-level logic can run without root. Production never sets it.
# shellcheck disable=SC2034
{
  readonly ROOT_PREFIX=${MONSTER_INSTALL_ROOT:-}
  readonly SERVICE_USER=monster
  readonly SERVICE_GROUP=monster
  readonly APP_DIR=$ROOT_PREFIX/opt/monster-ai
  readonly APP_PREV=$ROOT_PREFIX/opt/monster-ai.prev
  readonly APP_NEW=$ROOT_PREFIX/opt/monster-ai.new
  readonly APP_FAILED=$ROOT_PREFIX/opt/monster-ai.failed
  readonly DATA_DIR=$ROOT_PREFIX/var/lib/monster-ai
  readonly LOG_DIR=$ROOT_PREFIX/var/log/monster-ai
  readonly CONF_DIR=$ROOT_PREFIX/etc/monster-ai
  readonly ENV_FILE=$CONF_DIR/monster.env
  readonly VENV_DIR=$APP_DIR/.venv
  readonly VENV_PY=$VENV_DIR/bin/python
  readonly HASH_FILE=$APP_DIR/.requirements.sha256
  readonly COMMITTED_FILE=$APP_DIR/.committed
  readonly UNIT_DIR=$ROOT_PREFIX/etc/systemd/system
  readonly INSTALL_LOG=$ROOT_PREFIX/var/log/monster-ai-install.log
  readonly LOCK_FILE=$ROOT_PREFIX/var/lock/monster-ai-install.lock
  readonly CLI_WRAPPER=$ROOT_PREFIX/usr/local/bin/monster-ai
  readonly SWAPFILE=$ROOT_PREFIX/swapfile
  readonly SYSCTL_FILE=$ROOT_PREFIX/etc/sysctl.d/99-monster.conf
  readonly JOURNALD_DROPIN=$ROOT_PREFIX/etc/systemd/journald.conf.d/monster-ai.conf
  readonly LOGROTATE_FILE=$ROOT_PREFIX/etc/logrotate.d/monster-ai
}
WARNINGS=()
REDACT_VALUES=()

# ---------------------------------------------------------------------------------------------- output
# All output goes through these helpers. Writes never abort the script: a vanished SSH terminal must not
# kill an installer that is half-way through apt or pip.
_say() { printf '%s\n' "$*" || true; }
info() { _say "  [info] $*"; }
ok() { _say "  [ ok ] $*"; }
warn() {
  WARNINGS+=("$*")
  _say "  [WARN] $*"
}
err() { _say "  [FAIL] $*"; }
step() {
  _say ""
  _say "[$1/$TOTAL_STEPS] $2"
}
indent() { sed 's/^/        /'; }

# die MESSAGE [EXIT_CODE]
die() {
  DIE_CALLED=1
  err "$1"
  exit "${2:-1}"
}

# fatal MESSAGE [EXIT_CODE]: abort - except in --dry-run, where the problem is only reported.
fatal() {
  if ((DRY_RUN)); then
    warn "[dry-run] would abort here: $1"
    return 0
  fi
  die "$@"
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# quote_cmd ARG...: shell-quoted rendering of a command line (for log lines; never pass secrets here).
quote_cmd() { printf '%q ' "$@"; }

# run CMD ARG...: execute, or only describe the command under --dry-run.
run() {
  if ((DRY_RUN)); then
    info "would run: $(quote_cmd "$@")"
    return 0
  fi
  "$@"
}

# confirm QUESTION: true with --yes; otherwise asks on a terminal; false when there is nobody to ask.
confirm() {
  local answer=""
  if ((ASSUME_YES)); then return 0; fi
  if [[ -t 0 ]]; then
    read -r -p "$1 [y/N] " answer || true
    [[ $answer == [yY] || $answer == [yY][eE][sS] ]]
    return
  fi
  return 1
}

# retry ATTEMPTS FIRST_DELAY_SECONDS CMD ARG...: exponential backoff (the delay doubles after each failure).
retry() {
  local attempts=$1 delay=$2 n=1 rc=0
  shift 2
  while true; do
    "$@" && return 0
    rc=$?
    if ((n >= attempts)); then return "$rc"; fi
    info "attempt $n/$attempts failed (exit $rc): $1 ...; retrying in ${delay}s"
    sleep "$delay"
    delay=$((delay * 2))
    n=$((n + 1))
  done
}

# version_ge A B: true when dotted version A >= B (numeric compare, e.g. 3.10 >= 3.9).
version_ge() {
  [[ $(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1) == "$2" ]]
}

# normalize_line_endings DIR...: strip CR from shell scripts, units and config files so a CRLF checkout
# (Windows) cannot break the server.
normalize_line_endings() {
  local dir
  for dir in "$@"; do
    [[ -d $dir ]] || continue
    find "$dir" -type f \( -name '*.sh' -o -name '*.service' -o -name '*.timer' -o -name '*.conf' \
      -o -name 'logrotate-monster' \) -exec sed -i 's/\r$//' {} +
  done
}

# human_mb MB: "1.5 GB" / "512 MB".
human_mb() {
  awk -v mb="$1" 'BEGIN { if (mb >= 1024) printf "%.1f GB", mb / 1024; else printf "%d MB", mb }'
}

# ------------------------------------------------------------------------------------------ redaction
# Secret values are loaded from the env file (see load_redaction_values in envfile.sh) and masked, together
# with token-shaped strings, in every piece of command output that is echoed to the console.
describe_secret() {
  if [[ -z $1 ]]; then printf '<empty>'; else printf '<set, %d chars>' "${#1}"; fi
}

redact_stream() {
  local line secret
  while IFS= read -r line || [[ -n $line ]]; do
    for secret in ${REDACT_VALUES[@]+"${REDACT_VALUES[@]}"}; do
      if [[ -n $secret ]]; then line=${line//"$secret"/<redacted>}; fi
    done
    printf '%s\n' "$line"
  done | sed -E \
    -e 's/bot[0-9]{5,}:[A-Za-z0-9_-]{20,}/bot<redacted>/g' \
    -e 's/[0-9]{5,}:[A-Za-z0-9_-]{30,}/<redacted-token>/g'
}
