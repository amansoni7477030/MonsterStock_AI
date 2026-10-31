#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# =====================================================================================================
# MONSTER AI - installer / upgrader      Ubuntu 22.04 / 24.04 and Debian 12, native systemd, NO containers
#
#   sudo bash deployment/install.sh [options]          (see --help)
#
# ONE idempotent command: installs the OS packages, creates the `monster` user and directories, stages
# the release into /opt/monster-ai (the previous one stays in /opt/monster-ai.prev), builds the
# virtualenv from binary wheels, merges the configuration, validates it, discovers the Telegram chat id,
# initialises the database, installs and starts the systemd units, runs the first-time bootstrap in the
# background and verifies the result. Re-running upgrades code and dependencies, keeps data and secrets,
# restarts the services and rolls back automatically when the new release does not come up.
#
# Exit codes: 0 ok | 1 failure | 2 bad usage | 3 failed and rolled back | 4 installed but not running
#             75 another installer is running.
# Secrets live only in /etc/monster-ai/monster.env (root:monster, 0640) and are never printed.
# =====================================================================================================
set -Eeuo pipefail
# The service user must be able to read the code and the virtualenv even when root's umask is 027 or 077.
umask 022

export LC_ALL=C.UTF-8 LANG=C.UTF-8
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONDONTWRITEBYTECODE

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=${SCRIPT_PATH%/*}
ORIG_ARGS=("$@")

ENV_SEED_ARG=""
SOURCE_DIR=""
NO_BOOTSTRAP=0
SKIP_SWAP=0
ENABLE_DASHBOARD=0
NO_START=0
ACTION=install
DRY_RUN=0
ASSUME_YES=0

# Pure-Python packages that have no wheel on PyPI (feedparser -> sgmllib3k). They are built from their
# sdist, which needs no compiler; EVERYTHING else must come from a binary wheel.
readonly PIP_SDIST_OK=sgmllib3k
readonly BOOTSTRAP_MAX_MINUTES=120
readonly READY_WAIT_SEC=120

# ------------------------------------------------------------------------- early phase (before the libs)
early_die() {
  printf '[FAIL] %s\n' "$1" >&2
  exit "${2:-1}"
}

usage() {
  cat <<'EOF'
MONSTER AI installer / upgrader (Ubuntu 22.04 / 24.04, Debian 12; native systemd, no containers)

Usage: sudo bash deployment/install.sh [options]

Options:
  --env-file PATH     seed the configuration from PATH. New install: the base of monster.env. Existing
                      install: only keys that are non-empty there and empty/missing in monster.env.
  --source DIR        release tree to install (default: the directory above deployment/)
  --no-bootstrap      do not start the first-time bootstrap (the daemon starts it itself later)
  --skip-swap         do not create a swap file and do not touch kernel memory settings
  --enable-dashboard  install the optional read-only dashboard (loopback only)
  --yes, -y           non-interactive: assume "yes" to every question
  --dry-run           only report what would be done (read-only checks run; root is not required)
  --no-start          install and enable everything but start nothing
  --rollback          swap /opt/monster-ai with /opt/monster-ai.prev and restart (manual rollback)
  -h, --help          this text

Environment: MONSTER_INSTALL_DEBUG=1         shell trace (secrets stay masked)
             MONSTER_INSTALL_IGNORE_WINDOW=1 do not wait for the 09:00-09:16 IST weekday report window
EOF
}

usage_error() {
  printf '[FAIL] %s\n\n' "$1" >&2
  usage >&2
  exit 2
}

parse_args() {
  while (($#)); do
    case $1 in
      --env-file)
        (($# >= 2)) || usage_error "--env-file needs a path"
        ENV_SEED_ARG=$2
        shift 2
        ;;
      --env-file=*)
        ENV_SEED_ARG=${1#*=}
        shift
        ;;
      --source)
        (($# >= 2)) || usage_error "--source needs a directory"
        SOURCE_DIR=$2
        shift 2
        ;;
      --source=*)
        SOURCE_DIR=${1#*=}
        shift
        ;;
      --no-bootstrap)
        NO_BOOTSTRAP=1
        shift
        ;;
      --skip-swap)
        SKIP_SWAP=1
        shift
        ;;
      --enable-dashboard)
        ENABLE_DASHBOARD=1
        shift
        ;;
      --yes | -y)
        ASSUME_YES=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --no-start)
        NO_START=1
        shift
        ;;
      --rollback)
        ACTION=rollback
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *) usage_error "unknown option: $1" ;;
    esac
  done
  if [[ -n $ENV_SEED_ARG ]]; then
    [[ -f $ENV_SEED_ARG ]] || early_die "--env-file: no such file: $ENV_SEED_ARG" 2
    ENV_SEED_ARG=$(readlink -f -- "$ENV_SEED_ARG")
  fi
  if [[ -z $SOURCE_DIR ]]; then SOURCE_DIR=${SCRIPT_DIR%/*}; fi
  [[ -d $SOURCE_DIR ]] || early_die "--source: no such directory: $SOURCE_DIR" 2
  SOURCE_DIR=$(readlink -f -- "$SOURCE_DIR")
}

reexec_with_sudo() {
  if ((EUID == 0)) || ((DRY_RUN)); then return 0; fi
  command -v sudo >/dev/null 2>&1 || early_die "this installer must run as root and sudo is not installed - log in as root and run it again"
  printf 'Not running as root: re-running with sudo ...\n' >&2
  exec sudo env "MONSTER_INSTALL_DEBUG=${MONSTER_INSTALL_DEBUG:-0}" \
    "MONSTER_INSTALL_IGNORE_WINDOW=${MONSTER_INSTALL_IGNORE_WINDOW:-0}" bash "$SCRIPT_PATH" "${ORIG_ARGS[@]}"
}

# A checkout made on Windows can carry CRLF line endings: repair the libraries before sourcing them.
repair_libs_crlf() {
  local f
  if ((EUID != 0)); then return 0; fi
  for f in "$SCRIPT_DIR"/lib/*.sh; do
    if [[ -f $f ]] && grep -q $'\r' "$f"; then sed -i 's/\r$//' "$f"; fi
  done
}

if [[ ${MONSTER_INSTALL_SOURCE_ONLY:-0} != 1 ]]; then
  parse_args "$@"
  reexec_with_sudo
  [[ -r $SCRIPT_DIR/lib/common.sh ]] || early_die "deployment/lib/ is missing next to install.sh - run the installer from the full project tree"
  repair_libs_crlf
fi
# MONSTER_INSTALL_SOURCE_ONLY=1 (tests): define the functions without parsing arguments or running anything.
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/envfile.sh
source "$SCRIPT_DIR/lib/envfile.sh"
# shellcheck source=lib/decisions.sh
source "$SCRIPT_DIR/lib/decisions.sh"
# shellcheck source=lib/system.sh
source "$SCRIPT_DIR/lib/system.sh"

# --------------------------------------------------------------------------------------- session state
LAST_ERROR=""
HAD_PREV=0
REQ_KEY=""
PY_BIN=""
PY_VER=""
PY_TAG=""
ARCH=""
OS_ID=""
OS_VERSION_ID=""
OS_PRETTY=""
WORKDIR=""
SEED_FILE=""
TEE_PID=""
NTP_PLAN=""
BOOTSTRAP_RESULT=skipped
TELEGRAM_STATE=unknown
HEALTH_NOTE=""

# ---------------------------------------------------------------------------- startup, traps and logging
make_workdir() {
  local base=/run
  if [[ ! -d $base || ! -w $base ]]; then base=${TMPDIR:-/tmp}; fi
  WORKDIR=$(mktemp -d "$base/monster-ai-install.XXXXXX")
  chmod 700 "$WORKDIR"
}

# Everything is also written to $INSTALL_LOG. SIGHUP is ignored: a dropped SSH session must not kill
# apt/dpkg/pip half-way (that would leave a broken package state that needs manual repair).
setup_install_log() {
  local -a tee_args=(-a)
  trap '' HUP
  if ! { (umask 077 && : >>"$INSTALL_LOG"); } 2>/dev/null; then
    warn "cannot write $INSTALL_LOG; continuing without a transcript"
    return 0
  fi
  chmod 600 "$INSTALL_LOG"
  if tee --output-error=warn /dev/null </dev/null >/dev/null 2>&1; then tee_args=(--output-error=warn -a); fi
  exec 3>&1
  exec > >(tee "${tee_args[@]}" "$INSTALL_LOG" >&3) 2>&1
  TEE_PID=$!
  printf '\n===== MONSTER AI install started %s (pid %d) =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$$"
}

close_install_log() {
  if [[ -z $TEE_PID ]]; then return 0; fi
  exec 1>&- 2>&-
  wait "$TEE_PID" 2>/dev/null || true
}

on_error() { LAST_ERROR="exit $1 at install.sh line $2: $3"; }

on_signal() {
  DIE_CALLED=1
  err "interrupted by SIG$1"
  exit "$2"
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM ERR
  set +e +u +o pipefail
  if ((rc != 0 && !DIE_CALLED)) && [[ -n $LAST_ERROR ]]; then err "unexpected error: $LAST_ERROR"; fi
  if ((rc != 0 && RELEASE_SWAPPED && !RELEASE_COMMITTED && !ROLLED_BACK && !DRY_RUN)); then
    rollback_release "the installation failed (exit $rc)" || true
    if ((ROLLED_BACK && rc == 1)); then rc=3; fi
  fi
  restore_services
  if [[ -n $WORKDIR && -d $WORKDIR ]]; then rm -rf -- "$WORKDIR"; fi
  if ((rc != 0)); then
    _say ""
    _say "  The installer stopped (exit $rc). It is safe to run it again."
    if ((!DRY_RUN)); then _say "  Full transcript: $INSTALL_LOG"; fi
  fi
  close_install_log
  exit "$rc"
}

startup() {
  if ((DRY_RUN)); then
    info "DRY RUN: nothing will be changed (read-only checks only)"
  else
    ((EUID == 0)) || die "must run as root"
    acquire_lock
    setup_install_log
  fi
  make_workdir
  trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
  trap on_exit EXIT
  trap 'on_signal INT 130' INT
  trap 'on_signal TERM 143' TERM
  if [[ ${MONSTER_INSTALL_DEBUG:-0} == 1 ]]; then
    PS4='+ install.sh:${LINENO}: '
    set -x
  fi
}

# ============================================================================================ [1/14]
detect_platform() {
  ARCH=$(uname -m)
  OS_ID=$(os_field ID 2>/dev/null) || OS_ID=""
  OS_VERSION_ID=$(os_field VERSION_ID 2>/dev/null) || OS_VERSION_ID=""
  OS_PRETTY=$(os_field PRETTY_NAME 2>/dev/null) || OS_PRETTY=""
  if [[ -z $OS_PRETTY ]]; then OS_PRETTY="unknown Linux"; fi
}

check_systemd() {
  local init
  init=$(ps -p 1 -o comm= 2>/dev/null | tr -d ' ') || init=""
  if [[ -d /run/systemd/system && $init == systemd ]]; then
    ok "systemd is PID 1 ($(systemctl --version 2>/dev/null | sed -n 1p))"
  else
    fatal "systemd is not running as PID 1 (container or WSL1?). MONSTER AI is managed by systemd and needs a real systemd host."
  fi
}

check_source_tree() {
  local f missing=()
  for f in requirements.txt .env.example app/__init__.py app/cli.py deployment/systemd/monster-ai.service \
    deployment/journald-monster.conf deployment/logrotate-monster; do
    [[ -f $SOURCE_DIR/$f ]] || missing+=("$f")
  done
  if ((${#missing[@]})); then
    fatal "the release tree $SOURCE_DIR is incomplete; missing: ${missing[*]}"
  else
    ok "release tree: $SOURCE_DIR"
  fi
}

check_python_early() {
  local v
  if ! have_cmd python3; then
    info "python3 is not installed yet; the next step installs it"
    return 0
  fi
  v=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || v=""
  if [[ -z $v ]]; then
    warn "python3 is present but did not run"
  elif py_version_supported "$v"; then
    ok "python3 $v"
  elif version_ge "$v" 3.14; then
    fatal "python3 $v is newer than the pinned dependencies support (CPython 3.10 - 3.13 have binary wheels). Use Ubuntu 22.04/24.04 or Debian 12."
  else
    fatal "python3 $v is too old (3.10 or newer is required). Use Ubuntu 22.04/24.04 or Debian 12 - do not replace the system Python."
  fi
}

num_or_zero() { if [[ ${1:-} =~ ^[0-9]+$ ]]; then printf '%s' "$1"; else printf '0'; fi; }

check_resources() {
  local root_free data_free ram
  root_free=$(num_or_zero "$(free_disk_mb / 2>/dev/null)")
  data_free=$(num_or_zero "$(free_disk_mb "$DATA_DIR" 2>/dev/null)")
  if ((root_free < 3072)); then
    fatal "only $(human_mb "$root_free") free on / - at least 3 GB are required. Free some disk space (apt-get clean, old kernels, logs) and re-run."
  else
    ok "free disk on /: $(human_mb "$root_free")"
  fi
  if ((data_free < 3072)); then
    fatal "only $(human_mb "$data_free") free on the data path ($DATA_DIR) - at least 3 GB are required."
  else
    ok "free disk on $DATA_DIR: $(human_mb "$data_free")"
  fi
  ram=$(num_or_zero "$(meminfo_mb MemTotal 2>/dev/null)")
  if ((ram < 900)); then
    warn "only $(human_mb "$ram") RAM: the 1 GB target is already tight. Training and backfill will be slow and rely on swap."
  else
    ok "RAM: $(human_mb "$ram")"
  fi
}

ensure_download_tool() {
  if have_cmd curl; then return 0; fi
  info "curl is missing; installing it (needed for the connectivity checks)"
  if ((DRY_RUN)); then return 0; fi
  repair_dpkg
  apt_get update
  apt_get install curl ca-certificates
}

# probe_endpoint NAME URL required|optional [HINT]: any HTTP answer proves the route works; 000 means no answer.
# Required endpoints get 3 attempts, optional ones 2, so a blocked optional feed costs about half a minute.
probe_endpoint() {
  local name=$1 url=$2 mode=$3 hint=${4:-} code="000" attempt attempts=2
  if [[ $mode == required ]]; then attempts=3; fi
  if ! have_cmd curl; then
    warn "cannot test $name: curl is not available"
    return 0
  fi
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    code=$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 8 --max-time 20 "$url" 2>/dev/null) || code="000"
    if [[ $code != 000 ]]; then
      ok "$name reachable (HTTP $code)"
      return 0
    fi
    if ((attempt < attempts)); then sleep $((attempt * 3)); fi
  done
  if [[ $mode == required ]]; then
    fatal "$name ($url) is unreachable over HTTPS. Check DNS, outbound firewall rules and any proxy, then re-run. $hint"
  else
    warn "$name ($url) is unreachable. $hint"
  fi
}

check_network() {
  probe_endpoint "PyPI index" https://pypi.org/simple/pip/ required "Python packages cannot be downloaded without it."
  probe_endpoint "PyPI files" https://files.pythonhosted.org/ required "Python packages cannot be downloaded without it."
  probe_endpoint "Telegram API" https://api.telegram.org/ optional "Reports and alerts cannot be delivered until it is reachable."
  probe_endpoint "Yahoo Finance" https://query1.finance.yahoo.com/ optional "MONSTER AI degrades gracefully (fewer data sources)."
  probe_endpoint "NSE archives" https://nsearchives.nseindia.com/ optional "MONSTER AI degrades gracefully (degraded mode, fewer data sources)."
}

step_preflight() {
  step 1 "Preflight checks"
  if ((EUID != 0)); then
    if ((DRY_RUN)); then warn "not running as root (fine for --dry-run)"; else fatal "must run as root"; fi
  fi
  detect_platform
  if os_supported "$OS_ID" "$OS_VERSION_ID"; then
    ok "OS: $OS_PRETTY"
  elif have_cmd apt-get; then
    warn "OS '$OS_PRETTY' is not a tested platform (Ubuntu 22.04/24.04, Debian 12); continuing because apt is available"
  else
    fatal "unsupported OS '$OS_PRETTY': an apt-based Linux with systemd is required"
  fi
  if arch_supported "$ARCH"; then
    ok "CPU architecture: $ARCH"
  else
    fatal "unsupported CPU architecture '$ARCH': binary wheels of the pinned dependencies exist for x86_64 and aarch64 only"
  fi
  check_systemd
  check_source_tree
  check_python_early
  check_resources
  ensure_download_tool
  check_network
  if apt_busy; then wait_for_apt "$APT_LOCK_TIMEOUT"; fi
}

# ============================================================================================ [2/14]
select_python() {
  PY_BIN=$(command -v python3 2>/dev/null) || PY_BIN=""
  if [[ -z $PY_BIN ]]; then
    fatal "python3 is not available even after the package installation"
    return 0
  fi
  PY_VER=$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  PY_TAG=$("$PY_BIN" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
  if ! py_version_supported "$PY_VER"; then
    fatal "python3 is $PY_VER but 3.10 - 3.13 is required (the pinned wheels). Use Ubuntu 22.04/24.04 or Debian 12; do not replace the system Python."
    return 0
  fi
  if ! "$PY_BIN" -c 'import ensurepip, venv' 2>/dev/null; then
    fatal "the Python venv module is missing (apt install python3-venv)"
    return 0
  fi
  ok "using $PY_BIN ($PY_VER)"
}

apply_ntp_plan() {
  local waited=0
  case $NTP_PLAN in
    install-chrony)
      run systemctl enable --now chrony.service || warn "could not start chrony; check 'systemctl status chrony'"
      ;;
    keep:chrony+disable:systemd-timesyncd)
      info "chrony and systemd-timesyncd are both active; keeping chrony only"
      run systemctl disable --now systemd-timesyncd.service || warn "could not disable systemd-timesyncd"
      ;;
    *warn:multiple)
      warn "several time-sync services are active (${NTP_ACTIVE[*]}): keep exactly one of them"
      ;;
    keep:*) ok "time synchronisation: ${NTP_PLAN#keep:} is active" ;;
  esac
  if ((DRY_RUN)); then return 0; fi
  if have_cmd chronyc && svc_active chrony.service; then chronyc -a makestep >/dev/null 2>&1 || true; fi
  while ((waited < 60)) && ! ntp_synchronized; do
    sleep 3
    waited=$((waited + 3))
  done
  if ntp_synchronized; then
    ok "system clock is NTP-synchronised"
  else
    warn "the clock is not NTP-synchronised yet. The 09:12 IST cut-off is time critical: check 'timedatectl' and 'chronyc tracking' (UDP 123 must be allowed outbound)."
  fi
}

step_packages() {
  step 2 "System packages"
  local pkg missing=()
  local -a base=(python3 python3-venv python3-pip libgomp1 sqlite3 ca-certificates curl gnupg tzdata logrotate)
  for pkg in "${base[@]}"; do
    if ! pkg_installed "$pkg"; then missing+=("$pkg"); fi
  done
  mapfile -t NTP_ACTIVE < <(ntp_active_services)
  NTP_PLAN=$(ntp_plan "${NTP_ACTIVE[@]}")
  if [[ $NTP_PLAN == install-chrony ]] && ! pkg_installed chrony; then missing+=(chrony); fi
  if ((${#missing[@]})); then
    info "installing: ${missing[*]}"
    if ((!DRY_RUN)); then
      repair_dpkg
      apt_get update
      apt_get install --no-install-recommends "${missing[@]}"
      apt-get clean || true
    fi
  else
    ok "all required packages are already installed"
  fi
  if ((!DRY_RUN)) || have_cmd python3; then select_python; fi
  apply_ntp_plan
}

# ============================================================================================ [3/14]
step_swap() {
  step 3 "Swap (safety net) and kernel memory settings"
  local total free size
  if ((SKIP_SWAP)); then
    info "--skip-swap: swap and kernel memory settings are left alone"
    return 0
  fi
  total=$(num_or_zero "$(meminfo_mb SwapTotal 2>/dev/null)")
  free=$(num_or_zero "$(free_disk_mb / 2>/dev/null)")
  size=$(swap_target_mb "$total" "$free" 0)
  if ((DRY_RUN)); then
    info "would apply: swap total ${total} MB, free disk ${free} MB -> create ${size} MB swap file, write $SYSCTL_FILE"
    return 0
  fi
  if swap_file_active; then
    ok "$SWAPFILE is already active"
    ensure_fstab_entry
  elif ((total >= 512)); then
    ok "swap is already configured ($(human_mb "$total"))"
  elif ((size == 0)); then
    warn "not enough free disk ($(human_mb "$free")) to add a swap file safely; continuing without it"
  else
    info "total swap is only $(human_mb "$total"): creating a $(human_mb "$size") swap file at $SWAPFILE"
    if create_swapfile "$size"; then
      ensure_fstab_entry
      ok "swap file active ($(human_mb "$size"), persisted in /etc/fstab)"
    else
      warn "this file system ($(fs_type /)) cannot hold a swap file; continuing without swap"
    fi
  fi
  write_file "$SYSCTL_FILE" 0644 <<'EOF'
# MONSTER AI memory tuning (written by deployment/install.sh)
vm.swappiness = 10
vm.vfs_cache_pressure = 50
vm.overcommit_memory = 0
EOF
  sysctl -q -p "$SYSCTL_FILE" >/dev/null 2>&1 || warn "could not apply $SYSCTL_FILE now (it takes effect at the next boot)"
  info "Swap is a safety net for short memory spikes, NOT a substitute for RAM: sustained swapping slows the 09:12 IST report path. If it is used constantly, move to a larger server."
}

# ============================================================================================ [4/14]
step_user_dirs() {
  step 4 "Service user, directories and permissions"
  local d nologin
  nologin=$(command -v nologin 2>/dev/null) || nologin=/usr/sbin/nologin
  if ((DRY_RUN)); then
    info "would ensure user/group $SERVICE_USER (no login, no sudo), $DATA_DIR $LOG_DIR $CONF_DIR with 0750"
    return 0
  fi
  if ! getent group "$SERVICE_GROUP" >/dev/null; then groupadd --system "$SERVICE_GROUP"; fi
  if id -u "$SERVICE_USER" >/dev/null 2>&1; then
    ok "user '$SERVICE_USER' already exists"
  else
    useradd --system --gid "$SERVICE_GROUP" --home-dir "$DATA_DIR" --no-create-home \
      --shell "$nologin" --comment "MONSTER AI service" "$SERVICE_USER"
    ok "created system user '$SERVICE_USER' (no login shell, no sudo)"
  fi
  # Journal access lets the failure alert (and manual diagnostics) include the last log lines.
  if ! getent group systemd-journal >/dev/null; then groupadd --system systemd-journal; fi
  if ! id -nG "$SERVICE_USER" | tr ' ' '\n' | grep -qx systemd-journal; then
    usermod -aG systemd-journal "$SERVICE_USER"
  fi
  install -d -m 0750 -o root -g "$SERVICE_GROUP" "$CONF_DIR"
  install -d -m 0750 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$DATA_DIR" "$LOG_DIR"
  for d in models cache tmp locks; do install -d -m 0750 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$DATA_DIR/$d"; done
  install -d -m 0700 -o "$SERVICE_USER" -g "$SERVICE_GROUP" "$DATA_DIR/backups"
  if [[ -n $(find "$DATA_DIR" "$LOG_DIR" ! -user "$SERVICE_USER" -print -quit) ]]; then
    info "repairing ownership below $DATA_DIR and $LOG_DIR"
    chown -R "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR" "$LOG_DIR"
  fi
  ok "$DATA_DIR and $LOG_DIR are owned by $SERVICE_USER (0750); $CONF_DIR is root:$SERVICE_GROUP (0750)"
}

# ============================================================================================ [5/14]
# Snapshot the environment seed into the private work dir NOW: --source may be the very tree that is
# replaced below, and --env-file may be deleted by the caller afterwards.
prepare_seed() {
  local src=""
  SEED_FILE=""
  if [[ -n $ENV_SEED_ARG ]]; then
    src=$ENV_SEED_ARG
  elif [[ -f $SOURCE_DIR/.env ]]; then
    src=$SOURCE_DIR/.env
  fi
  if [[ -z $src ]]; then return 0; fi
  SEED_FILE=$WORKDIR/seed.env
  env_sanitize_seed "$src" "$SEED_FILE"
  info "configuration seed: $src (values are never printed)"
}

resolve_dashboard_wanted() {
  local v=""
  DASHBOARD_WANTED=0
  if ((ENABLE_DASHBOARD)); then
    DASHBOARD_WANTED=1
    return 0
  fi
  if [[ -r $ENV_FILE ]]; then v=$(env_get DASHBOARD_ENABLED "$ENV_FILE" 2>/dev/null || true); fi
  if [[ -z $v && -n $SEED_FILE ]]; then v=$(env_get DASHBOARD_ENABLED "$SEED_FILE" 2>/dev/null || true); fi
  if [[ -n $v ]] && env_is_true "$v"; then DASHBOARD_WANTED=1; fi
}

# A crash between the two renames of a previous swap leaves no /opt/monster-ai: bring the old release back.
recover_interrupted_swap() {
  if [[ ! -d $APP_DIR && -d $APP_PREV ]]; then
    warn "found an interrupted release swap: restoring $APP_PREV as $APP_DIR"
    mv -- "$APP_PREV" "$APP_DIR"
  fi
  rm -rf -- "$APP_NEW" "$APP_FAILED"
}

stage_release() {
  info "staging $SOURCE_DIR -> $APP_NEW"
  stage_tree "$SOURCE_DIR" "$APP_NEW"
  normalize_line_endings "$APP_NEW"
  chown -R root:root "$APP_NEW"
  chmod -R u=rwX,go=rX "$APP_NEW"
  chmod 0755 "$APP_NEW"/deployment/*.sh
  assert_no_env_files "$APP_NEW" || die "refusing to install a release that contains env files"
  ok "staged $(du -sh "$APP_NEW" | cut -f 1) of code (no tests, no .git, no env files, no runtime state)"
}

# On weekdays the pre-open session and the report run 09:00-09:16 IST: never stop the daemon then.
wait_for_quiet_window() {
  local dow hhmm secs
  if [[ ${MONSTER_INSTALL_IGNORE_WINDOW:-0} == 1 ]] || ! svc_active monster-ai.service; then return 0; fi
  read -r dow hhmm < <(ist_dow_hhmm)
  secs=$(seconds_until_window_end "$dow" "$hhmm")
  if ((secs > 0)); then
    warn "it is $hhmm IST on a weekday: the report window (09:00-09:16 IST) is protected. Waiting ${secs}s before restarting the daemon (MONSTER_INSTALL_IGNORE_WINDOW=1 skips this)."
    sleep "$secs"
  fi
}

swap_release() {
  local note=""
  if [[ -d $APP_DIR ]]; then
    HAD_PREV=1
    RELEASE_SWAPPED=1
    if [[ -d $APP_PREV && ! -f $COMMITTED_FILE ]]; then
      # A previous run was killed before its release started: the last good release is still in .prev.
      warn "$APP_DIR is an unfinished installation (no .committed marker): discarding it; $APP_PREV stays the rollback target"
      rm -rf -- "$APP_DIR"
    else
      rm -rf -- "$APP_PREV"
      mv -- "$APP_DIR" "$APP_PREV"
    fi
    note=", the previous release is kept in $APP_PREV"
  fi
  RELEASE_SWAPPED=1
  mv -- "$APP_NEW" "$APP_DIR"
  ok "release swapped: $APP_DIR is the new release$note"
}

step_release() {
  step 5 "Release staging and atomic swap"
  if ((DRY_RUN)); then
    info "would stage $SOURCE_DIR into $APP_NEW (excluding .git .venv __pycache__ tests .github runtime state *.db *.log .env), stop the services, swap with $APP_DIR (previous -> $APP_PREV)"
    return 0
  fi
  prepare_seed
  resolve_dashboard_wanted
  recover_interrupted_swap
  stage_release
  wait_for_quiet_window
  stop_services
  swap_release
}

# ============================================================================================ [6/14]
reuse_venv_from_prev() {
  rm -rf -- "$VENV_DIR"
  # Hard links: the previous release keeps a fully working venv (rollback) and nothing is duplicated on disk.
  if cp -al -- "$APP_PREV/.venv" "$VENV_DIR" 2>/dev/null; then return 0; fi
  rm -rf -- "$VENV_DIR"
  cp -a -- "$APP_PREV/.venv" "$VENV_DIR"
}

build_venv() {
  local -a pip_args=(--disable-pip-version-check --no-cache-dir --progress-bar off --retries 5 --timeout 60)
  local -a reqs=(-r "$APP_DIR/requirements.txt")
  if ((DASHBOARD_WANTED)); then reqs+=(-r "$APP_DIR/requirements-dashboard.txt"); fi
  info "creating the virtualenv $VENV_DIR with Python $PY_VER"
  rm -rf -- "$VENV_DIR"
  "$PY_BIN" -m venv "$VENV_DIR"
  retry 3 10 "$VENV_PY" -m pip install "${pip_args[@]}" --upgrade pip
  info "installing the dependencies from binary wheels (no compiler is ever used; sdist exception: $PIP_SDIST_OK)"
  retry 3 15 "$VENV_PY" -m pip install "${pip_args[@]}" --only-binary=:all: "--no-binary=$PIP_SDIST_OK" "${reqs[@]}"
}

verify_venv() {
  local out
  if ! out=$("$VENV_PY" -m pip check 2>&1); then
    warn "pip check failed: $out"
    return 1
  fi
  if ! out=$("$VENV_PY" -c 'import numpy, pandas, scipy, lightgbm, requests, feedparser' 2>&1); then
    warn "dependency smoke import failed: $out"
    return 1
  fi
  if ((DASHBOARD_WANTED)); then
    if ! out=$("$VENV_PY" -c 'import fastapi, uvicorn, starlette, pydantic, httpx' 2>&1); then
      warn "dashboard dependency import failed: $out"
      return 1
    fi
  fi
}

step_venv() {
  step 6 "Python virtual environment (binary wheels only)"
  local -a reqs=("$APP_DIR/requirements.txt")
  if ((DRY_RUN)); then
    info "would build $VENV_DIR with: pip install --no-cache-dir --only-binary=:all: --no-binary=$PIP_SDIST_OK -r requirements.txt (3 tries, backoff), reuse it when the requirements hash is unchanged, pip check, smoke import, compileall"
    return 0
  fi
  if ((DASHBOARD_WANTED)); then reqs+=("$APP_DIR/requirements-dashboard.txt"); fi
  REQ_KEY=$(requirements_key "$PY_TAG" "$ARCH" "${reqs[@]}")
  if venv_can_reuse "$REQ_KEY" "$APP_PREV/.requirements.sha256" "$APP_PREV/.venv" && reuse_venv_from_prev; then
    ok "requirements unchanged (hash ${REQ_KEY:0:12}): reusing the existing virtualenv"
    if ! verify_venv; then
      warn "the reused virtualenv failed its checks; rebuilding it"
      build_venv
    fi
  else
    build_venv
  fi
  verify_venv || die "the virtualenv failed its checks (messages above)"
  ok "pip check and the smoke import (numpy pandas scipy lightgbm requests feedparser) passed"
  info "precompiling bytecode (the service cannot write __pycache__)"
  "$VENV_PY" -m compileall -q -x '/\.venv/' "$APP_DIR"
  printf '%s\n' "$REQ_KEY" >"$HASH_FILE"
  rm -rf -- "${HOME:-/root}/.cache/pip" 2>/dev/null || true
  ok "virtualenv ready ($(du -sh "$VENV_DIR" | cut -f 1))"
}

# ============================================================================================ [7/14]
force_env_paths() {
  set_env_var MONSTER_ENV production "$ENV_FILE"
  set_env_var MONSTER_HOME "$APP_DIR" "$ENV_FILE"
  set_env_var MONSTER_DATA_DIR "$DATA_DIR" "$ENV_FILE"
  set_env_var MONSTER_LOG_DIR "$LOG_DIR" "$ENV_FILE"
}

print_secret_summary() (
  set +x
  local key
  for key in "${ENV_SECRET_KEYS[@]}"; do
    printf '        %-24s %s\n' "$key" "$(describe_secret "$(env_get "$key" "$ENV_FILE" 2>/dev/null || true)")"
  done
)

step_envfile() {
  step 7 "Configuration $ENV_FILE"
  local template=$APP_DIR/.env.example tmp names count
  if ((DRY_RUN)); then
    info "would create $ENV_FILE from --env-file, else <source>/.env, else .env.example (secrets empty); an existing file keeps every non-empty value and only gains missing keys; paths forced; root:$SERVICE_GROUP 0640"
    return 0
  fi
  [[ -f $template ]] || die "$template is missing from the release"
  if [[ ! -f $ENV_FILE ]]; then
    tmp=$(mktemp "$CONF_DIR/.monster.env.XXXXXX")
    env_create_from_template "$template" "$SEED_FILE" "$tmp"
    chown "root:$SERVICE_GROUP" "$tmp"
    chmod 0640 "$tmp"
    mv -f -- "$tmp" "$ENV_FILE"
    if [[ -n $ENV_SEED_ARG ]]; then
      ok "created $ENV_FILE from --env-file"
    elif [[ -n $SEED_FILE ]]; then
      ok "created $ENV_FILE from the .env of the source tree"
    else
      ok "created $ENV_FILE from .env.example (every key present, secrets empty)"
    fi
  else
    ok "$ENV_FILE exists: existing non-empty values are never overwritten"
    if [[ -n $ENV_SEED_ARG && -n $SEED_FILE ]]; then
      names=$(env_import_seed "$SEED_FILE" "$ENV_FILE" "$template" | tr '\n' ' ') || names=""
      count=$(wc -w <<<"$names" | tr -d ' ')
      info "imported $count key(s) from --env-file (they were empty or missing): ${names:-none}"
    fi
    names=$(env_append_missing "$template" "$ENV_FILE" | tr '\n' ' ') || names=""
    count=$(wc -w <<<"$names" | tr -d ' ')
    if ((count > 0)); then info "appended $count missing key(s) with their defaults: $names"; fi
  fi
  force_env_paths
  if ((ENABLE_DASHBOARD)); then set_env_var DASHBOARD_ENABLED true "$ENV_FILE"; fi
  chown "root:$SERVICE_GROUP" "$ENV_FILE"
  chmod 0640 "$ENV_FILE"
  if [[ -n $SEED_FILE ]]; then
    names=$(env_unknown_keys "$SEED_FILE" "$template" | tr '\n' ' ') || names=""
    if [[ -n $names ]]; then warn "ignored keys of the seed file that MONSTER AI does not know (typos?): $names"; fi
  fi
  load_redaction_values "$ENV_FILE"
  info "secrets in $ENV_FILE (values are never shown):"
  print_secret_summary
}

# ============================================================================================ [8/14]
# doctor_verdict (decisions.sh) reads both the exit code and the [FAIL]/[warn] markers, because the exit
# code semantics differ between the written contract (0/1/2) and the implementation (0/1). A failure on a
# brand-new host can merely be missing state, so init is tried once before the verdict is final.
step_validate_config() {
  step 8 "Validating the configuration (doctor --offline, as $SERVICE_USER)"
  local out rc=0 verdict
  if ((DRY_RUN)); then
    info "would run: python -m app.cli doctor --offline"
    return 0
  fi
  out=$(run_as_monster "$VENV_PY" -m app.cli doctor --offline 2>&1) || rc=$?
  printf '%s\n' "$out" | redact_stream | indent
  verdict=$(doctor_verdict "$rc" "$out")
  case $verdict in
    ok) ok "doctor: all checks passed" ;;
    warn) warn "doctor reported warnings (listed above); they are optional improvements and the installation continues" ;;
    *)
      info "doctor reported a failure; running init once to see whether it only reflects a first run"
      run_as_monster "$VENV_PY" -m app.cli init 2>&1 | redact_stream | indent || true
      rc=0
      out=$(run_as_monster "$VENV_PY" -m app.cli doctor --offline 2>&1) || rc=$?
      printf '%s\n' "$out" | redact_stream | indent
      if [[ $(doctor_verdict "$rc" "$out") == fail ]]; then
        die "the configuration or the host failed validation (doctor exit $rc, report above). Fix $ENV_FILE and run again: sudo bash $APP_DIR/deployment/install.sh --yes"
      fi
      warn "doctor passes after init; continuing"
      ;;
  esac
}

# ============================================================================================ [9/14]
env_is_set() ( set +x; [[ -n $(env_get "$1" "$2" 2>/dev/null || true) ]] )

telegram_missing_help() {
  warn "TELEGRAM_BOT_TOKEN is not set: reports and alerts CANNOT be delivered until you add it."
  info "  1. Create a bot with @BotFather in Telegram and copy its token."
  info "  2. Edit the file:  sudo nano $ENV_FILE   and set   TELEGRAM_BOT_TOKEN=<your token>"
  info "  3. Run once more:  sudo bash $APP_DIR/deployment/install.sh --yes   (finds the chat id, restarts the services)"
}

step_telegram() {
  step 9 "Telegram"
  local out chat rc=0
  if ((DRY_RUN)); then
    info "would discover TELEGRAM_CHAT_ID with 'telegram-setup --print-only --wait 180' when only the token is set"
    return 0
  fi
  if ! env_is_set TELEGRAM_BOT_TOKEN "$ENV_FILE"; then
    TELEGRAM_STATE=no-token
    telegram_missing_help
    return 0
  fi
  if env_is_set TELEGRAM_CHAT_ID "$ENV_FILE"; then
    TELEGRAM_STATE=configured
    ok "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set"
    return 0
  fi
  info "TELEGRAM_BOT_TOKEN is set but TELEGRAM_CHAT_ID is empty."
  info "Send any message to your bot now (waiting up to 180 s) ..."
  out=$(run_as_monster "$VENV_PY" -m app.cli telegram-setup --print-only --wait 180) || rc=$?
  chat=$(parse_chat_id "$out")
  if ((rc == 0)) && [[ -n $chat ]]; then
    set_env_var TELEGRAM_CHAT_ID "$chat" "$ENV_FILE"
    TELEGRAM_STATE=configured
    ok "discovered the chat id and stored it as TELEGRAM_CHAT_ID"
  else
    TELEGRAM_STATE=no-chat-id
    warn "no message from your bot arrived (or the bot is used elsewhere): TELEGRAM_CHAT_ID is still empty."
    info "  Send any message to the bot, then run: sudo bash $APP_DIR/deployment/install.sh --yes"
    info "  (or put the chat id into $ENV_FILE by hand: TELEGRAM_CHAT_ID=<number>)"
  fi
}

# =========================================================================================== [10/14]
step_initialise() {
  step 10 "Initialising the database (init, as $SERVICE_USER)"
  local out rc=0
  if ((DRY_RUN)); then
    info "would run: python -m app.cli init"
    return 0
  fi
  out=$(run_as_monster "$VENV_PY" -m app.cli init 2>&1) || rc=$?
  printf '%s\n' "$out" | redact_stream | indent
  if ((rc != 0)); then die "'init' failed (exit $rc): see the message above"; fi
  ok "directories, database schema and seed data are in place"
}

# =========================================================================================== [11/14]
install_cli_wrapper() {
  write_file "$CLI_WRAPPER" 0755 <<'EOF'
#!/usr/bin/env bash
# MONSTER AI CLI wrapper (generated by deployment/install.sh - do not edit).
# Runs `python -m app.cli ...` as the service user with the production environment, e.g.
#   sudo monster-ai health
set -Eeuo pipefail
if [[ $EUID -ne 0 ]]; then exec sudo "$0" "$@"; fi
cd /opt/monster-ai
umask 027
exec runuser -u monster -- env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  HOME=/var/lib/monster-ai LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  MONSTER_ENV=production MONSTER_HOME=/opt/monster-ai MONSTER_DATA_DIR=/var/lib/monster-ai \
  MONSTER_LOG_DIR=/var/log/monster-ai MONSTER_ENV_FILE=/etc/monster-ai/monster.env \
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/monster-ai/.venv/bin/python -m app.cli "$@"
EOF
}

install_system_files() {
  mkdir -p "$(dirname -- "$JOURNALD_DROPIN")"
  if install_if_changed "$APP_DIR/deployment/journald-monster.conf" "$JOURNALD_DROPIN" 0644; then
    systemctl restart systemd-journald || warn "could not restart systemd-journald; the limits apply after the next restart"
    ok "journald limits installed (SystemMaxUse=100M, RuntimeMaxUse=50M, MaxRetentionSec=2week)"
  else
    ok "journald drop-in is up to date"
  fi
  install -m 0644 -o root -g root -- "$APP_DIR/deployment/logrotate-monster" "$LOGROTATE_FILE"
  if have_cmd logrotate && ! logrotate -d "$LOGROTATE_FILE" >/dev/null 2>&1; then
    warn "logrotate rejected $LOGROTATE_FILE; check 'logrotate -d $LOGROTATE_FILE'"
  fi
  install_cli_wrapper
}

verify_unit_files() {
  local out unit
  have_cmd systemd-analyze || return 0
  for unit in monster-ai.service monster-ai-bootstrap.service monster-ai-health.service monster-ai-health.timer; do
    out=$(systemd-analyze verify "$UNIT_DIR/$unit" 2>&1 || true)
    if [[ -n $out ]]; then
      info "systemd-analyze verify $unit:"
      printf '%s\n' "$out" | sed -n '1,12p' | indent
    fi
  done
}

step_systemd() {
  step 11 "systemd units, journald and logrotate"
  local -a core_enable=(monster-ai.service monster-ai-health.timer)
  local units_msg="${CORE_UNITS[*]}" fail_code=4
  if ((HAD_PREV)); then fail_code=3; fi
  if ((DASHBOARD_WANTED)); then units_msg+=" $DASHBOARD_UNIT"; fi
  if ((DRY_RUN)); then
    info "would install $units_msg into $UNIT_DIR, the journald drop-in, logrotate and $CLI_WRAPPER; then systemctl enable --now ${core_enable[*]}"
    return 0
  fi
  install_units "$APP_DIR"
  install_system_files
  systemctl daemon-reload
  verify_unit_files
  systemctl reset-failed 'monster-ai*' 2>/dev/null || true
  if ((NO_START)); then
    systemctl enable "${core_enable[@]}"
    if ((DASHBOARD_WANTED)); then systemctl enable "$DASHBOARD_UNIT"; fi
    ok "units installed and enabled; nothing was started (--no-start)"
    mark_committed
    return 0
  fi
  info "starting the daemon (Type=notify: waits for READY=1) ..."
  if ! systemctl enable --now "${core_enable[@]}"; then
    show_unit_diagnostics monster-ai.service 40
    die "the daemon failed to start" "$fail_code"
  fi
  SERVICES_RESTARTED=1
  if ! wait_daemon_ready "$READY_WAIT_SEC"; then
    show_unit_diagnostics monster-ai.service 40
    die "monster-ai.service did not become (and stay) active within ${READY_WAIT_SEC}s" "$fail_code"
  fi
  mark_committed
  rm -rf -- "$APP_FAILED"
  ok "monster-ai.service is active and stable; the new release is committed"
  if ((DASHBOARD_WANTED)); then
    if systemctl enable --now "$DASHBOARD_UNIT"; then
      ok "dashboard started on loopback only (DASHBOARD_HOST stays 127.0.0.1)"
    else
      warn "the dashboard failed to start (it is optional; the daemon is unaffected): journalctl -u monster-ai-dashboard"
    fi
  fi
}

# =========================================================================================== [12/14]
wait_for_bootstrap() {
  local started now elapsed state unit last
  local max=$((BOOTSTRAP_MAX_MINUTES * 60)) interval=15
  started=$(date +%s)
  while true; do
    state=$(bootstrap_state)
    unit=$(svc_state monster-ai-bootstrap.service)
    now=$(date +%s)
    elapsed=$((now - started))
    last=$({ journalctl -u monster-ai-bootstrap.service -u monster-ai.service -n 1 --no-pager -o cat 2>/dev/null || true; } |
      redact_stream | cut -c 1-150)
    info "[bootstrap +$((elapsed / 60))m] state=$state unit=$unit | $last"
    if [[ $state == "done" ]]; then
      BOOTSTRAP_RESULT="done"
      return 0
    fi
    if [[ $state == failed ]] || { [[ $unit == failed ]] && [[ $state != running ]]; }; then
      BOOTSTRAP_RESULT=failed
      return 1
    fi
    if [[ $state == unknown || $state == none ]] && [[ $unit == active ]] && ((elapsed > 45)); then
      BOOTSTRAP_RESULT="done"
      return 0
    fi
    if ((elapsed >= max)); then
      BOOTSTRAP_RESULT=timeout
      return 2
    fi
    sleep "$interval"
  done
}

step_bootstrap() {
  step 12 "First-time bootstrap (history backfill and initial models)"
  local state unit rc=0
  if ((NO_BOOTSTRAP || NO_START)); then
    BOOTSTRAP_RESULT=skipped
    info "skipped. The daemon starts the bootstrap by itself when it never completed (or: sudo systemctl start monster-ai-bootstrap)."
    return 0
  fi
  if ((DRY_RUN)); then
    info "would run 'systemctl start --no-block monster-ai-bootstrap.service' and poll every 15 s for up to $BOOTSTRAP_MAX_MINUTES minutes"
    return 0
  fi
  state=$(bootstrap_state)
  if [[ $state == "done" ]]; then
    BOOTSTRAP_RESULT="done"
    ok "the bootstrap already completed on this host; nothing to do"
    return 0
  fi
  unit=$(svc_state monster-ai-bootstrap.service)
  if [[ $state == running ]]; then
    # The daemon launches the same resumable job by itself right after READY; a second instance would
    # only be refused by the job lock, so just watch the one that is running.
    info "the bootstrap is already running (started by the daemon); watching it"
  elif [[ $unit == activating ]]; then
    info "the bootstrap unit is already running; watching it"
  else
    systemctl reset-failed monster-ai-bootstrap.service 2>/dev/null || true
    # RemainAfterExit=yes keeps a finished run "active": stop it first so that start really runs it again.
    if [[ $unit == active ]]; then systemctl stop monster-ai-bootstrap.service || true; fi
    systemctl start --no-block monster-ai-bootstrap.service
    info "bootstrap started in the background (systemd keeps it running if this SSH session ends)"
  fi
  info "it takes about 10-60 minutes on a small server. Ctrl-C only stops watching - the bootstrap keeps running."
  wait_for_bootstrap || rc=$?
  case $rc in
    0) ok "bootstrap finished: universe, history, F&O data and initial models are in place" ;;
    1)
      warn "the bootstrap reported a failure. The last 40 log lines follow."
      show_unit_diagnostics monster-ai-bootstrap.service 40
      info "The daemon keeps retrying automatically: the bootstrap unit restarts itself every 5 minutes (6 times per 6 h) and the daemon relaunches the bootstrap whenever it never completed. Nothing for you to do unless it persists."
      ;;
    *)
      warn "the bootstrap is still running after $BOOTSTRAP_MAX_MINUTES minutes; it continues in the background (journalctl -u monster-ai-bootstrap -f)"
      ;;
  esac
}

# =========================================================================================== [13/14]
# Feed / service lines of `doctor --json` ({"ok": bool, "checks": [{"name", "status": ok|warn|fail, "detail"}]}).
readonly FEEDS_SUMMARY_PY='
import json, re, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(3)
checks = data.get("checks") if isinstance(data, dict) else None
if not isinstance(checks, list):
    sys.exit(4)
want = re.compile(r"network|telegram|clock|bootstrap|models", re.I)
rows = [c for c in checks if isinstance(c, dict) and want.search(str(c.get("name", "")))]
for c in rows:
    print("%-26s %-5s %s" % (str(c.get("name", "?"))[:26], str(c.get("status", "?")), str(c.get("detail", ""))[:64]))
sys.exit(0 if rows else 4)
'

step_verify() {
  step 13 "Verification"
  local out rc=0
  if ((DRY_RUN)); then
    info "would check monster-ai.service, run 'health --json', 'telegram-test' (when configured) and summarise 'doctor --json'"
    return 0
  fi
  if ((NO_START)); then
    info "skipped (--no-start)"
    return 0
  fi
  if ! wait_daemon_ready 60; then
    show_unit_diagnostics monster-ai.service 40
    die "monster-ai.service is not running (it was healthy right after the start). It restarts automatically; investigate with: journalctl -u monster-ai -n 100" 4
  fi
  ok "monster-ai.service is active"
  out=$(run_as_monster timeout 120 "$VENV_PY" -m app.cli health --json 2>&1) || rc=$?
  case $rc in
    0)
      HEALTH_NOTE="OK"
      ok "health: OK"
      ;;
    1)
      HEALTH_NOTE="WARNING"
      warn "health: WARNING (not fatal; typical while data is still being collected)"
      run_as_monster timeout 120 "$VENV_PY" -m app.cli health 2>&1 | redact_stream | indent || true
      ;;
    2)
      HEALTH_NOTE="CRITICAL"
      if [[ $BOOTSTRAP_RESULT == "done" ]]; then
        warn "health: CRITICAL even though the bootstrap finished - details below"
      else
        warn "health: CRITICAL, expected until the bootstrap has finished (state: $BOOTSTRAP_RESULT) - details below"
      fi
      run_as_monster timeout 120 "$VENV_PY" -m app.cli health 2>&1 | redact_stream | indent || true
      ;;
    *)
      HEALTH_NOTE="exit $rc"
      warn "health returned exit $rc: $(printf '%s' "$out" | redact_stream | sed -n 1p)"
      ;;
  esac
  if [[ $TELEGRAM_STATE == configured ]]; then
    if run_as_monster timeout 90 "$VENV_PY" -m app.cli telegram-test >/dev/null 2>&1; then
      ok "Telegram test message delivered"
    else
      warn "the Telegram test message could not be delivered: check the token/chat id in $ENV_FILE and that api.telegram.org is reachable"
    fi
  else
    info "Telegram is not fully configured, so no test message was sent"
  fi
  out=$(run_as_monster timeout 180 "$VENV_PY" -m app.cli doctor --json 2>/dev/null) || true
  info "data feeds and services (doctor --json; 'warn' on a network line = optional source unreachable -> degraded mode):"
  if ! printf '%s' "$out" | run_as_monster "$VENV_PY" -c "$FEEDS_SUMMARY_PY" 2>/dev/null | indent; then
    info "  (no feed details found in the doctor output; run 'sudo monster-ai doctor' to see them)"
  fi
}

# =========================================================================================== [14/14]
step_summary() {
  step 14 "Summary"
  local version
  if ((DRY_RUN)); then
    info "dry run finished: re-run without --dry-run to install"
    return 0
  fi
  version=$(run_as_monster "$VENV_PY" -m app.cli version 2>/dev/null | sed -n 1p) || version="unknown"
  _say ""
  _say "=============================================================================="
  _say " MONSTER AI is installed"
  _say "=============================================================================="
  _say " Version        : ${version:-unknown}   (Python ${PY_VER:-?}, $ARCH, $OS_PRETTY)"
  _say " Mode           : SHADOW - research output only; this system NEVER places orders"
  _say " Code           : $APP_DIR   (previous release: $APP_PREV)"
  _say " State / data   : $DATA_DIR"
  _say " Logs           : $LOG_DIR   and the journal"
  _say " Configuration  : $ENV_FILE   (root:$SERVICE_GROUP 0640 - the only place for secrets)"
  _say " Bootstrap      : $BOOTSTRAP_RESULT"
  _say " Health         : ${HEALTH_NOTE:-not checked}"
  _say " Telegram       : $TELEGRAM_STATE"
  if ((DASHBOARD_WANTED)); then
    _say " Dashboard      : enabled on 127.0.0.1 only -> ssh -L 8080:127.0.0.1:8080 <user>@<server>, then http://127.0.0.1:8080"
  fi
  if ((${#WARNINGS[@]})); then
    _say ""
    _say " ${#WARNINGS[@]} warning(s) during the installation:"
    local w
    for w in "${WARNINGS[@]}"; do _say "   - $w"; done
  fi
  _say ""
  _say " Useful commands"
  _say "   journalctl -u monster-ai -f                      follow the daemon"
  _say "   sudo monster-ai health                           health check (exit 0 ok / 1 warn / 2 critical)"
  _say "   cd $APP_DIR && sudo -u monster env MONSTER_ENV_FILE=$ENV_FILE $VENV_PY -m app.cli health   (same, no wrapper)"
  _say "   systemctl list-timers monster-ai-health.timer    the 5-minute health schedule"
  _say "   sudo bash $APP_DIR/deployment/install.sh --yes   upgrade / re-run safely at any time"
  _say "   sudo bash $APP_DIR/deployment/install.sh --rollback   go back to the previous release"
  _say "   sudo bash $APP_DIR/deployment/uninstall.sh       remove it (data and secrets are kept unless --purge)"
  _say ""
}

# ============================================================================================ rollback
main_rollback() {
  info "manual rollback: $APP_PREV becomes the live release again"
  if ((DRY_RUN)); then
    info "would swap $APP_DIR and $APP_PREV, restore the unit files of the older release and restart the services"
    return 0
  fi
  [[ -d $APP_PREV ]] || die "there is no previous release at $APP_PREV"
  [[ -d $APP_DIR ]] || die "$APP_DIR does not exist; run the installer instead"
  resolve_dashboard_wanted
  stop_services
  rm -rf -- "$APP_FAILED"
  mv -- "$APP_DIR" "$APP_FAILED"
  mv -- "$APP_PREV" "$APP_DIR"
  mv -- "$APP_FAILED" "$APP_PREV"
  install_units "$APP_DIR" 0
  systemctl daemon-reload
  systemctl reset-failed 'monster-ai*' 2>/dev/null || true
  SERVICES_STOPPED=1
  SERVICES_RESTARTED=0
  restore_services
  if wait_daemon_ready "$READY_WAIT_SEC"; then
    mark_committed
    ok "rolled back: monster-ai.service is active again ($APP_PREV now holds the release you rolled away from)"
  else
    show_unit_diagnostics monster-ai.service 40
    die "the restored release did not come up; see the diagnostics above" 4
  fi
}

main() {
  startup
  if [[ $ACTION == rollback ]]; then
    main_rollback
    return 0
  fi
  _say "MONSTER AI installer - $(date -u '+%Y-%m-%d %H:%M:%S UTC') - host $(hostname 2>/dev/null || echo unknown)"
  if ((!DRY_RUN)) && ! confirm "Install / upgrade MONSTER AI on this host now?"; then
    die "not confirmed. Re-run with --yes for a non-interactive install." 2
  fi
  step_preflight
  step_packages
  step_swap
  step_user_dirs
  step_release
  step_venv
  step_envfile
  step_validate_config
  step_telegram
  step_initialise
  step_systemd
  step_bootstrap
  step_verify
  step_summary
}

if [[ ${MONSTER_INSTALL_SOURCE_ONLY:-0} != 1 ]]; then main; fi
