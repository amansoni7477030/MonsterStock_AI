#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# MONSTER AI deployment - system helpers: apt, services, swap, release staging, units, rollback.
#
# Sourced by install.sh and uninstall.sh. Needs root for everything that changes the system; the pure
# file helpers (stage_tree, assert_no_env_files, write_file) are also exercised by the unit tests.
# Requires common.sh, envfile.sh, decisions.sh (sourced below).
set -Eeuo pipefail

if [[ -n ${_MONSTER_SYSTEM_LOADED:-} ]]; then return 0; fi
_MONSTER_SYSTEM_LOADED=1
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=envfile.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/envfile.sh"
# shellcheck source=decisions.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/decisions.sh"

# Install-session state (the installer's EXIT trap reads these to decide about rollback / restart).
: "${NO_START:=0}"
: "${DASHBOARD_WANTED:=0}"
: "${SERVICES_STOPPED:=0}"
: "${SERVICES_RESTARTED:=0}"
: "${RELEASE_SWAPPED:=0}"
: "${RELEASE_COMMITTED:=0}"
: "${ROLLED_BACK:=0}"
WAS_ACTIVE_UNITS=()

readonly CORE_UNITS=(monster-ai.service monster-ai-bootstrap.service monster-ai-health.service
  monster-ai-health.timer 'monster-ai-alert@.service')
readonly DASHBOARD_UNIT=monster-ai-dashboard.service
readonly STOP_ORDER=(monster-ai-health.timer monster-ai-health.service monster-ai-dashboard.service
  monster-ai-bootstrap.service monster-ai.service)

# ------------------------------------------------------------------------------------------ small tools
# acquire_lock: one install/uninstall at a time. flock on fd 9; the kernel releases it when the shell exits.
acquire_lock() {
  mkdir -p -- "$(dirname -- "$LOCK_FILE")"
  { exec 9>"$LOCK_FILE"; } 2>/dev/null || die "cannot open the lock file $LOCK_FILE"
  if ! flock -n 9; then
    die "another MONSTER AI install/uninstall is already running (lock $LOCK_FILE). Wait for it to finish." 75
  fi
}

# write_file DEST MODE: write stdin to DEST atomically as root:root (the content may be generated text).
write_file() {
  local dest=$1 mode=$2 tmp
  if ((DRY_RUN)); then
    info "would write $dest"
    cat >/dev/null
    return 0
  fi
  mkdir -p -- "$(dirname -- "$dest")"
  tmp=$(mktemp "${dest}.XXXXXX")
  cat >"$tmp"
  chmod "$mode" "$tmp"
  chown root:root "$tmp"
  mv -f -- "$tmp" "$dest"
}

# install_if_changed SRC DEST MODE: copy only when the content differs; returns 0 when DEST changed.
install_if_changed() {
  local src=$1 dest=$2 mode=$3
  if [[ -f $dest ]] && cmp -s -- "$src" "$dest"; then return 1; fi
  install -D -m "$mode" -o root -g root -- "$src" "$dest"
  return 0
}

svc_state() { systemctl is-active "$1" 2>/dev/null || true; }
svc_active() { [[ $(svc_state "$1") == active ]]; }

# run_as_monster CMD ARG...: run in /opt/monster-ai as the service user with exactly the environment of the
# systemd units (never as root: root-owned database/log files would lock the service out).
run_as_monster() {
  (
    cd "$APP_DIR"
    umask 027
    runuser -u "$SERVICE_USER" -- env -i \
      PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
      HOME="$DATA_DIR" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      MONSTER_ENV=production MONSTER_HOME="$APP_DIR" MONSTER_DATA_DIR="$DATA_DIR" \
      MONSTER_LOG_DIR="$LOG_DIR" MONSTER_ENV_FILE="$ENV_FILE" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$@"
  )
}

# bootstrap_state: running | done | failed | none (never started) | unknown (database not readable).
# The key lives in the application's kv table; it is read as the service user so no root-owned
# -wal/-shm files can appear next to the database.
bootstrap_state() {
  local db out
  db=$(env_get MONSTER_DB_PATH "$ENV_FILE" 2>/dev/null || true)
  if [[ -z $db ]]; then db=$DATA_DIR/monster.db; fi
  if [[ ! -f $db ]]; then
    echo none
    return 0
  fi
  if ! out=$(run_as_monster sqlite3 -readonly -batch -noheader -cmd '.timeout 3000' "$db" \
    "SELECT value FROM kv WHERE key = 'bootstrap.state'" 2>/dev/null); then
    echo unknown
    return 0
  fi
  out=$(printf '%s' "$out" | tr -d '[:space:]')
  echo "${out:-none}"
}

# show_unit_diagnostics UNIT [LINES]: status and last journal lines (secrets masked).
show_unit_diagnostics() {
  local unit=$1 lines=${2:-40}
  info "systemctl status $unit:"
  { systemctl status --no-pager -l -n 0 "$unit" 2>&1 || true; } | redact_stream | indent
  info "last $lines journal lines of $unit:"
  { journalctl -u "$unit" -n "$lines" --no-pager -o short-iso 2>&1 || true; } | redact_stream | indent
}

# ----------------------------------------------------------------------------------------------- apt
export APT_LOCK_TIMEOUT=300
readonly APT_ARGS=(-y -q -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}"
  -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold -o APT::Install-Recommends=false)

pkg_installed() {
  [[ $(dpkg-query -W -f='${Status}' "$1" 2>/dev/null || true) == 'install ok installed' ]]
}

# The apt/dpkg lock files: a package manager that really works holds one of them open. Process names alone cannot tell: the resident
# "unattended-upgrade-shutdown" helper (comm "unattended-upgr") runs on every Ubuntu server and is no upgrade in progress, so a
# name check made the installer wait the full timeout on a stock image and fail.
readonly APT_LOCK_FILES=(/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock)

# apt_busy: true while a package manager holds the apt/dpkg locks (needs root to see every process).
apt_busy() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -s "${APT_LOCK_FILES[@]}" >/dev/null 2>&1
    return
  fi
  # No psmisc: fall back to the process names, with the unattended upgrade itself matched by its command line (never the helper).
  pgrep -x 'apt|apt-get|aptitude|dpkg|packagekitd|apt.systemd.dai' >/dev/null 2>&1 ||
    pgrep -f '(^|/)unattended-upgrade( |$)' >/dev/null 2>&1
}

# wait_for_apt [SECONDS]: a fresh VM often runs unattended-upgrades for minutes after first boot.
wait_for_apt() {
  local limit=${1:-300} waited=0
  while apt_busy; do
    if ((waited == 0)); then info "another package manager is running; waiting up to ${limit}s for it ..."; fi
    if ((waited >= limit)); then
      fatal "apt/dpkg is still busy after ${limit}s. Wait for it to finish (or reboot) and re-run the installer."
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

# repair_dpkg: finish a dpkg run that was interrupted (for example by a dropped SSH session).
repair_dpkg() {
  if [[ -n $(dpkg --audit 2>/dev/null || true) ]]; then
    info "dpkg was interrupted earlier; completing it (dpkg --configure -a)"
    dpkg --configure -a
  fi
}

apt_get() {
  wait_for_apt
  retry 3 15 apt-get "${APT_ARGS[@]}" "$@"
}

# ------------------------------------------------------------------------------------------------- NTP
ntp_active_services() {
  local svc
  for svc in chrony systemd-timesyncd ntp ntpsec openntpd; do
    if svc_active "$svc.service"; then printf '%s\n' "$svc"; fi
  done
}

ntp_synchronized() {
  [[ $(timedatectl show -p NTPSynchronized --value 2>/dev/null || true) == yes ]]
}

# ------------------------------------------------------------------------------------------------ swap
swap_file_active() { swapon --noheadings --show=NAME 2>/dev/null | grep -qx "$SWAPFILE"; }

ensure_fstab_entry() {
  if awk -v f="$SWAPFILE" '$1 == f && $3 == "swap" { found = 1 } END { exit !found }' /etc/fstab; then return 0; fi
  if [[ -s /etc/fstab && -n $(tail -c 1 /etc/fstab) ]]; then printf '\n' >>/etc/fstab; fi
  printf '%s none swap sw 0 0\n' "$SWAPFILE" >>/etc/fstab
}

_swap_fill_dd() {
  dd if=/dev/zero of="$SWAPFILE" bs=1M count="$1" status=none conv=fsync
}

# create_swapfile MB: fallocate where it is reliable, dd on btrfs/zfs/overlay or when the file system
# rejects fallocate / the resulting file ("swapfile has holes"). Returns 1 (and leaves nothing behind)
# when the file system cannot hold a swap file at all.
create_swapfile() {
  local mb=$1 fstype use_dd=0
  fstype=$(fs_type "$(dirname -- "$SWAPFILE")")
  case $fstype in
    tmpfs | ramfs) return 1 ;;
    btrfs | zfs | overlay*) use_dd=1 ;;
  esac
  swapoff "$SWAPFILE" 2>/dev/null || true
  rm -f -- "$SWAPFILE"
  if [[ $fstype == btrfs ]]; then
    : >"$SWAPFILE"
    chattr +C "$SWAPFILE" 2>/dev/null || true
  fi
  if ((use_dd)) || ! fallocate -l "${mb}M" "$SWAPFILE" 2>/dev/null; then
    _swap_fill_dd "$mb"
  fi
  chmod 600 "$SWAPFILE"
  mkswap -q "$SWAPFILE" >/dev/null
  if swapon "$SWAPFILE" 2>/dev/null; then return 0; fi
  if ((!use_dd)); then
    rm -f -- "$SWAPFILE"
    _swap_fill_dd "$mb"
    chmod 600 "$SWAPFILE"
    mkswap -q "$SWAPFILE" >/dev/null
    if swapon "$SWAPFILE" 2>/dev/null; then return 0; fi
  fi
  rm -f -- "$SWAPFILE"
  return 1
}

# -------------------------------------------------------------------------------------- release staging
# Everything that must never reach /opt/monster-ai: VCS data, caches, runtime state, tests, CI files and -
# above all - env files (secrets). Patterns starting with ./ only match at the top of the tree.
readonly STAGE_EXCLUDES=(
  --exclude=.git --exclude=.github --exclude=.venv --exclude=__pycache__ '--exclude=*.pyc'
  --exclude=./data/runtime --exclude=./data/cache --exclude=./logs --exclude=./tests
  '--exclude=*.db' '--exclude=*.db-wal' '--exclude=*.db-shm' '--exclude=*.log'
  --exclude=.env '--exclude=*.env' '--exclude=*.tar.gz' '--exclude=*.zip'
  --exclude=.ruff_cache --exclude=.pytest_cache --exclude=.mypy_cache --exclude=.coverage --exclude=htmlcov
  --exclude=.scratch
  # Installer state files: they only exist in a release that was built and started by this installer.
  --exclude=./.committed --exclude=./.requirements.sha256
)

# stage_tree SRC DEST: copy the release tree (see STAGE_EXCLUDES); ownership is not preserved.
stage_tree() {
  local src=$1 dest=$2
  mkdir -p -- "$dest"
  tar -C "$src" "${STAGE_EXCLUDES[@]}" -cf - . | tar -C "$dest" --no-same-owner -xf -
}

# assert_no_env_files DIR: fail (listing the offenders) if an env file slipped into the staged tree.
assert_no_env_files() {
  local found
  found=$(find "$1" \( -name '.env' -o -name '*.env' \) -print)
  if [[ -n $found ]]; then
    err "env files must never be part of a release (secrets):"
    printf '%s\n' "$found" | indent
    return 1
  fi
}

# mark_committed: the live release started successfully. Only a committed release may replace .prev, so a
# killed installer can never push the last known-good release out of the rollback slot.
mark_committed() {
  RELEASE_COMMITTED=1
  : >"$COMMITTED_FILE" 2>/dev/null || true
}

# --------------------------------------------------------------------------------------------- services
# wait_units_stopped SECONDS UNIT...: true once none of the units is active, activating or deactivating.
wait_units_stopped() {
  local limit=$1 waited=0 busy unit state
  shift
  while ((waited < limit)); do
    busy=0
    for unit in "$@"; do
      state=$(svc_state "$unit")
      if [[ $state != inactive && $state != failed ]]; then busy=1; fi
    done
    if ((!busy)); then return 0; fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

# stop_services: gracefully stop every MONSTER AI unit that is running (daemon: wait up to 60 s, then
# SIGKILL) and remember which ones were active so restore_services can bring exactly those back.
stop_services() {
  local unit state
  WAS_ACTIVE_UNITS=()
  for unit in "${STOP_ORDER[@]}"; do
    state=$(svc_state "$unit")
    if [[ $state == active || $state == activating || $state == reloading ]]; then WAS_ACTIVE_UNITS+=("$unit"); fi
  done
  if ((${#WAS_ACTIVE_UNITS[@]} == 0)); then
    info "no MONSTER AI service is running"
    return 0
  fi
  info "stopping: ${WAS_ACTIVE_UNITS[*]}"
  SERVICES_STOPPED=1
  SERVICES_RESTARTED=0
  systemctl stop --no-block "${WAS_ACTIVE_UNITS[@]}" || true
  if wait_units_stopped 60 "${WAS_ACTIVE_UNITS[@]}"; then return 0; fi
  warn "the daemon did not stop within 60 s; killing it"
  systemctl kill --kill-whom=all --signal=SIGKILL monster-ai.service 2>/dev/null || true
  wait_units_stopped 15 "${WAS_ACTIVE_UNITS[@]}" || true
}

# restore_services: start again what stop_services stopped (also the safety net of the EXIT trap).
restore_services() {
  local unit
  if ((!SERVICES_STOPPED)) || ((SERVICES_RESTARTED)) || ((NO_START)); then return 0; fi
  info "restarting the services that were stopped for the upgrade"
  systemctl daemon-reload || true
  systemctl reset-failed 'monster-ai*' 2>/dev/null || true
  for unit in monster-ai.service monster-ai-health.timer "$DASHBOARD_UNIT" monster-ai-bootstrap.service; do
    if env_in_list "$unit" "${WAS_ACTIVE_UNITS[@]}"; then systemctl start --no-block "$unit" || true; fi
  done
  SERVICES_RESTARTED=1
}

# install_units TREE [PRUNE_DASHBOARD]: copy the unit files of a release tree into /etc/systemd/system.
# The dashboard unit is installed only when the dashboard is wanted; with PRUNE_DASHBOARD=1 (default) a
# previously installed dashboard unit is removed when it is not.
install_units() {
  local tree=$1 prune=${2:-1} unit src
  for unit in "${CORE_UNITS[@]}"; do
    src=$tree/deployment/systemd/$unit
    [[ -f $src ]] || die "release is incomplete: $src is missing"
    install -m 0644 -o root -g root -- "$src" "$UNIT_DIR/$unit"
  done
  if ((DASHBOARD_WANTED)) && [[ -f $tree/deployment/systemd/$DASHBOARD_UNIT ]]; then
    install -m 0644 -o root -g root -- "$tree/deployment/systemd/$DASHBOARD_UNIT" "$UNIT_DIR/$DASHBOARD_UNIT"
  elif ((prune)) && ! ((DASHBOARD_WANTED)) && [[ -f $UNIT_DIR/$DASHBOARD_UNIT ]]; then
    systemctl disable --now "$DASHBOARD_UNIT" 2>/dev/null || true
    rm -f -- "$UNIT_DIR/$DASHBOARD_UNIT"
    info "dashboard disabled: removed $DASHBOARD_UNIT"
  fi
}

# wait_daemon_ready [SECONDS]: true when the daemon reached READY and then stays up (same MainPID, no
# restart) for 15 more seconds - a crash loop must not pass as "active".
wait_daemon_ready() {
  local limit=${1:-120} waited=0 state="" pid1 pid2 restarts1 restarts2
  while ((waited < limit)); do
    state=$(svc_state monster-ai.service)
    if [[ $state == active ]]; then break; fi
    if [[ $state == failed ]]; then return 1; fi
    sleep 2
    waited=$((waited + 2))
  done
  if [[ $state != active ]]; then return 1; fi
  pid1=$(systemctl show -p MainPID --value monster-ai.service 2>/dev/null || true)
  restarts1=$(systemctl show -p NRestarts --value monster-ai.service 2>/dev/null || true)
  sleep 15
  state=$(svc_state monster-ai.service)
  pid2=$(systemctl show -p MainPID --value monster-ai.service 2>/dev/null || true)
  restarts2=$(systemctl show -p NRestarts --value monster-ai.service 2>/dev/null || true)
  [[ $state == active && $pid1 == "$pid2" && ${restarts1:-0} == "${restarts2:-0}" ]]
}

# rollback_release REASON: put /opt/monster-ai.prev back as the live release, restore its unit files and
# restart what was running. The failed tree is kept (without its venv) as /opt/monster-ai.failed.
rollback_release() {
  local reason=$1
  if [[ ! -d $APP_PREV ]]; then
    warn "no previous release at $APP_PREV: nothing to roll back to"
    return 1
  fi
  warn "ROLLING BACK to the previous release ($reason)"
  systemctl stop --no-block "${STOP_ORDER[@]}" 2>/dev/null || true
  if ! wait_units_stopped 60 "${STOP_ORDER[@]}"; then
    systemctl kill --kill-whom=all --signal=SIGKILL monster-ai.service 2>/dev/null || true
    wait_units_stopped 15 "${STOP_ORDER[@]}" || true
  fi
  rm -rf -- "$APP_FAILED"
  if [[ -d $APP_DIR ]]; then mv -- "$APP_DIR" "$APP_FAILED"; fi
  mv -- "$APP_PREV" "$APP_DIR"
  rm -rf -- "$APP_FAILED/.venv"
  install_units "$APP_DIR" 0
  systemctl daemon-reload || true
  ROLLED_BACK=1
  SERVICES_STOPPED=1
  SERVICES_RESTARTED=0
  restore_services
  ok "previous release restored; the failed one was kept (without its venv) in $APP_FAILED"
}
