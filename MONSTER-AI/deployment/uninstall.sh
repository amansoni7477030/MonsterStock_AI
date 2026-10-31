#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# =====================================================================================================
# MONSTER AI - uninstaller
#
#   sudo bash deployment/uninstall.sh [--purge] [--purge-swap] [--yes] [--dry-run]
#
# Default: stops and disables the services, removes the systemd units, the journald / logrotate / sysctl
# drop-ins, the `monster-ai` command and the code (/opt/monster-ai and its .prev/.new/.failed siblings).
# It KEEPS /var/lib/monster-ai (database, models, backups), /var/log/monster-ai and /etc/monster-ai
# (configuration and secrets), so a later install picks up exactly where this one stopped.
# =====================================================================================================
set -Eeuo pipefail

export LC_ALL=C.UTF-8 LANG=C.UTF-8

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
SCRIPT_DIR=${SCRIPT_PATH%/*}
ORIG_ARGS=("$@")
PURGE=0
PURGE_SWAP=0
DRY_RUN=0
ASSUME_YES=0

early_die() {
  printf '[FAIL] %s\n' "$1" >&2
  exit "${2:-1}"
}

usage() {
  cat <<'EOF'
MONSTER AI uninstaller

Usage: sudo bash deployment/uninstall.sh [options]

Stops and removes the services, units, drop-ins and code. KEEPS /var/lib/monster-ai (data),
/var/log/monster-ai and /etc/monster-ai (configuration and secrets) unless --purge is given.

Options:
  --purge       ALSO delete the data, the logs, the configuration (secrets!) and the 'monster' user.
                Asks for confirmation unless --yes is given.
  --purge-swap  ALSO remove /swapfile (swapoff, /etc/fstab entry, file). Never done otherwise.
  --yes, -y     do not ask
  --dry-run     only print what would be removed (no root needed)
  -h, --help    this text
EOF
}

parse_args() {
  while (($#)); do
    case $1 in
      --purge) PURGE=1 ;;
      --purge-swap) PURGE_SWAP=1 ;;
      --yes | -y) ASSUME_YES=1 ;;
      --dry-run) DRY_RUN=1 ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        printf '[FAIL] unknown option: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done
}

reexec_with_sudo() {
  if ((EUID == 0)) || ((DRY_RUN)); then return 0; fi
  command -v sudo >/dev/null 2>&1 || early_die "this script must run as root and sudo is not installed - log in as root and run it again"
  printf 'Not running as root: re-running with sudo ...\n' >&2
  exec sudo bash "$SCRIPT_PATH" "${ORIG_ARGS[@]}"
}

parse_args "$@"
reexec_with_sudo
[[ -r $SCRIPT_DIR/lib/common.sh ]] || early_die "deployment/lib/ is missing next to uninstall.sh"
if ((EUID == 0)); then
  for lib in "$SCRIPT_DIR"/lib/*.sh; do
    if [[ -f $lib ]] && grep -q $'\r' "$lib"; then sed -i 's/\r$//' "$lib"; fi
  done
fi
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=lib/system.sh
source "$SCRIPT_DIR/lib/system.sh"

REMOVED=()
KEPT=()

heading() {
  _say ""
  _say "$1"
}

# remove_path PATH...: delete files/directories that exist and remember them for the summary.
remove_path() {
  local p
  for p in "$@"; do
    if [[ -e $p || -L $p ]]; then
      run rm -rf -- "$p"
      REMOVED+=("$p")
    fi
  done
}

stop_and_disable() {
  local unit present=0
  for unit in "${STOP_ORDER[@]}"; do
    if [[ -f $UNIT_DIR/$unit ]] || [[ $(svc_state "$unit") != inactive ]]; then present=1; fi
  done
  if ((!present)); then
    info "no MONSTER AI service is installed or running"
    return 0
  fi
  info "stopping and disabling the services"
  run systemctl stop "${STOP_ORDER[@]}" 'monster-ai-alert@*' 2>/dev/null || true
  if ((!DRY_RUN)) && ! wait_units_stopped 60 "${STOP_ORDER[@]}"; then
    warn "the daemon did not stop within 60 s; killing it"
    systemctl kill --kill-whom=all --signal=SIGKILL monster-ai.service 2>/dev/null || true
    wait_units_stopped 15 "${STOP_ORDER[@]}" || true
  fi
  run systemctl disable monster-ai.service monster-ai-health.timer "$DASHBOARD_UNIT" 2>/dev/null || true
}

remove_units() {
  local unit
  for unit in "${CORE_UNITS[@]}" "$DASHBOARD_UNIT"; do remove_path "$UNIT_DIR/$unit"; done
  remove_path "$UNIT_DIR"/multi-user.target.wants/monster-ai*.service "$UNIT_DIR"/timers.target.wants/monster-ai*.timer
  run systemctl daemon-reload || true
  run systemctl reset-failed 'monster-ai*' 2>/dev/null || true
}

remove_system_files() {
  local had_journald=0
  if [[ -e $JOURNALD_DROPIN ]]; then had_journald=1; fi
  remove_path "$JOURNALD_DROPIN" "$LOGROTATE_FILE" "$SYSCTL_FILE" "$CLI_WRAPPER"
  if ((had_journald)); then run systemctl restart systemd-journald || warn "could not restart systemd-journald"; fi
}

remove_swap() {
  if [[ ! -e $SWAPFILE ]] && ! grep -q "^${SWAPFILE}[[:space:]]" /etc/fstab 2>/dev/null; then
    info "no $SWAPFILE to remove"
    return 0
  fi
  run swapoff "$SWAPFILE" 2>/dev/null || true
  if ((!DRY_RUN)) && [[ -f /etc/fstab ]]; then sed -i "\\|^${SWAPFILE}[[:space:]]|d" /etc/fstab; fi
  remove_path "$SWAPFILE"
}

purge_data() {
  remove_path "$DATA_DIR" "$LOG_DIR" "$CONF_DIR" "$INSTALL_LOG"
  if id -u "$SERVICE_USER" >/dev/null 2>&1; then
    run userdel "$SERVICE_USER" || warn "could not delete the user $SERVICE_USER"
    REMOVED+=("user $SERVICE_USER")
  fi
  if getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
    run groupdel "$SERVICE_GROUP" 2>/dev/null || true
  fi
}

main() {
  local keep title="MONSTER AI uninstaller" verb="Removed"
  if ((DRY_RUN)); then
    title="$title (dry run: nothing is changed)"
    verb="Would remove"
  else
    ((EUID == 0)) || die "must run as root"
    acquire_lock
  fi
  _say "$title"
  if ((PURGE)); then
    _say ""
    _say "  --purge will PERMANENTLY DELETE:"
    _say "    $DATA_DIR   (database, models, backups)"
    _say "    $LOG_DIR"
    _say "    $CONF_DIR   (configuration AND SECRETS such as the Telegram token)"
    _say "    and the '$SERVICE_USER' user. There is no undo."
    if ! confirm "Delete all of that?"; then die "aborted: nothing was changed" 2; fi
  elif ! confirm "Remove the MONSTER AI services and code (data, logs and configuration are kept)?"; then
    die "aborted: nothing was changed (use --yes for a non-interactive run)" 2
  fi
  heading "[1/4] Services"
  stop_and_disable
  heading "[2/4] systemd units and system drop-ins"
  remove_units
  remove_system_files
  heading "[3/4] Code"
  remove_path "$APP_DIR" "$APP_PREV" "$APP_NEW" "$APP_FAILED"
  heading "[4/4] Data, configuration and swap"
  if ((PURGE)); then
    purge_data
  else
    for keep in "$DATA_DIR" "$LOG_DIR" "$CONF_DIR"; do
      if [[ -e $keep ]]; then KEPT+=("$keep"); fi
    done
    if ((${#KEPT[@]})); then info "kept: ${KEPT[*]} (use --purge to delete them)"; fi
  fi
  if ((PURGE_SWAP)); then
    remove_swap
  elif [[ -e $SWAPFILE ]]; then
    info "kept $SWAPFILE (it is harmless; --purge-swap removes it)"
  fi
  _say ""
  if ((${#REMOVED[@]})); then
    _say "$verb:"
    printf '  - %s\n' "${REMOVED[@]}"
  else
    _say "Nothing was installed - nothing to remove."
  fi
  _say "Kernel settings from $SYSCTL_FILE (if they were applied) stay active until the next reboot."
  _say "Reinstall at any time with: sudo bash deployment/install.sh --yes"
}

main
