#!/usr/bin/env bash
# MONSTER AI deployment - decision helpers. They only read the system (or the files you pass in) and print
# a verdict, so the trickiest rules are testable without root: swap sizing, requirements hash / venv
# reuse, the report-window guard, NTP plan, interpreter and OS checks.
#
# Requires common.sh (sourced below).
set -Eeuo pipefail

if [[ -n ${_MONSTER_DECISIONS_LOADED:-} ]]; then return 0; fi
_MONSTER_DECISIONS_LOADED=1
# shellcheck source-path=SCRIPTDIR
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# ------------------------------------------------------------------------------------ measurements
# meminfo_mb FIELD [FILE]: MB (rounded) of a /proc/meminfo field (MemTotal, SwapTotal, ...); 0 when absent.
# Rounded, not truncated: a 512 MB swap file reports SwapTotal 524284 kB, which must still count as 512.
meminfo_mb() {
  local value
  value=$(awk -v f="$1:" '$1 == f { printf "%d", $2 / 1024 + 0.5; found = 1 } END { if (!found) printf "0" }' \
    "${2:-/proc/meminfo}")
  printf '%s' "$value"
}

# free_disk_mb PATH: free MB of the file system holding PATH (nearest existing parent when PATH is new).
free_disk_mb() {
  local p=$1
  while [[ ! -e $p && $p != / && $p != . ]]; do p=$(dirname -- "$p"); done
  df -Pm -- "$p" | awk 'NR == 2 { print $4 }'
}

# fs_type PATH: file system type name (ext2/ext3 = ext4, xfs, btrfs, zfs, overlayfs, tmpfs ...).
fs_type() {
  local p=$1
  while [[ ! -e $p && $p != / && $p != . ]]; do p=$(dirname -- "$p"); done
  stat -f -c %T -- "$p" 2>/dev/null || echo unknown
}

# os_field FIELD [FILE]: value of a KEY=value line in /etc/os-release without sourcing the file.
os_field() {
  sed -n "s/^$1=//p" "${2:-/etc/os-release}" | head -n 1 | tr -d "\"'"
}

# os_supported ID VERSION_ID: the tested platforms.
os_supported() {
  case "$1:$2" in
    ubuntu:22.04 | ubuntu:24.04 | debian:12) return 0 ;;
    *) return 1 ;;
  esac
}

# arch_supported UNAME_M: architectures with binary wheels for the pinned requirements.
arch_supported() {
  [[ $1 == x86_64 || $1 == aarch64 ]]
}

# py_version_supported MAJOR.MINOR: the pinned wheels cover CPython 3.10 - 3.13.
py_version_supported() {
  version_ge "$1" 3.10 && ! version_ge "$1" 3.14
}

# ------------------------------------------------------------------------------------ CLI output parsing
# doctor_verdict EXIT_CODE REPORT_TEXT: ok | warn | fail. The CLI contract says doctor exits 0/1/2 for
# ok/warnings/critical, the current implementation exits 1 when a check FAILED and 0 otherwise (warnings
# included). The "[FAIL]" / "[warn]" markers of the report settle both readings.
doctor_verdict() {
  local rc=$1 report=$2
  if ((rc >= 2)) || [[ $report == *'[FAIL]'* ]]; then
    echo fail
  elif ((rc == 1)) || [[ $report == *'[warn]'* ]]; then
    echo warn
  else
    echo ok
  fi
}

# parse_chat_id TEXT: the Telegram chat id printed by `telegram-setup --print-only`. It is the last stdout
# line, either the bare number (contract) or `TELEGRAM_CHAT_ID=<number>` (current CLI). Empty when invalid.
parse_chat_id() {
  local last
  last=$(printf '%s\n' "$1" | sed -n '$p' | tr -d '[:space:]')
  last=${last#TELEGRAM_CHAT_ID=}
  if [[ $last =~ ^-?[0-9]+$ ]]; then printf '%s' "$last"; fi
}

# ------------------------------------------------------------------------------------------------ swap
# swap_target_mb TOTAL_SWAP_MB FREE_DISK_MB [SKIP]: size of the swap file to create (0 = none).
# Rules: nothing when skipped or when >= 512 MB swap already exists; 1024 MB, or 512 MB when less than
# 3.5 GB of disk would remain; nothing at all when even 512 MB would leave less than 2 GB.
swap_target_mb() {
  local swap=$1 free=$2 skip=${3:-0} size=1024
  if ((skip)) || ((swap >= 512)); then
    echo 0
    return 0
  fi
  if ((free - size < 3584)); then size=512; fi
  if ((free - size < 2048)); then size=0; fi
  echo "$size"
}

# --------------------------------------------------------------------------------- requirements / venv
# requirements_key PY_TAG ARCH FILE...: sha256 over the requirement files (CR stripped, so a CRLF checkout
# hashes like an LF one), the interpreter tag and the CPU architecture. A venv is only reusable when all
# three are unchanged.
requirements_key() {
  local tag=$1 arch=$2 f
  shift 2
  {
    for f in "$@"; do
      tr -d '\r' <"$f"
      printf '\n-- end of %s --\n' "${f##*/}"
    done
    printf 'python=%s\narch=%s\n' "$tag" "$arch"
  } | sha256sum | awk '{ print $1 }'
}

# venv_can_reuse KEY HASH_FILE VENV_DIR: true when HASH_FILE holds KEY and the venv interpreter still runs
# (a distro upgrade that replaced the system Python silently breaks old venvs).
venv_can_reuse() {
  local key=$1 hashfile=$2 venv=$3 stored
  [[ -r $hashfile && -x $venv/bin/python ]] || return 1
  stored=$(tr -d '[:space:]' <"$hashfile")
  [[ -n $stored && $stored == "$key" ]] || return 1
  "$venv/bin/python" -c 'import sys' >/dev/null 2>&1
}

# ------------------------------------------------------------------------------------- report window
# in_report_window DOW HHMM: true on weekdays (1-5) from 09:00 up to 09:16 IST - the time-critical
# pre-open / report window in which the daemon must not be restarted. HHMM like 0912.
in_report_window() {
  local dow=$1 hhmm=$2
  ((dow >= 1 && dow <= 5)) || return 1
  ((10#$hhmm >= 900 && 10#$hhmm < 916))
}

# ist_dow_hhmm [EPOCH]: "DOW HHMM" in IST (UTC+05:30, no DST), computed without a tz database.
ist_dow_hhmm() {
  local now=${1:-}
  if [[ -z $now ]]; then now=$(date -u +%s); fi
  date -u -d "@$((now + 19800))" '+%u %H%M'
}

# seconds_until_window_end DOW HHMM: how long to wait until 09:16 IST (0 outside the window).
seconds_until_window_end() {
  local dow=$1 hhmm=$2 now_min end_min
  if ! in_report_window "$dow" "$hhmm"; then
    echo 0
    return 0
  fi
  now_min=$((10#${hhmm:0:2} * 60 + 10#${hhmm:2:2}))
  end_min=$((9 * 60 + 16))
  echo $(((end_min - now_min) * 60))
}

# ------------------------------------------------------------------------------------------------ NTP
# ntp_plan ACTIVE_SERVICE...: exactly one time-sync service must run. Prints:
#   install-chrony                            none active
#   keep:<service>                            one active
#   keep:chrony+disable:systemd-timesyncd     both active (chrony wins)
#   keep:<first>+warn:multiple                several others active
ntp_plan() {
  local svc has_chrony=0 has_timesyncd=0
  if (($# == 0)); then
    echo "install-chrony"
    return 0
  fi
  if (($# == 1)); then
    echo "keep:$1"
    return 0
  fi
  for svc in "$@"; do
    if [[ $svc == chrony ]]; then has_chrony=1; fi
    if [[ $svc == systemd-timesyncd ]]; then has_timesyncd=1; fi
  done
  if ((has_chrony && has_timesyncd && $# == 2)); then
    echo "keep:chrony+disable:systemd-timesyncd"
  else
    echo "keep:$1+warn:multiple"
  fi
}
