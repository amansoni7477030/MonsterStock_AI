#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# MONSTER AI deployment - the env-file seed that leaves the workstation (used by deploy_remote.sh).
#
# Only the settings MONSTER AI defines - the keys assigned in .env.example - are uploaded. Everything else in
# the file stays on this machine: the broker's login values (KOTAK_MPIN, KOTAK_TOTP_SECRET, ...), the secrets of
# other projects, typos. A value is never printed; only the NAMES of the keys that were left out are.
#
# This runs on the workstation (Linux, macOS, Git Bash), not on the server: bash 3.2, awk, od and iconv are all it needs.
set -Eeuo pipefail

# envseed_filter SEED TEMPLATE OUT
#   OUT (mode 0600) receives the assignments of SEED whose key TEMPLATE defines, as clean UTF-8 / LF text (BOM,
#   CRLF and UTF-16 are handled; comments and every other line are dropped). The names of the keys that were left
#   out are printed once each, in file order. Returns 1 and leaves OUT empty when the template defines no setting
#   or SEED cannot be read: the caller must then not upload the original. OUT may be neither SEED nor TEMPLATE.
envseed_filter() {
  local seed=$1 template=$2 out=$3 text magic status=0
  text=$seed
  if [[ $out -ef $seed || $out -ef $template ]]; then
    printf 'envseed_filter: the output file must not be the seed or the template\n' >&2
    return 1
  fi
  (umask 077 && : >"$out") || return 1
  if [[ ! -r $seed ]]; then
    printf 'envseed_filter: cannot read %s\n' "$seed" >&2
    return 1
  fi
  if ! grep -q '^[A-Z][A-Z0-9_]*=' "$template" 2>/dev/null; then
    printf 'envseed_filter: %s defines no setting, so the seed cannot be filtered\n' "$template" >&2
    return 1
  fi
  magic=$(head -c 2 "$seed" | od -An -tx1 | tr -d ' \n')
  if [[ $magic == fffe || $magic == feff ]]; then
    text=$out.utf8
    (umask 077 && iconv -f UTF-16 -t UTF-8 "$seed" >"$text") || status=1
  fi
  if ((status == 0)); then
    (umask 077 && : >"$out.names") || status=1
  fi
  if ((status == 0)); then
    # The two paths travel in the environment (awk -v would process backslashes); LC_ALL=C makes every awk count bytes.
    ENVSEED_OUT=$out ENVSEED_NAMES=$out.names LC_ALL=C awk '
      {
        line = $0
        sub(/\r$/, "", line)
        if (FNR == 1 && substr(line, 1, 3) == "\357\273\277") line = substr(line, 4)
      }
      FNR == NR {
        if (match(line, /^[A-Z][A-Z0-9_]*=/)) known[substr(line, 1, RLENGTH - 1)] = 1
        next
      }
      {
        work = line
        sub(/^[ \t]+/, "", work)
        if (work ~ /^export[ \t]/) sub(/^export[ \t]+/, "", work)
        if (!match(work, /^[A-Z][A-Z0-9_]*[ \t]*=/)) next
        key = substr(work, 1, RLENGTH - 1)
        sub(/[ \t]+$/, "", key)
        if (key in known) {
          print line > ENVIRON["ENVSEED_OUT"]
        } else if (!(key in seen)) {
          seen[key] = 1
          print key > ENVIRON["ENVSEED_NAMES"]
        }
      }
    ' "$template" "$text" || status=1
  fi
  if ((status == 0)); then cat -- "$out.names" || status=1; fi
  if [[ $text != "$seed" ]]; then rm -f -- "$text"; fi
  rm -f -- "$out.names"
  if ((status != 0)); then
    : >"$out" || true
    printf 'envseed_filter: the env file could not be filtered\n' >&2
    return 1
  fi
}
