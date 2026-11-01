#!/usr/bin/env bash
# shellcheck source-path=SCRIPTDIR
# MONSTER AI deployment - env file handling: parse, encode, atomic update, merge, import.
#
# /etc/monster-ai/monster.env is read by TWO parsers that must agree: systemd's EnvironmentFile= and the
# application's own dotenv reader (app/config.py). Values are therefore written either raw or wrapped in
# one kind of quotes, never with escape sequences, and every function that touches values runs in a
# subshell with xtrace off so `MONSTER_INSTALL_DEBUG=1` can never print a secret.
#
# Files are parsed once into associative arrays (no subprocess per key). Requires common.sh (sourced
# below) and bash >= 4.3. No function in this file ever prints a value.
set -Eeuo pipefail

if [[ -n ${_MONSTER_ENVFILE_LOADED:-} ]]; then return 0; fi
_MONSTER_ENVFILE_LOADED=1
# shellcheck source=common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

# Secret keys: must equal the entries of app.config.VARS with secret=True (a test enforces this).
ENV_SECRET_KEYS=(TELEGRAM_BOT_TOKEN LLM_API_KEY MODEL_SIGNING_KEY BACKUP_PASSPHRASE
  S3_ACCESS_KEY_ID S3_SECRET_ACCESS_KEY DASHBOARD_TOKEN)

# Keys that describe one machine's file system or runtime: never imported from another machine's file.
ENV_MACHINE_LOCAL_KEYS=(MONSTER_ENV MONSTER_HOME MONSTER_DATA_DIR MONSTER_LOG_DIR MONSTER_DB_PATH
  MONSTER_MODELS_DIR MONSTER_CACHE_DIR MONSTER_BACKUP_DIR NEWS_FEEDS_FILE)

# env_is_key NAME: the application only accepts [A-Z][A-Z0-9_]*.
env_is_key() { [[ $1 =~ ^[A-Z][A-Z0-9_]*$ ]]; }

# env_in_list NEEDLE ITEM...
env_in_list() {
  local needle=$1 item
  shift
  for item in "$@"; do
    if [[ $item == "$needle" ]]; then return 0; fi
  done
  return 1
}

# env_is_true VALUE: same truthy words as the application (1 true yes on y).
env_is_true() {
  local v=${1,,}
  [[ $v == 1 || $v == true || $v == yes || $v == on || $v == y ]]
}

# env_parse_value_to VAR RHS: the logical value of the text right of KEY= stored in VAR - exactly what
# app/config.py computes (trim; one pair of matching quotes is removed; for unquoted values ` #...` is a
# comment). No subshell.
env_parse_value_to() {
  local -n _parsed=$1
  local v=$2 first last
  v=${v#"${v%%[![:space:]]*}"}
  v=${v%"${v##*[![:space:]]}"}
  if ((${#v} >= 2)); then
    first=${v:0:1}
    last=${v: -1}
    if [[ $first == "$last" && ($first == '"' || $first == "'") ]]; then
      _parsed=${v:1:${#v}-2}
      return 0
    fi
  fi
  if [[ $v == *' #'* ]]; then
    v=${v%%' #'*}
    v=${v%"${v##*[![:space:]]}"}
  fi
  _parsed=$v
}

# env_parse_value RHS: printing form of env_parse_value_to.
env_parse_value() {
  local _printed
  env_parse_value_to _printed "$1"
  printf '%s' "$_printed"
}

# env_encode_value_to VAR VALUE: VAR gets the text to store right of KEY= so that systemd and the application
# both read VALUE back unchanged: raw when "plain", else single quotes, else double quotes. Returns 1 for
# values that cannot be stored safely (newline, or an unquotable mix of quotes and backslashes).
env_encode_value_to() {
  local -n _encoded=$1
  local v=$2
  case $v in
    *$'\n'* | *$'\r'*) return 1 ;;
  esac
  if [[ -z $v ]]; then
    _encoded=""
    return 0
  fi
  if [[ $v != *[[:space:]\'\"\\]* ]]; then
    _encoded=$v
    return 0
  fi
  if [[ $v != *\'* && $v != *\\ ]]; then
    _encoded="'$v'"
    return 0
  fi
  if [[ $v != *\"* && $v != *\\* ]]; then
    _encoded="\"$v\""
    return 0
  fi
  return 1
}

# env_encode_value VALUE: printing form of env_encode_value_to (exit 1 when unsupported).
env_encode_value() {
  local _printed
  env_encode_value_to _printed "$1" || return 1
  printf '%s' "$_printed"
}

# _env_assignment LINE KEYVAR RHSVAR: split a dotenv line; true for `KEY=value` / `export KEY=value`.
# (KEYVAR / RHSVAR are variable NAMES of the caller; the locals here use names callers never use, because a
# nameref to a name that is shadowed by a local would silently assign to the local.)
_env_assignment() {
  local -n _out_key=$2 _out_rhs=$3
  local _line=${1%$'\r'} _name
  _line=${_line#"${_line%%[![:space:]]*}"}
  if [[ -z $_line || $_line == \#* ]]; then return 1; fi
  if [[ $_line == export[[:space:]]* ]]; then
    _line=${_line#export}
    _line=${_line#"${_line%%[![:space:]]*}"}
  fi
  if [[ $_line != *=* ]]; then return 1; fi
  _name=${_line%%=*}
  _name=${_name%"${_name##*[![:space:]]}"}
  _out_key=$_name
  _out_rhs=${_line#*=}
}

# _env_read_into ARRAY FILE: fill the associative array ARRAY (key -> parsed value; the last assignment wins).
_env_read_into() {
  local -n _map=$1
  local line k rhs value
  [[ -r $2 ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    if _env_assignment "$line" k rhs && env_is_key "$k"; then
      env_parse_value_to value "$rhs"
      _map["$k"]=$value
    fi
  done <"$2"
}

# _env_keys_into ARRAY FILE: fill the indexed array ARRAY with every valid key assigned in FILE, once, in order.
_env_keys_into() {
  local -n _keys=$1
  local -A _seen=()
  local line k rhs
  _keys=()
  [[ -r $2 ]] || return 0
  while IFS= read -r line || [[ -n $line ]]; do
    if _env_assignment "$line" k rhs && env_is_key "$k" && [[ -z ${_seen[$k]+x} ]]; then
      _seen[$k]=1
      _keys+=("$k")
    fi
  done <"$2"
}

# env_get KEY FILE: prints the effective value (last assignment wins). Exit 1 when KEY is absent.
env_get() (
  set +x
  local -A map=()
  _env_read_into map "$2"
  if [[ -n ${map[$1]+x} ]]; then
    printf '%s' "${map[$1]}"
    return 0
  fi
  return 1
)

# env_has_key KEY FILE: true when KEY is assigned (even to an empty value).
env_has_key() { env_get "$1" "$2" >/dev/null; }

# env_file_keys FILE: every valid key assigned in FILE, once, in file order.
env_file_keys() (
  set +x
  local -a keys=()
  _env_keys_into keys "$1"
  if ((${#keys[@]})); then printf '%s\n' "${keys[@]}"; fi
)

# set_env_var KEY VALUE FILE: atomic, injection-free update. The value is only ever handled by printf and
# bash expansion (no sed/awk replacement text), so / & | $ quotes and backslashes are safe. A duplicate
# assignment is collapsed into one; the file keeps its owner and mode (new files are 0600).
set_env_var() (
  set +x
  local key=$1 value=$2 file=$3 rhs tmp line found=0
  if ! env_is_key "$key"; then
    err "set_env_var: invalid key '$key'"
    return 2
  fi
  if ! env_encode_value_to rhs "$value"; then
    err "set_env_var: the value for $key has characters that cannot be stored safely; edit $file manually"
    return 3
  fi
  if ! tmp=$(mktemp "${file}.XXXXXX"); then
    err "set_env_var: cannot create a temporary file next to $file"
    return 1
  fi
  trap 'rm -f -- "$tmp"' EXIT
  {
    if [[ -f $file ]]; then
      while IFS= read -r line || [[ -n $line ]]; do
        line=${line%$'\r'}
        if [[ $line =~ ^[[:space:]]*(export[[:space:]]+)?"$key"[[:space:]]*= ]]; then
          if ((!found)); then
            printf '%s=%s\n' "$key" "$rhs"
            found=1
          fi
        else
          printf '%s\n' "$line"
        fi
      done <"$file"
    fi
    if ((!found)); then printf '%s=%s\n' "$key" "$rhs"; fi
  } >"$tmp"
  if [[ -e $file ]]; then
    chmod --reference="$file" "$tmp" 2>/dev/null || true
    chown --reference="$file" "$tmp" 2>/dev/null || true
  else
    chmod 600 "$tmp"
  fi
  mv -f -- "$tmp" "$file"
  trap - EXIT
)

# env_sanitize_seed SRC DEST: copy SRC as clean UTF-8 / LF text (Windows editors write BOMs, CRLF and
# sometimes UTF-16). DEST is created 0600 because it can contain secrets.
env_sanitize_seed() {
  local src=$1 dest=$2 magic
  : >"$dest"
  chmod 600 "$dest"
  magic=$(head -c 2 -- "$src" | od -An -tx1 | tr -d ' \n')
  if [[ $magic == fffe || $magic == feff ]]; then
    iconv -f UTF-16 -t UTF-8 "$src" >"$dest"
  else
    cat -- "$src" >"$dest"
  fi
  sed -i -e '1s/^\xEF\xBB\xBF//' -e 's/\r$//' "$dest"
}

# env_create_from_template TEMPLATE SEED OUT: OUT gets every key of TEMPLATE (with its comments) in
# template order. A key takes the SEED value when SEED has a non-empty one (machine-local keys excepted),
# otherwise the template default - secrets therefore start empty. SEED may be empty/absent.
env_create_from_template() (
  set +x
  local template=$1 seed=${2:-} out=$3 line key rhs enc
  local -A seed_map=()
  if [[ -n $seed && -f $seed ]]; then _env_read_into seed_map "$seed"; fi
  : >"$out"
  chmod 600 "$out"
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    if [[ $line =~ ^([A-Z][A-Z0-9_]*)=(.*)$ ]]; then
      key=${BASH_REMATCH[1]}
      rhs=${BASH_REMATCH[2]}
      if [[ -n ${seed_map[$key]:-} ]] && ! env_in_list "$key" "${ENV_MACHINE_LOCAL_KEYS[@]}"; then
        if env_encode_value_to enc "${seed_map[$key]}"; then
          rhs=$enc
        else
          err "$key: the value in the seed file has unsupported characters; left at its default"
        fi
      fi
      printf '%s=%s\n' "$key" "$rhs"
    else
      printf '%s\n' "$line"
    fi
  done <"$template" >>"$out"
)

# env_import_seed SEED TARGET TEMPLATE: for every key known to TEMPLATE that is non-empty in SEED and
# empty/missing in TARGET, copy the value. Existing non-empty values are never touched. Prints the names
# of the keys that were imported.
env_import_seed() (
  set +x
  local seed=$1 target=$2 template=$3 key
  local -A seed_map=() target_map=()
  local -a keys=()
  _env_read_into seed_map "$seed"
  _env_read_into target_map "$target"
  _env_keys_into keys "$template"
  for key in "${keys[@]}"; do
    if env_in_list "$key" "${ENV_MACHINE_LOCAL_KEYS[@]}"; then continue; fi
    if [[ -z ${seed_map[$key]:-} ]]; then continue; fi
    if [[ -z ${target_map[$key]:-} ]]; then
      if set_env_var "$key" "${seed_map[$key]}" "$target"; then printf '%s\n' "$key"; fi
    fi
  done
)

# env_unknown_keys SEED TEMPLATE: names assigned in SEED that the template does not know (typos, dev-only).
env_unknown_keys() (
  set +x
  local -a seed_keys=() template_keys=()
  local -A known=()
  local key
  _env_keys_into seed_keys "$1"
  _env_keys_into template_keys "$2"
  for key in "${template_keys[@]}"; do known[$key]=1; done
  for key in "${seed_keys[@]}"; do
    if [[ -z ${known[$key]+x} ]]; then printf '%s\n' "$key"; fi
  done
)

# env_append_missing TEMPLATE TARGET: append every TEMPLATE key that TARGET does not assign at all
# (with the comment block above it). Prints the appended key names.
env_append_missing() (
  set +x
  local template=$1 target=$2 line key pending="" additions="" tmp stamp
  local -A present=()
  local -a target_keys=()
  _env_keys_into target_keys "$target"
  for key in "${target_keys[@]}"; do present[$key]=1; done
  while IFS= read -r line || [[ -n $line ]]; do
    line=${line%$'\r'}
    if [[ $line =~ ^([A-Z][A-Z0-9_]*)= ]]; then
      key=${BASH_REMATCH[1]}
      if [[ -z ${present[$key]+x} ]]; then
        additions+="${pending}${line}"$'\n'
        printf '%s\n' "$key"
      fi
      pending=""
    elif [[ $line == \#* ]]; then
      pending+="${line}"$'\n'
    else
      pending=""
    fi
  done <"$template"
  if [[ -z $additions ]]; then return 0; fi
  stamp=$(date -u '+%Y-%m-%d')
  tmp=$(mktemp "${target}.XXXXXX")
  trap 'rm -f -- "$tmp"' EXIT
  {
    if [[ -f $target ]]; then
      while IFS= read -r line || [[ -n $line ]]; do
        printf '%s\n' "${line%$'\r'}"
      done <"$target"
    fi
    printf '\n# ---- keys added by deployment/install.sh on %s ----\n%s' "$stamp" "$additions"
  } >"$tmp"
  if [[ -e $target ]]; then
    chmod --reference="$target" "$tmp" 2>/dev/null || true
    chown --reference="$target" "$tmp" 2>/dev/null || true
  else
    chmod 600 "$tmp"
  fi
  mv -f -- "$tmp" "$target"
  trap - EXIT
)

# load_redaction_values FILE: fill REDACT_VALUES with the secret values stored in FILE (min. 6 chars, as in
# the application's logger) so redact_stream can mask them.
load_redaction_values() {
  local file=$1 key value xtrace=0
  local -A map=()
  case $- in *x*)
    xtrace=1
    set +x
    ;;
  esac
  REDACT_VALUES=()
  _env_read_into map "$file"
  for key in "${ENV_SECRET_KEYS[@]}"; do
    value=${map[$key]:-}
    if ((${#value} >= 6)); then REDACT_VALUES+=("$value"); fi
  done
  if ((xtrace)); then set -x; fi
  return 0
}
