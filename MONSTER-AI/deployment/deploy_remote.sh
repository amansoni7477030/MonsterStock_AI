#!/usr/bin/env bash
# =====================================================================================================
# MONSTER AI - one-command remote deploy from Linux / macOS
#
#   deployment/deploy_remote.sh user@host [options]
#
#   --port N             SSH port (default 22)
#   --identity FILE      SSH private key
#   --env-file FILE      configuration to seed /etc/monster-ai/monster.env from (default: ./.env of the
#                        project when it exists). Secrets are copied over SSH only, never printed, and the
#                        remote copy is shredded afterwards.
#   --no-bootstrap | --enable-dashboard | --skip-swap    passed on to install.sh
#   --keep-remote-temp   keep the uploaded release on the server (the env copy is deleted regardless)
#
# What it does: packs the project (without .git, .venv, tests, runtime data and env files), uploads it
# together with the env file into a private temporary directory on the server, runs
# `sudo bash deployment/install.sh --yes` there with a terminal (so sudo can ask for a password and you
# see the progress), and finally removes the temporary directory. Needs only ssh, scp and tar.
# Exit code: that of the installer (0 = installed).
# =====================================================================================================
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

TARGET=""
PORT=22
IDENTITY=""
ENV_FILE=""
INSTALL_FLAGS=()
KEEP_REMOTE_TEMP=0

ARCHIVE=""
REMOTE_DIR=""
SSH_OPTS=()
SCP_OPTS=()

say() { printf '%s\n' "$*"; }
die() {
  printf '[FAIL] %s\n' "$1" >&2
  exit "${2:-1}"
}

usage() {
  cat <<'EOF'
Usage: deployment/deploy_remote.sh user@host [options]

  --port N             SSH port (default 22)
  --identity FILE      SSH private key
  --env-file FILE      configuration seed (default: ./.env of the project when present)
  --no-bootstrap       do not run the first-time data bootstrap now
  --enable-dashboard   install the optional read-only dashboard (loopback only)
  --skip-swap          do not create a swap file
  --keep-remote-temp   keep the uploaded release on the server (the env copy is always deleted)
  -h, --help           this text
EOF
}

parse_args() {
  while (($#)); do
    case $1 in
      --target | -t)
        (($# >= 2)) || die "$1 needs user@host" 2
        TARGET=$2
        shift
        ;;
      --port | -p)
        (($# >= 2)) || die "$1 needs a port number" 2
        PORT=$2
        shift
        ;;
      --identity | -i)
        (($# >= 2)) || die "$1 needs a key file" 2
        IDENTITY=$2
        shift
        ;;
      --env-file | -e)
        (($# >= 2)) || die "$1 needs a file" 2
        ENV_FILE=$2
        shift
        ;;
      --no-bootstrap) INSTALL_FLAGS+=(--no-bootstrap) ;;
      --enable-dashboard) INSTALL_FLAGS+=(--enable-dashboard) ;;
      --skip-swap) INSTALL_FLAGS+=(--skip-swap) ;;
      --keep-remote-temp) KEEP_REMOTE_TEMP=1 ;;
      -h | --help)
        usage
        exit 0
        ;;
      -*)
        usage >&2
        die "unknown option: $1" 2
        ;;
      *)
        [[ -z $TARGET ]] || die "unexpected argument: $1 (the target is already $TARGET)" 2
        TARGET=$1
        ;;
    esac
    shift
  done
}

validate() {
  local target_re='^[A-Za-z0-9._-]+@[][A-Za-z0-9._:%-]+$' tool
  [[ -n $TARGET ]] || {
    usage >&2
    die "the target (user@host) is required" 2
  }
  [[ $TARGET =~ $target_re ]] || die "the target must look like user@host (letters, digits, . _ - : only): $TARGET" 2
  if ! [[ $PORT =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then die "the port must be 1-65535: $PORT" 2; fi
  for tool in ssh scp tar; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' was not found. Install the OpenSSH client and tar."
  done
  [[ -f $PROJECT_ROOT/deployment/install.sh ]] || die "deployment/install.sh not found below $PROJECT_ROOT"
  if [[ -n $IDENTITY ]]; then
    [[ -f $IDENTITY ]] || die "identity file not found: $IDENTITY" 2
  fi
  if [[ -n $ENV_FILE ]]; then
    [[ -f $ENV_FILE ]] || die "env file not found: $ENV_FILE" 2
  elif [[ -f $PROJECT_ROOT/.env ]]; then
    ENV_FILE=$PROJECT_ROOT/.env
  fi
}

build_ssh_options() {
  SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
  SCP_OPTS=(-P "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20)
  if [[ -n $IDENTITY ]]; then
    SSH_OPTS+=(-i "$IDENTITY")
    SCP_OPTS+=(-i "$IDENTITY")
  fi
}

# scp needs [ ] around IPv6 literals: user@[::1]:path
scp_target() {
  local user=${TARGET%%@*} host=${TARGET#*@}
  case $host in
    *:*) if [[ $host != \[* ]]; then host="[$host]"; fi ;;
  esac
  printf '%s@%s' "$user" "$host"
}

# The remote temporary directory holds the release and (briefly) the env file with secrets: delete the env
# copy first (shred when available), then the directory. Runs on every exit path.
cleanup() {
  local rc=$? q="'"
  trap - EXIT INT TERM
  set +e
  if [[ -n $ARCHIVE && -f $ARCHIVE ]]; then rm -f -- "$ARCHIVE"; fi
  if [[ -n $REMOTE_DIR ]]; then
    if ! ssh "${SSH_OPTS[@]}" "$TARGET" bash -s -- "$REMOTE_DIR" "$KEEP_REMOTE_TEMP" <<'EOF'
set -u
d=$1
keep=$2
case $d in /*/monster-ai-deploy.*) ;; *) exit 2 ;; esac
if [ -f "$d/monster.env" ]; then
  if command -v shred >/dev/null 2>&1; then shred -u "$d/monster.env" 2>/dev/null || rm -f "$d/monster.env"; else rm -f "$d/monster.env"; fi
fi
if [ "$keep" != 1 ]; then rm -rf "$d"; fi
EOF
    then
      printf '[WARN] could not clean up %s on the server. It may still hold a copy of your env file - remove it with:\n' "$REMOTE_DIR" >&2
      printf '       ssh %s %sshred -u %s/monster.env; rm -rf %s%s\n' "$TARGET" "$q" "$REMOTE_DIR" "$REMOTE_DIR" "$q" >&2
    fi
  fi
  exit "$rc"
}

build_archive() {
  ARCHIVE=${TMPDIR:-/tmp}/monster-ai-$(date +%Y%m%d-%H%M%S).tar.gz
  say "Packing the project -> $ARCHIVE"
  # COPYFILE_DISABLE keeps macOS tar from adding ._* AppleDouble files.
  COPYFILE_DISABLE=1 tar -czf "$ARCHIVE" \
    --exclude=.venv --exclude=.git --exclude=__pycache__ '--exclude=*.pyc' \
    --exclude=data/runtime --exclude=data/cache --exclude=logs --exclude=tests --exclude=.github \
    '--exclude=*.db' '--exclude=*.db-wal' '--exclude=*.db-shm' '--exclude=*.log' \
    --exclude=.env '--exclude=*.env' '--exclude=*.tar.gz' '--exclude=*.zip' \
    --exclude=.ruff_cache --exclude=.pytest_cache --exclude=.mypy_cache --exclude=.coverage --exclude=htmlcov \
    -C "$PROJECT_ROOT" .
  say "  archive size: $(du -h "$ARCHIVE" | cut -f 1)"
}

upload() {
  local archive_name dest
  archive_name=$(basename -- "$ARCHIVE")
  dest="$(scp_target):$REMOTE_DIR"
  say "Uploading to $TARGET:$REMOTE_DIR ..."
  scp "${SCP_OPTS[@]}" -- "$ARCHIVE" "$dest/$archive_name"
  if [[ -n $ENV_FILE ]]; then
    say "  env file: $ENV_FILE (copied over SSH, contents are never shown, deleted from the server afterwards)"
    scp "${SCP_OPTS[@]}" -- "$ENV_FILE" "$dest/monster.env"
  else
    say "  no env file given and no ./.env found: the server starts from .env.example (add your Telegram token later)"
  fi
  # Unpack on the server and make shell/unit files LF-only (a Windows checkout may carry CRLF).
  ssh "${SSH_OPTS[@]}" "$TARGET" bash -s -- "$REMOTE_DIR" "$archive_name" <<'EOF'
set -eu
d=$1
a=$2
case $d in /*/monster-ai-deploy.*) ;; *) echo "unexpected directory $d" >&2; exit 2 ;; esac
cd "$d"
tar -xzf "$a"
rm -f "$a"
find . -type f \( -name '*.sh' -o -name '*.service' -o -name '*.timer' -o -name '*.conf' -o -name 'logrotate-monster' \) -exec sed -i 's/\r$//' {} +
chmod 700 "$d"
EOF
}

main() {
  local sudo_prefix="sudo " remote_cmd flag rc=0
  parse_args "$@"
  validate
  build_ssh_options
  build_archive
  say "Connecting to $TARGET (port $PORT) ..."
  REMOTE_DIR=$(ssh "${SSH_OPTS[@]}" "$TARGET" 'mktemp -d -t monster-ai-deploy.XXXXXXXXXX')
  [[ $REMOTE_DIR =~ ^/[A-Za-z0-9._/-]+$ ]] || die "unexpected answer from the server instead of a temporary directory: $REMOTE_DIR"
  upload
  if [[ ${TARGET%%@*} == root ]]; then sudo_prefix=""; fi
  remote_cmd="${sudo_prefix}bash $REMOTE_DIR/deployment/install.sh --yes"
  if [[ -n $ENV_FILE ]]; then remote_cmd="$remote_cmd --env-file $REMOTE_DIR/monster.env"; fi
  for flag in ${INSTALL_FLAGS[@]+"${INSTALL_FLAGS[@]}"}; do remote_cmd="$remote_cmd $flag"; done
  say "Running the installer on the server (you may be asked for the sudo password) ..."
  say "------------------------------------------------------------------------------"
  ssh -tt "${SSH_OPTS[@]}" "$TARGET" "$remote_cmd" || rc=$?
  say "------------------------------------------------------------------------------"
  if ((rc == 0)); then
    say "Deployment finished successfully."
  else
    say "[FAIL] the installer exited with code $rc (see the output above; the transcript is in /var/log/monster-ai-install.log on the server)."
    say "       Fixing the cause and running this command again is safe."
  fi
  return "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
main "$@"
