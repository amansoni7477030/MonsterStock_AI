"""Checks of the deployment layer (no network; the installer is never run as a whole).

Static: line endings, shebang / strict mode, systemd units (tolerant INI reader, known-directive and
hardening checks), CLI commands and environment variables referenced by the scripts, secrets, forbidden
tooling, installer step functions, workflows, PowerShell script.

Executable (needs bash - Git Bash on Windows): the pure helper functions and the file-level release logic of
``install.sh`` run in a sandbox (``MONSTER_INSTALL_ROOT``) with stubbed system commands, so the trickiest string
and directory handling is really executed: injection-free ``set_env_var``, env merge, swap sizing, requirements
hash / venv reuse, release staging excludes, atomic swap, rollback, the exit trap (auto-rollback, exit codes,
services always restarted), systemd step, bootstrap polling, verification verdicts, Telegram chat-id discovery
and secret masking. The heavy ones carry ``@pytest.mark.slow``: Git Bash needs ~0.15 s per ``$(...)`` and ~0.3 s
per external command, so the full file takes minutes on Windows (seconds on Linux); ``pytest -m "not slow"``
skips them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import VARS, _read_env_file

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deployment"
SYSTEMD = DEPLOY / "systemd"
WORKFLOWS = ROOT / ".github" / "workflows"

SHELL_FILES = sorted([*DEPLOY.glob("*.sh"), *(DEPLOY / "lib").glob("*.sh")])
UNIT_FILES = sorted([*SYSTEMD.glob("*.service"), *SYSTEMD.glob("*.timer")])
CONF_FILES = [DEPLOY / "journald-monster.conf", DEPLOY / "logrotate-monster"]
LF_FILES = [*SHELL_FILES, *UNIT_FILES, *CONF_FILES]
SCRIPT_AND_UNIT_FILES = [*SHELL_FILES, *UNIT_FILES, *CONF_FILES]

EXPECTED_UNITS = {
    "monster-ai.service",
    "monster-ai-bootstrap.service",
    "monster-ai-health.service",
    "monster-ai-health.timer",
    "monster-ai-alert@.service",
    "monster-ai-dashboard.service",
}

# The CLI contract (everything the deployment layer may call).
CLI_CONTRACT = {
    "version", "env-example", "init", "doctor", "bootstrap", "daemon", "health", "alert-unit",
    "telegram-setup", "telegram-test", "dashboard", "backup",
}
CLI_WORKFLOW_EXTRA = {"train", "logs", "diag", "storage", "ai-test"}  # operator commands: documented, never called by units or scripts

# Names that are not application settings (app.config.VARS) but are set / read by the deployment layer.
RUNTIME_VARIABLES = {
    "MONSTER_ENV_FILE", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "MALLOC_ARENA_MAX", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PATH", "HOME", "LANG", "LC_ALL", "DEBIAN_FRONTEND",
    "NEEDRESTART_MODE", "APT_LOCK_TIMEOUT", "MONSTER_INSTALL_DEBUG", "MONSTER_INSTALL_IGNORE_WINDOW",
    "MONSTER_INSTALL_ROOT", "MONSTER_INSTALL_SOURCE_ONLY",
}
VAR_NAMES = {v.env for v in VARS}
SECRET_NAMES = {v.env for v in VARS if v.secret}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------- bash discovery
def _find_bash() -> str | None:
    if os.name != "nt":
        return shutil.which("bash")
    roots: list[Path] = []
    git = shutil.which("git")
    if git:
        roots.append(Path(git).resolve().parent.parent)
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            roots += [Path(base) / "Git", Path(base) / "Programs" / "Git"]
    for root in roots:
        for rel in ("bin/bash.exe", "usr/bin/bash.exe"):
            candidate = root / rel
            if candidate.is_file():
                return str(candidate)
    return None  # the WSL launcher in System32 / WindowsApps is deliberately ignored


BASH = _find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="bash (Git Bash on Windows) is not available")


def _tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in (Path(sys.executable).parent / name, Path(sys.executable).parent / f"{name}.exe"):
        if candidate.is_file():
            return str(candidate)
    return None


# =================================================================================================== static
def test_expected_files_exist():
    names = {p.name for p in SYSTEMD.iterdir()}
    assert EXPECTED_UNITS <= names, EXPECTED_UNITS - names
    for path in [DEPLOY / "install.sh", DEPLOY / "uninstall.sh", DEPLOY / "deploy_remote.sh",
                 DEPLOY / "deploy_remote.ps1", DEPLOY / "README.md", *CONF_FILES,
                 WORKFLOWS / "ci.yml", WORKFLOWS / "train.yml"]:
        assert path.is_file(), path
    assert {p.name for p in (DEPLOY / "lib").glob("*.sh")} >= {"common.sh", "envfile.sh", "decisions.sh", "system.sh"}


@pytest.mark.parametrize("path", LF_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_files_are_lf_only(path: Path):
    data = path.read_bytes()
    assert b"\r" not in data, f"{path.name} contains a CR byte (CRLF line endings break the server)"
    assert not data.startswith(b"\xef\xbb\xbf"), f"{path.name} starts with a BOM"
    assert data.endswith(b"\n"), f"{path.name} does not end with a newline"


def test_gitattributes_force_lf_for_deployment_files():
    text = read(ROOT / ".gitattributes")
    for pattern in ("*.sh", "*.service", "*.timer", "*.conf"):
        assert re.search(rf"^{re.escape(pattern)}\s+text\s+eol=lf", text, re.M), pattern
    assert re.search(r"^\*\s+text=auto\s+eol=lf", text, re.M)  # covers logrotate-monster, README, workflows


@pytest.mark.parametrize("path", SHELL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_shell_scripts_start_strict(path: Path):
    lines = read(path).split("\n")
    assert lines[0] == "#!/usr/bin/env bash", path.name
    first_command = next(line for line in lines[1:] if line.strip() and not line.lstrip().startswith("#"))
    assert first_command == "set -Eeuo pipefail", f"{path.name}: first command is {first_command!r}"


@needs_bash
@pytest.mark.parametrize("path", SHELL_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_bash_syntax(path: Path):
    result = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_shellcheck_clean():
    shellcheck = _tool("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck is not installed (CI runs it in its own job)")
    result = subprocess.run([shellcheck, "-x", *map(str, SHELL_FILES)], capture_output=True, text=True, check=False,
                            cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_powershell_script_parses_and_is_ascii():
    path = DEPLOY / "deploy_remote.ps1"
    data = path.read_bytes()
    assert all(b < 128 for b in data), "deploy_remote.ps1 must be pure ASCII (Windows PowerShell 5.1 reads it as ANSI)"
    shell = _tool("pwsh") or _tool("powershell")
    if shell is None:
        pytest.skip("no PowerShell available")
    command = (
        "$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{path}',[ref]$t,[ref]$e);if($e.Count){{$e|ForEach-Object{{$_.Message}};exit 1}}"
    )
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_powershell_script_never_reads_or_prints_the_env_file():
    text = read(DEPLOY / "deploy_remote.ps1")
    forbidden = ("Get-Content", "ReadAllText", "ReadAllLines", "ReadAllBytes", "[System.IO.File]::Open", "Select-String",
                 "ConvertFrom-StringData", "Import-Csv", "[IO.File]")
    for token in forbidden:
        assert token not in text, f"deploy_remote.ps1 must not read files ({token}): the env file is only copied"
    for line in text.splitlines():
        if re.search(r"Write-(Host|Output|Verbose|Debug|Warning)", line):
            assert not re.search(r"\$\(|\$envText|\$content|\$secret|\$token|\$values", line, re.I), line
    assert "StrictHostKeyChecking=accept-new" in text
    assert "ServerAliveInterval=30" in text
    assert "ConnectTimeout=20" in text
    assert "'-tt'" in text or '"-tt"' in text
    assert "shred -u" in text and "mktemp -d" in text
    assert "System32" in text and "tar.exe" in text


def test_remote_deploy_scripts_mirror_each_other():
    bash_text, ps_text = read(DEPLOY / "deploy_remote.sh"), read(DEPLOY / "deploy_remote.ps1")
    for token in ("--exclude=.venv", "--exclude=.git", "--exclude=__pycache__", "--exclude=data/runtime",
                  "--exclude=logs", "--exclude=tests", "--exclude=.github"):
        assert token in bash_text, token
        assert token.removeprefix("--exclude=") in ps_text, token
    for token in ("shred -u", "mktemp -d", "StrictHostKeyChecking=accept-new", "ServerAliveInterval=30",
                  "ConnectTimeout=20", "-tt", "--env-file", "--yes", "monster.env", r"sed -i 's/\r$//'"):
        assert token in bash_text, token
        assert token in ps_text or token.replace("--", "") in ps_text, token


# ------------------------------------------------------------------------------------------ systemd units
def parse_unit(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Tolerant INI reader: duplicate keys are kept (Environment=, After= ...), comments are skipped."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for number, raw in enumerate(read(path).split("\n"), 1):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        assert current is not None, f"{path.name}:{number}: setting before any [Section]"
        key, sep, value = line.partition("=")
        assert sep, f"{path.name}:{number}: not a Key=Value line: {raw!r}"
        sections[current].append((key.strip(), value.strip()))
    return sections


def values(unit: dict[str, list[tuple[str, str]]], section: str, key: str) -> list[str]:
    return [v for k, v in unit.get(section, []) if k == key]


COMMON_ENV = {
    "MONSTER_ENV": "production",
    "MONSTER_HOME": "/opt/monster-ai",
    "MONSTER_DATA_DIR": "/var/lib/monster-ai",
    "MONSTER_LOG_DIR": "/var/log/monster-ai",
    "MONSTER_ENV_FILE": "/etc/monster-ai/monster.env",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "MALLOC_ARENA_MAX": "2",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
HARDENING = {
    "NoNewPrivileges": "yes", "ProtectSystem": "strict", "ProtectHome": "yes", "PrivateTmp": "yes",
    "PrivateDevices": "yes", "ProtectKernelTunables": "yes", "ProtectKernelModules": "yes",
    "ProtectControlGroups": "yes", "ProtectClock": "yes", "RestrictSUIDSGID": "yes", "RestrictNamespaces": "yes",
    "LockPersonality": "yes", "RestrictRealtime": "yes", "SystemCallArchitectures": "native",
    "CapabilityBoundingSet": "", "AmbientCapabilities": "", "RestrictAddressFamilies": "AF_INET AF_INET6 AF_UNIX",
    "UMask": "0027", "ReadWritePaths": "/var/lib/monster-ai /var/log/monster-ai",
    "ReadOnlyPaths": "/opt/monster-ai /etc/monster-ai",
}
PY = "/opt/monster-ai/.venv/bin/python -m app.cli"

SERVICE_EXPECT: dict[str, dict[str, dict[str, str]]] = {
    "monster-ai.service": {
        "Service": {
            "Type": "notify", "NotifyAccess": "main", "User": "monster", "Group": "monster",
            "ExecStart": f"{PY} daemon", "WorkingDirectory": "/opt/monster-ai",
            "EnvironmentFile": "/etc/monster-ai/monster.env", "Restart": "always", "RestartSec": "10",
            "WatchdogSec": "180", "TimeoutStopSec": "90", "KillMode": "control-group", "OOMPolicy": "continue",
            "OOMScoreAdjust": "-300", "MemoryHigh": "750M", "MemoryMax": "880M", "TasksMax": "256",
            "LimitNOFILE": "4096", "Nice": "0", "CPUSchedulingPolicy": "other",
        },
        "Unit": {"OnFailure": "monster-ai-alert@%n.service", "StartLimitIntervalSec": "0"},
        "Install": {"WantedBy": "multi-user.target"},
    },
    "monster-ai-bootstrap.service": {
        "Service": {
            "Type": "oneshot", "RemainAfterExit": "yes", "User": "monster", "Group": "monster",
            "ExecStart": f"{PY} bootstrap --resume", "EnvironmentFile": "/etc/monster-ai/monster.env",
            "TimeoutStartSec": "infinity", "Nice": "10", "IOSchedulingClass": "idle", "MemoryHigh": "750M",
            "MemoryMax": "880M", "Restart": "on-failure", "RestartSec": "300",
        },
        "Unit": {"OnFailure": "monster-ai-alert@%n.service", "StartLimitBurst": "6", "StartLimitIntervalSec": "6h"},
    },
    "monster-ai-health.service": {
        "Service": {
            "Type": "oneshot", "User": "monster", "Nice": "10", "ExecStart": f"{PY} health --notify",
            "SuccessExitStatus": "1 2", "EnvironmentFile": "/etc/monster-ai/monster.env",
        },
    },
    "monster-ai-alert@.service": {
        "Service": {
            "Type": "oneshot", "User": "monster", "ExecStart": f"{PY} alert-unit %i", "TimeoutStartSec": "60",
            "EnvironmentFile": "/etc/monster-ai/monster.env",
        },
    },
    "monster-ai-dashboard.service": {
        "Service": {
            "User": "monster", "ExecStart": f"{PY} dashboard", "Restart": "on-failure", "MemoryMax": "250M",
            "EnvironmentFile": "/etc/monster-ai/monster.env",
        },
        "Unit": {"After": "monster-ai.service"},
    },
}

KNOWN_KEYS = {
    "Unit": {"Description", "Documentation", "After", "Before", "Wants", "Requires", "OnFailure",
             "StartLimitIntervalSec", "StartLimitBurst", "ConditionPathExists"},
    "Service": {
        "Type", "NotifyAccess", "User", "Group", "SupplementaryGroups", "WorkingDirectory", "SyslogIdentifier",
        "EnvironmentFile", "Environment", "ExecStart", "Restart", "RestartSec", "WatchdogSec", "TimeoutStartSec",
        "TimeoutStopSec", "KillMode", "OOMPolicy", "OOMScoreAdjust", "MemoryHigh", "MemoryMax", "TasksMax",
        "LimitNOFILE", "Nice", "CPUSchedulingPolicy", "IOSchedulingClass", "UMask", "RemainAfterExit",
        "SuccessExitStatus", "StateDirectory", "StateDirectoryMode", "LogsDirectory", "LogsDirectoryMode",
        "NoNewPrivileges", "ProtectSystem", "ProtectHome", "PrivateTmp", "PrivateDevices", "ProtectKernelTunables",
        "ProtectKernelModules", "ProtectKernelLogs", "ProtectControlGroups", "ProtectClock", "ProtectHostname",
        "RestrictSUIDSGID", "RestrictNamespaces", "LockPersonality", "RestrictRealtime", "SystemCallArchitectures",
        "CapabilityBoundingSet", "AmbientCapabilities", "RestrictAddressFamilies", "ReadWritePaths", "ReadOnlyPaths",
    },
    "Timer": {"OnBootSec", "OnUnitActiveSec", "AccuracySec", "RandomizedDelaySec", "Unit", "Persistent"},
    "Install": {"WantedBy", "RequiredBy", "Alias"},
}
SERVICES = sorted(p for p in UNIT_FILES if p.suffix == ".service")


@pytest.mark.parametrize("path", UNIT_FILES, ids=lambda p: p.name)
def test_unit_has_a_description_comment_block_and_known_directives(path: Path):
    text = read(path)
    header = []
    for line in text.split("\n"):
        if line.startswith("["):
            break
        header.append(line)
    assert sum(1 for line in header if line.startswith("#")) >= 4, f"{path.name}: describe the unit in a comment block"
    unit = parse_unit(path)
    assert "Description" in dict(unit["Unit"])
    for section, entries in unit.items():
        assert section in KNOWN_KEYS, f"{path.name}: unexpected section [{section}]"
        for key, _value in entries:
            assert key in KNOWN_KEYS[section], f"{path.name} [{section}]: unknown/misplaced directive {key}"


@pytest.mark.parametrize("name", sorted(SERVICE_EXPECT))
def test_unit_required_settings(name: str):
    unit = parse_unit(SYSTEMD / name)
    for section, expected in SERVICE_EXPECT[name].items():
        for key, wanted in expected.items():
            found = values(unit, section, key)
            assert found, f"{name} [{section}] {key}= is missing"
            assert wanted in found or wanted in " ".join(found).split(), f"{name} {key}={found} expected {wanted!r}"


@pytest.mark.parametrize("path", SERVICES, ids=lambda p: p.name)
def test_service_is_hardened_and_has_the_runtime_environment(path: Path):
    unit = parse_unit(path)
    for key, wanted in HARDENING.items():
        assert values(unit, "Service", key) == [wanted], f"{path.name}: {key} must be {wanted!r}"
    env = {}
    for item in values(unit, "Service", "Environment"):
        name, _, value = item.partition("=")
        env[name] = value
    assert env == COMMON_ENV, path.name
    assert values(unit, "Service", "EnvironmentFile") == ["/etc/monster-ai/monster.env"]
    for forbidden in ("SystemCallFilter", "MemoryDenyWriteExecute"):
        assert forbidden not in dict(unit["Service"]), f"{path.name}: {forbidden} breaks numpy/LightGBM"
    assert values(unit, "Service", "User") == ["monster"]


def test_start_limits_are_in_the_unit_section_for_systemd_249():
    for path in SERVICES:
        unit = parse_unit(path)
        for key in ("StartLimitIntervalSec", "StartLimitBurst"):
            assert key not in dict(unit.get("Service", [])), f"{path.name}: {key} belongs to [Unit] (systemd 249)"


def test_failure_alert_has_no_on_failure_and_daemon_never_gives_up():
    alert = parse_unit(SYSTEMD / "monster-ai-alert@.service")
    assert "OnFailure" not in dict(alert["Unit"]), "an alert that fails must not trigger another alert"
    daemon = parse_unit(SYSTEMD / "monster-ai.service")
    after = " ".join(values(daemon, "Unit", "After")).split()
    wants = " ".join(values(daemon, "Unit", "Wants")).split()
    for target in ("network-online.target", "time-sync.target"):
        assert target in after and target in wants


def test_dashboard_unit_does_not_bind_publicly_and_is_not_enabled_by_default():
    text = "\n".join(line for line in read(SYSTEMD / "monster-ai-dashboard.service").split("\n")
                     if not line.lstrip().startswith("#"))
    assert "DASHBOARD_HOST" not in text and "0.0.0.0" not in text
    unit = parse_unit(SYSTEMD / "monster-ai-dashboard.service")
    assert values(unit, "Install", "WantedBy") == ["multi-user.target"]


def test_health_timer_schedule():
    timer = parse_unit(SYSTEMD / "monster-ai-health.timer")
    expected = {"OnBootSec": "3min", "OnUnitActiveSec": "5min", "AccuracySec": "30s", "RandomizedDelaySec": "0",
                "Unit": "monster-ai-health.service"}
    for key, wanted in expected.items():
        assert values(timer, "Timer", key) == [wanted], key
    assert values(timer, "Install", "WantedBy") == ["timers.target"]


def test_journald_and_logrotate_configs():
    journald = read(DEPLOY / "journald-monster.conf")
    for line in ("[Journal]", "SystemMaxUse=100M", "RuntimeMaxUse=50M", "MaxRetentionSec=2week"):
        assert line in journald.split("\n"), line
    rotate = read(DEPLOY / "logrotate-monster")
    assert "/var/log/monster-ai/*.out {" in rotate
    # the application owns its daily *.log files: logrotate's copytruncate would empty yesterday's file
    assert not re.search(r"^/var/log/monster-ai/\*\.log\b", rotate, re.M)
    assert "/var/log/monster-ai-install.log {" in rotate
    for token in ("weekly", "rotate 4", "compress", "missingok", "notifempty", "copytruncate", "su monster monster"):
        assert token in rotate, token


# --------------------------------------------------------------------------- CLI commands and variables
def cli_commands(text: str) -> set[str]:
    return set(re.findall(r"app\.cli\s+([a-z][a-z0-9-]*)", text))


def test_only_contract_cli_commands_are_used_by_units_and_scripts():
    used: set[str] = set()
    for path in SCRIPT_AND_UNIT_FILES:
        commands = cli_commands(read(path))
        assert commands <= CLI_CONTRACT, f"{path.name} uses commands outside the CLI contract: {commands - CLI_CONTRACT}"
        used |= commands
    assert {"init", "doctor", "bootstrap", "daemon", "health", "alert-unit", "telegram-setup", "telegram-test",
            "dashboard", "version"} <= used, used


def test_workflows_and_docs_use_only_documented_cli_commands():
    for path in [*WORKFLOWS.glob("*.yml"), DEPLOY / "README.md"]:
        commands = cli_commands(read(path))
        assert commands <= CLI_CONTRACT | CLI_WORKFLOW_EXTRA, f"{path.name}: {commands}"


def test_cli_commands_exist_in_the_real_cli():
    cli = ROOT / "app" / "cli.py"
    if not cli.is_file():
        pytest.skip("app/cli.py is not there yet")
    defined = set(re.findall(r'\badd\("([a-z][a-z0-9-]*)"', read(cli)))
    used = set()
    for path in [*SCRIPT_AND_UNIT_FILES, *WORKFLOWS.glob("*.yml")]:
        used |= cli_commands(read(path))
    assert used <= defined, f"the deployment layer calls CLI commands that do not exist: {used - defined}"


def referenced_env_names(path: Path) -> set[str]:
    text = "\n".join(line for line in read(path).split("\n") if not line.lstrip().startswith("#"))
    names: set[str] = set()
    if path.suffix in {".service", ".timer"}:
        names |= {m.group(1) for m in re.finditer(r"^Environment=([A-Z][A-Z0-9_]*)=", text, re.M)}
        return names
    for block in re.findall(r"env -i((?:[^\n]*\\\n)*[^\n]*)", text):
        names |= set(re.findall(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9_]{2,})=", block))
    names |= set(re.findall(r"\b(?:env_get|env_has_key|env_is_set|set_env_var)\s+\"?([A-Z][A-Z0-9_]+)", text))
    for array in re.findall(r"(?:ENV_SECRET_KEYS|ENV_MACHINE_LOCAL_KEYS)=\(([^)]*)\)", text, re.S):
        names |= set(re.findall(r"[A-Z][A-Z0-9_]+", array))
    for line in re.findall(r"^\s*export ([^\n]+)$", text, re.M):
        names |= set(re.findall(r"([A-Z][A-Z0-9_]+)=", line))
    names |= set(re.findall(r"\b(MONSTER_[A-Z0-9_]+)", text))
    return names


@pytest.mark.parametrize("path", [*SHELL_FILES, *UNIT_FILES], ids=lambda p: p.name)
def test_environment_variables_are_known(path: Path):
    unknown = referenced_env_names(path) - VAR_NAMES - RUNTIME_VARIABLES
    assert not unknown, f"{path.name} references variables that are neither settings nor runtime variables: {unknown}"


def test_train_workflow_environment_names_are_settings():
    text = read(WORKFLOWS / "train.yml")
    names = set(re.findall(r"^\s+([A-Z][A-Z0-9_]+):", text, re.M))
    unknown = names - VAR_NAMES - {"PYTHONUNBUFFERED", "OMP_NUM_THREADS", "STAGE"}
    assert not unknown, unknown
    for secret in ("MODEL_SIGNING_KEY", "S3_BUCKET", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "TELEGRAM_BOT_TOKEN"):
        assert f"${{{{ secrets.{secret} }}}}" in text, secret


def test_secret_key_list_matches_the_application():
    text = read(DEPLOY / "lib" / "envfile.sh")
    block = re.search(r"ENV_SECRET_KEYS=\(([^)]*)\)", text, re.S)
    assert block
    assert set(re.findall(r"[A-Z][A-Z0-9_]+", block.group(1))) == SECRET_NAMES


# ------------------------------------------------------------------------------------ hygiene / policy
SCANNED = [*SCRIPT_AND_UNIT_FILES, DEPLOY / "deploy_remote.ps1", DEPLOY / "README.md", *WORKFLOWS.glob("*.yml")]


def test_no_secret_looking_literals():
    patterns = [
        r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b",  # Telegram bot token
        r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key id
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?i)\b(password|passphrase|secret|token|api[_-]?key)\b\s*[:=]\s*['\"]?[A-Za-z0-9/+_-]{16,}",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    ]
    for path in SCANNED:
        text = read(path)
        for pattern in patterns:
            match = re.search(pattern, text)
            assert not match, f"{path.name} contains a secret-looking literal: {match.group(0)[:12]}..."


def test_secrets_never_appear_in_unit_files_or_process_arguments():
    for path in UNIT_FILES:
        text = read(path)
        for name in SECRET_NAMES:
            assert not re.search(rf"^Environment={name}=", text, re.M), f"{path.name}: {name} must come from the env file"
    install = read(DEPLOY / "install.sh") + read(DEPLOY / "lib" / "system.sh")
    for name in SECRET_NAMES:
        assert not re.search(rf"(?<![A-Z_]){name}=\S*\$", install), f"a command line carries {name}"
    assert "set +x" in read(DEPLOY / "lib" / "envfile.sh")  # value handling runs with xtrace off


def test_no_container_tooling():
    forbidden = re.compile(
        r"\b(docker|dockerfile|docker-compose|podman|buildah|nerdctl|containerd|crictl|kubectl|k8s|lxc|lxd|"
        r"systemd-nspawn|devcontainer|ghcr\.io)\b|^\s*container\s*:|^\s*services\s*:|^\s*image\s*:", re.I | re.M)
    for path in [*SCANNED, *DEPLOY.rglob("*")]:
        if path.is_file() and path.suffix not in {".pyc"}:
            for match in forbidden.finditer(read(path)):
                pytest.fail(f"{path.relative_to(ROOT)} references container tooling: {match.group(0)!r}")
    for stray in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yaml", "compose.yml",
                  ".devcontainer", "Containerfile", ".dockerignore"):
        assert not (ROOT / stray).exists(), stray
    assert "no containers" in read(DEPLOY / "README.md").lower()


# --------------------------------------------------------------------------------------- installer shape
STEP_FUNCTIONS = [
    "step_preflight", "step_packages", "step_swap", "step_user_dirs", "step_release", "step_venv", "step_envfile",
    "step_validate_config", "step_telegram", "step_initialise", "step_systemd", "step_bootstrap", "step_verify",
    "step_summary",
]


def test_install_script_defines_one_function_per_step_in_order():
    text = read(DEPLOY / "install.sh")
    defined = re.findall(r"^(step_[a-z_]+)\(\) \{", text, re.M)
    assert defined == STEP_FUNCTIONS
    for number, name in enumerate(STEP_FUNCTIONS, 1):
        body = text.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]
        assert re.search(rf"\bstep {number} \"", body), f"{name} must announce itself as step {number}"
    main_body = text.split("\nmain() {", 1)[1].split("\n}\n", 1)[0]
    assert re.findall(r"^\s+(step_[a-z_]+)$", main_body, re.M) == STEP_FUNCTIONS
    assert "TOTAL_STEPS" in read(DEPLOY / "lib" / "common.sh") and ": \"${TOTAL_STEPS:=14}\"" in read(
        DEPLOY / "lib" / "common.sh")


def test_install_script_implements_the_specified_behaviour():
    install = read(DEPLOY / "install.sh")
    system = read(DEPLOY / "lib" / "system.sh")
    everything = install + system + read(DEPLOY / "lib" / "common.sh") + read(DEPLOY / "lib" / "decisions.sh")
    for flag in ("--env-file", "--source", "--no-bootstrap", "--skip-swap", "--enable-dashboard", "--yes",
                 "--dry-run", "--no-start"):
        assert flag in install, flag
    for token in (
        "MONSTER_INSTALL_DEBUG", "set -x", "LC_ALL=C.UTF-8", "DEBIAN_FRONTEND=noninteractive", "NEEDRESTART_MODE=a",
        "DPkg::Lock::Timeout", "flock", "monster-ai-install.lock", "--only-binary=:all:", "--no-cache-dir",
        "pip check", "compileall", ".requirements.sha256", "fallocate", "dd if=/dev/zero", "chmod 600",
        "mkswap", "swapon", "/etc/fstab", "vm.swappiness", "vm.vfs_cache_pressure", "vm.overcommit_memory",
        "journald.conf.d", "logrotate", "NTPSynchronized", "chrony", "systemd-timesyncd", "runuser -u",
        "doctor --offline", "telegram-setup --print-only --wait 180", "Send any message to your bot now",
        "waiting up to 180 s", "monster-ai-bootstrap.service", "systemctl start --no-block", "rollback_release",
        "telegram-test", "health --json", "doctor --json", "SHADOW", "journalctl -u monster-ai -f", "trap ",
        "umask 022",
    ):
        assert token in everything, f"install scripts lack: {token}"
    assert "ssh-copy" not in everything
    for package in ("python3-venv", "python3-pip", "libgomp1", "sqlite3", "ca-certificates", "curl", "gnupg", "tzdata",
                    "logrotate", "chrony"):
        assert package in install, package
    for host in ("pypi.org", "api.telegram.org", "query1.finance.yahoo.com", "nsearchives.nseindia.com"):
        assert host in install, host
    # binary wheels only, with exactly one audited pure-Python exception
    assert "PIP_SDIST_OK=sgmllib3k" in install
    assert "--no-binary=$PIP_SDIST_OK" in install
    assert "--no-binary=:all:" not in install and "--only-binary=:none:" not in install


def test_cli_is_always_run_as_the_service_user():
    install = read(DEPLOY / "install.sh")
    for line in install.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("#", "info ", "warn ", "die ", "_say ")) or not re.search(r"app\.cli\s+[a-z]", line):
            continue
        assert "run_as_monster" in line, f"app.cli must never run as root: {line.strip()}"
    wrapper = install.split("install_cli_wrapper() {", 1)[1].split("\n}\n", 1)[0]
    assert "runuser -u monster" in wrapper and "env -i" in wrapper


def test_uninstall_script_keeps_data_by_default():
    text = read(DEPLOY / "uninstall.sh")
    for token in ("--purge", "--purge-swap", "--yes", "confirm", "SWAPFILE", "systemctl disable", "journald",
                  "LOGROTATE_FILE", "SYSCTL_FILE"):
        assert token in text, token
    main_body = text.split("\nmain() {", 1)[1]
    purge_block = main_body.split("if ((PURGE)); then\n    purge_data", 1)
    assert len(purge_block) == 2, "data deletion must be behind --purge"
    assert "remove_path \"$APP_DIR\"" in main_body
    assert re.search(r"remove_path \"\$DATA_DIR\"", text.split("purge_data() {", 1)[1].split("\n}\n", 1)[0])
    swap_removal = re.findall(r"^\s*remove_swap$", main_body, re.M)
    assert swap_removal and "if ((PURGE_SWAP)); then\n    remove_swap" in main_body


def test_readme_covers_the_required_topics():
    text = read(DEPLOY / "README.md")
    for token in ("Quickstart", "deploy_remote.ps1", "deploy_remote.sh", "Upgrade", "rollback", "uninstall",
                  "Troubleshooting", "apt", "swap", "PyPI", "NSE", "degraded", "chat id", "NTP", "Security notes",
                  "ssh -L 8080:127.0.0.1:8080", "Resource budget", "External training", "TRAINING_MODE=external",
                  "MODEL_SIGNING_KEY", "SHADOW", "No firewall"):
        assert token.lower() in text.lower(), token


def test_ci_workflow_contents():
    text = read(WORKFLOWS / "ci.yml")
    for token in ("ubuntu-latest", '"3.10"', '"3.12"', '"3.13"', "cache: pip", "pip install -r requirements-dev.txt",
                  "ruff check .", "pytest --cov --cov-fail-under=80", "shellcheck", "pip download",
                  "--only-binary=:all:", "x86_64", "aarch64", "pull_request", "push"):
        assert token in text, token
    assert "container:" not in text


def test_train_workflow_contents():
    text = read(WORKFLOWS / "train.yml")
    for token in ("schedule:", "cron:", "workflow_dispatch", "setup-python", '"3.12"', "python -m app.cli init",
                  "python -m app.cli train", "--publish", "upload-artifact", "MODEL_SIGNING_KEY", "S3_", "TELEGRAM_"):
        assert token in text, token
    assert "bootstrap" not in text.lower().replace("# ", "")  # only documented commands


# ============================================================================================== executable
def _bash_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MONSTER_", "BASH_"))}
    env.update(extra)
    return env


def run_bash(script: str, work: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Run a bash harness; ``LIBDIR``, ``PROJECT_DIR`` and ``WORKDIR`` are provided to the script."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / "harness.sh"
    path.write_text(script, encoding="utf-8", newline="\n")
    variables = {"LIBDIR": str(DEPLOY / "lib"), "PROJECT_DIR": str(ROOT), "WORKDIR": str(work), **env}
    return subprocess.run([BASH, str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env=_bash_env(variables), check=False, timeout=300)


def assert_harness_ok(result: subprocess.CompletedProcess[str]) -> list[str]:
    oks = [line for line in result.stdout.splitlines() if line.startswith("ok ")]
    failures = [line for line in result.stdout.splitlines() if line.startswith("FAIL")]
    assert not failures and result.returncode == 0, (
        f"harness failures: {failures}\n--- stdout ---\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-3000:]}")
    assert oks, "the harness ran no checks"
    return oks


HARNESS_HEADER = r"""
set -Eeuo pipefail
to_posix() { cygpath -u "$1" 2>/dev/null || printf '%s' "$1"; }
LIB=$(to_posix "$LIBDIR")
WORK=$(to_posix "$WORKDIR")
PROJECT=$(to_posix "$PROJECT_DIR")
fail=0
check() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: expected [$2] got [$3]"; fail=1; fi; }
yes_() { if "${@:2}"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
no_() { if "${@:2}"; then echo "FAIL $1 (expected failure)"; fail=1; else echo "ok   $1"; fi; }
count_cr() { tr -cd '\r' <"$1" | wc -c | tr -d ' '; }
"""
HARNESS_FOOTER = "\nexit \"$fail\"\n"


# --- systemd model of the EnvironmentFile= parser ---------------------------------------------------------
def systemd_env_parse(text: str) -> dict[str, str]:
    """Model of systemd's EnvironmentFile= reader (src/basic/env-file.c): quotes, backslashes, comments."""
    out: dict[str, str] = {}
    state, key, value, trailing = "pre_key", "", "", None
    i = 0

    def finish() -> None:
        nonlocal key, value, trailing
        if trailing is not None:
            value = value[:trailing]
        if key:
            out[key.rstrip()] = value
        key, value, trailing = "", "", None

    while i < len(text):
        c = text[i]
        i += 1
        if state == "pre_key":
            if c in " \t\r\n":
                continue
            if c in "#;":
                state = "comment"
            else:
                key, state = c, "key"
        elif state == "comment":
            state = "comment_esc" if c == "\\" else ("pre_key" if c in "\r\n" else "comment")
        elif state == "comment_esc":
            state = "comment"
        elif state == "key":
            if c in "\r\n":
                key, state = "", "pre_key"
            elif c == "=":
                state = "pre_value"
            else:
                key += c
        elif state == "pre_value":
            if c in " \t":
                continue
            i -= 1
            state = "value"
        elif state == "value":
            if c in "\r\n":
                finish()
                state = "pre_key"
            elif c == "\\":
                state = "value_esc"
            elif c in "'\"":
                state, trailing = ("single" if c == "'" else "double"), None
            elif c in " \t":
                trailing = len(value) if trailing is None else trailing
                value += c
            else:
                value, trailing = value + c, None
        elif state == "value_esc":
            state = "value"
            if c not in "\r\n":
                value, trailing = value + "\\" + c, None
        elif state == "single":
            if c == "'":
                state = "value"
            elif c == "\\":
                state = "single_esc"
            else:
                value += c
        elif state == "single_esc":
            state = "single"
            if c not in "\r\n":
                value += "\\" + c
        elif state == "double":
            if c == '"':
                state = "value"
            elif c == "\\":
                state = "double_esc"
            else:
                value += c
        elif state == "double_esc":
            state = "double"
            if c in '"\\`$':
                value += c
            elif c not in "\r\n":
                value += "\\" + c
    if state in {"value", "pre_value"}:
        finish()
    return out


@needs_bash
@pytest.mark.slow
def test_env_values_survive_both_parsers(tmp_path: Path):
    """set_env_var must write values that systemd (EnvironmentFile=) and the application read identically."""
    good = ["plain", "a/b&c|d$e", "it's", 'say "hi"', "back\\slash", "two words", "  lead", "trail  ", "C:\\dir\\x",
            "a#b", "-1001234567890", "x=y", "$HOME", "%n", "pa ss # no comment", "https://h/p?a=1&b=2", "ünï", "",
            "a'b", "'", '"', "x\\yz"]
    (tmp_path / "values.txt").write_text("\n".join(good) + "\n", encoding="utf-8", newline="\n")
    script = HARNESS_HEADER + r"""
source "$LIB/envfile.sh"
f="$WORK/rt.env"
: >"$f"
i=0
while IFS= read -r v; do
  set_env_var "K$i" "$v" "$f"
  i=$((i+1))
done <"$WORK/values.txt"
check count "$i" "$(env_file_keys "$f" | wc -l | tr -d ' ')"
before=$(cat "$f")
no_ both-quotes set_env_var BAD "a'b\"c" "$f"
no_ trailing-backslash-in-quotes set_env_var BAD 'ends\' "$f"
no_ newline set_env_var BAD $'a\nb' "$f"
no_ lowercase-key set_env_var bad v "$f"
check rejected-values-leave-file-untouched "$before" "$(cat "$f")"
check no-leftover-temp-files "0" "$(ls "$WORK" | grep -c '^rt\.env\.' || true)"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    assert_harness_ok(result)
    text = (tmp_path / "rt.env").read_text(encoding="utf-8")
    app_values = _read_env_file(tmp_path / "rt.env")
    systemd_values = systemd_env_parse(text)
    for index, expected in enumerate(good):
        key = f"K{index}"
        assert systemd_values.get(key, "") == expected, (key, expected, systemd_values.get(key))
        assert app_values.get(key, "") == expected, (key, expected, app_values.get(key))


@needs_bash
def test_env_functions(tmp_path: Path):
    template = (ROOT / ".env.example").read_text(encoding="utf-8")
    (tmp_path / "tpl").write_text(template, encoding="utf-8", newline="\n")
    token = "123456789:" + "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"[:30]
    (tmp_path / "seed").write_bytes(
        ("\ufeff# dev file\r\nTOP_N=7\r\nTELEGRAM_BOT_TOKEN=\"" + token + "\"\r\nMONSTER_DB_PATH=C:\\dev\\monster.db\r\n"
         "UNIVERSE_MODE=custom # inline comment\r\nTYPO_KEY=1\r\nexport LOG_LEVEL=DEBUG\r\n").encode())
    script = HARNESS_HEADER + r"""
source "$LIB/envfile.sh"
tpl="$WORK/tpl"; seed="$WORK/seed"; clean="$WORK/seed.clean"
env_sanitize_seed "$seed" "$clean"
check sanitize-no-cr "0" "$(count_cr "$clean")"
check sanitize-no-bom "TOP_N=7" "$(sed -n '2p' "$clean")"

# new file from template + seed: every template key, seed values win, machine-local keys never imported
out="$WORK/created.env"
env_create_from_template "$tpl" "$clean" "$out"
check created-keys "$(env_file_keys "$tpl" | sort | tr '\n' ' ')" "$(env_file_keys "$out" | sort | tr '\n' ' ')"
check top-n "7" "$(env_get TOP_N "$out")"
check export-prefix-handled "DEBUG" "$(env_get LOG_LEVEL "$out")"
check inline-comment-stripped "custom" "$(env_get UNIVERSE_MODE "$out")"
check db-path-not-imported "" "$(env_get MONSTER_DB_PATH "$out")"
check token-imported "$TOKEN" "$(env_get TELEGRAM_BOT_TOKEN "$out")"
check unknown-keys "TYPO_KEY" "$(env_unknown_keys "$clean" "$tpl" | tr '\n' ' ' | sed 's/ $//')"
check secrets-default-empty "" "$(env_get BACKUP_PASSPHRASE "$out")"
check comments-kept "1" "$(grep -c '^# Bot token from @BotFather' "$out")"

# existing file: never overwrite a non-empty value; import only empty/missing; append only missing keys
tgt="$WORK/target.env"
printf 'TOP_N=3\nUNIVERSE_MODE=\nLOG_LEVEL=WARNING\n' >"$tgt"
imported=$(env_import_seed "$clean" "$tgt" "$tpl" | sort | tr '\n' ' ')
check import-only-empty-or-missing "TELEGRAM_BOT_TOKEN UNIVERSE_MODE " "$imported"
check existing-kept "3" "$(env_get TOP_N "$tgt")"
check existing-log-level-kept "WARNING" "$(env_get LOG_LEVEL "$tgt")"
check empty-filled "custom" "$(env_get UNIVERSE_MODE "$tgt")"
before=$(env_file_keys "$tgt" | wc -l | tr -d ' ')
appended=$(env_append_missing "$tpl" "$tgt" | wc -l | tr -d ' ')
total=$(env_file_keys "$tpl" | wc -l | tr -d ' ')
check append-only-missing "$((total - before))" "$appended"
check all-keys-present "$total" "$(env_file_keys "$tgt" | wc -l | tr -d ' ')"
check second-append-is-noop "0" "$(env_append_missing "$tpl" "$tgt" | wc -l | tr -d ' ')"
check existing-still-kept "3" "$(env_get TOP_N "$tgt")"

# redaction: exact secret values and token shapes are masked, nothing else
load_redaction_values "$tgt"
check redact-values-loaded "1" "${#REDACT_VALUES[@]}"
shown=$(printf 'cfg=%s bot%s\n' "$TOKEN" "$TOKEN" | redact_stream)
check redact "cfg=<redacted> bot<redacted>" "$shown"
check describe-secret "<set, 6 chars>" "$(describe_secret abcdef)"
check describe-empty "<empty>" "$(describe_secret '')"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path, TOKEN=token)
    oks = assert_harness_ok(result)
    assert len(oks) >= 20


@needs_bash
def test_decision_helpers(tmp_path: Path):
    script = HARNESS_HEADER + r"""
source "$LIB/decisions.sh"
# swap sizing: skip / already enough / 1 GB / 512 MB when < 3.5 GB would remain / nothing when disk is too small
check swap-existing "0" "$(swap_target_mb 512 9000)"
check swap-1g "1024" "$(swap_target_mb 0 5000)"
check swap-boundary "1024" "$(swap_target_mb 0 4608)"
check swap-below-boundary "512" "$(swap_target_mb 0 4607)"
check swap-small-existing "1024" "$(swap_target_mb 256 6000)"
check swap-tight "512" "$(swap_target_mb 0 3300)"
check swap-too-tight "0" "$(swap_target_mb 0 2500)"
check swap-skipped "0" "$(swap_target_mb 0 9000 1)"
printf 'MemTotal: 1013480 kB\nSwapTotal: 524284 kB\n' >"$WORK/meminfo"
check meminfo-rounds-up "512" "$(meminfo_mb SwapTotal "$WORK/meminfo")"
check meminfo-total "990" "$(meminfo_mb MemTotal "$WORK/meminfo")"

# requirements hash: CRLF-insensitive, sensitive to content, python and arch
mkdir -p "$WORK/d1" "$WORK/d2"
printf 'numpy==1\nrequests==2\n' >"$WORK/d1/requirements.txt"
printf 'numpy==1\r\nrequests==2\r\n' >"$WORK/d2/requirements.txt"
k1=$(requirements_key cp312 x86_64 "$WORK/d1/requirements.txt")
check key-crlf-insensitive "$k1" "$(requirements_key cp312 x86_64 "$WORK/d2/requirements.txt")"
check key-python-sensitive "differs" "$([[ $k1 != "$(requirements_key cp313 x86_64 "$WORK/d1/requirements.txt")" ]] && echo differs || echo same)"
check key-arch-sensitive "differs" "$([[ $k1 != "$(requirements_key cp312 aarch64 "$WORK/d1/requirements.txt")" ]] && echo differs || echo same)"
printf 'numpy==2\n' >"$WORK/d2/requirements.txt"
check key-content-sensitive "differs" "$([[ $k1 != "$(requirements_key cp312 x86_64 "$WORK/d2/requirements.txt")" ]] && echo differs || echo same)"

# venv reuse decision
mkdir -p "$WORK/venv/bin"
printf '#!/bin/sh\nexit 0\n' >"$WORK/venv/bin/python"; chmod +x "$WORK/venv/bin/python"
printf '%s\n' "$k1" >"$WORK/hash"
yes_ reuse-when-hash-matches venv_can_reuse "$k1" "$WORK/hash" "$WORK/venv"
no_ rebuild-when-hash-differs venv_can_reuse "other" "$WORK/hash" "$WORK/venv"
no_ rebuild-when-hash-file-missing venv_can_reuse "$k1" "$WORK/nohash" "$WORK/venv"
printf '#!/bin/sh\nexit 3\n' >"$WORK/venv/bin/python"
no_ rebuild-when-interpreter-broken venv_can_reuse "$k1" "$WORK/hash" "$WORK/venv"

# report window (weekdays 09:00 <= t < 09:16 IST) and the IST conversion
yes_ window-monday-0912 in_report_window 1 0912
yes_ window-start-inclusive in_report_window 5 0900
no_ window-end-exclusive in_report_window 3 0916
no_ window-before in_report_window 3 0859
no_ window-saturday in_report_window 6 0912
check window-wait-seconds "240" "$(seconds_until_window_end 2 0912)"
check window-wait-outside "0" "$(seconds_until_window_end 2 1000)"
check ist-conversion "1 0830" "$(ist_dow_hhmm "$EPOCH_MONDAY_0300Z")"

# time sync plan: exactly one NTP service
check ntp-none "install-chrony" "$(ntp_plan)"
check ntp-timesyncd-kept "keep:systemd-timesyncd" "$(ntp_plan systemd-timesyncd)"
check ntp-both "keep:chrony+disable:systemd-timesyncd" "$(ntp_plan systemd-timesyncd chrony)"
check ntp-many "keep:chrony+warn:multiple" "$(ntp_plan chrony ntp systemd-timesyncd)"

# platform checks
yes_ os-ubuntu-2204 os_supported ubuntu 22.04
yes_ os-ubuntu-2404 os_supported ubuntu 24.04
yes_ os-debian-12 os_supported debian 12
no_ os-ubuntu-2004 os_supported ubuntu 20.04
yes_ arch-x86 arch_supported x86_64
yes_ arch-arm arch_supported aarch64
no_ arch-armv7 arch_supported armv7l
yes_ py-310 py_version_supported 3.10
yes_ py-313 py_version_supported 3.13
no_ py-39 py_version_supported 3.9
no_ py-314 py_version_supported 3.14
yes_ version-ge version_ge 3.10 3.9
no_ version-lt version_ge 3.9 3.10

# CLI output parsing: doctor semantics differ between contract (0/1/2) and implementation (0/1)
check verdict-ok "ok" "$(doctor_verdict 0 '[ ok ] python')"
check verdict-warn-by-marker "warn" "$(doctor_verdict 0 '[warn] swap')"
check verdict-warn-by-code "warn" "$(doctor_verdict 1 '[ ok ] python')"
check verdict-fail-impl "fail" "$(doctor_verdict 1 '[FAIL] configuration')"
check verdict-fail-contract "fail" "$(doctor_verdict 2 'anything')"
check chat-id-bare "424242" "$(parse_chat_id $'bot: @x\n424242')"
check chat-id-prefixed "-1001234567890" "$(parse_chat_id $'bot: @x\nchat: g (supergroup)\nTELEGRAM_CHAT_ID=-1001234567890')"
check chat-id-garbage "" "$(parse_chat_id $'bot: @x\nno message received')"

# retry with exponential backoff (sleep is stubbed)
sleep() { :; }
calls=0; flaky() { calls=$((calls+1)); [[ $calls -ge 3 ]]; }
yes_ retry-succeeds retry 5 1 flaky
check retry-attempts "3" "$calls"
calls=0; hopeless() { calls=$((calls+1)); return 7; }
rc=0; retry 3 1 hopeless >/dev/null || rc=$?
check retry-exit-code "7" "$rc"
check retry-bounded "3" "$calls"
""" + HARNESS_FOOTER
    epoch = str(int(datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc).timestamp()))  # Monday 08:30 IST
    result = run_bash(script, tmp_path, EPOCH_MONDAY_0300Z=epoch)
    oks = assert_harness_ok(result)
    assert len(oks) >= 50


# --- sandboxed file-level logic of install.sh -------------------------------------------------------------
SANDBOX_HEADER = HARNESS_HEADER + r"""
SBX="$WORK/sbx"
mkdir -p "$SBX/home"
export HOME="$SBX/home"   # step_venv removes ~/.cache/pip: never the developer's own
export MONSTER_INSTALL_ROOT="$SBX"
export MONSTER_INSTALL_SOURCE_ONLY=1
source "$PROJECT/deployment/install.sh"
mkdir -p "$UNIT_DIR"
CALLS="$WORK/calls.log"; : >"$CALLS"
FAKE_STATE=inactive
STOPPED=0
# stubs for everything that would touch the real system
systemctl() {
  echo "systemctl $*" >>"$CALLS"
  case "$1" in
    stop) STOPPED=1 ;;
    start | enable) STOPPED=0 ;;
    is-active) if ((STOPPED)); then echo inactive; else echo "$FAKE_STATE"; fi ;;
    show) echo 4242 ;;
  esac
  return 0
}
sleep() { :; }
chown() { :; }
runuser() { shift 3; "$@"; }
install() { local a=(); while (($#)); do case $1 in -o|-g) shift 2 ;; *) a+=("$1"); shift ;; esac; done; command install "${a[@]}"; }
WORKDIR="$WORK/wd"; mkdir -p "$WORKDIR"
make_release() { # make_release DIR LABEL: a minimal installed release tree
  mkdir -p "$1/deployment/systemd" "$1/.venv/bin" "$1/app"
  printf '%s\n' "$2" >"$1/VERSION"
  cp "$PROJECT"/deployment/systemd/* "$1/deployment/systemd/"
  printf '#!/bin/sh\nexit 0\n' >"$1/.venv/bin/python"; chmod +x "$1/.venv/bin/python"
}
"""


def _source_tree_script() -> str:
    return r"""
SRC="$WORK/src"
mkdir -p "$SRC/app" "$SRC/data/seeds" "$SRC/data/runtime" "$SRC/data/cache" "$SRC/logs" "$SRC/tests" "$SRC/.git" \
  "$SRC/.github/workflows" "$SRC/.venv/bin" "$SRC/pkg/__pycache__" "$SRC/deployment"
cp -R "$PROJECT/deployment/." "$SRC/deployment/"
cp "$PROJECT/.env.example" "$SRC/.env.example"
printf 'numpy==1\n' >"$SRC/requirements.txt"; printf 'fastapi==1\n' >"$SRC/requirements-dashboard.txt"
printf 'x = 1\n' >"$SRC/app/__init__.py"; printf 'y = 2\n' >"$SRC/app/cli.py"; printf 'v2\n' >"$SRC/VERSION"
printf '{}' >"$SRC/data/seeds/feeds.json"
printf 'SECRET=1\n' >"$SRC/.env"; printf 'SECRET=2\n' >"$SRC/monster.env"; printf 'x' >"$SRC/release.tar.gz"
printf 'db' >"$SRC/data/runtime/monster.db"; printf 'c' >"$SRC/data/cache/c.bin"; printf 'l' >"$SRC/logs/a.log"
printf 'db' >"$SRC/app/local.db"; printf 'l' >"$SRC/app/x.log"
printf 't' >"$SRC/tests/test_x.py"; printf 'g' >"$SRC/.git/config"; printf 'w' >"$SRC/.github/workflows/ci.yml"
printf 'p' >"$SRC/.venv/bin/python"; printf 'c' >"$SRC/pkg/__pycache__/a.pyc"; printf 'p' >"$SRC/pkg/mod.py"
printf 'p' >"$SRC/pkg/mod.pyc"
printf 'c' >"$SRC/.committed"; printf 'h' >"$SRC/.requirements.sha256"; printf 'n' >"$SRC/pkg/.committed"
"""


@needs_bash
@pytest.mark.slow
def test_release_staging_swap_and_rollback(tmp_path: Path):
    script = SANDBOX_HEADER + _source_tree_script() + r"""
make_release "$APP_DIR" v1

# --- staging: what is copied and what must never be copied
SOURCE_DIR="$SRC"
step_release >"$WORK/step5.out" 2>&1
check new-release-live "v2" "$(cat "$APP_DIR/VERSION")"
check previous-release-kept "v1" "$(cat "$APP_PREV/VERSION")"
check had-previous "1" "$HAD_PREV"
check swapped-flag "1" "$RELEASE_SWAPPED"
check staging-dir-gone "no" "$([[ -e $APP_NEW ]] && echo yes || echo no)"
for keep in app/__init__.py app/cli.py data/seeds/feeds.json .env.example requirements.txt deployment/install.sh \
            deployment/lib/common.sh deployment/systemd/monster-ai.service pkg/mod.py pkg/.committed; do
  check "staged-$keep" "yes" "$([[ -f $APP_DIR/$keep ]] && echo yes || echo no)"
done
for drop in .env monster.env release.tar.gz data/runtime data/cache logs tests .git .github .venv pkg/__pycache__ \
            app/local.db app/x.log pkg/mod.pyc .committed .requirements.sha256; do
  check "excluded-$drop" "no" "$([[ -e $APP_DIR/$drop ]] && echo yes || echo no)"
done
check no-env-files "" "$(find "$APP_DIR" \( -name '.env' -o -name '*.env' \) -print)"
check deployment-scripts-executable "yes" "$([[ -x $APP_DIR/deployment/install.sh || $(uname -s) == MINGW* ]] && echo yes || echo no)"

# --- automatic rollback restores the old tree, keeps the failed one (without its venv)
mkdir -p "$APP_DIR/.venv/bin"; : >"$APP_DIR/.venv/bin/python"
rollback_release "unit test" >"$WORK/rb.out" 2>&1
check rolled-back-live "v1" "$(cat "$APP_DIR/VERSION")"
check rolled-back-flag "1" "$ROLLED_BACK"
check failed-tree-kept "v2" "$(cat "$APP_FAILED/VERSION")"
check failed-venv-dropped "no" "$([[ -e $APP_FAILED/.venv ]] && echo yes || echo no)"
check prev-consumed "no" "$([[ -e $APP_PREV ]] && echo yes || echo no)"
check units-restored-from-old-release "yes" "$([[ -f $UNIT_DIR/monster-ai.service ]] && echo yes || echo no)"
check no-previous-no-rollback "no" "$(rollback_release again >/dev/null 2>&1 && echo yes || echo no)"

# --- an interrupted swap (crash between the two renames) is repaired on the next run
rm -rf "$APP_DIR" "$APP_FAILED"
mkdir -p "$APP_PREV" "$APP_NEW"; echo v1 >"$APP_PREV/VERSION"
recover_interrupted_swap >/dev/null 2>&1
check recovered "v1" "$(cat "$APP_DIR/VERSION")"
check recovery-cleans-staging "no" "$([[ -e $APP_NEW ]] && echo yes || echo no)"

# --- a killed installer leaves an uncommitted release: it must not push the last good one out of .prev
rm -rf "$APP_DIR" "$APP_PREV" "$APP_FAILED" "$APP_NEW"
make_release "$APP_DIR" killed
make_release "$APP_PREV" good; : >"$APP_PREV/.committed"
mkdir -p "$APP_NEW"; echo v3 >"$APP_NEW/VERSION"
HAD_PREV=0; swap_release >"$WORK/sw1.out" 2>&1
check unfinished-new-live "v3" "$(cat "$APP_DIR/VERSION")"
check unfinished-last-good-stays-prev "good" "$(cat "$APP_PREV/VERSION")"
check unfinished-had-prev "1" "$HAD_PREV"
check unfinished-explained "1" "$(grep -c 'unfinished installation' "$WORK/sw1.out" || true)"

# --- a committed release becomes .prev as usual; a first-install leftover without .prev does too
rm -rf "$APP_DIR" "$APP_PREV" "$APP_NEW"
make_release "$APP_DIR" live; : >"$APP_DIR/.committed"
make_release "$APP_PREV" older; : >"$APP_PREV/.committed"
mkdir -p "$APP_NEW"; echo v4 >"$APP_NEW/VERSION"
swap_release >"$WORK/sw2.out" 2>&1
check committed-becomes-prev "live" "$(cat "$APP_PREV/VERSION")"
check committed-new-live "v4" "$(cat "$APP_DIR/VERSION")"
check committed-marker-not-inherited "no" "$([[ -e $APP_DIR/.committed ]] && echo yes || echo no)"
rm -rf "$APP_DIR" "$APP_PREV" "$APP_NEW"
make_release "$APP_DIR" leftover
mkdir -p "$APP_NEW"; echo v5 >"$APP_NEW/VERSION"
swap_release >"$WORK/sw3.out" 2>&1
check leftover-without-prev-becomes-prev "leftover" "$(cat "$APP_PREV/VERSION")"
check leftover-new-live "v5" "$(cat "$APP_DIR/VERSION")"
mark_committed
check mark-committed-flag "1" "$RELEASE_COMMITTED"
check mark-committed-file "yes" "$([[ -f $APP_DIR/.committed ]] && echo yes || echo no)"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    oks = assert_harness_ok(result)
    assert len(oks) >= 45


@needs_bash
def test_manual_rollback_swaps_releases(tmp_path: Path):
    script = SANDBOX_HEADER + r"""
make_release "$APP_DIR" v2
make_release "$APP_PREV" v1
FAKE_STATE=active
main_rollback >"$WORK/mr.out" 2>&1 || { cat "$WORK/mr.out"; fail=1; }
check live-is-old "v1" "$(cat "$APP_DIR/VERSION")"
check old-is-prev-now "v2" "$(cat "$APP_PREV/VERSION")"
check no-failed-dir-left "no" "$([[ -e $APP_FAILED ]] && echo yes || echo no)"
check daemon-restart-requested "yes" "$(grep -q 'systemctl daemon-reload' "$CALLS" && echo yes || echo no)"
check committed "1" "$RELEASE_COMMITTED"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    assert_harness_ok(result)


@needs_bash
def test_apt_busy_follows_the_lock_files_not_the_resident_helper(tmp_path: Path):
    """On a stock Ubuntu server the idle ``unattended-upgrade-shutdown`` helper (comm ``unattended-upgr``) always runs. The
    installer once took it for an upgrade in progress, waited the whole timeout on a fresh server and failed; it must wait only
    while a package manager really holds the apt/dpkg locks."""
    script = SANDBOX_HEADER + r"""
PROCS=()   # the fake process table: "comm|command line"
pgrep() {  # pgrep -x REGEX (the process name) and pgrep -f REGEX (the command line) against PROCS
  local mode=$1 re=$2 line comm cmd
  for line in ${PROCS[@]+"${PROCS[@]}"}; do
    comm=${line%%|*}; cmd=${line#*|}
    case $mode in
      -x) [[ $comm =~ ^($re)$ ]] && return 0 ;;
      -f) [[ $cmd =~ $re ]] && return 0 ;;
    esac
  done
  return 1
}
HELPER="unattended-upgr|/usr/bin/python3 /usr/share/unattended-upgrades/unattended-upgrade-shutdown --wait-for-signal"

# with fuser: the lock files decide, whatever the process names say
PROCS=("$HELPER")
fuser() { printf '%s\n' "$*" >"$WORK/fuser.args"; return 1; }
no_ idle-server-with-the-helper-is-not-busy apt_busy
check fuser-asked-about-the-locks "-s ${APT_LOCK_FILES[*]}" "$(cat "$WORK/fuser.args")"
check lock-files "/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock" "${APT_LOCK_FILES[*]}"
fuser() { return 0; }
yes_ lock-held-is-busy apt_busy

# without psmisc the names decide, with the helper excluded
unset -f fuser
command() { if [[ ${1:-} == -v && ${2:-} == fuser ]]; then return 1; fi; builtin command "$@"; }
PROCS=()
no_ no-fuser-nothing-running "apt_busy"
PROCS=("$HELPER")
no_ no-fuser-the-helper-alone-is-not-busy apt_busy
PROCS=("$HELPER" "unattended-upgr|/usr/bin/python3 /usr/bin/unattended-upgrade")
yes_ no-fuser-a-real-unattended-upgrade-is-busy apt_busy
PROCS=("apt-get|apt-get -y install libgomp1")
yes_ no-fuser-apt-get-is-busy apt_busy
PROCS=("dpkg|/usr/bin/dpkg --configure -a")
yes_ no-fuser-dpkg-is-busy apt_busy
PROCS=("packagekitd|/usr/libexec/packagekitd")
yes_ no-fuser-packagekit-is-busy apt_busy
unset -f command
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    assert_harness_ok(result)


@needs_bash
def test_connectivity_probe_asks_like_a_browser(tmp_path: Path):
    """NSE's CDN gives no answer at all to curl's own User-Agent (the installer reported a blocked NSE that was not blocked), so the
    preflight probe sends a browser's and still treats any HTTP answer as proof that the route works."""
    script = SANDBOX_HEADER + r"""
curl() { printf '%s\n' "$*" >>"$CALLS"; printf '404'; }
probe_endpoint "NSE archives" https://nsearchives.nseindia.com/ optional hint >"$WORK/probe.out" 2>&1
check probe-sends-a-browser-user-agent "yes" "$(grep -q -- '-A Mozilla/5.0 (X11; Linux x86_64)' "$CALLS" && echo yes || echo no)"
check any-http-answer-is-reachable "yes" "$(grep -q 'reachable (HTTP 404)' "$WORK/probe.out" && echo yes || echo no)"
curl() { printf '000'; return 7; }
probe_endpoint "NSE archives" https://nsearchives.nseindia.com/ optional "hint text" >"$WORK/probe2.out" 2>&1
check no-answer-is-a-warning "yes" "$(grep -q 'is unreachable. hint text' "$WORK/probe2.out" && echo yes || echo no)"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    assert_harness_ok(result)


FAKE_PYTHON = r"""#!/usr/bin/env bash
# Fake interpreter: records what the installer asks pip / venv / compileall to do.
LOG="${FAKE_LOG:?}"
case "$*" in
  "-m venv "*) mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; chmod +x "$3/bin/python"; echo "venv $3" >>"$LOG" ;;
  "-m pip install"*) echo "pip $*" >>"$LOG" ;;
  "-m pip check"*) echo "No broken requirements found." ;;
  "-m compileall"*) echo "compileall $*" >>"$LOG" ;;
  -c*) : ;;
  *) echo "unexpected call: $*" >&2; exit 9 ;;
esac
"""


@needs_bash
@pytest.mark.slow
def test_virtualenv_reuse_and_binary_wheel_policy(tmp_path: Path):
    (tmp_path / "fakepy").write_text(FAKE_PYTHON, encoding="utf-8", newline="\n")
    script = SANDBOX_HEADER + r"""
chmod +x "$WORK/fakepy"
export FAKE_LOG="$WORK/py.log"; : >"$FAKE_LOG"
PY_BIN="$WORK/fakepy"; PY_VER=3.12; PY_TAG=cp312; ARCH=x86_64
write_release() { make_release "$1" "$2"; printf 'numpy==1\n' >"$1/requirements.txt"; printf 'fastapi==1\n' >"$1/requirements-dashboard.txt"; }
rm -rf "$APP_DIR" "$APP_PREV"

# --- fresh install: no previous venv -> build from binary wheels only
write_release "$APP_DIR" v2
rm -rf "$APP_DIR/.venv"
DASHBOARD_WANTED=0
step_venv >"$WORK/venv1.out" 2>&1 || { cat "$WORK/venv1.out"; fail=1; }
pipline=$(grep '^pip .* -r ' "$FAKE_LOG" | head -n 1)
check build-uses-only-binary "yes" "$([[ $pipline == *--only-binary=:all:* ]] && echo yes || echo no)"
check build-exception-is-sgmllib3k "yes" "$([[ $pipline == *--no-binary=sgmllib3k* ]] && echo yes || echo no)"
check build-never-all-source "no" "$([[ $pipline == *--no-binary=:all:* ]] && echo yes || echo no)"
check build-no-cache "yes" "$([[ $pipline == *--no-cache-dir* ]] && echo yes || echo no)"
check build-requirements-file "yes" "$([[ $pipline == *"-r $APP_DIR/requirements.txt"* ]] && echo yes || echo no)"
check build-without-dashboard "no" "$([[ $pipline == *requirements-dashboard* ]] && echo yes || echo no)"
check pip-upgraded-first "yes" "$(grep -q -- '--upgrade pip' "$FAKE_LOG" && echo yes || echo no)"
check precompiled "yes" "$(grep -q '^compileall' "$FAKE_LOG" && echo yes || echo no)"
check hash-written "64" "$(tr -d '[:space:]' <"$HASH_FILE" | wc -c | tr -d ' ')"
first_hash=$(tr -d '[:space:]' <"$HASH_FILE")

# --- upgrade with unchanged requirements: venv is reused (hard-linked), pip is not run again
rm -rf "$APP_PREV"; mv "$APP_DIR" "$APP_PREV"
write_release "$APP_DIR" v3
rm -rf "$APP_DIR/.venv"
: >"$FAKE_LOG"
step_venv >"$WORK/venv2.out" 2>&1 || { cat "$WORK/venv2.out"; fail=1; }
check reuse-no-pip-install "0" "$(grep -c '^pip ' "$FAKE_LOG" || true)"
check reuse-no-new-venv "0" "$(grep -c '^venv ' "$FAKE_LOG" || true)"
check reuse-venv-present "yes" "$([[ -x $VENV_PY ]] && echo yes || echo no)"
check previous-keeps-a-working-venv "yes" "$([[ -x $APP_PREV/.venv/bin/python ]] && echo yes || echo no)"
check reuse-same-hash "$first_hash" "$(tr -d '[:space:]' <"$HASH_FILE")"

# --- changed requirements -> rebuilt
rm -rf "$APP_PREV"; mv "$APP_DIR" "$APP_PREV"
write_release "$APP_DIR" v4
printf 'numpy==2\n' >"$APP_DIR/requirements.txt"
rm -rf "$APP_DIR/.venv"
: >"$FAKE_LOG"
step_venv >"$WORK/venv3.out" 2>&1 || { cat "$WORK/venv3.out"; fail=1; }
check changed-requirements-rebuilds "1" "$(grep -c '^venv ' "$FAKE_LOG" || true)"
check changed-hash "differs" "$([[ $(tr -d '[:space:]' <"$HASH_FILE") != "$first_hash" ]] && echo differs || echo same)"

# --- dashboard enabled: its requirements are installed too and change the hash
rm -rf "$APP_PREV"; mv "$APP_DIR" "$APP_PREV"
write_release "$APP_DIR" v5
rm -rf "$APP_DIR/.venv"
: >"$FAKE_LOG"
DASHBOARD_WANTED=1
step_venv >"$WORK/venv4.out" 2>&1 || { cat "$WORK/venv4.out"; fail=1; }
check dashboard-requirements-installed "yes" "$(grep '^pip .* -r ' "$FAKE_LOG" | grep -q 'requirements-dashboard.txt' && echo yes || echo no)"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    oks = assert_harness_ok(result)
    assert len(oks) >= 17


FAKE_CLI_PYTHON = r"""#!/usr/bin/env bash
# Fake interpreter running a fake `python -m app.cli <command>`: the output / exit code of each command is
# read from $MONSTER_DATA_DIR/fake/<command>.{out,rc}; invocations are logged.
fake="$MONSTER_DATA_DIR/fake"
[[ $1 == -m && $2 == app.cli ]] || exit 9
cmd=$3
echo "$cmd ${*:4}" >>"$fake/invocations.log"
[[ -f $fake/$cmd.out ]] && cat "$fake/$cmd.out"
if [[ -f $fake/$cmd.err ]]; then cat "$fake/$cmd.err" >&2; fi
rc=0; [[ -f $fake/$cmd.rc ]] && rc=$(cat "$fake/$cmd.rc")
exit "$rc"
"""


@needs_bash
@pytest.mark.slow
def test_configuration_step_creates_merges_and_never_prints_secrets(tmp_path: Path):
    token = "987654321:" + "ZyXwVuTsRqPoNmLkJiHgFeDcBa9876543210"[:30]
    (tmp_path / "seed.env").write_text(
        f"# my dev file\nTELEGRAM_BOT_TOKEN={token}\nTOP_N=7\nMONSTER_DATA_DIR=C:\\dev\\data\nDASHBOARD_ENABLED=false\n",
        encoding="utf-8", newline="\n")
    script = SANDBOX_HEADER + r"""
mkdir -p "$APP_DIR" "$CONF_DIR"
cp "$PROJECT/.env.example" "$APP_DIR/.env.example"
ENV_SEED_ARG="$WORK/seed.env"; DRY_RUN=0; ENABLE_DASHBOARD=0
SEED_FILE="$WORKDIR/seed.env"; env_sanitize_seed "$ENV_SEED_ARG" "$SEED_FILE"

# --- first install: created from the seed, trace on to prove that secrets never reach xtrace either
( set -x; step_envfile ) >"$WORK/env1.out" 2>"$WORK/env1.err" || { cat "$WORK/env1.out" "$WORK/env1.err"; fail=1; }
check created "yes" "$([[ -f $ENV_FILE ]] && echo yes || echo no)"
check token-stored "$TOKEN" "$(env_get TELEGRAM_BOT_TOKEN "$ENV_FILE")"
check seed-value-imported "7" "$(env_get TOP_N "$ENV_FILE")"
check forced-env "production" "$(env_get MONSTER_ENV "$ENV_FILE")"
check forced-home "$APP_DIR" "$(env_get MONSTER_HOME "$ENV_FILE")"
check forced-data "$DATA_DIR" "$(env_get MONSTER_DATA_DIR "$ENV_FILE")"
check forced-logs "$LOG_DIR" "$(env_get MONSTER_LOG_DIR "$ENV_FILE")"
check foreign-path-not-imported "$DATA_DIR" "$(env_get MONSTER_DATA_DIR "$ENV_FILE")"
if [[ $(uname -s) != MINGW* && $(uname -s) != MSYS* ]]; then check mode "640" "$(stat -c %a "$ENV_FILE")"; fi
check token-not-in-stdout "0" "$(grep -c -F "$TOKEN" "$WORK/env1.out" || true)"
check token-not-in-xtrace "0" "$(grep -c -F "$TOKEN" "$WORK/env1.err" || true)"
check secret-summary-shown "1" "$(grep -c 'TELEGRAM_BOT_TOKEN .*<set, ' "$WORK/env1.out" || true)"

# --- re-run with another seed: existing values survive, empty ones are filled, new keys appended
printf 'TOP_N=9\nTELEGRAM_CHAT_ID=555\nBACKUP_PASSPHRASE=newsecretvalue\n' >"$WORK/seed2.env"
ENV_SEED_ARG="$WORK/seed2.env"; SEED_FILE="$WORKDIR/seed2.env"; env_sanitize_seed "$ENV_SEED_ARG" "$SEED_FILE"
printf 'EXTRA_FROM_NEWER_RELEASE=1\n' >>"$APP_DIR/.env.example"
step_envfile >"$WORK/env2.out" 2>&1
check existing-top-n-kept "7" "$(env_get TOP_N "$ENV_FILE")"
check existing-token-kept "$TOKEN" "$(env_get TELEGRAM_BOT_TOKEN "$ENV_FILE")"
check empty-chat-id-filled "555" "$(env_get TELEGRAM_CHAT_ID "$ENV_FILE")"
check empty-passphrase-filled "newsecretvalue" "$(env_get BACKUP_PASSPHRASE "$ENV_FILE")"
check new-key-appended "1" "$(env_get EXTRA_FROM_NEWER_RELEASE "$ENV_FILE")"
check passphrase-not-printed "0" "$(grep -c -F newsecretvalue "$WORK/env2.out" || true)"
check one-assignment-per-key "0" "$(env_file_keys "$ENV_FILE" | sort | uniq -d | wc -l | tr -d ' ')"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path, TOKEN=token)
    oks = assert_harness_ok(result)
    assert len(oks) >= 18
    assert token not in result.stdout + result.stderr


@needs_bash
@pytest.mark.slow
def test_telegram_and_validation_steps_with_a_fake_cli(tmp_path: Path):
    (tmp_path / "fakecli").write_text(FAKE_CLI_PYTHON, encoding="utf-8", newline="\n")
    token = "555000111:" + "QwErTyUiOpAsDfGhJkLzXcVbNm0123456789"[:30]
    script = SANDBOX_HEADER + r"""
chmod +x "$WORK/fakecli"
mkdir -p "$APP_DIR/.venv/bin" "$DATA_DIR/fake" "$CONF_DIR"
cp "$WORK/fakecli" "$APP_DIR/.venv/bin/python"
fake() { printf '%s' "$2" >"$DATA_DIR/fake/$1"; }
reset_fake() { rm -f "$DATA_DIR"/fake/*; : >"$DATA_DIR/fake/invocations.log"; }
env_file_with() { printf '%s\n' "$@" >"$ENV_FILE"; }
REDACT_VALUES=()

# --- no token at all: loud warning naming the file, the key and the command to re-run; never a failure
reset_fake; env_file_with 'TELEGRAM_BOT_TOKEN=' 'TELEGRAM_CHAT_ID='
WARNINGS=()
step_telegram >"$WORK/t1.out" 2>&1
check no-token-state "no-token" "$TELEGRAM_STATE"
check no-token-names-file "yes" "$(grep -qF "$ENV_FILE" "$WORK/t1.out" && echo yes || echo no)"
check no-token-names-key "yes" "$(grep -q 'TELEGRAM_BOT_TOKEN=' "$WORK/t1.out" && echo yes || echo no)"
check no-token-names-command "yes" "$(grep -q 'deployment/install.sh --yes' "$WORK/t1.out" && echo yes || echo no)"
check no-token-warns-loudly "yes" "$(grep -q 'WARN.*CANNOT be delivered' "$WORK/t1.out" && echo yes || echo no)"
check no-token-cli-not-called "0" "$(grep -c telegram-setup "$DATA_DIR/fake/invocations.log" || true)"

# --- token + chat id present: nothing to discover
reset_fake; env_file_with "TELEGRAM_BOT_TOKEN=$TOKEN" 'TELEGRAM_CHAT_ID=77'
step_telegram >"$WORK/t2.out" 2>&1
check configured "configured" "$TELEGRAM_STATE"
check no-discovery-needed "0" "$(grep -c telegram-setup "$DATA_DIR/fake/invocations.log" || true)"

# --- token only: the chat id is discovered (the CLI prints TELEGRAM_CHAT_ID=<id> as its last line)
reset_fake; env_file_with "TELEGRAM_BOT_TOKEN=$TOKEN" 'TELEGRAM_CHAT_ID='
fake telegram-setup.out $'bot: @my_bot\nchat: Me (private)\nTELEGRAM_CHAT_ID=424242\n'
fake telegram-setup.err $'Open Telegram, find @my_bot and send it any message\n'
step_telegram >"$WORK/t3.out" 2>&1
check discovered-state "configured" "$TELEGRAM_STATE"
check discovered-stored "424242" "$(env_get TELEGRAM_CHAT_ID "$ENV_FILE")"
check token-kept "$TOKEN" "$(env_get TELEGRAM_BOT_TOKEN "$ENV_FILE")"
check prompt-shown "1" "$(grep -c 'Send any message to your bot now (waiting up to 180 s)' "$WORK/t3.out" || true)"
check setup-args "telegram-setup --print-only --wait 180" "$(grep telegram-setup "$DATA_DIR/fake/invocations.log")"
check token-not-printed "0" "$(grep -c -F "$TOKEN" "$WORK/t3.out" || true)"

# --- group chats have negative ids; the bare-number contract form works as well
reset_fake; env_file_with "TELEGRAM_BOT_TOKEN=$TOKEN" 'TELEGRAM_CHAT_ID='
fake telegram-setup.out $'-1001234567890\n'
step_telegram >"$WORK/t4.out" 2>&1
check negative-id "-1001234567890" "$(env_get TELEGRAM_CHAT_ID "$ENV_FILE")"

# --- nobody wrote to the bot: not fatal, chat id stays empty, the next step is explained
reset_fake; env_file_with "TELEGRAM_BOT_TOKEN=$TOKEN" 'TELEGRAM_CHAT_ID='
fake telegram-setup.rc 1
step_telegram >"$WORK/t5.out" 2>&1
check timeout-state "no-chat-id" "$TELEGRAM_STATE"
check chat-id-still-empty "" "$(env_get TELEGRAM_CHAT_ID "$ENV_FILE")"
check timeout-advice "1" "$(grep -c 'install.sh --yes' "$WORK/t5.out" || true)"

# --- doctor: ok / warnings continue, failure stops with the application's own text
reset_fake; env_file_with 'TOP_N=5'
fake doctor.out $'[ ok ] python  3.12\n[warn] swap  no swap\n\nREADY (warnings, if any, are optional improvements)'
step_validate_config >"$WORK/v1.out" 2>&1
check doctor-warnings-continue "yes" "$(grep -q 'optional improvements' "$WORK/v1.out" && echo yes || echo no)"
reset_fake; env_file_with 'TOP_N=5'
fake doctor.out $'[FAIL] configuration  invalid configuration: TOP_N: boom\n\nNOT READY: fix the [FAIL] items above'
fake doctor.rc 1
rc=0; ( step_validate_config >"$WORK/v2.out" 2>&1 ) || rc=$?
check doctor-failure-aborts "1" "$rc"
check doctor-own-text-shown "yes" "$(grep -q 'invalid configuration: TOP_N: boom' "$WORK/v2.out" && echo yes || echo no)"
check doctor-failure-tries-init-once "1" "$(grep -c '^init' "$DATA_DIR/fake/invocations.log" || true)"
reset_fake; env_file_with 'TOP_N=5'
fake doctor.out $'[ ok ] python 3.12'
step_validate_config >"$WORK/v3.out" 2>&1
check doctor-ok "yes" "$(grep -q 'all checks passed' "$WORK/v3.out" && echo yes || echo no)"
check doctor-run-offline "doctor --offline" "$(grep '^doctor' "$DATA_DIR/fake/invocations.log" | head -n 1)"

# --- init failure is fatal and shows the app's message
reset_fake; fake init.out $'configuration error: bad\n'; fake init.rc 2
rc=0; ( step_initialise >"$WORK/i1.out" 2>&1 ) || rc=$?
check init-failure-aborts "1" "$rc"
check init-message-shown "1" "$(grep -c 'configuration error: bad' "$WORK/i1.out" || true)"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path, TOKEN=token)
    oks = assert_harness_ok(result)
    assert len(oks) >= 25
    assert token not in result.stdout


@needs_bash
def test_stage_excludes_are_root_anchored_where_needed(tmp_path: Path):
    """Nested packages named like excluded top-level folders are kept; secrets are dropped everywhere."""
    script = SANDBOX_HEADER + r"""
SRC="$WORK/src2"
mkdir -p "$SRC/pkg/logs" "$SRC/pkg/tests" "$SRC/logs" "$SRC/tests" "$SRC/sub"
printf 'k' >"$SRC/pkg/logs/__init__.py"; printf 'k' >"$SRC/pkg/tests/__init__.py"
printf 'x' >"$SRC/logs/a"; printf 'x' >"$SRC/tests/a"; printf 'S' >"$SRC/sub/.env"; printf 'S' >"$SRC/sub/prod.env"
printf 'e' >"$SRC/.env.example"
stage_tree "$SRC" "$WORK/staged"
check nested-logs-package-kept "yes" "$([[ -f $WORK/staged/pkg/logs/__init__.py ]] && echo yes || echo no)"
check nested-tests-package-kept "yes" "$([[ -f $WORK/staged/pkg/tests/__init__.py ]] && echo yes || echo no)"
check top-logs-dropped "no" "$([[ -e $WORK/staged/logs ]] && echo yes || echo no)"
check top-tests-dropped "no" "$([[ -e $WORK/staged/tests ]] && echo yes || echo no)"
check env-example-kept "yes" "$([[ -f $WORK/staged/.env.example ]] && echo yes || echo no)"
check nested-env-dropped "no" "$([[ -e $WORK/staged/sub/.env || -e $WORK/staged/sub/prod.env ]] && echo yes || echo no)"
mkdir -p "$WORK/bad/x"; printf 'S' >"$WORK/bad/x/leak.env"
no_ guard-detects-leaked-env assert_no_env_files "$WORK/bad"
yes_ guard-accepts-clean-tree assert_no_env_files "$WORK/staged"
# line endings: CRLF in scripts, units and configs is repaired, other files are left alone
mkdir -p "$WORK/crlf"; printf 'a\r\nb\r\n' >"$WORK/crlf/x.service"; printf 'a\r\n' >"$WORK/crlf/x.sh"
printf 'a\r\n' >"$WORK/crlf/logrotate-monster"; printf 'a\r\n' >"$WORK/crlf/data.txt"
normalize_line_endings "$WORK/crlf"
check crlf-service "0" "$(count_cr "$WORK/crlf/x.service")"
check crlf-sh "0" "$(count_cr "$WORK/crlf/x.sh")"
check crlf-logrotate "0" "$(count_cr "$WORK/crlf/logrotate-monster")"
check crlf-other-untouched "1" "$(count_cr "$WORK/crlf/data.txt")"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    assert_harness_ok(result)


@needs_bash
def test_install_and_uninstall_entry_points_handle_arguments(tmp_path: Path):
    for script, argument, code in (("install.sh", "--bogus", 2), ("uninstall.sh", "--bogus", 2),
                                   ("install.sh", "--help", 0), ("uninstall.sh", "--help", 0),
                                   ("deploy_remote.sh", "--help", 0), ("deploy_remote.sh", "--port", 2)):
        result = subprocess.run([BASH, str(DEPLOY / script), argument], capture_output=True, text=True, check=False,
                                env=_bash_env({}), timeout=60)
        assert result.returncode == code, (script, argument, result.stderr)
    result = subprocess.run([BASH, str(DEPLOY / "install.sh"), "--env-file", str(tmp_path / "missing.env")],
                            capture_output=True, text=True, check=False, env=_bash_env({}), timeout=60)
    assert result.returncode == 2 and "no such file" in result.stderr
    result = subprocess.run([BASH, str(DEPLOY / "deploy_remote.sh"), "bad host"], capture_output=True, text=True,
                            check=False, env=_bash_env({}), timeout=60)
    assert result.returncode == 2
    result = subprocess.run([BASH, str(DEPLOY / "deploy_remote.sh"), "-evil@host"], capture_output=True, text=True,
                            check=False, env=_bash_env({}), timeout=60)
    assert result.returncode == 2, "a target starting with '-' could inject ssh options"


def test_feed_summary_snippet_reads_doctor_json():
    match = re.search(r"readonly FEEDS_SUMMARY_PY='\n(.*?)\n'\n", read(DEPLOY / "install.sh"), re.S)
    assert match, "FEEDS_SUMMARY_PY is missing from install.sh"
    snippet = match.group(1)

    def run(stdin: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-c", snippet], input=stdin, capture_output=True, text=True,
                              check=False, timeout=60)

    payload = {"ok": False, "checks": [
        {"name": "python", "status": "ok", "detail": "3.12"},
        {"name": "network: Yahoo Finance", "status": "warn", "detail": "timed out"},
        {"name": "network: NSE archives", "status": "ok", "detail": "HTTP 200"},
        {"name": "telegram bot", "status": "fail", "detail": "401 Unauthorized"},
        {"name": "system clock", "status": "ok", "detail": "skew 0.2 s"},
        "not a check",
    ]}
    good = run(json.dumps(payload))
    assert good.returncode == 0, good.stderr
    lines = good.stdout.splitlines()
    assert len(lines) == 4, lines
    assert not any(line.startswith("python") for line in lines)
    assert any("network: Yahoo Finance" in line and "warn" in line and "timed out" in line for line in lines)
    assert run("this is not json").returncode == 3
    assert run(json.dumps({"ok": True, "checks": []})).returncode == 4
    assert run(json.dumps(["unexpected"])).returncode == 4


@needs_bash
@pytest.mark.slow
def test_bootstrap_and_verification_steps_with_stubs(tmp_path: Path):
    """Step 12 (start, watch, give up politely) and step 13 (health / Telegram verdicts) against stubs."""
    script = SANDBOX_HEADER + r"""
# Git Bash forks slowly (~0.15 s per $(...), ~0.3 s per external command): the stubs and helpers below are
# bash builtins only.
has() { local t; t=$(<"$1"); [[ $t == *"$2"* ]]; }
count_in() { local l; N=0; while IFS= read -r l; do if [[ $l == *"$2"* ]]; then N=$((N + 1)); fi; done <"$1"; }
count_eq() { local l; N=0; while IFS= read -r l; do if [[ $l == "$2" ]]; then N=$((N + 1)); fi; done <"$1"; }
FAKE="$DATA_DIR/fake"; mkdir -p "$FAKE"
fake() { printf '%s' "$2" >"$FAKE/$1"; }
reset_fake() { rm -f "$FAKE"/*; : >"$FAKE/invocations.log"; }
reset_fake
# `runuser ... python -m app.cli <cmd>` replaced by a function (the real call path is covered by the Telegram test)
run_as_monster() {
  local -a a=("$@"); local cmd rc=0 t
  if [[ ${a[0]} == timeout ]]; then a=("${a[@]:2}"); fi
  if [[ ${a[1]:-} == -c ]]; then return 9; fi
  cmd=${a[3]}
  printf '%s\n' "${a[*]:3}" >>"$FAKE/invocations.log"
  if [[ -f $FAKE/$cmd.out ]]; then t=$(<"$FAKE/$cmd.out"); printf '%s\n' "$t"; fi
  if [[ -f $FAKE/$cmd.rc ]]; then rc=$(<"$FAKE/$cmd.rc"); fi
  return "$rc"
}
indent() { local l; while IFS= read -r l; do printf '        %s\n' "$l"; done; }
redact_stream() { local l; while IFS= read -r l; do printf '%s\n' "$l"; done; }
cut() { local l; IFS= read -r l && printf '%s\n' "${l:0:150}"; }
journalctl() { echo "2025-01-01T09:00:00+0530 monster-ai[1]: history backfill 120/500 symbols"; }
REDACT_VALUES=()

# the application state is a queue: every call pops one value, the last one repeats
set_states() { printf '%s\n' "$@" >"$WORK/states"; }
bootstrap_state() {
  local -a q
  mapfile -t q <"$WORK/states"
  echo "${q[0]}"
  if ((${#q[@]} > 1)); then printf '%s\n' "${q[@]:1}" >"$WORK/states"; fi
}
FAKE_BOOT_UNIT=inactive
svc_state() { if [[ $1 == monster-ai-bootstrap.service ]]; then echo "$FAKE_BOOT_UNIT"; else echo "$FAKE_STATE"; fi; }
# time moves only through sleep, so the two-hour limit costs two loop iterations
FAKE_NOW=1700000000; TICK=15
date() { if [[ ${1:-} == +%s ]]; then echo "$FAKE_NOW"; else command date "$@"; fi; }
sleep() { FAKE_NOW=$((FAKE_NOW + TICK)); }
starts() { count_in "$CALLS" 'systemctl start --no-block monster-ai-bootstrap.service'; }
scenario() { : >"$CALLS"; BOOTSTRAP_RESULT=skipped; NO_BOOTSTRAP=0; NO_START=0; DRY_RUN=0; TICK=15; FAKE_BOOT_UNIT=inactive; }

# --- fresh host: started detached, then watched until the database says done
scenario; set_states none running running done
step_bootstrap >"$WORK/b1.out" 2>&1
check fresh-result done "$BOOTSTRAP_RESULT"
starts; check fresh-started-once-detached 1 "$N"
count_in "$WORK/b1.out" '[bootstrap +'; check fresh-polled-until-done 3 "$N"
yes_ fresh-progress-shows-the-log has "$WORK/b1.out" 'history backfill 120/500'
yes_ fresh-ssh-drop-is-harmless has "$WORK/b1.out" 'Ctrl-C only stops watching'
yes_ fresh-announces-success has "$WORK/b1.out" 'bootstrap finished'

# --- already completed on this host: nothing to start
scenario; set_states done
step_bootstrap >"$WORK/b2.out" 2>&1
check done-result done "$BOOTSTRAP_RESULT"
starts; check done-not-started 0 "$N"

# --- the daemon launched the (resumable) job itself right after READY: watch it, never start a second one
scenario; set_states running running done
step_bootstrap >"$WORK/b3.out" 2>&1
check daemon-run-result done "$BOOTSTRAP_RESULT"
starts; check daemon-run-not-duplicated 0 "$N"
yes_ daemon-run-explained has "$WORK/b3.out" 'started by the daemon'

# --- the unit itself is already running
scenario; set_states none running done; FAKE_BOOT_UNIT=activating
step_bootstrap >"$WORK/b4.out" 2>&1
starts; check unit-run-not-duplicated 0 "$N"
check unit-run-result done "$BOOTSTRAP_RESULT"

# --- failure: not fatal for the installer, last 40 log lines, promise of automatic retries
scenario; set_states none running failed
rc=0; step_bootstrap >"$WORK/b5.out" 2>&1 || rc=$?
check failed-result failed "$BOOTSTRAP_RESULT"
check failed-not-fatal 0 "$rc"
yes_ failed-shows-last-40-lines has "$WORK/b5.out" 'last 40 journal lines'
yes_ failed-promises-retry has "$WORK/b5.out" 'keeps retrying automatically'

# --- the unit failed and the database is unreadable: still a failure; the failed state is reset first
scenario; set_states none unknown; FAKE_BOOT_UNIT=failed
rc=0; step_bootstrap >"$WORK/b6.out" 2>&1 || rc=$?
check unit-failed-result failed "$BOOTSTRAP_RESULT"
yes_ unit-failed-reset has "$CALLS" 'systemctl reset-failed monster-ai-bootstrap.service'

# --- a finished run (RemainAfterExit) is stopped first so that start really runs it again; an unreadable
# database after the 45 s grace period is read as "the oneshot finished"
scenario; set_states none unknown; FAKE_BOOT_UNIT=active
step_bootstrap >"$WORK/b7.out" 2>&1
yes_ finished-run-stopped-first has "$CALLS" 'systemctl stop monster-ai-bootstrap.service'
starts; check finished-run-started 1 "$N"
check unit-finished-result done "$BOOTSTRAP_RESULT"

# --- two hours without a result: stop watching, keep running
scenario; set_states none running; TICK=8000
rc=0; step_bootstrap >"$WORK/b8.out" 2>&1 || rc=$?
check timeout-result timeout "$BOOTSTRAP_RESULT"
check timeout-not-fatal 0 "$rc"
yes_ timeout-keeps-running has "$WORK/b8.out" 'continues in the background'

# --- flags
scenario; NO_BOOTSTRAP=1; set_states none
step_bootstrap >"$WORK/b9.out" 2>&1
check no-bootstrap-skipped skipped "$BOOTSTRAP_RESULT"
starts; check no-bootstrap-nothing-started 0 "$N"
scenario; NO_START=1; set_states none
step_bootstrap >"$WORK/b10.out" 2>&1
check no-start-skipped skipped "$BOOTSTRAP_RESULT"
scenario
FAKE_STATE=active

# =========================================================================================== step 13
verify() { # verify TELEGRAM_STATE BOOTSTRAP_RESULT OUTFILE
  TELEGRAM_STATE=$1; BOOTSTRAP_RESULT=$2; HEALTH_NOTE=""; WARNINGS=()
  step_verify >"$3" 2>&1
}
reset_fake; fake health.out '{"level":"ok"}'; fake doctor.out '{"ok":true,"checks":[]}'
verify configured done "$WORK/s1.out"
check health-ok OK "$HEALTH_NOTE"
count_eq "$FAKE/invocations.log" 'health --json'; check asks-for-json 1 "$N"
count_eq "$FAKE/invocations.log" telegram-test; check telegram-test-sent 1 "$N"
yes_ telegram-delivered has "$WORK/s1.out" 'test message delivered'
count_eq "$FAKE/invocations.log" 'doctor --json'; check reads-doctor-json 1 "$N"

reset_fake; fake health.rc 1
verify configured done "$WORK/s2.out"
check health-warning WARNING "$HEALTH_NOTE"
count_eq "$FAKE/invocations.log" health; check health-warning-shows-details 1 "$N"

reset_fake; fake health.rc 2
verify configured running "$WORK/s3.out"
check health-critical CRITICAL "$HEALTH_NOTE"
yes_ critical-is-expected-during-bootstrap has "$WORK/s3.out" 'expected until the bootstrap has finished'

reset_fake; fake health.rc 2
verify configured done "$WORK/s3b.out"
yes_ critical-after-bootstrap-is-flagged has "$WORK/s3b.out" 'CRITICAL even though the bootstrap finished'

reset_fake; fake health.rc 9
verify configured done "$WORK/s4.out"
check health-unknown-code "exit 9" "$HEALTH_NOTE"

reset_fake; fake telegram-test.rc 1
verify configured done "$WORK/s5.out"
yes_ telegram-failure-is-a-warning has "$WORK/s5.out" 'test message could not be delivered'

reset_fake
verify no-token done "$WORK/s6.out"
count_eq "$FAKE/invocations.log" telegram-test; check no-test-message-without-config 0 "$N"

FAKE_STATE=failed
reset_fake
rc=0; ( verify configured done "$WORK/s7.out" ) || rc=$?
check dead-daemon-exit-code 4 "$rc"
yes_ dead-daemon-shows-journal has "$WORK/s7.out" 'last 40 journal lines'
count_in "$FAKE/invocations.log" health; check dead-daemon-no-health-call 0 "$N"
FAKE_STATE=active

NO_START=1; reset_fake
verify configured done "$WORK/s8.out"
NO_START=0
count_in "$FAKE/invocations.log" health; check no-start-skips-verification 0 "$N"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    oks = assert_harness_ok(result)
    assert len(oks) >= 40


@needs_bash
@pytest.mark.slow
def test_systemd_step_installs_units_and_commits_or_fails_with_the_right_code(tmp_path: Path):
    script = SANDBOX_HEADER + r"""
# bash builtins only (Git Bash forks slowly)
has() { local t; t=$(<"$1"); [[ $t == *"$2"* ]]; }
same_file() { [[ $(<"$1") == "$(<"$2")" ]]; }
count_in() { local l; N=0; while IFS= read -r l; do if [[ $l == *"$2"* ]]; then N=$((N + 1)); fi; done <"$1"; }
line_of() { local l i=0; N=0; while IFS= read -r l; do i=$((i + 1)); if [[ $l == *"$1"* ]]; then N=$i; return 0; fi; done <"$CALLS"; }
make_release "$APP_DIR" v2
cp "$PROJECT/deployment/journald-monster.conf" "$PROJECT/deployment/logrotate-monster" "$APP_DIR/deployment/"
mkdir -p "$(dirname "$LOGROTATE_FILE")"
journalctl() { echo "journal line"; }
reset() { : >"$CALLS"; RELEASE_COMMITTED=0; SERVICES_RESTARTED=0; NO_START=0; DASHBOARD_WANTED=0; HAD_PREV=1; STOPPED=0; FAKE_STATE=active; }

# --- upgrade: units, journald, logrotate and the CLI wrapper are installed, then the daemon is enabled/started
reset
step_systemd >"$WORK/s1.out" 2>&1
for unit in monster-ai.service monster-ai-bootstrap.service monster-ai-health.service monster-ai-health.timer 'monster-ai-alert@.service'; do
  yes_ "installed-$unit" same_file "$PROJECT/deployment/systemd/$unit" "$UNIT_DIR/$unit"
done
no_ dashboard-unit-not-installed test -e "$UNIT_DIR/monster-ai-dashboard.service"
yes_ journald-dropin-installed same_file "$PROJECT/deployment/journald-monster.conf" "$JOURNALD_DROPIN"
yes_ logrotate-installed same_file "$PROJECT/deployment/logrotate-monster" "$LOGROTATE_FILE"
yes_ wrapper-is-valid-bash bash -n "$CLI_WRAPPER"
yes_ wrapper-runs-as-service-user has "$CLI_WRAPPER" 'exec runuser -u monster'
yes_ wrapper-restricts-umask has "$CLI_WRAPPER" $'\numask 027\n'
line_of 'systemctl daemon-reload'; reload=$N
line_of 'systemctl enable --now monster-ai.service monster-ai-health.timer'; enable=$N
if ((reload > 0 && enable > reload)); then order=yes; else order=no; fi
check reload-before-enable yes "$order"
count_in "$CALLS" 'systemctl restart systemd-journald'; check journald-restarted-once 1 "$N"
check release-committed 1 "$RELEASE_COMMITTED"
yes_ commit-marker-written test -f "$COMMITTED_FILE"
count_in "$CALLS" 'monster-ai-dashboard.service'; check dashboard-not-started 0 "$N"

# --- the same run again changes nothing (idempotent: journald is not restarted for an identical drop-in);
# this time with the dashboard
reset; DASHBOARD_WANTED=1
step_systemd >"$WORK/s2.out" 2>&1
count_in "$CALLS" 'systemctl restart systemd-journald'; check rerun-keeps-journald 0 "$N"
check rerun-committed 1 "$RELEASE_COMMITTED"
yes_ dashboard-unit-installed test -f "$UNIT_DIR/monster-ai-dashboard.service"
count_in "$CALLS" 'systemctl enable --now monster-ai-dashboard.service'; check dashboard-started 1 "$N"

# --- dashboard no longer wanted (removed again) and --no-start (enabled, never started, still committed)
reset; NO_START=1
step_systemd >"$WORK/s3.out" 2>&1
no_ dashboard-unit-removed test -e "$UNIT_DIR/monster-ai-dashboard.service"
count_in "$CALLS" 'systemctl disable --now monster-ai-dashboard.service'; check dashboard-disabled 1 "$N"
count_in "$CALLS" 'enable --now'; check no-start-nothing-started 0 "$N"
count_in "$CALLS" 'systemctl enable monster-ai.service monster-ai-health.timer'; check no-start-enabled 1 "$N"
check no-start-committed 1 "$RELEASE_COMMITTED"

# --- the daemon never becomes active: exit 3 when a previous release exists (the trap rolls back), else 4
rm -f "$COMMITTED_FILE"
reset; FAKE_STATE=failed; HAD_PREV=1
rc=0; ( trap 'echo "committed=$RELEASE_COMMITTED"' EXIT; step_systemd ) >"$WORK/s4.out" 2>&1 || rc=$?
check failed-upgrade-exit-code 3 "$rc"
yes_ failed-upgrade-not-committed has "$WORK/s4.out" 'committed=0'
no_ failed-upgrade-not-marked test -e "$COMMITTED_FILE"
yes_ failed-upgrade-shows-diagnostics has "$WORK/s4.out" 'last 40 journal lines'
reset; FAKE_STATE=failed; HAD_PREV=0
rc=0; ( step_systemd ) >"$WORK/s5.out" 2>&1 || rc=$?
check failed-first-install-exit-code 4 "$rc"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    oks = assert_harness_ok(result)
    assert len(oks) >= 28


@needs_bash
@pytest.mark.slow
def test_exit_trap_rolls_back_failed_upgrades_and_always_restarts_services(tmp_path: Path):
    script = SANDBOX_HEADER + r"""
# bash builtins only (Git Bash forks slowly); scenarios are ordered so that only two release trees are built
has() { local t; t=$(<"$1"); [[ $t == *"$2"* ]]; }
count_in() { local l; N=0; while IFS= read -r l; do if [[ $l == *"$2"* ]]; then N=$((N + 1)); fi; done <"$1"; }
journalctl() { echo "journal line"; }
WORKDIR=""
flags() {
  : >"$CALLS"; WAS_ACTIVE_UNITS=()
  RELEASE_SWAPPED=0; RELEASE_COMMITTED=0; ROLLED_BACK=0; DIE_CALLED=0; HAD_PREV=1; SERVICES_STOPPED=0
  SERVICES_RESTARTED=0; DRY_RUN=0; NO_START=0; LAST_ERROR=""
}
make_release "$APP_DIR" v2; make_release "$APP_PREV" v1

# --- a committed release is never rolled back
flags
rc=0; ( trap on_exit EXIT; RELEASE_SWAPPED=1; RELEASE_COMMITTED=1; exit 1 ) >"$WORK/e1.out" 2>&1 || rc=$?
check committed-exit-code 1 "$rc"
check committed-release-kept v2 "$(<"$APP_DIR/VERSION")"

# --- success and dry runs never roll back
flags
rc=0; ( trap on_exit EXIT; RELEASE_SWAPPED=1; exit 0 ) >"$WORK/e2.out" 2>&1 || rc=$?
check success-exit-code 0 "$rc"
check success-release-kept v2 "$(<"$APP_DIR/VERSION")"
flags
rc=0; ( trap on_exit EXIT; DRY_RUN=1; RELEASE_SWAPPED=1; exit 1 ) >"$WORK/e3.out" 2>&1 || rc=$?
check dry-run-exit-code 1 "$rc"
check dry-run-release-kept v2 "$(<"$APP_DIR/VERSION")"

# --- failure while the services are stopped but before the swap: exactly the stopped ones are started again
flags
rc=0; ( trap on_exit EXIT; SERVICES_STOPPED=1; WAS_ACTIVE_UNITS=(monster-ai.service monster-ai-health.timer); exit 1 ) >"$WORK/e4.out" 2>&1 || rc=$?
check restart-exit-code 1 "$rc"
yes_ restart-daemon has "$CALLS" 'systemctl start --no-block monster-ai.service'
yes_ restart-timer has "$CALLS" 'systemctl start --no-block monster-ai-health.timer'
count_in "$CALLS" 'monster-ai-bootstrap'; check restart-not-the-units-that-were-off 0 "$N"
check release-untouched-before-swap v2 "$(<"$APP_DIR/VERSION")"

# --- services already restarted by the happy path are not started twice; --no-start starts nothing at all
flags
( trap on_exit EXIT; SERVICES_STOPPED=1; SERVICES_RESTARTED=1; WAS_ACTIVE_UNITS=(monster-ai.service); exit 0 ) >"$WORK/e5.out" 2>&1
count_in "$CALLS" 'systemctl start'; check no-double-restart 0 "$N"
flags
( trap on_exit EXIT; NO_START=1; SERVICES_STOPPED=1; WAS_ACTIVE_UNITS=(monster-ai.service); exit 0 ) >"$WORK/e6.out" 2>&1
count_in "$CALLS" 'systemctl start'; check no-start-trap-starts-nothing 0 "$N"

# --- failure after the swap and before the commit: previous release back, exit code 3, services restarted
flags
rc=0; ( trap on_exit EXIT; RELEASE_SWAPPED=1; SERVICES_STOPPED=1; WAS_ACTIVE_UNITS=(monster-ai.service); exit 1 ) >"$WORK/e7.out" 2>&1 || rc=$?
check swap-failure-exit-code 3 "$rc"
check swap-failure-previous-release-live v1 "$(<"$APP_DIR/VERSION")"
check swap-failure-failed-release-kept v2 "$(<"$APP_FAILED/VERSION")"
yes_ swap-failure-says-it-is-safe-to-rerun has "$WORK/e7.out" 'safe to run it again'
yes_ swap-failure-announces-rollback has "$WORK/e7.out" 'ROLLING BACK'
yes_ swap-failure-restarts-the-daemon has "$CALLS" 'systemctl start --no-block monster-ai.service'

# --- first install / nothing to roll back to: the exit code stays 1 and the live tree is left alone
flags
rc=0; ( trap on_exit EXIT; RELEASE_SWAPPED=1; exit 1 ) >"$WORK/e8.out" 2>&1 || rc=$?
check no-previous-exit-code 1 "$rc"
check no-previous-release-untouched v1 "$(<"$APP_DIR/VERSION")"

# --- the exit code of another failure kind is kept (2 = bad usage is not turned into 3), still rolled back
rm -rf "$APP_DIR" "$APP_PREV" "$APP_FAILED"; make_release "$APP_DIR" v4; make_release "$APP_PREV" v3
flags
rc=0; ( trap on_exit EXIT; RELEASE_SWAPPED=1; exit 2 ) >"$WORK/e9.out" 2>&1 || rc=$?
check other-exit-code-kept 2 "$rc"
check other-exit-code-still-rolled-back v3 "$(<"$APP_DIR/VERSION")"
""" + HARNESS_FOOTER
    result = run_bash(script, tmp_path)
    oks = assert_harness_ok(result)
    assert len(oks) >= 20
