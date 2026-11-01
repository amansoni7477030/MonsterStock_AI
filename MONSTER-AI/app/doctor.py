"""``doctor``: verifies that this machine can run MONSTER AI. Offline checks are always safe; online checks only read."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from app.config import ConfigError, Settings, load_settings, validate
from app.logging_setup import redact
from app.resources import check_resources, disk_info, memory_info

MIN_PYTHON = (3, 10)
PACKAGES = ("numpy", "pandas", "scipy", "lightgbm", "requests", "curl_cffi", "feedparser", "tzdata")
CLOCK_SKEW_WARN_SEC = 30.0
REQUIRED_SEEDS = ("universe_seed.csv", "nse_holidays.json", "feeds.json")


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str = ""


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": not self.failed, "checks": [c.__dict__ for c in self.checks]}


def _python(report: DoctorReport) -> None:
    ok = sys.version_info[:2] >= MIN_PYTHON
    report.add("python", "ok" if ok else "fail", f"{platform.python_version()} on {platform.system()} {platform.machine()}"
               + ("" if ok else f" (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})"))


def _packages(report: DoctorReport) -> None:
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report.add(f"package {name}", "ok", str(getattr(module, "__version__", "")))
        except Exception as exc:  # noqa: BLE001
            report.add(f"package {name}", "fail", f"{type(exc).__name__}: {str(exc)[:80]}")
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo("Asia/Kolkata")
        report.add("timezone data", "ok", "Asia/Kolkata")
    except Exception as exc:  # noqa: BLE001
        report.add("timezone data", "fail", f"{type(exc).__name__}: install the tzdata package")
    _server_time(report)


def _server_time(report: DoctorReport) -> None:
    """The server may run in any time zone: show that the application's own clock (UTC, IST) is right regardless."""
    from app.clock import IST, get_clock

    now = get_clock().now()
    local = now.astimezone()
    report.add("server time", "ok", f"UTC {now:%Y-%m-%d %H:%M:%S}, IST {now.astimezone(IST):%Y-%m-%d %H:%M:%S}; the server's own "
               f"zone is {local.tzname()} (UTC{local:%z}) - irrelevant, all scheduling and dates use IST")


def _http_transport(report: DoctorReport, s: Settings) -> None:
    from ingestion.http import curl_cffi_info

    info = curl_cffi_info()
    if s.http_transport == "requests":
        report.add("http transport", "warn", "HTTP_TRANSPORT=requests: no browser fingerprint (NSE / Yahoo may block)")
    elif info["available"]:
        profile = s.http_impersonate or "no impersonation"
        report.add("http transport", "ok", f"curl_cffi {info['version']} ({profile}); {str(info['libcurl']).split(' ')[0]}")
    else:
        report.add("http transport", "warn", f"curl_cffi unavailable ({info['error'][:100]}): falling back to requests, "
                   "NSE / Yahoo may block it (re-run the installer)")


def _config(report: DoctorReport, env_file: str | None) -> Settings | None:
    try:
        settings = load_settings(env_file=env_file)
    except ConfigError as exc:
        report.add("configuration", "fail", str(exc)[:300])
        return None
    problems = validate(settings)
    report.add("configuration", "fail" if problems else "ok", "; ".join(problems) or f"mode={settings.mode}, env={settings.env}")
    return settings


def _paths(report: DoctorReport, s: Settings) -> None:
    for label, path in (("data dir", s.data_dir), ("log dir", s.log_dir), ("models dir", s.models_dir),
                        ("backup dir", s.backup_dir)):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            report.add(label, "ok", str(path))
        except OSError as exc:
            report.add(label, "fail", f"{path}: {exc.strerror or exc}")
    seeds = s.home / "data" / "seeds"
    missing = [n for n in REQUIRED_SEEDS if not (seeds / n).is_file()]
    report.add("seed data", "fail" if missing else "ok", ("missing: " + ", ".join(missing)) if missing else str(seeds))


def _database(report: DoctorReport, s: Settings) -> None:
    from database.migrate import check_integrity, current_version
    from database.store import Store

    try:
        store = Store.open(s.db_path)
    except Exception as exc:  # noqa: BLE001
        report.add("database", "fail", f"{type(exc).__name__}: {str(exc)[:160]}")
        return
    try:
        ok = check_integrity(store.conn, quick=True)
        report.add("database", "ok" if ok else "fail", f"schema v{current_version(store.conn)}, integrity "
                   f"{'ok' if ok else 'FAILED'}, {s.db_path.stat().st_size / 1048576:.1f} MB")
        state = store.kv_get("bootstrap.state")
        report.add("bootstrap", "ok" if state == "done" else "warn", state or "not started")
        models = {stage: (store.active_model(stage) or {}).get("status", "none") for stage in ("pre", "final")}
        report.add("models", "ok" if all(v != "none" for v in models.values()) else "warn",
                   ", ".join(f"{k}: {v}" for k, v in models.items()) + ("" if all(v != 'none' for v in models.values())
                                                                          else " (heuristic baseline in use)"))
    finally:
        store.close()


def _resources(report: DoctorReport, s: Settings) -> None:
    status = check_resources(s.data_dir, warn_disk_mb=s.min_free_disk_mb_warn, crit_disk_mb=s.min_free_disk_mb_crit,
                             min_ram_mb=s.min_available_ram_mb)
    disk = disk_info(s.data_dir)
    mem = memory_info()
    detail = f"{disk['free_mb']:.0f} MB disk free"
    if mem["total_mb"]:
        detail += f", RAM {mem['total_mb']:.0f} MB total / {mem['available_mb'] or 0:.0f} MB available"
    report.add("resources", {"ok": "ok", "unknown": "ok", "warn": "warn", "crit": "fail"}[status.level],
               "; ".join(status.messages) or detail)
    swap, total = mem.get("swap_total_mb"), mem.get("total_mb")
    if total and total < 1600 and swap is not None and swap < 256:
        report.add("swap", "warn", "no swap on a small server: training can be OOM-killed (the installer adds a swap file)")


def _telegram_config(report: DoctorReport, s: Settings) -> None:
    if s.telegram_configured:
        report.add("telegram settings", "ok", "token and chat id present")
    elif s.telegram_bot_token:
        report.add("telegram settings", "warn", "token set but TELEGRAM_CHAT_ID missing (run telegram-setup)")
    else:
        report.add("telegram settings", "warn", "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; reports cannot be delivered")


def _ai_config(report: DoctorReport, s: Settings) -> None:
    """The AI services are optional enrichment: being off is fine (a half-set configuration is caught by ``validate``)."""
    on = []
    if s.llm_configured:
        on.append(f"LLM via {s.llm_provider} ({', '.join(m for m in (s.llm_model_name, s.llm_fallback_name) if m)})")
    if s.llm_web_configured:
        on.append(f"web check (up to {s.llm_web_max_per_run} per news job)")
    if s.jev_configured:
        on.append(f"Jev ({s.jev_model})")
    report.add("ai services", "ok", " + ".join(on) or "off (optional): GROQ_API_KEY / TYPESAFE_API_KEY not set, the rules run alone")


def _ai_network(report: DoctorReport, s: Settings) -> None:
    """Free model-list calls prove each configured key without spending tokens (``ai-test`` sends a real sample). An optional
    service that fails is a warning, never a reason to call the machine not ready."""
    try:
        from news_intelligence.ai_check import run_checks

        checks = run_checks(s, deep=False)
    except Exception as exc:  # noqa: BLE001
        report.add("ai check", "warn", f"{type(exc).__name__}: {redact(str(exc))[:120]}")
        return
    for check in checks:
        if check.status != "off":
            report.add(f"ai: {check.label}", "ok" if check.status == "ok" else "warn", check.detail)


def _network(report: DoctorReport, s: Settings) -> None:
    from app.errors import DataUnavailable
    from ingestion.http import HttpClient

    probes = (
        ("Yahoo Finance", "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=5d&interval=1d", True),
        ("NSE archives", "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv", False),
        ("Telegram API", "https://api.telegram.org/", False),
    )
    for name, url, critical in probes:
        client = HttpClient.from_settings("doctor", s, max_attempts=1, timeout=(10.0, 15.0), headers={"Accept": "*/*"})
        try:
            response = client.get(url, soft_statuses=range(300, 600))  # any HTTP answer proves reachability; judge the status
            ok = response.status < 400 or (name == "Telegram API" and response.status in (401, 404))
            report.add(f"network: {name}", "ok" if ok else ("fail" if critical else "warn"), f"HTTP {response.status}")
            if name == "Telegram API":
                _clock_skew(report, response.headers.get("Date"))
        except DataUnavailable as exc:
            report.add(f"network: {name}", "fail" if critical else "warn", str(exc)[:100])
        finally:
            client.close()
    if s.telegram_bot_token:
        try:
            from notifications.telegram import TelegramClient

            me = TelegramClient(s).get_me()
            report.add("telegram bot", "ok", f"@{me.get('username', '?')}")
        except Exception as exc:  # noqa: BLE001
            reason = " ".join(redact(str(exc)).split())[:150] or type(exc).__name__
            report.add("telegram bot", "fail", f"getMe failed: {reason}")


def _clock_skew(report: DoctorReport, header: str | None) -> None:
    if not header:
        return
    try:
        remote = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return
    skew = abs((datetime.now(timezone.utc) - remote).total_seconds())
    report.add("system clock", "ok" if skew <= CLOCK_SKEW_WARN_SEC else "warn",
               f"{skew:.0f} s from server time" + ("" if skew <= CLOCK_SKEW_WARN_SEC else " (enable NTP: the scans are time-critical)"))


def run_doctor(*, offline: bool = False, env_file: str | None = None) -> DoctorReport:
    report = DoctorReport()
    _python(report)
    _packages(report)
    settings = _config(report, env_file)
    if settings is not None:
        _paths(report, settings)
        _database(report, settings)
        _resources(report, settings)
        _http_transport(report, settings)
        _telegram_config(report, settings)
        _ai_config(report, settings)
        if not offline:
            _network(report, settings)
            _ai_network(report, settings)
    return report


def render(report: DoctorReport) -> str:
    marks = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
    width = max((len(c.name) for c in report.checks), default=10)
    lines = [f"{marks[c.status]} {c.name.ljust(width)}  {c.detail}" for c in report.checks]
    verdict = "NOT READY: fix the [FAIL] items above" if report.failed else "READY (warnings, if any, are optional improvements)"
    return "\n".join([*lines, "", verdict])
