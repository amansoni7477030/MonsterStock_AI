"""Typed, validated configuration loaded from environment variables (12-factor).

Precedence: real environment > ``.env`` file (development convenience) > defaults. On the server the
installer writes ``/etc/monster-ai/monster.env`` (mode 600) which systemd injects, so secrets never live in
the code tree. ``Settings.redacted()`` is the only safe way to print configuration.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.errors import ConfigError

HOME_DEFAULT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------------- parsers


def _str(value: str) -> str:
    return value.strip()


def _opt_str(value: str) -> str | None:
    value = value.strip()
    return value or None


def _bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on", "y"}:
        return True
    if v in {"0", "false", "no", "off", "n", ""}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def _int(value: str) -> int:
    return int(value.strip())


def _float(value: str) -> float:
    return float(value.strip())


def _floats(value: str) -> tuple[float, ...]:
    items = [p.strip() for p in value.split(",") if p.strip()]
    return tuple(float(p) for p in items)


def _strs(value: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in re.split(r"[,\s]+", value) if p.strip())


def _path(value: str) -> Path:
    return Path(value.strip()).expanduser()


def _choice(*options: str) -> Callable[[str], str]:
    def parse(value: str) -> str:
        v = value.strip().lower()
        if v not in options:
            raise ValueError(f"must be one of {', '.join(options)}")
        return v

    return parse


def _hhmm(value: str) -> str:
    """Validate an IST wall-clock time and normalise it to HH:MM:SS so string ordering is valid."""
    v = value.strip()
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?", v):
        raise ValueError("expected HH:MM or HH:MM:SS (24h, IST)")
    return v if len(v) == 8 else v + ":00"


# ------------------------------------------------------------------------------------------ definition


@dataclass(frozen=True)
class Var:
    attr: str
    env: str
    default: str
    parse: Callable[[str], Any]
    doc: str
    group: str
    secret: bool = False


# ``default`` for path variables may reference other settings with ``{home}``, ``{data_dir}``.
VARS: tuple[Var, ...] = (
    # -- runtime -----------------------------------------------------------------------------------
    Var("env", "MONSTER_ENV", "production", _choice("production", "development", "test"),
        "Runtime environment.", "Runtime"),
    Var("mode", "MONSTER_MODE", "shadow", _choice("shadow", "live"),
        "shadow = research output only (default, mandatory until a live shadow record exists). The system never places orders in either mode.",
        "Runtime"),
    Var("log_level", "LOG_LEVEL", "INFO", lambda v: _choice("debug", "info", "warning", "error")(v).upper(),
        "DEBUG / INFO / WARNING / ERROR. DEBUG adds one line per HTTP request and per skipped fetch.", "Runtime"),
    Var("log_format", "LOG_FORMAT", "text", _choice("text", "json"),
        "text = one readable line per record; json = one JSON object per line (for log shippers).", "Runtime"),
    Var("home", "MONSTER_HOME", "{home_default}", _path, "Code root (auto-detected).", "Runtime"),
    Var("data_dir", "MONSTER_DATA_DIR", "{home}/data/runtime", _path,
        "Writable state: SQLite DB, models, cache, backups. Server: /var/lib/monster-ai.", "Runtime"),
    Var("log_dir", "MONSTER_LOG_DIR", "{home}/logs", _path,
        "Daily log files (<role>-YYYY-MM-DD.log plus errors-YYYY-MM-DD.log). Server: /var/log/monster-ai.", "Runtime"),
    Var("db_path", "MONSTER_DB_PATH", "{data_dir}/monster.db", _path, "SQLite database file.", "Runtime"),
    Var("models_dir", "MONSTER_MODELS_DIR", "{data_dir}/models", _path, "Approved model artifacts.", "Runtime"),
    Var("cache_dir", "MONSTER_CACHE_DIR", "{data_dir}/cache", _path, "Disposable cache (size-limited).", "Runtime"),
    Var("backup_dir", "MONSTER_BACKUP_DIR", "{data_dir}/backups", _path, "Database backups.", "Runtime"),
    Var("timezone", "MONSTER_TIMEZONE", "Asia/Kolkata", _str,
        "Time zone of log stamps and operator-facing displays; always printed with an explicit UTC offset. The server's own "
        "zone is never used, exchange logic is always IST and storage is always UTC.", "Runtime"),
    # -- telegram ----------------------------------------------------------------------------------
    Var("telegram_bot_token", "TELEGRAM_BOT_TOKEN", "", _opt_str, "Bot token from @BotFather.", "Telegram", True),
    Var("telegram_chat_id", "TELEGRAM_CHAT_ID", "", _opt_str,
        "Destination chat id (run `python -m app.cli telegram-setup` to discover it automatically).", "Telegram"),
    Var("telegram_alert_chat_id", "TELEGRAM_ALERT_CHAT_ID", "", _opt_str,
        "Chat for operational alerts (defaults to TELEGRAM_CHAT_ID).", "Telegram"),
    Var("send_prelim_report", "SEND_PRELIM_REPORT", "false", _bool,
        "Also send the 08:45 preliminary scan (off by default to avoid noise).", "Telegram"),
    Var("send_evening_report", "SEND_EVENING_REPORT", "true", _bool, "Send the post-session evaluation summary.", "Telegram"),
    Var("telegram_timeout_sec", "TELEGRAM_TIMEOUT_SEC", "20", _int, "HTTP timeout per Telegram call.", "Telegram"),
    Var("telegram_max_attempts", "TELEGRAM_MAX_ATTEMPTS", "5", _int, "Bounded retries per message.", "Telegram"),
    # -- universe ----------------------------------------------------------------------------------
    Var("universe_mode", "UNIVERSE_MODE", "fno_nifty100", _choice("fno_nifty100", "nifty200", "nifty500", "custom"),
        "Instrument universe policy.", "Universe"),
    Var("universe_symbols", "UNIVERSE_SYMBOLS", "", _strs, "Symbols for UNIVERSE_MODE=custom (comma separated).", "Universe"),
    Var("universe_max_size", "UNIVERSE_MAX_SIZE", "260", _int, "Hard cap on universe size (RAM guard).", "Universe"),
    Var("min_median_turnover_cr", "MIN_MEDIAN_TURNOVER_CR", "25", _float,
        "Liquidity gate: minimum 20-day median traded value in INR crore.", "Universe"),
    Var("min_price", "MIN_PRICE", "50", _float, "Liquidity gate: minimum last close in INR.", "Universe"),
    Var("history_years", "HISTORY_YEARS", "10", _int, "Years of daily history to backfill.", "Universe"),
    Var("train_years", "TRAIN_YEARS", "6", _int, "Years of history used when training.", "Universe"),
    # -- targets / ranking ---------------------------------------------------------------------------
    Var("range_thresholds", "RANGE_THRESHOLDS", "1,2,3,5", _floats, "Intraday range thresholds in % of open.", "Targets"),
    Var("primary_threshold", "PRIMARY_THRESHOLD", "3", _float, "Threshold (%) whose calibrated probability drives the ranking.", "Targets"),
    Var("unusual_k", "UNUSUAL_K", "1.5", _float, "Volatility-adjusted threshold: range > k x trailing mean range.", "Targets"),
    Var("top_n", "TOP_N", "5", _int, "Number of candidates reported.", "Targets"),
    Var("tier_strong_p", "TIER_STRONG_P", "0.55", _float, "Calibrated P(range >= primary) for STRONG.", "Targets"),
    Var("tier_moderate_p", "TIER_MODERATE_P", "0.40", _float, "Calibrated P for MODERATE.", "Targets"),
    Var("tier_watch_p", "TIER_WATCH_P", "0.25", _float, "Calibrated P for WATCH; below this a name is not listed.", "Targets"),
    Var("max_uncertainty_strong", "MAX_UNCERTAINTY_STRONG", "0.35", _float,
        "Fallback uncertainty ceiling for STRONG, used only for a model with no learned reference "
        "(trained models use the median uncertainty of held-out names of similar probability).", "Targets"),
    Var("max_uncertainty_moderate", "MAX_UNCERTAINTY_MODERATE", "0.55", _float,
        "Fallback uncertainty ceiling for MODERATE (trained models use the 85th percentile of those names).", "Targets"),
    Var("max_cost_to_range_ratio", "MAX_COST_TO_RANGE_RATIO", "0.25", _float,
        "Reject candidates whose round-trip cost exceeds this share of the median expected range.", "Targets"),
    # -- schedule (IST) ----------------------------------------------------------------------------------
    Var("prelim_scan_time", "PRELIM_SCAN_TIME", "08:45", _hhmm, "Preliminary scan (IST).", "Schedule"),
    Var("final_scan_time", "FINAL_SCAN_TIME", "09:08:45", _hhmm,
        "Final scan start (IST); must be after the pre-open equilibrium price is published (~09:08).", "Schedule"),
    Var("report_cutoff_time", "REPORT_CUTOFF_TIME", "09:12:00", _hhmm, "Point-in-time cutoff of the final snapshot (IST).", "Schedule"),
    Var("report_deadline_time", "REPORT_DEADLINE_TIME", "09:14:30", _hhmm,
        "Hard deadline: the report is never sent after this time (IST).", "Schedule"),
    # -- providers -----------------------------------------------------------------------------------------
    Var("yahoo_min_interval_sec", "YAHOO_MIN_INTERVAL_SEC", "1.0", _float, "Minimum spacing between Yahoo requests.", "Providers"),
    Var("nse_min_interval_sec", "NSE_MIN_INTERVAL_SEC", "1.0", _float, "Minimum spacing between NSE requests.", "Providers"),
    Var("nse_enabled", "NSE_ENABLED", "true", _bool, "Use NSE public endpoints (pre-open, option chain, filings).", "Providers"),
    Var("http_timeout_sec", "HTTP_TIMEOUT_SEC", "30", _float, "Default HTTP read timeout.", "Providers"),
    Var("http_max_attempts", "HTTP_MAX_ATTEMPTS", "4", _int, "Bounded attempts per request.", "Providers"),
    Var("http_max_response_mb", "HTTP_MAX_RESPONSE_MB", "40", _int, "Abort responses larger than this (RAM guard).", "Providers"),
    Var("http_transport", "HTTP_TRANSPORT", "auto", _choice("auto", "curl_cffi", "requests"),
        "HTTP client for scraped providers (NSE, Yahoo, RSS, GDELT): auto = curl_cffi (real browser TLS / HTTP-2 "
        "fingerprint, fewer blocks) with an automatic requests fallback.", "Providers"),
    Var("http_impersonate", "HTTP_IMPERSONATE", "chrome", _str,
        "Browser profile curl_cffi impersonates: chrome, safari, firefox, edge or a pinned version such as chrome131 "
        "(empty = no impersonation). Proxies: set HTTPS_PROXY in the environment file.", "Providers"),
    Var("fred_api_key", "FRED_API_KEY", "", _opt_str, "Optional FRED key (enables vintage-aware macro data).", "Providers", True),
    Var("gdelt_enabled", "GDELT_ENABLED", "true", _bool, "Use the GDELT DOC API for global news discovery.", "Providers"),
    Var("news_feeds_file", "NEWS_FEEDS_FILE", "{home}/data/seeds/feeds.json", _path, "RSS/Atom feed registry.", "Providers"),
    Var("news_lookback_hours", "NEWS_LOOKBACK_HOURS", "36", _int, "How far back a collection run looks.", "Providers"),
    Var("news_retention_days", "NEWS_RETENTION_DAYS", "45", _int, "Raw article text retention (structured events are kept).", "Providers"),
    # -- llm -----------------------------------------------------------------------------------------------
    Var("llm_api_base", "LLM_API_BASE", "", _opt_str,
        "OpenAI-compatible base URL (e.g. https://api.openai.com/v1). Empty = deterministic rules only.", "LLM"),
    Var("llm_api_key", "LLM_API_KEY", "", _opt_str, "API key for the LLM endpoint.", "LLM", True),
    Var("llm_model", "LLM_MODEL", "", _opt_str, "Model name for structured extraction.", "LLM"),
    Var("llm_timeout_sec", "LLM_TIMEOUT_SEC", "40", _float, "LLM request timeout.", "LLM"),
    Var("llm_max_articles_per_run", "LLM_MAX_ARTICLES_PER_RUN", "60", _int, "Cost cap per extraction run.", "LLM"),
    Var("llm_max_output_tokens", "LLM_MAX_OUTPUT_TOKENS", "700", _int, "Max tokens per extraction response.", "LLM"),
    # -- training / governance ---------------------------------------------------------------------------
    Var("training_mode", "TRAINING_MODE", "local-light", _choice("local-light", "external", "disabled"),
        "local-light = small capped training on the server; external = only pull signed artifacts; disabled.", "Training"),
    Var("training_dow", "TRAINING_DOW", "sat", _choice("mon", "tue", "wed", "thu", "fri", "sat", "sun"),
        "Weekday of scheduled (re)training.", "Training"),
    Var("model_artifact_url", "MODEL_ARTIFACT_URL", "", _opt_str, "HTTPS URL of the external manifest.json (external mode).", "Training"),
    Var("model_signing_key", "MODEL_SIGNING_KEY", "", _opt_str, "HMAC-SHA256 key shared with the training worker.", "Training", True),
    Var("train_max_rows", "TRAIN_MAX_ROWS", "320000", _int, "Row cap for in-process training (RAM guard).", "Training"),
    Var("gate_min_days", "GATE_MIN_DAYS", "120", _int, "Min out-of-time days before a model may be called validated.", "Training"),
    Var("gate_max_ece", "GATE_MAX_ECE", "0.06", _float, "Calibration gate (expected calibration error).", "Training"),
    Var("gate_min_brier_skill", "GATE_MIN_BRIER_SKILL", "0.0", _float, "Brier skill vs climatology required.", "Training"),
    Var("min_news_model_days", "MIN_NEWS_MODEL_DAYS", "120", _int,
        "Accumulated live snapshot days required before the news/event specialist is trained.", "Training"),
    # -- storage / backups -------------------------------------------------------------------------------------
    Var("backup_keep_daily", "BACKUP_KEEP_DAILY", "7", _int, "Daily backups retained.", "Storage"),
    Var("backup_keep_weekly", "BACKUP_KEEP_WEEKLY", "4", _int, "Weekly backups retained.", "Storage"),
    Var("backup_max_mb", "BACKUP_MAX_MB", "1500", _int,
        "Size budget of the local backup folder: the oldest backups beyond the daily / weekly policy are dropped until it fits "
        "(the newest is always kept). 0 = no budget.", "Storage"),
    Var("data_max_mb", "DATA_MAX_MB", "4000", _int,
        "Size budget of the whole data directory (database, models, backups, cache). Above 80 % health warns; above 100 % "
        "the optional data is trimmed harder. 0 = no budget.", "Storage"),
    Var("event_retention_days", "EVENT_RETENTION_DAYS", "1100", _int,
        "Structured news events are kept this long (about 3 years). 0 = forever.", "Storage"),
    Var("quote_retention_days", "QUOTE_RETENTION_DAYS", "400", _int,
        "Pre-open / live quote rows and the economic calendar are kept this long. Intermediate pre-open polls are thinned to "
        "the last one per symbol and day after 14 days. 0 = forever.", "Storage"),
    Var("record_retention_days", "RECORD_RETENTION_DAYS", "1825", _int,
        "Frozen snapshots, predictions, outcomes and evaluation metrics are kept this long (5 years). 0 = forever.", "Storage"),
    Var("ops_retention_days", "OPS_RETENTION_DAYS", "180", _int,
        "Job runs, delivery records and undelivered alerts are kept this long (delivered alerts half of it). 0 = forever.", "Storage"),
    Var("log_retention_days", "LOG_RETENTION_DAYS", "30", _int,
        "Daily log files older than this are deleted (the journal keeps its own two-week window). Minimum 1.", "Storage"),
    Var("log_max_mb", "LOG_MAX_MB", "150", _int,
        "Size budget of the whole log directory; the oldest files go first, today's files are never removed. 0 = no budget.", "Storage"),
    Var("backup_passphrase", "BACKUP_PASSPHRASE", "", _opt_str, "If set, backups are gpg-encrypted (AES256).", "Storage", True),
    Var("s3_endpoint_url", "S3_ENDPOINT_URL", "", _opt_str, "S3-compatible endpoint (AWS, R2, B2, MinIO, Spaces).", "Storage"),
    Var("s3_bucket", "S3_BUCKET", "", _opt_str, "Bucket for off-box backups / cold archive / artifacts.", "Storage"),
    Var("s3_region", "S3_REGION", "us-east-1", _str, "Signing region ('auto' for Cloudflare R2).", "Storage"),
    Var("s3_access_key_id", "S3_ACCESS_KEY_ID", "", _opt_str, "S3 access key id.", "Storage", True),
    Var("s3_secret_access_key", "S3_SECRET_ACCESS_KEY", "", _opt_str, "S3 secret key.", "Storage", True),
    Var("s3_prefix", "S3_PREFIX", "monster-ai/", _str, "Key prefix inside the bucket.", "Storage"),
    # -- dashboard ---------------------------------------------------------------------------------------------
    Var("dashboard_enabled", "DASHBOARD_ENABLED", "false", _bool, "Run the optional read-only dashboard service.", "Dashboard"),
    Var("dashboard_host", "DASHBOARD_HOST", "127.0.0.1", _str, "Bind address. Keep loopback and use an SSH tunnel.", "Dashboard"),
    Var("dashboard_port", "DASHBOARD_PORT", "8080", _int, "Dashboard port.", "Dashboard"),
    Var("dashboard_token", "DASHBOARD_TOKEN", "", _opt_str, "Bearer token (required when not bound to loopback).", "Dashboard", True),
    # -- resource guards -----------------------------------------------------------------------------------------
    Var("min_free_disk_mb_warn", "MIN_FREE_DISK_MB_WARN", "1500", _int, "Warn below this free disk.", "Resources"),
    Var("min_free_disk_mb_crit", "MIN_FREE_DISK_MB_CRIT", "600", _int, "Critical: emergency cleanup, block heavy jobs.", "Resources"),
    Var("min_available_ram_mb", "MIN_AVAILABLE_RAM_MB", "120", _int, "Do not start heavy jobs below this MemAvailable.", "Resources"),
    Var("job_max_rss_mb", "JOB_MAX_RSS_MB", "720", _int, "Soft RSS ceiling per job process (watchdog aborts the job).", "Resources"),
    Var("job_timeout_minutes", "JOB_TIMEOUT_MINUTES", "60", _int, "Default wall-clock limit for non-critical jobs.", "Resources"),
)


@dataclass(frozen=True)
class Settings:
    env: str = "production"
    mode: str = "shadow"
    log_level: str = "INFO"
    log_format: str = "text"
    home: Path = HOME_DEFAULT
    data_dir: Path = HOME_DEFAULT / "data" / "runtime"
    log_dir: Path = HOME_DEFAULT / "logs"
    db_path: Path = HOME_DEFAULT / "data" / "runtime" / "monster.db"
    models_dir: Path = HOME_DEFAULT / "data" / "runtime" / "models"
    cache_dir: Path = HOME_DEFAULT / "data" / "runtime" / "cache"
    backup_dir: Path = HOME_DEFAULT / "data" / "runtime" / "backups"
    timezone: str = "Asia/Kolkata"
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    telegram_alert_chat_id: str | None = None
    send_prelim_report: bool = False
    send_evening_report: bool = True
    telegram_timeout_sec: int = 20
    telegram_max_attempts: int = 5
    universe_mode: str = "fno_nifty100"
    universe_symbols: tuple[str, ...] = ()
    universe_max_size: int = 260
    min_median_turnover_cr: float = 25.0
    min_price: float = 50.0
    history_years: int = 10
    train_years: int = 6
    range_thresholds: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
    primary_threshold: float = 3.0
    unusual_k: float = 1.5
    top_n: int = 5
    tier_strong_p: float = 0.55
    tier_moderate_p: float = 0.40
    tier_watch_p: float = 0.25
    max_uncertainty_strong: float = 0.35
    max_uncertainty_moderate: float = 0.55
    max_cost_to_range_ratio: float = 0.25
    prelim_scan_time: str = "08:45:00"
    final_scan_time: str = "09:08:45"
    report_cutoff_time: str = "09:12:00"
    report_deadline_time: str = "09:14:30"
    yahoo_min_interval_sec: float = 1.0
    nse_min_interval_sec: float = 1.0
    nse_enabled: bool = True
    http_timeout_sec: float = 30.0
    http_max_attempts: int = 4
    http_max_response_mb: int = 40
    http_transport: str = "auto"
    http_impersonate: str = "chrome"
    fred_api_key: str | None = None
    gdelt_enabled: bool = True
    news_feeds_file: Path = HOME_DEFAULT / "data" / "seeds" / "feeds.json"
    news_lookback_hours: int = 36
    news_retention_days: int = 45
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_sec: float = 40.0
    llm_max_articles_per_run: int = 60
    llm_max_output_tokens: int = 700
    training_mode: str = "local-light"
    training_dow: str = "sat"
    model_artifact_url: str | None = None
    model_signing_key: str | None = None
    train_max_rows: int = 320000
    gate_min_days: int = 120
    gate_max_ece: float = 0.06
    gate_min_brier_skill: float = 0.0
    min_news_model_days: int = 120
    backup_keep_daily: int = 7
    backup_keep_weekly: int = 4
    backup_max_mb: int = 1500
    data_max_mb: int = 4000
    event_retention_days: int = 1100
    quote_retention_days: int = 400
    record_retention_days: int = 1825
    ops_retention_days: int = 180
    log_retention_days: int = 30
    log_max_mb: int = 150
    backup_passphrase: str | None = None
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_prefix: str = "monster-ai/"
    dashboard_enabled: bool = False
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8080
    dashboard_token: str | None = None
    min_free_disk_mb_warn: int = 1500
    min_free_disk_mb_crit: int = 600
    min_available_ram_mb: int = 120
    job_max_rss_mb: int = 720
    job_timeout_minutes: int = 60
    extra: dict = field(default_factory=dict, compare=False)

    # ------------------------------------------------------------------------------------------ helpers
    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def is_test(self) -> bool:
        return self.env == "test"

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def alert_chat_id(self) -> str | None:
        return self.telegram_alert_chat_id or self.telegram_chat_id

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_base and self.llm_model)

    @property
    def s3_configured(self) -> bool:
        return bool(self.s3_bucket and self.s3_access_key_id and self.s3_secret_access_key)

    def secrets(self) -> list[str]:
        """All secret values (for log redaction)."""
        out = []
        for var in VARS:
            if var.secret:
                value = getattr(self, var.attr)
                if value:
                    out.append(str(value))
        return out

    def __repr__(self) -> str:
        # the generated dataclass repr would print every secret (bot token, API keys, signing key) into logs and tracebacks
        return "Settings(" + ", ".join(f"{name}={value!r}" for name, value in self.redacted().items()) + ")"

    def redacted(self) -> dict[str, Any]:
        """Configuration safe to print or log."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            value = getattr(self, f.name)
            if f.name in {v.attr for v in VARS if v.secret}:
                out[f.name] = "***set***" if value else None
            else:
                out[f.name] = str(value) if isinstance(value, Path) else value
        return out

    def ensure_dirs(self) -> None:
        """Create writable directories. Secrets-bearing paths get restrictive permissions."""
        for path in (self.data_dir, self.log_dir, self.models_dir, self.cache_dir, self.backup_dir, self.tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


_PLACEHOLDER = re.compile(r"\{(home_default|home|data_dir)\}")


def _read_env_file(path: Path) -> dict[str, str]:
    """Minimal dotenv reader (no shell expansion, no export, no command substitution)."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            values[key] = value
    return values


def load_settings(environ: Mapping[str, str] | None = None, env_file: str | Path | None = None) -> Settings:
    """Build ``Settings``. ``environ`` defaults to ``os.environ``; ``env_file`` defaults to ``$MONSTER_ENV_FILE``
    or ``<home>/.env`` when present. Raises ``ConfigError`` listing every problem at once."""
    real_env: Mapping[str, str] = os.environ if environ is None else environ
    merged: dict[str, str] = {}
    candidate = env_file or real_env.get("MONSTER_ENV_FILE") or (HOME_DEFAULT / ".env")
    if candidate and Path(candidate).is_file():
        merged.update(_read_env_file(Path(candidate)))
    merged.update({k: v for k, v in real_env.items() if v is not None})

    resolved: dict[str, Any] = {}
    errors: list[str] = []
    context = {"home_default": str(HOME_DEFAULT), "home": str(HOME_DEFAULT), "data_dir": ""}

    for var in VARS:
        raw = merged.get(var.env)
        if raw is None or (raw.strip() == "" and var.default != ""):
            raw = var.default
        raw = _PLACEHOLDER.sub(lambda m: context.get(m.group(1), m.group(0)), raw)
        try:
            value = var.parse(raw)
        except (ValueError, TypeError) as exc:
            errors.append(f"{var.env}: {exc}")
            continue
        resolved[var.attr] = value
        if var.attr == "home":
            context["home"] = str(value)
        if var.attr == "data_dir":
            context["data_dir"] = str(value)

    if errors:
        raise ConfigError("invalid configuration: " + "; ".join(errors))

    settings = Settings(**resolved)
    problems = validate(settings)
    if problems:
        raise ConfigError("invalid configuration: " + "; ".join(problems))
    return settings


def validate(s: Settings) -> list[str]:
    """Cross-field validation; returns a list of problems (empty when valid)."""
    problems: list[str] = []
    if not s.range_thresholds or any(t <= 0 for t in s.range_thresholds):
        problems.append("RANGE_THRESHOLDS must be positive numbers")
    elif list(s.range_thresholds) != sorted(set(s.range_thresholds)):
        problems.append("RANGE_THRESHOLDS must be strictly ascending")
    if s.primary_threshold not in s.range_thresholds:
        problems.append("PRIMARY_THRESHOLD must be one of RANGE_THRESHOLDS")
    if not 1 <= s.top_n <= 20:
        problems.append("TOP_N must be between 1 and 20")
    if not 0 < s.tier_watch_p < s.tier_moderate_p < s.tier_strong_p < 1:
        problems.append("TIER_WATCH_P < TIER_MODERATE_P < TIER_STRONG_P must lie within (0, 1)")
    if s.unusual_k <= 1.0:
        problems.append("UNUSUAL_K must be greater than 1")
    if s.universe_mode == "custom" and not s.universe_symbols:
        problems.append("UNIVERSE_SYMBOLS is required when UNIVERSE_MODE=custom")
    if not 50 <= s.universe_max_size <= 600:
        problems.append("UNIVERSE_MAX_SIZE must be between 50 and 600")
    if not 1 <= s.history_years <= 20 or not 1 <= s.train_years <= s.history_years:
        problems.append("HISTORY_YEARS must be 1-20 and TRAIN_YEARS must not exceed it")
    if s.http_max_attempts < 1 or s.http_max_attempts > 10:
        problems.append("HTTP_MAX_ATTEMPTS must be between 1 and 10")
    if s.telegram_max_attempts < 1 or s.telegram_max_attempts > 10:
        problems.append("TELEGRAM_MAX_ATTEMPTS must be between 1 and 10")
    if s.min_free_disk_mb_crit >= s.min_free_disk_mb_warn:
        problems.append("MIN_FREE_DISK_MB_CRIT must be lower than MIN_FREE_DISK_MB_WARN")
    if not (s.prelim_scan_time < s.final_scan_time < s.report_cutoff_time < s.report_deadline_time < "09:15:00"):
        problems.append("schedule must satisfy PRELIM < FINAL < CUTOFF < DEADLINE < 09:15 (IST)")
    if s.report_deadline_time >= "09:15:00":
        problems.append("REPORT_DEADLINE_TIME must be before the 09:15 IST regular session")
    if s.dashboard_enabled and s.dashboard_host not in {"127.0.0.1", "::1", "localhost"} and not s.dashboard_token:
        problems.append("DASHBOARD_TOKEN is required when the dashboard is not bound to loopback")
    if s.dashboard_token and len(s.dashboard_token) < 16:
        problems.append("DASHBOARD_TOKEN must be at least 16 characters")
    if not 1 <= s.dashboard_port <= 65535:
        problems.append("DASHBOARD_PORT must be a valid TCP port")
    if (s.llm_api_base or s.llm_model) and not (s.llm_api_base and s.llm_model):
        problems.append("LLM_API_BASE and LLM_MODEL must be set together")
    if s.llm_api_base and not s.llm_api_base.lower().startswith("https://") and not s.llm_api_base.lower().startswith(
        ("http://localhost", "http://127.0.0.1")
    ):
        problems.append("LLM_API_BASE must use https (or a loopback http endpoint)")
    if s.training_mode == "external" and not (s.model_artifact_url and s.model_signing_key):
        problems.append("TRAINING_MODE=external requires MODEL_ARTIFACT_URL and MODEL_SIGNING_KEY")
    if s.model_artifact_url and not s.model_artifact_url.lower().startswith("https://"):
        problems.append("MODEL_ARTIFACT_URL must use https")
    if (s.s3_bucket or s.s3_access_key_id or s.s3_secret_access_key) and not s.s3_configured:
        problems.append("S3_BUCKET, S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set together")
    try:
        ZoneInfo(s.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        problems.append(f"unknown MONSTER_TIMEZONE {s.timezone!r}")
    if s.telegram_bot_token and not re.fullmatch(r"\d{5,}:[A-Za-z0-9_-]{20,}", s.telegram_bot_token):
        problems.append("TELEGRAM_BOT_TOKEN does not look like a BotFather token (123456:ABC...)")
    for name in ("backup_max_mb", "data_max_mb", "event_retention_days", "quote_retention_days", "record_retention_days",
                 "ops_retention_days", "log_max_mb"):
        if getattr(s, name) < 0:
            problems.append(f"{name.upper()} must not be negative (0 = unlimited)")
    if s.log_retention_days < 1:
        problems.append("LOG_RETENTION_DAYS must be at least 1")
    if s.data_max_mb and s.backup_max_mb and s.backup_max_mb >= s.data_max_mb:
        problems.append("BACKUP_MAX_MB must be smaller than DATA_MAX_MB (the backups live inside the data directory)")
    return problems


def env_example_text() -> str:
    """Render the canonical ``.env.example`` from ``VARS`` (single source of truth)."""
    lines = [
        "# MONSTER AI configuration. Copy to .env (local) or /etc/monster-ai/monster.env (server, mode 600).",
        "# Only the Telegram values are required; every other setting has a safe default.",
        "",
    ]
    group = None
    for var in VARS:
        if var.group != group:
            group = var.group
            lines += ["", f"# ---- {group} " + "-" * max(4, 70 - len(group))]
        default = var.default
        if default.startswith("{"):
            default = ""
        lines.append(f"# {var.doc}")
        lines.append(f"{var.env}={default}")
    return "\n".join(lines) + "\n"
