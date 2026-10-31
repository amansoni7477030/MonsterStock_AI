"""Optional read-only dashboard (FastAPI). Install the extras from ``requirements-dashboard.txt`` first.

Design rules:

* Read-only by construction: only GET routes exist, and every request opens the SQLite file with
  ``Store.open(path, readonly=True)`` (``mode=ro`` + ``query_only``) and closes it again.
* Default-deny: one ASGI middleware authenticates every path except ``/healthz``, so a new route can never
  forget the check. With ``DASHBOARD_TOKEN`` set, ``Authorization: Bearer <token>`` or HTTP Basic (any user name,
  the token as password) is required; repeated failures lock the client IP out for a while. Without a token the
  app only answers for loopback ``Host`` headers (DNS-rebinding guard) and ``run_dashboard`` refuses to bind
  anywhere but loopback.
* Hardened responses: strict security headers on every response, no docs/openapi outside development, generic
  JSON errors (never a traceback), validated query parameters, and every dynamic value HTML-escaped.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import datetime as dt
import hashlib
import ipaddress
import logging
import math
import re
import secrets
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from fastapi import FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__, jsonutil
from app.clock import Clock, get_clock, iso, parse_ts
from app.config import Settings, load_settings
from app.errors import ConfigError
from app.logging_setup import redact, setup_logging
from dashboard.templates import LOW_N, render_page
from database.store import Store

log = logging.getLogger(__name__)

HEARTBEAT_KEY = "daemon.heartbeat"
HEARTBEAT_MAX_AGE_SEC = 900
MAX_FAILURES = 5
LOCKOUT_SEC = 300.0
MAX_TRACKED_CLIENTS = 1024
DEFAULT_JOBS = 50
MAX_JOBS = 200
MAX_SCOPES = 20
MAX_METRIC_ROWS = 500
COUNTS_TTL_SEC = 30.0
STAGES = ("pre", "final")
REALM = "MONSTER AI"

_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
# Development only: the interactive docs pages load their assets from a CDN.
_CSP_DOCS = (
    "default-src 'none'; script-src 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'unsafe-inline' https://cdn.jsdelivr.net; img-src data: https://fastapi.tiangolo.com; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": _CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
_DOCS_PATHS = frozenset({"/docs", "/redoc"})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_INT = re.compile(r"-?[0-9]{1,9}")


class ApiError(Exception):
    """A client- or availability-level failure that is reported as generic JSON (``message`` is safe text)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _json(payload: Any, status: int = 200) -> Response:
    return Response(content=jsonutil.dumps(payload), status_code=status, media_type="application/json")


# ----------------------------------------------------------------------------------------- validation
def parse_date_param(raw: str | None) -> str | None:
    """A strict ``YYYY-MM-DD`` calendar date (regex first, then real date parsing); ``None`` when absent."""
    if raw is None or raw == "":
        return None
    try:
        if not _DATE.fullmatch(raw):
            raise ValueError
        parsed = dt.date.fromisoformat(raw)
    except ValueError:
        raise ApiError(400, "invalid date: expected YYYY-MM-DD") from None
    if not 2000 <= parsed.year <= 2100:
        raise ApiError(400, "date out of range")
    return parsed.isoformat()


def clamp_int(raw: str | None, *, default: int, low: int, high: int, name: str) -> int:
    """Parse an integer query parameter and clamp it into ``[low, high]``; garbage is a 400."""
    if raw is None or raw == "":
        return default
    if not _INT.fullmatch(raw):
        raise ApiError(400, f"invalid {name}: expected an integer")
    return max(low, min(high, int(raw)))


def _parse_flag(raw: str | None, name: str) -> bool:
    value = (raw or "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ApiError(400, f"invalid {name}: expected true or false")


def _parse_stage(raw: str | None) -> str | None:
    if raw is None or raw == "":
        return None
    if raw not in STAGES:
        raise ApiError(400, "invalid stage: expected pre or final")
    return raw


# ---------------------------------------------------------------------------------------- authentication
def _presented_token(header: str) -> str | None:
    """The secret carried by a Bearer or Basic ``Authorization`` header (any Basic user name is accepted)."""
    if len(header) > 2048:
        return None
    scheme, _, value = header.partition(" ")
    scheme, value = scheme.lower(), value.strip()
    if scheme == "bearer":
        return value or None
    if scheme == "basic":
        try:
            decoded = base64.b64decode(value, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None
        _user, separator, password = decoded.partition(":")
        return password if separator else None
    return None


def _unauthorized() -> Response:
    response = _json({"error": "unauthorized"}, 401)
    response.headers["WWW-Authenticate"] = f'Basic realm="{REALM}", charset="UTF-8"'
    response.headers.append("WWW-Authenticate", f'Bearer realm="{REALM}"')
    return response


def _locked(remaining: float) -> Response:
    response = _json({"error": "too many failed attempts; try again later"}, 429)
    response.headers["Retry-After"] = str(max(1, math.ceil(remaining)))
    return response


class _Failures:
    __slots__ = ("count", "last", "locked_until")

    def __init__(self) -> None:
        self.count = 0
        self.last = 0.0
        self.locked_until = 0.0


class _Auth:
    """Token check with a per-client-IP failed-attempt lockout (in memory, bounded)."""

    def __init__(self, token: str, clock: Callable[[], Clock]) -> None:
        self._digest = hashlib.sha256(token.encode("utf-8")).digest()
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: dict[str, _Failures] = {}

    def _matches(self, presented: str) -> bool:
        return secrets.compare_digest(hashlib.sha256(presented.encode("utf-8")).digest(), self._digest)

    def check(self, request: Request) -> Response | None:
        """``None`` when the request may proceed, otherwise the 401 / 429 response to send."""
        client = request.client.host if request.client else "unknown"
        now = self._clock().monotonic()
        with self._lock:
            state = self._failures.get(client)
            if state is not None and state.locked_until > now:
                return _locked(state.locked_until - now)
        header = request.headers.get("authorization")
        if header is None:
            return _unauthorized()
        presented = _presented_token(header)
        if presented is not None and self._matches(presented):
            with self._lock:
                self._failures.pop(client, None)
            return None
        self._record_failure(client, now)
        return _unauthorized()

    def _record_failure(self, client: str, now: float) -> None:
        with self._lock:
            state = self._failures.get(client)
            if state is None or now - state.last >= LOCKOUT_SEC:
                state = self._failures[client] = _Failures()
            state.count += 1
            state.last = now
            if state.count >= MAX_FAILURES:
                state.locked_until = now + LOCKOUT_SEC
            if len(self._failures) > MAX_TRACKED_CLIENTS:
                self._prune(now)

    def _prune(self, now: float) -> None:
        for client in [c for c, s in self._failures.items() if now - s.last >= LOCKOUT_SEC]:
            del self._failures[client]
        if len(self._failures) > MAX_TRACKED_CLIENTS:
            oldest = sorted(self._failures, key=lambda c: self._failures[c].last)
            for client in oldest[: len(oldest) // 2]:
                del self._failures[client]


def is_loopback_host(host: str) -> bool:
    name = host.strip().strip("[]").lower()
    if name in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _request_hostname(request: Request) -> str:
    raw = request.headers.get("host", "").strip().lower()
    if raw.startswith("["):
        return raw[1:].split("]", 1)[0]
    return raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw


def _allowed_hosts(settings: Settings) -> frozenset[str] | None:
    """``None`` (any Host) when a token protects the app; otherwise loopback names only (DNS-rebinding guard)."""
    if settings.dashboard_token:
        return None
    hosts = set(_LOOPBACK_HOSTS)
    if settings.dashboard_host.strip():
        hosts.add(settings.dashboard_host.strip().strip("[]").lower())
    if settings.is_test:
        hosts.add("testserver")
    return frozenset(hosts)


# -------------------------------------------------------------------------------------------- context
class _Context:
    """Per-app state: settings, clock, how to open the database and a short TTL cache of table counts."""

    def __init__(
        self, settings: Settings, store_factory: Callable[[], Store] | None, clock: Clock | None
    ) -> None:
        self.settings = settings
        self._factory = store_factory or (lambda: Store.open(settings.db_path, readonly=True))
        self._clock = clock
        self._counts: tuple[float, dict[str, int]] | None = None
        self._lock = threading.Lock()

    @property
    def clock(self) -> Clock:
        return self._clock or get_clock()

    def now(self) -> dt.datetime:
        return self.clock.now()

    @contextlib.contextmanager
    def store(self) -> Iterator[Store]:
        try:
            opened = self._factory()
        except (sqlite3.Error, OSError) as exc:
            log.warning("dashboard: cannot open the database: %s", redact(str(exc)))
            raise ApiError(503, "database unavailable") from None
        try:
            yield opened
        except sqlite3.Error as exc:
            log.warning("dashboard: database query failed: %s", redact(str(exc)))
            raise ApiError(503, "database unavailable") from None
        finally:
            opened.close()

    def table_counts(self, store: Store) -> dict[str, int]:
        now = self.clock.monotonic()
        with self._lock:
            if self._counts is not None and now - self._counts[0] < COUNTS_TTL_SEC:
                return self._counts[1]
        counts = store.table_counts()
        with self._lock:
            self._counts = (now, counts)
        return counts

    def healthy(self) -> bool:
        try:
            with self.store() as store:
                heartbeat = heartbeat_status(store, self.now())
        except ApiError:
            return False
        return heartbeat["status"] == "ok"


# ----------------------------------------------------------------------------------------- data views
def heartbeat_status(store: Store, now: dt.datetime) -> dict[str, Any]:
    """Age of the daemon heartbeat (``kv['daemon.heartbeat']``, judged by its last write time)."""
    row = store.conn.execute("SELECT value, updated_at FROM kv WHERE key = ?", (HEARTBEAT_KEY,)).fetchone()
    at = (parse_ts(row["updated_at"]) or parse_ts(row["value"])) if row else None
    if at is None:
        return {"at": None, "age_seconds": None, "status": "missing"}
    age = max(0, round((now - at).total_seconds()))
    return {"at": iso(at), "age_seconds": age, "status": "ok" if age <= HEARTBEAT_MAX_AGE_SEC else "stale"}


def _source_status(row: dict[str, Any]) -> str:
    if not row.get("enabled"):
        return "disabled"
    if not row.get("last_attempt_at") and not row.get("last_success_at"):
        return "unknown"
    failures = row.get("consecutive_failures") or 0
    if failures >= 3:
        return "failing"
    return "degraded" if failures > 0 else "ok"


def sources_data(store: Store) -> dict[str, Any]:
    rows = []
    summary = {"total": 0, "ok": 0, "degraded": 0, "failing": 0, "unknown": 0, "disabled": 0}
    critical_failing = []
    for row in store.source_health():
        status = _source_status(row)
        summary["total"] += 1
        summary[status] += 1
        if status == "failing" and row.get("criticality") == "critical":
            critical_failing.append(row["source_id"])
        rows.append({
            "source_id": row["source_id"], "provider": row.get("provider"), "criticality": row.get("criticality"),
            "enabled": bool(row.get("enabled")), "status": status, "last_attempt_at": row.get("last_attempt_at"),
            "last_success_at": row.get("last_success_at"), "last_data_ts": row.get("last_data_ts"),
            "consecutive_failures": row.get("consecutive_failures") or 0,
            "total_success": row.get("total_success") or 0, "total_failure": row.get("total_failure") or 0,
            "last_latency_ms": row.get("last_latency_ms"),
            "last_error": redact(str(row["last_error"]))[:300] if row.get("last_error") else None,
        })
    summary["critical_failing"] = critical_failing
    return {"summary": summary, "sources": rows}


_MODEL_FIELDS = (
    "model_version", "stage", "status", "created_at", "approved_at", "activated_at", "train_start", "train_end",
    "feature_version", "parent_version",
)


def _model_view(row: dict[str, Any]) -> dict[str, Any]:
    out = {name: row.get(name) for name in _MODEL_FIELDS}
    out["is_active"] = bool(row.get("is_active"))
    out["metrics"] = row.get("metrics") or {}
    out["notes"] = redact(str(row["notes"]))[:300] if row.get("notes") else None
    return out


def models_data(store: Store) -> list[dict[str, Any]]:
    return [_model_view(row) for row in store.list_models()[:200]]


_JOB_FIELDS = (
    "run_id", "job_name", "run_key", "scheduled_for", "started_at", "finished_at", "status", "attempt",
    "exit_code", "peak_rss_mb",
)


def _job_view(row: dict[str, Any]) -> dict[str, Any]:
    out = {name: row.get(name) for name in _JOB_FIELDS}
    out["message"] = redact(str(row["message"]))[:500] if row.get("message") else None
    return out


def jobs_data(store: Store, limit: int) -> list[dict[str, Any]]:
    return [_job_view(row) for row in store.recent_jobs(limit)]


_PREDICTION_FIELDS = (
    "prediction_id", "stage", "rank", "symbol", "tier", "model_version", "model_status", "uncertainty",
    "data_quality", "baseline_name", "created_at",
)


def _prediction_view(row: dict[str, Any]) -> dict[str, Any]:
    out = {name: row.get(name) for name in _PREDICTION_FIELDS}
    out["is_baseline"] = bool(row.get("is_baseline"))
    for name in ("p", "range", "liquidity", "explanation"):
        out[name] = row.get(name) or {}
    return out


def predictions_data(
    store: Store, *, day: str | None = None, stage: str | None = None, baselines: bool = False
) -> dict[str, Any]:
    """Predictions of ``day`` (default: the latest date that has any). ``baselines`` adds the baseline picks."""
    dates = store.prediction_dates()
    latest = dates[-1] if dates else None
    target = day or latest
    rows = store.predictions_for(target, stage=stage, baselines=None if baselines else False) if target else []
    return {
        "date": target, "latest_date": latest, "available_dates": dates[-60:][::-1], "stage": stage,
        "include_baselines": baselines, "n": len(rows), "predictions": [_prediction_view(row) for row in rows],
    }


_GROUP_ORDER = {"hit rate": 0, "calibration": 1, "vs baselines": 2, "other": 3}


def _metric_group(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("baseline", "lift", "skill", "_vs_", "vs_")):
        return "vs baselines"
    if any(token in lowered for token in ("brier", "ece", "calib", "reliab", "logloss")):
        return "calibration"
    return "hit rate" if "hit" in lowered else "other"


def _metric_view(row: dict[str, Any]) -> dict[str, Any]:
    n = row.get("n")
    return {
        "slice": row["slice"], "name": row["name"], "group": _metric_group(row["name"]), "value": row.get("value"),
        "n": n, "ci_low": row.get("ci_low"), "ci_high": row.get("ci_high"), "low_n": n is None or n < LOW_N,
        "extra": jsonutil.loads(row.get("extra_json")),
    }


def performance_data(store: Store) -> dict[str, Any]:
    """Latest evaluation metrics per scope (hit rate with CI, calibration, baselines, slices), always with n."""
    scopes: dict[str, Any] = {}
    names = [r["scope"] for r in store.conn.execute(
        "SELECT DISTINCT scope FROM eval_metrics ORDER BY scope LIMIT ?", (MAX_SCOPES,)
    ).fetchall()]
    for scope in names:
        rows = store.latest_metrics(scope)[:MAX_METRIC_ROWS]
        metrics = sorted(
            (_metric_view(row) for row in rows),
            key=lambda m: (_GROUP_ORDER[m["group"]], m["slice"] != "all", m["slice"], m["name"]),
        )
        scopes[scope] = {"as_of_date": rows[0]["as_of_date"] if rows else None, "metrics": metrics}
    counts = store.conn.execute(
        "SELECT COALESCE(SUM(validated = 1), 0) AS validated, COALESCE(SUM(validated = 0), 0) AS provisional, "
        "COUNT(DISTINCT CASE WHEN validated = 1 THEN trade_date END) AS days FROM outcomes"
    ).fetchone()
    return {
        "scopes": scopes,
        "outcomes": {"validated": counts["validated"], "provisional": counts["provisional"], "days": counts["days"]},
        "min_reliable_n": LOW_N,
    }


def overview_data(store: Store, ctx: _Context) -> dict[str, Any]:
    now = ctx.now()
    active: dict[str, Any] = {}
    for stage in STAGES:
        model = store.active_model(stage)
        active[stage] = _model_view(model) if model else None
    latest = store.conn.execute("SELECT MAX(trade_date) AS latest FROM predictions").fetchone()["latest"]
    return {
        "mode": ctx.settings.mode,
        "generated_at": iso(now),
        "latest_prediction_date": latest,
        "heartbeat": heartbeat_status(store, now),
        "active_models": active,
        "last_jobs": jobs_data(store, 10),
        "sources": sources_data(store)["summary"],
        "table_counts": ctx.table_counts(store),
    }


# --------------------------------------------------------------------------------------------- the app
def _secure(response: Response, path: str, docs_enabled: bool) -> Response:
    for name, value in SECURITY_HEADERS.items():
        response.headers[name] = value
    if docs_enabled and path in _DOCS_PATHS:
        response.headers["Content-Security-Policy"] = _CSP_DOCS
    return response


async def _guarded(call_next: Callable[[Request], Awaitable[Response]], request: Request) -> Response:
    try:
        return await call_next(request)
    except Exception:
        log.exception("dashboard: unhandled error while serving %s", request.url.path)
        return _json({"error": "internal error"}, 500)


def create_app(
    settings: Settings, *, store_factory: Callable[[], Store] | None = None, clock: Clock | None = None
) -> FastAPI:
    """Build the read-only dashboard app. ``store_factory`` (default: read-only ``Store.open``) is called once
    per request and the store it returns is closed afterwards; ``clock`` is injectable for tests."""
    development = settings.env == "development"
    app = FastAPI(
        title="MONSTER AI dashboard", version=__version__,
        docs_url="/docs" if development else None,
        redoc_url="/redoc" if development else None,
        openapi_url="/openapi.json" if development else None,
    )
    ctx = _Context(settings, store_factory, clock)
    auth = _Auth(settings.dashboard_token, lambda: ctx.clock) if settings.dashboard_token else None
    allowed_hosts = _allowed_hosts(settings)
    if auth is None:
        log.warning("dashboard: no DASHBOARD_TOKEN set; authentication is off (loopback access only)")

    @app.middleware("http")
    async def guard(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        path = request.url.path
        if allowed_hosts is not None and _request_hostname(request) not in allowed_hosts:
            response = _json({"error": "invalid host header"}, 400)
        else:
            denied = auth.check(request) if auth is not None and path != "/healthz" else None
            response = denied if denied is not None else await _guarded(call_next, request)
        return _secure(response, path, development)

    @app.exception_handler(ApiError)
    async def api_error(_request: Request, exc: ApiError) -> Response:
        return _json({"error": exc.message}, exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError) -> Response:
        return _json({"error": "invalid request parameters"}, 400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, exc: StarletteHTTPException) -> Response:
        messages = {404: "not found", 405: "method not allowed"}
        return _json({"error": messages.get(exc.status_code, "request failed")}, exc.status_code)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> Response:
        ok = ctx.healthy()
        return _json({"status": "ok" if ok else "degraded"}, 200 if ok else 503)

    @app.get("/api/overview")
    def api_overview() -> Response:
        with ctx.store() as store:
            return _json(overview_data(store, ctx))

    @app.get("/api/predictions")
    def api_predictions(
        day: str | None = Query(None, alias="date", max_length=10, description="YYYY-MM-DD; default: latest"),
        stage: str | None = Query(None, max_length=8, description="pre or final; default: both"),
        baselines: str | None = Query(None, max_length=8, description="true to include baseline picks"),
    ) -> Response:
        parsed = (parse_date_param(day), _parse_stage(stage), _parse_flag(baselines, "baselines"))
        with ctx.store() as store:
            return _json(predictions_data(store, day=parsed[0], stage=parsed[1], baselines=parsed[2]))

    @app.get("/api/performance")
    def api_performance() -> Response:
        with ctx.store() as store:
            return _json(performance_data(store))

    @app.get("/api/models")
    def api_models() -> Response:
        with ctx.store() as store:
            models = models_data(store)
        return _json({"n": len(models), "models": models})

    @app.get("/api/jobs")
    def api_jobs(limit: str | None = Query(None, max_length=12, description="1-200, default 50")) -> Response:
        size = clamp_int(limit, default=DEFAULT_JOBS, low=1, high=MAX_JOBS, name="limit")
        with ctx.store() as store:
            jobs = jobs_data(store, size)
        return _json({"limit": size, "n": len(jobs), "jobs": jobs})

    @app.get("/api/sources")
    def api_sources() -> Response:
        with ctx.store() as store:
            return _json(sources_data(store))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> Response:
        with ctx.store() as store:
            page = {
                "generated_at": iso(ctx.now()),
                "mode": settings.mode,
                "settings": {"top_n": settings.top_n, "primary_threshold": settings.primary_threshold},
                "overview": overview_data(store, ctx),
                "predictions": predictions_data(store),
                "performance": performance_data(store),
                "jobs": jobs_data(store, 25),
                "sources": sources_data(store)["sources"],
                "models": models_data(store),
            }
        return HTMLResponse(render_page(page))

    return app


def run_dashboard(settings: Settings, *, store_factory: Callable[[], Store] | None = None) -> None:
    """Serve the dashboard with uvicorn (single worker, no access log). Refuses to start without a token unless
    bound to a loopback address."""
    if not settings.dashboard_token and not is_loopback_host(settings.dashboard_host):
        raise ConfigError("DASHBOARD_TOKEN is required unless the dashboard is bound to a loopback address")
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to serve the dashboard: pip install -r requirements-dashboard.txt"
        ) from exc
    uvicorn.run(
        create_app(settings, store_factory=store_factory), host=settings.dashboard_host,
        port=settings.dashboard_port, log_level="warning", access_log=False, workers=1, server_header=False,
    )


def serve(settings: Settings) -> int:
    """``run_dashboard`` for command-line use: returns a process exit code (0 after a clean stop, 1 when the
    configuration forbids starting)."""
    try:
        run_dashboard(settings)
    except ConfigError as exc:
        log.error("dashboard: %s", exc)
        return 1
    return 0


def main() -> int:
    """``python -m dashboard.app``: serve the dashboard from the environment configuration."""
    settings = load_settings()
    setup_logging(settings.log_dir, settings.log_level, secrets=settings.secrets(), filename="dashboard.log")
    if not settings.dashboard_enabled:
        log.info("dashboard is disabled (DASHBOARD_ENABLED=false); nothing to run")
        return 0
    return serve(settings)


if __name__ == "__main__":
    raise SystemExit(main())
