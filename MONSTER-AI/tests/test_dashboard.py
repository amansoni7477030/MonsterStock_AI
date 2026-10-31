"""Dashboard tests: auth matrix, lockout, headers, XSS, validation, read-only guarantees, health, startup rules."""

from __future__ import annotations

import base64
import dataclasses
import importlib
import json
import sqlite3
import sys
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import dashboard
from app.errors import ConfigError
from dashboard import app as dash
from dashboard import templates as tpl
from database.store import Store

TOKEN = "dashboard-token-0123456789"
BEARER = {"Authorization": f"Bearer {TOKEN}"}
XSS = "<script>alert(1)</script>"
XSS_ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"
SECURITY_HEADERS = {
    "content-security-policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
                               "frame-ancestors 'none'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "x-frame-options": "DENY",
}


def basic(user: str, password: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


def authed_settings(settings, **changes):
    return dataclasses.replace(settings, dashboard_token=TOKEN, **changes)


def client_for(settings, fake_clock, **kwargs) -> TestClient:
    return TestClient(dash.create_app(settings, clock=fake_clock), **kwargs)


# -------------------------------------------------------------------------------------------- fixtures
def _source(store, source_id, criticality):
    store.register_source(source_id=source_id, provider=f"{source_id}-provider", endpoint="x", coverage="c",
                          timestamp_meaning="t", update_frequency="d", history_depth="10y", rate_limits="1/s",
                          license_notes="research", failure_behavior="retry", backup_source=None,
                          criticality=criticality)


def _predictions(store, trade_date, stage, symbols, baselines=()):
    snapshot = store.insert_snapshot(
        trade_date=trade_date, stage=stage, cutoff_ts=f"{trade_date}T03:42:00Z", feature_version="fv1",
        model_version="m-final-1", code_version="c1", payload={"d": trade_date, "s": stage}, data_quality={})
    rows = [{"symbol": symbol, "rank": rank, "tier": "STRONG" if rank == 1 else "WATCH", "model_version": "m-final-1",
             "model_status": "shadow", "p": {"hit_3": 0.52 - 0.05 * rank, "unusual": 0.31},
             "range": {"q50": 2.5, "q90": 5.25}, "uncertainty": 0.21, "data_quality": "ok",
             "explanation": {"catalysts": [{"headline": f"headline for {symbol}"}]}}
            for rank, symbol in enumerate(symbols, start=1)]
    rows += [{"symbol": "BASELINE_PICK", "rank": 1, "tier": "WATCH", "model_version": "baseline",
              "model_status": "baseline", "p": {"hit_3": 0.2}, "range": {}, "explanation": {}, "is_baseline": True,
              "baseline_name": name} for name in baselines]
    return store.insert_predictions(snapshot, trade_date, stage, rows)


@pytest.fixture()
def db(settings, store, fake_clock):
    """A database with one of everything, including hostile strings that must be escaped."""
    _source(store, "yahoo_daily", "critical")
    _source(store, "nse_preopen", "critical")
    _source(store, "news_rss", "optional")
    _source(store, "legacy_feed", "optional")
    _source(store, "never_run", "important")
    store.record_source_result("yahoo_daily", True, latency_ms=80.0, data_ts="2026-10-30")
    for _ in range(3):
        store.record_source_result("nse_preopen", False, error=f"HTTP 503 {XSS} token=abcdef0123456789")
    store.record_source_result("news_rss", False, error="timeout")
    store.record_source_result("legacy_feed", True)
    store.set_source_enabled("legacy_feed", False)

    store.register_model(model_version="m-final-1", stage="final", status="shadow", metrics={"brier": 0.2},
                         notes=f"{XSS} secret=abcdef0123456789", train_end="2026-10-30")
    store.register_model(model_version="m-final-2", stage="final")
    store.register_model(model_version="m-pre-1", stage="pre", status="validated")
    store.activate_model("m-final-1")
    store.activate_model("m-pre-1")

    store.job_record("final_scan", "2026-11-02", "success", message="ok")
    store.job_record(f"evening {XSS}", "2026-11-02", "failed", message=f"{XSS} Bearer abcdefghijklmnop0123")
    _predictions(store, "2026-10-30", "final", ["OLD1"])
    final_ids = _predictions(store, "2026-11-02", "final", [XSS, "BBB", "CCC"], baselines=("atr", "random"))
    _predictions(store, "2026-11-02", "pre", ["PRE1"])
    ohlc = {"open": 100, "high": 104, "low": 99, "close": 103}
    store.upsert_outcome(final_ids[0], "2026-11-02", XSS, source="nse_bhavcopy", validated=True, ohlcv=ohlc,
                         metrics={"range_pct": 5.0})
    store.upsert_outcome(final_ids[1], "2026-11-02", "BBB", source="yahoo", validated=False, ohlcv=ohlc,
                         metrics={"range_pct": 3.0})

    store.insert_metrics("2026-11-02", "rolling_30d", [
        {"slice": "all", "name": "top5_hit_rate", "value": 0.4, "n": 24, "ci_low": 0.22, "ci_high": 0.61},
        {"slice": "all", "name": "brier", "value": 0.171, "n": 120},
        {"slice": "all", "name": "brier_skill", "value": 0.08, "n": 120},
        {"slice": "all", "name": "baseline_atr_top5_hit_rate", "value": 0.31, "n": 24, "ci_low": 0.15, "ci_high": 0.5},
        {"slice": "tier:STRONG", "name": "top5_hit_rate", "value": 0.5, "n": 8, "ci_low": 0.2, "ci_high": 0.8},
        {"slice": "regime:high_vol", "name": "top5_hit_rate", "value": 0.45, "n": 40, "ci_low": 0.3, "ci_high": 0.6,
         "extra": {"note": "x"}},
        {"slice": "all", "name": f"odd {XSS}", "value": 1.0},
    ])
    store.insert_metrics("2026-11-02", "daily", [{"name": "top5_hit_rate", "value": 0.6, "n": 5}])
    store.kv_set("daemon.heartbeat", fake_clock.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    return store


@pytest.fixture()
def client(settings, db, fake_clock):
    return client_for(settings, fake_clock)


@pytest.fixture()
def secured(settings, db, fake_clock):
    return client_for(authed_settings(settings), fake_clock)


# ------------------------------------------------------------------------------------- authentication
def test_without_a_token_the_loopback_app_is_open(client):
    assert client.get("/api/overview").status_code == 200
    assert client.get("/").status_code == 200


def test_auth_matrix_with_a_token(secured):
    anonymous = secured.get("/api/overview")
    assert anonymous.status_code == 401 and anonymous.json() == {"error": "unauthorized"}
    challenges = anonymous.headers.get_list("www-authenticate")
    assert challenges[0].startswith("Basic realm=") and challenges[1].startswith("Bearer realm=")
    assert secured.get("/").status_code == 401

    assert secured.get("/api/overview", headers=BEARER).status_code == 200
    assert secured.get("/api/overview", headers={"Authorization": f"bearer {TOKEN}"}).status_code == 200
    assert secured.get("/api/overview", headers=basic("anyone", TOKEN)).status_code == 200
    assert secured.get("/api/overview", headers=basic("", TOKEN)).status_code == 200
    assert secured.get("/", headers=basic("browser", TOKEN)).status_code == 200  # what a browser sends

    wrong = {
        "bearer wrong": {"Authorization": "Bearer nope"},
        "bearer empty": {"Authorization": "Bearer "},
        "basic wrong": basic("u", "nope"),
        "token as user name": basic(TOKEN, ""),
        "basic no colon": {"Authorization": "Basic " + base64.b64encode(TOKEN.encode()).decode()},
        "basic bad base64": {"Authorization": "Basic !!!notbase64!!!"},
        "basic not utf8": {"Authorization": "Basic " + base64.b64encode(b"\xff\xfe:\xff").decode()},
        "other scheme": {"Authorization": f"Digest {TOKEN}"},
        "no scheme": {"Authorization": TOKEN},
        "token prefix": {"Authorization": f"Bearer {TOKEN[:-1]}"},
        "token suffix": {"Authorization": f"Bearer {TOKEN}x"},
        "oversized header": {"Authorization": "Bearer " + "a" * 5000},
    }
    for index, (label, headers) in enumerate(wrong.items()):
        stranger = TestClient(secured.app, client=(f"192.0.2.{index + 1}", 4000))  # own address: no lockout interplay
        assert stranger.get("/api/overview", headers=headers).status_code == 401, label
    probe = TestClient(secured.app, client=("192.0.2.200", 4000))
    assert probe.get(f"/api/overview?token={TOKEN}").status_code == 401  # never accepted in the URL
    assert probe.get("/api/overview", headers={"X-Auth-Token": TOKEN}).status_code == 401


def test_all_routes_are_protected_by_default(secured):
    for path in ("/", "/api/overview", "/api/predictions", "/api/performance", "/api/models", "/api/jobs",
                 "/api/sources", "/nonexistent", "/docs", "/openapi.json"):
        assert secured.get(path).status_code == 401, path
    assert secured.post("/api/overview").status_code == 401
    assert secured.options("/api/overview").status_code == 401


def test_token_comparison_is_constant_time(secured, monkeypatch):
    calls = []
    real = dash.secrets.compare_digest
    monkeypatch.setattr(dash.secrets, "compare_digest", lambda a, b: calls.append((len(a), len(b))) or real(a, b))
    secured.get("/api/overview", headers={"Authorization": "Bearer short"})
    secured.get("/api/overview", headers=BEARER)
    assert calls == [(32, 32), (32, 32)]  # always equal-length digests, whatever the token length


def test_lockout_after_repeated_failures_then_recovery(secured, fake_clock):
    bad = {"Authorization": "Bearer wrong-token"}
    for _ in range(dash.MAX_FAILURES):
        assert secured.get("/api/overview", headers=bad).status_code == 401
    locked = secured.get("/api/overview", headers=BEARER)  # even the right token is refused while locked
    assert locked.status_code == 429 and locked.json()["error"].startswith("too many failed attempts")
    assert 1 <= int(locked.headers["retry-after"]) <= 300
    assert secured.get("/api/overview").status_code == 429  # and so is an anonymous request
    assert secured.get("/healthz").status_code == 200  # health checks stay available
    fake_clock.advance(dash.LOCKOUT_SEC - 1)
    assert secured.get("/api/overview", headers=BEARER).status_code == 429
    fake_clock.advance(2)
    assert secured.get("/api/overview", headers=BEARER).status_code == 200
    for _ in range(dash.MAX_FAILURES - 1):  # a success cleared the counter
        assert secured.get("/api/overview", headers=bad).status_code == 401
    assert secured.get("/api/overview", headers=BEARER).status_code == 200


def test_lockout_is_per_client_and_anonymous_requests_do_not_count(settings, db, fake_clock):
    app = dash.create_app(authed_settings(settings), clock=fake_clock)
    attacker = TestClient(app, client=("203.0.113.9", 1111))
    operator = TestClient(app, client=("198.51.100.7", 2222))
    for _ in range(dash.MAX_FAILURES):
        attacker.get("/api/overview", headers={"Authorization": "Bearer wrong"})
    assert attacker.get("/api/overview", headers=BEARER).status_code == 429
    assert operator.get("/api/overview", headers=BEARER).status_code == 200
    for _ in range(20):  # a browser's first, credential-less request is just a challenge
        assert operator.get("/api/overview").status_code == 401
    assert operator.get("/api/overview", headers=BEARER).status_code == 200


def test_failure_window_expires_and_memory_is_bounded(fake_clock, monkeypatch):
    auth = dash._Auth(TOKEN, lambda: fake_clock)

    def request(ip, header="Bearer wrong"):
        return SimpleNamespace(client=SimpleNamespace(host=ip), headers={"authorization": header})

    for _ in range(dash.MAX_FAILURES - 1):
        auth.check(request("10.0.0.1"))
    fake_clock.advance(dash.LOCKOUT_SEC)  # old failures do not accumulate forever
    auth.check(request("10.0.0.1"))
    assert auth.check(request("10.0.0.1", f"Bearer {TOKEN}")) is None

    monkeypatch.setattr(dash, "MAX_TRACKED_CLIENTS", 10)
    for index in range(40):  # an attacker rotating addresses cannot grow the table without bound
        fake_clock.advance(1)
        auth.check(request(f"10.1.0.{index}"))
    assert len(auth._failures) <= 11
    assert auth.check(SimpleNamespace(client=None, headers={})).status_code == 401  # no peer address at all

    fake_clock.advance(dash.LOCKOUT_SEC)  # now every tracked entry is stale and is dropped when the table is pruned
    for index in range(12):
        auth.check(request(f"10.2.0.{index}"))
    assert len(auth._failures) <= 11 and not any(ip.startswith("10.1.") for ip in auth._failures)


# ------------------------------------------------------------------------------------------- headers
def assert_secure(response):
    for name, value in SECURITY_HEADERS.items():
        assert response.headers[name] == value, name


def test_security_headers_on_every_kind_of_response(secured, fake_clock):
    responses = [
        secured.get("/api/overview", headers=BEARER), secured.get("/", headers=BEARER),
        secured.get("/healthz"), secured.get("/api/overview"),  # 401
        secured.get("/missing", headers=BEARER),  # 404
        secured.post("/api/overview", headers=BEARER),  # 405
        secured.get("/api/predictions?date=bogus", headers=BEARER),  # 400
        secured.get("/api/jobs?limit=" + "9" * 30, headers=BEARER),  # 400 from request validation
    ]
    for _ in range(dash.MAX_FAILURES):
        secured.get("/api/overview", headers={"Authorization": "Bearer wrong"})
    responses.append(secured.get("/api/overview", headers=BEARER))  # 429
    assert {r.status_code for r in responses} == {200, 401, 404, 405, 400, 429}
    for response in responses:
        assert_secure(response)
    assert "unsafe-inline" in responses[0].headers["content-security-policy"]
    assert "script-src" not in responses[0].headers["content-security-policy"]
    assert responses[1].headers["content-type"] == "text/html; charset=utf-8"
    assert responses[0].headers["content-type"] == "application/json"


def test_unhandled_errors_are_generic_json_with_headers(settings, fake_clock, caplog):
    def broken():
        raise RuntimeError("boom /secret/path token=abcdef0123456789")

    app = dash.create_app(settings, store_factory=broken, clock=fake_clock)
    response = TestClient(app).get("/api/overview")
    assert response.status_code == 500 and response.json() == {"error": "internal error"}
    assert "boom" not in response.text and "Traceback" not in response.text and "secret" not in response.text
    assert_secure(response)
    assert "unhandled error" in caplog.text


def test_host_header_is_checked_only_when_unauthenticated(client, secured):
    for host in ("localhost:8080", "127.0.0.1", "[::1]:8080", "LOCALHOST", "testserver"):
        assert client.get("/api/overview", headers={"Host": host}).status_code == 200, host
    for host in ("evil.example.com", "evil.example.com:8080", "127.0.0.1.evil.com"):
        response = client.get("/api/overview", headers={"Host": host})
        assert response.status_code == 400 and response.json() == {"error": "invalid host header"}, host
        assert_secure(response)
    assert client.get("/healthz", headers={"Host": "evil.example.com"}).status_code == 400
    assert secured.get("/api/overview", headers={**BEARER, "Host": "dash.example.com"}).status_code == 200


def test_host_allow_list_includes_the_configured_bind_address(settings, db, fake_clock):
    bound = dataclasses.replace(settings, dashboard_host="127.0.0.1", env="production")
    prod = client_for(bound, fake_clock)
    assert prod.get("/api/overview", headers={"Host": "127.0.0.1:8080"}).status_code == 200
    assert prod.get("/api/overview", headers={"Host": "testserver"}).status_code == 400  # test shortcut is test-only
    assert dash._allowed_hosts(dataclasses.replace(settings, dashboard_host="  ", env="production")) == {
        "localhost", "127.0.0.1", "::1"}
    assert dash._allowed_hosts(dataclasses.replace(settings, dashboard_host="[::1]")) >= {"::1"}
    assert dash._allowed_hosts(authed_settings(settings)) is None  # a token makes the Host header irrelevant


# --------------------------------------------------------------------------------------- docs / schema
def test_docs_and_openapi_are_disabled_in_production(settings, db, fake_clock):
    prod = client_for(authed_settings(dataclasses.replace(settings, env="production")), fake_clock)
    for path in ("/docs", "/redoc", "/openapi.json"):
        response = prod.get(path, headers=BEARER)
        assert response.status_code == 404 and response.json() == {"error": "not found"}, path


def test_docs_are_available_in_development_behind_auth(settings, db, fake_clock):
    dev = client_for(authed_settings(settings, env="development"), fake_clock)
    assert dev.get("/openapi.json").status_code == 401
    schema = dev.get("/openapi.json", headers=BEARER)
    assert schema.status_code == 200 and "/api/predictions" in schema.json()["paths"]
    assert "/healthz" not in schema.json()["paths"] and "/" not in schema.json()["paths"]
    docs = dev.get("/docs", headers=BEARER)
    assert docs.status_code == 200 and "cdn.jsdelivr.net" in docs.headers["content-security-policy"]
    assert_secure_except_csp = {k: v for k, v in SECURITY_HEADERS.items() if k != "content-security-policy"}
    for name, value in assert_secure_except_csp.items():
        assert docs.headers[name] == value
    assert "cdn.jsdelivr.net" not in dev.get("/api/overview", headers=BEARER).headers["content-security-policy"]


# --------------------------------------------------------------------------------------- read-only
def test_only_get_routes_exist_and_every_write_verb_is_refused(client, db):
    methods = {m for route in client.app.routes for m in (getattr(route, "methods", None) or set())}
    assert methods <= {"GET", "HEAD"}, methods
    before = db.table_counts()
    paths = ("/", "/healthz", "/api/overview", "/api/predictions", "/api/performance", "/api/models",
             "/api/jobs", "/api/sources", "/api/anything")
    for path in paths:
        for verb in ("post", "put", "patch", "delete"):
            response = getattr(client, verb)(path)
            assert response.status_code in {404, 405}, (verb, path)
        assert client.head(path).status_code in {404, 405}
    assert db.table_counts() == before


def test_every_request_opens_the_database_read_only_and_closes_it(settings, db, fake_clock, monkeypatch):
    original = Store.open
    opened, write_probes, closed = [], [], []

    def spy(path, **kwargs):
        instance = original(path, **kwargs)
        opened.append(kwargs)
        try:
            instance.kv_set("probe", "1")
        except sqlite3.OperationalError as exc:
            write_probes.append(str(exc))
        real_close = instance.close
        instance.close = lambda: (closed.append(1), real_close())[1]
        return instance

    monkeypatch.setattr(dash.Store, "open", staticmethod(spy))
    app_client = client_for(settings, fake_clock)
    for path in ("/api/overview", "/api/predictions", "/api/performance", "/api/models", "/api/jobs",
                 "/api/sources", "/healthz", "/"):
        assert app_client.get(path).status_code in {200, 503}, path
    assert len(opened) == len(closed) == 8
    assert all(kwargs == {"readonly": True} for kwargs in opened)
    assert len(write_probes) == 8 and all("readonly" in text for text in write_probes)
    monkeypatch.undo()
    assert db.kv_get("probe") is None


def test_stores_are_closed_even_when_the_query_fails(settings, db, fake_clock):
    closed = []

    def factory():
        instance = Store.open(settings.db_path, readonly=True)
        instance.conn.close()  # every query now fails with sqlite3.ProgrammingError
        instance.close = lambda: closed.append(1)
        return instance

    response = TestClient(dash.create_app(settings, store_factory=factory, clock=fake_clock)).get("/api/performance")
    assert response.status_code == 503 and response.json() == {"error": "database unavailable"}
    assert closed == [1]


def test_database_problems_become_generic_503s(settings, tmp_path, fake_clock):
    missing = dataclasses.replace(settings, db_path=tmp_path / "does-not-exist.db")
    app_client = client_for(missing, fake_clock)
    for path in ("/api/overview", "/api/predictions", "/api/performance", "/api/models", "/api/jobs",
                 "/api/sources", "/"):
        response = app_client.get(path)
        assert response.status_code == 503 and response.json() == {"error": "database unavailable"}, path
        assert str(tmp_path) not in response.text
    unmigrated = tmp_path / "empty.db"
    sqlite3.connect(unmigrated).close()
    bare = client_for(dataclasses.replace(settings, db_path=unmigrated), fake_clock)
    assert bare.get("/api/overview").status_code == 503
    for error in (sqlite3.OperationalError("unable to open database file"), OSError("denied")):
        def failing(error=error):
            raise error

        failing_client = TestClient(dash.create_app(settings, store_factory=failing, clock=fake_clock))
        assert failing_client.get("/api/models").status_code == 503


# --------------------------------------------------------------------------------------------- health
def test_healthz_reports_ok_with_nothing_else(client, secured):
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}
    assert secured.get("/healthz").json() == {"status": "ok"}  # no credentials needed
    assert secured.get("/healthz", headers={"Authorization": "Bearer wrong"}).status_code == 200
    text = response.text
    for forbidden in (str(dash.__file__), "monster.db", "version", "m-final-1", "yahoo", "table", "heartbeat"):
        assert forbidden not in text


def test_healthz_degrades_on_stale_missing_heartbeat_or_dead_database(settings, db, fake_clock, tmp_path):
    app_client = client_for(settings, fake_clock)
    fake_clock.advance(dash.HEARTBEAT_MAX_AGE_SEC - 1)
    assert app_client.get("/healthz").status_code == 200
    fake_clock.advance(2)
    stale = app_client.get("/healthz")
    assert stale.status_code == 503 and stale.json() == {"status": "degraded"}
    db.kv_set("daemon.heartbeat", "alive")  # a fresh write time is what counts, not the value
    assert app_client.get("/healthz").json() == {"status": "ok"}
    db.kv_delete("daemon.heartbeat")
    assert app_client.get("/healthz").json() == {"status": "degraded"}
    dead = client_for(dataclasses.replace(settings, db_path=tmp_path / "gone.db"), fake_clock)
    response = dead.get("/healthz")
    assert response.status_code == 503 and response.json() == {"status": "degraded"}
    assert_secure(response)


def test_heartbeat_status_values(db, fake_clock):
    now = fake_clock.now()
    assert dash.heartbeat_status(db, now) == {"at": "2026-11-02T03:00:00Z", "age_seconds": 0, "status": "ok"}
    assert dash.heartbeat_status(db, now.replace(year=2025))["age_seconds"] == 0  # clock skew never goes negative
    db.kv_delete("daemon.heartbeat")
    assert dash.heartbeat_status(db, now)["status"] == "missing"
    db.conn.execute("INSERT INTO kv (key, value, updated_at) VALUES ('daemon.heartbeat', '2026-11-02T02:59:00Z', 'x')")
    assert dash.heartbeat_status(db, now) == {"at": "2026-11-02T02:59:00Z", "age_seconds": 60, "status": "ok"}


# ------------------------------------------------------------------------------------------ JSON API
def test_overview(client, db, fake_clock):
    data = client.get("/api/overview").json()
    assert data["mode"] == "shadow" and data["generated_at"] == "2026-11-02T03:00:00Z"
    assert data["latest_prediction_date"] == "2026-11-02"
    assert data["heartbeat"]["status"] == "ok"
    assert data["active_models"]["final"]["model_version"] == "m-final-1"
    assert data["active_models"]["final"]["status"] == "shadow" and data["active_models"]["pre"]["status"] == "validated"
    assert len(data["last_jobs"]) == 2 and data["last_jobs"][0]["status"] == "failed"
    assert data["sources"] == {"total": 5, "ok": 1, "degraded": 1, "failing": 1, "unknown": 1, "disabled": 1,
                               "critical_failing": ["nse_preopen"]}
    assert data["table_counts"]["predictions"] == 7 and "kv" in data["table_counts"]
    assert "artifact_path" not in json.dumps(data) and "abcdef0123456789" not in json.dumps(data)


def test_overview_table_counts_are_cached_briefly(client, db, fake_clock):
    first = client.get("/api/overview").json()["table_counts"]["kv"]
    db.kv_set("extra", "1")
    assert client.get("/api/overview").json()["table_counts"]["kv"] == first  # served from the 30 s cache
    fake_clock.advance(dash.COUNTS_TTL_SEC + 1)
    assert client.get("/api/overview").json()["table_counts"]["kv"] == first + 1


def test_overview_on_an_empty_database(settings, store, fake_clock):
    data = client_for(settings, fake_clock).get("/api/overview").json()
    assert data["latest_prediction_date"] is None and data["active_models"] == {"pre": None, "final": None}
    assert data["heartbeat"] == {"at": None, "age_seconds": None, "status": "missing"}
    assert data["last_jobs"] == [] and data["sources"]["total"] == 0


def test_predictions_default_to_the_latest_date_and_hide_baselines(client):
    data = client.get("/api/predictions").json()
    assert data["date"] == "2026-11-02" and data["latest_date"] == "2026-11-02"
    assert data["available_dates"] == ["2026-11-02", "2026-10-30"] and data["include_baselines"] is False
    assert data["n"] == 4 and all(row["is_baseline"] is False for row in data["predictions"])
    final = [row for row in data["predictions"] if row["stage"] == "final"]
    assert [row["rank"] for row in final] == [1, 2, 3] and final[0]["symbol"] == XSS
    assert final[0]["p"]["hit_3"] == pytest.approx(0.47) and final[0]["range"]["q90"] == 5.25
    assert final[0]["explanation"]["catalysts"][0]["headline"] == f"headline for {XSS}"
    assert {"prediction_id", "tier", "model_version", "model_status", "uncertainty", "data_quality"} <= set(final[0])


def test_predictions_filters_and_baselines_flag(client):
    older = client.get("/api/predictions?date=2026-10-30").json()
    assert older["date"] == "2026-10-30" and [row["symbol"] for row in older["predictions"]] == ["OLD1"]
    empty = client.get("/api/predictions?date=2026-11-03").json()
    assert empty["date"] == "2026-11-03" and empty["n"] == 0 and empty["predictions"] == []
    with_baselines = client.get("/api/predictions?baselines=true&stage=final").json()
    assert with_baselines["include_baselines"] is True and with_baselines["stage"] == "final"
    flags = {(row["baseline_name"], row["is_baseline"]) for row in with_baselines["predictions"]}
    assert flags == {("", False), ("atr", True), ("random", True)}
    assert [row["stage"] for row in client.get("/api/predictions?stage=pre").json()["predictions"]] == ["pre"]
    for truthy in ("1", "TRUE", "yes", "on"):
        assert client.get(f"/api/predictions?baselines={truthy}").json()["include_baselines"] is True
    for falsy in ("0", "false", "no", "off", ""):
        assert client.get(f"/api/predictions?baselines={falsy}").json()["include_baselines"] is False


@pytest.mark.parametrize("value", [
    "2026-13-45", "2026-02-30", "20261102", "2026-1-2", "abc", "2026-11-02T00:00", "\uff12\uff10\uff12\uff16-11-02",
    "1999-01-01", "2101-01-01", " 2026-11-02", "2026-11-02 ", "2026-11-02'", "x' OR '1'='1", "../../etc", "%00",
])
def test_invalid_dates_are_rejected_without_echoing_the_input(client, value):
    response = client.get("/api/predictions", params={"date": value})
    assert response.status_code == 400 and set(response.json()) == {"error"}
    assert value.strip() not in response.text or value.strip() == ""
    assert_secure(response)


def test_overlong_or_malformed_parameters_are_generic_400s(client):
    for query in ("date=2026-11-02-extra", "stage=finalized", "baselines=maybe", "stage=x", "limit=" + "1" * 13):
        path = "/api/jobs" if query.startswith("limit") else "/api/predictions"
        response = client.get(f"{path}?{query}")
        assert response.status_code == 400 and set(response.json()) == {"error"}, query
    assert "detail" not in client.get("/api/predictions?date=" + "x" * 50).text


def test_parameter_parsers_directly():
    assert dash.parse_date_param(None) is None and dash.parse_date_param("") is None
    assert dash.parse_date_param("2026-11-02") == "2026-11-02"
    assert dash.clamp_int(None, default=50, low=1, high=200, name="limit") == 50
    assert dash.clamp_int("999999", default=50, low=1, high=200, name="limit") == 200
    assert dash.clamp_int("-3", default=50, low=1, high=200, name="limit") == 1
    with pytest.raises(dash.ApiError):
        dash.clamp_int("1e3", default=50, low=1, high=200, name="limit")


def test_jobs_limit_is_clamped(settings, db, fake_clock):
    for index in range(250):
        db.job_record("tick", f"run-{index}", "success", message="x")
    app_client = client_for(settings, fake_clock)
    sizes = {query: len(app_client.get(f"/api/jobs{query}").json()["jobs"]) for query in (
        "", "?limit=100000", "?limit=200", "?limit=201", "?limit=0", "?limit=-5", "?limit=7", "?limit=")}
    assert sizes == {"": 50, "?limit=100000": 200, "?limit=200": 200, "?limit=201": 200, "?limit=0": 1,
                     "?limit=-5": 1, "?limit=7": 7, "?limit=": 50}
    payload = app_client.get("/api/jobs?limit=3").json()
    assert payload["limit"] == 3 and payload["n"] == 3 and "pid" not in payload["jobs"][0]
    assert payload["jobs"][0]["run_key"] == "run-249"  # newest first
    for query in ("?limit=abc", "?limit=1.5", "?limit=" + "9" * 20):
        assert app_client.get(f"/api/jobs{query}").status_code == 400


def test_jobs_messages_are_redacted(client):
    jobs = client.get("/api/jobs").json()["jobs"]
    assert "abcdefghijklmnop0123" not in json.dumps(jobs) and "Bearer <redacted>" in jobs[0]["message"]


def test_models_endpoint(client):
    data = client.get("/api/models").json()
    assert data["n"] == 3
    by_version = {m["model_version"]: m for m in data["models"]}
    assert by_version["m-final-1"]["is_active"] is True and by_version["m-final-2"]["is_active"] is False
    assert by_version["m-final-2"]["status"] == "candidate" and by_version["m-pre-1"]["stage"] == "pre"
    assert by_version["m-final-1"]["metrics"] == {"brier": 0.2} and "abcdef0123456789" not in json.dumps(data)
    assert "artifact_path" not in by_version["m-final-1"] and "artifact_sha256" not in by_version["m-final-1"]


def test_sources_endpoint(client):
    data = client.get("/api/sources").json()
    by_id = {s["source_id"]: s for s in data["sources"]}
    assert {sid: s["status"] for sid, s in by_id.items()} == {
        "yahoo_daily": "ok", "nse_preopen": "failing", "news_rss": "degraded", "legacy_feed": "disabled",
        "never_run": "unknown"}
    assert by_id["nse_preopen"]["consecutive_failures"] == 3 and by_id["yahoo_daily"]["last_data_ts"] == "2026-10-30"
    assert "abcdef0123456789" not in json.dumps(data) and "token=<redacted>" in by_id["nse_preopen"]["last_error"]
    assert data["summary"]["critical_failing"] == ["nse_preopen"]


def test_performance_reports_every_metric_with_its_sample_size(client):
    data = client.get("/api/performance").json()
    assert data["min_reliable_n"] == 30 and data["outcomes"] == {"validated": 1, "provisional": 1, "days": 1}
    assert set(data["scopes"]) == {"daily", "rolling_30d"}
    block = data["scopes"]["rolling_30d"]
    assert block["as_of_date"] == "2026-11-02"
    metrics = block["metrics"]
    assert all({"n", "low_n", "ci_low", "ci_high", "group", "slice", "value"} <= set(m) for m in metrics)
    groups = [m["group"] for m in metrics]
    assert groups == sorted(groups, key=lambda g: dash._GROUP_ORDER[g])  # hit rates, calibration, baselines, other
    top = next(m for m in metrics if m["name"] == "top5_hit_rate" and m["slice"] == "all")
    assert (top["n"], top["low_n"], top["ci_low"], top["ci_high"]) == (24, True, 0.22, 0.61)
    brier = next(m for m in metrics if m["name"] == "brier")
    assert brier["group"] == "calibration" and brier["low_n"] is False and brier["n"] == 120
    assert next(m for m in metrics if m["name"] == "brier_skill")["group"] == "vs baselines"
    assert next(m for m in metrics if m["name"].startswith("baseline_atr"))["group"] == "vs baselines"
    slices = {m["slice"] for m in metrics}
    assert {"all", "tier:STRONG", "regime:high_vol"} <= slices
    overall_first = [m["slice"] != "all" for m in metrics if m["group"] == "hit rate"]
    assert overall_first == sorted(overall_first)
    unsized = next(m for m in metrics if m["name"].startswith("odd"))
    assert unsized["n"] is None and unsized["low_n"] is True  # an unknown sample size is flagged, never hidden
    assert next(m for m in metrics if m["slice"] == "regime:high_vol")["extra"] == {"note": "x"}


def test_performance_on_an_empty_database(settings, store, fake_clock):
    data = client_for(settings, fake_clock).get("/api/performance").json()
    assert data["scopes"] == {} and data["outcomes"] == {"validated": 0, "provisional": 0, "days": 0}


# ------------------------------------------------------------------------------------------- HTML page
class TagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: set[str] = set()
        self.attributes: set[str] = set()
        self.classes: set[str] = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        for name, value in attrs:
            self.attributes.add(name)
            if name == "class":
                self.classes.update((value or "").split())


def test_html_page_escapes_every_dynamic_value(client):
    response = client.get("/")
    html = response.text
    assert response.status_code == 200 and response.headers["content-type"] == "text/html; charset=utf-8"
    assert XSS not in html and "<script" not in html.lower() and "javascript:" not in html.lower()
    assert XSS_ESCAPED in html  # the hostile symbol, job name, job message, source error, headline are all escaped
    assert html.count(XSS_ESCAPED) >= 5
    parser = TagCollector()
    parser.feed(html)
    allowed_tags = {"html", "head", "meta", "title", "style", "body", "header", "h1", "h2", "h3", "p", "span", "main",
                    "div", "table", "thead", "tbody", "tr", "th", "td", "footer"}
    assert parser.tags <= allowed_tags, parser.tags - allowed_tags
    assert parser.attributes <= {"charset", "name", "content", "http-equiv", "lang", "class"}, parser.attributes
    assert parser.classes <= {"note", "muted", "badge", "ok", "warn", "bad", "wrap"}, parser.classes
    assert "http://" not in html and "https://" not in html  # no external assets of any kind
    assert "abcdef0123456789" not in html and "abcdefghijklmnop0123" not in html  # secrets redacted


def test_html_page_content(client):
    html = client.get("/").text
    for expected in ("MONSTER AI", "SHADOW", "not trading advice", "never places orders", "Latest candidates (2026-11-02)",
                     "Final scan - top 3", "Pre-scan - top 1", "P(range &gt;= 3%)", "47.0%", "headline for BBB",
                     "m-final-1", "Feed health", "yahoo_daily", "failing", "Recent jobs", "Performance",
                     "Validated outcomes: 1 over 1 trading day(s)", "top5_hit_rate", "n &lt; 30", "tier:STRONG",
                     "22.0% - 61.0%", "regime:high_vol", "rolling_30d (as of 2026-11-02)"):
        assert expected in html, expected
    assert "OLD1" not in html and "BASELINE_PICK" not in html  # older dates and baseline picks are API-only


def test_html_page_with_an_empty_database(settings, store, fake_clock):
    html = client_for(settings, fake_clock).get("/").text
    for expected in ("No predictions recorded yet.", "No models registered yet.", "No jobs have run yet.",
                     "No data sources registered yet.", "No evaluation metrics yet", "daemon"):
        assert expected in html


def test_html_page_requires_the_token_and_respects_top_n(settings, db, fake_clock):
    one = client_for(authed_settings(settings, top_n=1), fake_clock)
    assert one.get("/").status_code == 401
    html = one.get("/", headers=basic("me", TOKEN)).text
    assert "Final scan - top 1" in html and "headline for BBB" not in html


# ----------------------------------------------------------------------------------------- templates
def test_template_helpers_escape_and_format():
    assert tpl.esc(None) == "" and tpl.esc('<a href="x">&\'') == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;"
    assert tpl.badge("<b>", "ok") == '<span class="badge ok">&lt;b&gt;</span>'
    assert tpl.badge("x", "evil\" onmouseover=\"alert(1)") == '<span class="badge muted">x</span>'
    assert tpl.status_badge("FAILED").startswith('<span class="badge bad">')
    assert tpl.table(["h"], [[tpl.Raw("<i>ok</i>"), "<i>no</i>"]]).count("<i>") == 1
    assert "empty here" in tpl.table(["h"], [], empty="empty here")
    assert [tpl.fmt_pct(v) for v in (0.5, "0.25", None, "x", float("nan"), True)] == [
        "50.0%", "25.0%", "n/a", "n/a", "n/a", "n/a"]
    assert tpl.fmt_num(1.239) == "1.24" and tpl.fmt_num(None) == "n/a"
    assert tpl.fmt_ts("2026-11-02T03:00:00Z") == "2026-11-02 03:00:00Z" and tpl.fmt_ts(None) == "n/a"
    assert tpl.fmt_text("a  b\n c") == "a b c" and tpl.fmt_text("x" * 300, 10) == "xxxxxxxxx..."
    assert tpl._metric_cell("top5_hit_rate", 0.5) == "50.0%" and tpl._metric_cell("brier", 0.17123) == "0.1712"


def test_explanation_summaries():
    assert tpl.explain("plain text") == "plain text" and tpl.explain(None) == "" and tpl.explain(5) == ""
    assert tpl.explain({"catalysts": [{"headline": "H"}]}) == "H" and tpl.explain({"catalysts": ["raw"]}) == "raw"
    assert tpl.explain({"catalysts": [], "summary": "S"}) == "S" and tpl.explain({"reason": "R"}) == "R"
    assert tpl.explain({"catalysts": [{"headline": ""}], "summary": "S"}) == "S"  # an empty headline falls through
    assert tpl.explain({"drivers": ["a", "b", "c", "d"]}) == "a, b, c" and tpl.explain({"other": 1}) == ""


def test_render_page_tolerates_sparse_data():
    page = {"overview": {}, "predictions": {"predictions": [{"symbol": "S", "stage": "final", "p": None}]}}
    html = tpl.render_page(page)
    assert "<html" in html and "n/a" in html and ">S<" in html


# ----------------------------------------------------------------------------------------- start-up rules
@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "[::1]", "127.0.0.5", "LOCALHOST"])
def test_loopback_hosts(host):
    assert dash.is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "", "::", "10.0.0.1"])
def test_non_loopback_hosts(host):
    assert not dash.is_loopback_host(host)


@pytest.fixture()
def uvicorn_calls(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    return calls


def test_run_dashboard_refuses_to_serve_unauthenticated_off_loopback(settings, uvicorn_calls):
    exposed = dataclasses.replace(settings, dashboard_host="0.0.0.0", dashboard_token=None)
    with pytest.raises(ConfigError, match="DASHBOARD_TOKEN is required"):
        dash.run_dashboard(exposed)
    with pytest.raises(ConfigError):
        dash.run_dashboard(dataclasses.replace(exposed, dashboard_host="192.168.1.10"))
    assert uvicorn_calls == []


def test_run_dashboard_starts_hardened_uvicorn(settings, uvicorn_calls):
    dash.run_dashboard(dataclasses.replace(settings, dashboard_host="127.0.0.1", dashboard_port=8123))
    app, kwargs = uvicorn_calls[0]
    assert app.title == "MONSTER AI dashboard"
    assert kwargs == {"host": "127.0.0.1", "port": 8123, "log_level": "warning", "access_log": False, "workers": 1,
                      "server_header": False}
    for host in ("::1", "localhost"):
        dash.run_dashboard(dataclasses.replace(settings, dashboard_host=host))
    exposed = dataclasses.replace(settings, dashboard_host="0.0.0.0", dashboard_token=TOKEN)
    dash.run_dashboard(exposed)  # allowed because a token protects it
    assert len(uvicorn_calls) == 4 and uvicorn_calls[-1][1]["host"] == "0.0.0.0"


def test_run_dashboard_reports_missing_uvicorn(settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(ImportError, match="requirements-dashboard.txt"):
        dash.run_dashboard(settings)


def test_main_honours_the_enabled_flag(settings, monkeypatch, uvicorn_calls):
    monkeypatch.setattr(dash, "load_settings", lambda: dataclasses.replace(settings, dashboard_enabled=False))
    assert dash.main() == 0 and uvicorn_calls == []
    monkeypatch.setattr(dash, "load_settings", lambda: dataclasses.replace(settings, dashboard_enabled=True))
    assert dash.main() == 0 and len(uvicorn_calls) == 1


def test_serve_returns_exit_codes_for_the_command_line(settings, uvicorn_calls, caplog):
    assert dashboard.serve(settings) == 0 and len(uvicorn_calls) == 1
    refused = dataclasses.replace(settings, dashboard_host="0.0.0.0", dashboard_token=None)
    assert dash.serve(refused) == 1 and len(uvicorn_calls) == 1  # nothing was started
    assert "DASHBOARD_TOKEN is required" in caplog.text


# ----------------------------------------------------------------------------------- optional import
def test_importing_without_the_extras_fails_with_a_clear_message(monkeypatch):
    for name in ("dashboard", "dashboard.app", "dashboard.templates"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(ImportError, match=r"optional dependencies.*requirements-dashboard\.txt"):
        importlib.import_module("dashboard")


def test_unrelated_import_errors_are_not_disguised(monkeypatch):
    for name in ("dashboard", "dashboard.app", "dashboard.templates"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "dashboard.templates", None)
    with pytest.raises(ImportError) as err:
        importlib.import_module("dashboard")
    assert "requirements-dashboard" not in str(err.value)


def test_package_exports():
    assert dashboard.create_app is dash.create_app and dashboard.run_dashboard is dash.run_dashboard
    assert dashboard.serve is dash.serve and set(dashboard.__all__) == {"create_app", "run_dashboard", "serve"}


def test_core_does_not_import_the_dashboard_at_module_level():
    import pathlib
    import re

    root = pathlib.Path(dash.__file__).resolve().parent.parent
    pattern = re.compile(r"^(import|from)\s+(dashboard|fastapi|starlette|pydantic|uvicorn|httpx)\b", re.M)
    for package in ("app", "database", "models", "ranking", "features", "ingestion", "automation", "backtesting",
                    "notifications", "news_intelligence", "event_graph"):
        for path in (root / package).rglob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), path  # lazy, function-level imports are fine


def test_importing_the_storage_and_signing_modules_leaves_the_web_stack_unloaded():
    import pathlib
    import subprocess

    root = pathlib.Path(dash.__file__).resolve().parent.parent
    code = ("import sys, database.object_store, models.signing, models.bundle; "
            "print(','.join(m for m in ('fastapi', 'starlette', 'pydantic', 'uvicorn', 'httpx', 'dashboard') "
            "if m in sys.modules))")
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=120,
                            check=True)
    assert result.stdout.strip() == ""
