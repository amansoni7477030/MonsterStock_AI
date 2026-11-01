"""HTTP transport: curl_cffi first (browser impersonation), requests as the fallback, adaptive politeness, safe logs.

Two layers of tests: scripted fakes (fast, deterministic) and the REAL curl_cffi against a loopback ``http.server``
(streaming, gzip, cookies, redirects, size cap, retries, connection errors) so the library integration itself is proven
without any external network."""

from __future__ import annotations

import gzip
import json
import logging
import random
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import requests

from app.config import load_settings
from app.errors import DataUnavailable
from ingestion import http as http_mod
from ingestion.http import (
    CFFI_IMPORT_ERROR,
    Headers,
    HttpClient,
    RateLimiter,
    classify_error,
    curl_cffi_info,
    http_stats,
    http_summary,
    reset_http_stats,
    stats_for,
)
from tests.helpers import FakeResponse, FakeSession

needs_curl_cffi = pytest.mark.skipif(http_mod.cffi_requests is None, reason=f"curl_cffi unavailable: {CFFI_IMPORT_ERROR}")


@pytest.fixture(autouse=True)
def _clean_state():
    reset_http_stats()
    http_mod._WARNED.clear()
    yield
    reset_http_stats()
    http_mod._WARNED.clear()


# ------------------------------------------------------------------------------------------ headers
def test_headers_are_case_insensitive_and_read_only():
    headers = Headers({"ETag": '"abc"', "Content-Type": "text/csv"})
    assert headers.get("etag") == '"abc"' and headers["CONTENT-TYPE"] == "text/csv" and headers.get("missing") is None
    assert len(headers) == 2 and sorted(headers) == ["content-type", "etag"]
    assert "etag" in headers and "ETAG" in headers
    with pytest.raises(TypeError):
        headers["x"] = "y"  # type: ignore[index]


def test_response_headers_of_a_client_are_case_insensitive(fake_clock):
    session = FakeSession([FakeResponse(200, b"x", headers={"Last-Modified": "Sat, 31 Oct 2026 10:00:00 GMT"})])
    client = HttpClient("unit", session=session, clock=fake_clock)
    assert client.get("https://example.com/x").headers.get("last-modified").startswith("Sat, 31 Oct 2026")


# ------------------------------------------------------------------------------------- classification
@needs_curl_cffi
@pytest.mark.parametrize("name, kind", [
    ("Timeout", "transient"), ("ConnectionError", "transient"), ("SSLError", "transient"), ("IncompleteRead", "transient"),
    ("HTTPError", "transient"), ("DNSError", "transient"), ("ProxyError", "transient"), ("RequestException", "transient"),
    ("InvalidURL", "fatal"), ("InvalidSchema", "fatal"), ("MissingSchema", "fatal"), ("InvalidHeader", "fatal"),
    ("TooManyRedirects", "fatal"), ("ImpersonateError", "impersonate"),
])
def test_curl_cffi_errors_are_classified(name, kind):
    assert classify_error(getattr(http_mod.cffi_exc, name)("boom")) == kind


@pytest.mark.parametrize("exc, kind", [
    (requests.exceptions.Timeout("x"), "transient"), (requests.exceptions.ConnectionError("x"), "transient"),
    (requests.exceptions.ChunkedEncodingError("x"), "transient"), (requests.exceptions.SSLError("x"), "transient"),
    (requests.exceptions.InvalidURL("x"), "fatal"), (requests.exceptions.MissingSchema("x"), "fatal"),
    (AssertionError("unexpected request"), None), (ValueError("x"), None), (KeyError("x"), None),
])
def test_requests_errors_are_classified_and_foreign_errors_propagate(exc, kind):
    assert classify_error(exc) == kind


def test_a_foreign_exception_from_the_session_is_not_swallowed(fake_clock):
    client = HttpClient("unit", session=FakeSession([KeyError("bug in a test double")]), clock=fake_clock)
    with pytest.raises(KeyError):
        client.get("https://example.com/x")


# ------------------------------------------------------------------------------------ rate limiting
def test_a_rate_limit_answer_slows_the_provider_down_and_successes_restore_the_pace(fake_clock):
    script = [FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(200, b"a"), FakeResponse(200, b"b"),
              FakeResponse(200, b"c")]
    client = HttpClient("unit", session=FakeSession(script), clock=fake_clock, min_interval=1.0, rng=random.Random(3))
    client.get("https://example.com/1")
    assert client._limiter.min_interval == pytest.approx(2.0 * 0.9)  # doubled by the 429, nudged back by the success
    for _ in range(2):
        client.get("https://example.com/2")
    assert 1.0 <= client._limiter.min_interval < 1.8
    assert stats_for("unit").rate_limited == 1


def test_adaptive_spacing_is_capped_and_never_below_the_base():
    limiter = RateLimiter(1.0, max_interval=5.0)
    for _ in range(6):
        limiter.slow_down()
    assert limiter.min_interval == 5.0
    for _ in range(80):
        limiter.speed_up()
    assert limiter.min_interval == 1.0
    unlimited = RateLimiter(0.0)
    unlimited.slow_down()
    assert unlimited.min_interval == 0.5 and unlimited.max_interval == 10.0  # a provider without spacing gets some


def test_jitter_spreads_requests_without_breaking_the_minimum(fake_clock):
    limiter = RateLimiter(2.0, fake_clock, jitter=0.5, rng=random.Random(7))
    limiter.wait()
    for _ in range(5):
        limiter.wait()
    assert len(fake_clock.sleeps) == 5 and all(2.0 <= s <= 3.0 for s in fake_clock.sleeps) and len(set(fake_clock.sleeps)) > 1


# ----------------------------------------------------------------------------------------- stats
def test_counters_and_the_summary_line(fake_clock):
    session = FakeSession([FakeResponse(503), FakeResponse(200, b"x" * 2_000_000), FakeResponse(304),
                           FakeResponse(500), FakeResponse(500)])
    client = HttpClient("yahoo", session=session, clock=fake_clock, max_attempts=2, rng=random.Random(1))
    client.get("https://example.com/a")
    client.get("https://example.com/a", soft_statuses=(304,))
    with pytest.raises(DataUnavailable):
        client.get("https://example.com/a")
    s = http_stats()["yahoo"]
    assert (s["requests"], s["retries"], s["ok"], s["not_modified"], s["failures"]) == (5, 2, 2, 1, 1)
    assert s["bytes"] == 2_000_000 and s["last_error"].startswith("HTTP 500")
    text = http_summary()
    assert text.startswith("yahoo 5 req / 2 retry / 0 x429") and "1 failed" in text and "1 not-modified" in text and "2.0 MB" in text
    reset_http_stats()
    assert http_summary() == "" and http_stats() == {}


def test_a_client_keeps_counting_after_the_counters_are_reset(fake_clock):
    client = HttpClient("yahoo", session=FakeSession([FakeResponse(200, b"a"), FakeResponse(200, b"b")]), clock=fake_clock)
    client.get("https://example.com/a")
    reset_http_stats()  # what the job runner does at the start of every job
    assert http_stats() == {}
    client.get("https://example.com/b")
    assert http_stats()["yahoo"]["requests"] == 1 and http_summary().startswith("yahoo 1 req")


# --------------------------------------------------------------------------------------------- logs
def test_retries_and_give_ups_are_logged_with_host_and_path_but_never_the_query(fake_clock, caplog):
    client = HttpClient("fred", session=FakeSession([FakeResponse(503), FakeResponse(503)]), clock=fake_clock, max_attempts=2)
    with caplog.at_level(logging.DEBUG, logger="ingestion.http"), pytest.raises(DataUnavailable):
        client.get("https://api.example.com/fred/series?api_key=SUPERSECRET123&x=1")
    text = caplog.text
    assert "api.example.com/fred/series" in text and "SUPERSECRET123" not in text and "api_key" not in text
    assert "retrying in" in text and "giving up after 2 attempts" in text and "HTTP 503" in text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_a_successful_request_is_logged_at_debug_with_timing_and_transport(fake_clock, caplog):
    client = HttpClient("nse_api", session=FakeSession([FakeResponse(200, b"hello")]), clock=fake_clock)
    with caplog.at_level(logging.DEBUG, logger="ingestion.http"):
        client.get("https://www.nseindia.com/api/marketStatus?secret=1")
    line = next(r.getMessage() for r in caplog.records if "marketStatus" in r.getMessage())
    assert "nse_api GET www.nseindia.com/api/marketStatus -> 200" in line and "5 bytes" in line and "injected" in line
    assert "secret=1" not in line


# ------------------------------------------------------------------------------- transport selection
class FakeCffiSession:
    """Stand-in for curl_cffi.requests.Session: records how it was built and what it was asked to send."""

    instances: list[FakeCffiSession] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.headers: dict[str, str] = {}
        self.calls: list[dict] = []
        self.closed = False
        FakeCffiSession.instances.append(self)

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.kwargs.get("impersonate") == "chrome-from-the-future":
            raise http_mod.cffi_exc.ImpersonateError("unsupported profile")
        return FakeResponse(200, b"ok")

    def close(self):
        self.closed = True


@pytest.fixture()
def fake_cffi(monkeypatch):
    FakeCffiSession.instances = []
    monkeypatch.setattr(http_mod, "cffi_requests", SimpleNamespace(Session=FakeCffiSession))
    return FakeCffiSession


@needs_curl_cffi
def test_auto_prefers_curl_cffi_and_impersonates_a_browser(fake_cffi, fake_clock):
    client = HttpClient("nse_api", clock=fake_clock, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    assert client.info()["transport"] == "curl_cffi" and client.info()["impersonate"] == "chrome"
    session = fake_cffi.instances[0]
    assert session.kwargs == {"impersonate": "chrome"}


@needs_curl_cffi
def test_the_browsers_own_headers_are_never_overridden_while_impersonating(fake_cffi, fake_clock):
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip", "Connection": "keep-alive", "Accept-Language": "en-US"}
    client = HttpClient("nse_api", clock=fake_clock, headers=headers)
    assert fake_cffi.instances[0].headers == {} or "User-Agent" not in fake_cffi.instances[0].headers
    client.session.headers.update({})
    client.get("https://www.nseindia.com/api/x", headers={"User-Agent": "curl/8", "Accept": "application/json",
                                                           "Sec-Fetch-Mode": "cors", "Referer": "https://www.nseindia.com/"})
    sent = fake_cffi.instances[0].calls[0]
    assert sent["headers"] == {"Accept": "application/json", "Sec-Fetch-Mode": "cors", "Referer": "https://www.nseindia.com/"}
    assert sent["max_redirects"] == http_mod.MAX_REDIRECTS and sent["stream"] is True
    assert set(fake_cffi.instances[0].headers) == {"Accept-Language"}  # session defaults keep everything except the fingerprint


@needs_curl_cffi
def test_without_impersonation_our_own_user_agent_is_sent(fake_cffi, fake_clock):
    client = HttpClient("fred", clock=fake_clock, impersonate=None, headers={"User-Agent": "monster-ai"})
    client.get("https://api.example.com/x", headers={"User-Agent": "monster-ai"})
    assert fake_cffi.instances[0].kwargs == {} and fake_cffi.instances[0].headers == {"User-Agent": "monster-ai"}
    assert fake_cffi.instances[0].calls[0]["headers"] == {"User-Agent": "monster-ai"}


@needs_curl_cffi
def test_an_unsupported_browser_profile_degrades_to_plain_curl_cffi_without_losing_the_attempt(fake_cffi, fake_clock, caplog):
    client = HttpClient("yahoo", clock=fake_clock, impersonate="chrome-from-the-future", max_attempts=1)
    with caplog.at_level(logging.WARNING, logger="ingestion.http"):
        response = client.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI")
    assert response.ok and client.impersonate is None and client.info()["transport"] == "curl_cffi"
    assert len(fake_cffi.instances) == 2 and fake_cffi.instances[0].closed  # the rejected session was replaced
    assert "not supported by this curl_cffi" in caplog.text


@needs_curl_cffi
def test_when_curl_cffi_keeps_rejecting_requests_takes_over(monkeypatch, fake_clock, caplog):
    class Hostile(FakeCffiSession):
        def request(self, method, url, **kwargs):
            raise http_mod.cffi_exc.ImpersonateError("never works")

    monkeypatch.setattr(http_mod, "cffi_requests", SimpleNamespace(Session=Hostile))
    monkeypatch.setattr(http_mod.requests, "Session", lambda: FakeSession([FakeResponse(200, b"fallback")]))
    client = HttpClient("yahoo", clock=fake_clock, max_attempts=1)
    with caplog.at_level(logging.WARNING, logger="ingestion.http"):
        assert client.get("https://example.com/x").content == b"fallback"
    assert client.info()["transport"] == "requests" and client.impersonate is None
    assert "continuing without impersonation" in caplog.text and "falling back to requests" in caplog.text


def test_requests_can_be_forced(fake_cffi, fake_clock):
    client = HttpClient("unit", clock=fake_clock, transport="requests")
    assert client.info()["transport"] == "requests" and isinstance(client.session, requests.Session) and not fake_cffi.instances
    assert client.session.headers["User-Agent"] == "Mozilla/5.0"


def test_without_curl_cffi_everything_still_works_through_requests(monkeypatch, fake_clock, caplog):
    monkeypatch.setattr(http_mod, "cffi_requests", None)
    with caplog.at_level(logging.WARNING, logger="ingestion.http"):
        first = HttpClient("yahoo", clock=fake_clock)
        second = HttpClient("nse_api", clock=fake_clock)
    assert first.info()["transport"] == second.info()["transport"] == "requests"
    assert caplog.text.count("curl_cffi is not usable") == 1  # warned once per process, not per client


def test_an_unknown_transport_name_means_auto(fake_cffi, fake_clock, caplog):
    with caplog.at_level(logging.WARNING, logger="ingestion.http"):
        client = HttpClient("unit", clock=fake_clock, transport="carrier-pigeon")
    assert client.info()["transport"] in {"curl_cffi", "requests"} and "unknown HTTP transport" in caplog.text


@needs_curl_cffi
def test_a_session_that_cannot_be_created_falls_back_to_requests(monkeypatch, fake_clock):
    def broken(**kwargs):
        raise OSError("native library refused to load")

    monkeypatch.setattr(http_mod, "cffi_requests", SimpleNamespace(Session=broken))
    client = HttpClient("unit", clock=fake_clock)
    assert client.info()["transport"] == "requests" and isinstance(client.session, requests.Session)


def test_reset_session_keeps_the_transport_and_closes_the_old_one(fake_cffi, fake_clock):
    if http_mod.cffi_requests is None:
        pytest.skip("curl_cffi unavailable")
    client = HttpClient("nse_api", clock=fake_clock)
    first = client.session
    client.reset_session()
    assert client.session is not first and first.closed and client.info()["transport"] == "curl_cffi"
    client.close()
    assert client.session.closed


def test_from_settings_applies_the_configured_transport_and_overrides(settings, fake_clock):
    cfg = settings.__class__(**{**settings.__dict__, "http_transport": "requests", "http_impersonate": "", "http_timeout_sec": 12.0,
                                "http_max_attempts": 2, "http_max_response_mb": 3})
    client = HttpClient.from_settings("nse_api", cfg, clock=fake_clock, min_interval=1.5, max_attempts=3)
    assert client.info()["transport"] == "requests" and client.impersonate is None
    assert (client.timeout, client.max_attempts, client.max_bytes, client.jitter) == ((10.0, 12.0), 3, 3 * 1024 * 1024, 0.2)
    assert client.info()["min_interval"] == 1.5


def test_the_environment_selects_the_transport(tmp_path):
    env = {"MONSTER_ENV": "test", "MONSTER_DATA_DIR": str(tmp_path / "d"), "MONSTER_LOG_DIR": str(tmp_path / "l"),
           "HTTP_TRANSPORT": "curl_cffi", "HTTP_IMPERSONATE": "safari"}
    cfg = load_settings(env, env_file=tmp_path / "none.env")
    assert (cfg.http_transport, cfg.http_impersonate) == ("curl_cffi", "safari")
    with pytest.raises(Exception, match="HTTP_TRANSPORT"):
        load_settings({**env, "HTTP_TRANSPORT": "carrier-pigeon"}, env_file=tmp_path / "none.env")


def test_curl_cffi_info_reports_availability():
    info = curl_cffi_info()
    assert info["available"] is (http_mod.cffi_requests is not None)
    if info["available"]:
        assert info["version"] and "curl" in info["libcurl"].lower()
    else:
        assert info["error"]


# ===================================================================== real curl_cffi, loopback server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits: dict[str, int] = {}

    def log_message(self, *args):  # silence
        pass

    def _send(self, status, body=b"", headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        if "Content-Length" not in (headers or {}) and "Transfer-Encoding" not in (headers or {}):
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        Handler.hits[path] = Handler.hits.get(path, 0) + 1
        if path == "/headers":
            self._send(200, json.dumps(dict(self.headers.items())).encode(), {"Content-Type": "application/json"})
        elif path == "/gzip":
            self._send(200, gzip.compress(b'{"hello": "world"}' * 50), {"Content-Encoding": "gzip", "Content-Type": "application/json"})
        elif path == "/flaky":
            if Handler.hits[path] < 3:
                self._send(503, b"later", {"Retry-After": "1"})
            else:
                self._send(200, b"finally")
        elif path == "/cookie":
            self._send(200, b"set", {"Set-Cookie": "session=abc123; Path=/"})
        elif path == "/whoami":
            self._send(200, (self.headers.get("Cookie") or "").encode())
        elif path == "/redirect":
            self._send(302, b"", {"Location": "/final"})
        elif path == "/final":
            self._send(200, b"arrived")
        elif path == "/loop":
            self._send(302, b"", {"Location": "/loop"})
        elif path == "/etag":
            if self.headers.get("If-None-Match") == '"v1"':
                self._send(304, b"", {"ETag": '"v1"'})
            else:
                self._send(200, b"payload", {"ETag": '"v1"', "Last-Modified": "Sat, 31 Oct 2026 10:00:00 GMT"})
        elif path == "/huge":  # chunked, no Content-Length: only the streaming cap can stop it
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for _ in range(40):
                    chunk = b"x" * 65536
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            except OSError:
                pass  # the client hung up once it had seen enough
        elif path == "/slow":
            time.sleep(8.0)  # far beyond the client's timeout: the test must not depend on libcurl's one-second tick
            self._send(200, b"too late")
        elif path == "/missing":
            self._send(404, b"nope")
        else:
            self._send(200, b"ok")


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # a client hanging up mid-stream (size-cap test) is expected
        pass


@pytest.fixture(scope="module")
def server():
    httpd = QuietServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture()
def live(fake_clock):
    Handler.hits.clear()
    made: list[HttpClient] = []

    def build(**kw):
        client = HttpClient("loopback", clock=fake_clock, allow_http_hosts=("127.0.0.1",), transport="curl_cffi",
                            rng=random.Random(1), **kw)
        made.append(client)
        return client

    yield build
    for client in made:
        client.close()


@needs_curl_cffi
def test_real_curl_cffi_sends_the_impersonated_browser_headers_not_ours(server, live):
    client = live(headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
    assert client.info()["transport"] == "curl_cffi"
    seen = client.get(f"{server}/headers", headers={"Accept": "application/json", "User-Agent": "curl/8.0"}).json()
    lowered = {k.lower(): v for k, v in seen.items()}
    assert "chrome" in lowered["user-agent"].lower() and lowered["user-agent"] != "Mozilla/5.0"
    assert lowered["accept"] == "application/json" and "sec-ch-ua" in lowered
    assert "br" in lowered["accept-encoding"] or "gzip" in lowered["accept-encoding"]


@needs_curl_cffi
def test_real_curl_cffi_without_impersonation_sends_the_configured_user_agent(server, live):
    client = live(impersonate=None, headers={"User-Agent": "monster-ai/1.0"})
    assert {k.lower(): v for k, v in client.get(f"{server}/headers").json().items()}["user-agent"] == "monster-ai/1.0"


@needs_curl_cffi
def test_real_curl_cffi_decodes_compressed_bodies_while_streaming(server, live):
    response = live().get(f"{server}/gzip")
    assert response.json() == {"hello": "world"} if len(response.content) < 30 else response.content.startswith(b'{"hello"')
    assert len(response.content) == len(b'{"hello": "world"}' * 50)


@needs_curl_cffi
def test_real_curl_cffi_keeps_cookies_until_the_session_is_reset(server, live):
    client = live()
    client.get(f"{server}/cookie")
    assert client.get(f"{server}/whoami").text() == "session=abc123"
    client.reset_session()
    assert client.get(f"{server}/whoami").text() == ""


@needs_curl_cffi
def test_real_curl_cffi_follows_redirects_but_gives_up_on_loops(server, live):
    client = live(max_attempts=1)
    assert client.get(f"{server}/redirect").text() == "arrived"
    with pytest.raises(DataUnavailable) as err:
        client.get(f"{server}/loop")
    assert err.value.retryable is False and "TooManyRedirects" in str(err.value)


@needs_curl_cffi
def test_real_curl_cffi_retries_a_flaky_provider_honouring_retry_after(server, live, fake_clock):
    client = live(max_attempts=4)
    assert client.get(f"{server}/flaky").text() == "finally"
    assert Handler.hits["/flaky"] == 3 and fake_clock.sleeps == [1.0, 1.0]
    assert stats_for("loopback").retries == 2 and stats_for("loopback").rate_limited == 2


@needs_curl_cffi
def test_real_curl_cffi_supports_conditional_requests(server, live):
    client = live()
    first = client.get(f"{server}/etag")
    assert first.status == 200 and first.headers.get("etag") == '"v1"' and first.headers.get("LAST-MODIFIED")
    again = client.get(f"{server}/etag", headers={"If-None-Match": first.headers["etag"]}, soft_statuses=(304,))
    assert again.status == 304 and again.content == b"" and stats_for("loopback").not_modified == 1


@needs_curl_cffi
def test_real_curl_cffi_enforces_the_size_cap_on_a_chunked_stream(server, live):
    client = live(max_bytes=500_000, max_attempts=1)
    with pytest.raises(DataUnavailable) as err:
        client.get(f"{server}/huge")
    assert "size cap" in str(err.value) and err.value.retryable is False


@needs_curl_cffi
def test_real_curl_cffi_soft_and_hard_statuses(server, live):
    client = live(max_attempts=1)
    assert client.get(f"{server}/missing", soft_statuses=(404,)).status == 404
    with pytest.raises(DataUnavailable) as err:
        client.get(f"{server}/missing")
    assert err.value.retryable is False and "HTTP 404" in str(err.value)


@needs_curl_cffi
def test_real_curl_cffi_connection_errors_are_retried_and_never_leak_the_url(live, fake_clock):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]  # nothing listens here any more
    client = live(max_attempts=3, min_interval=0.0)
    with pytest.raises(DataUnavailable) as err:
        client.get(f"http://127.0.0.1:{dead_port}/secret-path?api_key=SUPERSECRET123")
    text = str(err.value)
    assert err.value.retryable is True and "3 attempts" in text and "SUPERSECRET123" not in text and "secret-path" not in text
    assert len(fake_clock.sleeps) == 2 and stats_for("loopback").failures == 1


@needs_curl_cffi
def test_real_curl_cffi_times_out_slow_providers(server, live):
    client = live(max_attempts=1, timeout=(1.0, 0.5))
    started = time.monotonic()
    with pytest.raises(DataUnavailable) as err:
        client.get(f"{server}/slow")
    assert err.value.retryable is True and "Timeout" in str(err.value) and time.monotonic() - started < 6.0


@needs_curl_cffi
def test_real_curl_cffi_refuses_plain_http_to_other_hosts(live):
    client = HttpClient("loopback", transport="curl_cffi")
    with pytest.raises(DataUnavailable, match="non-HTTPS"):
        client.get("http://example.com/x")
    client.close()
