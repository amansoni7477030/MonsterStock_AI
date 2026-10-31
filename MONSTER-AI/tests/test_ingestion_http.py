from __future__ import annotations

import random

import pytest
import requests

from app.errors import CircuitOpen, DataUnavailable
from ingestion.http import CircuitBreaker, HttpClient, RateLimiter
from tests.helpers import FakeResponse, FakeSession, timeout_error


def make(session, clock, **kw):
    kw.setdefault("rng", random.Random(1))
    return HttpClient("unit", session=session, clock=clock, **kw)


def test_success_returns_json_and_text(fake_clock):
    client = make(FakeSession([FakeResponse(200, {"a": 1})]), fake_clock)
    response = client.get("https://example.com/x")
    assert response.ok and response.json() == {"a": 1} and response.host == "example.com"
    assert client.get_json.__name__ == "get_json"


def test_retries_5xx_with_exponential_backoff_then_succeeds(fake_clock):
    session = FakeSession([FakeResponse(503), FakeResponse(502), FakeResponse(200, b"ok")])
    client = make(session, fake_clock, max_attempts=4)
    assert client.get_bytes("https://example.com/x") == b"ok"
    assert len(session.calls) == 3 and len(fake_clock.sleeps) == 2
    assert fake_clock.sleeps[1] > fake_clock.sleeps[0] * 0.9  # backoff grows (jittered)


def test_retry_after_header_is_honoured_and_capped(fake_clock):
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(429, headers={"Retry-After": "999"}),
                           FakeResponse(200, b"ok")])
    client = make(session, fake_clock, max_attempts=3)
    client.get("https://example.com/x")
    assert fake_clock.sleeps == [7.0, 60.0]


def test_timeouts_exhaust_attempts_and_never_leak_the_url(fake_clock):
    session = FakeSession([timeout_error(), timeout_error(), timeout_error()])
    client = make(session, fake_clock, max_attempts=3)
    with pytest.raises(DataUnavailable) as err:
        client.get("https://example.com/x?api_key=SUPERSECRET123")
    text = str(err.value)
    assert "SUPERSECRET123" not in text and "example.com/x" not in text and "3 attempts" in text
    assert err.value.retryable is True


def test_client_error_is_not_retried_unless_soft(fake_clock):
    session = FakeSession([FakeResponse(404), FakeResponse(404)])
    client = make(session, fake_clock)
    with pytest.raises(DataUnavailable) as err:
        client.get("https://example.com/missing")
    assert err.value.retryable is False and len(session.calls) == 1
    soft = client.get("https://example.com/missing", soft_statuses=(404,))
    assert soft.status == 404


def test_non_https_is_refused_except_loopback(fake_clock):
    client = make(FakeSession([FakeResponse(200, b"x")]), fake_clock)
    with pytest.raises(DataUnavailable):
        client.get("http://example.com/x")
    assert client.get("http://localhost:8080/x").ok


def test_response_size_cap_header_and_stream(fake_clock):
    client = make(FakeSession([FakeResponse(200, b"x" * 100, headers={"Content-Length": "100"})]), fake_clock, max_bytes=50)
    with pytest.raises(DataUnavailable):
        client.get("https://example.com/big")
    client = make(FakeSession([FakeResponse(200, b"x" * 200_000)]), fake_clock, max_bytes=100_000)
    with pytest.raises(DataUnavailable) as err:
        client.get("https://example.com/big")
    assert "size cap" in str(err.value)


def test_invalid_json_is_reported_as_retryable_data_error(fake_clock):
    client = make(FakeSession([FakeResponse(200, b"<html>not json</html>")]), fake_clock)
    with pytest.raises(DataUnavailable):
        client.get_json("https://example.com/x")


def test_circuit_breaker_opens_then_half_opens(fake_clock):
    breaker = CircuitBreaker(threshold=2, cooldown=100, clock=fake_clock)
    session = FakeSession(lambda url, kw: FakeResponse(503))
    client = make(session, fake_clock, max_attempts=1, breaker=breaker)
    for _ in range(2):
        with pytest.raises(DataUnavailable):
            client.get("https://example.com/x")
    calls = len(session.calls)
    with pytest.raises(CircuitOpen):
        client.get("https://example.com/x")
    assert len(session.calls) == calls  # fail fast, no network call
    fake_clock.advance(101)
    session.handler = lambda url, kw: FakeResponse(200, b"ok")
    assert client.get("https://example.com/x").ok and not breaker.is_open


def test_auth_failure_triggers_rewarm_then_retry(fake_clock):
    rewarmed = []
    session = FakeSession([FakeResponse(403), FakeResponse(200, b"ok")])
    client = make(session, fake_clock, max_attempts=3, on_auth_failure=lambda: rewarmed.append(1))
    assert client.get("https://example.com/x").ok and rewarmed == [1]


def test_rate_limiter_spacing(fake_clock):
    limiter = RateLimiter(2.0, fake_clock)
    limiter.wait()
    fake_clock.advance(0.5)
    limiter.wait()
    assert fake_clock.sleeps == [1.5]
    RateLimiter(0, fake_clock).wait()  # disabled


def test_request_exception_is_non_retryable_and_redacted(fake_clock):
    session = FakeSession([requests.exceptions.InvalidURL("bad https://x?token=ABCDEFGH12345")])
    client = make(session, fake_clock)
    with pytest.raises(DataUnavailable) as err:
        client.get("https://example.com/x")
    assert "ABCDEFGH12345" not in str(err.value) and err.value.retryable is False


def test_reset_session_replaces_owned_session():
    client = HttpClient("unit")
    first = client.session
    client.reset_session()
    assert client.session is not first
    injected = FakeSession()
    HttpClient("unit", session=injected).reset_session()  # injected sessions are never replaced
