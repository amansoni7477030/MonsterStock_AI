"""Resilient HTTP for data providers.

Every external call goes through ``HttpClient`` which provides: per-provider spacing (rate limit), bounded
retries with exponential backoff + jitter, ``Retry-After`` handling, a circuit breaker so a dead provider fails
fast instead of burning the 09:12 cutoff, a hard response-size cap (RAM guard) and secret-safe errors (a URL may
embed an API key, so exceptions only ever carry host + status).
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from app.clock import Clock, get_clock
from app.errors import CircuitOpen, DataUnavailable
from app.logging_setup import redact

log = logging.getLogger(__name__)

DEFAULT_RETRY_STATUSES = (408, 425, 429, 500, 502, 503, 504)
USER_AGENT_SIMPLE = "Mozilla/5.0"
USER_AGENT_BROWSER = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class HttpResponse:
    status: int
    content: bytes
    headers: Mapping[str, str]
    host: str
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self, encoding: str | None = None) -> str:
        return self.content.decode(encoding or "utf-8", errors="replace")

    def json(self) -> Any:
        try:
            return json.loads(self.content)
        except ValueError as exc:
            raise DataUnavailable(self.host, "response is not valid JSON", retryable=True) from exc


class RateLimiter:
    """Enforces a minimum spacing between consecutive requests (thread-safe)."""

    def __init__(self, min_interval: float, clock: Clock | None = None) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        clock = self._clock or get_clock()
        with self._lock:
            now = clock.monotonic()
            if self._last is not None:
                remaining = self.min_interval - (now - self._last)
                if remaining > 0:
                    clock.sleep(remaining)
                    now = clock.monotonic()
            self._last = now


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; half-opens after ``cooldown`` seconds."""

    def __init__(self, threshold: int = 6, cooldown: float = 300.0, clock: Clock | None = None) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self.failures = 0
        self._opened_at: float | None = None

    def _now(self) -> float:
        return (self._clock or get_clock()).monotonic()

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if self._now() - self._opened_at >= self.cooldown:
            return True  # half-open: let one probe through
        return False

    def record_success(self) -> None:
        self.failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self._opened_at = self._now()

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None and not self.allow()


@dataclass
class HttpClient:
    source: str
    headers: Mapping[str, str] = field(default_factory=lambda: {"User-Agent": USER_AGENT_SIMPLE, "Accept": "*/*"})
    min_interval: float = 0.0
    timeout: tuple[float, float] = (10.0, 30.0)
    max_attempts: int = 4
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    max_bytes: int = 40 * 1024 * 1024
    session: Any = None
    clock: Clock | None = None
    breaker: CircuitBreaker | None = None
    retry_statuses: tuple[int, ...] = DEFAULT_RETRY_STATUSES
    allow_http_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    rng: random.Random = field(default_factory=random.Random)
    on_auth_failure: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.min_interval, self.clock)
        if self.breaker is None:
            self.breaker = CircuitBreaker(clock=self.clock)
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update(dict(self.headers))
            self._own_session = True
        else:
            self._own_session = False

    # ------------------------------------------------------------------------------------- plumbing
    def _clock(self) -> Clock:
        return self.clock or get_clock()

    def reset_session(self) -> None:
        """Drop cookies / connection state (used for NSE re-handshakes)."""
        if self._own_session:
            with contextlib.suppress(Exception):
                self.session.close()
            self.session = requests.Session()
            self.session.headers.update(dict(self.headers))

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), 60.0)
        base = min(self.backoff_cap, self.backoff_base * (2 ** (attempt - 1)))
        return base * (0.5 + self.rng.random() * 0.5)

    @staticmethod
    def _retry_after(headers: Mapping[str, str]) -> float | None:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _read_body(self, response: Any) -> bytes:
        declared = response.headers.get("Content-Length") if hasattr(response, "headers") else None
        try:
            if declared is not None and int(declared) > self.max_bytes:
                raise DataUnavailable(self.source, "response exceeds size cap", retryable=False)
        except (TypeError, ValueError):
            pass
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > self.max_bytes:
                raise DataUnavailable(self.source, "response exceeds size cap", retryable=False)
            chunks.append(chunk)
        return b"".join(chunks)

    # ---------------------------------------------------------------------------------------- request
    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        data: Any = None,
        json_body: Any = None,
        soft_statuses: Iterable[int] = (),
        timeout: tuple[float, float] | None = None,
    ) -> HttpResponse:
        """Perform a request. Statuses in ``soft_statuses`` (e.g. 404 for "file not published yet") are returned
        to the caller; other non-2xx statuses raise ``DataUnavailable``."""
        parsed = urlparse(url)
        host = parsed.hostname or "?"
        if parsed.scheme != "https" and host not in self.allow_http_hosts:
            raise DataUnavailable(self.source, "refusing non-HTTPS URL", retryable=False)
        assert self.breaker is not None
        if not self.breaker.allow():
            raise CircuitOpen(self.source, "circuit open after repeated failures", retryable=True)
        soft = set(soft_statuses)
        last_error = "unknown error"
        retryable = True

        for attempt in range(1, self.max_attempts + 1):
            self._limiter.wait()
            started = self._clock().monotonic()
            retry_after: float | None = None
            try:
                response = self.session.request(
                    method, url, params=params, headers=dict(headers) if headers else None, data=data, json=json_body,
                    timeout=timeout or self.timeout, stream=True, allow_redirects=True,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError, requests.exceptions.SSLError) as exc:
                last_error = f"{type(exc).__name__}"
                retryable = True
            except requests.exceptions.RequestException as exc:
                self.breaker.record_failure()
                raise DataUnavailable(self.source, f"request failed: {type(exc).__name__}", retryable=False) from None
            else:
                try:
                    status = int(response.status_code)
                    if status in self.retry_statuses and attempt < self.max_attempts:
                        retry_after = self._retry_after(response.headers)
                        last_error = f"HTTP {status}"
                        retryable = True
                        response.close()
                    else:
                        try:
                            body = self._read_body(response)
                        except (requests.exceptions.RequestException, OSError) as exc:
                            last_error = f"body read failed: {type(exc).__name__}"
                            retryable = True
                            body = None
                        if body is not None:
                            elapsed = (self._clock().monotonic() - started) * 1000.0
                            result = HttpResponse(status, body, dict(response.headers), host, elapsed)
                            if 200 <= status < 300 or status in soft:
                                self.breaker.record_success()
                                return result
                            if status in (401, 403) and self.on_auth_failure and attempt < self.max_attempts:
                                self.on_auth_failure()
                                last_error, retryable = f"HTTP {status}", True
                            else:
                                if status >= 500 or status in self.retry_statuses:
                                    self.breaker.record_failure()
                                raise DataUnavailable(self.source, f"HTTP {status}",
                                                      retryable=status in self.retry_statuses or status >= 500)
                finally:
                    with contextlib.suppress(Exception):
                        response.close()
            if attempt < self.max_attempts:
                delay = self._delay(attempt, retry_after)
                log.debug("%s: %s (attempt %d/%d), retrying in %.1fs", self.source, redact(last_error), attempt,
                          self.max_attempts, delay)
                self._clock().sleep(delay)

        self.breaker.record_failure()
        raise DataUnavailable(self.source, f"{last_error} after {self.max_attempts} attempts", retryable=retryable)

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        return self.request("GET", url, **kwargs)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.get(url, **kwargs).json()

    def get_bytes(self, url: str, **kwargs: Any) -> bytes:
        return self.get(url, **kwargs).content
