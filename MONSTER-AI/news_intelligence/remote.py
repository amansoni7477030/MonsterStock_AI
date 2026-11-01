"""Plumbing shared by the optional AI services of the news pipeline (the Groq/OpenAI-compatible LLM extractor and the Jev judge).

Both services are *enrichment*: they may add information to what the deterministic rules found and they may never block or
fabricate. This base gives them the same safety rails:

* a result cache in the ``kv`` table (content-hash keys, 14 days, pruned by the nightly housekeeping),
* a per-run call budget and a wall-clock budget, so a slow or rate-limited provider cannot delay the morning schedule,
* a circuit breaker that opens after repeated failures (or at once on a bad key),
* a rate-limit *cooldown* that survives between runs: when a provider says "come back in an hour" (an exhausted daily quota)
  the next news run does not hammer it again,
* counters in ``kv`` (calls, failures, tokens) that ``monster-ai diag`` shows.

Every failure returns ``None`` to the caller, which falls back to the rules.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any, TypeVar

import requests

from app import __version__, jsonutil
from app.clock import Clock, get_clock, iso, parse_ts
from app.logging_setup import redact

log = logging.getLogger(__name__)
T = TypeVar("T")

RETRY_AFTER_CAP_SEC = 30.0  # a short rate-limit wait is slept through; a longer one is a quota and opens a cooldown
COOLDOWN_MAX_SEC = 6 * 3600.0  # never trust a "come back in 20 hours" for more than this: re-probe with one request instead
MIN_ATTEMPT_SEC = 3.0  # a request that would get less of the run's time budget than this is not started
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def clamp(value: Any, low: float, high: float, default: float | None = None) -> float | None:
    """``value`` as a float limited to [low, high]; ``default`` for anything that is not a finite number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return max(low, min(high, number))


def retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """Seconds a rate-limited server asks for (``Retry-After``: plain seconds, or an ``1m30s`` / ``250ms`` duration); ``None``
    when the header is missing or unreadable."""
    headers = headers or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    seconds = clamp(raw, 0.0, 7 * 86400.0)
    if seconds is not None:
        return seconds
    parts = _DURATION.findall(str(raw).strip().lower())
    if not parts:
        return None
    return min(7 * 86400.0, sum(float(number) * _UNIT_SECONDS[unit] for number, unit in parts))


def error_reason(payload: Any, text: str = "", limit: int = 140) -> str:
    """The provider's own short explanation of an error response, so a failure says *why* ("Unknown model: x", "Invalid API Key")
    and not just "http 400". Reads ``{"error": {"message": ...}}`` (Groq, OpenAI), ``{"detail": {"message": ...}}`` and
    ``{"detail": "..."}`` (Jev); a body that is not JSON contributes only when it is plain text (an HTML error page does not).
    Single-line, redacted and capped; empty when the provider said nothing useful."""
    message: Any = None
    if isinstance(payload, dict):
        for key in ("error", "detail"):
            value = payload.get(key)
            if isinstance(value, dict):
                message = value.get("message") or value.get("error_type")
            elif isinstance(value, str):
                message = value
            if message:
                break
    elif text and not text.lstrip().startswith("<"):
        message = text
    return redact(re.sub(r"\s+", " ", str(message or "")).strip())[:limit]


def describe_status(status: int, reason: str = "") -> str:
    """``http 400: Unknown model: x`` (just ``http 400`` when the provider gave no reason)."""
    return f"http {status}: {reason}" if reason else f"http {status}"


def prune_prefix(store: Any, now: Any, prefix: str, ttl_days: int) -> int:
    """Delete the cache entries ``prefix*`` last written more than ``ttl_days`` ago; returns how many were removed."""
    upper = prefix[:-1] + chr(ord(prefix[-1]) + 1)  # the key range [prefix, prefix + 1): every key that starts with it
    cursor = store.conn.execute(
        "DELETE FROM kv WHERE key >= ? AND key < ? AND updated_at < ?", (prefix, upper, iso(now - timedelta(days=ttl_days)))
    )
    return int(cursor.rowcount)


class OutOfTime(Exception):
    """The wall-clock budget of this run is spent."""


class Quota(Exception):
    """The provider's quota is exhausted for ``seconds``: no point in asking again during this run (or the next ones)."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"quota exhausted for {seconds:.0f} s")
        self.seconds = seconds


class Backoff(Exception):
    """The service cannot do what was asked on this account or model (a tool it does not offer, a request larger than the plan's
    per-minute token limit): asking again will fail the same way, so the caller stays away for ``seconds``."""

    def __init__(self, seconds: float, reason: str) -> None:
        super().__init__(reason)
        self.seconds = seconds


class Fatal(Exception):
    """Configuration-level failure (bad key / endpoint): no point retrying during this run."""


class RemoteJudge:
    """Cache, budgets, circuit breaker and cooldown of one optional AI service. Subclasses set the class attributes."""

    CACHE_PREFIX = "ai:"  # kv keys of the cached results: ``<prefix><hash>``
    COUNTER = "ai"  # kv counters: ``<counter>.calls``, ``.failures``, ``.tokens_in``, ``.tokens_out``, ``.cooldown_until``
    SOURCE_ID = "ai"  # the service's row in the source register: every outcome is also written to ``source_health``
    CACHE_TTL_DAYS = 14

    def __init__(
        self, *, clock: Clock | None = None, store: Any = None, resolver: Any = None, session: Any = None,
        max_consecutive_failures: int = 5, max_seconds: float = 0.0,
    ) -> None:
        self.clock = clock or get_clock()
        self.store = store
        self.resolver = resolver
        self.session = session  # created on first use (a ``requests.Session``) unless one is injected
        self.max_consecutive_failures = max_consecutive_failures
        self.max_seconds = float(max_seconds)
        self.run_calls = self.run_failures = self.run_cache_hits = self.run_budget_skipped = 0
        self.run_tokens_in = self.run_tokens_out = 0
        self.consecutive_failures = 0
        self.tripped = False
        self.last_error: str | None = None  # why the latest request failed (``monster-ai ai-test`` prints it)
        self._run_started: float | None = None  # set by start_run(): without a run there is no wall-clock budget

    # ------------------------------------------------------------------------------------- run state
    def bind(self, *, store: Any = None, resolver: Any = None) -> None:
        """Attach the store (cache + counters) and resolver when they were not given at construction (never replaced)."""
        if store is not None and self.store is None:
            self.store = store
        if resolver is not None and self.resolver is None:
            self.resolver = resolver

    def start_run(self) -> None:
        self.run_calls = self.run_failures = self.run_cache_hits = self.run_budget_skipped = 0
        self.run_tokens_in = self.run_tokens_out = 0
        self.consecutive_failures = 0
        self.tripped = False
        self._run_started = self.clock.monotonic()

    def mark(self) -> tuple[int, int, int, int]:
        """The run counters now (calls, failures, cache hits, budget skips): :meth:`since` turns a mark into what happened
        after it, so one batch of a multi-batch run can be reported on its own."""
        return self.run_calls, self.run_failures, self.run_cache_hits, self.run_budget_skipped

    def since(self, mark: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        calls, failures, hits, skipped = self.mark()
        return calls - mark[0], failures - mark[1], hits - mark[2], skipped - mark[3]

    def elapsed(self) -> float:
        """Seconds since ``start_run()`` (0 when no run has started)."""
        return 0.0 if self._run_started is None else self.clock.monotonic() - self._run_started

    @property
    def out_of_time(self) -> bool:
        return self._run_started is not None and self.max_seconds > 0 and self.elapsed() >= self.max_seconds

    def sleep_within_budget(self, seconds: float) -> None:
        """Sleep, unless the wall-clock budget of the run would be spent first (then raise :class:`OutOfTime`)."""
        if self._run_started is not None and self.max_seconds > 0 and self.elapsed() + seconds > self.max_seconds:
            raise OutOfTime()
        self.clock.sleep(seconds)

    def budgeted_timeout(self, limit: float, minimum: float = MIN_ATTEMPT_SEC) -> float:
        """The timeout for the next request: ``limit``, cut to what is left of the run's wall-clock budget so that no single
        request can run past it. Raises :class:`OutOfTime` when less than ``minimum`` seconds are left (a request that cannot
        finish is not started). Without a run there is no budget and ``limit`` is returned unchanged."""
        if self._run_started is None or self.max_seconds <= 0:
            return limit
        remaining = self.max_seconds - self.elapsed()
        if remaining < minimum:
            raise OutOfTime()
        return min(limit, remaining)

    # --------------------------------------------------------------------------------------- counters
    def _bump(self, key: str, amount: int = 1) -> None:
        if self.store is not None and amount:
            self.store.kv_set(key, str(int(self.store.kv_get(key, "0") or 0) + int(amount)))

    def _count_tokens(self, tokens_in: Any, tokens_out: Any) -> None:
        low_in, low_out = int(clamp(tokens_in, 0, 10**9, 0) or 0), int(clamp(tokens_out, 0, 10**9, 0) or 0)
        self.run_tokens_in += low_in
        self.run_tokens_out += low_out
        self._bump(f"{self.COUNTER}.tokens_in", low_in)
        self._bump(f"{self.COUNTER}.tokens_out", low_out)

    def _failed(self, reason: str, *, trip: bool = False) -> None:
        self.run_failures += 1
        self.consecutive_failures += 1
        self.last_error = reason
        self._bump(f"{self.COUNTER}.failures")
        self._record_health(False, error=reason)
        if trip or self.consecutive_failures >= self.max_consecutive_failures:
            self.tripped = True
        log.warning("%s failed: %s", self.COUNTER, reason)

    def _record_health(self, ok: bool, latency_ms: float | None = None, error: str | None = None) -> None:
        """Report one outcome to ``source_health``, where the health check warns after repeated failures. Never raises."""
        if self.store is None:
            return
        try:
            self.store.record_source_result(self.SOURCE_ID, ok, latency_ms=latency_ms, error=redact(error) if error else None)
        except Exception:  # noqa: BLE001  # bookkeeping must never lose an item (e.g. the source is not registered yet)
            log.debug("could not record the health of %s", self.SOURCE_ID, exc_info=True)

    # ------------------------------------------------------------------------------------------- requests
    def send(
        self, method: str, url: str, key: str | None, timeout: float, body: dict[str, Any] | None = None
    ) -> tuple[int, Any, str, dict[str, str]]:
        """One HTTP call authenticated with the service's bearer ``key``: ``(status, decoded JSON or None, the first 500
        characters of the text, response headers)``. Transport errors propagate to the caller's retry loop."""
        if self.session is None:
            self.session = requests.Session()
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": f"monster-ai/{__version__}"}
        if key:
            headers["Authorization"] = "Bearer " + key
        if method == "POST":
            response = self.session.post(url, headers=headers, json=body, timeout=timeout)
        else:
            response = self.session.get(url, headers=headers, timeout=timeout)
        try:
            payload = response.json()
        except (ValueError, AttributeError):
            payload = None
        text = str(getattr(response, "text", "") or "")[:500]
        return int(response.status_code), payload, text, dict(getattr(response, "headers", {}) or {})

    def _admit(self, max_calls: int) -> bool:
        """True when a request may be made now. Otherwise ``False`` (circuit open, provider in cooldown, or this run's call /
        time budget spent) and the caller falls back to the rules. An admitted request counts as a call."""
        if self.tripped:
            return False
        if self.cooldown_active():
            self.tripped = True  # the provider asked us to stay away: no request at all until its cooldown ends
            return False
        if self.run_calls >= max_calls or self.out_of_time:
            self.run_budget_skipped += 1
            return False
        self.run_calls += 1
        self._bump(f"{self.COUNTER}.calls")
        return True

    def _attempt(self, call: Callable[[], T]) -> T | None:
        """Run one admitted request. Any failure is counted, logged and answered with ``None`` (never raised): a rejected key
        or endpoint opens the circuit, an exhausted quota starts a cooldown, and a spent time budget is no failure at all."""
        started = self.clock.monotonic()
        try:
            result = call()
        except Fatal as exc:
            self._failed(f"fatal: {exc}", trip=True)
            return None
        except Quota as exc:
            self._failed(f"rate limit: {exc}", trip=True)
            self.start_cooldown(exc.seconds)
            return None
        except Backoff as exc:
            self._failed(f"unavailable: {exc}", trip=True)
            self.start_cooldown(exc.seconds)
            return None
        except OutOfTime:
            self.run_calls -= 1  # not a failure and not a use: the run's time ran out before the answer
            self.run_budget_skipped += 1
            self.tripped = True
            return None
        except Exception as exc:  # noqa: BLE001  # an AI service must never lose an item: any failure falls back to the rules
            self._failed(redact(f"{type(exc).__name__}: {exc}"))
            return None
        self.consecutive_failures = 0
        self._record_health(True, (self.clock.monotonic() - started) * 1000.0)
        return result

    # ----------------------------------------------------------------------------------------- cooldown
    def cooldown_active(self) -> bool:
        """True while the provider asked us (in an earlier run) to stay away; the first call after it ends re-probes."""
        if self.store is None:
            return False
        until = parse_ts(self.store.kv_get(f"{self.COUNTER}.cooldown_until"))
        return until is not None and self.clock.now() < until

    def start_cooldown(self, seconds: float) -> None:
        if self.store is not None:
            wait = max(1.0, min(float(seconds), COOLDOWN_MAX_SEC))
            self.store.kv_set(f"{self.COUNTER}.cooldown_until", iso(self.clock.now() + timedelta(seconds=wait)))

    # --------------------------------------------------------------------------------------------- cache
    def _cache_read(self, key: str) -> Any | None:
        """The ``res`` payload cached under ``key`` while it is younger than the TTL, else ``None``."""
        if self.store is None:
            return None
        entry = jsonutil.loads(self.store.kv_get(key))
        if not isinstance(entry, dict):
            return None
        stamp = parse_ts(entry.get("ts"))
        if stamp is None or self.clock.now() - stamp > timedelta(days=self.CACHE_TTL_DAYS):
            return None
        return entry.get("res")

    def _cache_write(self, key: str, payload: Any) -> None:
        if self.store is not None:
            self.store.kv_set(key, jsonutil.dumps({"ts": iso(self.clock.now()), "res": payload}))

    def prune_cache(self) -> int:
        """Delete cache entries older than the TTL; returns the number removed."""
        if self.store is None:
            return 0
        return prune_prefix(self.store, self.clock.now(), self.CACHE_PREFIX, self.CACHE_TTL_DAYS)
