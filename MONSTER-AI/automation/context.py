"""Job execution context: settings, store, clock, calendar and lazily created provider clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.clock import Clock, get_clock, ist_date
from app.config import Settings
from automation.calendar import ExchangeCalendar, seed_calendar
from database.store import Store


@dataclass
class JobResult:
    status: str  # success | skipped | failed
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobContext:
    settings: Settings
    store: Store
    clock: Clock
    calendar: ExchangeCalendar
    run_key: str
    attempt: int = 1
    max_attempts: int = 1
    last_attempt: bool = True
    dry_run: bool = False
    force: bool = False
    deadline: float | None = None  # monotonic timestamp
    options: dict[str, str] = field(default_factory=dict)  # free-form per-run switches (run-job --opt KEY=VALUE)
    _cache: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, settings: Settings, store: Store, *, run_key: str, clock: Clock | None = None, **kw: Any) -> JobContext:
        from ingestion.registry import register_all

        register_all(store)  # idempotent: source health and data-quality checks are driven by the registered catalogue
        clk = clock or get_clock()
        calendar = ExchangeCalendar(store)
        return cls(settings, store, clk, calendar, run_key, **kw)

    # ------------------------------------------------------------------------------------------------ dates
    @property
    def run_date(self) -> date:
        """The trading date this run is about: the leading ``YYYY-MM-DD`` of the run key (weekly keys fall back to today)."""
        try:
            return date.fromisoformat(self.run_key[:10])
        except ValueError:
            return ist_date(self.clock.now())

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - self.clock.monotonic()

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def budget(self, default_seconds: float) -> float:
        """Monotonic deadline for a bounded step: the job's own deadline or ``default_seconds`` from now, whichever is sooner."""
        own = self.clock.monotonic() + default_seconds
        return own if self.deadline is None else min(own, self.deadline)

    # ---------------------------------------------------------------------------------------------- clients
    def _lazy(self, key: str, factory: Any) -> Any:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def yahoo(self) -> Any:
        from ingestion.yahoo import YahooClient

        return self._lazy("yahoo", lambda: YahooClient(self.settings, clock=self.clock))

    def nse_archive(self) -> Any:
        from ingestion.nse import NseArchive

        return self._lazy("nse_archive", lambda: NseArchive(self.settings, clock=self.clock))

    def nse_api(self) -> Any:
        from ingestion.nse import NseApi

        return self._lazy("nse_api", lambda: NseApi(self.settings, clock=self.clock))

    def kotak(self) -> Any:
        """The broker's market-data client. Only for ``settings.kotak_configured`` (it refuses to start without a key)."""
        from ingestion.kotak import KotakApi

        return self._lazy("kotak", lambda: KotakApi(self.settings, clock=self.clock))

    def news(self) -> Any:
        from ingestion.news_sources import NewsCollector

        return self._lazy("news", lambda: NewsCollector(self.store, self.settings, clock=self.clock))

    def intel(self) -> Any:
        from automation.intel import Intel

        return self._lazy("intel", lambda: Intel(self.store, self.settings, self.clock))

    def registry(self) -> Any:
        from models.registry import ModelRegistry

        return self._lazy("registry", lambda: ModelRegistry(self.store, self.settings))

    def notifier(self) -> Any:
        from notifications.service import NotificationService

        return self._lazy("notifier", lambda: NotificationService(self.settings, self.store, clock=self.clock))

    def ensure_calendar(self) -> None:
        seed_calendar(self.store, self.settings, self.clock.now())
        self.calendar.reload()
