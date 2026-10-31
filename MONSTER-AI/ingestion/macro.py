"""Optional FRED client. Market-based yields come from Yahoo; FRED adds first-release macro values (with an API key).

Revised values are never exposed as point-in-time history: ``output_type=4`` returns the INITIAL release of each
observation and ``realtime_start`` records when that value first became public."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.clock import Clock, iso
from app.config import Settings
from app.errors import DataUnavailable
from app.logging_setup import redact
from database.store import Store
from ingestion.http import USER_AGENT_SIMPLE, HttpClient

log = logging.getLogger(__name__)

DEFAULT_SERIES = ("CPIAUCSL", "UNRATE", "PAYEMS", "FEDFUNDS", "GDPC1", "DGS10", "DGS2")


@dataclass
class FredClient:
    settings: Settings
    store: Store
    http: HttpClient | None = None
    clock: Clock | None = None

    def __post_init__(self) -> None:
        self.http = self.http or HttpClient(
            "fred", headers={"User-Agent": USER_AGENT_SIMPLE}, min_interval=0.6, timeout=(8.0, 25.0), max_attempts=3,
            max_bytes=8 * 1024 * 1024, clock=self.clock)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.fred_api_key)

    def collect(self, series: tuple[str, ...] = DEFAULT_SERIES, *, observation_start: str = "2018-01-01") -> int:
        """Store first-release observations. Returns rows written; a no-op without an API key."""
        if not self.enabled:
            return 0
        assert self.http is not None
        written = 0
        for series_id in series:
            try:
                payload = self.http.get_json("https://api.stlouisfed.org/fred/series/observations", params={
                    "series_id": series_id, "api_key": self.settings.fred_api_key or "", "file_type": "json",
                    "output_type": "4", "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                    "observation_start": observation_start})
            except DataUnavailable as exc:
                self.store.record_source_result("fred", False, error=redact(str(exc)))
                log.warning("FRED %s failed: %s", series_id, redact(str(exc)))
                continue
            now = iso(self.clock.now()) if self.clock else self.store.now()
            rows = []
            for obs in payload.get("observations", []) if isinstance(payload, dict) else []:
                try:
                    value = float(obs["value"])
                except (KeyError, TypeError, ValueError):
                    continue  # FRED uses "." for missing values
                rows.append((series_id, obs["date"], obs["realtime_start"], obs["realtime_end"], value, now))
            if rows:
                with self.store.tx():
                    self.store.conn.executemany(
                        "INSERT OR IGNORE INTO macro_observations (series_id, observation_date, realtime_start, realtime_end, "
                        "value, first_seen_at) VALUES (?, ?, ?, ?, ?, ?)", rows)
                written += len(rows)
            self.store.record_source_result("fred", True)
        return written
