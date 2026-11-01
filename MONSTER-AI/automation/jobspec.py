"""Job metadata only (no implementations): the scheduler daemon imports this, never the data stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobSpec:
    name: str
    description: str
    heavy: bool = False  # exclusive: at most one heavy job at a time (shared lock)
    critical: bool = False  # time-critical; may preempt a running non-critical job
    timeout_min: int = 30


JOBS: dict[str, JobSpec] = {
    j.name: j for j in (
        JobSpec("collect_global", "Global closes, live context, GIFT Nifty", timeout_min=25),
        JobSpec("collect_news", "News/filings collection and event extraction", timeout_min=20),
        JobSpec("collect_india", "Previous-session completeness (official file first)", timeout_min=20),
        JobSpec("prelim_scan", "08:45 preliminary scan (first repairs an incomplete previous session)", critical=True, timeout_min=6),
        JobSpec("refresh_preopen", "Pre-open book collection", critical=True, timeout_min=10),
        JobSpec("final_scan", "09:08 final scan and Telegram report", critical=True, timeout_min=8),
        JobSpec("ingest_eod", "Official end-of-day data", timeout_min=25),
        JobSpec("evaluate", "Outcome evaluation and evening report", heavy=True, timeout_min=15),
        JobSpec("backup", "Online database backup", timeout_min=20),
        JobSpec("housekeeping", "Retention, disk guard, DB maintenance", timeout_min=30),
        JobSpec("train", "Walk-forward retraining and champion/challenger", heavy=True, timeout_min=150),
        JobSpec("model_sync", "Fetch externally trained signed models", heavy=True, timeout_min=20),
        JobSpec("refresh_calendar", "NSE holiday calendar refresh", timeout_min=10),
        JobSpec("refresh_universe", "Weekly universe/F&O membership refresh", heavy=True, timeout_min=45),
        JobSpec("health_check", "Operational health alerts", timeout_min=5),
        JobSpec("bootstrap", "First-run setup: universe, history, features, first models", heavy=True, timeout_min=480),
    )
}
