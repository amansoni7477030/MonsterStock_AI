"""Exception hierarchy. Every expected failure mode has a dedicated type so callers can apply the
documented fallback (retry, downgrade, abstain, alert) instead of catching bare ``Exception``."""

from __future__ import annotations


class MonsterError(Exception):
    """Base class for all application errors."""


class ConfigError(MonsterError):
    """Invalid or missing configuration."""


class DataUnavailable(MonsterError):
    """A data source could not deliver usable data after bounded retries."""

    def __init__(self, source: str, message: str = "", *, retryable: bool = True) -> None:
        super().__init__(f"[{source}] {message}" if message else f"[{source}] unavailable")
        self.source = source
        self.retryable = retryable


class CircuitOpen(DataUnavailable):
    """A source is temporarily short-circuited after repeated consecutive failures."""


class StaleDataError(MonsterError):
    """Data exists but is older than the allowed freshness window."""


class ValidationFailed(MonsterError):
    """Data failed a schema / sanity validation."""


class LeakageError(MonsterError):
    """A look-ahead (point-in-time) violation was detected."""


class ModelArtifactError(MonsterError):
    """Model artifact is missing, corrupt, unsigned or incompatible with the feature schema."""


class CutoffMissed(MonsterError):
    """A time-critical job could not finish before its cutoff."""


class JobTimeout(MonsterError):
    """A job exceeded its allotted wall-clock time."""


class ResourceExhausted(MonsterError):
    """RAM or disk is below the configured safety threshold."""


class DatabaseCorrupt(MonsterError):
    """SQLite integrity check failed."""
