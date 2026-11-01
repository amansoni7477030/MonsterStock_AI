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


class ModelArtifactError(MonsterError):
    """Model artifact is missing, corrupt, unsigned or incompatible with the feature schema."""


class ResourceExhausted(MonsterError):
    """RAM or disk is below the configured safety threshold."""


class DatabaseCorrupt(MonsterError):
    """SQLite integrity check failed."""
