-- Macro vintages and scheduled economic events.

-- First-release aware macro observations (realtime_start = when this value became public).
CREATE TABLE macro_observations (
    series_id        TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    realtime_start   TEXT NOT NULL,
    realtime_end     TEXT NOT NULL,
    value            REAL,
    first_seen_at    TEXT NOT NULL,
    PRIMARY KEY (series_id, observation_date, realtime_start)
) WITHOUT ROWID;

CREATE TABLE econ_calendar (
    event_key     TEXT PRIMARY KEY,
    country       TEXT,
    title         TEXT NOT NULL,
    impact        TEXT,
    scheduled_at  TEXT NOT NULL,
    forecast      TEXT,
    previous      TEXT,
    actual        TEXT,
    source        TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX idx_econ_calendar_time ON econ_calendar (scheduled_at);
