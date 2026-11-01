-- Drop the two tables nothing reads: macro_observations (FRED first-release values) and econ_calendar (the public
-- economic-calendar mirror). They were stored for research only: no feature, report line or health check ever read them, and
-- the collectors that filled them (ingestion/macro.py, NewsCollector.collect_econ_calendar) were removed with this migration.
-- This is the first migration that removes data instead of adding it, and it cannot be undone (restore a backup to get the
-- rows back). Nothing else is touched.

DROP INDEX IF EXISTS idx_econ_calendar_time;
DROP TABLE IF EXISTS econ_calendar;
DROP TABLE IF EXISTS macro_observations;

-- The registry and health rows of the two retired sources go too: without them a failure counter left behind by the old
-- collector would keep raising a "source ... failed N times in a row" warning for a source that no longer exists.
DELETE FROM source_health WHERE source_id IN ('fred', 'econ_calendar');
DELETE FROM sources WHERE source_id IN ('fred', 'econ_calendar');
