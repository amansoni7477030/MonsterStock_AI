-- Versioned relationship graph, plus the event metadata the blueprint asks for (sections 7 and 23.6). Additive only.
--
-- graph_edge_history is append-only: one row per real change of a graph edge (created / updated / removed), written in the
-- same transaction as the change, so the edges that were valid at any past instant can be rebuilt
-- (event_graph.repository.GraphRepository.edges_as_of) instead of replaying history with today's edges. Edges that already
-- existed when this migration ran are recorded once as 'baseline' at their last_updated time (older versions were never kept).

CREATE TABLE IF NOT EXISTS graph_edge_history (
    history_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    src               TEXT NOT NULL,
    dst               TEXT NOT NULL,
    relation          TEXT NOT NULL,
    version           INTEGER NOT NULL,
    change            TEXT NOT NULL CHECK (change IN ('baseline', 'created', 'updated', 'removed')),
    direction         INTEGER NOT NULL DEFAULT 0,
    sensitivity       REAL,
    lag_days_mean     REAL,
    lag_days_sd       REAL,
    confidence        REAL NOT NULL DEFAULT 0.3,
    regime_dependence TEXT,
    rationale         TEXT NOT NULL DEFAULT '',
    source            TEXT NOT NULL DEFAULT 'prior',
    evidence_count    INTEGER NOT NULL DEFAULT 0,
    valid_from        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_graph_edge_history_key ON graph_edge_history (src, dst, relation, valid_from);
CREATE INDEX IF NOT EXISTS idx_graph_edge_history_time ON graph_edge_history (valid_from);

CREATE TRIGGER IF NOT EXISTS graph_edge_history_no_update BEFORE UPDATE ON graph_edge_history
BEGIN
    SELECT RAISE(ABORT, 'graph edge history is append-only');
END;

CREATE TRIGGER IF NOT EXISTS graph_edge_history_no_delete BEFORE DELETE ON graph_edge_history
WHEN COALESCE((SELECT value FROM kv WHERE key = 'maintenance.allow_prune'), '0') <> '1'
BEGIN
    SELECT RAISE(ABORT, 'graph edge history is append-only');
END;

INSERT INTO graph_edge_history (
    src, dst, relation, version, change, direction, sensitivity, lag_days_mean, lag_days_sd, confidence, regime_dependence,
    rationale, source, evidence_count, valid_from
)
SELECT src, dst, relation, version, 'baseline', direction, sensitivity, lag_days_mean, lag_days_sd, confidence, regime_dependence,
       rationale, source, evidence_count, last_updated
FROM graph_edges
WHERE NOT EXISTS (SELECT 1 FROM graph_edge_history)
ORDER BY edge_id;

-- Event metadata (all NULL on events written before this migration = unknown). n_independent_sources already exists.
-- novelty_class: new | follow_up | resurfaced | rumour
ALTER TABLE events ADD COLUMN novelty_class TEXT;
-- rumour: 1 while no non-rumour report has joined the event, 0 once one has
ALTER TABLE events ADD COLUMN rumour INTEGER;
-- source_class of the best source so far: exchange_filing | official | company | media | social | unknown
ALTER TABLE events ADD COLUMN source_class TEXT;
-- n_conflicts: reports that contradict the event (the length of contradictions_json)
ALTER TABLE events ADD COLUMN n_conflicts INTEGER;
