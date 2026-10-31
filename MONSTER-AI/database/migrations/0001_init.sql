-- MONSTER AI schema v1. Applied exactly once by database/migrate.py (statement by statement, in a transaction).
-- Conventions: timestamps are UTC ISO-8601 'YYYY-MM-DDTHH:MM:SSZ'; trade dates are exchange-local 'YYYY-MM-DD'.

CREATE TABLE kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- ------------------------------------------------------------------------------------------- sources
CREATE TABLE sources (
    source_id         TEXT PRIMARY KEY,
    provider          TEXT NOT NULL,
    endpoint          TEXT NOT NULL,
    coverage          TEXT NOT NULL,
    timestamp_meaning TEXT NOT NULL,
    update_frequency  TEXT NOT NULL,
    history_depth     TEXT NOT NULL,
    rate_limits       TEXT NOT NULL,
    license_notes     TEXT NOT NULL,
    failure_behavior  TEXT NOT NULL,
    backup_source     TEXT,
    criticality       TEXT NOT NULL CHECK (criticality IN ('critical', 'important', 'optional')),
    enabled           INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE source_health (
    source_id            TEXT PRIMARY KEY,
    last_attempt_at      TEXT,
    last_success_at      TEXT,
    last_error           TEXT,
    last_error_at        TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    total_success        INTEGER NOT NULL DEFAULT 0,
    total_failure        INTEGER NOT NULL DEFAULT 0,
    last_latency_ms      REAL,
    last_data_ts         TEXT
);

-- --------------------------------------------------------------------------------- instrument master
CREATE TABLE instruments (
    instrument_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('equity', 'index', 'global', 'fx', 'commodity', 'rate', 'volatility')),
    exchange      TEXT NOT NULL DEFAULT 'NSE',
    isin          TEXT,
    name          TEXT,
    sector        TEXT,
    industry      TEXT,
    yahoo_symbol  TEXT,
    is_fno        INTEGER NOT NULL DEFAULT 0,
    lot_size      INTEGER,
    status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'delisted', 'quarantined')),
    status_reason TEXT,
    listed_from   TEXT,
    delisted_on   TEXT,
    in_universe   INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT NOT NULL,
    UNIQUE (symbol, kind)
);
CREATE INDEX idx_instruments_universe ON instruments (in_universe, kind, status);

-- Point-in-time attribute versions (sector, F&O eligibility, lot size, universe membership).
CREATE TABLE instrument_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id  INTEGER NOT NULL REFERENCES instruments (instrument_id) ON DELETE CASCADE,
    effective_from TEXT NOT NULL,
    effective_to   TEXT,
    attrs_json     TEXT NOT NULL,
    UNIQUE (instrument_id, effective_from)
);

CREATE TABLE corporate_actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id INTEGER NOT NULL REFERENCES instruments (instrument_id) ON DELETE CASCADE,
    action_type   TEXT NOT NULL,
    ex_date       TEXT NOT NULL,
    ratio         REAL,
    amount        REAL,
    details       TEXT,
    source        TEXT,
    first_seen_at TEXT NOT NULL,
    UNIQUE (instrument_id, action_type, ex_date)
);

-- ------------------------------------------------------------------------------------- market data
CREATE TABLE bars_daily (
    instrument_id INTEGER NOT NULL REFERENCES instruments (instrument_id) ON DELETE CASCADE,
    trade_date    TEXT NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    adj_close     REAL,
    volume        REAL,
    prev_close    REAL,
    turnover      REAL,
    delivery_pct  REAL,
    vwap          REAL,
    trades        REAL,
    source        TEXT NOT NULL,
    source_rank   INTEGER NOT NULL DEFAULT 1,
    quality       TEXT NOT NULL DEFAULT 'ok',
    received_at   TEXT NOT NULL,
    PRIMARY KEY (instrument_id, trade_date)
) WITHOUT ROWID;

CREATE TABLE fno_daily (
    symbol             TEXT NOT NULL,
    trade_date         TEXT NOT NULL,
    underlying         REAL,
    lot_size           INTEGER,
    fut_close          REAL,
    fut_settle         REAL,
    fut_oi             REAL,
    fut_oi_chg         REAL,
    fut_volume         REAL,
    fut_oi_all         REAL,
    fut_oi_chg_all     REAL,
    next_fut_oi        REAL,
    basis_pct          REAL,
    rollover_pct       REAL,
    call_oi            REAL,
    put_oi             REAL,
    call_oi_chg        REAL,
    put_oi_chg         REAL,
    call_vol           REAL,
    put_vol            REAL,
    pcr_oi             REAL,
    pcr_vol            REAL,
    atm_strike         REAL,
    atm_straddle       REAL,
    expected_move_pct  REAL,
    iv_atm             REAL,
    iv_skew            REAL,
    max_call_oi_strike REAL,
    max_put_oi_strike  REAL,
    dte                INTEGER,
    received_at        TEXT NOT NULL,
    PRIMARY KEY (symbol, trade_date)
) WITHOUT ROWID;

-- Append-only live observations (each fetch keeps source and received timestamps).
CREATE TABLE preopen_quotes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date  TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    source_ts   TEXT,
    received_at TEXT NOT NULL,
    iep         REAL,
    final_price REAL,
    prev_close  REAL,
    final_qty   REAL,
    buy_qty     REAL,
    sell_qty    REAL,
    pchange     REAL
);
CREATE INDEX idx_preopen_date_symbol ON preopen_quotes (trade_date, symbol, received_at);

CREATE TABLE live_quotes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    source_ts   TEXT,
    received_at TEXT NOT NULL,
    price       REAL,
    prev_close  REAL,
    change_pct  REAL,
    extra_json  TEXT
);
CREATE INDEX idx_live_quotes_key ON live_quotes (key, received_at);

CREATE TABLE calendar_days (
    trade_date     TEXT PRIMARY KEY,
    is_trading_day INTEGER NOT NULL,
    session_kind   TEXT NOT NULL CHECK (session_kind IN ('regular', 'special', 'holiday', 'weekend')),
    open_time      TEXT,
    close_time     TEXT,
    description    TEXT,
    source         TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE backfill_progress (
    key        TEXT PRIMARY KEY,
    last_date  TEXT,
    status     TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL
);

-- ----------------------------------------------------------------------------------- news / events
CREATE TABLE news_raw (
    article_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT NOT NULL,
    feed          TEXT,
    url           TEXT,
    url_hash      TEXT NOT NULL UNIQUE,
    content_hash  TEXT NOT NULL,
    title         TEXT NOT NULL,
    summary       TEXT,
    text          TEXT,
    language      TEXT,
    published_at  TEXT,
    first_seen_at TEXT NOT NULL,
    region        TEXT,
    category      TEXT,
    processed     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_news_seen ON news_raw (first_seen_at);
CREATE INDEX idx_news_content ON news_raw (content_hash);
CREATE INDEX idx_news_unprocessed ON news_raw (processed, first_seen_at);

CREATE TABLE filings (
    filing_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange       TEXT NOT NULL,
    symbol         TEXT,
    isin           TEXT,
    company        TEXT,
    category       TEXT,
    subject        TEXT,
    text           TEXT,
    attachment_url TEXT,
    disseminated_at TEXT NOT NULL,
    first_seen_at  TEXT NOT NULL,
    source_ref     TEXT NOT NULL,
    processed      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (exchange, source_ref)
);
CREATE INDEX idx_filings_symbol_time ON filings (symbol, disseminated_at);
CREATE INDEX idx_filings_unprocessed ON filings (processed, disseminated_at);

CREATE TABLE board_meetings (
    symbol        TEXT NOT NULL,
    meeting_date  TEXT NOT NULL,
    purpose       TEXT NOT NULL DEFAULT '',
    description   TEXT,
    is_results    INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    PRIMARY KEY (symbol, meeting_date, purpose)
);
CREATE INDEX idx_board_meetings_date ON board_meetings (meeting_date);

CREATE TABLE events (
    event_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_key              TEXT NOT NULL UNIQUE,
    event_type               TEXT NOT NULL,
    headline                 TEXT NOT NULL,
    summary                  TEXT,
    region                   TEXT,
    scope                    TEXT,
    entities_json            TEXT NOT NULL DEFAULT '[]',
    source_quality           REAL,
    confidence               REAL,
    novelty                  REAL,
    severity                 REAL,
    horizon                  TEXT,
    event_time               TEXT,
    published_at             TEXT,
    first_seen_at            TEXT NOT NULL,
    last_updated_at          TEXT NOT NULL,
    n_articles               INTEGER NOT NULL DEFAULT 1,
    n_independent_sources    INTEGER NOT NULL DEFAULT 1,
    market_reacted           INTEGER,
    reaction_json            TEXT,
    evidence_json            TEXT NOT NULL DEFAULT '[]',
    contradictions_json      TEXT NOT NULL DEFAULT '[]',
    extraction_method        TEXT NOT NULL DEFAULT 'rules'
);
CREATE INDEX idx_events_first_seen ON events (first_seen_at);
CREATE INDEX idx_events_type ON events (event_type, first_seen_at);

CREATE TABLE event_articles (
    event_id   INTEGER NOT NULL REFERENCES events (event_id) ON DELETE CASCADE,
    article_id INTEGER REFERENCES news_raw (article_id) ON DELETE SET NULL,
    filing_id  INTEGER REFERENCES filings (filing_id) ON DELETE SET NULL,
    added_at   TEXT NOT NULL
);
CREATE INDEX idx_event_articles_event ON event_articles (event_id);

CREATE TABLE graph_nodes (
    node_id    TEXT PRIMARY KEY,
    node_type  TEXT NOT NULL,
    name       TEXT,
    attrs_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE graph_edges (
    edge_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    src               TEXT NOT NULL,
    dst               TEXT NOT NULL,
    relation          TEXT NOT NULL,
    direction         INTEGER NOT NULL DEFAULT 0,
    sensitivity       REAL,
    lag_days_mean     REAL,
    lag_days_sd       REAL,
    confidence        REAL NOT NULL DEFAULT 0.3,
    regime_dependence TEXT,
    rationale         TEXT NOT NULL DEFAULT '',
    source            TEXT NOT NULL DEFAULT 'prior',
    evidence_count    INTEGER NOT NULL DEFAULT 0,
    version           INTEGER NOT NULL DEFAULT 1,
    last_updated      TEXT NOT NULL,
    UNIQUE (src, dst, relation)
);
CREATE INDEX idx_graph_edges_src ON graph_edges (src);

-- ------------------------------------------------------------------- immutable snapshots / predictions
CREATE TABLE snapshots (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      TEXT NOT NULL,
    stage           TEXT NOT NULL CHECK (stage IN ('pre', 'final')),
    cutoff_ts       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    model_version   TEXT,
    code_version    TEXT,
    payload         BLOB NOT NULL,
    data_quality_json TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    UNIQUE (trade_date, stage, content_hash)
);
CREATE INDEX idx_snapshots_date ON snapshots (trade_date, stage);

CREATE TABLE predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id      INTEGER NOT NULL REFERENCES snapshots (snapshot_id),
    trade_date       TEXT NOT NULL,
    stage            TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    rank             INTEGER NOT NULL,
    tier             TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    model_status     TEXT NOT NULL,
    p_json           TEXT NOT NULL,
    range_json       TEXT NOT NULL,
    explanation_json TEXT NOT NULL,
    uncertainty      REAL,
    data_quality     TEXT NOT NULL,
    liquidity_json   TEXT,
    features_hash    TEXT,
    is_baseline      INTEGER NOT NULL DEFAULT 0,
    baseline_name    TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    UNIQUE (snapshot_id, symbol, is_baseline, baseline_name)
);
CREATE INDEX idx_predictions_date ON predictions (trade_date, stage, is_baseline);

CREATE TABLE outcomes (
    outcome_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions (prediction_id),
    trade_date    TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL,
    volume        REAL,
    range_pct     REAL,
    exc_pct       REAL,
    dir_ret_pct   REAL,
    mfe_pct       REAL,
    mae_pct       REAL,
    hit_json      TEXT,
    extra_json    TEXT,
    source        TEXT NOT NULL,
    validated     INTEGER NOT NULL DEFAULT 0,
    recorded_at   TEXT NOT NULL,
    UNIQUE (prediction_id, source)
);
CREATE INDEX idx_outcomes_date ON outcomes (trade_date);

-- Frozen predictions can never be edited; outcomes are attached separately.
CREATE TRIGGER snapshots_no_update BEFORE UPDATE ON snapshots
BEGIN
    SELECT RAISE(ABORT, 'snapshots are immutable');
END;

CREATE TRIGGER snapshots_no_delete BEFORE DELETE ON snapshots
WHEN COALESCE((SELECT value FROM kv WHERE key = 'maintenance.allow_prune'), '0') <> '1'
BEGIN
    SELECT RAISE(ABORT, 'snapshots are immutable');
END;

CREATE TRIGGER predictions_no_update BEFORE UPDATE ON predictions
BEGIN
    SELECT RAISE(ABORT, 'predictions are immutable');
END;

CREATE TRIGGER predictions_no_delete BEFORE DELETE ON predictions
WHEN COALESCE((SELECT value FROM kv WHERE key = 'maintenance.allow_prune'), '0') <> '1'
BEGIN
    SELECT RAISE(ABORT, 'predictions are immutable');
END;

-- --------------------------------------------------------------------------------------- evaluation
CREATE TABLE eval_metrics (
    metric_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date  TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    scope       TEXT NOT NULL,
    slice       TEXT NOT NULL DEFAULT 'all',
    name        TEXT NOT NULL,
    value       REAL,
    n           INTEGER,
    ci_low      REAL,
    ci_high     REAL,
    extra_json  TEXT
);
CREATE INDEX idx_eval_metrics ON eval_metrics (as_of_date, scope, name);

CREATE TABLE model_registry (
    model_version       TEXT PRIMARY KEY,
    stage               TEXT NOT NULL CHECK (stage IN ('pre', 'final')),
    created_at          TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('candidate', 'shadow', 'validated', 'rejected', 'retired')),
    is_active           INTEGER NOT NULL DEFAULT 0,
    train_start         TEXT,
    train_end           TEXT,
    feature_schema_hash TEXT,
    feature_version     TEXT,
    code_commit         TEXT,
    metrics_json        TEXT,
    artifact_path       TEXT,
    artifact_sha256     TEXT,
    artifact_signature  TEXT,
    approved_at         TEXT,
    activated_at        TEXT,
    parent_version      TEXT,
    notes               TEXT
);
CREATE UNIQUE INDEX idx_model_one_active ON model_registry (stage) WHERE is_active = 1;

-- ------------------------------------------------------------------------------------- operations
CREATE TABLE job_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name      TEXT NOT NULL,
    run_key       TEXT NOT NULL,
    scheduled_for TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    status        TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed', 'skipped', 'missed', 'timeout')),
    attempt       INTEGER NOT NULL DEFAULT 1,
    exit_code     INTEGER,
    message       TEXT,
    peak_rss_mb   REAL,
    pid           INTEGER
);
CREATE INDEX idx_job_runs_key ON job_runs (job_name, run_key);
CREATE INDEX idx_job_runs_started ON job_runs (started_at);

CREATE TABLE deliveries (
    delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    trade_date  TEXT,
    dedupe_key  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed', 'withheld')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    message_ids TEXT,
    error       TEXT,
    body_hash   TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    sent_at     TEXT,
    UNIQUE (channel, dedupe_key)
);

CREATE TABLE alerts (
    alert_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    level        TEXT NOT NULL,
    code         TEXT NOT NULL,
    message      TEXT NOT NULL,
    dedupe_key   TEXT,
    delivered    INTEGER NOT NULL DEFAULT 0,
    delivered_at TEXT
);
CREATE INDEX idx_alerts_pending ON alerts (delivered, created_at);
CREATE INDEX idx_alerts_dedupe ON alerts (dedupe_key, created_at);
