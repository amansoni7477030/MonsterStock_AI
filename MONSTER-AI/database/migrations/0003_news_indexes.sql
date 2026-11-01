-- Indexes for the retention and lookup paths of the news / event tables.
-- event_articles(article_id|filing_id): the ON DELETE SET NULL actions of news_raw / filings scan this table on every
-- delete without them (the nightly news retention deletes thousands of rows).

CREATE INDEX IF NOT EXISTS idx_event_articles_article ON event_articles (article_id);
CREATE INDEX IF NOT EXISTS idx_event_articles_filing ON event_articles (filing_id);
CREATE INDEX IF NOT EXISTS idx_filings_first_seen ON filings (first_seen_at);
CREATE INDEX IF NOT EXISTS idx_events_last_updated ON events (last_updated_at);
