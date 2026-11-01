"""News, filings and calendar collection (blueprint sections 4, 7, 17).

Raw items are stored with BOTH the publisher timestamp and our own first-seen timestamp (the latter is what point-in-time
queries use). Per-feed circuit breakers isolate flaky publishers; duplicates are rejected at the database level by URL hash.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser

from app.clock import Clock, get_clock, iso, parse_ts
from app.config import Settings
from app.errors import CircuitOpen, DataUnavailable
from app.logging_setup import redact
from database.store import Store
from ingestion.http import USER_AGENT_SIMPLE, HttpClient
from ingestion.nse import NseApi

log = logging.getLogger(__name__)

FEED_DISABLE_AFTER = 6
FEED_DISABLE_HOURS = 24
FEED_FULL_REFRESH_HOURS = 6  # conditional GETs are trusted for this long; then one unconditional fetch (a publisher may answer 304 forever)
MAX_ITEMS_PER_FEED = 60
TRACKING_PARAMS = {"fbclid", "gclid", "ref", "ocid", "cmpid", "taid", "mc_cid", "mc_eid", "guccounter", "guce_referrer"}


@dataclass
class FeedSpec:
    id: str
    url: str
    publisher: str
    region: str
    category: str
    kind: str = "media"


@dataclass
class CollectReport:
    attempted: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    not_modified: int = 0  # feeds answered with 304: nothing new, nothing downloaded
    new_items: int = 0
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if len(self.notes) < 10:
            self.notes.append(text)

    @property
    def success_ratio(self) -> float:
        return self.ok / self.attempted if self.attempted else 1.0


def load_feed_registry(path: Path) -> tuple[list[FeedSpec], list[dict[str, str]]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataUnavailable("rss_news", f"feed registry unreadable ({type(exc).__name__})", retryable=False) from None
    feeds = []
    for item in data.get("feeds", []):
        if not item.get("url", "").startswith("https://") or not item.get("id"):
            continue
        feeds.append(FeedSpec(item["id"], item["url"], item.get("publisher", ""), item.get("region", "global"),
                              item.get("category", ""), item.get("kind", "media")))
    return feeds, list(data.get("gdelt_queries", []))


def canonical_url(url: str) -> str:
    """Lower-case host, drop tracking params and fragments, sort the remaining params."""
    parsed = urlparse(url.strip())
    query = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                   if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS)
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/") or "/", "", urlencode(query), ""))


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324 - identity key, not security


def _clean(text: Any, limit: int) -> str:
    try:
        from news_intelligence.text import clean_text

        return clean_text(str(text or ""), max_chars=limit)
    except ImportError:  # pragma: no cover - the news package is installed in every deployment
        import html
        import re

        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(text or "")))).strip()[:limit]


def _release_day(raw: Any, when: datetime) -> str:
    """Calendar date of an economic release as the feed states it (in the feed's own UTC offset, so a release moved by an hour
    keeps its date even when the UTC date flips); the UTC date when the text is not an ISO date."""
    try:
        return date.fromisoformat(str(raw or "").strip()[:10]).isoformat()
    except ValueError:
        return when.date().isoformat()


class NewsCollector:
    def __init__(self, store: Store, settings: Settings, *, http: HttpClient | None = None, clock: Clock | None = None) -> None:
        self.store, self.settings, self.clock = store, settings, clock
        self.http = http or HttpClient.from_settings(
            "rss", settings, headers={"User-Agent": USER_AGENT_SIMPLE, "Accept": "application/rss+xml, application/xml, text/xml, */*"},
            min_interval=0.4, timeout=(8.0, 20.0), max_attempts=2, max_bytes=6 * 1024 * 1024, clock=clock,
            breaker=_NoBreaker(),
        )

    def _now(self) -> datetime:
        return (self.clock or get_clock()).now()

    # -------------------------------------------------------------------------------------------- RSS
    def _feed_disabled(self, feed: FeedSpec) -> bool:
        until = parse_ts(self.store.kv_get(f"feed:{feed.id}:disabled_until"))
        return until is not None and until > self._now()

    def _feed_failure(self, feed: FeedSpec) -> None:
        fails = int(self.store.kv_get(f"feed:{feed.id}:fails", "0") or 0) + 1
        self.store.kv_set(f"feed:{feed.id}:fails", str(fails))
        if fails >= FEED_DISABLE_AFTER:
            self.store.kv_set(f"feed:{feed.id}:disabled_until", iso(self._now() + timedelta(hours=FEED_DISABLE_HOURS)) or "")
            self.store.kv_set(f"feed:{feed.id}:fails", "0")
            log.warning("feed %s disabled for %d h after %d failures", feed.id, FEED_DISABLE_HOURS, fails)

    def _validators(self, feed: FeedSpec) -> dict[str, str]:
        """``If-None-Match`` / ``If-Modified-Since`` for a feed fetched in full recently: an unchanged feed then costs one tiny
        304 answer instead of a download (fewer requests bytes and fewer chances of a rate limit)."""
        full_at = parse_ts(self.store.kv_get(f"feed:{feed.id}:full_at"))
        if full_at is None or self._now() - full_at > timedelta(hours=FEED_FULL_REFRESH_HOURS):
            return {}
        headers: dict[str, str] = {}
        etag = self.store.kv_get(f"feed:{feed.id}:etag")
        modified = self.store.kv_get(f"feed:{feed.id}:lastmod")
        if etag:
            headers["If-None-Match"] = etag
        if modified:
            headers["If-Modified-Since"] = modified
        return headers

    def _remember_validators(self, feed: FeedSpec, response: Any) -> None:
        self.store.kv_set(f"feed:{feed.id}:etag", response.headers.get("etag") or "")
        self.store.kv_set(f"feed:{feed.id}:lastmod", response.headers.get("last-modified") or "")
        self.store.kv_set(f"feed:{feed.id}:full_at", iso(self._now()) or "")

    def collect_rss(self, feeds: Iterable[FeedSpec], *, lookback_hours: int | None = None, deadline: float | None = None,
                    region: str | None = None) -> CollectReport:
        report = CollectReport()
        clk = self.clock or get_clock()
        horizon = self._now() - timedelta(hours=lookback_hours or self.settings.news_lookback_hours)
        for feed in feeds:
            if region and feed.region != region:
                continue
            if deadline is not None and clk.monotonic() > deadline:
                report.note("deadline reached; remaining feeds deferred")
                break
            if self._feed_disabled(feed):
                report.skipped += 1
                continue
            report.attempted += 1
            try:
                response = self.http.get(feed.url, headers=self._validators(feed), soft_statuses=(304,))
                if response.status == 304:
                    report.ok += 1
                    report.not_modified += 1
                    self.store.kv_set(f"feed:{feed.id}:fails", "0")
                    continue
                parsed = feedparser.parse(response.content)
                if not parsed.entries and getattr(parsed, "bozo", 0):
                    raise DataUnavailable("rss_news", "unparseable feed", retryable=True)
            except (DataUnavailable, CircuitOpen) as exc:
                report.failed += 1
                report.note(f"{feed.id}: {redact(str(exc))}")
                self._feed_failure(feed)
                continue
            except Exception as exc:  # noqa: BLE001 - one malformed feed must never abort collection
                log.warning("rss %s: unexpected %s while collecting; the feed is skipped this round", feed.id,
                            type(exc).__name__, exc_info=True)
                report.failed += 1
                report.note(f"{feed.id}: {type(exc).__name__}")
                self._feed_failure(feed)
                continue
            self.store.kv_set(f"feed:{feed.id}:fails", "0")
            self._remember_validators(feed, response)
            report.ok += 1
            report.new_items += self._store_entries(feed, parsed.entries[:MAX_ITEMS_PER_FEED], horizon)
        self.store.record_source_result(
            "rss_news", report.success_ratio >= 0.5 or report.attempted == 0,
            error=None if report.success_ratio >= 0.5 else f"{report.failed}/{report.attempted} feeds failed")
        return report

    def _store_entries(self, feed: FeedSpec, entries: list[Any], horizon: datetime) -> int:
        now_iso = iso(self._now())
        rows = []
        for entry in entries:
            title = _clean(entry.get("title"), 300)
            link = str(entry.get("link") or "").strip()
            if not title or not link.startswith(("https://", "http://")):
                continue
            stamp = entry.get("published_parsed") or entry.get("updated_parsed")
            published = datetime.fromtimestamp(calendar.timegm(stamp), timezone.utc) if stamp else None
            if published is not None and published < horizon:
                continue
            if published is not None and published > self._now() + timedelta(hours=6):
                published = None  # nonsense future timestamp: treat as unknown rather than trusting it
            summary = _clean(entry.get("summary") or entry.get("description"), 1500)
            rows.append((feed.publisher or feed.id, feed.id, link, _hash(canonical_url(link)), _hash(f"{title}|{summary[:300]}"), title,
                         summary or None, iso(published), now_iso, feed.region, feed.category))
        if not rows:
            return 0
        before = self.store.conn.total_changes
        with self.store.tx():
            self.store.conn.executemany(
                "INSERT OR IGNORE INTO news_raw (source, feed, url, url_hash, content_hash, title, summary, published_at, "
                "first_seen_at, region, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return self.store.conn.total_changes - before

    # ------------------------------------------------------------------------------------------ GDELT
    def collect_gdelt(self, queries: Iterable[dict[str, str]], *, lookback_hours: int | None = None,
                      deadline: float | None = None) -> CollectReport:
        """GDELT DOC API (discovery only: titles, domains, times). Honours the 1-request-per-5-seconds rule."""
        report = CollectReport()
        if not self.settings.gdelt_enabled:
            return report
        client = HttpClient.from_settings(
            "gdelt", self.settings, headers={"User-Agent": USER_AGENT_SIMPLE}, min_interval=6.0, timeout=(8.0, 25.0),
            max_attempts=2, backoff_base=7.0, clock=self.clock, max_bytes=4 * 1024 * 1024)
        clk = self.clock or get_clock()
        hours = lookback_hours or self.settings.news_lookback_hours
        now_iso = iso(self._now())
        for q in queries:
            if deadline is not None and clk.monotonic() > deadline:
                report.note("deadline reached")
                break
            report.attempted += 1
            try:
                payload = client.get("https://api.gdeltproject.org/api/v2/doc/doc", params={
                    "query": q["query"], "mode": "artlist", "format": "json", "maxrecords": "60", "timespan": f"{hours}h",
                    "sort": "datedesc"}).json()
            except (DataUnavailable, CircuitOpen) as exc:
                report.failed += 1
                report.note(f"{q.get('id')}: {redact(str(exc))}")
                continue
            report.ok += 1
            rows = []
            for article in (payload.get("articles") or []) if isinstance(payload, dict) else []:
                url = str(article.get("url") or "")
                title = _clean(article.get("title"), 300)
                if not title or not url.startswith(("https://", "http://")) or str(article.get("language", "English")) != "English":
                    continue
                seen = parse_ts(article.get("seendate"))
                rows.append((str(article.get("domain") or "gdelt"), f"gdelt:{q.get('id')}", url, _hash(canonical_url(url)),
                             _hash(f"{title}|"), title, None, iso(seen), now_iso, "global", q.get("category")))
            if rows:
                before = self.store.conn.total_changes
                with self.store.tx():
                    self.store.conn.executemany(
                        "INSERT OR IGNORE INTO news_raw (source, feed, url, url_hash, content_hash, title, summary, published_at, "
                        "first_seen_at, region, category) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
                report.new_items += self.store.conn.total_changes - before
        self.store.record_source_result("gdelt", report.success_ratio >= 0.5 or report.attempted == 0,
                                        error=None if report.success_ratio >= 0.5 else "queries failing (rate limit?)")
        return report

    # --------------------------------------------------------------------------------- NSE filings etc.
    def collect_filings(self, nse: NseApi, start: date, end: date, symbols: set[str]) -> int:
        try:
            items = nse.announcements(start, end, symbols)
        except DataUnavailable as exc:
            self.store.record_source_result("nse_announcements", False, error=redact(str(exc)))
            log.warning("filings collection failed: %s", redact(str(exc)))
            return 0
        now_iso = iso(self._now())
        before = self.store.conn.total_changes
        with self.store.tx():
            self.store.conn.executemany(
                "INSERT OR IGNORE INTO filings (exchange, symbol, isin, company, category, subject, text, attachment_url, "
                "disseminated_at, first_seen_at, source_ref) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(f["exchange"], f["symbol"], f["isin"], f["company"], f["category"], f["subject"], f["text"],
                  f["attachment_url"], f["disseminated_at"], now_iso, f["source_ref"]) for f in items])
        new_rows = self.store.conn.total_changes - before  # counted before the source-health bookkeeping below adds changes
        self.store.record_source_result("nse_announcements", True)
        return new_rows

    def collect_board_meetings(self, nse: NseApi, start: date, end: date, symbols: set[str]) -> int:
        try:
            items = nse.event_calendar(start, end, symbols)
        except DataUnavailable as exc:
            self.store.record_source_result("nse_announcements", False, error=redact(str(exc)))
            log.warning("board-meeting collection failed: %s", redact(str(exc)))
            return 0
        written = self.store.upsert_board_meetings(items)
        if items:
            earliest = min(i["meeting_date"] for i in items)
            current = self.store.kv_get("board_meetings.coverage_start")
            if current is None or earliest < current:
                self.store.kv_set("board_meetings.coverage_start", earliest)
        return written

    # -------------------------------------------------------------------------------- economic calendar
    def collect_econ_calendar(self) -> int:
        """Upsert this week's releases. A release is keyed on (country, title, release DATE), so a release the publisher moved
        to another time updates its row (time, impact, forecast) instead of adding a duplicate."""
        client = HttpClient.from_settings(
            "econ_calendar", self.settings, headers={"User-Agent": USER_AGENT_SIMPLE}, min_interval=1.0, timeout=(8.0, 20.0),
            max_attempts=2, clock=self.clock, max_bytes=2 * 1024 * 1024)
        try:
            data = client.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json").json()
        except (DataUnavailable, CircuitOpen) as exc:
            self.store.record_source_result("econ_calendar", False, error=redact(str(exc)))
            log.warning("economic calendar collection failed: %s", redact(str(exc)))
            return 0
        if not isinstance(data, list):  # e.g. an error object answered with HTTP 200: not "a quiet week"
            self.store.record_source_result("econ_calendar", False, error="unexpected payload")
            log.warning("economic calendar collection failed: unexpected payload (%s)", type(data).__name__)
            return 0
        now_iso = iso(self._now())
        releases: dict[str, tuple] = {}
        legacy_keys: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            when = parse_ts(item.get("date"))
            title = _clean(item.get("title"), 160)
            if when is None or not title:
                continue
            country = str(item.get("country") or "")[:8]
            key = _hash(f"{country}|{title}|{_release_day(item.get('date'), when)}")
            row = (key, country, title, str(item.get("impact") or "")[:10], iso(when), str(item.get("forecast") or "")[:40],
                   str(item.get("previous") or "")[:40], now_iso, now_iso)
            if key not in releases or row[4] < releases[key][4]:  # same title twice in one day: the earlier time, whatever the order
                releases[key] = row
            legacy_keys[_hash(f"{iso(when)}|{country}|{title}")] = key
        if data and not releases:  # a feed that changed shape must not look like a quiet week
            self.store.record_source_result("econ_calendar", False, error="no readable items")
            log.warning("economic calendar collection failed: none of the %d items could be read", len(data))
            return 0
        with self.store.tx():
            for legacy, key in legacy_keys.items():  # rows written before the key ignored the time keep their history
                self.store.conn.execute("UPDATE OR IGNORE econ_calendar SET event_key = ? WHERE event_key = ?", (key, legacy))
                self.store.conn.execute("DELETE FROM econ_calendar WHERE event_key = ?", (legacy,))
            self.store.conn.executemany(
                "INSERT INTO econ_calendar (event_key, country, title, impact, scheduled_at, forecast, previous, source, first_seen_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'faireconomy', ?, ?) ON CONFLICT(event_key) DO UPDATE SET "
                "impact = excluded.impact, scheduled_at = excluded.scheduled_at, forecast = excluded.forecast, "
                "previous = excluded.previous, updated_at = excluded.updated_at", list(releases.values()))
        self.store.record_source_result("econ_calendar", True)
        return len(releases)

    # ---------------------------------------------------------------------------------------- retention
    def prune_raw(self, retention_days: int | None = None) -> int:
        days = retention_days or self.settings.news_retention_days
        cutoff = iso(self._now() - timedelta(days=days))
        cur = self.store.conn.execute("DELETE FROM news_raw WHERE first_seen_at < ? AND processed <> 0", (cutoff,))
        return cur.rowcount


class _NoBreaker:
    """Per-feed isolation happens in the collector; the shared HTTP client must not trip globally on one bad feed."""

    def allow(self) -> bool:
        return True

    def record_success(self) -> None:
        return None

    def record_failure(self) -> None:
        return None

    @property
    def is_open(self) -> bool:
        return False
