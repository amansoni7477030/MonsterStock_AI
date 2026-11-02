"""``sources-check``: reads every data endpoint the system depends on, through the production clients, and says which of them still
answer with data.

A provider renames or retires an endpoint now and then (NSE retired ``equity-stockIndices`` in 2026). This is the command that finds
that out before a scan does, and it names the reason of every failure. It only reads: nothing is stored and no report is touched.
Sections: ``nse`` (the JSON API behind www.nseindia.com), ``archive`` (the end-of-day files), ``yahoo``, ``rss`` (the news feeds) and
``gdelt`` (global news discovery: GDELT and the Google News search that stands in for it). An endpoint the system cannot do without
that does not answer is ``fail``; one that is optional (GDELT, its fallback), one of many (a single feed) or merely not published
yet is ``warn``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, timedelta
from typing import Any

from app.clock import get_clock, ist_date
from app.config import Settings
from app.doctor import DoctorReport
from app.doctor import render as render_report
from app.logging_setup import redact

SECTIONS = ("nse", "archive", "yahoo", "rss", "gdelt")
FILE_DAYS = 4  # weekdays an end-of-day file is looked for: today's is published in the evening
MAX_FEED_ROWS = 10  # failing feeds named one by one
Verdict = tuple[str, str]  # (status, detail)


def _reason(exc: BaseException) -> str:
    return " ".join(redact(str(exc)).split())[:140] or type(exc).__name__


def _count(n: int, noun: str, *, empty: str = "fail") -> Verdict:
    return ("ok", f"{n} {noun}") if n else (empty, f"answered with no {noun}")


def _probe(report: DoctorReport, name: str, call: Callable[[], Any], judge: Callable[[Any], Verdict], *, optional: bool = False) -> Any:
    """One read. ``judge`` turns the answer into a verdict; an endpoint that does not answer at all is ``fail``, or ``warn`` when it
    is optional. One broken endpoint never hides the others."""
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001
        report.add(name, "warn" if optional else "fail", _reason(exc))
        return None
    try:
        status, detail = judge(result)
    except Exception as exc:  # noqa: BLE001 - an answer of an unexpected shape is a finding, not a crash
        status, detail = "fail", f"unexpected answer ({type(exc).__name__})"
    report.add(name, "warn" if optional and status == "fail" else status, detail)
    return result


# ------------------------------------------------------------------------------------------------ NSE
def check_nse(report: DoctorReport, s: Settings, today: date) -> None:
    from ingestion.nse import LIVE_LISTS, NseApi

    if not s.nse_enabled:
        report.add("NSE API", "ok", "off (NSE_ENABLED=false)")
        return
    api = NseApi(s)
    try:
        _probe(report, "NSE market status", api.market_status, lambda r: _count(len(r), "blocks"))
        _probe(report, "NSE GIFT Nifty", api.gift_nifty,
               lambda r: ("ok", f"{r['price']:g}") if r else ("warn", "the market status has no GIFT Nifty block"))
        _probe(report, "NSE pre-open book", lambda: api.preopen("ALL"), lambda r: _count(len(r.rows), "stocks"))
        _probe(report, "NSE all indices", api.all_indices,
               lambda r: ("ok", f"{len(r)} indices") if "NIFTY 50" in r and "INDIA VIX" in r
               else ("fail", f"{len(r)} indices, NIFTY 50 or INDIA VIX is missing"))
        for name in LIVE_LISTS.get(s.universe_mode, LIVE_LISTS["fno_nifty100"]):
            _probe(report, f"NSE live quotes ({name})", lambda name=name: api.live_equities(name),
                   lambda r: _count(len(r.rows), f"stocks, answer stamped {r.source_ts}"))
        _probe(report, "NSE announcements", lambda: api.announcements(today - timedelta(days=3), today),
               lambda r: _count(len(r), "filings in 3 days", empty="warn"))
        _probe(report, "NSE board meetings", lambda: api.event_calendar(today, today + timedelta(days=14)),
               lambda r: _count(len(r), "meetings in 14 days", empty="warn"))
        _probe(report, "NSE trading holidays", api.holidays, lambda r: _count(len(r), "dates"))
    finally:
        api.http.close()


# -------------------------------------------------------------------------------------------- archive
def _newest(report: DoctorReport, name: str, read: Callable[[date], Any], days: Sequence[date]) -> None:
    """Pass when the file exists for any of ``days`` (the newest first): today's may simply not be published yet."""
    problem = f"no file for the last {len(days)} weekdays"
    for day in days:
        try:
            rows = read(day)
        except Exception as exc:  # noqa: BLE001
            problem = _reason(exc)
            continue
        if rows:
            late = "" if day == days[0] else f" (the file of {days[0]} is not published yet)"
            report.add(name, "ok", f"{len(rows)} rows for {day}{late}")
            return
    report.add(name, "fail", problem)


def check_archive(report: DoctorReport, s: Settings, today: date) -> None:
    from ingestion.nse import INDEX_FILES, NseArchive

    arc = NseArchive(s)
    try:
        weekdays = [day for day in (today - timedelta(days=n) for n in range(FILE_DAYS + 4)) if day.weekday() < 5][:FILE_DAYS]
        _newest(report, "NSE bhavcopy (cash market)", arc.bhavcopy, weekdays)
        _newest(report, "NSE bhavcopy (F&O, UDiFF)", arc.fo_bhavcopy, weekdays)
        for key in INDEX_FILES:
            _probe(report, f"NSE index list {key}", lambda key=key: arc.index_constituents(key), lambda r: _count(len(r), "stocks"))
        _probe(report, "NSE F&O lot sizes", arc.fo_lot_sizes, lambda r: _count(len(r), "underlyings"))
    finally:
        arc.http.close()


# ---------------------------------------------------------------------------------------------- Yahoo
def check_yahoo(report: DoctorReport, s: Settings) -> None:
    from ingestion.pipeline import LIVE_CONTEXT
    from ingestion.series import SERIES
    from ingestion.yahoo import YahooClient

    client = YahooClient(s)
    keys = [spec.key for spec in SERIES]

    def spark(found: dict[str, Any]) -> Verdict:
        missing = [key for key in keys if key not in found]
        if not missing:
            return "ok", f"{len(keys)} series"
        return ("warn" if found else "fail"), f"{len(found)} of {len(keys)} series answered; missing {', '.join(missing)[:90]}"

    def bars(series: Any) -> Verdict:
        return ("ok", f"{len(series.bars)} daily bars, the last of {series.bars[-1]['trade_date']}") if series.bars else ("fail", "no bars")

    try:
        _probe(report, "Yahoo spark (bulk probe)", lambda: client.spark_latest(keys), spark)
        for symbol in ("^NSEI", "RELIANCE.NS"):
            _probe(report, f"Yahoo chart {symbol}", lambda symbol=symbol: client.fetch_daily(symbol, range_="5d"), bars)
        silent: list[str] = []
        for symbol in LIVE_CONTEXT:
            try:
                if client.fetch_quote(symbol).get("price") is None:
                    silent.append(symbol)
            except Exception:  # noqa: BLE001
                silent.append(symbol)
        report.add("Yahoo live-context quotes", "warn" if silent else "ok",
                   f"no quote for {', '.join(silent)}" if silent else f"{len(LIVE_CONTEXT)} of {len(LIVE_CONTEXT)}")
    finally:
        client.http.close()


# ------------------------------------------------------------------------------------------ RSS, GDELT
def check_rss(report: DoctorReport, s: Settings) -> None:
    import feedparser

    from ingestion.news_sources import NewsCollector, load_feed_registry

    try:
        feeds, _queries = load_feed_registry(s.news_feeds_file)
    except Exception as exc:  # noqa: BLE001
        report.add("RSS feed registry", "fail", _reason(exc))
        return
    if not feeds:
        report.add("RSS feeds", "warn", "the registry lists no feeds")
        return
    http = NewsCollector(None, s).http  # the very client (headers, spacing, size cap) the news job uses
    failed: list[tuple[str, str]] = []
    try:
        for feed in feeds:
            try:
                parsed = feedparser.parse(http.get(feed.url).content)
                if not parsed.entries:  # a page that is not a feed parses without an error, but it has nothing to read
                    raise ValueError("not a readable feed" if getattr(parsed, "bozo", 0) else "the feed has no items")
            except Exception as exc:  # noqa: BLE001
                failed.append((feed.id, _reason(exc)))
    finally:
        http.close()
    answered = len(feeds) - len(failed)
    report.add("RSS feeds", "ok" if not failed else "warn" if answered * 2 >= len(feeds) else "fail", f"{answered} of {len(feeds)} answer")
    for feed_id, reason in failed[:MAX_FEED_ROWS]:
        report.add(f"RSS {feed_id}", "warn", reason)


def check_gdelt(report: DoctorReport, s: Settings) -> None:
    """Global news discovery: the GDELT DOC API and the Google News search that stands in for it. Both are optional: a topic is lost
    only when neither answers, and the RSS feeds do not depend on either."""
    from ingestion.news_sources import GDELT_URL, NewsCollector, gdelt_client, gnews_search, read_gnews

    query = "Nifty sourcelang:eng"
    if s.gdelt_enabled:
        client = gdelt_client(s)
        params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": "5", "timespan": "24h", "sort": "datedesc"}
        try:
            _probe(report, "GDELT DOC API (optional)", lambda: client.get(GDELT_URL, params=params).json(),
                   lambda r: _count(len(r.get("articles") or []) if isinstance(r, dict) else 0, "articles", empty="warn"), optional=True)
        finally:
            client.close()
    else:
        report.add("GDELT DOC API", "ok", "off (GDELT_ENABLED=false)")
    if not s.gdelt_fallback:
        report.add("Google News search (GDELT fallback)", "ok", "off (GDELT_FALLBACK=false)")
        return
    http = NewsCollector(None, s).http  # the very client (headers, spacing, size cap) the news job uses
    try:
        _probe(report, "Google News search (GDELT fallback)", lambda: read_gnews(http, gnews_search(query, 24)),
               lambda r: _count(len(r), "items", empty="warn"), optional=True)
    finally:
        http.close()


# --------------------------------------------------------------------------------------------- runner
def run_checks(settings: Settings, sections: Sequence[str] | None = None, *, today: date | None = None) -> DoctorReport:
    """The checks of ``sections`` (all of them by default), in the order of ``SECTIONS``."""
    report = DoctorReport()
    day = today or ist_date(get_clock().now())
    wanted = set(sections or SECTIONS)
    runners: dict[str, Callable[[], None]] = {
        "nse": lambda: check_nse(report, settings, day), "archive": lambda: check_archive(report, settings, day),
        "yahoo": lambda: check_yahoo(report, settings), "rss": lambda: check_rss(report, settings),
        "gdelt": lambda: check_gdelt(report, settings),
    }
    for name in SECTIONS:
        if name in wanted:
            runners[name]()
    return report


def render(report: DoctorReport) -> str:
    """The report as the doctor prints it, with a verdict about endpoints instead of about the machine."""
    failing = sum(check.status == "fail" for check in report.checks)
    return render_report(report, failed=f"{failing} endpoint(s) FAIL: see above",
                         passed="every endpoint answers (warnings, if any, are optional or temporary)")
