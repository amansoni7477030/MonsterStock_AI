"""Data-collection jobs: orchestration, coverage decisions, degradation to vendor data, retries via failed status."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.errors import DataUnavailable
from automation import jobs
from automation.context import JobResult
from ingestion.pipeline import IngestReport
from tests.seed_data import FakeNews, FakeNseApi, make_job_context, panel_ending, populate_store, preopen_rows

UTC = timezone.utc


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=300, n_symbols=24, seed=5)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    populate_store(store, panel)
    return type("Env", (), {"settings": settings, "store": store, "clock": fake_clock, "panel": panel})


def ctx_for(env, run_key="2026-11-02", **kw):
    return make_job_context(env.settings, env.store, env.clock, run_key, **kw)


def fill_bars(day: str):
    """Fake official bhavcopy: writes a rank-2 bar for every universe symbol still missing ``day``."""

    def fake(store, _archive, d):
        written = 0
        for row in store.list_instruments(kind="equity", in_universe=True):
            if store.latest_bar_date(int(row["instrument_id"])) != d.isoformat():
                store.upsert_bars(int(row["instrument_id"]), [{"trade_date": d.isoformat(), "open": 100, "high": 104, "low": 99,
                                                              "close": 103, "volume": 1e6}], source="nse_bhavcopy", source_rank=2,
                                  quality="official")
                written += 1
        return written, written

    return fake


# ------------------------------------------------------------------------------------------ collect_global
def test_collect_global_success_and_failure_modes(env, monkeypatch):
    monkeypatch.setattr(jobs, "collect_live_context", lambda *a, **k: 9)
    monkeypatch.setattr(jobs, "collect_gift", lambda *a, **k: {"price": 1})
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=33, failed=1))
    ok = jobs.collect_global(ctx_for(env))
    assert ok.status == "success" and ok.metrics["series_ok"] == 33
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=5, failed=20))
    assert jobs.collect_global(ctx_for(env)).status == "failed"
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=33, partial=True, notes=["deadline"]))
    partial = jobs.collect_global(ctx_for(env))
    assert partial.status == "failed" and "deadline" in partial.message


def test_collect_global_survives_optional_step_failures(env, monkeypatch):
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=33))
    monkeypatch.setattr(jobs, "collect_live_context", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yahoo quote api down")))
    monkeypatch.setattr(jobs, "collect_gift", lambda *a, **k: (_ for _ in ()).throw(DataUnavailable("nse", "blocked")))
    assert jobs.collect_global(ctx_for(env)).status == "success"


# --------------------------------------------------------------------------------------------- collect_news
def test_collect_news_tolerates_partial_feed_failure(env, monkeypatch):
    from ingestion import news_sources

    monkeypatch.setattr(news_sources, "load_feed_registry", lambda path: ([object()], [{"id": "q"}]))
    ctx = ctx_for(env)
    result = jobs.collect_news(ctx)
    assert result.status == "success" and "9/10 feeds ok" in result.message
    assert result.metrics["rss_new"] == 25 and result.metrics["gdelt_new"] == 7
    ctx._cache["news"] = FakeNews(rss_ok=2, rss_failed=8)
    assert jobs.collect_news(ctx).status == "failed"


def test_the_news_job_says_what_the_ai_services_contributed(env, monkeypatch):
    from ingestion import news_sources

    class Intel:
        def process(self):
            return {"llm_used": 12, "llm_cached": 3, "llm_failed": 0, "jev_used": 40, "jev_cached": 0, "jev_failed": 1}

    monkeypatch.setattr(news_sources, "load_feed_registry", lambda path: ([object()], [{"id": "q"}]))
    ctx = ctx_for(env)
    ctx._cache["intel"] = Intel()
    result = jobs.collect_news(ctx)
    assert result.status == "success" and result.message.endswith("; AI: llm 12 used, 3 cached; jev 40 used, 1 failed")
    assert result.metrics["processed"]["jev_used"] == 40


@pytest.mark.parametrize(("processed", "expected"), [
    ({}, ""), ({"error": "RuntimeError"}, ""), ({"llm_used": 0, "jev_used": 0}, ""),
    ({"jev_cached": 4}, "; AI: jev 0 used, 4 cached"), ({"llm_failed": 2}, "; AI: llm 0 used, 2 failed"),
    ({"llm_used": None, "jev_used": 1}, "; AI: jev 1 used"),
])
def test_the_ai_note_only_mentions_services_that_did_something(processed, expected):
    assert jobs.ai_note(processed) == expected


def test_collect_news_reports_unreadable_registry(env, monkeypatch):
    from ingestion import news_sources

    def broken(path):
        raise DataUnavailable("rss_news", "feed registry unreadable", retryable=False)

    monkeypatch.setattr(news_sources, "load_feed_registry", broken)
    assert jobs.collect_news(ctx_for(env)).status == "failed"


def test_filings_window_reaches_back_over_weekends_and_holidays(env):
    def window(run_key):
        ctx = ctx_for(env, run_key)
        ctx.ensure_calendar()
        return tuple(d.isoformat() for d in jobs.filings_window(ctx))

    assert window("2026-11-02") == ("2026-10-29", "2026-11-02")  # Monday: Friday's after-hours filings are included
    assert window("2026-11-03") == ("2026-11-01", "2026-11-03")  # Tuesday: from the day before Monday's session
    assert window("2026-10-31") == ("2026-10-29", "2026-10-31")  # a Saturday run: previous session Friday 30 Oct
    assert window("2026-11-11") == ("2026-11-08", "2026-11-11")  # Diwali (10 Nov) is no session: Monday 9 Nov is the anchor


def test_collect_news_asks_for_the_whole_window(env, monkeypatch):
    from ingestion import news_sources

    class Recording(FakeNews):
        def collect_filings(self, nse, start, end, symbols):
            self.window = (start.isoformat(), end.isoformat())
            return 3

    monkeypatch.setattr(news_sources, "load_feed_registry", lambda path: ([object()], [{"id": "q"}]))
    ctx = ctx_for(env, "2026-11-02")
    ctx._cache["news"] = recording = Recording()
    result = jobs.collect_news(ctx)
    assert recording.window == ("2026-10-29", "2026-11-02")
    assert result.metrics["filings_new"] == 3 and result.metrics["filings_window"] == "2026-10-29..2026-11-02"


# --------------------------------------------------------------------------------------------- collect_india
def test_collect_india_is_a_noop_when_the_session_is_complete(env, monkeypatch):
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=3))
    monkeypatch.setattr(jobs, "ingest_bhavcopy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    result = jobs.collect_india(ctx_for(env))
    assert result.status == "success" and result.metrics["coverage"] == 1.0


def test_collect_india_repairs_missing_symbols_from_the_official_file(settings, store, fake_clock, panel, monkeypatch):
    populate_store(store, panel, skip_last_for=8)
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=3))
    monkeypatch.setattr(jobs, "ingest_bhavcopy", fill_bars("2026-10-30"))
    ctx = make_job_context(settings, store, fake_clock)
    result = jobs.collect_india(ctx)
    assert result.status == "success", result.message
    assert result.metrics["official_rows"] == 8 and result.metrics["coverage"] == 1.0


def test_collect_india_falls_back_to_vendor_then_reports_failure(settings, store, fake_clock, panel, monkeypatch):
    populate_store(store, panel, skip_last_for=8)
    monkeypatch.setattr(jobs, "ingest_bhavcopy", lambda *a, **k: (_ for _ in ()).throw(DataUnavailable("nse", "blocked")))
    calls = []

    def vendor(store_, yahoo, **kw):
        calls.append(kw.get("symbols"))
        return IngestReport(ok=0, failed=len(kw.get("symbols") or []))

    monkeypatch.setattr(jobs, "refresh_daily_bars", vendor)
    result = jobs.collect_india(make_job_context(settings, store, fake_clock))
    assert result.status == "failed" and "67%" in result.message
    assert len(calls[0]) == 8  # only the missing names are requested from the vendor
    assert set(calls[1]) == {"^NSEI", "^NSEBANK", "^INDIAVIX"}


def test_unexpected_provider_error_degrades_instead_of_crashing(settings, store, fake_clock, panel, monkeypatch):
    populate_store(store, panel, skip_last_for=2)
    monkeypatch.setattr(jobs, "ingest_bhavcopy", lambda *a, **k: (_ for _ in ()).throw(KeyError("unexpected column")))
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=0, failed=2))
    result = jobs.collect_india(make_job_context(settings, store, fake_clock))
    assert result.status == "success"  # 22 of 24 names (92%) is above the 90% threshold; the KeyError was absorbed
    assert result.metrics["coverage"] == pytest.approx(22 / 24, abs=1e-3) and "official_error" in result.metrics


def test_trading_jobs_are_skipped_on_holidays(env):
    result = jobs.collect_india(ctx_for(env, run_key="2026-11-08"))  # Muhurat Sunday: closed for the regular session
    assert result.status == "skipped" and "closed" in result.message


# -------------------------------------------------------------------------------------------- prelim_scan
def stub_scan(monkeypatch, order):
    """The scan itself is covered by test_scan.py: record only that (and when) it is asked to run."""

    def fake(ctx, stage):
        order.append(f"scan:{stage}")
        return JobResult("success", "scanned")

    monkeypatch.setattr(jobs.scan, "run_scan", fake)


def test_prelim_scan_repairs_an_incomplete_previous_session_before_scanning(settings, store, fake_clock, panel, monkeypatch):
    populate_store(store, panel, skip_last_for=10)  # 14 of 24 names (58 %) have Friday's bar
    order = []
    official, repair = fill_bars("2026-10-30"), jobs.ensure_session_bars

    def fetch_official(store_, archive, day):
        order.append("official file")
        return official(store_, archive, day)

    def recording_repair(ctx, session, **kw):
        order.append(("repair", session.isoformat(), kw))
        return repair(ctx, session, **kw)

    monkeypatch.setattr(jobs, "ingest_bhavcopy", fetch_official)
    monkeypatch.setattr(jobs, "ensure_session_bars", recording_repair)
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=0))
    stub_scan(monkeypatch, order)
    ctx = make_job_context(settings, store, fake_clock)
    assert jobs.prelim_scan(ctx).status == "success"
    assert order == [("repair", "2026-10-30", {"budget_sec": 120}), "official file", "scan:pre"]
    assert jobs.session_coverage(ctx, date(2026, 10, 30))[0] == 1.0


@pytest.mark.parametrize(("missing", "repaired"), [(2, False), (3, True)])
def test_prelim_scan_repairs_only_below_the_coverage_threshold(settings, store, fake_clock, panel, monkeypatch, missing, repaired):
    populate_store(store, panel, skip_last_for=missing)  # 22 of 24 = 92 % is complete enough; 21 of 24 = 88 % is not
    order = []
    monkeypatch.setattr(jobs, "ensure_session_bars", lambda ctx, session, **kw: order.append("repair") or {})
    stub_scan(monkeypatch, order)
    jobs.prelim_scan(make_job_context(settings, store, fake_clock))
    assert order == (["repair", "scan:pre"] if repaired else ["scan:pre"])


def test_a_failed_repair_never_stops_the_preliminary_scan(settings, store, fake_clock, panel, monkeypatch):
    populate_store(store, panel, skip_last_for=10)
    monkeypatch.setattr(jobs, "ensure_session_bars", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("provider exploded")))
    order = []
    stub_scan(monkeypatch, order)
    assert jobs.prelim_scan(make_job_context(settings, store, fake_clock)).status == "success"
    assert order == ["scan:pre"]


def test_prelim_scan_is_skipped_on_holidays_without_touching_the_providers(env, monkeypatch):
    monkeypatch.setattr(jobs, "ensure_session_bars", lambda *a, **k: pytest.fail("the exchange is closed"))
    order = []
    stub_scan(monkeypatch, order)
    result = jobs.prelim_scan(ctx_for(env, run_key="2026-11-08"))  # Muhurat Sunday: closed for the regular session
    assert result.status == "skipped" and order == []


# ------------------------------------------------------------------------------------------------ ingest_eod
def test_ingest_eod_after_the_close_uses_the_official_file(env, monkeypatch):
    env.clock.set(datetime(2026, 11, 2, 12, 30, tzinfo=UTC))  # 18:00 IST
    monkeypatch.setattr(jobs, "ingest_bhavcopy", fill_bars("2026-11-02"))
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=3))
    monkeypatch.setattr(jobs, "ingest_fno_day", lambda *a, **k: 120)
    result = jobs.ingest_eod(ctx_for(env))
    assert result.status == "success" and "official bhavcopy ingested" in result.message
    assert env.store.latest_bar_date(env.store.instrument_id("S00")) == "2026-11-02"


def test_ingest_eod_waits_for_the_close_and_flags_vendor_only_data(env, monkeypatch):
    env.clock.set(datetime(2026, 11, 2, 10, 10, tzinfo=UTC))  # 15:40 IST: bhavcopy window not open yet
    assert jobs.ingest_eod(ctx_for(env)).status == "failed"
    env.clock.set(datetime(2026, 11, 2, 12, 30, tzinfo=UTC))
    monkeypatch.setattr(jobs, "ingest_bhavcopy", lambda *a, **k: (0, 0))  # not published yet

    def vendor(store, yahoo, **kw):
        for row in store.list_instruments(kind="equity", in_universe=True):
            store.upsert_bars(int(row["instrument_id"]), [{"trade_date": "2026-11-02", "open": 100, "high": 104, "low": 99,
                                                          "close": 103, "volume": 1e6}], source="yahoo")
        return IngestReport(ok=24)

    monkeypatch.setattr(jobs, "refresh_daily_bars", vendor)
    monkeypatch.setattr(jobs, "ingest_fno_day", lambda *a, **k: 0)
    result = jobs.ingest_eod(ctx_for(env))
    assert result.status == "success" and "provisional" in result.message


# -------------------------------------------------------------------------------------------- refresh_preopen
def test_refresh_preopen_stops_as_soon_as_final_prices_are_complete(env):
    env.clock.set(datetime(2026, 11, 2, 3, 30, 30, tzinfo=UTC))  # 09:00:30 IST
    symbols = env.panel.symbols
    ctx = ctx_for(env)
    api = FakeNseApi(books=[preopen_rows(env.panel, symbols[:4], final=False), preopen_rows(env.panel, symbols)],
                     gift={"price": 24100.0, "prev_close": 24000.0, "change_pct": 0.4, "source_ts": "x", "expiry": "y"})
    ctx._cache["nse_api"] = api
    result = jobs.refresh_preopen(ctx)
    assert result.status == "success" and result.metrics["snapshots"] == 2 and api.calls == 2


def test_refresh_preopen_reports_an_unavailable_book(env):
    env.clock.set(datetime(2026, 11, 2, 3, 30, 30, tzinfo=UTC))
    ctx = ctx_for(env)
    ctx._cache["nse_api"] = FakeNseApi(fail=True)
    result = jobs.refresh_preopen(ctx)
    assert result.status == "failed" and "unavailable" in result.message


def test_refresh_preopen_treats_the_previous_sessions_book_as_unavailable(env):
    env.clock.set(datetime(2026, 11, 2, 3, 30, 30, tzinfo=UTC))
    ctx = ctx_for(env)
    ctx._cache["nse_api"] = FakeNseApi(books=[preopen_rows(env.panel, env.panel.symbols)], book_ts="30-Oct-2026 09:07:48")
    result = jobs.refresh_preopen(ctx)
    assert result.status == "failed" and "unavailable" in result.message
    assert env.store.latest_preopen("2026-11-02") == {}  # Friday's equilibrium prices must not become Monday's opening gap


def test_refresh_preopen_skips_when_nse_is_disabled(env, settings):
    cfg = settings.__class__(**{**settings.__dict__, "nse_enabled": False})
    assert jobs.refresh_preopen(make_job_context(cfg, env.store, env.clock)).status == "skipped"
