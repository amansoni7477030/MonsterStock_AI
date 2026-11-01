"""End-to-end scan tests on synthetic data: quality gate, freeze, idempotency, fallbacks, delivery."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from automation.context import JobContext
from automation.scan import PREOPEN_POLL_SEC, preopen_coverage, run_scan, scan_cutoff, wait_for_preopen
from notifications.service import NotificationService
from tests.seed_data import FakeIntel, FakeNseApi, FakeTelegram, panel_ending, populate_store, preopen_rows

UTC = timezone.utc
PRELIM_TIME = datetime(2026, 11, 2, 3, 15, tzinfo=UTC)  # 08:45 IST
FINAL_TIME = datetime(2026, 11, 2, 3, 39, tzinfo=UTC)  # 09:09 IST


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=420, n_symbols=24, seed=11)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42",
                                "send_prelim_report": True})
    populate_store(store, panel)
    telegram = FakeTelegram()

    def make(run_key: str = "2026-11-02", **kw) -> JobContext:
        ctx = JobContext.create(cfg, store, run_key=run_key, clock=fake_clock, **kw)
        ctx.ensure_calendar()
        ctx._cache["notifier"] = NotificationService(cfg, store, client=telegram, clock=fake_clock)
        ctx._cache["intel"] = FakeIntel()
        ctx._cache["nse_api"] = FakeNseApi(gift={"price": 24100.0, "prev_close": 24000.0, "change_pct": 0.4,
                                                 "source_ts": "02-Nov-2026 08:44:00", "expiry": "26Nov2026"})
        return ctx

    return type("Env", (), {"make": staticmethod(make), "telegram": telegram, "store": store, "clock": fake_clock, "cfg": cfg,
                            "panel": panel})


def test_scan_cutoff_is_bounded(settings):
    day = datetime(2026, 11, 2, tzinfo=UTC).date()
    early = scan_cutoff(settings, day, "pre", datetime(2026, 11, 2, 1, 0, tzinfo=UTC))
    assert early.hour == 3 and early.minute == 15  # nominal 08:45 IST
    late = scan_cutoff(settings, day, "final", datetime(2026, 11, 2, 6, 0, tzinfo=UTC))
    assert (late.hour, late.minute) == (3, 42)  # clipped to the 09:12 IST report cutoff


def test_prelim_scan_freezes_snapshot_and_reports_baseline(env):
    env.clock.set(PRELIM_TIME)
    result = run_scan(env.make(), "pre")
    assert result.status == "success", result.message
    snap = env.store.latest_snapshot("2026-11-02", "pre")
    assert snap is not None and snap["model_version"] == "baseline-pre"
    payload = snap["payload"]
    assert {"report", "table", "features", "baselines", "excluded", "meta"} <= set(payload)
    assert payload["report"]["model_status"] == "baseline"
    assert payload["report"]["uncertainty_note"] == ""  # the heuristic has no learned reference to explain
    assert len(payload["table"]) > 10 and payload["features"]["columns"]
    model_rows = env.store.predictions_for("2026-11-02", stage="pre", baselines=False)
    base_rows = env.store.predictions_for("2026-11-02", stage="pre", baselines=True)
    assert len(model_rows) == 5 and all(r["tier"] in {"WATCH", "NONE"} for r in model_rows)
    assert {r["baseline_name"] for r in base_rows} >= {"atr", "range20", "volume", "random"}
    assert len(env.telegram.sent) >= 1
    assert "PRELIMINARY" in env.telegram.sent[0][1]


def scan_with_gift(env, source_ts):
    """A preliminary scan whose GIFT Nifty quote carries the ISO-UTC stamp the real NSE parser produces."""
    env.clock.set(PRELIM_TIME)
    ctx = env.make()
    ctx._cache["nse_api"] = FakeNseApi(gift={"price": 24100.0, "prev_close": 24000.0, "change_pct": 0.4, "source_ts": source_ts,
                                             "expiry": "26Nov2026"})
    result = run_scan(ctx, "pre")
    assert result.status == "success", result.message
    return env.store.latest_snapshot("2026-11-02", "pre")["payload"]["report"]


def test_the_report_shows_gift_nifty_while_its_quote_is_fresh(env):
    report = scan_with_gift(env, "2026-11-02T03:14:00Z")  # 08:44 IST, one minute before the scan
    assert report["regime"]["gift_nifty"] == "+0.40%"
    assert "GIFT Nifty: +0.40%" in "\n".join(text for _, text in env.telegram.sent)


def test_a_stale_gift_nifty_quote_never_reads_as_todays_hint(env):
    report = scan_with_gift(env, "2026-11-02T01:00:00Z")  # 06:30 IST: over two hours old
    assert "gift_nifty" not in report["regime"]
    assert any("GIFT Nifty" in feed for feed in report["missing_feeds"])


def test_scan_is_idempotent_and_retry_resends_frozen_report(env):
    env.clock.set(PRELIM_TIME)
    env.telegram.fail = True
    first = run_scan(env.make(attempt=1, max_attempts=3, last_attempt=False), "pre")
    assert first.status == "failed" and "delivery failed" in first.message
    snap_id = env.store.latest_snapshot("2026-11-02", "pre")["snapshot_id"]
    count_before = len(env.store.predictions_for("2026-11-02", stage="pre", baselines=None))

    env.telegram.fail = False
    second = run_scan(env.make(attempt=2, max_attempts=3, last_attempt=False), "pre")
    assert second.status == "success"
    assert env.store.latest_snapshot("2026-11-02", "pre")["snapshot_id"] == snap_id
    assert len(env.store.predictions_for("2026-11-02", stage="pre", baselines=None)) == count_before
    delivered = len(env.telegram.sent)  # a report may span several Telegram messages
    assert delivered >= 1
    third = run_scan(env.make(), "pre")  # nothing left to do
    assert third.status == "success" and len(env.telegram.sent) == delivered


def test_dry_run_freezes_and_sends_nothing(env):
    env.clock.set(PRELIM_TIME)
    result = run_scan(env.make(dry_run=True), "pre")
    assert result.status == "success" and "report" in result.metrics
    assert env.store.latest_snapshot("2026-11-02", "pre") is None
    assert env.telegram.sent == []


def test_closed_day_is_skipped(env):
    env.clock.set(PRELIM_TIME)
    result = run_scan(env.make(run_key="2026-11-01"), "pre")  # Sunday
    assert result.status == "skipped" and "closed" in result.message


def test_blocked_by_stale_universe_alerts_and_notifies_on_last_attempt(settings, store, fake_clock, panel):
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42"})
    populate_store(store, panel, skip_last_for=8)  # 8 of 24 symbols lack the last session -> 67% coverage
    telegram = FakeTelegram()
    fake_clock.set(FINAL_TIME)
    ctx = JobContext.create(cfg, store, run_key="2026-11-02", clock=fake_clock, attempt=3, max_attempts=3, last_attempt=True)
    ctx.ensure_calendar()
    ctx._cache.update(notifier=NotificationService(cfg, store, client=telegram, clock=fake_clock), intel=FakeIntel(),
                      nse_api=FakeNseApi(fail=True))
    result = run_scan(ctx, "final")
    assert result.status == "failed" and "data quality gate" in result.message
    assert store.latest_snapshot("2026-11-02", "final") is None
    assert any(a["code"] == "scan_final_failed" for a in store.pending_alerts(50)) or telegram.sent
    assert any("final" in text.lower() for _, text in telegram.sent)  # failure notice reached the operator


def test_final_scan_uses_preopen_book_with_preliminary_fallback(env):
    env.clock.set(FINAL_TIME)
    symbols = env.panel.symbols
    covered = symbols[:18]  # 75% have a pre-open price, 25% fall back to the preliminary model
    env.store.add_preopen_quotes("2026-11-02", preopen_rows(env.panel, covered), "02-Nov-2026 09:08:30")
    env.clock.advance(5)
    ctx = env.make()
    ctx._cache["nse_api"] = FakeNseApi(books=[preopen_rows(env.panel, covered)])
    result = run_scan(ctx, "final")
    assert result.status == "success", result.message
    snap = env.store.latest_snapshot("2026-11-02", "final")
    assert snap["payload"]["meta"]["n_fallback"] == len(symbols) - len(covered)
    assert set(snap["payload"]["open_source"]) == set(covered)
    notes = snap["payload"]["report"]["notes"]
    assert any("no pre-open price" in n for n in notes)
    assert any("FINAL" in text for _, text in env.telegram.sent)
    assert ctx.intel().calls == ["refresh_reactions"]  # pre-open quotes are known: re-check what is already priced in


def test_a_split_ex_date_name_is_scored_without_its_misleading_opening_gap(env):
    env.clock.set(FINAL_TIME)
    rows = preopen_rows(env.panel, env.panel.symbols)
    half = rows[0]["prev_close"] / 2  # a 1:2 split with an ex-date today: the exchange's previous close is already halved
    rows[0].update(prev_close=half, iep=half * 1.004, final_price=half * 1.004)
    name = rows[0]["symbol"]
    env.store.add_preopen_quotes("2026-11-02", rows, "02-Nov-2026 09:08:30")
    env.clock.advance(5)
    ctx = env.make()
    ctx._cache["nse_api"] = FakeNseApi(books=[rows])
    assert run_scan(ctx, "final").status == "success"
    payload = env.store.latest_snapshot("2026-11-02", "final")["payload"]
    assert payload["meta"]["n_fallback"] == 1  # only the re-based name loses its gap
    assert any(name in note and "ex-date today" in note for note in payload["report"]["notes"])
    assert name not in payload["baselines"]["gap"]  # a -50 % "gap" would otherwise top the opening-gap baseline


def test_the_preliminary_scan_does_not_recheck_market_reactions(env):
    env.clock.set(PRELIM_TIME)
    ctx = env.make()
    assert run_scan(ctx, "pre").status == "success"
    assert ctx.intel().calls == []  # the 08:10 news job already did it; nothing new is known before the open


def test_final_scan_ignores_a_pre_open_book_from_the_previous_session(env):
    env.clock.set(FINAL_TIME)
    ctx = env.make()
    ctx._cache["nse_api"] = FakeNseApi(books=[preopen_rows(env.panel, env.panel.symbols)], book_ts="30-Oct-2026 09:07:48")
    result = run_scan(ctx, "final")
    assert result.status == "success", result.message  # gives up on the book after a few empty polls instead of waiting for it
    snap = env.store.latest_snapshot("2026-11-02", "final")
    assert snap["payload"]["open_source"] == {} and snap["payload"]["meta"]["n_fallback"] == len(env.panel.symbols)
    assert any("no pre-open price" in n for n in snap["payload"]["report"]["notes"])
    assert env.store.latest_preopen("2026-11-02") == {}


def test_final_report_after_deadline_is_withheld_but_snapshot_is_frozen(env):
    env.clock.set(datetime(2026, 11, 2, 3, 52, tzinfo=UTC))  # 09:22 IST, past the 09:14:30 deadline
    result = run_scan(env.make(), "final")
    assert result.status == "success" and "withheld" in result.message
    snap = env.store.latest_snapshot("2026-11-02", "final")
    assert snap["payload"]["meta"]["late"] is True
    assert env.telegram.sent == []


def test_preopen_wait_stops_when_final_prices_arrive(env):
    env.clock.set(datetime(2026, 11, 2, 3, 38, 45, tzinfo=UTC))  # 09:08:45
    symbols = env.panel.symbols
    partial = preopen_rows(env.panel, symbols[:5])
    full = preopen_rows(env.panel, symbols)
    ctx = env.make()
    api = FakeNseApi(books=[partial, full])
    ctx._cache["nse_api"] = api
    start = env.clock.now()
    coverage = wait_for_preopen(ctx, symbols)
    assert coverage >= 0.9 and api.calls == 2
    assert (env.clock.now() - start).total_seconds() == PREOPEN_POLL_SEC


def test_preopen_wait_gives_up_after_repeated_failures(env):
    env.clock.set(datetime(2026, 11, 2, 3, 38, 45, tzinfo=UTC))
    ctx = env.make()
    api = FakeNseApi(fail=True)
    ctx._cache["nse_api"] = api
    assert wait_for_preopen(ctx, env.panel.symbols) == 0.0
    assert api.calls == 3


def test_preopen_coverage_counts_final_and_indicative(env):
    env.clock.set(FINAL_TIME)
    rows = preopen_rows(env.panel, env.panel.symbols[:6], final=True) + preopen_rows(env.panel, env.panel.symbols[6:12], final=False)
    env.store.add_preopen_quotes("2026-11-02", rows, None)
    usable, final = preopen_coverage(env.store, "2026-11-02", env.panel.symbols, env.clock.now())
    assert usable == pytest.approx(0.5) and final == pytest.approx(0.25)
