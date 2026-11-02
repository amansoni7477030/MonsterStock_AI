"""End-to-end tests of the closing-hour scan on synthetic data: NSE's live quotes over the stored history, the frozen report, its
delivery and retries, and every way the scan must refuse to guess."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from automation import ideas as scan
from automation import jobs
from automation.context import JobContext
from notifications.service import NotificationService
from tests.ideas_data import catalyst
from tests.seed_data import (
    LIVE_STAMP,
    FakeIntel,
    FakeNseApi,
    FakeTelegram,
    live_rows,
    panel_ending,
    populate_store,
)

UTC = timezone.utc
DAY = "2026-11-02"
SCAN_TIME = datetime(2026, 11, 2, 9, 1, tzinfo=UTC)  # 14:31 IST, inside the 14:30-15:05 window
LATE_TIME = datetime(2026, 11, 2, 9, 40, tzinfo=UTC)  # 15:10 IST, after the 15:05 report deadline
LATE_STAMP = "2026-11-02T09:39:00Z"
EARLY = {"attempt": 1, "max_attempts": 5, "last_attempt": False}  # not the last try: a failure then sends no notice yet


def index_quotes(panel, change=0.4):
    prev = float(panel.nifty["close"].iloc[-1])
    return {
        "NIFTY 50": {"name": "NIFTY 50", "last": prev * (1 + change / 100), "prev_close": prev, "change_pct": change, "row_ts": LIVE_STAMP},
        "INDIA VIX": {"name": "INDIA VIX", "last": 14.0, "prev_close": 14.2, "change_pct": -1.4, "row_ts": LIVE_STAMP},
    }


class NewsIntel(FakeIntel):
    """An intel with scripted catalysts that remembers which symbols it was asked about."""

    def __init__(self, found=None, note=None):
        super().__init__()
        self.found, self.note, self.asked = found or {}, note, []

    def catalysts(self, symbols, *_, **__):
        self.asked.append(list(symbols))
        return {symbol: items for symbol, items in self.found.items() if symbol in symbols}

    def coverage_note(self):
        return self.note


class PickyTelegram(FakeTelegram):
    """Delivers everything except the ideas report itself (to see what the operator is told when only that fails)."""

    def send_message(self, chat_id, text, **kw):
        if "Closing-hour trade ideas" in text:
            from app.errors import DataUnavailable

            raise DataUnavailable("telegram", "simulated outage", retryable=True)
        return super().send_message(chat_id, text, **kw)


def build_env(settings, store, clock, panel, telegram=None, **over):
    """A store holding ``panel`` as the history, a Telegram double, and a ``make`` that builds job contexts of the closing scan with
    a scripted NSE: by default today is quiet except for ``star``, a strong close on heavy volume."""
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42", **over})
    populate_store(store, panel)
    clock.set(SCAN_TIME)
    telegram = telegram or FakeTelegram()
    star = panel.symbols[0]
    default = live_rows(panel, moves={star: 2.6}, strength={star: 0.95}, volume={star: 1.8})

    def make(*, live=None, indices=None, run_key=DAY, intel=None, nse_fail=False, live_ts=LIVE_STAMP, **kw) -> JobContext:
        ctx = JobContext.create(cfg, store, run_key=run_key, clock=clock, **kw)
        ctx.ensure_calendar()
        api = FakeNseApi(live=default if live is None else live, indices=index_quotes(panel) if indices is None else indices,
                         fail=nse_fail, live_ts=live_ts)
        ctx._cache.update(notifier=NotificationService(cfg, store, client=telegram, clock=clock), intel=intel or FakeIntel(),
                          nse_api=api)
        return ctx

    return SimpleNamespace(make=make, telegram=telegram, store=store, clock=clock, cfg=cfg, panel=panel, star=star, live=default)


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=420, n_symbols=24, seed=11)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    return build_env(settings, store, fake_clock, panel)


def frozen(store, day=DAY):
    payload = store.kv_get_json(scan.run_key(day))
    return payload["report"] if payload else None


def said(env) -> str:
    return "\n".join(text for _, text in env.telegram.sent)


def alerts(env, code):
    return [a for a in env.store.pending_alerts(100) if a["code"] == code]


def star_btst(report, env):
    return [i for i in report["ideas"] if i["symbol"] == env.star and i["horizon"] == "BTST"]


# --------------------------------------------------------------------------------------- the happy path
def test_the_scan_freezes_the_report_and_sends_it(env):
    result = scan.run_closing_scan(env.make())
    assert result.status == "success", result.message
    report = frozen(env.store)
    assert report["trade_date"] == DAY and report["late"] is False and report["universe_size"] == 24
    assert report["entry_by"] == env.cfg.ideas_entry_by_time[:5] and report["market"]["regime"] == "bullish"
    (star,) = star_btst(report, env)
    assert star["side"] == "LONG" and star["stop"] < star["entry"] < star["target1"] < star["target2"]
    assert star["score"] >= env.cfg.ideas_min_score
    text = said(env)
    assert "Closing-hour trade ideas" in text and env.star in text and "NOT a probability" in text
    assert result.metrics["ideas"] == len(report["ideas"]) and result.metrics["late"] is False


def test_every_idea_is_a_plan_with_a_stop_and_costs(env):
    scan.run_closing_scan(env.make())
    ideas = frozen(env.store)["ideas"]
    assert ideas
    for idea in ideas:
        assert idea["horizon"] in {"BTST", "SWING", "WATCH"} and idea["side"] in {"LONG", "SHORT"}
        if idea["horizon"] == "WATCH":
            continue
        assert idea["risk_pct"] <= env.cfg.ideas_max_risk_pct and idea["est_cost_pct"] > 0
        assert (idea["stop"] < idea["entry"]) == (idea["side"] == "LONG")


def test_bearish_ideas_are_futures_and_options_names_only(env):
    scan.run_closing_scan(env.make())
    shorts = [i for i in frozen(env.store)["ideas"] if i["side"] == "SHORT" and i["horizon"] in ("BTST", "SWING")]
    assert all(i["is_fno"] for i in shorts)


# -------------------------------------------------------------------------- freeze, retry, dry run, force
def test_a_retry_resends_the_frozen_report_without_scanning_again(env):
    env.telegram.fail = True
    first = scan.run_closing_scan(env.make(attempt=1, max_attempts=5, last_attempt=False))
    assert first.status == "failed" and "delivery failed" in first.message
    kept = env.store.kv_get_json(scan.run_key(DAY))
    assert kept is not None and env.telegram.sent == []

    env.telegram.fail = False
    retry = env.make(nse_fail=True, attempt=2, max_attempts=5)  # NSE is down now: a new scan could not succeed
    second = scan.run_closing_scan(retry)
    assert second.status == "success", second.message
    assert env.store.kv_get_json(scan.run_key(DAY)) == kept and retry.nse_api().live_calls == []
    delivered = len(env.telegram.sent)
    assert delivered >= 1
    third = scan.run_closing_scan(env.make())  # nothing left to do
    assert third.status == "success" and len(env.telegram.sent) == delivered


def test_the_last_failed_delivery_sends_the_operator_a_notice(settings, store, fake_clock, panel):
    env = build_env(settings, store, fake_clock, panel, telegram=PickyTelegram())
    early = scan.run_closing_scan(env.make(attempt=1, max_attempts=5, last_attempt=False))
    assert early.status == "failed" and "delivery failed" in early.message and env.telegram.sent == []
    last = scan.run_closing_scan(env.make(attempt=5, max_attempts=5, last_attempt=True))
    assert last.status == "failed"
    assert len(env.telegram.sent) == 1 and "REPORT WITHHELD" in said(env) and "could not be delivered" in said(env)


def test_a_dry_run_freezes_and_sends_nothing(env):
    result = scan.run_closing_scan(env.make(dry_run=True))
    assert result.status == "success" and result.metrics["report"]["trade_date"] == DAY
    assert env.store.kv_get_json(scan.run_key(DAY)) is None and env.telegram.sent == []


def test_a_forced_scan_after_the_report_was_delivered_leaves_the_record_alone(env):
    assert scan.run_closing_scan(env.make()).status == "success"
    kept, delivered = env.store.kv_get_json(scan.run_key(DAY)), len(env.telegram.sent)
    other = live_rows(env.panel, moves={env.star: -1.0})

    again = env.make(live=other, force=True)
    result = scan.run_closing_scan(again)
    assert result.status == "skipped" and "stays the record" in result.message
    assert env.store.kv_get_json(scan.run_key(DAY)) == kept and len(env.telegram.sent) == delivered
    assert again.nse_api().live_calls == []  # NSE is not even asked

    preview = scan.run_closing_scan(env.make(live=other, force=True, dry_run=True))  # a fresh look is still possible
    assert preview.status == "success" and not star_btst(preview.metrics["report"], env)
    assert env.store.kv_get_json(scan.run_key(DAY)) == kept


def test_a_forced_scan_replaces_a_frozen_report_that_nobody_received(env):
    env.telegram.fail = True
    assert scan.run_closing_scan(env.make()).status == "failed"
    before = frozen(env.store)
    assert star_btst(before, env) and env.telegram.sent == []

    env.telegram.fail = False
    result = scan.run_closing_scan(env.make(live=live_rows(env.panel, moves={env.star: -1.0}), force=True))
    assert result.status == "success", result.message
    after = frozen(env.store)
    assert after != before and not star_btst(after, env)
    assert env.telegram.sent  # what is now the record is what was sent


# ------------------------------------------------------------------------------------ refusing to guess
def test_an_nse_failure_alerts_and_only_the_last_attempt_sends_a_failure_notice(env):
    early = scan.run_closing_scan(env.make(nse_fail=True, attempt=1, max_attempts=5, last_attempt=False))
    assert early.status == "failed" and "NSE live quotes unavailable" in early.message
    assert alerts(env, "ideas_scan_failed") and env.telegram.sent == []
    assert env.store.kv_get_json(scan.run_key(DAY)) is None

    last = scan.run_closing_scan(env.make(nse_fail=True, attempt=5, max_attempts=5, last_attempt=True))
    assert last.status == "failed"
    notice = said(env)
    assert "REPORT WITHHELD" in notice and "CLOSING-HOUR IDEAS" in notice and "closing_scan" in notice


def test_rows_stamped_on_an_earlier_session_are_not_todays_prices(env):
    stamp = "2026-10-30T09:00:00Z"
    result = scan.run_closing_scan(env.make(live=live_rows(env.panel, stamp=stamp), live_ts=stamp, **EARLY))
    assert result.status == "failed" and "too few live quotes" in result.message and "earlier session" in result.message
    assert env.store.kv_get_json(scan.run_key(DAY)) is None and env.telegram.sent == []


def test_too_few_live_quotes_is_a_failure_not_a_thin_report(env):
    result = scan.run_closing_scan(env.make(live=env.live[:10], **EARLY))
    assert result.status == "failed" and "too few live quotes: 10 of 24" in result.message
    assert env.store.kv_get_json(scan.run_key(DAY)) is None and env.telegram.sent == []


def test_an_incomplete_live_row_is_counted_and_never_used(env):
    rows = [{**r, "volume": None} if r["symbol"] == env.star else r for r in env.live]
    result = scan.run_closing_scan(env.make(live=rows))
    assert result.status == "success", result.message
    report = frozen(env.store)
    assert not any(i["symbol"] == env.star for i in report["ideas"])
    assert any("1 incomplete live row" in note for note in report["notes"])


def test_a_name_whose_previous_close_differs_from_the_history_is_left_out(env):
    rows = [{**r, "prev_close": r["prev_close"] / 2} if r["symbol"] == env.star else r for r in env.live]
    result = scan.run_closing_scan(env.make(live=rows))
    assert result.status == "success", result.message
    report = frozen(env.store)
    assert not any(i["symbol"] == env.star for i in report["ideas"])
    assert any("1 name(s) left out" in n and "price basis differs" in n for n in report["notes"])


def test_a_missing_regular_session_in_the_history_stops_the_scan(settings, store, fake_clock):
    old = panel_ending("2026-10-29", n_dates=420, n_symbols=24, seed=11)  # no bar of Friday 30 October, a regular session
    env = build_env(settings, store, fake_clock, old)
    result = scan.run_closing_scan(env.make(**EARLY))
    assert result.status == "failed" and "2026-10-30 was a regular session" in result.message and "collect_india" in result.message
    assert frozen(env.store) is None and env.telegram.sent == []


def test_a_special_session_in_the_gap_is_not_a_missing_session(settings, store, fake_clock):
    (settings.data_dir / "calendar_overrides.json").write_text(json.dumps(
        {"sessions": [{"date": "2026-10-30", "open": "17:45", "close": "18:45", "description": "Muhurat trading"}]}), encoding="utf-8")
    old = panel_ending("2026-10-29", n_dates=420, n_symbols=24, seed=11)
    env = build_env(settings, store, fake_clock, old)
    result = scan.run_closing_scan(env.make())
    assert result.status == "success", result.message
    assert frozen(env.store) is not None


def test_a_history_without_the_previous_closes_stops_the_scan(settings, store, fake_clock, panel):
    env = build_env(settings, store, fake_clock, panel)
    store.conn.execute("DELETE FROM bars_daily WHERE trade_date = ? AND instrument_id IN "
                       "(SELECT instrument_id FROM instruments WHERE kind = 'equity' AND symbol != ?)", ("2026-10-30", env.star))
    result = scan.run_closing_scan(env.make(**EARLY))
    assert result.status == "failed" and "no stored bar of" in result.message and "collect_india" in result.message
    assert frozen(env.store) is None


def test_a_closed_day_is_skipped(env):
    result = scan.run_closing_scan(env.make(run_key="2026-11-01"))  # a Sunday
    assert result.status == "skipped" and "closed" in result.message
    assert env.telegram.sent == [] and env.store.kv_get_json(scan.run_key("2026-11-01")) is None


def test_a_disabled_feature_is_skipped(settings, store, fake_clock, panel):
    env = build_env(settings, store, fake_clock, panel, ideas_enabled=False)
    result = scan.run_closing_scan(env.make())
    assert result.status == "skipped" and "IDEAS_ENABLED" in result.message
    assert env.telegram.sent == [] and frozen(env.store) is None


# ------------------------------------------------------------------------------------------ the deadline
def test_a_scan_after_the_deadline_is_frozen_late_and_withheld_unless_forced(env):
    env.clock.set(LATE_TIME)
    late_live = live_rows(env.panel, moves={env.star: 2.6}, strength={env.star: 0.95}, volume={env.star: 1.8}, stamp=LATE_STAMP)
    result = scan.run_closing_scan(env.make(live=late_live, live_ts=LATE_STAMP))
    assert result.status == "success" and "withheld" in result.message and env.telegram.sent == []
    assert frozen(env.store)["late"] is True and alerts(env, "ideas_deadline_missed")

    retry = scan.run_closing_scan(env.make(live=late_live, live_ts=LATE_STAMP))  # a retry does not get past the deadline either
    assert retry.status == "success" and env.telegram.sent == []

    forced = scan.run_closing_scan(env.make(live=late_live, live_ts=LATE_STAMP, force=True))
    assert forced.status == "success", forced.message
    assert "built after the report deadline" in said(env) and frozen(env.store)["late"] is True


def test_the_prices_age_is_stated_when_the_last_quotes_are_old(env):
    env.clock.set(datetime(2026, 11, 2, 9, 20, tzinfo=UTC))  # 14:50 IST, the newest quote is 20 minutes old
    scan.run_closing_scan(env.make())
    assert any("20 minutes old" in note for note in frozen(env.store)["notes"])


# ----------------------------------------------------------------------------------------- the record
def past_idea(symbol, horizon="BTST", side="LONG"):
    return {"symbol": symbol, "horizon": horizon, "side": side, "entry": 100.0, "stop": 98.0, "target1": 102.0, "target2": 104.0,
            "est_cost_pct": 0.3}


def freeze_day(store, day, ideas, *, late=False):
    outcome = {"complete": True, "official": True, "status": "target1", "net_pct": 1.7, "ret_pct": 2.0, "t1_hit": True, "t2_hit": False,
               "gap_pct": 0.2, "sessions": 1}
    store.kv_set_json(scan.run_key(day), {"report": {"trade_date": day, "ideas": ideas, "late": late}})
    store.kv_set_json(scan.outcome_key(day), {"ideas": {f"{i['horizon']}:{i['side']}:{i['symbol']}": outcome for i in ideas}})


def test_the_live_record_of_earlier_days_stands_behind_the_report_and_late_reports_do_not_count(env):
    freeze_day(env.store, "2026-10-29", [past_idea("S01"), past_idea("S02")])
    freeze_day(env.store, "2026-10-28", [past_idea("S03")], late=True)
    assert scan.run_closing_scan(env.make()).status == "success"
    record = frozen(env.store)["record"]
    assert record["BTST"]["n"] == 2 and record["BTST"]["days"] == 1 and record["BTST"]["validated"] is False
    assert "Live record: 2 judged over 1 days" in said(env) and "not validated" in said(env)


def test_todays_own_report_is_not_part_of_its_record(env):
    freeze_day(env.store, DAY, [past_idea("S01")])
    assert scan.live_record(env.store, date(2026, 11, 2)) == {}


# ---------------------------------------------------------------------------------------------- news
def test_confirmed_bad_news_vetoes_the_idea_and_reduced_news_coverage_is_stated(env):
    bad = catalyst("Demo regulator halts output", polarity=-0.9, severity=0.9, confidence=0.9)
    intel = NewsIntel({env.star: [bad]}, note="RSS feeds degraded")
    result = scan.run_closing_scan(env.make(intel=intel))
    assert result.status == "success", result.message
    assert intel.asked and env.star in intel.asked[0]
    report = frozen(env.store)
    assert not star_btst(report, env)
    assert "news coverage reduced: RSS feeds degraded" in report["notes"]


def test_confirmed_good_news_is_shown_with_the_idea(env):
    good = catalyst("Demo order win", polarity=0.7, severity=0.7, confidence=0.8)
    result = scan.run_closing_scan(env.make(intel=NewsIntel({env.star: [good]})))
    assert result.status == "success", result.message
    (star,) = star_btst(frozen(env.store), env)
    assert any("news in its favour" in reason for reason in star["reasons"]) and star["catalysts"]
    assert "Demo order win" in said(env)


# ----------------------------------------------------------------------------------------------- market
def test_missing_index_quotes_are_stated_and_the_scan_goes_on(env):
    result = scan.run_closing_scan(env.make(indices={}))
    assert result.status == "success", result.message
    report = frozen(env.store)
    assert "nifty_chg" not in report["market"] and any("NSE index quotes unavailable" in n for n in report["notes"])
    assert "Nifty n/a" in said(env)


def test_a_bearish_day_withholds_the_opposite_side_of_the_overnight_trade(env):
    quotes = index_quotes(env.panel, change=-2.4)
    down = live_rows(env.panel, moves={s: -2.4 for s in env.panel.symbols}, default_move=-2.4)
    result = scan.run_closing_scan(env.make(live=down, indices=quotes))
    assert result.status == "success", result.message
    report = frozen(env.store)
    assert report["market"]["regime"] == "bearish"
    assert not [i for i in report["ideas"] if i["horizon"] == "BTST" and i["side"] == "LONG"]
    assert any("overnight" in n.lower() for n in report["notes"])


# --------------------------------------------------------------------------------------- the scheduler
def test_the_scheduler_job_delegates_to_the_scan_and_skips_a_closed_day(env):
    assert jobs.FUNCTIONS["closing_scan"](env.make()).status == "success" and env.telegram.sent
    closed = jobs.FUNCTIONS["closing_scan"](env.make(run_key="2026-11-01"))
    assert closed.status == "skipped"
