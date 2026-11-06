"""End-to-end tests of the 09:30 opening read on synthetic data: NSE's live quotes over the stored history, the morning picks read again, the
frozen report and its delivery, the timing window, every way the scan must refuse to guess, and the evening judging with its live record."""

from __future__ import annotations

import html
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from app.errors import DataUnavailable
from automation import jobs
from automation import opening as scan
from automation.context import JobContext
from notifications.opening import LATE_BANNER
from notifications.service import NotificationService
from ranking import opening as op
from ranking.types import Candidate, ReportContext
from tests.ideas_data import catalyst
from tests.seed_data import FakeIntel, FakeNseApi, FakeTelegram, panel_ending, populate_store

UTC = timezone.utc
DAY = "2026-11-02"
OPEN_STAMP = "2026-11-02T04:00:00Z"  # 09:30:00 IST: exactly fifteen minutes after the open
SCAN_TIME = datetime(2026, 11, 2, 4, 0, 30, tzinfo=UTC)  # 09:30:30 IST
EARLY_STAMP = "2026-11-02T03:55:00Z"  # 09:25 IST: ten minutes into the session
LATE_STAMP = "2026-11-02T04:10:00Z"  # 09:40 IST: twenty-five minutes in
LATE_NOW = datetime(2026, 11, 2, 4, 10, 30, tzinfo=UTC)
TOO_LATE_STAMP = "2026-11-02T04:30:00Z"  # 10:00 IST: forty-five minutes in
TOO_LATE_NOW = datetime(2026, 11, 2, 4, 30, 30, tzinfo=UTC)
PAST_DEADLINE = datetime(2026, 11, 2, 4, 16, 0, tzinfo=UTC)  # 09:46 IST, after the 09:45 report deadline
NOT_LAST = {"attempt": 1, "max_attempts": 5, "last_attempt": False}  # not the last try: a failure then sends no notice yet
LAST = {"attempt": 5, "max_attempts": 5, "last_attempt": True}
STRICT = dict(opening_min_x=0.6, max_cost_to_range_ratio=0.5)  # only the star is a mover; the cost gate never decides these tests
STAR_X = 0.95  # the star's opening range as a multiple of its normal day: an EXPLOSIVE opening whose range (2.6 %) has not yet passed 3 %


def opening_rows(panel, *, x=None, gaps=None, last_at=None, stamp=OPEN_STAMP, default_x=0.2, default_gap=0.1, base_prev=None):
    """The live rows of the first fifteen minutes: every name opened ``default_gap`` % from its previous close, its range so far is ``x``
    times its own 20-session mean daily range (``default_x``: a quiet opening), and the last price sits at ``last_at`` of that range."""
    hist = op.history_stats(panel.o, panel.h, panel.low, panel.c, panel.v)
    rows = []
    for symbol in panel.symbols:
        prev = float((base_prev or {}).get(symbol, panel.c[symbol].iloc[-1]))
        opened = prev * (1 + (gaps or {}).get(symbol, default_gap) / 100.0)
        width = (x or {}).get(symbol, default_x) * float(hist.loc[symbol, "rm20"])
        high, low = opened * (1 + width / 200.0), opened * (1 - width / 200.0)
        last = low + (last_at or {}).get(symbol, 0.5) * (high - low)
        rows.append({"symbol": symbol, "open": opened, "high": high, "low": low, "last": last, "prev_close": prev,
                     "change_pct": (last / prev - 1) * 100, "volume": 1e6, "turnover": 1e6 * last, "year_high": None, "year_low": None,
                     "row_ts": stamp})
    return rows


def index_quotes(panel, change=0.3, stamp=OPEN_STAMP):
    prev = float(panel.nifty["close"].iloc[-1])
    return {
        "NIFTY 50": {"name": "NIFTY 50", "last": prev * (1 + change / 100), "prev_close": prev, "change_pct": change, "open": prev,
                     "high": prev * 1.004, "low": prev * 0.999, "row_ts": stamp},
        "INDIA VIX": {"name": "INDIA VIX", "last": 14.0, "prev_close": 14.2, "change_pct": -1.4, "row_ts": stamp},
    }


class NewsIntel(FakeIntel):
    """An intel with scripted catalysts that remembers which symbols it was asked about."""

    def __init__(self, found=None, note=None, fail=False):
        super().__init__()
        self.found, self.note, self.fail, self.asked = found or {}, note, fail, []

    def catalysts(self, symbols, *_, **__):
        self.asked.append(list(symbols))
        if self.fail:
            raise RuntimeError("news store locked")
        return {symbol: items for symbol, items in self.found.items() if symbol in symbols}

    def coverage_note(self):
        return self.note


class FakeYahoo:
    """Yahoo's five-minute feed: ``series`` maps a Yahoo symbol to the bars it answers (an Exception value makes it fail)."""

    def __init__(self, series=None):
        self.series, self.calls = series or {}, []

    def fetch_intraday(self, symbol, **_):
        self.calls.append(symbol)
        got = self.series.get(symbol)
        if isinstance(got, Exception):
            raise got
        if got is None:
            raise DataUnavailable("yahoo", f"symbol not found: {symbol}", retryable=False)
        return got


def build_env(settings, store, clock, panel, telegram=None, **over):
    """A store holding ``panel`` as the history, a Telegram double, and a ``make`` that builds job contexts of the opening scan with a
    scripted NSE: by default every name has a quiet opening except ``star``, whose first fifteen minutes are ``STAR_X`` times its normal day."""
    cfg = settings.__class__(**{**settings.__dict__, "telegram_bot_token": "123:TEST", "telegram_chat_id": "42", **{**STRICT, **over}})
    populate_store(store, panel)
    clock.set(SCAN_TIME)
    telegram = telegram or FakeTelegram()
    star, pick1, pick2 = panel.symbols[0], panel.symbols[5], panel.symbols[3]
    default = opening_rows(panel, x={star: STAR_X})

    def make(*, live=None, indices=None, run_key=DAY, intel=None, nse_fail=False, live_ts=OPEN_STAMP, yahoo=None, **kw) -> JobContext:
        ctx = JobContext.create(cfg, store, run_key=run_key, clock=clock, **kw)
        ctx.ensure_calendar()
        api = FakeNseApi(live=default if live is None else live, indices=index_quotes(panel) if indices is None else indices,
                         fail=nse_fail, live_ts=live_ts, clock=clock)
        ctx._cache.update(notifier=NotificationService(cfg, store, client=telegram, clock=clock), intel=intel or FakeIntel(), nse_api=api,
                          yahoo=yahoo or FakeYahoo())
        return ctx

    return SimpleNamespace(make=make, telegram=telegram, store=store, clock=clock, cfg=cfg, panel=panel, star=star, pick1=pick1, pick2=pick2,
                           live=default)


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=420, n_symbols=24, seed=11)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    return build_env(settings, store, fake_clock, panel)


def seed_morning(store, symbols, *, stage="final", day=DAY):
    """Freeze a morning snapshot whose report lists ``symbols`` as its candidates, in that order."""
    candidates = [Candidate(symbol=s, rank=i, tier="STRONG" if i == 1 else "MODERATE", p={"hit_3": round(0.7 - 0.05 * i, 2), "hit_2": 0.8},
                            primary_threshold=3.0, range={"q10": 1.0, "q50": 2.5, "q90": 5.0}, name=f"{s} Limited", sector="Banks")
                  for i, s in enumerate(symbols, 1)]
    report = ReportContext(trade_date=day, stage=stage, snapshot_ist=f"{day} 09:12:00 IST", model_version="m1", model_status="shadow",
                           candidates=candidates)
    store.insert_snapshot(trade_date=day, stage=stage, cutoff_ts=f"{day}T03:42:00Z", feature_version="fv-test", model_version="m1",
                          code_version="test", payload={"report": report.to_dict(), "table": {}}, data_quality={})


def frozen(store, day=DAY):
    payload = store.kv_get_json(scan.run_key(day))
    return payload["report"] if payload else None


def said(env) -> str:
    return html.unescape("\n".join(text for _, text in env.telegram.sent))


def alerts(env, code):
    return [a for a in env.store.pending_alerts(100) if a["code"] == code]


# --------------------------------------------------------------------------------------- the happy path
def test_the_scan_freezes_the_report_and_sends_it(env):
    seed_morning(env.store, [env.pick1, env.pick2])
    result = scan.run_opening_scan(env.make())
    assert result.status == "success", result.message
    report = frozen(env.store)
    assert report["trade_date"] == DAY and report["late"] is False and report["elapsed_min"] == 15.0 and report["pre_stage"] == "final"
    assert report["universe_size"] == 24 and report["n_eligible"] == 24 and report["mode"] == "shadow" and report["model"]["id"] == op.MODEL_ID
    assert [p["symbol"] for p in report["picks"]] == [env.pick1, env.pick2] and [m["symbol"] for m in report["movers"]] == [env.star]
    star = report["movers"][0]
    assert star["kind"] == "MOVER" and star["strength"] == "EXPLOSIVE" and star["x"] == pytest.approx(STAR_X, abs=0.01)
    assert 0.0 < star["metrics"]["or_raw_pct"] < 3.0 and 0.5 < star["p3"] < 1.0  # not past 3 % yet: a chance is estimated, not stated as certain
    assert star["levels"]["trigger_long"] > star["or_high"] > star["or_low"] > star["levels"]["trigger_short"]
    assert star["day_range"][0] < star["day_range"][1] < star["day_range"][2] and star["ahead"][0] < star["ahead"][2]
    first = report["picks"][0]
    assert first["pre"] == {"rank": 1, "tier": "STRONG", "p": 0.65, "threshold": 3.0, "stage": "final"} and first["strength"] == "QUIET"
    market = report["market"]
    assert market["day_type"] == "calm" and market["advancers"] + market["decliners"] == 24 and market["nifty_chg"] == 0.3 and market["vix"] == 14.0
    assert len(market["sectors_up"]) == 3 and market["nifty_x"] is not None
    text = said(env)
    assert "Opening read" in text and env.star in text and env.pick1 in text and "How to read the opening strength" in text
    assert "SHADOW MODE" in text and "Live record:</b> none judged yet" in text and "not investment advice" in text


def test_the_scored_table_of_all_eligible_names_is_frozen_beside_the_report(env):
    assert scan.run_opening_scan(env.make()).status == "success"
    state = env.store.kv_get_json(scan.state_key(DAY))
    assert state["cols"][0] == "symbol" and len(state["rows"]) == 24
    row = dict(zip(state["cols"], next(r for r in state["rows"] if r[0] == env.star), strict=True))
    assert row["p3"] > 0.5 and row["day_q10"] < row["day_q50"] < row["day_q90"] and row["or_high"] > row["or_low"]
    assert row["or_raw_pct"] == pytest.approx(row["or_pct"], abs=1e-3)  # frozen so the record can tell a name already past 3 % from the others


def test_a_retry_a_late_run_and_a_forced_run_send_the_frozen_report_and_never_scan_again(env):
    assert scan.run_opening_scan(env.make()).status == "success"
    first = frozen(env.store)
    sent = len(env.telegram.sent)
    again = env.make(nse_fail=True, live=[], **NOT_LAST)
    result = scan.run_opening_scan(again)
    assert result.status == "success" and "duplicate" in result.message and again.nse_api().live_calls == []
    forced = scan.run_opening_scan(env.make(nse_fail=True, force=True))
    assert forced.status == "success" and len(env.telegram.sent) == sent and frozen(env.store) == first


def test_a_dry_run_previews_without_freezing_or_sending(env):
    seed_morning(env.store, [env.pick1])
    result = scan.run_opening_scan(env.make(dry_run=True))
    assert result.status == "success" and "dry run" in result.message and result.metrics["report"]["picks"][0]["symbol"] == env.pick1
    assert frozen(env.store) is None and env.store.kv_get_json(scan.state_key(DAY)) is None and env.telegram.sent == []


# ----------------------------------------------------------------------------------- the morning picks
def test_the_picks_are_read_in_their_morning_order_and_a_missing_quote_is_stated(env):
    seed_morning(env.store, [env.pick2, env.pick1, "GONE"])
    assert scan.run_opening_scan(env.make()).status == "success"
    report = frozen(env.store)
    assert [p["symbol"] for p in report["picks"]] == [env.pick2, env.pick1] and [p["pre"]["rank"] for p in report["picks"]] == [1, 2]
    assert any("no usable live quote for the morning pick(s): GONE" in n for n in report["notes"])
    assert report["picks"][0]["name"] == f"{env.pick2} Limited" and report["picks"][0]["is_fno"] in (True, False)


def test_without_a_morning_report_only_the_movers_are_shown_and_it_is_said(env):
    assert scan.run_opening_scan(env.make()).status == "success"
    report = frozen(env.store)
    assert report["picks"] == [] and report["pre_stage"] == "" and any("no 09:12 report was stored for today" in n for n in report["notes"])
    assert "09:12 report was stored" in said(env) and "Your " not in said(env).split("New opening movers")[0].split("Reference levels")[-1]


def test_the_silent_preliminary_scan_stands_in_only_when_there_is_no_final_one(env):
    seed_morning(env.store, [env.pick1], stage="pre")
    assert scan.run_opening_scan(env.make()).status == "success"
    assert frozen(env.store)["pre_stage"] == "pre" and frozen(env.store)["picks"][0]["pre"]["stage"] == "pre"


def test_a_final_scan_that_found_no_setup_is_not_replaced_by_the_superseded_preliminary_names(env):
    seed_morning(env.store, [env.pick1], stage="pre")
    seed_morning(env.store, [], stage="final")
    assert scan.run_opening_scan(env.make()).status == "success"
    report = frozen(env.store)
    assert report["picks"] == [] and report["pre_stage"] == "final" and any("final scan listed no candidates" in n for n in report["notes"])


def test_an_unreadable_morning_snapshot_is_skipped_not_fatal(env):
    env.store.insert_snapshot(trade_date=DAY, stage="final", cutoff_ts=f"{DAY}T03:42:00Z", feature_version="fv-test", model_version="m1",
                              code_version="test", payload={"report": {"candidates": [{"symbol": "X", "rank": "first"}], "trade_date": 5}},
                              data_quality={})
    assert scan.run_opening_scan(env.make()).status == "success"


# ------------------------------------------------------------------------------------------ the timing
def test_before_the_third_candle_has_closed_the_scan_waits_and_the_last_try_says_so(env):
    early = opening_rows(env.panel, stamp=EARLY_STAMP)
    result = scan.run_opening_scan(env.make(live=early, live_ts=EARLY_STAMP, **NOT_LAST))
    assert result.status == "failed" and "not formed yet" in result.message and "10 minutes" in result.message
    assert frozen(env.store) is None and env.telegram.sent == [] and alerts(env, "opening_scan_failed")
    last = scan.run_opening_scan(env.make(live=early, live_ts=EARLY_STAMP, **LAST))
    assert last.status == "failed" and "09:30 OPENING READ" in said(env)


def test_a_feed_that_lags_a_little_still_shows_the_formed_range(env):
    stamp = "2026-11-02T03:59:10Z"  # 14.2 minutes into the session
    assert scan.run_opening_scan(env.make(live=opening_rows(env.panel, stamp=stamp), live_ts=stamp)).status == "success"
    assert frozen(env.store)["elapsed_min"] == pytest.approx(14.2, abs=0.05)


def test_a_scan_that_runs_late_is_scaled_back_to_fifteen_minutes_and_says_so(settings, store, fake_clock, panel):
    on_time = build_env(settings, store, fake_clock, panel)
    seed_morning(store, [on_time.star])  # a pick is listed whatever its chance, so the same name can be compared in both scans
    assert scan.run_opening_scan(on_time.make()).status == "success"
    base = frozen(store)["picks"][0]
    store.kv_delete(scan.run_key(DAY))
    fake_clock.set(LATE_NOW)
    result = scan.run_opening_scan(on_time.make(live=opening_rows(panel, x={on_time.star: STAR_X}, stamp=LATE_STAMP), live_ts=LATE_STAMP,
                                                indices=index_quotes(panel, stamp=LATE_STAMP)))
    assert result.status == "success", result.message
    report = frozen(store)
    late = report["picks"][0]
    assert report["elapsed_min"] == 25.0 and any("first 25 minutes" in n and "1.13" in n for n in report["notes"])
    assert late["or_pct"] == pytest.approx(base["or_pct"] / 1.131, rel=0.01) and late["x"] < base["x"]  # the same live range counts for less


def test_after_thirty_minutes_it_is_no_opening_read_and_nothing_is_sent(env):
    env.clock.set(TOO_LATE_NOW)
    result = scan.run_opening_scan(env.make(live=opening_rows(env.panel, stamp=TOO_LATE_STAMP), live_ts=TOO_LATE_STAMP, **NOT_LAST))
    assert result.status == "failed" and "too late for an opening read" in result.message and "45 minutes" in result.message
    assert frozen(env.store) is None and env.telegram.sent == []


def test_a_dry_run_outside_the_window_is_a_clearly_marked_preview(env):
    env.clock.set(TOO_LATE_NOW)
    result = scan.run_opening_scan(env.make(live=opening_rows(env.panel, stamp=TOO_LATE_STAMP), live_ts=TOO_LATE_STAMP, dry_run=True))
    assert result.status == "success"
    notes = result.metrics["report"]["notes"]
    assert any(n.startswith("PREVIEW: the live quotes are 45 minutes after the open") for n in notes)
    assert frozen(env.store) is None and env.telegram.sent == []


# ------------------------------------------------------------------------------ what it must not guess
def test_nse_being_unavailable_fails_with_an_alert_and_nothing_is_guessed(env):
    plain = scan.run_opening_scan(env.make(nse_fail=True, **NOT_LAST))
    assert plain.status == "failed" and "NSE live quotes unavailable" in plain.message and "nothing is guessed" in plain.message
    assert alerts(env, "opening_scan_failed") and frozen(env.store) is None and env.telegram.sent == []
    last = scan.run_opening_scan(env.make(nse_fail=True, **LAST))
    assert last.status == "failed" and "OPENING" in said(env).upper()


def test_too_few_live_quotes_stop_the_scan(env):
    few = env.live[:5]
    result = scan.run_opening_scan(env.make(live=few, **NOT_LAST))
    assert result.status == "failed" and "too few live quotes" in result.message and "5 of 24" in result.message and frozen(env.store) is None


def test_quotes_stamped_on_an_earlier_session_are_not_todays_opening(env):
    stale = opening_rows(env.panel, stamp="2026-10-30T09:59:00Z")
    result = scan.run_opening_scan(env.make(live=stale, **NOT_LAST))
    assert result.status == "failed" and "stamped on an earlier session" in result.message


def test_a_name_whose_previous_close_does_not_match_the_stored_one_is_left_out(env):
    base = {s: float(env.panel.c[s].iloc[-1]) for s in env.panel.symbols}
    base[env.star] *= 1.05  # an ex-date today: the exchange's previous close is 5 % away from the stored one
    live = opening_rows(env.panel, x={env.star: STAR_X}, base_prev=base)
    assert scan.run_opening_scan(env.make(live=live)).status == "success"
    report = frozen(env.store)
    assert report["movers"] == [] and report["n_eligible"] == 23
    assert any("1 name(s) left out" in n and "price basis differs" in n for n in report["notes"])


def test_when_most_previous_closes_disagree_the_stored_history_is_blamed(env):
    base = {s: float(env.panel.c[s].iloc[-1]) * 1.08 for s in env.panel.symbols}
    result = scan.run_opening_scan(env.make(live=opening_rows(env.panel, base_prev=base), **NOT_LAST))
    assert result.status == "failed" and "differs from the exchange's previous close" in result.message and "collect_india" in result.message
    assert frozen(env.store) is None


def test_a_history_that_stops_before_the_previous_session_stops_the_scan(settings, store, fake_clock):
    old = panel_ending("2026-10-29", n_dates=420, n_symbols=24, seed=11)  # nothing stored for Friday 30 October
    env = build_env(settings, store, fake_clock, old)
    result = scan.run_opening_scan(env.make(live=opening_rows(env.panel), **NOT_LAST))
    assert result.status == "failed" and "2026-10-30 was a regular session" in result.message and "collect_india" in result.message
    assert frozen(env.store) is None and env.telegram.sent == []


def test_a_special_session_in_the_gap_is_not_a_missing_session(settings, store, fake_clock):
    (settings.data_dir / "calendar_overrides.json").write_text(json.dumps(
        {"sessions": [{"date": "2026-10-30", "open": "17:45", "close": "18:45", "description": "Muhurat trading"}]}), encoding="utf-8")
    env = build_env(settings, store, fake_clock, panel_ending("2026-10-29", n_dates=420, n_symbols=24, seed=11))
    assert scan.run_opening_scan(env.make()).status == "success"


def test_a_special_session_day_is_skipped_because_the_open_is_not_09_15(settings, store, fake_clock, panel):
    (settings.data_dir / "calendar_overrides.json").write_text(json.dumps(
        {"sessions": [{"date": DAY, "open": "17:45", "close": "18:45", "description": "Muhurat trading"}]}), encoding="utf-8")
    env = build_env(settings, store, fake_clock, panel)
    result = scan.run_opening_scan(env.make())
    assert result.status == "skipped" and "special session" in result.message and "09:15" in result.message
    assert frozen(env.store) is None and env.telegram.sent == []


def test_a_closed_day_and_a_disabled_feature_are_skipped(settings, store, fake_clock, panel):
    env = build_env(settings, store, fake_clock, panel)
    closed = scan.run_opening_scan(env.make(run_key="2026-11-01"))  # a Sunday
    assert closed.status == "skipped" and "closed" in closed.message and env.store.kv_get_json(scan.run_key("2026-11-01")) is None
    off = build_env(settings, store, fake_clock, panel, opening_enabled=False)
    result = scan.run_opening_scan(off.make())
    assert result.status == "skipped" and "OPENING_ENABLED" in result.message and frozen(store) is None


# -------------------------------------------------------------------------------------------- the news
def test_news_is_asked_only_for_the_new_movers_and_shown_with_them(env):
    seed_morning(env.store, [env.pick1])
    intel = NewsIntel({env.star: [catalyst("Big order win announced")]})
    assert scan.run_opening_scan(env.make(intel=intel)).status == "success"
    assert intel.asked == [[env.star]]
    notes = frozen(env.store)["movers"][0]["notes"]
    assert any(n.startswith("news: Big order win announced") for n in notes)
    assert "Big order win announced" in said(env)


def test_news_trouble_never_stops_the_read(env):
    broken = NewsIntel(fail=True)
    assert scan.run_opening_scan(env.make(intel=broken)).status == "success"
    assert any("news unavailable for the new movers" in n for n in frozen(env.store)["notes"])


def test_reduced_news_coverage_is_stated(env):
    assert scan.run_opening_scan(env.make(intel=NewsIntel(note="RSS feeds degraded"))).status == "success"
    assert any("news coverage reduced: RSS feeds degraded" in n for n in frozen(env.store)["notes"])


def test_a_day_without_a_standout_says_nothing_is_forced(settings, store, fake_clock, panel):
    env = build_env(settings, store, fake_clock, panel)
    assert scan.run_opening_scan(env.make(live=opening_rows(panel))).status == "success"  # every opening quiet
    report = frozen(store)
    assert report["movers"] == [] and any("no new name opened at 0.60x or more of its normal daily range: nothing is forced" in n for n in report["notes"])
    assert "No new name passed the gates today: nothing is forced" in said(env)


def test_the_mover_threshold_is_the_opening_strength_setting(settings, store, fake_clock, panel):
    strict = build_env(settings, store, fake_clock, panel, opening_min_x=1.0)  # the star opened at 0.95x: just short
    assert scan.run_opening_scan(strict.make()).status == "success"
    assert frozen(store)["movers"] == [] and any("opened at 1.00x or more" in n for n in frozen(store)["notes"])


def test_a_mover_already_past_3_percent_says_so_and_is_no_test_of_the_model(settings, store, fake_clock, panel):
    env = build_env(settings, store, fake_clock, panel)
    big = 1.3  # 1.3 x 2.77 % = 3.6 %: the star's range so far has passed 3 % when the scan reads it
    assert scan.run_opening_scan(env.make(live=opening_rows(panel, x={env.star: big}))).status == "success"
    star = frozen(store)["movers"][0]
    assert star["metrics"]["or_raw_pct"] > 3.0 and star["p3"] == 1.0 and star["p5"] < 1.0
    assert star["day_range"][0] >= star["metrics"]["or_raw_pct"] - 0.01  # never below what it has already done
    movers = said(env).split("New opening movers")[1]
    assert "already above 3%" in movers and "Chance of a range above 3%" not in movers and "100%" not in movers
    env.clock.set(EVENING)
    add_day_bars(env, {env.star: 5.5})
    assert scan.evaluate_opening(env.make(yahoo=FakeYahoo({}))).status == "success"
    stored = store.kv_get_json(scan.outcome_key(DAY))
    assert stored["ideas"][f"MOVER:{env.star}"]["decided3"] is True and stored["universe"]["n"] == 23 and stored["universe"]["n_decided"] == 1
    record = scan.live_record(store, date(2026, 11, 3))
    assert record["model"]["n"] == 23 and record["model"]["n_decided"] == 1  # the 23 other names are the test; the star is counted apart
    assert (record["MOVER"]["n"], record["MOVER"]["n_open"], record["MOVER"]["n_decided"], record["MOVER"]["hit3"]) == (1, 0, 1, None)


def test_a_missing_index_feed_is_stated_and_the_read_goes_on(env):
    result = scan.run_opening_scan(env.make(indices={}))
    assert result.status == "success"
    report = frozen(env.store)
    assert "nifty_chg" not in report["market"] and any("NSE index quotes unavailable" in n for n in report["notes"])


# ---------------------------------------------------------------------------------------- the deadline
def test_a_scan_after_the_deadline_is_frozen_late_and_withheld_unless_forced(env):
    env.clock.set(PAST_DEADLINE)
    stamp = "2026-11-02T04:14:00Z"  # 09:44 IST: still inside the thirty-minute limit
    kw = dict(live=opening_rows(env.panel, x={env.star: STAR_X}, stamp=stamp), live_ts=stamp)
    result = scan.run_opening_scan(env.make(**kw))
    assert result.status == "success" and "withheld" in result.message and env.telegram.sent == []
    assert frozen(env.store)["late"] is True and alerts(env, "opening_deadline_missed")
    retry = scan.run_opening_scan(env.make(**kw))  # a retry does not get past the deadline either
    assert retry.status == "success" and env.telegram.sent == []
    forced = env.make(nse_fail=True, force=True)  # the frozen report is sent as it is: NSE is not asked again
    assert scan.run_opening_scan(forced).status == "success"
    assert forced.nse_api().live_calls == [] and LATE_BANNER in said(env)


def test_the_report_can_be_switched_off_and_is_still_frozen_for_the_record(settings, store, fake_clock, panel):
    env = build_env(settings, store, fake_clock, panel, send_opening_report=False)
    result = scan.run_opening_scan(env.make())
    assert result.status == "success" and "withheld (opening_disabled)" in result.message
    assert frozen(store) is not None and store.kv_get_json(scan.state_key(DAY)) is not None and env.telegram.sent == []


def test_a_failed_delivery_is_retried_with_the_same_frozen_report(settings, store, fake_clock, panel):
    telegram = FakeTelegram()
    telegram.fail = True
    env = build_env(settings, store, fake_clock, panel, telegram=telegram)
    first = scan.run_opening_scan(env.make(**NOT_LAST))
    assert first.status == "failed" and "report delivery failed" in first.message and frozen(store) is not None
    frozen_before = frozen(store)
    telegram.fail = False
    second = scan.run_opening_scan(env.make(nse_fail=True))
    assert second.status == "success" and telegram.sent and frozen(store) == frozen_before


# ----------------------------------------------------------------------------------------- the schedule
def test_the_scheduler_runs_the_scan_through_the_job_registry(env):
    assert jobs.FUNCTIONS["opening_scan"](env.make()).status == "success" and env.telegram.sent
    assert jobs.FUNCTIONS["opening_scan"](env.make(run_key="2026-11-01")).status == "skipped"
    assert jobs.FUNCTIONS["evaluate_opening"](env.make(run_key="2026-11-01")).status == "skipped"


# ----------------------------------------------------------------------------------------- evaluation
EVENING = datetime(2026, 11, 2, 13, 5, tzinfo=UTC)  # 18:35 IST


def add_day_bars(env, ranges, *, day=DAY, rank=2, default=1.5):
    """The day's daily bars of every name: open = the live open, a range of ``ranges.get(symbol, default)`` % of the open."""
    for row in env.live:
        r = ranges.get(row["symbol"], default)
        opened = row["open"]
        bar = {"trade_date": day, "open": opened, "high": opened * (1 + r / 200.0), "low": opened * (1 - r / 200.0), "close": opened,
               "volume": 1e6}
        env.store.upsert_bars(env.store.instrument_id(row["symbol"]), [bar], source="nse_bhavcopy" if rank >= 2 else "yahoo",
                              source_rank=rank, quality="official" if rank >= 2 else "provisional")


def five_minute_bars(levels, ltp, *, hit="target", day=DAY):
    """Bars of the day whose first candle breaks above the long trigger and whose next one then reaches the first target (or the stop)."""
    trig, stop, target = levels["trigger_long"], levels["long_stop"], levels["long_target1"]
    third = {"hhmm": "09:40", "open": trig, "high": target + 0.5, "low": trig - 0.01, "close": target} if hit == "target" else \
        {"hhmm": "09:40", "open": trig, "high": trig + 0.1, "low": stop - 0.5, "close": stop}
    bars = [{"hhmm": "09:25", "open": ltp, "high": ltp, "low": ltp, "close": ltp},
            {"hhmm": "09:30", "open": ltp, "high": trig + 0.05, "low": max(ltp, stop + 0.5), "close": trig},
            third, {"hhmm": "15:10", "open": third["close"], "high": third["close"], "low": third["close"] * 0.999, "close": third["close"]}]
    return [{"trade_date": day, **bar} for bar in bars] + [{"trade_date": "2026-10-30", "hhmm": "09:35", "open": 1.0, "high": 9e9, "low": 0.1,
                                                            "close": 1.0}]  # an earlier session in the same feed is never this day's


@pytest.fixture()
def judged_env(env):
    """A frozen report with two picks and the star, and the evening of the same day with the official bars in."""
    seed_morning(env.store, [env.pick1, env.pick2])
    assert scan.run_opening_scan(env.make()).status == "success"
    env.clock.set(EVENING)
    add_day_bars(env, {env.star: 5.5, env.pick1: 4.0})
    return env


def evening_ctx(env, bars=None):
    star = frozen(env.store)["movers"][0]
    series = {f"{env.star}.NS": five_minute_bars(star["levels"], star["ltp"])} if bars is None else bars
    return env.make(yahoo=FakeYahoo(series))


def test_the_evening_judges_every_listed_name_and_the_whole_table(judged_env):
    env = judged_env
    ctx = evening_ctx(env)
    result = scan.evaluate_opening(ctx)
    assert result.status == "success" and "1 run(s) settled" in result.message and "model record: 1 session(s)" in result.message
    stored = env.store.kv_get_json(scan.outcome_key(DAY))
    assert stored["official"] is True and stored["universe"]["n"] == 24 and set(stored["ideas"]) == {
        f"PICK:{env.pick1}", f"PICK:{env.pick2}", f"MOVER:{env.star}"}
    star = stored["ideas"][f"MOVER:{env.star}"]
    assert star["status"] == "judged" and star["hit3"] is True and star["hit5"] is True and star["range_pct"] == pytest.approx(5.5, abs=0.01)
    assert star["plan"]["status"] == "target1" and star["plan"]["side"] == "LONG" and star["ahead_pct"] > 0 and "ahead_in_band" in star
    assert stored["ideas"][f"PICK:{env.pick1}"]["hit3"] is True and stored["ideas"][f"PICK:{env.pick2}"]["hit3"] is False
    assert "plan" not in stored["ideas"][f"PICK:{env.pick2}"]  # Yahoo had no five-minute bars for this name: judged on the daily bar only
    assert stored["universe"]["top_n"] == 6 and 0 <= stored["universe"]["hit3"] <= 1 and stored["universe"]["band_cover"] is not None


def test_a_second_evening_run_keeps_the_intraday_figures_without_asking_yahoo_again(judged_env):
    env = judged_env
    first_ctx = evening_ctx(env)
    scan.evaluate_opening(first_ctx)
    before = env.store.kv_get_json(scan.outcome_key(DAY))
    assert sorted(first_ctx._cache["yahoo"].calls) == sorted([f"{env.pick1}.NS", f"{env.pick2}.NS", f"{env.star}.NS"])
    second = evening_ctx(env, bars={})
    result = scan.evaluate_opening(second)
    after = env.store.kv_get_json(scan.outcome_key(DAY))
    assert result.status == "success"
    assert after["ideas"][f"MOVER:{env.star}"]["plan"] == before["ideas"][f"MOVER:{env.star}"]["plan"]
    assert set(second._cache["yahoo"].calls) == {f"{env.pick1}.NS", f"{env.pick2}.NS"}  # only the names that still had no intraday figures


def test_a_stop_is_recorded_as_a_stop_with_the_cost_taken_off(judged_env):
    env = judged_env
    star = frozen(env.store)["movers"][0]
    ctx = env.make(yahoo=FakeYahoo({f"{env.star}.NS": five_minute_bars(star["levels"], star["ltp"], hit="stop")}))
    scan.evaluate_opening(ctx)
    plan = env.store.kv_get_json(scan.outcome_key(DAY))["ideas"][f"MOVER:{env.star}"]["plan"]
    assert plan["status"] == "stop" and plan["ret_pct"] < 0 and plan["net_pct"] == pytest.approx(plan["ret_pct"] - star["est_cost_pct"], abs=0.002)


def test_yahoo_trouble_leaves_the_daily_outcomes_in_place(judged_env):
    env = judged_env
    ctx = env.make(yahoo=FakeYahoo({f"{env.star}.NS": RuntimeError("boom"), f"{env.pick1}.NS": DataUnavailable("yahoo", "blocked")}))
    result = scan.evaluate_opening(ctx)
    assert result.status == "success"
    stored = env.store.kv_get_json(scan.outcome_key(DAY))
    assert stored["ideas"][f"MOVER:{env.star}"]["hit3"] is True and "plan" not in stored["ideas"][f"MOVER:{env.star}"]


def test_provisional_bars_are_judged_but_wait_for_the_official_file_before_they_count(env):
    seed_morning(env.store, [env.pick1])
    assert scan.run_opening_scan(env.make()).status == "success"
    env.clock.set(EVENING)
    add_day_bars(env, {env.star: 5.0}, rank=1)
    result = scan.evaluate_opening(env.make())
    assert result.status == "success" and "0 run(s) settled, 1 waiting" in result.message
    stored = env.store.kv_get_json(scan.outcome_key(DAY))
    assert stored["official"] is False and stored["ideas"][f"MOVER:{env.star}"]["hit3"] is True
    record = scan.live_record(env.store, date(2026, 11, 3))
    assert record["model"]["days"] == 0 and record["MOVER"]["n"] == 0 and record["MOVER"]["pending"] == 1
    add_day_bars(env, {env.star: 5.0}, rank=2)  # the official file arrives: the same night's next run replaces the outcome
    scan.evaluate_opening(env.make())
    assert env.store.kv_get_json(scan.outcome_key(DAY))["official"] is True
    assert scan.live_record(env.store, date(2026, 11, 3))["model"]["days"] == 1


def test_a_run_without_its_days_bar_waits_and_an_empty_ledger_is_skipped(env):
    assert scan.evaluate_opening(env.make()).status == "skipped"
    assert scan.run_opening_scan(env.make()).status == "success"
    env.clock.set(EVENING)
    result = scan.evaluate_opening(env.make())
    assert result.status == "success" and "0 run(s) settled, 1 waiting" in result.message and env.store.kv_get_json(scan.outcome_key(DAY)) is None


def test_the_next_morning_the_report_shows_the_record_and_a_late_report_stays_out_of_it(judged_env):
    env = judged_env
    scan.evaluate_opening(evening_ctx(env))
    env.clock.set(datetime(2026, 11, 3, 4, 0, 30, tzinfo=UTC))
    stamp = "2026-11-03T04:00:00Z"
    previous = {s: float(row["open"]) for s, row in ((r["symbol"], r) for r in env.live)}  # yesterday's stored close is its open
    live = opening_rows(env.panel, x={env.star: STAR_X}, stamp=stamp, base_prev=previous)
    assert scan.run_opening_scan(env.make(run_key="2026-11-03", live=live, live_ts=stamp, indices=index_quotes(env.panel, stamp=stamp))).status == "success"
    record = frozen(env.store, "2026-11-03")["record"]
    assert record["model"]["days"] == 1 and record["model"]["n"] == 24 and record["MOVER"]["n"] == 1 and record["PICK"]["n"] == 2
    assert record["plan"]["n"] == 1 and record["MOVER"]["hit3"] == 1.0
    assert "Live record:</b> 1 session(s), 24 names judged" in said(env)
    # a report built after its deadline is kept out of the record
    payload = env.store.kv_get_json(scan.run_key(DAY))
    payload["report"]["late"] = True
    env.store.kv_set_json(scan.run_key(DAY), payload)
    assert scan.live_record(env.store, date(2026, 11, 3)) == {"model": {"days": 0, "n": 0, "validated": False}}


def test_a_split_on_the_day_is_not_an_outcome(judged_env):
    env = judged_env
    row = next(r for r in env.live if r["symbol"] == env.star)
    env.store.upsert_bars(env.store.instrument_id(env.star), [{"trade_date": DAY, "open": row["open"] / 2, "high": row["open"] / 2 * 1.02,
                                                                "low": row["open"] / 2 * 0.98, "close": row["open"] / 2, "volume": 1e6}],
                          source="nse_bhavcopy", source_rank=2, quality="official")
    scan.evaluate_opening(evening_ctx(env, bars={}))
    stored = env.store.kv_get_json(scan.outcome_key(DAY))
    assert stored["ideas"][f"MOVER:{env.star}"]["status"] == "invalid" and stored["universe"]["n"] == 23
