"""The broker feed inside the real scans and the evening judge: both feeds agreeing, each one failing alone, both failing, a lagging broker, names
the feeds disagree on, the index fallback, the daily instrument-map cache, and the evening judging on the broker's candles with Yahoo behind it."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from app.errors import DataUnavailable
from automation import opening as scan
from automation.ideas import NSE_GRACE_SEC
from ingestion import pipeline as pl
from ingestion.registry import register_all
from tests import test_closing_scan as closing_tests
from tests import test_opening_scan as opening_tests
from tests.seed_data import FakeKotakApi, panel_ending
from tests.test_opening_scan import (
    DAY,
    EVENING,
    OPEN_STAMP,
    FakeYahoo,
    add_day_bars,
    five_minute_bars,
    frozen,
    index_quotes,
    opening_rows,
    seed_morning,
)

UTC = timezone.utc
KEY = "kotak-test-token-0123456789abcdef0123"
NSE_ONLY = "NSE (SECURITIES IN F&O): 24 of 24 names with a complete live row"


@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=420, n_symbols=24, seed=11)


@pytest.fixture(autouse=True)
def no_real_broker(monkeypatch):
    """Tests are hermetic: a context that reaches for the real broker client (and so the network) is a test that forgot its double."""
    def refuse(self, *args, **kwargs):
        raise AssertionError("a test reached for the real Kotak client: give the context a FakeKotakApi")

    monkeypatch.setattr("ingestion.kotak.KotakApi.__init__", refuse)


@pytest.fixture()
def env(settings, store, fake_clock, panel):
    base = opening_tests.build_env(settings, store, fake_clock, panel, kotak_consumer_key=KEY)
    plain_make = base.make

    def make(**kw):
        ctx = plain_make(**kw)
        ctx._cache.setdefault("kotak", FakeKotakApi([dict(r) for r in base.live], clock=ctx.clock, index_rows=index_quotes(base.panel)))
        return ctx

    base.make = make
    return base


def with_broker(env, ctx, rows=None, **kw):
    """Give a context a scripted broker whose rows are, by default, a copy of NSE's (so the two feeds agree on everything)."""
    kw.setdefault("index_rows", index_quotes(env.panel))
    ctx._cache["kotak"] = FakeKotakApi([dict(r) for r in (env.live if rows is None else rows)], clock=ctx.clock, **kw)
    return ctx


def dry(env, **kw):
    """The report of a dry run (nothing is frozen or sent), so that two runs on the same day can be compared."""
    result = scan.run_opening_scan(with_broker(env, env.make(dry_run=True, **kw.pop("ctx_kw", {})), **kw))
    assert result.status == "success", result.message
    return result.metrics["report"]


# ------------------------------------------------------------------------------------------- the opening scan
def test_a_broker_that_agrees_changes_nothing_but_the_source_line(env):
    seed_morning(env.store, [env.pick1, env.pick2])
    both = dry(env)
    plain_ctx = env.make(dry_run=True)
    plain_ctx.settings = replace(env.cfg, kotak_consumer_key=None)  # the same day, the broker switched off
    plain = scan.run_opening_scan(plain_ctx).metrics["report"]
    assert both["source"] == "NSE (SECURITIES IN F&O) + Kotak broker feed: 24 of 24 names with a complete live row; 24 confirmed by both"
    assert plain["source"] == NSE_ONLY
    assert both["movers"] == plain["movers"] and both["picks"] == plain["picks"] and both["market"] == plain["market"]
    assert both["notes"] == plain["notes"]  # nothing to say about the feeds when they agree


def test_the_fresher_row_of_a_name_is_the_one_that_is_scored(env):
    star = env.star
    wider = opening_rows(env.panel, x={star: 1.05})  # the broker has seen more trades: a wider range so far and a bigger volume
    for row in wider:
        row["volume"] = 2e6 if row["symbol"] == star else 1e6
    mine = dry(env, rows=wider)
    plain = dry(env, rows=[dict(r) for r in env.live])
    assert mine["movers"][0]["x"] > plain["movers"][0]["x"] and mine["movers"][0]["symbol"] == star
    assert mine["movers"][0]["x"] == pytest.approx(1.05, abs=0.01)


def test_names_the_feeds_disagree_on_are_left_out_and_said(env):
    broken = [dict(r, prev_close=r["prev_close"] * 1.05) if r["symbol"] == env.star else dict(r) for r in env.live]  # a basis only one feed adjusted
    report = dry(env, rows=broken)
    assert env.star not in [m["symbol"] for m in report["movers"]]
    assert report["source"].endswith("23 of 24 names with a complete live row; 23 confirmed by both")
    assert any("disagreed on 1 name" in n and env.star in n for n in report["notes"])


def test_a_lagging_broker_feed_is_not_used_during_the_session(env):
    old = [dict(r, row_ts="2026-11-02T03:55:00Z") for r in env.live]  # 09:25 IST, while the scan runs at 09:30:30
    report = dry(env, rows=old)
    assert report["source"] == NSE_ONLY and any("Kotak live feed not used: its newest trade was 330 s old" in n for n in report["notes"])


def test_with_nse_down_the_broker_alone_carries_the_scan_and_the_indices_come_from_it_too(env):
    report = dry(env, ctx_kw={"nse_fail": True})
    assert report["source"] == "Kotak broker feed (NSE did not answer): 24 of 24 names with a complete live row"
    assert any("NSE live feed unavailable: the Kotak broker feed alone was used" in n for n in report["notes"])
    assert report["market"]["nifty_chg"] == 0.3 and report["market"]["vix"] == 14.0  # NSE's index call was never even tried
    assert [m["symbol"] for m in report["movers"]] == [env.star]


def test_the_broker_is_asked_first_and_nse_gets_only_a_grace_period_once_the_broker_has_answered(env):
    def spied(ctx):
        seen = {}
        nse = ctx._cache["nse_api"]
        original = nse.live_equities

        def spy(*args, **kwargs):
            seen["broker_calls_before_nse"] = len(ctx._cache["kotak"].quote_calls)
            seen["deadline"] = kwargs.get("deadline")
            return original(*args, **kwargs)

        nse.live_equities = spy
        return seen

    ctx = with_broker(env, env.make(dry_run=True))
    seen = spied(ctx)
    assert scan.run_opening_scan(ctx).status == "success"
    assert seen["broker_calls_before_nse"] == 1  # a blocked NSE can use up the whole budget: the broker must not wait behind it
    assert seen["deadline"] == pytest.approx(ctx.clock.monotonic() + NSE_GRACE_SEC)

    down = with_broker(env, env.make(dry_run=True), fail=True)
    seen = spied(down)
    assert scan.run_opening_scan(down).status == "success"
    assert seen["deadline"] > down.clock.monotonic() + NSE_GRACE_SEC  # without the broker's answer NSE keeps the full budget

    direct = with_broker(env, env.make(dry_run=True))
    seen = spied(direct)
    assert scan.fetch_live(direct, env.panel.symbols, None) is not None
    assert seen["deadline"] == pytest.approx(direct.clock.monotonic() + NSE_GRACE_SEC)  # no deadline given: the grace is still a bound


def test_a_slow_nse_does_not_cost_the_broker_its_answer(env):
    ctx = with_broker(env, env.make(dry_run=True))
    nse = ctx._cache["nse_api"]
    original = nse.live_equities

    def slow(*args, **kwargs):
        ctx.clock.advance(NSE_GRACE_SEC + 1)  # NSE answers, but only after the grace period: its call is refused at the deadline
        return original(*args, **kwargs)

    nse.live_equities = slow
    live = scan.fetch_live(ctx, env.panel.symbols, None)
    assert live is not None and live.broker == "used" and len(live.rows) == len(env.panel.symbols)


def test_with_the_broker_down_the_nse_scan_runs_as_it_always_did_and_says_so(env):
    report = dry(env, fail=True)
    assert report["source"] == NSE_ONLY and report["notes"][0] == "Kotak live feed unavailable: NSE alone was used"
    assert [m["symbol"] for m in report["movers"]] == [env.star]
    health = {h["source_id"]: h for h in env.store.source_health()}
    assert health["kotak_quotes"]["consecutive_failures"] == 1 and "simulated outage" in health["kotak_quotes"]["last_error"]


def test_when_neither_feed_answers_the_failure_names_both_and_nothing_is_guessed(env):
    ctx = with_broker(env, env.make(nse_fail=True), fail=True)
    result = scan.run_opening_scan(ctx)
    assert result.status == "failed" and result.message.startswith("NSE and Kotak live quotes unavailable (")
    assert "simulated block" in result.message and "simulated outage" in result.message and "nothing is guessed" in result.message
    assert env.store.kv_get_json(scan.run_key(DAY)) is None


def test_without_a_key_the_broker_is_never_asked(settings, store, fake_clock, panel):
    plain = opening_tests.build_env(settings, store, fake_clock, panel)
    ctx = plain.make(dry_run=True)
    ctx._cache["kotak"] = FakeKotakApi([dict(r) for r in plain.live], clock=fake_clock)
    result = scan.run_opening_scan(ctx)
    assert result.status == "success" and ctx._cache["kotak"].quote_calls == [] and result.metrics["report"]["source"] == NSE_ONLY
    assert "kotak_quotes" not in {h["source_id"]: h for h in store.source_health() if h["last_attempt_at"]}


def test_the_index_quotes_come_from_the_broker_when_nses_index_call_gives_nothing(env):
    ctx = env.make(dry_run=True)
    ctx._cache["nse_api"].indices_fail = True
    report = scan.run_opening_scan(with_broker(env, ctx)).metrics["report"]
    assert report["market"]["nifty_chg"] == 0.3 and not any("index quotes unavailable" in n for n in report["notes"])


def test_a_broker_feed_that_is_not_used_does_not_supply_the_indices_either(env):
    ctx = env.make(dry_run=True)
    ctx._cache["nse_api"].indices_fail = True
    report = scan.run_opening_scan(with_broker(env, ctx, fail=True)).metrics["report"]
    assert any("index quotes unavailable" in n for n in report["notes"])


def test_an_unexpected_error_in_the_brokers_handling_costs_the_scan_only_that_feed(env):
    class Exploding(FakeKotakApi):
        def quotes(self, tokens, *, deadline=None):
            raise KeyError("boom")

    ctx = env.make(dry_run=True)
    ctx._cache["kotak"] = Exploding([dict(r) for r in env.live], clock=ctx.clock, index_rows=index_quotes(env.panel))
    report = scan.run_opening_scan(ctx).metrics["report"]
    assert report["source"] == NSE_ONLY and report["notes"][0] == "Kotak live feed unavailable: NSE alone was used"
    health = {h["source_id"]: h for h in env.store.source_health()}["kotak_quotes"]
    assert health["consecutive_failures"] == 1 and "unexpected KeyError" in health["last_error"]

    class BadIndices(FakeKotakApi):
        def indices(self, *, deadline=None):
            raise TypeError("boom")

    ctx = env.make(dry_run=True)
    ctx._cache["nse_api"].indices_fail = True
    ctx._cache["kotak"] = BadIndices([dict(r) for r in env.live], clock=ctx.clock)
    report = scan.run_opening_scan(ctx).metrics["report"]
    assert any("index quotes unavailable" in n for n in report["notes"]) and "Kotak broker feed" in report["source"]


def test_the_frozen_report_keeps_the_source_it_was_read_from_and_a_retry_does_not_ask_either_feed_again(env):
    first = env.make()
    with_broker(env, first)
    assert scan.run_opening_scan(first).status == "success"
    assert "Kotak broker feed" in frozen(env.store)["source"]
    retry = with_broker(env, env.make(nse_fail=True, **opening_tests.NOT_LAST), fail=True)
    assert scan.run_opening_scan(retry).status == "success" and retry._cache["kotak"].quote_calls == []


# --------------------------------------------------------------------------------------------- the closing scan
def test_the_closing_scan_uses_the_same_feeds_and_names_them_in_its_report(settings, store, fake_clock, panel):
    env = closing_tests.build_env(settings, store, fake_clock, panel, kotak_consumer_key=KEY)
    ctx = env.make(dry_run=True)
    ctx._cache["kotak"] = FakeKotakApi([dict(r) for r in env.live], clock=fake_clock, index_rows=closing_tests.index_quotes(panel))
    result = closing_tests.scan.run_closing_scan(ctx)
    assert result.status == "success", result.message
    assert result.metrics["report"]["source"] == "NSE (SECURITIES IN F&O) + Kotak broker feed: 24 of 24 names with a complete live row; 24 confirmed by both"
    down = env.make(dry_run=True, nse_fail=True)
    down._cache["kotak"] = FakeKotakApi(fail=True, clock=fake_clock)
    failed = closing_tests.scan.run_closing_scan(down)
    assert failed.status == "failed" and failed.message.startswith("NSE and Kotak live quotes unavailable (")


# ------------------------------------------------------------------------------------------- the token cache
SYMBOLS = ["AAA", "BBB", "CCC"]


def test_the_instrument_map_is_read_from_the_broker_once_a_day_and_kept_in_the_database(store):
    first = FakeKotakApi(tokens={"AAA": "11", "BBB": "22", "CCC": "33", "ZZZ": "99"})
    assert pl.kotak_tokens(store, first, SYMBOLS, date(2026, 11, 2)) == {"AAA": "11", "BBB": "22", "CCC": "33"}
    again = FakeKotakApi(tokens={"AAA": "11"})
    assert pl.kotak_tokens(store, again, ["AAA", "BBB"], date(2026, 11, 2)) == {"AAA": "11", "BBB": "22"} and again.master_calls == 0
    stored = store.kv_get_json(pl.TOKENS_KEY)
    assert stored["day"] == "2026-11-02" and stored["tokens"] == {"AAA": "11", "BBB": "22", "CCC": "33"} and stored["absent"] == []


def test_a_new_name_triggers_a_new_read_and_a_name_the_broker_does_not_list_is_not_asked_for_again(store):
    api = FakeKotakApi(tokens={"AAA": "11", "BBB": "22"})
    assert pl.kotak_tokens(store, api, ["AAA", "NOPE"], date(2026, 11, 2)) == {"AAA": "11"} and api.master_calls == 1
    assert store.kv_get_json(pl.TOKENS_KEY)["absent"] == ["NOPE"]
    assert pl.kotak_tokens(store, api, ["AAA", "NOPE"], date(2026, 11, 2)) == {"AAA": "11"} and api.master_calls == 1  # known to be unlisted
    assert pl.kotak_tokens(store, api, ["AAA", "BBB"], date(2026, 11, 2)) == {"AAA": "11", "BBB": "22"} and api.master_calls == 2
    assert pl.kotak_tokens(store, api, ["AAA"], date(2026, 11, 3)) == {"AAA": "11"} and api.master_calls == 3  # a new day: read again


def test_when_the_master_cannot_be_read_the_older_map_serves_and_with_none_the_failure_is_raised(store):
    with pytest.raises(DataUnavailable):
        pl.kotak_tokens(store, FakeKotakApi(master_fail=True), SYMBOLS, date(2026, 11, 2))
    assert store.kv_get_json(pl.TOKENS_KEY) is None
    pl.kotak_tokens(store, FakeKotakApi(tokens={"AAA": "11", "BBB": "22", "CCC": "33"}), SYMBOLS, date(2026, 11, 2))
    down = FakeKotakApi(master_fail=True)
    assert pl.kotak_tokens(store, down, SYMBOLS, date(2026, 11, 3)) == {"AAA": "11", "BBB": "22", "CCC": "33"} and down.master_calls == 1
    assert store.kv_get_json(pl.TOKENS_KEY)["day"] == "2026-11-02"  # the older map is not re-stamped as fresh


@pytest.mark.parametrize("damaged", [
    "text", ["AAA"], {"day": "2026-11-02", "tokens": ["AAA"], "absent": []}, {"day": "2026-11-02", "tokens": {"AAA": 11}, "absent": []},
    {"day": "2026-11-02", "tokens": {"AAA": ""}, "absent": []}, {"day": "2026-11-02", "tokens": {"AAA": "11"}, "absent": "BBB"},
    {"day": "2026-11-02", "tokens": {"AAA": "11"}, "absent": [1]}, {"day": 20261102, "tokens": {"AAA": "11"}, "absent": []},
    {"tokens": {"AAA": "11"}, "absent": []}, {"day": "2026-11-02", "absent": []},
])
def test_a_recorded_map_of_the_wrong_shape_is_no_map_and_is_read_again_instead_of_crashing_the_scan(store, damaged):
    store.kv_set_json(pl.TOKENS_KEY, damaged)
    api = FakeKotakApi(tokens={"AAA": "11", "BBB": "22", "CCC": "33"})
    assert pl.kotak_tokens(store, api, SYMBOLS, date(2026, 11, 2)) == {"AAA": "11", "BBB": "22", "CCC": "33"} and api.master_calls == 1
    assert pl.kotak_tokens(store, api, SYMBOLS, date(2026, 11, 2)) and api.master_calls == 1  # the repaired record serves from now on


def test_a_damaged_record_or_an_older_map_without_a_wanted_name_does_not_stand_in_for_a_master_that_is_down(store):
    store.kv_set_json(pl.TOKENS_KEY, {"day": "2026-11-01", "tokens": ["AAA"], "absent": []})
    with pytest.raises(DataUnavailable, match="simulated master outage"):
        pl.kotak_tokens(store, FakeKotakApi(master_fail=True), SYMBOLS, date(2026, 11, 2))
    store.kv_set_json(pl.TOKENS_KEY, {"day": "2026-11-01", "tokens": {"ZZZ": "99"}, "absent": []})  # valid, but of an unrelated name
    with pytest.raises(DataUnavailable, match="simulated master outage"):  # the real reason, not "the broker lists none of the symbols"
        pl.kotak_tokens(store, FakeKotakApi(master_fail=True), SYMBOLS, date(2026, 11, 2))
    errors: list[str] = []
    assert pl.collect_kotak_quotes(store, FakeKotakApi(master_fail=True), SYMBOLS, date(2026, 11, 2), errors=errors) is None
    assert errors == ["[kotak_quotes] simulated master outage"]


def test_collecting_quotes_records_the_sources_health_and_a_failure_returns_none_with_the_reason(store, fake_clock):
    register_all(store)  # a job context does this: source health is kept for the registered sources
    fake_clock.set(datetime(2026, 11, 2, 4, 0, 30, tzinfo=UTC))  # after the rows' stamps: a stamp from the future would not be believed
    rows = [{"symbol": s, "open": 100.0, "high": 101.0, "low": 99.0, "last": 100.5, "prev_close": 100.0, "volume": 10.0, "row_ts": OPEN_STAMP}
            for s in SYMBOLS]
    ok = pl.collect_kotak_quotes(store, FakeKotakApi(rows, clock=fake_clock), SYMBOLS, date(2026, 11, 2))
    assert ok is not None and len(ok.rows) == 3
    health = {h["source_id"]: h for h in store.source_health()}["kotak_quotes"]
    assert health["consecutive_failures"] == 0 and health["last_data_ts"] == OPEN_STAMP and health["last_success_at"]
    errors: list[str] = []
    assert pl.collect_kotak_quotes(store, FakeKotakApi(rows, fail=True), SYMBOLS, date(2026, 11, 2), errors=errors) is None
    assert errors == ["[kotak_quotes] simulated outage"]
    assert {h["source_id"]: h for h in store.source_health()}["kotak_quotes"]["consecutive_failures"] == 1
    none_listed = FakeKotakApi(rows, tokens={"OTHER": "1"})
    errors.clear()
    assert pl.collect_kotak_quotes(store, none_listed, ["QQQ"], date(2026, 11, 4), errors=errors) is None and "lists none" in errors[0]


# ------------------------------------------------------------------------------------------ the evening judge
def broker_day(levels, ltp, *, day=DAY, until="15:10"):
    """A full broker day: a five-minute bar every five minutes from 09:15 to ``until``, flat at the previous close where ``five_minute_bars``
    (the breakout, the target) has none."""
    special = {b["hhmm"]: b for b in five_minute_bars(levels, ltp, day=day) if b["trade_date"] == day}
    bars, close = [], ltp
    minute = 9 * 60 + 15
    while minute <= int(until[:2]) * 60 + int(until[3:]):
        hhmm = f"{minute // 60:02d}:{minute % 60:02d}"
        bar = special.get(hhmm) or {"hhmm": hhmm, "open": close, "high": close, "low": close, "close": close}
        close = bar["close"]
        bars.append({"ts": f"{day}T{hhmm}:00Z", "trade_date": day, **bar})
        minute += 5
    return bars


@pytest.fixture()
def judged(env):
    seed_morning(env.store, [env.pick1, env.pick2])
    assert scan.run_opening_scan(env.make()).status == "success"
    env.clock.set(EVENING)
    add_day_bars(env, {env.star: 5.5, env.pick1: 4.0})
    env.star_report = frozen(env.store)["movers"][0]
    return env


def evening(env, *, kotak=None, yahoo=None, **kw):
    ctx = env.make(yahoo=FakeYahoo(yahoo or {}), **kw)
    if kotak is not None:
        ctx._cache["kotak"] = kotak
    return ctx


def star_outcome(env):
    return env.store.kv_get_json(scan.outcome_key(DAY))["ideas"][f"MOVER:{env.star}"]


def test_the_evening_judges_on_the_brokers_candles_and_never_asks_yahoo_for_a_name_it_has(judged):
    env = judged
    star = env.star_report
    kotak = FakeKotakApi(bars={env.star: broker_day(star["levels"], star["ltp"])}, clock=env.clock, tokens={env.star: env.star})
    ctx = evening(env, kotak=kotak)
    assert scan.evaluate_opening(ctx).status == "success"
    plan = star_outcome(env)["plan"]
    assert plan["status"] == "target1" and plan["side"] == "LONG" and star_outcome(env)["ahead_pct"] > 0
    assert ctx._cache["yahoo"].calls == [f"{env.pick1}.NS", f"{env.pick2}.NS"]  # the broker had only the star: Yahoo is asked for the others
    assert (env.star, "5min", DAY) in kotak.candle_calls and kotak.master_calls == 0  # the day's instrument map was read by the morning scan
    assert {h["source_id"]: h for h in env.store.source_health()}["kotak_history"]["consecutive_failures"] == 0


def test_an_unfinished_broker_day_is_never_judged_as_a_quiet_one_and_yahoo_is_asked_instead(judged):
    env = judged
    star = env.star_report
    kotak = FakeKotakApi(bars={env.star: broker_day(star["levels"], star["ltp"], until="14:20")}, clock=env.clock, tokens={env.star: env.star})
    ctx = evening(env, kotak=kotak, yahoo={f"{env.star}.NS": five_minute_bars(star["levels"], star["ltp"])})
    scan.evaluate_opening(ctx)
    assert f"{env.star}.NS" in ctx._cache["yahoo"].calls and star_outcome(env)["plan"]["status"] == "target1"
    env.store.kv_delete(scan.outcome_key(DAY))
    thin = FakeKotakApi(bars={env.star: broker_day(star["levels"], star["ltp"])[:30]}, clock=env.clock, tokens={env.star: env.star})
    ctx = evening(env, kotak=thin, yahoo={})
    scan.evaluate_opening(ctx)
    assert f"{env.star}.NS" in ctx._cache["yahoo"].calls and "plan" not in star_outcome(env)  # 30 bars of 72: not judged as a day


def test_a_day_that_stops_at_15_00_or_has_a_hole_in_it_is_not_judged_as_a_complete_one(judged):
    env = judged
    star = env.star_report
    yahoo = {f"{env.star}.NS": five_minute_bars(star["levels"], star["ltp"])}
    stops_early = broker_day(star["levels"], star["ltp"], until="15:00")  # 70 of 72 bars, the last one at 15:00: enough for a looser rule
    with_hole = [b for b in broker_day(star["levels"], star["ltp"]) if b["hhmm"] != "12:30"]  # 71 bars: a stretch nobody can call quiet
    for bars in (stops_early, with_hole):
        env.store.kv_delete(scan.outcome_key(DAY))
        ctx = evening(env, kotak=FakeKotakApi(bars={env.star: bars}, clock=env.clock, tokens={env.star: env.star}), yahoo=yahoo)
        scan.evaluate_opening(ctx)
        assert f"{env.star}.NS" in ctx._cache["yahoo"].calls  # Yahoo is asked instead
    assert {h["source_id"]: h for h in env.store.source_health()}["kotak_history"]["consecutive_failures"] == 0  # an incomplete day is no outage


def test_bars_after_15_10_are_never_judged_on_so_a_day_reads_the_same_when_it_is_judged_later(judged):
    env = judged
    star = env.star_report
    clean = broker_day(star["levels"], star["ltp"])
    after = [{"ts": f"{DAY}T{h}:00Z", "trade_date": DAY, "hhmm": h, "open": 1.0, "high": 9e6, "low": 0.5, "close": 1.0}
             for h in ("15:15", "15:20", "15:25")]  # the archive's late bars (or its flat artefact) must not move the verdict
    outcomes = []
    for bars in (clean, clean + after):
        env.store.kv_delete(scan.outcome_key(DAY))
        ctx = evening(env, kotak=FakeKotakApi(bars={env.star: bars}, clock=env.clock, tokens={env.star: env.star}))
        scan.evaluate_opening(ctx)
        outcomes.append(star_outcome(env))
    assert outcomes[0] == outcomes[1] and outcomes[0]["plan"]["status"] == "target1"


def test_the_evening_shares_one_five_minute_budget_across_every_report_it_judges(judged):
    env = judged
    older = "2026-10-30"
    env.store.kv_set_json(scan.run_key(older), env.store.kv_get_json(scan.run_key(DAY)))  # a second report waiting to be judged
    add_day_bars(env, {env.star: 5.5, env.pick1: 4.0}, day=older)
    names = [env.star, env.pick1, env.pick2]
    kotak = FakeKotakApi(bars={name: [] for name in names}, clock=env.clock, tokens={name: name for name in names})
    answer = kotak.candles

    def slow(*args, **kwargs):
        try:
            return answer(*args, **kwargs)
        finally:
            env.clock.advance(scan.INTRADAY_BUDGET_SEC / 2 + 1)  # every request costs a little more than half of the evening's budget

    kotak.candles = slow
    assert scan.evaluate_opening(evening(env, kotak=kotak)).status == "success"
    assert len(kotak.candle_calls) == 2  # the first report used the budget up: the second one was given nothing, not a fresh 120 s


def test_a_broker_failure_falls_back_to_yahoo_name_by_name_and_is_recorded(judged):
    env = judged
    star = env.star_report
    kotak = FakeKotakApi(bars={env.star: DataUnavailable("kotak_history", "HTTP 429", retryable=True)}, clock=env.clock, tokens={env.star: env.star})
    ctx = evening(env, kotak=kotak, yahoo={f"{env.star}.NS": five_minute_bars(star["levels"], star["ltp"])})
    assert scan.evaluate_opening(ctx).status == "success" and star_outcome(env)["plan"]["status"] == "target1"
    assert {h["source_id"]: h for h in env.store.source_health()}["kotak_history"]["consecutive_failures"] >= 1
    env.store.kv_delete(scan.outcome_key(DAY))
    env.store.kv_delete(pl.TOKENS_KEY)  # no instrument map at all and the master is down: the broker cannot be asked about anyone
    down = FakeKotakApi(master_fail=True, clock=env.clock)
    ctx = evening(env, kotak=down, yahoo={f"{env.star}.NS": five_minute_bars(star["levels"], star["ltp"])})
    assert scan.evaluate_opening(ctx).status == "success"
    assert f"{env.star}.NS" in ctx._cache["yahoo"].calls and star_outcome(env)["plan"]["status"] == "target1" and down.candle_calls == []


def test_an_unexpected_error_while_reading_candles_falls_back_to_yahoo_like_any_other_failure(judged):
    env = judged
    star = env.star_report
    kotak = FakeKotakApi(bars={env.star: KeyError("boom")}, clock=env.clock, tokens={env.star: env.star})
    ctx = evening(env, kotak=kotak, yahoo={f"{env.star}.NS": five_minute_bars(star["levels"], star["ltp"])})
    assert scan.evaluate_opening(ctx).status == "success" and star_outcome(env)["plan"]["status"] == "target1"
    assert "unexpected KeyError" in {h["source_id"]: h for h in env.store.source_health()}["kotak_history"]["last_error"]


def test_with_the_broker_an_older_report_is_still_judged_intraday_and_without_it_not(judged):
    env = judged
    star = env.star_report
    late = datetime(2026, 11, 12, 13, 5, tzinfo=UTC)  # ten days after the report: Yahoo's feed no longer reaches back so far
    env.clock.set(late)
    bars = {env.star: broker_day(star["levels"], star["ltp"])}
    ctx = evening(env, kotak=FakeKotakApi(bars=bars, clock=env.clock, tokens={env.star: env.star}), run_key="2026-11-12")
    scan.evaluate_opening(ctx)
    assert star_outcome(env)["plan"]["status"] == "target1"
    env.store.kv_delete(scan.outcome_key(DAY))
    off = evening(env, kotak=FakeKotakApi(bars=bars, clock=env.clock, tokens={env.star: env.star}), run_key="2026-11-12")
    off.settings = replace(env.cfg, kotak_consumer_key=None)
    scan.evaluate_opening(off)
    assert "plan" not in star_outcome(env) and off._cache["yahoo"].calls == []  # too old for Yahoo: not even asked
    assert timedelta(days=10) > timedelta(days=scan.INTRADAY_REACH_DAYS) and scan.EVALUATE_DAYS >= 10
