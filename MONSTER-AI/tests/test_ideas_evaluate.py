"""``evaluate_ideas``: the frozen ideas judged against the stored bars, the outcomes kept apart from the reports, and what the live
record makes of them."""

from __future__ import annotations

from datetime import date

import pytest

from automation import ideas as scan
from tests.seed_data import make_job_context

TRADE_DAY = "2026-10-28"  # the day of the frozen ideas (a Wednesday)
NEXT_DAY, SECOND_DAY = "2026-10-29", "2026-10-30"


def idea(symbol, horizon, side, entry, stop, target1, target2=None, **extra):
    return {"symbol": symbol, "horizon": horizon, "side": side, "entry": entry, "stop": stop, "target1": target1, "target2": target2,
            "est_cost_pct": 0.3, **extra}


IDEAS = [
    idea("AAA", "BTST", "LONG", 100.0, 98.0, 102.5, 105.0),
    idea("BBB", "BTST", "SHORT", 200.0, 204.0, 196.0, 192.0),
    idea("CCC", "SWING", "LONG", 50.0, 48.0, 55.0, 58.0),
    idea("DDD", "WATCH", "LONG", 100.0, 97.0, 104.0, trigger_long=101.0, trigger_short=98.0, metrics={"long_target1": 103.0}),
    idea("EEE", "BTST", "LONG", 80.0, 78.0, 82.0, 84.0),  # no bars at all
    idea("FFF", "BTST", "LONG", 100.0, 98.0, 102.5, 105.0),  # the same move as AAA, but only on provisional bars
]
BARS = {  # symbol -> (rank, [(date, open, high, low, close)])
    "AAA": (2, [(NEXT_DAY, 100.4, 103.0, 99.5, 102.0)]),  # reaches the first target, never the stop
    "BBB": (2, [(NEXT_DAY, 201.0, 205.0, 199.0, 203.0)]),  # runs into the stop
    "CCC": (2, [(NEXT_DAY, 50.0, 51.0, 49.5, 50.5), (SECOND_DAY, 50.5, 51.5, 50.0, 51.0)]),  # neither level yet: 2 of 5 sessions
    "DDD": (2, [(NEXT_DAY, 100.0, 102.0, 99.0, 101.5)]),  # breaks the long trigger, no first target
    "FFF": (1, [(NEXT_DAY, 100.4, 103.0, 99.5, 102.0)]),
}


def add_bars(store, symbol, rank, bars):
    """Daily bars of ``symbol``: rank 2 is the official end-of-day file, rank 1 a vendor's provisional figure."""
    iid = store.upsert_instrument(symbol, "equity", exchange="NSE", name=f"{symbol} Limited", sector="Demo", is_fno=1, in_universe=1,
                                  yahoo_symbol=f"{symbol}.NS", lot_size=100)
    rows = [{"trade_date": d, "open": o, "high": h, "low": low, "close": c, "volume": 1e6} for d, o, h, low, c in bars]
    if rank == 2:
        store.upsert_bars(iid, rows, source="nse_bhavcopy", source_rank=2, quality="official")
    else:
        store.upsert_bars(iid, rows, source="yahoo", source_rank=1)


@pytest.fixture()
def ctx(settings, store, fake_clock):
    for symbol, (rank, bars) in BARS.items():
        add_bars(store, symbol, rank, bars)
    store.kv_set_json(scan.run_key(TRADE_DAY), {"report": {"trade_date": TRADE_DAY, "late": False, "ideas": IDEAS}})
    return make_job_context(settings, store, fake_clock, run_key=SECOND_DAY)


def outcomes(store, day=TRADE_DAY):
    return store.kv_get_json(scan.outcome_key(day))["ideas"]


def test_every_idea_is_judged_on_the_sessions_after_its_day(ctx):
    result = scan.evaluate_ideas(ctx)
    assert result.status == "success", result.message
    found = outcomes(ctx.store)
    assert set(found) == {"BTST:LONG:AAA", "BTST:SHORT:BBB", "SWING:LONG:CCC", "WATCH:LONG:DDD", "BTST:LONG:FFF"}  # EEE has no bars

    target = found["BTST:LONG:AAA"]
    assert target["status"] == "target1" and target["exit_price"] == 102.5 and target["ret_pct"] == 2.5
    assert target["net_pct"] == pytest.approx(2.2) and target["complete"] and target["official"]

    stopped = found["BTST:SHORT:BBB"]
    assert stopped["status"] == "stop" and stopped["exit_price"] == 204.0 and stopped["ret_pct"] == -2.0
    assert stopped["net_pct"] == pytest.approx(-2.3)

    swing = found["SWING:LONG:CCC"]
    assert swing["status"] == "time" and swing["sessions"] == 2 and swing["complete"] is False

    watch = found["WATCH:LONG:DDD"]
    assert watch["status"] == "long" and watch["long_t1_hit"] is False and watch["complete"]
    assert watch["long_close_pct"] == pytest.approx((101.5 / 101.0 - 1) * 100, abs=1e-3)


def test_the_summary_counts_what_is_settled_and_what_is_still_waiting(ctx):
    result = scan.evaluate_ideas(ctx)
    # AAA, BBB and DDD are settled on official bars; CCC is inside its horizon, EEE has no session yet, FFF waits for official bars
    assert result.metrics == {"judged": 3, "pending": 3}
    assert "3 idea(s) settled over 1 run(s), 3 still waiting" in result.message
    assert "BTST 1" in result.message and "SHORT 1" in result.message and "WATCH 1" in result.message


def test_outcomes_are_kept_apart_from_the_frozen_report(ctx):
    before = ctx.store.kv_get(scan.run_key(TRADE_DAY))
    scan.evaluate_ideas(ctx)
    assert ctx.store.kv_get(scan.run_key(TRADE_DAY)) == before  # what was said is never edited
    assert ctx.store.kv_get_json(scan.outcome_key(TRADE_DAY))["evaluated_at"]


def test_the_live_record_counts_only_settled_outcomes_judged_on_official_bars(ctx):
    scan.evaluate_ideas(ctx)
    record = scan.live_record(ctx.store, date(2026, 10, 31))
    assert record["BTST"]["n"] == 1 and record["BTST"]["pending"] == 1  # AAA is settled; FFF's bars are provisional
    assert record["SHORT"]["n"] == 1 and record["SHORT"]["mean_net_pct"] == pytest.approx(-2.3)
    assert record["SWING"]["n"] == 0 and record["SWING"]["pending"] == 1
    assert record["WATCH"]["n"] == 1 and record["WATCH"]["triggered_rate"] == 1.0
    assert all(summary["validated"] is False for summary in record.values())  # one day of ideas proves nothing


def test_a_provisional_outcome_is_replaced_once_the_official_bars_arrive(ctx):
    scan.evaluate_ideas(ctx)
    assert outcomes(ctx.store)["BTST:LONG:FFF"]["official"] is False
    add_bars(ctx.store, "FFF", 2, [(NEXT_DAY, 100.4, 103.0, 99.5, 102.0)])  # the official file replaces the vendor's figure
    scan.evaluate_ideas(ctx)
    assert outcomes(ctx.store)["BTST:LONG:FFF"]["official"] is True
    assert scan.live_record(ctx.store, date(2026, 10, 31))["BTST"]["n"] == 2


def test_a_later_evaluation_completes_a_swing_that_was_still_inside_its_horizon(ctx):
    scan.evaluate_ideas(ctx)
    assert outcomes(ctx.store)["SWING:LONG:CCC"]["complete"] is False
    add_bars(ctx.store, "CCC", 2, [("2026-11-02", 51.0, 56.0, 50.5, 55.5)])  # the first target (55) is reached on the third session
    ctx.run_key = "2026-11-02"
    scan.evaluate_ideas(ctx)
    swing = outcomes(ctx.store)["SWING:LONG:CCC"]
    assert swing["status"] == "target1" and swing["complete"] and swing["sessions"] == 3 and swing["official"]


def test_an_idea_with_a_price_basis_change_is_not_counted(ctx):
    add_bars(ctx.store, "GGG", 2, [(NEXT_DAY, 50.0, 52.0, 49.0, 51.0)])  # a 1:2 split: the price halves
    report = ctx.store.kv_get_json(scan.run_key(TRADE_DAY))
    report["report"]["ideas"].append(idea("GGG", "BTST", "LONG", 100.0, 98.0, 102.5, 105.0))
    ctx.store.kv_set_json(scan.run_key(TRADE_DAY), report)
    scan.evaluate_ideas(ctx)
    assert outcomes(ctx.store)["BTST:LONG:GGG"]["status"] == "invalid"
    assert scan.live_record(ctx.store, date(2026, 10, 31))["BTST"]["n"] == 1  # still only AAA


def test_a_late_report_is_judged_but_does_not_count_for_the_record(ctx):
    report = ctx.store.kv_get_json(scan.run_key(TRADE_DAY))
    report["report"]["late"] = True
    ctx.store.kv_set_json(scan.run_key(TRADE_DAY), report)
    assert scan.evaluate_ideas(ctx).status == "success"
    assert "BTST:LONG:AAA" in outcomes(ctx.store)
    assert scan.live_record(ctx.store, date(2026, 10, 31)) == {}


def test_without_frozen_ideas_there_is_nothing_to_evaluate(settings, store, fake_clock):
    result = scan.evaluate_ideas(make_job_context(settings, store, fake_clock, run_key=SECOND_DAY))
    assert result.status == "skipped" and "no frozen trade ideas" in result.message


def test_old_runs_outside_the_window_are_left_alone(ctx):
    ctx.store.kv_set_json(scan.run_key("2026-09-01"), {"report": {"trade_date": "2026-09-01", "late": False, "ideas": IDEAS}})
    scan.evaluate_ideas(ctx)
    assert ctx.store.kv_get_json(scan.outcome_key("2026-09-01")) is None
