"""The merge of NSE's and the broker's live rows: who supplies a name, what must agree, which of two agreeing rows is the fresher, when a
lagging broker feed is left out, and what the report is told."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from ingestion.kotak import KotakQuotes
from ingestion.livefeed import LAST_TOLERANCE, OPEN_TOLERANCE, PREV_TOLERANCE, disagreement, merge_live
from ingestion.nse import LiveEquities

UTC = timezone.utc
NOW = datetime(2026, 11, 2, 4, 0, 30, tzinfo=UTC)  # 09:30:30 IST
NSE_STAMP = "2026-11-02T04:00:05Z"


def row(symbol, **over):
    base = {"symbol": symbol, "open": 100.0, "high": 103.0, "low": 99.0, "last": 102.0, "prev_close": 100.0, "change_pct": 2.0,
            "volume": 1000.0, "turnover": 1.0e8, "year_high": None, "year_low": None, "row_ts": NSE_STAMP}
    base.update(over)
    return base


def nse(*rows, stamp=NSE_STAMP):
    return LiveEquities(stamp, list(rows))


def broker(*rows, asof=NOW - timedelta(seconds=3), received=NOW):
    return KotakQuotes(list(rows), received, len(rows), asof)


def by_symbol(merged):
    return {r["symbol"]: r for r in merged.rows}


def test_with_the_broker_off_the_nse_rows_pass_through_exactly_as_before():
    merged = merge_live(nse(row("A"), row("B")), None, configured=False)
    assert [r["symbol"] for r in merged.rows] == ["A", "B"] and {r["feed"] for r in merged.rows} == {"nse"}
    assert merged.source_ts == NSE_STAMP and merged.broker == "off" and merged.notes == [] and merged.conflicts == {}
    assert merged.feeds == {"nse": 2, "kotak": 0} and merged.cross_checked == 0


def test_a_name_only_one_feed_has_comes_from_that_feed():
    merged = merge_live(nse(row("A"), row("ONLYNSE")), broker(row("A"), row("ONLYKOTAK")), configured=True)
    got = by_symbol(merged)
    assert got["ONLYNSE"]["feed"] == "nse" and got["ONLYKOTAK"]["feed"] == "kotak" and merged.cross_checked == 1
    assert merged.broker == "used" and merged.notes == []


def test_of_two_agreeing_rows_the_one_with_the_larger_volume_is_taken_whole_and_a_tie_goes_to_nse():
    fresher = row("A", volume=1500.0, last=102.4, high=103.2, low=98.9)
    merged = merge_live(nse(row("A"), row("B")), broker(fresher, row("B")), configured=True)
    got = by_symbol(merged)
    assert got["A"]["feed"] == "kotak" and (got["A"]["last"], got["A"]["high"], got["A"]["low"], got["A"]["volume"]) == (102.4, 103.2, 98.9, 1500.0)
    assert got["B"]["feed"] == "nse"  # equal volumes: the feed every model was validated on
    assert merged.cross_checked == 2 and merged.feeds == {"nse": 1, "kotak": 1}


def test_a_row_without_a_volume_does_not_compete_the_other_feeds_complete_row_is_used_even_if_older():
    merged = merge_live(nse(row("A"), row("B", volume=None)), broker(row("A", volume=None, row_ts="2026-11-02T04:00:29Z"), row("B", volume=7.0)),
                        configured=True)
    got = by_symbol(merged)
    assert got["A"]["feed"] == "nse" and got["B"]["feed"] == "kotak"  # each name is carried by the feed that has all its figures
    assert merged.cross_checked == 0 and merged.source_ts == NSE_STAMP  # the broker's unused row, stamped later, lends nothing
    assert merged.incomplete == 0 and merged.stale == 0  # both names are covered: nothing is left out


def test_a_nse_row_with_more_volume_wins_and_fields_are_never_mixed_between_feeds():
    older = row("A", volume=900.0, high=104.0)  # the broker's row is older but, by itself, has the higher high: it must not leak in
    merged = merge_live(nse(row("A", volume=1000.0, high=103.0)), broker(older), configured=True)
    assert by_symbol(merged)["A"]["high"] == 103.0 and by_symbol(merged)["A"]["feed"] == "nse"


@pytest.mark.parametrize(("over", "word"), [
    ({"prev_close": 100.0 * (1 + PREV_TOLERANCE * 2)}, "previous close"),
    ({"open": 100.0 * (1 + OPEN_TOLERANCE * 2)}, "open"),
    ({"last": 102.0 * (1 + LAST_TOLERANCE * 2)}, "last price"),
])
def test_names_the_feeds_disagree_on_are_left_out_and_said(over, word):
    merged = merge_live(nse(row("A"), row("B")), broker(row("A", **over), row("B")), configured=True)
    assert set(by_symbol(merged)) == {"B"} and word in merged.conflicts["A"] and merged.cross_checked == 1
    assert any("disagreed on 1 name" in n and "A" in n for n in merged.notes)


def test_small_differences_that_two_snapshots_seconds_apart_really_have_are_no_conflict():
    near = row("A", last=102.0 * 1.01, open=100.0 * 1.001, prev_close=100.0 * 1.001, volume=1200.0)
    merged = merge_live(nse(row("A")), broker(near), configured=True)
    assert merged.conflicts == {} and by_symbol(merged)["A"]["feed"] == "kotak"


def test_a_missing_open_is_no_comparison_the_row_is_incomplete_and_the_other_feeds_complete_row_carries_the_name():
    merged = merge_live(nse(row("A", open=None)), broker(row("A", open=140.0, last=141.0, high=142.0, low=139.0, prev_close=139.0)), configured=True)
    assert merged.conflicts == {} and by_symbol(merged)["A"]["feed"] == "kotak" and merged.cross_checked == 0
    many = [f"S{i}" for i in range(8)]
    merged = merge_live(nse(*[row(s) for s in many]), broker(*[row(s, prev_close=50.0) for s in many]), configured=True)
    assert len(merged.conflicts) == 8 and merged.rows == [] and merged.notes[0].endswith("S0, S1, S2, S3, S4 ...")


def test_disagreement_reads_the_two_rows_without_changing_them():
    a, b = row("A"), row("A", prev_close=120.0)
    assert disagreement(a, b).startswith("previous close 100 (NSE) against 120 (Kotak)") and disagreement(a, row("A")) == ""


def test_the_scan_goes_on_with_nse_alone_when_the_broker_fails_and_says_so():
    merged = merge_live(nse(row("A")), None, configured=True)
    assert merged.broker == "failed" and by_symbol(merged)["A"]["feed"] == "nse"
    assert merged.notes == ["Kotak live feed unavailable: NSE alone was used"]
    empty = merge_live(nse(row("A")), broker(), configured=True)
    assert empty.broker == "failed"  # an answer without a single row is no answer


def test_the_broker_alone_carries_the_scan_when_nse_does_not_answer():
    merged = merge_live(None, broker(row("A", row_ts="2026-11-02T04:00:27Z"), row("B")), configured=True)
    assert {r["feed"] for r in merged.rows} == {"kotak"} and merged.feeds == {"nse": 0, "kotak": 2}
    assert merged.source_ts == "2026-11-02T04:00:27Z"  # the broker's freshest update, not NSE's (there is none)
    assert merged.notes == ["NSE live feed unavailable: the Kotak broker feed alone was used"]


def test_neither_feed_gives_an_empty_table():
    merged = merge_live(None, None, configured=True)
    assert merged.rows == [] and merged.source_ts is None and merged.broker == "failed"
    assert merge_live(None, None, configured=False).broker == "off"


def test_during_the_session_a_lagging_broker_feed_is_not_used_at_all():
    stale = broker(row("A", volume=5000.0), row("B"), asof=NOW - timedelta(seconds=240))
    merged = merge_live(nse(row("A"), row("B")), stale, configured=True, lag_limit_sec=90, judge_lag=True)
    assert merged.broker == "lagging" and {r["feed"] for r in merged.rows} == {"nse"}
    assert merged.notes == ["Kotak live feed not used: its newest trade was 240 s old"]
    assert merged.source_ts == NSE_STAMP and merged.cross_checked == 0


def test_a_feed_within_the_limit_outside_the_session_or_with_the_check_off_is_used():
    stale = broker(row("A"), asof=NOW - timedelta(seconds=240))
    assert merge_live(nse(row("A")), stale, configured=True, lag_limit_sec=90, judge_lag=False).broker == "used"  # the evening: final figures
    assert merge_live(nse(row("A")), stale, configured=True, lag_limit_sec=0, judge_lag=True).broker == "used"  # 0 = no check
    fresh = broker(row("A"), asof=NOW - timedelta(seconds=89))
    assert merge_live(nse(row("A")), fresh, configured=True, lag_limit_sec=90, judge_lag=True).broker == "used"


def test_a_broker_answer_with_no_trade_time_cannot_be_judged_and_is_not_used_during_the_session():
    unstamped = broker(row("A", row_ts=None), asof=None)
    merged = merge_live(nse(row("A")), unstamped, configured=True, lag_limit_sec=90, judge_lag=True)
    assert merged.broker == "unstamped" and merged.notes == ["Kotak live feed not used: its quotes carry no trade time"]
    assert merge_live(nse(row("A")), unstamped, configured=True, lag_limit_sec=90, judge_lag=False).broker == "used"


def test_the_stamp_is_that_of_the_rows_taken_not_of_a_feed_whose_rows_all_lost():
    newer = broker(row("A", volume=2000.0, row_ts="2026-11-02T04:00:20Z"), asof=datetime(2026, 11, 2, 4, 0, 20, tzinfo=UTC))
    assert merge_live(nse(row("A")), newer, configured=True).source_ts == "2026-11-02T04:00:20Z"
    # the broker answered but NSE's rows were all the fresher ones: NSE's stamp stands, not the broker's later "as of"
    only_nse = broker(row("A", volume=10.0, row_ts="2026-11-02T04:00:29Z"), asof=datetime(2026, 11, 2, 4, 0, 29, tzinfo=UTC))
    assert merge_live(nse(row("A")), only_nse, configured=True).source_ts == NSE_STAMP
    # only the newest of the broker rows actually taken counts: A lost (less volume), B is the broker's alone
    mixed = broker(row("A", volume=10.0, row_ts="2026-11-02T04:00:29Z"), row("B", row_ts="2026-11-02T04:00:12Z"),
                   asof=datetime(2026, 11, 2, 4, 0, 29, tzinfo=UTC))
    assert merge_live(nse(row("A")), mixed, configured=True).source_ts == "2026-11-02T04:00:12Z"


def test_a_stamp_from_the_future_does_not_make_the_table_look_fresher():
    skewed = broker(row("A", volume=2000.0, row_ts="2026-11-02T04:00:50Z"), asof=NOW)  # seconds of clock skew: at most the moment of the answer
    assert merge_live(nse(row("A")), skewed, configured=True).source_ts == "2026-11-02T04:00:30Z"
    absurd = broker(row("A", volume=2000.0, row_ts="2026-11-02T05:30:00Z"), asof=NOW)  # an hour and a half ahead is a fault: not believed
    assert merge_live(nse(row("A")), absurd, configured=True).source_ts == NSE_STAMP


DAY = date(2026, 11, 2)
YESTERDAY = "2026-10-30T09:50:00Z"  # the last trade of a name that has not traded today


def test_a_stale_row_of_one_feed_cannot_conflict_with_or_outvote_a_current_row_of_the_other():
    yesterday = row("A", volume=9_000_000.0, last=130.0, prev_close=125.0, open=127.0, row_ts=YESTERDAY)  # a whole day's volume, another price
    merged = merge_live(nse(row("A"), row("B")), broker(yesterday, row("B")), configured=True, day=DAY)
    got = by_symbol(merged)
    assert got["A"]["feed"] == "nse" and got["A"]["volume"] == 1000.0 and merged.conflicts == {}
    assert merged.cross_checked == 1 and merged.stale == 0  # the name is covered by NSE: nothing is "left out"
    reverse = merge_live(nse(yesterday, row("B")), broker(row("A"), row("B")), configured=True, day=DAY)
    assert by_symbol(reverse)["A"]["feed"] == "kotak" and reverse.conflicts == {}


def test_names_without_a_usable_row_in_either_feed_are_counted_as_stale_or_incomplete():
    merged = merge_live(
        nse(row("OLD", row_ts=YESTERDAY), row("BOTHOLD", row_ts=YESTERDAY), row("HOLE", high=None), row("MIXED", row_ts=YESTERDAY), row("OK")),
        broker(row("BOTHOLD", row_ts=YESTERDAY), row("HOLE", row_ts=YESTERDAY), row("MIXED", volume=None), row("OK")), configured=True, day=DAY)
    assert set(by_symbol(merged)) == {"OK"}
    # HOLE: today's row with a figure missing outranks a stale one; MIXED likewise; OLD and BOTHOLD only ever had stale rows
    assert (merged.stale, merged.incomplete) == (2, 2)


def test_without_a_day_the_date_is_not_judged_but_completeness_is():
    merged = merge_live(nse(row("OLD", row_ts=YESTERDAY), row("HOLE", last=None)), None, configured=False)
    assert set(by_symbol(merged)) == {"OLD"} and (merged.stale, merged.incomplete) == (0, 1)
