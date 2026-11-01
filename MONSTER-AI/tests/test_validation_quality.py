"""Bars that look wrong are never dropped silently: impossible ones are rejected loudly (WARNING with the count and the first
reasons, plus a per-symbol counter); extreme but internally consistent days - exactly the days this system targets - are kept
with quality ``suspect``. A vendor reload fetches first and replaces the vendor rows atomically."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from app.errors import DataUnavailable
from database.store import Store
from ingestion import pipeline as pl
from ingestion.validation import MAX_RANGE_FRACTION, SUSPECT, validate_bars
from ingestion.yahoo import YahooSeries

NOW_AFTER_CLOSE = datetime(2026, 11, 2, 11, 0, tzinfo=timezone.utc)  # 16:30 IST Monday: every session up to Monday is final
WEEK = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]


def vbar(day, close=100.0, spread=0.02, volume=1000.0, **override):
    row = {"trade_date": day, "open": close * 0.998, "high": close * (1 + spread), "low": close * (1 - spread), "close": close,
           "adj_close": close, "volume": volume}
    row.update(override)
    return row


def series(bars, splits=None):
    return YahooSeries("X", {"exchangeTimezoneName": "Asia/Kolkata"}, bars, splits or [], [])


class FakeYahoo:
    def __init__(self, script):
        self.script = script  # symbol -> results / exceptions in call order (the last one repeats)
        self.calls: list[tuple[str, str]] = []

    def fetch_daily(self, symbol, range_="1mo"):
        self.calls.append((symbol, range_))
        queue = self.script[symbol]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item


def equity(store, symbol="AAA"):
    store.upsert_instrument(symbol, "equity", yahoo_symbol=f"{symbol}.NS", in_universe=1)
    return store.get_instrument(symbol)


def messages(caplog, level):
    return [r.getMessage() for r in caplog.records if r.levelno == level]


# ============================================================================================== validation
IMPOSSIBLE = [
    ({"open": -1, "high": 1, "low": 1, "close": 1, "volume": 5}, "non-positive price"),
    ({"open": 0, "high": 101, "low": 99, "close": 100, "volume": 5}, "non-positive price"),
    ({"open": 100, "high": 90, "low": 89, "close": 100, "volume": 5}, "high below open/close"),
    ({"open": 100, "high": 101, "low": 102, "close": 100.5, "volume": 5}, None),  # high below low
    ({"open": 100, "high": 105, "low": 101, "close": 103, "volume": 5}, "low above open/close"),  # open below the low
    ({"open": 100, "high": 105, "low": 99, "close": 110, "volume": 5}, "high below open/close"),  # close above the high
    ({"open": 100, "high": 105, "low": 99, "close": 101, "volume": -3}, "negative volume"),
    ({"open": None, "high": 105, "low": 99, "close": 101, "volume": 5}, "missing OHLC"),
]


@pytest.mark.parametrize("keep_extreme", [False, True])
@pytest.mark.parametrize("bar, reason", IMPOSSIBLE)
def test_impossible_bars_are_always_rejected(bar, reason, keep_extreme):
    clean, report = validate_bars([{"trade_date": "2026-10-05", **bar}], keep_extreme=keep_extreme)
    assert clean == [] and report.rejected_count == 1 and report.suspect == []
    assert reason is None or report.rejected == [("2026-10-05", reason)]


def test_a_negative_volume_is_impossible_for_equities():
    clean, report = validate_bars([{"trade_date": "2026-10-05", "open": 100, "high": 101, "low": 99, "close": 100, "volume": -1}],
                                  kind="equity", keep_extreme=True)
    assert clean == [] and report.rejected == [("2026-10-05", "negative volume")]


@pytest.mark.parametrize("kind", ["index", "global", "fx", "commodity", "rate", "volatility"])
def test_a_junk_volume_of_another_series_costs_the_volume_not_the_price_bar(kind):
    """Only equity features use volume: dropping a critical global bar (a yield, crude oil) for a junk volume would cost more."""
    clean, report = validate_bars([{"trade_date": "2026-10-05", "open": 100, "high": 101, "low": 99, "close": 100, "volume": -1}],
                                  kind=kind, keep_extreme=True)
    assert len(clean) == 1 and clean[0]["volume"] is None and clean[0]["close"] == 100
    assert report.rejected == [] and report.repaired == 1


EXTREME = {"trade_date": "2026-10-05", "open": 100, "high": 400, "low": 99, "close": 150, "volume": 1}  # +304 % range, consistent


def test_an_extreme_but_consistent_day_is_rejected_by_default_and_kept_as_suspect_on_request():
    clean, report = validate_bars([EXTREME])
    assert clean == [] and report.rejected == [("2026-10-05", "implausible range")] and report.suspect == []  # legacy default

    clean, report = validate_bars([EXTREME], keep_extreme=True)
    assert [b["trade_date"] for b in clean] == ["2026-10-05"] and clean[0]["quality"] == SUSPECT
    assert (clean[0]["open"], clean[0]["high"], clean[0]["low"], clean[0]["close"]) == (100.0, 400.0, 99.0, 150.0)  # unchanged
    assert report.rejected == [] and report.suspect == [("2026-10-05", "implausible range")]
    assert report.accepted == 1 and report.suspect_count == 1


def test_a_range_exactly_at_the_bound_is_not_extreme():
    at_limit = {"trade_date": "2026-10-05", "open": 100, "high": 160, "low": 100, "close": 150, "volume": 1}
    clean, report = validate_bars([at_limit], keep_extreme=True)
    assert len(clean) == 1 and "quality" not in clean[0] and report.suspect == []


@pytest.mark.parametrize("kind", sorted(MAX_RANGE_FRACTION))
def test_the_bound_belongs_to_the_kind_and_extreme_days_of_every_kind_are_kept(kind):
    limit = MAX_RANGE_FRACTION[kind]
    wide = {"trade_date": "2026-10-05", "open": 100.0, "high": 100.0 * (1 + limit + 0.2), "low": 100.0, "close": 101.0, "volume": 1}
    assert validate_bars([wide], kind=kind)[0] == []
    clean, report = validate_bars([wide], kind=kind, keep_extreme=True)
    assert len(clean) == 1 and clean[0]["quality"] == SUSPECT and report.suspect_count == 1


def test_a_later_duplicate_of_the_same_day_decides_whether_the_day_is_suspect():
    normal = {"trade_date": "2026-10-05", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 5}
    clean, report = validate_bars([EXTREME, normal], keep_extreme=True)
    assert len(clean) == 1 and "quality" not in clean[0] and report.suspect == []
    clean, report = validate_bars([normal, EXTREME], keep_extreme=True)
    assert clean[0]["quality"] == SUSPECT and report.suspect_count == 1


def test_rounding_noise_is_still_repaired_next_to_a_suspect_day():
    noisy = {"trade_date": "2026-10-04", "open": 100, "high": 99.99, "low": 98, "close": 100.0, "volume": 10}
    noisy_low = {"trade_date": "2026-10-03", "open": 100, "high": 101, "low": 100.02, "close": 100.5, "volume": 10}
    clean, report = validate_bars([noisy_low, noisy, EXTREME], keep_extreme=True)
    assert [b["trade_date"] for b in clean] == ["2026-10-03", "2026-10-04", "2026-10-05"]
    assert clean[0]["low"] == 100.0 and clean[1]["high"] == 100.0 and "quality" not in clean[1] and clean[2]["quality"] == SUSPECT
    assert report.repaired == 2 and report.suspect_count == 1 and report.rejected == []


# =================================================================================== vendor ingestion
def test_rejected_vendor_bars_are_logged_counted_and_extreme_days_are_kept_as_suspect(store, caplog):
    """Five bars, ONE impossible: the old code warned only above max(3, 20 %) rejections, so this vanished without a trace."""
    row = equity(store)
    bars = [vbar(WEEK[0]), vbar(WEEK[1], open=-1), vbar(WEEK[2], close=150.0, high=300.0, low=100.0), vbar(WEEK[3]), vbar(WEEK[4])]
    with caplog.at_level(logging.INFO, logger="ingestion.pipeline"):
        pl.ingest_symbol_daily(store, FakeYahoo({"AAA.NS": [series(bars)]}), row, now=NOW_AFTER_CLOSE, years=1)
    frame = store.get_bars(int(row["instrument_id"]))
    assert list(frame.index.strftime("%Y-%m-%d")) == [WEEK[0], WEEK[2], WEEK[3], WEEK[4]]  # the impossible day only is gone
    assert frame.loc[WEEK[2], "quality"] == "suspect" and frame.loc[WEEK[2], "high"] == 300.0
    assert set(frame.loc[frame.index != pd.Timestamp(WEEK[2]), "quality"]) == {"ok"}
    warning = messages(caplog, logging.WARNING)
    assert len(warning) == 1 and "AAA" in warning[0] and "1 of 5" in warning[0]
    assert f"{WEEK[1]}: non-positive price" in warning[0]
    assert any("suspect" in m and f"{WEEK[2]}" in m for m in messages(caplog, logging.INFO))
    assert store.kv_get("bars:rejected:AAA") == "1"


def test_the_rejection_counter_accumulates_per_symbol_and_a_clean_run_leaves_it_alone(store):
    a, b = equity(store, "AAA"), equity(store, "BBB")
    bad = series([vbar(WEEK[0]), vbar(WEEK[1], open=-1), vbar(WEEK[2], volume=-5)])
    yahoo = FakeYahoo({"AAA.NS": [bad], "BBB.NS": [series([vbar(WEEK[0])])]})
    pl.ingest_symbol_daily(store, yahoo, a, now=NOW_AFTER_CLOSE, years=1)
    assert store.kv_get("bars:rejected:AAA") == "2"
    pl.ingest_symbol_daily(store, yahoo, a, now=NOW_AFTER_CLOSE, years=1)
    assert store.kv_get("bars:rejected:AAA") == "4"
    pl.ingest_symbol_daily(store, yahoo, b, now=NOW_AFTER_CLOSE, years=1)
    assert store.kv_get("bars:rejected:BBB") is None  # nothing rejected: no key, no noise


def test_an_unreadable_counter_is_reset_with_a_warning_instead_of_failing_the_ingestion(store, caplog):
    row = equity(store)
    store.kv_set("bars:rejected:AAA", "not a number")
    with caplog.at_level(logging.WARNING, logger="ingestion.pipeline"):
        pl.ingest_symbol_daily(store, FakeYahoo({"AAA.NS": [series([vbar(WEEK[0]), vbar(WEEK[1], open=-1)])]}), row,
                               now=NOW_AFTER_CLOSE, years=1)
    assert store.kv_get("bars:rejected:AAA") == "1"
    assert any("unreadable rejection counter" in m for m in messages(caplog, logging.WARNING))
    assert store.latest_bar_date(int(row["instrument_id"])) == WEEK[0]  # the good bar was stored regardless


def test_only_the_first_reasons_are_quoted(store, caplog):
    row = equity(store)
    days = [f"2026-10-{d:02d}" for d in range(19, 27)]
    bars = [vbar(days[0])] + [vbar(d, open=-1) for d in days[1:]]  # seven impossible bars
    with caplog.at_level(logging.WARNING, logger="ingestion.pipeline"):
        pl.ingest_symbol_daily(store, FakeYahoo({"AAA.NS": [series(bars)]}), row, now=NOW_AFTER_CLOSE, years=1)
    (line,) = messages(caplog, logging.WARNING)
    assert "7 of 8" in line and days[1] in line and days[3] in line and days[4] not in line and line.rstrip(")").endswith("...")


def test_nothing_is_logged_when_everything_validates(store, caplog):
    row = equity(store)
    with caplog.at_level(logging.INFO, logger="ingestion.pipeline"):
        pl.ingest_symbol_daily(store, FakeYahoo({"AAA.NS": [series([vbar(d) for d in WEEK])]}), row, now=NOW_AFTER_CLOSE, years=1)
    assert not [r for r in caplog.records if r.name == "ingestion.pipeline"] and store.kv_get("bars:rejected:AAA") is None


def test_extreme_days_of_non_equity_series_are_kept_too(store):
    store.upsert_instrument("^VIX", "volatility", yahoo_symbol="^VIX")
    wide = vbar(WEEK[0], close=40.0, high=170.0, low=30.0)  # a 4x intraday range: beyond the 3.0 bound, but consistent
    pl.ingest_symbol_daily(store, FakeYahoo({"^VIX": [series([wide])]}), store.get_instrument("^VIX", "volatility"),
                           now=NOW_AFTER_CLOSE, years=1)
    assert store.get_bars(store.instrument_id("^VIX", "volatility")).loc[WEEK[0], "quality"] == "suspect"


# ===================================================================================== official file
class Archive:
    def __init__(self, rows):
        self.rows = rows

    def bhavcopy(self, day):
        return self.rows


def official(symbol, **fields):
    row = {"symbol": symbol, "series": "EQ", "trade_date": "2026-10-30", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
           "volume": 1000.0, "prev_close": None, "turnover": 1e7, "delivery_pct": 50.0, "vwap": 100.5, "trades": 10.0}
    row.update(fields)
    return row


def test_the_official_file_keeps_extreme_rows_as_suspect_and_reports_the_rejected_ones(store, caplog):
    for symbol in ("AAA", "BBB", "CCC", "DDD"):
        store.upsert_instrument(symbol, "equity")
    rows = [official("AAA"), official("BBB", open=-5.0), official("CCC", high=400.0, close=150.0),
            official("DDD", volume=-1.0), official("ZZZ")]  # ZZZ is not tracked
    with caplog.at_level(logging.INFO, logger="ingestion.pipeline"):
        written, matched = pl.ingest_bhavcopy(store, Archive(rows), date(2026, 10, 30))
    assert (written, matched) == (2, 4)
    quality = {s: store.get_bars(store.instrument_id(s)) for s in ("AAA", "CCC")}
    assert quality["AAA"].loc["2026-10-30", "quality"] == "official" and quality["CCC"].loc["2026-10-30", "quality"] == "suspect"
    assert quality["CCC"].loc["2026-10-30", "source_rank"] == 2 and quality["CCC"].loc["2026-10-30", "high"] == 400.0
    assert store.get_bars(store.instrument_id("BBB")).empty and store.get_bars(store.instrument_id("DDD")).empty
    (warning,) = messages(caplog, logging.WARNING)
    assert "2 of 4 rows rejected" in warning and "BBB: non-positive price" in warning and "DDD: negative volume" in warning
    assert store.kv_get("bars:rejected:BBB") == "1" and store.kv_get("bars:rejected:DDD") == "1"
    assert store.kv_get("bars:rejected:AAA") is None and any("suspect" in m and "CCC" in m for m in messages(caplog, logging.INFO))


# =============================================================================== reload is all or nothing
def stored_closes(store, row):
    return store.get_bars(int(row["instrument_id"]))["close"].round(6).tolist()


def seed_old_basis(store, row):
    store.upsert_bars(int(row["instrument_id"]), [vbar(d, close=200.0 + i) for i, d in enumerate(WEEK)], source="yahoo", source_rank=1)


SPLIT = [{"ex_date": "2026-10-28", "numerator": 2.0, "denominator": 1.0}]


def halved():
    return series([vbar(d, close=(200.0 + i) / 2) for i, d in enumerate(WEEK)], SPLIT)


def test_a_failed_reload_leaves_history_and_events_alone_and_the_next_run_retries(store):
    row = equity(store)
    seed_old_basis(store, row)
    before = stored_closes(store, row)
    failing = FakeYahoo({"AAA.NS": [halved(), DataUnavailable("yahoo", "boom")]})  # incremental ok, full reload fails
    with pytest.raises(DataUnavailable):
        pl.ingest_symbol_daily(store, failing, row, now=NOW_AFTER_CLOSE, years=1)
    assert stored_closes(store, row) == before and store.corporate_actions(int(row["instrument_id"]), "split") == []

    pl.ingest_symbol_daily(store, FakeYahoo({"AAA.NS": [halved(), halved()]}), row, now=NOW_AFTER_CLOSE, years=1)
    assert stored_closes(store, row) == [100.0, 100.5, 101.0, 101.5, 102.0]  # the retry still saw the split as new
    assert [r["ratio"] for r in store.corporate_actions(int(row["instrument_id"]), "split")] == [2.0]


def test_a_reload_that_yields_no_usable_bars_keeps_the_stored_history(store, caplog):
    row = equity(store)
    seed_old_basis(store, row)
    before = stored_closes(store, row)
    garbage = series([vbar(d, open=-1) for d in WEEK], SPLIT)
    with caplog.at_level(logging.WARNING, logger="ingestion.pipeline"):
        written = pl.ingest_symbol_daily(store, FakeYahoo({"AAA.NS": [halved(), garbage]}), row, now=NOW_AFTER_CLOSE, years=1)
    assert written == 0 and stored_closes(store, row) == before
    assert any("full reload returned no usable bars" in m for m in messages(caplog, logging.WARNING))
    assert store.corporate_actions(int(row["instrument_id"]), "split") == []


def test_the_reload_is_one_transaction(store, monkeypatch):
    row = equity(store)
    seed_old_basis(store, row)
    before = stored_closes(store, row)

    def boom(self, *args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Store, "add_corporate_action", boom)
    with pytest.raises(RuntimeError):
        pl.ingest_symbol_daily(store, FakeYahoo({"AAA.NS": [halved(), halved()]}), row, now=NOW_AFTER_CLOSE, years=1)
    assert stored_closes(store, row) == before  # the deleted vendor rows came back with the rollback


def test_a_reload_replaces_vendor_rows_only_and_records_the_split_with_it(store):
    row = equity(store)
    seed_old_basis(store, row)
    iid = int(row["instrument_id"])
    store.upsert_bars(iid, [vbar(WEEK[1], close=777.0)], source="nse_bhavcopy", source_rank=2, quality="official")
    yahoo = FakeYahoo({"AAA.NS": [halved(), halved()]})
    pl.ingest_symbol_daily(store, yahoo, row, now=NOW_AFTER_CLOSE, years=1)
    assert [c[1] for c in yahoo.calls] == ["5d", "1y"]
    frame = store.get_bars(iid)
    assert frame.loc[WEEK[1], "close"] == 777.0 and frame.loc[WEEK[1], "source"] == "nse_bhavcopy"  # the official row survives raw
    assert frame.loc[WEEK[0], "close"] == 100.0 and frame.loc[WEEK[0], "source"] == "yahoo"
    assert [(r["ex_date"], r["ratio"], r["source"]) for r in store.corporate_actions(iid, "split")] == [("2026-10-28", 2.0, "yahoo")]


def test_a_known_split_does_not_trigger_another_reload(store):
    row = equity(store)
    iid = int(row["instrument_id"])
    store.upsert_bars(iid, [vbar(d, close=(200.0 + i) / 2) for i, d in enumerate(WEEK)], source="yahoo", source_rank=1)
    store.add_corporate_action(iid, "split", "2026-10-28", ratio=2.0, source="yahoo")
    yahoo = FakeYahoo({"AAA.NS": [halved()]})
    pl.ingest_symbol_daily(store, yahoo, row, now=NOW_AFTER_CLOSE, years=1)
    assert [c[1] for c in yahoo.calls] == ["5d"]
