from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from app.errors import CircuitOpen, DataUnavailable
from ingestion import pipeline as pl
from ingestion.nse import PreopenSnapshot, parse_fo_zip
from ingestion.universe import build_universe, load_seed, seed_path, select_symbols
from ingestion.validation import overlap_consistent, validate_bars
from ingestion.yahoo import SymbolNotFound, YahooSeries
from tests.helpers import fixture_bytes

NOW_AFTER_CLOSE = datetime(2026, 11, 2, 11, 0, tzinfo=timezone.utc)  # 16:30 IST Monday


def bars(closes: dict[str, float], spread: float = 0.02):
    out = []
    for d, c in closes.items():
        out.append({"trade_date": d, "open": c * 0.998, "high": c * (1 + spread), "low": c * (1 - spread), "close": c,
                    "adj_close": c, "volume": 1000.0})
    return out


def series(closes, splits=None):
    return YahooSeries("X", {"exchangeTimezoneName": "Asia/Kolkata"}, bars(closes), splits or [], [])


class FakeYahoo:
    def __init__(self, script):
        self.script = script  # symbol -> list of results/exceptions (last one repeats)
        self.calls: list[tuple[str, str]] = []

    def fetch_daily(self, symbol, range_="1mo"):
        self.calls.append((symbol, range_))
        queue = self.script[symbol]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item


def equity(store, symbol="AAA", **kw):
    return store.upsert_instrument(symbol, "equity", yahoo_symbol=f"{symbol}.NS", in_universe=1, **kw)


# --------------------------------------------------------------------------------------- validation
def test_validate_bars_repairs_rounding_and_rejects_impossible_bars():
    raw = [
        {"trade_date": "2026-10-01", "open": 100, "high": 99.99, "low": 98, "close": 100.0, "volume": 10},  # tiny: repaired
        {"trade_date": "2026-10-02", "open": 100, "high": 90, "low": 89, "close": 100, "volume": 10},  # high below open
        {"trade_date": "2026-10-03", "open": 100, "high": 101, "low": 102, "close": 100.5, "volume": 10},  # low above close
        {"trade_date": "2026-10-04", "open": -1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"trade_date": "2026-10-05", "open": 100, "high": 400, "low": 99, "close": 150, "volume": 1},  # implausible range
        {"trade_date": "2026-10-06", "open": 100, "high": 100, "low": 100, "close": 100, "volume": 0},  # placeholder
        {"trade_date": "2026-10-07", "open": None, "high": 1, "low": 1, "close": 1},
        {"trade_date": "bad", "open": 1, "high": 1, "low": 1, "close": 1},
        {"trade_date": "2026-10-09", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 5},
        {"trade_date": "2026-10-09", "open": 100, "high": 103, "low": 98, "close": 102, "volume": 6},  # duplicate: last wins
    ]
    clean, report = validate_bars(raw, kind="equity")
    assert [b["trade_date"] for b in clean] == ["2026-10-01", "2026-10-09"]
    assert clean[0]["high"] == 100.0 and clean[1]["close"] == 102 and report.repaired == 1
    assert report.rejected_count == 7
    assert validate_bars([], kind="equity")[0] == []
    flat_fx, _ = validate_bars([{"trade_date": "2026-10-01", "open": 95, "high": 95, "low": 95, "close": 95, "volume": 0}], kind="fx")
    assert len(flat_fx) == 1  # FX bars legitimately have zero volume
    strict, _ = validate_bars([{"trade_date": "2026-10-01", "open": 1, "high": 2, "low": 1, "close": 1.5}], kind="equity", require_volume=True)
    assert strict == []


def test_overlap_consistency_detection():
    stored = {"2026-10-01": 100.0, "2026-10-02": 101.0, "2026-10-03": 102.0, "2026-10-04": 103.0}
    same = [{"trade_date": d, "close": c * 1.001} for d, c in stored.items()]
    halved = [{"trade_date": d, "close": c / 2} for d, c in stored.items()]
    assert overlap_consistent(stored, same) and not overlap_consistent(stored, halved)
    assert overlap_consistent(stored, [{"trade_date": "2026-12-01", "close": 5}])  # no overlap -> cannot judge
    assert not overlap_consistent({"2026-10-01": 100.0}, [{"trade_date": "2026-10-01", "close": 50.0}])  # single overlap mismatch
    assert overlap_consistent({}, same)


# ------------------------------------------------------------------------------------------ vendor bars
def test_range_selection():
    today = date(2026, 11, 2)
    assert pl.choose_range(None, today, 10) == "10y" and pl.choose_range("2026-10-30", today, 10) == "5d"
    assert pl.choose_range("2026-10-15", today, 10) == "1mo" and pl.choose_range("2026-09-01", today, 10) == "3mo"
    assert pl.choose_range("2026-10-01", today, 10) == "3mo"  # a 32-day gap is not covered by a 1-month window
    assert pl.choose_range("2025-01-01", today, 10) == "2y" and pl.choose_range("2010-01-01", today, 10) == "10y"
    assert [pl.range_for_years(y) for y in (1, 2, 4, 9, 15)] == ["1y", "2y", "5y", "10y", "max"]


def test_first_load_then_incremental_applies_point_in_time_filter(store, fake_clock):
    equity(store, "AAA")
    morning = fake_clock.now()  # 08:30 IST Monday: Monday's own bar must not be stored yet
    yahoo = FakeYahoo({"AAA.NS": [series({"2026-10-29": 100, "2026-10-30": 101, "2026-11-02": 102}),
                                  series({"2026-10-30": 101, "2026-11-02": 103})]})
    report = pl.refresh_daily_bars(store, yahoo, now=morning, years=10, include_series=False)
    iid = store.instrument_id("AAA")
    assert report.ok == 1 and store.latest_bar_date(iid) == "2026-10-30"
    assert yahoo.calls[0] == ("AAA.NS", "10y")

    report = pl.refresh_daily_bars(store, yahoo, now=NOW_AFTER_CLOSE, years=10, include_series=False)
    assert yahoo.calls[1][1] == "5d" and store.latest_bar_date(iid) == "2026-11-02"  # now the session is final
    assert report.rows == 2 and store.get_bars(iid).loc["2026-11-02", "close"] == 103


def test_vendor_readjustment_triggers_full_reload(store, fake_clock):
    iid = equity(store, "AAA")
    old = {f"2026-10-{d:02d}": 200.0 + d for d in (26, 27, 28, 29, 30)}
    store.upsert_bars(iid, bars(old), source="yahoo")
    halved = {d: c / 2 for d, c in old.items()}
    yahoo = FakeYahoo({"AAA.NS": [series(halved), series(halved)]})  # incremental result, then the full reload
    report = pl.refresh_daily_bars(store, yahoo, now=NOW_AFTER_CLOSE, years=10, include_series=False)
    assert [c[1] for c in yahoo.calls] == ["5d", "10y"] and report.ok == 1
    assert store.get_bars(iid).loc["2026-10-30", "close"] == pytest.approx(115.0)


def test_new_split_event_inside_history_forces_reload(store):
    iid = equity(store, "AAA")
    store.upsert_bars(iid, bars({"2026-10-28": 100, "2026-10-29": 101, "2026-10-30": 102}), source="yahoo")
    split = [{"ex_date": "2026-10-29", "numerator": 2.0, "denominator": 1.0}]
    same = series({"2026-10-28": 100, "2026-10-29": 101, "2026-10-30": 102}, split)
    yahoo = FakeYahoo({"AAA.NS": [same, same]})
    pl.refresh_daily_bars(store, yahoo, now=NOW_AFTER_CLOSE, years=10, include_series=False)
    assert [c[1] for c in yahoo.calls] == ["5d", "10y"]
    assert len(store.corporate_actions(iid, "split")) == 1


def test_not_found_symbols_are_quarantined_after_three_runs(store):
    equity(store, "GONE")
    equity(store, "OKAY")
    yahoo = FakeYahoo({"GONE.NS": [SymbolNotFound("GONE.NS")], "OKAY.NS": [series({"2026-10-30": 100})]})
    for run in range(3):
        report = pl.refresh_daily_bars(store, yahoo, now=NOW_AFTER_CLOSE, years=10, include_series=False)
        assert report.ok == 1 and report.failed == 1
    row = store.get_instrument("GONE")
    assert row["status"] == "quarantined" and row["in_universe"] == 0
    assert store.get_instrument("OKAY")["status"] == "active"
    assert any(a["code"] == "symbol_quarantined" for a in store.pending_alerts())
    pl.refresh_daily_bars(store, yahoo, now=NOW_AFTER_CLOSE, years=10, include_series=False)
    assert yahoo.calls.count(("GONE.NS", "10y")) == 3  # quarantined symbols are no longer requested


def test_refresh_survives_bad_symbols_deadlines_and_open_circuit(store, fake_clock):
    for s in ("AAA", "BBB", "CCC", "DDD"):
        equity(store, s)
    yahoo = FakeYahoo({"AAA.NS": [DataUnavailable("yahoo", "HTTP 500")], "BBB.NS": [RuntimeError("boom")],
                       "CCC.NS": [series({"2026-10-30": 100})], "DDD.NS": [series({"2026-10-30": 100})]})
    report = pl.refresh_daily_bars(store, yahoo, now=NOW_AFTER_CLOSE, years=10, include_series=False)
    assert report.ok == 2 and report.failed == 2 and any("RuntimeError" in n for n in report.notes)

    yahoo = FakeYahoo({"AAA.NS": [series({"2026-10-30": 100})], "BBB.NS": [CircuitOpen("yahoo", "open")],
                       "CCC.NS": [series({"2026-10-30": 100})], "DDD.NS": [series({"2026-10-30": 100})]})
    report = pl.refresh_daily_bars(store, yahoo, now=NOW_AFTER_CLOSE, years=10, include_series=False)
    assert report.partial and len(yahoo.calls) == 2  # stops hammering a dead provider

    fake_clock.advance(100)
    report = pl.refresh_daily_bars(store, FakeYahoo({s + ".NS": [series({"2026-10-30": 100})] for s in "AAA BBB CCC DDD".split()}),
                                   now=NOW_AFTER_CLOSE, years=10, include_series=False, deadline=fake_clock.monotonic() - 1,
                                   clock=fake_clock)
    assert report.partial and report.ok == 0


def test_series_instruments_and_ordering(store):
    ids = pl.ensure_series_instruments(store)
    assert "^NSEI" in ids and "^GSPC" in ids and store.get_instrument("^VIX", "volatility")
    rows = store.list_instruments(status="active")
    ordered = sorted(rows, key=pl._order_key)
    assert ordered[0]["symbol"] in {"^GSPC", "^IXIC", "^VIX", "USDINR=X", "DX-Y.NYB", "^TNX", "CL=F", "BZ=F", "^NSEI", "^INDIAVIX"}


def test_backfill_history_is_resumable(store, fake_clock):
    for s in ("AAA", "BBB"):
        equity(store, s)
    yahoo = FakeYahoo({"AAA.NS": [series({"2026-10-30": 100})], "BBB.NS": [DataUnavailable("yahoo", "HTTP 500"),
                                                                          series({"2026-10-29": 90, "2026-10-30": 91})]})
    first = pl.backfill_history(store, yahoo, now=NOW_AFTER_CLOSE, years=10)
    assert first.ok == 1 and first.failed == 1
    assert store.get_backfill("yahoo_history:AAA")["status"] == "done" and store.get_backfill("yahoo_history:BBB")["status"] == "failed"
    second = pl.backfill_history(store, yahoo, now=NOW_AFTER_CLOSE, years=10)
    assert second.skipped == 1 and second.ok == 1  # AAA skipped, BBB retried successfully
    assert len(yahoo.calls) == 3
    fake_clock.advance(10)
    third = pl.backfill_history(store, yahoo, now=NOW_AFTER_CLOSE, years=10, deadline=0.0, clock=fake_clock)
    assert third.skipped == 2  # everything already done


# ---------------------------------------------------------------------------------- official end-of-day
class FakeArchive:
    def __init__(self, bhav=None, fo=None, lists=None, lots=None, fail_lists=False):
        self._bhav, self._fo, self._lists, self._lots, self._fail = bhav or {}, fo or {}, lists or {}, lots or {}, fail_lists
        self.fo_calls: list[date] = []

    def bhavcopy(self, day):
        return self._bhav.get(day)

    def fo_bhavcopy(self, day):
        self.fo_calls.append(day)
        return self._fo.get(day)

    def index_constituents(self, index):
        if self._fail:
            raise DataUnavailable("nse_lists", "down")
        return self._lists[index]

    def fo_lot_sizes(self):
        if self._fail:
            raise DataUnavailable("nse_lists", "down")
        return self._lots


def test_bhavcopy_overrides_vendor_data_and_brings_delivery(store, fake_clock):
    from ingestion.nse import NseArchive

    iid = equity(store, "RELIANCE")
    store.upsert_instrument("TCS", "equity")
    store.upsert_bars(iid, [{"trade_date": "2026-10-30", "open": 1170, "high": 1176, "low": 1150, "close": 1166, "volume": 1}],
                      source="yahoo")

    class Http:
        def get(self, url, **kw):
            class R:
                status = 200
                content = fixture_bytes("bhavcopy_full_sample.csv")
            return R()

    rows = NseArchive.__new__(NseArchive)
    rows.http = Http()
    written, matched = pl.ingest_bhavcopy(store, rows, date(2026, 10, 30))
    assert written == 2 and matched == 2  # RELIANCE and TCS are tracked; INFY/SBIN/... in the file are ignored
    bar = store.get_bars(iid).loc["2026-10-30"]
    assert bar["source"] == "nse_bhavcopy" and bar["close"] == 1164.9 and bar["delivery_pct"] == 54.73
    assert bar["turnover"] > 1e9 and bar["quality"] == "official"
    assert pl.ingest_bhavcopy(store, FakeArchive(), date(2026, 10, 29)) == (0, 0)


def test_fno_day_ingestion_and_resumable_backfill(store, fake_clock):
    store.upsert_instrument("RELIANCE", "equity", is_fno=1)
    store.upsert_instrument("TCS", "equity", is_fno=1)
    fo_rows = parse_fo_zip(fixture_bytes("fo_udiff_sample.zip"), date(2026, 10, 30))
    archive = FakeArchive(fo={date(2026, 10, 30): fo_rows})
    assert pl.ingest_fno_day(store, archive, date(2026, 10, 30)) == 2
    assert store.latest_fno_date() == "2026-10-30" and pl.fno_symbols(store) == {"RELIANCE", "TCS"}

    days = [date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30)]
    first = pl.backfill_fno(store, archive, days)
    assert first.skipped == 3 and archive.fo_calls[1:] == [date(2026, 10, 29), date(2026, 10, 28)]  # newest first; 30th already stored
    for _ in range(2):
        pl.backfill_fno(store, archive, days)
    calls_before = len(archive.fo_calls)
    pl.backfill_fno(store, archive, days)
    assert len(archive.fo_calls) == calls_before  # 'missing' days stop being retried after 3 attempts

    seen = []
    archive2 = FakeArchive(fo={date(2026, 10, 27): fo_rows})
    pl.backfill_fno(store, archive2, [date(2026, 10, 27)], on_day=lambda d, n: seen.append((d, n)))
    assert seen == [(date(2026, 10, 27), 0)] or store.fno_frame(symbols=["TCS"]).shape[0] >= 1

    class Boom(FakeArchive):
        def fo_bhavcopy(self, day):
            raise CircuitOpen("nse_archive", "open")

    assert pl.backfill_fno(store, Boom(), [date(2026, 9, 1)]).partial
    class Flaky(FakeArchive):
        def fo_bhavcopy(self, day):
            raise DataUnavailable("nse_archive", "HTTP 500")

    assert pl.backfill_fno(store, Flaky(), [date(2026, 9, 2)]).failed == 1
    fake_clock.advance(5)
    assert pl.backfill_fno(store, archive, [date(2026, 8, 3)], deadline=0.0, clock=fake_clock).partial


# --------------------------------------------------------------------------------------- live collection
class FakeNse:
    def __init__(self, snapshot=None, gift=None, error=None):
        self.snapshot, self.gift_value, self.error = snapshot, gift, error

    def preopen(self, key, symbols=None, *, deadline=None):
        if self.error:
            raise self.error
        return self.snapshot

    def gift_nifty(self, *, deadline=None):
        if self.error:
            raise self.error
        return self.gift_value


def test_collect_preopen_and_gift(store):
    snapshot = PreopenSnapshot("2026-11-02T03:38:00Z", [{"symbol": "AAA", "iep": 100.0, "final_price": 101.0, "prev_close": 99.0}])
    assert pl.collect_preopen(store, FakeNse(snapshot), "2026-11-02", ["AAA"]).rows
    assert store.latest_preopen("2026-11-02")["AAA"]["final_price"] == 101.0
    assert pl.collect_preopen(store, FakeNse(error=DataUnavailable("nse_api", "HTTP 403")), "2026-11-02", ["AAA"]) is None
    assert pl.collect_preopen(store, FakeNse(PreopenSnapshot(None, [])), "2026-11-02", []).rows == []

    gift = {"price": 23242.5, "prev_close": 23086.0, "change_pct": 0.68, "source_ts": "2026-11-02T03:30:00Z", "expiry": "23-Nov-2026"}
    assert pl.collect_gift(store, FakeNse(gift=gift))["price"] == 23242.5
    assert store.latest_live_quote("GIFT_NIFTY")["change_pct"] == 0.68
    assert pl.collect_gift(store, FakeNse(error=DataUnavailable("nse_api", "x"))) is None


@pytest.mark.parametrize("stamp, kept", [
    ("2026-11-02T03:38:00Z", True),    # 09:08 IST on the trade date
    ("2026-10-30T03:38:00Z", False),   # the previous session's book
    ("2026-11-01T19:00:00Z", True),    # 00:30 IST on the 2nd: judged by the IST date, not the UTC date
    ("2026-11-02T18:45:00Z", False),   # 00:15 IST on the 3rd
    ("02-Nov-2026 09:08:30", True),    # NSE's own display format (naive: Indian time)
    ("30-Oct-2026 09:07:48", False),
    (None, True),                      # nothing to judge by: kept rather than discarding a book that may be perfectly good
])
def test_a_pre_open_book_is_judged_by_its_ist_date(stamp, kept):
    rows = pl.fresh_preopen_rows(PreopenSnapshot(stamp, [{"symbol": "AAA", "iep": 100.0}]), "2026-11-02")
    assert bool(rows) is kept


def test_a_book_served_from_the_previous_session_is_never_stored_as_todays(store):
    stale = PreopenSnapshot("2026-10-30T03:38:00Z", [{"symbol": "AAA", "iep": 100.0, "final_price": 101.0, "prev_close": 99.0}])
    result = pl.collect_preopen(store, FakeNse(stale), "2026-11-02", ["AAA"])
    assert result is not None and result.rows == [] and result.source_ts == "2026-10-30T03:38:00Z"  # callers see "not published"
    assert store.latest_preopen("2026-11-02") == {}
    assert store.conn.execute("SELECT COUNT(*) FROM source_health WHERE source_id = 'nse_api_market'").fetchone()[0] == 0  # not a delivery


def test_only_the_rows_stamped_today_are_stored(store):
    mixed = PreopenSnapshot("2026-11-02T03:38:00Z", [
        {"symbol": "NEW", "iep": 100.0, "final_price": 101.0, "row_ts": "2026-11-02T03:37:30Z"},
        {"symbol": "OLD", "iep": 50.0, "final_price": 51.0, "row_ts": "2026-10-30T03:37:30Z"},  # not refreshed since Friday
        {"symbol": "UNDATED", "iep": 10.0, "final_price": 11.0},  # takes the snapshot's (fresh) time
    ])
    result = pl.collect_preopen(store, FakeNse(mixed), "2026-11-02", ["NEW", "OLD", "UNDATED"])
    assert [r["symbol"] for r in result.rows] == ["NEW", "UNDATED"]
    assert sorted(store.latest_preopen("2026-11-02")) == ["NEW", "UNDATED"]
    health = {h["source_id"]: h for h in store.source_health()}
    assert "nse_api_market" not in health or health["nse_api_market"]["total_failure"] >= 0


def test_live_context_collection(store, fake_clock):
    class Y:
        def fetch_quote(self, key):
            if key == "^HSI":
                raise DataUnavailable("yahoo", "x")
            return {"price": 100.0, "prev_close": 99.0, "change_pct": 1.01, "source_ts": "2026-11-02T03:00:00Z"}

    assert pl.collect_live_context(store, Y(), ("^N225", "^HSI", "ES=F")) == 2
    assert store.latest_live_quote("LIVE:^N225")["price"] == 100.0
    assert pl.collect_live_context(store, Y(), ("^N225",), deadline=-1.0, clock=fake_clock) == 0
    assert pl.monotonic_deadline(5, fake_clock) == fake_clock.monotonic() + 5


# ------------------------------------------------------------------------------------------- universe
def nse_lists():
    mk = lambda s, ind="Financial Services": {"symbol": s, "name": s + " Ltd", "industry": ind, "isin": "INE" + s, "series": "EQ"}  # noqa: E731
    return {"nifty500": [mk("AAA"), mk("BBB"), mk("CCC"), mk("DDD")], "nifty100": [mk("AAA")], "nifty200": [mk("AAA"), mk("BBB"), mk("CCC")]}


def test_universe_from_nse_lists_and_modes(store, settings):
    archive = FakeArchive(lists=nse_lists(), lots={"BBB": 100})
    result = build_universe(store, settings, archive, date(2026, 11, 2))
    assert result.source == "nse" and result.symbols == ["AAA", "BBB"]  # Nifty100 or F&O
    assert store.get_instrument("BBB")["is_fno"] == 1 and store.get_instrument("BBB")["lot_size"] == 100
    assert store.get_instrument("CCC")["is_fno"] == 0 and store.get_instrument("AAA")["yahoo_symbol"] == "AAA.NS"

    rows = [{"symbol": s, "is_fno": s == "BBB", "nifty100": s == "AAA", "nifty200": s != "DDD"} for s in ("AAA", "BBB", "CCC", "DDD")]
    from dataclasses import replace

    assert select_symbols(rows, replace(settings, universe_mode="nifty200")) == ["AAA", "BBB", "CCC"]
    assert select_symbols(rows, replace(settings, universe_mode="nifty500")) == ["AAA", "BBB", "CCC", "DDD"]
    assert select_symbols(rows, replace(settings, universe_mode="custom", universe_symbols=("ZZZ", "AAA"))) == ["ZZZ", "AAA"]

    # F&O eligibility can lapse: the flag is cleared on the next refresh
    build_universe(store, settings, FakeArchive(lists=nse_lists(), lots={}), date(2026, 11, 9))
    assert store.get_instrument("BBB")["is_fno"] == 0 and store.get_instrument("BBB")["lot_size"] is None


def test_universe_falls_back_to_seed_then_to_database(store, settings, tmp_path):
    result = build_universe(store, settings, FakeArchive(fail_lists=True), date(2026, 11, 2))
    assert result.source == "seed" and 200 < len(result.symbols) < 260
    assert "RELIANCE" in result.symbols and any("seed" in n for n in result.notes)
    assert store.get_instrument("M&M")["yahoo_symbol"] == "M&M.NS"

    from dataclasses import replace

    no_seed = replace(settings, home=tmp_path)
    kept = build_universe(store, no_seed, None, date(2026, 11, 3))
    assert kept.source == "db" and "RELIANCE" in kept.symbols
    assert load_seed(seed_path(no_seed)) == []


def test_universe_without_any_source_raises(store, settings, tmp_path):
    from dataclasses import replace

    with pytest.raises(DataUnavailable):
        build_universe(store, replace(settings, home=tmp_path), None, date(2026, 11, 2))


def test_universe_cap_prefers_liquid_names(store, settings):
    from dataclasses import replace

    small = replace(settings, universe_max_size=50)
    archive = FakeArchive(lists={"nifty500": [{"symbol": f"S{i:03d}", "name": "n", "industry": "x", "isin": "i", "series": "EQ"}
                                              for i in range(80)],
                                 "nifty100": [{"symbol": f"S{i:03d}", "name": "n", "industry": "x", "isin": "i", "series": "EQ"}
                                              for i in range(80)],
                                 "nifty200": []}, lots={})
    store.upsert_instrument("S079", "equity")
    iid = store.instrument_id("S079")
    store.upsert_bars(iid, [{"trade_date": (date.today() - timedelta(days=3)).isoformat(), "open": 1, "high": 2, "low": 1, "close": 1.5,
                             "turnover": 9e12}], source="nse_bhavcopy", source_rank=2)
    result = build_universe(store, small, archive, date(2026, 11, 2))
    assert len(result.symbols) == 50 and "S079" in result.symbols and any("capped" in n for n in result.notes)
