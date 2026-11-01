"""Price-basis breaks: a split the vendor history does not reflect yet is spotted from the exchange's own previous close and
repaired by re-asking the vendor; provisional vendor bars are replaced by the official file once it is published."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from automation import jobs
from ingestion import basis
from ingestion.pipeline import IngestReport
from tests.seed_data import make_job_context, panel_ending, populate_store

UTC = timezone.utc
SESSION = "2026-10-30"  # the Friday before the fake clock's Monday
PREVIOUS = "2026-10-29"


def bar(day: str, close: float, **extra) -> dict:
    return {"trade_date": day, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6, **extra}


@pytest.fixture()
def book(store):
    """Three stocks with a vendor close on 29 Oct; the official file of 30 Oct is added per test."""
    ids = {}
    for symbol in ("AAA", "BBB", "CCC"):
        iid = store.upsert_instrument(symbol, "equity", in_universe=1)
        ids[symbol] = iid
        store.upsert_bars(iid, [bar(PREVIOUS, 100.0)], source="yahoo")
    return ids


def official(store, iid, close: float, prev_close: float) -> None:
    store.upsert_bars(iid, [bar(SESSION, close, prev_close=prev_close)], source="nse_bhavcopy", source_rank=2, quality="official")


# ------------------------------------------------------------------------------------------- detection
def test_only_stocks_whose_exchange_previous_close_disagrees_are_flagged(store, book):
    official(store, book["AAA"], 51.0, 50.0)  # 1:2 split: the exchange's base price halved
    official(store, book["BBB"], 101.0, 100.0)  # an ordinary day
    official(store, book["CCC"], 102.0, 98.5)  # a 1.5 % difference is inside the tolerance (vendor / official rounding)
    found = basis.official_breaks(store, SESSION)
    assert [(b.symbol, b.day, b.prev_close, b.stored_close) for b in found] == [("AAA", SESSION, 50.0, 100.0)]
    assert found[0].factor == 2.0


def test_a_small_bonus_is_still_caught_and_a_vendor_row_is_never_the_reference(store, book):
    official(store, book["AAA"], 91.0, 90.9)  # bonus 1:10 -> previous close 100 / 1.1
    assert [b.symbol for b in basis.official_breaks(store, SESSION)] == ["AAA"]
    store.upsert_bars(book["BBB"], [bar(SESSION, 40.0, prev_close=100.0)], source="yahoo")  # vendor row: no official claim
    assert [b.symbol for b in basis.official_breaks(store, SESSION)] == ["AAA"]


def test_rows_without_a_previous_close_or_history_are_ignored(store, book):
    store.upsert_bars(book["AAA"], [bar(SESSION, 50.0)], source="nse_bhavcopy", source_rank=2, quality="official")  # no prev_close
    new = store.upsert_instrument("NEW", "equity", in_universe=1)
    official(store, new, 10.0, 5.0)  # a fresh listing: nothing stored before it to compare with
    assert basis.official_breaks(store, SESSION) == []
    assert basis.official_breaks(store, date.fromisoformat(SESSION)) == []  # a date object works too


def test_index_and_other_series_are_not_equities(store, book):
    series = store.upsert_instrument("^NSEI", "index")
    store.upsert_bars(series, [bar(PREVIOUS, 24000.0)], source="yahoo")
    store.upsert_bars(series, [bar(SESSION, 12000.0, prev_close=12000.0)], source="nse_bhavcopy", source_rank=2)
    assert basis.official_breaks(store, SESSION) == []


def test_the_preopen_book_reveals_an_ex_date_before_the_session_starts(store, book):
    store.add_preopen_quotes(SESSION, [{"symbol": "AAA", "iep": 51.0, "prev_close": 50.0},
                                       {"symbol": "BBB", "iep": 100.5, "prev_close": 100.0}], "30-Oct-2026 09:00:10")
    store.add_preopen_quotes(SESSION, [{"symbol": "AAA", "iep": 51.2, "prev_close": 50.0},
                                       {"symbol": "BBB", "iep": 100.7, "prev_close": 100.0}], "30-Oct-2026 09:07:50")
    found = basis.preopen_breaks(store, SESSION)
    assert [(b.symbol, b.prev_close, b.stored_close) for b in found] == [("AAA", 50.0, 100.0)]
    assert basis.preopen_breaks(store, "2026-10-28") == []


# ------------------------------------------------------------------------------------------- pending list
def test_pending_breaks_keep_their_first_sighting_and_clear(store, book):
    first = basis.BasisBreak("AAA", SESSION, 50.0, 100.0)
    assert basis.mark_pending(store, [first]) == ["AAA"]
    assert basis.mark_pending(store, [basis.BasisBreak("AAA", "2026-11-02", 50.0, 100.0), basis.BasisBreak("BBB", "2026-11-02", 1.0, 2.0)]) == ["BBB"]
    assert basis.pending(store) == {"AAA": SESSION, "BBB": "2026-11-02"}
    basis.clear(store, "AAA")
    assert basis.pending(store) == {"BBB": "2026-11-02"}


def test_a_break_is_explained_by_a_recorded_split_near_or_after_it(store, book):
    assert not basis.is_resolved(store, "AAA", SESSION)
    store.add_corporate_action(book["AAA"], "split", "2026-10-20", ratio=2.0, source="yahoo")  # an old, unrelated split
    assert not basis.is_resolved(store, "AAA", SESSION)
    store.add_corporate_action(book["AAA"], "split", SESSION, ratio=2.0, source="yahoo")
    assert basis.is_resolved(store, "AAA", SESSION)
    store.add_corporate_action(book["BBB"], "split", "2026-10-28", ratio=2.0, source="yahoo")  # within the 3-day grace
    assert basis.is_resolved(store, "BBB", SESSION)
    assert basis.is_resolved(store, "GHOST", SESSION)  # unknown instrument: nothing left to repair


def test_overdue_after_the_give_up_window():
    assert not basis.overdue("2026-10-30", date(2026, 11, 4)) and basis.overdue("2026-10-30", date(2026, 11, 5))


# ---------------------------------------------------------------------------------------------- the job
@pytest.fixture()
def ctx(settings, store, fake_clock, book):
    fake_clock.set(datetime(2026, 11, 2, 12, 30, tzinfo=UTC))
    return make_job_context(settings, store, fake_clock, run_key="2026-11-02")


def test_a_split_the_vendor_reports_is_repaired_and_forgotten(ctx, store, book, monkeypatch):
    official(store, book["AAA"], 51.0, 50.0)
    asked = []

    def vendor(store_, yahoo, **kw):
        asked.append(set(kw["symbols"]))
        store_.add_corporate_action(book["AAA"], "split", SESSION, ratio=2.0, source="yahoo")  # the vendor has caught up
        return IngestReport(ok=1)

    monkeypatch.setattr(jobs, "refresh_daily_bars", vendor)
    result = jobs.repair_basis_breaks(ctx, date.fromisoformat(SESSION))
    assert asked == [{"AAA"}]  # exactly the affected stock, not the whole universe
    assert result == {"basis_breaks": 1, "basis_repaired": 1, "basis_given_up": 0, "basis_waiting": 0}
    assert basis.pending(store) == {}
    assert jobs.repair_basis_breaks(ctx, date.fromisoformat(SESSION)) == {}  # nothing pending, nothing asked
    assert len(asked) == 1


def test_a_slow_vendor_is_asked_again_every_session_until_the_operator_is_told(ctx, store, book, monkeypatch):
    official(store, book["AAA"], 51.0, 50.0)
    asked = []
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: asked.append(k["symbols"]) or IngestReport(ok=1))
    first = jobs.repair_basis_breaks(ctx, date.fromisoformat(SESSION))
    assert first["basis_waiting"] == 1 and first["basis_repaired"] == 0 and basis.pending(store) == {"AAA": SESSION}
    assert store.pending_alerts(10) == []
    later = jobs.repair_basis_breaks(ctx, date.fromisoformat("2026-11-02"))  # the next session: asked again
    assert later["basis_waiting"] == 1 and len(asked) == 2
    final = jobs.repair_basis_breaks(ctx, date.fromisoformat("2026-11-05"))  # six days after the first sighting
    assert final["basis_given_up"] == 1 and basis.pending(store) == {}
    alert = [a for a in store.pending_alerts(10) if a["code"] == "price_basis_unresolved"]
    assert len(alert) == 1 and "AAA" in alert[0]["message"]


def test_an_ordinary_session_asks_the_vendor_nothing(ctx, store, book, monkeypatch):
    official(store, book["AAA"], 101.0, 100.0)
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert jobs.repair_basis_breaks(ctx, date.fromisoformat(SESSION)) == {}


# ------------------------------------------------------------------- provisional -> official upgrade
@pytest.fixture(scope="module")
def panel():
    return panel_ending(n_dates=300, n_symbols=24, seed=21)


@pytest.fixture()
def full(settings, store, fake_clock, panel):
    populate_store(store, panel)  # official (rank 2) bars for every name up to the last session
    fake_clock.set(datetime(2026, 11, 2, 2, 30, tzinfo=UTC))  # 08:00 IST: collect_india's window
    return make_job_context(settings, store, fake_clock, run_key="2026-11-02")


@pytest.fixture()
def provisional(full):
    """The same, but the last two sessions only have the vendor's provisional bars (the official file was not out yet)."""
    full.store.conn.execute(
        "UPDATE bars_daily SET source_rank = 1, quality = 'ok', source = 'yahoo' WHERE trade_date IN (?, ?) AND instrument_id IN "
        "(SELECT instrument_id FROM instruments WHERE kind = 'equity')", (PREVIOUS, SESSION))
    return full


def promote(store, session: str, *, prev_close: bool = False):
    """A fake bhavcopy: every vendor row of ``session`` becomes the official one (with the exchange's previous close when asked)."""

    def fake(store_, _archive, day):
        rows = store_.conn.execute(
            "SELECT b.instrument_id AS instrument_id, b.close AS close FROM bars_daily b JOIN instruments i USING (instrument_id) "
            "WHERE b.trade_date = ? AND b.source_rank < 2 AND i.kind = 'equity'", (day.isoformat(),)).fetchall()
        for r in rows:
            extra = {}
            if prev_close:
                before = store_.conn.execute("SELECT close FROM bars_daily WHERE instrument_id = ? AND trade_date < ? "
                                             "ORDER BY trade_date DESC LIMIT 1", (r["instrument_id"], day.isoformat())).fetchone()
                extra["prev_close"] = float(before["close"]) if before else None
            store_.upsert_bars(int(r["instrument_id"]), [bar(day.isoformat(), float(r["close"]), **extra)],
                               source="nse_bhavcopy", source_rank=2, quality="official")
        return len(rows), len(rows)

    return fake


def test_complete_and_official_means_nothing_to_do(full, monkeypatch):
    assert jobs.provisional_symbols(full, date.fromisoformat(SESSION)) == []
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=3))
    monkeypatch.setattr(jobs, "ingest_bhavcopy", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    result = jobs.collect_india(full)
    assert result.status == "success" and result.metrics["coverage"] == 1.0 and "official_error" not in result.metrics


def test_the_morning_job_replaces_provisional_vendor_bars_by_the_official_file(provisional, monkeypatch):
    full = provisional
    assert len(jobs.provisional_symbols(full, date.fromisoformat(SESSION))) == 24  # complete, but all provisional
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=3))
    monkeypatch.setattr(jobs, "ingest_bhavcopy", promote(full.store, SESSION))
    result = jobs.collect_india(full)
    assert result.status == "success" and result.metrics["official_rows"] == 24 and result.metrics["coverage"] == 1.0
    assert jobs.provisional_symbols(full, date.fromisoformat(SESSION)) == []


def test_the_nightly_housekeeping_upgrades_the_last_sessions_too(provisional, monkeypatch):
    full = provisional
    seen = []

    def fake(store, _archive, day):
        seen.append(day.isoformat())
        return promote(store, day.isoformat())(store, _archive, day)

    monkeypatch.setattr(jobs, "ingest_bhavcopy", fake)
    full.clock.set(datetime(2026, 11, 2, 18, 20, tzinfo=UTC))  # 23:50 IST: housekeeping's slot
    result = jobs.housekeeping(full)
    assert result.status == "success" and result.metrics["official_rows"] == 48
    assert seen == ["2026-10-29", "2026-10-30"]  # Nov 2 has no bars yet, so there is nothing provisional to replace
    assert jobs.provisional_symbols(full, date(2026, 10, 30)) == []


def test_a_missing_official_file_never_fails_the_job(provisional, monkeypatch):
    full = provisional
    monkeypatch.setattr(jobs, "refresh_daily_bars", lambda *a, **k: IngestReport(ok=3))
    monkeypatch.setattr(jobs, "ingest_bhavcopy", lambda *a, **k: (0, 0))  # not published yet
    result = jobs.collect_india(full)
    assert result.status == "success" and result.metrics["official_rows"] == 0
    assert len(jobs.provisional_symbols(full, date.fromisoformat(SESSION))) == 24  # still provisional; tried again next time


def test_the_official_file_also_reveals_a_split_of_a_stock_that_has_history(provisional, monkeypatch):
    """The whole path in one go: provisional vendor bars, the official file with the exchange's halved previous close, the
    vendor asked about that one stock only, the split recorded."""
    full = provisional
    store = full.store
    victim = store.instrument_id("S03")
    last = float(store.conn.execute("SELECT close FROM bars_daily WHERE instrument_id = ? AND trade_date = ?",
                                    (victim, PREVIOUS)).fetchone()["close"])
    raw = promote(store, SESSION, prev_close=True)

    def official_file(store_, archive, day):
        written, matched = raw(store_, archive, day)
        if day.isoformat() == SESSION:  # S03 went ex-split 1:2 on 30 Oct
            store_.conn.execute("UPDATE bars_daily SET prev_close = ? WHERE instrument_id = ? AND trade_date = ?",
                                (last / 2, victim, SESSION))
        return written, matched

    asked = []

    def vendor(store_, yahoo, **kw):
        if kw.get("symbols") and "S03" in kw["symbols"]:
            asked.append(set(kw["symbols"]))
            store_.add_corporate_action(victim, "split", SESSION, ratio=2.0, source="yahoo")
        return IngestReport(ok=3)

    monkeypatch.setattr(jobs, "ingest_bhavcopy", official_file)
    monkeypatch.setattr(jobs, "refresh_daily_bars", vendor)
    result = jobs.collect_india(full)
    assert result.status == "success", result.message
    assert asked == [{"S03"}] and result.metrics["basis_breaks"] == 1 and result.metrics["basis_repaired"] == 1
    assert [r["ex_date"] for r in store.corporate_actions(victim, "split")] == [SESSION]
    assert basis.pending(store) == {}
