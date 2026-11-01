"""Store: the effective snapshot of a (trade date, stage), idempotent metrics, realised ranges and source freshness."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

DAY = "2026-11-02"
UTC = timezone.utc


def pred(symbol: str, rank: int, **extra):
    base = {"symbol": symbol, "rank": rank, "tier": "WATCH", "model_version": "m1", "model_status": "shadow",
            "p": {"hit_3": 0.31}, "range": {"q10": 1.0, "q50": 2.4, "q90": 5.0}, "uncertainty": 0.2, "data_quality": "ok"}
    base.update(extra)
    return base


def freeze(store, *, day: str = DAY, stage: str = "final", tag: str = "a", symbols=("AAA", "BBB"), baselines=("atr",)):
    sid = store.insert_snapshot(trade_date=day, stage=stage, cutoff_ts=f"{day}T03:40:00Z", feature_version="fv1", model_version="m1",
                                code_version="c1", payload={"tag": tag}, data_quality={})
    rows = [pred(symbol, rank) for rank, symbol in enumerate(symbols, 1)]
    rows += [pred(symbols[0], 1, tier="BASELINE", model_status="baseline", is_baseline=True, baseline_name=name) for name in baselines]
    return sid, store.insert_predictions(sid, day, stage, rows)


def attach(store, prediction_id: int, symbol: str, *, validated: bool = True, range_pct: float = 4.0, day: str = DAY):
    store.upsert_outcome(prediction_id, day, symbol, source="nse_bhavcopy" if validated else "yahoo", validated=validated,
                         ohlcv={"open": 100, "high": 104, "low": 100 - range_pct + 4, "close": 102},
                         metrics={"range_pct": range_pct, "dir_ret_pct": 2.0})


# ----------------------------------------------------------------------------- the effective snapshot
def test_a_forced_rescan_supersedes_the_earlier_snapshot_for_every_read_path(store):
    first, ids_first = freeze(store, tag="first")
    second, ids_second = freeze(store, tag="rescan")
    assert second > first and len(ids_first) == len(ids_second) == 3
    got = store.predictions_for(DAY, stage="final", baselines=None)
    assert {p["snapshot_id"] for p in got} == {second} and len(got) == 3  # the latest freeze only, not both
    assert len(store.predictions_for(DAY)) == 2 and len(store.predictions_for(DAY, baselines=True)) == 1
    both = store.predictions_for(DAY, stage="final", baselines=None, all_snapshots=True)
    assert {p["snapshot_id"] for p in both} == {first, second} and len(both) == 6  # an explicit way to read everything


def test_evaluated_rows_and_pending_outcomes_ignore_superseded_predictions(store):
    first, ids_first = freeze(store, tag="first")
    second, ids_second = freeze(store, tag="rescan")
    for pid, symbol in zip(ids_first[:2] + ids_second[:2], ["AAA", "BBB", "AAA", "BBB"], strict=True):
        attach(store, pid, symbol)
    rows = store.evaluated_rows(since=DAY, until=DAY)
    assert {r["snapshot_id"] for r in rows} == {second} and len(rows) == 2  # n = 2 for two picks, not 4
    assert len(store.evaluated_rows(since=DAY, until=DAY, all_snapshots=True)) == 4
    assert {p["snapshot_id"] for p in store.predictions_without_validated_outcome(DAY)} == {second}  # only the baseline pick left
    assert len(store.predictions_without_validated_outcome(DAY)) == 1
    assert len(store.predictions_without_validated_outcome(DAY, all_snapshots=True)) == 2


def test_the_preliminary_and_final_stage_each_keep_their_own_latest_snapshot(store):
    pre_old, _ = freeze(store, stage="pre", tag="pre-1")
    pre_new, _ = freeze(store, stage="pre", tag="pre-2")
    final, _ = freeze(store, stage="final", tag="final-1")
    assert {p["snapshot_id"] for p in store.predictions_for(DAY, stage="pre", baselines=None)} == {pre_new}
    assert {p["snapshot_id"] for p in store.predictions_for(DAY, stage="final", baselines=None)} == {final}
    assert {p["snapshot_id"] for p in store.predictions_for(DAY, baselines=None)} == {pre_new, final}
    assert pre_old < pre_new


def test_effective_snapshot_is_what_latest_snapshot_returns(store):
    assert store.effective_snapshot(DAY, "final") is None and store.effective_snapshots() == []
    freeze(store, tag="first")
    second, _ = freeze(store, tag="rescan")
    pre, _ = freeze(store, stage="pre", tag="pre")
    freeze(store, day="2026-11-03", tag="next-day")
    assert store.effective_snapshot(DAY, "final")["snapshot_id"] == store.latest_snapshot(DAY, "final")["snapshot_id"] == second
    assert store.effective_snapshot(DAY, "final")["payload"] == {"tag": "rescan"}
    listed = store.effective_snapshots(since=DAY, until=DAY)
    assert [(s["trade_date"], s["stage"], s["snapshot_id"]) for s in listed] == [(DAY, "final", second), (DAY, "pre", pre)]
    assert all("payload" not in s for s in listed)  # listing never decompresses the payloads
    assert [s["stage"] for s in store.effective_snapshots(stage="pre")] == ["pre"]
    assert len(store.effective_snapshots()) == 3 and store.effective_snapshots(since="2027-01-01") == []


def test_outcome_counts_cover_effective_predictions_only(store):
    first, ids_first = freeze(store, tag="first")
    second, ids_second = freeze(store, tag="rescan")
    attach(store, ids_first[0], "AAA")
    attach(store, ids_first[1], "BBB", validated=False)
    attach(store, ids_second[0], "AAA")
    attach(store, ids_second[1], "BBB", validated=False)
    assert store.outcome_counts() == {"validated": 1, "provisional": 1, "days": 1}


def test_outcome_counts_on_an_empty_store(store):
    assert store.outcome_counts() == {"validated": 0, "provisional": 0, "days": 0}


# --------------------------------------------------------------------------------- idempotent metrics
def test_insert_metrics_replaces_the_same_metric_of_the_same_day_instead_of_duplicating_it(store):
    row = {"name": "topn_hit_rate", "value": 0.4, "n": 5, "ci_low": 0.1, "ci_high": 0.8}
    assert store.insert_metrics(DAY, "daily", [row, {"name": "mae", "slice": "rank:1", "value": 1.0, "n": 9}]) == 2
    assert store.insert_metrics(DAY, "daily", [{**row, "value": 0.6, "n": 10}]) == 1  # a re-run with updated numbers
    rows = {(m["slice"], m["name"]): m for m in store.latest_metrics("daily")}
    assert len(store.latest_metrics("daily")) == 2 and rows[("all", "topn_hit_rate")]["value"] == 0.6
    assert rows[("all", "topn_hit_rate")]["n"] == 10 and rows[("rank:1", "mae")]["value"] == 1.0  # others untouched
    store.insert_metrics("2026-11-03", "daily", [{**row, "value": 0.7}])  # another day keeps its own history
    count = store.conn.execute("SELECT COUNT(*) AS n FROM eval_metrics WHERE name = 'topn_hit_rate'").fetchone()["n"]
    assert count == 2
    store.insert_metrics(DAY, "rolling_20d", [{**row, "value": 0.1}])  # another scope is a different metric
    assert store.conn.execute("SELECT COUNT(*) AS n FROM eval_metrics WHERE as_of_date = ?", (DAY,)).fetchone()["n"] == 3


def test_insert_metrics_keeps_the_last_of_duplicates_within_one_call(store):
    written = store.insert_metrics(DAY, "daily", [{"name": "x", "value": 1.0}, {"name": "x", "value": 2.0}, {"name": "y", "value": 3.0}])
    assert written == 2
    assert {m["name"]: m["value"] for m in store.latest_metrics("daily")} == {"x": 2.0, "y": 3.0}


def test_insert_metrics_is_atomic_when_a_row_cannot_be_stored(store):
    store.insert_metrics(DAY, "daily", [{"name": "x", "value": 1.0}])
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_metrics(DAY, "daily", [{"name": "x", "value": 9.0}, {"name": None, "value": 1.0}])
    assert [m["value"] for m in store.latest_metrics("daily")] == [1.0]  # the replaced row came back: all or nothing


# --------------------------------------------------------------------------------------- realised ranges
def test_realised_ranges_cover_equities_with_a_usable_bar(store):
    for symbol, kind in (("AAA", "equity"), ("BBB", "equity"), ("CCC", "equity"), ("^SPX", "global")):
        store.upsert_instrument(symbol, kind)
    bars = {"AAA": (100, 104, 99), "BBB": (50, 50, 49), "CCC": (0, 5, 4), "^SPX": (100, 110, 90)}
    for symbol, (o, h, low) in bars.items():
        store.upsert_bars(store.instrument_id(symbol, "global" if symbol.startswith("^") else "equity"),
                          [{"trade_date": DAY, "open": o, "high": h, "low": low, "close": o}], source="t")
    store.upsert_bars(store.instrument_id("AAA"), [{"trade_date": "2026-11-03", "open": 10, "high": 11, "low": 10, "close": 10}], source="t")
    got = store.realised_ranges(DAY, "2026-11-03")
    assert got[DAY] == {"AAA": pytest.approx(5.0), "BBB": pytest.approx(2.0)}  # no index series, no zero-open bar
    assert got["2026-11-03"] == {"AAA": pytest.approx(10.0)}
    assert store.realised_ranges("2026-12-01", "2026-12-02") == {}


# -------------------------------------------------------------------------------------- source freshness
def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, 2, hour, minute, tzinfo=UTC)


CUTOFF = "2026-11-02T03:15:00Z"


def seed_sources(store, fake_clock):
    fake_clock.set(datetime(2026, 10, 30, 14, 0, tzinfo=UTC))  # the 30 Oct bars arrive in the evening
    for symbol, kind in (("AAA", "equity"), ("BBB", "equity"), ("^GSPC", "global")):
        store.upsert_instrument(symbol, kind, in_universe=1 if kind == "equity" else 0)
        store.upsert_bars(store.instrument_id(symbol, kind), [{"trade_date": "2026-10-30", "open": 1, "high": 2, "low": 1, "close": 2}], source="t")
    store.upsert_fno([{"symbol": "AAA", "trade_date": "2026-10-30"}])
    fake_clock.set(at(2, 30))
    store.add_live_quote("GIFT_NIFTY", price=24000.0, source_ts="02-Nov-2026 08:00:00")
    fake_clock.set(at(3, 10))
    store.add_preopen_quotes(DAY, [{"symbol": "AAA", "iep": 1.0}, {"symbol": "BBB", "iep": 2.0}], "02-Nov-2026 08:40:10")
    fake_clock.set(at(2, 0))
    store.conn.execute("INSERT INTO news_raw (source, url_hash, content_hash, title, published_at, first_seen_at) "
                       "VALUES ('rss', 'u1', 'c1', 't', '2026-11-02T01:30:00Z', '2026-11-02T01:45:00Z')")
    store.conn.execute("INSERT INTO filings (exchange, symbol, disseminated_at, first_seen_at, source_ref) "
                       "VALUES ('NSE', 'AAA', '2026-11-01T10:00:00Z', '2026-11-01T10:05:00Z', 'r1')")


def test_source_freshness_reports_the_latest_source_time_and_arrival_of_every_input(store, fake_clock):
    seed_sources(store, fake_clock)
    got = store.source_freshness(trade_date=DAY, cutoff_ts=CUTOFF)
    assert set(got) == {"market_bars", "global_closes", "fno", "preopen", "gift", "news", "filings"}
    assert got["market_bars"] == {"latest": "2026-10-30", "received_at": "2026-10-30T14:00:00Z", "n": 2}
    assert got["global_closes"] == {"latest": "2026-10-30", "received_at": "2026-10-30T14:00:00Z", "n": 1}
    assert got["fno"] == {"latest": "2026-10-30", "received_at": "2026-10-30T14:00:00Z", "n": 1}
    assert got["preopen"] == {"latest": "02-Nov-2026 08:40:10", "received_at": "2026-11-02T03:10:00Z", "n": 2}
    assert got["gift"] == {"latest": "02-Nov-2026 08:00:00", "received_at": "2026-11-02T02:30:00Z", "n": 1}
    assert got["news"] == {"latest": "2026-11-02T01:30:00Z", "received_at": "2026-11-02T01:45:00Z", "n": 1}
    assert got["filings"] == {"latest": "2026-11-01T10:00:00Z", "received_at": "2026-11-01T10:05:00Z", "n": 1}


def test_source_freshness_only_counts_what_had_arrived_by_the_cutoff(store, fake_clock):
    seed_sources(store, fake_clock)
    fake_clock.set(at(3, 20))  # after the 03:15 cutoff: must not appear in a snapshot taken at the cutoff
    store.upsert_bars(store.instrument_id("AAA"), [{"trade_date": "2026-10-30", "open": 1, "high": 3, "low": 1, "close": 3}], source="t")
    store.add_preopen_quotes(DAY, [{"symbol": "AAA", "iep": 9.0}], "02-Nov-2026 08:50:00")
    store.add_live_quote("GIFT_NIFTY", price=24100.0, source_ts="02-Nov-2026 08:50:00")
    store.conn.execute("INSERT INTO news_raw (source, url_hash, content_hash, title, published_at, first_seen_at) "
                       "VALUES ('rss', 'u2', 'c2', 't', '2026-11-02T03:19:00Z', '2026-11-02T03:19:30Z')")
    got = store.source_freshness(trade_date=DAY, cutoff_ts=CUTOFF)
    assert got["market_bars"]["received_at"] == "2026-10-30T14:00:00Z"  # the evening revision (03:20) came too late
    assert got["preopen"] == {"latest": "02-Nov-2026 08:40:10", "received_at": "2026-11-02T03:10:00Z", "n": 2}
    assert got["gift"]["latest"] == "02-Nov-2026 08:00:00" and got["news"]["n"] == 1
    later = store.source_freshness(trade_date=DAY, cutoff_ts="2026-11-02T03:30:00Z")
    assert later["preopen"]["latest"] == "02-Nov-2026 08:50:00" and later["news"]["n"] == 2


def test_source_freshness_of_an_empty_database_is_explicitly_empty(store):
    got = store.source_freshness(trade_date=DAY, cutoff_ts=CUTOFF)
    assert all(v == {"latest": None, "received_at": None, "n": 0} for v in got.values()) and len(got) == 7


def test_source_freshness_ignores_data_older_than_its_lookback(store, fake_clock):
    fake_clock.set(datetime(2026, 9, 1, 14, 0, tzinfo=UTC))
    store.upsert_instrument("AAA", "equity", in_universe=1)
    store.upsert_bars(store.instrument_id("AAA"), [{"trade_date": "2026-09-01", "open": 1, "high": 2, "low": 1, "close": 2}], source="t")
    assert store.source_freshness(trade_date=DAY, cutoff_ts=CUTOFF)["market_bars"]["n"] == 0  # two months stale: nothing recent
