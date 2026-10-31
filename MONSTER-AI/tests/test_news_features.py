from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from event_graph.seed import seed_graph
from news_intelligence import features as feat
from news_intelligence.dedup import novelty_at
from news_intelligence.pipeline import NewsPipeline
from ranking.types import Catalyst
from tests.test_news_support import add_article, add_filing, add_instruments, events, insert_event

H = "2026-11-01T{:02d}:00:00Z"
TITLE_ORDER = "Reliance bags Rs 5,000 crore order from NTPC"


def count(store, table, where="1=1"):
    return store.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


def ent(symbol, *, polarity=0.0, type="company", confidence=0.9, in_title=True, **extra):
    return {"type": type, "id": symbol, "name": symbol, "count": 1, "confidence": confidence, "polarity": polarity,
            "in_title": in_title, **extra}


def add(store, key, entities, hour=10, **fields):
    stamp = H.format(hour)
    fields.setdefault("first_seen_at", stamp)
    fields.setdefault("last_updated_at", fields["first_seen_at"])
    return insert_event(store, key, entities_json=json.dumps(entities), **fields)


def evidence(owner, hour, *, q=0.75, conf=0.7, pol=0.7, ref=None):
    return {"kind": "article", "ref": ref or f"a:{owner}{hour}", "t": H.format(hour), "owner": owner, "domains": [owner], "n": 1,
            "q": q, "conf": conf, "pol": pol, "title": f"{owner} story", "url": f"https://{owner}.example/{hour}"}


@pytest.fixture()
def pipe(store, settings, fake_clock):
    add_instruments(store)
    return NewsPipeline(store, settings, clock=fake_clock)


# ------------------------------------------------------------------------------------------ feature vectors
def test_every_symbol_always_gets_every_key(pipe):
    feats = pipe.event_features(["TCS", "NOPE", "TCS"], H.format(14))
    assert list(feats) == ["TCS", "NOPE"]
    for vec in feats.values():
        assert tuple(vec) == feat.FEATURE_KEYS and len(vec) == 20
        assert all(isinstance(v, float) and v == 0.0 for v in vec.values())
    assert pipe.event_features([], H.format(14)) == {}
    glob = pipe.global_event_features(H.format(14))
    assert tuple(glob) == feat.GLOBAL_KEYS and all(v == 0.0 for v in glob.values())


def test_company_event_features_have_the_documented_meaning(store, pipe):
    add(store, "order", [ent("TCS", polarity=0.7)], severity=0.6, confidence=0.7, source_quality=0.9)
    add(store, "results", [ent("TCS", polarity=-0.5)], hour=11, event_type="earnings_result", severity=0.4, confidence=0.8,
        source_quality=0.6)
    out = pipe.event_features(["TCS", "RELIANCE"], H.format(12))
    tcs = out["TCS"]
    assert tcs["ev_n_36h"] == 2 and tcs["ev_n_24h"] == 2
    assert tcs["ev_max_severity"] == 0.6 and tcs["ev_conf_weighted_severity"] == pytest.approx(1 - (1 - 0.42) * (1 - 0.32), abs=1e-3)
    assert tcs["ev_mean_polarity"] == pytest.approx((0.42 * 0.7 - 0.32 * 0.5) / 0.74, abs=1e-3)
    assert tcs["ev_abs_polarity"] == pytest.approx((0.42 * 0.7 + 0.32 * 0.5) / 0.74, abs=1e-3)
    assert tcs["ev_has_order"] == 1.0 and tcs["ev_has_results"] == 1.0
    assert tcs["ev_has_ma"] == tcs["ev_has_regulatory"] == tcs["ev_has_rating"] == tcs["ev_has_mgmt"] == 0.0
    assert tcs["ev_max_novelty"] == pytest.approx(novelty_at(1.0, 0.0, 1 * 3600.0, 1), abs=1e-3)
    assert tcs["ev_hours_since_latest"] == 1.0 and tcs["ev_source_quality_max"] == 0.9
    assert tcs["ev_contradiction_flag"] == 0.0 and tcs["ev_market_reacted_share"] == 0.0
    assert all(v == 0.0 for v in out["RELIANCE"].values())


def test_every_flag_feature_follows_its_event_types(store, pipe):
    for i, kind in enumerate(("merger_acquisition", "litigation_regulatory", "rating_action", "management_change", "government_contract")):
        add(store, kind, [ent("INFY", polarity=0.3)], event_type=kind, severity=0.5, confidence=0.7, first_seen_at=H.format(6 + i))
    out = pipe.event_features(["INFY"], H.format(14))["INFY"]
    assert out["ev_has_ma"] == out["ev_has_regulatory"] == out["ev_has_rating"] == out["ev_has_mgmt"] == out["ev_has_order"] == 1.0


def test_passing_mentions_and_other_events_do_not_count(store, pipe):
    add(store, "mention", [ent("TCS", polarity=0.7), ent("INFY", polarity=0.0, in_title=False)], severity=0.6)
    add(store, "other", [ent("WIPRO", polarity=0.4)], event_type="other", severity=0.9)
    add(store, "weak", [ent("SBIN", polarity=0.4, confidence=0.3)], severity=0.9)
    out = pipe.event_features(["TCS", "INFY", "WIPRO", "SBIN"], H.format(12))
    assert out["TCS"]["ev_n_36h"] == 1
    assert out["INFY"]["ev_n_36h"] == 0 and out["WIPRO"]["ev_n_36h"] == 0 and out["SBIN"]["ev_n_36h"] == 0
    assert feat.is_subject({"polarity": -0.2}) and feat.is_subject({"in_title": True}) and not feat.is_subject({"polarity": 0})


def test_lookback_window_and_the_24h_count(store, pipe):
    add(store, "old", [ent("TCS", polarity=0.7)], first_seen_at="2026-10-30T20:00:00Z")  # 40h before the as-of time
    add(store, "mid", [ent("TCS", polarity=0.7)], first_seen_at="2026-10-31T04:00:00Z")  # 32h before
    add(store, "new", [ent("TCS", polarity=0.7)], first_seen_at="2026-11-01T10:00:00Z")
    out = pipe.event_features(["TCS"], H.format(12))["TCS"]
    assert out["ev_n_36h"] == 2 and out["ev_n_24h"] == 1
    assert pipe.event_features(["TCS"], H.format(12), lookback_hours=48)["TCS"]["ev_n_36h"] == 3
    assert pipe.event_features(["TCS"], H.format(12), lookback_hours=6)["TCS"]["ev_n_36h"] == 1


# ------------------------------------------------------------------------------------------ point in time
def test_items_first_seen_after_as_of_are_invisible_everywhere(store, pipe):
    seed_graph(store)
    add(store, "a", [ent("TCS", polarity=0.7)], hour=10, severity=0.6)
    add(store, "b", [ent("TCS", polarity=-0.5)], hour=12, event_type="earnings_result", severity=0.5)
    add(store, "c", [ent("INFY", polarity=0.4)], hour=14, severity=0.5)
    add(store, "m", [ent("macro:brent", type="commodity", polarity=0.6)], hour=13, scope="macro", event_type="commodity_supply",
        severity=0.8, confidence=0.8)

    def tcs(hour, minute=0):
        return pipe.event_features(["TCS", "INFY"], f"2026-11-01T{hour:02d}:{minute:02d}:00Z")

    assert tcs(9, 59)["TCS"]["ev_n_36h"] == 0
    assert tcs(10)["TCS"]["ev_n_36h"] == 1  # as_of is inclusive
    assert tcs(11, 59)["TCS"]["ev_has_results"] == 0 and tcs(12)["TCS"]["ev_has_results"] == 1
    assert tcs(13, 59)["INFY"]["ev_n_36h"] == 0 and tcs(14)["INFY"]["ev_n_36h"] == 1
    assert [e["headline"] for e in pipe.recent_events(H.format(11), 36)] == ["a"]
    assert {e["headline"] for e in pipe.recent_events(H.format(14), 36)} == {"a", "b", "c", "m"}
    assert pipe.global_event_features(H.format(12))["gl_n_events"] == 0.0
    assert pipe.global_event_features(H.format(13))["gl_n_events"] == 1.0
    assert [c.headline for c in pipe.catalysts_for("TCS", H.format(11))] == ["a"]
    assert pipe.catalysts_for("TCS", H.format(9)) == []
    assert pipe.event_features(["HINDPETRO"], H.format(12))["HINDPETRO"]["ev_global_macro_severity"] == 0.0
    assert pipe.event_features(["HINDPETRO"], H.format(13))["HINDPETRO"]["ev_global_macro_severity"] > 0.0


def test_article_counts_corroboration_and_confidence_use_only_what_was_known(store, pipe):
    ev = add(store, "story", [ent("TCS", polarity=0.7)], severity=0.6, n_articles=3, n_independent_sources=3, confidence=0.9,
             evidence_json=json.dumps([evidence("pti", 10), evidence("reuters", 12), evidence("mint", 16)]),
             last_updated_at=H.format(16))
    for hour in (10, 12, 16):
        store.conn.execute("INSERT INTO event_articles (event_id, added_at) VALUES (?, ?)", (ev, H.format(hour)))

    def view(hour):
        return pipe.recent_events(H.format(hour), 36)[0]

    assert (view(10)["n_articles"], view(10)["n_independent_sources"]) == (1, 1)
    assert (view(13)["n_articles"], view(13)["n_independent_sources"]) == (2, 2)
    assert (view(17)["n_articles"], view(17)["n_independent_sources"]) == (3, 3)
    assert view(10)["confidence"] < view(13)["confidence"] < view(17)["confidence"]
    assert pipe.event_features(["TCS"], H.format(10))["TCS"]["ev_max_independent_sources"] == 1.0
    assert pipe.event_features(["TCS"], H.format(13))["TCS"]["ev_max_independent_sources"] == 2.0
    assert len(view(13)["evidence"]) == 2


def test_contradictions_and_late_entities_appear_only_once_known(store, pipe):
    add(store, "story", [ent("TCS", polarity=0.7), ent("INFY", polarity=0.3, in_title=False, t=H.format(12))], severity=0.6,
        evidence_json=json.dumps([evidence("pti", 10, pol=0.7), evidence("reuters", 12, pol=-0.7)]),
        contradictions_json=json.dumps([{"kind": "polarity", "t": H.format(12), "ref": "a:x", "source": "reuters", "pol": -0.7,
                                         "against": 0.7}]),
        last_updated_at=H.format(12))
    early = pipe.event_features(["TCS", "INFY"], H.format(11))
    late = pipe.event_features(["TCS", "INFY"], H.format(13))
    assert early["TCS"]["ev_contradiction_flag"] == 0.0 and late["TCS"]["ev_contradiction_flag"] == 1.0
    assert early["INFY"]["ev_n_36h"] == 0 and late["INFY"]["ev_n_36h"] == 1
    assert early["TCS"]["ev_mean_polarity"] == pytest.approx(0.7, abs=1e-3)
    assert abs(late["TCS"]["ev_mean_polarity"]) < 0.1  # disagreeing origins: direction is no longer asserted
    assert late["TCS"]["ev_conf_weighted_severity"] < early["TCS"]["ev_conf_weighted_severity"]


def test_agreement_scale_edge_cases():
    contradiction = [{"kind": "polarity"}]
    assert feat._agreement_scale([], contradiction) is None
    assert feat._agreement_scale([{"pol": 0.5}], [{"kind": "denial"}]) is None
    assert feat._agreement_scale([{"pol": 0.0, "q": 1, "conf": 1}], contradiction) is None
    even = [{"pol": 0.6, "q": 0.5, "conf": 0.5}, {"pol": -0.6, "q": 0.5, "conf": 0.5}]
    assert feat._agreement_scale(even, contradiction) == pytest.approx(0.0)
    lopsided = [{"pol": 0.6, "q": 0.2, "conf": 0.5}, {"pol": -0.6, "q": 0.9, "conf": 0.9}]
    assert feat._agreement_scale(lopsided, contradiction) < 0


def test_market_reaction_is_only_visible_after_it_was_assessed(store, pipe):
    add(store, "a", [ent("TCS", polarity=0.7)], severity=0.6, market_reacted=1,
        reaction_json=json.dumps({"reacted": True, "as_of": H.format(15)}))
    assert pipe.event_features(["TCS"], H.format(12))["TCS"]["ev_market_reacted_share"] == 0.0
    assert pipe.event_features(["TCS"], H.format(16))["TCS"]["ev_market_reacted_share"] == 1.0
    assert pipe.recent_events(H.format(12), 36)[0]["market_reacted"] is None
    assert pipe.recent_events(H.format(16), 36)[0]["market_reacted"] is True


def test_legacy_events_without_evidence_fall_back_to_their_columns(store, pipe):
    add(store, "plain", [ent("TCS", polarity=0.7)], severity=0.6, confidence=0.55, n_articles=0, n_independent_sources=2,
        source_quality=0.8)
    view = pipe.recent_events(H.format(11), 36)[0]
    assert view["confidence"] == 0.55 and view["n_independent_sources"] == 2 and view["n_articles"] == 1
    assert view["source_quality"] == 0.8 and view["source"] == "" and view["url"] is None


# ----------------------------------------------------------------------------------- sector and macro features
def test_sector_and_broad_macro_events_feed_their_features(store, pipe):
    add(store, "it", [ent("sector:information_technology", type="sector", polarity=0.0, in_title=False)], scope="sector",
        event_type="sector_regulation", severity=0.5, confidence=0.8)
    add(store, "stress", [ent("macro:vix", type="index", polarity=0.7)], scope="macro", event_type="market_stress", severity=0.8,
        confidence=0.9, hour=11)
    add(store, "weak-sector", [ent("sector:pharma", type="sector", confidence=0.2)], scope="sector", severity=0.9)
    out = pipe.event_features(["TCS", "RELIANCE"], H.format(12))
    assert out["TCS"]["ev_sector_severity"] == pytest.approx(0.4) and out["RELIANCE"]["ev_sector_severity"] == 0.0
    assert out["TCS"]["ev_global_macro_severity"] == pytest.approx(0.72) == out["RELIANCE"]["ev_global_macro_severity"]


def test_macro_events_reach_stocks_through_the_graph(store, pipe):
    seed_graph(store)
    add(store, "brent", [ent("macro:brent", type="commodity", polarity=0.7, name="Brent crude", move_pct=3.1)], scope="macro",
        event_type="commodity_supply", severity=0.8, confidence=0.9, source_quality=0.75)
    out = pipe.event_features(["HINDPETRO", "TCS"], H.format(12))
    assert 0.0 < out["HINDPETRO"]["ev_global_macro_severity"] <= 0.72 + 1e-9
    assert out["TCS"]["ev_global_macro_severity"] == 0.0
    glob = pipe.global_event_features(H.format(12))
    assert glob["gl_n_events"] == 1.0 and glob["gl_commodity_severity"] == pytest.approx(0.72)
    assert glob["gl_max_severity"] == 0.8 and glob["gl_hours_since_latest"] == 2.0 and glob["gl_source_quality_max"] == 0.75


def test_global_features_cover_each_macro_family(store, pipe):
    families = {
        "central_bank": "gl_central_bank_severity", "geopolitical_conflict": "gl_geopolitical_severity",
        "macro_release": "gl_macro_release_severity", "trade_policy": "gl_trade_policy_severity",
        "elections_politics": "gl_politics_severity", "global_company": "gl_global_company_severity",
        "market_stress": "gl_stress_severity", "sanctions_tariffs": "gl_geopolitical_severity",
    }
    for i, kind in enumerate(families):
        add(store, kind, [ent("macro:fed", type="institution", polarity=0.3)], scope="macro", event_type=kind, severity=0.6,
            confidence=0.5, hour=2 + i, contradictions_json=json.dumps([{"kind": "denial", "t": H.format(2 + i)}]))
    glob = pipe.global_event_features(H.format(20))
    for key in set(families.values()):
        assert glob[key] == pytest.approx(0.3), key
    assert glob["gl_n_events"] == 8.0 and glob["gl_contradiction_flag"] == 1.0
    assert 0.0 < glob["gl_max_novelty"] <= 1.0 and 0.0 < glob["gl_conf_weighted_severity"] < 1.0


# ------------------------------------------------------------------------------------------ recent_events
def test_recent_events_rank_by_importance_and_respect_scope_and_limit(store, pipe):
    add(store, "minor", [ent("TCS", polarity=0.2)], severity=0.2, confidence=0.5)
    add(store, "major", [ent("INFY", polarity=0.7)], severity=0.9, confidence=0.9)
    add(store, "macro", [ent("macro:fed", type="institution", polarity=0.4)], scope="macro", event_type="central_bank", severity=0.7,
        confidence=0.9)
    out = pipe.recent_events(H.format(12), 36)
    assert [e["headline"] for e in out] == ["major", "macro", "minor"]
    assert [e["headline"] for e in pipe.recent_events(H.format(12), 36, scope="macro")] == ["macro"]
    assert len(pipe.recent_events(H.format(12), 36, limit=2)) == 2 and pipe.recent_events(H.format(12), 36, limit=0) == []
    first = out[0]
    assert set(first) >= {"event_id", "event_type", "scope", "headline", "entities", "severity", "confidence", "novelty",
                          "n_articles", "n_independent_sources", "first_seen_at", "hours_since", "evidence", "market_reacted"}
    assert first["hours_since"] == 2.0


def test_queries_are_cached_until_the_events_change(store, pipe):
    add(store, "a", [ent("TCS", polarity=0.7)], severity=0.6)
    first = pipe.event_features(["TCS"], H.format(12))
    assert pipe.event_features(["TCS"], H.format(12)) == first
    add(store, "b", [ent("TCS", polarity=0.7)], hour=11, severity=0.6)
    assert pipe.event_features(["TCS"], H.format(12))["TCS"]["ev_n_36h"] == 2
    for hour in range(6):  # the cache stays bounded
        pipe.recent_events(H.format(12 + hour), 36)
    assert len(pipe._cache) <= 6


def test_as_of_is_validated_and_accepts_several_forms(store, pipe):
    add(store, "a", [ent("TCS", polarity=0.7)], severity=0.6)
    aware = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 11, 1, 12, 0)
    for value in (aware, naive, "2026-11-01T12:00:00Z", "2026-11-01T17:30:00+05:30", pd.Timestamp("2026-11-01T12:00:00Z").to_pydatetime()):
        assert pipe.event_features(["TCS"], value)["TCS"]["ev_n_36h"] == 1
    for bad in ("garbage", None, ""):
        with pytest.raises(ValueError):
            pipe.event_features(["TCS"], bad)
        with pytest.raises(ValueError):
            pipe.catalysts_for("TCS", bad)


# ------------------------------------------------------------------------------------------ catalysts
def test_catalysts_cover_own_sector_and_macro_events_best_first(store, pipe):
    seed_graph(store)
    add(store, "own", [ent("HINDPETRO", polarity=-0.6)], severity=0.7, confidence=0.8, event_type="litigation_regulatory",
        evidence_json=json.dumps([evidence("reuters", 10)]))
    add(store, "oilsector", [ent("sector:oil_gas_and_consumable_fuels", type="sector", polarity=-0.5, in_title=False)], scope="sector",
        event_type="sector_regulation", severity=0.6, confidence=0.7, hour=11)
    add(store, "brent", [ent("macro:brent", type="commodity", polarity=0.7, name="Brent crude", move_pct=3.1)], scope="macro",
        event_type="commodity_supply", severity=0.8, confidence=0.9, hour=11)
    add(store, "tcs", [ent("TCS", polarity=0.7)], severity=0.9, confidence=0.9, hour=11)
    add(store, "trivia", [ent("HINDPETRO", polarity=0.0)], event_type="other", severity=0.1, hour=11)
    cats = pipe.catalysts_for("HINDPETRO", H.format(12), limit=5)
    assert all(isinstance(c, Catalyst) for c in cats)
    assert {c.headline for c in cats} == {"own", "oilsector", "brent"}
    own = next(c for c in cats if c.headline == "own")
    assert own.event_type == "litigation_regulatory" and own.polarity < 0 and own.source == "reuters"
    assert own.url == "https://reuters.example/10" and own.severity == 0.7 and 0.0 < own.novelty <= 1.0
    brent = next(c for c in cats if c.headline == "brent")
    assert brent.polarity < 0  # crude up is a headwind for an oil marketer: sign of the effect on THIS stock
    sector = next(c for c in cats if c.headline == "oilsector")
    assert sector.polarity == -0.5
    assert cats[0].headline in {"own", "brent"} and len(pipe.catalysts_for("HINDPETRO", H.format(12), limit=1)) == 1
    assert pipe.catalysts_for("HINDPETRO", H.format(12), limit=0) == []
    assert [c.headline for c in cats] == [c.headline for c in pipe.catalysts_for("HINDPETRO", H.format(12), limit=5)]  # deterministic
    assert [c.headline for c in pipe.catalysts_for("TCS", H.format(12))] == ["tcs"]


def test_market_reacted_events_are_dampened_in_the_ranking(store, pipe):
    add(store, "fresh", [ent("TCS", polarity=0.7)], severity=0.6, confidence=0.7, hour=10)
    add(store, "priced", [ent("TCS", polarity=0.7)], severity=0.62, confidence=0.7, hour=10, market_reacted=1,
        reaction_json=json.dumps({"reacted": True, "as_of": H.format(10)}))
    cats = pipe.catalysts_for("TCS", H.format(12), limit=2)
    assert [c.headline for c in cats] == ["fresh", "priced"]


def test_catalysts_can_include_low_severity_other_events_only_above_the_floor(store, pipe):
    add(store, "floor", [ent("TCS", polarity=0.0)], event_type="other", severity=0.25)
    assert [c.headline for c in pipe.catalysts_for("TCS", H.format(12))] == ["floor"]


# ----------------------------------------------------------------------------------------- reactions
def _seed_bars(store, symbol, last_day, n=60, *, gap=None):
    inst = store.instrument_id(symbol)
    days = pd.bdate_range(end=last_day, periods=n)
    rows = []
    price = 100.0
    for i, day in enumerate(days):
        price *= 1.0 + (0.004 if i % 2 else -0.004)
        rows.append({"trade_date": day.strftime("%Y-%m-%d"), "open": price, "high": price * 1.003, "low": price * 0.997,
                     "close": price, "volume": 1000.0})
    if gap is not None:
        next_day = (days[-1] + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        rows.append({"trade_date": next_day, "open": price * (1 + gap), "high": price * (1 + gap) * 1.01,
                     "low": price * (1 + gap) * 0.99, "close": price * (1 + gap * 1.05), "volume": 4000.0})
    store.upsert_bars(inst, rows, source="test", source_rank=2)


def test_refresh_reactions_marks_events_the_market_already_moved_on(store, pipe):
    _seed_bars(store, "TCS", "2026-11-02", gap=0.06)  # Monday history, Tuesday gap up
    _seed_bars(store, "INFY", "2026-11-02")
    add(store, "tcs", [ent("TCS", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:30:00Z", last_updated_at="2026-11-02T11:30:00Z")
    add(store, "infy", [ent("INFY", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:31:00Z", last_updated_at="2026-11-02T11:31:00Z")
    add(store, "nosym", [ent("macro:fed", type="institution", polarity=0.7)], scope="macro", event_type="central_bank",
        first_seen_at="2026-11-02T11:32:00Z")
    add(store, "other", [ent("TCS", polarity=0.7)], event_type="other", first_seen_at="2026-11-02T11:33:00Z")
    as_of = "2026-11-03T12:00:00Z"  # Tuesday 17:30 IST: the gap session's bar is known
    assert pipe.refresh_reactions(as_of) == 2
    rows = {e["headline"]: e for e in events(store)}
    assert rows["tcs"]["market_reacted"] == 1 and rows["infy"]["market_reacted"] is None
    assert rows["nosym"]["market_reacted"] is None and rows["other"]["market_reacted"] is None
    payload = json.loads(rows["tcs"]["reaction_json"])
    assert payload["reacted"] is True and payload["as_of"] == as_of and payload["symbol"] == "TCS" and payload["gap_pct"] > 5
    assert json.loads(rows["infy"]["reaction_json"])["reacted"] is None  # no Tuesday bar: unknown, never guessed
    assert pipe.refresh_reactions(as_of) == 0  # idempotent for the same as_of
    # features see the reaction only from the moment it was assessed
    assert pipe.event_features(["TCS"], "2026-11-03T12:30:00Z")["TCS"]["ev_market_reacted_share"] == 1.0
    assert pipe.event_features(["TCS"], "2026-11-02T13:00:00Z")["TCS"]["ev_market_reacted_share"] == 0.0


def test_refresh_reactions_uses_a_custom_provider_and_survives_its_failures(store, pipe):
    add(store, "tcs", [ent("TCS", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:30:00Z")
    add(store, "infy", [ent("INFY", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:31:00Z")
    calls = []

    def provider(symbol, start, end):
        calls.append((symbol, start, end))
        if symbol == "INFY":
            raise RuntimeError("vendor down")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    assert pipe.refresh_reactions("2026-11-03T12:00:00Z", provider) == 1
    assert [c[0] for c in calls] == ["INFY", "TCS"] and calls[0][2] == "2026-11-03"
    assert events(store)[0]["market_reacted"] is None
    assert json.loads(events(store)[0]["reaction_json"])["note"] == "no price bars"


def test_refresh_reactions_uses_the_preopen_quote_when_the_session_has_not_started(store, pipe):
    _seed_bars(store, "TCS", "2026-11-02")
    inst_close = float(store.get_bars(store.instrument_id("TCS")).iloc[-1]["close"])
    add(store, "tcs", [ent("TCS", polarity=0.7)], severity=0.6, first_seen_at="2026-11-02T11:30:00Z")
    store.add_preopen_quotes("2026-11-03", [{"symbol": "TCS", "iep": inst_close * 1.07}], "2026-11-03T03:40:00Z")
    assert pipe.refresh_reactions("2026-11-03T03:40:00Z") == 1  # 09:10 IST: pre-open, the session has not started
    assert events(store)[0]["market_reacted"] == 1
    assert "pre-open" in json.loads(events(store)[0]["reaction_json"])["note"]


def test_event_symbol_prefers_the_signed_subject():
    rows = json.dumps([ent("NTPC", polarity=0.0, confidence=0.95, in_title=False), ent("RELIANCE", polarity=0.6, confidence=0.8),
                       ent("global:NVDA", polarity=0.9), ent("sector:pharma", type="sector")])
    assert NewsPipeline._event_symbol(rows) == "RELIANCE"
    assert NewsPipeline._event_symbol(json.dumps([ent("sector:pharma", type="sector")])) is None
    assert NewsPipeline._event_symbol(None) is None


# ----------------------------------------------------------------------------------------------- pruning
def test_prune_removes_old_raw_rows_but_keeps_events(store, pipe):
    add_article(store, 1, "Reliance bags Rs 5,000 crore order from NTPC", "Reliance Industries received an order from NTPC.",
                source="Mint", feed="mint", url="https://www.livemint.com/old", seen="2026-08-01T10:00:00Z")
    add_article(store, 2, "TCS bags Rs 900 crore deal from European bank", "Multi-year deal.", source="Mint", feed="mint",
                url="https://www.livemint.com/new", seen="2026-11-01T10:00:00Z")
    add_filing(store, "OLD", symbol="INFY", category="Award of Order / Receipt of Order", subject="Infosys - Award of order",
               text="Infosys has received an order worth Rs 700 crore.", seen="2026-08-01T10:00:00Z", disseminated="2026-08-01T09:50:00Z")
    pipe.process_pending()
    add_article(store, 3, "Ancient unprocessed story", "Nothing.", source="Mint", feed="mint", url="https://www.livemint.com/c",
                seen="2026-07-01T10:00:00Z")
    add_filing(store, "PENDING", symbol="TCS", category="Award of Order / Receipt of Order", subject="TCS - Award of order",
               text="TCS has received an order worth Rs 900 crore.", seen="2026-08-02T10:00:00Z")
    assert len(events(store)) == 3 and count(store, "event_articles") == 3
    out = pipe.prune()
    assert out["news_raw"] == 2 and out["filings"] == 1 and out["event_articles_detached"] == 2
    assert [r[0] for r in store.conn.execute("SELECT url FROM news_raw")] == ["https://www.livemint.com/new"]
    assert [r[0] for r in store.conn.execute("SELECT source_ref FROM filings")] == ["PENDING"]  # unprocessed filings are kept
    assert len(events(store)) == 3 and count(store, "event_articles") == 3  # events and their links survive
    assert count(store, "event_articles", "article_id IS NULL AND filing_id IS NULL") == 2
    assert pipe.prune() == {"news_raw": 0, "filings": 0, "event_articles_detached": 0, "signatures": 0, "llm_cache": 0}


def test_prune_honours_the_retention_argument_and_cleans_kv(store, pipe, fake_clock):
    add_article(store, 1, "Reliance bags Rs 5,000 crore order from NTPC", "Reliance Industries received an order from NTPC.",
                source="Mint", feed="mint", url="https://www.livemint.com/a", seen="2026-10-25T10:00:00Z")
    pipe.process_pending()
    assert store.kv_get("news.sig:1") is not None
    assert pipe.prune(retention_days=30)["news_raw"] == 0
    out = pipe.prune(retention_days=2)
    assert out["news_raw"] == 1
    assert store.kv_get("news.sig:1") is not None  # the event is within the 14-day clustering window
    fake_clock.advance(20 * 86400)
    store.kv_set("llm:stale", "{}")
    fake_clock.advance(20 * 86400)
    out = pipe.prune()
    assert out["signatures"] == 1 and out["llm_cache"] >= 1 and store.kv_get("news.sig:1") is None
    assert len(events(store)) == 1  # structured events are kept
    assert store.kv_get("news.last_run") is not None  # unrelated keys survive


def test_prune_handles_many_rows(store, pipe):
    for i in range(30):
        add_article(store, i, f"Story {i}", "x", seen="2026-06-01T10:00:00Z", source="Mint", feed="mint", url=f"https://www.livemint.com/{i}")
    assert pipe.prune()["news_raw"] == 30 and count(store, "news_raw") == 0


def test_bulk_delete_skips_foreign_key_lookups_only_when_safe(store, pipe):
    conn = store.conn

    def fk():
        return conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert fk() == 1
    with pipe._bulk_delete():
        assert fk() == 0
    assert fk() == 1
    with store.tx(), pipe._bulk_delete():
        assert fk() == 1  # the pragma cannot change inside a transaction, so nothing is toggled
    with pytest.raises(RuntimeError), pipe._bulk_delete():
        raise RuntimeError("boom")
    assert fk() == 1  # restored even when the delete fails
    conn.execute("PRAGMA foreign_keys = OFF")
    with pipe._bulk_delete():
        assert fk() == 0
    assert fk() == 0  # it was off before, so it stays off
    conn.execute("PRAGMA foreign_keys = ON")


def test_prune_inside_a_caller_transaction_still_detaches_links(store, pipe):
    add_article(store, 1, TITLE_ORDER, "Reliance Industries received an order from NTPC.", source="Mint", feed="mint",
                url="https://www.livemint.com/old", seen="2026-08-01T10:00:00Z")
    pipe.process_pending()
    with store.tx():
        out = pipe.prune()
    assert out["news_raw"] == 1 and out["event_articles_detached"] == 1
    assert count(store, "event_articles", "article_id IS NULL") == 1 and store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
