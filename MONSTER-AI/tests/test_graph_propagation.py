from __future__ import annotations

import math
import random
import re

import pytest

from event_graph import propagation as prop
from event_graph.propagation import GraphImpactEngine, StockImpact
from event_graph.repository import GraphRepository
from event_graph.seed import seed_graph

AS_OF = "2026-11-01T12:00:00Z"


@pytest.fixture()
def repo(store):
    repo = GraphRepository(store)
    for node_id, kind, name in (
        ("macro:brent", "commodity", "Brent crude oil"), ("macro:opec", "institution", "OPEC"), ("macro:usdinr", "fx", "USD/INR"),
        ("sector:oil_marketing", "sector", "Oil marketing"), ("sector:upstream_oil", "sector", "Upstream oil"),
        ("sector:it_services", "sector", "IT services"), ("HINDPETRO", "stock", "HPCL"), ("BPCL", "stock", "BPCL"),
        ("ONGC", "stock", "ONGC"), ("TCS", "stock", "TCS"),
    ):
        repo.upsert_node(node_id, kind, name)
    causal = (
        ("macro:brent", "sector:oil_marketing", "input_cost", -1, "crude input costs: refiners and OMCs buy crude"),
        ("macro:brent", "sector:upstream_oil", "exposure", 1, "realisations: producers sell crude"),
        ("macro:opec", "macro:brent", "co_movement", 1, "supply policy: OPEC output sets the crude balance"),
        ("macro:usdinr", "sector:it_services", "fx_translation", 1, "rupee revenue: IT services bill in dollars"),
        ("sector:oil_marketing", "macro:brent", "co_movement", 1, "feedback: refiner demand loops back into crude"),
    )
    for src, dst, relation, direction, why in causal:
        repo.upsert_edge(src, dst, relation, direction=direction, sensitivity=0.5 * direction, lag_days_mean=0.0, lag_days_sd=0.5,
                         confidence=0.4, rationale=why, source="prior_expert", regime_dependence="depends on regime")
    for stock, sector in (("HINDPETRO", "sector:oil_marketing"), ("BPCL", "sector:oil_marketing"), ("ONGC", "sector:upstream_oil"),
                          ("TCS", "sector:it_services")):
        repo.upsert_edge(stock, sector, "membership", direction=1, sensitivity=1.0, confidence=0.9, rationale="member of sector",
                         source="membership_curated")
    return repo


def entity(ident, polarity=0.7, *, type="commodity", name=None, move=None, confidence=0.9):
    item = {"id": ident, "type": type, "name": name or ident, "polarity": polarity, "confidence": confidence}
    if move is not None:
        item["move_pct"] = move
    return item


def event(event_id=1, entities=None, *, severity=0.8, confidence=0.9, novelty=1.0, first_seen="2026-11-01T10:00:00Z", **extra):
    base = {"event_id": event_id, "event_type": "commodity_supply", "headline": "Brent jumps", "severity": severity,
            "confidence": confidence, "novelty": novelty, "first_seen_at": first_seen,
            "entities": entities if entities is not None else [entity("macro:brent", move=3.1, name="Brent crude oil")]}
    base.update(extra)
    return base


def weight(sev=0.8, conf=0.9, nov=1.0, polarity=0.7):
    return sev * conf * nov * (0.5 + 0.5 * abs(polarity))


# ---------------------------------------------------------------------------------------------- chains
def test_chains_are_readable_and_built_from_the_stored_rationale(store, repo):
    out = GraphImpactEngine(store).impacts([event()], AS_OF)
    assert set(out) == {"HINDPETRO", "BPCL", "ONGC"} and all(isinstance(v, StockImpact) for v in out.values())
    assert out["HINDPETRO"].chains[0] == "Brent +3.1% -> crude input costs [expert prior] -> sector:oil_marketing -> HINDPETRO"
    assert out["ONGC"].chains[0] == "Brent +3.1% -> realisations [expert prior] -> sector:upstream_oil -> ONGC"
    assert "TCS" not in out


def test_signed_exposure_strength_follows_the_edge_signs(store, repo):
    out = GraphImpactEngine(store).impacts([event()], AS_OF)
    expected = weight() * 0.4 * 0.9  # strength x edge confidence x membership confidence
    assert out["HINDPETRO"].direction == -1 and out["HINDPETRO"].score == pytest.approx(-math.tanh(expected), abs=1e-4)
    assert out["ONGC"].direction == 1 and out["ONGC"].score == pytest.approx(math.tanh(expected), abs=1e-4)
    assert out["HINDPETRO"].confidence == pytest.approx(0.4 * 0.9 * 0.9, abs=1e-4)
    assert out["HINDPETRO"].top_event_ids == [1] and out["HINDPETRO"].n_events == 1
    assert out["HINDPETRO"].event_effects == {1: pytest.approx(-expected, abs=1e-4)} and out["HINDPETRO"].basis == "prior"
    down = GraphImpactEngine(store).impacts([event(entities=[entity("macro:brent", -0.7, move=-3.1)])], AS_OF)
    assert down["HINDPETRO"].direction == 1 and down["ONGC"].direction == -1
    assert down["HINDPETRO"].chains[0].startswith("Brent -3.1% ->")


def test_chain_labels_without_a_percentage_or_rationale_label(store, repo):
    repo.upsert_edge("macro:usdinr", "sector:oil_marketing", "fx_translation", direction=1, sensitivity=0.3, confidence=0.4,
                     rationale="no label here", source="prior_expert")
    out = GraphImpactEngine(store).impacts([event(entities=[entity("macro:usdinr", 0.6, type="fx", name="USD/INR")])], AS_OF)
    assert out["HINDPETRO"].chains[0] == "USD/INR up -> fx translation [expert prior] -> sector:oil_marketing -> HINDPETRO"
    assert out["TCS"].chains[0] == "USD/INR up -> rupee revenue [expert prior] -> sector:it_services -> TCS"
    assert prop._short("macro:in10y", "x") == "India 10y yield" and prop._short("macro:custom_thing", "x") == "Custom thing"
    assert prop._short("sector:x", "Nifty (index)") == "Nifty"


def test_strength_decays_with_causal_depth_and_depth_is_bounded(store, repo):
    engine = GraphImpactEngine(store)
    direct = engine.impacts([event()], AS_OF)["HINDPETRO"]
    opec = engine.impacts([event(entities=[entity("macro:opec", 0.7, type="institution", name="OPEC")])], AS_OF)["HINDPETRO"]
    assert abs(opec.event_effects[1]) < abs(direct.event_effects[1])
    assert opec.event_effects[1] == pytest.approx(-weight() * 0.4 * (0.4 * 0.6) * 0.9, abs=1e-4)  # second causal hop x decay
    assert opec.chains[0] == (
        "OPEC up -> supply policy [expert prior] -> macro:brent -> crude input costs [expert prior] -> sector:oil_marketing -> HINDPETRO")
    assert engine.impacts([event()], AS_OF, max_depth=1) == {}
    assert set(engine.impacts([event()], AS_OF, max_depth=2)) == {"HINDPETRO", "BPCL", "ONGC"}
    assert engine.impacts([event(entities=[entity("macro:opec", 0.7, type="institution")])], AS_OF, max_depth=3) != {}
    assert engine.impacts([event(entities=[entity("macro:opec", 0.7, type="institution")])], AS_OF, max_depth=2) == {}
    harsher = engine.impacts([event(entities=[entity("macro:opec", 0.7, type="institution")])], AS_OF, decay=0.3, min_weight=0.001)
    assert abs(harsher["HINDPETRO"].event_effects[1]) < abs(opec.event_effects[1])


def test_weak_paths_are_pruned_by_min_weight(store, repo):
    engine = GraphImpactEngine(store)
    opec = [event(entities=[entity("macro:opec", 0.7, type="institution")])]
    assert "HINDPETRO" in engine.impacts(opec, AS_OF, min_weight=0.05)
    assert engine.impacts(opec, AS_OF, min_weight=0.06) == {}
    assert engine.impacts([event()], AS_OF, min_weight=0.9) == {}


def test_cycles_terminate_and_never_revisit_a_node(store, repo):
    engine = GraphImpactEngine(store)
    out = engine.impacts([event(), event(2, [entity("macro:opec", 0.9, type="institution")])], AS_OF, max_depth=8, min_weight=1e-6)
    assert out
    for impact in out.values():
        for chain in impact.chains:
            nodes = [part for part in chain.split(" -> ") if re.fullmatch(r"(?:macro|sector):\w+|[A-Z]+", part)]
            assert len(nodes) == len(set(nodes)), chain


# ------------------------------------------------------------------------------------- event weighting
def test_events_first_seen_after_as_of_are_ignored(store, repo):
    late = event(first_seen="2026-11-01T12:00:01Z")
    assert GraphImpactEngine(store).impacts([late], AS_OF) == {}
    assert GraphImpactEngine(store).impacts([late], "2026-11-01T12:00:01Z") != {}
    undated = event()
    undated.pop("first_seen_at")
    assert GraphImpactEngine(store).impacts([undated], AS_OF) != {}


def test_priced_in_events_are_discounted_and_novelty_scales_the_weight(store, repo):
    engine = GraphImpactEngine(store)
    base = engine.impacts([event()], AS_OF)["HINDPETRO"].event_effects[1]
    for reacted in (True, 1):
        assert engine.impacts([event(market_reacted=reacted)], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(base * 0.5, rel=1e-3)
    assert engine.impacts([event(reaction={"reacted": True})], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(base * 0.5, rel=1e-3)
    for untouched in (False, None, 0):
        assert engine.impacts([event(market_reacted=untouched)], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(base, rel=1e-3)
    assert engine.impacts([event(novelty=0.5)], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(base * 0.5, rel=1e-3)
    assert engine.impacts([event(severity=0.4)], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(base * 0.5, rel=1e-3)
    assert engine.impacts([event(confidence=0.45)], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(base * 0.5, rel=1e-3)


def test_larger_polarity_gives_a_larger_strength(store, repo):
    engine = GraphImpactEngine(store)
    mild = engine.impacts([event(entities=[entity("macro:brent", 0.2)])], AS_OF)["HINDPETRO"].event_effects[1]
    strong = engine.impacts([event(entities=[entity("macro:brent", 1.0)])], AS_OF)["HINDPETRO"].event_effects[1]
    assert abs(strong) > abs(mild) and abs(strong) == pytest.approx(weight(polarity=1.0) * 0.36, abs=1e-4)
    assert engine.impacts([event(entities=[entity("macro:brent", 9.0)])], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(strong)


def test_several_events_aggregate_with_a_bounded_score(store, repo):
    engine = GraphImpactEngine(store)
    events = [event(1), event(2, [entity("macro:opec", 0.9, type="institution")]), event(3, severity=1.0, confidence=1.0)]
    out = engine.impacts(events, AS_OF)["HINDPETRO"]
    assert out.n_events == 3 and set(out.top_event_ids) == {1, 2, 3} and len(out.chains) <= prop.MAX_CHAINS
    total = sum(out.event_effects.values())
    assert out.score == pytest.approx(math.tanh(total), abs=1e-3) and -1.0 < out.score < 0.0
    assert abs(out.event_effects[3]) > abs(out.event_effects[1]) > abs(out.event_effects[2])
    assert out.top_event_ids[0] == 3 and out.confidence <= 0.99
    many = engine.impacts([event(i, severity=1.0, confidence=1.0) for i in range(1, 60)], AS_OF)["HINDPETRO"]
    assert -1.0 <= many.score < 0 and abs(many.score) > 0.9


def test_results_do_not_depend_on_input_order(store, repo):
    engine = GraphImpactEngine(store)
    events = [event(1), event(2, [entity("macro:opec", 0.9, type="institution")]), event(3, [entity("macro:usdinr", 0.5, type="fx")]),
              event(4, [entity("macro:brent", -0.4), entity("macro:opec", 0.3, type="institution")], first_seen="2026-11-01T09:00:00Z")]
    baseline = engine.impacts(events, AS_OF)
    for seed in range(6):
        shuffled = events[:]
        random.Random(seed).shuffle(shuffled)
        assert engine.impacts(shuffled, AS_OF) == baseline
    assert list(baseline) == sorted(baseline)


def test_direct_company_events_and_global_names(store, repo):
    engine = GraphImpactEngine(store)
    direct = event(5, [entity("TCS", 0.6, type="company", name="TCS")], event_type="order_win", headline="TCS wins a large deal")
    out = engine.impacts([direct], AS_OF)["TCS"]
    assert out.basis == "direct" and out.direction == 1 and out.chains == ["order_win: TCS wins a large deal -> TCS"]
    assert out.score == pytest.approx(math.tanh(weight(polarity=0.6)), abs=1e-4)
    glob = event(6, [entity("global:AAPL", 0.8, type="company", name="Apple")])
    assert engine.impacts([glob], AS_OF) == {}  # a foreign company is not an NSE stock
    both = engine.impacts([direct, event(7, [entity("macro:usdinr", 0.5, type="fx")])], AS_OF)["TCS"]
    assert both.n_events == 2 and both.basis in ("direct", "prior")


def test_invalid_events_and_entities_are_ignored(store, repo):
    engine = GraphImpactEngine(store)
    junk = [
        "not a mapping", None, 5,
        event(1, [entity("macro:brent", 0.0)]), event(2, [entity("macro:brent", float("nan"))]), event(3, [entity("macro:brent", "abc")]),
        event(4, [{"polarity": 0.7}]), event(5, [entity("macro:unknown", 0.7)]), event(6, "not a list"), event(7, ["junk", 5]),
        event(8, severity=0.0), event(9, severity=float("nan")), event(10, confidence="bad"), event(11, novelty=-3),
    ]
    assert engine.impacts(junk, AS_OF) == {}
    ok = engine.impacts([event(1, [entity("macro:brent", "0.7")])], AS_OF)
    assert "HINDPETRO" in ok  # numeric strings are accepted
    crowded = event(1, [entity(f"macro:unknown{i}", 0.7) for i in range(30)] + [entity("macro:brent", 0.7)])
    assert engine.impacts([crowded], AS_OF) == {}  # only the first entities of an event are considered


def test_zero_direction_edges_carry_nothing(store, repo):
    repo.upsert_edge("macro:usdinr", "sector:oil_marketing", "co_movement", direction=0, sensitivity=0.0, confidence=0.4,
                     rationale="unclear: sign unknown", source="prior_expert")
    out = GraphImpactEngine(store).impacts([event(entities=[entity("macro:usdinr", 0.7, type="fx")])], AS_OF)
    assert "HINDPETRO" not in out and "TCS" in out


def mark(repo, src, dst, relation, source, **over):
    """Re-label an edge's provenance (optionally with new values) the way graph learning would."""
    edge = repo.get_edge(src, dst, relation)
    values = {"direction": edge.direction, "sensitivity": edge.sensitivity, "lag_days_mean": edge.lag_days_mean,
              "lag_days_sd": edge.lag_days_sd, "confidence": edge.confidence, "regime_dependence": edge.regime_dependence,
              "rationale": edge.rationale, "source": source, "evidence_count": 300 if source != "prior_expert" else 0}
    repo.upsert_edge(src, dst, relation, **{**values, **over})


def test_basis_reports_the_provenance_of_the_edges_used(store, repo):
    engine = GraphImpactEngine(store)
    evt = [event()]

    assert engine.impacts(evt, AS_OF)["HINDPETRO"].basis == "prior"
    mark(repo, "macro:brent", "sector:oil_marketing", "input_cost", "measured")
    assert engine.impacts(evt, AS_OF)["HINDPETRO"].basis == "measured"
    mixed = engine.impacts([event(entities=[entity("macro:opec", 0.7, type="institution")])], AS_OF)["HINDPETRO"]
    assert mixed.basis == "mixed"
    mark(repo, "macro:brent", "sector:oil_marketing", "input_cost", "relationship_break")
    assert engine.impacts(evt, AS_OF)["HINDPETRO"].basis == "relationship_break"
    assert engine.impacts(evt, AS_OF)["ONGC"].basis == "prior"


def test_chains_say_for_every_edge_whether_it_is_a_measured_estimate_or_an_expert_prior(store, repo):
    mark(repo, "macro:brent", "sector:oil_marketing", "input_cost", "measured")
    engine = GraphImpactEngine(store)
    out = engine.impacts([event()], AS_OF)
    assert out["HINDPETRO"].chains[0] == "Brent +3.1% -> crude input costs [measured] -> sector:oil_marketing -> HINDPETRO"
    assert out["ONGC"].chains[0] == "Brent +3.1% -> realisations [expert prior] -> sector:upstream_oil -> ONGC"
    opec = engine.impacts([event(entities=[entity("macro:opec", 0.7, type="institution", name="OPEC")])], AS_OF)["HINDPETRO"]
    assert opec.chains[0] == (
        "OPEC up -> supply policy [expert prior] -> macro:brent -> crude input costs [measured] -> sector:oil_marketing -> HINDPETRO")
    assert opec.basis == "mixed"  # the tags and the basis tell the same story
    repo.upsert_edge("macro:brent", "ONGC", "exposure", direction=1, sensitivity=0.3, confidence=0.6, rationale="direct: spot crude",
                     source="prior")  # the default provenance of an edge nobody measured is an expert prior too
    assert engine.impacts([event()], AS_OF)["ONGC"].chains[0].endswith("direct [expert prior] -> ONGC")


def test_a_broken_edge_propagates_with_no_direction_and_is_labelled_broken(store, repo):
    mark(repo, "macro:brent", "sector:oil_marketing", "input_cost", "relationship_break", direction=0, sensitivity=None,
         confidence=0.05)
    out = GraphImpactEngine(store).impacts([event()], AS_OF)
    broken = out["HINDPETRO"]
    assert broken.chains == ["Brent +3.1% -> crude input costs [broken] -> sector:oil_marketing -> HINDPETRO"]
    assert broken.score == 0.0 and broken.direction == 0 and broken.basis == "relationship_break"
    assert broken.confidence <= 0.05  # the confidence floor is carried along, not hidden
    assert broken.top_event_ids == [] and broken.event_effects == {} and broken.n_events == 0  # nothing is claimed as an effect
    assert out["BPCL"].chains[0].endswith("[broken] -> sector:oil_marketing -> BPCL")
    assert out["ONGC"].direction == 1 and "[broken]" not in out["ONGC"].chains[0]  # unrelated links are unaffected


def test_a_legacy_break_row_that_still_carries_a_direction_asserts_nothing(store, repo):
    mark(repo, "macro:brent", "sector:oil_marketing", "input_cost", "relationship_break")  # keeps its old sign and confidence
    assert repo.get_edge("macro:brent", "sector:oil_marketing", "input_cost").direction == -1
    impact = GraphImpactEngine(store).impacts([event()], AS_OF)["HINDPETRO"]
    assert impact.score == 0.0 and impact.direction == 0 and impact.event_effects == {}
    assert impact.chains == ["Brent +3.1% -> crude input costs [broken] -> sector:oil_marketing -> HINDPETRO"]


def test_a_shock_that_met_a_broken_link_stays_directionless_further_down(store, repo):
    mark(repo, "macro:opec", "macro:brent", "co_movement", "relationship_break", direction=0, sensitivity=None, confidence=0.05)
    opec = [event(entities=[entity("macro:opec", 0.7, type="institution", name="OPEC")])]
    impact = GraphImpactEngine(store).impacts(opec, AS_OF)["HINDPETRO"]
    assert impact.chains == [
        "OPEC up -> supply policy [broken] -> macro:brent -> crude input costs [expert prior] -> sector:oil_marketing -> HINDPETRO"]
    assert impact.score == 0.0 and impact.event_effects == {} and impact.basis == "mixed"
    assert GraphImpactEngine(store).impacts(opec, AS_OF, min_weight=0.9) == {}  # a shock too weak to matter is not reported


def test_a_broken_link_adds_nothing_next_to_a_working_one(store, repo):
    mark(repo, "macro:brent", "sector:oil_marketing", "input_cost", "relationship_break", direction=0, sensitivity=None,
         confidence=0.05)
    repo.upsert_edge("macro:usdinr", "sector:oil_marketing", "fx_translation", direction=1, sensitivity=0.3, confidence=0.4,
                     rationale="fx translation: weaker rupee", source="prior_expert")
    both = event(entities=[entity("macro:brent", 0.7, move=3.1, name="Brent crude oil"),
                           entity("macro:usdinr", 0.6, type="fx", name="USD/INR")])
    impact = GraphImpactEngine(store).impacts([both], AS_OF)["HINDPETRO"]
    expected = weight(polarity=0.6) * 0.4 * 0.9
    assert impact.direction == 1 and impact.score == pytest.approx(math.tanh(expected), abs=1e-4)
    assert impact.event_effects == {1: pytest.approx(expected, abs=1e-4)} and impact.top_event_ids == [1] and impact.n_events == 1
    assert impact.chains == [
        "USD/INR up -> fx translation [expert prior] -> sector:oil_marketing -> HINDPETRO",
        "Brent +3.1% -> crude input costs [broken] -> sector:oil_marketing -> HINDPETRO"]
    assert impact.basis == "mixed"


def test_the_graph_cache_follows_edge_changes(store, repo):
    engine = GraphImpactEngine(store)
    before = engine.impacts([event()], AS_OF)["HINDPETRO"].event_effects[1]
    repo.upsert_edge("macro:brent", "sector:oil_marketing", "input_cost", direction=-1, sensitivity=-0.5, lag_days_mean=0.0,
                     lag_days_sd=0.5, confidence=0.2, rationale="crude input costs: halved belief", source="prior_expert")
    assert engine.impacts([event()], AS_OF)["HINDPETRO"].event_effects[1] == pytest.approx(before / 2, rel=1e-2)


# ----------------------------------------------------------------------------------- the shipped priors
def test_with_the_shipped_priors_crude_hurts_marketers_and_helps_producers_but_carries_regime_caveats(store):
    from tests.test_news_support import add_instruments

    add_instruments(store)
    seed_graph(store)
    out = GraphImpactEngine(store).impacts([event(entities=[entity("macro:brent", 0.7, name="Brent crude oil", move=3.1)])], AS_OF)
    assert out["HINDPETRO"].direction == -1 and out["ONGC"].direction == 1
    assert out["HINDPETRO"].chains[0].startswith("Brent +3.1% -> ") and out["HINDPETRO"].chains[0].endswith("-> HINDPETRO")
    assert out["HINDPETRO"].basis == "prior" and out["HINDPETRO"].confidence < 0.4
    omc = GraphRepository(store).get_edge("macro:brent", "sector:oil_marketing", "input_cost")
    assert omc.regime_dependence and omc.source == "prior_expert" and omc.evidence_count == 0
    usd = GraphImpactEngine(store).impacts([event(entities=[entity("macro:usdinr", 0.7, type="fx", name="USD/INR")])], AS_OF)
    assert usd["INFY"].direction == 1 and usd["TCS"].direction == 1


def test_every_curated_stock_can_be_reached_by_some_event(store):
    seed_graph(store)
    repo = GraphRepository(store)
    graph = repo.load()
    stocks = {n for n, node in graph.nodes.items() if node.node_type == "stock"}
    drivers = [n for n, node in graph.nodes.items() if node.node_type in ("macro", "commodity", "fx", "rate", "index", "country", "institution")]
    reached: set[str] = set()
    engine = GraphImpactEngine(store)
    for start in range(0, len(drivers), 10):
        events = [event(i, [entity(d, 0.9, type=graph.nodes[d].node_type)], severity=1.0, confidence=1.0)
                  for i, d in enumerate(drivers[start : start + 10], start=1)]
        reached |= set(engine.impacts(events, AS_OF, max_depth=5, min_weight=0.001))
    assert not (stocks - reached), sorted(stocks - reached)
