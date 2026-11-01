from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from database.store import Store
from event_graph import sensitivity as sens
from event_graph.repository import GraphRepository
from event_graph.sensitivity import (
    EdgeEstimate,
    benjamini_hochberg,
    bh_adjust,
    estimate_edge,
    shrink_towards_prior,
    update_graph_from_returns,
)


def synth(n=900, lag=2, beta=0.8, noise=0.01, seed=1, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    x = pd.Series(rng.normal(0, 0.01, n), index=index)
    y = pd.Series(beta * x.shift(lag).fillna(0.0).to_numpy() + rng.normal(0, noise, n), index=index)
    return x, y


def null_series(n=900, seed=3, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    return pd.Series(rng.normal(0, 0.01, n), index=index)


# ------------------------------------------------------------------------------------------- estimation
def test_a_known_lag_and_beta_are_recovered():
    x, y = synth(lag=2, beta=0.8)
    est = estimate_edge(x, y, max_lag=3, min_obs=250)
    assert est.estimable and est.best_lag == 2 and est.beta == pytest.approx(0.8, abs=0.08)
    assert est.p_value < 1e-10 and est.t_stat > 10 and est.sign_stability >= 0.9 and est.oos_r2 > 0.3 and est.r2 > 0.3
    assert est.lags_tested == (0, 1, 2, 3) and est.n_obs >= 890
    assert est.betas[2] == pytest.approx(0.8, abs=0.08) and all(abs(est.betas[k]) < 0.1 for k in (0, 1, 3))
    assert {"driver_up", "driver_down", "high_vol", "low_vol"} <= set(est.regime_betas)
    assert all(v == pytest.approx(0.8, abs=0.25) for v in est.regime_betas.values())


def test_a_contemporaneous_negative_relationship_is_found_at_lag_zero():
    x, y = synth(lag=0, beta=-0.6, seed=4)
    est = estimate_edge(x, y)
    assert est.best_lag == 0 and est.beta == pytest.approx(-0.6, abs=0.08) and est.t_stat < -10 and est.sign_stability >= 0.9


def test_lead_only_edges_never_use_lag_zero():
    x, y = synth(lag=0, beta=0.9, seed=5)
    est = estimate_edge(x, y, min_lag=1)
    assert est.lags_tested == (1, 2, 3) and 0 not in est.betas and est.best_lag in (1, 2, 3)
    assert abs(est.beta) < 0.15 and est.p_value > 0.001  # the contemporaneous link cannot masquerade as a lead


def test_unrelated_series_are_not_significant():
    est = estimate_edge(null_series(seed=11), null_series(seed=12))
    assert est.estimable and est.p_value > 0.05 and est.oos_r2 < 0.01 and est.r2 < 0.01


def test_too_little_data_is_not_estimated():
    x, y = synth(n=100)
    est = estimate_edge(x, y, min_obs=250)
    assert not est.estimable and est.best_lag is None and "fewer than 250" in est.reason
    assert est.p_value == 1.0 and est.sign_stability == 0.0 and est.n_obs == 0
    assert estimate_edge(x, y, min_obs=50).estimable


def test_a_relationship_that_flips_sign_has_low_sign_stability():
    n = 900
    rng = np.random.default_rng(6)
    index = pd.bdate_range("2020-01-01", periods=n)
    x = pd.Series(rng.normal(0, 0.01, n), index=index)
    beta_t = np.where(np.arange(n) < 500, 0.8, -0.6)  # the channel reverses for the last 45% of the sample
    y = pd.Series(beta_t * x.to_numpy() + rng.normal(0, 0.01, n), index=index)
    est = estimate_edge(x, y, max_lag=0)
    assert est.beta == pytest.approx(0.8 * 500 / n - 0.6 * 400 / n, abs=0.1) and est.beta > 0
    assert est.sign_stability < 0.7
    steady = estimate_edge(*synth(lag=0, beta=0.8, seed=6), max_lag=0)
    assert steady.sign_stability >= 0.9


def test_bad_values_arrays_and_degenerate_inputs():
    x, y = synth(lag=1, beta=0.7, seed=7)
    dirty_x, dirty_y = x.copy(), y.copy()
    dirty_x.iloc[10] = np.nan
    dirty_y.iloc[20] = np.inf
    dirty_y.iloc[30] = -np.inf
    assert estimate_edge(dirty_x, dirty_y).best_lag == 1
    assert estimate_edge(x.to_numpy(), y.to_numpy()).best_lag == 1  # plain arrays are accepted
    flat = pd.Series(0.0, index=x.index)
    assert not estimate_edge(flat, y).estimable
    assert estimate_edge(x, y, regimes=False).regime_betas == {}
    assert sens._ols(np.zeros(5), np.arange(5.0)) is None
    assert sens._oos_r2(np.arange(10.0), np.arange(10.0), 4) == float("-inf")
    assert sens._sign_stability(np.arange(10.0), np.arange(10.0), 1, (120,)) == 0.0
    assert sens._robust_se(np.arange(5.0), np.zeros(5), 10.0) == float("inf")
    assert EdgeEstimate(None).estimable is False


# ----------------------------------------------------------------------------------------------- shrinkage
def test_shrinkage_blends_data_and_prior_by_precision():
    strong = estimate_edge(*synth(lag=0, beta=0.9, noise=0.003, seed=8))
    sensitivity, confidence = shrink_towards_prior(strong, 0.3, 0.3)
    assert 0.3 < sensitivity <= strong.beta + 1e-9 and sensitivity == pytest.approx(strong.beta, abs=0.05)
    assert 0.3 < confidence <= sens.CONFIDENCE_CEILING
    weak = EdgeEstimate(best_lag=0, beta=0.9, std_err=3.0, t_stat=0.3, p_value=0.8, n_obs=300, sign_stability=0.5)
    weak_sens, weak_conf = shrink_towards_prior(weak, 0.3, 0.3)
    assert weak_sens == pytest.approx(0.3, abs=0.05) and weak_conf < 0.35
    high_conf = shrink_towards_prior(strong, 0.3, 0.9)[0]
    low_conf = shrink_towards_prior(strong, 0.3, 0.2)[0]
    assert abs(high_conf - 0.3) <= abs(low_conf - 0.3)  # a more confident prior resists the data more
    assert shrink_towards_prior(strong, 0.3, 0.3, prior_sd=0.001)[0] == pytest.approx(0.3, abs=0.01)
    assert shrink_towards_prior(strong, None, 0.3)[0] > 0.8  # no prior sensitivity: the data speaks


def test_shrinkage_returns_the_prior_when_nothing_was_estimated():
    assert shrink_towards_prior(EdgeEstimate(None), 0.4, 0.33) == (0.4, 0.33)
    assert shrink_towards_prior(EdgeEstimate(0, beta=1.0, std_err=0.0), -0.2, 0.3) == (-0.2, 0.3)
    assert shrink_towards_prior(EdgeEstimate(0, beta=1.0, std_err=float("inf")), None, 0.3) == (0.0, 0.3)


# ------------------------------------------------------------------------------------------- multiple tests
def reference_bh(p, alpha):
    order = sorted(range(len(p)), key=lambda i: p[i])
    cutoff = -1
    for rank, i in enumerate(order, start=1):
        if p[i] <= alpha * rank / len(p):
            cutoff = rank
    rejected = set(order[:cutoff]) if cutoff > 0 else set()
    return [i in rejected for i in range(len(p))]


def test_p_values_are_adjusted_for_searching_over_lags():
    x, y = synth(lag=1, beta=0.05, noise=0.01, seed=13)
    searched = estimate_edge(x, y, max_lag=3)
    assert searched.p_raw <= searched.p_value <= 1.0
    assert searched.p_value == pytest.approx(1 - (1 - searched.p_raw) ** len(searched.betas), rel=1e-6)
    single = estimate_edge(x, y, max_lag=0)
    assert single.p_value == pytest.approx(single.p_raw)
    strong = estimate_edge(*synth(lag=2, beta=0.8))
    assert strong.p_value < 1e-10 and strong.p_raw <= strong.p_value
    # under the null the adjusted p-values are calibrated (the raw ones are inflated by picking the best lag)
    nulls = [estimate_edge(null_series(seed=300), null_series(seed=400 + i)) for i in range(80)]
    assert sum(e.p_value < 0.05 for e in nulls) < sum(e.p_raw < 0.05 for e in nulls)


def test_benjamini_hochberg_matches_the_reference_procedure():
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
    rng = np.random.default_rng(0)
    shuffled = [float(v) for v in rng.permutation(p)]
    for alpha in (0.01, 0.05, 0.10, 0.25):
        assert benjamini_hochberg(p, alpha) == reference_bh(p, alpha)
        assert benjamini_hochberg(shuffled, alpha) == reference_bh(shuffled, alpha)  # input order is preserved
    assert sum(benjamini_hochberg(p, 0.05)) == 4  # the textbook example from Benjamini & Hochberg (1995)
    assert benjamini_hochberg([], 0.1) == [] and benjamini_hochberg([0.9], 0.1) == [False] and benjamini_hochberg([0.01], 0.1) == [True]


def test_bh_adjusted_values_are_monotone_bounded_and_ordered():
    p = np.array([0.5, 0.001, 0.04, 0.03, 0.9])
    q = bh_adjust(p)
    assert q.shape == p.shape and (q >= p - 1e-12).all() and (q <= 1.0).all()
    order = np.argsort(p)
    assert (np.diff(q[order]) >= -1e-12).all()
    assert bh_adjust([]).size == 0


def test_fdr_control_rejects_far_fewer_nulls_than_a_naive_threshold():
    nulls = [estimate_edge(null_series(seed=100), null_series(seed=200 + i)) for i in range(60)]
    real = estimate_edge(*synth(lag=1, beta=0.7, seed=9))
    naive = sum(e.p_raw < 0.05 for e in nulls)
    mask = benjamini_hochberg([real.p_value, *(e.p_value for e in nulls)], 0.10)
    assert naive >= 3 and sum(mask[1:]) < naive and mask[0]


# ----------------------------------------------------------------------------------------- graph updates
@pytest.fixture()
def repo(store):
    repo = GraphRepository(store)
    for node_id, kind in (("macro:brent", "commodity"), ("macro:usdinr", "fx"), ("macro:noise", "macro"), ("macro:fed", "institution"),
                          ("HINDPETRO", "stock"), ("INFY", "stock"), ("INDIGO", "stock"), ("TCS", "stock"), ("sector:it_services", "sector")):
        repo.upsert_node(node_id, kind, node_id)
    repo.upsert_edge("macro:brent", "HINDPETRO", "input_cost", direction=-1, sensitivity=-0.5, lag_days_mean=0.0, lag_days_sd=0.5,
                     confidence=0.3, regime_dependence="policy", rationale="crude input costs: refiners buy crude", source="prior_expert")
    repo.upsert_edge("macro:usdinr", "INFY", "fx_translation", direction=1, sensitivity=0.4, lag_days_mean=0.0, lag_days_sd=0.5,
                     confidence=0.3, rationale="rupee revenue: weaker rupee lifts margins", source="prior_expert")
    repo.upsert_edge("macro:noise", "INDIGO", "exposure", direction=1, sensitivity=0.2, lag_days_mean=0.0, lag_days_sd=0.5,
                     confidence=0.3, rationale="spurious link: no real channel", source="prior_expert")
    repo.upsert_edge("macro:fed", "TCS", "rate_sensitivity", direction=1, sensitivity=0.1, lag_days_mean=1.0, lag_days_sd=0.5,
                     confidence=0.3, rationale="rate channel: lead link", source="prior_expert")
    repo.upsert_edge("INFY", "sector:it_services", "membership", direction=1, sensitivity=1.0, confidence=0.9, source="membership_curated")
    return repo


def frames(n=900, seed=21):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    drivers = pd.DataFrame({c: rng.normal(0, 0.01, n) for c in ("macro:brent", "macro:usdinr", "macro:noise", "macro:fed")}, index=index)
    targets = pd.DataFrame(
        {
            "HINDPETRO": -0.7 * drivers["macro:brent"] + rng.normal(0, 0.01, n),
            "INFY": -0.9 * drivers["macro:usdinr"] + rng.normal(0, 0.01, n),
            "INDIGO": rng.normal(0, 0.01, n),
            "TCS": 0.9 * drivers["macro:fed"] + rng.normal(0, 0.01, n),  # contemporaneous, but the prior says the driver leads
            "UNUSED": rng.normal(0, 0.01, n),
        },
        index=index,
    )
    return drivers, targets


def test_a_confirmed_prior_becomes_a_measured_edge_with_more_confidence(store, repo):
    drivers, targets = frames()
    summary = update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    edge = repo.get_edge("macro:brent", "HINDPETRO", "input_cost")
    assert edge.source == "measured" and edge.direction == -1 and edge.version == 2
    assert -0.75 < edge.sensitivity < -0.5 and edge.sensitivity == pytest.approx(-0.7, abs=0.08)
    assert 0.3 < edge.confidence <= 0.9 and edge.evidence_count >= 250 and edge.lag_days_mean == 0.0
    assert "crude input costs" in edge.rationale and edge.rationale.count("| measured beta") == 1
    assert "macro:brent|HINDPETRO|input_cost" in summary["updated"]
    record = store.kv_get_json(sens.EST_PREFIX + edge.key)
    assert record["prior"]["sensitivity"] == -0.5 and record["prior"]["confidence"] == 0.3 and record["passed"] is True
    assert record["best_lag"] == 0 and record["n_obs"] == edge.evidence_count and record["q"] <= 0.10


def test_an_opposite_sign_becomes_a_relationship_break_that_asserts_no_direction(store, repo):
    drivers, targets = frames()
    summary = update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    edge = repo.get_edge("macro:usdinr", "INFY", "fx_translation")
    assert edge.key in summary["breaks"] and edge.source == "relationship_break"
    # neither the stale prior sign nor a silent flip: a relationship that stopped working claims no direction at all
    assert edge.direction == 0 and edge.sensitivity is None and edge.confidence == sens.CONFIDENCE_FLOOR
    assert "RELATIONSHIP BREAK" in edge.rationale and "prior sign +1" in edge.rationale and "no directional claim" in edge.rationale
    assert re.search(r"measured beta -0\.[89]\d", edge.rationale) and edge.rationale.startswith("rupee revenue")
    assert edge.evidence_count >= 250
    record = store.kv_get_json(sens.EST_PREFIX + edge.key)
    assert record["prior"]["direction"] == 1 and record["prior"]["sensitivity"] == 0.4  # the prior stays on record
    assert [(h["change"], h["direction"], h["source"]) for h in repo.edge_history(edge.src, edge.dst, edge.relation)] == [
        ("created", 1, "prior_expert"), ("updated", 0, "relationship_break")]  # and the history shows when it broke


def test_a_relationship_break_is_logged_as_a_warning(store, repo, caplog):
    drivers, targets = frames()
    with caplog.at_level("WARNING", logger="event_graph.sensitivity"):
        update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    assert "relationship break macro:usdinr|INFY|fx_translation" in caplog.text and "prior sign +1" in caplog.text
    assert [r.levelname for r in caplog.records if "relationship break" in r.getMessage()] == ["WARNING"]


def test_a_broken_relationship_that_the_data_confirms_again_regains_its_prior_direction(store, repo):
    drivers, targets = frames()
    update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    assert repo.get_edge("macro:usdinr", "INFY", "fx_translation").direction == 0
    recovered = targets.copy()
    recovered["INFY"] = 0.9 * drivers["macro:usdinr"] + np.random.default_rng(5).normal(0, 0.01, len(targets))
    summary = update_graph_from_returns(store, drivers, recovered, as_of="2023-07-03")
    edge = repo.get_edge("macro:usdinr", "INFY", "fx_translation")
    assert "macro:usdinr|INFY|fx_translation" in summary["updated"] and summary["breaks"] == []
    assert edge.source == "measured" and edge.direction == 1 and edge.sensitivity > 0.3 and edge.confidence > 0.3
    assert "RELATIONSHIP BREAK" not in edge.rationale


def test_failing_the_gates_leaves_the_prior_and_decays_confidence_slightly(store, repo):
    drivers, targets = frames()
    first = update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    edge = repo.get_edge("macro:noise", "INDIGO", "exposure")
    assert edge.key in first["failed_gates"] and edge.key in first["decayed"]
    assert edge.source == "prior_expert" and edge.sensitivity == 0.2 and edge.direction == 1 and edge.evidence_count == 0
    assert edge.confidence == pytest.approx(0.3 * sens.DECAY) and edge.rationale == "spurious link: no real channel"
    again = update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    assert repo.get_edge("macro:noise", "INDIGO", "exposure").confidence == pytest.approx(0.3 * sens.DECAY)  # once per as_of
    assert edge.key in again["failed_gates"] and edge.key not in again["decayed"]
    update_graph_from_returns(store, drivers, targets, as_of="2023-07-03")
    assert repo.get_edge("macro:noise", "INDIGO", "exposure").confidence == pytest.approx(0.3 * sens.DECAY**2)


def test_a_driver_that_should_lead_is_not_confirmed_by_a_contemporaneous_link(store, repo):
    drivers, targets = frames()
    summary = update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    edge = repo.get_edge("macro:fed", "TCS", "rate_sensitivity")
    assert edge.key in summary["failed_gates"] and edge.source == "prior_expert"
    record = store.kv_get_json(sens.EST_PREFIX + edge.key)
    assert record["best_lag"] >= 1 and record["passed"] is False


def test_only_candidate_edges_are_tested(store, repo):
    drivers, targets = frames()
    summary = update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    assert summary["tested"] == 4  # the membership edge and columns without edges are ignored
    assert summary["passed"] == 2 and summary["alpha"] == 0.10 and summary["as_of"] == "2023-06-30"
    assert summary["skipped_insufficient"] == []
    assert update_graph_from_returns(store, drivers[["macro:brent"]], targets[["INDIGO"]], as_of="2023-06-30")["tested"] == 0


def test_estimation_is_idempotent_and_does_not_accumulate_text(store, repo):
    drivers, targets = frames()
    update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    versions = {e.key: (e.version, e.rationale, e.confidence) for e in repo.get_edges()}
    update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    assert {e.key: (e.version, e.rationale, e.confidence) for e in repo.get_edges()} == versions
    update_graph_from_returns(store, drivers, targets, as_of="2023-07-03")
    edge = repo.get_edge("macro:brent", "HINDPETRO", "input_cost")
    assert edge.rationale.count("| measured") == 1 and "as of 2023-07-03" in edge.rationale
    brk = repo.get_edge("macro:usdinr", "INFY", "fx_translation")
    assert brk.rationale.count("RELATIONSHIP BREAK") == 1


def test_insufficient_history_leaves_edges_untouched(store, repo):
    drivers, targets = frames(n=900)
    summary = update_graph_from_returns(store, drivers, targets, as_of="2020-06-30")  # only ~130 rows are known by then
    assert summary["tested"] == 4 and len(summary["skipped_insufficient"]) == 4 and summary["passed"] == 0
    assert {e.version for e in repo.get_edges()} == {1} and all(e.source != "measured" for e in repo.get_edges())
    assert not summary["decayed"] and store.kv_get_json(sens.EST_PREFIX + "macro:brent|HINDPETRO|input_cost") is None


def test_only_data_up_to_as_of_is_used(store, settings, fake_clock):
    drivers, targets = frames(n=1100)
    targets = targets.copy()
    late = targets.index >= targets.index[800]
    targets.loc[late, "INDIGO"] = 3.0 * drivers.loc[late, "macro:noise"] + 0.001  # a relationship that only exists in the future
    cutoff = targets.index[799]

    def run(path, d, t):
        db = Store.open(path, clock=fake_clock)
        try:
            r = GraphRepository(db)
            for node_id, kind in (("macro:noise", "macro"), ("INDIGO", "stock")):
                r.upsert_node(node_id, kind, node_id)
            r.upsert_edge("macro:noise", "INDIGO", "exposure", direction=1, sensitivity=0.2, lag_days_mean=0.0, lag_days_sd=0.5,
                          confidence=0.3, rationale="spurious link: none", source="prior_expert")
            summary = update_graph_from_returns(db, d, t, as_of=cutoff)
            return summary, r.get_edge("macro:noise", "INDIGO", "exposure"), db.kv_get_json(sens.EST_PREFIX + "macro:noise|INDIGO|exposure")
        finally:
            db.close()

    full = run(settings.data_dir / "full.db", drivers, targets)
    truncated = run(settings.data_dir / "cut.db", drivers.loc[:cutoff], targets.loc[:cutoff])
    assert full[0] == truncated[0] and full[1] == truncated[1] and full[2] == truncated[2]
    assert full[1].source == "prior_expert"  # the future relationship never leaked into the past estimate


def test_as_of_accepts_several_types(store, repo):
    drivers, targets = frames()
    stamps = {"2023-06-30", date(2023, 6, 30), datetime(2023, 6, 30, 12, 0), pd.Timestamp("2023-06-30T12:00:00+05:30"),
              datetime(2023, 6, 30, tzinfo=timezone.utc)}
    for value in stamps:
        assert update_graph_from_returns(store, drivers, targets, as_of=value)["as_of"] == "2023-06-30"


def test_timezone_aware_return_frames_are_accepted(store, repo):
    drivers, targets = frames()
    drivers.index = drivers.index.tz_localize("Asia/Kolkata")
    targets.index = targets.index.tz_localize("Asia/Kolkata")
    assert update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")["passed"] == 2


def test_the_multiple_testing_guard_protects_against_many_spurious_edges(store):
    repo = GraphRepository(store)
    rng = np.random.default_rng(55)
    index = pd.bdate_range("2020-01-01", periods=900)
    repo.upsert_node("macro:real", "macro", "real")
    drivers = {"macro:real": pd.Series(rng.normal(0, 0.01, 900), index=index)}
    targets = {}
    for i in range(40):
        repo.upsert_node(f"S{i}", "stock", f"S{i}")
        targets[f"S{i}"] = pd.Series(rng.normal(0, 0.01, 900), index=index)
        repo.upsert_edge("macro:real", f"S{i}", "exposure", direction=1, sensitivity=0.1, confidence=0.3, rationale="probe: noise",
                         source="prior_expert", lag_days_mean=0.0, lag_days_sd=0.5)
    targets["S0"] = 0.6 * drivers["macro:real"] + targets["S0"]
    summary = update_graph_from_returns(store, pd.DataFrame(drivers), pd.DataFrame(targets), as_of="2023-06-30")
    assert summary["tested"] == 40 and summary["updated"] == ["macro:real|S0|exposure"] and summary["breaks"] == []
    raw_hits = sum(1 for k in range(1, 40) if estimate_edge(drivers["macro:real"], targets[f"S{k}"]).p_raw < 0.05)
    assert raw_hits >= 1 and summary["passed"] == 1  # naive 5% testing would have "found" spurious links


def test_summary_keys_and_json_safety(store, repo):
    drivers, targets = frames()
    summary = update_graph_from_returns(store, drivers, targets, as_of="2023-06-30")
    assert set(summary) == {"as_of", "tested", "passed", "updated", "breaks", "decayed", "failed_gates", "skipped_insufficient", "alpha"}
    import json

    json.dumps(summary)
    for key in (e.key for e in repo.get_edges() if e.relation != "membership"):
        record = store.kv_get_json(sens.EST_PREFIX + key)
        assert record is not None and math.isfinite(record["p"]) and set(record["prior"]) >= {"sensitivity", "confidence", "direction"}
