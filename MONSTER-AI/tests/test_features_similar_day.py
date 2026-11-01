"""Similar-day engine: training and live scans must look at the SAME sessions. Each date is matched against the most recent POOL
sessions before it and standardised over the previous STD_WINDOW sessions, so the x_sim_* features do not depend on how much
history a caller happens to load (a live scan loads a bounded window; training sees years). The end-to-end check that a live
window reproduces the training row lives in test_features_live_window.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import similar_day as sd
from features.builder import build_feature_set
from tests.synthetic import make_panel

SMALL = dict(k=5, pool=40, window=40, warmup=10, min_history=15)  # the same rules at a size that is cheap to reason about


def vectors(n: int = 200, dims: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, dims)), index=pd.bdate_range("2024-01-01", periods=n),
                        columns=[f"d{i}" for i in range(dims)])


def labels(n: int = 200, stocks: int = 3, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: (rng.random((n, stocks)) * 4.0).astype(np.float32) for k in sd.LABELS_USED}


def run(vec: pd.DataFrame, lab: dict[str, np.ndarray], rows=None, **kw):
    stock, market = sd.similar_day_features(vec, lab, rows=rows, **{**SMALL, **kw})
    return {**stock, **market}


# ------------------------------------------------------------------------------------------ standardisation
def test_z_scores_use_a_trailing_window_ending_at_the_previous_session():
    vec = vectors(120, 3)
    z = sd.standardize(vec, min_periods=10, window=20)
    r = 70
    past = vec.iloc[r - 20:r]
    expected = ((vec.iloc[r] - past.mean()) / past.std()).clip(-5, 5).to_numpy()
    np.testing.assert_allclose(z[r], expected, rtol=1e-5)

    outside = vec.copy()
    outside.iloc[r - 21] += 100.0  # the session just before the window: irrelevant
    np.testing.assert_array_equal(sd.standardize(outside, min_periods=10, window=20)[r], z[r])
    inside = vec.copy()
    inside.iloc[r - 20] += 100.0  # the oldest session of the window: matters
    assert not np.allclose(sd.standardize(inside, min_periods=10, window=20)[r], z[r])


def test_a_session_is_not_part_of_its_own_statistics():
    vec = vectors(60, 2)
    vec.iloc[40, 0] = 1e6  # a 20-session window that contained it would cap the z-score near 4.3; excluded, it clips at 5
    z = sd.standardize(vec, min_periods=10, window=20)
    assert z[40, 0] == 5.0 and np.isfinite(z).all()
    assert abs(z[41, 0]) < 1.0  # ... while it inflates the statistics of the sessions that FOLLOW it


def test_rows_before_the_warm_up_are_neutral():
    z = sd.standardize(vectors(80, 2), min_periods=10, window=30)
    assert (z[:10] == 0.0).all() and (z[10:] != 0.0).any()


def test_the_default_window_is_bounded_so_old_history_cannot_matter():
    vec = vectors(700, 2)
    z = sd.standardize(vec)
    older = vec.copy()
    older.iloc[:100] *= 50.0  # more than STD_WINDOW sessions before row 400
    np.testing.assert_array_equal(sd.standardize(older)[400:], z[400:])
    assert not np.allclose(sd.standardize(older)[150:300], z[150:300])


# ------------------------------------------------------------------------------------- the candidate pool
def twin_case():
    """Row 150 is a once-seen extreme day; its only twin (row 70) is older than the pool. Every other day is ordinary."""
    vec = vectors(160, 4, seed=3)
    vec.iloc[70] = 6.0
    vec.iloc[150] = 6.0
    lab = labels(160, 2)
    for name in lab:
        lab[name][:] = 1.0
    lab["y_range"][70] = 99.0
    return vec, lab


def test_only_the_most_recent_sessions_before_the_date_are_candidates():
    vec, lab = twin_case()
    pooled = run(vec, lab, rows=[150])
    assert pooled["x_sim_range_mean"][150].tolist() == [1.0, 1.0]  # the old twin is outside the 40-session pool
    unbounded = run(vec, lab, rows=[150], pool=10_000)
    assert (unbounded["x_sim_range_mean"][150] > 1.0).all()  # ... and would have been the nearest neighbour otherwise


def test_sessions_beyond_the_pool_cannot_change_a_date():
    vec, lab = twin_case()
    base = run(vec, lab, rows=[150])
    older_v, older_l = vec.copy(), {k: v.copy() for k, v in lab.items()}
    older_v.iloc[:60] = np.random.default_rng(9).normal(size=(60, 4)) * 50  # more than pool + window before the date
    for v in older_l.values():
        v[:110] = 77.0  # outcomes older than the pool
    perturbed = run(older_v, older_l, rows=[150])
    for name, values in base.items():
        np.testing.assert_array_equal(values[150], perturbed[name][150], err_msg=name)
    recent_v = vec.copy()
    recent_v.iloc[140] += 4.0  # a session inside the pool does matter
    assert not np.allclose(run(recent_v, lab, rows=[150])["x_sim_dist_mean"][150], base["x_sim_dist_mean"][150])


def test_no_features_until_there_are_enough_candidates():
    out = run(vectors(40), labels(40), pool=3)  # fewer candidates than neighbours wanted
    assert all(np.isnan(v).all() for v in out.values())
    early = run(vectors(60), labels(60))
    assert np.isnan(early["x_sim_dist_mean"][:15]).all() and np.isfinite(early["x_sim_dist_mean"][15:]).all()


def test_neighbours_are_earlier_sessions_only():
    vec, lab = twin_case()
    lab["y_range"][:] = np.arange(160, dtype=np.float32)[:, None]  # a label that identifies the session it belongs to
    picked = run(vec, lab, rows=[100])["x_sim_range_mean"][100]
    assert (picked >= 60.0).all() and (picked < 100.0).all()  # mean of sessions from 60..99, never a later one


def test_rows_limit_the_computation_without_changing_it():
    vec, lab = vectors(), labels()
    full = run(vec, lab)
    one = run(vec, lab, rows=[150, 199])
    for name, values in full.items():
        np.testing.assert_array_equal(one[name][[150, 199]], values[[150, 199]], err_msg=name)
        assert np.isnan(one[name][[0, 100, 198]]).all()


def test_the_regime_penalty_keeps_neighbours_in_the_targets_volatility_regime():
    vec, lab = vectors(), labels()
    lab["y_range"][:] = (np.arange(200) % 2).astype(np.float32)[:, None]  # 1.0 for odd sessions
    regime = (np.arange(200) % 2) * 3.0  # alternating calm (0) and crisis (3) sessions
    plain = sd.similar_day_features(vec, lab, rows=[199], **SMALL)
    penalised = sd.similar_day_features(vec, lab, rows=[199], volreg=regime, penalty=50.0, **SMALL)
    assert penalised[0]["x_sim_range_mean"][199].tolist() == [1.0] * 3  # session 199 is odd: only odd sessions are chosen
    assert plain[0]["x_sim_range_mean"][199][0] < 1.0  # ... which is not what plain distance does
    assert penalised[1]["x_sim_dist_mean"][199] > plain[1]["x_sim_dist_mean"][199]


def test_an_unknown_regime_counts_as_neutral():
    vec, lab = vectors(), labels()
    unknown = sd.similar_day_features(vec, lab, rows=[199], volreg=np.full(200, np.nan), **SMALL)
    neutral = sd.similar_day_features(vec, lab, rows=[199], **SMALL)
    for name, values in neutral[0].items():
        np.testing.assert_array_equal(unknown[0][name], values, err_msg=name)
    np.testing.assert_array_equal(unknown[1]["x_sim_dist_mean"], neutral[1]["x_sim_dist_mean"])


def test_a_history_covering_the_pool_and_the_window_reproduces_the_full_history_result():
    vec, lab = vectors(260), labels(260)
    need = SMALL["pool"] + SMALL["window"]
    full = run(vec, lab, rows=[259])
    for length in (need + 1, need + 30):
        live = run(vec.iloc[-length:], {k: v[-length:] for k, v in lab.items()}, rows=[length - 1])
        for name, values in full.items():
            np.testing.assert_allclose(live[name][length - 1], values[259], rtol=1e-6, err_msg=f"{name} ({length} sessions)")
    short = SMALL["pool"] + SMALL["window"] // 2
    skewed = run(vec.iloc[-short:], {k: v[-short:] for k, v in lab.items()}, rows=[short - 1])
    assert not np.allclose(skewed["x_sim_dist_mean"][short - 1], full["x_sim_dist_mean"][259])  # too little history: different


def test_what_the_engine_may_depend_on_covers_the_pool_and_its_statistics():
    assert sd.HISTORY_SESSIONS >= sd.POOL + sd.STD_WINDOW
    assert sd.HISTORY_SESSIONS >= sd.POOL + 252 + 20  # the volatility regime of the oldest candidate ranks the VIX over a year


# ------------------------------------------------------------------------------------------ sample size
def test_the_match_count_and_the_per_stock_valid_count_are_reported():
    vec, lab = vectors(), labels()
    for name in lab:
        lab[name][:, 1] = np.nan  # a stock with no outcomes at all
        lab[name][:185, 2] = np.nan  # a stock whose history starts late
        lab[name][185:, 2] = 10.0
    out = run(vec, lab, rows=[199])
    assert out["x_sim_n"][199] == SMALL["k"]
    valid = out["x_sim_nvalid"][199]
    assert valid[0] == SMALL["k"] and valid[1] == 0 and 0 <= valid[2] <= SMALL["k"]
    assert np.isnan(out["x_sim_hit3"][199][1]) and np.isfinite(out["x_sim_hit3"][199][0])
    assert np.isfinite(out["x_sim_range_mean"][199][2]) == (valid[2] > 0)  # statistics exist exactly where outcomes do
    if valid[2] > 0:
        assert out["x_sim_range_mean"][199][2] == 10.0


def test_the_match_count_is_a_market_level_value_and_the_valid_count_stays_a_feature():
    panel = make_panel(n_dates=330, n_symbols=10, seed=5, with_fno=False, with_meetings=False)
    fs = build_feature_set(panel, stage="pre", thresholds=(1.0, 2.0, 3.0, 5.0), unusual_k=1.5, cutoff_hhmm="08:45",
                           with_labels=False, row_slice=slice(320, 330))
    assert {"x_sim_n", "x_sim_dist_mean"} <= set(fs.market.columns) and "x_sim_nvalid" not in fs.market.columns
    assert (fs.market["x_sim_n"] == sd.K).all() and fs.X["x_sim_nvalid"].between(0, sd.K).all()
