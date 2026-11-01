"""Day-block bootstrap, rank IC per day and the random-ranking reference: intervals that respect dependence between picks."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from backtesting import metrics as M

nan = float("nan")


def isnan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def clustered_days(days: int = 30, picks: int = 5) -> np.ndarray:
    """Every pick of a day shares its outcome (all hit or all miss): the extreme of within-day dependence."""
    return np.array([[float(d % 2 == 0)] * picks for d in range(days)])


# ------------------------------------------------------------------------------------------- the helper
def test_day_block_bootstrap_needs_a_minimum_number_of_days_and_is_deterministic():
    few = np.ones((M.MIN_BOOT_DAYS - 1, 5))
    assert all(isnan(v) for v in M.day_block_bootstrap(few))
    data = np.random.default_rng(1).normal(size=(40, 5))
    first = M.day_block_bootstrap(data, n_boot=200)
    assert first == M.day_block_bootstrap(data, n_boot=200)  # fixed seed: a re-run reproduces the stored interval
    assert first != M.day_block_bootstrap(data, n_boot=200, seed=8)
    low, high = first
    assert low < data.mean() < high


def test_the_interval_of_a_constant_is_that_constant_and_empty_input_is_nan():
    low, high = M.day_block_bootstrap(np.full((20, 5), 0.25))
    assert low == high == 0.25
    assert all(isnan(v) for v in M.day_block_bootstrap(np.empty((0, 5))))
    assert all(isnan(v) for v in M.day_block_bootstrap(np.full((20, 5), nan)))


def test_picks_of_one_day_are_not_treated_as_independent():
    data = clustered_days()  # 150 picks, but really only 30 independent days
    wilson = M.wilson_interval(data.sum(), data.size)
    low, high = M.day_block_bootstrap(data, block=1, n_boot=600)
    assert (high - low) > 2.0 * (wilson[1] - wilson[0])  # the pick-level Wilson interval is far too narrow


def test_consecutive_days_stay_together_so_clustered_regimes_widen_the_interval():
    trending = np.array([[0.0]] * 50 + [[1.0]] * 50)  # a calm spell followed by a hot spell
    iid = M.day_block_bootstrap(trending, block=1, n_boot=800)
    blocked = M.day_block_bootstrap(trending, n_boot=800)  # default block ~ days ** (1/3)
    assert (blocked[1] - blocked[0]) > 1.5 * (iid[1] - iid[0])
    assert M.day_block_bootstrap(trending, block=500, n_boot=100) == M.day_block_bootstrap(trending, block=100, n_boot=100)


def test_ragged_days_ignore_missing_picks_and_any_reduction_can_be_used():
    ragged = np.array([[1.0, nan, nan], [0.0, 0.0, nan], [1.0, 1.0, 1.0], [0.0, nan, nan], [1.0, 1.0, nan], [0.0, 0.0, 0.0]])
    mean_low, mean_high = M.day_block_bootstrap(ragged, n_boot=300)
    assert 0.0 <= mean_low <= np.nanmean(ragged) <= mean_high <= 1.0
    values = np.array([[1.0, 2.0, 9.0]] * 6)
    low, high = M.day_block_bootstrap(values, np.nanmedian, n_boot=100)
    assert low == high == 2.0
    narrow = M.day_block_bootstrap(np.random.default_rng(3).normal(size=(60, 3)), n_boot=400, level=0.5)
    wide = M.day_block_bootstrap(np.random.default_rng(3).normal(size=(60, 3)), n_boot=400, level=0.95)
    assert narrow[0] > wide[0] and narrow[1] < wide[1]


def test_a_single_column_of_per_day_values_is_accepted():
    per_day = np.random.default_rng(5).normal(0.3, 1.0, size=50)
    low, high = M.day_block_bootstrap(per_day, n_boot=300)
    assert low < per_day.mean() < high


# ------------------------------------------------------------------------------- the pooled top-N hit rate
def daily_frame(hits: list[int], per_day: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"date": [f"2026-01-{d + 1:02d}" for d in range(len(hits))], "mean": [h / per_day for h in hits],
                         "sum": [float(h) for h in hits], "n": [per_day] * len(hits)})


def test_topn_hit_rate_interval_uses_day_blocks_not_the_pick_level_wilson_interval():
    daily = daily_frame(([5] * 5 + [0] * 5) * 3)  # hot and cold spells: every pick of a day shares its day's fate
    rate, low, high, n = M.topn_hit_rate(daily)
    wilson = M.wilson_interval(75, 150)
    assert (rate, n) == (0.5, 150) and low < rate < high
    assert (high - low) > 2.0 * (wilson[1] - wilson[0])


def test_topn_hit_rate_with_too_few_days_gives_the_rate_without_an_interval():
    rate, low, high, n = M.topn_hit_rate(daily_frame([3, 2]))
    assert rate == 0.5 and n == 10 and isnan(low) and isnan(high)
    assert all(isnan(v) for v in M.topn_hit_rate(pd.DataFrame())[:3])


def test_topn_hit_rate_handles_uneven_pick_counts_per_day():
    daily = pd.DataFrame({"date": list("abcdefgh"), "mean": [1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0],
                          "sum": [3.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0], "n": [3, 2, 3, 2, 2, 3, 1, 4]})
    rate, low, high, n = M.topn_hit_rate(daily)
    assert n == 20 and rate == pytest.approx(8 / 20) and 0.0 <= low <= rate <= high <= 1.0


# ---------------------------------------------------------------------------------- the paired bootstrap
def test_paired_bootstrap_default_still_resamples_days_independently():
    frame = pd.DataFrame({"date": range(40), "mean_m": np.random.default_rng(2).random(40), "mean_b": np.random.default_rng(9).random(40)})
    model = frame[["date", "mean_m"]].rename(columns={"mean_m": "mean"})
    base = frame[["date", "mean_b"]].rename(columns={"mean_b": "mean"})
    got = M.paired_day_bootstrap(model, base, n_boot=300, seed=7)
    diff = (frame["mean_m"] - frame["mean_b"]).to_numpy()
    idx = np.random.default_rng(7).integers(0, len(diff), size=(300, len(diff)))  # the historical algorithm
    means = diff[idx].mean(axis=1)
    assert got["ci_low"] == float(np.percentile(means, 2.5)) and got["p10"] == float(np.percentile(means, 10))
    assert got["mean_diff"] == float(diff.mean()) and got["n_days"] == 40


def test_paired_bootstrap_can_resample_blocks_of_consecutive_days():
    dates = list(range(100))
    model = pd.DataFrame({"date": dates, "mean": [1.0] * 50 + [0.0] * 50})
    base = pd.DataFrame({"date": dates, "mean": [0.5] * 100})
    iid = M.paired_day_bootstrap(model, base, n_boot=500)
    blocked = M.paired_day_bootstrap(model, base, n_boot=500, block=5)
    assert (blocked["ci_high"] - blocked["ci_low"]) > 1.5 * (iid["ci_high"] - iid["ci_low"])
    assert blocked["mean_diff"] == iid["mean_diff"] == 0.0


# -------------------------------------------------------------------------------------------- rank IC
def ranking_days(days: int = 12, names: int = 10, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"date": f"d{d:02d}", "score": float(n), "outcome": n + rng.normal(0, 3.0) * (d % 3)} for d in range(days)
            for n in range(names)]
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------- rank IC
def test_rank_correlation_of_one_day_matches_the_textbook_value_and_degrades_to_nan():
    scores = np.arange(10, dtype=float)
    assert M.rank_correlation(scores, scores * 2 + 1) == pytest.approx(1.0)
    assert M.rank_correlation(scores, -scores) == pytest.approx(-1.0)
    assert M.rank_correlation(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0])) \
        == pytest.approx(1 - 6 * 8 / (8 * 63))  # 1 - 6 * sum(d^2) / (n (n^2 - 1)) with every rank off by one
    tied = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    assert M.rank_correlation(tied, np.arange(8.0)) == pytest.approx(pd.Series(tied).corr(pd.Series(np.arange(8.0)), method="spearman"))
    assert isnan(M.rank_correlation(scores[:7], scores[:7]))  # fewer than eight names
    assert isnan(M.rank_correlation(np.full(10, 0.4), scores)) and isnan(M.rank_correlation(scores, np.full(10, 2.0)))
    with_gap = scores.copy()
    with_gap[3] = nan
    assert M.rank_correlation(with_gap, scores) == pytest.approx(1.0)  # a missing pair is dropped, the rest still correlates
    assert isnan(M.rank_correlation(scores, scores, min_names=11))


def test_daily_rank_ic_gives_one_correlation_per_usable_day_and_matches_the_aggregate():
    frame = ranking_days()
    per_day = M.daily_rank_ic(frame, "score", "outcome")
    assert list(per_day.index) == sorted(frame["date"].unique()) and per_day.between(-1, 1).all()
    assert per_day["d00"] == pytest.approx(1.0)  # an outcome that is a perfect monotone function of the score
    aggregate = M.spearman_ic(frame, "score", "outcome")
    assert aggregate["n_days"] == len(per_day) and aggregate["ic_mean"] == pytest.approx(per_day.mean())
    tiny = frame[frame["score"] < 4]
    assert M.daily_rank_ic(tiny, "score", "outcome").empty  # fewer than 8 names a day: no usable correlation
    flat = frame.assign(outcome=1.0)
    assert M.daily_rank_ic(flat, "score", "outcome").empty


# ------------------------------------------------------------------------- the random-ranking reference
def test_ndcg_of_a_random_ranking_is_its_exact_expectation():
    relevance = np.array([3.0, 2.0, 1.0, 0.0])
    for k in (1, 2, 4, 99):
        brute = np.mean([M.ndcg_at_k(np.array(order, dtype=float), relevance, k) for order in itertools.permutations(range(4))])
        assert M.ndcg_random_expectation(relevance, k) == pytest.approx(brute)
    assert M.ndcg_random_expectation(np.array([5.0] * 6), 3) == pytest.approx(1.0)  # no ranking can be better than another
    assert isnan(M.ndcg_random_expectation(np.zeros(4), 2)) and isnan(M.ndcg_random_expectation(np.array([]), 3))
    assert M.ndcg_random_expectation(np.array([3.0, nan, 1.0]), 2) == M.ndcg_random_expectation(np.array([3.0, 1.0]), 2)
