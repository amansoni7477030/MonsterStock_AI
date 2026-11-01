"""Relative volume (blueprint Appendix A): current volume over COMPARABLE HISTORICAL volume. The baseline is the mean of
the sessions BEFORE the one measured, so a spike never dilutes its own yardstick."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.technical import stock_features

N = 60
BASE = 1000.0


def volume_features(volume: np.ndarray) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2025-01-01", periods=len(volume))
    price = pd.DataFrame(100.0, index=index, columns=["AAA"])
    v = pd.DataFrame(volume, index=index, columns=["AAA"])
    flat = pd.Series(0.0, index=index)
    return stock_features(price, price, price, price, v, flat, flat + 1.0)


def spiked(size: float, row: int = 30) -> np.ndarray:
    volume = np.full(N, BASE)
    volume[row] = size
    return volume


def test_a_ten_times_spike_measures_ten_times_the_prior_average():
    out = volume_features(spiked(10 * BASE))
    assert out["t_rvol1"]["AAA"].iloc[30] == pytest.approx(10.0)  # it read 6.9x while the baseline contained the spike
    assert out["t_rvol1"]["AAA"].iloc[29] == pytest.approx(1.0)


def test_the_five_session_variant_and_the_acceleration_share_that_baseline():
    out = volume_features(spiked(10 * BASE))
    assert out["t_rvol5"]["AAA"].iloc[30] == pytest.approx((10 * BASE + 4 * BASE) / 5 / BASE)
    assert out["t_vol_acc"]["AAA"].iloc[30] == pytest.approx(10.0 - 2.8)
    assert out["t_vol_acc"]["AAA"].iloc[29] == pytest.approx(0.0)


def test_the_baseline_is_independent_of_the_session_it_measures():
    small, large = volume_features(spiked(5 * BASE)), volume_features(spiked(20 * BASE))
    ratio = large["t_rvol1"]["AAA"].iloc[30] / small["t_rvol1"]["AAA"].iloc[30]
    assert ratio == pytest.approx(4.0)  # exactly proportional to the spike: the denominator did not move


def test_the_spike_then_enters_the_baseline_of_the_following_sessions():
    out = volume_features(spiked(10 * BASE))
    assert out["t_rvol1"]["AAA"].iloc[31] == pytest.approx(BASE / ((19 * BASE + 10 * BASE) / 20))
    assert out["t_rvol1"]["AAA"].iloc[51] == pytest.approx(1.0)  # 21 sessions later the spike has left the window


def test_the_baseline_needs_ten_prior_sessions():
    rvol = volume_features(np.full(N, BASE))["t_rvol1"]["AAA"]
    assert rvol.iloc[:10].isna().all() and rvol.iloc[10] == pytest.approx(1.0)


def test_a_missing_volume_stays_missing_and_does_not_poison_the_baseline():
    volume = np.full(N, BASE)
    volume[25] = np.nan
    rvol = volume_features(volume)["t_rvol1"]["AAA"]
    assert np.isnan(rvol.iloc[25]) and rvol.iloc[26] == pytest.approx(1.0)
