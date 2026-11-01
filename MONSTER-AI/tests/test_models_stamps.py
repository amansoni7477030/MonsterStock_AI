"""Training bookkeeping that must not depend on the machine: the version stamp (UTC, from the application clock), the climatology
the Brier skill is measured against (training days only) and the heuristic baseline's "unusual" multiple."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from app.clock import IST, Clock, FakeClock, set_clock
from features.builder import build_feature_set
from models.inference import DEFAULT_UNUSUAL_K, BaselineBundle
from models.training import TrainConfig, prior_base_rate, train_stage, version_stamp
from tests.seed_data import SMALL_TRAIN, panel_ending

THRESHOLDS = (1.0, 2.0, 3.0, 5.0)


# --------------------------------------------------------------------------------------- version stamp
def test_the_stamp_comes_from_the_application_clock_in_utc(fake_clock):
    assert version_stamp("pre") == ("2026-11-02T03:00:00Z", "m-20261102-030000-pre")
    fake_clock.advance(3661)
    assert version_stamp("final") == ("2026-11-02T04:01:01Z", "m-20261102-040101-final")  # it moves with the clock, not the OS


def test_the_stamp_does_not_depend_on_the_time_zone_of_the_instant_given():
    utc = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)
    ist = datetime(2026, 11, 2, 8, 30, tzinfo=IST)  # the same instant, written in Indian time
    far = datetime(2026, 11, 1, 19, 0, tzinfo=timezone(timedelta(hours=-8)))
    assert version_stamp("pre", utc) == version_stamp("pre", ist) == version_stamp("pre", far)
    assert version_stamp("pre", ist)[1] == "m-20261102-030000-pre"


def test_a_naive_time_is_refused_rather_than_read_as_local_time():
    with pytest.raises(ValueError, match="naive"):
        version_stamp("pre", datetime(2026, 11, 2, 8, 30))


def test_a_clock_that_reports_another_zone_still_stamps_utc():
    class WestCoastClock(Clock):
        def now(self) -> datetime:
            return datetime(2026, 12, 31, 23, 0, tzinfo=timezone(timedelta(hours=-5)))

    set_clock(WestCoastClock())
    assert version_stamp("pre") == ("2027-01-01T04:00:00Z", "m-20270101-040000-pre")


# ----------------------------------------------------------------------------- the Brier-skill reference
def test_the_climatology_is_the_base_rate_of_the_training_days_only():
    day_ids = np.repeat(np.arange(10), 5)
    label = np.where(day_ids < 6, 0.0, 1.0)  # a holdout (days 6-9) that is far more eventful than the training period
    assert label.mean() == pytest.approx(0.4)
    assert prior_base_rate(label, day_ids, 6) == 0.0
    label[:5] = 1.0
    assert prior_base_rate(label, day_ids, 6) == pytest.approx(5 / 30)


def test_unknown_outcomes_do_not_count_and_the_first_holdout_day_is_excluded():
    day_ids = np.repeat(np.arange(4), 2)
    label = np.array([1.0, np.nan, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert prior_base_rate(label, day_ids, 3) == pytest.approx(3 / 5)
    assert prior_base_rate(label, day_ids, 2) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- the baseline's unusual multiple
def frame_with_range(value: float) -> pd.DataFrame:
    return pd.DataFrame({"t_range_mean20": [value], "t_atr14_pct": [value]}, index=["AAA"])


def expected_unusual(k: float) -> float:
    return float(1.0 - ndtr(np.log(k) / BaselineBundle.SIGMA))  # centre == normal: only the multiple matters


def test_the_default_multiple_is_the_one_it_always_used():
    assert DEFAULT_UNUSUAL_K == 1.5
    bundle = BaselineBundle("pre", THRESHOLDS, 3.0)
    assert bundle.predict(frame_with_range(2.0))["p_unusual"].iloc[0] == pytest.approx(expected_unusual(1.5))
    assert bundle.meta["unusual_k"] == 1.5


@pytest.mark.parametrize("k", [1.2, 2.0, 3.0])
def test_the_configured_multiple_drives_the_unusual_probability(k):
    bundle = BaselineBundle("pre", THRESHOLDS, 3.0, unusual_k=k)
    out = bundle.predict(frame_with_range(2.0))
    assert bundle.meta["unusual_k"] == k and out["p_unusual"].iloc[0] == pytest.approx(expected_unusual(k))


def test_a_higher_multiple_means_a_lower_unusual_probability_and_nothing_else_changes():
    low, high = (BaselineBundle("final", THRESHOLDS, 3.0, unusual_k=k).predict(frame_with_range(2.0)) for k in (1.5, 2.5))
    assert high["p_unusual"].iloc[0] < low["p_unusual"].iloc[0]
    other = [c for c in low.columns if c != "p_unusual"]
    assert low[other].equals(high[other])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_multiple_falls_back_to_the_default_with_a_warning(bad, caplog):
    with caplog.at_level(logging.WARNING, logger="models.inference"):
        bundle = BaselineBundle("pre", THRESHOLDS, 3.0, unusual_k=bad)
    assert bundle.unusual_k == DEFAULT_UNUSUAL_K and "unusual_k" in caplog.text


# -------------------------------------------------------------------- the real training run uses all of it
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    panel = panel_ending(n_dates=470, n_symbols=40, seed=21)
    fs = build_feature_set(panel, stage="pre", thresholds=THRESHOLDS, unusual_k=1.5, cutoff_hhmm="08:45")
    x, y = fs.X.copy(), fs.labels.copy()
    set_clock(FakeClock(datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)))
    cfg = TrainConfig(stage="pre", thresholds=THRESHOLDS, primary_threshold=3.0, unusual_k=1.5,
                      **{**SMALL_TRAIN, "gate_max_ece": 0.5, "gate_min_brier_skill": -1.0})
    result = train_stage(fs, cfg, tmp_path_factory.mktemp("models"))  # no ``now``: the application clock
    return result, x, y, cfg


@pytest.mark.slow
def test_a_training_run_is_stamped_from_the_clock_in_utc(trained):
    result = trained[0]
    assert result.version == "m-20261102-030000-pre" and result.meta["created_at"] == "2026-11-02T03:00:00Z"
    assert result.directory.name == result.version


@pytest.mark.slow
def test_a_training_run_measures_skill_against_the_training_period_base_rate(trained):
    result, x, y, cfg = trained
    keep = ((x["u_hist_days"] >= cfg.min_hist_days) & y["y_range"].notna() & y["y_unusual"].notna()
            & x["t_atr14_pct"].notna() & x["t_range_mean20"].notna()).to_numpy()
    dates = x.index.get_level_values("date")[keep]
    days = np.sort(dates.unique())
    assert len(days) == result.meta["n_days"]
    first_holdout = days[result.meta["n_days"] - result.meta["holdout_days"]]
    for head in ("hit_3", "unusual"):
        label = y["y_unusual" if head == "unusual" else f"y_{head}"].to_numpy(dtype=float)[keep]
        expected = float(np.nanmean(label[dates < first_holdout]))
        assert result.meta["base_rates"][head] == pytest.approx(expected, rel=1e-9)
        assert abs(expected - float(np.nanmean(label))) > 1e-4  # the holdout really would have moved the old reference
