"""Point-in-time guarantees of the feature builder: the future, the current session's path and not-yet-final global bars
must never influence features, and live (single-row) computation must equal batch (training) computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.builder import build_feature_set
from tests.synthetic import copy_panel, make_panel

THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
T0 = 260  # a date with enough history for every rolling window and for similar-day matching


@pytest.fixture(scope="module")
def panel():
    return make_panel(n_dates=330, n_symbols=20, seed=11)


def build(p, stage, **kw):
    return build_feature_set(p, stage=stage, thresholds=THRESHOLDS, unusual_k=1.5, cutoff_hhmm="08:45", **kw)


def assert_same(a: pd.DataFrame, b: pd.DataFrame, columns=None):
    cols = list(columns) if columns is not None else list(a.columns)
    np.testing.assert_allclose(a[cols].to_numpy(dtype=float), b[cols].to_numpy(dtype=float), rtol=1e-5, atol=1e-5, equal_nan=True)


def garbage(frame: pd.DataFrame | pd.Series, start: int, rng: np.random.Generator):
    target = frame.iloc[start:]
    frame.iloc[start:] = rng.uniform(5, 5000, size=target.shape)


@pytest.mark.parametrize("stage", ["pre", "final"])
def test_future_data_cannot_change_past_features(panel, stage):
    base = build(panel, stage)
    q = copy_panel(panel)
    rng = np.random.default_rng(3)
    cutoff_date = q.dates[T0]
    for frame in (q.o, q.h, q.low, q.c, q.v, q.nifty):
        garbage(frame, T0 + 1, rng)
    garbage(q.vix, T0 + 1, rng)
    # global bars dated on/after the cutoff session are not final at 08:45 IST of that date -> must be invisible
    for key, series in q.closes.items():
        mask = series.index >= cutoff_date
        series.loc[mask] = rng.uniform(1, 9999, mask.sum())
    q.fno.loc[q.fno["trade_date"] >= cutoff_date, ["fut_oi", "fut_oi_chg", "iv_atm", "pcr_oi", "dte"]] = 12345.0
    perturbed = build(q, stage)
    upto = base.X.index.get_level_values("date") <= cutoff_date
    assert_same(base.X[upto], perturbed.X[upto])
    # sanity: the perturbation really changed the later rows (the test would be vacuous otherwise)
    later = ~upto
    assert not np.allclose(base.X[later].to_numpy(dtype=float), perturbed.X[later].to_numpy(dtype=float), equal_nan=True)


def test_current_session_path_does_not_leak_into_pre_features(panel):
    base = build(panel, "pre")
    q = copy_panel(panel)
    rng = np.random.default_rng(4)
    for frame in (q.o, q.h, q.low, q.c, q.v):
        frame.iloc[T0] = rng.uniform(5, 5000, size=frame.shape[1])
    q.nifty.iloc[T0] = rng.uniform(5000, 30000, size=4)
    q.vix.iloc[T0] = 99.0
    perturbed = build(q, "pre")
    day = base.X.index.get_level_values("date") == q.dates[T0]
    assert_same(base.X[day], perturbed.X[day])


def test_final_stage_uses_only_the_open_of_the_current_session(panel):
    base = build(panel, "final")
    day = base.X.index.get_level_values("date") == panel.dates[T0]

    q = copy_panel(panel)
    rng = np.random.default_rng(5)
    for frame in (q.h, q.low, q.c, q.v):
        frame.iloc[T0] = rng.uniform(5, 5000, size=frame.shape[1])
    q.nifty.iloc[T0] = rng.uniform(5000, 30000, size=4)
    perturbed = build(q, "final")
    assert_same(base.X[day], perturbed.X[day])  # high/low/close/volume of today are invisible

    q2 = copy_panel(panel)
    q2.o.iloc[T0] = q2.o.iloc[T0] * 1.05  # a 5% different opening price changes ONLY the pre-open group (+ similar-day via market gap)
    changed = build(q2, "final")
    p_cols = [c for c in base.X.columns if c.startswith("p_")]
    assert not np.allclose(base.X.loc[day, p_cols].to_numpy(dtype=float), changed.X.loc[day, p_cols].to_numpy(dtype=float))
    stable = [c for c in base.X.columns if not c.startswith(("p_", "x_"))]
    assert_same(base.X[day], changed.X[day], stable)


@pytest.mark.parametrize("stage", ["pre", "final"])
def test_live_row_equals_batch_row_even_with_unknown_session_data(panel, stage):
    """Train/serve parity: today's high/low/close/volume are unknown live (NaN) and only the final-stage open exists."""
    batch = build(panel, stage)
    live_panel = copy_panel(panel)
    for frame in (live_panel.h, live_panel.low, live_panel.c, live_panel.v):
        frame.iloc[-1] = np.nan
    if stage == "pre":
        live_panel.o.iloc[-1] = np.nan
    live_panel.nifty.iloc[-1] = np.nan
    live_panel.vix.iloc[-1] = np.nan
    last = slice(len(panel.dates) - 1, len(panel.dates))
    live = build(live_panel, stage, with_labels=False, row_slice=last)
    last_date = panel.dates[-1]
    batch_last = batch.X[batch.X.index.get_level_values("date") == last_date]
    assert len(live.X) == len(panel.symbols) and (live.X.index.get_level_values("date") == last_date).all()
    assert_same(batch_last, live.X)


def test_global_features_follow_availability_not_calendar_date(panel):
    """The S&P bar dated D closes after the Indian cutoff of D; the Indian feature for D must carry the move of D-1."""
    q = copy_panel(panel)
    d = q.dates[T0]
    series = q.closes["^GSPC"]
    prev_idx = series.index[series.index < d][-2:]
    series.loc[prev_idx[0]] = 4000.0
    series.loc[prev_idx[1]] = 4100.0  # +2.5% on D-1
    series.loc[series.index[series.index == d]] = 9999.0  # D itself: must not be visible
    fs = build(q, "pre")
    value = fs.X.xs(d, level="date")["g_us_ret"].iloc[0]
    expected = (4100.0 / 4000.0 - 1.0) * 100.0
    ixic = q.closes["^IXIC"]
    ix_prev = ixic.index[ixic.index < d][-2:]
    ix_move = (ixic.loc[ix_prev[1]] / ixic.loc[ix_prev[0]] - 1.0) * 100.0
    assert value == pytest.approx((expected + ix_move) / 2.0, rel=1e-3)


def test_results_features_are_unknown_before_coverage_not_zero(panel):
    q = copy_panel(panel)
    q.meetings_coverage_start = q.dates[100].date()
    fs = build(q, "pre")
    early = fs.X.xs(q.dates[50], level="date")["c_results_today"]
    late = fs.X.xs(q.dates[150], level="date")["c_results_today"]
    assert early.isna().all() and late.notna().all()


def test_a_meeting_further_ahead_than_the_results_horizon_is_invisible(panel):
    """Live only knows meetings NSE has announced (a week or two ahead); training must not know further than that either."""
    meeting = T0 + 30
    q = copy_panel(panel)
    extra = {"symbol": "S00", "meeting_date": q.dates[meeting].date().isoformat(), "purpose": "Results", "is_results": 1}
    q.meetings = pd.concat([q.meetings, pd.DataFrame([extra])], ignore_index=True)
    base, perturbed = build(panel, "pre"), build(q, "pre")
    calendar_cols = [c for c in base.X.columns if c.startswith("c_")]
    dates = base.X.index.get_level_values("date")
    unaware = dates <= q.dates[meeting - 11]  # 11+ sessions before the meeting
    assert_same(base.X[unaware], perturbed.X[unaware], calendar_cols)
    edge = (q.dates[meeting - 10], "S00")
    assert perturbed.X.loc[edge, "c_days_to_results"] == 10.0 and np.isnan(base.X.loc[edge, "c_days_to_results"])


def test_relative_volume_compares_with_the_sessions_before_it(panel):
    """The feature of date t describes session t-1 against the 20 sessions before THAT one (never including it)."""
    fs = build(panel, "pre")
    sym = panel.symbols[3]
    v = panel.v[sym]
    baseline = v.iloc[T0 - 21:T0 - 1].mean()
    row = fs.X.loc[(panel.dates[T0], sym)]
    assert row["t_rvol1"] == pytest.approx(v.iloc[T0 - 1] / baseline, rel=1e-4)
    assert row["t_rvol5"] == pytest.approx(v.iloc[T0 - 5:T0].mean() / baseline, rel=1e-4)
    q = copy_panel(panel)
    q.v.iloc[T0 - 1, 3] = 10.0 * baseline  # a ten-fold spike in the latest finished session
    assert build(q, "pre").X.loc[(q.dates[T0], sym), "t_rvol1"] == pytest.approx(10.0, rel=1e-4)  # it read 6.9 before


def test_a_stocks_own_return_stays_out_of_its_sector_dispersion(panel):
    base = build(panel, "pre")
    q = copy_panel(panel)
    q.c.iloc[T0 - 1, 0] *= 1.08  # S00 jumps 8% in the latest finished session
    moved = build(q, "pre")
    day = panel.dates[T0]
    assert panel.sectors["S00"] == panel.sectors["S04"]
    assert moved.X.loc[(day, "S00"), "s_disp1"] == base.X.loc[(day, "S00"), "s_disp1"]
    assert moved.X.loc[(day, "S04"), "s_disp1"] > base.X.loc[(day, "S04"), "s_disp1"]  # its peers do see the outlier


def test_labels_describe_the_session_and_baseline_is_prior_only(panel):
    fs = build(panel, "pre")
    t = panel.dates[T0]
    sym = panel.symbols[0]
    o, h, low, c = (getattr(panel, k).loc[t, sym] for k in ("o", "h", "low", "c"))
    row = fs.labels.loc[(t, sym)]
    assert row["y_range"] == pytest.approx((h - low) / o * 100, rel=1e-4)
    assert row["y_exc"] == pytest.approx(max(h - o, o - low) / o * 100, rel=1e-4)
    assert row["y_up"] == float(c > o) and row["y_hit_3"] == float((h - low) / o * 100 > 3.0)
    base = fs.X.loc[(t, sym), "t_range_mean20"]
    assert row["y_unusual"] == float((h - low) / o * 100 > 1.5 * base)
    assert row["y_range_z"] == pytest.approx((h - low) / o * 100 / base, rel=1e-3)
