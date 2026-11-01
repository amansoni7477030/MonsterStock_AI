"""Stale global series: a feed whose latest final bar is older than its tolerance must stop feeding the model (features blank,
global coverage lower) instead of its last move being forward-filled for months. Weekends and ordinary exchange holidays must not
trigger that."""

from __future__ import annotations

import logging
import zlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.clock import at_ist
from features import global_features as gf
from ingestion.series import SERIES_BY_KEY

DATES = pd.bdate_range("2026-01-05", "2026-06-30")
MEMBERS = sorted({m for members, _ in gf.COMPOSITES.values() for m in members})
HHMM = "08:45"


def walk(key: str, index: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(zlib.crc32(key.encode()) % 10_000)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index)))), index=index)


def all_closes(dead: dict[str, str] | None = None, missing: tuple[str, ...] = ()) -> dict[str, pd.Series]:
    """Daily closes for every series the composites use; ``dead`` maps a key to the last date it still has a bar."""
    index = pd.bdate_range("2025-10-01", "2026-07-01")
    out = {}
    for key in MEMBERS:
        if key in missing:
            continue
        series = walk(key, index)
        if key in (dead or {}):
            series = series[series.index <= pd.Timestamp(dead[key])]
        out[key] = series
    return out


def features(closes: dict[str, pd.Series]) -> dict[str, np.ndarray]:
    return gf.global_features(closes, DATES, HHMM)


def at(values: np.ndarray, day: str) -> float:
    return float(values[DATES.get_loc(pd.Timestamp(day))])


# ------------------------------------------------------------------------------------- one series
def aligned(key: str, close: pd.Series, ages: list[float], unit: str = "pct"):
    """The series as seen at cutoffs ``ages`` days after its latest bar became available."""
    available = SERIES_BY_KEY[key].available_at(close.index[-1].date()).timestamp()
    return gf.aligned_series(SERIES_BY_KEY[key], close, np.array([available + a * 86400.0 for a in ages]), unit)


def test_a_series_is_blank_once_its_latest_bar_is_older_than_the_limit():
    close = walk("^GSPC", pd.bdate_range("2026-01-02", "2026-03-31"))
    ages = [0.3, 2.0, 3.9, 4.0, 4.01, 10.0, 60.0]
    out = aligned("^GSPC", close, ages)
    assert gf.STALE_AFTER_DAYS["us_equity"] == 4.0
    for name in ("ret1", "ret5", "level"):
        assert np.isfinite(out[name]).tolist() == [a <= 4.0 for a in ages], name  # forward-filled forever before
    np.testing.assert_allclose(out["age_days"], ages, rtol=1e-9)  # the age itself is still reported
    assert out["ret1"][0] == pytest.approx((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0)


def test_the_basis_point_unit_is_blanked_like_the_percent_unit():
    out = aligned("^TNX", walk("^TNX", pd.bdate_range("2026-01-02", "2026-03-31")), [1.0, 9.0], unit="bp")
    assert np.isfinite(out["ret1"][0]) and np.isnan(out["ret1"][1]) and np.isnan(out["level"][1])


@pytest.mark.parametrize("key, last_bar, cutoff_day", [
    ("^GSPC", "2026-05-22", "2026-05-26"),      # US Memorial Day: Friday's bar is 3.3 days old on Tuesday
    ("^VIX", "2026-05-22", "2026-05-26"),
    ("^TNX", "2026-05-22", "2026-05-26"),
    ("CL=F", "2026-05-22", "2026-05-26"),
    ("USDINR=X", "2026-05-22", "2026-05-25"),   # an ordinary weekend
    ("^FTSE", "2026-04-02", "2026-04-07"),      # Easter: closed Good Friday and Easter Monday (4.5 days)
    ("^GDAXI", "2025-12-23", "2025-12-29"),     # Xetra closes 24-26 December (5.4 days)
    ("^N225", "2025-12-30", "2026-01-05"),      # Tokyo's New Year closure (5.8 days)
    ("^AXJO", "2026-04-03", "2026-04-07"),      # a Monday holiday in Sydney
])
def test_weekends_and_ordinary_holiday_gaps_are_not_stale(key, last_bar, cutoff_day):
    close = walk(key, pd.bdate_range(end=last_bar, periods=40))
    cutoff = at_ist(date.fromisoformat(cutoff_day), HHMM).timestamp()
    out = gf.aligned_series(SERIES_BY_KEY[key], close, np.array([cutoff]), "bp" if key == "^TNX" else "pct")
    assert np.isfinite(out["ret1"]).all() and np.isfinite(out["level"]).all()
    assert 2.0 < out["age_days"][0] <= 6.0


def test_the_limit_depends_on_the_kind_of_market():
    assert gf.STALE_AFTER_DAYS["eu_equity"] > gf.STALE_AFTER_DAYS["us_equity"] == gf.DEFAULT_STALE_AFTER_DAYS
    assert {SERIES_BY_KEY[m].category for m in MEMBERS} <= set(gf.STALE_AFTER_DAYS)  # every category used has its own entry
    span = pd.bdate_range("2026-01-02", "2026-03-31")
    assert np.isfinite(aligned("^FTSE", walk("^FTSE", span), [5.5])["ret1"]).all()
    assert np.isnan(aligned("^GSPC", walk("^GSPC", span), [5.5])["ret1"]).all()
    assert np.isnan(aligned("^FTSE", walk("^FTSE", span), [8.0])["ret1"]).all()


# ------------------------------------------------------------------------------------ composites + cover
def test_all_series_fresh_means_full_coverage_and_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="features.global_features"):
        out = features(all_closes())
    assert (out["g_cover"] == 1.0).all() and not caplog.records
    assert np.isfinite(out["g_us_ret"]).all() and np.isfinite(out["g_vix_lvl"]).all()


def test_a_dead_single_series_composite_goes_blank_and_coverage_drops_by_its_share():
    out = features(all_closes(dead={"HG=F": "2026-02-20"}))
    assert np.isfinite(at(out["g_copper_ret"], "2026-02-23")) and at(out["g_cover"], "2026-02-23") == 1.0  # still recent
    for day in ("2026-03-05", "2026-04-15", "2026-06-30"):
        assert np.isnan(at(out["g_copper_ret"], day))  # it used to be forward-filled with the last move
        assert at(out["g_cover"], day) == pytest.approx(13 / 14)
    healthy = features(all_closes())
    for name in ("g_us_ret", "g_oil_ret", "g_eu_ret", "g_vix_chg"):
        np.testing.assert_array_equal(out[name], healthy[name])  # nothing else is touched


def test_dead_vix_and_yield_feeds_blank_their_levels_too():
    out = features(all_closes(dead={"^VIX": "2026-03-02", "^TNX": "2026-03-02"}))
    for name in ("g_vix_chg", "g_vix_lvl", "g_us10y_bp", "g_us10y_lvl", "g_us10y_bp5"):
        assert np.isfinite(at(out[name], "2026-03-03")) and np.isnan(at(out[name], "2026-05-20")), name


def test_one_dead_member_does_not_contaminate_its_composite_but_is_counted_in_coverage():
    dead = features(all_closes(dead={"^GSPC": "2026-02-20"}))
    nasdaq_only = features(all_closes(missing=("^GSPC",)))
    late = DATES >= pd.Timestamp("2026-03-10")
    np.testing.assert_allclose(dead["g_us_ret"][late], nasdaq_only["g_us_ret"][late])  # the stale S&P move is not averaged in
    assert at(dead["g_cover"], "2026-06-30") == pytest.approx(1.0 - 0.5 / 14)


def test_a_series_that_was_never_loaded_lowers_coverage_as_well():
    out = features(all_closes(missing=("^FCHI",)))
    assert np.isfinite(out["g_eu_ret"]).all()
    np.testing.assert_allclose(out["g_cover"], 1.0 - 0.25 / 14)


def test_a_composite_without_any_series_counts_as_uncovered():
    out = features(all_closes(missing=("HG=F", "NG=F")))
    assert np.isnan(out["g_copper_ret"]).all() and np.isnan(out["g_natgas_ret"]).all()
    np.testing.assert_allclose(out["g_cover"], 12 / 14)


def test_the_sensitivities_built_on_a_dead_factor_stop_updating():
    glob = features(all_closes(dead={"HG=F": "2026-05-01"}))
    ret1 = pd.DataFrame(np.random.default_rng(2).normal(0, 1, (len(DATES), 3)), index=DATES, columns=list("ABC"))
    expected = gf.sensitivity_features(ret1, glob)["g_exp_copper"]
    assert expected.loc["2026-05-04"].notna().all() and expected.loc["2026-06-15":].isna().all().all()


def test_a_stale_feed_is_reported_once_at_warning_level(caplog):
    with caplog.at_level(logging.WARNING, logger="features.global_features"):
        features(all_closes(dead={"CL=F": "2026-02-02"}))
    stale = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(stale) == 1 and "CL=F" in stale[0] and "stale" in stale[0]


def test_a_dead_series_no_composite_uses_is_not_reported(caplog):
    closes = all_closes()
    closes["^FVX"] = walk("^FVX", pd.bdate_range("2025-10-01", "2026-01-02"))  # dead, but no feature is built from it
    with caplog.at_level(logging.WARNING, logger="features.global_features"):
        features(closes)
    assert not caplog.records
