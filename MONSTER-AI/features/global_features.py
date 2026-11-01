"""Global overnight features, aligned point-in-time (blueprint sections 4, 5, 9, 11).

For every Indian trading date the latest *final* bar of each global series at the scan cutoff is selected with its
own close time and daylight-saving rules (ingestion.series). Composite factors are then correlated with each stock's
returns using trailing windows only; same-day AND one-day-lagged sensitivities are both measured (lead/lag), and the
resulting "expected move from factor" features embody the global-to-India transmission without hard-coded rules.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from app.clock import at_ist
from ingestion.series import SERIES_BY_KEY, SeriesSpec

log = logging.getLogger(__name__)

# composite -> (member series, unit)  unit: 'pct' = percent return, 'bp' = basis-point change of a yield in percent
COMPOSITES: dict[str, tuple[tuple[str, ...], str]] = {
    "us": (("^GSPC", "^IXIC"), "pct"),
    "semis": (("^SOX",), "pct"),
    "eu": (("^STOXX50E", "^GDAXI", "^FTSE", "^FCHI"), "pct"),
    "asia": (("^N225", "^HSI", "000001.SS", "^KS11", "^TWII", "^AXJO"), "pct"),
    "oil": (("BZ=F", "CL=F"), "pct"),
    "gold": (("GC=F", "SI=F"), "pct"),
    "copper": (("HG=F",), "pct"),
    "natgas": (("NG=F",), "pct"),
    "usdinr": (("USDINR=X",), "pct"),
    "dxy": (("DX-Y.NYB",), "pct"),
    "us10y": (("^TNX",), "bp"),
    "vix": (("^VIX",), "pct"),
    "cny": (("USDCNY=X",), "pct"),
    "jpy": (("USDJPY=X",), "pct"),
}
FACTORS_WITH_SENSITIVITY = ("us", "semis", "eu", "oil", "usdinr", "us10y", "copper")
FIVE_DAY = ("us", "oil", "usdinr", "semis", "us10y")
COMPOSITE_MEMBERS = frozenset(m for members, _ in COMPOSITES.values() for m in members)

# Calendar days a series' latest final bar may be older than the scan cutoff before its features are blank (NaN): a feed that
# stopped must not have its last move carried forward for months. The age runs from the moment the bar became final, so
# ordinary gaps stay inside: a US three-day weekend is 3.3 days, a Monday holiday in Tokyo or Sydney 3.9; the European
# exchanges' Easter and Christmas closures reach 4.5-5.4 and Tokyo's New Year 5.8, hence a week for non-US equity indices.
STALE_AFTER_DAYS: dict[str, float] = {
    "us_equity": 4.0, "volatility": 4.0, "rates": 4.0, "commodity": 4.0, "fx": 4.0, "eu_equity": 7.0, "asia_equity": 7.0,
}
DEFAULT_STALE_AFTER_DAYS = 4.0


def stale_after_days(spec: SeriesSpec) -> float:
    return STALE_AFTER_DAYS.get(spec.category, DEFAULT_STALE_AFTER_DAYS)


def _cutoffs(dates: pd.DatetimeIndex, hhmm: str) -> np.ndarray:
    return np.array([at_ist(d.date(), hhmm).timestamp() for d in dates], dtype=float)


def aligned_series(
    spec: SeriesSpec, close: pd.Series, cutoffs: np.ndarray, unit: str
) -> dict[str, np.ndarray]:
    """Return (1-day change, 5-day change, level) of the latest bar final at each cutoff, plus that bar's age in days.

    The changes and the level are NaN where there is no such bar or it is stale (older than ``stale_after_days``); ``age_days``
    is always the true age."""
    close = close.dropna().sort_index()
    n = len(cutoffs)
    if close.empty:
        nan = np.full(n, np.nan)
        return {"ret1": nan, "ret5": nan, "level": nan, "age_days": nan}
    available = np.array([spec.available_at(d.date()).timestamp() for d in close.index], dtype=float)
    pos = np.searchsorted(available, cutoffs, side="right") - 1
    valid = pos >= 0
    safe = np.clip(pos, 0, None)
    if unit == "bp":
        change1, change5 = close.diff() * 100.0, close.diff(5) * 100.0
    else:
        change1, change5 = close.pct_change(fill_method=None) * 100.0, close.pct_change(5, fill_method=None) * 100.0
    level = close.to_numpy(dtype=float)
    age = (cutoffs - available[safe]) / 86400.0
    fresh = valid & (age <= stale_after_days(spec))
    return {
        "ret1": np.where(fresh, change1.to_numpy(dtype=float)[safe], np.nan),
        "ret5": np.where(fresh, change5.to_numpy(dtype=float)[safe], np.nan),
        "level": np.where(fresh, level[safe], np.nan),
        "age_days": np.where(valid, age, np.nan),
    }


def global_features(
    closes: Mapping[str, pd.Series], dates: pd.DatetimeIndex, cutoff_hhmm: str
) -> dict[str, np.ndarray]:
    """Composite factor moves available at ``cutoff_hhmm`` IST of each date (values are NOT shifted: they are, by
    construction, already known before the open of that date). A stale or missing series contributes nothing and lowers
    ``g_cover``, the share of the composites' series with a fresh 1-day change (each composite weighs the same)."""
    cutoffs = _cutoffs(dates, cutoff_hhmm)
    per_series: dict[str, dict[str, np.ndarray]] = {}
    for key, series in closes.items():
        spec = SERIES_BY_KEY.get(key)
        if spec is None:
            continue
        unit = "bp" if COMPOSITES["us10y"][0][0] == key else "pct"
        per_series[key] = aligned_series(spec, series, cutoffs, unit)
        age = per_series[key]["age_days"][-1] if len(cutoffs) else np.nan
        if key in COMPOSITE_MEMBERS and age > stale_after_days(spec):
            log.warning("global series %s is stale: its latest final bar is %.1f days old at the last cutoff (limit %.1f); "
                        "its features are blank", key, age, stale_after_days(spec))

    out: dict[str, np.ndarray] = {}
    n = len(dates)
    cover = np.zeros(n)
    for name, (members, _unit) in COMPOSITES.items():
        stacks1 = [per_series[m]["ret1"] for m in members if m in per_series]
        stacks5 = [per_series[m]["ret5"] for m in members if m in per_series]
        cover += np.mean([np.isfinite(per_series[m]["ret1"]) if m in per_series else np.zeros(n, dtype=bool) for m in members],
                         axis=0)
        if stacks1:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # "Mean of empty slice": a day with no data is NaN by design
                out_name = {"us10y": "g_us10y_bp", "vix": "g_vix_chg"}.get(name, f"g_{name}_ret")
                out[out_name] = np.nanmean(np.vstack(stacks1), axis=0) if len(stacks1) > 1 else stacks1[0]
                if name in FIVE_DAY:
                    five = {"us10y": "g_us10y_bp5"}.get(name, f"g_{name}_ret5")
                    out[five] = np.nanmean(np.vstack(stacks5), axis=0) if len(stacks5) > 1 else stacks5[0]
        else:
            out[{"us10y": "g_us10y_bp", "vix": "g_vix_chg"}.get(name, f"g_{name}_ret")] = np.full(n, np.nan)
    out["g_vix_lvl"] = per_series["^VIX"]["level"] if "^VIX" in per_series else np.full(n, np.nan)
    out["g_us10y_lvl"] = per_series["^TNX"]["level"] if "^TNX" in per_series else np.full(n, np.nan)
    out["g_cover"] = cover / len(COMPOSITES)
    return out


def sensitivity_features(
    ret1: pd.DataFrame, factors: Mapping[str, np.ndarray], *, window: int = 120, min_periods: int = 60
) -> dict[str, pd.DataFrame]:
    """Trailing same-day and lag-1 sensitivities of each stock to each factor -> expected moves for *today*.

    ``factors[name][t]`` is the factor move known before the open of date t. For stock return r_s on session s the
    pairs (r_s, f_s) with s <= t-1 estimate beta; the expected move today is beta_{t-1} * f_t. The lag-1 variant uses
    (r_s, f_{s-1}) and f_{t-1}. Outputs are already aligned to t: the builder must NOT shift them again.
    """
    out: dict[str, pd.DataFrame] = {}
    for name in FACTORS_WITH_SENSITIVITY:
        raw = factors.get(_factor_key(name))
        if raw is None or not np.isfinite(raw).any():
            continue
        f0 = pd.Series(raw, index=ret1.index)
        f1 = f0.shift(1)
        var0 = f0.rolling(window, min_periods=min_periods).var()
        var1 = f1.rolling(window, min_periods=min_periods).var()
        beta0 = ret1.rolling(window, min_periods=min_periods).cov(f0).div(var0.replace(0.0, np.nan), axis=0)
        beta1 = ret1.rolling(window, min_periods=min_periods).cov(f1).div(var1.replace(0.0, np.nan), axis=0)
        corr0 = ret1.rolling(window, min_periods=min_periods).corr(f0)
        out[f"g_exp_{name}"] = beta0.shift(1).mul(f0, axis=0)
        out[f"g_exp1_{name}"] = beta1.shift(1).mul(f1, axis=0)
        out[f"g_corr_{name}"] = corr0.shift(1)
    return out


def _factor_key(name: str) -> str:
    return {"us10y": "g_us10y_bp", "vix": "g_vix_chg"}.get(name, f"g_{name}_ret")
