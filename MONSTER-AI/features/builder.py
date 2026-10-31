"""Feature builder: ONE code path for training (all historical dates) and live inference (the current date).

Point-in-time rules enforced here:
  * every price/volume/F&O feature is computed "as of the previous session" and shifted by one trading day;
  * global overnight features use each series' own final-bar availability at the scan cutoff;
  * calendar/expiry/results features are known in advance and therefore unshifted;
  * the pre-open gap (stage ``final``) uses only the session's opening price, never its high/low/close;
  * the similar-day engine only looks at strictly earlier sessions.
tests/test_features_leakage.py verifies this by perturbing the future and the current session.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from features import calendar_features as cal
from features import fno_features as fnof
from features import global_features as gf
from features import regime as rg
from features import similar_day as sd
from features.labels import make_labels
from features.schema import FEATURE_VERSION, sector_code
from features.sector import sector_features, sector_loo
from features.technical import TRADING_DAYS, ema, stock_features

log = logging.getLogger(__name__)


@dataclass
class Panel:
    """Wide market data aligned on the master trading-date index (rows) and symbols (columns)."""

    dates: pd.DatetimeIndex
    symbols: list[str]
    o: pd.DataFrame
    h: pd.DataFrame
    low: pd.DataFrame
    c: pd.DataFrame
    v: pd.DataFrame
    nifty: pd.DataFrame  # open/high/low/close indexed by dates
    vix: pd.Series
    closes: Mapping[str, pd.Series]  # global series closes keyed by canonical symbol
    sectors: Mapping[str, str | None]
    is_fno: Mapping[str, int]
    fno: pd.DataFrame | None = None
    meetings: pd.DataFrame | None = None
    meetings_coverage_start: date | None = None
    next_after_last: date | None = None
    future_dates: list[date] = field(default_factory=list)


@dataclass
class FeatureSet:
    X: pd.DataFrame  # MultiIndex (date, symbol), float32
    labels: pd.DataFrame | None
    market: pd.DataFrame  # date-indexed market-level features / regime codes
    stage: str
    version: str = FEATURE_VERSION

    @property
    def columns(self) -> list[str]:
        return list(self.X.columns)

    def release(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Hand the matrices to the caller and drop this object's references (no second copy stays alive)."""
        x, y = self.X, self.labels
        self.X, self.labels = pd.DataFrame(), None
        return x, y


def _f32(frame: pd.DataFrame | pd.Series, rows: slice) -> np.ndarray:
    return frame.iloc[rows].to_numpy(dtype=np.float32)


class _Columns:
    """The float32 feature matrix, filled one column at a time into a single preallocated Fortran-ordered buffer.

    The previous design (a dict of per-feature arrays copied into a DataFrame at the end) held two full copies at its peak,
    which does not fit a 1 GB server at 5 years x 220 stocks. Rows are date-major then symbol, matching
    ``MultiIndex.from_product``; the buffer is over-allocated but untouched capacity never becomes resident."""

    def __init__(self, n_dates: int, n_sym: int, capacity: int = 320) -> None:
        self.n_sym = n_sym
        self.buf = np.empty((n_dates * n_sym, capacity), dtype=np.float32, order="F")
        self.names: list[str] = []
        self._index: dict[str, int] = {}

    def add(self, name: str, flat: np.ndarray) -> None:
        j = self._index.get(name)
        if j is None:
            if len(self.names) >= self.buf.shape[1]:
                raise ValueError(f"feature capacity {self.buf.shape[1]} exceeded at {name!r}")
            j = self._index[name] = len(self.names)
            self.names.append(name)
        self.buf[:, j] = flat

    def per_date(self, name: str) -> np.ndarray:
        """Value of a market-level feature for each date (the first symbol's row; they are identical across symbols)."""
        return np.array(self.buf[:: self.n_sym, self._index[name]])

    def frame(self, index: pd.MultiIndex) -> pd.DataFrame:
        return pd.DataFrame(self.buf[:, : len(self.names)], index=index, columns=self.names, copy=False)


def build_feature_set(
    panel: Panel, *, stage: str, thresholds: tuple[float, ...], unusual_k: float, cutoff_hhmm: str,
    with_labels: bool = True, row_slice: slice | None = None,
) -> FeatureSet:
    """Assemble the feature matrix. ``row_slice`` limits which dates are emitted (live: the last row)."""
    dates, symbols = panel.dates, panel.symbols
    n_dates, n_sym = len(dates), len(symbols)
    rows = row_slice or slice(0, n_dates)
    out_dates = dates[rows]
    cols = _Columns(len(out_dates), n_sym)

    def add_stock(name: str, frame: pd.DataFrame, *, shift: bool = True) -> None:
        cols.add(name, _f32(frame.shift(1) if shift else frame, rows).reshape(-1))

    def add_market(name: str, values: pd.Series | np.ndarray, *, shift: bool = True) -> None:
        series = values if isinstance(values, pd.Series) else pd.Series(values, index=dates)
        series = series.shift(1) if shift else series
        cols.add(name, np.repeat(series.iloc[rows].to_numpy(dtype=np.float32), n_sym))

    nifty = panel.nifty.reindex(dates)
    mkt_ret1 = nifty["close"].pct_change(fill_method=None) * 100.0
    mkt_range = (nifty["high"] - nifty["low"]) / nifty["open"] * 100.0

    # --- technical (streamed into the matrix; only the frames other groups read back are kept) --------------------
    tech = stock_features(panel.o, panel.h, panel.low, panel.c, panel.v, mkt_ret1, mkt_range, sink=add_stock)
    ret1 = tech["t_ret1"]

    # --- sector --------------------------------------------------------------------------------------
    sect = sector_features(ret1, tech["t_ret5"], tech["t_ret20"], tech["t_range_prev"], tech["t_rvol1"], panel.sectors,
                           nifty["close"].pct_change(20, fill_method=None) * 100.0)
    for name, frame in sect.items():
        add_stock(name, frame)
    del sect

    # --- market / regime ---------------------------------------------------------------------------------
    market = rg.market_features(nifty, panel.vix, ret1, tech["t_range_prev"], tech["t_rvol1"], tech["t_dist_ema50"])
    market_arrays: dict[str, pd.Series] = {}
    for name, series in market.items():
        add_market(name, series)
        market_arrays[name] = series.shift(1)

    # --- F&O -------------------------------------------------------------------------------------------------
    fmat = fnof.fno_matrices(panel.fno if panel.fno is not None else pd.DataFrame(), dates, symbols)
    for name, frame in fnof.fno_features(fmat, ret1, tech["t_rv20"], tech["t_atr14_pct"]).items():
        add_stock(name, frame)
    exp_ord = fnof.expiry_ordinals(fmat, dates)
    del fmat

    # --- global overnight (already aligned to the date: not shifted) -------------------------------------------
    glob = gf.global_features(panel.closes, dates, cutoff_hhmm)
    for name, values in glob.items():
        add_market(name, values, shift=False)
    sens = gf.sensitivity_features(ret1, glob)
    for name, frame in sens.items():
        add_stock(name, frame, shift=False)
    del sens
    risk = rg.risk_regime(glob["g_us_ret"], glob["g_vix_chg"], glob["g_vix_lvl"])
    add_market("m_risk", risk, shift=False)

    # --- calendar / expiry / results (known in advance: not shifted) -------------------------------------------
    for name, values in cal.calendar_market_features(dates, panel.next_after_last).items():
        add_market(name, values, shift=False)
    expiry, share = cal.expiry_features(exp_ord, dates)
    for name, frame in expiry.items():
        add_stock(name, frame, shift=False)
    add_market("m_expiry_share", share, shift=False)
    meetings = panel.meetings
    for name, frame in cal.results_features(meetings, dates, panel.future_dates, symbols, panel.meetings_coverage_start).items():
        add_stock(name, frame, shift=False)

    # --- pre-market (stage final only): uses the session's OPEN, never high/low/close -----------------------------
    gap_pct: pd.DataFrame | None = None
    if stage == "final":
        prev_close = panel.c.shift(1)
        gap_pct = (panel.o / prev_close - 1.0) * 100.0
        gap_pct = gap_pct.where((prev_close > 0) & (panel.o > 0))
        atr_prev = tech["t_atr14_pct"].shift(1)
        mkt_gap = gap_pct.mean(axis=1, skipna=True)
        add_stock("p_gap_pct", gap_pct, shift=False)
        add_stock("p_gap_abs", gap_pct.abs(), shift=False)
        add_stock("p_gap_atr", gap_pct / atr_prev.where(atr_prev > 0), shift=False)
        add_market("p_mkt_gap", mkt_gap, shift=False)
        add_stock("p_gap_rel_mkt", gap_pct.sub(mkt_gap, axis=0), shift=False)
        add_stock("p_gap_rel_sector", gap_pct - sector_loo(gap_pct, panel.sectors), shift=False)
        hi20 = panel.h.rolling(20, min_periods=10).max().shift(1)
        lo20 = panel.low.rolling(20, min_periods=10).min().shift(1)
        add_stock("p_gap_to_hi20", (panel.o / hi20 - 1.0) * 100.0, shift=False)
        add_stock("p_gap_to_lo20", (panel.o / lo20 - 1.0) * 100.0, shift=False)
        add_stock("p_open_vs_ema20", (panel.o / ema(panel.c, 20).shift(1) - 1.0) * 100.0, shift=False)

    # --- static descriptors ---------------------------------------------------------------------------------------
    cols.add("u_sector_code", np.tile(np.array([sector_code(panel.sectors.get(s)) for s in symbols], dtype=np.float32), len(out_dates)))
    cols.add("u_is_fno", np.tile(np.array([float(panel.is_fno.get(s, 0)) for s in symbols], dtype=np.float32), len(out_dates)))
    cov250 = ret1.rolling(TRADING_DAYS, min_periods=120).cov(mkt_ret1)
    var250 = mkt_ret1.rolling(TRADING_DAYS, min_periods=120).var()
    add_stock("u_beta250", cov250.div(var250, axis=0))
    add_stock("u_range_mean250", tech["t_range_prev"].rolling(TRADING_DAYS, min_periods=120).mean())
    add_stock("u_hist_days", panel.c.notna().cumsum().astype(float))

    # --- labels (unshifted: they describe the session itself) ---------------------------------------------------------
    baseline = tech["t_range_mean20"].shift(1)
    labels = make_labels(panel.o, panel.h, panel.low, panel.c, thresholds=thresholds, unusual_k=unusual_k, baseline=baseline)

    # --- similar-day engine -----------------------------------------------------------------------------------------------
    dims = list(sd.DIMS_FINAL if stage == "final" else sd.DIMS_PRE)
    # vectors must span ALL dates (not just emitted rows) so earlier sessions can be matched
    full_vectors = pd.DataFrame(
        {d: _full_market_vector(d, market_arrays, glob, panel, dates, gap_pct) for d in dims}, index=dates
    )
    label_arrays = {
        k: labels[k].to_numpy(dtype=np.float32) if k in labels else np.full((n_dates, n_sym), np.nan, dtype=np.float32)
        for k in sd.LABELS_USED
    }
    volreg = market_arrays.get("m_volreg")
    target_rows = list(range(rows.start or 0, rows.stop if rows.stop is not None else n_dates)) if row_slice else None
    sim_stock, sim_market = sd.similar_day_features(
        full_vectors, label_arrays, k=30, min_history=120, volreg=volreg.to_numpy(dtype=np.float32) if volreg is not None else None,
        rows=target_rows,
    )
    for name, matrix in sim_stock.items():
        cols.add(name, matrix[rows].reshape(-1))
    for name, vector in sim_market.items():
        cols.add(name, np.repeat(vector[rows], n_sym))
    del sim_stock, sim_market, label_arrays, full_vectors, tech

    index = pd.MultiIndex.from_product([out_dates, symbols], names=["date", "symbol"])
    x = cols.frame(index)
    label_frame = None
    if with_labels:
        label_frame = pd.DataFrame({k: _f32(v, rows).reshape(-1) for k, v in labels.items()}, index=index)

    market_frame = pd.DataFrame({name: cols.per_date(name) for name in cols.names
                                 if name.startswith(("m_", "g_", "c_dow", "c_month", "x_sim_d"))}, index=out_dates)
    return FeatureSet(x, label_frame, market_frame, stage)


def _full_market_vector(
    name: str, market_arrays: Mapping[str, pd.Series], glob: Mapping[str, np.ndarray], panel: Panel,
    dates: pd.DatetimeIndex, gap_pct: pd.DataFrame | None,
) -> np.ndarray:
    if name in glob:
        return np.asarray(glob[name], dtype=np.float64)
    if name in market_arrays:
        return market_arrays[name].to_numpy(dtype=np.float64)
    if name == "p_mkt_gap" and gap_pct is not None:
        return gap_pct.mean(axis=1, skipna=True).to_numpy(dtype=np.float64)
    return np.full(len(dates), np.nan)


def finite_mask(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Rows where every named column is finite."""
    return frame[columns].notna().all(axis=1)
