"""F&O features from end-of-day futures/options data (blueprint section 8).

Price/OI relationships and PCR are *features, not facts*: they are heuristics about positioning, never proof of
participant intent. Unknown stays NaN (no F&O history -> NaN, never 0) so the model can tell "missing" from "flat".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FIELDS = (
    "underlying", "fut_close", "fut_settle", "fut_oi", "fut_oi_chg", "fut_volume", "fut_oi_all", "fut_oi_chg_all",
    "basis_pct", "rollover_pct", "call_oi", "put_oi", "call_oi_chg", "put_oi_chg", "pcr_oi", "pcr_vol", "iv_atm",
    "iv_skew", "max_call_oi_strike", "max_put_oi_strike", "dte",
)


def fno_matrices(
    fno: pd.DataFrame, dates: pd.DatetimeIndex, symbols: list[str]
) -> dict[str, pd.DataFrame]:
    """Pivot the long ``fno_daily`` frame into (dates x symbols) matrices aligned to the master calendar."""
    out: dict[str, pd.DataFrame] = {}
    for field in FIELDS:
        if fno is None or fno.empty or field not in fno.columns:
            out[field] = pd.DataFrame(np.nan, index=dates, columns=symbols)
            continue
        wide = fno.pivot_table(index="trade_date", columns="symbol", values=field, aggfunc="last")
        out[field] = wide.reindex(index=dates, columns=symbols).astype(float)
    return out


def _safe_pct(change: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    prior = level - change
    return (change / prior.where(prior > 0) * 100.0).where(level.notna() & change.notna())


def fno_features(m: dict[str, pd.DataFrame], ret1: pd.DataFrame, rv20: pd.DataFrame, atr14_pct: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derived F&O features as of each session's close (shifted by the builder)."""
    out: dict[str, pd.DataFrame] = {}
    out["f_fut_oi_chg_pct"] = _safe_pct(m["fut_oi_chg"], m["fut_oi"])
    out["f_fut_oi_all_chg_pct"] = _safe_pct(m["fut_oi_chg_all"], m["fut_oi_all"])
    out["f_basis_pct"] = m["basis_pct"]
    out["f_rollover_pct"] = m["rollover_pct"]
    out["f_pcr_oi"] = m["pcr_oi"]
    out["f_pcr_vol"] = m["pcr_vol"]
    out["f_call_oi_chg_pct"] = _safe_pct(m["call_oi_chg"], m["call_oi"])
    out["f_put_oi_chg_pct"] = _safe_pct(m["put_oi_chg"], m["put_oi"])
    iv = m["iv_atm"]
    out["f_iv_atm"] = iv
    out["f_iv_chg5"] = iv - iv.shift(5)
    out["f_iv_rank120"] = iv.rolling(120, min_periods=40).rank(pct=True)
    out["f_iv_skew"] = m["iv_skew"]
    exp_move = iv / 100.0 * np.sqrt(1.0 / 252.0) * 100.0
    out["f_exp_move_1d"] = exp_move
    out["f_iv_vs_rv"] = iv / rv20.where(rv20 > 0)
    out["f_exp_vs_atr"] = exp_move / atr14_pct.where(atr14_pct > 0)
    underlying = m["underlying"].where(m["underlying"] > 0)
    out["f_dist_maxcall"] = (m["max_call_oi_strike"] / underlying - 1.0) * 100.0
    out["f_dist_maxput"] = (m["max_put_oi_strike"] / underlying - 1.0) * 100.0
    out["f_dte"] = m["dte"]
    price_up = ret1 > 0
    oi_up = m["fut_oi_chg"] > 0
    known = ret1.notna() & m["fut_oi_chg"].notna()
    code = np.where(price_up & oi_up, 1.0, np.where(~price_up & oi_up, 2.0, np.where(price_up & ~oi_up, 3.0, 4.0)))
    out["f_oi_price_code"] = pd.DataFrame(code, index=ret1.index, columns=ret1.columns).where(known)
    out["f_oi_price_signed"] = np.sign(ret1) * out["f_fut_oi_chg_pct"]
    out["f_fut_vol_rel"] = m["fut_volume"] / m["fut_volume"].rolling(20, min_periods=8).mean()
    out["f_fut_oi_rel60"] = m["fut_oi_all"] / m["fut_oi_all"].rolling(60, min_periods=20).mean()
    return out


def expiry_ordinals(m: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar ordinal of the near expiry as known on each session (date + dte)."""
    ordinals = np.array([d.toordinal() for d in dates], dtype=float)[:, None]
    return m["dte"] + ordinals
