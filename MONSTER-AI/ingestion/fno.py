"""F&O aggregation: from raw bhavcopy contract rows to one compact feature row per stock per day.

Everything is computed from *end-of-day settlement data of the previous session*, identical in training and at
inference time (point-in-time safe). Implied volatility is inverted from settlement prices with Black-Scholes
(European, r = 6.5%, no dividends) and is therefore an approximation - a feature, not a fact.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
from scipy.special import ndtr

RISK_FREE = 0.065
MIN_OPTION_PRICE = 1.0  # rupees; below this IV is dominated by tick-size noise


def bs_price(spot: np.ndarray, strike: np.ndarray, t: float, sigma: np.ndarray, is_call: np.ndarray,
             r: float = RISK_FREE) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    sqrt_t = math.sqrt(max(t, 1e-9))
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = math.exp(-r * t)
    call = spot * ndtr(d1) - strike * disc * ndtr(d2)
    put = strike * disc * ndtr(-d2) - spot * ndtr(-d1)
    return np.where(is_call, call, put)


def implied_vol(price: np.ndarray, spot: np.ndarray, strike: np.ndarray, t: float, is_call: np.ndarray,
                r: float = RISK_FREE, iterations: int = 60) -> np.ndarray:
    """Vectorised bisection; NaN where the price is below intrinsic, above the no-arbitrage cap or non-positive."""
    price = np.asarray(price, dtype=float)
    spot = np.broadcast_to(np.asarray(spot, dtype=float), price.shape)
    strike = np.asarray(strike, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    disc = math.exp(-r * t)
    intrinsic = np.where(is_call, np.maximum(spot - strike * disc, 0.0), np.maximum(strike * disc - spot, 0.0))
    upper_cap = np.where(is_call, spot, strike * disc)
    valid = np.isfinite(price) & (price > intrinsic + 1e-9) & (price < upper_cap)
    lo = np.full(price.shape, 0.005)
    hi = np.full(price.shape, 4.0)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        too_high = bs_price(spot, strike, t, mid, is_call, r) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    out = 0.5 * (lo + hi)
    out = np.where(valid & (out > 0.0051) & (out < 3.99), out, np.nan)
    return out


def _tfrac(dte: int) -> float:
    return max(float(dte), 0.5) / 365.0


def _pick(values: Sequence[Any]) -> float | None:
    for v in values:
        if v is not None and v == v:
            return float(v)
    return None


def aggregate_stock_fno(
    rows: Iterable[Mapping[str, Any]], trade_date: date, wanted: set[str] | None = None
) -> list[dict[str, Any]]:
    """One ``fno_daily`` row per stock with futures/options activity on ``trade_date``."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("instrument") not in {"STF", "STO"}:
            continue
        symbol = row.get("symbol")
        if not symbol or (wanted is not None and symbol not in wanted):
            continue
        groups[symbol].append(row)

    out: list[dict[str, Any]] = []
    for symbol, items in groups.items():
        futures = sorted((r for r in items if r["instrument"] == "STF"), key=lambda r: r["expiry"])
        options = [r for r in items if r["instrument"] == "STO"]
        expiries = sorted({r["expiry"] for r in futures} or {r["expiry"] for r in options})
        if not expiries:
            continue
        near_expiry = expiries[0]
        dte = (date.fromisoformat(near_expiry) - trade_date).days
        underlying = _pick([r.get("underlying") for r in items])
        record: dict[str, Any] = {"symbol": symbol, "trade_date": trade_date.isoformat(), "underlying": underlying,
                                  "lot_size": _pick([r.get("lot_size") for r in items]), "dte": dte}

        near_future = next((r for r in futures if r["expiry"] == near_expiry), None)
        if near_future is not None:
            settle = near_future.get("settle") or near_future.get("close")
            record.update(
                fut_close=near_future.get("close"), fut_settle=settle, fut_oi=near_future.get("oi"),
                fut_oi_chg=near_future.get("oi_chg"), fut_volume=near_future.get("volume"),
                fut_oi_all=sum(r.get("oi") or 0.0 for r in futures),
                fut_oi_chg_all=sum(r.get("oi_chg") or 0.0 for r in futures),
            )
            if underlying and settle:
                record["basis_pct"] = (settle / underlying - 1.0) * 100.0
            nxt = futures[1] if len(futures) > 1 else None
            if nxt is not None:
                record["next_fut_oi"] = nxt.get("oi")
                near_oi, next_oi = near_future.get("oi") or 0.0, nxt.get("oi") or 0.0
                if near_oi + next_oi > 0:
                    record["rollover_pct"] = next_oi / (near_oi + next_oi) * 100.0

        record.update(_option_metrics([r for r in options if r["expiry"] == near_expiry], underlying, dte))
        out.append(record)
    return out


def _option_metrics(options: Sequence[Mapping[str, Any]], spot: float | None, dte: int) -> dict[str, Any]:
    if not options or not spot or spot <= 0:
        return {}
    strikes = np.array([o["strike"] if o.get("strike") is not None else np.nan for o in options], dtype=float)
    is_call = np.array([o.get("option_type") == "CE" for o in options], dtype=bool)
    is_put = np.array([o.get("option_type") == "PE" for o in options], dtype=bool)
    price = np.array([(o.get("settle") if o.get("settle") else o.get("close")) or np.nan for o in options], dtype=float)
    oi = np.array([o.get("oi") or 0.0 for o in options], dtype=float)
    oi_chg = np.array([o.get("oi_chg") or 0.0 for o in options], dtype=float)
    vol = np.array([o.get("volume") or 0.0 for o in options], dtype=float)
    valid_strike = np.isfinite(strikes)
    if not valid_strike.any():
        return {}

    result: dict[str, Any] = {
        "call_oi": float(oi[is_call].sum()), "put_oi": float(oi[is_put].sum()),
        "call_oi_chg": float(oi_chg[is_call].sum()), "put_oi_chg": float(oi_chg[is_put].sum()),
        "call_vol": float(vol[is_call].sum()), "put_vol": float(vol[is_put].sum()),
    }
    if result["call_oi"] > 0:
        result["pcr_oi"] = result["put_oi"] / result["call_oi"]
    if result["call_vol"] > 0:
        result["pcr_vol"] = result["put_vol"] / result["call_vol"]
    if is_call.any() and oi[is_call].max() > 0:
        result["max_call_oi_strike"] = float(strikes[is_call][np.nanargmax(np.where(valid_strike[is_call], oi[is_call], -1))])
    if is_put.any() and oi[is_put].max() > 0:
        result["max_put_oi_strike"] = float(strikes[is_put][np.nanargmax(np.where(valid_strike[is_put], oi[is_put], -1))])

    call_strikes = {float(k): float(p) for k, p, c in zip(strikes, price, is_call, strict=True) if c and np.isfinite(k) and np.isfinite(p) and p > 0}
    put_strikes = {float(k): float(p) for k, p, c in zip(strikes, price, is_put, strict=True) if c and np.isfinite(k) and np.isfinite(p) and p > 0}
    common = sorted(set(call_strikes) & set(put_strikes))
    t = _tfrac(dte)
    if common:
        atm = min(common, key=lambda k: (abs(k - spot), k))
        straddle = call_strikes[atm] + put_strikes[atm]
        result.update(atm_strike=atm, atm_straddle=straddle, expected_move_pct=straddle / spot * 100.0)
        ivs = implied_vol(np.array([call_strikes[atm], put_strikes[atm]]), spot, np.array([atm, atm]), t,
                          np.array([True, False]))
        finite = ivs[np.isfinite(ivs)]
        if finite.size:
            result["iv_atm"] = float(finite.mean() * 100.0)
        else:  # Brenner-Subrahmanyam approximation from the straddle
            result["iv_atm"] = float(straddle / (0.7979 * spot * math.sqrt(t)) * 100.0)
    skew = _skew(strikes, price, is_call, is_put, spot, t)
    if skew is not None:
        result["iv_skew"] = skew
    return result


def _skew(strikes: np.ndarray, price: np.ndarray, is_call: np.ndarray, is_put: np.ndarray, spot: float, t: float) -> float | None:
    """IV(OTM put near 95% moneyness) minus IV(OTM call near 105%), in vol points. Positive = downside premium."""

    def side(mask: np.ndarray, target: float, call: bool) -> float | None:
        ok = mask & np.isfinite(strikes) & np.isfinite(price) & (price >= MIN_OPTION_PRICE) & (np.abs(strikes / spot - 1) <= 0.10)
        if not ok.any():
            return None
        candidates = np.flatnonzero(ok)
        order = candidates[np.argsort(np.abs(strikes[candidates] - target))]
        for idx in order[:4]:
            iv = implied_vol(np.array([price[idx]]), spot, np.array([strikes[idx]]), t, np.array([call]))[0]
            if np.isfinite(iv):
                return float(iv * 100.0)
        return None

    put_iv = side(is_put, spot * 0.95, False)
    call_iv = side(is_call, spot * 1.05, True)
    if put_iv is None or call_iv is None:
        return None
    return put_iv - call_iv


def summarize_option_chain(payload: Any, expiry: str) -> dict[str, Any] | None:
    """Condense the live ``option-chain-v3`` payload (descriptive confirmation only, never a model input)."""
    if not isinstance(payload, Mapping):
        return None
    records = payload.get("records")
    if not isinstance(records, Mapping):
        return None
    spot = _pick([records.get("underlyingValue")])
    rows = [r for r in (records.get("data") or []) if isinstance(r, Mapping) and r.get("expiryDates") in (expiry, None)]
    if not rows or not spot:
        return None

    def leg(row: Mapping[str, Any], side: str, key: str) -> float:
        value = (row.get(side) or {}).get(key)
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    call_oi = sum(leg(r, "CE", "openInterest") for r in rows)
    put_oi = sum(leg(r, "PE", "openInterest") for r in rows)
    call_vol = sum(leg(r, "CE", "totalTradedVolume") for r in rows)
    put_vol = sum(leg(r, "PE", "totalTradedVolume") for r in rows)
    strikes = [r for r in rows if r.get("strikePrice") is not None]
    if not strikes:
        return None
    atm_row = min(strikes, key=lambda r: abs(float(r["strikePrice"]) - spot))
    ce_iv, pe_iv = leg(atm_row, "CE", "impliedVolatility"), leg(atm_row, "PE", "impliedVolatility")
    ivs = [v for v in (ce_iv, pe_iv) if v > 0]
    straddle = leg(atm_row, "CE", "lastPrice") + leg(atm_row, "PE", "lastPrice")
    return {
        "underlying": spot, "expiry": expiry, "atm_strike": float(atm_row["strikePrice"]),
        "atm_iv": sum(ivs) / len(ivs) if ivs else None, "straddle": straddle or None,
        "expected_move_pct": straddle / spot * 100.0 if straddle else None,
        "pcr_oi": put_oi / call_oi if call_oi else None, "pcr_vol": put_vol / call_vol if call_vol else None,
        "call_oi": call_oi, "put_oi": put_oi,
        "call_oi_chg": sum(leg(r, "CE", "changeinOpenInterest") for r in rows),
        "put_oi_chg": sum(leg(r, "PE", "changeinOpenInterest") for r in rows),
        "max_call_oi_strike": float(max(strikes, key=lambda r: leg(r, "CE", "openInterest"))["strikePrice"]),
        "max_put_oi_strike": float(max(strikes, key=lambda r: leg(r, "PE", "openInterest"))["strikePrice"]),
        "source_ts": records.get("timestamp"),
    }
