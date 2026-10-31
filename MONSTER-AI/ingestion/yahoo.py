"""Yahoo Finance chart API client (unofficial; research-grade - see ingestion.registry for licence notes).

Bars are keyed by the *exchange-local* session date derived from the bar timestamp and the exchange timezone
reported by Yahoo, which is how point-in-time availability is later computed (ingestion.series).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.clock import Clock, iso
from app.config import Settings
from app.errors import DataUnavailable
from ingestion.http import USER_AGENT_SIMPLE, HttpClient

log = logging.getLogger(__name__)

BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"
FALLBACK_BASE = "https://query2.finance.yahoo.com/v8/finance/chart/"


@dataclass
class YahooSeries:
    symbol: str
    meta: Mapping[str, Any]
    bars: list[dict[str, Any]]
    splits: list[dict[str, Any]] = field(default_factory=list)
    dividends: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tz_name(self) -> str:
        return str(self.meta.get("exchangeTimezoneName") or "UTC")


class SymbolNotFound(DataUnavailable):
    """Yahoo does not know the symbol (delisted / renamed): caller may quarantine after repeated hits."""

    def __init__(self, symbol: str) -> None:
        super().__init__("yahoo", f"symbol not found: {symbol}", retryable=False)
        self.symbol = symbol


def _tz(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


class YahooClient:
    def __init__(self, settings: Settings, *, http: HttpClient | None = None, clock: Clock | None = None) -> None:
        self.clock = clock
        self.http = http or HttpClient(
            "yahoo",
            headers={"User-Agent": USER_AGENT_SIMPLE, "Accept": "application/json"},
            min_interval=settings.yahoo_min_interval_sec,
            timeout=(10.0, settings.http_timeout_sec),
            max_attempts=settings.http_max_attempts,
            max_bytes=settings.http_max_response_mb * 1024 * 1024,
            clock=clock,
        )

    # ------------------------------------------------------------------------------------------ fetch
    def _chart(self, symbol: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        path = quote(symbol, safe="")
        last: Exception | None = None
        for base in (BASE, FALLBACK_BASE):
            try:
                response = self.http.get(base + path, params=params, soft_statuses=(404,))
            except DataUnavailable as exc:
                last = exc
                continue
            payload = response.json()
            chart = payload.get("chart") if isinstance(payload, dict) else None
            if not isinstance(chart, dict):
                last = DataUnavailable("yahoo", "unexpected response shape")
                continue
            error = chart.get("error")
            if error:
                code = str(error.get("code", "")) if isinstance(error, dict) else str(error)
                if "not found" in code.lower() or response.status == 404:
                    raise SymbolNotFound(symbol)
                last = DataUnavailable("yahoo", f"api error: {code[:60]}")
                continue
            results = chart.get("result") or []
            if not results:
                raise SymbolNotFound(symbol)
            return results[0]
        raise last or DataUnavailable("yahoo", "no usable response")

    def fetch_daily(self, symbol: str, *, range_: str = "1mo") -> YahooSeries:
        """Daily OHLCV bars (+ split/dividend events). Callers apply the point-in-time rule (a bar is usable only
        once its session is final, see ingestion.series) so in-progress bars never reach the database."""
        result = self._chart(symbol, {"range": range_, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"})
        return self._parse_daily(symbol, result)

    def _parse_daily(self, symbol: str, result: Mapping[str, Any]) -> YahooSeries:
        meta = result.get("meta") or {}
        tz = _tz(str(meta.get("exchangeTimezoneName") or "UTC"))
        stamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote_block = (indicators.get("quote") or [{}])[0]
        adj_block = ((indicators.get("adjclose") or [{}])[0] or {}).get("adjclose") or []

        bars: list[dict[str, Any]] = []
        for idx, stamp in enumerate(stamps):
            try:
                moment = datetime.fromtimestamp(int(stamp), timezone.utc)
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            o, h, low, c = (_num(_at(quote_block.get(k), idx)) for k in ("open", "high", "low", "close"))
            if None in (o, h, low, c):
                continue
            bars.append(
                {
                    "trade_date": moment.astimezone(tz).date().isoformat(), "open": o, "high": h, "low": low, "close": c,
                    "adj_close": _num(_at(adj_block, idx)), "volume": _num(_at(quote_block.get("volume"), idx)),
                }
            )

        events = result.get("events") or {}
        splits = []
        for item in (events.get("splits") or {}).values():
            try:
                splits.append({
                    "ex_date": datetime.fromtimestamp(int(item["date"]), timezone.utc).astimezone(tz).date().isoformat(),
                    "numerator": float(item.get("numerator", 1) or 1), "denominator": float(item.get("denominator", 1) or 1),
                })
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                continue
        dividends = []
        for item in (events.get("dividends") or {}).values():
            try:
                dividends.append({
                    "ex_date": datetime.fromtimestamp(int(item["date"]), timezone.utc).astimezone(tz).date().isoformat(),
                    "amount": float(item["amount"]),
                })
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                continue
        return YahooSeries(symbol, meta, bars, sorted(splits, key=lambda s: s["ex_date"]),
                           sorted(dividends, key=lambda d: d["ex_date"]))

    def fetch_intraday(self, symbol: str, *, interval: str = "5m", range_: str = "1d") -> list[dict[str, Any]]:
        """Intraday bars (UTC timestamps) for post-session analysis of the day's path (time-to-extreme, MFE/MAE)."""
        result = self._chart(symbol, {"range": range_, "interval": interval, "includePrePost": "false"})
        stamps = result.get("timestamp") or []
        q = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        out = []
        for idx, stamp in enumerate(stamps):
            o, h, low, c = (_num(_at(q.get(k), idx)) for k in ("open", "high", "low", "close"))
            if None in (o, h, low, c):
                continue
            out.append({"ts": iso(datetime.fromtimestamp(int(stamp), timezone.utc)), "open": o, "high": h, "low": low,
                        "close": c, "volume": _num(_at(q.get("volume"), idx))})
        return out

    def fetch_quote(self, symbol: str) -> dict[str, Any]:
        """Latest price snapshot from the chart meta (descriptive context only)."""
        result = self._chart(symbol, {"range": "1d", "interval": "5m"})
        meta = result.get("meta") or {}
        price = _num(meta.get("regularMarketPrice"))
        prev = _num(meta.get("chartPreviousClose")) or _num(meta.get("previousClose"))
        stamp = meta.get("regularMarketTime")
        source_ts = iso(datetime.fromtimestamp(int(stamp), timezone.utc)) if stamp else None
        change = ((price / prev - 1.0) * 100.0) if price and prev else None
        return {"symbol": symbol, "price": price, "prev_close": prev, "change_pct": change, "source_ts": source_ts}


def _at(seq: Any, idx: int) -> Any:
    try:
        return seq[idx]
    except (TypeError, IndexError, KeyError):
        return None


def bar_date_range(bars: list[dict[str, Any]]) -> tuple[date, date] | None:
    if not bars:
        return None
    return date.fromisoformat(bars[0]["trade_date"]), date.fromisoformat(bars[-1]["trade_date"])
