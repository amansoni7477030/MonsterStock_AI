"""Yahoo Finance chart API client (unofficial; research-grade - see ingestion.registry for licence notes).

Bars are keyed by the *exchange-local* session date derived from the bar timestamp and the exchange timezone
reported by Yahoo, which is how point-in-time availability is later computed (ingestion.series).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
SPARK_URL = "https://query1.finance.yahoo.com/v8/finance/spark"
SPARK_FALLBACK_URL = "https://query2.finance.yahoo.com/v8/finance/spark"
SPARK_MAX_SYMBOLS = 20  # hard limit of the endpoint ("Number of symbols needs to be less than or equal to 20")


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
        self.http = http or HttpClient.from_settings(
            "yahoo", settings, headers={"User-Agent": USER_AGENT_SIMPLE, "Accept": "application/json"},
            min_interval=settings.yahoo_min_interval_sec, clock=clock,
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

    def fetch_intraday(self, symbol: str, *, range_: str = "5d", interval: str = "5m") -> list[dict[str, Any]]:
        """Intraday bars (five-minute by default, the last few sessions), oldest first. Used after the close to judge intraday levels
        against the path the price took; never to build a prediction (Yahoo's feed omits the opening auction's volume and some
        closing bars)."""
        return self._parse_intraday(self._chart(symbol, {"range": range_, "interval": interval}))

    @staticmethod
    def _parse_intraday(result: Mapping[str, Any]) -> list[dict[str, Any]]:
        meta = result.get("meta") or {}
        tz = _tz(str(meta.get("exchangeTimezoneName") or "UTC"))
        stamps = result.get("timestamp") or []
        block = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
        bars: list[dict[str, Any]] = []
        for idx, stamp in enumerate(stamps):
            try:
                moment = datetime.fromtimestamp(int(stamp), timezone.utc)
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            o, h, low, c = (_num(_at(block.get(k), idx)) for k in ("open", "high", "low", "close"))
            if None in (o, h, low, c):  # Yahoo leaves a null bar where nothing traded (and in the closing minutes)
                continue
            local = moment.astimezone(tz)
            bars.append({"ts": iso(moment), "trade_date": local.date().isoformat(), "hhmm": local.strftime("%H:%M"), "open": o, "high": h,
                         "low": low, "close": c, "volume": _num(_at(block.get("volume"), idx))})
        bars.sort(key=lambda b: b["ts"])
        return bars

    def spark_latest(self, symbols: Sequence[str]) -> dict[str, SparkPoint]:
        """Newest daily point of MANY symbols in ``ceil(n / 20)`` requests (the spark endpoint answers up to 20 symbols per
        call; a chart call answers one). Used as a freshness probe so unchanged series are not fetched one by one."""
        unique = list(dict.fromkeys(symbols))
        out: dict[str, SparkPoint] = {}
        for start in range(0, len(unique), SPARK_MAX_SYMBOLS):
            chunk = unique[start:start + SPARK_MAX_SYMBOLS]
            last: Exception | None = None
            for base in (SPARK_URL, SPARK_FALLBACK_URL):
                try:
                    payload = self.http.get(base, params={"symbols": ",".join(chunk), "range": "5d", "interval": "1d"}).json()
                except DataUnavailable as exc:
                    last = exc
                    continue
                out.update(parse_spark(payload))
                break
            else:
                raise last or DataUnavailable("yahoo", "spark probe failed")
        return out


@dataclass(frozen=True)
class SparkPoint:
    """The newest point of a spark series: when its bar started (UTC) and its last / closing price."""

    ts: datetime
    close: float


def parse_spark(payload: Any) -> dict[str, SparkPoint]:
    """``{symbol: SparkPoint}`` from a ``v8/finance/spark`` payload. Entries without a usable (timestamp, close) pair are
    omitted, so the caller simply fetches those symbols the normal way."""
    out: dict[str, SparkPoint] = {}
    if not isinstance(payload, dict):
        return out
    for symbol, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        stamps, closes = entry.get("timestamp") or [], entry.get("close") or []
        for idx in range(min(len(stamps), len(closes)) - 1, -1, -1):  # newest non-null point
            close = _num(closes[idx])
            if close is None:
                continue
            try:
                out[str(symbol)] = SparkPoint(datetime.fromtimestamp(int(stamps[idx]), timezone.utc), close)
            except (TypeError, ValueError, OverflowError, OSError):
                pass
            break
    return out


def _at(seq: Any, idx: int) -> Any:
    try:
        return seq[idx]
    except (TypeError, IndexError, KeyError):
        return None
