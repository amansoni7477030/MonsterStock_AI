"""Kotak Neo market data, read-only: live quotes of many stocks per request and exchange-grade intraday candles.

Why it exists. The live scans read NSE's website API, one source behind one cookie handshake that datacentre addresses are often
refused. The broker feeds the same exchange data through a documented, authenticated API: a second, independent live source to
cross-check NSE against (and to carry a scan when NSE is down), and 5-minute candles built from the exchange's own trades. Yahoo's
5-minute bars are not: measured on the same stock-days, Yahoo's first-15-minute range is about a tenth narrower than the exchange's
(its first bar misses the opening spike), and its prices carry float32 noise.

What it uses. Only the application token (``KOTAK_CONSUMER_KEY``, sent as the ``Authorization`` header). The quotes, history and
instrument-master endpoints need no login: no MPIN, no TOTP secret, no mobile number and no UCC, which can place orders and are never
read, stored or sent. This system never trades. The token is sent to https hosts on ``kotaksecurities.com`` only (``is_kotak_url``);
a file URL the API returns is followed only when it is on such a host too, and then without the token.

Measured on the live API (2026-11-06, ``docs/DATA_SOURCES.md``):

* ``GET script-details/1.0/quotes/neosymbol/nse_cm|<token>,nse_cm|<token>/all``: the documented limit is 50 symbols per request, but 50
  is answered with HTTP 400 and 49 works; this client asks for 40. A row has ``ltp``, ``ohlc`` (``close`` is the PREVIOUS close),
  ``last_volume`` (the day's volume so far), ``avg_cost`` (VWAP) and ``lstup_time`` (UTC epoch seconds of the stock's last update).
  After the close every figure equals NSE's own row exactly (18 of 18 stocks, volumes included).
* ``GET market-data/1.0/historical/details``: 1/3/5-minute candles for at most 30 days per request, start-stamped, with the volume of
  the opening auction in the first bar; the archive reaches back years (2023 answered). A session normally ends with the 15:10
  five-minute bar: the last 15 minutes are missing, or appear only as a flat artefact after the day, and on some days the 15:10 bar
  carries a negative volume (dropped by the parser). About one request a second is accepted.
* Five-minute candles are exactly the aggregate of the one-minute candles; the day's low matched NSE's official low for 60 of 60 stocks.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import quote, urlsplit

from app.clock import IST, UTC, Clock, get_clock, iso, parse_ts
from app.config import Settings, is_kotak_url
from app.errors import DataUnavailable
from app.logging_setup import redact
from ingestion.http import HttpClient, HttpResponse

log = logging.getLogger(__name__)

SEGMENT = "nse_cm"
FILE_PATHS = "script-details/1.0/masterscrip/file-paths"
QUOTES = "script-details/1.0/quotes/neosymbol/{pairs}/all"
HISTORY = "market-data/1.0/historical/details"
QUOTE_BATCH = 40  # the documented limit is 50 and the live API refuses exactly 50 (HTTP 400): leave a margin
MAX_SPAN_DAYS = 30  # the longest window of one 1/3/5-minute request
INTERVALS = frozenset({"1min", "3min", "5min", "10min", "15min", "30min", "60min"})
FIRST_BAR_MIN = 9 * 60 + 15  # IST minute of the day of the session's first bar
LAST_BAR_END_MIN = 15 * 60 + 30  # a bar must START before this
MIN_STOCKS = 500  # an instrument master with fewer stocks than this is not the real file
# The broker quotes an index by its (case-sensitive) name, not by a token; the scans key the same indices by NSE's upper-case names.
INDEX_QUOTES = {"Nifty 50": "NIFTY 50", "INDIA VIX": "INDIA VIX"}
PRICE_SLACK = 1e-6  # relative slack when checking that a bar's high / low contain its open and close
FUTURE_SLACK_SEC = 60.0  # how far ahead of this machine's clock a quote's stamp may be (clock skew) before it is not believed
AWARE_STAMP_FORMATS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z")  # an offset in the text: 2026-11-06T09:15:00+0530
NAIVE_STAMP_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")  # no offset: the exchange's clock, IST
AUTH_HINT = "Kotak refused the application token (HTTP {code}): copy KOTAK_CONSUMER_KEY again from NEO app > More > Trade API"


class _BadRequest(Exception):
    """HTTP 400 with a fault body: the request itself was refused (an unknown token in a batch, a window the API does not serve)."""


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _positive(value: Any) -> float | None:
    out = _num(value)
    return out if out is not None and out > 0 else None


def _epoch(value: Any) -> datetime | None:
    """A UTC epoch (seconds; milliseconds are recognised) as an aware datetime, ``None`` when it is not a plausible 21st-century stamp."""
    seconds = _num(value)
    if seconds is None or seconds <= 0:
        return None
    stamp = parse_ts(seconds)
    return stamp if stamp is not None and 2020 <= stamp.year <= 2100 else None


def _bar_stamp(value: Any) -> datetime | None:
    """A candle's start: ISO text with an offset (``2026-11-06T09:15:00+0530``), naive text (IST) or an epoch."""
    if isinstance(value, (int, float)):
        return _epoch(value)
    text = str(value or "").strip()
    for fmt in AWARE_STAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt).astimezone(UTC)
        except ValueError:
            continue
    for fmt in NAIVE_STAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=IST).astimezone(UTC)
        except ValueError:
            continue
    return parse_ts(text, assume_tz=IST)


# ---------------------------------------------------------------------------------------------------- parsers
def parse_token_map(text: str) -> dict[str, str]:
    """``{NSE symbol: instrument token}`` of the cash-market equities (the ``-EQ`` series) of an ``nse_cm`` instrument master. The master's
    last header cell carries a stray ``;`` and some cells are padded, so keys and values are stripped."""
    out: dict[str, str] = {}
    for raw in csv.DictReader(io.StringIO(text)):
        row = {(k or "").strip().rstrip(";"): (v or "").strip() for k, v in raw.items()}
        if row.get("pExchSeg", SEGMENT).lower() != SEGMENT:
            continue
        trading = row.get("pTrdSymbol", "").upper()
        if not (row.get("pGroup", "").upper() == "EQ" or trading.endswith("-EQ")):
            continue
        try:
            token = str(int(float(row["pSymbol"])))
        except (KeyError, ValueError):
            continue
        for name in (row.get("pSymbolName", "").upper(), trading[:-3] if trading.endswith("-EQ") else ""):
            if name:
                out.setdefault(name, token)
    return out


def parse_quotes(data: Any, by_token: Mapping[str, str]) -> list[dict[str, Any]]:
    """Live rows in the shape of NSE's (``symbol``, ``open``, ``high``, ``low``, ``last``, ``prev_close``, ``volume`` ...), one per stock
    that answered with a last price and a previous close. ``row_ts`` is the stock's own last update (UTC), not the time of the call. A
    figure the broker leaves out, or sends as 0 before the session opens, stays ``None``."""
    rows: list[dict[str, Any]] = []
    if not isinstance(data, list):
        return rows
    for item in data:
        if not isinstance(item, Mapping):
            continue
        symbol = by_token.get(str(item.get("exchange_token") or "").strip())
        ohlc = item.get("ohlc") if isinstance(item.get("ohlc"), Mapping) else {}
        last, prev = _positive(item.get("ltp")), _positive(ohlc.get("close"))
        if symbol is None or last is None or prev is None:
            continue
        volume, vwap = _num(item.get("last_volume")), _positive(item.get("avg_cost"))
        stamp = _epoch(item.get("lstup_time"))
        rows.append({
            "symbol": symbol, "open": _positive(ohlc.get("open")), "high": _positive(ohlc.get("high")), "low": _positive(ohlc.get("low")),
            "last": last, "prev_close": prev, "change_pct": (last / prev - 1.0) * 100.0, "volume": volume,
            "turnover": volume * vwap if volume is not None and vwap is not None else None, "vwap": vwap,
            "year_high": _positive(item.get("year_high")), "year_low": _positive(item.get("year_low")),
            "row_ts": iso(stamp) if stamp else None,
        })
    return rows


def parse_indices(data: Any) -> dict[str, dict[str, Any]]:
    """``{"NIFTY 50": {...}, "INDIA VIX": {...}}`` in the shape of NSE's ``allIndices`` rows (``name``, ``last``, ``prev_close``, ``open``,
    ``high``, ``low``, ``change_pct``, ``row_ts``) from an index quote answer; an index without a last price is left out."""
    out: dict[str, dict[str, Any]] = {}
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, Mapping):
            continue
        name = INDEX_QUOTES.get(str(item.get("exchange_token") or "").strip())
        ohlc = item.get("ohlc") if isinstance(item.get("ohlc"), Mapping) else {}
        last, prev = _positive(item.get("ltp")), _positive(ohlc.get("close"))
        if name is None or last is None:
            continue
        stamp = _epoch(item.get("lstup_time"))
        out[name] = {"name": name, "last": last, "prev_close": prev, "open": _positive(ohlc.get("open")),
                     "high": _positive(ohlc.get("high")), "low": _positive(ohlc.get("low")),
                     "change_pct": (last / prev - 1.0) * 100.0 if prev else None, "row_ts": iso(stamp) if stamp else None}
    return out


def parse_candles(payload: Any) -> list[dict[str, Any]]:
    """The bars of one history answer, oldest first, in the shape Yahoo's intraday bars have (``ts``, ``trade_date``, ``hhmm`` in IST,
    ``open`` / ``high`` / ``low`` / ``close`` / ``volume``). Rows that are not a bar of the regular session are dropped: no usable price,
    a high or low that does not contain the open and close, a negative volume (an artefact of the vendor's archive), a start outside
    09:15-15:30 IST. A timestamp that appears twice keeps its last row (an archive copy glued in front of the live one)."""
    candles = ((payload.get("data") or {}).get("candles") if isinstance(payload, Mapping) and isinstance(payload.get("data"), Mapping)
               else None)
    if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
        return []
    found: dict[datetime, dict[str, Any]] = {}
    for row in candles:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 5:
            continue
        stamp = _bar_stamp(row[0])
        o, h, low, c = (_positive(row[i]) for i in (1, 2, 3, 4))
        volume = _num(row[5]) if len(row) > 5 else None
        if stamp is None or None in (o, h, low, c) or (volume is not None and volume < 0):
            continue
        local = stamp.astimezone(IST)
        minute = local.hour * 60 + local.minute
        if not FIRST_BAR_MIN <= minute < LAST_BAR_END_MIN:
            continue
        slack = 1.0 + PRICE_SLACK
        if h * slack < max(o, c, low) or low > min(o, c, h) * slack:
            continue
        found[stamp] = {"ts": iso(stamp), "trade_date": local.date().isoformat(), "hhmm": f"{local.hour:02d}:{local.minute:02d}",
                        "open": o, "high": h, "low": low, "close": c, "volume": volume}
    return [found[stamp] for stamp in sorted(found)]


def newest_trade(rows: Iterable[Mapping[str, Any]], received: datetime) -> datetime | None:
    """When the freshest of ``rows`` last traded, as of an answer that arrived at ``received``. A stamp that lies in the future by more than
    the allowed clock skew is not believed (it would make a feed of stale quotes look current: the age check is the guard against a lagging
    feed) and a stamp within the skew is capped at ``received``; with no believable stamp the age cannot be judged (``None``)."""
    ceiling = received + timedelta(seconds=FUTURE_SLACK_SEC)
    stamps = [stamp for stamp in (parse_ts(row.get("row_ts")) for row in rows) if stamp is not None and stamp <= ceiling]
    return min(max(stamps), received) if stamps else None


# -------------------------------------------------------------------------------------------------- results
@dataclass
class KotakQuotes:
    """One pass over many stocks: the rows that answered, the newest update among them and when the answer arrived."""

    rows: list[dict[str, Any]]
    received: datetime
    asked: int
    asof: datetime | None = None
    failed: list[str] = field(default_factory=list)

    @property
    def lag_sec(self) -> float | None:
        """How old the freshest trade of the answer was when it arrived. A stock that is not trading is not stale, but the freshest of
        about two hundred liquid stocks is: this is the age of the FEED."""
        return None if self.asof is None else (self.received - self.asof).total_seconds()


class KotakApi:
    """The broker's read-only market-data endpoints behind the shared HTTP rails (spacing, bounded retries, circuit breaker, deadlines)."""

    def __init__(self, settings: Settings, *, http: HttpClient | None = None, history_http: HttpClient | None = None,
                 clock: Clock | None = None) -> None:
        key = settings.kotak_consumer_key
        if not key:
            raise DataUnavailable("kotak", "KOTAK_CONSUMER_KEY is not set", retryable=False)
        if not is_kotak_url(settings.kotak_base_url):
            raise DataUnavailable("kotak", "KOTAK_BASE_URL is not an https host on kotaksecurities.com: the key is not sent", retryable=False)
        self._key = key
        self.base = settings.kotak_base_url.strip().rstrip("/")
        self.clock = clock
        # plain ``requests``: the API needs no browser fingerprint, and it keeps the connection alive (about 0.1 s a request after the
        # first) where curl_cffi's streamed responses close it every time and pay the host's 2.3 s handshake again (measured)
        common: dict[str, Any] = {"headers": {"Accept": "application/json"}, "transport": "requests", "clock": clock,
                                  "max_attempts": min(settings.http_max_attempts, 3)}
        self.http = http or HttpClient.from_settings("kotak_quotes", settings, min_interval=settings.kotak_quote_interval_sec, **common)
        self.history_http = history_http or HttpClient.from_settings(
            "kotak_history", settings, min_interval=settings.kotak_history_interval_sec, **common)

    def close(self) -> None:
        self.http.close()
        self.history_http.close()

    def _now(self) -> datetime:
        return (self.clock or get_clock()).now()

    # ----------------------------------------------------------------------------------------------- plumbing
    def _call(self, client: HttpClient, path: str, *, params: Mapping[str, str] | None = None, deadline: float | None = None) -> Any:
        """The JSON answer of one authenticated GET. HTTP 400 is raised as ``_BadRequest`` (with the broker's own words), a refused
        token as a ``DataUnavailable`` that says what to do, every other failure as the client's ``DataUnavailable``."""
        headers = {"Authorization": self._key, "Content-Type": "application/x-www-form-urlencoded"}
        try:
            response = client.get(f"{self.base}/{path}", params=params, headers=headers, soft_statuses=(400,), deadline=deadline)
        except DataUnavailable as exc:
            text = str(exc)
            for code in ("401", "403"):
                if f"HTTP {code}" in text:
                    raise DataUnavailable(client.source, AUTH_HINT.format(code=code), retryable=False) from None
            raise
        body = response.json()
        if response.status == 400:
            raise _BadRequest(self._fault(body) or "HTTP 400")
        fault = self._fault(body) if isinstance(body, Mapping) else ""
        if fault:
            raise DataUnavailable(client.source, f"the broker answered with an error: {fault}", retryable=False)
        return body

    @staticmethod
    def _fault(body: Any) -> str:
        """The broker's own reason of a failed answer, whichever of its error shapes it used (``{"fault": {...}}`` with HTTP 400,
        ``{"status": "ERROR", ...}``, ``{"error": ...}``). A successful answer also carries a ``message`` ("Data fetched
        successfully"): that is not a fault."""
        if not isinstance(body, Mapping):
            return ""
        if isinstance(body.get("fault"), Mapping):
            source: Mapping[str, Any] = body["fault"]
        elif any(key in body for key in ("error", "Error", "emsg")) or str(body.get("status") or body.get("stat") or "").lower() in {
                "error", "failure", "failed", "not_ok"}:
            source = body
        else:
            return ""
        text = (source.get("description") or source.get("message") or source.get("error") or source.get("Error")
                or source.get("emsg") or "error")
        return redact(" ".join(str(text).split()))[:160]

    # ----------------------------------------------------------------------------------------- instrument master
    def token_map(self, *, deadline: float | None = None) -> dict[str, str]:
        """``{symbol: token}`` of every NSE equity, from today's instrument master (a file of about 3 MB: fetch it once a day)."""
        listing = self._call(self.http, FILE_PATHS, deadline=deadline)
        data = listing.get("data") if isinstance(listing, Mapping) and isinstance(listing.get("data"), Mapping) else listing
        paths = [str(p) for p in (data.get("filesPaths") or [])] if isinstance(data, Mapping) else []
        url = next((p for p in paths if p.lower().endswith("nse_cm-v1.csv")), None) or next(
            (p for p in paths if "/nse_cm" in p.lower() and "nse_com" not in p.lower()), None)
        if url is None:
            raise DataUnavailable("kotak_quotes", "the instrument master list has no nse_cm file", retryable=True)
        parts = urlsplit(url)
        if not is_kotak_url(f"{parts.scheme}://{parts.netloc}"):
            raise DataUnavailable("kotak_quotes", "the instrument master is on a host outside kotaksecurities.com: not fetched", retryable=False)
        response: HttpResponse = self.http.get(url, headers={"Accept": "text/csv,*/*"}, timeout=(10.0, 60.0), deadline=deadline)
        tokens = parse_token_map(response.content.decode("utf-8-sig", errors="replace"))
        if len(tokens) < MIN_STOCKS:
            raise DataUnavailable("kotak_quotes", f"the instrument master holds only {len(tokens)} stocks: not the real file", retryable=True)
        return tokens

    # ---------------------------------------------------------------------------------------------- live quotes
    def quotes(self, tokens: Mapping[str, str], *, deadline: float | None = None) -> KotakQuotes:
        """Live rows of the stocks in ``tokens`` (symbol -> token), ``QUOTE_BATCH`` per request. A batch the broker refuses is halved
        until the one stock it cannot quote is found and left out (it is in ``failed``); a batch that cannot be read at all (timeout,
        server error) puts its stocks into ``failed`` too. Raises only when nothing at all could be read."""
        by_token = {token: symbol for symbol, token in tokens.items()}
        pairs = list(tokens.items())
        rows: list[dict[str, Any]] = []
        failed: list[str] = []
        last_error: DataUnavailable | None = None
        for start in range(0, len(pairs), QUOTE_BATCH):
            batch = pairs[start:start + QUOTE_BATCH]
            try:
                got = self._quote_batch(batch, by_token, failed, deadline)
            except DataUnavailable as exc:
                last_error = exc
                failed += [symbol for symbol, _ in batch]
                if "refused the application token" in str(exc) or not exc.retryable:
                    failed += [symbol for symbol, _ in pairs[start + QUOTE_BATCH:]]  # the rest would only be refused again: not asked
                    break
                continue
            rows += got
            answered = {row["symbol"] for row in got}
            failed += [symbol for symbol, _ in batch if symbol not in answered]  # the API silently omits a token it does not know
        if not rows and pairs:
            raise last_error or DataUnavailable("kotak_quotes", "no stock answered", retryable=True)
        received = self._now()
        asof = newest_trade(rows, received)
        log.info("kotak quotes: %d of %d stocks, newest update %s%s", len(rows), len(pairs), iso(asof) or "-",
                 f", {len(set(failed))} not read" if failed else "")
        return KotakQuotes(rows, received, len(pairs), asof, sorted(set(failed)))

    def _quote_batch(self, batch: list[tuple[str, str]], by_token: Mapping[str, str], failed: list[str],
                     deadline: float | None) -> list[dict[str, Any]]:
        path = QUOTES.format(pairs=quote(",".join(f"{SEGMENT}|{token}" for _, token in batch), safe="|,"))
        try:
            data = self._call(self.http, path, deadline=deadline)
        except _BadRequest as exc:
            if len(batch) == 1:
                log.warning("kotak: no quote for %s (%s)", batch[0][0], exc)
                failed.append(batch[0][0])
                return []
            middle = len(batch) // 2
            return (self._quote_batch(batch[:middle], by_token, failed, deadline)
                    + self._quote_batch(batch[middle:], by_token, failed, deadline))
        if not isinstance(data, list):
            raise DataUnavailable("kotak_quotes", "unexpected quote answer (not a list)", retryable=True)
        return parse_quotes(data, by_token)

    # ------------------------------------------------------------------------------------------------- candles
    def indices(self, *, deadline: float | None = None) -> dict[str, dict[str, Any]]:
        """Nifty 50 and India VIX live (see ``parse_indices``): the fallback when NSE's index call does not answer."""
        path = QUOTES.format(pairs=quote(",".join(f"{SEGMENT}|{name}" for name in INDEX_QUOTES), safe="|,"))
        try:
            return parse_indices(self._call(self.http, path, deadline=deadline))
        except _BadRequest as exc:
            raise DataUnavailable("kotak_quotes", f"the broker refused the index quote: {exc}", retryable=False) from None

    def candles(self, token: str, interval: str, start: date, end: date, *, deadline: float | None = None) -> list[dict[str, Any]]:
        """The bars of one stock from ``start`` to ``end`` (inclusive, IST dates), oldest first (see ``parse_candles``). One request
        covers at most ``MAX_SPAN_DAYS`` days; a window the broker refuses raises ``DataUnavailable``."""
        if interval not in INTERVALS:
            raise ValueError(f"unsupported candle interval {interval!r}")
        if end < start or (end - start).days > MAX_SPAN_DAYS - 1:
            raise ValueError(f"a candle window is 1 to {MAX_SPAN_DAYS} days, got {start} .. {end}")
        params = {"neosymbol": f"{SEGMENT}|{token}", "interval": interval, "fromdate": start.isoformat(), "todate": end.isoformat()}
        try:
            payload = self._call(self.history_http, HISTORY, params=params, deadline=deadline)
        except _BadRequest as exc:
            raise DataUnavailable("kotak_history", f"the broker refused the window: {exc}", retryable=False) from None
        return parse_candles(payload)
