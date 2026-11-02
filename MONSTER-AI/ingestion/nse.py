"""NSE data access: public archive files (no cookies) and the public JSON API (cookie handshake).

Both are undocumented/public and may change; every parser is defensive (``.get`` chains, finite-number coercion)
and returns "missing" rather than guessing. ``monster-ai doctor`` live-checks them on the server.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.clock import IST, Clock, iso, parse_ts
from app.config import Settings
from app.errors import DataUnavailable
from ingestion.http import USER_AGENT_BROWSER, HttpClient

log = logging.getLogger(__name__)

ARCHIVE = "https://nsearchives.nseindia.com"
HOME = "https://www.nseindia.com/"
API = "https://www.nseindia.com/api/"
UDIFF_START = date(2024, 7, 8)
MAX_ZIP_UNCOMPRESSED = 120 * 1024 * 1024
# The stock lists the live-quote endpoint serves for each universe policy: ``SECURITIES IN F&O`` is the whole F&O segment in one
# request; the broad index lists cover the larger universes.
LIVE_LISTS = {"fno_nifty100": ("SECURITIES IN F&O",), "nifty200": ("SECURITIES IN F&O", "NIFTY 200"),
              "nifty500": ("SECURITIES IN F&O", "NIFTY 500"), "custom": ("SECURITIES IN F&O", "NIFTY 500")}
# NSE retired ``equity-stockIndices`` in 2026. An index list is now served by ``equity-stock-indices``, the F&O segment (and the
# "permitted to trade" list) by ``equity-stockIndex``. The retired name stays last in line: it costs one 404 when it is gone and
# keeps working where it is still served. The endpoint that answered is remembered for the next call.
LIVE_INDEX_ENDPOINTS = ("equity-stock-indices", "equity-stockIndices")
LIVE_SEGMENT_ENDPOINTS = ("equity-stockIndex", *LIVE_INDEX_ENDPOINTS)
LIVE_SEGMENT_LISTS = frozenset({"SECURITIES IN F&O", "PERMITTED TO TRADE"})

BROWSER_HEADERS = {
    "User-Agent": USER_AGENT_BROWSER,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}
API_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": HOME,
    "X-Requested-With": "XMLHttpRequest",
    # what a browser sends for an XHR/fetch call from the NSE site itself (the impersonation profile defaults to a page load)
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}
INDEX_FILES = {
    "nifty50": "ind_nifty50list.csv", "nifty100": "ind_nifty100list.csv", "nifty200": "ind_nifty200list.csv",
    "nifty500": "ind_nifty500list.csv",
}


def num(value: Any) -> float | None:
    """Finite float or ``None`` (blank, '-', NaN, garbage)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if text in {"", "-", "--", "NA", "N/A", "null", "None"}:
            return None
        try:
            out = float(text)
        except ValueError:
            return None
    return out if out == out and abs(out) != float("inf") else None


def _parse_nse_date(text: str) -> date | None:
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")  # pragma: no cover


def _csv_rows(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), skipinitialspace=True)
    rows = []
    for row in reader:
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None}
        if any(clean.values()):
            rows.append(clean)
    return rows


# ---------------------------------------------------------------------------------------- archive files
class NseArchive:
    """Official end-of-day files from nsearchives.nseindia.com."""

    def __init__(self, settings: Settings, *, http: HttpClient | None = None, clock: Clock | None = None) -> None:
        self.http = http or HttpClient.from_settings(
            "nse_archive", settings, headers=BROWSER_HEADERS, min_interval=settings.nse_min_interval_sec,
            timeout=(10.0, max(settings.http_timeout_sec, 60.0)), clock=clock,
        )

    def bhavcopy(self, day: date) -> list[dict[str, Any]] | None:
        """Official cash-market bhavcopy with delivery data for ``day``; ``None`` when not (yet) published."""
        url = f"{ARCHIVE}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv"
        response = self.http.get(url, soft_statuses=(403, 404))
        if response.status != 200 or not response.content.strip():
            return None
        text = _decode(response.content)
        if text.lstrip().startswith("<"):  # HTML error page served with 200
            return None
        out: list[dict[str, Any]] = []
        for row in _csv_rows(text):
            parsed_date = _parse_nse_date(row.get("DATE1", ""))
            symbol = row.get("SYMBOL", "")
            if not symbol or parsed_date is None:
                continue
            if parsed_date != day:
                log.warning("bhavcopy for %s contains rows dated %s; treating file as unavailable", day, parsed_date)
                return None
            turnover_lacs = num(row.get("TURNOVER_LACS"))
            out.append(
                {
                    "symbol": symbol, "series": row.get("SERIES", ""), "trade_date": day.isoformat(),
                    "prev_close": num(row.get("PREV_CLOSE")), "open": num(row.get("OPEN_PRICE")),
                    "high": num(row.get("HIGH_PRICE")), "low": num(row.get("LOW_PRICE")),
                    "close": num(row.get("CLOSE_PRICE")), "last": num(row.get("LAST_PRICE")),
                    "vwap": num(row.get("AVG_PRICE")), "volume": num(row.get("TTL_TRD_QNTY")),
                    "turnover": turnover_lacs * 100000.0 if turnover_lacs is not None else None,
                    "trades": num(row.get("NO_OF_TRADES")), "delivery_qty": num(row.get("DELIV_QTY")),
                    "delivery_pct": num(row.get("DELIV_PER")),
                }
            )
        return out or None

    def fo_bhavcopy(self, day: date) -> list[dict[str, Any]] | None:
        """UDiFF F&O bhavcopy rows for ``day`` (``None`` if unpublished or before the UDiFF format existed)."""
        if day < UDIFF_START:
            return None
        url = f"{ARCHIVE}/content/fo/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
        response = self.http.get(url, soft_statuses=(403, 404))
        if response.status != 200 or len(response.content) < 100:
            return None
        return parse_fo_zip(response.content, day)

    def index_constituents(self, index: str = "nifty500") -> list[dict[str, str]]:
        name = INDEX_FILES.get(index)
        if name is None:
            raise ValueError(f"unknown index list {index!r}")
        response = self.http.get(f"{ARCHIVE}/content/indices/{name}", soft_statuses=(403, 404))
        if response.status != 200:
            raise DataUnavailable("nse_lists", f"{name} unavailable (HTTP {response.status})")
        out = []
        for row in _csv_rows(_decode(response.content)):
            symbol = row.get("Symbol", "")
            if not symbol:
                continue
            out.append({"symbol": symbol, "name": row.get("Company Name", ""), "industry": row.get("Industry", ""),
                        "isin": row.get("ISIN Code", ""), "series": row.get("Series", "EQ")})
        if not out:
            raise DataUnavailable("nse_lists", f"{name} parsed to zero rows")
        return out

    def fo_lot_sizes(self) -> dict[str, int]:
        """Lot size per F&O underlying symbol (current first expiry column)."""
        response = self.http.get(f"{ARCHIVE}/content/fo/fo_mktlots.csv", soft_statuses=(403, 404))
        if response.status != 200:
            raise DataUnavailable("nse_lists", f"fo_mktlots unavailable (HTTP {response.status})")
        out: dict[str, int] = {}
        for row in _csv_rows(_decode(response.content)):
            symbol = row.get("SYMBOL", "")
            if not symbol:
                continue
            lot = None
            for key, value in row.items():
                if key in {"UNDERLYING", "SYMBOL"}:
                    continue
                lot = num(value)
                if lot:
                    break
            if lot:
                out[symbol] = int(lot)
        return out


def parse_fo_zip(content: bytes, day: date) -> list[dict[str, Any]] | None:
    """Parse a UDiFF F&O zip safely (member count and uncompressed size are capped against zip bombs)."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return None
    members = archive.infolist()
    if not members or len(members) > 3 or sum(m.file_size for m in members) > MAX_ZIP_UNCOMPRESSED:
        log.warning("F&O archive rejected (members=%d)", len(members))
        return None
    member = next((m for m in members if m.filename.lower().endswith(".csv")), None)
    if member is None:
        return None
    text = io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="")
    out: list[dict[str, Any]] = []
    for row in csv.DictReader(text, skipinitialspace=True):
        instrument = (row.get("FinInstrmTp") or "").strip()
        if instrument not in {"STF", "STO", "IDF", "IDO"}:
            continue
        traded = _parse_nse_date((row.get("TradDt") or "").strip())
        if traded is not None and traded != day:
            return None
        expiry = _parse_nse_date((row.get("XpryDt") or "").strip())
        if expiry is None:
            continue
        option_type = (row.get("OptnTp") or "").strip() or None
        out.append(
            {
                "instrument": instrument, "symbol": (row.get("TckrSymb") or "").strip(), "expiry": expiry.isoformat(),
                "strike": num(row.get("StrkPric")), "option_type": option_type, "open": num(row.get("OpnPric")),
                "high": num(row.get("HghPric")), "low": num(row.get("LwPric")), "close": num(row.get("ClsPric")),
                "settle": num(row.get("SttlmPric")), "underlying": num(row.get("UndrlygPric")),
                "oi": num(row.get("OpnIntrst")), "oi_chg": num(row.get("ChngInOpnIntrst")),
                "volume": num(row.get("TtlTradgVol")), "lot_size": num(row.get("NewBrdLotQty")),
            }
        )
    return out or None


# ---------------------------------------------------------------------------------------------- live API
@dataclass
class PreopenSnapshot:
    source_ts: str | None
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LiveEquities:
    """One live stock-list answer: today's running figures of every stock of a list and the list's own index row (if it has one)."""

    source_ts: str | None
    rows: list[dict[str, Any]] = field(default_factory=list)
    index: dict[str, Any] | None = None


class NseApi:
    """Public JSON endpoints behind www.nseindia.com. A cookie handshake is required; on 401/403 the session is
    reset and re-warmed automatically (bounded by the HTTP client's attempts and circuit breaker)."""

    def __init__(self, settings: Settings, *, http: HttpClient | None = None, clock: Clock | None = None) -> None:
        self.clock = clock
        self._warm_mark: float | None = None
        self._live_endpoint: dict[str, str] = {}  # list name -> the live-quote endpoint that answered for it
        self.http = http or HttpClient.from_settings(
            "nse_api", settings, headers=BROWSER_HEADERS, min_interval=settings.nse_min_interval_sec,
            max_attempts=min(settings.http_max_attempts, 3), clock=clock,
        )
        self.http.on_auth_failure = self._rewarm
        self._deadline: float | None = None

    def _now_mono(self) -> float:
        from app.clock import get_clock

        return (self.clock or get_clock()).monotonic()

    def _warm(self, deadline: float | None = None) -> None:
        self.http.reset_session()
        try:
            self.http.get(HOME, headers={"Accept": "text/html,application/xhtml+xml,*/*"}, soft_statuses=(401, 403, 404, 429),
                          deadline=deadline)
        except DataUnavailable:
            log.debug("NSE warm-up failed", exc_info=True)
        self._warm_mark = self._now_mono()

    def _rewarm(self) -> None:
        self._warm(self._deadline)

    def _get(self, path: str, params: Mapping[str, str] | None = None, *, timeout: tuple[float, float] | None = None,
             deadline: float | None = None, soft_statuses: tuple[int, ...] = ()) -> Any:
        """The JSON answer of one endpoint; ``None`` when it answers with one of ``soft_statuses`` (a 404 the caller expects)."""
        self._deadline = deadline  # also bounds the re-handshake the HTTP client triggers on a 401 / 403
        try:
            if self._warm_mark is None or self._now_mono() - self._warm_mark > 20 * 60:
                self._warm(deadline)
            response = self.http.get(API + path, params=params, headers=API_HEADERS, timeout=timeout, deadline=deadline,
                                     soft_statuses=soft_statuses)
        finally:
            self._deadline = None
        return response.json() if response.ok else None

    # --------------------------------------------------------------------------------------- endpoints
    def market_status(self, *, deadline: float | None = None) -> Mapping[str, Any]:
        data = self._get("marketStatus", deadline=deadline)
        if not isinstance(data, dict):
            raise DataUnavailable("nse_api", "unexpected marketStatus shape")
        return data

    def gift_nifty(self, *, deadline: float | None = None) -> dict[str, Any] | None:
        """GIFT Nifty last price, change and exchange timestamp (``None`` if the block is absent)."""
        return parse_gift_nifty(self.market_status(deadline=deadline))

    def preopen(self, key: str = "ALL", symbols: Iterable[str] | None = None, *, deadline: float | None = None) -> PreopenSnapshot:
        """The pre-open book. ``deadline`` (``clock.monotonic()`` instant) bounds every attempt, so a slow NSE can never push
        the final scan past its cutoff."""
        data = self._get("market-data-pre-open", {"key": key}, timeout=(10.0, 60.0), deadline=deadline)
        return parse_preopen(data, set(symbols) if symbols is not None else None)

    def announcements(self, start: date, end: date, symbols: set[str] | None = None) -> list[dict[str, Any]]:
        params = {"index": "equities", "from_date": f"{start:%d-%m-%Y}", "to_date": f"{end:%d-%m-%Y}"}
        data = self._get("corporate-announcements", params, timeout=(10.0, 120.0))
        return parse_announcements(data, symbols)

    def live_equities(self, index: str = "SECURITIES IN F&O", symbols: Iterable[str] | None = None, *,
                      deadline: float | None = None) -> LiveEquities:
        """Today's running open / high / low / last price / volume of every stock of ``index`` in ONE call. ``deadline`` (a
        ``clock.monotonic()`` instant) bounds every attempt, so a slow NSE cannot push the closing scan past its report deadline.
        The endpoints are tried in order (see ``LIVE_SEGMENT_ENDPOINTS``): only a 404 moves on to the next, any other failure is
        raised at once; a list that no endpoint knows is an error that says so."""
        name = index.strip().upper()
        endpoints = LIVE_SEGMENT_ENDPOINTS if name in LIVE_SEGMENT_LISTS else LIVE_INDEX_ENDPOINTS
        known = self._live_endpoint.get(name)
        order = [known, *(e for e in endpoints if e != known)] if known else list(endpoints)
        for endpoint in order:
            data = self._get(endpoint, {"index": name}, timeout=(10.0, 60.0), deadline=deadline, soft_statuses=(404,))
            if data is not None:
                self._live_endpoint[name] = endpoint
                return parse_live_equities(data, set(symbols) if symbols is not None else None)
        raise DataUnavailable("nse_api", f"no live-quote endpoint knows {name} (HTTP 404 from {', '.join(order)}): did NSE change its API?",
                              retryable=False)

    def all_indices(self, *, deadline: float | None = None) -> dict[str, dict[str, Any]]:
        """Every NSE index live (Nifty 50, India VIX and the sector indices) keyed by its name, in one call."""
        return parse_all_indices(self._get("allIndices", timeout=(10.0, 60.0), deadline=deadline))

    def event_calendar(self, start: date, end: date, symbols: set[str] | None = None) -> list[dict[str, Any]]:
        params = {"index": "equities", "from_date": f"{start:%d-%m-%Y}", "to_date": f"{end:%d-%m-%Y}"}
        data = self._get("event-calendar", params, timeout=(10.0, 120.0))
        return parse_event_calendar(data, symbols)

    def holidays(self) -> list[dict[str, Any]]:
        return parse_holidays(self._get("holiday-master", {"type": "trading"}))


# ------------------------------------------------------------------------------------------- pure parsers
def parse_gift_nifty(status: Mapping[str, Any]) -> dict[str, Any] | None:
    block = status.get("giftnifty") if isinstance(status, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    price = num(block.get("LASTPRICE"))
    if price is None:
        return None
    change = num(block.get("DAYCHANGE"))
    stamp = parse_ts(block.get("TIMESTMP"), assume_tz=IST)
    return {
        "price": price, "change": change, "prev_close": price - change if change is not None else None,
        "change_pct": num(block.get("PERCHANGE")), "source_ts": iso(stamp) if stamp else None,
        "expiry": block.get("EXPIRYDATE"),
    }


def parse_preopen(data: Any, wanted: set[str] | None = None) -> PreopenSnapshot:
    """Normalise ``market-data-pre-open``. Fields come from ``metadata`` and ``detail.preOpenMarket``; anything absent
    stays ``None`` (e.g. ``final_price`` before the equilibrium price is published)."""
    if not isinstance(data, Mapping):
        return PreopenSnapshot(None, [])
    top_ts = parse_ts(data.get("timestamp"), assume_tz=IST)
    rows: list[dict[str, Any]] = []
    latest_ts = top_ts
    for item in data.get("data") or []:
        if not isinstance(item, Mapping):
            continue
        meta = item.get("metadata") or {}
        detail = ((item.get("detail") or {}).get("preOpenMarket")) or {}
        symbol = str(meta.get("symbol") or "").strip()
        if not symbol or (wanted is not None and symbol not in wanted):
            continue
        row_ts = parse_ts(detail.get("lastUpdateTime"), assume_tz=IST) or top_ts
        if row_ts and (latest_ts is None or row_ts > latest_ts):
            latest_ts = row_ts
        rows.append(
            {
                "symbol": symbol, "prev_close": num(meta.get("previousClose")),
                "iep": num(detail.get("IEP")) or num(meta.get("iep")) or num(meta.get("lastPrice")),
                "final_price": num(detail.get("finalPrice")) or num(meta.get("finalPrice")),
                "final_qty": num(detail.get("finalQuantity")) or num(meta.get("finalQuantity")),
                "buy_qty": num(detail.get("totalBuyQuantity")), "sell_qty": num(detail.get("totalSellQuantity")),
                "pchange": num(meta.get("pChange")), "row_ts": iso(row_ts) if row_ts else None,
            }
        )
    return PreopenSnapshot(iso(latest_ts) if latest_ts else None, rows)


def _positive(value: Any) -> float | None:
    """A finite number above zero, or ``None``: a live open of 0 is the state before the session opens, not a price."""
    out = num(value)
    return out if out is not None and out > 0 else None


def _change_pct(last: float | None, prev: float | None) -> float | None:
    return (last / prev - 1.0) * 100.0 if last is not None and prev is not None else None


def parse_live_equities(data: Any, wanted: set[str] | None = None) -> LiveEquities:
    """Normalise the stock lists of NSE's live market page (``equity-stockIndex`` for the F&O segment, ``equity-stock-indices`` for
    an index list). The answer's own ``timestamp`` is the stamp of every row: the per-row ``lastUpdateTime`` is a placeholder
    (``31-DEC-2999``) or the date of some old update for most stocks, never the time of a trade, and is not read. An index list
    starts with the index itself, a row without a ``symbol`` whose ``identifier`` is the index name: it is kept apart, never mixed
    with the stocks (NSE stock symbols never contain a space). A stock row needs a last price and a previous close; every other
    figure that is absent stays ``None``. NSE writes the prices of the F&O list as text, hence the defensive number reading."""
    if not isinstance(data, Mapping):
        return LiveEquities(None, [])
    stamp = parse_ts(data.get("timestamp"), assume_tz=IST)
    stamp_text = iso(stamp) if stamp else None
    metadata = data.get("metadata")
    list_name = str(data.get("name") or (metadata.get("indexName") if isinstance(metadata, Mapping) else "") or "").strip().upper()
    rows: list[dict[str, Any]] = []
    index_row: dict[str, Any] | None = None
    for item in data.get("data") or []:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol") or "").strip()
        last, prev = _positive(item.get("lastPrice")), _positive(item.get("previousClose"))
        if not symbol:
            identifier = str(item.get("identifier") or "").strip()
            if " " in identifier and index_row is None and last is not None:
                index_row = {"name": identifier, "last": last, "prev_close": prev, "change_pct": _change_pct(last, prev),
                             "row_ts": stamp_text}
            continue
        if " " in symbol or symbol.upper() == list_name:
            if index_row is None and last is not None:
                index_row = {"name": symbol, "last": last, "prev_close": prev, "change_pct": _change_pct(last, prev),
                             "row_ts": stamp_text}
            continue
        if last is None or prev is None or (wanted is not None and symbol not in wanted):
            continue
        rows.append(
            {
                "symbol": symbol, "open": _positive(item.get("open")), "high": _positive(item.get("dayHigh")),
                "low": _positive(item.get("dayLow")), "last": last, "prev_close": prev, "change_pct": _change_pct(last, prev),
                "volume": num(item.get("totalTradedVolume")), "turnover": num(item.get("totalTradedValue")),
                "year_high": _positive(item.get("yearHigh")), "year_low": _positive(item.get("yearLow")),
                "row_ts": stamp_text,
            }
        )
    return LiveEquities(stamp_text, rows, index_row)


def parse_all_indices(data: Any) -> dict[str, dict[str, Any]]:
    """Normalise ``allIndices`` to ``{"NIFTY 50": {...}, "INDIA VIX": {...}, ...}`` (names upper-cased); rows without a last
    value are dropped."""
    if not isinstance(data, Mapping):
        return {}
    stamp = parse_ts(data.get("timestamp"), assume_tz=IST)
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("data") or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("index") or item.get("indexSymbol") or "").strip()
        last, prev = _positive(item.get("last")), _positive(item.get("previousClose"))
        if not name or last is None:
            continue
        out[name.upper()] = {
            "name": name, "last": last, "prev_close": prev, "open": _positive(item.get("open")),
            "high": _positive(item.get("high")), "low": _positive(item.get("low")),
            "change_pct": _change_pct(last, prev), "row_ts": iso(stamp) if stamp else None,
        }
    return out


def parse_announcements(data: Any, symbols: set[str] | None = None) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if not symbol or (symbols is not None and symbol not in symbols):
            continue
        stamp = parse_ts(item.get("exchdisstime"), assume_tz=IST) or parse_ts(item.get("an_dt"), assume_tz=IST)
        if stamp is None:
            continue
        text = str(item.get("attchmntText") or "").strip()
        attachment = str(item.get("attchmntFile") or "").strip() or None
        ref = str(item.get("seq_id") or "").strip() or hashlib.sha1(  # noqa: S324 - identity key, not security
            f"{symbol}|{item.get('an_dt')}|{attachment}|{text[:80]}".encode(), usedforsecurity=False
        ).hexdigest()
        out.append(
            {
                "exchange": "NSE", "symbol": symbol, "isin": item.get("sm_isin"), "company": item.get("sm_name"),
                "category": str(item.get("desc") or "").strip(), "subject": text[:200], "text": text[:2000],
                "attachment_url": attachment if attachment and attachment.lower().startswith("https://") else None,
                "disseminated_at": iso(stamp), "source_ref": ref,
            }
        )
    return out


def parse_event_calendar(data: Any, symbols: set[str] | None = None) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol") or "").strip()
        meeting = _parse_nse_date(str(item.get("date") or ""))
        if not symbol or meeting is None or (symbols is not None and symbol not in symbols):
            continue
        purpose = str(item.get("purpose") or "").strip()
        description = str(item.get("bm_desc") or "").strip()
        out.append(
            {
                "symbol": symbol, "meeting_date": meeting.isoformat(), "purpose": purpose[:120],
                "description": description[:400],
                "is_results": "result" in purpose.lower() or "result" in description.lower()[:200],
            }
        )
    return out


def parse_holidays(data: Any) -> list[dict[str, Any]]:
    block = data.get("CM") if isinstance(data, Mapping) else None
    out = []
    for item in block or []:
        if not isinstance(item, Mapping):
            continue
        day = _parse_nse_date(str(item.get("tradingDate") or ""))
        if day is None:
            continue
        description = str(item.get("description") or "").strip()
        out.append({"trade_date": day.isoformat(), "description": description, "special": description.endswith("*")})
    return out
