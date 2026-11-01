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
from datetime import date, datetime, timezone
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


class NseApi:
    """Public JSON endpoints behind www.nseindia.com. A cookie handshake is required; on 401/403 the session is
    reset and re-warmed automatically (bounded by the HTTP client's attempts and circuit breaker)."""

    def __init__(self, settings: Settings, *, http: HttpClient | None = None, clock: Clock | None = None) -> None:
        self.clock = clock
        self._warm_mark: float | None = None
        self.http = http or HttpClient.from_settings(
            "nse_api", settings, headers=BROWSER_HEADERS, min_interval=settings.nse_min_interval_sec,
            max_attempts=min(settings.http_max_attempts, 3), clock=clock,
        )
        self.http.on_auth_failure = self._rewarm
        self._deadline: float | None = None
        self._expiry_cache: dict[str, list[str]] = {}

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
             deadline: float | None = None) -> Any:
        self._deadline = deadline  # also bounds the re-handshake the HTTP client triggers on a 401 / 403
        try:
            if self._warm_mark is None or self._now_mono() - self._warm_mark > 20 * 60:
                self._warm(deadline)
            response = self.http.get(API + path, params=params, headers=API_HEADERS, timeout=timeout, deadline=deadline)
        finally:
            self._deadline = None
        return response.json()

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

    def option_expiries(self, symbol: str) -> list[str]:
        if symbol not in self._expiry_cache:
            info = self._get("option-chain-contract-info", {"symbol": symbol})
            dates = info.get("expiryDates") if isinstance(info, dict) else None
            self._expiry_cache[symbol] = [d for d in (dates or []) if isinstance(d, str)]
        return self._expiry_cache[symbol]

    def option_chain_summary(self, symbol: str) -> dict[str, Any] | None:
        expiries = self.option_expiries(symbol)
        if not expiries:
            return None
        chain = self._get("option-chain-v3", {"type": "Equity", "symbol": symbol, "expiry": expiries[0]}, timeout=(10.0, 60.0))
        from ingestion.fno import summarize_option_chain

        return summarize_option_chain(chain, expiries[0])

    def announcements(self, start: date, end: date, symbols: set[str] | None = None) -> list[dict[str, Any]]:
        params = {"index": "equities", "from_date": f"{start:%d-%m-%Y}", "to_date": f"{end:%d-%m-%Y}"}
        data = self._get("corporate-announcements", params, timeout=(10.0, 120.0))
        return parse_announcements(data, symbols)

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


def utc_now_iso(clock: Clock | None = None) -> str:
    from app.clock import get_clock

    return iso((clock or get_clock()).now()) or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
