"""Registry of global / India market series and their point-in-time availability.

A daily bar dated ``d`` (in the exchange's local calendar) is only usable once that market has closed plus a
publication lag. Daylight-saving shifts are handled with ``zoneinfo``, so the same rule reproduces exactly what
was knowable at any historical cutoff (training) and at the live cutoff (inference): no train/serve skew.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache

from app.clock import local_close_utc


@dataclass(frozen=True)
class SeriesSpec:
    key: str  # canonical symbol stored in instruments.symbol
    yahoo: str
    name: str
    kind: str  # instruments.kind: index | global | fx | commodity | rate | volatility
    category: str  # us_equity | eu_equity | asia_equity | india_index | fx | rates | commodity | volatility
    tz: str  # tz in which Yahoo dates the bar
    close_hhmm: str  # local time after which the bar is final
    lag_min: int = 30  # provider publication delay after the close
    critical: bool = False  # missing/stale -> degrade the whole report

    def available_at(self, bar_date: date) -> datetime:
        return _available_at(self.tz, self.close_hhmm, self.lag_min, bar_date)


@lru_cache(maxsize=200_000)
def _available_at(tz: str, close_hhmm: str, lag_min: int, bar_date: date) -> datetime:
    return local_close_utc(bar_date, tz, close_hhmm) + timedelta(minutes=lag_min)


NY, LON, BER, PAR, TYO, HKG, SHA, SEL, TPE, SYD, KOL = (
    "America/New_York", "Europe/London", "Europe/Berlin", "Europe/Paris", "Asia/Tokyo", "Asia/Hong_Kong",
    "Asia/Shanghai", "Asia/Seoul", "Asia/Taipei", "Australia/Sydney", "Asia/Kolkata",
)

SERIES: tuple[SeriesSpec, ...] = (
    # United States
    SeriesSpec("^GSPC", "^GSPC", "S&P 500", "global", "us_equity", NY, "16:00", critical=True),
    SeriesSpec("^IXIC", "^IXIC", "Nasdaq Composite", "global", "us_equity", NY, "16:00", critical=True),
    SeriesSpec("^DJI", "^DJI", "Dow Jones", "global", "us_equity", NY, "16:00"),
    SeriesSpec("^RUT", "^RUT", "Russell 2000", "global", "us_equity", NY, "16:00"),
    SeriesSpec("^SOX", "^SOX", "PHLX Semiconductor", "global", "us_equity", NY, "16:00"),
    SeriesSpec("^VIX", "^VIX", "CBOE VIX", "volatility", "volatility", NY, "16:15", critical=True),
    # Europe
    SeriesSpec("^STOXX50E", "^STOXX50E", "Euro Stoxx 50", "global", "eu_equity", BER, "17:30"),
    SeriesSpec("^FTSE", "^FTSE", "FTSE 100", "global", "eu_equity", LON, "16:30"),
    SeriesSpec("^GDAXI", "^GDAXI", "DAX", "global", "eu_equity", BER, "17:30"),
    SeriesSpec("^FCHI", "^FCHI", "CAC 40", "global", "eu_equity", PAR, "17:30"),
    # Asia-Pacific
    SeriesSpec("^N225", "^N225", "Nikkei 225", "global", "asia_equity", TYO, "15:30"),
    SeriesSpec("^HSI", "^HSI", "Hang Seng", "global", "asia_equity", HKG, "16:00"),
    SeriesSpec("000001.SS", "000001.SS", "Shanghai Composite", "global", "asia_equity", SHA, "15:00"),
    SeriesSpec("^KS11", "^KS11", "KOSPI", "global", "asia_equity", SEL, "15:30"),
    SeriesSpec("^TWII", "^TWII", "TAIEX", "global", "asia_equity", TPE, "13:30"),
    SeriesSpec("^AXJO", "^AXJO", "ASX 200", "global", "asia_equity", SYD, "16:00"),
    # FX (24h; Yahoo dates the bar in London time, final at the end of that day)
    SeriesSpec("USDINR=X", "USDINR=X", "USD/INR", "fx", "fx", LON, "23:59", critical=True),
    SeriesSpec("DX-Y.NYB", "DX-Y.NYB", "US Dollar Index", "fx", "fx", NY, "17:00", critical=True),
    SeriesSpec("EURUSD=X", "EURUSD=X", "EUR/USD", "fx", "fx", LON, "23:59"),
    SeriesSpec("USDJPY=X", "USDJPY=X", "USD/JPY", "fx", "fx", LON, "23:59"),
    SeriesSpec("USDCNY=X", "USDCNY=X", "USD/CNY", "fx", "fx", LON, "23:59"),
    # US rates
    SeriesSpec("^TNX", "^TNX", "US 10Y yield", "rate", "rates", NY, "16:15", critical=True),
    SeriesSpec("^FVX", "^FVX", "US 5Y yield", "rate", "rates", NY, "16:15"),
    SeriesSpec("^IRX", "^IRX", "US 13W yield", "rate", "rates", NY, "16:15"),
    # Commodities (CME Globex trading day ends 17:00 New York)
    SeriesSpec("CL=F", "CL=F", "WTI crude", "commodity", "commodity", NY, "17:00", critical=True),
    SeriesSpec("BZ=F", "BZ=F", "Brent crude", "commodity", "commodity", NY, "17:00", critical=True),
    SeriesSpec("NG=F", "NG=F", "Natural gas", "commodity", "commodity", NY, "17:00"),
    SeriesSpec("GC=F", "GC=F", "Gold", "commodity", "commodity", NY, "17:00"),
    SeriesSpec("SI=F", "SI=F", "Silver", "commodity", "commodity", NY, "17:00"),
    SeriesSpec("HG=F", "HG=F", "Copper", "commodity", "commodity", NY, "17:00"),
    # India
    SeriesSpec("^NSEI", "^NSEI", "Nifty 50", "index", "india_index", KOL, "15:30", lag_min=10, critical=True),
    SeriesSpec("^NSEBANK", "^NSEBANK", "Nifty Bank", "index", "india_index", KOL, "15:30", lag_min=10),
    SeriesSpec("^INDIAVIX", "^INDIAVIX", "India VIX", "volatility", "volatility", KOL, "15:30", lag_min=10, critical=True),
)

# Every NSE cash-market equity follows the same rule as the India index series.
NSE_EQUITY_SPEC = SeriesSpec("NSE_EQUITY", "", "NSE equity", "equity", "india_equity", KOL, "15:30", lag_min=10)

SERIES_BY_KEY: dict[str, SeriesSpec] = {s.key: s for s in SERIES}
CRITICAL_KEYS: tuple[str, ...] = tuple(s.key for s in SERIES if s.critical)
INDIA_MARKET = "^NSEI"


def latest_available_date(spec: SeriesSpec, dates: list[date], cutoff: datetime) -> date | None:
    """Most recent bar date whose bar is final at ``cutoff`` (``dates`` ascending)."""
    for d in reversed(dates):
        if spec.available_at(d) <= cutoff:
            return d
    return None
