"""Populate a ``Store`` from a synthetic panel so jobs, scans and training run end to end without any network."""

from __future__ import annotations

from typing import Any

import pandas as pd

from database.store import Store
from features.builder import Panel
from ingestion.pipeline import ensure_series_instruments
from ingestion.series import INDIA_MARKET
from models.gbm import GbmConfig
from tests.synthetic import make_panel

LAST_SESSION = "2026-10-30"  # the session before the FakeClock "today" (Monday 2026-11-02)

# a training configuration small enough for unit tests (seconds), still exercising every stage of the pipeline
SMALL_TRAIN = dict(
    n_folds=4, fold_days=50, min_train_days=60, embargo_days=3, holdout_days=60, gate_min_days=40, min_rows=1500,
    spec_min_rows=300, n_boot=60, max_rows_per_fit=30000,
    gbm=GbmConfig(n_estimators=20, meta_estimators=15, num_leaves=7, min_data_in_leaf=40, meta_min_data=60),
)


def panel_ending(last: str = LAST_SESSION, **kwargs: Any) -> Panel:
    n_dates = kwargs.pop("n_dates", 420)
    start = pd.bdate_range(end=last, periods=n_dates)[0].strftime("%Y-%m-%d")
    return make_panel(n_dates=n_dates, start=start, **kwargs)


def _bar_rows(index: pd.DatetimeIndex, frame: dict[str, pd.Series]) -> list[dict[str, Any]]:
    rows = []
    for d in index:
        row = {"trade_date": d.date().isoformat()}
        for key, series in frame.items():
            value = series.get(d)
            row[key] = None if value is None or pd.isna(value) else float(value)
        rows.append(row)
    return rows


def populate_store(store: Store, panel: Panel, *, universe: bool = True, skip_last_for: int = 0) -> dict[str, int]:
    """Write equities, index, VIX, global series, F&O and board meetings. ``skip_last_for`` drops the latest session's bar
    for that many symbols (to exercise the data-quality gate)."""
    ids = ensure_series_instruments(store)
    for j, symbol in enumerate(panel.symbols):
        iid = store.upsert_instrument(
            symbol, "equity", exchange="NSE", name=f"{symbol} Limited", sector=panel.sectors[symbol],
            is_fno=int(panel.is_fno[symbol]), in_universe=1 if universe else 0, yahoo_symbol=f"{symbol}.NS", lot_size=500,
        )
        ids[symbol] = iid
        frame = {"open": panel.o[symbol], "high": panel.h[symbol], "low": panel.low[symbol], "close": panel.c[symbol],
                 "volume": panel.v[symbol]}
        rows = _bar_rows(panel.dates, frame)
        if j < skip_last_for:
            rows = rows[:-1]
        store.upsert_bars(iid, rows, source="nse_bhavcopy", source_rank=2, quality="official")

    store.upsert_bars(ids[INDIA_MARKET], _bar_rows(panel.dates, {c: panel.nifty[c] for c in ("open", "high", "low", "close")}),
                      source="yahoo", source_rank=1)
    bank = {c: panel.nifty[c] * 2.3 for c in ("open", "high", "low", "close")}
    store.upsert_bars(ids["^NSEBANK"], _bar_rows(panel.dates, bank), source="yahoo", source_rank=1)
    store.upsert_bars(ids["^INDIAVIX"], _bar_rows(panel.dates, {c: panel.vix for c in ("open", "high", "low", "close")}),
                      source="yahoo", source_rank=1)
    for key, series in panel.closes.items():
        if key in ids:
            store.upsert_bars(ids[key], _bar_rows(series.index, {c: series for c in ("open", "high", "low", "close")}),
                              source="yahoo", source_rank=1)
    if panel.fno is not None:
        store.upsert_fno(panel.fno.to_dict("records"))
    if panel.meetings is not None and len(panel.meetings):
        store.upsert_board_meetings(panel.meetings.to_dict("records"))
        store.kv_set("board_meetings.coverage_start", str(panel.meetings_coverage_start))
    return ids


def preopen_rows(panel: Panel, symbols: list[str], *, final: bool = True, gap: float = 0.004) -> list[dict[str, Any]]:
    """Pre-open book rows: price = last close * (1 + gap)."""
    last = panel.c.iloc[-1]
    rows = []
    for symbol in symbols:
        price = float(last[symbol]) * (1 + gap)
        rows.append({"symbol": symbol, "iep": price, "final_price": price if final else None, "prev_close": float(last[symbol]),
                     "final_qty": 1000, "buy_qty": 1500, "sell_qty": 1400, "pchange": gap * 100})
    return rows


class FakeTelegram:
    configured = True
    last_attempts = 1

    def __init__(self) -> None:
        self.sent: list[tuple[Any, str]] = []
        self.fail = False

    def send_message(self, chat_id: Any, text: str, **_: Any) -> list[int]:
        if self.fail:
            from app.errors import DataUnavailable

            raise DataUnavailable("telegram", "simulated outage", retryable=True)
        self.sent.append((chat_id, text))
        return [len(self.sent)]


class FakeIntel:
    """Stand-in for ``automation.intel.Intel`` in scan tests (no news available)."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.pipeline = None
        self.calls: list[str] = []

    def catalysts(self, *_: Any, **__: Any) -> dict:
        return {}

    def chains(self, *_: Any, **__: Any) -> dict:
        return {}

    def reacted(self, *_: Any, **__: Any) -> dict:
        return {}

    def coverage_note(self) -> None:
        return None

    def process(self) -> dict:
        return {}

    def refresh_reactions(self, *_: Any, **__: Any) -> int:
        self.calls.append("refresh_reactions")
        return 0

    def reseed_graph(self) -> dict:
        self.calls.append("reseed_graph")
        return {}

    def learn_graph(self, *_: Any, **__: Any) -> dict:
        self.calls.append("learn_graph")
        return {}

    def prune(self) -> dict:
        self.calls.append("prune")
        return {}


LIVE_STAMP = "2026-11-02T09:00:00Z"  # 14:30 IST on the FakeClock day


def live_rows(panel: Panel, *, moves: dict[str, float] | None = None, strength: dict[str, float] | None = None,
              volume: dict[str, float] | None = None, fraction: float = 0.79, stamp: str = LIVE_STAMP, default_move: float = 0.1,
              ) -> list[dict[str, Any]]:
    """The live stock-list rows (as ``parse_live_equities`` returns them) of today's session in progress for every
    symbol of ``panel``, which ends on the previous session. ``moves`` is the change in % (default a flat day), ``strength`` the
    position of the last price in the day's range (0 = at the low, 1 = at the high) and ``volume`` today's projected-day volume
    relative to the 20-day average (default 1.0); ``fraction`` is the share of that volume done by the stamp."""
    rows = []
    for symbol in panel.symbols:
        prev = float(panel.c[symbol].iloc[-1])
        move = (moves or {}).get(symbol, default_move)
        last = prev * (1 + move / 100.0)
        opened = prev * (1 + move / 300.0)
        span = prev * 0.02
        low = last - (strength or {}).get(symbol, 0.5) * span
        low, high = min(low, opened), max(low + span, opened)
        shares = float(panel.v[symbol].iloc[-20:].mean()) * (volume or {}).get(symbol, 1.0) * fraction
        rows.append({"symbol": symbol, "open": opened, "high": high, "low": low, "last": last, "prev_close": prev,
                     "change_pct": move, "volume": shares, "turnover": shares * last, "year_high": None, "year_low": None,
                     "row_ts": stamp})
    return rows


class FakeNseApi:
    """Scripted NSE API: ``books`` is a list of row-lists returned one per call (the last one repeats), stamped ``book_ts``
    (the book's own timestamp; set it to an earlier day to simulate NSE serving the previous session's book). ``live`` are the
    rows ``live_equities`` answers (see ``live_rows``), ``indices`` what ``all_indices`` answers; ``fail`` breaks every endpoint,
    ``indices_fail`` only the index one. With a ``clock`` it behaves like the real HTTP client: no call starts at or after the
    ``deadline`` it is given (a scan that hands over a deadline in the past gets nothing, exactly as in production)."""

    def __init__(self, books: list[list[dict[str, Any]]] | None = None, *, fail: bool = False, gift: dict | None = None,
                 holidays: list[dict[str, Any]] | None = None, book_ts: str = "02-Nov-2026 09:08:30",
                 live: list[dict[str, Any]] | None = None, live_ts: str | None = LIVE_STAMP,
                 indices: dict[str, dict[str, Any]] | None = None, indices_fail: bool = False, clock: Any = None) -> None:
        self.books = books or []
        self.calls = 0
        self.fail = fail
        self.gift = gift
        self._holidays = holidays
        self.book_ts = book_ts
        self.live = live or []
        self.live_ts = live_ts
        self.live_calls: list[str] = []
        self.indices = indices or {}
        self.indices_fail = indices_fail
        self.clock = clock

    def _admit(self, deadline: float | None) -> None:
        from app.errors import DataUnavailable

        if self.clock is not None and deadline is not None and self.clock.monotonic() >= deadline:
            raise DataUnavailable("nse_api", "deadline reached", retryable=True)

    def live_equities(self, index: str = "SECURITIES IN F&O", symbols: Any = None, *, deadline: float | None = None):
        from app.errors import DataUnavailable
        from ingestion.nse import LiveEquities

        self._admit(deadline)
        self.live_calls.append(index)
        if self.fail:
            raise DataUnavailable("nse", "simulated block", retryable=True)
        return LiveEquities(self.live_ts, [row for row in self.live if symbols is None or row["symbol"] in symbols])

    def all_indices(self, *, deadline: float | None = None):
        from app.errors import DataUnavailable

        self._admit(deadline)
        if self.fail or self.indices_fail:
            raise DataUnavailable("nse", "simulated block", retryable=True)
        return self.indices

    def preopen(self, key: str = "ALL", symbols: Any = None, *, deadline: float | None = None):
        from app.errors import DataUnavailable
        from ingestion.nse import PreopenSnapshot

        self.calls += 1
        if self.fail:
            raise DataUnavailable("nse", "simulated block", retryable=True)
        rows = self.books[min(self.calls - 1, len(self.books) - 1)] if self.books else []
        return PreopenSnapshot(rows=rows, source_ts=self.book_ts)

    def gift_nifty(self, *, deadline: float | None = None):
        return self.gift

    def holidays(self):
        from app.errors import DataUnavailable

        if self.fail:
            raise DataUnavailable("nse", "simulated block", retryable=True)
        return self._holidays or []

    def announcements(self, start, end, symbols=None):
        return []

    def event_calendar(self, start, end, symbols=None):
        return []


class FakeNews:
    """Stand-in for ``NewsCollector`` in job tests."""

    def __init__(self, rss_ok: int = 9, rss_failed: int = 1) -> None:
        self.calls: list[str] = []
        self.rss_ok, self.rss_failed = rss_ok, rss_failed

    def collect_rss(self, feeds: Any, **_: Any):
        from ingestion.news_sources import CollectReport

        self.calls.append("rss")
        return CollectReport(attempted=self.rss_ok + self.rss_failed, ok=self.rss_ok, failed=self.rss_failed, new_items=25)

    def collect_gdelt(self, queries: Any, **_: Any):
        from ingestion.news_sources import CollectReport

        self.calls.append("gdelt")
        return CollectReport(attempted=2, ok=2, new_items=7, fallback=1)

    def collect_board_meetings(self, *a: Any, **k: Any) -> int:
        self.calls.append("meetings")
        return 2

    def collect_filings(self, *a: Any, **k: Any) -> int:
        self.calls.append("filings")
        return 1

    def prune_raw(self) -> int:
        self.calls.append("prune")
        return 0


def make_job_context(settings: Any, store: Any, clock: Any, run_key: str = "2026-11-02", **kw: Any):
    """A ``JobContext`` whose providers are all fakes (no network); tests replace individual entries as needed."""
    from automation.context import JobContext
    from notifications.service import NotificationService

    telegram = FakeTelegram()
    ctx = JobContext.create(settings, store, run_key=run_key, clock=clock, **kw)
    ctx.ensure_calendar()
    ctx._cache.update(notifier=NotificationService(settings, store, client=telegram, clock=clock), intel=FakeIntel(),
                      nse_api=FakeNseApi(), news=FakeNews(), yahoo=object(), nse_archive=object())
    ctx.telegram = telegram  # type: ignore[attr-defined]
    return ctx
