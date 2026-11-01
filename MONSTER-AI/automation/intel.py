"""Adapter to the news-intelligence and event-graph packages.

News is an *enrichment*: if the packages are unavailable or raise, scans continue with empty catalysts and the report
says so (blueprint: failure of the LLM/news provider must reduce coverage, never block or fabricate)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any

from app.clock import Clock, get_clock, iso
from app.config import Settings
from database.store import Store
from ranking.types import Catalyst

log = logging.getLogger(__name__)

PROCESS_BATCH = 600  # rows handled per pipeline call (the pipeline's own default)
MAX_PROCESS_BATCHES = 4  # a first run or a long outage leaves a backlog: drain up to 2 400 rows in one job, not over several days


class Intel:
    def __init__(self, store: Store, settings: Settings, clock: Clock | None = None) -> None:
        self.store, self.settings, self.clock = store, settings, clock
        self.pipeline: Any = None
        self.graph: Any = None
        self.errors: list[str] = []
        try:
            from news_intelligence.pipeline import NewsPipeline

            llm = None
            if settings.llm_configured:
                from news_intelligence.llm import LLMExtractor

                llm = LLMExtractor(settings, clock=clock)
            self.pipeline = NewsPipeline(store, settings, clock=clock, llm=llm)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"news pipeline unavailable ({type(exc).__name__})")
            log.warning("news pipeline unavailable: %s", exc)
        try:
            from event_graph.propagation import GraphImpactEngine

            self.graph = GraphImpactEngine(store)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"event graph unavailable ({type(exc).__name__})")
            log.warning("event graph unavailable: %s", exc)
        self._ensure_graph()

    def _now(self) -> datetime:
        return (self.clock or get_clock()).now()

    def _degrade(self, note: str) -> None:
        """Record a coverage problem (once per run); ``coverage_note`` hands it to the report."""
        if note not in self.errors:
            self.errors.append(note)

    # ------------------------------------------------------------------------------------------- graph
    def _ensure_graph(self) -> None:
        """Seed the event graph on first use and whenever the shipped seed changed (a cheap fingerprint check otherwise)."""
        if self.graph is None:
            return
        try:
            from event_graph.seed import SEED_KEY, load_seed, seed_fingerprint

            if self.store.kv_get(SEED_KEY) != seed_fingerprint(load_seed()):
                self.reseed_graph()
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"event graph seed unavailable ({type(exc).__name__})")
            log.warning("event graph seed check failed: %s", exc)

    def reseed_graph(self) -> dict[str, Any]:
        """(Re)load the expert priors and the sector memberships of the current instrument master (idempotent; measured
        edges are never overwritten). Call after the universe changed."""
        if self.graph is None:
            return {}
        try:
            from event_graph.repository import GraphRepository
            from event_graph.seed import seed_graph

            summary = seed_graph(self.store)
            size = GraphRepository(self.store).counts()
            out = {k: v for k, v in summary.items() if k != "unmapped"} | {"unmapped": len(summary.get("unmapped", []))}
            # what changed in THIS call is usually nothing (the graph is also seeded on first use); the size is what matters
            return out | {"nodes": size["nodes"], "edges": size["edges"]}
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"event graph seeding failed ({type(exc).__name__})")
            log.warning("event graph seeding failed: %s", exc)
            return {}

    def learn_graph(self, as_of: date) -> dict[str, Any]:
        """Weekly: re-estimate the graph's sensitivities from market returns (never raises; the priors simply stay)."""
        if self.graph is None:
            return {}
        try:
            from automation.calendar import ExchangeCalendar
            from automation.graph_learning import learn_graph

            return learn_graph(self.store, ExchangeCalendar(self.store), as_of)
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"event graph learning failed ({type(exc).__name__})")
            log.warning("event graph learning failed: %s", exc)
            return {}

    # -------------------------------------------------------------------------------------- processing
    def process(self) -> dict[str, Any]:
        """Turn new articles and filings into events (draining a backlog in bounded batches), then check which events the
        market has already reacted to."""
        if self.pipeline is None:
            return {"error": "pipeline unavailable"}
        totals: dict[str, Any] = {}
        try:
            batches = 0
            while batches < MAX_PROCESS_BATCHES:
                batches += 1
                stats = self.pipeline.process_pending(PROCESS_BATCH)
                counts = stats.as_dict() if hasattr(stats, "as_dict") else dict(vars(stats))
                for key, value in counts.items():
                    totals[key] = totals.get(key, 0) + value
                handled = counts.get("articles_in", 0) + counts.get("filings_in", 0)
                if handled < PROCESS_BATCH or counts.get("errors", 0):
                    break  # drained, or something is wrong: do not hammer a failing database
            totals["batches"] = batches
        except Exception as exc:  # noqa: BLE001
            log.exception("news processing failed")
            self._degrade(f"news processing failed ({type(exc).__name__})")
            return {"error": type(exc).__name__}
        totals["reactions"] = self.refresh_reactions(self._now())
        return totals

    def refresh_reactions(self, as_of: datetime) -> int:
        """Fill in whether the market already reacted to recent company events (feeds the "already priced in" logic)."""
        if self.pipeline is None:
            return 0
        try:
            return int(self.pipeline.refresh_reactions(as_of))
        except Exception as exc:  # noqa: BLE001
            self._degrade(f"market-reaction check unavailable ({type(exc).__name__})")
            log.warning("refreshing market reactions failed: %s", exc)
            return 0

    def prune(self) -> dict[str, int]:
        """Retention for raw news / filings and the extraction caches (events themselves are kept)."""
        if self.pipeline is None:
            return {}
        try:
            return dict(self.pipeline.prune())
        except Exception as exc:  # noqa: BLE001
            log.warning("news retention failed: %s", exc)
            return {}

    # ------------------------------------------------------------------------------------------ queries
    def catalysts(self, symbols: Sequence[str], as_of: datetime, limit: int = 3) -> dict[str, list[Catalyst]]:
        out: dict[str, list[Catalyst]] = {}
        if self.pipeline is None:
            return out
        failures: list[str] = []
        for symbol in symbols:
            try:
                items = self.pipeline.catalysts_for(symbol, as_of, limit=limit)
            except Exception as exc:  # noqa: BLE001
                log.warning("catalysts_for failed for %s", symbol, exc_info=True)
                failures.append(type(exc).__name__)
                continue
            if items:
                out[symbol] = list(items)
        if failures:
            self._degrade(f"catalysts unavailable for {len(failures)} of {len(symbols)} stock(s) ({', '.join(sorted(set(failures)))})")
        return out

    def chains(self, symbols: Sequence[str], as_of: datetime) -> dict[str, list[str]]:
        if self.graph is None or self.pipeline is None:
            return {}
        try:
            events = self.pipeline.recent_events(as_of, self.settings.news_lookback_hours, limit=80)
            impacts = self.graph.impacts(events, as_of)
        except Exception as exc:  # noqa: BLE001
            log.warning("graph propagation failed", exc_info=True)
            self._degrade(f"event chains unavailable ({type(exc).__name__})")
            return {}
        wanted = set(symbols)
        return {s: list(getattr(i, "chains", []))[:3] for s, i in impacts.items() if s in wanted and getattr(i, "chains", None)}

    def reacted(self, symbols: Sequence[str], as_of: datetime) -> dict[str, tuple[int, int]]:
        """Per symbol: (events already showing a market reaction, events) within the lookback, point-in-time. Only the events
        a stock is the SUBJECT of count (signed by the event or named in its headline): a passing mention of the stock in
        another company's story is not an event for it."""
        since = iso(as_of - timedelta(hours=self.settings.news_lookback_hours))
        until = iso(as_of)
        wanted = set(symbols)
        tally: dict[str, list[int]] = {}
        try:
            from news_intelligence.features import subject_companies

            rows = self.store.conn.execute(
                "SELECT entities_json, market_reacted FROM events WHERE first_seen_at >= ? AND first_seen_at <= ?", (since, until)
            ).fetchall()
            for row in rows:
                for symbol in subject_companies(row["entities_json"], until) & wanted:
                    counts = tally.setdefault(symbol, [0, 0])
                    counts[0] += 1 if row["market_reacted"] == 1 else 0
                    counts[1] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("already-priced-in check failed", exc_info=True)
            self._degrade(f"already-priced-in check unavailable ({type(exc).__name__})")
            return {}
        return {symbol: (counts[0], counts[1]) for symbol, counts in tally.items()}

    def event_features(self, symbols: Sequence[str], as_of: datetime) -> dict[str, dict[str, float]]:
        if self.pipeline is None:
            return {}
        try:
            return self.pipeline.event_features(list(symbols), as_of, self.settings.news_lookback_hours)
        except Exception as exc:  # noqa: BLE001
            log.warning("event features failed", exc_info=True)
            self._degrade(f"event features unavailable ({type(exc).__name__})")
            return {}

    def coverage_note(self) -> str | None:
        reduced = self.store.kv_get("news.reduced_coverage")
        notes = list(self.errors)
        if reduced == "1":
            notes.append("LLM extraction degraded (rule-based extraction used)")
        return "; ".join(notes) or None
