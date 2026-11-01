"""Pre-open scans (blueprint sections 12-14).

One scan = assess data quality -> build point-in-time features -> score with the active model (verified, with automatic
fallbacks) -> rank with abstention -> FREEZE an immutable snapshot and predictions -> deliver the report.

The snapshot is written exactly once per (trade date, stage). A retry after a crash or a failed delivery re-sends the
frozen report instead of recomputing, so what is evaluated later is precisely what was reported.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from app import __version__, jsonutil
from app.clock import at_ist, iso
from app.config import Settings
from app.logging_setup import redact
from automation.context import JobContext, JobResult
from backtesting.evaluate import feature_payload, snapshot_table
from database.store import Store
from features import regime
from features.live import LiveFeatures, build_live_features
from features.schema import FEATURE_VERSION
from ingestion.freshness import DataQuality, assess_data_quality, ist_label
from ingestion.pipeline import collect_gift, collect_preopen
from models.inference import Bundle, ModelBundle, check_columns
from ranking.ranker import RankInput, RankResult, p_and_range, rank
from ranking.types import ReportContext

log = logging.getLogger(__name__)

PREOPEN_TARGET_COVERAGE = 0.90
PREOPEN_POLL_SEC = 15.0
PREOPEN_MAX_CONSECUTIVE_FAILURES = 3
ENRICH_POOL = 15  # best-scored names that get news / event-graph enrichment before the final ranking
MISSING_INPUT_WARN_RATIO = 0.25
STATUS_ORDER = {"baseline": 0, "shadow": 1, "validated": 2}


@dataclass
class Scored:
    features: pd.DataFrame
    preds: pd.DataFrame
    explainers: dict[str, Callable[[], list[str]]]
    version: str
    status: str
    notes: list[str] = field(default_factory=list)
    n_fallback: int = 0


# ------------------------------------------------------------------------------------------------ timing
def scan_cutoff(settings: Settings, trade_date: date, stage: str, now: datetime) -> datetime:
    """Information cutoff: the scan's own start time, bounded to [nominal scan time, report cutoff]."""
    nominal = at_ist(trade_date, settings.prelim_scan_time if stage == "pre" else settings.final_scan_time)
    ceiling = at_ist(trade_date, settings.report_cutoff_time)
    return min(max(now, nominal), ceiling)


def preopen_coverage(store: Store, trade_date: str, symbols: list[str], before: datetime) -> tuple[float, float]:
    """(share of symbols with any usable pre-open price, share with the final equilibrium price)."""
    if not symbols:
        return 0.0, 0.0
    book = store.latest_preopen(trade_date, before=iso(before))
    wanted = set(symbols)
    usable = sum(1 for s, q in book.items() if s in wanted and (q.get("final_price") or q.get("iep")))
    final = sum(1 for s, q in book.items() if s in wanted and q.get("final_price"))
    return usable / len(wanted), final / len(wanted)


def wait_for_preopen(ctx: JobContext, symbols: list[str]) -> float:
    """Poll the NSE pre-open book until the equilibrium prices are published or the report cutoff is reached."""
    s, day = ctx.settings, ctx.run_date
    if not s.nse_enabled:
        return 0.0
    limit = at_ist(day, s.report_cutoff_time)

    def final_share() -> float:
        return preopen_coverage(ctx.store, day.isoformat(), symbols, ctx.clock.now())[1]

    failures = 0
    final = final_share()
    while final < PREOPEN_TARGET_COVERAGE and ctx.clock.now() < limit:
        snapshot = _best_effort("pre-open collection", collect_preopen, ctx.store, ctx.nse_api(), day.isoformat(), symbols)
        failures = 0 if snapshot is not None and snapshot.rows else failures + 1
        if failures >= PREOPEN_MAX_CONSECUTIVE_FAILURES:
            log.warning("pre-open book unavailable after %d attempts; continuing without it", failures)
            break
        final = final_share()
        remaining = (limit - ctx.clock.now()).total_seconds()
        if final >= PREOPEN_TARGET_COVERAGE or remaining <= 0:
            break
        ctx.clock.sleep(min(PREOPEN_POLL_SEC, remaining))
    return final_share()


# ------------------------------------------------------------------------------------------------ scoring
def _weakest(*statuses: str) -> str:
    return min(statuses, key=lambda s: STATUS_ORDER.get(s, 0))


def _predict(bundle: Bundle, frame: pd.DataFrame, registry: Any, stage: str, notes: list[str], store: Store) -> tuple[Bundle, pd.DataFrame]:
    """Predict with ``bundle``; any failure of a trained model degrades to the heuristic baseline (never aborts a scan)."""
    if isinstance(bundle, ModelBundle):
        missing = check_columns(bundle, frame)
        total = max(len(bundle.meta.get("feature_columns", [])), 1)
        if missing:
            notes.append(f"{len(missing)} of {total} model inputs unavailable (treated as missing)")
        if len(missing) / total > MISSING_INPUT_WARN_RATIO:
            store.add_alert("warning", "model_inputs_missing", f"{bundle.version}: {len(missing)} of {total} inputs missing",
                            dedupe_key=f"inputs-missing:{bundle.version}", dedupe_minutes=720)
        try:
            return bundle, bundle.predict(frame)
        except Exception as exc:  # noqa: BLE001 - a broken model must not stop the report
            log.exception("model %s failed to predict", bundle.version)
            store.add_alert("error", "model_predict_failed", f"{bundle.version}: {type(exc).__name__}",
                            dedupe_key=f"predict-failed:{bundle.version}", dedupe_minutes=720)
            notes.append(f"model {bundle.version} failed at inference; heuristic baseline used")
            bundle = registry.baseline(stage)
    return bundle, bundle.predict(frame)


def score(ctx: JobContext, stage: str, live: LiveFeatures) -> Scored:
    registry = ctx.registry()
    notes: list[str] = []
    X = live.X
    main = registry.active_bundle(stage)
    parts: list[tuple[Bundle, pd.DataFrame]] = []
    n_fallback = 0
    if stage == "final" and live.X_pre is not None:
        has_gap = pd.Series([live.open_source.get(sym, "none") != "none" for sym in X.index], index=X.index)
        if has_gap.all():
            parts.append((main, X))
        else:
            fallback = registry.active_bundle("pre")
            if has_gap.any():
                parts.append((main, X[has_gap]))
            lacking = X.index[~has_gap]
            parts.append((fallback, live.X_pre.reindex(lacking)))
            n_fallback = len(lacking)
            notes.append(f"{n_fallback} of {len(X)} names have no pre-open price; scored with the preliminary model")
    else:
        parts.append((main, X))

    frames, predictions, explainers = [], [], {}
    versions, statuses = [], []
    for bundle, frame in parts:
        used, preds = _predict(bundle, frame, registry, bundle.stage, notes, ctx.store)
        frames.append(frame)
        predictions.append(preds)
        versions.append(used.version)
        statuses.append(used.status)
        for position, symbol in enumerate(frame.index):
            explainers[symbol] = _explainer(used, frame, position)
    features = pd.concat(frames).reindex(X.index)
    preds = pd.concat(predictions).reindex(X.index)
    return Scored(features, preds, explainers, versions[0], _weakest(*statuses), notes, n_fallback)


def _explainer(bundle: Bundle, frame: pd.DataFrame, position: int) -> Callable[[], list[str]]:
    return lambda: bundle.explain(frame, position)


def _best_effort(label: str, fn: Callable[..., Any], *args: Any) -> Any:
    """Enrichment steps (GIFT quote, pre-open book) must never abort a scan."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s failed: %s", label, redact(str(exc)))
        return None


# ------------------------------------------------------------------------------------------- persistence
def _last_closes(store: Store, session: str | None) -> dict[str, float]:
    if not session:
        return {}
    rows = store.conn.execute(
        "SELECT i.symbol AS symbol, b.close AS close FROM bars_daily b JOIN instruments i USING (instrument_id) "
        "WHERE b.trade_date = ? AND i.kind = 'equity' AND b.close IS NOT NULL", (session,)).fetchall()
    return {r["symbol"]: float(r["close"]) for r in rows}


def _prediction_rows(result: RankResult, scored: Scored, settings: Settings) -> list[dict[str, Any]]:
    """Model picks (listed and shadow-only), plus same-day baseline picks for later comparison."""
    rows: list[dict[str, Any]] = []
    version, status = scored.version, scored.status
    for c in result.candidates:
        rows.append({
            "symbol": c.symbol, "rank": c.rank, "tier": c.tier, "model_version": version, "model_status": status,
            "p": c.p, "range": c.range, "uncertainty": c.uncertainty, "data_quality": c.data_quality,
            "liquidity": {"turnover_cr": c.turnover_cr, "est_cost_pct": c.est_cost_pct, "note": c.liquidity_note},
            "explanation": {"drivers": c.drivers, "confirmations": c.confirmations, "contradictions": c.contradictions,
                            "priced_in": c.priced_in, "chain": c.chain, "catalysts": [asdict(x) for x in c.catalysts]},
            "features_hash": jsonutil.content_hash(scored.features.loc[c.symbol].to_dict()),
        })
    listed = {c.symbol for c in result.candidates}
    pool = result.table[result.table["in_pool"]].sort_values("p_primary", ascending=False, kind="mergesort")
    rank_no = len(rows)
    for symbol in pool.index:
        if rank_no >= settings.top_n:
            break
        if symbol in listed:
            continue
        rank_no += 1
        p, rng = p_and_range(scored.preds.loc[symbol])
        rows.append({"symbol": symbol, "rank": rank_no, "tier": "NONE", "model_version": version, "model_status": status,
                     "p": p, "range": rng, "uncertainty": float(scored.preds.at[symbol, "uncertainty"]), "data_quality": "ok",
                     "explanation": {"note": "below the listing threshold; recorded for shadow evaluation only"},
                     "features_hash": jsonutil.content_hash(scored.features.loc[symbol].to_dict())})
    for name, symbols in result.baselines.items():
        for position, symbol in enumerate(symbols, 1):
            rows.append({"symbol": symbol, "rank": position, "tier": "BASELINE", "model_version": version,
                         "model_status": "baseline", "p": {}, "range": {}, "uncertainty": None, "data_quality": "ok",
                         "explanation": {}, "is_baseline": True, "baseline_name": name})
    return rows


# --------------------------------------------------------------------------------------------- delivery
def deliver(ctx: JobContext, stage: str, report: ReportContext) -> JobResult:
    """Send (or re-send, idempotently) a frozen report and translate the outcome into a job result."""
    if ctx.dry_run:
        return JobResult("success", "dry run: report not sent")
    outcome = ctx.notifier().send_report(report)
    if outcome.ok:
        return JobResult("success", f"report {outcome.status}", {"message_ids": outcome.message_ids})
    if outcome.status == "withheld":
        return JobResult("success", f"snapshot frozen; report withheld ({outcome.reason})")
    if ctx.last_attempt:
        ctx.notifier().send_failure_notice(report.trade_date, stage, "report could not be delivered", outcome.reason)
    return JobResult("failed", f"report delivery failed: {outcome.reason}")


def _fail(ctx: JobContext, stage: str, reason: str, details: Any = None) -> JobResult:
    ctx.store.add_alert("error", f"scan_{stage}_failed", reason, dedupe_key=f"scan-failed:{stage}:{ctx.run_key}",
                        dedupe_minutes=30)
    if ctx.last_attempt and not ctx.dry_run and stage == "final":
        ctx.notifier().send_failure_notice(ctx.run_key, stage, reason, details)
    return JobResult("failed", reason)


# ------------------------------------------------------------------------------------------------ the scan
def run_scan(ctx: JobContext, stage: str) -> JobResult:
    if stage not in {"pre", "final"}:
        raise ValueError(f"unknown scan stage {stage!r}")
    ctx.ensure_calendar()
    day = ctx.run_date
    if not ctx.calendar.is_trading_day(day):
        return JobResult("skipped", f"exchange closed on {day} ({ctx.calendar.holiday_name(day)})")
    existing = None if ctx.force else ctx.store.latest_snapshot(day.isoformat(), stage)
    if existing and not ctx.dry_run:
        stored = existing["payload"].get("report")
        if not stored:
            return JobResult("failed", f"frozen {stage} snapshot {existing['snapshot_id']} has no report payload")
        return deliver(ctx, stage, ReportContext.from_dict(stored))
    try:
        return _scan(ctx, stage)
    except Exception as exc:  # noqa: BLE001 - one bad scan must be reported, not crash the daemon
        log.exception("%s scan failed", stage)
        return _fail(ctx, stage, f"{type(exc).__name__}: {redact(str(exc))[:200]}")


def _scan(ctx: JobContext, stage: str) -> JobResult:
    s, store, clock, day = ctx.settings, ctx.store, ctx.clock, ctx.run_date
    universe = store.list_instruments(kind="equity", in_universe=True, status="active")
    symbols = [r["symbol"] for r in universe]
    if not symbols:
        return _fail(ctx, stage, "no instruments in the universe (run bootstrap)")

    if s.nse_enabled:
        _best_effort("GIFT Nifty collection", collect_gift, store, ctx.nse_api())
    if stage == "final":
        wait_for_preopen(ctx, symbols)
    cutoff = scan_cutoff(s, day, stage, clock.now())

    quality = assess_data_quality(store, ctx.calendar, cutoff=cutoff, stage=stage, trade_date=day)
    if quality.blocked:
        return _fail(ctx, stage, "data quality gate blocked the scan: " + "; ".join(quality.messages[:4]), quality.messages)

    live = build_live_features(store, s, ctx.calendar, trade_date=day, stage=stage, cutoff=cutoff)
    scored = score(ctx, stage, live)

    intel = ctx.intel()
    if stage == "final":
        intel.refresh_reactions(cutoff)  # the pre-open quotes are in by now: sharper "already priced in" detection
    top = scored.preds["p_primary"].dropna().nlargest(ENRICH_POOL).index.tolist()
    catalysts = intel.catalysts(top, cutoff)
    result = rank(
        RankInput(
            stage=stage, trade_date=day.isoformat(), model_version=scored.version, model_status=scored.status,
            primary_threshold=s.primary_threshold, features=scored.features, preds=scored.preds,
            instruments={r["symbol"]: {"name": r["name"], "sector": r["sector"], "is_fno": bool(r["is_fno"])} for r in universe},
            last_close=_last_closes(store, live.last_bar), catalysts=catalysts, chains=intel.chains(top, cutoff),
            reacted=intel.reacted(top, cutoff), quality_status=quality.status, quality_notes=quality.messages[:3],
            explainer=lambda symbol: scored.explainers[symbol]() if symbol in scored.explainers else [],
        ),
        s,
    )

    report = _build_report(ctx, stage, cutoff, quality, live, scored, result, intel.coverage_note())
    if ctx.dry_run:
        return JobResult("success", "dry run: nothing frozen or sent", {"report": report.to_dict()})

    created = clock.now()
    late = created > at_ist(day, s.report_deadline_time)
    payload = {
        "report": report.to_dict(), "table": snapshot_table(result.table), "features": feature_payload(scored.features),
        "baselines": result.baselines, "excluded": result.excluded, "open_source": live.open_source,
        "meta": {"late": late, "n_fallback": scored.n_fallback, "market": live.market, "code": __version__},
    }
    snapshot_id = store.insert_snapshot(
        trade_date=day.isoformat(), stage=stage, cutoff_ts=iso(cutoff) or "", feature_version=FEATURE_VERSION,
        model_version=scored.version, code_version=__version__, payload=payload, data_quality=quality.to_dict(),
    )
    ids = store.insert_predictions(snapshot_id, day.isoformat(), stage, _prediction_rows(result, scored, s))
    log.info("%s scan frozen: snapshot=%s predictions=%d model=%s (%s) candidates=%s", stage, snapshot_id, len(ids),
             scored.version, scored.status, [c.symbol for c in result.candidates])
    outcome = deliver(ctx, stage, report)
    outcome.metrics.update(snapshot_id=snapshot_id, predictions=len(ids), candidates=len(result.candidates), late=late)
    return outcome


def _build_report(
    ctx: JobContext, stage: str, cutoff: datetime, quality: DataQuality, live: LiveFeatures, scored: Scored,
    result: RankResult, news_note: str | None,
) -> ReportContext:
    m = live.market
    notes = list(scored.notes)
    if news_note:
        notes.append(f"news coverage reduced: {news_note}")
    if scored.status == "baseline":
        notes.append("no trained model is active yet; heuristic baseline in use")
    return ReportContext(
        trade_date=ctx.run_date.isoformat(), stage=stage, snapshot_ist=ist_label(cutoff), model_version=scored.version,
        model_status=scored.status, mode=ctx.settings.mode, candidates=result.candidates, no_setup=result.no_setup,
        no_setup_reason=result.no_setup_reason,
        regime=regime.describe(m.get("m_trend"), m.get("m_volreg"), m.get("m_risk"), m.get("m_expiry_share")),
        freshness=quality.freshness(), missing_feeds=quality.missing_feeds(), notes=notes[:5],
        universe_size=result.n_universe, n_eligible=result.n_eligible,
    )
