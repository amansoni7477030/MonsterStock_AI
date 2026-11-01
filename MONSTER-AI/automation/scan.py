"""Pre-open scans (blueprint sections 12-14).

One scan = assess data quality -> build point-in-time features -> score with the active model (verified, with automatic
fallbacks) -> rank with abstention -> FREEZE an immutable snapshot and predictions -> deliver the report.

The snapshot and its predictions are written in ONE transaction, exactly once per (trade date, stage). A retry after a crash or
a failed delivery re-sends the frozen report instead of recomputing (and first restores the predictions of a snapshot that an
older version froze without them), so what is evaluated later is precisely what was reported. A forced re-scan freezes a second
snapshot; only the latest one is effective (``Store.effective_snapshot``) and ever scored.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from app.clock import at_ist, iso
from app.config import Settings
from app.logging_setup import redact
from app.release import release_id
from automation.context import JobContext, JobResult
from backtesting.evaluate import feature_payload, features_hash, snapshot_table
from database.store import Store
from features import regime
from features.live import LiveFeatures, build_live_features
from features.schema import FEATURE_VERSION
from ingestion import basis
from ingestion.freshness import DataQuality, assess_data_quality, ist_label
from ingestion.pipeline import collect_gift, collect_preopen
from models.inference import Bundle, ModelBundle, check_columns, disagreement_limit, pick_uncertainty_ceilings
from ranking.ranker import RankInput, RankResult, enrichment_pool, p_and_range, rank
from ranking.types import Candidate, ReportContext

log = logging.getLogger(__name__)

PREOPEN_TARGET_COVERAGE = 0.90
PREOPEN_POLL_SEC = 15.0
PREOPEN_MAX_CONSECUTIVE_FAILURES = 3
PREOPEN_CONFIRM_AFTER = "09:06:00"  # IST: from here on an empty pre-open book means the exchange is closed today
ENRICH_POOL = 15  # best-ranked names the report can list (after eligibility / cost filters) that get news / event-graph enrichment
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
    ceilings: dict[str, tuple[float, float]] = field(default_factory=dict)  # symbol -> (STRONG, MODERATE) uncertainty limits
    spec_std_limits: dict[str, float] = field(default_factory=dict)  # symbol -> specialist-disagreement limit of its model


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


def monotonic_at(ctx: JobContext, moment: datetime) -> float:
    """The ``clock.monotonic()`` instant corresponding to the wall-clock ``moment`` (HTTP deadlines are monotonic)."""
    return ctx.clock.monotonic() + max((moment - ctx.clock.now()).total_seconds(), 0.0)


@dataclass(frozen=True)
class PreopenWait:
    share: float      # share of the universe with a final equilibrium price
    usable: float     # share with any usable pre-open price for the trade date
    reachable: bool   # NSE answered at least once (an answered-but-stale book means no session today)


def await_preopen(ctx: JobContext, symbols: list[str]) -> PreopenWait:
    """Poll the NSE pre-open book until the equilibrium prices are published or the report cutoff is reached. Every request
    carries the cutoff as its deadline, so a slow NSE can never push the final report past 09:12."""
    s, day = ctx.settings, ctx.run_date
    if not s.nse_enabled:
        return PreopenWait(0.0, 0.0, False)
    limit = at_ist(day, s.report_cutoff_time)

    def coverage() -> tuple[float, float]:
        usable, final = preopen_coverage(ctx.store, day.isoformat(), symbols, ctx.clock.now())
        return final, usable

    failures = 0
    reachable = False
    final, usable = coverage()
    reachable = usable > 0  # rows already stored today prove NSE answered with a fresh book
    while final < PREOPEN_TARGET_COVERAGE and ctx.clock.now() < limit:
        snapshot = _best_effort("pre-open collection", collect_preopen, ctx.store, ctx.nse_api(), day.isoformat(), symbols,
                                deadline=monotonic_at(ctx, limit))
        reachable = reachable or snapshot is not None
        failures = 0 if snapshot is not None and snapshot.rows else failures + 1
        if failures >= PREOPEN_MAX_CONSECUTIVE_FAILURES:
            log.warning("pre-open book unavailable after %d attempts; continuing without it", failures)
            break
        final, usable = coverage()
        remaining = (limit - ctx.clock.now()).total_seconds()
        if final >= PREOPEN_TARGET_COVERAGE or remaining <= 0:
            break
        ctx.clock.sleep(min(PREOPEN_POLL_SEC, remaining))
    final, usable = coverage()
    return PreopenWait(final, usable, reachable or usable > 0)


def wait_for_preopen(ctx: JobContext, symbols: list[str]) -> float:
    """``await_preopen`` reduced to the share of the universe that has its final equilibrium price."""
    return await_preopen(ctx, symbols).share


def suspected_holiday(ctx: JobContext, wait: PreopenWait) -> JobResult | None:
    """The calendar of a year without an official NSE list holds only rule-based holidays, so before the final report goes
    out the exchange itself is asked: NSE answered but published no pre-open book for today = a closed day. The day is then
    recorded as an inferred closure and the operator is told. A merely unreachable NSE proves nothing: the scan goes on."""
    s, day = ctx.settings, ctx.run_date
    if not s.nse_enabled or ctx.force or ctx.calendar.is_official(day):
        return None
    if wait.usable > 0 or not wait.reachable or ctx.clock.now() < at_ist(day, PREOPEN_CONFIRM_AFTER):
        return None
    ctx.store.upsert_calendar_days([{"trade_date": day.isoformat(), "is_trading_day": False, "session_kind": "holiday",
                                     "description": "no NSE pre-open book for the day (inferred closed)"}], source="inferred")
    ctx.calendar.reload()
    message = (f"{day} looks like an exchange holiday: NSE published no pre-open book and the {day.year} holiday list is not "
               "official yet. No report was sent.")
    ctx.store.add_alert("warning", "holiday_inferred", message, dedupe_key=f"holiday-inferred:{day}", dedupe_minutes=24 * 60)
    log.warning("%s", message)
    return JobResult("skipped", message)


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


def _rebased_today(ctx: JobContext, symbols: Any) -> list[str]:
    """Names with an ex-date today (split or bonus) whose stored prices are still on the old basis: the exchange's previous close
    in the pre-open book differs from the stored one. Their opening gap would compare two price bases (a 1:2 split looks like
    a -50 % gap), so the final scan must not use it. A failed check never stops the scan."""
    found = _best_effort("price basis check", basis.preopen_breaks, ctx.store, ctx.run_date.isoformat()) or []
    wanted = set(symbols)
    return sorted(item.symbol for item in found if item.symbol in wanted)


def score(ctx: JobContext, stage: str, live: LiveFeatures) -> Scored:
    registry = ctx.registry()
    notes: list[str] = []
    X = live.X
    main = registry.active_bundle(stage)
    parts: list[tuple[Bundle, pd.DataFrame]] = []
    n_fallback = 0
    if stage == "final" and live.X_pre is not None:
        has_gap = pd.Series([live.open_source.get(sym, "none") != "none" for sym in X.index], index=X.index)
        rebased = _rebased_today(ctx, X.index)
        if rebased:
            has_gap.loc[rebased] = False
            log.warning("final scan: %s have an ex-date today (price basis changed); scored without the opening gap", rebased)
            notes.append(f"{', '.join(rebased[:4])}{' ...' if len(rebased) > 4 else ''}: ex-date today (split or bonus), the opening "
                         "gap is not comparable with the stored prices; scored without it")
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
    ceilings: dict[str, tuple[float, float]] = {}
    spec_limits: dict[str, float] = {}
    versions, statuses = [], []
    for bundle, frame in parts:
        used, preds = _predict(bundle, frame, registry, bundle.stage, notes, ctx.store)
        frames.append(frame)
        predictions.append(preds)
        versions.append(used.version)
        statuses.append(used.status)
        ceilings.update(_uncertainty_limits(used, preds))
        limit = disagreement_limit(used.meta)  # a model's own normal level of specialist disagreement
        if limit is not None:
            spec_limits.update(dict.fromkeys(frame.index, limit))
        for position, symbol in enumerate(frame.index):
            explainers[symbol] = _explainer(used, frame, position)
    features = pd.concat(frames).reindex(X.index)
    preds = pd.concat(predictions).reindex(X.index)
    return Scored(features, preds, explainers, versions[0], _weakest(*statuses), notes, n_fallback, ceilings, spec_limits)


def _uncertainty_limits(bundle: Bundle, preds: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """symbol -> (STRONG, MODERATE) uncertainty ceilings for the names this model scored; empty when it learned none."""
    limits = pick_uncertainty_ceilings(bundle.meta, preds["p_primary"].to_numpy(dtype=float))
    if limits is None:
        return {}
    strong, moderate = limits
    return {symbol: (float(s), float(m)) for symbol, s, m in zip(preds.index, strong, moderate, strict=True) if math.isfinite(s)}


def _explainer(bundle: Bundle, frame: pd.DataFrame, position: int) -> Callable[[], list[str]]:
    return lambda: bundle.explain(frame, position)


def _best_effort(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Enrichment steps (GIFT quote, pre-open book) must never abort a scan."""
    try:
        return fn(*args, **kwargs)
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


def _candidate_row(c: Candidate, version: str, status: str, stored: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen prediction row of one listed candidate: shared by a fresh freeze and by the restore of an interrupted one."""
    return {
        "symbol": c.symbol, "rank": c.rank, "tier": c.tier, "model_version": version, "model_status": status,
        "p": c.p, "range": c.range, "uncertainty": c.uncertainty, "data_quality": c.data_quality,
        "liquidity": {"turnover_cr": c.turnover_cr, "est_cost_pct": c.est_cost_pct, "note": c.liquidity_note},
        "explanation": {"drivers": c.drivers, "confirmations": c.confirmations, "contradictions": c.contradictions,
                        "priced_in": c.priced_in, "chain": c.chain, "catalysts": [x.to_dict() for x in c.catalysts]},
        "features_hash": features_hash(stored, c.symbol),
    }


def _baseline_rows(baselines: Mapping[str, Sequence[str]], version: str) -> list[dict[str, Any]]:
    return [{"symbol": symbol, "rank": position, "tier": "BASELINE", "model_version": version, "model_status": "baseline",
             "p": {}, "range": {}, "uncertainty": None, "data_quality": "ok", "explanation": {}, "is_baseline": True,
             "baseline_name": name}
            for name, symbols in baselines.items() for position, symbol in enumerate(symbols, 1)]


def _prediction_rows(
    result: RankResult, scored: Scored, settings: Settings, stored: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Model picks (listed and shadow-only), plus same-day baseline picks for later comparison. ``stored`` is the snapshot's
    feature payload (built from ``scored.features`` when omitted): a pick's ``features_hash`` covers exactly what the
    snapshot keeps for it, so it can be re-verified from the snapshot alone."""
    stored = feature_payload(scored.features) if stored is None else stored
    version, status = scored.version, scored.status
    rows = [_candidate_row(c, version, status, stored) for c in result.candidates]
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
                     "features_hash": features_hash(stored, symbol)})
    return rows + _baseline_rows(result.baselines, version)


def restore_predictions(store: Store, snapshot: Mapping[str, Any]) -> int:
    """Re-create the predictions of a snapshot that was frozen without them, from its own payload: the report's listed
    candidates and the baseline picks, row for row what the original freeze stored. The shadow-only extras below the listing
    threshold were never part of the payload and are not invented. Returns the number of rows written."""
    payload = snapshot["payload"]
    report = ReportContext.from_dict(payload["report"])
    version = snapshot.get("model_version") or report.model_version
    rows = [_candidate_row(c, version, report.model_status, payload.get("features") or {}) for c in report.candidates]
    rows += _baseline_rows(payload.get("baselines") or {}, version)
    return len(store.insert_predictions(snapshot["snapshot_id"], snapshot["trade_date"], snapshot["stage"], rows))


def _ensure_predictions(store: Store, snapshot: Mapping[str, Any]) -> None:
    """A frozen snapshot must have its predictions. One that lost them (an interruption between the two writes of an older
    version) is repaired from its payload. A failed repair is reported but must not stop the frozen report being re-sent."""
    day, stage, snapshot_id = snapshot["trade_date"], snapshot["stage"], snapshot["snapshot_id"]
    if store.predictions_for(day, stage=stage, baselines=None):
        return
    key = f"snapshot-repair:{snapshot_id}"
    try:
        restored = restore_predictions(store, snapshot)
    except Exception as exc:  # noqa: BLE001
        message = (f"{stage} snapshot {snapshot_id} of {day} has no predictions and they could not be restored: "
                   f"{redact(str(exc))[:160]}")
        log.warning("%s", message, exc_info=True)
        store.add_alert("error", "snapshot_repair_failed", message, dedupe_key=key, dedupe_minutes=24 * 60)
        return
    message = (f"{stage} snapshot {snapshot_id} of {day} had no predictions (interrupted freeze): restored {restored} "
               "from its frozen payload")
    log.warning("%s", message)
    store.add_alert("warning", "snapshot_repaired", message, dedupe_key=key, dedupe_minutes=24 * 60)


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
        _ensure_predictions(ctx.store, existing)
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
        gift_limit = at_ist(day, s.report_cutoff_time) if stage == "final" else at_ist(day, s.final_scan_time)
        _best_effort("GIFT Nifty collection", collect_gift, store, ctx.nse_api(), deadline=monotonic_at(ctx, gift_limit))
    if stage == "final":
        closed = suspected_holiday(ctx, await_preopen(ctx, symbols))
        if closed is not None:
            return closed
    cutoff = scan_cutoff(s, day, stage, clock.now())

    quality = assess_data_quality(store, ctx.calendar, cutoff=cutoff, stage=stage, trade_date=day)
    if quality.blocked:
        return _fail(ctx, stage, "data quality gate blocked the scan: " + "; ".join(quality.messages[:4]), quality.messages)

    live = build_live_features(store, s, ctx.calendar, trade_date=day, stage=stage, cutoff=cutoff)
    scored = score(ctx, stage, live)

    intel = ctx.intel()
    if stage == "final":
        intel.refresh_reactions(cutoff)  # the pre-open quotes are in by now: sharper "already priced in" detection
    last_close = _last_closes(store, live.last_bar)
    top = enrichment_pool(scored.features, scored.preds, last_close, s, ENRICH_POOL)  # after the eligibility / cost filters
    catalysts = intel.catalysts(top, cutoff)
    result = rank(
        RankInput(
            stage=stage, trade_date=day.isoformat(), model_version=scored.version, model_status=scored.status,
            primary_threshold=s.primary_threshold, features=scored.features, preds=scored.preds,
            instruments={r["symbol"]: {"name": r["name"], "sector": r["sector"], "is_fno": bool(r["is_fno"])} for r in universe},
            last_close=last_close, catalysts=catalysts, chains=intel.chains(top, cutoff),
            reacted=intel.reacted(top, cutoff), quality_status=quality.status, quality_notes=quality.messages[:3],
            explainer=lambda symbol: scored.explainers[symbol]() if symbol in scored.explainers else [],
            uncertainty_ceilings=scored.ceilings, spec_std_limits=scored.spec_std_limits,
        ),
        s,
    )

    report = _build_report(ctx, stage, cutoff, quality, live, scored, result, intel.coverage_note())
    if ctx.dry_run:
        return JobResult("success", "dry run: nothing frozen or sent", {"report": report.to_dict()})

    created = clock.now()
    late = created > at_ist(day, s.report_deadline_time)
    code_version = release_id()  # version + code fingerprint: any frozen report can be traced to the exact code that made it
    stored = feature_payload(scored.features)
    freshness = _best_effort("freshness inputs", store.source_freshness, trade_date=day.isoformat(), cutoff_ts=iso(cutoff) or "")
    payload = {
        "report": report.to_dict(), "table": snapshot_table(result.table), "features": stored,
        "baselines": result.baselines, "excluded": result.excluded, "open_source": live.open_source,
        "meta": {"late": late, "n_fallback": scored.n_fallback, "market": live.market, "code": code_version,
                 "expected_session": quality.expected_session, "freshness": freshness or {}},
    }
    with store.tx():  # frozen together or not at all: a snapshot without its predictions could never be evaluated
        snapshot_id = store.insert_snapshot(
            trade_date=day.isoformat(), stage=stage, cutoff_ts=iso(cutoff) or "", feature_version=FEATURE_VERSION,
            model_version=scored.version, code_version=code_version, payload=payload, data_quality=quality.to_dict(),
        )
        ids = store.insert_predictions(snapshot_id, day.isoformat(), stage, _prediction_rows(result, scored, s, stored))
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
    if not ctx.calendar.is_official(ctx.run_date):
        notes.append(f"the {ctx.run_date.year} NSE holiday list is not published yet: holidays are rule-based")
    return ReportContext(
        trade_date=ctx.run_date.isoformat(), stage=stage, snapshot_ist=ist_label(cutoff), model_version=scored.version,
        model_status=scored.status, mode=ctx.settings.mode, candidates=result.candidates, no_setup=result.no_setup,
        no_setup_reason=result.no_setup_reason, regime=_regime_labels(ctx.store, quality, cutoff, m),
        freshness=quality.freshness(), missing_feeds=quality.missing_feeds(), notes=notes[:5],
        universe_size=result.n_universe, n_eligible=result.n_eligible, uncertainty_note=_uncertainty_note(scored),
    )


def _regime_labels(store: Store, quality: DataQuality, cutoff: datetime, market: Mapping[str, Any]) -> dict[str, str]:
    """Regime labels plus GIFT Nifty's change versus its previous close (the market's own hint of the opening). The quote is shown
    only while the freshness gate rates it ok, so a stale quote never reads as today's signal."""
    labels = regime.describe(market.get("m_trend"), market.get("m_volreg"), market.get("m_risk"), market.get("m_expiry_share"))
    state = quality.feeds.get("GIFT Nifty")
    if state is not None and state.level == "ok":
        quote = _best_effort("GIFT Nifty quote", store.latest_live_quote, "GIFT_NIFTY", before=iso(cutoff) or "")
        change = (quote or {}).get("change_pct")
        if isinstance(change, (int, float)) and math.isfinite(change):
            labels["gift_nifty"] = f"{change:+.2f}%"
    return labels


def _uncertainty_note(scored: Scored) -> str:
    """Tells the reader what the conviction tiers compare each pick's uncertainty with (empty: the fixed fallback ceilings)."""
    if not scored.ceilings:
        return ""
    return ("tiers compare each pick's uncertainty with the model's held-out names of similar probability "
            "(STRONG: at or below their median, MODERATE: at or below their 85th percentile)")
