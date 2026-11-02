"""Made-up sample messages for ``monster-ai telegram-preview``: one of every kind, built from obviously fake data.

Nothing here comes from the market or the database. Every preview starts with ``PREVIEW_BANNER`` so a sample can never be
mistaken for a real report, and the fake names (``DEMOONE``...) and example.com links cannot be traded or followed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.clock import IST, iso, ist_date
from notifications import charts, report
from notifications import ideas as idea_messages
from notifications.telegram import MAX_CAPTION_CHARS, MAX_MESSAGE_CHARS, split_message
from ranking.ideas import BTST_PLAN, SWING_PLAN, WATCH_PLAN
from ranking.types import (
    Candidate,
    Catalyst,
    EveningRow,
    EveningSummary,
    IdeasReport,
    ReportContext,
    TradeIdea,
)

PREVIEW_BANNER = "🧪 <b>SAMPLE PREVIEW</b> - made-up data, not a market signal"

# kind -> what it shows; the order is the order they are sent in
KINDS = {
    "final": "the 09:12 report with a validated model: summary table, chart and the detail of every candidate",
    "prelim": "the optional 08:45 preliminary report (shadow model: tiers capped at WATCH)",
    "no-setup": "the report on a morning without a high-conviction setup",
    "ideas": "the 14:30 closing-hour trade ideas (BTST, swing, bearish F&O, tomorrow's intraday watchlist) with level tables, charts and their live record",
    "evening": "the after-close evaluation with the live record, a table and a chart",
    "alert": "one operational alert of each severity",
    "digest": "several queued alerts in one message",
    "failure": "the notice that a report was withheld",
    "unit": "the alert for a failed systemd service",
    "online": "the connectivity test message",
}
ALERT_KINDS = frozenset({"alert", "digest", "unit"})  # these go to the alert chat, the rest to the report chat


def _candidates() -> list[Candidate]:
    return [
        Candidate(
            symbol="DEMOONE", rank=1, tier="STRONG", name="Demo One Industries Ltd", sector="Energy", is_fno=True,
            p={"hit_1": 0.93, "hit_2": 0.81, "hit_3": 0.64, "hit_5": 0.28, "unusual": 0.42, "up": 0.54, "down": 0.46},
            primary_threshold=3.0, range={"q10": 2.2, "q50": 3.5, "q90": 5.4, "exc_q50": 2.3, "exc_q90": 3.9},
            catalysts=[Catalyst(headline="Demo refinery outage lifts crack spreads", source="Demo Wire", url="https://example.com/demo",
                                source_class="media", n_independent=3)],
            chain=["Demo crude spike -> demo refiners -> DEMOONE"],
            confirmations=["Pre-open volume 3x average", "Futures open interest up"], contradictions=[],
            priced_in="Partly priced in (gap +1.2%)", liquidity_ok=True, liquidity_note="liquid", turnover_cr=412.5,
            est_cost_pct=0.12, uncertainty=0.31, drivers=["gap_pct", "atr_ratio", "news_novelty"],
        ),
        Candidate(
            symbol="DEMOTWO", rank=2, tier="MODERATE", name="Demo Two Metals Ltd", sector="Metals", is_fno=True,
            p={"hit_1": 0.88, "hit_2": 0.69, "hit_3": 0.52, "hit_5": 0.19, "unusual": 0.37, "up": 0.5, "down": 0.5},
            primary_threshold=3.0, range={"q10": 1.9, "q50": 3.1, "q90": 4.8, "exc_q50": 2.0, "exc_q90": 3.4},
            catalysts=[Catalyst(headline="Demo chatter about a large export order", source="Demo Forum", rumour=True,
                                source_class="social", n_independent=2, n_conflicts=1)],
            chain=["Demo steel prices -> demo metals -> DEMOTWO"], confirmations=["Sector futures active"],
            contradictions=["Pre-open price below the previous close"], priced_in="Not priced in",
            liquidity_ok=True, liquidity_note="liquid", turnover_cr=188.0, est_cost_pct=0.18, uncertainty=0.44,
            drivers=["atr_ratio", "sector_momentum"],
        ),
        Candidate(
            symbol="DEMOTRE", rank=3, tier="WATCH", name="Demo Three Bank Ltd", sector="Banks", is_fno=False,
            p={"hit_1": 0.8, "hit_2": 0.55, "hit_3": 0.4, "hit_5": 0.1, "unusual": 0.3, "up": 0.48, "down": 0.52},
            primary_threshold=3.0, range={"q10": 1.4, "q50": 2.6, "q90": 4.1, "exc_q50": 1.7, "exc_q90": 3.0},
            chain=[], confirmations=[], contradictions=[], priced_in="not assessed", liquidity_ok=False,
            liquidity_note="thin pre-open book", data_quality="warning", turnover_cr=54.2, est_cost_pct=0.41, uncertainty=0.58,
            drivers=["gap_pct"],
        ),
    ]


def _context(today: str, *, stage: str, status: str, model: str, candidates: list[Candidate], **over: Any) -> ReportContext:
    base: dict[str, Any] = dict(
        trade_date=today, stage=stage, snapshot_ist=f"{today} {'08:45:00' if stage == 'pre' else '09:12:00'} IST",
        model_version=model, model_status=status, candidates=candidates,
        regime={"trend": "bull", "volatility": "normal-vol", "risk": "risk-on", "gift_nifty": "+0.40%"},
        freshness={"India daily bars": "ok (98% of universe)", "Global closes": "ok (31/33 fresh)",
                   "Pre-open book": "ok (96% final prices)", "GIFT Nifty": "ok (4 min old)",
                   "News feeds": "WARN - 1 of 6 feeds is late"},
        missing_feeds=[], uncertainty_note="Intervals are wide for 2 names", universe_size=212, n_eligible=187,
        notes=["Sample preview: nothing in this message is real"],
    )
    base.update(over)
    return ReportContext(**base)


def _row(name: str, value: float, n: int | None = None, low: float | None = None, high: float | None = None, **extra: Any) -> dict:
    return {"name": name, "slice": "all", "value": value, "n": n, "ci_low": low, "ci_high": high, "extra": extra or None}


def _evening(today: str) -> EveningSummary:
    window = [
        _row("topn_hit_rate", 0.6, 100, 0.52, 0.68), _row("range_mae_q50", 1.42, 100, 1.2, 1.65),
        _row("range_coverage_80", 0.71, 100, 0.62, 0.79), _row("direction_accuracy", 0.54, 96, 0.45, 0.63),
        _row("pick_range_mean", 4.1, 100), _row("pick_range_median", 3.8, 100),
        _row("baseline_atr_range_mean", 3.6, 100), _row("baseline_atr_range_median", 3.4, 100),
        _row("baseline_random_range_mean", 2.4, 100), _row("baseline_random_range_median", 2.2, 100),
        _row("ndcg_at_5", 0.71, 20, 0.6, 0.8, random=0.45), _row("rank_ic", 0.12, 18, 0.02, 0.22),
    ]
    return EveningSummary(
        trade_date=today, stage="final", primary_threshold=3.0, model_version="demo-model-0.1",
        rows=[EveningRow("DEMOONE", 1, "STRONG", 0.64, 4.2, 3.1, 1.2, True), EveningRow("DEMOTWO", 2, "MODERATE", 0.52, 2.4, 1.6, -0.8, False),
              EveningRow("DEMOTRE", 3, "WATCH", 0.4, 3.3, 2.5, 0.4, True)],
        n=3, model_hit_rate=2 / 3, model_hit_ci=(0.21, 0.94), baseline_hit_rates={"atr": 0.33, "random": 0.2},
        mean_range_pct=3.3, baseline_mean_range_pct={"atr": 3.0, "random": 2.2},
        rolling={"hit_rate_30d": 0.58, "brier_30d": 0.19, "n_days": 30}, calibration_note="Well calibrated across the top bins",
        data_quality_notes=["sample preview: nothing in this message is real"], provisional=True, missing_symbols=[],
        live_record={"20d": window},
    )


def _alerts() -> list[tuple[str, str, str]]:
    return [
        ("critical", "health_db_corrupt", "SQLite integrity check failed: database disk image is malformed"),
        ("error", "scan_final_failed", "final scan blocked by the data-quality gate: the previous session's bars are incomplete"),
        ("warning", "health_storage_budget", "the data directory uses 82% of DATA_MAX_MB"),
        ("info", "bootstrap_complete", "Bootstrap finished; the daily timetable is now active."),
    ]


def _ideas_report(today: str, now: datetime) -> IdeasReport:
    """A closing-hour report with every kind of list, an unvalidated record and a validated one."""
    cut = now.astimezone(IST)
    btst = TradeIdea(
        symbol="DEMOONE", rank=1, horizon="BTST", side="LONG", setup="closing strength with a breakout", score=82.4, grade="A",
        ltp=842.1, name="Demo One Motors Ltd", sector="Automobile", is_fno=True, change_pct=2.9, entry=842.1, max_entry=845.4,
        stop=831.6, target1=852.0, target2=861.5, risk_pct=1.25, reward1_pct=1.18, reward2_pct=2.30, rr1=0.94, rr2=1.84,
        hold=BTST_PLAN.hold,
        reasons=["closing near the day's high (position in the day's range 0.93)", "volume 2.1x its 20-day average (projected for the full day)",
                 "trading beyond its prior 20-day high", "sector peers +1.1% today"],
        cautions=["RSI 71: already stretched"],
        catalysts=[Catalyst(headline="Demo order win of Rs 900 crore", source="Demo Wire", url="https://example.com/demo-order",
                            source_class="exchange_filing", polarity=0.7, severity=0.6, confidence=0.8)],
        turnover_cr=412.5, est_cost_pct=0.31,
    )
    swing = TradeIdea(
        symbol="DEMOTWO", rank=1, horizon="SWING", side="LONG", setup="pullback to the 20-day average in an uptrend", score=71.2,
        grade="B", ltp=1530.4, name="Demo Two Bank Ltd", sector="Financial Services", is_fno=True, change_pct=0.8, entry=1530.4,
        max_entry=1537.0, stop=1488.0, target1=1590.0, target2=1648.0, risk_pct=2.77, reward1_pct=3.89, reward2_pct=7.69, rr1=1.4,
        rr2=2.78, hold=SWING_PLAN.hold,
        reasons=["trend aligned: above its 20 and 50-day averages", "pullback to the 20-day average in an uptrend"],
        cautions=["results / board meeting in 3 sessions: gap risk"], turnover_cr=188.0, est_cost_pct=0.33,
    )
    short = TradeIdea(
        symbol="DEMOTRE", rank=1, horizon="BTST", side="SHORT", setup="weak close on volume", score=66.0, grade="C", ltp=310.2,
        name="Demo Three Metals Ltd", sector="Metals & Mining", is_fno=True, change_pct=-2.4, entry=310.2, max_entry=309.6, stop=315.0,
        target1=305.6, target2=301.4, risk_pct=1.55, reward1_pct=1.48, reward2_pct=2.84, rr1=0.96, rr2=1.84,
        hold=BTST_PLAN.hold,
        reasons=["closing near the day's low (position in the day's range 0.08)"], cautions=["market regime is bullish: this side is discounted"],
        turnover_cr=96.0, est_cost_pct=0.36,
    )
    watch = TradeIdea(
        symbol="DEMOFOR", rank=1, horizon="WATCH", side="BOTH", setup="tight range below the 20-day high", score=74.0, grade="B",
        ltp=2204.0, name="Demo Four Pharma Ltd", sector="Healthcare", is_fno=False, change_pct=0.2, trigger_long=2216.5, trigger_short=2181.0,
        hold=WATCH_PLAN.hold,
        reasons=["typical daily range 2.1% (ATR): room to move", "last 5 sessions' range 0.71x the 20-day average: coiled"],
        cautions=["a gap at the open can jump past the level: do not chase it"], turnover_cr=64.0,
        metrics={"long_stop": 2193.6, "long_target1": 2229.8, "long_target2": 2250.2,
                 "short_stop": 2203.4, "short_target1": 2168.0, "short_target2": 2147.0},
    )
    return IdeasReport(
        trade_date=today, snapshot_ist=f"{cut:%Y-%m-%d %H:%M:%S} IST", entry_by="15:15", source="NSE (SECURITIES IN F&O): 196 of 201 names with a complete live row",
        ideas=[btst, swing, short, watch],
        market={"nifty_chg": 0.37, "vix": 13.2, "vix_chg": -2.9, "advancers": 133, "decliners": 61, "regime": "bullish"},
        record={
            "BTST": {"group": "BTST", "n": 64, "days": 22, "pending": 0, "validated": True, "win_rate": 0.58, "win_ci": [0.51, 0.65],
                     "mean_net_pct": 0.21, "mean_net_ci": [0.04, 0.38], "t1_rate": 0.44, "stop_rate": 0.31, "mean_gap_pct": 0.18},
            "SWING": {"group": "SWING", "n": 12, "days": 8, "pending": 3, "validated": False, "win_rate": 0.5, "win_ci": [0.3, 0.7],
                      "mean_net_pct": 0.4, "mean_net_ci": [-0.8, 1.6], "t1_rate": 0.33, "stop_rate": 0.25, "mean_gap_pct": 0.1},
        },
        notes=["sample preview: nothing in this message is real", "volume is projected to a full day: 79% of a session's volume is normally traded by now (an estimate)"],
        universe_size=201, n_eligible=164,
    )


def _final_context(today: str) -> ReportContext:
    return _context(today, stage="final", status="validated", model="demo-model-0.1", candidates=_candidates())


def _prelim_context(today: str) -> ReportContext:
    return _context(today, stage="pre", status="shadow", model="demo-model-0.1", candidates=_candidates()[:2])


def build_samples(settings: Any, now: datetime) -> list[tuple[str, list[str]]]:
    """``[(kind, messages)]`` in sending order; ``now`` (aware) supplies the date and the alert times."""
    today = ist_date(now).isoformat()
    stamp = iso(now)
    alerts = _alerts()
    digest = [{"level": level, "code": code, "message": message, "created_at": stamp} for level, code, message in alerts[:3]]
    journal = ("Nov 02 08:30:01 demo systemd[1]: monster-ai.service: Main process exited, code=killed, status=9/KILL\n"
               "Nov 02 08:30:01 demo systemd[1]: monster-ai.service: Failed with result 'signal'.")
    return [
        ("final", report.render_final_report(_final_context(today))),
        ("prelim", report.render_prelim_report(_prelim_context(today))),
        ("no-setup", report.render_final_report(_context(
            today, stage="final", status="validated", model="demo-model-0.1", candidates=[], no_setup=True,
            no_setup_reason="the top probability is only 33% and no name cleared the conviction gates"))),
        ("ideas", idea_messages.render_ideas_report(_ideas_report(today, now))),
        ("evening", report.render_evening_report(_evening(today))),
        ("alert", [report.render_alert(level, code, message, created_at=stamp) for level, code, message in alerts]),
        ("digest", report.render_alert_digest(digest)),
        ("failure", report.render_failure_notice(today, "final", "deadline missed",
                                                  {"feed": "yahoo", "age": "2 d", "gate": "global block stale"})),
        ("unit", [report.render_unit_failure("monster-ai.service", journal, now)]),
        ("online", [report.render_test_message(settings)]),
    ]


def build_sample_charts(now: datetime) -> dict[str, list[charts.Chart]]:
    """``{kind: [chart]}`` for the kinds that come with images, drawn from the same made-up data as ``build_samples``. Every image
    carries the SAMPLE strip, so none can pass for a real one."""
    today = ist_date(now).isoformat()
    return {
        "final": charts.report_charts(_final_context(today), sample=True),
        "prelim": charts.report_charts(_prelim_context(today), sample=True),
        "ideas": charts.ideas_charts(_ideas_report(today, now), sample=True),
        "evening": charts.evening_charts(_evening(today), sample=True),
    }


def marked_caption(caption: str) -> str:
    """A chart caption introduced by the preview banner (the caption of a photo is limited to ``MAX_CAPTION_CHARS``)."""
    return f"{PREVIEW_BANNER}\n{caption}"[:MAX_CAPTION_CHARS]


def marked(messages: list[str]) -> list[str]:
    """Every message of a sample introduced by the preview banner, so no single message can pass for a real one (a message the
    banner pushes over the limit is split again)."""
    out: list[str] = []
    for text in messages:
        full = f"{PREVIEW_BANNER}\n\n{text}"
        out.extend([full] if len(full) <= MAX_MESSAGE_CHARS else split_message(full))
    return out
