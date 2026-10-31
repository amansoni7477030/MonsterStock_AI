"""Walk-forward backtest report (blueprint section 15).

The evaluation is the same purged, embargoed walk-forward used for training: every probability is out-of-sample, calibration is
fitted on earlier data than the holdout, and the model's daily top-N is compared with simple baselines (ATR, 20-day range,
volume, gap, random) using a paired day-block bootstrap. Nothing is registered or activated; the report states honestly whether
an edge over baselines exists. Survivorship bias (today's index membership applied to history) and vendor-data quality are
documented limitations, not hidden.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from app import jsonutil
from app.clock import get_clock
from app.config import Settings
from automation.calendar import ExchangeCalendar, seed_calendar
from database.store import Store

log = logging.getLogger(__name__)

LIMITATIONS = (
    "Universe = today's F&O + Nifty 100 membership applied to history (survivorship bias: delisted/removed names are absent).",
    "Prices are vendor-adjusted (Yahoo) before official bhavcopy history exists; corporate-action errors are possible.",
    "Costs are modelled estimates (brokerage, statutory charges, slippage by liquidity), not realised fills.",
    "A large intraday range is not a tradable edge by itself: the report measures range prediction, not profitability.",
)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown(stage: str, result: Any, years: int) -> str:
    gates = result.gates
    ranking = result.metrics.get("ranking", {})
    heads = result.metrics.get("heads", {})
    lines = [f"## Stage `{stage}` - status **{result.status.upper()}** ({years} years of history)", "",
             f"*Reason:* {gates.get('reason')}", "",
             f"- Holdout days: {gates.get('holdout_days')} (required {gates.get('limits', {}).get('min_days')})",
             f"- Calibration error (ECE): {_fmt(gates.get('ece'))} (limit {gates.get('limits', {}).get('max_ece')})",
             f"- Brier skill vs base rate: {_fmt(gates.get('brier_skill'))}",
             f"- Best simple baseline: {gates.get('best_baseline')}", ""]
    boot = gates.get("vs_best_baseline") or {}
    lines += [f"Model minus best baseline (daily top-{ranking.get('top_n')} hit rate): mean {_fmt(boot.get('mean_diff'))}, "
              f"10th percentile {_fmt(boot.get('p10'))}, 90th percentile {_fmt(boot.get('p90'))}", "",
              "| scorer | hit rate | 95% CI | mean range % | picks |", "|---|---|---|---|---|"]
    for name, row in (ranking.get("scorers") or {}).items():
        lines.append(f"| {name} | {_fmt(row.get('hit_rate'))} | {_fmt(row.get('ci_low'))} - {_fmt(row.get('ci_high'))} | "
                     f"{_fmt(row.get('mean_range'), 2)} | {row.get('n_selected')} |")
    lines += ["", "| head | AUC | Brier | skill | ECE | base rate |", "|---|---|---|---|---|---|"]
    for name, row in heads.items():
        lines.append(f"| {name} | {_fmt(row.get('auc'))} | {_fmt(row.get('brier'))} | {_fmt(row.get('brier_skill'))} | "
                     f"{_fmt(row.get('ece'))} | {_fmt(row.get('base_rate'))} |")
    return "\n".join(lines)


def run_backtest(settings: Settings, store: Store, *, stages: tuple[str, ...], years: int, out_dir: Path | None = None) -> int:
    from features.live import build_training_features
    from models.training import TrainConfig, TrainingError, train_stage

    seed_calendar(store, settings)
    calendar = ExchangeCalendar(store)
    end = calendar.last_completed_session(get_clock().now())
    target = out_dir or (settings.data_dir / "reports")
    target.mkdir(parents=True, exist_ok=True)
    stamp = end.isoformat()
    sections: list[str] = [f"# MONSTER AI walk-forward backtest (data through {stamp})", ""]
    summary: dict[str, Any] = {"through": stamp, "years": years, "stages": {}}
    exit_code = 0
    for stage in stages:
        try:
            fs = build_training_features(store, settings, calendar, stage=stage, years=years, end=end)
            cfg = TrainConfig(
                stage=stage, thresholds=settings.range_thresholds, primary_threshold=settings.primary_threshold,
                unusual_k=settings.unusual_k, top_n=settings.top_n, gate_min_days=settings.gate_min_days,
                gate_max_ece=settings.gate_max_ece, gate_min_brier_skill=settings.gate_min_brier_skill,
            )
            with tempfile.TemporaryDirectory(dir=str(settings.tmp_dir)) as scratch:
                result = train_stage(fs, cfg, Path(scratch), now=get_clock().now())
        except TrainingError as exc:
            sections.append(f"## Stage `{stage}`\n\nNot evaluated: {exc}\n")
            summary["stages"][stage] = {"error": str(exc)}
            exit_code = 1
            continue
        sections.append(render_markdown(stage, result, years))
        summary["stages"][stage] = {"status": result.status, "gates": result.gates, "metrics": result.metrics}
        print(f"{stage}: {result.status} - {result.gates.get('reason')}")
    sections += ["", "### Limitations", *[f"- {item}" for item in LIMITATIONS], ""]
    (target / f"backtest-{stamp}.md").write_text("\n".join(sections), encoding="utf-8")
    (target / f"backtest-{stamp}.json").write_text(json.dumps(jsonutil.to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"report written to {target / f'backtest-{stamp}.md'}")
    return exit_code
