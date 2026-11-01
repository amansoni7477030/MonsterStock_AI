"""Server-rendered HTML for the dashboard: one page, no JavaScript, no external assets.

Every dynamic value passes through ``esc`` (``html.escape``); ``Raw`` marks fragments assembled here from
already escaped parts and is the only thing that bypasses escaping. Attribute values are never dynamic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from typing import Any

LOW_N = 30
_MAX_TEXT = 160

CSS = """
:root{color-scheme:light dark}
body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;background:#10161d;color:#d8dee6}
header,main,footer{max-width:1180px;margin:0 auto;padding:0 16px}
header{padding-top:18px}
h1{font-size:22px;margin:0 0 6px}
h2{font-size:17px;margin:26px 0 8px;border-bottom:1px solid #2a3542;padding-bottom:4px}
h3{font-size:14px;margin:14px 0 6px;color:#aab6c3}
table{border-collapse:collapse;width:100%;margin:6px 0 4px;font-size:13px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #222c38;vertical-align:top}
th{color:#9fb0c1;font-weight:600;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#8392a3}
.note{background:#1a2430;border-left:3px solid #4f6d8f;padding:6px 10px;margin:8px 0}
.badge{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;font-weight:600;background:#2a3542}
.ok{background:#1f4d36;color:#9ef0c2}
.warn{background:#5a4616;color:#ffd987}
.bad{background:#5c2328;color:#ffb3ba}
.muted.badge{background:#2a3542;color:#9fb0c1}
.wrap{overflow-x:auto}
footer{padding:18px 16px 30px;color:#8392a3;font-size:12px}
"""

_BADGE_KINDS = frozenset({"ok", "warn", "bad", "muted"})
_STATUS_KIND = {
    "success": "ok", "ok": "ok", "validated": "ok", "active": "ok",
    "running": "warn", "shadow": "warn", "degraded": "warn", "stale": "warn", "candidate": "warn",
    "failed": "bad", "timeout": "bad", "missed": "bad", "failing": "bad", "rejected": "bad", "missing": "bad",
}


class Raw(str):
    """A fragment of HTML that has already been escaped; never build one from untrusted text."""


def esc(value: Any) -> str:
    """HTML-escape any value (``None`` becomes the empty string)."""
    return escape("" if value is None else str(value), quote=True)


def _cell(value: Any) -> str:
    return str(value) if isinstance(value, Raw) else esc(value)


def badge(text: Any, kind: str = "muted") -> Raw:
    css = kind if kind in _BADGE_KINDS else "muted"
    return Raw(f'<span class="badge {css}">{esc(text)}</span>')


def status_badge(status: Any) -> Raw:
    return badge(status, _STATUS_KIND.get(str(status).lower(), "muted"))


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], *, empty: str = "No data yet.") -> Raw:
    body = "".join("<tr>" + "".join(f"<td>{_cell(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    if not body:
        return Raw(f'<p class="muted">{esc(empty)}</p>')
    head = "".join(f"<th>{esc(name)}</th>" for name in headers)
    return Raw(f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


# ------------------------------------------------------------------------------------------- formatting
def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt_pct(value: Any, digits: int = 1) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def fmt_ts(value: Any) -> str:
    return "n/a" if not value else str(value).replace("T", " ")


def fmt_text(value: Any, limit: int = _MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def explain(explanation: Any) -> str:
    """One short line out of a free-form explanation payload (catalyst headline, summary or drivers)."""
    if isinstance(explanation, str):
        return explanation
    if not isinstance(explanation, Mapping):
        return ""
    catalysts = explanation.get("catalysts")
    if isinstance(catalysts, list) and catalysts:
        first = catalysts[0]
        headline = first.get("headline") if isinstance(first, Mapping) else first
        if headline:
            return str(headline)
    for name in ("summary", "headline", "catalyst", "reason"):
        if isinstance(explanation.get(name), str) and explanation[name]:
            return explanation[name]
    drivers = explanation.get("drivers")
    if isinstance(drivers, list) and drivers:
        return ", ".join(str(item) for item in drivers[:3])
    return ""


def _metric_cell(name: str, value: Any) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("hit", "rate", "precision", "recall")):
        return fmt_pct(value)
    return fmt_num(value, 4)


# ---------------------------------------------------------------------------------------------- sections
def _header(page: Mapping[str, Any]) -> str:
    overview = page["overview"]
    heartbeat = overview.get("heartbeat") or {}
    counts = overview.get("table_counts") or {}
    shown = ("predictions", "outcomes", "snapshots", "events", "news_raw", "bars_daily")
    count_text = ", ".join(f"{name}: {counts[name]}" for name in shown if name in counts)
    mode = str(page.get("mode", "shadow"))
    return (
        "<header>"
        "<h1>MONSTER AI</h1>"
        f"<p>Mode {badge(mode.upper(), 'warn' if mode == 'shadow' else 'ok')} "
        f"daemon {status_badge(heartbeat.get('status', 'missing'))} "
        f'<span class="muted">snapshot {esc(fmt_ts(page.get("generated_at")))} UTC; latest predictions '
        f"{esc(overview.get('latest_prediction_date') or 'none')}</span></p>"
        '<p class="note">Research output only. Probabilities are calibrated estimates with uncertainty, not '
        "trading advice; the system never places orders. Shadow mode means results are tracked, not relied upon."
        "</p>"
        f'<p class="muted">{esc(count_text)}</p>'
        "</header>"
    )


def _models_section(models: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        [
            m.get("stage"), m.get("model_version"), status_badge(m.get("status")),
            badge("active", "ok") if m.get("is_active") else "", fmt_ts(m.get("created_at")),
            fmt_ts(m.get("activated_at")), m.get("train_end") or "n/a",
        ]
        for m in models[:12]
    ]
    headers = ["Stage", "Version", "Status", "", "Created (UTC)", "Activated (UTC)", "Train end"]
    return "<h2>Models</h2>" + table(headers, rows, empty="No models registered yet.")


def _prediction_rows(preds: Sequence[Mapping[str, Any]], primary: float) -> list[list[Any]]:
    key = f"hit_{primary:g}"
    rows = []
    for p in preds:
        probs = p.get("p") or {}
        span = p.get("range") or {}
        rows.append([
            p.get("rank"), p.get("symbol"), p.get("tier"), fmt_pct(probs.get(key)), fmt_pct(probs.get("unusual")),
            fmt_num(span.get("q50")), fmt_num(span.get("q90")), fmt_num(p.get("uncertainty")),
            p.get("data_quality"), f"{p.get('model_version')} ({p.get('model_status')})",
            fmt_text(explain(p.get("explanation"))),
        ])
    return rows


def _predictions_section(page: Mapping[str, Any]) -> str:
    data = page["predictions"]
    settings = page.get("settings") or {}
    primary = float(settings.get("primary_threshold", 3.0))
    top_n = int(settings.get("top_n", 5))
    out = [f"<h2>Latest candidates ({esc(data.get('date') or 'no date')})</h2>"]
    preds = [p for p in data.get("predictions", []) if not p.get("is_baseline")]
    if not preds:
        out.append(table([], [], empty="No predictions recorded yet."))
    for stage in ("final", "pre"):
        subset = [p for p in preds if p.get("stage") == stage][:top_n]
        if not subset:
            continue
        label = "Final scan" if stage == "final" else "Pre-scan"
        out.append(f"<h3>{label} - top {len(subset)}</h3>")
        headers = ["#", "Symbol", "Tier", f"P(range > {primary:g}%)", "P(unusual)", "Range q50 %", "Range q90 %",
                   "Uncertainty", "Data", "Model", "Catalyst"]
        out.append(table(headers, _prediction_rows(subset, primary)))
    return "".join(out)


def _performance_section(performance: Mapping[str, Any]) -> str:
    outcomes = performance.get("outcomes") or {}
    out = [
        "<h2>Performance</h2>",
        f'<p class="muted">Validated outcomes: {esc(outcomes.get("validated", 0))} over '
        f'{esc(outcomes.get("days", 0))} trading day(s); provisional: {esc(outcomes.get("provisional", 0))}. '
        f"Every metric shows its sample size n; rows with n &lt; {LOW_N} are not statistically reliable.</p>",
    ]
    scopes = performance.get("scopes") or {}
    if not scopes:
        out.append(table([], [], empty="No evaluation metrics yet; they appear after the first evaluated days."))
    for scope, block in scopes.items():
        out.append(f"<h3>{esc(scope)} (as of {esc(block.get('as_of_date'))})</h3>")
        rows = []
        for m in block.get("metrics", []):
            low = bool(m.get("low_n"))
            low_n = badge(f"n < {LOW_N}", "warn") if low else ""
            interval = (
                f"{_metric_cell(m['name'], m.get('ci_low'))} - {_metric_cell(m['name'], m.get('ci_high'))}"
                if m.get("ci_low") is not None and m.get("ci_high") is not None else "n/a"
            )
            rows.append([
                m.get("group"), m.get("slice"), m.get("name"), _metric_cell(str(m.get("name")), m.get("value")),
                m.get("n") if m.get("n") is not None else "n/a", interval, low_n,
            ])
        out.append(table(["Group", "Slice", "Metric", "Value", "n", "95% CI", ""], rows, empty="No metrics."))
    return "".join(out)


def _jobs_section(jobs: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        [j.get("job_name"), j.get("run_key"), status_badge(j.get("status")), fmt_ts(j.get("started_at")),
         fmt_ts(j.get("finished_at")), j.get("attempt"), fmt_text(j.get("message"))]
        for j in jobs[:25]
    ]
    headers = ["Job", "Run", "Status", "Started (UTC)", "Finished (UTC)", "Attempt", "Message"]
    return "<h2>Recent jobs</h2>" + table(headers, rows, empty="No jobs have run yet.")


def _sources_section(sources: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        [s.get("source_id"), s.get("provider"), s.get("criticality"), status_badge(s.get("status")),
         fmt_ts(s.get("last_success_at")), s.get("last_data_ts") or "n/a", s.get("consecutive_failures"),
         fmt_text(s.get("last_error"), 100)]
        for s in sources
    ]
    headers = ["Source", "Provider", "Criticality", "Status", "Last success (UTC)", "Last data", "Failures",
               "Last error"]
    return "<h2>Feed health</h2>" + table(headers, rows, empty="No data sources registered yet.")


def render_page(page: Mapping[str, Any]) -> str:
    """Render the full dashboard page from the same data the JSON API serves."""
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="robots" content="noindex, nofollow">',
        '<meta http-equiv="refresh" content="120">',
        f"<title>MONSTER AI ({esc(page.get('mode', 'shadow'))})</title>",
        f"<style>{CSS}</style></head><body>",
        _header(page),
        "<main>",
        _models_section(page.get("models", [])),
        _predictions_section(page),
        _performance_section(page.get("performance", {})),
        _jobs_section(page.get("jobs", [])),
        _sources_section(page.get("sources", [])),
        "</main>",
        "<footer>Read-only view. Times are UTC. Source of truth: the local SQLite database.</footer>",
        "</body></html>",
    ]
    return "".join(parts)
