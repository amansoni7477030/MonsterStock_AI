"""Live self-check of the optional AI services: ``monster-ai ai-test`` and the network part of ``doctor``.

``deep=True`` sends ONE made-up news item through the same code path production uses (context block included), so a wrong model
name, a rejected key, a schema the provider refuses or an unreadable answer shows up now and not at 08:10 on a trading day.
``deep=False`` only lists the account's models, which is free (no tokens are spent): it proves the key and shows whether the
configured models exist. The web check has no free probe: ``deep=True`` makes one real search (it takes tens of seconds and bills
a search), ``deep=False`` only reports how it is configured.

Nothing is cached, counted or written: the services run without a store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import requests

from app.clock import Clock
from app.config import Settings
from news_intelligence.context import NewsContext
from news_intelligence.jev import JEV_BASE_URL, JevJudge
from news_intelligence.llm import LLMExtractor
from news_intelligence.remote import RemoteJudge
from news_intelligence.web import WebResearcher

SERVICES = ("llm", "web", "jev")
SAMPLE_TITLE = "Reliance Industries bags Rs 5,000 crore order from NTPC for solar modules"
SAMPLE_TEXT = (
    "Reliance Industries has received a Rs 5,000 crore order from NTPC for supply of solar modules over three years, the "
    "company said in a stock exchange filing on Monday."
)
SAMPLE_EVENT = "order_win"  # what any sound model answers for the sample
WEB_SAMPLE_TITLE = "Reliance in talks to acquire a stake in a media company, sources say"
WEB_SAMPLE_TEXT = (
    "Reliance Industries is in talks to buy a minority stake, two people familiar with the matter said. The company declined "
    "to comment."
)
LIST_TIMEOUT_SEC = 15.0


def sample_context(now: Any) -> NewsContext:
    """A context like production's for the made-up item: a wire story published two hours ago."""
    return NewsContext(
        kind="article", source="Sample Wire", source_class="media", source_tier="wire", category="markets",
        published_at=now - timedelta(hours=2), now=now, mentioned=("Reliance Industries Limited (RELIANCE)",),
    )


@dataclass
class AiCheck:
    service: str  # llm | web | jev
    label: str
    status: str  # ok | warn | fail | off
    detail: str
    seconds: float | None = None

    @property
    def usable(self) -> bool:
        """The service answers (possibly with a remark)."""
        return self.status in ("ok", "warn")

    def to_dict(self) -> dict[str, Any]:
        return {"service": self.service, "label": self.label, "status": self.status, "detail": self.detail,
                "seconds": None if self.seconds is None else round(self.seconds, 2)}


def _explain(reason: str | None, key_name: str) -> str:
    """The recorded failure reason with the usual remedy attached."""
    reason = reason or "no answer"
    if "http 401" in reason or "http 403" in reason:
        return f"{reason}: the key was rejected, check {key_name}"
    if "http 404" in reason:
        return f"{reason}: model or endpoint not found, check the model name"
    if "rate limit" in reason:
        return f"{reason}: the provider's quota is used up, try again later"
    return reason


def _prelude(
    service: str, label: str, judge: RemoteJudge, url: str, key: str | None, key_name: str, wanted: Sequence[str]
) -> tuple[AiCheck | None, list[str]]:
    """List the account's models. Returns a failed check when the service cannot be used at all, otherwise ``None`` and the
    remarks to show (for instance a configured model that the account does not list)."""
    started = judge.clock.monotonic()
    try:
        status, payload, _text, _headers = judge.send("GET", url, key, LIST_TIMEOUT_SEC)
    except (requests.RequestException, OSError) as exc:
        return AiCheck(service, label, "fail", f"cannot reach the service ({type(exc).__name__})"), []
    seconds = judge.clock.monotonic() - started
    if status in (401, 403):
        return AiCheck(service, label, "fail", f"HTTP {status}: the key was rejected, check {key_name}", seconds), []
    rows = payload.get("data", payload.get("models")) if isinstance(payload, dict) else None
    if status != 200 or not isinstance(rows, list):
        return None, [f"model list unavailable (HTTP {status})"]
    names = {str(row.get("id") or row.get("name")) for row in rows if isinstance(row, dict)}
    missing = [name for name in wanted if name not in names]
    return None, [f"not listed for this key: {', '.join(missing)}"] if missing else []


def _verdict(service: str, label: str, remarks: list[str], summary: str, seconds: float | None) -> AiCheck:
    return AiCheck(service, label, "warn" if remarks else "ok", "; ".join([summary, *remarks]), seconds)


def check_llm(
    settings: Settings, *, deep: bool = True, session: Any = None, clock: Clock | None = None, resolver: Any = None
) -> AiCheck:
    if not settings.llm_configured:
        return AiCheck("llm", "LLM", "off", "not configured (set GROQ_API_KEY, or LLM_API_BASE + LLM_MODEL for another provider)")
    groq = settings.llm_provider == "groq"
    label = "Groq LLM" if groq else "LLM (custom provider)"
    key_name = "GROQ_API_KEY" if groq else "LLM_API_KEY"
    judge = LLMExtractor(settings, session=session, clock=clock, resolver=resolver)
    base = (settings.llm_base_url or "").rstrip("/").removesuffix("/chat/completions")
    failed, remarks = _prelude("llm", label, judge, base + "/models", settings.llm_key, key_name, judge.models)
    if failed is not None:
        return failed
    if not deep:
        return _verdict("llm", label, remarks, "key accepted", None)
    started = judge.clock.monotonic()
    result = judge.extract(SAMPLE_TITLE, SAMPLE_TEXT, sample_context(judge.clock.now()))
    seconds = judge.clock.monotonic() - started
    if result is None:
        return AiCheck("llm", label, "fail", _explain(judge.last_error, key_name), seconds)
    if result.event_type != SAMPLE_EVENT:
        remarks.append(f"expected {SAMPLE_EVENT} for the sample, got {result.event_type}")
    if not result.quote_verified:
        remarks.append("its evidence quote was not found in the text (the pipeline ignores such answers)")
    entities = ", ".join(f"{e.id} {e.polarity:+.1f}" for e in result.entities) or "none resolved"
    severity = "-" if result.severity is None else f"{result.severity:.2f}"
    summary = f"model {judge.last_model}: {result.event_type}, severity {severity}, entities {entities}"
    return _verdict("llm", label, remarks, summary, seconds)


def check_web(
    settings: Settings, *, deep: bool = True, session: Any = None, clock: Clock | None = None, resolver: Any = None
) -> AiCheck:
    """The optional web check. A failure is only ever a warning: every item is then judged without it."""
    label = "Groq web check"
    if not settings.llm_configured:
        return AiCheck("web", label, "off", "needs the LLM: set GROQ_API_KEY")
    if not settings.llm_web_models:
        return AiCheck("web", label, "off", "needs Groq with a gpt-oss model: browser_search exists on no other provider or model")
    if not settings.llm_web_search:
        return AiCheck("web", label, "off", "disabled (LLM_WEB_SEARCH=false)")
    if settings.llm_web_max_per_run <= 0:
        return AiCheck("web", label, "off", "disabled (LLM_WEB_MAX_PER_RUN=0)")
    enabled = f"enabled on {', '.join(settings.llm_web_models)}, up to {settings.llm_web_max_per_run} per news job"
    if not deep:
        return AiCheck("web", label, "ok", enabled + " (ai-test makes one real search to prove it)")
    researcher = WebResearcher(settings, session=session, clock=clock, resolver=resolver)
    started = researcher.clock.monotonic()
    brief = researcher.research(WEB_SAMPLE_TITLE, WEB_SAMPLE_TEXT, sample_context(researcher.clock.now()))
    seconds = researcher.clock.monotonic() - started
    if brief is None:
        reason = _explain(researcher.last_error, "GROQ_API_KEY")
        return AiCheck("web", label, "warn", f"{reason}; items are judged without a web check", seconds)
    return AiCheck("web", label, "ok", f"model {researcher.last_model}: {brief.summary()}", seconds)


def check_jev(
    settings: Settings, *, deep: bool = True, session: Any = None, clock: Clock | None = None, resolver: Any = None
) -> AiCheck:
    if not settings.jev_configured:
        return AiCheck("jev", "Jev", "off", "not configured (set TYPESAFE_API_KEY)")
    label = "Jev (TypeSafe AI)"
    judge = JevJudge(settings, session=session, clock=clock, resolver=resolver)
    wanted = [str(settings.jev_model)]
    failed, remarks = _prelude("jev", label, judge, JEV_BASE_URL + "/v1/models", settings.typesafe_api_key, "TYPESAFE_API_KEY", wanted)
    if failed is not None:
        return failed
    if not deep:
        return _verdict("jev", label, remarks, "key accepted", None)
    started = judge.clock.monotonic()
    judgment = judge.judge(SAMPLE_TITLE, SAMPLE_TEXT, company="Reliance Industries Limited", source_kind="media",
                           context=sample_context(judge.clock.now()))
    seconds = judge.clock.monotonic() - started
    if judgment is None:
        return AiCheck("jev", label, "fail", _explain(judge.last_error, "TYPESAFE_API_KEY"), seconds)
    if judgment.event_type != SAMPLE_EVENT:
        remarks.append(f"expected {SAMPLE_EVENT} for the sample, got {judgment.event_type}")
    direction = "n/a" if judgment.polarity is None else f"{judgment.polarity:+.2f}"
    summary = (
        f"model {judgment.model or settings.jev_model}: {judgment.event_type} (p {judgment.event_prob:.2f}, confidence "
        f"{judgment.event_confidence:.2f}), rumour {judgment.rumour:.2f}, denial {judgment.denial:.2f}, severity "
        f"{judgment.severity:.2f}, direction {direction}"
    )
    return _verdict("jev", label, remarks, summary, seconds)


def run_checks(
    settings: Settings, *, only: Sequence[str] | None = None, deep: bool = True, sessions: Mapping[str, Any] | None = None,
    clock: Clock | None = None, resolver: Any = None,
) -> list[AiCheck]:
    """Check the wanted services (default: all), in a fixed order. ``sessions`` injects an HTTP session per service. The web
    check shares the LLM's key, so it is skipped when the LLM check has just failed."""
    sessions = sessions or {}
    checks = {"llm": check_llm, "web": check_web, "jev": check_jev}
    results: list[AiCheck] = []
    for name in SERVICES:
        if only is not None and name not in only:
            continue
        llm_failed = any(check.service == "llm" and check.status == "fail" for check in results)
        if name == "web" and llm_failed:
            results.append(AiCheck("web", "Groq web check", "off", "skipped: the LLM check failed first (it uses the same key)"))
            continue
        results.append(checks[name](settings, deep=deep, session=sessions.get(name), clock=clock, resolver=resolver))
    return results
