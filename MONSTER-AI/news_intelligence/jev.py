"""Optional Jev judge: calibrated judgments on a news item from TypeSafe's System One model.

Jev does not write text. It answers typed *questions* about a *state* (the article) and returns a probability for every option
plus a confidence, so code can act when it is sure and fall back when it is not. That is what the keyword rules of
:mod:`news_intelligence.events` do worst: telling an announced fact from a rumour or a denial, picking the event type of a
free-text headline, and judging how much a piece of news matters. Five atomic questions go out in ONE request (the docs'
"speculative fan-out": extra questions cost almost nothing):

``event_type``  Choice over the project's own taxonomy, with probabilities and confidence
``rumour``      Noul: unconfirmed claim, speculation or hearsay
``denial``      Noul: the text denies or refutes an earlier report
``severity``    Score on a 5-level rubric (how strongly the news could move the share price)
``polarity``    Choice good / bad / neutral for the company in focus (only asked when there is one)

The judgments are *merged into* the rule draft by :func:`merge_jev` under conservative rules (a rumour or denial flag set by
the rules is never cleared, a type is replaced only when Jev is confident and the rules are not), and a result below
``JEV_MIN_CONFIDENCE`` is ignored. Any failure returns ``None`` and the rules decide. Nothing here touches the price model:
it improves the catalysts that the report shows and that a later news model can learn from.

API contract (https://docs.typesafe.ai, OpenAPI at https://api.typesafe.ai/openapi.json): ``POST /v1/systemone`` with
``{"model", "state", "questions"}``; the answer holds ``choice`` / ``score`` / ``noul``, ``probabilities`` and ``confidence`` per
question; 429 and 529 mean "retry later" (``Retry-After``), 401 a bad key.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import requests

from app import __version__
from app.clock import Clock
from app.config import Settings
from app.logging_setup import redact, register_secret
from news_intelligence.events import EVENT_TYPES, EventDraft
from news_intelligence.remote import (
    RETRY_AFTER_CAP_SEC,
    Fatal,
    OutOfTime,
    Quota,
    RemoteJudge,
    clamp,
    prune_prefix,
    retry_after_seconds,
)

log = logging.getLogger(__name__)

JEV_BASE_URL = "https://api.typesafe.ai"
JEV_PATH = "/v1/systemone"
PROMPT_VERSION = "j1"
CACHE_PREFIX = "jev:"
CACHE_TTL_DAYS = 14
MAX_ARTICLE_CHARS = 2500
MAX_HEADLINE_CHARS = 300
MAX_SECONDS_PER_RUN = 180.0  # Jev answers in about a tenth of a second: a run that needs minutes is a provider problem
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
FATAL_STATUS = frozenset({401, 403, 404})

MIN_EVENT_PROB = 0.6  # the chosen event type must carry at least this probability...
RULES_KEEP_CONFIDENCE = 0.75  # ...and the rules keep their own type when they are at least this sure of it
MIN_POLARITY = 0.2  # a direction weaker than this is "unclear", not a signal
RUMOUR_SEVERITY, DENIAL_SEVERITY = 0.85, 0.5  # severity factors of an unconfirmed report / a denial (as in events._severity)

EVENT_DESCRIPTIONS: dict[str, str] = {
    "earnings_result": "Quarterly or annual financial results were announced or reported (profit, revenue, margins).",
    "earnings_guidance": "Management gave or changed a forecast, guidance or outlook for future earnings, volumes or margins.",
    "order_win": "The company won or was awarded a contract, order or tender.",
    "order_loss": "The company lost a contract, order or tender, or one was cancelled, terminated or reduced.",
    "merger_acquisition": "A merger, acquisition, stake purchase or sale, demerger, joint venture or takeover bid.",
    "fundraising": "Capital is being raised: equity issue, QIP, rights issue, preferential allotment, bonds, debt or a loan.",
    "buyback_dividend": "A dividend, bonus issue, buyback or other payout to shareholders.",
    "promoter_transaction": "Promoters, insiders or large holders bought or sold shares or changed their holding.",
    "pledge": "Promoter or insider shares were pledged, invoked or released.",
    "management_change": "A senior management, board or auditor change: someone was appointed, resigned or removed.",
    "rating_action": "A credit rating or an analyst rating or target price was assigned, upgraded or downgraded.",
    "litigation_regulatory": "A court case, regulator action, investigation, penalty, notice, raid or fraud allegation.",
    "government_contract": "A contract, licence, subsidy or incentive from a government body.",
    "board_meeting": "A board meeting date or agenda was announced, with no outcome yet.",
    "corporate_action": "A routine corporate action: record date, stock split, name change, listing, delisting or AGM.",
    "bulk_block_deal": "A bulk deal, block deal or other large trade in the shares.",
    "macro_release": "A scheduled economic data release such as inflation, GDP, PMI, trade, jobs or industrial output.",
    "central_bank": "A central bank decision or statement: policy rate, liquidity or guidance (RBI, Fed, ECB, ...).",
    "geopolitical_conflict": "War, military conflict, a terror attack or a serious diplomatic crisis.",
    "sanctions_tariffs": "Sanctions, tariffs, export bans or import duties were imposed, changed or lifted.",
    "commodity_supply": "A supply or price shock in a commodity: crude oil, gas, metals, farm goods, power or coal.",
    "trade_policy": "Trade agreements or trade policy, import-export rules or supply-chain policy.",
    "elections_politics": "Elections, government formation or other political events.",
    "sector_regulation": "A new law, regulation, tax or policy that affects a whole sector.",
    "global_company": "News about a large foreign company (results, outlook, an event) that may affect Indian peers or suppliers.",
    "market_stress": "Broad market stress: a sharp index fall, volatility spike, liquidity or banking scare, or a crash.",
    "other": "None of the above, or the text reports no market-relevant event.",
}
SEVERITY_LEVELS = (
    "No price relevance: routine, repeated, old or purely promotional information.",
    "Minor: small, expected or already known information.",
    "Moderate: a clear but ordinary development.",
    "Major: a surprise or a large change in earnings, orders, ownership, regulation or outlook.",
    "Extreme: market-moving, such as a takeover, fraud, default, trading halt, or a central-bank or war shock.",
)
POLARITY_CRITERIA = {
    "positive": "Good for the company's share price: the economic consequence is favourable.",
    "negative": "Bad for the company's share price: the economic consequence is unfavourable.",
    "neutral": "Neutral or unclear: no clear consequence for the share price, or mixed.",
}
SOURCE_LABELS = {
    "exchange_filing": "exchange filing", "official": "official statement", "company": "company statement",
    "media": "news media", "social": "social media",
}
RUMOUR_QUESTION = (
    "The text reports something unconfirmed: a rumour, speculation, an anonymous-source claim or a mere possibility "
    "('may', 'likely to', 'in talks', 'sources say') rather than a confirmed fact or an official announcement."
)
DENIAL_QUESTION = "The text denies, refutes or dismisses an earlier report, rumour or market speculation."
SEVERITY_QUESTION = (
    "How strongly could this news move the share price of the company in focus (or the market when no company is in focus) "
    "on the next trading day?"
)
POLARITY_QUESTION = (
    "What does this news mean for the share price of the company in focus? Judge the economic consequence, not the tone of "
    "the wording."
)
EVENT_QUESTION = (
    "What kind of event does the text report? Pick the closest category; use 'other' when none fits or when the text reports "
    "no market-relevant event."
)


@dataclass
class JevJudgment:
    """The calibrated answers for one item. Probabilities are 0..1; ``severity`` is the expected rubric level scaled to 0..1
    and ``polarity`` is P(good) - P(bad). Confidences follow Jev's own definition (0 = spread out, 1 = certain)."""

    event_type: str
    event_prob: float
    event_confidence: float
    rumour: float
    denial: float
    severity: float
    severity_confidence: float
    polarity: float | None = None
    polarity_confidence: float | None = None
    model: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type, "event_prob": self.event_prob, "event_confidence": self.event_confidence,
            "rumour": self.rumour, "denial": self.denial, "severity": self.severity,
            "severity_confidence": self.severity_confidence, "polarity": self.polarity,
            "polarity_confidence": self.polarity_confidence, "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Any, *, cached: bool = False) -> JevJudgment | None:
        """Rebuild from the cache; ``None`` when the entry is malformed (it is then simply asked again)."""
        if not isinstance(data, dict) or data.get("event_type") not in EVENT_DESCRIPTIONS:
            return None
        numbers = {}
        for name in ("event_prob", "event_confidence", "rumour", "denial", "severity", "severity_confidence"):
            number = clamp(data.get(name), 0.0, 1.0)
            if number is None:
                return None
            numbers[name] = number
        polarity = clamp(data.get("polarity"), -1.0, 1.0)
        return cls(
            event_type=str(data["event_type"]), polarity=polarity,
            polarity_confidence=clamp(data.get("polarity_confidence"), 0.0, 1.0) if polarity is not None else None,
            model=str(data.get("model") or "")[:60], cached=cached, **numbers,
        )


@dataclass
class JevEffect:
    """What a judgment changed in a rule draft (for the run statistics and the tests)."""

    retyped: bool = False
    rumour_flagged: bool = False
    denial_flagged: bool = False
    severity_blended: bool = False
    polarity: float | None = None  # a usable direction for the company in focus, else None


def prune_jev_cache(store: Any, now: Any) -> int:
    """Delete Jev cache entries older than the TTL (works without a judge instance)."""
    return prune_prefix(store, now, CACHE_PREFIX, CACHE_TTL_DAYS)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def build_request(model: str, title: str, text: str, *, company: str | None = None, source_kind: str | None = None) -> dict[str, Any]:
    """The ``/v1/systemone`` body: the article as a named-field state and the atomic questions (the content never mixes with
    the questions, so a sentence inside an article cannot act as an instruction)."""
    state: dict[str, Any] = {"headline": (title or "")[:MAX_HEADLINE_CHARS], "article": (text or "")[:MAX_ARTICLE_CHARS]}
    if source_kind in SOURCE_LABELS:
        state["source_type"] = SOURCE_LABELS[str(source_kind)]
    if company:
        state["company_in_focus"] = company[:120]
    questions: dict[str, Any] = {
        "event_type": {"type": "choice", "instructions": EVENT_QUESTION, "criteria": dict(EVENT_DESCRIPTIONS)},
        "rumour": {"type": "noul", "instructions": RUMOUR_QUESTION},
        "denial": {"type": "noul", "instructions": DENIAL_QUESTION},
        "severity": {"type": "score", "instructions": SEVERITY_QUESTION, "criteria": list(SEVERITY_LEVELS)},
    }
    if company:
        questions["polarity"] = {"type": "choice", "instructions": POLARITY_QUESTION, "criteria": dict(POLARITY_CRITERIA)}
    return {"model": model, "state": state, "questions": questions}


def _answer(answers: dict[str, Any], name: str, kind: str) -> dict[str, Any]:
    item = answers.get(name)
    if not isinstance(item, dict) or item.get("type") != kind:
        raise ValueError(f"no {kind} answer for {name}")
    return item


def _chosen(item: dict[str, Any], options: Any) -> tuple[str, float, float]:
    """(choice, its probability, confidence) of a Choice answer, checked against the options that were offered."""
    choice, probabilities = item.get("choice"), item.get("probabilities")
    if not isinstance(choice, str) or choice not in options or not isinstance(probabilities, dict):
        raise ValueError("choice is not one of the options")
    probability = clamp(probabilities.get(choice), 0.0, 1.0)
    confidence = clamp(item.get("confidence"), 0.0, 1.0)
    if probability is None or confidence is None:
        raise ValueError("probability or confidence is not a number")
    return choice, probability, confidence


def parse_response(payload: Any, *, company_in_focus: bool) -> JevJudgment:
    """Validate a ``/v1/systemone`` response. Raises ``ValueError`` for anything that is not exactly what was asked: every
    answer must exist with the right type, a choice must be one of the options offered, every number must be finite."""
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise ValueError("no answers")
    answers = payload["answers"]
    event, event_prob, event_conf = _chosen(_answer(answers, "event_type", "choice"), EVENT_DESCRIPTIONS)
    rumour, denial = (clamp(_answer(answers, name, "noul").get("noul"), 0.0, 1.0) for name in ("rumour", "denial"))
    score = _answer(answers, "severity", "score")
    level = clamp(score.get("score"), 0.0, len(SEVERITY_LEVELS) - 1.0)
    severity_conf = clamp(score.get("confidence"), 0.0, 1.0)
    if rumour is None or denial is None or level is None or severity_conf is None:
        raise ValueError("a probability or score is not a number")
    polarity = polarity_conf = None
    if company_in_focus:
        side, _, polarity_conf = _chosen(_answer(answers, "polarity", "choice"), POLARITY_CRITERIA)
        probabilities = answers["polarity"]["probabilities"]
        good, bad = clamp(probabilities.get("positive"), 0.0, 1.0), clamp(probabilities.get("negative"), 0.0, 1.0)
        if good is None or bad is None:
            raise ValueError("polarity probabilities are not numbers")
        polarity = round(good - bad, 3)
    return JevJudgment(
        event_type=event, event_prob=round(event_prob, 3), event_confidence=round(event_conf, 3), rumour=round(rumour, 3),
        denial=round(denial, 3), severity=round(level / (len(SEVERITY_LEVELS) - 1), 3), severity_confidence=round(severity_conf, 3),
        polarity=polarity, polarity_confidence=None if polarity_conf is None else round(polarity_conf, 3),
        model=str(payload.get("model") or "")[:60],
    )


def merge_jev(draft: EventDraft, judgment: JevJudgment, *, min_confidence: float, official: bool) -> JevEffect:
    """Fold a judgment into the rule draft (in place) and say what it changed. Conservative by construction:

    * the event type is replaced only when Jev is confident (probability >= 0.6, confidence >= the setting) and the rules are
      not (their own confidence below 0.75, or no type at all); when both agree the rules' confidence is lifted;
    * a rumour or denial flag is *added* when Jev is sure (never for exchange filings or official statements, never cleared),
      with the same severity / direction discounts the rules apply to those;
    * severity is the mean of the rules' and Jev's when Jev is confident;
    * a direction for the company in focus (halved for a rumour or denial) is returned for the caller to apply, and only when
      it is clear and confident.
    """
    effect = JevEffect()
    yes = 0.5 + min_confidence / 2.0  # a Noul has no confidence of its own: |2p - 1| >= min_confidence is p >= this
    if judgment.event_type != "other" and judgment.event_prob >= MIN_EVENT_PROB and judgment.event_confidence >= min_confidence:
        if judgment.event_type == draft.event_type:
            draft.confidence = round(max(draft.confidence, min(0.95, judgment.event_prob)), 3)
        elif draft.event_type == "other" or (draft.confidence < RULES_KEEP_CONFIDENCE and judgment.event_prob > draft.confidence):
            draft.event_type = judgment.event_type
            draft.scope, draft.polarity_hint = EVENT_TYPES[judgment.event_type].scope, 0.0
            draft.confidence = round(min(0.95, judgment.event_prob), 3)
            effect.retyped = True
    if not official:  # the same discounts the rules apply (events.classify_rules): a denial halves, a rumour trims the severity
        if judgment.denial >= yes and not draft.is_denial:
            draft.severity = round(draft.severity * (DENIAL_SEVERITY / RUMOUR_SEVERITY if draft.is_rumour else DENIAL_SEVERITY), 3)
            if not draft.is_rumour:  # a rumour's direction is already halved
                draft.polarity_hint = round(draft.polarity_hint * 0.5, 3)
            draft.is_denial, draft.is_rumour = True, False
            effect.denial_flagged = True
        elif judgment.rumour >= yes and not draft.is_rumour and not draft.is_denial:
            draft.is_rumour = True
            draft.severity = round(draft.severity * RUMOUR_SEVERITY, 3)
            draft.confidence = round(draft.confidence * 0.8, 3)
            draft.polarity_hint = round(draft.polarity_hint * 0.5, 3)
            effect.rumour_flagged = True
    if judgment.severity_confidence >= min_confidence:
        draft.severity = round(0.5 * draft.severity + 0.5 * judgment.severity, 3)
        effect.severity_blended = True
    if (judgment.polarity is not None and (judgment.polarity_confidence or 0.0) >= min_confidence
            and abs(judgment.polarity) >= MIN_POLARITY):
        effect.polarity = round(judgment.polarity * (0.5 if draft.is_rumour or draft.is_denial else 1.0), 3)
    return effect


class JevJudge(RemoteJudge):
    """Validated Jev judgments with retries, a per-run budget, a result cache and health counters."""

    CACHE_PREFIX = CACHE_PREFIX
    COUNTER = "jev"
    CACHE_TTL_DAYS = CACHE_TTL_DAYS

    def __init__(
        self,
        settings: Settings,
        *,
        session: Any = None,
        clock: Clock | None = None,
        store: Any = None,
        resolver: Any = None,
        max_attempts: int = 3,
        backoff_base: float = 1.0,
        backoff_cap: float = 8.0,
        max_consecutive_failures: int = 5,
    ) -> None:
        super().__init__(clock=clock, store=store, resolver=resolver, max_consecutive_failures=max_consecutive_failures,
                         max_seconds=MAX_SECONDS_PER_RUN)
        self.settings = settings
        self.session = session
        self.max_attempts = max(1, max_attempts)
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        if settings.typesafe_api_key:
            register_secret(settings.typesafe_api_key)

    @property
    def configured(self) -> bool:
        return self.settings.jev_configured

    @property
    def min_confidence(self) -> float:
        return float(self.settings.jev_min_confidence)

    def _cache_key(self, title: str, text: str, company: str | None, source_kind: str | None) -> str:
        raw = f"{PROMPT_VERSION}|{self.settings.jev_model}|{_norm(title)}|{_norm(text)[:MAX_ARTICLE_CHARS]}|{company or ''}|{source_kind or ''}"
        return CACHE_PREFIX + hashlib.sha256(raw.encode()).hexdigest()[:40]

    def _url(self) -> str:
        return JEV_BASE_URL + JEV_PATH

    def _post_once(self, body: dict[str, Any]) -> tuple[int, Any, dict[str, str]]:
        if self.session is None:
            self.session = requests.Session()
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": f"monster-ai/{__version__}"}
        if self.settings.typesafe_api_key:
            headers["Authorization"] = "Bearer " + self.settings.typesafe_api_key
        response = self.session.post(self._url(), headers=headers, json=body, timeout=float(self.settings.jev_timeout_sec))
        try:
            payload = response.json()
        except (ValueError, AttributeError):
            payload = None
        return int(response.status_code), payload, dict(getattr(response, "headers", {}) or {})

    def _request(self, body: dict[str, Any]) -> Any:
        """POST with bounded retries; returns the decoded response or raises ``Fatal`` / ``Quota`` / ``OutOfTime`` /
        ``RuntimeError``. 429 and 529 are "retry later": a short ``Retry-After`` is slept through, a long one is a quota."""
        last = "no attempt"
        attempt = 0
        while attempt < self.max_attempts:
            if self.out_of_time:
                raise OutOfTime()
            try:
                status, payload, headers = self._post_once(body)
            except (requests.RequestException, OSError) as exc:
                last = f"transport error {type(exc).__name__}"
                status, payload, headers = -1, None, {}
            if status == 200:
                usage = payload.get("usage") if isinstance(payload, dict) else None
                if isinstance(usage, dict):
                    self._count_tokens(usage.get("input_tokens"), usage.get("output_tokens"))
                return payload
            if status in FATAL_STATUS:
                raise Fatal(f"http {status}")
            if status in (429, 529):
                wait = retry_after_seconds(headers)
                if wait is not None and wait > RETRY_AFTER_CAP_SEC:
                    raise Quota(wait)
            if status != -1 and status not in RETRYABLE_STATUS:
                raise RuntimeError(f"http {status}")
            if status != -1:
                last = f"http {status}"
            attempt += 1
            if attempt < self.max_attempts:
                delay = min(self.backoff_cap, self.backoff_base * 2 ** (attempt - 1))
                self.sleep_within_budget(max(delay, min(retry_after_seconds(headers) or 0.0, RETRY_AFTER_CAP_SEC)))
        raise RuntimeError(f"gave up after {self.max_attempts} attempts ({last})")

    def judge(self, title: str, text: str = "", *, company: str | None = None, source_kind: str | None = None) -> JevJudgment | None:
        """The judgment for one item, or ``None`` (not configured, budget spent, circuit open, any failure)."""
        if not self.configured:
            return None
        title, text = (title or "").strip(), (text or "").strip()
        if not (title or text):
            return None
        company = (company or "").strip() or None
        key = self._cache_key(title, text, company, source_kind)
        cached = JevJudgment.from_dict(self._cache_read(key), cached=True)
        if cached is not None:
            self.run_cache_hits += 1
            return cached
        if self.tripped:
            return None
        if self.cooldown_active():
            self.tripped = True
            return None
        if self.run_calls >= int(self.settings.jev_max_items_per_run) or self.out_of_time:
            self.run_budget_skipped += 1
            return None
        self.run_calls += 1
        self._bump("jev.calls")
        try:
            payload = self._request(build_request(str(self.settings.jev_model), title, text, company=company, source_kind=source_kind))
            judgment = parse_response(payload, company_in_focus=company is not None)
        except Fatal as exc:
            self._failed(f"fatal: {exc}", trip=True)
            return None
        except Quota as exc:
            self._failed(f"rate limit: {exc}", trip=True)
            self.start_cooldown(exc.seconds)
            return None
        except OutOfTime:
            self.run_calls -= 1
            self.run_budget_skipped += 1
            self.tripped = True
            return None
        except Exception as exc:  # the judge must never lose an item: any failure falls back to the rules
            self._failed(redact(f"{type(exc).__name__}: {exc}"))
            return None
        self.consecutive_failures = 0
        self._cache_write(key, judgment.to_dict())
        return judgment
