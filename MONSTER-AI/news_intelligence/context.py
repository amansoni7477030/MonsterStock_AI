"""Point-in-time facts about one news item that both AI readers receive next to its text.

A reader that sees only a headline cannot tell this morning's filing from last week's rumour, cannot resolve "on Monday" or
"yesterday", and does not know whether the source speaks first-hand. :class:`NewsContext` carries those facts to the LLM (as a
prompt block) and to Jev (as extra fields of its state). :class:`WebBrief` is what the optional web check found out about the
item; it travels the same way.

The context holds only what was knowable when the item was published (who said it, when, about which company) plus the web
brief. It never holds prices, outcomes or the verdict of the rule classifier: the AI is meant to read the text independently of
the rules (the merge then compares the two readings), and a price or a hindsight fact would let it explain a move instead of
reading the news.

Everything that reaches a prompt here is untrusted text (feed names, search results): it is single-line, length-capped and
stripped of control characters and of the tags that structure the prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from app.clock import IST, UTC

SOURCE_LABELS = {
    "exchange_filing": "exchange filing", "official": "official statement", "company": "company statement",
    "media": "news media", "social": "social media",
}
TIER_LABELS = {"wire": "wire service", "business_media": "business media", "aggregator": "aggregator relaying a publisher"}
CONFIRMED = ("yes", "no", "unclear")
SESSION_NOTE = "NSE regular session 09:15-15:30 IST, Monday to Friday except exchange holidays"

MAX_FIELD_CHARS = 140
MAX_MENTIONED = 8
MAX_FACTS = 3
MAX_FACT_CHARS = 180
MAX_WEB_SOURCES = 5
MAX_OUTLETS = 10
PUBLISHED_YEARS = (1990, 2100)  # a publication time outside this is a feed's placeholder, not a date to show a model

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
_LINK = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_CLOSING_TAG = re.compile(r"</(article|context)\s*>", re.IGNORECASE)
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST = re.compile(rf"{_LABEL}(?:\.{_LABEL})*\.[a-z]{{2,24}}")
_INVISIBLE = re.compile(r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")  # zero-width and bidi characters hide words
_DIRECTIVE = (
    r"(?:ignore|disregard|forget|override|bypass|overrule|discard|do not follow|don't follow|stop following)"
)
_PROMPT_NOUN = r"(?:instructions?|prompts?|directives?|system messages?|guidelines)"
_QUALIFIER = r"(?:all|any|the|these|those|your|my|our|previous|prior|above|earlier|preceding|former|original|initial|every)"
# What an attacker writes to an AI that reads the article, and a news article never contains. Each pattern is deliberately narrow: a
# false alarm only withholds one item from the AI services (the rules still read it), while a miss lets a text dictate an answer.
INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an order to ignore earlier instructions",
     re.compile(rf"\b{_DIRECTIVE}\W+(?:{_QUALIFIER}\W+){{1,4}}{_PROMPT_NOUN}\b")),
    ("a role marker (system / assistant)",
     re.compile(r"(?:^|\n|[.!?]\s)\s*(?:system|assistant|developer|admin(?:istrator)?)\s*(?:prompt|message)?\s*:", re.MULTILINE)),
    ("a chat-template token", re.compile(r"<\|[a-z_ ]{1,24}\|>|\[/?(?:inst|sys)\]|<<\s*/?sys\s*>>|###\s*(?:instruction|system)")),
    ("a dictated answer",
     re.compile(r"\b(?:report|set|output|return|respond|reply|answer|assign|mark|rate|classify|label|score|rank)\b[^.\n]{0,30}"
                r"\b(?:severity|polarity|confidence|event[_ ]type|evidence[_ ]quote)\b\W{0,4}(?:to|as|of|=|:)?\W{0,4}"
                r"(?:[-+]?\d|merger|acquisition|positive|negative|bullish|bearish|other)")),
    ("a prompt-block tag", re.compile(r"</?\s*(?:article|context)\s*>")),
    ("a request addressed to an AI",
     re.compile(r"\b(?:as an? (?:ai|language model|assistant)|you are (?:now )?(?:an? )?(?:ai|assistant|language model|chatbot)|"
                r"new instructions?\s*:|respond (?:only )?with (?:the )?json)\b")),
)


def injection_reason(*texts: Any) -> str | None:
    """What makes the text look like an attempt to instruct the AI that reads it (see ``INJECTION_PATTERNS``), or ``None``. Used
    to withhold such an item from the AI services and to drop such "facts" from a web brief."""
    blob = "\n".join(str(text) for text in texts if text)
    if not blob:
        return None
    blob = unicodedata.normalize("NFKC", _INVISIBLE.sub("", blob)).lower()
    blob = re.sub(r"[ \t\r\f\v]+", " ", blob)
    for label, pattern in INJECTION_PATTERNS:
        if pattern.search(blob):
            return label
    return None


def defang(text: str) -> str:
    """``text`` with the closing tags of the prompt blocks neutralised, so a sentence inside it cannot end the block it sits in."""
    return _CLOSING_TAG.sub(r"< /\1>", text)


def one_line(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    """``value`` as single-line text without control characters or prompt tags, at most ``limit`` characters long."""
    text = defang(_CONTROL.sub(" ", "" if value is None else str(value)))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def host_of(value: Any) -> str | None:
    """The lower-case host name in a URL or bare domain (``www.`` dropped), or ``None`` when it is not a plausible host."""
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if "//" in raw:
        try:
            raw = urlsplit(raw).hostname or ""
        except ValueError:
            return None
    raw = raw.split("/")[0].split(":")[0].removeprefix("www.")
    return raw if _HOST.fullmatch(raw) else None


def _count(value: Any, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    return max(0, min(high, int(value)))


def _aware(moment: datetime | None) -> datetime | None:
    """Instants are UTC in this project: a naive one is read as UTC, never as the zone of the machine that runs the code."""
    if moment is None or moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=UTC)


def stamp(moment: datetime) -> str:
    """An instant as exchange time for a human or a model: ``Mon 02 Nov 2026 08:12 IST``."""
    return moment.astimezone(IST).strftime("%a %d %b %Y %H:%M") + " IST"


def age_text(hours: float) -> str:
    if hours < 1.0:
        return f"{max(1, round(hours * 60))} minutes"
    return f"{hours:.1f} hours" if hours < 48.0 else f"{hours / 24.0:.1f} days"


@dataclass(frozen=True)
class WebBrief:
    """What a web search found about one item. Built only through :meth:`from_raw`, which treats the model's reply as untrusted:
    the verdict must be one of the allowed words, counts are bounded integers, flags must be real booleans, facts are short
    single lines without links and sources must be plausible host names."""

    confirmed: str = "unclear"  # do independent outlets report the same facts: yes | no | unclear
    outlets: int = 0  # how many independent outlets were found
    denial_found: bool = False  # the company or an authority denied or corrected the report
    earlier_coverage: bool = False  # the same event was public well before this item (old news)
    facts: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()  # host names the search relied on (the researcher keeps only hosts the search actually returned)
    denial_verified: bool = False  # a reputable page the search returned contains denial wording; set by the researcher, never by the model
    cached: bool = field(default=False, compare=False)

    @classmethod
    def from_raw(cls, raw: Any, *, cached: bool = False) -> WebBrief | None:
        """A brief from a reply of the search model (untrusted) or, with ``cached=True``, from our own cache. ``denial_verified``
        is read from the cache only: a reply that claims it is ignored."""
        if not isinstance(raw, dict):
            return None
        confirmed = str(raw.get("confirmed") or "").strip().lower()
        facts: list[str] = []
        for item in raw.get("facts") if isinstance(raw.get("facts"), list) else ():
            text = one_line(_LINK.sub("", item), MAX_FACT_CHARS) if isinstance(item, str) else ""
            if text and injection_reason(text) is None and len(facts) < MAX_FACTS:  # a page that instructs the reader is not a fact
                facts.append(text)
        sources: list[str] = []
        for item in raw.get("sources") if isinstance(raw.get("sources"), list) else ():
            host = host_of(item) if isinstance(item, str) else None
            if host and host not in sources and len(sources) < MAX_WEB_SOURCES:
                sources.append(host)
        return cls(
            confirmed=confirmed if confirmed in CONFIRMED else "unclear",
            outlets=_count(raw.get("independent_outlets"), MAX_OUTLETS),
            denial_found=raw.get("denial_found") is True,
            earlier_coverage=raw.get("earlier_coverage") is True,
            facts=tuple(facts), sources=tuple(sources),
            denial_verified=cached and raw.get("denial_verified") is True, cached=cached,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed": self.confirmed, "independent_outlets": self.outlets, "denial_found": self.denial_found,
            "earlier_coverage": self.earlier_coverage, "facts": list(self.facts), "sources": list(self.sources),
            "denial_verified": self.denial_verified,
        }

    @property
    def refutes(self) -> bool:
        """A denial or correction was found, the facts were not confirmed, and a reputable page that the search actually returned
        and that the reply cites contains denial wording: the only outcome that changes a draft by itself (see
        :func:`news_intelligence.events.apply_refutation`)."""
        return self.denial_found and self.denial_verified and self.confirmed != "yes" and bool(self.sources)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    def summary(self) -> str:
        denial = "no" if not self.denial_found else ("yes (seen on a reputable page)" if self.denial_verified else "yes (unverified)")
        text = (
            f"independent outlets report the same facts: {self.confirmed}" + (f" ({self.outlets} found)" if self.outlets else "")
            + f"; denial or correction found: {denial}"
            + f"; earlier coverage of the same event: {'yes' if self.earlier_coverage else 'no'}"
        )
        if self.facts:
            text += ". Facts found: " + " | ".join(self.facts)
        if self.sources:
            text += ". Sources: " + ", ".join(self.sources)
        return text


@dataclass(frozen=True)
class NewsContext:
    """The facts about one item that help a reader interpret it. Every field is optional: what is unknown is left out."""

    kind: str = "article"  # article | filing
    source: str | None = None  # publisher or exchange as named by the feed
    domain: str | None = None  # host of the item's URL
    source_class: str | None = None  # exchange_filing | official | company | media | social | unknown
    source_tier: str | None = None  # wire | business_media | aggregator ... (a finer split of ``media``)
    category: str | None = None  # the exchange's filing category or the feed's category
    published_at: datetime | None = None
    now: datetime | None = None  # the moment of this check
    company: str | None = None  # the company in focus (the filer, or the most prominent company)
    symbol: str | None = None
    sector: str | None = None
    mentioned: tuple[str, ...] = ()  # ``Name (SYMBOL)`` of the companies the entity matcher found in the text
    web: WebBrief | None = None

    def __post_init__(self) -> None:
        # Both instants become timezone-aware UTC (a naive one would otherwise break the age and be read in the machine's zone),
        # and a publication time that cannot be real is dropped instead of being shown to a model.
        published = _aware(self.published_at)
        if published is not None and not PUBLISHED_YEARS[0] <= published.year <= PUBLISHED_YEARS[1]:
            published = None
        object.__setattr__(self, "published_at", published)
        object.__setattr__(self, "now", _aware(self.now))

    def with_web(self, brief: WebBrief | None) -> NewsContext:
        return replace(self, web=brief)

    def age_hours(self) -> float | None:
        """Hours between publication and the check, ``None`` when either is unknown or the clocks disagree."""
        if self.published_at is None or self.now is None:
            return None
        hours = (self.now - self.published_at).total_seconds() / 3600.0
        return hours if hours >= 0.0 else None

    def fingerprint(self) -> str:
        """What of this context can change an answer and stays the same between runs: it goes into the cache keys. Everything the
        prompt shows is in (the publication minute, the source, the matched companies, the web brief...) except the time of the
        check and the age, which would make a cache useless."""
        published = self.published_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ") if self.published_at else ""
        web = self.web.digest() if self.web is not None else ""
        parts = (self.kind, self.source_class or "", self.source_tier or "", self.domain or "", one_line(self.source or "", 80),
                 self.category or "", published, self.company or "", self.sector or "", ";".join(self.mentioned), web)
        return "|".join(parts)

    def _source_text(self) -> str:
        label = SOURCE_LABELS.get(self.source_class or "", "")
        tier = TIER_LABELS.get(self.source_tier or "", "")
        kind = ", ".join(part for part in (label, tier) if part)
        name = one_line(self.source or "", 80)
        host = one_line(self.domain or "", 80)
        who = f"{name} ({host})" if name and host and host != name.lower() else (name or host)
        return f"{who}, {kind}" if who and kind else (who or kind)

    def lines(self) -> list[str]:
        """The facts as prompt lines for the LLM (the user message wraps them in a ``<context>`` block)."""
        out = ["Item: " + ("exchange filing" if self.kind == "filing" else "news article")]
        source = self._source_text()
        if source:
            out.append("Source: " + source)
        if self.published_at is not None:
            age = self.age_hours()
            out.append("Published: " + stamp(self.published_at) + (f" ({age_text(age)} before this check)" if age is not None else ""))
        if self.now is not None:
            out.append(f"Check time: {stamp(self.now)}. {SESSION_NOTE}.")
        if self.category:
            out.append("Category: " + one_line(self.category, 80))
        if self.company:
            sector = f"; sector: {one_line(self.sector, 60)}" if self.sector else ""
            out.append("Company in focus: " + one_line(self.company, 90) + sector)
        elif self.sector:
            out.append("Sector: " + one_line(self.sector, 60))
        if self.mentioned:
            out.append("Companies our entity matcher found in the text: " + "; ".join(one_line(m, 90) for m in self.mentioned[:MAX_MENTIONED]))
        if self.web is not None:
            out.append("Web check (an automated search made just now; it may be wrong): " + self.web.summary())
        return out

    def state(self) -> dict[str, Any]:
        """The same facts as extra fields of Jev's ``state`` (``headline``, ``article``, ``source_type`` and ``company_in_focus``
        are set by the caller)."""
        out: dict[str, Any] = {}
        if self.published_at is not None:
            out["published"] = stamp(self.published_at)
        age = self.age_hours()
        if age is not None:
            out["hours_between_publication_and_check"] = round(age, 1)
        if self.now is not None:
            out["check_time"] = stamp(self.now)
        if self.source:
            out["source"] = one_line(self.source, 80)
        if self.source_tier in TIER_LABELS:
            out["source_tier"] = TIER_LABELS[str(self.source_tier)]
        if self.category:
            out["category"] = one_line(self.category, 80)
        if self.sector:
            out["sector"] = one_line(self.sector, 60)
        if self.mentioned:
            out["companies_mentioned"] = [one_line(m, 90) for m in self.mentioned[:MAX_MENTIONED]]
        if self.web is not None:
            out["web_check"] = self.web.to_dict()
        return out
