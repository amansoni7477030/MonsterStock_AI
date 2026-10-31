"""Small embedded finance sentiment lexicon (Loughran-McDonald style) with negation handling.

Polarity describes the *wording* only. It is a weak auxiliary feature: the sign of an event for an entity is decided
by event semantics in ``events.assign_polarity``, never by positive or negative wording alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from news_intelligence.text import tokenize

POSITIVE = frozenset(
    """gain gains gained rise rises rose rising jump jumps jumped surge surges surged soar soars soared rally rallies
    rallied climb climbs climbed advance advances advanced higher record strong stronger strength robust growth grow
    grows grew improve improves improved improvement profit profits profitable beat beats outperform outperforms
    upgrade upgraded upgrades bullish boost boosts boosted win wins won bags secures secured approve approves approved
    approval positive optimistic optimism recovery recover recovers rebound rebounds expand expands expansion upbeat
    opportunity benefit benefits favourable favorable stable relief easing ease dividend bonus buyback milestone
    breakthrough award awarded successful success efficient resilient resilience tailwind accelerate accelerates
    exceed exceeds exceeded tops healthy solid surplus inflow inflows""".split()
)
NEGATIVE = frozenset(
    """fall falls fell falling drop drops dropped slump slumps slumped plunge plunges plunged tumble tumbles tumbled
    slide slides slid decline declines declined crash crashes crashed tank tanks tanked sink sinks sank lower loss
    losses lose loses lost weak weaker weakness weaken weakens miss misses missed downgrade downgraded downgrades
    bearish fraud scam probe investigation raid raids penalty penalties fine fined ban banned bans default defaults
    defaulted bankruptcy insolvency concern concerns worry worries warning warns warn threat threatens crisis stress
    turmoil selloff slowdown slows slowing recession shortage disruption disrupted delay delays delayed halt halts
    halted suspend suspends suspended resign resigns resigned resignation quit quits struggle struggles struggling
    pressure pressures headwind headwinds deficit outflow outflows cancel cancels cancelled canceled terminate
    terminated lawsuit litigation violation violations breach breaches attack attacks war conflict sanction sanctions
    fire explosion accident recall recalls shutdown closure layoff layoffs writeoff impairment restate restatement
    adverse unfavourable unfavorable disappoint disappoints disappointing disappointed poor worst plummet plummets""".split()
)
UNCERTAINTY = frozenset(
    """may might could possibly perhaps uncertain uncertainty unclear volatile volatility risk risks pending
    unconfirmed speculation speculate approximately depends contingent""".split()
)
NEGATORS = frozenset(
    """not no never without neither nor none cannot hardly barely scarcely fails failed fail unable lack lacks
    lacking absence""".split()
)
INTENSIFIERS = frozenset(
    """sharply steeply significantly substantially massively heavily strongly hugely drastically dramatically
    biggest""".split()
)
DIMINISHERS = frozenset("slightly marginally modestly mildly somewhat narrowly".split())

NEGATION_WINDOW = 3
_CONTRACTION = re.compile(r"\b(\w+)n['\u2019]t\b")
_CLAUSE_BREAK = re.compile(r"[.;:!?()\[\]]|\s[-\u2013\u2014]\s|,")


@dataclass(frozen=True)
class Sentiment:
    polarity: float
    positive: float
    negative: float
    uncertainty: float
    n_tokens: int


def score(text: str | None) -> Sentiment:
    """Lexicon score of ``text``; negators flip the next three words of the same clause."""
    if not text:
        return Sentiment(0.0, 0.0, 0.0, 0.0, 0)
    pos = neg = unc = 0.0
    n_tokens = 0
    for clause in _CLAUSE_BREAK.split(_CONTRACTION.sub(r"\1 not", text[:8000].lower())):
        tokens = tokenize(clause, max_tokens=400)
        n_tokens += len(tokens)
        negated_until = -1
        scale = 1.0
        last: tuple[int, str, float] | None = None  # (token index, 'pos'|'neg', weight) of the previous scored word
        for i, tok in enumerate(tokens):
            if tok in NEGATORS:
                negated_until = i + NEGATION_WINDOW
                continue
            if tok in INTENSIFIERS or tok in DIMINISHERS:
                factor = 1.5 if tok in INTENSIFIERS else 0.6
                if last is not None and last[0] == i - 1:  # "rises sharply": the adverb follows the word it modifies
                    delta = (factor - 1.0) * last[2]
                    pos, neg = (pos + delta, neg) if last[1] == "pos" else (pos, neg + delta)
                else:
                    scale = factor
                continue
            negated = i <= negated_until
            if tok in POSITIVE:
                weight, side = (0.7 * scale, "neg") if negated else (scale, "pos")
            elif tok in NEGATIVE:
                weight, side = (0.4 * scale, "pos") if negated else (scale, "neg")
            elif tok in UNCERTAINTY:
                unc += 1.0
                continue
            else:
                continue
            if side == "pos":
                pos += weight
            else:
                neg += weight
            last = (i, side, weight)
            scale = 1.0
    polarity = (pos - neg) / (pos + neg + 2.0)
    return Sentiment(round(polarity, 4), pos, neg, unc, n_tokens)


def polarity(text: str | None) -> float:
    """Wording polarity in (-1, 1); 0.0 for empty or balanced text."""
    return score(text).polarity
