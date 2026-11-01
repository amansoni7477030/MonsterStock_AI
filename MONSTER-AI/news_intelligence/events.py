"""Event taxonomy and the deterministic rule classifier.

``classify_rules`` is the always-available extractor: keyword/regex rules (NSE announcement categories count
heavily), rumour/denial detection, magnitude-aware severity and a polarity hint. ``assign_polarity`` then gives
every entity its *own* sign for the event - positive wording alone is never treated as a bullish signal.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from news_intelligence.entities import Entity

HORIZONS = ("intraday", "days", "weeks", "months")
SCOPES = ("company", "sector", "macro", "global")


@dataclass(frozen=True)
class EventType:
    name: str
    category: str
    scope: str
    horizon: str
    severity: float
    scheduled: bool = False


_TYPES = (
    EventType("earnings_result", "earnings", "company", "days", 0.60, True),
    EventType("earnings_guidance", "earnings", "company", "weeks", 0.55),
    EventType("order_win", "orders", "company", "days", 0.50),
    EventType("order_loss", "orders", "company", "days", 0.45),
    EventType("merger_acquisition", "corporate", "company", "weeks", 0.70),
    EventType("fundraising", "corporate", "company", "days", 0.45),
    EventType("buyback_dividend", "capital_return", "company", "days", 0.40),
    EventType("promoter_transaction", "ownership", "company", "days", 0.40),
    EventType("pledge", "ownership", "company", "days", 0.50),
    EventType("management_change", "governance", "company", "weeks", 0.40),
    EventType("rating_action", "credit", "company", "days", 0.45),
    EventType("litigation_regulatory", "legal", "company", "weeks", 0.60),
    EventType("government_contract", "orders", "company", "weeks", 0.55),
    EventType("board_meeting", "calendar", "company", "days", 0.25, True),
    EventType("corporate_action", "corporate", "company", "days", 0.25),
    EventType("bulk_block_deal", "flow", "company", "intraday", 0.35),
    EventType("macro_release", "macro", "macro", "intraday", 0.40, True),
    EventType("central_bank", "macro", "macro", "days", 0.65, True),
    EventType("geopolitical_conflict", "geopolitics", "macro", "days", 0.70),
    EventType("sanctions_tariffs", "geopolitics", "macro", "weeks", 0.60),
    EventType("commodity_supply", "commodity", "macro", "weeks", 0.55),
    EventType("trade_policy", "policy", "macro", "months", 0.50),
    EventType("elections_politics", "politics", "macro", "weeks", 0.50),
    EventType("sector_regulation", "policy", "sector", "months", 0.55),
    EventType("global_company", "global", "global", "days", 0.45),
    EventType("market_stress", "market", "global", "days", 0.70),
    EventType("other", "other", "company", "days", 0.15),
)
EVENT_TYPES: dict[str, EventType] = {t.name: t for t in _TYPES}
EVENT_TYPE_NAMES = tuple(EVENT_TYPES)

# What each type means, in words: both AI readers (the LLM prompt and Jev's choice question) classify against the same
# definitions as the rules do, so their labels mean the same thing.
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

RUMOUR_SEVERITY = 0.85  # an unconfirmed report is worth this share of the severity of a confirmed one
DENIAL_SEVERITY = 0.5  # a denial or a refuted report, this share
REFUTED_RULE = "web_refuted"  # recorded in ``EventDraft.rules`` when a web check found the report denied elsewhere

# Tie-break order when two types score equally (more specific / more consequential first).
_PRIORITY = (
    "litigation_regulatory", "pledge", "government_contract", "merger_acquisition", "promoter_transaction",
    "earnings_result", "earnings_guidance", "order_loss", "order_win", "rating_action", "management_change",
    "buyback_dividend", "fundraising", "global_company", "central_bank", "market_stress", "geopolitical_conflict",
    "sanctions_tariffs", "commodity_supply", "macro_release", "trade_policy", "sector_regulation",
    "elections_politics", "bulk_block_deal", "corporate_action", "board_meeting", "other",
)


@dataclass
class EventDraft:
    event_type: str = "other"
    severity: float = 0.15
    confidence: float = 0.3
    horizon: str = "days"
    polarity_hint: float = 0.0
    is_rumour: bool = False
    scheduled: bool = False
    scope: str = "company"
    is_denial: bool = False
    routine: bool = False
    rules: tuple[str, ...] = ()


def apply_refutation(draft: EventDraft) -> bool:
    """A reliable source has denied or corrected the report this draft was made from: the draft becomes an unconfirmed rumour
    with a denial's discounts (half the severity, a halved direction, a trimmed confidence). A draft that is itself a denial is
    left alone, and so is one that was refuted already. Returns whether the draft changed."""
    if draft.is_denial or REFUTED_RULE in draft.rules:
        return False
    if draft.is_rumour:  # its direction was halved when it was flagged: only the severity still has to go down
        draft.severity = round(max(0.02, draft.severity * DENIAL_SEVERITY / RUMOUR_SEVERITY), 3)
    else:
        draft.severity = round(max(0.02, draft.severity * DENIAL_SEVERITY), 3)
        draft.polarity_hint = round(draft.polarity_hint * 0.5, 3)
        draft.is_rumour = True
    if draft.horizon == "months":  # classify_rules shortens the horizon of every rumour: an unconfirmed report is no long-term fact
        draft.horizon = "weeks"
    draft.confidence = round(max(0.05, draft.confidence * 0.7), 3)
    draft.rules = (*draft.rules, REFUTED_RULE)
    return True


# ------------------------------------------------------------------------------------------- fragments
_UP = (
    r"(?:rises?|rose|rising|jumps?|jumped|surges?|surged|soars?|soared|climbs?|climbed|gains?|gained|rall(?:y|ies|ied)"
    r"|spikes?|spiked|advances?|advanced|up|higher|grows?|grew|improves?|improved|expands?|expanded|doubles?|doubled"
    r"|zooms?|zoomed|leaps?|leapt|records?|strengthens?|strengthened|appreciates?|appreciated|hardens?|hardened)"
)
_DOWN = (
    r"(?:falls?|fell|falling|drops?|dropped|slumps?|slumped|plunges?|plunged|tumbles?|tumbled|slides?|slid|declines?"
    r"|declined|dips?|dipped|sinks?|sank|slips?|slipped|crash(?:es|ed)?|tanks?|tanked|shrinks?|shrank|down|lower"
    r"|plummets?|plummeted|halves|halved|erodes?|eroded|widens?|widened|contracts?|contracted|weakens?|weakened"
    r"|depreciates?|depreciated|softens?|softened|retreats?|retreated|eased)"
)
_PROFIT = r"profit(?!\s+(?:booking|taking))"
_METRIC = r"(?:net profit|profit after tax|" + _PROFIT + r"|pat|net income|earnings|revenue|sales|ebitda|nii)"
_MONEY = r"(?:\u20b9|\brs\.?|\binr\b|\busd\b|us\$|\$|\u20ac|\beur\b)"
_GOV = (
    r"(?:ministry|government|govt|nhai|indian railways|railway board|railways|defence|\bmod\b|state government|municipal"
    r"|discoms?|isro|drdo|navy|army|air force|coast guard|\bpsus?\b|central government|public sector|bsnl|mtnl)"
)


@dataclass(frozen=True)
class Rule:
    name: str
    event_type: str
    weight: float
    pattern: re.Pattern[str]
    polarity: float | None = None
    severity: float | None = None
    exclusive: bool = False  # when matched, this rule alone decides the polarity of its event type


def _r(name: str, event_type: str, weight: float, pattern: str, polarity: float | None = None,
       severity: float | None = None, *, exclusive: bool = False) -> Rule:
    return Rule(name, event_type, weight, re.compile(pattern, re.IGNORECASE), polarity, severity, exclusive)


_RULES: tuple[Rule, ...] = (
    # ------------------------------------------------------------------------------------- earnings
    _r("period_results", "earnings_result", 2.0,
       r"\b(?:q[1-4]\b|[1-4]q\b|q[1-4]\s?fy\s?'?\d{2,4}|fy\s?'?\d{2}(?:-\d{2})?|(?:first|second|third|fourth|1st|2nd|3rd|4th)[- ]quarter"
       r"|quarterly|half[- ]year(?:ly)?|annual|full[- ]year|h[12]\s?fy\s?\d{2}|(?:june|september|december|march) quarter)"
       r".{0,45}\b(?:results?|earnings|profit|net profit|pat|net income|revenue|loss|ebitda|margins?|numbers|performance)\b"),
    _r("financial_results", "earnings_result", 2.5,
       r"\b(?:financial|audited|unaudited|standalone|consolidated|quarterly)\s+(?:financial\s+)?results?\b"),
    _r("profit_up", "earnings_result", 2.0, r"\b" + _METRIC + r"\b.{0,40}\b" + _UP + r"\b", 0.7),
    _r("profit_down", "earnings_result", 2.0, r"\b" + _METRIC + r"\b.{0,40}\b" + _DOWN + r"\b", -0.7),
    _r("net_loss", "earnings_result", 2.2,
       r"\b(?:net loss|(?:swings?|slips?|slumps?|falls?|turns?|posts?|reports?) (?:in)?to (?:a )?(?:net )?loss|loss widens?|widens? (?:its )?(?:net )?loss"
       r"|posts? (?:a )?(?:net )?loss|reports? (?:a )?(?:net )?loss)\b", -0.7),
    _r("turns_profit", "earnings_result", 2.2,
       r"\b(?:swings?|turns?|returns?|moves?) (?:in)?to (?:a )?(?:net )?profit\b|\bturns? profitable\b", 0.7),
    _r("beat_estimates", "earnings_result", 1.6,
       r"\b(?:beats?|tops?|surpass(?:es|ed)?|exceed(?:s|ed)?|above|ahead of)\s+(?:the\s+)?(?:street|market|analyst|analysts|consensus)?\s*"
       r"(?:estimates?|expectations?|forecasts?|views?|consensus|projections?)\b", 0.7),
    _r("miss_estimates", "earnings_result", 1.6,
       r"\b(?:misses?|lags?|trails?|below|short of|falls? short of)\s+(?:the\s+)?(?:street|market|analyst|analysts|consensus)?\s*"
       r"(?:estimates?|expectations?|forecasts?|views?|consensus|projections?)\b", -0.7),
    _r("record_results", "earnings_result", 1.6,
       r"\b(?:record|highest[- ]ever|best[- ]ever)\b.{0,25}\b(?:profit|revenue|earnings|quarter|sales)\b", 0.6),
    _r("business_update", "earnings_result", 1.3,
       r"\b(?:monthly|quarterly|september|october|november|december|january|february|march|april|may|june|july|august)?\s*"
       r"(?:sales|volumes?|despatches|dispatches|production|registrations|collections?)\b.{0,25}\b" + _UP + r"\b", 0.4),
    _r("business_update_down", "earnings_result", 1.3,
       r"\b(?:sales|volumes?|despatches|dispatches|production|registrations|collections?)\b.{0,25}\b" + _DOWN + r"\b", -0.4),
    _r("results_preview", "earnings_result", 1.5, r"\b(?:results?|earnings)\s+(?:preview|calendar|today|on)\b"),
    # ------------------------------------------------------------------------------------- guidance
    _r("guidance_raise", "earnings_guidance", 2.3,
       r"\b(?:raises?|lifts?|hikes?|boosts?|upgrades?|increases?|ups|lifted|raised)\b.{0,25}\b(?:guidance|outlook|forecast|target|estimates?)\b", 0.6),
    _r("guidance_cut", "earnings_guidance", 2.5,
       r"\b(?:cuts?|lowers?|slashes?|trims?|reduces?|downgrades?|withdraws?|scraps?|cut|lowered|slashed)\b.{0,25}\b(?:guidance|outlook|forecast|target|estimates?)\b", -0.6),
    _r("profit_warning", "earnings_guidance", 2.5,
       r"\bprofit warning\b|\bwarns?\b.{0,20}\b(?:profit|earnings|revenue|margins?|demand|slowdown)\b", -0.6),
    _r("guidance_generic", "earnings_guidance", 1.4, r"\b(?:guidance|management outlook|fy\d{2} outlook)\b"),
    # ----------------------------------------------------------------------------------------- orders
    _r("order_win", "order_win", 2.0,
       r"\b(?:bags?|bagged|wins?|won|secures?|secured|receives?|received|gets?|got|lands?|landed|awards?|awarded|clinches?|clinched|inks?|inked)\b"
       r".{0,50}\b(?:orders?|contracts?|deals?|projects?|work orders?|letters? of (?:award|intent)|loa|loi|tenders?|mandates?|assignments?)\b", 0.6),
    _r("order_value", "order_win", 1.2,
       r"\b(?:orders?|contracts?)\b.{0,20}\b(?:worth|valued at|of)\b.{0,12}" + _MONEY, 0.5),
    _r("order_inflow", "order_win", 1.0, r"\border (?:inflows?|wins?)\b", 0.5),
    _r("l1_bidder", "order_win", 2.0, r"\bemerges? as (?:the )?(?:l1|lowest|preferred) bidder\b", 0.6),
    _r("award_letter", "order_win", 2.0, r"\b(?:letter of award|letter of intent)\b", 0.6),
    _r("order_loss", "order_loss", 2.3,
       r"\b(?:cancel(?:s|led|ed|lation)?|terminat(?:es|ed|ion)|scraps?|scrapped|revokes?|revoked|withdraws?|withdrawn|loses?|lost|rejects?|rejected)\b"
       r".{0,40}\b(?:orders?|contracts?|deals?|tenders?|bids?)\b", -0.6),
    _r("order_cancelled", "order_loss", 2.3,
       r"\b(?:orders?|contracts?|deals?)\b.{0,25}\b(?:cancel(?:led|ed)|terminated|scrapped|revoked|withdrawn)\b", -0.6),
    _r("gov_contract", "government_contract", 3.2,
       r"\b(?:orders?|contracts?|tenders?|loa|projects?|deals?)\b.{0,60}\b" + _GOV + r"|" + _GOV + r".{0,60}\b(?:orders?|contracts?|tenders?|loa)\b", 0.6),
    # ------------------------------------------------------------------------------------------ M&A
    _r("mna_strong", "merger_acquisition", 2.3,
       r"\b(?:acquires?|acquired|acquisitions?|to acquire|acquiring|takeovers?|takes over|to take over|buy[- ]?outs?|mergers?|merges?|merged"
       r"|amalgamation|demergers?|demerge|scheme of arrangement|open offer|swap ratio)\b"),
    _r("mna_soft", "merger_acquisition", 1.6,
       r"\b(?:divest(?:s|ed|iture|ment)?|spin[- ]?off|slump sale|joint venture|stake sale|sells? (?:its )?(?:entire |remaining )?(?:stake|business|unit|division)"
       r"|buys? (?:a )?(?:\d{1,2}(?:\.\d+)?%\s)?stake|picks? up (?:a )?(?:\d{1,2}(?:\.\d+)?%\s)?stake)\b"),
    # ------------------------------------------------------------------------------------ fundraising
    _r("equity_raise", "fundraising", 2.2,
       r"\b(?:qip|qualified institutional placement|rights issue|preferential (?:issue|allotment)|private placement|follow[- ]on public offer|fpo"
       r"|offer for sale|equity (?:infusion|raise)|capital infusion|fund[- ]?rais(?:e|es|ed|ing)|raises?\s+(?:" + _MONEY + r")?\s?[\d,.]+\s?(?:crore|cr|billion|bn|million|mn))\b", -0.2),
    _r("debt_raise", "fundraising", 1.8,
       r"\b(?:ncds?|non[- ]convertible debentures?|bond issue|convertible (?:bonds?|debentures?|warrants?)|external commercial borrowing|ecbs?|debt raising)\b"),
    _r("ipo", "fundraising", 1.5, r"\b(?:ipo|initial public offering|listing)\b"),
    # --------------------------------------------------------------------------- buyback / dividend
    _r("buyback", "buyback_dividend", 2.3, r"\b(?:buy[- ]?backs?|share repurchase|repurchase of (?:equity )?shares)\b", 0.5),
    _r("dividend", "buyback_dividend", 1.9, r"\b(?:interim |final |special |annual )?dividends?\b", 0.3),
    _r("dividend_omitted", "buyback_dividend", 2.6,
       r"\b(?:omits?|skips?|cuts?|suspends?|passes|no)\s+(?:the\s+)?(?:interim |final |special )?dividends?\b", -0.4, exclusive=True),
    # ------------------------------------------------------------------ promoter / pledge / mgmt
    _r("promoter_buy", "promoter_transaction", 2.4,
       r"\b(?:promoters?|promoter group|insiders?)\b.{0,40}\b(?:buys?|bought|acquires?|acquired|raises?|increases?|picks? up|purchases?|infuses?)\b", 0.5),
    _r("promoter_sell", "promoter_transaction", 2.4,
       r"\b(?:promoters?|promoter group|insiders?)\b.{0,40}\b(?:sells?|sold|offloads?|offloaded|reduces?|cuts?|trims?|dumps?|exits?|stake sale)\b", -0.4),
    _r("promoter_generic", "promoter_transaction", 1.4,
       r"\b(?:promoter (?:holding|stake|shareholding)|sast|regulation 29|insider trading|substantial acquisition)\b"),
    _r("pledge", "pledge", 2.6, r"\b(?:pledg(?:e|es|ed|ing)|invocation of pledge|encumbrance|encumbered)\b", -0.6),
    _r("pledge_release", "pledge", 3.2,
       r"\b(?:release|released|releases|revocation|revoked|revokes)\s+(?:of\s+)?(?:the\s+)?(?:pledge|pledged|encumbrance)", 0.4, exclusive=True),
    _r("mgmt_exit", "management_change", 2.3,
       r"\b(?:resign(?:s|ed|ation)|steps? down|quits?|cessation|retires?|retirement|exits?)\b.{0,50}"
       r"\b(?:ceo|cfo|coo|cto|md|chairman|chairperson|managing director|director|chief executive|chief financial officer|chief operating officer"
       r"|whole[- ]time director|company secretary|compliance officer|key managerial|kmp|independent director|auditor|president|head)\b", -0.4),
    _r("mgmt_exit_rev", "management_change", 2.3,
       r"\b(?:ceo|cfo|coo|cto|md|chairman|chairperson|managing director|director|chief executive|chief financial officer|chief operating officer"
       r"|whole[- ]time director|company secretary|compliance officer|key managerial|kmp|independent director|auditor|president|head)\b.{0,40}"
       r"\b(?:resign(?:s|ed|ation)|steps? down|quits?|cessation|retires?|exits?)\b", -0.4),
    _r("auditor_exit", "management_change", 3.2,
       r"\b(?:statutory )?auditors?\b.{0,30}\b(?:resign(?:s|ed|ation)|quits?|steps? down)\b", -0.7, exclusive=True),
    _r("mgmt_appoint", "management_change", 2.0,
       r"\b(?:appoint(?:s|ed|ment)|re-?appoint(?:s|ed|ment)|names?|elevates?|promotes?|new)\b.{0,50}"
       r"\b(?:ceo|cfo|coo|cto|md|chairman|chairperson|managing director|director|chief executive|chief financial officer|chief operating officer"
       r"|whole[- ]time director|company secretary|compliance officer|key managerial|kmp|independent director|auditor|president|head)\b", 0.1),
    # ----------------------------------------------------------------------------------------- ratings
    _r("credit_rating", "rating_action", 2.2,
       r"\b(?:credit rating|rating action|crisil|icra|care ratings|india ratings|brickwork|acuite|fitch|moody'?s|s&p global)\b.{0,50}"
       r"\b(?:upgrades?|upgraded|downgrades?|downgraded|reaffirms?|reaffirmed|assigns?|assigned|revises?|revised|outlook|watch|withdraws?|withdrawn|places?)\b"),
    _r("rating_up", "rating_action", 1.2, r"\b(?:upgrades?|upgraded)\b.{0,30}\b(?:rating|to (?:aaa|aa\+?|a\+?|bbb))\b", 0.5),
    _r("rating_down", "rating_action", 1.2, r"\b(?:downgrades?|downgraded)\b.{0,30}\b(?:rating|to (?:bbb|bb|b|c|d)\b)", -0.6),
    _r("outlook_negative", "rating_action", 1.6, r"\boutlook\b.{0,20}\b(?:negative|to negative|cut)\b|\bnegative (?:outlook|watch)\b", -0.4),
    _r("outlook_positive", "rating_action", 1.6, r"\boutlook\b.{0,20}\b(?:positive|to positive)\b|\bpositive outlook\b", 0.3),
    _r("analyst_rating", "rating_action", 1.6,
       r"\b(?:upgrades?|downgrades?|initiates? coverage|target price|price target|reiterates?|maintains?)\b.{0,40}"
       r"\b(?:buy|sell|hold|neutral|overweight|underweight|outperform|underperform|accumulate|reduce)\b", None, 0.30),
    _r("analyst_up", "rating_action", 1.0, r"\b(?:upgrades?|upgraded|raises? target|target raised|initiates? .{0,20}buy)\b", 0.4),
    _r("analyst_down", "rating_action", 1.0, r"\b(?:downgrades?|downgraded|cuts? target|target cut|initiates? .{0,20}sell)\b", -0.4),
    # ---------------------------------------------------------------------- litigation / regulatory
    _r("fraud", "litigation_regulatory", 2.8,
       r"\b(?:fraud|scam|money laundering|pmla|siphon(?:ing)?|diversion of funds|forensic audit|whistle-?blower|accounting irregularit(?:y|ies)|misappropriation)\b", -0.8, 0.7),
    _r("probe_raid", "litigation_regulatory", 2.3,
       r"\b(?:probe[sd]?|investigat(?:ion|ing)|raid(?:s|ed)?|arrest(?:s|ed)?|charge[- ]?sheet|summon(?:s|ed)|questioned|detain(?:s|ed))\b", -0.6),
    _r("agency_action", "litigation_regulatory", 2.2,
       r"\b(?:enforcement directorate|cbi|sfio|serious fraud investigation|income[- ]tax department|competition commission|sebi|nclt|nclat|nppa|dgft)\b.{0,50}"
       r"\b(?:notice|penalty|fine[sd]?|probe|raid|summon|bans?|bars?|barred|directs?|issues?|slaps?|imposes?|levies|attach(?:es|ed|ment)|orders?)\b", -0.5),
    _r("penalty", "litigation_regulatory", 2.3,
       r"\b(?:penalt(?:y|ies)|fined?|show[- ]cause notice|slaps? (?:a )?(?:fine|penalty)|imposes? (?:a )?(?:fine|penalty)|levies (?:a )?(?:fine|penalty))\b", -0.5),
    _r("insolvency", "litigation_regulatory", 2.8,
       r"\b(?:insolvency|bankruptcy|defaults? on|defaulted|wilful defaulter|debt default|missed (?:a )?(?:loan |interest |bond )?payment|loan default|liquidation)\b", -0.8, 0.75),
    _r("court", "litigation_regulatory", 1.6,
       r"\b(?:supreme court|high court|sat|nclat|nclt|tribunal|court)\b.{0,50}"
       r"\b(?:orders?|rules?|ruled|verdict|judgment|dismisses|dismissed|upholds|upheld|quashes|quashed|stays?|stayed|relief|acquits?|acquitted|directs?|hearing|petition|appeal)\b"),
    _r("court_relief", "litigation_regulatory", 1.8,
       r"\b(?:quash(?:es|ed)|acquit(?:s|ted)|sets? aside|stayed|exonerat(?:es|ed)|wins? (?:case|appeal|legal)|gets? relief|relief (?:from|for))\b", 0.5),
    _r("court_adverse", "litigation_regulatory", 1.8,
       r"\b(?:dismiss(?:es|ed) (?:the )?(?:plea|petition|appeal)|orders? (?:to )?pay|directs? (?:to )?pay|convict(?:s|ed))\b", -0.5),
    _r("pharma_adverse", "litigation_regulatory", 2.9,
       r"\b(?:usfda|fda|ema|dcgi)\b.{0,50}\b(?:warning letter|form 483|observations?|import alert|oai|ban|recall|rejects?|complete response letter|crl)\b"
       r"|\b(?:warning letter|import alert|form 483)\b", -0.6),
    _r("pharma_approval", "litigation_regulatory", 2.6,
       r"\b(?:usfda|fda|ema|dcgi|cdsco)\b.{0,50}\b(?:approval|approves?|approved|nod|clears?|cleared|tentative approval)\b", 0.4),
    _r("licence_action", "litigation_regulatory", 2.2,
       r"\b(?:licen[sc]e|registration|approval|permit)\b.{0,25}\b(?:cancel+ed|revoked|suspended|withdrawn|rejected)\b", -0.6),
    _r("ban_action", "litigation_regulatory", 2.2,
       r"\b(?:bars?|barred|bans?|banned|debar(?:s|red))\b.{0,40}\b(?:trading|markets?|securities|company|promoters?|directors?|from)\b", -0.7),
    _r("tax_demand", "litigation_regulatory", 1.8,
       r"\b(?:tax|gst|customs|excise)\b.{0,25}\b(?:demand|notice|evasion|dues|liability|assessment order)\b", -0.4),
    _r("approval_granted", "litigation_regulatory", 2.0,
       r"\b(?:cci|sebi|nclt|rbi|cerc|cdsco|dgca|trai|irdai|mca|dpiit)\b.{0,40}"
       r"\b(?:approv(?:es|ed|al)|nod|clears?|cleared|clearance|go[- ]ahead|green signal|green[- ]lights?)\b", 0.4),
    # ---------------------------------------------------------------------------- calendar / actions
    _r("board_meeting", "board_meeting", 2.2,
       r"\b(?:intimation of (?:the )?board meeting|date of board meeting|board meeting (?:on|to be held|scheduled|to consider|will)"
       r"|meeting of the board (?:on|to consider|will)|board of directors? (?:will|to|shall)\s+(?:meet|consider)"
       r"|board to (?:consider|meet|approve))\b"),
    _r("corporate_action", "corporate_action", 2.0,
       r"\b(?:stock split|share split|sub[- ]?division|bonus (?:issue|shares?)|issue of bonus|record date|ex[- ]date|ex[- ]dividend|ex[- ]bonus|ex[- ]split"
       r"|book closure|face value|delist(?:ing|ed|s)?|change (?:of|in) name|name change|reclassification|consolidation of shares)\b"),
    _r("bulk_block", "bulk_block_deal", 2.2,
       r"\b(?:bulk deals?|block deals?|bulk/block deals?|bulk and block|open market (?:purchase|sale|transaction)|stake sold (?:via|through))\b"),
    # ------------------------------------------------------------------------------------ macro etc.
    _r("macro_data", "macro_release", 2.0,
       r"\b(?:cpi|consumer price index|wpi|wholesale price index|retail inflation|wholesale inflation|inflation (?:data|print|rate|numbers|eases|cools|rises|jumps|falls)"
       r"|core inflation|gdp (?:data|growth|numbers|estimates?|print)|(?:india|us|china|euro ?zone)(?:'s)? gdp|iip|index of industrial production"
       r"|industrial (?:output|production)|pmi|purchasing managers|trade deficit|trade data|current account|jobs report|non[- ]?farm payrolls?|nfp"
       r"|unemployment rate|jobless claims|retail sales|gst collections?|pce|durable goods|consumer confidence|fiscal deficit|core sector|producer price)\b"),
    _r("fx_rates_move", "macro_release", 1.2,
       r"\b(?:rupee|dollar index|dxy|yen|treasury yields?|bond yields?|10-year yield)\b.{0,30}\b(?:" + _UP[3:-1] + "|" + _DOWN[3:-1] + r"|record low|record high|all-time low|lifetime low)\b"),
    _r("central_bank", "central_bank", 2.2,
       r"\b(?:rbi|reserve bank of india|mpc|monetary policy committee|repo rate|reverse repo|policy rate|crr|cash reserve ratio|federal reserve|fomc|powell"
       r"|ecb|lagarde|boj|bank of japan|bank of england|pboc|rate (?:cut|hike|decision)|interest rate (?:decision|cut|hike)|monetary policy|dot plot"
       r"|quantitative (?:tightening|easing)|hawkish|dovish|the fed|fed (?:rate|chair|holds?|cuts?|hikes?|raises?|signals?|minutes|officials?|policy|decision|meeting))\b"),
    _r("rate_hike", "central_bank", 1.4,
       r"\b(?:rate hikes?|hikes?|raises?|lifts?) (?:the )?(?:key |benchmark |policy |repo )?(?:interest )?rates?\b|\bhawkish\b|\bhigher for longer\b", 0.7),
    _r("rate_cut", "central_bank", 1.4,
       r"\b(?:rate cuts?|cuts?|lowers?|slashes?|eases?) (?:the )?(?:key |benchmark |policy |repo )?(?:interest )?(?:rates?|crr|cash reserve ratio)\b|\bdovish\b", -0.7),
    _r("geo_strong", "geopolitical_conflict", 2.2,
       r"\b(?:war|missiles?|air ?strikes?|drone (?:attack|strike)s?|invasion|invades?|ceasefire|cease-fire|truce|troops|military (?:action|operation|strike|escalation)"
       r"|shelling|bombing|terror(?:ist)? attacks?|hostages?|hostilities|houthis?|hezbollah|hamas|red sea|strait of hormuz|hormuz|blockade|coup)\b"),
    _r("geo_weak", "geopolitical_conflict", 1.0, r"\b(?:tensions?|clashes?|conflict|unrest|standoff|escalat(?:e|es|ed|ion|ing))\b"),
    _r("geo_escalate", "geopolitical_conflict", 0.8,
       r"\b(?:escalat(?:e|es|ed|ion|ing)|attacks?|strikes?|invades?|invasion|missiles?|air ?strikes?|blockade|hostilities|flare[- ]?up|retaliat(?:e|es|ed|ion))\b", 0.7),
    _r("geo_deescalate", "geopolitical_conflict", 0.8,
       r"\b(?:ceasefire|cease-fire|truce|peace (?:deal|talks|agreement)|de-?escalat(?:e|es|ed|ion)|talks resume)\b", -0.7),
    _r("sanctions_tariffs", "sanctions_tariffs", 2.2,
       r"\b(?:sanctions?|sanctioned|tariffs?|(?:import|export|customs|countervailing|anti-?dumping|safeguard) dut(?:y|ies)|export (?:ban|curbs?|restrictions?|controls?)"
       r"|import (?:ban|curbs?|restrictions?)|embargo|trade war|reciprocal tariffs?|blacklist(?:s|ed)?|entity list)\b"),
    _r("tariff_impose", "sanctions_tariffs", 0.8,
       r"\b(?:impos(?:es|ed|ing)|hikes?|slaps?|threatens?|announces?|levies|raises?)\b.{0,30}\b(?:tariffs?|sanctions?|dut(?:y|ies)|curbs?)\b", 0.6),
    _r("tariff_ease", "sanctions_tariffs", 0.8,
       r"\b(?:lifts?|eases?|waives?|removes?|pauses?|rolls? back|exempts?|cuts?|suspends?)\b.{0,30}\b(?:tariffs?|sanctions?|dut(?:y|ies)|curbs?|ban)\b", -0.6),
    _r("trade_policy", "trade_policy", 2.0,
       r"\b(?:trade (?:deal|pact|agreement|talks|negotiations?)|free trade agreement|ftas?|bilateral trade agreement|wto|export policy|import policy|customs duty"
       r"|duty cut|duty hike|duty-free|pli scheme|production[- ]linked incentive|dgft|export incentives?|rodtep|quota)\b"),
    _r("commodity_supply", "commodity_supply", 2.0,
       r"\b(?:opec\+?|output cuts?|production cuts?|supply (?:cuts?|disruptions?|shortage|glut|surplus|crunch|concerns?)|pipeline (?:outage|attack|leak|shutdown)"
       r"|refinery (?:outage|fire|shutdown|attack)|crude inventories|oil inventories|stockpiles?|inventory (?:draw|build|drawdown|buildup)|eia (?:report|data)"
       r"|export quotas?|mine (?:strike|closure|shutdown|collapse)|coal shortage|gas shortage|lng (?:supply|shortage)|bumper crop|crop (?:damage|failure)"
       r"|monsoon (?:deficit|shortfall|delay)|drought|heatwave)\b"),
    _r("commodity_move", "commodity_supply", 1.2,
       r"\b(?:brent|crude|oil|natural gas|gold|silver|copper|alumin(?:i)?um|zinc|nickel|iron ore|coal)\b.{0,30}\b(?:" + _UP[3:-1] + "|" + _DOWN[3:-1] + r"|record high|record low)\b"),
    _r("elections", "elections_politics", 2.0,
       r"\b(?:elections?|exit polls?|poll results?|assembly polls?|lok sabha|vote counting|coalition government|no-?confidence (?:motion|vote)|hung parliament"
       r"|snap polls?|bypolls?|impeachment|cabinet reshuffle)\b"),
    _r("budget", "sector_regulation", 2.0, r"\b(?:union budget|budget (?:20\d{2}|proposals?|announcements?|speech)|finance minister)\b"),
    _r("regulator_rules", "sector_regulation", 2.0,
       r"\b(?:rbi|sebi|trai|irdai|cerc|nppa|fssai|dgca|pfrda|ibbi|cci)\b.{0,50}"
       r"\b(?:norms?|guidelines?|circular|framework|rules?|regulations?|directions?|draft|consultation paper|mandate[sd]?|curbs?|tightens?|relax(?:es|ed)?|eases?|caps?|new (?:rules?|norms?))\b"),
    _r("policy_levers", "sector_regulation", 1.8,
       r"\b(?:windfall tax|excise duty|gst (?:rate|council|cut|hike|slab)|price cap|ceiling price|subsid(?:y|ies)|minimum support price|msp|ethanol blending"
       r"|ev (?:policy|subsid(?:y|ies))|scrappage policy|spectrum (?:auction|pricing)|agr dues|fuel price (?:cut|hike))\b"),
    _r("policy_positive", "sector_regulation", 0.8,
       r"\b(?:relief|cuts? (?:tax|duty|gst)|reduces? (?:tax|duty|gst)|incentives?|boost|extends? (?:the )?(?:scheme|deadline)|relax(?:es|ed)|eases?"
       r"|approves? (?:scheme|package)|waives?|lifts? (?:ban|curbs?|restrictions?)|allows?)\b", 0.5),
    _r("policy_negative", "sector_regulation", 0.8,
       r"\b(?:curbs?|tighten(?:s|ed|ing)|caps?|capped|windfall tax|stricter|restricts?|restrictions?|hikes? (?:tax|duty|gst)|crack ?down|clampdown|penal)\b", -0.5),
    _r("global_company", "global_company", 4.0,
       r"\b(?:nvidia|apple|microsoft|alphabet|google|amazon|meta|tesla|tsmc|taiwan semiconductor|samsung|intel|amd|micron|asml|broadcom|qualcomm|oracle|openai)\b.{0,60}"
       r"\b(?:results?|earnings|guidance|revenue|profit|forecast|quarter|sales|shipments|iphone|chips?|ai (?:demand|spending|capex)|capex|recall|layoffs|antitrust)\b"),
    _r("market_stress", "market_stress", 2.2,
       r"\b(?:sell[- ]?off|market (?:crash|rout|meltdown|turmoil|panic)|stocks? (?:crash|plunge|tumble|tank|rout)|circuit breakers?|bloodbath"
       r"|black (?:monday|tuesday|swan)|flash crash|panic (?:selling|sell-?off)|vix (?:spikes?|jumps?|surges?|soars?)|volatility (?:spikes?|surges?)|fear gauge"
       r"|bank run|contagion|credit crunch|liquidity (?:crisis|crunch|squeeze)|margin calls?|systemic risk|recession (?:fears?|worries|risk)|global (?:rout|selloff)"
       r"|risk[- ]off|flight to safety)\b", -0.7),
    _r("sensex_crash", "market_stress", 2.0,
       r"\b(?:sensex|nifty|dow|nasdaq|s&p 500)\b.{0,40}\b(?:crash(?:es|ed)?|plunges?|plunged|tumbles?|tumbled|tanks?|tanked|sinks?|slumps?|slumped)\b.{0,30}\b(?:points?|%|per ?cent)\b", -0.7),
    _r("operational_disruption", "other", 1.8,
       r"\b(?:fire|explosion|blast|accident|plant (?:shutdown|closure)|production (?:halt|halted|suspended)|operations? (?:suspended|halted)|force majeure"
       r"|lockout|cyber ?attack|data breach|ransomware)\b", -0.6, 0.5),
    _r("commentary", "other", 1.0, r"\b(?:stocks? to watch|market wrap|closing bell|opening bell|stock market today|share market today)\b"),
)

_PRIORITY_INDEX = {name: i for i, name in enumerate(_PRIORITY)}

# NSE corporate-announcement categories ('desc'): (substring, event type, weight). First match wins.
_CATEGORY_RULES: tuple[tuple[str, str, float], ...] = (
    ("financial results", "earnings_result", 3.0),
    ("outcome of board meeting", "board_meeting", 1.0),
    ("board meeting", "board_meeting", 3.0),
    ("substantial acquisition", "promoter_transaction", 2.0),
    ("takeover", "promoter_transaction", 2.0),
    ("amalgamation", "merger_acquisition", 3.0),
    ("scheme of arrangement", "merger_acquisition", 3.0),
    ("acquisition", "merger_acquisition", 3.0),
    ("award of order", "order_win", 3.0),
    ("receipt of order", "order_win", 3.0),
    ("bagging", "order_win", 3.0),
    ("credit rating", "rating_action", 3.0),
    ("resignation", "management_change", 3.0),
    ("change in directorate", "management_change", 3.0),
    ("change in management", "management_change", 3.0),
    ("cessation", "management_change", 2.5),
    ("appointment", "management_change", 2.5),
    ("dividend", "buyback_dividend", 3.0),
    ("buyback", "buyback_dividend", 3.0),
    ("bonus", "corporate_action", 3.0),
    ("stock split", "corporate_action", 3.0),
    ("sub-division", "corporate_action", 3.0),
    ("record date", "corporate_action", 2.5),
    ("fund raising", "fundraising", 3.0),
    ("preferential", "fundraising", 3.0),
    ("rights issue", "fundraising", 3.0),
    ("qip", "fundraising", 3.0),
    ("issue of securities", "fundraising", 2.0),
    ("insider trading", "promoter_transaction", 2.0),
    ("sast", "promoter_transaction", 2.0),
    ("encumbrance", "pledge", 3.0),
    ("pledge", "pledge", 3.0),
    ("action(s) initiated", "litigation_regulatory", 3.0),
    ("litigation", "litigation_regulatory", 3.0),
    ("fraud", "litigation_regulatory", 3.0),
    ("default", "litigation_regulatory", 3.0),
    ("arrest", "litigation_regulatory", 3.0),
    ("insolvency", "litigation_regulatory", 3.0),
)
_ROUTINE_CATEGORIES = (
    "trading window", "newspaper publication", "copy of newspaper", "investor meet", "con. call", "conference call",
    "certificate under", "loss of share", "duplicate share", "reg. 74", "reg 74", "reg. 24", "compliances",
    "annual report", "annual general meeting", "agm", "postal ballot", "scrutinizer", "voting results",
    "statement of deviation", "monitoring agency", "investor presentation", "investor grievance", "share transfer",
    "e-voting", "esop", "employee stock", "shareholders meeting", "transcript", "audio recording", "schedule of",
)

_RUMOUR_STRONG = re.compile(
    r"\b(?:reportedly|according to (?:sources|reports?|people)|sources? (?:said|say|told|close|familiar|aware)|people (?:familiar|aware|close)"
    r"|in (?:early |advanced )?talks|unconfirmed|rumou?rs?|rumou?red|speculat(?:ion|ed|es)|market talk|whispers?)\b",
    re.IGNORECASE,
)
_RUMOUR_WEAK = re.compile(
    r"\b(?:likely to|poised to|may (?:acquire|buy|sell|raise|launch|bid|hike|cut|cross|face|see|list|consider|announce|approve|merge|exit"
    r"|invest|get|report|post|clock|record|clear|issue|pay|declare|lose|win|bag|secure)|could (?:acquire|buy|sell|raise|launch|bid|hike|cut|face|see|list"
    r"|consider|merge|exit|invest|bag|secure|lose|win))\b",
    re.IGNORECASE,
)
_DENIAL = re.compile(
    r"\b(?:denies|denied|dismisses|dismissed|refutes|refuted|rubbishes|clarifies|clarification|baseless|false (?:reports?|news|claims?)|no truth|categorically denies|not true)\b",
    re.IGNORECASE,
)


def mentions_denial(text: str | None) -> bool:
    """True when the text uses the wording of a denial or correction (the rules' own test, offered to the web check so that it
    judges a page with the same words the classifier uses)."""
    return bool(text) and _DENIAL.search(str(text)) is not None
_SCHEDULED_CUES = re.compile(
    r"\b(?:scheduled|due (?:on|today|tomorrow|this week)|to be (?:held|announced|released|published)|will (?:be held|announce|meet|consider|release)"
    r"|to (?:consider|announce|meet|release)|ahead of|awaits?|upcoming|preview|calendar|intimation)\b",
    re.IGNORECASE,
)
_OUTCOME_CUES = re.compile(
    r"\b(?:outcome|approved|approves|declared|reported|posts?|posted|announces results|net profit|net loss|profit (?:rises|falls)|considered and approved)\b",
    re.IGNORECASE,
)
_SEVERE_WORDS = re.compile(
    r"\b(?:fraud|scam|default|insolvency|bankruptcy|arrest|raid|ban(?:s|ned)?|suspends?|halts?|fire|explosion|recall|delist(?:ing)?|forensic|plunges?|crash(?:es|ed)?"
    r"|collapse|emergency|unscheduled|surprise|biggest|largest|first[- ]ever|historic|massive|record)\b",
    re.IGNORECASE,
)
_DAMPENERS = re.compile(r"\b(?:routine|clarification|intimation|reaffirm(?:s|ed)?|no impact|not material|immaterial)\b", re.IGNORECASE)
_LONG_TERM = re.compile(r"\b(?:long[- ]term|multi[- ]year|over the next (?:\d+|few|several) years|structural)\b", re.IGNORECASE)
_INTRADAY = re.compile(r"\b(?:intraday|immediately|after[- ]hours|today|this morning)\b", re.IGNORECASE)

_PERCENT = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d+)?)\s?(?:%|per ?cent|percent)", re.IGNORECASE)
_AMOUNT = re.compile(
    r"(?:(\u20b9|\brs\.?|\binr\b)|(\$|\busd\b|us\$|\u20ac|\beur\b))?\s?(\d[\d,]*(?:\.\d+)?)\s?"
    r"(lakh crore|crore|cr\b|lakh|million|mn\b|billion|bn\b|trillion|tn\b)?",
    re.IGNORECASE,
)
_BODY_LIMIT = 1500
_USD_CRORE = 8.5


def max_percent(text: str) -> float | None:
    """Largest percentage mentioned (0-1000), ``None`` when there is none."""
    values = [float(m.group(1)) for m in _PERCENT.finditer(text[:3000])]
    values = [v for v in values if v <= 1000]
    return max(values) if values else None


def amount_in_crore(text: str) -> float | None:
    """Largest money amount in INR crore (USD converted at a coarse constant); amounts need a currency or unit."""
    best: float | None = None
    for m in _AMOUNT.finditer(text[:3000]):
        inr, usd, raw, unit = m.group(1), m.group(2), m.group(3), (m.group(4) or "").lower()
        if not (inr or usd or unit):
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        scale = {"lakh crore": 1e5, "crore": 1.0, "cr": 1.0, "lakh": 0.01, "million": 0.1, "mn": 0.1, "billion": 100.0,
                 "bn": 100.0, "trillion": 1e5, "tn": 1e5}.get(unit, 1e-7 if not unit else 1.0)
        crore = value * scale
        if usd:
            crore *= _USD_CRORE if unit in ("crore", "cr", "lakh crore", "lakh") else _USD_CRORE * 10
        best = crore if best is None else max(best, crore)
    return best


# ----------------------------------------------------------------------------------------- classifier
def _category_hint(category: str | None) -> tuple[str | None, float, bool]:
    cat = re.sub(r"\s+", " ", (category or "").strip().lower())
    if not cat:
        return None, 0.0, False
    routine = any(key in cat for key in _ROUTINE_CATEGORIES)
    for key, event_type, weight in _CATEGORY_RULES:
        if key in cat:
            return event_type, weight, False
    return None, 0.0, routine


def _severity(base: float, event_type: str, full: str, rule_sev: float, rumour: bool, denial: bool) -> float:
    sev = max(base, rule_sev)
    pct = max_percent(full)
    if pct is not None and event_type not in ("central_bank", "macro_release"):
        sev += 0.25 if pct >= 20 else 0.15 if pct >= 10 else 0.08 if pct >= 5 else 0.0
    if event_type in ("order_win", "government_contract", "order_loss", "merger_acquisition", "fundraising", "buyback_dividend"):
        crore = amount_in_crore(full)
        if crore is not None:
            sev += 0.2 if crore >= 5000 else 0.12 if crore >= 1000 else 0.06 if crore >= 300 else 0.0
    if _SEVERE_WORDS.search(full):
        sev += 0.1
    if _DAMPENERS.search(full):
        sev -= 0.1
    if rumour:
        sev *= RUMOUR_SEVERITY
    if denial:
        sev *= DENIAL_SEVERITY
    return round(min(1.0, max(0.02, sev)), 3)


def classify_rules(
    title: str | None, text: str | None = "", *, category: str | None = None, source_kind: str | None = None
) -> EventDraft:
    """Deterministic classification of one article or exchange filing."""
    title = (title or "").strip()[:400]
    body = (text or "").strip()[:_BODY_LIMIT]
    full = f"{title}. {body}" if body else title
    scores: dict[str, float] = {}
    hits: dict[str, int] = {}
    pol_num: dict[str, float] = {}
    pol_den: dict[str, float] = {}
    pol_override: dict[str, float] = {}
    rule_sev: dict[str, float] = {}
    names: list[str] = []
    title_hits: dict[str, int] = {}
    for rule in _RULES:
        in_title = bool(title) and rule.pattern.search(title) is not None
        if not (in_title or (body and rule.pattern.search(body))):
            continue
        weight = rule.weight * (1.5 if in_title else 1.0)
        et = rule.event_type
        scores[et] = scores.get(et, 0.0) + weight
        hits[et] = hits.get(et, 0) + 1
        if in_title:
            title_hits[et] = title_hits.get(et, 0) + 1
        if rule.polarity is not None:
            if rule.exclusive:
                pol_override[et] = rule.polarity
            pol_num[et] = pol_num.get(et, 0.0) + weight * rule.polarity
            pol_den[et] = pol_den.get(et, 0.0) + weight
        if rule.severity is not None:
            rule_sev[et] = max(rule_sev.get(et, 0.0), rule.severity)
        names.append(rule.name)

    cat_type, cat_weight, routine_cat = _category_hint(category)
    rules_best = max(scores.values()) if scores else 0.0
    if cat_type:
        scores[cat_type] = scores.get(cat_type, 0.0) + cat_weight
        hits[cat_type] = hits.get(cat_type, 0) + 1
        names.append(f"category:{cat_type}")

    ranked = sorted(scores.items(), key=lambda kv: (-round(kv[1], 4), _PRIORITY_INDEX.get(kv[0], 99)))
    best = ranked[0][0] if ranked else "other"
    best_score = ranked[0][1] if ranked else 0.0
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    scheduled_cue = _SCHEDULED_CUES.search(full) is not None
    if best == "order_win" and "gov_contract" in names:
        best = "government_contract"
    elif best == "earnings_result" and scores.get("board_meeting", 0.0) > 0 and scheduled_cue and not _OUTCOME_CUES.search(full):
        best = "board_meeting"
    elif best == "board_meeting" and scores.get("earnings_result", 0.0) >= 2.5 and _OUTCOME_CUES.search(full):
        best = "earnings_result"
    if ranked and best != ranked[0][0]:
        best_score = scores.get(best, best_score)

    info = EVENT_TYPES[best]
    routine = bool(routine_cat and rules_best < 3.0 and source_kind == "exchange_filing" and best in ("other", "board_meeting"))
    if routine:
        best, info = "other", EVENT_TYPES["other"]

    official = source_kind in ("exchange_filing", "official")
    strong_rumour = len(_RUMOUR_STRONG.findall(title)) + len(_RUMOUR_STRONG.findall(body))
    weak_rumour = len(_RUMOUR_WEAK.findall(title)) * 2 + len(_RUMOUR_WEAK.findall(body))
    is_rumour = (not official) and (strong_rumour >= 1 or weak_rumour >= 2)
    denial = _DENIAL.search(full) is not None and not official
    if denial:
        is_rumour = False

    polarity = 0.0
    if not routine and best in pol_override:
        polarity = pol_override[best]
    elif not routine and pol_den.get(best, 0.0) > 0:
        polarity = max(-1.0, min(1.0, pol_num[best] / pol_den[best]))
    if denial or is_rumour:
        polarity *= 0.5

    base_severity = info.severity
    if best == "rating_action" and "analyst_rating" in names and "credit_rating" not in names:
        base_severity = 0.30
    severity = 0.02 if routine else _severity(base_severity, best, full, rule_sev.get(best, 0.0), is_rumour, denial)

    if routine or not scores:
        confidence = 0.35 if routine else 0.25
    else:
        confidence = 0.30 + 0.12 * min(hits.get(best, 0), 3) + (0.10 if title_hits.get(best) else 0.0)
        if cat_type == best:
            confidence += 0.25
        if second_score >= 0.8 * best_score and len(ranked) > 1 and ranked[1][0] != best:
            confidence -= 0.12
        if official:
            confidence += 0.10
        if best == "other":
            confidence = min(confidence, 0.35)
    if is_rumour:
        confidence *= 0.6 if strong_rumour else 0.8
    confidence = round(min(0.95, max(0.05, confidence)), 3)

    horizon = info.horizon
    if _LONG_TERM.search(full) and best not in ("macro_release", "bulk_block_deal"):
        horizon = "months"
    elif _INTRADAY.search(full) and best in ("macro_release", "bulk_block_deal", "central_bank"):
        horizon = "intraday"
    if is_rumour and horizon == "months":
        horizon = "weeks"
    scheduled = info.scheduled or (scheduled_cue and best in ("board_meeting", "earnings_result", "macro_release", "central_bank"))
    return EventDraft(
        event_type=best,
        severity=severity,
        confidence=confidence,
        horizon=horizon,
        polarity_hint=round(polarity, 3),
        is_rumour=is_rumour,
        scheduled=bool(scheduled),
        scope=info.scope,
        is_denial=denial,
        routine=routine,
        rules=tuple(names[:12]),
    )


# ------------------------------------------------------------------------------------ entity polarity
_DIR_DOWN_EXTRA = r"cuts?|eases?|slash(?:es|ed)?|lowers?|lowered|pares?|pared|trims?|trimmed"
_DIR_UP_EXTRA = r"hikes?|raises?|lifts?|boosts?|boosted|lifted"
_DIR_RX = re.compile(
    r"\b(?:(?P<down>de-?escalat\w*|dovish|lifts?\s+(?:the\s+)?(?:ban|sanctions?|curbs?|restrictions?|embargo|tariffs?|dut(?:y|ies))|"
    + _DOWN[3:-1]
    + r"|" + _DIR_DOWN_EXTRA + r")"
    r"|(?P<up>hawkish|escalat\w*|impos(?:es|ed)|" + _UP[3:-1] + r"|" + _DIR_UP_EXTRA + r"))\b",
    re.IGNORECASE,
)
_DEMAND_DIR_RX = re.compile(
    r"\b(?:(?P<down>slowdown|slows|slowing|weak demand|contraction|outflows?|selling|deficient|below[- ]normal|drought|delayed|de-?escalat\w*|dovish|"
    + _DOWN[3:-1]
    + r"|" + _DIR_DOWN_EXTRA + r")|(?P<up>stimulus|recovery|rebound|pickup|boost|inflows?|buying|strong demand|bumper|good monsoon|above[- ]normal|hawkish|"
    + _UP[3:-1]
    + r"|" + _DIR_UP_EXTRA + r"))\b",
    re.IGNORECASE,
)
_PRICE_DIR_RX = re.compile(r"\b(?:(?P<down>" + _DOWN[3:-1] + r")|(?P<up>" + _UP[3:-1] + r"))\b", re.IGNORECASE)
_STRONG_RX = re.compile(r"\b(?:surg\w+|soar\w*|plung\w+|crash\w*|spik\w+|tumbl\w+|slump\w*|jump\w*|plummet\w*|skyrocket\w*|tank\w*)\b", re.IGNORECASE)
_NEUTRAL_RX = re.compile(r"\b(?:holds?|keeps?|maintains?|pauses?|unchanged|status quo|steady|on hold)\b", re.IGNORECASE)
_RESTRICTIVE = re.compile(
    r"\b(?:output cuts?|production cuts?|cuts? (?:output|production|supply)|supply (?:cuts?|disruptions?|shortage|crunch|concerns?)|outage|embargo|attack|strike|"
    r"inventory draws?|stockpiles? (?:fall|drop|decline|draw)|shortage|disruption|extends? (?:output )?cuts?|export (?:ban|curbs?))\b",
    re.IGNORECASE,
)
_EXPANSIVE = re.compile(
    r"\b(?:supply (?:glut|surplus)|glut|surplus|inventory (?:builds?|buildup)|stockpiles? (?:rise|build|jump)|raises? (?:output|production)|ramps? up (?:output|production)|"
    r"boosts? (?:output|production)|output increase|oversupply)\b",
    re.IGNORECASE,
)
_ACQUIRE_VERB = re.compile(
    r"\b(?:to acquire|acquires?|acquired|buys?|to buy|bought|takes over|to take over|to merge with|merges? with|picks? up|to purchase|purchases?|snaps? up|bids? for)\b",
    re.IGNORECASE,
)
_CENTRAL_BANKS = frozenset({"macro:rbi", "macro:fed", "macro:ecb", "macro:boj"})
_ESCALATE_RX = re.compile(
    r"\b(?:escalat\w*|attacks?|strikes?|invades?|invasion|missiles?|air ?strikes?|blockade|hostilities|flare[- ]?up|retaliat\w*|tensions? (?:rise|mount|grow|flare))\b",
    re.IGNORECASE,
)
_DEESCALATE_RX = re.compile(
    r"\b(?:ceasefire|cease-fire|truce|peace (?:deal|talks|agreement)|de-?escalat\w*|tensions? (?:ease|cool|subside)|talks resume)\b", re.IGNORECASE
)
_ECON_WORDS = re.compile(
    r"\b(?:economy|economic|demand|growth|gdp|stimulus|slowdown|recession|data|inflation|jobs|exports|imports|activity|pmi|factory|property|consumption|spending)\b",
    re.IGNORECASE,
)
_DEMAND_IDS = frozenset({"macro:china_demand", "macro:us_economy", "macro:india_growth", "macro:monsoon", "macro:govt_capex", "macro:fii_flows"})
_SIGNED_SECTOR_TYPES = frozenset({"sector_regulation"})
_MACRO_TYPES = frozenset({"macro", "commodity", "fx", "rate", "index", "country", "institution"})
_PRICE_TYPES = frozenset({"commodity", "fx", "rate", "index"})  # judged by price verbs, not action verbs like 'cuts'
_AWARD_TYPES = frozenset({"order_win", "government_contract"})
_AWARD_VERB = re.compile(
    r"\b(?:awards?|awarded|places?|placed|gives?|gave|hands?|handed|selects?|selected|picks?|picked|assigns?|assigned|grants?|granted)\b",
    re.IGNORECASE,
)
_CUSTOMER_VERB = re.compile(r"(?:places?|placed|gives?|gave|hands?|handed)", re.IGNORECASE)
_AGENT_PREP = re.compile(r"\b(?:by|from)\s*$", re.IGNORECASE)
_MOVE_PCT = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d+)?)\s?(?:%|per ?cent|percent)", re.IGNORECASE)
_CLAUSE_SEPARATORS = ("; ", ". ", ": ", " | ")


def primary_company(entities: Sequence[Entity]) -> Entity | None:
    """The subject of the text: title mentions first, then the earliest mention, then confidence and frequency."""
    companies = [e for e in entities if e.type == "company"]
    if not companies:
        return None
    return max(companies, key=lambda e: (e.in_title, -(e.spans[0][0] if e.spans else 10**9), e.confidence, e.count))


def _award_winner(entities: Sequence[Entity], text: str) -> tuple[Entity | None, bool]:
    """(party the order goes TO, subject is the customer) from "X awards Y ..." / "order awarded to Y by X"."""
    verb = _AWARD_VERB.search(text)
    if verb is None:
        return None, False
    best: tuple[int, Entity] | None = None
    for entity in entities:
        if entity.type != "company":
            continue
        for start, _end, _invert in entity.spans:
            if start < verb.end() or start - verb.end() > 60 or _AGENT_PREP.search(text[max(0, start - 6) : start]):
                continue
            if best is None or start < best[0]:
                best = (start, entity)
            break
    return (best[1] if best else None), _CUSTOMER_VERB.fullmatch(verb.group(0)) is not None


def _clauses(text: str, spans: Sequence[tuple[int, int, bool]]) -> list[tuple[str, int, int, bool]]:
    """(clause, mention start, mention end, invert) per mention; clauses end at '; ', '. ', ': ' or ' | ' so that one
    entity's verbs are not read as another's."""
    out: list[tuple[str, int, int, bool]] = []
    for start, end, invert in spans:
        left = max(0, start - 160)
        for sep in _CLAUSE_SEPARATORS:
            i = text.rfind(sep, left, start)
            if i >= 0:
                left = max(left, i + len(sep))
        right = min(len(text), end + 200)
        for sep in _CLAUSE_SEPARATORS:
            j = text.find(sep, end, right)
            if j >= 0:
                right = min(right, j)
        out.append((text[left:right], start - left, end - left, invert))
    return out


def _nearest_direction(clause: str, start: int, end: int, rx: re.Pattern[str], *, neutral_stops: bool) -> int:
    """+1/-1 from the first direction word after the mention (else the last one just before it); 0 when unclear."""
    after = rx.search(clause, end)
    if neutral_stops:
        hold = _NEUTRAL_RX.search(clause, end)
        if hold and (after is None or hold.start() < after.start()):
            return 0
    if after is not None and after.start() - end <= 90:
        return 1 if after.lastgroup == "up" else -1
    earlier = [m for m in rx.finditer(clause) if m.start() < start]
    if earlier and start - earlier[-1].end() <= 40:
        return 1 if earlier[-1].lastgroup == "up" else -1
    return 0


def _intensity(text: str) -> float:
    """Escalation (+) vs de-escalation (-) of a geopolitical theme from the whole text."""
    up, down = _ESCALATE_RX.search(text), _DEESCALATE_RX.search(text)
    return 0.6 if up and not down else (-0.6 if down and not up else 0.0)


def _move_direction(entity: Entity, text: str, draft: EventDraft) -> float:
    """Direction of the variable named by a macro-type entity (up = positive) from the words around its mentions."""
    eid = entity.id
    if eid == "macro:opec":
        restrictive, expansive = _RESTRICTIVE.search(text), _EXPANSIVE.search(text)
        return 0.6 if restrictive and not expansive else (-0.6 if expansive and not restrictive else 0.0)
    if eid.startswith("macro:geopolitics"):
        return draft.polarity_hint if draft.event_type == "geopolitical_conflict" and draft.polarity_hint else _intensity(text)
    rx = _DEMAND_DIR_RX if eid in _DEMAND_IDS else (_PRICE_DIR_RX if entity.type in _PRICE_TYPES else _DIR_RX)
    mentions = _clauses(text, entity.spans) if entity.spans else [(text[:300], 0, 0, False)]
    total = 0.0
    strength = 0.0
    for clause, start, end, invert in mentions:
        if eid in ("macro:us_economy", "macro:china_demand") and not _ECON_WORDS.search(clause):
            continue
        sign = _nearest_direction(clause, start, end, rx, neutral_stops=eid in _CENTRAL_BANKS)
        if sign == 0:
            continue
        mag = 0.5 + (0.25 if _STRONG_RX.search(clause) else 0.0)
        pct = [float(m.group(1)) for m in _MOVE_PCT.finditer(clause)]
        if pct and max(pct) >= 2.0:
            mag += 0.15
        total += (-sign if invert else sign) * mag
        strength = max(strength, mag)
    if total == 0.0:
        if entity.type == "commodity" and draft.event_type == "commodity_supply":
            if _RESTRICTIVE.search(text) and not _EXPANSIVE.search(text):
                return 0.6
            if _EXPANSIVE.search(text) and not _RESTRICTIVE.search(text):
                return -0.6
        return 0.0
    return round(max(-1.0, min(1.0, strength if total > 0 else -strength)), 3)


def extract_moves(text: str, entities: Sequence[Entity]) -> dict[str, float]:
    """Signed percentage move quoted next to macro-type entities (``{'macro:brent': 3.1}``)."""
    out: dict[str, float] = {}
    for entity in entities:
        if entity.type not in _MACRO_TYPES or not entity.spans or entity.polarity == 0.0:
            continue
        for _start, end, _invert in entity.spans:
            chunk = text[end : end + 70]
            match = _MOVE_PCT.search(chunk)
            if match:
                value = float(match.group(1))
                if value <= 100:
                    out[entity.id] = round(value if entity.polarity > 0 else -value, 2)
                    break
    return out


def assign_polarity(draft: EventDraft, entities: Sequence[Entity], text: str) -> dict[str, float]:
    """Sign of the event for each entity (-1..1). ``text`` must be the string the entities were extracted from.

    Companies take the event's polarity hint only when they are the subject of a company-scope event (M&A distinguishes
    target from acquirer); sectors are signed only by regulation events; macro-type entities carry the direction of the
    variable itself. Everything else is 0 - unknown, not neutral wording.
    """
    out: dict[str, float] = {}
    primary = primary_company(entities)
    if draft.event_type in _AWARD_TYPES:
        winner, customer_subject = _award_winner(entities, text)
        if winner is not None:
            primary = winner
        elif customer_subject:
            primary = None  # "X places an order ...": X is the buyer, nobody here is signed
    company_scope = draft.scope in ("company", "global") or draft.event_type == "global_company"
    verb = _ACQUIRE_VERB.search(text) if draft.event_type == "merger_acquisition" else None
    n_companies = sum(1 for e in entities if e.type == "company")
    for entity in entities:
        value = 0.0
        if entity.type == "company":
            if draft.event_type == "merger_acquisition" and verb and entity.spans and n_companies >= 2:
                value = 0.1 if entity.spans[0][0] < verb.start() else 0.5
            elif company_scope and primary is not None and entity.id == primary.id:
                value = draft.polarity_hint
        elif entity.type == "sector":
            if draft.event_type in _SIGNED_SECTOR_TYPES:
                value = draft.polarity_hint
        elif entity.type in _MACRO_TYPES:
            value = _move_direction(entity, text, draft)
        out[entity.id] = round(max(-1.0, min(1.0, value)), 3)
    return out
