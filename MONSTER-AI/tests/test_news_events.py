from __future__ import annotations

import time

import pytest

from news_intelligence import events as ev
from news_intelligence.entities import Entity, EntityResolver
from tests.test_news_support import add_instruments

# (title, text, NSE category, source kind, accepted event types, expected polarity sign or None)
CASES = [
    ("Reliance Industries Q2 net profit rises 12% to Rs 19,000 crore, beats estimates", "", None, None, "earnings_result", 1),
    ("TCS Q2 profit falls 3% as deal ramp-ups slow; misses estimates", "", None, None, "earnings_result", -1),
    ("Tata Steel Q2 loss widens to Rs 1,200 crore", "", None, None, "earnings_result", -1),
    ("Zomato shares jump 8% on strong Q2 results", "", None, None, "earnings_result", None),
    ("Maruti Suzuki September sales rise 8%", "", None, None, "earnings_result", 1),
    ("L&T bags orders worth Rs 5,000-10,000 crore for its infrastructure business", "", None, None, "order_win", 1),
    ("BHEL wins Rs 2,500 crore order from NTPC for power project", "", None, None, "order_win", 1),
    ("HAL gets Rs 7,000 crore defence ministry contract for helicopters", "", None, None, "government_contract", 1),
    ("Company loses Rs 800 crore order after client cancels contract", "", None, None, "order_loss", -1),
    ("Adani Ports to acquire stake in Australian terminal for Rs 3,000 crore", "", None, None, "merger_acquisition", None),
    ("Tata Motors to demerge commercial vehicle business", "", None, None, "merger_acquisition", None),
    ("Vedanta announces QIP to raise Rs 5,000 crore", "", None, None, "fundraising", -1),
    ("Hindustan Zinc plans to raise Rs 2,000 crore via NCDs", "", None, None, "fundraising", None),
    ("Coal India board approves interim dividend of Rs 15 per share", "", None, None, "buyback_dividend", 1),
    ("Infosys announces Rs 18,000 crore share buyback", "", None, None, "buyback_dividend", 1),
    ("Company omits final dividend after weak year", "", None, None, "buyback_dividend", -1),
    ("Promoter sells 2% stake in Zomato via block deal", "", None, None, "promoter_transaction|bulk_block_deal", -1),
    ("Promoter group buys additional 1% stake in the company", "", None, None, "promoter_transaction", 1),
    ("Adani Enterprises promoter pledges additional shares", "", None, None, "pledge", -1),
    ("Promoter releases pledge on 5% of shares", "", None, None, "pledge", 1),
    ("Wipro CFO resigns", "", None, None, "management_change", -1),
    ("Statutory auditor resigns citing lack of information", "", None, None, "management_change", -1),
    ("Board appoints new managing director for the lender", "", None, None, "management_change", None),
    ("ICRA downgrades Vodafone Idea rating to BBB", "", None, None, "rating_action", -1),
    ("CRISIL upgrades outlook to positive for the steel maker", "", None, None, "rating_action", 1),
    ("Jefferies initiates coverage with buy rating and target price of Rs 1,800", "", None, None, "rating_action", None),
    ("SEBI bans Gensol promoters from securities market for fund diversion", "", None, None, "litigation_regulatory", -1),
    ("ED raids premises of XYZ in money laundering case", "", None, None, "litigation_regulatory", -1),
    ("Sun Pharma gets USFDA warning letter for Halol plant", "", None, None, "litigation_regulatory", -1),
    ("Dr Reddy's receives USFDA approval for generic drug", "", None, None, "litigation_regulatory", 1),
    ("High Court quashes tax demand against the company, grants relief", "", None, None, "litigation_regulatory", 1),
    ("NCLT admits insolvency petition against the company", "", None, None, "litigation_regulatory", -1),
    ("RBI keeps repo rate unchanged at 6.5%", "", None, None, "central_bank", None),
    ("RBI cuts repo rate by 25 bps to 6.25%", "", None, None, "central_bank", -1),
    ("Fed raises rates by 25 bps; Powell signals more hikes", "", None, None, "central_bank", 1),
    ("India's CPI inflation eases to 4.3% in September", "", None, None, "macro_release", None),
    ("US nonfarm payrolls beat estimates; unemployment rate at 3.8%", "", None, None, "macro_release", None),
    ("Rupee falls to record low of 88.5 against dollar", "", None, None, "macro_release", None),
    ("Brent crude jumps 4% as Houthi attacks disrupt Red Sea shipping", "", None, None, "geopolitical_conflict|commodity_supply", None),
    ("Ceasefire announced; truce holds along the border", "", None, None, "geopolitical_conflict", -1),
    ("OPEC+ agrees to extend output cuts through Q1", "", None, None, "commodity_supply", None),
    ("Gold hits record high as dollar weakens", "", None, None, "commodity_supply", None),
    ("US imposes 25% tariffs on steel and aluminium imports", "", None, None, "sanctions_tariffs", 1),
    ("US lifts sanctions on exports of the metals", "", None, None, "sanctions_tariffs", -1),
    ("India, UK sign free trade agreement", "", None, None, "trade_policy", None),
    ("Government cuts windfall tax on crude petroleum", "", None, None, "sector_regulation", None),
    ("SEBI tightens F&O norms for retail traders", "", None, None, "sector_regulation", -1),
    ("Union Budget 2026: Finance Minister announces capex boost", "", None, None, "sector_regulation", 1),
    ("Lok Sabha election results: NDA leads in early trends", "", None, None, "elections_politics", None),
    ("Nvidia Q3 revenue beats estimates on AI chip demand", "", None, None, "global_company", None),
    ("Sensex crashes 1,200 points as global selloff deepens", "", None, None, "market_stress", -1),
    ("Profit booking drags Nifty lower", "", None, None, "other", None),
    ("Nifty ends at record high; banks lead", "", None, None, "other", None),
    # NSE filings (category counts heavily)
    ("Intimation of Board Meeting to consider financial results for Q2", "", "Board Meeting", "exchange_filing", "board_meeting", None),
    ("Financial Results for the quarter ended September 30, 2026", "", "Financial Results", "exchange_filing", "earnings_result", None),
    ("Outcome of Board Meeting", "The Board approved the unaudited financial results for the quarter and half year ended Sep 30", "Outcome of Board Meeting", "exchange_filing", "earnings_result", None),
    ("Outcome of Board Meeting", "The Board recommended a final dividend of Rs 5 per share", "Outcome of Board Meeting", "exchange_filing", "buyback_dividend", 1),
    ("Award of order", "Company has received an order worth Rs 450 crore from Indian Railways", "Award of Order / Receipt of Order", "exchange_filing", "government_contract|order_win", 1),
    ("Credit Rating", "CRISIL reaffirmed AA+ rating with stable outlook", "Credit Rating", "exchange_filing", "rating_action", None),
    ("Appointment of Mr X as Additional Director", "", "Appointment", "exchange_filing", "management_change", None),
    ("Acquisition of stake in subsidiary", "", "Acquisition", "exchange_filing", "merger_acquisition", None),
    ("Dividend", "Record date for final dividend of Rs 5", "Dividend", "exchange_filing", "buyback_dividend", 1),
    ("Q2 PAT up 25%", "ABC Ltd reports 25% growth in Q2 PAT", "Press Release", "exchange_filing", "earnings_result", 1),
    ("Disclosure under SEBI (Substantial Acquisition of Shares and Takeovers) Regulations", "", "Disclosure under SEBI Takeover Regulations", "exchange_filing", "promoter_transaction", None),
    ("Fire at plant", "Fire broke out at the company's plant; operations suspended", "Updates", "exchange_filing", "other", -1),
]


@pytest.mark.parametrize(("title", "text", "category", "kind", "types", "sign"), CASES)
def test_rule_classifier(title, text, category, kind, types, sign):
    draft = ev.classify_rules(title, text, category=category, source_kind=kind)
    assert draft.event_type in types.split("|"), (draft.event_type, draft.rules)
    assert 0.0 <= draft.severity <= 1.0 and 0.0 <= draft.confidence <= 0.95
    assert draft.horizon in ev.HORIZONS and draft.scope in ev.SCOPES
    assert draft.event_type in ev.EVENT_TYPES
    if sign is not None:
        assert draft.polarity_hint * sign > 0, draft.polarity_hint
    if draft.event_type == "central_bank" and "unchanged" in title:
        assert draft.polarity_hint == 0.0


def test_the_corpus_has_enough_cases():
    assert len(CASES) >= 40


def test_taxonomy_is_complete_and_consistent():
    expected = {
        "earnings_result", "earnings_guidance", "order_win", "order_loss", "merger_acquisition", "fundraising",
        "buyback_dividend", "promoter_transaction", "pledge", "management_change", "rating_action", "litigation_regulatory",
        "government_contract", "board_meeting", "corporate_action", "bulk_block_deal", "macro_release", "central_bank",
        "geopolitical_conflict", "sanctions_tariffs", "commodity_supply", "trade_policy", "elections_politics",
        "sector_regulation", "global_company", "market_stress", "other",
    }
    assert set(ev.EVENT_TYPES) == expected == set(ev.EVENT_TYPE_NAMES)
    for info in ev.EVENT_TYPES.values():
        assert info.scope in ev.SCOPES and info.horizon in ev.HORIZONS and 0.0 < info.severity < 1.0
    assert set(ev._PRIORITY) == expected  # tie-break order covers every type


@pytest.mark.parametrize(
    "title",
    [
        "Reliance reportedly in talks to buy stake in rival, sources said",
        "Tata Motors in talks to acquire a stake, unconfirmed reports say",
        "Company may acquire competitor, likely to announce next week",
    ],
)
def test_rumour_words_flag_and_lower_confidence(title):
    rumour = ev.classify_rules(title)
    firm = ev.classify_rules("Reliance acquires a stake in rival")
    assert rumour.is_rumour and not firm.is_rumour
    assert rumour.confidence < firm.confidence and rumour.severity <= firm.severity


def test_rumour_detection_is_skipped_for_filings_and_ignores_the_month_may():
    filing = ev.classify_rules("Acquisition", "The company may consider acquiring a stake", category="Acquisition", source_kind="exchange_filing")
    assert not filing.is_rumour
    assert not ev.classify_rules("Results scheduled in May 2026 for Q4").is_rumour


def test_denial_is_not_a_rumour_and_is_flagged():
    draft = ev.classify_rules("Company denies reports of acquisition talks")
    assert draft.is_denial and not draft.is_rumour and draft.severity < ev.classify_rules("Company acquires rival firm").severity


def test_routine_filings_are_flagged_not_dropped():
    trading = ev.classify_rules("Closure of Trading Window", "", category="Trading Window", source_kind="exchange_filing")
    meet = ev.classify_rules("Analyst meet", "Schedule of call", category="Analysts/Institutional Investor Meet/Con. Call Updates", source_kind="exchange_filing")
    assert trading.routine and meet.routine and trading.severity <= 0.05
    # a routine-looking category with a strong textual signal is not routine
    strong = ev.classify_rules("Investor Presentation", "Company receives order worth Rs 900 crore from Ministry of Defence", category="Investor Presentation", source_kind="exchange_filing")
    assert not strong.routine and strong.event_type in ("government_contract", "order_win")


def test_severity_grows_with_magnitude():
    small = ev.classify_rules("Firm bags order worth Rs 50 crore")
    large = ev.classify_rules("Firm bags order worth Rs 9,000 crore")
    assert large.severity > small.severity
    assert ev.classify_rules("Profit rises 3%").severity < ev.classify_rules("Profit surges 45%").severity
    fraud = ev.classify_rules("SEBI probes alleged fraud at the company")
    assert fraud.severity >= 0.7


def test_scheduled_and_horizon():
    assert ev.classify_rules("Board meeting on Nov 5 to consider results").scheduled
    assert ev.classify_rules("RBI MPC meeting outcome: repo rate unchanged").scheduled
    assert not ev.classify_rules("Firm bags order worth Rs 900 crore").scheduled
    assert ev.classify_rules("Structural multi-year reform in power distribution announced by regulator").horizon == "months"
    assert ev.classify_rules("Bulk deal today: fund buys 2% stake via block deal").horizon == "intraday"


def test_empty_text_defaults():
    draft = ev.classify_rules("")
    assert draft.event_type == "other" and draft.confidence <= 0.3 and not draft.routine


def test_amount_and_percent_parsing():
    assert ev.amount_in_crore("order worth Rs 5,000 crore") == 5000.0
    assert ev.amount_in_crore("worth Rs 12000 crore") == 12000.0
    assert ev.amount_in_crore("a $2 billion deal") == pytest.approx(17000.0)
    assert ev.amount_in_crore("USD 300 million contract") == pytest.approx(2550.0)
    assert ev.amount_in_crore("Rs 2.5 lakh crore outlay") == pytest.approx(250000.0)
    assert ev.amount_in_crore("no money here, only 25 bps") is None
    assert ev.max_percent("up 3.5% and down 12 per cent") == 12.0 and ev.max_percent("nothing") is None
    assert ev.max_percent("share of 5000%") is None


@pytest.mark.parametrize("payload", ["profit " * 900, "a " * 3000, "rises " * 900, "net profit " + "x" * 5000, "(" * 3000])
def test_rules_are_not_vulnerable_to_regex_backtracking(payload):
    started = time.perf_counter()
    ev.classify_rules(payload[:400], payload)
    assert time.perf_counter() - started < 6.0  # exponential backtracking would take minutes


# --------------------------------------------------------------------------------------- polarity
@pytest.fixture()
def resolver(store):
    add_instruments(store)
    return EntityResolver(store)


def _polarity(resolver, text, **kwargs):
    entities = resolver.extract(text)
    draft = ev.classify_rules(text, "", **kwargs)
    pol = ev.assign_polarity(draft, entities, text)
    for entity in entities:
        entity.polarity = pol[entity.id]
    return draft, entities, pol


def test_company_event_polarity_goes_to_the_subject_only(resolver):
    _draft, _ents, pol = _polarity(resolver, "TCS wins Rs 4,000 crore order; Infosys shares rise in sympathy")
    assert pol["TCS"] > 0 and pol["INFY"] == 0.0


@pytest.mark.parametrize(
    ("text", "winner", "customer"),
    [
        ("NTPC awards Reliance Rs 5,000 crore solar module contract", "RELIANCE", "NTPC"),
        ("Reliance bags Rs 5,000 crore order from NTPC for solar modules", "RELIANCE", "NTPC"),
        ("Reliance awarded Rs 5,000 crore order by NTPC for solar modules", "RELIANCE", "NTPC"),
        ("Order awarded to Reliance by NTPC: Rs 5,000 crore solar module contract", "RELIANCE", "NTPC"),
    ],
)
def test_order_beneficiary_is_the_winner_not_the_customer(resolver, text, winner, customer):
    draft, _ents, pol = _polarity(resolver, text)
    assert draft.event_type in {"order_win", "government_contract"}
    assert pol[winner] > 0 and pol[customer] == 0.0


def test_a_buyer_placing_an_order_is_not_signed(resolver):
    draft, _ents, pol = _polarity(resolver, "IndiGo places order for 500 aircraft worth $50 billion")
    assert draft.event_type == "order_win"
    assert pol["INDIGO"] == 0.0


def test_wording_alone_never_signs_an_unrelated_event(resolver):
    _draft, _ents, pol = _polarity(resolver, "Analysts say Reliance is a strong and positive story, bullish outlook")
    assert pol["RELIANCE"] == 0.0


def test_acquirer_and_target_roles(resolver):
    text = "Reliance to acquire stake in Wipro for Rs 3,000 crore"
    draft, _ents, pol = _polarity(resolver, text)
    assert draft.event_type == "merger_acquisition"
    assert pol["WIPRO"] > pol["RELIANCE"] > 0


def test_macro_directions_and_moves(resolver):
    text = "Brent crude jumps 3.1% as Red Sea attack escalates; rupee slumps; RBI cuts rate; OPEC extends output cuts"
    draft, entities, pol = _polarity(resolver, text)
    assert pol["macro:brent"] > 0 and pol["macro:usdinr"] > 0  # rupee weakness = USDINR up
    assert pol["macro:rbi"] < 0 and pol["macro:opec"] > 0 and pol["macro:geopolitics_mideast"] > 0
    moves = ev.extract_moves(text, entities)
    assert moves["macro:brent"] == pytest.approx(3.1)


def test_central_bank_hold_is_neutral_and_demand_words_sign_countries(resolver):
    _d, _e, pol = _polarity(resolver, "RBI keeps repo rate unchanged at 6.5%")
    assert pol["macro:rbi"] == 0.0
    _d, _e, pol = _polarity(resolver, "China stimulus boosts demand; US growth data slows sharply")
    assert pol["macro:china_demand"] > 0


def test_verbs_of_other_entities_do_not_leak_across_clauses(resolver):
    _d, _e, pol = _polarity(resolver, "HPCL and IOC fall as crude jumps 4%; gold falls")
    assert pol["macro:brent"] > 0 > pol["macro:gold"]


def test_lifting_sanctions_is_a_decrease_and_supply_cuts_lift_commodities(resolver):
    _d, _e, pol = _polarity(resolver, "US lifts sanctions on Russia; ceasefire holds in Ukraine")
    assert pol["macro:sanctions"] < 0
    _d, _e, pol = _polarity(resolver, "OPEC announces production cuts as crude inventories draw")
    assert pol["macro:opec"] > 0 and pol["macro:brent"] > 0


def test_sector_polarity_only_for_regulation_events(resolver):
    _d, _e, pol = _polarity(resolver, "RBI eases norms for banks, relief for lenders")
    assert pol["sector:financial_services"] > 0
    _d, _e, pol = _polarity(resolver, "Banks rally as IT services stocks slump")
    assert all(v == 0.0 for v in pol.values())


def test_assign_polarity_with_hand_built_entities():
    draft = ev.EventDraft(event_type="order_win", polarity_hint=0.6, scope="company")
    ents = [Entity("company", "AAA", "A", 1, 0.9, in_title=True), Entity("company", "BBB", "B", 1, 0.9)]
    assert ev.assign_polarity(draft, ents, "text") == {"AAA": 0.6, "BBB": 0.0}
    macro = ev.EventDraft(event_type="central_bank", polarity_hint=0.7, scope="macro")
    assert ev.assign_polarity(macro, ents, "text") == {"AAA": 0.0, "BBB": 0.0}
    assert ev.extract_moves("x", ents) == {}
