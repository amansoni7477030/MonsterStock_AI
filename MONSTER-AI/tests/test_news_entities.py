from __future__ import annotations

import json
import time

import pytest

from news_intelligence import entities as en
from news_intelligence.entities import Entity, EntityResolver, entity_from_dict
from tests.test_news_support import INSTRUMENTS, add_instruments

EXTRA = [
    ("DRREDDY", "Dr. Reddy's Laboratories Limited", "Healthcare", 1),
    ("SAIL", "Steel Authority of India Limited", "Metals & Mining", 1),
    ("TITAN", "Titan Company Limited", "Consumer Durables", 1),
    ("GAIL", "GAIL (India) Limited", "Oil Gas & Consumable Fuels", 1),
    ("J&KBANK", "Jammu & Kashmir Bank Limited", "Financial Services", 0),
    ("NAM-INDIA", "Nippon Life India Asset Management Limited", "Financial Services", 0),
]


@pytest.fixture()
def resolver(store):
    add_instruments(store, INSTRUMENTS + EXTRA)
    return EntityResolver(store)


def ids(resolver, text, **kwargs):
    return {e.id: e for e in resolver.extract(text, **kwargs)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reliance Industries and TCS gain", {"RELIANCE", "TCS"}),
        ("RIL's Jio hikes mobile tariffs", {"RELIANCE", "sector:telecom"}),
        ("Tata Consultancy Services wins deal", {"TCS"}),
        ("HDFC Bank and State Bank of India lead", {"HDFCBANK", "SBIN"}),
        ("SBI posts record profit", {"SBIN"}),
        ("L&T bags order; Larsen & Toubro shares up; Larsen and Toubro", {"LT"}),
        ("M&M and Mahindra & Mahindra report sales", {"M&M"}),
        ("M & M launches new SUV", {"M&M"}),
        ("BAJAJ-AUTO exports rise", {"BAJAJ-AUTO"}),
        ("Bajaj Auto exports rise", {"BAJAJ-AUTO"}),
        ("Dr. Reddy's gets nod; Dr Reddys shares up", {"DRREDDY"}),
        ("J&KBANK jumps on results", {"J&KBANK"}),
        ("NAM-INDIA gains as markets rally", {"NAM-INDIA"}),
        ("Infosys, Wipro and Hindustan Petroleum fall", {"INFY", "WIPRO", "HINDPETRO"}),
        ("HPCL and Indian Oil face under-recoveries", {"HINDPETRO", "IOC"}),
    ],
)
def test_company_extraction(resolver, text, expected):
    found = {e.id for e in resolver.extract(text) if e.type == "company"}
    assert expected & {e for e in expected if not e.startswith("sector:")} <= found
    assert found <= {"RELIANCE", "TCS", "HDFCBANK", "SBIN", "LT", "M&M", "BAJAJ-AUTO", "DRREDDY", "J&KBANK", "NAM-INDIA",
                     "INFY", "WIPRO", "HINDPETRO", "IOC"}


def test_company_entities_carry_type_name_count_and_confidence(resolver):
    found = ids(resolver, "TCS shares rise after Tata Consultancy Services wins deal; TCS leads")
    tcs = found["TCS"]
    assert tcs.type == "company" and tcs.name == "Tata Consultancy Services Limited"
    assert tcs.count == 3 and 0.9 <= tcs.confidence <= 0.98
    assert tcs.to_dict()["id"] == "TCS" and set(tcs.to_dict()) >= {"type", "id", "name", "count", "confidence", "polarity"}


@pytest.mark.parametrize(
    ("text", "present"),
    [
        ("TCS rules: tax collected at source on foreign remittance to rise", False),
        ("TCS on foreign tours: new rate announced", False),
        ("TCS shares rise as the IT major wins a deal", True),
        ("ITC cigarette volumes rise", True),
        ("Input tax credit claims under GST: ITC mismatch notices", False),
        ("itc and tcs are lowercase words here", False),
        ("Olympic committee: IOC announces hosts", False),
        ("IOC and BPCL face under-recoveries", True),
    ],
)
def test_ambiguous_tickers_need_context(resolver, text, present):
    symbols = {"TCS", "ITC", "IOC"} & set(ids(resolver, text))
    assert bool(symbols) is present, symbols


def test_all_caps_text_keeps_acronym_tickers_but_drops_word_like_ones(resolver):
    found = ids(resolver, "SENSEX ENDS HIGHER AS TCS AND ITC LEAD GAINERS WHILE SAIL AND TITAN SLIP ON PROFIT TAKING")
    assert {"TCS", "ITC"} <= set(found) and not ({"SAIL", "TITAN"} & set(found))
    mixed = ids(resolver, "Sensex ends higher: SAIL slips while TITAN gains")
    assert {"SAIL", "TITAN"} <= set(mixed)
    assert "SAIL" not in ids(resolver, "The ship will sail on Tuesday")


def test_common_words_never_become_company_aliases(resolver):
    assert ids(resolver, "The titan of industry spoke about reliance on imports and gas prices") .get("TITAN") is None
    assert "RELIANCE" not in ids(resolver, "Reliance on imports is rising")
    assert "RELIANCE" in ids(resolver, "Reliance shares gain; Reliance Retail expands")


def test_sector_and_macro_entities(resolver):
    found = ids(resolver, "Banks and IT services stocks rally; oil marketing companies fall as crude oil prices rise. Nasdaq slumps")
    assert {"sector:financial_services", "sector:it_services", "sector:oil_marketing", "macro:brent", "macro:nasdaq"} <= set(found)
    assert found["macro:brent"].type == "commodity" and found["sector:it_services"].type == "sector"
    macro = ids(resolver, "The rupee fell as the dollar index surged; US Treasury yields rose; Fed holds rates; RBI watches; China demand weak")
    assert {"macro:usdinr", "macro:dxy", "macro:us10y", "macro:fed", "macro:rbi", "macro:china_demand"} <= set(macro)
    assert "macro:us_economy" not in macro  # 'US' is consumed by the longer 'US Treasury yields' alias
    assert macro["macro:usdinr"].type == "fx" and macro["macro:us10y"].type == "rate" and macro["macro:rbi"].type == "institution"
    assert macro["macro:china_demand"].type == "country"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Brent crude rises", "macro:brent"), ("WTI futures fall", "macro:wti"), ("crude oil prices", "macro:brent"),
        ("natural gas prices spike", "macro:natgas"), ("gold hits record", "macro:gold"), ("silver rallies", "macro:silver"),
        ("copper slumps", "macro:copper"), ("aluminium prices up", "macro:aluminium"), ("rupee weakens past 85", "macro:usdinr"),
        ("USD/INR hits 86", "macro:usdinr"), ("dollar index climbs", "macro:dxy"), ("US 10-year yield jumps", "macro:us10y"),
        ("Federal Reserve holds rates", "macro:fed"), ("RBI keeps repo unchanged", "macro:rbi"),
        ("SEBI tightens F&O norms", "macro:sebi"), ("ECB cuts rates", "macro:ecb"), ("Bank of Japan hikes", "macro:boj"),
        ("OPEC+ extends cuts", "macro:opec"), ("Trump tariffs on India", "macro:tariffs"),
        ("sanctions on Russia", "macro:sanctions"), ("China slowdown", "macro:china_demand"),
        ("Red Sea attacks disrupt shipping", "macro:geopolitics_mideast"), ("Strait of Hormuz tensions", "macro:geopolitics_mideast"),
        ("Taiwan tensions hit chip stocks", "macro:geopolitics_taiwan"), ("semiconductor stocks fall", "macro:sox"),
        ("Nasdaq falls", "macro:nasdaq"), ("Nikkei slumps", "macro:nikkei"), ("Hang Seng falls", "macro:hsi"),
        ("VIX spikes", "macro:vix"), ("Nifty Bank gains", "sector:financial_services"), ("PSU banks rally", "sector:psu_banks"),
        ("OMCs face under-recoveries", "sector:oil_marketing"), ("airlines rally", "sector:aviation"),
        ("telecom tariffs hike", "sector:telecom"), ("cement demand", "sector:cement"), ("pharma stocks", "sector:pharma"),
        ("Nvidia earnings", "global:NVDA"),
    ],
)
def test_macro_sector_and_global_vocabulary(resolver, text, expected):
    assert expected in ids(resolver, text)


def test_crude_pattern_and_tariff_pattern_have_exclusions(resolver):
    assert "macro:brent" in ids(resolver, "Crude jumps on supply fears")
    assert "macro:brent" not in ids(resolver, "Crude palm oil and crude steel output rise")
    assert "macro:tariffs" in ids(resolver, "US imposes tariffs on imports")
    assert "macro:tariffs" not in ids(resolver, "Jio hikes mobile tariffs and telecom tariffs")


def test_case_sensitive_aliases_for_us_and_fed(resolver):
    assert "macro:us_economy" in ids(resolver, "Stocks in the US fell")
    assert "macro:us_economy" not in ids(resolver, "Give us a call; let us know")
    assert "macro:fed" in ids(resolver, "The Fed signals caution")
    assert "macro:fed" not in ids(resolver, "He was fed up with the delays")


def test_overlapping_mentions_are_counted_once(resolver):
    found = ids(resolver, "Fed holds rates after the Federal Reserve meeting; Reliance Industries (Reliance) gains")
    assert found["macro:fed"].count == 2 and found["RELIANCE"].count == 2


def test_title_mentions_raise_confidence_and_spans_index_the_scanned_text(resolver):
    with_title = ids(resolver, "Reliance Industries reported results", title="TCS Q2 results")
    assert with_title["TCS"].in_title and not with_title["RELIANCE"].in_title
    scan = "TCS Q2 results. Reliance Industries reported results"
    start, end, invert = with_title["RELIANCE"].spans[0]
    assert scan[start:end] == "Reliance Industries" and invert is False
    rupee = ids(resolver, "rupee slumps")["macro:usdinr"]
    assert rupee.spans[0][2] is True  # 'rupee' reads inverted: weaker rupee = USDINR up
    assert ids(resolver, "USD/INR rises")["macro:usdinr"].spans[0][2] is False


def test_extract_is_bounded_and_fast(resolver):
    assert resolver.extract("") == [] and resolver.extract(None) == []
    big = " ".join(["Reliance Industries and the market moved as analysts debated crude oil prices and HDFC Bank results"] * 60)
    started = time.perf_counter()
    for _ in range(15):
        out = resolver.extract(big[:6000], max_entities=3)
    assert time.perf_counter() - started < 10.0  # a catastrophic pattern would take minutes, not seconds
    assert len(out) <= 3
    many = " ".join(row[1] for row in INSTRUMENTS + EXTRA)
    assert len(resolver.extract(many)) <= en.MAX_ENTITIES


def test_resolver_rebuilds_when_the_instrument_count_changes(store):
    add_instruments(store, INSTRUMENTS[:2])
    resolver = EntityResolver(store)
    assert "QUASARTEC" not in ids(resolver, "Quasar Widgets Limited shares gain")
    builds = resolver.builds
    resolver.extract("anything")
    assert resolver.builds == builds  # cached
    store.upsert_instrument("QUASARTEC", "equity", name="Quasar Widgets Limited", industry="Capital Goods")
    assert "QUASARTEC" in ids(resolver, "Quasar Widgets Limited shares gain")
    assert resolver.builds == builds + 1
    assert resolver.refresh() is False and resolver.refresh(force=True) is True


def test_resolve_by_id_symbol_and_name(resolver):
    assert resolver.resolve("RELIANCE").id == "RELIANCE"
    assert resolver.resolve("reliance").id == "RELIANCE"
    assert resolver.resolve("Reliance Industries").id == "RELIANCE"
    assert resolver.resolve("sector:pharma").type == "sector"
    assert resolver.resolve("macro:brent").type == "commodity"
    assert resolver.resolve("Brent crude").id == "macro:brent"
    assert resolver.resolve("global:NVDA").type == "company"
    assert resolver.resolve("TOTALLY UNKNOWN INC") is None and resolver.resolve("") is None and resolver.resolve(None) is None
    assert resolver.resolve(123) is None  # type: ignore[arg-type]


def test_lookup_helpers(resolver):
    assert resolver.company_name("TCS") == "Tata Consultancy Services Limited" and resolver.company_name("NOPE") is None
    assert resolver.industry_node("TCS") == "sector:information_technology"
    assert resolver.industry_node("M&M") == "sector:automobile_and_auto_components"
    assert resolver.industry_node("NOPE") is None
    assert {"TCS", "sector:pharma", "macro:brent", "global:NVDA"} <= resolver.known_ids()


def test_entity_from_dict_is_lenient():
    assert entity_from_dict("TCS").id == "TCS" and entity_from_dict("TCS").type == "company"
    assert entity_from_dict("sector:pharma").type == "sector" and entity_from_dict("macro:brent").type == "macro"
    full = entity_from_dict({"id": "X", "type": "company", "name": "X Ltd", "count": "3", "confidence": 5, "polarity": -9, "move_pct": "2.5"})
    assert full == Entity("company", "X", "X Ltd", 3, 1.0, -1.0, 2.5)
    assert entity_from_dict({"symbol": "Y"}).id == "Y" and entity_from_dict({"entity_id": "Z"}).id == "Z"
    bad = entity_from_dict({"id": "X", "confidence": "nan", "count": "oops", "polarity": None, "move_pct": "bad", "type": "alien"})
    assert bad.count == 1 and bad.move_pct is None and bad.type == "company"
    assert entity_from_dict({"no": "id"}) is None and entity_from_dict(5) is None and entity_from_dict(None) is None


def test_custom_alias_file_and_failure_modes(store, tmp_path):
    add_instruments(store, INSTRUMENTS[:2])
    seed = tmp_path / "aliases.json"
    seed.write_text(json.dumps({
        "companies": [{"symbol": "TCS", "aliases": ["Tata Consulting"]}, {"symbol": ""}],
        "sectors": [{"id": "sector:custom", "name": "Custom", "keywords": ["widget makers"]}, {"keywords": ["no id"]}],
        "macros": [{"id": "macro:custom", "type": "weird", "name": "Custom macro", "keywords": ["gizmo index"], "cs_keywords": ["GZ"]}, {}],
        "patterns": [{"id": "macro:custom", "regex": "(", "flags": "ci"}, {"id": "missing", "regex": "x"}, {"id": "macro:custom", "regex": "\\bgizmo\\b"}],
        "global_companies": [{"id": "global:ACME", "name": "Acme", "aliases": ["Acme Corp"]}, {}],
        "ambiguous_symbols": ["TCS"],
    }), encoding="utf-8")
    resolver = EntityResolver(store, aliases_path=seed)
    found = ids(resolver, "Tata Consulting and widget makers rise; GZ and gizmo index up; Acme Corp wins")
    assert {"TCS", "sector:custom", "macro:custom", "global:ACME"} <= set(found)
    assert found["macro:custom"].type == "macro"
    missing = EntityResolver(store, aliases_path=tmp_path / "missing.json")
    assert "TCS" in ids(missing, "Tata Consultancy Services wins")  # instrument names still work
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    assert "TCS" in ids(EntityResolver(store, aliases_path=broken), "Tata Consultancy Services wins")
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]", encoding="utf-8")
    assert "TCS" in ids(EntityResolver(store, aliases_path=listy), "Tata Consultancy Services wins")


def test_seed_files_are_internally_consistent():
    from pathlib import Path

    root = Path(en.__file__).resolve().parent.parent / "data" / "seeds"
    aliases = json.loads((root / "aliases.json").read_text(encoding="utf-8"))
    graph = json.loads((root / "graph_seed.json").read_text(encoding="utf-8"))
    graph_ids = {n["id"] for n in graph["nodes"]} | {s["symbol"] for s in graph["stocks"]}
    for section in ("sectors", "macros", "global_companies"):
        for item in aliases[section]:
            assert item["id"] in graph_ids, f"{item['id']} is not a node in the graph seed"
            assert item.get("type", "sector" if section == "sectors" else "company") in en.ENTITY_TYPES
    assert len(aliases["companies"]) >= 150
    assert {c["symbol"] for c in aliases["companies"]} >= {"RELIANCE", "HDFCBANK", "TATAMOTORS", "LT", "M&M", "SBIN", "ITC", "TCS"}
    assert set(aliases["ambiguous_symbols"]) >= {"LT", "ITC", "TCS", "IOC", "GAIL", "BEL"}
