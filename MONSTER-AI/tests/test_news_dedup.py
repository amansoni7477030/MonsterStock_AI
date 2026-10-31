from __future__ import annotations

import pytest

from news_intelligence import dedup
from news_intelligence.dedup import (
    Clusterer,
    ClusterEvent,
    Decision,
    Item,
    Origin,
    Signature,
    base_novelty,
    event_confidence,
    novelty_at,
)

T0 = 1_793_000_000.0  # arbitrary epoch second
H = 3600.0
DAY = 86400.0

TITLE = "Reliance Industries bags Rs 5,000 crore order from NTPC for solar module supply"
BODY = (
    "Reliance Industries has received a Rs 5,000 crore order from NTPC for supply of solar modules over three years, "
    "the company said in a stock exchange filing on Monday. Execution begins next quarter."
)


def make_item(
    ref: str = "a1",
    title: str = TITLE,
    body: str = BODY,
    *,
    seen: float = T0,
    owner: str = "economictimes",
    quality: float = 0.6,
    confidence: float = 0.7,
    event_type: str = "order_win",
    polarity: float = 0.6,
    primary: str | None = "RELIANCE",
    rumour: bool = False,
    denial: bool = False,
    uhash: str = "",
    domain: str = "economictimes.indiatimes.com",
    kind: str = "article",
    filing_key: str | None = None,
) -> Item:
    return Item(
        ref, kind, seen, Signature.build(title, body), owner, quality, confidence, event_type, polarity, primary,
        rumour, denial, uhash, domain, filing_key,
    )


class Harness:
    def __init__(self, **kwargs) -> None:
        self.clusterer = Clusterer(**kwargs)
        self.counter = 0

    def feed(self, item: Item) -> tuple[Decision, ClusterEvent]:
        decision = self.clusterer.decide(item)
        self.counter += 1
        key = f"k{self.counter}" if decision.action == "new" else decision.event.key
        return decision, self.clusterer.commit(item, decision, key=key, event_id=self.counter)


def test_fifty_syndicated_copies_are_one_event_with_one_origin():
    h = Harness()
    first, event = h.feed(make_item("a0", owner="pti", domain="ptinews.com"))
    assert first.action == "new" and first.novelty == 1.0
    for i in range(1, 50):
        decision, event = h.feed(make_item(f"a{i}", seen=T0 + i * 60, owner=f"pub{i}", domain=f"pub{i}.example.com"))
        assert decision.action == "attach" and decision.event.key == "k1"
        assert decision.match in {"exact_content", "near_simhash", "near_cosine"}
    assert len(h.clusterer) == 1
    assert event.n_articles == 50 and event.n_independent == 1  # all copies of the same text share one origin
    assert len(event.origins) == 1 and event.origins[0].n == 50
    assert len(event.origins[0].domains) == dedup.MAX_DOMAINS  # bounded


def test_lightly_edited_copies_still_merge():
    h = Harness()
    h.feed(make_item("a0", owner="pti"))
    variants = [
        (TITLE + " - Business Standard", BODY),
        (TITLE.replace("bags", "wins"), BODY),
        (TITLE, BODY + " Shares were up 2% in early trade."),
        (TITLE.upper(), BODY.lower()),
    ]
    for i, (title, body) in enumerate(variants, start=1):
        decision, event = h.feed(make_item(f"v{i}", title, body, seen=T0 + i * 600, owner=f"pub{i}"))
        assert decision.action == "attach", (title, decision.match)
    assert len(h.clusterer) == 1 and event.n_independent == 1


def test_independent_writers_of_one_story_are_corroboration_not_new_events():
    h = Harness()
    h.feed(make_item("a0", owner="pti"))
    rewrite = make_item(
        "a1",
        "NTPC awards Reliance Rs 5,000 crore solar module supply contract",
        "NTPC has placed an order worth Rs 5,000 crore with Reliance Industries for the supply of solar modules; "
        "deliveries will run for three years, people aware of the matter said.",
        seen=T0 + 2 * H, owner="mint", domain="livemint.com",
    )
    decision, event = h.feed(rewrite)
    assert decision.action == "attach" and decision.match == "story_link" and decision.origin is None
    third = make_item(
        "a2", "Reliance gets NTPC solar module order worth Rs 5,000 crore",
        "Reliance Industries won a Rs 5,000 crore NTPC order to supply solar modules over three years.",
        seen=T0 + 3 * H, owner="reuters", domain="reuters.com",
    )
    decision, event = h.feed(third)
    assert decision.action == "attach"
    assert len(h.clusterer) == 1 and event.n_articles == 3 and event.n_independent == 3
    assert len(event.origins) == 3


def test_different_company_or_type_is_a_different_event():
    h = Harness()
    h.feed(make_item("a0"))
    other_company, _ = h.feed(
        make_item("b0", "TCS bags Rs 900 crore deal from a European bank", "Multi-year transformation deal.", primary="TCS",
                  seen=T0 + H)
    )
    assert other_company.action == "new"
    unrelated, _ = h.feed(
        make_item("c0", "Reliance Industries to raise Rs 3,000 crore via NCDs", "The board approved fundraising.",
                  event_type="fundraising", polarity=0.0, seen=T0 + H)
    )
    assert unrelated.action == "new"
    assert len(h.clusterer) == 3


def test_exact_url_match_attaches_an_updated_article():
    h = Harness()
    h.feed(make_item("a0", uhash="url1"))
    updated = make_item(
        "a1", "Reliance order win: what analysts say", "Brokerages raise targets after the order announcement.",
        seen=T0 + H, uhash="url1", primary=None, event_type="other", polarity=0.0,
    )
    decision, event = h.feed(updated)
    assert decision.action == "attach" and decision.match == "exact_url" and decision.event.key == "k1"
    assert event.n_articles == 2


def test_exact_content_hash_match_wins_before_similarity():
    h = Harness()
    h.feed(make_item("a0"))
    decision, _ = h.feed(make_item("a1", seen=T0 + 5 * H, owner="other"))
    assert decision.match == "exact_content" and decision.similarity == 1.0


def test_near_simhash_match_uses_hamming_distance_three():
    base = Signature(0b1011 << 20, {1: 1.0}, "chash-a")
    near = Signature((0b1011 << 20) ^ 0b111, {99: 1.0}, "chash-b")  # 3 bits away, no shared terms
    far = Signature((0b1011 << 20) ^ 0b1111, {98: 1.0}, "chash-c")  # 4 bits away
    c = Clusterer()
    event = ClusterEvent("k1", 1, "order_win", T0, T0, "RELIANCE", 0.5, origins=[Origin(base, "pti", T0, 0.6)])
    c.add_event(event)

    def probe(sig):
        return c.decide(Item("p", "article", T0 + H, sig, "x", 0.6, 0.7, "other"))

    assert probe(near).match == "near_simhash" and probe(near).event is event
    assert probe(far).action == "new"


def test_resurfacing_and_followups_depend_on_age_since_last_sighting():
    h = Harness()
    h.feed(make_item("a0"))
    within, _ = h.feed(make_item("a1", seen=T0 + 70 * H))
    assert within.action == "attach"
    followup, ev2 = h.feed(make_item("a2", seen=T0 + 70 * H + 4 * DAY))
    assert followup.action == "new" and followup.followup_of is not None and followup.novelty == dedup.NOVELTY_FOLLOWUP
    resurfaced, _ = h.feed(make_item("a3", seen=T0 + 70 * H + 4 * DAY + 9 * DAY))
    assert resurfaced.action == "new" and resurfaced.resurfaced_of is not None
    assert resurfaced.novelty == dedup.NOVELTY_RESURFACED
    assert ev2.key != "k1"


def test_rumours_are_penalised():
    assert base_novelty() == 1.0
    assert base_novelty(rumour=True) == pytest.approx(0.7)
    assert base_novelty(followup=True) == pytest.approx(0.35)
    assert base_novelty(resurfaced=True, rumour=True) == pytest.approx(0.07)
    h = Harness()
    decision, _ = h.feed(make_item("r0", "Reliance in talks to acquire stake, sources said", "No official comment.",
                                   rumour=True, event_type="merger_acquisition", polarity=0.4))
    assert decision.novelty == pytest.approx(0.7)


def test_novelty_decays_with_age_and_coverage():
    fresh = novelty_at(1.0, T0, T0, 1)
    assert fresh == 1.0
    assert novelty_at(1.0, T0, T0 + 48 * H, 1) == pytest.approx(0.5, abs=1e-3)
    assert novelty_at(1.0, T0, T0, 6) < novelty_at(1.0, T0, T0, 1)
    assert novelty_at(1.0, T0, T0 - 5 * H, 1) == 1.0  # as_of before first sighting never exceeds 1
    assert 0.0 <= novelty_at(0.7, T0, T0 + 10 * DAY, 50) <= 0.01


def test_event_confidence_rewards_corroboration_and_contradiction_lowers_it():
    base = event_confidence(0.7, 0.6, 1, 0)
    assert event_confidence(0.7, 0.6, 3, 0) > base
    assert event_confidence(0.7, 0.6, 3, 1) < event_confidence(0.7, 0.6, 3, 0)
    assert event_confidence(0.7, 0.6, 1, 9) == pytest.approx(base * 0.45, abs=2e-3)  # contradictions never zero it out
    assert event_confidence(0.7, 1.0, 1, 0) > event_confidence(0.7, 0.2, 1, 0)
    assert event_confidence(2.0, 1.0, 9, 0) <= 0.97
    assert event_confidence(0.0, 1.0, 1, 0) == 0.0


def test_opposite_polarity_from_a_different_origin_is_a_contradiction():
    h = Harness()
    h.feed(make_item("a0", "Reliance wins Rs 5,000 crore NTPC order", "Order to supply solar modules.", polarity=0.7))
    cancel = make_item(
        "a1", "Reliance loses Rs 5,000 crore NTPC order after dispute", "NTPC cancelled the order to supply solar modules.",
        seen=T0 + 2 * H, owner="mint", polarity=-0.7,
    )
    decision, event = h.feed(cancel)
    assert decision.action == "attach" and decision.match == "story_link"
    assert [c.kind for c in decision.own_contradictions] == ["polarity"]
    assert event.contradictions == 1


def test_cross_event_contradiction_records_both_sides():
    h = Harness()
    h.feed(make_item("a0", "TCS bags Rs 900 crore deal from European bank", "Multi-year deal.", primary="TCS", polarity=0.7,
                     event_type="order_win"))
    decision, other = h.feed(make_item(
        "a1", "TCS loses European bank contract to rival Accenture", "The mandate was awarded elsewhere.",
        primary="TCS", polarity=-0.6, event_type="order_win", seen=T0 + 3 * H, owner="mint",
    ))
    assert decision.action == "new"
    assert [c.kind for c in decision.own_contradictions] == ["polarity"]
    assert len(decision.other_contradictions) == 1 and decision.other_contradictions[0][0].key == "k1"
    assert h.clusterer.events["k1"].contradictions == 1 and other.contradictions == 1


def test_denials_of_rumours_are_recorded_and_same_polarity_is_not_a_contradiction():
    h = Harness()
    h.feed(make_item("r0", "Reliance may acquire Disney India stake, sources say", "Talks at early stage.", rumour=True,
                     event_type="merger_acquisition", polarity=0.3))
    denial = make_item(
        "d0", "Reliance denies report on Disney India stake acquisition", "Reliance said the report is speculative.",
        event_type="merger_acquisition", polarity=0.0, denial=True, seen=T0 + 4 * H, owner="reuters",
    )
    decision, _ = h.feed(denial)
    kinds = [c.kind for c in decision.own_contradictions]
    assert "denial" in kinds
    h2 = Harness()
    h2.feed(make_item("a0", polarity=0.7))
    agree, _ = h2.feed(make_item("a1", "NTPC awards Reliance Rs 5,000 crore solar module supply contract",
                                 "NTPC placed an order worth Rs 5,000 crore with Reliance for the supply of solar modules.",
                                 polarity=0.5, owner="mint", seen=T0 + H))
    assert agree.own_contradictions == []


def test_filing_key_groups_same_day_filings_and_links_news():
    h = Harness()
    d1, _ = h.feed(make_item("f1", "Reliance: Award of order", "Receipt of order from NTPC.", kind="filing", owner="nse",
                             quality=1.0, filing_key="filing:RELIANCE:order_win:2026-11-02"))
    assert d1.action == "new"
    d2, event = h.feed(make_item("f2", "Reliance: Award of order (addendum)", "Order value Rs 5,000 crore.", kind="filing",
                                 owner="nse", quality=1.0, seen=T0 + H, filing_key="filing:RELIANCE:order_win:2026-11-02"))
    assert d2.action == "attach" and d2.match == "filing_key" and event.n_independent == 1
    other_day, _ = h.feed(make_item("f3", "Reliance: Award of order", "Another order from a different customer entirely.",
                                    kind="filing", owner="nse", quality=1.0, seen=T0 + DAY,
                                    filing_key="filing:RELIANCE:order_win:2026-11-03"))
    assert other_day.action == "new"  # a filing on another exchange day is a distinct disclosure
    news = make_item("n1", "Reliance wins Rs 5,000 crore NTPC solar module order", "Reliance Industries order NTPC solar modules.",
                     seen=T0 + 2 * H, owner="mint")
    decision, event = h.feed(news)
    assert decision.action == "attach" and decision.event.key == "k1"
    assert event.filing_keys == {"filing:RELIANCE:order_win:2026-11-02"}
    again = make_item("f4", "Reliance: Award of order", "Receipt of order.", kind="filing", owner="nse", quality=1.0,
                      seen=T0 + 3 * H, filing_key="filing:RELIANCE:order_win:2026-11-02")
    assert h.clusterer.decide(again).match == "filing_key"


def test_story_link_requires_same_primary_type_and_window():
    h = Harness()
    h.feed(make_item("a0"))
    similar_text = ("NTPC awards Reliance Rs 5,000 crore solar module supply contract",
                    "NTPC placed an order worth Rs 5,000 crore with Reliance for supply of solar modules.")
    wrong_primary = h.clusterer.decide(make_item("b0", *similar_text, primary="ADANIGREEN", owner="mint", seen=T0 + H))
    assert wrong_primary.match != "story_link"
    wrong_type = h.clusterer.decide(make_item("b1", *similar_text, event_type="government_contract", owner="mint", seen=T0 + H))
    assert wrong_type.match != "story_link"
    late = h.clusterer.decide(make_item("b2", *similar_text, owner="mint", seen=T0 + 30 * H))
    assert late.match != "story_link"
    unlinked = h.clusterer.decide(make_item("b3", *similar_text, primary=None, owner="mint", seen=T0 + H))
    assert unlinked.match != "story_link"
    other_type = h.clusterer.decide(make_item("b4", *similar_text, event_type="other", owner="mint", seen=T0 + H))
    assert other_type.match != "story_link"


def test_decide_does_not_mutate_and_commit_updates_the_index():
    h = Harness()
    h.feed(make_item("a0"))
    probe = make_item("a1", seen=T0 + H, owner="x")
    before = h.clusterer.events["k1"].n_articles
    h.clusterer.decide(probe)
    assert h.clusterer.events["k1"].n_articles == before
    decision = h.clusterer.decide(probe)
    event = h.clusterer.commit(probe, decision, key="zzz", event_id=9)
    assert event.n_articles == before + 1 and event.last_seen == T0 + H
    bad = Decision("attach", "story_link", None)
    with pytest.raises(ValueError):
        h.clusterer.commit(probe, bad, key="k", event_id=None)


def test_origins_are_capped_and_late_arrivals_move_first_seen_back():
    h = Harness()
    title = "Reliance NTPC solar module order crore supply contract awarded"
    stem = "Reliance gets NTPC solar module order of Rs 5,000 crore for supply "
    h.feed(make_item("a0", title, stem + "alpha0 bravo0 charlie0 delta0 echo0", seen=T0 + 5 * H))
    c = h.clusterer
    event = c.events["k1"]
    for i in range(1, dedup.MAX_ORIGINS + 6):
        unique = " ".join(f"{word}{i}" for word in ("kilo", "lima", "mike", "oscar", "papa"))
        decision, _ = h.feed(make_item(f"o{i}", title, stem + unique, seen=T0 + 5 * H + i, owner=f"pub{i}"))
        assert decision.action == "attach" and decision.event is event and decision.match == "story_link"
    assert len(event.origins) == dedup.MAX_ORIGINS and event.n_articles == dedup.MAX_ORIGINS + 6
    assert event.n_independent == dedup.MAX_ORIGINS
    early = make_item("e0", title, stem + "alpha0 bravo0 charlie0 delta0 echo0", seen=T0, owner="early")
    decision, _ = h.feed(early)
    assert decision.action == "attach" and decision.match == "exact_content"
    assert event.first_seen == T0


def test_blob_round_trip_rebuilds_an_equivalent_index():
    h = Harness()
    h.feed(make_item("a0", uhash="u1", owner="pti", domain="ptinews.com"))
    h.feed(make_item("a1", seen=T0 + H, owner="x", domain="x.com"))
    original = h.clusterer.events["k1"]
    blob = original.to_blob()
    assert blob["p"] == "RELIANCE" and len(blob["o"]) == 1
    origins = [Origin.from_json(o) for o in blob["o"]]
    assert origins[0].n == 2 and origins[0].owner == "pti" and origins[0].domains == ["ptinews.com", "x.com"]
    assert origins[0].sig.simhash == original.origins[0].sig.simhash
    assert origins[0].sig.chash == original.origins[0].sig.chash[:20]
    rebuilt = ClusterEvent("k1", 1, "order_win", T0, T0 + H, blob["p"], blob["pol"], bool(blob["rum"]), bool(blob["den"]),
                           2, origins, set(blob["fk"]))
    fresh = Clusterer()
    fresh.add_event(rebuilt)
    decision = fresh.decide(make_item("a2", seen=T0 + 2 * H, owner="y", domain="y.com"))
    assert decision.action == "attach" and decision.event is rebuilt
    assert fresh.decide(make_item("a3", seen=T0 + 2 * H, uhash="u1", title="Other", body="Other", primary=None,
                                  event_type="other")).match == "exact_url"
    empty = Signature.from_json({})
    assert empty.simhash == 0 and empty.vec == {} and empty.chash == ""
    assert Origin.from_json({}).quality == 0.2


def test_signature_json_is_compact_and_ordered():
    sig = Signature.build(TITLE, BODY)
    data = sig.to_json()
    assert set(data) == {"h", "v", "c"} and len(data["v"]) <= dedup.VEC_TERMS and len(data["h"]) == 16
    back = Signature.from_json(data)
    assert back.simhash == sig.simhash and set(back.vec) == set(sig.vec)
    assert sig.terms() == frozenset(sig.vec)


def test_n_independent_is_at_least_one_for_empty_events():
    event = ClusterEvent("k", None, "other", T0, T0, None, 0.0)
    assert event.n_independent == 1
