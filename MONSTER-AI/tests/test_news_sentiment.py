from __future__ import annotations

import pytest

from news_intelligence import sentiment as s


def test_empty_text_is_neutral():
    assert s.polarity("") == 0.0 and s.polarity(None) == 0.0
    assert s.score("").n_tokens == 0


def test_positive_and_negative_wording():
    assert s.polarity("Profit surges as sales jump and margins improve") > 0.3
    assert s.polarity("Shares plunge after fraud probe and weak earnings") < -0.3
    assert s.polarity("The board met on Tuesday") == 0.0


@pytest.mark.parametrize(
    ("text", "sign"),
    [
        ("Revenue did not rise", -1),
        ("Sales are not strong", -1),
        ("Company didn't fall as feared", 1),
        ("No losses reported this quarter", 1),
        ("Margins failed to improve", -1),
    ],
)
def test_negation_flips_the_nearby_word(text, sign):
    assert s.polarity(text) * sign > 0


def test_negation_does_not_cross_clause_boundaries():
    assert s.polarity("Not good, but profit rises sharply") > 0  # the comma ends the negation scope


def test_intensifiers_diminishers_and_uncertainty():
    plain, strong, weak = s.score("profit rises"), s.score("profit rises sharply"), s.score("profit rises slightly")
    assert strong.positive > plain.positive > weak.positive
    assert s.score("sharply higher profit").positive > s.score("higher profit").positive
    assert s.score("shares fell slightly").negative < s.score("shares fell").negative
    assert s.score("may possibly face risks").uncertainty == 3.0


def test_polarity_stays_inside_the_unit_interval():
    text = " ".join(["surge rally jump gain profit"] * 200)
    assert -1.0 < s.polarity(text) < 1.0
    assert -1.0 < s.polarity(" ".join(["fraud default crash"] * 200)) < 0
