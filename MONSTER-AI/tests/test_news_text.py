from __future__ import annotations

import time

import pytest

from news_intelligence import text as t

# ------------------------------------------------------------------------------------------ clean_text


def test_clean_text_strips_markup_and_entities():
    raw = "<p>Reliance &amp; <b>Jio</b> profit &#8377;5,000 cr&nbsp;up</p><br/>Next&lt;line&gt;"
    assert t.clean_text(raw) == "Reliance & Jio profit \u20b95,000 cr up Next"  # entity-encoded tags are markup too
    assert t.clean_text("P&amp;L &lt; 0 and 5 &gt; 3") == "P&L < 0 and 5 > 3"


def test_clean_text_removes_script_style_comments_and_entity_encoded_markup():
    raw = (
        "A<script>alert(1)</script>B<style>p{color:red}</style>C<!-- hidden -->D"
        "&lt;script&gt;steal()&lt;/script&gt;E<noscript>nojs</noscript>F"
    )
    assert t.clean_text(raw) == "A B C D E F"


def test_clean_text_unclosed_constructs_are_bounded():
    assert t.clean_text("start <script>never closed " + "x" * 5000) == "start"
    assert t.clean_text("a <!-- never closed") == "a"
    assert t.clean_text("1 < 2 and 3 > 2, I <3 India") == "1 < 2 and 3 > 2, I <3 India"
    assert t.clean_text("x <b" + "y" * 5000) .startswith("x <b")
    assert t.clean_text('<script src="a.js"/>keep <b>this</b>') == "keep this"


def test_clean_text_normalises_unicode_and_drops_invisible_characters():
    assert t.clean_text("\uff34\uff23\uff33 \ufb01nance\u200b\u2060 zero\u00adwidth \x00ctrl") == "TCS finance zerowidth ctrl"
    assert t.clean_text("  many\n\n  spaces\t here ") == "many spaces here"


def test_clean_text_truncates_and_accepts_odd_inputs():
    assert len(t.clean_text("word " * 5000, max_chars=100)) <= 100
    assert t.clean_text(None) == "" and t.clean_text("") == "" and t.clean_text("abc", max_chars=0) == ""
    assert t.clean_text(b"<b>bytes</b> \xff ok") == "bytes ok"
    assert t.clean_text(12345) == "12345"


@pytest.mark.parametrize(
    "payload",
    ["<a" * 60_000, "<script>" * 8_000, "&amp;" * 60_000, "<!--" * 30_000, "<" + "a" * 100_000, "<p>" * 40_000],
)
def test_clean_text_is_linear_on_adversarial_input(payload):
    started = time.perf_counter()
    out = t.clean_text(payload)
    assert time.perf_counter() - started < 9.0  # linear work: quadratic or worse would be far slower on 100k chars
    assert len(out) <= 6000


def test_lead_cuts_at_sentence_or_word():
    text = "First sentence is here. Second sentence follows and is much longer than the limit allows."
    assert t.lead(text, 40) == "First sentence is here."
    assert t.lead("short", 40) == "short"
    assert t.lead("word " * 30, 22).endswith("word")


# ------------------------------------------------------------------------------------------------ urls
def test_canonical_url_normalises_identity():
    a = t.canonical_url("HTTP://www.Example.com:80/Path/a/?utm_source=x&b=2&a=1&fbclid=zz&gclid=1&ref=tw#frag")
    assert a == "https://example.com/Path/a?a=1&b=2"
    assert t.canonical_url("https://user:pw@news.example.com:8443/x/") == "https://news.example.com:8443/x"
    assert t.canonical_url("news.example.com/x//y/") == "https://news.example.com/x/y"
    assert t.canonical_url("https://example.com") == "https://example.com/"
    assert t.canonical_url("") == "" and t.canonical_url(None) == ""
    assert t.canonical_url("https://[bad") == "https://[bad"  # unparseable input is returned lower-cased, never raises


def test_url_hash_is_stable_across_tracking_variants():
    base = t.url_hash("https://www.site.com/a?id=1")
    assert base == t.url_hash("http://site.com/a/?id=1&utm_campaign=z#top")
    assert base != t.url_hash("https://site.com/b?id=1")
    assert t.domain_of("https://m.Example.com/x") == "m.example.com" and t.domain_of("nonsense") == "nonsense"
    assert t.domain_of("") == ""


# ------------------------------------------------------------------------------------- tokens / hashes
def test_tokenize_and_content_hash():
    assert t.tokenize("Q2 profit, up 12%!  India's gain") == ["q2", "profit", "up", "12", "india", "s", "gain"]
    assert t.tokenize(None) == [] and t.tokenize("x " * 100, max_tokens=5) == ["x"] * 5
    assert t.content_tokens(["the", "profit", "of", "a", "12", "tcs"]) == ["profit", "12", "tcs"]
    one = t.content_hash("Hello, World!", "Same   body text.")
    assert one == t.content_hash("hello world", "same body TEXT")
    assert one != t.content_hash("hello world", "other body")


def test_shingles_and_jaccard():
    assert t.shingles(["a", "b", "c", "d"], 3) == {"a b c", "b c d"}
    assert t.shingles(["a", "b"], 3) == {"a b"} and t.shingles([], 3) == set()
    assert t.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5 and t.jaccard([], []) == 0.0
    assert t.overlap_coefficient({1, 2}, {1, 2, 3, 4}) == 1.0 and t.overlap_coefficient([], [1]) == 0.0


def test_simhash_distance_orders_similarity():
    base = t.tokenize("Reliance Industries Q2 net profit rises 12 percent to Rs 17,000 crore beating street estimates on retail and Jio gains")
    same = t.tokenize("Reliance Industries Q2 net profit rises 12 percent to Rs 17,000 crore beating street estimates on retail and Jio gains")
    other = t.tokenize("Monsoon rains lift tractor sales as rural demand improves across several states in the south and west of the country")
    assert t.hamming(t.text_simhash(base), t.text_simhash(same)) == 0
    assert t.hamming(t.text_simhash(base), t.text_simhash(other)) > 10
    assert t.simhash([]) == 0 and t.text_simhash(["only", "two"]) != 0
    assert t.hamming(0b1011, 0b0001) == 2


def test_tfidf_cosine_and_idf_weighting():
    a = t.tokenize("Reliance Industries Q2 profit rises 12 percent beating estimates")
    b = t.tokenize("Reliance Industries Q2 profit rises 12 percent, beats estimates")
    c = t.tokenize("Monsoon rains lift tractor sales in rural India")
    assert t.tfidf_cosine(a, b) > 0.85
    assert t.tfidf_cosine(a, c) < 0.1
    assert t.cosine({}, {1: 1.0}) == 0.0 and t.cosine({1: 1.0}, {2: 1.0}) == 0.0
    df = t.DocFreq()
    for doc in (a, b, c, c):
        df.add(t.term_vector(doc))
    va, vc = t.term_vector(a), t.term_vector(c)
    assert df.idf(999_999) > df.idf(next(iter(va)))  # unseen terms are rarer than common ones
    assert 0.0 <= t.cosine(va, va, df) <= 1.0 and abs(t.cosine(va, va, df) - 1.0) < 1e-9
    assert t.cosine(va, vc, df) < 0.1
    assert len(t.term_vector(["alpha%d" % i for i in range(200)], top_k=10)) == 10


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sensex, Nifty close higher as Infosys and TCS gain", True),
        ("Sensex Nifty close higher Infosys TCS gain", True),
        ("RBI keeps repo rate unchanged at 6.5 percent as inflation eases", True),
        ("El gobierno anunci\u00f3 que los precios del petr\u00f3leo suben por la guerra", False),
        ("Le gouvernement a annonc\u00e9 que les prix du p\u00e9trole sont en hausse pour les march\u00e9s", False),
        ("\u092d\u093e\u0930\u0924\u0940\u092f \u0930\u093f\u091c\u093c\u0930\u094d\u0935 \u092c\u0948\u0902\u0915 \u0928\u0947 \u092c\u094d\u092f\u093e\u091c \u0926\u0930\u0947\u0902 \u0938\u094d\u0925\u093f\u0930 \u0930\u0916\u0940\u0902", False),
        ("Der Markt ist nicht mit den neuen Zahlen von der Bank zufrieden und die Aktie f\u00e4llt", False),
        ("short", True),
        ("", True),
    ],
)
def test_is_probably_english(text, expected):
    assert t.is_probably_english(text) is expected
