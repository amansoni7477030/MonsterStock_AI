from __future__ import annotations

import json

import pytest

from news_intelligence import source_quality as sq


@pytest.mark.parametrize(
    ("source", "feed", "url", "category", "kind", "score"),
    [
        ("NSE", None, "https://nsearchives.nseindia.com/corporate/a.pdf", None, "exchange_filing", 1.0),
        ("NSE", "nse_announcements", None, "filing", "exchange_filing", 1.0),
        ("Reserve Bank of India", "rbi_press", "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx", None, "official", 0.95),
        (None, None, "https://pib.gov.in/PressReleasePage.aspx", None, "official", 0.95),
        (None, None, "https://data.mospi.gov.in/x", None, "official", 0.95),
        (None, None, "https://www.whitehouse.gov/briefing", None, "official", 0.95),
        ("Federal Reserve", "fed_press", "https://www.federalreserve.gov/newsevents/pressreleases/x.htm", None, "official", 0.95),
        ("Acme Ltd", None, "https://www.businesswire.com/news/home/1", None, "company", 0.85),
        ("Acme Ltd", None, None, "press_release", "company", 0.85),
        ("Reuters", "gdelt:oil", "https://www.reuters.com/markets/oil-jumps", None, "wire", 0.75),
        ("reuters.com", "gdelt:oil", "https://www.reuters.com/markets/oil-jumps", None, "wire", 0.75),
        ("BBC", "bbc_world", "https://www.bbc.co.uk/news/business-1", None, "wire", 0.75),
        ("CNBC", "cnbc_top", "https://www.cnbc.com/2026/x.html", None, "wire", 0.75),
        ("Economic Times", "et_markets", "https://economictimes.indiatimes.com/markets/stocks/news/x.cms", "markets", "business_media", 0.6),
        ("Moneycontrol", None, "https://www.moneycontrol.com/news/x", None, "business_media", 0.6),
        ("Mint", "mint_markets", "https://www.livemint.com/market/x.html", None, "business_media", 0.6),
        ("Business Standard", None, "https://www.business-standard.com/x", None, "business_media", 0.6),
        ("BusinessLine", None, "https://www.thehindubusinessline.com/x", None, "business_media", 0.6),
        ("Google News", "gn_oil", "https://news.google.com/rss/articles/abc", None, "aggregator", 0.4),
        ("someone", "reddit_stocks", "https://www.reddit.com/r/IndiaInvestments/x", None, "social", 0.2),
        ("someone", None, "https://x.com/user/status/1", None, "social", 0.2),
        ("someone", None, "https://twitter.com/user/status/1", None, "social", 0.2),
        ("Random Blog", "misc", "https://randomblog.example.xyz/post", None, "unknown", 0.2),
        (None, None, None, None, "unknown", 0.2),
    ],
)
def test_quality_mapping(source, feed, url, category, kind, score):
    assert sq.quality_for(source, feed, url, category) == (score, kind)


def test_names_and_feed_prefixes_apply_when_the_url_is_unknown():
    assert sq.quality_for("Reuters", None, None) == (0.75, "wire")
    assert sq.quality_for("PTI", None, None) == (0.75, "wire")
    assert sq.quality_for("Unknown Daily", "rbi_notif", None) == (0.95, "official")
    assert sq.quality_for("Unknown Daily", "gn_nifty", None) == (0.4, "aggregator")
    assert sq.quality_for("Unknown Daily", "gdelt:oil", None) == (0.4, "aggregator")


def test_aggregator_link_inherits_a_known_publisher_only_up_to_business_media():
    link = "https://news.google.com/rss/articles/abc"
    assert sq.quality_for("Reuters", "gn_oil", link) == (0.6, "business_media")  # capped, never above business media
    assert sq.quality_for("Mint", "gn_oil", link) == (0.6, "business_media")
    assert sq.quality_for("Reddit", "gn_oil", link) == (0.2, "social")
    assert sq.quality_for("Google News", "gn_oil", link) == (0.4, "aggregator")


def test_publisher_key_identifies_independent_sources():
    assert sq.publisher_key("Economic Times", "et", "https://markets.economictimes.indiatimes.com/a") == "economictimes.indiatimes.com"
    assert sq.publisher_key(None, None, "https://www.bbc.co.uk/news/1") == "bbc.co.uk"
    assert sq.publisher_key(None, None, "https://blog.foo.example.org/a") == "example.org"
    assert sq.publisher_key(None, None, "https://news.foo.example.co.uk/a") == "example.co.uk"
    assert sq.publisher_key("Google News", "gn", "https://news.google.com/x") == "google news"
    assert sq.publisher_key(None, "feed_x", None) == "feed_x"
    assert sq.publisher_key(None, None, None) == "unknown"


def test_split_publisher_suffix():
    assert sq.split_publisher_suffix("Brent jumps 4% as Hormuz tension rises - Reuters") == (
        "Brent jumps 4% as Hormuz tension rises", "Reuters")
    assert sq.split_publisher_suffix("Sensex ends flat | Moneycontrol") == ("Sensex ends flat", "Moneycontrol")
    assert sq.split_publisher_suffix("No suffix here at all") == ("No suffix here at all", None)
    assert sq.split_publisher_suffix("Headline words - one two three four five six seven")[1] is None


def test_registry_survives_missing_or_bad_files_and_reloads_on_change(tmp_path):
    missing = sq.SourceRegistry.load(tmp_path / "nope.json")
    assert missing.resolve("Reuters", None, "https://reuters.com/x") == (0.2, "unknown")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert sq.SourceRegistry.load(bad).resolve(None, None, None) == (0.2, "unknown")
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"tiers": {"wire": 0.9, "bogus": 5, "social": 7}, "domains": {"a.com": "wire", "b.com": "nonsense"}}), encoding="utf-8")
    first = sq.get_registry(path)
    assert first.resolve(None, None, "https://a.com/x") == (0.9, "wire")
    assert first.resolve(None, None, "https://b.com/x") == (0.2, "unknown")  # unknown tier names are ignored
    assert first.scores["social"] == 0.2  # out-of-range scores are ignored
    assert sq.get_registry(path) is first  # cached
    path.write_text(json.dumps({"domains": {"a.com": "social"}}), encoding="utf-8")
    import os

    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    assert sq.get_registry(path).resolve(None, None, "https://a.com/x") == (0.2, "social")


def test_source_quality_enum_and_defaults_match_the_spec():
    assert sq.SourceQuality.EXCHANGE_FILING.value == "exchange_filing"
    assert sq.DEFAULT_SCORES == {
        "exchange_filing": 1.0, "official": 0.95, "company": 0.85, "wire": 0.75, "business_media": 0.6,
        "aggregator": 0.4, "social": 0.2, "unknown": 0.2,
    }
