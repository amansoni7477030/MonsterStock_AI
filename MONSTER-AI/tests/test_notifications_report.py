"""Renderer tests: golden lines on obviously fake data, escaping, NaN handling, caps and message limits."""

from __future__ import annotations

import dataclasses
import html
import math
import re
from types import SimpleNamespace

import pytest

from notifications import style
from notifications.report import (
    DISCLAIMER,
    EVENING_NOTE,
    SHADOW_BANNER,
    render_alert,
    render_alert_digest,
    render_evening_report,
    render_failure_notice,
    render_final_report,
    render_prelim_report,
    render_test_message,
    render_unit_failure,
)
from ranking.types import Candidate, Catalyst, EveningRow, EveningSummary, ReportContext

TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"
ALLOWED_TAGS = {"b", "i", "code", "pre", "a"}
_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)([^<>]*)>")
_RAW_AMP = re.compile(r"&(?!(?:amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);)")
_BAD_NUMBER = re.compile(r"\b(?:nan|NaN|inf|None)\b")


def assert_telegram_safe(messages):
    """Every message within limits, balanced, only whitelisted tags, and nothing unescaped in the text."""
    assert isinstance(messages, list) and messages and all(isinstance(m, str) for m in messages)
    for message in messages:
        assert 0 < len(message) <= 3900
        message.encode("utf-8")  # raises on a lone surrogate
        stack = []
        for closing, name, attrs in _TAG.findall(message):
            assert name in ALLOWED_TAGS, name
            if name == "a" and not closing:
                assert re.fullmatch(r' href="https?://[^"<>\s\']+"', attrs, re.IGNORECASE), attrs
            elif not attrs:
                pass
            else:
                raise AssertionError(f"unexpected attributes on <{name}>: {attrs}")
            if closing:
                assert stack and stack.pop() == name
            else:
                stack.append(name)
        assert not stack
        text = _TAG.sub("", message)
        assert "<" not in text and ">" not in text and not _RAW_AMP.search(text)


def lines(messages):
    return "\n".join(messages).split("\n")


def make_candidate(rank=1, symbol="FAKECO", **over):
    base = dict(
        symbol=symbol,
        rank=rank,
        tier="STRONG",
        p={"hit_1": 0.91, "hit_2": 0.78, "hit_3": 0.624, "hit_5": 0.31, "unusual": 0.4, "up": 0.52, "down": 0.48},
        primary_threshold=3.0,
        range={"q10": 2.1, "q50": 3.44, "q90": 5.6, "exc_q50": 2.2, "exc_q90": 3.8},
        name="Fake Company Ltd",
        sector="Fake Sector",
        catalysts=[Catalyst(headline="Fake plant outage lifts prices", source="Fake Wire", url="https://example.test/news/1")],
        chain=["Fake crude spike -> fake refiners -> FAKECO", "second chain"],
        confirmations=["Pre-open volume 3x", "Futures OI up"],
        contradictions=[],
        priced_in="Partly priced in (gap +1.2%)",
        liquidity_ok=True,
        liquidity_note="liquid",
        turnover_cr=412.5,
        est_cost_pct=0.12,
        data_quality="ok",
        uncertainty=0.42,
        drivers=["gap_pct", "atr_ratio", "news_novelty", "fourth_driver"],
        is_fno=True,
    )
    base.update(over)
    return Candidate(**base)


def make_ctx(candidates=None, **over):
    base = dict(
        trade_date="2026-11-02",
        stage="final",
        snapshot_ist="2026-11-02 09:12:00 IST",
        model_version="fake-model-0.1",
        model_status="validated",
        candidates=[make_candidate()] if candidates is None else candidates,
        regime={"vix": "calm", "breadth": "mixed"},
        freshness={"yahoo": "ok (3 min old)", "nse_preopen": "ok (1 min old)"},
        missing_feeds=[],
        uncertainty_note="Intervals are wide for 2 names",
        universe_size=212,
        n_eligible=187,
    )
    base.update(over)
    return ReportContext(**base)


# --------------------------------------------------------------------------------------- final report
def test_final_report_layout_golden_lines():
    messages = render_final_report(make_ctx())
    assert_telegram_safe(messages)
    assert len(messages) == 1
    got = lines(messages)
    expected = [
        "🚀 <b>MONSTER AI</b> · NSE large-move scan",
        "🎯 <b>FINAL</b> · 📅 Mon 02 Nov 2026 · 🕘 09:12 IST",
        "🧠 Model <code>fake-model-0.1</code> · ✅ <b>VALIDATED</b>",
        "🎯 <b>Top 1</b> · chance that the day's range (high - low) exceeds 3% of the open",
        "🥇 <b>#1 FAKECO</b> · 🔥 <b>STRONG</b>",
        "<i>Fake Company Ltd · Fake Sector · F&amp;O</i>",
        "📊 P(range &gt; 3%): <b>62%</b> (calibrated)",
        "<code>██████░░░░</code>",
        "🔢 1%: 91% · 2%: 78% · 3%: 62% · 5%: 31%",
        "📏 Expected intraday range: <b>3.4%</b> · 80% interval 2.1-5.6%",
        "📐 Max excursion from open (q50 / q90): 2.2% / 3.8%",
        "✨ P(unusual for this stock): 40%",
        "🧭 Direction (secondary, not a trade signal): P(up close) 52% · P(down close) 48%",
        '📰 <b>Main catalyst:</b> Fake plant outage lifts prices - Fake Wire (<a href="https://example.test/news/1">link</a>)',
        "🔗 <b>Global-to-India chain:</b> Fake crude spike → fake refiners → FAKECO",
        "✅ <b>Confirmations:</b> Pre-open volume 3x; Futures OI up",
        f"{style.WARN} <b>Contradictions:</b> none",
        "💸 <b>Priced-in assessment:</b> Partly priced in (gap +1.2%)",
        "💧 <b>Liquidity and data quality:</b> ✅ pass - liquid · turnover ₹412.5 cr · est. round-trip cost 0.12%",
        "🧩 <b>Model drivers</b> (diagnostic, not causal): gap_pct; atr_ratio; news_novelty",
        "🌍 <b>Market regime:</b> 🔹 vix: calm · 🔹 breadth: mixed",
        "📡 <b>Data freshness</b>",
        "🟢 yahoo: ok (3 min old)",
        "🟢 nse_preopen: ok (1 min old)",
        "✅ <b>Missing feeds:</b> none",
        "❓ <b>Model uncertainty:</b> Intervals are wide for 2 names · max candidate uncertainty 0.42 (0-1, higher = less reliable)",
        "🔬 <b>Universe:</b> 187 of 212 passed liquidity/data gates (88%)",
        f"{style.WARN} <i>{DISCLAIMER}</i>",
    ]
    for line in expected:
        assert line in got, line
    assert [got.index(line) for line in expected] == sorted(got.index(line) for line in expected)
    assert got.count(style.RULE) == 2  # one under the head, one above the footer
    assert "SHADOW MODE" not in messages[0]
    assert DISCLAIMER.startswith("Probabilistic research output; not investment advice and not a guarantee.")


@pytest.mark.parametrize(
    "status, badge, label",
    [
        ("validated", "✅", "VALIDATED"),
        ("shadow", "🧪", "SHADOW - unvalidated"),
        ("baseline", "📏", "BASELINE (no trained model)"),
        ("something-else", "🧪", "SHADOW - unvalidated"),
        (None, "🧪", "SHADOW - unvalidated"),
    ],
)
def test_model_status_labels_and_shadow_banner(status, badge, label):
    text = "\n".join(render_final_report(make_ctx(model_status=status)))
    assert f"🧠 Model <code>fake-model-0.1</code> · {badge} <b>{label}</b>" in text
    assert (f"🧪 <b>{SHADOW_BANNER}</b>" in text) == (status != "validated")
    if status != "validated":
        assert SHADOW_BANNER == "SHADOW MODE: no demonstrated edge over simple baselines yet - tiers are capped at WATCH."


def test_unvalidated_models_never_show_tiers_above_watch():
    ctx = make_ctx(
        [make_candidate(1, "AAA", tier="STRONG"), make_candidate(2, "BBB", tier="MODERATE"), make_candidate(3, "CCC", tier="WATCH")],
        model_status="shadow",
    )
    text = "\n".join(render_final_report(ctx))
    assert "STRONG" not in text and "MODERATE" not in text and text.count("👀 <b>WATCH</b>") == 3
    validated = "\n".join(render_final_report(dataclasses.replace(ctx, model_status="validated")))
    assert "<b>#1 AAA</b> · 🔥 <b>STRONG</b>" in validated and "<b>#2 BBB</b> · ⚡ <b>MODERATE</b>" in validated
    assert "<b>#3 CCC</b> · 👀 <b>WATCH</b>" in validated


@pytest.mark.parametrize("version", ["", "   ", None])
def test_empty_model_version_is_refused_everywhere(version):
    with pytest.raises(ValueError, match="model_version"):
        render_final_report(make_ctx(model_version=version))
    with pytest.raises(ValueError, match="model_version"):
        render_prelim_report(make_ctx(model_version=version, stage="pre"))
    with pytest.raises(ValueError, match="model_version"):
        render_evening_report(EveningSummary("2026-11-02", "final", 3.0, version))


def test_prelim_report_is_titled_preliminary_and_never_claims_finality():
    messages = render_prelim_report(make_ctx(stage="pre", snapshot_ist="2026-11-02 08:45:00 IST"))
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert "🚀 <b>MONSTER AI</b> · NSE large-move scan" in text
    assert "⏳ <b>PRELIMINARY</b> (08:45 scan) · 📅 Mon 02 Nov 2026 · 🕗 08:45 IST" in text
    assert "🎯 <b>FINAL</b>" not in text and "final" not in text.lower()
    assert "must not be treated as confirmed" in text
    assert "<b>#1 FAKECO</b> · 🔥 <b>STRONG</b>" in text and "<b>Universe:</b> 187 of 212" in text


def test_ranking_order_and_rank_fallbacks():
    cands = [make_candidate(3, "CCC"), make_candidate(1, "AAA"), make_candidate(2, "BBB"), make_candidate(None, "ZZZ")]
    heads = [ln for ln in lines(render_final_report(make_ctx(cands))) if "<b>#" in ln]
    assert [re.search(r"<b>#(\d+) (\w+)</b>", h).groups() for h in heads] == [
        ("1", "AAA"), ("2", "BBB"), ("3", "CCC"), ("4", "ZZZ"),
    ]
    odd = make_candidate(1, "ODD", tier="spicy<tier>")
    assert "<b>#1 ODD</b> · 🔹 <b>SPICY&lt;TIER&gt;</b>" in "\n".join(render_final_report(make_ctx([odd])))


def test_medals_go_to_the_first_three_places_only():
    cands = [make_candidate(i, f"S{i}") for i in range(1, 6)]
    heads = [ln for ln in lines(render_final_report(make_ctx(cands))) if "<b>#" in ln]
    assert [h.split(" ", 1)[0] for h in heads] == ["🥇", "🥈", "🥉", "🏅", "🏅"]


def test_plain_dict_candidates_render_like_objects():
    candidate = make_candidate()
    as_dict = dataclasses.asdict(candidate)
    assert render_final_report(make_ctx([as_dict])) == render_final_report(make_ctx([candidate]))


# --------------------------------------------------------------------------------------- no setup
def test_no_setup_block_labels_remaining_names_as_weak_watchlist():
    ctx = make_ctx([make_candidate(1, "AAA", tier="STRONG")], no_setup=True, no_setup_reason="Top probability only 33% <low>")
    messages = render_final_report(ctx)
    assert_telegram_safe(messages)
    got = lines(messages)
    assert "🛑 <b>NO HIGH-CONVICTION SETUP today</b>" in got
    assert "📝 <b>Reason:</b> Top probability only 33% &lt;low&gt;" in got
    assert "🔸 <b>WEAK / WATCHLIST ONLY</b> - shown for transparency only; these are not signals." in got
    assert "🔹 <b>#1 AAA</b> · 🔸 <b>WEAK / WATCHLIST ONLY</b>" in got  # a weak list never gets medals
    assert "STRONG" not in "\n".join(messages) and "🥇" not in "\n".join(messages)
    assert not any("Top 1" in ln for ln in got)  # no "top picks" headline over a list that is flagged weak


def test_no_candidates_means_no_setup_even_without_the_flag():
    text = "\n".join(render_final_report(make_ctx([], no_setup=False)))
    assert "🛑 <b>NO HIGH-CONVICTION SETUP today</b>" in text
    assert "📝 <b>Reason:</b> no candidate cleared the ranking gates" in text
    assert "WEAK / WATCHLIST ONLY" not in text and "<b>#1" not in text
    flagged = "\n".join(render_final_report(make_ctx([], no_setup=True)))
    assert "📝 <b>Reason:</b> not stated by the ranking stage" in flagged
    assert "🔬 <b>Universe:</b> 187 of 212 passed liquidity/data gates (88%)" in flagged  # the footer is still sent


# ----------------------------------------------------------------------------------- escaping / XSS
XSS_PAYLOADS = [
    '<script>alert("x")</script>',
    '"><img src=x onerror=alert(1)>',
    "<b>bold</b> & <i>it</i>",
    "a & b < c > d",
    "&lt;already&gt;",
    "'; DROP TABLE x; --",
    "</b></pre></a><a href=\"https://evil.test\">x</a>",
]


@pytest.mark.parametrize("payload", XSS_PAYLOADS)
def test_every_dynamic_value_is_escaped(payload):
    escaped = html.escape(payload, quote=False)
    cand = make_candidate(
        symbol="FAKE" + payload[:6],
        name=payload,
        sector=payload,
        catalysts=[Catalyst(headline=payload, source=payload, url="https://example.test/ok")],
        chain=[payload],
        confirmations=[payload],
        contradictions=[payload],
        priced_in=payload,
        liquidity_note=payload,
        drivers=[payload],
    )
    ctx = make_ctx(
        [cand],
        snapshot_ist=payload,
        model_version="v-" + payload,
        regime={payload: payload},
        freshness={payload: payload},
        missing_feeds=[payload],
        uncertainty_note=payload,
        notes=[payload],
    )
    messages = render_final_report(ctx)
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert escaped in text
    assert "<script" not in text and "<img" not in text
    assert all(href.startswith("https://example.test/") for href in re.findall(r'<a href="([^"]*)"', text))
    for message in render_prelim_report(ctx) + render_failure_notice(payload, payload, payload, payload):
        assert_telegram_safe([message])
    assert_telegram_safe([render_alert(payload, payload, payload, created_at=payload)])
    assert_telegram_safe(render_alert_digest([{"level": payload, "code": payload, "message": payload}]))
    assert_telegram_safe([render_unit_failure(payload, payload, payload)])


BAD_URLS = [
    "javascript:alert(1)",
    "JaVaScRiPt:alert(1)",
    "  javascript:alert(1)",
    "java\tscript:alert(1)",
    "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
    "vbscript:msgbox(1)",
    "ftp://example.test/file",
    "//example.test/protocol-relative",
    "/relative/path",
    "example.test/no-scheme",
    "http://",
    "https:///nohost",
    "http://[::1/unbalanced-ipv6",
    'https://example.test/"onmouseover="alert(1)',
    "https://example.test/a b",
    "https://example.test/<x>",
    "https://example.test/it's",
    "https://example.test/`tick`",
    "https://example.test/back\\slash",
    "https://example.test/" + "a" * 400,
    "",
    None,
    123,
    ["https://example.test/list"],
]


@pytest.mark.parametrize("url", BAD_URLS, ids=[repr(u)[:30] for u in BAD_URLS])
def test_unsafe_urls_are_dropped_and_never_printed(url):
    cand = make_candidate(catalysts=[Catalyst(headline="Fake headline", source="Fake Wire", url=url)])
    messages = render_final_report(make_ctx([cand]))
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert "📰 <b>Main catalyst:</b> Fake headline - Fake Wire" in lines(messages)
    assert "<a " not in text and "(link)" not in text and "javascript" not in text.lower()
    assert "example.test" not in text and "alert(1)" not in text


def test_http_and_https_links_are_kept_and_escaped_inside_href():
    for url, shown in (
        ("https://example.test/a?x=1&y=2", "https://example.test/a?x=1&amp;y=2"),
        ("http://example.test/plain", "http://example.test/plain"),
        ("HTTPS://EXAMPLE.TEST/Up", "HTTPS://EXAMPLE.TEST/Up"),
    ):
        cand = make_candidate(catalysts=[Catalyst(headline="Fake", url=url)])
        messages = render_final_report(make_ctx([cand]))
        assert_telegram_safe(messages)
        assert f'📰 <b>Main catalyst:</b> Fake (<a href="{shown}">link</a>)' in lines(messages)


# ---------------------------------------------------------------------------------- catalyst provenance
@pytest.mark.parametrize(("fields", "trust"), [
    (dict(rumour=True, source_class="social"), "🚩 <b>RUMOUR - unconfirmed</b> · social media"),
    (dict(rumour=False, source_class="exchange_filing", n_independent=3), "🔎 exchange filing · 3 independent sources"),
    (dict(source_class="media", n_independent=2, n_conflicts=1), "🔎 media report · 2 independent sources · 1 conflicting"),
    (dict(source_class="unknown"), "🔎 unverified source"),
    (dict(n_independent=1, n_conflicts=0), None),
    (dict(), None),
], ids=["rumour", "filing", "conflict", "unknown", "single-source", "no-provenance"])
def test_a_catalyst_shows_how_far_it_can_be_trusted(fields, trust):
    cand = make_candidate(catalysts=[Catalyst(headline="Fake headline", source="Fake Wire", **fields)])
    messages = render_final_report(make_ctx([cand]))
    assert_telegram_safe(messages)
    got = lines(messages)
    at = got.index("📰 <b>Main catalyst:</b> Fake headline - Fake Wire")
    if trust:
        assert got[at + 1] == trust  # the trust line sits directly under the catalyst
    else:
        assert not any(ln.startswith(("🚩", "🔎")) for ln in got)  # older snapshots do not grow a provenance line


def test_hostile_provenance_values_never_reach_the_message():
    cand = make_candidate(catalysts=[Catalyst(headline="Fake", source="Wire", rumour="yes", source_class="<b>x</b>",
                                              n_independent="many", n_conflicts=float("nan"))])
    messages = render_final_report(make_ctx([cand]))
    assert_telegram_safe(messages)
    assert "📰 <b>Main catalyst:</b> Fake - Wire" in lines(messages)
    assert "RUMOUR" not in "\n".join(messages)


def test_the_provenance_survives_the_snapshot_round_trip():
    original = Catalyst(headline="Fake", rumour=True, source_class="social", n_independent=2, n_conflicts=0)
    assert Catalyst.from_dict(original.to_dict()) == original
    assert Catalyst.from_dict({"headline": "Old snapshot", "unknown_future_key": 1}).rumour is None


# ------------------------------------------------------------------------------------ missing data
def test_missing_and_non_finite_values_render_as_n_a_never_as_numbers():
    cand = make_candidate(
        p={"hit_3": float("nan"), "hit_1": None, "up": "abc", "down": 1.5, "unusual": -0.2, "score": 0.1234},
        range={"q10": None, "q50": float("nan"), "q90": float("inf")},
        turnover_cr=None,
        est_cost_pct=float("nan"),
        uncertainty=float("nan"),
        priced_in=None,
        name=None,
        sector=None,
        is_fno=False,
        catalysts=[],
        chain=[],
        confirmations=None,
        contradictions=[None, ""],
        drivers=[],
        liquidity_note=None,
    )
    ctx = make_ctx([cand], regime={}, freshness=None, missing_feeds=None, uncertainty_note="", universe_size=0, n_eligible=0)
    messages = render_final_report(ctx)
    assert_telegram_safe(messages)
    got = lines(messages)
    for line in (
        "🥇 <b>#1 FAKECO</b> · 🔥 <b>STRONG</b>",
        "📊 P(range &gt; 3%): <b>n/a</b> (calibrated)",
        "🔢 1%: n/a · 3%: n/a",
        "✨ P(unusual for this stock): n/a",
        "📏 Expected intraday range: <b>n/a</b> · 80% interval n/a",
        "📐 Max excursion from open (q50 / q90): n/a / n/a",
        "🧭 Direction (secondary, not a trade signal): P(up close) n/a · P(down close) n/a",
        "📰 <b>Main catalyst:</b> none identified",
        "🔗 <b>Global-to-India chain:</b> n/a",
        "✅ <b>Confirmations:</b> none",
        f"{style.WARN} <b>Contradictions:</b> none",
        "💸 <b>Priced-in assessment:</b> not assessed",
        "💧 <b>Liquidity and data quality:</b> ✅ pass · turnover n/a · est. round-trip cost n/a",
        "🧩 <b>Model drivers</b> (diagnostic, not causal): n/a",
        "🌍 <b>Market regime:</b> n/a",
        "📡 <b>Data freshness:</b> n/a",
        "✅ <b>Missing feeds:</b> none",
        "❓ <b>Model uncertainty:</b> n/a",
        "🔬 <b>Universe:</b> n/a of n/a passed liquidity/data gates",
    ):
        assert line in got, line
    text = "\n".join(messages)
    assert not _BAD_NUMBER.search(text) and "0.1234" not in text
    assert "<code>█" not in text and "<code>░" not in text  # no probability, so no bar


def test_completely_empty_candidate_and_context_still_render():
    bare = Candidate(symbol="", rank=0, tier="", p={}, primary_threshold=float("nan"), range={})
    messages = render_final_report(ReportContext("", "final", "", "fake-model", "validated", candidates=[bare]))
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert "<b>#1 UNKNOWN</b> · 🔹 <b>n/a</b>" in text and "· 🕒 n/a" in text
    assert "📊 P(range &gt; n/a): <b>n/a</b> (calibrated)" in text and not _BAD_NUMBER.search(text)
    odd = "\n".join(render_final_report(make_ctx(trade_date="<b>bad</b>-and-long")))
    assert "📅 &lt;b&gt;bad&lt;/b&gt; ·" in odd and "-and-long" not in odd
    assert "📅 Mon 02 Nov 2026 ·" in "\n".join(render_final_report(make_ctx(trade_date="2026-11-02T00:00:00")))


def test_wrongly_typed_collections_degrade_gracefully_instead_of_crashing():
    ctx = make_ctx(candidates=3.14, missing_feeds=7, notes=None, regime=["x"])
    text = "\n".join(render_final_report(ctx))
    assert "🛑 <b>NO HIGH-CONVICTION SETUP today</b>" in text and "<b>Missing feeds:</b> 7" in text
    assert "<b>Market regime:</b> n/a" in text
    odd_candidate = make_candidate(catalysts=5, drivers=2.5, confirmations={"x": 1})
    assert "📰 <b>Main catalyst:</b> none identified" in "\n".join(render_final_report(make_ctx([odd_candidate])))
    summary = "\n".join(render_evening_report(make_summary(rows=5.0, missing_symbols=3.5, rolling=[1], baseline_hit_rates=None)))
    assert "📋 No evaluated picks for this date." in summary and "<b>Missing symbols:</b> 3.5" in summary
    assert "<b>Rolling record (out-of-sample):</b> n/a" in summary


def test_liquidity_warning_when_liquidity_or_data_quality_fails():
    for override in ({"liquidity_ok": False}, {"data_quality": "warning"}, {"data_quality": "stale"}):
        text = "\n".join(render_final_report(make_ctx([make_candidate(**override)])))
        assert f"💧 <b>Liquidity and data quality:</b> {style.WARN} warning - liquid · turnover ₹412.5 cr" in text


# ----------------------------------------------------------------------------- numbers and probabilities
def test_probabilities_are_rounded_percentages_never_raw_scores():
    p = {"hit_1": 0.625, "hit_2": 0.0, "hit_3": 1.0, "hit_5": 0.995, "up": 0.004, "down": 0.6234, "raw_score": 0.8765}
    text = "\n".join(render_final_report(make_ctx([make_candidate(p=p)])))
    assert "🔢 1%: 63% · 2%: 0% · 3%: 100% · 5%: 100%" in text
    assert "P(up close) 0% · P(down close) 62%" in text
    for raw in ("0.625", "0.6234", "62.3", "0.8765", "0.995"):
        assert raw not in text


def test_non_integer_thresholds_sort_numerically_and_pick_the_primary():
    p = {"hit_10": 0.05, "hit_2.5": 0.55, "hit_1": 0.9, "hit_2_5": 0.5, "unusual": 0.3, "up": 0.5, "down": 0.5}
    cand = make_candidate(p=p, primary_threshold=2.5)
    text = "\n".join(render_final_report(make_ctx([cand])))
    assert "P(range &gt; 2.5%):" in text and "🔢 1%: 90% · 2.5%: 5" in text and "· 10%: 5%" in text


def test_range_and_cost_formatting_edge_cases():
    cand = make_candidate(range={"q10": -0.04, "q50": 0.04, "q90": 12.345, "exc_q50": -0.0, "exc_q90": 7}, est_cost_pct=0.0, turnover_cr=1234567.89)
    text = "\n".join(render_final_report(make_ctx([cand])))
    assert "📏 Expected intraday range: <b>0.0%</b> · 80% interval 0.0-12.3%" in text
    assert "📐 Max excursion from open (q50 / q90): 0.0% / 7.0%" in text
    assert "turnover ₹1,234,567.9 cr · est. round-trip cost 0.00%" in text


# ------------------------------------------------------------------------------------ probability bar
@pytest.mark.parametrize(("p", "bar"), [(0.624, "██████░░░░"), (0.05, "█░░░░░░░░░"), (0.0, "░░░░░░░░░░"), (1.0, "██████████")])
def test_a_calibrated_probability_gets_a_bar_that_matches_the_percentage(p, bar):
    cand = make_candidate(p={"hit_3": p, "unusual": 0.4, "up": 0.5, "down": 0.5})
    assert f"<code>{bar}</code>" in lines(render_final_report(make_ctx([cand])))


def test_a_heuristic_score_never_gets_a_bar_and_says_it_is_not_calibrated():
    text = "\n".join(render_final_report(make_ctx([make_candidate()], model_status="baseline")))
    assert "(heuristic, NOT calibrated)" in text and "<code>" not in text.replace("<code>fake-model-0.1</code>", "")


def test_an_impossible_probability_draws_no_bar():
    cand = make_candidate(p={"hit_3": 1.7, "unusual": 0.4, "up": 0.5, "down": 0.5})
    text = "\n".join(render_final_report(make_ctx([cand])))
    assert "P(range &gt; 3%): <b>n/a</b>" in text and "█" not in text and "░" not in text


# ------------------------------------------------------------------------------- freshness and regime
def test_freshness_rows_carry_a_traffic_light_that_follows_the_status():
    fresh = {"A": "ok (3 min old)", "B": "WARN - one late", "C": "STALE - quote is 600 min old", "D": "MISSING - none", "E": "???"}
    got = lines(render_final_report(make_ctx(freshness=fresh)))
    for line in ("🟢 A: ok (3 min old)", "🟡 B: WARN - one late", "🟠 C: STALE - quote is 600 min old", "🔴 D: MISSING - none", "⚪ E: ???"):
        assert line in got, line


def test_missing_feeds_are_flagged_with_a_roadwork_sign_instead_of_a_tick():
    text = "\n".join(render_final_report(make_ctx(missing_feeds=["GIFT Nifty: quote is 600 min old"])))
    assert "🚧 <b>Missing feeds:</b> GIFT Nifty: quote is 600 min old" in text


def test_the_regime_shows_gift_nifty_and_the_known_labels_with_icons():
    regime = {"trend": "bull", "volatility": "normal-vol", "risk": "risk-on", "event": "F&O expiry day", "gift_nifty": "+0.40%"}
    text = "\n".join(render_final_report(make_ctx(regime=regime)))
    assert ("🌍 <b>Market regime:</b> 📈 Trend: bull · 🌊 Volatility: normal-vol · 🚦 Risk: risk-on · "
            "📌 Event: F&amp;O expiry day · 🌅 GIFT Nifty: +0.40%") in text


@pytest.mark.parametrize(("snapshot", "label"), [
    ("2026-11-02 09:12:00 IST", "🕘 09:12 IST"), ("2026-11-02 08:45:00 IST", "🕗 08:45 IST"),
    ("2026-11-02 00:05:00 IST", "🕛 00:05 IST"), ("2026-11-02 13:30:00 IST", "🕐 13:30 IST"),
    ("sometime", "🕒 sometime"),
])
def test_the_clock_face_follows_the_snapshot_hour(snapshot, label):
    assert f" · {label}" in "\n".join(render_final_report(make_ctx(snapshot_ist=snapshot)))


# --------------------------------------------------------------------------------------------- caps
def test_lengths_and_item_counts_are_capped():
    cats = [
        Catalyst(headline="H" * 500, source="S" * 100, url="https://example.test/1"),
        Catalyst(headline="Second headline"),
        Catalyst(headline="Third must not appear"),
        Catalyst(headline="   "),
    ]
    cand = make_candidate(
        catalysts=cats,
        confirmations=[f"conf {i}" for i in range(6)],
        contradictions=[f"contra {i}" for i in range(6)],
        drivers=[f"drv {i}" for i in range(6)],
        chain=[f"chain {i}" for i in range(4)],
        priced_in="P" * 900,
    )
    messages = render_final_report(make_ctx([cand]))
    assert_telegram_safe(messages)
    got = lines(messages)
    assert f"📰 <b>Main catalyst:</b> {'H' * 137}... - {'S' * 37}... (<a" in "\n".join(got)
    assert "H" * 141 not in "\n".join(got) and "Third must not appear" not in "\n".join(got)
    assert "📰 <b>Also:</b> Second headline" in got
    assert "✅ <b>Confirmations:</b> conf 0; conf 1; conf 2" in got
    assert f"{style.WARN} <b>Contradictions:</b> contra 0; contra 1; contra 2" in got
    assert "🧩 <b>Model drivers</b> (diagnostic, not causal): drv 0; drv 1; drv 2" in got
    assert "🔗 <b>Global-to-India chain:</b> chain 0" in got and "chain 1" not in "\n".join(got)
    assert f"💸 <b>Priced-in assessment:</b> {'P' * 197}..." in got


def test_control_characters_and_newlines_are_flattened():
    cand = make_candidate(catalysts=[Catalyst(headline="line one\nline\ttwo\x00\x1b[31m \u202eevil\u202c", source="Wire")])
    text = "\n".join(render_final_report(make_ctx([cand])))
    assert "📰 <b>Main catalyst:</b> line one line two[31m evil - Wire" in text
    assert "\x00" not in text and "\x1b" not in text and "\u202e" not in text and "\u202c" not in text


def test_lone_surrogates_are_stripped_so_every_message_is_valid_utf8():
    cand = make_candidate(catalysts=[Catalyst(headline="broken \ud83d feed \udc00 text", source="Wire")], name="Fa\ud800ke")
    messages = render_final_report(make_ctx([cand]))
    assert "📰 <b>Main catalyst:</b> broken feed text - Wire" in "\n".join(messages)
    for message in messages:
        message.encode("utf-8")  # raises on any lone surrogate


def heavy_candidate(rank):
    return make_candidate(
        rank,
        f"SYM{rank:02d}",
        catalysts=[
            Catalyst(headline="h" * 140, source="s" * 40, url="https://example.test/" + "u" * 250),
            Catalyst(headline="k" * 140, source="t" * 40, url="https://example.test/" + "v" * 250),
        ],
        chain=["c" * 200],
        confirmations=["x" * 120] * 3,
        contradictions=["y" * 120] * 3,
        priced_in="z" * 200,
        liquidity_note="l" * 100,
        drivers=["d" * 120] * 3,
    )


def test_long_reports_split_into_numbered_balanced_messages():
    cands = [heavy_candidate(i) for i in range(1, 13)]
    messages = render_final_report(make_ctx(cands))
    assert_telegram_safe(messages)
    assert len(messages) >= 4
    assert messages[0].startswith("🚀 <b>MONSTER AI</b> · NSE large-move scan")
    for index, message in enumerate(messages, 1):
        assert message.endswith(f"<i>📄 part {index}/{len(messages)}</i>")
    assert all("Universe:" not in m and DISCLAIMER not in m for m in messages[:-1])
    assert "Universe:" in messages[-1] and DISCLAIMER in messages[-1]
    text = "\n".join(messages)
    for rank in range(1, 13):
        assert text.count(f"<b>#{rank} SYM{rank:02d}</b> · 🔥 <b>STRONG</b>") == 1


def test_maximally_escaped_content_still_fits_every_message():
    nasty = make_candidate(
        priced_in="&" * 500,
        confirmations=["<" * 500] * 3,
        contradictions=["&" * 500] * 3,
        drivers=[">" * 500] * 3,
        chain=["&" * 500],
        liquidity_note="<" * 500,
        catalysts=[Catalyst(headline="&" * 500, source="<" * 100, url="https://example.test/x")],
    )
    ctx = make_ctx(
        [nasty, nasty],
        regime={str(i) * 24: "&" * 100 for i in range(10)},
        freshness={str(i) * 24: "<" * 100 for i in range(10)},
        uncertainty_note="&" * 600,
        notes=["<" * 600] * 8,
        missing_feeds=["&" * 300] * 20,
    )
    messages = render_final_report(ctx)
    assert_telegram_safe(messages)
    assert len(messages) >= 2


def test_regime_and_freshness_pairs_are_capped_with_a_remainder_note():
    ctx = make_ctx(regime={f"k{i}": "v" for i in range(14)}, missing_feeds=[f"feed{i}" for i in range(14)],
                   freshness={f"f{i}": "ok (x)" for i in range(14)})
    text = "\n".join(render_final_report(ctx))
    assert "🔹 k9: v · +4 more" in text and "k10" not in text
    assert "🟢 f9: ok (x)\n+4 more" in text and "f10" not in text
    assert "<b>Missing feeds:</b> feed0, feed1, feed2, feed3, feed4, feed5, feed6, feed7, feed8, feed9" in text
    assert "Notes:" not in text
    noted = "\n".join(render_final_report(make_ctx(notes=["first", "second"])))
    assert "📝 <b>Notes:</b> first; second" in noted


# ------------------------------------------------------------------------------------- evening report
def make_summary(**over):
    base = dict(
        trade_date="2026-11-02",
        stage="final",
        primary_threshold=3.0,
        model_version="fake-model-0.1",
        rows=[
            EveningRow("FAKECO", 1, "MODERATE", 0.62, 4.14, 3.1, 1.2, True, source="vendor"),
            EveningRow("DUMMY&CO", 2, "WATCH", 0.35, 2.0, 1.4, -0.84, False),
            EveningRow("NOTRADED", 3, "WATCH", None, None, None, None, None),
        ],
        n=2,
        model_hit_rate=0.5,
        model_hit_ci=(0.09, 0.91),
        baseline_hit_rates={"random": 0.2, "atr": 0.4},
        mean_range_pct=3.07,
        baseline_mean_range_pct={"atr": 2.9, "random": 2.2},
        rolling={"hit_rate_30d": 0.41, "brier_30d": 0.19, "n_days": 42, "weird": None},
        calibration_note="Slightly overconfident at the top bin",
        data_quality_notes=["note A", "note <B>"],
        provisional=True,
        missing_symbols=["MISS1", "MISS2"],
    )
    base.update(over)
    return EveningSummary(**base)


def test_evening_report_golden_lines_and_table():
    messages = render_evening_report(make_summary())
    assert_telegram_safe(messages)
    got = lines(messages)
    for line in (
        "🌆 <b>MONSTER AI</b> · Evening evaluation",
        "🎯 <b>FINAL</b> · 📅 Mon 02 Nov 2026",
        "🧠 Model <code>fake-model-0.1</code>",
        "📏 Target: intraday range &gt; 3% of the open",
        "🟡 <b>PROVISIONAL: vendor data, official file pending</b>",
        "🏁 <b>Top-3 hit rate:</b> 1/2 (CI 9%-91%)",
        "🆚 <b>Baseline hit rates:</b> atr 40%, random 20%",
        "📐 <b>Mean actual range:</b> model picks 3.1% | baselines: atr 2.9%, random 2.2%",
        "📈 <b>Rolling record (out-of-sample):</b> hit_rate_30d 41%, brier_30d 0.190, n_days 42, weird n/a",
        "🧮 <b>Calibration:</b> Slightly overconfident at the top bin",
        "🧹 <b>Data-quality notes:</b> note A; note &lt;B&gt;",
        "🚫 <b>Missing symbols:</b> MISS1, MISS2",
        f"💡 <i>{EVENING_NOTE}</i>",
    ):
        assert line in got, line
    assert EVENING_NOTE == "A single day is noise; judge the model only on the rolling out-of-sample record."
    table = re.search(r"<pre>(.*?)</pre>", messages[0], re.DOTALL).group(1).strip("\n").split("\n")
    fields = [ln.replace("&amp;", "&").split() for ln in table]
    assert fields[0] == ["#", "SYMBOL", "TIER", "P(3%)", "RANGE", "DIR", "HIT"]
    assert fields[1] == ["1", "FAKECO", "MODERATE", "62%", "4.1%", "+1.2%", "✅"]
    assert fields[2] == ["2", "DUMMY&CO", "WATCH", "35%", "2.0%", "-0.8%", "❌"]
    assert fields[3] == ["3", "NOTRADED", "WATCH", "n/a", "n/a", "n/a", "➖"]
    assert "DUMMY&amp;CO" in "\n".join(got) and "<pre>" in messages[0] and "</pre>" in messages[0]
    assert any(ln.startswith("📋 <b>Picks against what happened</b>") and "✅" in ln and "❌" in ln for ln in got)


def test_evening_report_validated_flag_and_missing_statistics():
    text = "\n".join(render_evening_report(make_summary(provisional=False)))
    assert "🟢 <b>VALIDATED: official exchange data</b>" in text and "PROVISIONAL" not in text
    bare = EveningSummary("2026-11-02", "pre", 2.5, "fake-model-0.1")
    got = lines(render_evening_report(bare))
    for line in (
        "📋 No evaluated picks for this date.",
        "🏁 <b>Top-N hit rate:</b> n/a (no evaluated picks)",
        "🆚 <b>Baseline hit rates:</b> n/a",
        "📐 <b>Mean actual range:</b> model picks n/a",
        "📈 <b>Rolling record (out-of-sample):</b> n/a",
        "🟡 <b>PROVISIONAL: vendor data, official file pending</b>",
    ):
        assert line in got, line
    assert "🎯 <b>PRELIMINARY</b>" in "\n".join(got) and "📏 Target: intraday range &gt; 2.5% of the open" in "\n".join(got)
    assert not any(ln.startswith(("🧮", "🧹", "🚫")) for ln in got)


def test_evening_report_confidence_interval_variants_and_row_cap():
    as_list = "\n".join(render_evening_report(make_summary(model_hit_ci=[0.2, 0.8])))
    assert "<b>Top-3 hit rate:</b> 1/2 (CI 20%-80%)" in as_list
    for impossible in (1.5, -0.1):
        invalid = "\n".join(render_evening_report(make_summary(model_hit_rate=impossible)))
        assert "<b>Top-3 hit rate:</b> n/a (no evaluated picks)" in invalid
    for ci in (None, (0.2,), ("a", "b"), (None, 0.5)):
        assert "<b>Top-3 hit rate:</b> 1/2" in "\n".join(render_evening_report(make_summary(model_hit_ci=ci)))
        assert "(CI" not in "\n".join(render_evening_report(make_summary(model_hit_ci=ci)))
    many = [EveningRow(f"S{i:02d}", i, "WATCH", 0.3, 1.0, 1.0, 0.1, False) for i in range(1, 26)]
    text = "\n".join(render_evening_report(make_summary(rows=many, n=25, model_hit_rate=0.0, model_hit_ci=None)))
    assert "S20" in text and "S21" not in text and "... +5 more" in text and "<b>Top-25 hit rate:</b> 0/25" in text


def test_evening_report_missing_symbols_are_capped_and_hit_rate_n_comes_from_summary():
    many = [f"M{i:02d}" for i in range(20)]
    text = "\n".join(render_evening_report(make_summary(missing_symbols=many)))
    assert "<b>Missing symbols:</b> M00, M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11, M12, M13, M14 (+5 more)" in text
    assert "<b>Top-3 hit rate:</b> 1/2" in text  # x/n uses the evaluated count, not the number of table rows


def test_evening_rows_are_sorted_by_rank_and_escape_everything():
    rows = [EveningRow("<B>2", 2, "WATCH", 0.3, 1, 1, 0, False), EveningRow("A&1", 1, "<I>", 0.3, 1, 1, 0, True)]
    messages = render_evening_report(make_summary(rows=rows, stage="<x>", rolling={"<k>": 0.5}, baseline_hit_rates={"<b>": 0.1}))
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert text.index("A&amp;1") < text.index("&lt;B&gt;2") and "🎯 <b>&lt;X&gt;</b>" in text
    assert "&lt;k&gt; 0.500" in text and "&lt;b&gt; 10%" in text


# ------------------------------------------------------------------------------ failure notice / alerts
def test_failure_notice_states_reason_and_never_implies_no_setups():
    messages = render_failure_notice("2026-11-02", "final", "deadline missed", {"feed": "yahoo", "age": "2 d"})
    assert_telegram_safe(messages)
    got = lines(messages)
    for line in (
        "🛑 <b>MONSTER AI</b> · REPORT WITHHELD",
        "🎯 <b>FINAL</b> · 📅 Mon 02 Nov 2026",
        "❗ <b>Reason:</b> deadline missed",
        "🔎 <b>Details:</b> feed=yahoo; age=2 d",
        "🛠 <b>Next step:</b> <code>sudo monster-ai logs --job final_scan</code>",
    ):
        assert line in got, line
    note = next(ln for ln in got if ln.startswith("💡"))
    assert note.startswith("💡 <b>No candidate list was published for this date.</b>") and "not a market signal" in note


@pytest.mark.parametrize(
    "details, expected",
    [
        (None, None),
        ("plain text", "🔎 <b>Details:</b> plain text"),
        (["a", "", "b"], "🔎 <b>Details:</b> a; b"),
        (("x",), "🔎 <b>Details:</b> x"),
        ({"k": "v<1>"}, "🔎 <b>Details:</b> k=v&lt;1&gt;"),
    ],
)
def test_failure_notice_details_variants(details, expected):
    got = lines(render_failure_notice("2026-11-02", "pre", "stale data", details))
    assert "🎯 <b>PRELIMINARY</b>" in "\n".join(got)
    assert "sudo monster-ai logs --job prelim_scan" in "\n".join(got)
    detail_lines = [ln for ln in got if ln.startswith("🔎")]
    assert detail_lines == ([expected] if expected else [])


def test_failure_notice_handles_odd_inputs_and_huge_details():
    messages = render_failure_notice("bad", "weird<stage>", "", "q" * 20000)
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert "<b>WEIRD&lt;STAGE&gt;</b>" in text and "<b>Reason:</b> n/a" in text and "📅 bad" in text
    assert "<b>Details:</b> " + "q" * 697 + "..." in text
    assert "<code>sudo monster-ai jobs</code>" in text  # an unknown stage: point at the job list, not at a guessed job
    assert "🎯 <b>n/a</b>" in "\n".join(render_failure_notice("2026-11-02", None, "x"))


def test_render_alert_golden_and_limits():
    assert render_alert("error", "feed_down", "yahoo <down> & out") == (
        "🟠 <b>ERROR</b> · <code>feed_down</code>\nyahoo &lt;down&gt; &amp; out"
    )
    stamped = render_alert("warning", "disk", "low", created_at="2026-11-02T03:00:00Z")
    assert stamped.endswith("\n<i>🕗 2026-11-02 08:30 IST</i>") and stamped.startswith("🟡 <b>WARNING</b>")
    assert render_alert("", "", "") == "🔵 <b>INFO</b> · <code>n/a</code>\nn/a"
    assert "IST" not in render_alert("info", "x", "y", created_at="garbage")
    huge = render_alert("critical", "x", "<" * 10000)
    assert_telegram_safe([huge])
    assert huge.splitlines()[1].endswith("...") and len(huge) <= 3900


def test_alerts_have_a_severity_colour_and_good_news_gets_a_tick():
    for level, icon in (("critical", "🔴"), ("error", "🟠"), ("warning", "🟡"), ("warn", "🟡"), ("info", "🔵"), ("weird", "⚪")):
        assert render_alert(level, "x", "y").startswith(f"{icon} <b>")
    assert render_alert("info", "bootstrap_complete", "done").startswith("🎉 <b>INFO</b>")
    assert render_alert("info", "health_recovered", "ok").startswith("✅ <b>INFO</b>")


def test_alerts_suggest_the_first_command_to_run_unless_the_message_already_does():
    got = lines([render_alert("error", "scan_final_failed", "blocked by the data-quality gate")])
    assert "🛠 <b>Next step:</b> <code>sudo monster-ai logs --job final_scan</code>" in got
    assert "🛠 <b>Next step:</b> <code>sudo monster-ai logs --job collect_india</code>" in lines(
        [render_alert("error", "job_failed_collect_india", "x")])
    assert "🛠" not in render_alert("warning", "daemon_restarted", "restarted. See: python -m app.cli logs --role daemon")
    assert "🛠" not in render_alert("error", "something_new", "no suggestion for an unknown code")
    assert "<code>sudo monster-ai logs</code>" in render_alert("error", "job_failed_bad name!", "x")  # an unsafe suffix is never echoed


def test_alert_digest_batches_into_one_message_with_ist_times():
    alerts = [
        {"level": "error", "code": "feed_down", "message": "yahoo down", "created_at": "2026-11-02T03:00:00Z"},
        {"level": "WARNING", "code": "disk_low", "message": "1.2 GB free <warn>", "created_at": None},
        {"level": "", "code": "", "message": ""},
    ]
    messages = render_alert_digest(alerts)
    assert_telegram_safe(messages)
    assert messages == [
        "🔔 <b>MONSTER AI</b> · 3 alerts\n"
        f"{style.RULE}\n"
        "🟠 <b>ERROR</b> · <code>feed_down</code> · 🕗 2026-11-02 08:30 IST\n"
        "yahoo down\n"
        "🟡 <b>WARNING</b> · <code>disk_low</code>\n"
        "1.2 GB free &lt;warn&gt;\n"
        "🔵 <b>INFO</b> · <code>n/a</code>\n"
        "n/a\n"
        "🛠 <b>Next step:</b> <code>sudo monster-ai health</code> · <code>sudo monster-ai logs</code>"
    ]
    many = [{"level": "error", "code": f"c{i}", "message": "m" * 400} for i in range(40)]
    split = render_alert_digest(many)
    assert_telegram_safe(split) and len(split) > 1


# ---------------------------------------------------------------------------------------- unit failure
def test_unit_failure_shows_the_unit_the_time_and_the_log_lines():
    text = render_unit_failure("monster-ai.service", "line <1>\nline & 2", "2026-11-02T03:00:00Z")
    assert_telegram_safe([text])
    got = lines([text])
    assert got[0] == "🚨 <b>Service failed</b> · <code>monster-ai.service</code>"
    assert "<i>🕗 2026-11-02 08:30 IST</i>" in got and "📜 <b>Last journal lines</b>" in got
    assert "<pre>line &lt;1&gt;" in text and "line &amp; 2</pre>" in text
    assert "🛠 <b>Next step:</b> <code>journalctl -u monster-ai.service -n 100 --no-pager</code>" in got
    assert "<pre>no log lines</pre>" in render_unit_failure("a.service", "")
    assert "🕗" not in render_unit_failure("a.service", "x")  # no time given: none invented


def test_unit_failure_keeps_the_newest_lines_of_a_huge_tail_and_never_echoes_an_unsafe_unit_name():
    tail = "\n".join(f"old line {i}" for i in range(2000)) + "\nthe newest line <&>"
    text = render_unit_failure("bad unit; rm -rf /", tail)
    assert_telegram_safe([text])
    assert "the newest line &lt;&amp;&gt;" in text and "old line 0\n" not in text and "<pre>..." in text
    assert "journalctl" not in text and "<code>bad unit; rm -rf /</code>" in text  # shown escaped, but never as a command


# ------------------------------------------------------------------------------------- test message
def test_test_message_has_mode_and_config_summary_but_no_secrets(settings):
    secrets = {
        "telegram_bot_token": TOKEN,
        "telegram_chat_id": "1001",
        "telegram_alert_chat_id": "2002",
        "llm_api_key": "sk-fake-llm-key-12345",
        "llm_api_base": "https://llm.example.test/v1",
        "llm_model": "fake-model",
        "typesafe_api_key": "ts-fake-typesafe-key-12345",
        "s3_bucket": "fake-bucket",
        "s3_access_key_id": "AKIAFAKEFAKEFAKE1234",
        "s3_secret_access_key": "fake/secret/access/key",
        "dashboard_token": "dashboard-token-1234567890",
        "model_signing_key": "signing-key-abcdef",
        "backup_passphrase": "backup-pass-phrase",
    }
    configured = dataclasses.replace(settings, **secrets)
    text = render_test_message(configured)
    assert_telegram_safe([text])
    got = lines([text])
    assert got[0] == "🤖 <b>MONSTER AI online</b> ✅"
    assert "🧭 <b>Mode:</b> shadow - research output only; the system never places orders" in got
    assert "🔹 Report chat: ✅ configured" in got and "🔹 Alerts go to: a separate chat" in got
    assert "🔹 Preliminary report: ⚪ off · evening report: ✅ on" in got
    assert any(ln.startswith("🎯 <b>Targets:</b> primary range threshold 3% · thresholds 1, 2, 3, 5% · top-N 5 · universe fno_nifty100") for ln in got)
    for line in ("🔹 08:45:00 preliminary scan", "🔹 09:08:45 final scan", "🔹 09:12:00 report cutoff · 09:14:30 delivery deadline"):
        assert line in got, line
    assert any(ln.startswith("🔌 <b>Integrations:</b> LLM ✅ configured · Jev ✅ configured · off-box backup ✅ configured") for ln in got)
    for secret in (*secrets.values(), "1001", "2002"):
        assert secret not in text
    bare = "\n".join(lines([render_test_message(settings)]))
    assert "Report chat: ❌ NOT configured" in bare and "Alerts go to: the same chat" in bare
    assert "LLM ⚪ off · Jev ⚪ off · off-box backup ⚪ off" in bare


def test_test_message_shows_each_ai_service_by_its_own_key(settings):
    groq_only = render_test_message(dataclasses.replace(settings, groq_api_key="gsk_fake_groq_key_1234567890"))
    assert "LLM ✅ configured · Jev ⚪ off" in groq_only and "gsk_fake" not in groq_only
    jev_only = render_test_message(dataclasses.replace(settings, typesafe_api_key="ts-fake-typesafe-key-12345"))
    assert "LLM ⚪ off · Jev ✅ configured" in jev_only


def test_test_message_tolerates_partial_settings_objects():
    text = render_test_message(SimpleNamespace(mode="live", range_thresholds=(1.0, math.nan, 2.0)))
    assert_telegram_safe([text])
    assert "<b>Mode:</b> live" in text and "thresholds 1, 2%" in text and "<b>Environment:</b> n/a" in text
