"""Renderer tests: golden lines on obviously fake data, escaping, NaN handling, caps and message limits."""

from __future__ import annotations

import dataclasses
import html
import math
import re
from types import SimpleNamespace

import pytest

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
        "<b>MONSTER AI | NSE LARGE-MOVE SCAN</b>",
        "Date: 2026-11-02 (Mon) | Snapshot: 2026-11-02 09:12:00 IST | Stage: FINAL | Model: fake-model-0.1 [VALIDATED]",
        "<b>#1 FAKECO - STRONG</b>  Fake Company Ltd | Fake Sector | F&amp;O",
        "P(range &gt;= 3%): 62% (calibrated)   [1%: 91% | 2%: 78% | 3%: 62% | 5%: 31%]",
        "P(unusual for this stock): 40%",
        "Expected intraday range: 3.4% (80% interval 2.1-5.6%)   Max excursion from open (q50/q90): 2.2% / 3.8%",
        "Direction (secondary, not a trade signal): P(up close) 52% / P(down close) 48%",
        'Main catalyst: Fake plant outage lifts prices - Fake Wire (<a href="https://example.test/news/1">link</a>)',
        "Global-to-India chain: Fake crude spike -&gt; fake refiners -&gt; FAKECO",
        "Confirmations: Pre-open volume 3x; Futures OI up",
        "Contradictions: none",
        "Priced-in assessment: Partly priced in (gap +1.2%)",
        "Liquidity and data quality: pass - liquid | turnover Rs 412.5 cr | est. round-trip cost 0.12%",
        "Model drivers (diagnostic, not causal): gap_pct; atr_ratio; news_novelty",
        "Market regime: vix=calm, breadth=mixed",
        "Data freshness: yahoo=ok (3 min old), nse_preopen=ok (1 min old)",
        "Missing feeds: none",
        "Model uncertainty: Intervals are wide for 2 names | max candidate uncertainty 0.42 (0-1, higher = less reliable)",
        "Universe: 187 of 212 passed liquidity/data gates.",
        f"<i>{DISCLAIMER}</i>",
    ]
    for line in expected:
        assert line in got, line
    assert [got.index(line) for line in expected] == sorted(got.index(line) for line in expected)
    assert "SHADOW MODE" not in messages[0]
    assert DISCLAIMER.startswith("Probabilistic research output; not investment advice and not a guarantee.")


@pytest.mark.parametrize(
    "status, label",
    [
        ("validated", "VALIDATED"),
        ("shadow", "SHADOW - unvalidated"),
        ("baseline", "BASELINE (no trained model)"),
        ("something-else", "SHADOW - unvalidated"),
        (None, "SHADOW - unvalidated"),
    ],
)
def test_model_status_labels_and_shadow_banner(status, label):
    text = "\n".join(render_final_report(make_ctx(model_status=status)))
    assert f"Model: fake-model-0.1 [{label}]" in text
    assert (f"<b>{SHADOW_BANNER}</b>" in text) == (status != "validated")
    if status != "validated":
        assert SHADOW_BANNER == "SHADOW MODE: no demonstrated edge over simple baselines yet - tiers are capped at WATCH."


def test_unvalidated_models_never_show_tiers_above_watch():
    ctx = make_ctx(
        [make_candidate(1, "AAA", tier="STRONG"), make_candidate(2, "BBB", tier="MODERATE"), make_candidate(3, "CCC", tier="WATCH")],
        model_status="shadow",
    )
    text = "\n".join(render_final_report(ctx))
    assert "STRONG" not in text and "MODERATE" not in text and text.count("- WATCH</b>") == 3
    validated = "\n".join(render_final_report(dataclasses.replace(ctx, model_status="validated")))
    assert "#1 AAA - STRONG" in validated and "#2 BBB - MODERATE" in validated


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
    assert "<b>MONSTER AI | NSE LARGE-MOVE SCAN - PRELIMINARY (08:45 scan)</b>" in text
    assert "Stage: PRELIMINARY" in text and "Stage: FINAL" not in text and "final" not in text.lower()
    assert "must not be treated as confirmed" in text
    assert "<b>#1 FAKECO - STRONG</b>" in text and "Universe: 187 of 212" in text


def test_ranking_order_and_rank_fallbacks():
    cands = [make_candidate(3, "CCC"), make_candidate(1, "AAA"), make_candidate(2, "BBB"), make_candidate(None, "ZZZ")]
    heads = [ln for ln in lines(render_final_report(make_ctx(cands))) if ln.startswith("<b>#")]
    assert [re.match(r"<b>#(\d+) (\w+)", h).groups() for h in heads] == [
        ("1", "AAA"), ("2", "BBB"), ("3", "CCC"), ("4", "ZZZ"),
    ]
    odd = make_candidate(1, "ODD", tier="spicy<tier>")
    assert "<b>#1 ODD - SPICY&lt;TIER&gt;</b>" in "\n".join(render_final_report(make_ctx([odd])))


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
    assert "<b>NO HIGH-CONVICTION SETUP today</b>" in got
    assert "Reason: Top probability only 33% &lt;low&gt;" in got
    assert "<b>WEAK / WATCHLIST ONLY</b> - shown for transparency only; these are not signals." in got
    assert "<b>#1 AAA - WEAK / WATCHLIST ONLY</b>  Fake Company Ltd | Fake Sector | F&amp;O" in got
    assert "STRONG" not in "\n".join(messages)


def test_no_candidates_means_no_setup_even_without_the_flag():
    text = "\n".join(render_final_report(make_ctx([], no_setup=False)))
    assert "<b>NO HIGH-CONVICTION SETUP today</b>" in text
    assert "Reason: no candidate cleared the ranking gates" in text
    assert "WEAK / WATCHLIST ONLY" not in text and "<b>#1" not in text
    flagged = "\n".join(render_final_report(make_ctx([], no_setup=True)))
    assert "Reason: not stated by the ranking stage" in flagged
    assert "Universe: 187 of 212 passed liquidity/data gates." in flagged  # the footer is still sent


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
    assert "Main catalyst: Fake headline - Fake Wire" in lines(messages)
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
        assert f'Main catalyst: Fake (<a href="{shown}">link</a>)' in lines(messages)


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
        "<b>#1 FAKECO - STRONG</b>",
        "P(range &gt;= 3%): n/a (calibrated)   [1%: n/a | 3%: n/a]",
        "P(unusual for this stock): n/a",
        "Expected intraday range: n/a (80% interval n/a)   Max excursion from open (q50/q90): n/a / n/a",
        "Direction (secondary, not a trade signal): P(up close) n/a / P(down close) n/a",
        "Main catalyst: none identified",
        "Global-to-India chain: n/a",
        "Confirmations: none",
        "Contradictions: none",
        "Priced-in assessment: not assessed",
        "Liquidity and data quality: pass | turnover n/a | est. round-trip cost n/a",
        "Model drivers (diagnostic, not causal): n/a",
        "Market regime: n/a",
        "Data freshness: n/a",
        "Missing feeds: none",
        "Model uncertainty: n/a",
        "Universe: n/a of n/a passed liquidity/data gates.",
    ):
        assert line in got, line
    text = "\n".join(messages)
    assert not _BAD_NUMBER.search(text) and "0.1234" not in text


def test_completely_empty_candidate_and_context_still_render():
    bare = Candidate(symbol="", rank=0, tier="", p={}, primary_threshold=float("nan"), range={})
    messages = render_final_report(ReportContext("", "final", "", "fake-model", "validated", candidates=[bare]))
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert "<b>#1 UNKNOWN - n/a</b>" in text and "Snapshot: n/a" in text
    assert "P(range &gt;= n/a): n/a (calibrated)" in text and not _BAD_NUMBER.search(text)
    odd = "\n".join(render_final_report(make_ctx(trade_date="<b>bad</b>-and-long")))
    assert "Date: &lt;b&gt;bad&lt;/b&gt; |" in odd and "-and-long" not in odd
    assert "Date: 2026-11-02 (Mon) |" in "\n".join(render_final_report(make_ctx(trade_date="2026-11-02T00:00:00")))


def test_wrongly_typed_collections_degrade_gracefully_instead_of_crashing():
    ctx = make_ctx(candidates=3.14, missing_feeds=7, notes=None, regime=["x"])
    text = "\n".join(render_final_report(ctx))
    assert "<b>NO HIGH-CONVICTION SETUP today</b>" in text and "Missing feeds: 7" in text and "Market regime: n/a" in text
    odd_candidate = make_candidate(catalysts=5, drivers=2.5, confirmations={"x": 1})
    assert "Main catalyst: none identified" in "\n".join(render_final_report(make_ctx([odd_candidate])))
    summary = "\n".join(render_evening_report(make_summary(rows=5.0, missing_symbols=3.5, rolling=[1], baseline_hit_rates=None)))
    assert "No evaluated picks for this date." in summary and "Missing symbols: 3.5" in summary
    assert "Rolling record (out-of-sample): n/a" in summary


def test_liquidity_warning_when_liquidity_or_data_quality_fails():
    for override in ({"liquidity_ok": False}, {"data_quality": "warning"}, {"data_quality": "stale"}):
        text = "\n".join(render_final_report(make_ctx([make_candidate(**override)])))
        assert "Liquidity and data quality: warning - liquid | turnover Rs 412.5 cr" in text


# ----------------------------------------------------------------------------- numbers and probabilities
def test_probabilities_are_rounded_percentages_never_raw_scores():
    p = {"hit_1": 0.625, "hit_2": 0.0, "hit_3": 1.0, "hit_5": 0.995, "up": 0.004, "down": 0.6234, "raw_score": 0.8765}
    text = "\n".join(render_final_report(make_ctx([make_candidate(p=p)])))
    assert "[1%: 63% | 2%: 0% | 3%: 100% | 5%: 100%]" in text
    assert "P(up close) 0% / P(down close) 62%" in text
    for raw in ("0.625", "0.6234", "62.3", "0.8765", "0.995"):
        assert raw not in text


def test_non_integer_thresholds_sort_numerically_and_pick_the_primary():
    p = {"hit_10": 0.05, "hit_2.5": 0.55, "hit_1": 0.9, "hit_2_5": 0.5, "unusual": 0.3, "up": 0.5, "down": 0.5}
    cand = make_candidate(p=p, primary_threshold=2.5)
    text = "\n".join(render_final_report(make_ctx([cand])))
    assert "P(range &gt;= 2.5%):" in text and "[1%: 90% | 2.5%: 5" in text and "| 10%: 5%]" in text


def test_range_and_cost_formatting_edge_cases():
    cand = make_candidate(range={"q10": -0.04, "q50": 0.04, "q90": 12.345, "exc_q50": -0.0, "exc_q90": 7}, est_cost_pct=0.0, turnover_cr=1234567.89)
    text = "\n".join(render_final_report(make_ctx([cand])))
    assert "Expected intraday range: 0.0% (80% interval 0.0-12.3%)" in text
    assert "Max excursion from open (q50/q90): 0.0% / 7.0%" in text
    assert "turnover Rs 1,234,567.9 cr | est. round-trip cost 0.00%" in text


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
    assert f"Main catalyst: {'H' * 137}... - {'S' * 37}... (<a" in "\n".join(got)
    assert "H" * 141 not in "\n".join(got) and "Third must not appear" not in "\n".join(got)
    assert "Also: Second headline" in got
    assert "Confirmations: conf 0; conf 1; conf 2" in got
    assert "Contradictions: contra 0; contra 1; contra 2" in got
    assert "Model drivers (diagnostic, not causal): drv 0; drv 1; drv 2" in got
    assert "Global-to-India chain: chain 0" in got and "chain 1" not in "\n".join(got)
    assert f"Priced-in assessment: {'P' * 197}..." in got


def test_control_characters_and_newlines_are_flattened():
    cand = make_candidate(catalysts=[Catalyst(headline="line one\nline\ttwo\x00\x1b[31m \u202eevil\u202c", source="Wire")])
    text = "\n".join(render_final_report(make_ctx([cand])))
    assert "Main catalyst: line one line two[31m evil - Wire" in text
    assert "\x00" not in text and "\x1b" not in text and "\u202e" not in text and "\u202c" not in text


def test_lone_surrogates_are_stripped_so_every_message_is_valid_utf8():
    cand = make_candidate(catalysts=[Catalyst(headline="broken \ud83d feed \udc00 text", source="Wire")], name="Fa\ud800ke")
    messages = render_final_report(make_ctx([cand]))
    assert "Main catalyst: broken feed text - Wire" in "\n".join(messages)
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
    assert messages[0].startswith("<b>MONSTER AI | NSE LARGE-MOVE SCAN</b>")
    for index, message in enumerate(messages, 1):
        assert message.endswith(f"<i>(part {index}/{len(messages)})</i>")
    assert all("Universe:" not in m and DISCLAIMER not in m for m in messages[:-1])
    assert "Universe:" in messages[-1] and DISCLAIMER in messages[-1]
    text = "\n".join(messages)
    for rank in range(1, 13):
        assert text.count(f"<b>#{rank} SYM{rank:02d} - STRONG</b>") == 1


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
    ctx = make_ctx(regime={f"k{i}": "v" for i in range(14)}, missing_feeds=[f"feed{i}" for i in range(14)])
    text = "\n".join(render_final_report(ctx))
    assert "k9=v, +4 more" in text and "k10" not in text
    assert "Missing feeds: feed0, feed1, feed2, feed3, feed4, feed5, feed6, feed7, feed8, feed9" in text
    assert "Notes:" not in text
    noted = "\n".join(render_final_report(make_ctx(notes=["first", "second"])))
    assert "Notes: first; second" in noted


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
        "<b>MONSTER AI | EVENING EVALUATION</b>",
        "Date: 2026-11-02 (Mon) | Stage: FINAL | Model: fake-model-0.1 | Target: intraday range &gt;= 3% of open",
        "<b>PROVISIONAL: vendor data, official file pending</b>",
        "Top-3 hit rate: 1/2 (CI 9%-91%)",
        "Baseline hit rates: atr 40%, random 20%",
        "Mean actual range: model picks 3.1% | baselines: atr 2.9%, random 2.2%",
        "Rolling record (out-of-sample): hit_rate_30d 41%, brier_30d 0.190, n_days 42, weird n/a",
        "Calibration: Slightly overconfident at the top bin",
        "Data-quality notes: note A; note &lt;B&gt;",
        "Missing symbols: MISS1, MISS2",
        f"<i>{EVENING_NOTE}</i>",
    ):
        assert line in got, line
    assert EVENING_NOTE == "A single day is noise; judge the model only on the rolling out-of-sample record."
    table = re.search(r"<pre>(.*?)</pre>", messages[0], re.DOTALL).group(1).strip("\n").split("\n")
    fields = [ln.replace("&amp;", "&").split() for ln in table]
    assert fields[0] == ["#", "SYMBOL", "TIER", "P(3%)", "RANGE", "DIR", "HIT"]
    assert fields[1] == ["1", "FAKECO", "MODERATE", "62%", "4.1%", "+1.2%", "yes"]
    assert fields[2] == ["2", "DUMMY&CO", "WATCH", "35%", "2.0%", "-0.8%", "no"]
    assert fields[3] == ["3", "NOTRADED", "WATCH", "n/a", "n/a", "n/a", "n/a"]
    assert "DUMMY&amp;CO" in "\n".join(got) and "<pre>" in messages[0] and "</pre>" in messages[0]


def test_evening_report_validated_flag_and_missing_statistics():
    text = "\n".join(render_evening_report(make_summary(provisional=False)))
    assert "<b>VALIDATED: official exchange data</b>" in text and "PROVISIONAL" not in text
    bare = EveningSummary("2026-11-02", "pre", 2.5, "fake-model-0.1")
    got = lines(render_evening_report(bare))
    for line in (
        "No evaluated picks for this date.",
        "Top-N hit rate: n/a (no evaluated picks)",
        "Baseline hit rates: n/a",
        "Mean actual range: model picks n/a",
        "Rolling record (out-of-sample): n/a",
        "<b>PROVISIONAL: vendor data, official file pending</b>",
    ):
        assert line in got, line
    assert "Stage: PRELIMINARY" in "\n".join(got) and "Target: intraday range &gt;= 2.5% of open" in "\n".join(got)
    assert not any(ln.startswith(("Calibration", "Data-quality", "Missing symbols")) for ln in got)


def test_evening_report_confidence_interval_variants_and_row_cap():
    as_list = "\n".join(render_evening_report(make_summary(model_hit_ci=[0.2, 0.8])))
    assert "Top-3 hit rate: 1/2 (CI 20%-80%)" in as_list
    for impossible in (1.5, -0.1):
        invalid = "\n".join(render_evening_report(make_summary(model_hit_rate=impossible)))
        assert "Top-3 hit rate: n/a (no evaluated picks)" in invalid
    for ci in (None, (0.2,), ("a", "b"), (None, 0.5)):
        assert "Top-3 hit rate: 1/2" in "\n".join(render_evening_report(make_summary(model_hit_ci=ci)))
        assert "(CI" not in "\n".join(render_evening_report(make_summary(model_hit_ci=ci)))
    many = [EveningRow(f"S{i:02d}", i, "WATCH", 0.3, 1.0, 1.0, 0.1, False) for i in range(1, 26)]
    text = "\n".join(render_evening_report(make_summary(rows=many, n=25, model_hit_rate=0.0, model_hit_ci=None)))
    assert "S20" in text and "S21" not in text and "... +5 more" in text and "Top-25 hit rate: 0/25" in text


def test_evening_report_missing_symbols_are_capped_and_hit_rate_n_comes_from_summary():
    many = [f"M{i:02d}" for i in range(20)]
    text = "\n".join(render_evening_report(make_summary(missing_symbols=many)))
    assert "Missing symbols: M00, M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11, M12, M13, M14 (+5 more)" in text
    assert "Top-3 hit rate: 1/2" in text  # x/n uses the evaluated count, not the number of table rows


def test_evening_rows_are_sorted_by_rank_and_escape_everything():
    rows = [EveningRow("<B>2", 2, "WATCH", 0.3, 1, 1, 0, False), EveningRow("A&1", 1, "<I>", 0.3, 1, 1, 0, True)]
    messages = render_evening_report(make_summary(rows=rows, stage="<x>", rolling={"<k>": 0.5}, baseline_hit_rates={"<b>": 0.1}))
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert text.index("A&amp;1") < text.index("&lt;B&gt;2") and "Stage: &lt;X&gt;" in text
    assert "&lt;k&gt; 0.500" in text and "&lt;b&gt; 10%" in text


# ------------------------------------------------------------------------------ failure notice / alerts
def test_failure_notice_states_reason_and_never_implies_no_setups():
    messages = render_failure_notice("2026-11-02", "final", "deadline missed", {"feed": "yahoo", "age": "2 d"})
    assert_telegram_safe(messages)
    got = lines(messages)
    for line in (
        "<b>MONSTER AI | REPORT WITHHELD</b>",
        "Date: 2026-11-02 (Mon) | Stage: FINAL",
        "Reason: deadline missed",
        "Details: feed=yahoo; age=2 d",
    ):
        assert line in got, line
    last = got[-1]
    assert last.startswith("<b>No candidate list was published for this date.</b>") and "not a market signal" in last


@pytest.mark.parametrize(
    "details, expected",
    [
        (None, None),
        ("plain text", "Details: plain text"),
        (["a", "", "b"], "Details: a; b"),
        (("x",), "Details: x"),
        ({"k": "v<1>"}, "Details: k=v&lt;1&gt;"),
    ],
)
def test_failure_notice_details_variants(details, expected):
    got = lines(render_failure_notice("2026-11-02", "pre", "stale data", details))
    assert "Stage: PRELIMINARY" in "\n".join(got)
    detail_lines = [ln for ln in got if ln.startswith("Details:")]
    assert detail_lines == ([expected] if expected else [])


def test_failure_notice_handles_odd_inputs_and_huge_details():
    messages = render_failure_notice("bad", "weird<stage>", "", "q" * 20000)
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert "Stage: WEIRD&lt;STAGE&gt;" in text and "Reason: n/a" in text and "Date: bad |" in text
    assert "Details: " + "q" * 697 + "..." in text
    assert "Stage: n/a" in "\n".join(render_failure_notice("2026-11-02", None, "x"))


def test_render_alert_golden_and_limits():
    assert render_alert("error", "feed_down", "yahoo <down> & out") == (
        "<b>[ERROR] MONSTER AI alert</b> <code>feed_down</code>\nyahoo &lt;down&gt; &amp; out"
    )
    stamped = render_alert("warning", "disk", "low", created_at="2026-11-02T03:00:00Z")
    assert stamped.endswith("\n<i>2026-11-02 08:30 IST</i>") and stamped.startswith("<b>[WARNING] MONSTER AI alert</b>")
    assert render_alert("", "", "") == "<b>[INFO] MONSTER AI alert</b> <code>n/a</code>\nn/a"
    assert "IST" not in render_alert("info", "x", "y", created_at="garbage")
    huge = render_alert("critical", "x", "<" * 10000)
    assert_telegram_safe([huge])
    assert huge.splitlines()[1].endswith("...") and len(huge) <= 3900


def test_alert_digest_batches_into_one_message_with_ist_times():
    alerts = [
        {"level": "error", "code": "feed_down", "message": "yahoo down", "created_at": "2026-11-02T03:00:00Z"},
        {"level": "WARNING", "code": "disk_low", "message": "1.2 GB free <warn>", "created_at": None},
        {"level": "", "code": "", "message": ""},
    ]
    messages = render_alert_digest(alerts)
    assert_telegram_safe(messages)
    assert messages == [
        "<b>MONSTER AI | 3 ALERTS</b>\n"
        "[ERROR] <code>feed_down</code> (2026-11-02 08:30 IST): yahoo down\n"
        "[WARNING] <code>disk_low</code>: 1.2 GB free &lt;warn&gt;\n"
        "[INFO] <code>n/a</code>: n/a"
    ]
    many = [{"level": "error", "code": f"c{i}", "message": "m" * 400} for i in range(40)]
    split = render_alert_digest(many)
    assert_telegram_safe(split) and len(split) > 1


# ------------------------------------------------------------------------------------- test message
def test_test_message_has_mode_and_config_summary_but_no_secrets(settings):
    secrets = {
        "telegram_bot_token": TOKEN,
        "telegram_chat_id": "1001",
        "telegram_alert_chat_id": "2002",
        "llm_api_key": "sk-fake-llm-key-12345",
        "llm_api_base": "https://llm.example.test/v1",
        "llm_model": "fake-model",
        "s3_bucket": "fake-bucket",
        "s3_access_key_id": "AKIAFAKEFAKEFAKE1234",
        "s3_secret_access_key": "fake/secret/access/key",
        "dashboard_token": "dashboard-token-1234567890",
        "model_signing_key": "signing-key-abcdef",
        "backup_passphrase": "backup-pass-phrase",
        "fred_api_key": "fred-fake-key-0001",
    }
    configured = dataclasses.replace(settings, **secrets)
    text = render_test_message(configured)
    assert_telegram_safe([text])
    got = lines([text])
    assert got[0] == "<b>MONSTER AI online</b>"
    assert "Mode: shadow (research output only; the system never places orders)" in got
    assert "Telegram: report chat configured | alerts to a separate chat | prelim report off | evening report on" in got
    assert any(ln.startswith("Targets: primary range threshold 3% | thresholds 1, 2, 3, 5% | top-N 5 | universe fno_nifty100") for ln in got)
    assert "Schedule (IST): prelim 08:45:00 | final scan 09:08:45 | cutoff 09:12:00 | deadline 09:14:30" in got
    assert any(ln.startswith("Integrations: LLM configured | off-box backup configured") for ln in got)
    for secret in (*secrets.values(), "1001", "2002"):
        assert secret not in text
    bare = "\n".join(lines([render_test_message(settings)]))
    assert "report chat NOT configured | alerts to the same chat" in bare and "LLM off | off-box backup off" in bare


def test_test_message_tolerates_partial_settings_objects():
    text = render_test_message(SimpleNamespace(mode="live", range_thresholds=(1.0, math.nan, 2.0)))
    assert_telegram_safe([text])
    assert "Mode: live" in text and "thresholds 1, 2%" in text and "Environment: n/a" in text
