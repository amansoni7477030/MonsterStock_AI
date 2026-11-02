"""The sample messages behind ``telegram-preview``: every kind is built, Telegram-safe, fake and always marked as a preview."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from notifications import samples
from tests.test_notifications_report import assert_telegram_safe

NOW = datetime(2026, 11, 2, 3, 42, tzinfo=timezone.utc)  # 09:12 IST on a Monday


@pytest.fixture()
def built(settings):
    return samples.build_samples(settings, NOW)


def test_every_kind_is_built_once_in_the_documented_order(built):
    assert [kind for kind, _ in built] == list(samples.KINDS)
    assert samples.ALERT_KINDS <= set(samples.KINDS)


def test_every_sample_is_valid_telegram_html_within_the_limits(built):
    for kind, messages in built:
        assert messages, kind
        assert_telegram_safe(messages)
        assert_telegram_safe(samples.marked(messages))


def test_every_single_message_carries_the_preview_banner(built):
    for kind, messages in built:
        marked = samples.marked(messages)
        assert len(marked) >= len(messages) and all(text.startswith(samples.PREVIEW_BANNER) for text in marked), kind
    assert "made-up data, not a market signal" in samples.PREVIEW_BANNER


def test_a_banner_that_pushes_a_message_over_the_limit_splits_it_again():
    marked = samples.marked(["x" * 3890, "short"])
    assert len(marked) == 3 and all(len(text) <= 3900 for text in marked)
    assert marked[0].startswith(samples.PREVIEW_BANNER) and marked[-1].endswith("short")


def test_the_samples_use_obviously_fake_names_and_only_example_dot_com_links(built):
    text = "\n".join("\n".join(messages) for _, messages in built)
    symbols = set(re.findall(r"<b>#\d (\w+)</b>", text)) | set(re.findall(r"^\s*\d (\w+)\s", text, re.MULTILINE))
    assert symbols and all(symbol.startswith("DEMO") for symbol in symbols), symbols
    assert set(re.findall(r'href="https?://([^/"]+)', text)) == {"example.com"}


def test_the_sample_dates_and_times_follow_the_clock_they_are_given(built):
    texts = {kind: "\n".join(messages) for kind, messages in built}
    assert "📅 Mon 02 Nov 2026" in texts["final"] and "🕘 09:12 IST" in texts["final"]
    assert "🕗 08:45 IST" in texts["prelim"] and "🕘 2026-11-02 09:12 IST" in texts["alert"]


def test_the_samples_show_every_feature_of_the_design(built):
    texts = {kind: "\n".join(messages) for kind, messages in built}
    final = texts["final"]
    assert all(token in final for token in ("🥇", "🥈", "🥉", "🔥", "⚡", "👀", "<code>██████░░░░</code>", "🚩", "🔎", "🟢", "🟡"))
    assert "🧪 <b>SHADOW MODE" in texts["prelim"] and "<b>PRELIMINARY</b>" in texts["prelim"] and "👀 <b>WATCH</b>" in texts["prelim"]
    assert "🛑 <b>NO HIGH-CONVICTION SETUP today</b>" in texts["no-setup"]
    ideas = texts["ideas"]
    assert all(token in ideas for token in ("BTST", "Swing", "Bearish setups", "intraday watchlist", "Buy above", "Sell below", "📚 Live record"))
    assert "record passes the gate" in ideas and "not validated" in ideas and "NOT a probability" in ideas
    assert "📚" in texts["evening"] and "✅" in texts["evening"] and "❌" in texts["evening"]
    assert all(icon in texts["alert"] for icon in ("🔴", "🟠", "🟡", "🎉")) and "🛠" in texts["alert"]
    assert "🔔 <b>MONSTER AI</b> · 3 alerts" in texts["digest"]
    assert "REPORT WITHHELD" in texts["failure"] and "🚨 <b>Service failed</b>" in texts["unit"]
    assert "MONSTER AI online" in texts["online"]
    assert "At a glance" in final and "At a glance" in ideas and "Tomorrow's levels" in ideas and "<pre>" in texts["evening"]


# ------------------------------------------------------------------------------------------- the images
@pytest.fixture(scope="module")
def pictures():
    pytest.importorskip("PIL")  # only the image tests need Pillow
    return samples.build_sample_charts(NOW)


def test_the_kinds_with_images_are_the_report_kinds_and_every_image_is_marked_as_a_sample(pictures):
    from notifications import charts

    assert list(pictures) == ["final", "prelim", "ideas", "evening"] and set(pictures) <= set(samples.KINDS)
    assert [len(items) for items in pictures.values()] == [1, 1, 2, 1]
    for kind, items in pictures.items():
        for chart in items:
            assert charts._SAMPLE_TEXT in chart.labels, (kind, chart.name)
            assert chart.png.startswith(b"\x89PNG")


def test_a_marked_caption_is_short_valid_html_that_starts_with_the_banner(pictures):
    for items in pictures.values():
        for chart in items:
            caption = samples.marked_caption(chart.caption)
            assert caption.startswith(samples.PREVIEW_BANNER) and len(caption) <= 1024
            assert_telegram_safe([caption])
    assert len(samples.marked_caption("x" * 5000)) == 1024


def test_the_sample_images_use_the_same_made_up_data_as_the_sample_texts(pictures):
    labels = " ".join(label for items in pictures.values() for chart in items for label in chart.labels)
    assert {"DEMOONE", "DEMOTWO", "DEMOTRE", "DEMOFOR"} <= set(re.findall(r"\bDEMO[A-Z]+\b", labels))
    assert "\u20b9" not in labels  # the font of the images has no rupee sign
