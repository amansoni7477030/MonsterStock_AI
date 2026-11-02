"""The chart images of the Telegram reports: what they claim, what they leave out, and that a failure never costs the report."""

from __future__ import annotations

import io
import logging
import re

import pytest

from notifications import canvas as cv
from notifications import charts
from notifications.charts import Chart, evening_charts, ideas_charts, report_charts
from notifications.report import DISCLAIMER, SHADOW_BANNER
from ranking.types import EveningRow, EveningSummary
from tests.test_notifications_ideas import make_idea, make_report
from tests.test_notifications_report import _BAD_NUMBER, assert_telegram_safe, make_candidate, make_ctx

Image = pytest.importorskip("PIL.Image")


def make_evening(**over) -> EveningSummary:
    base = dict(
        trade_date="2026-11-02", stage="final", primary_threshold=3.0, model_version="fake-model-0.1",
        rows=[EveningRow("AAA", 1, "STRONG", 0.64, 4.2, 3.1, 1.2, True), EveningRow("BBB", 2, "MODERATE", 0.5, 2.4, 1.6, -0.8, False),
              EveningRow("CCC", 3, "WATCH", 0.4, None, None, None, None)],
        n=3, model_hit_rate=0.5, baseline_hit_rates={"atr": 0.33, "random": 0.2},
    )
    return EveningSummary(**{**base, **over})


def decode(chart: Chart):
    image = Image.open(io.BytesIO(chart.png))
    image.load()
    return image


def said(chart: Chart) -> str:
    return " ".join(chart.labels)


# ------------------------------------------------------------------------------------------ the drawing layer
def test_ascii_label_swaps_what_the_font_cannot_draw():
    assert cv.ascii_label("\u20b91,000 \u2013 A\u2192B \u2265 2 \u00d7 3") == "Rs 1,000 - A->B >= 2 x 3"
    assert cv.ascii_label("Caf\u00e9 \U0001f600  tab\tline\nbreak") == "Cafe tab line break"
    assert cv.ascii_label("a \u00b7 b") == "a \u00b7 b" and cv.ascii_label(None) == "" and cv.ascii_label(float("nan")) == "nan"


def test_text_is_measured_fitted_and_wrapped_by_width():
    assert cv.text_width("WWWW", 30) > cv.text_width("iiii", 30) > 0
    long = "A very long company name that does not fit " * 4
    fitted = cv.fit(long, 24, 300)
    assert fitted.endswith("...") and cv.text_width(fitted, 24) <= 300 and cv.fit("short", 24, 300) == "short"
    lines = cv.wrap(long, 24, 300)
    assert len(lines) > 1 and all(cv.text_width(line, 24) <= 300 for line in lines) and " ".join(lines).replace("...", "")
    (only,) = cv.wrap("W" * 200, 24, 300)  # one word that is too long is cut, never overflows
    assert cv.text_width(only, 24) <= 300 and only.endswith("...")


def test_a_donut_is_cut_in_the_order_given_with_a_hole_in_the_middle():
    canvas = cv.Canvas(200)
    canvas.donut(100, 100, 80, [(1, cv.RED), (1, cv.GREEN), (0, cv.BLUE)])
    pixels = Image.open(io.BytesIO(canvas.png()))
    assert pixels.getpixel((160, 100)) == cv.RED and pixels.getpixel((40, 100)) == cv.GREEN  # clockwise from 12 o'clock
    assert pixels.getpixel((100, 100)) == cv.BG and pixels.getpixel((5, 5)) == cv.BG


def test_an_empty_or_single_donut_is_a_ring_not_an_error():
    empty, single = cv.Canvas(200), cv.Canvas(200)
    empty.donut(100, 100, 80, [(0, cv.RED)])
    single.donut(100, 100, 80, [(3, cv.AMBER)])
    assert Image.open(io.BytesIO(empty.png())).getpixel((160, 100)) == cv.TRACK
    assert Image.open(io.BytesIO(single.png())).getpixel((160, 100)) == cv.AMBER


def test_the_canvas_records_every_label_and_refuses_to_exist_without_pillow(monkeypatch):
    canvas = cv.Canvas(100)
    canvas.text(10, 10, "Rs 5 \u2192 6")
    canvas.text(10, 40, "")
    assert canvas.labels == ["Rs 5 -> 6"]
    monkeypatch.setattr(cv, "Image", None)
    assert cv.available() is False
    with pytest.raises(RuntimeError, match="Pillow"):
        cv.Canvas(100)


# --------------------------------------------------------------------------------------------- the charts
def test_every_chart_is_a_png_that_telegram_accepts():
    built = [*ideas_charts(make_report()), *report_charts(make_ctx()), *evening_charts(make_evening())]
    assert [chart.name for chart in built] == ["ideas-board", "ideas-plan", "candidates", "evening"]
    for chart in built:
        image = decode(chart)
        assert (image.format, image.mode, image.width) == ("PNG", "RGB", cv.WIDTH)
        assert 300 <= image.height <= 3000 and len(chart.png) < 2_000_000  # Telegram: 10 MB, width + height <= 10000, ratio <= 20
        assert chart.filename == f"{chart.name}.png" and 0 < len(chart.caption) <= charts.CAPTION_LIMIT
        assert_telegram_safe([chart.caption])
        assert "not investment advice" in said(chart).lower() or DISCLAIMER[:30] in said(chart)


def test_a_chart_is_deterministic_and_a_sample_carries_its_strip():
    first, again = ideas_charts(make_report()), ideas_charts(make_report())
    assert [chart.png for chart in first] == [chart.png for chart in again]
    sample = ideas_charts(make_report(), sample=True)
    assert charts._SAMPLE_TEXT in sample[0].labels and charts._SAMPLE_TEXT not in first[0].labels
    assert decode(sample[0]).getpixel((5, 5)) == cv.SAMPLE_RED and decode(first[0]).getpixel((5, 5)) == cv.BG


def test_a_closing_hour_score_is_a_score_and_never_a_probability():
    board, plan = ideas_charts(make_report())
    assert "NOT a probability" in said(board) and "ranking aid" in said(board)
    for chart in (board, plan):
        text = said(chart)
        assert all(text[max(0, hit.start() - 6):hit.start()] == "NOT a " for hit in re.finditer("probab", text)), text
        assert not re.search(r"chance|likel|win rate", text, re.IGNORECASE)
    assert "HEURISTIC SETUPS" in said(board)


def test_the_board_shows_the_lists_the_market_and_the_scores():
    (board, _plan) = ideas_charts(make_report())
    text = said(board)
    for expected in ("Closing-hour ideas", "Mon 02 Nov 2026", "snapshot 14:31 IST", "enter before 15:15 IST", "Nifty +0.40%",
                     "VIX 14.0 (-1.4%)", "Regime bullish", "120 up", "60 down", "Ideas by list", "BTST", "Swing", "Bearish F&O",
                     "Watchlist", "DEMOCO", "SWINGCO", "SHORTCO", "WATCHCO"):
        assert expected in text, expected
    assert [label for label in board.labels if re.fullmatch(r"\d{2}", label)] == ["78", "70", "66", "64"]  # scores, best first


def test_the_trade_plan_draws_the_levels_of_the_ideas_with_an_entry_only():
    _board, plan = ideas_charts(make_report())
    text = said(plan)
    for expected in ("stop -4.00%", "T1 +4.00%", "T2 +8.00%", "reward / risk 1.0 / 2.0", "SHORT", "LONG", "DEMOCO", "SHORTCO", "cost ~0.30%"):
        assert expected in text, expected
    assert "WATCHCO" not in text  # a watchlist level is a trigger, not a trade with a stop and targets


def test_the_candidates_chart_shows_a_probability_only_for_a_model_that_is_not_the_baseline():
    for status, banner in (("validated", False), ("shadow", True), ("baseline", True)):
        (chart,) = report_charts(make_ctx(model_status=status))
        text = said(chart)
        assert ("P 62%" in text) == (status != "baseline"), status
        assert ("heuristic" in text) == (status == "baseline") and (SHADOW_BANNER in text) == banner, status
        assert "2.1-5.6%" in text and "FAKECO" in text
    (validated,) = report_charts(make_ctx())
    assert "Model status: VALIDATED" in said(validated) and "STRONG" in said(validated)
    (capped,) = report_charts(make_ctx(model_status="shadow"))
    assert "WATCH" in said(capped) and "STRONG" not in said(capped)  # the tier is capped like it is in the text


def test_the_candidates_chart_adds_the_sector_mix_when_there_is_more_than_one_sector():
    mixed = [make_candidate(rank=1, symbol="AAA", sector="Energy"), make_candidate(rank=2, symbol="BBB", sector="Banks"),
             make_candidate(rank=3, symbol="CCC", sector="Energy")]
    (chart,) = report_charts(make_ctx(candidates=mixed))
    assert "Sector mix of the candidates" in said(chart) and "Energy" in chart.labels and "Banks" in chart.labels
    (plain,) = report_charts(make_ctx())
    assert "Sector mix of the candidates" not in said(plain)


def test_the_evening_chart_counts_hits_misses_and_picks_not_judged():
    (chart,) = evening_charts(make_evening())
    text = said(chart)
    for expected in ("1/3", "Hit", "Missed", "Not judged", "hit 4.2%", "miss 2.4%", "pending n/a", "Model", "ATR", "Random", "33%", "20%",
                     "PROVISIONAL: vendor data, official file pending", "A single day is noise"):
        assert expected in text, expected
    assert "1 of 3 picks beat the target" in chart.caption
    (final,) = evening_charts(make_evening(provisional=False))
    assert "VALIDATED: official exchange data" in said(final)


def test_nothing_is_drawn_when_there_is_nothing_honest_to_draw():
    assert report_charts(make_ctx(no_setup=True, no_setup_reason="weak")) == []  # a day without a setup gets no chart of weak names
    assert report_charts(make_ctx(candidates=[])) == []
    assert report_charts(make_ctx(candidates=[make_candidate(range={})])) == []
    assert ideas_charts(make_report(ideas=[])) == []
    assert evening_charts(make_evening(rows=[])) == []


def test_missing_and_non_finite_figures_are_left_out_never_invented():
    ideas = [make_idea(symbol="NOSCORE", score=None), make_idea(symbol="NANSCORE", score=float("nan")),
             make_idea(symbol="NOLEVELS", risk_pct=None, reward1_pct=None, reward2_pct=None, rr1=None, rr2=None, entry=float("nan"))]
    built = ideas_charts(make_report(ideas=ideas, market={"nifty_chg": None, "vix": float("inf"), "advancers": None}))
    assert [chart.name for chart in built] == ["ideas-board"]  # no idea has the levels a plan needs
    text = said(built[0])
    assert "NOLEVELS" in text and "NOSCORE" not in text and "NANSCORE" not in text and not _BAD_NUMBER.search(text)
    assert "Nifty" not in text and "VIX" not in text and "Breadth" not in text
    (evening,) = evening_charts(make_evening(model_hit_rate=None, baseline_hit_rates={"atr": float("nan")}))
    assert "Hit rate of this day's picks" not in said(evening) and not _BAD_NUMBER.search(said(evening))


def test_a_crowded_report_is_clipped_to_ten_rows_and_stays_inside_telegrams_limits():
    ideas = [make_idea(symbol=f"SYM{n:02d}", rank=n, score=95.0 - n) for n in range(1, 31)]
    board, plan = ideas_charts(make_report(ideas=ideas))
    assert "Setup score (top 10 of 30)" in said(board)
    assert sum(label.startswith("SYM") for label in board.labels) == 10
    assert sum(label.startswith("SYM") for label in plan.labels) == 8  # the plan is the 8 best
    assert all(decode(chart).height <= 3000 for chart in (board, plan))


def test_hostile_text_is_drawn_as_plain_ascii_and_never_breaks_a_chart():
    hostile = make_idea(symbol="\u00dcn\u00ef<b>&\u20b9\U0001f600" * 4, name="<script>", sector="R&D \u2192")
    candidate = make_candidate(symbol="<i>X</i>\u20b9" * 6, sector="\u20b9\u20b9", name="\U0001f600")
    built = [*ideas_charts(make_report(ideas=[hostile])), *report_charts(make_ctx(candidates=[candidate]))]
    assert len(built) == 3
    for chart in built:
        assert all(ch == "\u00b7" or " " <= ch <= "~" for label in chart.labels for ch in label)
        assert all(cv.text_width(label, 20) < cv.WIDTH for label in chart.labels)


def test_a_failing_chart_costs_only_itself(monkeypatch, caplog):
    def boom(*_args):
        raise RuntimeError("cannot draw")

    monkeypatch.setattr(charts, "_ideas_plan", boom)
    with caplog.at_level(logging.WARNING):
        built = ideas_charts(make_report())
    assert [chart.name for chart in built] == ["ideas-board"]
    assert "chart ideas-plan skipped: RuntimeError" in caplog.text


def test_without_pillow_nothing_is_drawn_and_nothing_fails(monkeypatch, caplog):
    monkeypatch.setattr(cv, "Image", None)
    with caplog.at_level(logging.INFO):
        assert ideas_charts(make_report()) == [] and report_charts(make_ctx()) == [] and evening_charts(make_evening()) == []
    assert "Pillow is not installed" in caplog.text


def test_report_dicts_are_accepted_like_report_objects():
    """The text renderers take dicts as well as dataclasses; so do the charts."""
    from dataclasses import asdict

    (from_object,) = report_charts(make_ctx())
    (from_dict,) = report_charts(asdict(make_ctx()))
    assert from_object.png == from_dict.png
