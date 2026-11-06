"""Telegram rendering and delivery of the 09:30 opening read: what every report must say, how it degrades on bad input, and the deadline rules."""

from __future__ import annotations

import dataclasses
import html
import math
import random
import re

import pytest

from app.clock import at_ist
from notifications import opening as renderer
from notifications.opening import EVIDENCE, LATE_BANNER, SHADOW_BANNER, render_opening_report
from notifications.service import NotificationService
from ranking.opening import MODEL_INFO
from ranking.types import OpeningIdea, OpeningReport
from tests.test_notifications_report import _BAD_NUMBER, assert_telegram_safe, lines
from tests.test_notifications_service import TOKEN, FakeClient, set_ist

KEY = "opening:2026-11-02"
LEVELS = {"trigger_long": 183.77, "long_stop": 182.13, "long_target1": 187.05, "long_target2": 190.70,
          "trigger_short": 179.21, "short_stop": 184.13, "short_target1": 174.47, "short_target2": 170.82}


def make_idea(**over) -> OpeningIdea:
    base = dict(
        symbol="TATASTEEL", rank=1, kind="PICK", ltp=182.4, open=180.2, prev_close=178.8, or_high=183.5, or_low=179.9, gap_pct=0.8, or_pct=1.9,
        move_pct=1.2, x=0.74, strength="HOT", p3=0.71, p5=0.16, day_range=[2.3, 3.6, 5.8], ahead=[1.5, 2.4, 4.1], name="Tata Steel Limited",
        sector="Metals", is_fno=True, atr_pct=2.4, turnover_cr=312.5, est_cost_pct=0.19, levels=dict(LEVELS),
        pre={"rank": 1, "tier": "STRONG", "p": 0.62, "threshold": 3.0, "stage": "final"},
        notes=["gap up +0.8%: still holding", "trading at the top of its opening range"], metrics={"clv": 0.9, "or_raw_pct": 1.9},
    )
    return OpeningIdea(**{**base, **over})


def make_report(**over) -> OpeningReport:
    base = dict(
        trade_date="2026-11-02", snapshot_ist="2026-11-02 09:30:42 IST", elapsed_min=15.2, source="NSE (SECURITIES IN F&O): 214 of 221 names",
        picks=[make_idea(), make_idea(symbol="HDFCBANK", rank=2, strength="QUIET", x=0.22, or_pct=0.45, p3=0.28, p5=0.02, name="HDFC Bank",
                                      metrics={"or_raw_pct": 0.45}, pre={"rank": 2, "tier": "MODERATE", "p": 0.41, "threshold": 3.0, "stage": "final"})],
        movers=[make_idea(symbol="ZOMATO", rank=1, kind="MOVER", strength="EXPLOSIVE", x=1.12, or_pct=3.4, p3=1.0, p5=0.41, pre={},
                          day_range=[3.4, 4.9, 7.8], metrics={"or_raw_pct": 3.4}, notes=["news: Board approves fund raise"])],
        pre_stage="final",
        market={"nifty_chg": 0.31, "nifty_or_pct": 0.42, "nifty_x": 0.71, "vix": 14.2, "vix_chg": -1.0, "advancers": 131, "decliners": 83,
                "median_x": 0.472, "day_type": "wide", "hot_share": 0.23,
                "sectors_up": [["Banks", 0.8], ["Auto", 0.5], ["Metals", 0.4]], "sectors_down": [["IT", -0.6], ["Pharma", -0.4], ["FMCG", -0.2]]},
        universe_size=221, n_eligible=198, notes=["no usable live quote for the morning pick(s): XYZ"],
    )
    return OpeningReport(**{**base, **over})


def text_of(report) -> str:
    messages = render_opening_report(report)
    assert_telegram_safe(messages)
    return "\n".join(messages)


# ----------------------------------------------------------------------------------------- rendering
def test_the_report_has_its_sections_in_order_and_every_fact_next_to_its_words():
    text = text_of(make_report())
    order = [text.index(marker) for marker in ("At a glance", "Reference levels", "picks, read again", "New opening movers", "Live record",
                                              "How to read the opening strength", "Universe:")]
    assert order == sorted(order)
    for line in (
        "🌅 <b>MONSTER AI</b> · 09:30 Opening read",
        "💹 ₹182.40 (+1.2% since the open) · gap +0.8% · opening range 1.90% = 0.74× a normal day",
        "🎯 Day range expected 3.6% (80%: 2.3-5.8%) · still ahead 2.4% (80%: 1.5-4.1%)",
        "📊 Chance of a range above 3%: <b>71%</b> · above 5%: 16%",
        "🔼 Above ₹183.77 · stop ₹182.13 · T1 ₹187.05 · T2 ₹190.70",
        "🔽 Below ₹179.21 · stop ₹184.13 · T1 ₹174.47 · T2 ₹170.82",
        "🗂 🔥 <b>STRONG</b> in the 09:12 report · rank #1 · it gave 62% for a range above 3%",
        "💧 est. round-trip cost 0.19% (intraday) · ATR 2.4% a day",
    ):
        assert line in lines([text]), line
    assert "first 15 min of trading" in text and "Nifty +0.31% (opening range 0.42% = 0.71× normal)" in text and "VIX 14.2 (-1.0%)" in text
    assert "breadth 131↑ / 83↓" in text and "Opening intensity: wide" in text and "23% of names are hot or explosive" in text
    assert "📈 Leading: Banks +0.8%, Auto +0.5%, Metals +0.4%" in text and "📉 Lagging: IT -0.6%, Pharma -0.4%, FMCG -0.2%" in text
    assert "🔬 <b>Universe:</b> 198 of 221 passed the liquidity and data gates" in text and "no usable live quote for the morning pick(s): XYZ" in text


def test_every_report_says_what_the_opening_does_and_does_not_tell():
    text = html.unescape(text_of(make_report()))
    assert SHADOW_BANNER in text and html.unescape(EVIDENCE) in text
    assert "Direction: no edge found" in text and "Breakout levels: no edge after costs" in text and "not investment advice" in text
    assert "still ahead" in text and "opening range" in text


def test_the_evidence_quotes_the_model_information_it_ships_with():
    from ranking.opening import MODEL_INFO

    for key in ("top6_hit", "top6_hit_history_only", "top6_hit_random"):
        assert f"{round(MODEL_INFO[key] * 100)}%" in EVIDENCE, key
    assert "93%" not in EVIDENCE and "already past a 3% range" in EVIDENCE  # the headline that was corrected: most of it was on the board already
    assert MODEL_INFO["top6_hit"] > MODEL_INFO["top6_hit_history_only"] > MODEL_INFO["top6_hit_random"]
    assert MODEL_INFO["ahead_top6_hit"] > MODEL_INFO["ahead_top6_hit_random"]


def test_a_probability_is_always_named_with_its_threshold():
    text = text_of(make_report())
    assert "Chance of a range above 3%" in text and "above 5%" in text and "chance of a range above 3%" in text  # the glance legend too
    assert re.findall(r"it gave \d+% for a range above 3%", text) and "score" not in text.lower()


def test_a_name_already_past_a_level_says_so_instead_of_a_trivial_chance():
    text = text_of(make_report(picks=[], movers=[make_idea(kind="MOVER", p3=1.0, p5=0.41, metrics={"or_raw_pct": 3.4})]))
    assert "📊 Range so far 3.4% · <b>already above 3%</b> · chance of above 5%: 41%" in lines([text])
    assert "Chance of a range above 3%" not in text.split("New opening movers")[1] and "100%" not in text.split("New opening movers")[1]
    both = text_of(make_report(picks=[], movers=[make_idea(kind="MOVER", p3=1.0, p5=1.0, metrics={"or_raw_pct": 5.6})]))
    assert "📊 Range so far 5.6% · <b>already above 3% and 5%</b>" in lines([both])
    exactly = text_of(make_report(picks=[], movers=[make_idea(kind="MOVER", p3=0.97, p5=0.3, metrics={"or_raw_pct": 3.0})]))
    assert "Chance of a range above 3%: <b>97%</b>" in exactly  # the range must EXCEED 3%: exactly 3.0 has not passed it


def test_a_band_bounded_by_the_range_so_far_reads_at_least():
    text = lines([text_of(make_report())])
    assert "🎯 Day range expected 4.9% (80%: at least 3.4%, up to 7.8%) · still ahead 2.4% (80%: 1.5-4.1%)" in text  # ZOMATO: low end = what it printed
    assert "🎯 Day range expected 3.6% (80%: 2.3-5.8%) · still ahead 2.4% (80%: 1.5-4.1%)" in text  # TATASTEEL: a real band
    unbounded = text_of(make_report(picks=[make_idea(day_range=[2.0, 2.9, 4.7], metrics={"or_raw_pct": 1.9})], movers=[]))
    assert "(80%: 2.0-4.7%)" in unbounded and "at least" not in unbounded


def test_the_glance_tables_are_compact_monospace_with_both_lists_marked():
    first, second = re.findall(r"<pre>(.*?)</pre>", text_of(make_report()), re.DOTALL)
    assert [row.split() for row in html.unescape(first).split("\n")] == [
        ["SYMBOL", "OR%", "×NORM", ">3%"], ["P", "TATASTEEL", "1.90", "0.74", "71%"], ["P", "HDFCBANK", "0.45", "0.22", "28%"],
        ["M", "ZOMATO", "3.40", "1.12", "yes"]]
    assert [row.split() for row in html.unescape(second).split("\n")] == [
        ["SYMBOL", "ABOVE", "BELOW"], ["P", "TATASTEEL", "183.77", "179.21"], ["P", "HDFCBANK", "183.77", "179.21"], ["M", "ZOMATO", "183.77", "179.21"]]
    assert all(len(line) <= 37 for block in (first, second) for line in html.unescape(block).split("\n"))  # a phone needs no sideways scrolling


def test_words_follow_the_opening_strength():
    hot = text_of(make_report(picks=[make_idea(strength="HOT")], movers=[]))
    quiet = text_of(make_report(picks=[make_idea(strength="QUIET", ahead=[0.8, 1.2, 2.1])], movers=[]))
    normal = text_of(make_report(picks=[make_idea(strength="NORMAL")], movers=[]))
    assert "🔥 <b>HOT</b> opening" in hot and "hot opening: a wider-than-usual day is more likely; which way it goes is not known" in hot
    assert "a wide day is likely" not in hot  # 42% of hot openings went on to a range above 3%: "likely" would overstate it
    assert "😴 <b>QUIET</b> opening" in quiet and "quiet opening: it lowers the expected range ahead only modestly (about 1.2% is still expected)" in quiet
    assert "wide day is likely" not in normal and "quiet opening" not in normal and "🙂 <b>NORMAL</b> opening" in normal
    explosive = text_of(make_report(picks=[], movers=[make_idea(strength="EXPLOSIVE")]))
    assert "💥 <b>EXPLOSIVE</b> opening" in explosive and "big opening: a wide day is likely; which way it goes is not known" in explosive


def test_a_mover_has_no_morning_line_and_a_pick_without_a_chance_still_says_what_the_morning_report_called_it():
    mover = text_of(make_report(picks=[], movers=[make_idea(kind="MOVER", pre={})]))
    assert "in the 09:12 report" not in mover and "picks, read again" not in mover
    no_p = text_of(make_report(picks=[make_idea(pre={"rank": 3, "tier": "WATCH", "p": None, "threshold": 2.5, "stage": "pre"})], movers=[]))
    assert "👀 <b>WATCH</b> in the 08:45 scan · rank #3" in no_p and "it gave" not in no_p


def test_a_day_without_movers_says_nothing_is_forced_and_a_day_without_picks_has_no_picks_section():
    text = text_of(make_report(movers=[], picks=[]))
    assert "No new name passed the gates today: nothing is forced" in text and "read again" not in text and "<pre>" not in text
    assert "🚀 <b>New opening movers</b>" in text


def test_a_section_title_never_ends_a_message_alone():
    many = [make_idea(symbol=f"LONGCO{i}", rank=i, notes=["a note " * 18] * 3) for i in range(1, 13)]
    messages = render_opening_report(make_report(picks=many, movers=[make_idea(kind="MOVER", symbol=f"MOVER{i}", rank=i) for i in range(1, 9)]))
    assert_telegram_safe(messages)
    assert len(messages) > 2
    for message in messages:
        last = message.rstrip().split("\n")[-1]
        assert not last.startswith(("🔎", "🚀")), last
    text = "\n".join(messages)
    assert all(f"LONGCO{i}" in text for i in range(1, 13)) and all(f"MOVER{i}" in text for i in range(1, 9))


# ------------------------------------------------------------------------------------------- the record
def test_without_a_record_the_report_says_the_figures_come_from_research():
    text = text_of(make_report(record={}))
    assert "Live record:</b> none judged yet" in text and f"{MODEL_INFO['sessions']} sessions of research" in text


def test_a_record_is_shown_with_its_sample_sizes_interval_and_verdict():
    record = {
        "model": {"days": 24, "n": 5104, "n_decided": 118, "mean_p3": 0.281, "hit3": 0.274, "top_n": 6, "top_hit": 0.88, "top_ci": [0.81, 0.93],
                  "validated": True},
        "PICK": {"n": 140, "n_open": 128, "n_decided": 12, "hit3": 0.46, "mean_p3": 0.44, "mean_range_pct": 3.1},
        "MOVER": {"n": 150, "n_open": 96, "n_decided": 54, "hit3": 0.61, "mean_p3": 0.58, "mean_range_pct": 4.4},
        "plan": {"n": 120, "t1_rate": 0.23, "stop_rate": 0.24, "mean_net_pct": -0.06, "net_ci": [-0.19, 0.07]},
    }
    got = lines([text_of(make_report(record=record))])
    assert ("📚 <b>Live record:</b> 24 session(s), 5,104 names judged (+118 already past 3%: not a test) · stated 28% for a range above 3%, "
            "happened 27% · its top 6 made one on 88% [81%-93%] · ✅ record passes the gate") in got
    assert "   ▫️ 09:12 picks read again: 140 judged · 128 not already past 3%: range above 3% on 46% (stated 44%) · mean range 3.10%" in got
    assert "   ▫️ new movers: 150 judged · 96 not already past 3%: range above 3% on 61% (stated 58%) · mean range 4.40%" in got
    assert "   ▫️ levels as a plan: 120 trades · first target 23% · stopped 24% · mean -0.06% net [-0.19% to +0.07%]" in got


def test_a_list_whose_names_were_all_past_3_percent_says_so_instead_of_a_rate():
    record = {"model": {}, "MOVER": {"n": 9, "n_open": 0, "n_decided": 9, "hit3": None, "mean_p3": None, "mean_range_pct": 5.2}}
    got = lines([text_of(make_report(record=record))])
    assert "   ▫️ new movers: 9 judged · all already past 3% at the scan · mean range 5.20%" in got
    assert not any("range above 3% on n/a" in line for line in got)


def test_a_record_that_has_not_passed_says_what_it_still_needs():
    short = {"model": {"days": 6, "n": 1200, "mean_p3": 0.28, "hit3": 0.30, "top_n": 6, "top_hit": 0.9, "top_ci": None, "validated": False}}
    text = text_of(make_report(record=short))
    assert "🧪 not validated: needs 20 judged sessions (has 6)" in text and "[" not in text.split("Live record")[1].split("\n")[0]
    off = {"model": {"days": 30, "n": 6000, "mean_p3": 0.28, "hit3": 0.45, "gap": 0.17, "top_n": 6, "top_hit": 0.9, "top_ci": [0.8, 0.95], "validated": False}}
    assert "the stated chance is off by 17 points" in text_of(make_report(record=off))


# ---------------------------------------------------------------------------------------- bad input
def test_untrusted_text_is_escaped_everywhere():
    hostile = make_idea(symbol="M&M<b>X", name="<script>alert(1)</script>", sector="a & b", notes=["<i>note</i> & more"],
                        pre={"rank": 1, "tier": "<b>STRONG</b>", "p": 0.5, "threshold": 3.0, "stage": "<x>"})
    text = text_of(make_report(picks=[hostile], movers=[], source="<b>NSE</b> & co", notes=["x < y & z"],
                               market={"sectors_up": [["R&D <b>", 1.0]] * 3, "sectors_down": [["a", -1.0]] * 3, "day_type": "wide", "median_x": 0.5}))
    assert "<script>" not in text and "&lt;script&gt;" in text and "M&amp;M" in text


def test_missing_and_non_finite_figures_render_as_na_and_are_never_invented():
    broken = make_idea(ltp=float("nan"), move_pct=None, gap_pct=float("inf"), or_pct="x", x=None, p3=None, p5=float("nan"), day_range=[1, 2],
                       ahead=None, levels={"trigger_long": float("nan")}, est_cost_pct=None, atr_pct=None, strength="", notes=[])
    text = text_of(make_report(picks=[broken], movers=[], market={}, record={}))
    assert "n/a" in text and not _BAD_NUMBER.search(text)
    assert "Reference levels n/a" in text and "Day range expected n/a" in text and "Chance of a range above 3%: <b>n/a</b>" in text


def test_odd_values_never_break_the_renderer():
    odd = make_idea(symbol=None, name=12345, sector=["x"], notes="just a string", pre="not a map", levels="oops", strength=7, rank=0)
    text = text_of(make_report(picks=[odd, "not an idea", None], movers=[object()], notes="not a list", market="not a map", record=["x"],
                               elapsed_min="soon", universe_size="many"))
    assert "UNKNOWN" in text


def test_a_report_without_a_trade_date_is_refused_and_one_that_went_through_json_renders_alike():
    with pytest.raises(ValueError, match="trade_date"):
        render_opening_report(make_report(trade_date=""))
    report = make_report(record={"model": {"days": 3, "n": 600, "mean_p3": 0.3, "hit3": 0.3, "top_n": 6, "top_hit": 0.9, "validated": False}})
    assert render_opening_report(OpeningReport.from_dict(report.to_dict())) == render_opening_report(report)
    assert render_opening_report(report.to_dict()) == render_opening_report(report)  # the frozen payload itself


def test_a_late_report_says_so_at_the_top_and_an_on_time_one_does_not():
    head = text_of(make_report(late=True)).split("━")[0]
    assert LATE_BANNER in html.unescape(head) and LATE_BANNER not in text_of(make_report())


def test_the_renderer_survives_random_garbage():
    rng = random.Random(7)
    junk = [None, float("nan"), float("inf"), -math.inf, 0, -1, 1e308, "", "x" * 500, "<b>&", ["a"], {"k": 1}, object(), True, 3.14, "\ud800"]

    def pick():
        return rng.choice(junk)

    for _ in range(150):
        idea = {f.name: pick() for f in dataclasses.fields(OpeningIdea) if rng.random() < 0.8}
        report = {f.name: pick() for f in dataclasses.fields(OpeningReport) if rng.random() < 0.7}
        report.update(trade_date="2026-11-02", picks=[idea, pick()], movers=[idea], market={"nifty_chg": pick(), "sectors_up": [pick()]})
        text = "\n".join(render_opening_report(report))
        assert len(text) > 0
        assert_telegram_safe(render_opening_report(report))


def test_the_legend_matches_the_stored_history():
    from ranking.opening import STRENGTH_HISTORY

    text = html.unescape(text_of(make_report()))
    for name, (_n, rate, ahead) in STRENGTH_HISTORY.items():
        assert f"{name.title()} (" in text and f"{round(rate * 100)}% ended with a range above 3%" in text
        assert f"median {ahead:.1f}% range still ahead" in text
    assert "partly already on the board" in text
    assert renderer.STRENGTH_HISTORY is STRENGTH_HISTORY


# ------------------------------------------------------------------------------------------- delivery
@pytest.fixture()
def cfg(settings):
    return dataclasses.replace(settings, telegram_bot_token=TOKEN, telegram_chat_id="1001", telegram_alert_chat_id="2002", telegram_max_attempts=3)


@pytest.fixture()
def fake_client():
    return FakeClient()


@pytest.fixture()
def svc(cfg, store, fake_clock, fake_client):
    return NotificationService(cfg, store, client=fake_client, clock=fake_clock)


def test_the_report_is_sent_once_and_recorded_in_the_ledger(svc, fake_client, store, fake_clock):
    set_ist(fake_clock, "09:30:40")
    expected = render_opening_report(make_report())
    result = svc.send_opening(make_report())
    assert (result.status, result.reason, result.message_ids) == ("sent", None, [101 + n for n in range(len(expected))])
    assert [text for _, text, _ in fake_client.sent] == expected
    chat, text, options = fake_client.sent[0]
    assert chat == "1001" and text.startswith("🌅 <b>MONSTER AI</b> · 09:30 Opening read")
    assert options == {"deadline": at_ist("2026-11-02", "09:45")}
    row = store.get_delivery("telegram", KEY)
    assert (row["status"], row["kind"], row["trade_date"]) == ("sent", "opening_report", "2026-11-02")


def test_a_second_send_is_a_duplicate_even_when_forced(svc, fake_client, fake_clock):
    set_ist(fake_clock, "09:30:40")
    svc.send_opening(make_report())
    sent_once = len(fake_client.sent)
    again = svc.send_opening(make_report(), force=True)
    assert (again.status, again.reason, again.attempts) == ("duplicate", "already_sent", 0) and len(fake_client.sent) == sent_once


@pytest.mark.parametrize("hms, extra, expected", [("09:30:40", 0.0, "sent"), ("09:45:00", 0.0, "sent"), ("09:45:00", 0.001, "withheld"),
                                                  ("10:05:00", 0.0, "withheld")])
def test_the_report_deadline_boundaries(svc, fake_client, store, fake_clock, hms, extra, expected):
    set_ist(fake_clock, hms, extra_seconds=extra)
    result = svc.send_opening(make_report())
    assert result.status == expected
    if expected == "withheld":
        assert result.reason == "deadline_missed" and fake_client.sent == []
        (alert,) = store.pending_alerts()
        assert alert["code"] == "opening_deadline_missed" and "09:45:00 IST deadline" in alert["message"]
        assert store.get_delivery("telegram", KEY)["status"] == "withheld"
    else:
        assert store.pending_alerts() == []


def test_a_forced_send_goes_out_after_the_deadline_without_a_send_by_time(svc, fake_client, fake_clock):
    set_ist(fake_clock, "10:20")
    assert svc.send_opening(make_report()).status == "withheld"
    assert svc.send_opening(make_report(late=True), force=True).status == "sent"
    chat, text, options = fake_client.sent[0]
    assert options == {} and LATE_BANNER in html.unescape(text)


def test_the_report_can_be_switched_off(cfg, store, fake_clock, fake_client):
    off = NotificationService(dataclasses.replace(cfg, send_opening_report=False), store, client=fake_client, clock=fake_clock)
    set_ist(fake_clock, "09:30:40")
    result = off.send_opening(make_report())
    assert (result.status, result.reason) == ("withheld", "opening_disabled") and fake_client.sent == []
    assert store.get_delivery("telegram", KEY)["error"] == "opening_disabled"


def test_unconfigured_telegram_withholds_and_never_raises(settings, store, fake_clock):
    quiet = FakeClient(configured=False)
    service = NotificationService(settings, store, client=quiet, clock=fake_clock)
    set_ist(fake_clock, "09:30:40")
    result = service.send_opening(make_report())
    assert (result.status, result.reason) == ("withheld", "telegram_not_configured") and quiet.sent == []


@pytest.mark.parametrize("trade_date", ["", "2026-13-45", "yesterday", None])
def test_a_report_with_an_invalid_date_is_rejected_without_a_ledger_row(svc, fake_client, store, trade_date):
    result = svc.send_opening(make_report(trade_date=trade_date))
    assert result.status == "failed" and "invalid trade_date" in result.reason
    assert store.table_counts()["deliveries"] == 0 and store.pending_alerts() == []
    assert len(fake_client.sent) == 1 and fake_client.sent[0][0] == "2002" and "report_invalid" in fake_client.sent[0][1]


def test_a_render_error_becomes_a_failed_result_not_an_exception(svc, fake_client, store, fake_clock, monkeypatch):
    def boom(_report):
        raise RuntimeError(f"cannot render {TOKEN}")

    monkeypatch.setattr(renderer, "render_opening_report", boom)
    set_ist(fake_clock, "09:30:40")
    result = svc.send_opening(make_report())
    assert result.status == "failed" and result.reason.startswith("render_error") and TOKEN not in result.reason
    assert store.get_delivery("telegram", KEY)["status"] == "failed"
    assert len(fake_client.sent) == 1 and fake_client.sent[0][0] == "2002" and "report_render_failed" in fake_client.sent[0][1]


def test_a_failed_delivery_is_retried_by_the_next_call(cfg, store, fake_clock):
    client = FakeClient(script=[RuntimeError("telegram is down")])
    service = NotificationService(cfg, store, client=client, clock=fake_clock)
    set_ist(fake_clock, "09:30:40")
    assert service.send_opening(make_report()).status == "failed" and store.get_delivery("telegram", KEY)["status"] == "failed"
    assert service.send_opening(make_report()).status == "sent" and store.get_delivery("telegram", KEY)["status"] == "sent"


def test_the_report_reaches_the_extra_chats_after_the_main_one_and_a_blocked_chat_does_not_stop_it(cfg, store, fake_clock):
    from tests.test_notifications_service import RoutedClient

    client = RoutedClient(broken={"3003"})
    service = NotificationService(dataclasses.replace(cfg, telegram_extra_chat_ids=("3003", "4004")), store, client=client, clock=fake_clock)
    set_ist(fake_clock, "09:30:40")
    expected = render_opening_report(make_report())
    assert service.send_opening(make_report()).status == "sent"
    texts = [(chat, text) for chat, text, _ in client.sent]
    assert [chat for chat, _ in texts] == ["1001"] * len(expected) + ["4004"] * len(expected) and client.photos == []  # no images for this report
    assert [text for _, text in texts[len(expected):]] == expected
    assert store.get_delivery("telegram", f"{KEY}@4004")["kind"] == "opening_report_extra"
    assert store.get_delivery("telegram", f"{KEY}@3003")["status"] == "failed"
    assert service.send_opening(make_report(), force=True).status == "duplicate"
