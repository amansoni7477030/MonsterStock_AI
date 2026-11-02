"""The closing-hour report as Telegram messages (rendering) and its delivery by the notification service."""

from __future__ import annotations

import dataclasses
import html
import math

import pytest

from app.clock import at_ist
from notifications import ideas as idea_messages
from notifications.ideas import HEURISTIC_BANNER, LATE_BANNER, RECORD_BANNER, render_ideas_report
from notifications.service import NotificationService
from ranking.types import Catalyst, IdeasReport, TradeIdea
from tests.test_notifications_report import _BAD_NUMBER, assert_telegram_safe, lines
from tests.test_notifications_service import TOKEN, FakeClient, set_ist

KEY = "ideas:2026-11-02"


def make_idea(**over) -> TradeIdea:
    base = dict(
        symbol="DEMOCO", rank=1, horizon="BTST", side="LONG", setup="strong close on volume", score=78.1, grade="A", ltp=1000.0,
        name="Demo Company Ltd", sector="Demo Sector", is_fno=True, change_pct=2.6, entry=1000.0, max_entry=1004.0, stop=960.0,
        target1=1040.0, target2=1080.0, risk_pct=4.0, reward1_pct=4.0, reward2_pct=8.0, rr1=1.0, rr2=2.0,
        hold="sell in the next session: at the first target or at the stop, otherwise at the close; T2 only for a part you keep",
        reasons=["closing near the day's high", "volume 1.8x its 20-day average"], cautions=["RSI 70: already stretched"],
        turnover_cr=250.0, est_cost_pct=0.3,
    )
    return TradeIdea(**{**base, **over})


def make_watch(**over) -> TradeIdea:
    return make_idea(
        symbol="WATCHCO", horizon="WATCH", side="BOTH", setup="tight range under the 20-day high", score=64.0, grade="B", ltp=500.0,
        entry=None, max_entry=None, stop=None, target1=None, target2=None, risk_pct=None, reward1_pct=None, reward2_pct=None, rr1=None,
        rr2=None, trigger_long=505.0, trigger_short=495.0, est_cost_pct=None, turnover_cr=None, cautions=[],
        metrics={"long_stop": 498.0, "long_target1": 512.0, "long_target2": 518.0, "short_stop": 502.0, "short_target1": 488.0,
                 "short_target2": 482.0},
        hold="levels for tomorrow's session: react only if one of them breaks", **over,
    )


def make_report(**over) -> IdeasReport:
    ideas = [
        make_idea(),
        make_idea(symbol="SWINGCO", horizon="SWING", setup="pullback to the 20-day average in an uptrend", score=66.0, grade="B",
                  hold="hold up to 5 sessions: sell at the first target or at the stop, otherwise at the 5th close"),
        make_idea(symbol="SHORTCO", side="SHORT", setup="weak close on volume", score=70.2, grade="B", stop=1040.0, target1=960.0,
                  target2=920.0, max_entry=996.0),
        make_watch(),
    ]
    base = dict(
        trade_date="2026-11-02", snapshot_ist="2026-11-02 14:31:00 IST", entry_by="15:15",
        source="NSE (SECURITIES IN F&O): 180 of 185 names with a complete live row", ideas=ideas,
        market={"nifty_chg": 0.4, "vix": 14.0, "vix_chg": -1.4, "advancers": 120, "decliners": 60, "regime": "bullish"},
        universe_size=185, n_eligible=150,
    )
    return IdeasReport(**{**base, **over})


def text_of(report) -> str:
    messages = render_ideas_report(report)
    assert_telegram_safe(messages)
    return "\n".join(messages)


# ----------------------------------------------------------------------------------------- rendering
def test_a_report_shows_the_four_lists_in_order_with_their_levels():
    text = text_of(make_report())
    order = [text.index(marker) for marker in ("BTST - buy before the close", "Swing - hold for 2-5 sessions", "Bearish setups - F&amp;O only",
                                              "Tomorrow's intraday watchlist")]
    assert order == sorted(order)
    for line in (
        "🌙 <b>MONSTER AI</b> · Closing-hour trade ideas",
        "🎯 Entry about ₹1,000.00 (do not chase beyond ₹1,004.00) · 🛑 Stop ₹960.00 (risk 4.00%)",
        "🏁 T1 ₹1,040.00 (+4.00%) · T2 ₹1,080.00 (+8.00%) · reward to risk 1.0 / 2.0",
        "🔼 Buy above ₹505.00 · stop ₹498.00 · T1 ₹512.00 · T2 ₹518.00",
        "🔽 Sell below ₹495.00 · stop ₹502.00 · T1 ₹488.00 · T2 ₹482.00",
        "💧 est. round-trip cost 0.30% (delivery-style) · turnover ₹250 cr a day",
    ):
        assert line in lines([text]), line
    assert "⏳ enter before 15:15 IST" in text and "Nifty +0.40%" in text and "VIX 14.0 (-1.4%)" in text and "breadth 120↑ / 60↓" in text
    assert "🔬 <b>Universe:</b> 150 of 185 passed the liquidity and data gates" in text


def test_a_bearish_idea_is_tagged_and_every_idea_keeps_its_own_hold_rule():
    got = lines([text_of(make_report())])
    assert any("#1 SHORTCO" in line and "BTST short" in line for line in got)
    assert any(line.startswith("⏱ hold up to 5 sessions") for line in got)
    assert any(line.startswith("⏱ sell in the next session") for line in got)


def test_the_score_is_a_score_and_never_drawn_or_worded_as_a_probability():
    text = text_of(make_report())
    assert "score 78.1" in text and HEURISTIC_BANNER in html.unescape(text)
    assert not any(bar in text for bar in "█▓▒░▮") and "chance" not in text.lower() and "probability" not in text.replace(
        "it is NOT a probability", "")


def test_missing_and_non_finite_figures_render_as_na_and_are_never_invented():
    broken = make_idea(entry=float("nan"), stop=None, target1=float("inf"), target2=-math.inf, risk_pct="x", rr1=None, score=None,
                       ltp=float("nan"), change_pct=None, est_cost_pct=None, turnover_cr=None)
    text = text_of(make_report(ideas=[broken]))
    assert "n/a" in text and not _BAD_NUMBER.search(text)
    assert "score n/a" in text and "Stop n/a" in text and "Entry about n/a" in text


def test_untrusted_text_is_escaped_everywhere():
    hostile = make_idea(symbol="A<B>&C", name="<script>alert(1)</script>", sector="R&D <b>", setup="x < y & z",
                        reasons=["<i>bold</i> & more"], cautions=["a > b"], hold="<u>hold</u>",
                        catalysts=[Catalyst(headline="<b>Fake</b> headline & more", source="Fake <Wire>", url="https://example.com/a?b=1&c=2")])
    report = make_report(ideas=[hostile], notes=["note <x> & y"], source="NSE <live> & more", entry_by="15:15<")
    text = text_of(report)
    assert "A&lt;B&gt;&amp;C" in text and "&lt;script&gt;" in text and "<script>" not in text
    assert "note &lt;x&gt; &amp; y" in text and "NSE &lt;live&gt; &amp; more" in text


def test_a_late_report_says_so_at_the_top_and_an_on_time_one_does_not():
    late = render_ideas_report(make_report(late=True))
    assert f"⏰ <b>{LATE_BANNER}</b>" in lines(late[:1])[:6]
    assert LATE_BANNER not in text_of(make_report())


def test_the_banner_follows_the_records_it_stands_behind():
    assert HEURISTIC_BANNER in html.unescape(text_of(make_report()))
    passed = {"BTST": {"n": 62, "days": 21, "validated": True, "win_rate": 0.58, "win_ci": [0.52, 0.64], "mean_net_pct": 0.42,
                       "mean_net_ci": [0.1, 0.7], "t1_rate": 0.4, "stop_rate": 0.3}}
    text = html.unescape(text_of(make_report(record=passed)))
    assert RECORD_BANNER in text and HEURISTIC_BANNER not in text


def test_every_list_shows_its_own_record_with_sample_sizes_and_intervals():
    record = {
        "BTST": {"n": 62, "days": 21, "validated": True, "win_rate": 0.58, "win_ci": [0.52, 0.64], "mean_net_pct": 0.42,
                 "mean_net_ci": [0.1, 0.7], "t1_rate": 0.4, "stop_rate": 0.3},
        "SWING": {"n": 0, "days": 0, "pending": 2, "validated": False},
        "SHORT": {"n": 12, "days": 7, "validated": False, "win_rate": 0.5, "win_ci": [0.2, 0.8], "mean_net_pct": -0.1,
                  "mean_net_ci": [-1.0, 0.8], "t1_rate": 0.25, "stop_rate": 0.4},
        "WATCH": {"n": 30, "days": 15, "validated": False, "triggered_rate": 0.6, "followed_through_rate": 0.55},
    }
    got = lines([text_of(make_report(record=record))])
    assert ("📚 Live record: 62 judged over 21 days · positive after costs 58% [52 to 64] · mean +0.42% net [+0.10 to +0.70] · "
            "first target 40% · stopped 30% · ✅ record passes the gate") in got
    assert "📚 Live record: none judged yet (2 still inside their horizon): heuristic, treat as an experiment" in got
    short = next(line for line in got if line.startswith("📚 Live record: 12 judged over 7 days"))
    assert "mean -0.10% net [-1.00 to +0.80]" in short and "🧪 not validated: needs 60 judged ideas over 20 days (has 12 over 7)" in short
    assert "📚 Live record: 30 judged over 15 days · a level broke on 60% of them · closed beyond it on 55% of the breaks" in got


def test_without_any_record_every_list_says_so():
    got = lines([text_of(make_report())])
    assert got.count("📚 Live record: none judged yet: heuristic, treat as an experiment") == 4


def test_an_empty_list_says_nothing_passed_the_gates():
    text = text_of(make_report(ideas=[make_idea()]))
    assert text.count("🛑 Nothing passed the gates for this list today.") == 3


def test_a_report_without_ideas_is_still_a_complete_message():
    messages = render_ideas_report(make_report(ideas=[], notes=["no BTST or swing setup passed the gates today: nothing is forced"]))
    assert_telegram_safe(messages)
    text = "\n".join(messages)
    assert text.count("🛑 Nothing passed the gates") == 4 and "nothing is forced" in text


def test_a_long_report_is_split_into_balanced_messages_without_losing_an_idea():
    ideas = [make_idea(symbol=f"LONGCO{i}", rank=i, reasons=["a reason " * 20] * 4, cautions=["a caution " * 20] * 3) for i in range(1, 31)]
    messages = render_ideas_report(make_report(ideas=ideas))
    assert_telegram_safe(messages)
    assert len(messages) > 1
    text = "\n".join(messages)
    assert all(f"LONGCO{i}" in text for i in range(1, 31))


def test_a_report_that_went_through_json_renders_like_the_original():
    report = make_report(record={"BTST": {"n": 3, "days": 2, "validated": False, "win_rate": 1.0, "win_ci": None, "mean_net_pct": 1.0,
                                          "mean_net_ci": None, "t1_rate": 1.0, "stop_rate": 0.0}})
    assert render_ideas_report(IdeasReport.from_dict(report.to_dict())) == render_ideas_report(report)
    assert render_ideas_report(report.to_dict()) == render_ideas_report(report)  # the frozen payload itself


def test_a_report_without_a_trade_date_is_refused():
    with pytest.raises(ValueError, match="trade_date"):
        render_ideas_report(make_report(trade_date=""))


def test_odd_values_never_break_the_renderer():
    odd = make_idea(symbol=None, name=12345, sector=["x"], reasons="just a string", cautions=None, catalysts=[object()], grade="",
                    horizon="BTST", side="SIDEWAYS", rank=0)
    text = text_of(make_report(ideas=[odd, "not an idea", None], notes="not a list", market="not a map", record=["x"]))
    assert "UNKNOWN" in text


# ------------------------------------------------------------------------------------------- delivery
@pytest.fixture()
def cfg(settings):
    return dataclasses.replace(settings, telegram_bot_token=TOKEN, telegram_chat_id="1001", telegram_alert_chat_id="2002",
                               telegram_max_attempts=3)


@pytest.fixture()
def fake_client():
    return FakeClient()


@pytest.fixture()
def svc(cfg, store, fake_clock, fake_client):
    return NotificationService(cfg, store, client=fake_client, clock=fake_clock)


def test_the_report_is_sent_once_and_recorded_in_the_ledger(svc, fake_client, store, fake_clock):
    set_ist(fake_clock, "14:31")
    expected = render_ideas_report(make_report())
    result = svc.send_ideas(make_report())
    assert (result.status, result.reason, result.message_ids) == ("sent", None, [101 + n for n in range(len(expected))])
    assert [text for _, text, _ in fake_client.sent] == expected
    chat, text, options = fake_client.sent[0]
    assert chat == "1001" and text.startswith("🌙 <b>MONSTER AI</b> · Closing-hour trade ideas")
    assert options == {"deadline": at_ist("2026-11-02", "15:05")}
    row = store.get_delivery("telegram", KEY)
    assert (row["status"], row["kind"], row["trade_date"]) == ("sent", "ideas_report", "2026-11-02")


def test_a_second_send_is_a_duplicate_even_when_forced(svc, fake_client, fake_clock):
    set_ist(fake_clock, "14:31")
    svc.send_ideas(make_report())
    sent_once = len(fake_client.sent)
    again = svc.send_ideas(make_report(), force=True)
    assert (again.status, again.reason, again.attempts) == ("duplicate", "already_sent", 0)
    assert len(fake_client.sent) == sent_once


@pytest.mark.parametrize("hms, extra, expected", [("14:31:00", 0.0, "sent"), ("15:05:00", 0.0, "sent"), ("15:05:00", 0.001, "withheld"),
                                                  ("15:10:00", 0.0, "withheld")])
def test_the_report_deadline_boundaries(svc, fake_client, store, fake_clock, hms, extra, expected):
    set_ist(fake_clock, hms, extra_seconds=extra)
    result = svc.send_ideas(make_report())
    assert result.status == expected
    if expected == "withheld":
        assert result.reason == "deadline_missed" and fake_client.sent == []
        (alert,) = store.pending_alerts()
        assert alert["code"] == "ideas_deadline_missed" and "15:05:00 IST deadline" in alert["message"]
        assert store.get_delivery("telegram", KEY)["status"] == "withheld"
    else:
        assert store.pending_alerts() == []


def test_a_forced_send_goes_out_after_the_deadline_without_a_send_by_time(svc, fake_client, fake_clock):
    set_ist(fake_clock, "15:20")
    assert svc.send_ideas(make_report()).status == "withheld"
    assert svc.send_ideas(make_report(late=True), force=True).status == "sent"
    chat, text, options = fake_client.sent[0]
    assert options == {} and LATE_BANNER in html.unescape(text)


def test_the_report_can_be_switched_off(cfg, store, fake_clock, fake_client):
    off = NotificationService(dataclasses.replace(cfg, send_ideas_report=False), store, client=fake_client, clock=fake_clock)
    set_ist(fake_clock, "14:31")
    result = off.send_ideas(make_report())
    assert (result.status, result.reason) == ("withheld", "ideas_disabled") and fake_client.sent == []
    assert store.get_delivery("telegram", KEY)["error"] == "ideas_disabled"


def test_unconfigured_telegram_withholds_and_never_raises(settings, store, fake_clock):
    quiet = FakeClient(configured=False)
    service = NotificationService(settings, store, client=quiet, clock=fake_clock)
    set_ist(fake_clock, "14:31")
    result = service.send_ideas(make_report())
    assert (result.status, result.reason) == ("withheld", "telegram_not_configured") and quiet.sent == []


@pytest.mark.parametrize("trade_date", ["", "2026-13-45", "yesterday", None])
def test_a_report_with_an_invalid_date_is_rejected_without_a_ledger_row(svc, fake_client, store, trade_date):
    result = svc.send_ideas(make_report(trade_date=trade_date))
    assert result.status == "failed" and "invalid trade_date" in result.reason
    assert store.table_counts()["deliveries"] == 0
    assert store.pending_alerts() == []  # the alert was delivered at once, to the alert chat only
    assert len(fake_client.sent) == 1 and fake_client.sent[0][0] == "2002" and "report_invalid" in fake_client.sent[0][1]


def test_a_render_error_becomes_a_failed_result_not_an_exception(svc, fake_client, store, fake_clock, monkeypatch):
    def boom(_report):
        raise RuntimeError(f"cannot render {TOKEN}")

    monkeypatch.setattr(idea_messages, "render_ideas_report", boom)
    set_ist(fake_clock, "14:31")
    result = svc.send_ideas(make_report())
    assert result.status == "failed" and result.reason.startswith("render_error") and TOKEN not in result.reason
    assert store.get_delivery("telegram", KEY)["status"] == "failed"
    assert len(fake_client.sent) == 1 and fake_client.sent[0][0] == "2002" and "report_render_failed" in fake_client.sent[0][1]


def test_a_failed_delivery_is_retried_by_the_next_call(cfg, store, fake_clock):
    client = FakeClient(script=[RuntimeError("telegram is down")])
    service = NotificationService(cfg, store, client=client, clock=fake_clock)
    set_ist(fake_clock, "14:31")
    assert service.send_ideas(make_report()).status == "failed" and store.get_delivery("telegram", KEY)["status"] == "failed"
    assert service.send_ideas(make_report()).status == "sent" and store.get_delivery("telegram", KEY)["status"] == "sent"
