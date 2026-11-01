"""Service tests: delivery ledger, deadline logic on FakeClock, failure handling, alert queue. No network."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import random
import sqlite3
from datetime import timedelta

import pytest
import requests

from app.clock import at_ist
from app.errors import DataUnavailable
from notifications import report as report_module
from notifications.service import LATE_PREFIX, DeliveryResult, NotificationService
from notifications.telegram import DeadlineExceeded, TelegramClient
from ranking.types import Candidate, EveningRow, EveningSummary, ReportContext

TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"
LEAKY = f"HTTPSConnectionPool(host='api.telegram.org'): Max retries exceeded with url: /bot{TOKEN}/sendMessage"
KEY = "final_report:2026-11-02"


def make_ctx(**over):
    cand = Candidate(
        symbol="FAKECO",
        rank=1,
        tier="WATCH",
        p={"hit_3": 0.4, "up": 0.5, "down": 0.5, "unusual": 0.3},
        primary_threshold=3.0,
        range={"q10": 1.0, "q50": 2.0, "q90": 4.0, "exc_q50": 1.5, "exc_q90": 3.0},
        name="Fake Company Ltd",
    )
    base = dict(
        trade_date="2026-11-02",
        stage="final",
        snapshot_ist="2026-11-02 09:12:00 IST",
        model_version="fake-model-0.1",
        model_status="validated",
        candidates=[cand],
        universe_size=100,
        n_eligible=80,
    )
    base.update(over)
    return ReportContext(**base)


def make_summary(**over):
    base = dict(
        trade_date="2026-11-02",
        stage="final",
        primary_threshold=3.0,
        model_version="fake-model-0.1",
        rows=[EveningRow("FAKECO", 1, "WATCH", 0.4, 3.5, 2.0, 1.1, True)],
        n=1,
        model_hit_rate=1.0,
        model_hit_ci=(0.2, 1.0),
    )
    base.update(over)
    return EveningSummary(**base)


class FakeClient:
    """Duck-typed TelegramClient: records sends; ``script`` entries are None (ok) or an exception to raise."""

    def __init__(self, *, configured=True, script=None):
        self.configured = configured
        self.script = list(script or [])
        self.sent = []
        self.last_attempts = 1
        self._next_id = 100

    def send_message(self, chat_id, text, **options):
        step = self.script.pop(0) if self.script else None
        if step is not None:
            raise step
        self.sent.append((chat_id, text, options))
        self._next_id += 1
        return [self._next_id]


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.headers = {}

    def json(self):
        return self._body


def ok(message_id=1):
    return FakeResponse(200, {"ok": True, "result": {"message_id": message_id}})


def err(code, description):
    return FakeResponse(code, {"ok": False, "error_code": code, "description": description})


class FakeSession:
    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def post(self, url, json=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.script.pop(0) if self.script else ok(len(self.calls))
        if isinstance(item, BaseException):
            raise item
        return item


def set_ist(clock, hms, *, extra_seconds=0.0):
    clock.set(at_ist("2026-11-02", hms) + timedelta(seconds=extra_seconds))


@pytest.fixture()
def cfg(settings):
    return dataclasses.replace(
        settings,
        telegram_bot_token=TOKEN,
        telegram_chat_id="1001",
        telegram_alert_chat_id="2002",
        telegram_max_attempts=3,
    )


@pytest.fixture()
def fake_client():
    return FakeClient()


@pytest.fixture()
def svc(cfg, store, fake_clock, fake_client):
    return NotificationService(cfg, store, client=fake_client, clock=fake_clock)


def real_service(cfg, store, fake_clock, *script):
    session = FakeSession(*script)
    client = TelegramClient(cfg, session=session, clock=fake_clock, rng=random.Random(1))
    return NotificationService(cfg, store, client=client, clock=fake_clock), session


def test_delivery_result_defaults_and_ok_flag():
    result = DeliveryResult("sent")
    assert result.reason is None and result.message_ids == [] and result.attempts == 0 and result.ok
    assert DeliveryResult("duplicate").ok
    assert not DeliveryResult("failed").ok and not DeliveryResult("withheld").ok


# ------------------------------------------------------------------------------- final report / ledger
def test_final_report_is_sent_and_recorded_in_the_ledger(svc, fake_client, store):
    result = svc.send_report(make_ctx())
    assert (result.status, result.reason, result.message_ids, result.attempts) == ("sent", None, [101], 1)
    chat, text, options = fake_client.sent[0]
    assert chat == "1001" and text.startswith("🚀 <b>MONSTER AI</b> · NSE large-move scan")
    assert options == {"deadline": at_ist("2026-11-02", "09:14:30")}
    row = store.get_delivery("telegram", KEY)
    assert (row["status"], row["kind"], row["trade_date"], row["attempts"]) == ("sent", "final_report", "2026-11-02", 1)
    assert json.loads(row["message_ids"]) == [101] and row["error"] is None and row["sent_at"]
    assert row["body_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_second_call_is_a_duplicate_even_when_forced(svc, fake_client):
    first = svc.send_report(make_ctx())
    second = svc.send_report(make_ctx())
    assert (second.status, second.reason, second.message_ids, second.attempts) == ("duplicate", "already_sent", [101], 0)
    assert second.message_ids == first.message_ids and svc.send_report(make_ctx(), force=True).status == "duplicate"
    assert len(fake_client.sent) == 1


def test_failed_delivery_is_recorded_alerted_deduplicated_and_retried(svc, fake_client, store):
    down = DataUnavailable("telegram", "gave up after 3 attempts: request timed out", retryable=True)
    fake_client.script = [down, down]
    fake_client.last_attempts = 3
    result = svc.send_report(make_ctx())
    assert result.status == "failed" and result.attempts == 3 and "gave up after 3 attempts" in result.reason
    row = store.get_delivery("telegram", KEY)
    assert (row["status"], row["attempts"]) == ("failed", 3) and "gave up" in row["error"]
    alerts = store.pending_alerts()
    assert [a["code"] for a in alerts] == ["telegram_delivery_failed"]
    assert "final_report for 2026-11-02 was not delivered" in alerts[0]["message"]
    assert fake_client.sent == []  # telegram is down: no immediate alert delivery attempted

    assert svc.send_report(make_ctx()).status == "failed"
    assert len(store.pending_alerts()) == 1  # the repeat failure within the hour is deduplicated

    fake_client.last_attempts = 1
    retried = svc.send_report(make_ctx())
    row = store.get_delivery("telegram", KEY)
    assert retried.status == "sent" and (row["status"], row["attempts"], row["error"]) == ("sent", 7, None)


def test_pending_rows_left_by_a_crash_are_retried(svc, store):
    store.save_delivery("telegram", "final_report", KEY, "pending", trade_date="2026-11-02")
    assert svc.send_report(make_ctx()).status == "sent"
    assert store.get_delivery("telegram", KEY)["status"] == "sent"


def test_multi_message_reports_are_sent_in_order_and_hashed_together(svc, fake_client, store, monkeypatch):
    monkeypatch.setattr(report_module, "render_final_report", lambda ctx: ["part one", "part two"])
    result = svc.send_report(make_ctx())
    assert result.status == "sent" and result.message_ids == [101, 102] and result.attempts == 2
    assert [text for _, text, _ in fake_client.sent] == ["part one", "part two"]
    row = store.get_delivery("telegram", KEY)
    assert row["body_hash"] == hashlib.sha256(b"part one\npart two").hexdigest()


def test_partial_delivery_records_the_ids_already_sent(svc, fake_client, store, monkeypatch):
    monkeypatch.setattr(report_module, "render_final_report", lambda ctx: ["part one", "part two"])
    fake_client.script = [None, DataUnavailable("telegram", "HTTP 400: Bad Request: nope", retryable=False)]
    result = svc.send_report(make_ctx())
    assert result.status == "failed" and result.message_ids == [101]
    row = store.get_delivery("telegram", KEY)
    assert row["status"] == "failed" and json.loads(row["message_ids"]) == [101]


def test_partial_ids_carried_by_the_client_error_are_kept(svc, fake_client, store):
    exc = DataUnavailable("telegram", "HTTP 502: x", retryable=True)
    exc.message_ids = [55]
    fake_client.script = [exc]
    result = svc.send_report(make_ctx())
    assert result.message_ids == [55] and json.loads(store.get_delivery("telegram", KEY)["message_ids"]) == [55]


def test_non_retryable_failure_alerts_the_alert_chat_immediately(svc, fake_client, store):
    fake_client.script = [DataUnavailable("telegram", "HTTP 403: Forbidden: bot was blocked by the user", retryable=False)]
    result = svc.send_report(make_ctx())
    assert result.status == "failed" and "403" in result.reason
    assert len(fake_client.sent) == 1
    chat, text, _ = fake_client.sent[0]
    assert chat == "2002" and text.startswith("🟠 <b>ERROR</b> · <code>telegram_delivery_failed</code>")
    assert store.pending_alerts() == []


def test_unexpected_client_exceptions_are_contained_and_redacted(svc, fake_client, store):
    fake_client.script = [RuntimeError(f"boom {TOKEN} /bot{TOKEN}/sendMessage")]
    result = svc.send_report(make_ctx())
    assert result.status == "failed" and result.reason.startswith("RuntimeError: ")
    assert TOKEN not in result.reason and TOKEN not in json.dumps(store.get_delivery("telegram", KEY), default=str)
    assert TOKEN not in json.dumps(store.pending_alerts(), default=str)


# ------------------------------------------------------------------------------------- prelim report
def test_prelim_report_respects_the_setting(cfg, store, fake_clock):
    client = FakeClient()
    off = NotificationService(cfg, store, client=client, clock=fake_clock)
    result = off.send_report(make_ctx(stage="pre"))
    assert (result.status, result.reason) == ("withheld", "prelim_disabled") and client.sent == []
    row = store.get_delivery("telegram", "pre_report:2026-11-02")
    assert (row["status"], row["kind"], row["error"]) == ("withheld", "prelim_report", "prelim_disabled")

    on = NotificationService(dataclasses.replace(cfg, send_prelim_report=True), store, client=client, clock=fake_clock)
    sent = on.send_report(make_ctx(stage="pre"))
    assert sent.status == "sent" and "<b>PRELIMINARY</b> (08:45 scan)" in client.sent[0][1]
    assert client.sent[0][2] == {}  # the preliminary report has no send-by deadline
    assert store.get_delivery("telegram", "pre_report:2026-11-02")["status"] == "sent"
    assert store.get_delivery("telegram", KEY) is None


# -------------------------------------------------------------------------------------------- deadline
@pytest.mark.parametrize(
    "hms, extra, expected",
    [
        ("08:30:00", 0.0, "sent"),
        ("09:14:29", 0.0, "sent"),
        ("09:14:30", 0.0, "sent"),
        ("09:14:30", 0.001, "withheld"),
        ("09:14:31", 0.0, "withheld"),
        ("09:30:00", 0.0, "withheld"),
    ],
)
def test_report_deadline_boundaries(svc, fake_client, store, fake_clock, hms, extra, expected):
    set_ist(fake_clock, hms, extra_seconds=extra)
    result = svc.send_report(make_ctx())
    assert result.status == expected
    row = store.get_delivery("telegram", KEY)
    if expected == "sent":
        assert len(fake_client.sent) == 1 and row["status"] == "sent" and store.pending_alerts() == []
    else:
        assert result.reason == "deadline_missed" and fake_client.sent == []  # nothing goes out in this path
        assert (row["status"], row["error"]) == ("withheld", "deadline_missed")


def test_a_missed_deadline_is_queued_as_an_alert_not_silent(svc, fake_client, store, fake_clock):
    set_ist(fake_clock, "09:20:00")
    assert svc.send_report(make_ctx()).reason == "deadline_missed"
    assert fake_client.sent == []
    (alert,) = store.pending_alerts()
    assert alert["code"] == "report_deadline_missed" and "09:14:30 IST deadline" in alert["message"]
    assert svc.send_report(make_ctx()).reason == "deadline_missed"
    assert len(store.pending_alerts()) == 1  # the repeat within the hour is deduplicated
    assert svc.flush_alerts() == 1  # what the scheduler tick does
    assert fake_client.sent[0][0] == "2002" and "report_deadline_missed" in fake_client.sent[0][1]


def test_force_sends_a_late_report_with_a_late_prefix(svc, fake_client, store, fake_clock):
    set_ist(fake_clock, "09:20:00")
    assert svc.send_report(make_ctx()).status == "withheld"
    result = svc.send_report(make_ctx(), force=True)
    assert result.status == "sent"
    chat, text, options = fake_client.sent[0]
    assert chat == "1001" and LATE_PREFIX == "⏰ <b>LATE</b> - sent after the deadline, for the record only"
    assert text.startswith(f"{LATE_PREFIX}\n\n🚀 <b>MONSTER AI</b> · NSE large-move scan")
    assert options == {} and store.get_delivery("telegram", KEY)["status"] == "sent"


def test_force_before_the_deadline_is_not_labelled_late(svc, fake_client):
    assert svc.send_report(make_ctx(), force=True).status == "sent"
    chat, text, options = fake_client.sent[0]
    assert text.startswith("🚀 <b>MONSTER AI") and "LATE" not in text and options == {}


def test_late_prefix_never_pushes_a_message_over_the_limit(svc, fake_client, fake_clock, monkeypatch):
    monkeypatch.setattr(report_module, "render_final_report", lambda ctx: ["x" * 3890, "second"])
    set_ist(fake_clock, "10:00:00")
    assert svc.send_report(make_ctx(), force=True).status == "sent"
    texts = [text for _, text, _ in fake_client.sent]
    assert len(texts) == 3 and all(len(t) <= 3900 for t in texts)
    assert texts[0].startswith(LATE_PREFIX) and texts[-1] == "second"


def test_a_report_for_an_earlier_trade_date_counts_as_late(svc, fake_client, fake_clock):
    fake_clock.set(at_ist("2026-11-03", "08:00"))
    result = svc.send_report(make_ctx())
    assert result.reason == "deadline_missed" and fake_client.sent == []


def test_retries_stop_at_the_deadline_with_the_real_client(cfg, store, fake_clock):
    service, session = real_service(cfg, store, fake_clock, err(500, "Internal Server Error"))
    set_ist(fake_clock, "09:14:29", extra_seconds=0.9)  # 0.1 s left; the first backoff is at least 0.5 s
    result = service.send_report(make_ctx())
    assert (result.status, result.reason, result.attempts) == ("withheld", "deadline_missed", 1)
    assert len(session.calls) == 1 and fake_clock.sleeps == []
    row = store.get_delivery("telegram", KEY)
    assert (row["status"], row["attempts"]) == ("withheld", 1)
    assert [a["code"] for a in store.pending_alerts()] == ["report_deadline_missed"]  # queued, not sent


def test_deadline_error_from_any_client_is_a_withheld_result(svc, fake_client, store):
    exc = DeadlineExceeded()
    exc.message_ids = [9]
    fake_client.script = [exc]
    result = svc.send_report(make_ctx())
    assert (result.status, result.reason, result.message_ids) == ("withheld", "deadline_missed", [9])
    assert store.get_delivery("telegram", KEY)["status"] == "withheld"


def test_global_clock_is_used_when_none_is_injected(cfg, store, fake_clock, fake_client):
    service = NotificationService(cfg, store, client=fake_client)
    assert service.clock is fake_clock
    set_ist(fake_clock, "09:14:31")
    assert service.send_report(make_ctx()).reason == "deadline_missed"


# ------------------------------------------------------------------------------- unconfigured / invalid
def test_unconfigured_telegram_withholds_and_never_raises(settings, store, fake_clock, caplog):
    caplog.set_level(logging.INFO)
    service = NotificationService(settings, store, clock=fake_clock)  # real client without a token
    assert isinstance(service.client, TelegramClient) and service.client.configured is False
    result = service.send_report(make_ctx())
    assert (result.status, result.reason) == ("withheld", "telegram_not_configured")
    assert any(r.levelno == logging.WARNING and "telegram_not_configured" in r.getMessage() for r in caplog.records)
    row = store.get_delivery("telegram", KEY)
    assert (row["status"], row["error"]) == ("withheld", "telegram_not_configured")
    assert service.send_evening(make_summary()).reason == "telegram_not_configured"
    assert service.send_failure_notice("2026-11-02", "final", "x").reason == "telegram_not_configured"
    assert service.send_text("custom", "custom:1", "x").reason == "telegram_not_configured"
    assert service.send_test().reason == "telegram_not_configured"
    assert service.flush_alerts() == 0 and service.alert("error", "x", "y") is True


def test_missing_chat_id_counts_as_unconfigured(cfg, store, fake_clock):
    client = FakeClient()
    service = NotificationService(dataclasses.replace(cfg, telegram_chat_id=None), store, client=client, clock=fake_clock)
    assert service.send_report(make_ctx()).reason == "telegram_not_configured" and client.sent == []


def test_missing_token_counts_as_unconfigured_for_the_real_client(cfg, store, fake_clock):
    session = FakeSession()
    tokenless = dataclasses.replace(cfg, telegram_bot_token=None)
    service = NotificationService(tokenless, store, client=TelegramClient(tokenless, session=session, clock=fake_clock), clock=fake_clock)
    assert service.send_report(make_ctx()).reason == "telegram_not_configured" and session.calls == []


def test_unconfigured_duck_typed_client_withholds(cfg, store, fake_clock):
    client = FakeClient(configured=False)
    service = NotificationService(cfg, store, client=client, clock=fake_clock)
    assert service.send_report(make_ctx()).status == "withheld" and client.sent == []


@pytest.mark.parametrize(
    "override, fragment",
    [({"stage": "weird"}, "invalid report stage"), ({"trade_date": "not-a-date"}, "invalid trade_date"), ({"trade_date": ""}, "invalid trade_date")],
)
def test_invalid_contexts_are_rejected_without_a_ledger_row(svc, fake_client, store, override, fragment):
    result = svc.send_report(make_ctx(**override))
    assert result.status == "failed" and fragment in result.reason
    assert store.table_counts()["deliveries"] == 0
    assert store.pending_alerts() == []  # the alert was delivered at once
    assert len(fake_client.sent) == 1 and "report_invalid" in fake_client.sent[0][1]


def test_render_errors_become_failed_results_not_exceptions(svc, fake_client, store, monkeypatch):
    result = svc.send_report(make_ctx(model_version=""))
    assert result.status == "failed" and result.reason.startswith("render_error: ") and "model_version" in result.reason
    row = store.get_delivery("telegram", KEY)
    assert row["status"] == "failed" and "model_version" in row["error"]
    assert len(fake_client.sent) == 1 and fake_client.sent[0][0] == "2002" and "report_render_failed" in fake_client.sent[0][1]

    def explode(ctx):
        raise TypeError("bad context")

    monkeypatch.setattr(report_module, "render_final_report", explode)
    other = svc.send_report(make_ctx(trade_date="2026-11-03"))
    assert other.status == "failed" and "TypeError: bad context" in other.reason


# --------------------------------------------------------------------------------- real client end to end
def test_attempts_are_accumulated_from_the_real_client(cfg, store, fake_clock):
    service, session = real_service(cfg, store, fake_clock, err(500, "Internal Server Error"), ok(31))
    result = service.send_report(make_ctx())
    assert (result.status, result.message_ids, result.attempts) == ("sent", [31], 2)
    assert store.get_delivery("telegram", KEY)["attempts"] == 2 and len(session.calls) == 2


def test_token_never_reaches_results_ledger_alerts_or_logs(cfg, store, fake_clock, caplog):
    caplog.set_level(logging.DEBUG)
    leaky = requests.exceptions.ConnectionError(LEAKY)
    service, session = real_service(cfg, store, fake_clock, leaky, leaky, leaky)
    result = service.send_report(make_ctx())
    assert result.status == "failed" and result.attempts == 3 and len(session.calls) == 3
    blobs = [
        repr(result),
        json.dumps(store.get_delivery("telegram", KEY), default=str),
        json.dumps(store.pending_alerts(), default=str),
        caplog.text,
    ]
    for blob in blobs:
        assert TOKEN not in blob and "api.telegram.org/bot" not in blob
    assert "ConnectionError" in result.reason


def test_unauthorised_token_is_final_and_alerts_the_alert_chat(cfg, store, fake_clock):
    service, session = real_service(cfg, store, fake_clock, err(401, "Unauthorized"))
    result = service.send_report(make_ctx())
    assert result.status == "failed" and "HTTP 401" in result.reason and result.attempts == 1
    assert len(session.calls) == 2 and session.calls[1]["json"]["chat_id"] == 2002  # immediate alert (non-retryable)
    assert store.pending_alerts() == []


# ------------------------------------------------------------------------------------------- evening
def test_evening_report_flow(svc, fake_client, store, cfg, fake_clock):
    summary = make_summary()
    result = svc.send_evening(summary)
    assert result.status == "sent" and fake_client.sent[0][0] == "1001"
    assert "Evening evaluation" in fake_client.sent[0][1]
    row = store.get_delivery("telegram", "evening:2026-11-02")
    assert (row["status"], row["kind"], row["trade_date"]) == ("sent", "evening_report", "2026-11-02")
    assert svc.send_evening(summary).status == "duplicate" and len(fake_client.sent) == 1

    quiet = NotificationService(dataclasses.replace(cfg, send_evening_report=False), store, client=FakeClient(), clock=fake_clock)
    skipped = quiet.send_evening(make_summary(trade_date="2026-11-03"))
    assert (skipped.status, skipped.reason) == ("withheld", "evening_disabled")
    assert store.get_delivery("telegram", "evening:2026-11-03")["status"] == "withheld"

    broken = svc.send_evening(make_summary(trade_date="2026-11-04", model_version=""))
    assert broken.status == "failed" and broken.reason.startswith("render_error: ")


# -------------------------------------------------------------------------------------- failure notice
def test_failure_notice_goes_to_the_alert_chat_once_per_stage_and_redacts_secrets(svc, fake_client, store):
    result = svc.send_failure_notice("2026-11-02", "final", f"stale data ({TOKEN})", {"feed": f"yahoo bot{TOKEN}"})
    assert result.status == "sent"
    chat, text, options = fake_client.sent[0]
    assert chat == "2002" and options == {}
    assert "🛑 <b>MONSTER AI</b> · REPORT WITHHELD" in text and "<b>Reason:</b> stale data (&lt;redacted&gt;)" in text
    assert TOKEN not in text
    row = store.get_delivery("telegram", "final_failure:2026-11-02")
    assert (row["status"], row["kind"]) == ("sent", "failure_notice")
    assert svc.send_failure_notice("2026-11-02", "final", "other reason").status == "duplicate"
    assert svc.send_failure_notice("2026-11-02", "pre", ["a", "b"]).status == "sent"
    assert store.get_delivery("telegram", "pre_failure:2026-11-02")["status"] == "sent"
    assert len(fake_client.sent) == 2


def test_failure_notice_falls_back_to_the_main_chat(cfg, store, fake_clock):
    client = FakeClient()
    service = NotificationService(dataclasses.replace(cfg, telegram_alert_chat_id=None), store, client=client, clock=fake_clock)
    assert service.send_failure_notice("2026-11-02", "final", "no model").status == "sent"
    assert client.sent[0][0] == "1001"


# ----------------------------------------------------------------------------------- send_text / test
def test_send_text_is_idempotent_and_uses_the_requested_chat(svc, fake_client, store):
    result = svc.send_text("custom", "custom:1", "<b>hi</b>", chat_id="555", trade_date="2026-11-02")
    assert result.status == "sent" and fake_client.sent[0][:2] == ("555", "<b>hi</b>")
    row = store.get_delivery("telegram", "custom:1")
    assert (row["kind"], row["trade_date"], row["status"]) == ("custom", "2026-11-02", "sent")
    assert svc.send_text("custom", "custom:1", "<b>hi</b>").status == "duplicate"
    assert svc.send_text("custom", "custom:2", "x").status == "sent" and fake_client.sent[1][0] == "1001"
    assert len(fake_client.sent) == 2


def test_text_with_lone_surrogates_can_still_be_ledgered_without_raising(svc, store):
    result = svc.send_text("custom", "custom:broken", "headline \ud800 broken")
    assert result.status == "sent" and store.get_delivery("telegram", "custom:broken")["body_hash"]


def test_send_test_uses_a_unique_key_each_time_and_hides_secrets(svc, fake_client, store):
    first, second = svc.send_test(), svc.send_test()
    assert first.status == second.status == "sent" and len(fake_client.sent) == 2
    text = fake_client.sent[0][1]
    assert "MONSTER AI online" in text and "<b>Mode:</b> shadow" in text and TOKEN not in text
    keys = [r["dedupe_key"] for r in store.conn.execute("SELECT dedupe_key FROM deliveries")]
    assert len(set(keys)) == 2 and all(k.startswith("test:") for k in keys)


def test_minimal_duck_typed_clients_are_supported(cfg, store, fake_clock):
    class Minimal:
        def __init__(self):
            self.sent = []

        def send_message(self, chat_id, text, **options):
            self.sent.append((chat_id, text))
            return [7]

    client = Minimal()
    service = NotificationService(cfg, store, client=client, clock=fake_clock)
    result = service.send_report(make_ctx())
    assert (result.status, result.message_ids, result.attempts) == ("sent", [7], 1)
    assert service.flush_alerts() == 0


# ---------------------------------------------------------------------------------------------- alerts
def test_alert_is_deduplicated_and_delivered_immediately(svc, fake_client, store, fake_clock):
    assert svc.alert("ERROR", "feed_down", "yahoo down", dedupe_key="yahoo") is True
    assert len(fake_client.sent) == 1
    chat, text, _ = fake_client.sent[0]
    assert chat == "2002" and text.startswith("🟠 <b>ERROR</b>") and "yahoo down" in text
    assert store.pending_alerts() == []

    assert svc.alert("error", "feed_down", "yahoo down", dedupe_key="yahoo") is False
    assert len(fake_client.sent) == 1
    fake_clock.advance(61 * 60)
    assert svc.alert("error", "feed_down", "yahoo down", dedupe_key="yahoo") is True and len(fake_client.sent) == 2
    assert svc.alert("warning", "x", "y") and svc.alert("warning", "x", "y") and len(fake_client.sent) == 4


def test_alert_dedupe_window_is_configurable(svc, fake_client, fake_clock):
    assert svc.alert("error", "x", "y", dedupe_key="k", dedupe_minutes=5)
    fake_clock.advance(4 * 60)
    assert not svc.alert("error", "x", "y", dedupe_key="k", dedupe_minutes=5)
    fake_clock.advance(2 * 60)
    assert svc.alert("error", "x", "y", dedupe_key="k", dedupe_minutes=5)


def test_alert_can_be_queued_without_delivery(svc, fake_client, store):
    assert svc.alert("error", "x", "y", deliver_now=False) is True
    assert fake_client.sent == [] and len(store.pending_alerts()) == 1


def test_alert_text_is_redacted_before_it_is_stored(svc, store):
    svc.alert("error", f"code {TOKEN}", f"POST https://api.telegram.org/bot{TOKEN}/sendMessage failed", deliver_now=False)
    stored = store.pending_alerts()[0]
    assert TOKEN not in stored["message"] and TOKEN not in stored["code"] and "bot<redacted>" in stored["message"]


def test_alert_swallows_storage_and_delivery_errors(cfg, store, fake_clock, monkeypatch):
    class BrokenStore:
        def add_alert(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

    service = NotificationService(cfg, BrokenStore(), client=FakeClient(), clock=fake_clock)
    assert service.alert("error", "x", "y") is False

    client = FakeClient(script=[DataUnavailable("telegram", "boom", retryable=True)])
    flaky = NotificationService(cfg, store, client=client, clock=fake_clock)
    assert flaky.alert("error", "x", "y") is True and len(store.pending_alerts()) == 1  # stays queued

    def broken_pending(limit=20):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "pending_alerts", broken_pending)
    assert flaky.alert("error", "x2", "y2") is True


def test_flush_batches_pending_alerts_into_one_message(svc, fake_client, store):
    for i in range(3):
        store.add_alert("error", f"code{i}", f"message {i}")
    assert svc.flush_alerts() == 3
    assert len(fake_client.sent) == 1 and fake_client.sent[0][0] == "2002"
    text = fake_client.sent[0][1]
    assert "🔔 <b>MONSTER AI</b> · 3 alerts" in text and all(f"code{i}" in text for i in range(3))
    assert store.pending_alerts() == [] and svc.flush_alerts() == 0 and len(fake_client.sent) == 1


def test_flush_is_capped_and_delivers_oldest_first(svc, fake_client, store):
    for i in range(5):
        store.add_alert("warning", f"c{i}", "m")
    assert svc.flush_alerts(max_items=2) == 2
    assert [a["code"] for a in store.pending_alerts()] == ["c2", "c3", "c4"]
    assert svc.flush_alerts(max_items=0) == 1 and [a["code"] for a in store.pending_alerts()] == ["c3", "c4"]


def test_flush_of_a_single_alert_uses_the_plain_alert_format(svc, fake_client, store):
    store.add_alert("error", "disk_low", "1.2 GB free")
    assert svc.flush_alerts() == 1
    assert fake_client.sent[0][1] == (
        "🟠 <b>ERROR</b> · <code>disk_low</code>\n1.2 GB free\n<i>🕗 2026-11-02 08:30 IST</i>"
    )


def test_flush_failure_leaves_alerts_pending(cfg, store, fake_clock):
    client = FakeClient(script=[DataUnavailable("telegram", "gave up", retryable=True)])
    service = NotificationService(cfg, store, client=client, clock=fake_clock)
    store.add_alert("error", "a", "m")
    store.add_alert("error", "b", "m")
    assert service.flush_alerts() == 0 and len(store.pending_alerts()) == 2
    assert service.flush_alerts() == 2 and store.pending_alerts() == []


def test_flush_is_rate_limited(svc, fake_client, store, fake_clock):
    store.add_alert("error", "a", "m")
    assert svc.flush_alerts() == 1 and fake_clock.sleeps == []
    store.add_alert("error", "b", "m")
    assert svc.flush_alerts() == 1 and fake_clock.sleeps == [1.0]
    fake_clock.advance(5)
    store.add_alert("error", "c", "m")
    assert svc.flush_alerts() == 1 and fake_clock.sleeps == [1.0]


def test_alerts_use_the_main_chat_when_no_alert_chat_is_set(cfg, store, fake_clock):
    client = FakeClient()
    service = NotificationService(dataclasses.replace(cfg, telegram_alert_chat_id=None), store, client=client, clock=fake_clock)
    assert service.alert("error", "x", "y") is True and client.sent[0][0] == "1001"
