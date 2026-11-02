"""Idempotent Telegram delivery on top of the ``deliveries`` and ``alerts`` ledgers.

Every outbound message has a dedupe key; once a key is ``sent`` the same message is never sent again, and any
other state (``pending``, ``failed``, ``withheld``) is re-evaluated on the next call. Delivery problems never
raise into the caller: they are returned as a ``DeliveryResult``, written to the ledger with a redacted
reason, and queued as an alert.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from app import jsonutil
from app.clock import Clock, at_ist, get_clock, iso, parse_date
from app.errors import DataUnavailable
from app.logging_setup import exception_site, redact, register_secret
from notifications import charts as chart_images
from notifications import ideas as idea_messages
from notifications import report
from notifications.telegram import MAX_MESSAGE_CHARS, DeadlineExceeded, TelegramClient, split_message
from ranking.types import EveningSummary, IdeasReport, ReportContext

log = logging.getLogger(__name__)

CHANNEL = "telegram"
LATE_PREFIX = "⏰ <b>LATE</b> - sent after the deadline, for the record only"
FLUSH_MIN_GAP_SEC = 1.0
CHART_BUDGET_SEC = 45.0  # the images are an extra: they never hold a job longer than this once the text is out


@dataclass
class DeliveryResult:
    """``status`` is sent | failed | withheld | duplicate; ``reason`` explains anything but a clean send."""

    status: str
    reason: str | None = None
    message_ids: list[int] = field(default_factory=list)
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status in ("sent", "duplicate")


def _safe_reason(exc: BaseException) -> str:
    text = " ".join(redact(str(exc)).split())
    if not isinstance(exc, DataUnavailable):
        text = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return text[:200]


def _redact_any(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): redact(str(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(str(v)) for v in value]
    return redact(str(value))


def _prefix_late(messages: list[str]) -> list[str]:
    first = f"{LATE_PREFIX}\n\n{messages[0]}"
    head = [first] if len(first) <= MAX_MESSAGE_CHARS else split_message(first)
    return [*head, *messages[1:]]


class NotificationService:
    def __init__(self, settings: Any, store: Any, *, client: Any = None, clock: Clock | None = None) -> None:
        self.settings = settings
        self.store = store
        self._clock = clock
        self.client = client if client is not None else TelegramClient(settings, clock=clock)
        self._last_flush: float | None = None
        register_secret(getattr(settings, "telegram_bot_token", None))

    @property
    def clock(self) -> Clock:
        return self._clock or get_clock()

    # --------------------------------------------------------------------------------- reports
    def send_report(self, ctx: ReportContext, *, force: bool = False) -> DeliveryResult:
        """Send the prelim (stage ``pre``) or final report once per trade date.

        A final report that would go out after ``report_deadline_time`` is withheld unless ``force`` is set,
        in which case it is prefixed as late. In-flight retries also stop at the deadline.
        """
        stage = str(getattr(ctx, "stage", "") or "").strip().lower()
        trade_date = str(getattr(ctx, "trade_date", "") or "").strip()
        if stage not in ("pre", "final"):
            return self._reject(f"invalid report stage {stage!r}")
        try:
            parse_date(trade_date)
        except (TypeError, ValueError):
            return self._reject(f"invalid trade_date {trade_date!r}")

        kind = "prelim_report" if stage == "pre" else "final_report"
        key = f"{stage}_report:{trade_date}"
        duplicate = self._duplicate(key)
        if duplicate:
            return duplicate
        if stage == "pre" and not self.settings.send_prelim_report:
            return self._withhold(kind, key, trade_date, "prelim_disabled")
        chat_id = self.settings.telegram_chat_id
        if not self._can_send(chat_id):
            return self._withhold(kind, key, trade_date, "telegram_not_configured", warn=True)

        deadline = at_ist(trade_date, self.settings.report_deadline_time) if stage == "final" else None
        late = deadline is not None and self.clock.now() > deadline
        if late and not force:
            withheld = self._withhold(kind, key, trade_date, "deadline_missed", warn=True)
            self.alert(  # queued only: the scheduler's flush delivers it, nothing is sent from this code path
                "error",
                "report_deadline_missed",
                f"{kind} for {trade_date} was withheld: the send attempt came after the "
                f"{self.settings.report_deadline_time} IST deadline",
                dedupe_key=f"delivery:{key}",
                deliver_now=False,
            )
            return withheld

        try:
            messages = report.render_prelim_report(ctx) if stage == "pre" else report.render_final_report(ctx)
        except Exception as exc:
            return self._render_failed(kind, key, trade_date, exc)
        if late:
            messages = _prefix_late(messages)
        return self._deliver(
            kind, key, messages, chat_id=chat_id, trade_date=trade_date, deadline=None if force else deadline,
            charts=None if late else lambda: chart_images.report_charts(ctx),
        )

    def send_evening(self, summary: EveningSummary) -> DeliveryResult:
        trade_date = str(getattr(summary, "trade_date", "") or "")
        key, kind = f"evening:{trade_date}", "evening_report"
        duplicate = self._duplicate(key)
        if duplicate:
            return duplicate
        if not self.settings.send_evening_report:
            return self._withhold(kind, key, trade_date, "evening_disabled")
        chat_id = self.settings.telegram_chat_id
        if not self._can_send(chat_id):
            return self._withhold(kind, key, trade_date, "telegram_not_configured", warn=True)
        try:
            messages = report.render_evening_report(summary)
        except Exception as exc:
            return self._render_failed(kind, key, trade_date, exc)
        return self._deliver(kind, key, messages, chat_id=chat_id, trade_date=trade_date,
                             charts=lambda: chart_images.evening_charts(summary))

    def send_ideas(self, ideas_report: IdeasReport, *, force: bool = False) -> DeliveryResult:
        """Send the closing-hour trade-idea report once per trade date. It is time-critical: a report that would go out after
        ``ideas_report_deadline_time`` is withheld (the entry window is nearly over) unless ``force`` is set, which a manual run
        uses; the report then says itself that it is late."""
        trade_date = str(getattr(ideas_report, "trade_date", "") or "").strip()
        try:
            parse_date(trade_date)
        except (TypeError, ValueError):
            return self._reject(f"invalid trade_date {trade_date!r}")
        kind, key = "ideas_report", f"ideas:{trade_date}"
        duplicate = self._duplicate(key)
        if duplicate:
            return duplicate
        if not self.settings.send_ideas_report:
            return self._withhold(kind, key, trade_date, "ideas_disabled")
        chat_id = self.settings.telegram_chat_id
        if not self._can_send(chat_id):
            return self._withhold(kind, key, trade_date, "telegram_not_configured", warn=True)
        deadline = at_ist(trade_date, self.settings.ideas_report_deadline_time)
        if self.clock.now() > deadline and not force:
            withheld = self._withhold(kind, key, trade_date, "deadline_missed", warn=True)
            self.alert(  # queued only: the scheduler's flush delivers it
                "warning",
                "ideas_deadline_missed",
                f"the closing-hour ideas for {trade_date} were withheld: the send attempt came after the "
                f"{self.settings.ideas_report_deadline_time} IST deadline",
                dedupe_key=f"delivery:{key}",
                deliver_now=False,
            )
            return withheld
        try:
            messages = idea_messages.render_ideas_report(ideas_report)
        except Exception as exc:
            return self._render_failed(kind, key, trade_date, exc)
        late = bool(getattr(ideas_report, "late", False)) or self.clock.now() > deadline  # too late to act on: no images either
        return self._deliver(kind, key, messages, chat_id=chat_id, trade_date=trade_date, deadline=None if force else deadline,
                             charts=None if late else lambda: chart_images.ideas_charts(ideas_report))

    def send_failure_notice(
        self, trade_date: str, stage: str, reason: str, details: Any = None
    ) -> DeliveryResult:
        """Tell the operator a report was withheld (sent to the alert chat, once per stage and date)."""
        key, kind = f"{stage}_failure:{trade_date}", "failure_notice"
        duplicate = self._duplicate(key)
        if duplicate:
            return duplicate
        chat_id = self.settings.alert_chat_id
        if not self._can_send(chat_id):
            return self._withhold(kind, key, str(trade_date), "telegram_not_configured", warn=True)
        messages = report.render_failure_notice(trade_date, stage, _redact_any(reason), _redact_any(details))
        return self._deliver(kind, key, messages, chat_id=chat_id, trade_date=str(trade_date))

    def send_text(
        self, kind: str, dedupe_key: str, text: str, *, chat_id: Any = None, trade_date: str | None = None
    ) -> DeliveryResult:
        """Idempotent send of arbitrary Telegram-HTML text (the client splits it if it is long)."""
        duplicate = self._duplicate(dedupe_key)
        if duplicate:
            return duplicate
        target = chat_id if chat_id is not None else self.settings.telegram_chat_id
        if not self._can_send(target):
            return self._withhold(kind, dedupe_key, trade_date, "telegram_not_configured", warn=True)
        return self._deliver(kind, dedupe_key, [text], chat_id=target, trade_date=trade_date)

    def send_test(self) -> DeliveryResult:
        key = f"test:{iso(self.clock.now())}:{uuid.uuid4().hex[:8]}"
        return self.send_text("test", key, report.render_test_message(self.settings))

    # ---------------------------------------------------------------------------------- alerts
    def alert(
        self,
        level: str,
        code: str,
        message: str,
        *,
        dedupe_key: str | None = None,
        dedupe_minutes: int = 60,
        deliver_now: bool = True,
    ) -> bool:
        """Queue an operational alert (deduplicated) and try to deliver it; never raises.

        Returns False when the alert was suppressed as a duplicate or could not be queued.
        """
        try:
            queued = self.store.add_alert(
                str(level).lower(),
                redact(str(code)),
                redact(str(message)),
                dedupe_key=dedupe_key,
                dedupe_minutes=dedupe_minutes,
            )
        except Exception as exc:
            log.warning("could not queue alert %s: %s", code, _safe_reason(exc))
            return False
        if queued and deliver_now:
            try:
                self.flush_alerts()
            except Exception as exc:
                log.warning("alert delivery attempt failed: %s", _safe_reason(exc))
        return bool(queued)

    def flush_alerts(self, max_items: int = 10) -> int:
        """Deliver up to ``max_items`` pending alerts as one message; returns how many were delivered."""
        chat_id = self.settings.alert_chat_id
        if not self._can_send(chat_id):
            return 0
        pending = self.store.pending_alerts(limit=max(1, int(max_items)))
        if not pending:
            return 0
        self._pace()
        if len(pending) == 1:
            first = pending[0]
            messages = [
                report.render_alert(
                    first["level"], first["code"], first["message"], created_at=first.get("created_at")
                )
            ]
        else:
            messages = report.render_alert_digest(pending)
        try:
            for text in messages:
                self.client.send_message(chat_id, text)
        except Exception as exc:
            log.warning("alert delivery failed: %s", _safe_reason(exc))
            return 0
        for alert in pending:
            self.store.mark_alert_delivered(alert["alert_id"])
        return len(pending)

    # -------------------------------------------------------------------------------- internals
    def _pace(self) -> None:
        """Keep consecutive alert flushes at least ``FLUSH_MIN_GAP_SEC`` apart (Telegram allows ~1 msg/s)."""
        if self._last_flush is not None:
            wait = FLUSH_MIN_GAP_SEC - (self.clock.monotonic() - self._last_flush)
            if wait > 0:
                self.clock.sleep(wait)
        self._last_flush = self.clock.monotonic()

    def _can_send(self, chat_id: Any) -> bool:
        return bool(getattr(self.client, "configured", True)) and bool(chat_id)

    def _attempts_used(self) -> int:
        used = getattr(self.client, "last_attempts", None)
        return used if isinstance(used, int) and not isinstance(used, bool) else 1

    def _duplicate(self, key: str) -> DeliveryResult | None:
        row = self.store.get_delivery(CHANNEL, key)
        if not row or row.get("status") != "sent":
            return None
        ids = jsonutil.loads(row.get("message_ids"), []) or []
        return DeliveryResult("duplicate", "already_sent", [i for i in ids if isinstance(i, int)], 0)

    def _withhold(
        self, kind: str, key: str, trade_date: str | None, reason: str, *, warn: bool = False
    ) -> DeliveryResult:
        self.store.save_delivery(CHANNEL, kind, key, "withheld", trade_date=trade_date, error=reason)
        (log.warning if warn else log.info)("%s withheld: %s", key, reason)
        return DeliveryResult("withheld", reason)

    def _reject(self, reason: str) -> DeliveryResult:
        log.error("report not sent: %s", reason)
        self.alert("error", "report_invalid", reason, dedupe_key=f"report_invalid:{reason}")
        return DeliveryResult("failed", reason)

    def _render_failed(self, kind: str, key: str, trade_date: str, exc: Exception) -> DeliveryResult:
        reason = f"render_error: {_safe_reason(exc)}"
        self.store.save_delivery(CHANNEL, kind, key, "failed", trade_date=trade_date, error=reason)
        log.error("%s could not be rendered: %s", key, reason)
        self.alert(
            "error", "report_render_failed", f"{kind} for {trade_date}: {reason}", dedupe_key=f"render:{key}"
        )
        return DeliveryResult("failed", reason)

    def _send_charts(self, build: Callable[[], list[Any]] | None, chat_id: Any, key: str) -> int:
        """Send a report's chart images after its text, one photo each. Best effort: a missing library, a failed drawing or a refused
        upload is logged and never changes the delivery that is already recorded. A client without ``send_photo`` (or
        ``SEND_CHARTS=false``) sends none. Returns how many images went out."""
        send = getattr(self.client, "send_photo", None)
        if build is None or not callable(send) or not getattr(self.settings, "send_charts", True):
            return 0
        sent = 0
        until = self.clock.now() + timedelta(seconds=CHART_BUDGET_SEC)
        try:
            for chart in build():
                send(chat_id, chart.png, caption=chart.caption, filename=chart.filename, deadline=until)
                sent += 1
        except Exception as exc:  # noqa: BLE001 - an image is an extra: the text report is already delivered
            log.warning("%s: chart images stopped after %d: %s at %s", key, sent, _safe_reason(exc), exception_site(exc))
        if sent:
            log.info("%s: %d chart image(s) delivered", key, sent)
        return sent

    def _deliver(
        self,
        kind: str,
        key: str,
        messages: list[str],
        *,
        chat_id: Any,
        trade_date: str | None,
        deadline: datetime | None = None,
        charts: Callable[[], list[Any]] | None = None,
    ) -> DeliveryResult:
        digest = hashlib.sha256("\n".join(messages).encode("utf-8", "replace")).hexdigest()
        self.store.save_delivery(CHANNEL, kind, key, "pending", trade_date=trade_date, body_hash=digest)
        options = {"deadline": deadline} if deadline is not None else {}
        sent_ids: list[int] = []
        attempts = 0
        status, reason, retryable = "sent", None, True
        for text in messages:
            try:
                sent_ids.extend(int(i) for i in self.client.send_message(chat_id, text, **options))
            except DeadlineExceeded as exc:
                status, reason, retryable = "withheld", "deadline_missed", False
                sent_ids.extend(getattr(exc, "message_ids", []))
            except DataUnavailable as exc:
                status, reason, retryable = "failed", _safe_reason(exc), exc.retryable
                sent_ids.extend(getattr(exc, "message_ids", []))
            except Exception as exc:  # delivery problems must never raise into the scheduler
                status, reason = "failed", _safe_reason(exc)
                log.warning("delivery %s [%s] failed unexpectedly: %s at %s", kind, key, reason, exception_site(exc))
            attempts += self._attempts_used()
            if status != "sent":
                break

        if status == "sent":
            self.store.save_delivery(
                CHANNEL, kind, key, "sent", trade_date=trade_date, attempts=attempts,
                message_ids=sent_ids, body_hash=digest,
            )
            log.info("%s delivered (%d message(s), %d attempt(s))", key, len(sent_ids), attempts)
            self._send_charts(charts, chat_id, key)
            return DeliveryResult("sent", None, sent_ids, attempts)

        self.store.save_delivery(
            CHANNEL, kind, key, status, trade_date=trade_date, attempts=attempts,
            message_ids=sent_ids or None, error=reason, body_hash=digest,
        )
        log.warning("%s not delivered (%s): %s", key, status, reason)
        subject = f"{kind} for {trade_date}" if trade_date else kind
        self.alert(
            "error",
            "report_deadline_missed" if status == "withheld" else "telegram_delivery_failed",
            f"{subject} was not delivered: {reason}",
            dedupe_key=f"delivery:{key}",
            deliver_now=not retryable and status == "failed",
        )
        return DeliveryResult(status, reason, sent_ids, attempts)
