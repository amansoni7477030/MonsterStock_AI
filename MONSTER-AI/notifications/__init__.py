"""Telegram delivery: pure HTML renderers, a resilient Bot API client and an idempotent delivery service."""

from __future__ import annotations

from notifications.report import (
    render_alert,
    render_alert_digest,
    render_evening_report,
    render_failure_notice,
    render_final_report,
    render_prelim_report,
    render_test_message,
)
from notifications.service import DeliveryResult, NotificationService
from notifications.telegram import DeadlineExceeded, TelegramClient, split_message, to_plain_text

__all__ = [
    "DeadlineExceeded",
    "DeliveryResult",
    "NotificationService",
    "TelegramClient",
    "render_alert",
    "render_alert_digest",
    "render_evening_report",
    "render_failure_notice",
    "render_final_report",
    "render_prelim_report",
    "render_test_message",
    "split_message",
    "to_plain_text",
]
