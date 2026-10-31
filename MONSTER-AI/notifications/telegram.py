"""Telegram Bot API client with bounded, secret-safe retries.

The bot token is part of every request URL, so this module never logs or raises a URL and never repeats an
exception message produced by the HTTP layer (those embed the URL). Failures are described with short fixed
phrases plus Telegram's own ``description`` field, always passed through ``redact``.
"""

from __future__ import annotations

import html
import logging
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests

from app.clock import Clock, get_clock
from app.errors import DataUnavailable
from app.logging_setup import redact, register_secret

log = logging.getLogger(__name__)

API_ROOT = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 3900
CONNECT_TIMEOUT_SEC = 10.0
BACKOFF_BASE_SEC = 1.0
BACKOFF_CAP_SEC = 30.0
RETRY_AFTER_CAP_SEC = 60.0
_SOURCE = "telegram"
_CHAT_TYPES = {"private": "private", "group": "group", "supergroup": "group", "channel": "channel"}
_UPDATE_KEYS = (
    "message", "edited_message", "channel_post", "edited_channel_post", "my_chat_member", "chat_member",
)

# --------------------------------------------------------------------------------------- HTML helpers

_TRACKED_TAGS = frozenset({"b", "i", "u", "s", "code", "pre", "a"})
_TOKEN_RE = re.compile(
    r"<(?P<end>/)?(?P<name>[A-Za-z][A-Za-z0-9-]*)(?P<attrs>(?:\s[^<>]*)?)>"
    r"|&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});"
    r"|[^<&]+"
    r"|[<&]"
)
_HREF_RE = re.compile(r'href\s*=\s*"([^"]*)"', re.IGNORECASE)
_PARSE_ERROR_MARKERS = ("can't parse entities", "can't find end of the entity")

Tag = tuple[str, str]


class _Chunker:
    """Packs lines into chunks, closing open tags at a chunk end and reopening them in the next chunk."""

    def __init__(self, limit: int, track_tags: bool) -> None:
        self.limit = limit
        self.track = track_tags
        self.chunks: list[str] = []
        self.stack: list[Tag] = []
        self.buf: list[str] = []
        self.size = 0
        self.has_content = False
        self._start_chunk()

    @staticmethod
    def _closers(stack: list[Tag]) -> str:
        return "".join(f"</{name}>" for name, _ in reversed(stack))

    def _start_chunk(self) -> None:
        prefix = "".join(opening for _, opening in self.stack)
        if len(prefix) + len(self._closers(self.stack)) > self.limit // 2:
            self.stack = []  # pathological tag soup: stop carrying it over so progress is guaranteed
            prefix = ""
        self.buf = [prefix]
        self.size = len(prefix)
        self.has_content = False

    def _emit(self) -> None:
        self.chunks.append("".join(self.buf).rstrip("\n") + self._closers(self.stack))
        self._start_chunk()

    def _apply(self, stack: list[Tag], match: re.Match[str] | None) -> tuple[bool, list[Tag]]:
        """Effect of one token on the open-tag stack; ``False`` means a stray closing tag to drop."""
        name = match.group("name") if match is not None else None
        if name is None or not self.track or name.lower() not in _TRACKED_TAGS:
            return True, stack
        name = name.lower()
        if not match.group("end"):
            return True, [*stack, (name, match.group(0))]
        for index in range(len(stack) - 1, -1, -1):
            if stack[index][0] == name:
                return True, stack[:index]
        return False, stack

    def _scan(self, line: str) -> tuple[str, list[Tag]]:
        """The line without stray closing tags, and the open-tag stack after it."""
        if not self.track:
            return line, self.stack
        stack = self.stack
        kept: list[str] = []
        for match in _TOKEN_RE.finditer(line):
            keep, stack = self._apply(stack, match)
            if keep:
                kept.append(match.group(0))
        return "".join(kept), stack

    def _pieces(self, text: str) -> Iterator[str]:
        size = max(1, self.limit // 2)
        while len(text) > size:
            cut = text.rfind(" ", 0, size)
            cut = cut + 1 if cut > 0 else size
            yield text[:cut]
            text = text[cut:]
        yield text

    def _tokens(self, line: str) -> Iterator[tuple[str, re.Match[str] | None]]:
        if not self.track:
            for piece in self._pieces(line):
                yield piece, None
            return
        for match in _TOKEN_RE.finditer(line):
            token = match.group(0)
            if match.group("name") is not None and len(token) <= self.limit // 4:
                yield token, match
            else:
                for piece in self._pieces(token):
                    yield piece, None

    def _fits(self, line: str, after: list[Tag]) -> bool:
        separator = 1 if self.has_content else 0
        return self.size + separator + len(line) + len(self._closers(after)) <= self.limit

    def _append(self, line: str, after: list[Tag]) -> None:
        separator = "\n" if self.has_content else ""
        self.buf.append(separator + line)
        self.size += len(separator) + len(line)
        self.has_content = True
        self.stack = after

    def _add_long(self, line: str) -> None:
        """Place a line that cannot fit in one chunk, token by token."""
        for token, match in self._tokens(line):
            keep, after = self._apply(self.stack, match)
            if keep and self.has_content and self.size + len(token) + len(self._closers(after)) > self.limit:
                self._emit()
                keep, after = self._apply(self.stack, match)  # the carried stack may have been dropped
            if not keep:
                continue
            self.buf.append(token)
            self.size += len(token)
            self.has_content = True
            self.stack = after

    def add_line(self, line: str) -> None:
        blank = not line.strip()
        if blank and not self.has_content:
            return
        clean, after = self._scan(line)
        if self._fits(clean, after):
            self._append(clean, after)
            return
        if self.has_content:
            self._emit()
            if blank:
                return
            clean, after = self._scan(line)
            if self._fits(clean, after):
                self._append(clean, after)
                return
        self._add_long(line)

    def run(self, text: str) -> list[str]:
        for line in text.split("\n"):
            self.add_line(line)
        if self.has_content:
            self._emit()
        return [chunk for chunk in self.chunks if chunk.strip()]


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS, *, html_mode: bool = True) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters, preferring line boundaries.

    In HTML mode the tags Telegram supports here (b, i, u, s, code, pre, a) stay balanced in every chunk:
    open tags are closed at the end of a chunk and reopened at the start of the next one.
    """
    body = "" if text is None else str(text)
    if limit < 40:
        raise ValueError("limit is too small to split a message")
    if len(body) <= limit:
        return [body]
    return _Chunker(limit, html_mode).run(body) or [body[:limit]]


def to_plain_text(text: str) -> str:
    """Strip Telegram-HTML to plain text: tags removed, entities unescaped, link targets in parentheses."""
    out: list[str] = []
    anchors: list[tuple[str, int]] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0)
        name = match.group("name")
        if name is None:
            out.append(html.unescape(token) if token.startswith("&") else token)
            continue
        if name.lower() != "a":
            continue
        if match.group("end"):
            if anchors:
                href, start = anchors.pop()
                label = "".join(out[start:])
                if href and href not in label:
                    out.append(f" ({href})")
        else:
            found = _HREF_RE.search(match.group("attrs") or "")
            anchors.append((html.unescape(found.group(1)) if found else "", len(out)))
    return "".join(out)


# ------------------------------------------------------------------------------------------- client


class DeadlineExceeded(DataUnavailable):
    """The caller's send-by time passed before the message could be delivered."""

    def __init__(self, message: str = "deadline passed before the message could be delivered") -> None:
        super().__init__(_SOURCE, message, retryable=False)


@dataclass
class _Reply:
    """Outcome of one HTTP round trip. ``transient`` replies are worth retrying; the rest are final."""

    ok: bool = False
    result: Any = None
    code: int = 0
    description: str = ""
    reason: str = ""
    transient: bool = False
    wait: float | None = None
    migrate_to: str | None = None


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def _short(value: Any, limit: int = 200) -> str:
    text = " ".join(redact(str(value if value is not None else "")).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _chat_value(chat_id: Any) -> int | str:
    text = str(chat_id).strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else text


def _message_id(result: Any) -> int | None:
    if isinstance(result, dict):
        value = result.get("message_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _first_chat(updates: list[dict[str, Any]]) -> dict[str, str] | None:
    for update in updates:
        candidates = [update.get(key) for key in _UPDATE_KEYS]
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            candidates.append(callback.get("message"))
        for item in candidates:
            chat = item.get("chat") if isinstance(item, dict) else None
            if not isinstance(chat, dict) or chat.get("id") is None:
                continue
            name = " ".join(str(chat[k]) for k in ("first_name", "last_name") if chat.get(k))
            title = chat.get("title") or name or (f"@{chat['username']}" if chat.get("username") else "")
            kind = str(chat.get("type") or "")
            return {
                "chat_id": str(chat["id"]),
                "title": _short(title, 80),
                "type": _CHAT_TYPES.get(kind, kind),
            }
    return None


def _keep_http_debug_logs_off() -> None:
    """urllib3 logs ``POST /bot<token>/sendMessage`` at DEBUG; never let that reach any handler."""
    for name in ("urllib3", "requests"):
        logger = logging.getLogger(name)
        if logger.getEffectiveLevel() < logging.WARNING:
            logger.setLevel(logging.WARNING)


class TelegramClient:
    """Thin Bot API wrapper; ``session`` is any requests-style object with ``post(url, json=, timeout=)``."""

    def __init__(
        self,
        settings: Any,
        *,
        session: Any = None,
        clock: Clock | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._settings = settings
        self._token = (getattr(settings, "telegram_bot_token", None) or "").strip()
        self._session = session if session is not None else requests.Session()
        self._clock = clock
        self._rng = rng or random.Random()
        self._read_timeout = float(getattr(settings, "telegram_timeout_sec", 20) or 20)
        self.last_attempts = 0
        self.migrated_chat_ids: dict[str, str] = {}
        if self._token:
            register_secret(self._token)
        _keep_http_debug_logs_off()

    def __repr__(self) -> str:
        return f"TelegramClient(configured={self.configured})"

    @property
    def configured(self) -> bool:
        return bool(self._token)

    @property
    def clock(self) -> Clock:
        return self._clock or get_clock()

    def close(self) -> None:
        closer = getattr(self._session, "close", None)
        if callable(closer):
            closer()

    # ------------------------------------------------------------------------------- public API
    def get_me(self) -> dict[str, Any]:
        self._require_configured()
        self.last_attempts = 0
        result = self._call("getMe", {}, self._read_timeout)
        if not isinstance(result, dict):
            raise DataUnavailable(_SOURCE, "unexpected getMe response", retryable=True) from None
        return result

    def send_message(
        self,
        chat_id: Any,
        text: str,
        *,
        parse_mode: str | None = "HTML",
        disable_preview: bool = True,
        deadline: datetime | None = None,
    ) -> list[int]:
        """Send ``text`` (split into several messages when long); returns the Telegram message ids.

        ``deadline`` (aware datetime) makes the call give up with ``DeadlineExceeded`` instead of sending
        or retrying after that instant. On failure the raised error carries ``message_ids`` already sent.
        """
        self._require_configured()
        self.last_attempts = 0
        body = "" if text is None else str(text)
        if not body.strip():
            raise DataUnavailable(_SOURCE, "refusing to send an empty message", retryable=False) from None
        if chat_id is None or not str(chat_id).strip():
            raise DataUnavailable(_SOURCE, "no destination chat id was given", retryable=False) from None
        target = _chat_value(self.migrated_chat_ids.get(str(chat_id).strip(), chat_id))
        html_mode = (parse_mode or "").upper() == "HTML"
        sent: list[int] = []
        try:
            for chunk in split_message(body, html_mode=html_mode):
                message_id = self._send_chunk(target, chunk, parse_mode, disable_preview, deadline)
                if message_id is not None:
                    sent.append(message_id)
        except DataUnavailable as exc:
            exc.message_ids = sent
            raise
        return sent

    def get_updates(self, offset: int | None = None, timeout: int = 0) -> list[dict[str, Any]]:
        self._require_configured()
        self.last_attempts = 0
        wait = max(0, int(timeout))
        payload: dict[str, Any] = {"timeout": wait}
        if offset is not None:
            payload["offset"] = int(offset)
        result = self._call("getUpdates", payload, self._read_timeout + wait)
        if not isinstance(result, list):
            raise DataUnavailable(_SOURCE, "unexpected getUpdates response", retryable=True) from None
        return [item for item in result if isinstance(item, dict)]

    def discover_chat_id(
        self, *, wait_seconds: float = 120, poll_interval: float = 3.0
    ) -> dict[str, str] | None:
        """Poll until the bot has seen a message and return that chat, or ``None`` after ``wait_seconds``."""
        started = self.clock.monotonic()
        while True:
            found = _first_chat(self.get_updates(timeout=0))
            if found:
                return found
            remaining = wait_seconds - (self.clock.monotonic() - started)
            if remaining <= 0:
                return None
            self.clock.sleep(min(max(poll_interval, 0.1), remaining))

    # ------------------------------------------------------------------------------- internals
    def _require_configured(self) -> None:
        if not self._token:
            raise DataUnavailable(_SOURCE, "bot token is not configured", retryable=False) from None

    def _send_chunk(
        self,
        chat: int | str,
        chunk: str,
        parse_mode: str | None,
        disable_preview: bool,
        deadline: datetime | None,
    ) -> int | None:
        text, mode = chunk, parse_mode
        migrated = plain = False
        while True:
            payload: dict[str, Any] = {"chat_id": chat, "text": text}
            if mode:
                payload["parse_mode"] = mode
            if disable_preview:
                payload["link_preview_options"] = {"is_disabled": True}
            reply = self._execute("sendMessage", payload, self._read_timeout, deadline)
            if reply.ok:
                return _message_id(reply.result)
            if reply.migrate_to and not migrated:
                migrated = True
                chat = _chat_value(reply.migrate_to)
                self.migrated_chat_ids[str(payload["chat_id"])] = str(chat)
                continue
            if reply.code == 400 and mode and not plain and _is_parse_error(reply.description):
                plain = True
                text, mode = to_plain_text(chunk), None
                log.warning("telegram rejected the HTML formatting; resending as plain text")
                continue
            raise DataUnavailable(_SOURCE, reply.reason, retryable=False) from None

    def _call(self, method: str, payload: dict[str, Any], read_timeout: float) -> Any:
        reply = self._execute(method, payload, read_timeout, None)
        if not reply.ok:
            raise DataUnavailable(_SOURCE, reply.reason, retryable=False) from None
        return reply.result

    def _execute(
        self, method: str, payload: dict[str, Any], read_timeout: float, deadline: datetime | None
    ) -> _Reply:
        """Run one API call with bounded retries; returns a success or a final (non-retryable) reply."""
        max_attempts = max(1, int(getattr(self._settings, "telegram_max_attempts", 5) or 1))
        last_reason = "no attempt was made"
        for attempt in range(1, max_attempts + 1):
            if deadline is not None and self.clock.now() > deadline:
                raise DeadlineExceeded() from None
            self.last_attempts += 1
            reply = self._post(method, payload, read_timeout)
            if reply.ok or not reply.transient:
                return reply
            last_reason = reply.reason
            log.warning("telegram %s attempt %d/%d failed: %s", method, attempt, max_attempts, reply.reason)
            if attempt < max_attempts:
                delay = reply.wait if reply.wait is not None else self._backoff(attempt)
                if deadline is not None and self.clock.now() + timedelta(seconds=delay) > deadline:
                    raise DeadlineExceeded() from None
                self.clock.sleep(delay)
        raise DataUnavailable(
            _SOURCE, f"gave up after {max_attempts} attempts: {last_reason}", retryable=True
        ) from None

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with equal jitter: between half and all of ``base * 2**(attempt-1)``."""
        ceiling = min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * (2 ** (attempt - 1)))
        return ceiling * (0.5 + 0.5 * self._rng.random())

    def _post(self, method: str, payload: dict[str, Any], read_timeout: float) -> _Reply:
        url = f"{API_ROOT}/bot{self._token}/{method}"
        timeout = (CONNECT_TIMEOUT_SEC, read_timeout)
        try:
            response = self._session.post(url, json=dict(payload), timeout=timeout)
        except requests.exceptions.Timeout:
            return _Reply(transient=True, reason="request timed out")
        except (
            requests.exceptions.InvalidURL,
            requests.exceptions.InvalidSchema,
            requests.exceptions.MissingSchema,
        ):
            return _Reply(reason="request could not be built (check TELEGRAM_BOT_TOKEN)")
        except (requests.exceptions.RequestException, OSError) as exc:
            return _Reply(transient=True, reason=f"network error ({type(exc).__name__})")
        except Exception as exc:  # never let an unknown error (whose text may embed the URL) escape
            return _Reply(reason=f"unexpected error ({type(exc).__name__})")
        return self._interpret(response)

    @staticmethod
    def _interpret(response: Any) -> _Reply:
        status = int(getattr(response, "status_code", 0) or 0)
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or "ok" not in body:
            reason = f"HTTP {status}: response was not Telegram JSON"
            return _Reply(transient=True, code=status, reason=reason)
        if body.get("ok"):
            return _Reply(ok=True, result=body.get("result"), code=status or 200)

        code = int(_number(body.get("error_code")) or status or 0)
        description = _short(body.get("description"), 160)
        parameters = body.get("parameters") if isinstance(body.get("parameters"), dict) else {}
        if code == 429:
            retry_after = _number(parameters.get("retry_after"))
            if retry_after is None:
                headers = getattr(response, "headers", None) or {}
                retry_after = _number(headers.get("Retry-After"))
            wait = min(RETRY_AFTER_CAP_SEC, max(0.0, retry_after)) if retry_after is not None else None
            return _Reply(transient=True, code=429, wait=wait, reason="HTTP 429: rate limited by Telegram")
        if code >= 500 or code == 408:
            return _Reply(transient=True, code=code, reason=_reason(code, description))
        migrate = parameters.get("migrate_to_chat_id")
        return _Reply(
            code=code,
            description=description,
            reason=_reason(code, description),
            migrate_to=str(migrate) if migrate not in (None, "") else None,
        )


def _is_parse_error(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in _PARSE_ERROR_MARKERS)


def _reason(code: int, description: str) -> str:
    text = f"HTTP {code}: {description}" if description else f"HTTP {code}"
    return text + " (check TELEGRAM_BOT_TOKEN)" if code == 401 else text
