"""Telegram client tests: scripted fake session, FakeClock (no real sleeping), no network."""

from __future__ import annotations

import dataclasses
import json
import logging
import random
import re
import traceback
from datetime import timedelta

import pytest
import requests

from app.errors import DataUnavailable
from app.logging_setup import redact
from notifications.telegram import (
    MAX_MESSAGE_CHARS,
    DeadlineExceeded,
    TelegramClient,
    split_message,
    to_plain_text,
)

TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"
LEAKY = (
    f"HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded with url: "
    f"/bot{TOKEN}/sendMessage (Caused by NewConnectionError('Failed to establish a new connection'))"
)


class FakeResponse:
    def __init__(self, status=200, body=None, *, headers=None, not_json=False):
        self.status_code = status
        self._body = body
        self._not_json = not_json
        self.headers = headers or {}
        self.text = "<html>gateway</html>" if not_json else json.dumps(body)

    def json(self):
        if self._not_json:
            raise ValueError("not json")
        return self._body


def ok(message_id=1):
    return FakeResponse(200, {"ok": True, "result": {"message_id": message_id, "chat": {"id": 1001}}})


def err(code, description, *, headers=None, **parameters):
    body = {"ok": False, "error_code": code, "description": description}
    if parameters:
        body["parameters"] = parameters
    return FakeResponse(code, body, headers=headers)


def updates(*items):
    return FakeResponse(200, {"ok": True, "result": list(items)})


class FakeSession:
    """Requests-style session that replays a script: a response, an exception to raise, or a callable."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def post(self, url, json=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.script.pop(0) if self.script else ok(len(self.calls))
        if callable(item):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


@pytest.fixture()
def tg_settings(settings):
    return dataclasses.replace(
        settings, telegram_bot_token=TOKEN, telegram_chat_id="1001", telegram_max_attempts=4, telegram_timeout_sec=15
    )


@pytest.fixture()
def make(tg_settings, fake_clock):
    def build(*script, settings=None, seed=0):
        session = FakeSession(*script)
        client = TelegramClient(settings or tg_settings, session=session, clock=fake_clock, rng=random.Random(seed))
        return client, session

    return build


_TAG = re.compile(r"<(/?)([a-z]+)(?:\s[^<>]*)?>")


def is_balanced(chunk):
    stack = []
    for closing, name in _TAG.findall(chunk):
        if closing:
            if not stack or stack.pop() != name:
                return False
        else:
            stack.append(name)
    return not stack


def squash(text):
    return "".join(re.sub(r"<[^>]+>", "", text).split())


def assert_no_secret(*texts):
    for text in texts:
        assert TOKEN not in text
        assert "api.telegram.org/bot" not in text


# ------------------------------------------------------------------------------------------ happy path
def test_send_message_posts_html_without_link_previews(make, fake_clock):
    client, session = make(ok(41))
    assert client.send_message("1001", "<b>hello</b>") == [41]
    call = session.calls[0]
    assert call["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert call["json"] == {
        "chat_id": 1001,
        "text": "<b>hello</b>",
        "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True},
    }
    assert call["timeout"] == (10.0, 15.0)
    assert fake_clock.sleeps == [] and client.last_attempts == 1


def test_chat_id_normalisation_and_options(make):
    client, session = make(ok(1), ok(2), ok(3))
    client.send_message("-1001234567890", "x")
    client.send_message("@fake_channel", "x", disable_preview=False, parse_mode=None)
    client.send_message(555, "x")
    first, second, third = (c["json"] for c in session.calls)
    assert first["chat_id"] == -1001234567890
    assert second == {"chat_id": "@fake_channel", "text": "x"}
    assert third["chat_id"] == 555


def test_unconfigured_client_refuses_without_touching_the_network(settings, fake_clock):
    session = FakeSession()
    client = TelegramClient(settings, session=session, clock=fake_clock)
    assert client.configured is False
    for call in (
        lambda: client.send_message("1", "hi"),
        client.get_me,
        client.get_updates,
        client.discover_chat_id,
    ):
        with pytest.raises(DataUnavailable) as info:
            call()
        assert info.value.retryable is False and "not configured" in str(info.value)
    assert session.calls == []


def test_empty_messages_and_missing_chats_are_rejected_locally(make):
    client, session = make()
    for text in ("", "  \n ", None):
        with pytest.raises(DataUnavailable) as info:
            client.send_message("1001", text)
        assert info.value.retryable is False
    for chat in (None, "", "  "):
        with pytest.raises(DataUnavailable) as info:
            client.send_message(chat, "hello")
        assert info.value.retryable is False and "chat id" in str(info.value)
    assert session.calls == []


def test_repr_registration_and_default_session_never_expose_the_token(tg_settings):
    client = TelegramClient(tg_settings)
    assert TOKEN not in repr(client) and client.configured is True
    assert isinstance(client._session, requests.Session)
    assert TOKEN not in redact(f"failure while calling {TOKEN}")  # registered as a secret on construction
    client.close()


def test_http_library_debug_logging_is_switched_off(tg_settings, fake_clock):
    names = ("urllib3", "requests")
    saved = {name: logging.getLogger(name).level for name in names}
    try:
        for name in names:
            logging.getLogger(name).setLevel(logging.DEBUG)
        TelegramClient(tg_settings, session=FakeSession(), clock=fake_clock)
        assert logging.getLogger("urllib3.connectionpool").getEffectiveLevel() >= logging.WARNING
        assert logging.getLogger("requests").getEffectiveLevel() >= logging.WARNING
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        TelegramClient(tg_settings, session=FakeSession(), clock=fake_clock)
        assert logging.getLogger("urllib3").level == logging.ERROR  # stricter settings are left alone
    finally:
        for name, level in saved.items():
            logging.getLogger(name).setLevel(level)


def test_close_is_safe_with_or_without_a_closable_session(make, tg_settings, fake_clock):
    client, session = make()
    client.close()
    assert session.closed is True
    TelegramClient(tg_settings, session=object(), clock=fake_clock).close()


# --------------------------------------------------------------------------------------------- 429
def test_rate_limit_sleeps_retry_after_then_succeeds(make, fake_clock):
    client, session = make(err(429, "Too Many Requests: retry after 3", retry_after=3), ok(7))
    assert client.send_message("1001", "hi") == [7]
    assert fake_clock.sleeps == [3.0] and len(session.calls) == 2 and client.last_attempts == 2


def test_retry_after_is_capped_at_sixty_seconds(make, fake_clock):
    client, _ = make(err(429, "Too Many Requests", retry_after=500), ok(1))
    client.send_message("1001", "hi")
    assert fake_clock.sleeps == [60.0]


def test_retry_after_header_is_the_fallback_then_backoff(make, fake_clock):
    client, _ = make(err(429, "Too Many Requests", headers={"Retry-After": "4"}), err(429, "Too Many Requests"), ok(1))
    client.send_message("1001", "hi")
    assert fake_clock.sleeps[0] == 4.0
    assert 1.0 <= fake_clock.sleeps[1] <= 2.0  # no hint at all: exponential backoff for attempt 2


def test_persistent_rate_limit_gives_up_as_retryable(make, fake_clock):
    client, session = make(*[err(429, "Too Many Requests", retry_after=2)] * 4)
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "hi")
    assert info.value.retryable is True and "gave up after 4 attempts" in str(info.value)
    assert len(session.calls) == 4 and fake_clock.sleeps == [2.0, 2.0, 2.0]


# ----------------------------------------------------------------------------- 5xx / timeouts / backoff
def test_server_errors_back_off_exponentially_with_jitter(make, fake_clock):
    client, session = make(err(502, "Bad Gateway"), err(500, "Internal"), err(503, "Unavailable"), ok(5))
    assert client.send_message("1001", "hi") == [5]
    assert len(session.calls) == 4 and len(fake_clock.sleeps) == 3
    for slept, ceiling in zip(fake_clock.sleeps, (1.0, 2.0, 4.0)):
        assert ceiling * 0.5 <= slept <= ceiling


def test_backoff_never_exceeds_thirty_seconds(tg_settings, make, fake_clock):
    many = dataclasses.replace(tg_settings, telegram_max_attempts=10)
    client, _ = make(*[err(500, "x")] * 10, settings=many)
    with pytest.raises(DataUnavailable):
        client.send_message("1001", "hi")
    assert len(fake_clock.sleeps) == 9 and max(fake_clock.sleeps) <= 30.0 and max(fake_clock.sleeps) > 15.0


def test_http_408_is_retryable(make):
    client, session = make(err(408, "Request Timeout"), ok(2))
    assert client.send_message("1001", "hi") == [2] and len(session.calls) == 2


def test_network_failures_exhaust_attempts_without_leaking_the_token(make, fake_clock, caplog):
    caplog.set_level(logging.DEBUG)
    client, session = make(
        requests.exceptions.ReadTimeout(LEAKY),
        requests.exceptions.ConnectionError(LEAKY),
        requests.exceptions.ConnectTimeout(LEAKY),
        OSError(LEAKY),
    )
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "hi")
    exc = info.value
    assert exc.retryable is True and exc.source == "telegram" and "gave up after 4 attempts" in str(exc)
    assert exc.__context__ is None and exc.__cause__ is None
    assert len(session.calls) == 4 and len(fake_clock.sleeps) == 3  # no sleep after the final attempt
    assert_no_secret(str(exc), repr(exc), caplog.text, "".join(traceback.format_exception(exc)))
    assert "timed out" in caplog.text and "ConnectionError" in caplog.text


def test_unknown_session_errors_are_final_and_never_echoed(make, caplog):
    caplog.set_level(logging.DEBUG)
    client, session = make(RuntimeError(LEAKY))
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "hi")
    assert info.value.retryable is False and "RuntimeError" in str(info.value) and len(session.calls) == 1
    assert_no_secret(str(info.value), caplog.text)


def test_unbuildable_request_is_not_retried_and_hides_the_url(make):
    client, session = make(requests.exceptions.InvalidURL(LEAKY))
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "hi")
    assert info.value.retryable is False and len(session.calls) == 1
    assert_no_secret(str(info.value))


def test_malformed_replies_are_retryable_failures(make, fake_clock):
    client, session = make(
        FakeResponse(200, not_json=True), FakeResponse(200, {"result": {}}), FakeResponse(200, ["x"]), ok(9)
    )
    assert client.send_message("1001", "hi") == [9] and len(session.calls) == 4
    broken, _ = make(*[FakeResponse(502, not_json=True)] * 4)
    with pytest.raises(DataUnavailable) as info:
        broken.send_message("1001", "hi")
    assert info.value.retryable is True and "HTTP 502" in str(info.value)


def test_ok_reply_without_a_message_id_still_counts_as_sent(make):
    client, session = make(FakeResponse(200, {"ok": True, "result": True}))
    assert client.send_message("1001", "hi") == [] and len(session.calls) == 1
    odd, _ = make(FakeResponse(200, {"ok": True, "result": {"message_id": "x"}}))
    assert odd.send_message("1001", "hi") == []


# ------------------------------------------------------------------------------------ final failures
@pytest.mark.parametrize(
    "response, expected",
    [
        (err(401, "Unauthorized"), "HTTP 401: Unauthorized (check TELEGRAM_BOT_TOKEN)"),
        (err(403, "Forbidden: bot was blocked by the user"), "HTTP 403: Forbidden: bot was blocked by the user"),
        (err(400, "Bad Request: chat not found"), "HTTP 400: Bad Request: chat not found"),
        (err(404, "Not Found"), "HTTP 404: Not Found"),
        (FakeResponse(400, {"ok": False, "description": "Bad Request: nope"}), "HTTP 400: Bad Request: nope"),
    ],
)
def test_client_errors_are_final_and_not_retried(make, fake_clock, response, expected):
    client, session = make(response)
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "hi")
    assert info.value.retryable is False and str(info.value) == f"[telegram] {expected}"
    assert len(session.calls) == 1 and fake_clock.sleeps == []
    assert info.value.__context__ is None


def test_telegram_descriptions_are_redacted_and_truncated(make):
    client, _ = make(err(400, f"Bad Request: chat {TOKEN} is {'very ' * 100}invalid"))
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "hi")
    assert TOKEN not in str(info.value) and len(str(info.value)) < 260 and str(info.value).endswith("...")


# --------------------------------------------------------------------------------------- parse errors
PARSE_ERROR = "Bad Request: can't parse entities: Can't find end tag corresponding to start tag \"b\""


def test_parse_error_retries_once_as_plain_text(make, fake_clock):
    html_text = '<b>Alpha &amp; <i>beta</i></b> &lt;ok&gt; <a href="https://example.test/a?x=1&amp;y=2">source</a>'
    client, session = make(err(400, PARSE_ERROR), ok(2))
    assert client.send_message("1001", html_text) == [2]
    first, second = (c["json"] for c in session.calls)
    assert first["parse_mode"] == "HTML" and "parse_mode" not in second
    assert second["text"] == "Alpha & beta <ok> source (https://example.test/a?x=1&y=2)"
    assert fake_clock.sleeps == []


def test_parse_error_twice_is_final(make):
    client, session = make(err(400, PARSE_ERROR), err(400, PARSE_ERROR))
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "<b>x")
    assert info.value.retryable is False and len(session.calls) == 2


def test_plain_text_messages_do_not_fall_back(make):
    client, session = make(err(400, PARSE_ERROR))
    with pytest.raises(DataUnavailable):
        client.send_message("1001", "a < b", parse_mode=None)
    assert len(session.calls) == 1 and "parse_mode" not in session.calls[0]["json"]


def test_only_the_rejected_chunk_is_downgraded(make):
    text = "\n".join(f"<b>line {i}</b> " + "x" * 200 for i in range(40))
    client, session = make(ok(1), err(400, PARSE_ERROR), ok(2), ok(3))
    assert client.send_message("1001", text) == [1, 2, 3]
    payloads = [c["json"] for c in session.calls]
    assert "parse_mode" in payloads[0] and "parse_mode" not in payloads[2] and "<b>" not in payloads[2]["text"]
    assert "parse_mode" in payloads[3]


# ----------------------------------------------------------------------------------------- migration
def test_migrate_to_chat_id_resends_and_is_remembered(make):
    client, session = make(
        err(400, "Bad Request: group chat was upgraded to a supergroup chat", migrate_to_chat_id=-1001234567890),
        ok(3),
        ok(4),
    )
    assert client.send_message("1001", "hi") == [3]
    assert [c["json"]["chat_id"] for c in session.calls] == [1001, -1001234567890]
    assert client.migrated_chat_ids == {"1001": "-1001234567890"}
    client.send_message("1001", "again")
    assert session.calls[-1]["json"]["chat_id"] == -1001234567890


def test_migration_is_followed_only_once_per_message(make):
    migrate = err(400, "Bad Request: group chat was upgraded", migrate_to_chat_id=-100222)
    client, session = make(migrate, err(400, "Bad Request: group chat was upgraded", migrate_to_chat_id=-100333))
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", "hi")
    assert info.value.retryable is False and len(session.calls) == 2


# ------------------------------------------------------------------------------------------ chunking
def long_markup(lines=70, width=120):
    body = [f"line {i:03d} " + ("word " * (width // 5)).strip() for i in range(lines)]
    return "<b>Header</b>\n<pre>\n" + "\n".join(body) + "\n</pre>\n<i>footer</i>"


def test_long_messages_split_on_line_boundaries_with_balanced_tags(make):
    text = long_markup()
    client, session = make()
    ids = client.send_message("1001", text)
    chunks = [c["json"]["text"] for c in session.calls]
    assert len(chunks) >= 2 and ids == list(range(1, len(chunks) + 1))
    assert all(len(c) <= MAX_MESSAGE_CHARS for c in chunks) and all(is_balanced(c) for c in chunks)
    assert all(c["json"]["parse_mode"] == "HTML" for c in session.calls)
    assert chunks[1].startswith("<pre>") and chunks[0].endswith("</pre>")  # the open block was closed and reopened
    assert squash("".join(chunks)) == squash(text)
    original_lines = {squash(line) for line in text.split("\n")}
    for chunk in chunks:
        assert all(squash(line) in original_lines for line in chunk.split("\n") if squash(line))


def test_split_message_basics():
    assert split_message("short") == ["short"] and split_message("") == [""] and split_message(None) == [""]
    with pytest.raises(ValueError):
        split_message("x" * 100, limit=10)


def test_split_message_hard_splits_unbroken_text_and_never_cuts_entities():
    word = "x" * 500
    chunks = split_message(word, limit=120)
    assert all(len(c) <= 120 for c in chunks) and "".join(chunks) == word
    entities = "&amp;" * 200
    pieces = split_message(entities, limit=100)
    assert all(len(c) <= 100 for c in pieces) and "".join(pieces) == entities
    assert all(c.count("&") == c.count(";") for c in pieces)


def test_split_message_prefers_spaces_for_long_lines():
    text = " ".join(f"word{i:03d}" for i in range(80))
    chunks = split_message(text, limit=100)
    assert all(len(c) <= 100 for c in chunks) and len(chunks) > 1
    assert all(re.fullmatch(r"(word\d{3} ?)+", c) for c in chunks)
    assert squash(" ".join(chunks)) == squash(text)


def test_split_message_reopens_links_and_nested_tags_in_order():
    link = '<a href="https://x.test/p">'
    text = f"<b><i>{link}" + " ".join(f"alpha{i:02d}" for i in range(60)) + "</a></i></b>"
    chunks = split_message(text, limit=160)
    assert len(chunks) > 1 and all(len(c) <= 160 and is_balanced(c) for c in chunks)
    for chunk in chunks:
        assert chunk.startswith(f"<b><i>{link}") and chunk.endswith("</a></i></b>")
    assert squash("".join(chunks)) == squash(text)


def test_split_message_balances_unclosed_tags_and_drops_stray_closers():
    unclosed = "<b>" + "\n".join(f"line {i:02d} " + "x" * 30 for i in range(20))
    chunks = split_message(unclosed, limit=200)
    assert len(chunks) > 1 and all(is_balanced(c) for c in chunks)
    stray = "\n".join(f"plain {i:02d} </b></i> " + "y" * 30 for i in range(20))
    cleaned = split_message(stray, limit=200)
    assert all("</b>" not in c and "</i>" not in c for c in cleaned) and all(is_balanced(c) for c in cleaned)


def test_split_message_drops_blank_lines_at_chunk_edges():
    text = "\n\n".join("para " + str(i) + " " + "z" * 40 for i in range(12))
    chunks = split_message(text, limit=150)
    assert all(not c.startswith("\n") and not c.endswith("\n") for c in chunks)
    assert squash("".join(chunks)) == squash(text)


def test_split_message_plain_mode_leaves_angle_brackets_alone():
    text = "\n".join(f"a < b > c {i:02d} " + "q" * 40 for i in range(12))
    chunks = split_message(text, limit=150, html_mode=False)
    assert len(chunks) > 1 and "".join(chunks).replace("\n", "") == text.replace("\n", "")
    assert all(len(c) <= 150 for c in chunks)


def test_split_message_survives_pathological_tags():
    huge = '<a href="' + "x" * 300 + '">'
    text = f"{huge}start " + "word " * 100 + "</a> tail"
    chunks = split_message(text, limit=100)
    assert all(len(c) <= 100 for c in chunks)
    assert squash("".join(chunks)).endswith("tail")
    deep = "<b><i><u><s><code>" + ("deep " * 60) + "</code></s></u></i></b>"
    nested = split_message(deep, limit=60)
    assert all(len(c) <= 60 and is_balanced(c) for c in nested)


def test_split_message_tolerates_misnested_and_all_stray_markup():
    text = "\n".join("<b><i>alpha</b> beta</i> " + "z" * 30 for _ in range(10))
    chunks = split_message(text, limit=200)
    assert len(chunks) > 1 and all(len(c) <= 200 and "</i>" not in c for c in chunks)
    stray = "</b>" * 100
    assert split_message(stray, limit=100) == [stray[:100]]  # nothing usable survives: never silently drop


def test_blank_line_after_a_full_chunk_does_not_create_an_empty_chunk():
    text = "a" * 40 + "\n\n" + "b" * 10 + "\n" + "c" * 45
    assert split_message(text, limit=40) == ["a" * 40, "b" * 10, "c" * 40, "c" * 5]


def test_split_message_plain_mode_hard_splits_long_lines():
    text = "x " * 100
    chunks = split_message(text, limit=60, html_mode=False)
    assert len(chunks) > 1 and all(len(c) <= 60 for c in chunks) and squash("".join(chunks)) == squash(text)


def random_markup(rng):
    tokens, stack = [], []
    for _ in range(rng.randint(40, 300)):
        roll = rng.random()
        if roll < 0.55:
            tokens.append(rng.choice(["alpha", "beta", "gamma", "delta", "x" * rng.randint(1, 30)]))
            tokens.append(" ")
        elif roll < 0.7:
            tokens.append("\n")
        elif roll < 0.75:
            tokens.append(rng.choice(["&amp;", "&lt;", "&gt;", "&quot;"]))
        elif roll < 0.87 and len(stack) < 3:
            name = rng.choice(["b", "i", "u", "s", "code", "a"])
            stack.append(name)
            tokens.append('<a href="https://x.test/p">' if name == "a" else f"<{name}>")
        elif stack:
            tokens.append(f"</{stack.pop()}>")
    tokens += [f"</{name}>" for name in reversed(stack)]
    return "".join(tokens)


def test_split_message_invariants_for_random_markup():
    rng = random.Random(20261102)
    for _ in range(300):
        text = random_markup(rng)
        limit = rng.choice([120, 200, 400, 1000])
        chunks = split_message(text, limit)
        assert all(len(c) <= limit for c in chunks), (limit, text)
        assert all(is_balanced(c) for c in chunks), (limit, text)
        assert squash("".join(chunks)) == squash(text), (limit, text)


# ------------------------------------------------------------------------------------- plain text
def test_to_plain_text_strips_tags_unescapes_entities_and_keeps_link_targets():
    assert to_plain_text("<b>a</b> &amp; <i>b</i> &lt;c&gt; &#39;d&#x27;") == "a & b <c> 'd'"
    assert to_plain_text('<a href="https://x.test/?a=1&amp;b=2">label</a>') == "label (https://x.test/?a=1&b=2)"
    assert to_plain_text('<a href="https://x.test/">https://x.test/</a>') == "https://x.test/"
    assert to_plain_text("<a>no href</a> 5 < 6 & more &nosuchentity; <script>x</script>") == (
        "no href 5 < 6 & more &nosuchentity; x"
    )
    assert to_plain_text("") == "" and to_plain_text(None) == ""
    assert to_plain_text("</a>stray") == "stray"


# ------------------------------------------------------------------------------------ other methods
def test_get_me_returns_the_bot_description(make):
    client, session = make(FakeResponse(200, {"ok": True, "result": {"id": 9, "username": "fake_bot"}}))
    assert client.get_me() == {"id": 9, "username": "fake_bot"}
    assert session.calls[0]["url"].endswith(f"/bot{TOKEN}/getMe")
    odd, _ = make(FakeResponse(200, {"ok": True, "result": "nope"}))
    with pytest.raises(DataUnavailable) as info:
        odd.get_me()
    assert info.value.retryable is True


def test_get_updates_sends_offset_and_extends_the_read_timeout(make):
    client, session = make(updates({"update_id": 1}, "junk", {"update_id": 2}))
    assert client.get_updates(offset=5, timeout=30) == [{"update_id": 1}, {"update_id": 2}]
    call = session.calls[0]
    assert call["json"] == {"timeout": 30, "offset": 5} and call["timeout"] == (10.0, 45.0)
    client2, session2 = make(updates())
    assert client2.get_updates() == [] and session2.calls[0]["json"] == {"timeout": 0}
    odd, _ = make(FakeResponse(200, {"ok": True, "result": {"x": 1}}))
    with pytest.raises(DataUnavailable):
        odd.get_updates()


@pytest.mark.parametrize(
    "update, expected",
    [
        (
            {"update_id": 1, "message": {"chat": {"id": 42, "type": "private", "first_name": "Fake", "last_name": "Person"}}},
            {"chat_id": "42", "title": "Fake Person", "type": "private"},
        ),
        (
            {"update_id": 1, "message": {"chat": {"id": -100555, "type": "supergroup", "title": "Fake Desk"}}},
            {"chat_id": "-100555", "title": "Fake Desk", "type": "group"},
        ),
        (
            {"update_id": 1, "message": {"chat": {"id": -77, "type": "group", "title": "Old Group"}}},
            {"chat_id": "-77", "title": "Old Group", "type": "group"},
        ),
        (
            {"update_id": 1, "channel_post": {"chat": {"id": -100888, "type": "channel", "title": "Fake Feed"}}},
            {"chat_id": "-100888", "title": "Fake Feed", "type": "channel"},
        ),
        (
            {"update_id": 1, "my_chat_member": {"chat": {"id": -100999, "type": "supergroup", "title": "Added Here"}}},
            {"chat_id": "-100999", "title": "Added Here", "type": "group"},
        ),
        (
            {"update_id": 1, "callback_query": {"message": {"chat": {"id": 7, "type": "private", "username": "fakeuser"}}}},
            {"chat_id": "7", "title": "@fakeuser", "type": "private"},
        ),
    ],
)
def test_discover_chat_id_understands_every_update_shape(make, update, expected):
    client, _ = make(updates({"update_id": 0, "poll": True}, update))
    assert client.discover_chat_id() == expected


def test_discover_chat_id_polls_until_a_message_arrives(make, fake_clock):
    message = {"update_id": 3, "message": {"chat": {"id": 42, "type": "private", "first_name": "Fake"}}}
    client, session = make(updates(), updates(), updates(message))
    assert client.discover_chat_id(wait_seconds=60, poll_interval=3.0)["chat_id"] == "42"
    assert len(session.calls) == 3 and fake_clock.sleeps == [3.0, 3.0]


def test_discover_chat_id_times_out_with_none(make, fake_clock):
    client, session = make(*[updates() for _ in range(5)])
    assert client.discover_chat_id(wait_seconds=10, poll_interval=3.0) is None
    assert len(session.calls) == 5 and fake_clock.sleeps == [3.0, 3.0, 3.0, 1.0]
    once, once_session = make(updates())
    assert once.discover_chat_id(wait_seconds=0) is None and len(once_session.calls) == 1


def test_discover_chat_id_propagates_auth_errors(make):
    client, _ = make(err(401, "Unauthorized"))
    with pytest.raises(DataUnavailable) as info:
        client.discover_chat_id()
    assert info.value.retryable is False


# --------------------------------------------------------------------------------------- deadlines
def test_deadline_in_the_past_aborts_before_any_request(make, fake_clock):
    client, session = make()
    with pytest.raises(DeadlineExceeded) as info:
        client.send_message("1001", "hi", deadline=fake_clock.now() - timedelta(seconds=1))
    assert isinstance(info.value, DataUnavailable) and info.value.retryable is False
    assert info.value.message_ids == [] and session.calls == []


def test_deadline_stops_retries_instead_of_sleeping_past_it(make, fake_clock):
    client, session = make(err(500, "Internal"))
    start = fake_clock.now()
    with pytest.raises(DeadlineExceeded):
        client.send_message("1001", "hi", deadline=start + timedelta(seconds=0.2))
    assert len(session.calls) == 1 and fake_clock.sleeps == [] and fake_clock.now() == start


def test_deadline_between_chunks_reports_what_was_already_sent(make, fake_clock):
    client, session = make(lambda: (fake_clock.advance(120), ok(1))[1])
    with pytest.raises(DeadlineExceeded) as info:
        client.send_message("1001", long_markup(), deadline=fake_clock.now() + timedelta(seconds=60))
    assert info.value.message_ids == [1] and len(session.calls) == 1


def test_failed_chunk_reports_ids_of_delivered_chunks(make):
    client, _ = make(ok(11), err(403, "Forbidden: bot was kicked"))
    with pytest.raises(DataUnavailable) as info:
        client.send_message("1001", long_markup())
    assert info.value.message_ids == [11] and info.value.retryable is False


def test_last_attempts_counts_http_calls_per_public_call(make):
    client, _ = make(err(500, "x"), ok(1), ok(2))
    client.send_message("1001", "a")
    assert client.last_attempts == 2
    client.send_message("1001", "b")
    assert client.last_attempts == 1
