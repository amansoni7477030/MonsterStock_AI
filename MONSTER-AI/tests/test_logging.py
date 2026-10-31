from __future__ import annotations

import logging

from app.logging_setup import (
    clear_secrets,
    job_context,
    redact,
    register_secret,
    reset_logging,
    setup_logging,
)

TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ0123456789"


def test_telegram_token_in_url_is_redacted():
    text = f"POST https://api.telegram.org/bot{TOKEN}/sendMessage failed"
    assert TOKEN not in redact(text)
    assert "bot<redacted>" in redact(text)


def test_credential_shapes_are_redacted():
    assert "supersecret1" not in redact("api_key=supersecret1&x=1")
    assert "abcdef123456" not in redact('{"Authorization": "Bearer abcdef123456"}')
    assert "AKIAABCDEFGHIJKLMNOP" not in redact("key AKIAABCDEFGHIJKLMNOP used")
    assert "deadbeef" not in redact("https://s3/x?X-Amz-Signature=deadbeef&a=1")
    assert redact("") == ""


def test_registered_secret_values_are_masked_everywhere():
    register_secret("my-very-private-value")
    assert redact("boom my-very-private-value boom") == "boom <redacted> boom"
    clear_secrets()
    assert redact("boom my-very-private-value") == "boom my-very-private-value"
    register_secret("abc")  # too short to be a meaningful secret; ignored
    assert redact("abc") == "abc"


def test_setup_logging_is_idempotent_and_writes_redacted_file(tmp_path):
    for _ in range(3):
        setup_logging(tmp_path, "INFO", console=False, secrets=[TOKEN])
    root = logging.getLogger()
    managed = [h for h in root.handlers if getattr(h, "_monster_managed", False)]
    assert len(managed) == 1
    with job_context("unit"):
        logging.getLogger("test").error("request to /bot%s/x failed: %s", TOKEN, TOKEN)
    for handler in managed:
        handler.flush()
    content = (tmp_path / "monster.log").read_text(encoding="utf-8")
    assert "[unit]" in content and "failed" in content
    assert TOKEN not in content
    reset_logging()


def test_unwritable_log_dir_falls_back_to_console(tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    setup_logging(blocker / "sub", "INFO")  # parent is a file -> mkdir fails
    root = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    reset_logging()
