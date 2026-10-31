"""Test helpers: scripted HTTP sessions and fixture loaders (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def fixture_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status: int = 200, content: bytes | str | dict | list = b"", headers: dict[str, str] | None = None):
        if isinstance(content, (dict, list)):
            content = json.dumps(content).encode()
        elif isinstance(content, str):
            content = content.encode()
        self.status_code = status
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size: int = 65536):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """Plays back scripted responses; an item may be a response, an exception, or a callable(url, kwargs)."""

    def __init__(self, script: list[Any] | Callable[[str, dict], Any] | None = None):
        self.script = script if isinstance(script, list) else []
        self.handler = script if callable(script) else None
        self.calls: list[tuple[str, str, dict]] = []
        self.headers: dict[str, str] = {}
        self.cookies_cleared = 0

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        if self.handler is not None:
            item = self.handler(url, kwargs)
        else:
            if not self.script:
                raise AssertionError(f"unexpected request: {method} {url}")
            item = self.script.pop(0)
        if callable(item) and not isinstance(item, FakeResponse):
            item = item(url, kwargs)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.cookies_cleared += 1


def timeout_error() -> requests.exceptions.Timeout:
    return requests.exceptions.Timeout("simulated timeout for https://secret.example/?api_key=SUPERSECRET123")
