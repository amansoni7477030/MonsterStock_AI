"""Object store tests: AWS SigV4 known answers, key rules, local/HTTP/S3 backends. Hermetic (fake sessions)."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import random
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlsplit

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from app.errors import ConfigError, DataUnavailable
from app.logging_setup import clear_secrets, redact
from database import object_store as osx
from database.object_store import (
    EMPTY_SHA256,
    HttpReadStore,
    LocalDirStore,
    ObjectStore,
    ObjectTooLarge,
    S3Store,
    atomic_write,
    build_artifact_source,
    build_object_store,
    canonical_headers,
    canonical_query_string,
    canonical_request,
    derive_signing_key,
    sign_request,
    string_to_sign,
    uri_encode,
    validate_key,
    validate_prefix,
)

# ------------------------------------------------------------------------------------ fakes / helpers
KEY_ID = "TESTKEYID0001"
SECRET = "s3cr3t/Key+value-0123456789abcdef"


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, *, length=None, fail_after=None, chunk=5):
        self.status_code = status
        merged = {"Content-Length": str(len(body) if length is None else length)} if body or length is not None else {}
        merged.update(headers or {})
        self.headers = CaseInsensitiveDict(merged)
        self._body, self._fail_after, self._chunk = body, fail_after, chunk
        self.closed = False

    def iter_content(self, chunk_size=1):
        sent = 0
        for start in range(0, len(self._body), self._chunk):
            piece = self._body[start:start + self._chunk]
            if self._fail_after is not None and sent + len(piece) > self._fail_after:
                raise requests.exceptions.ChunkedEncodingError("connection broken")
            sent += len(piece)
            yield piece

    def close(self):
        self.closed = True


@dataclass
class Call:
    method: str
    url: str
    headers: dict
    body: bytes | None
    streamed: bool
    kwargs: dict = field(default_factory=dict)


class FakeSession:
    """Plays back queued responses / exceptions / callables and records every request."""

    def __init__(self, *queue):
        self.queue = list(queue)
        self.calls: list[Call] = []
        self.closed = False

    def request(self, method, url, **kwargs):
        data = kwargs.get("data")
        streamed = data is not None and hasattr(data, "read")
        body = None
        if streamed:
            parts = []
            while chunk := data.read(7):
                parts.append(chunk)
            body = b"".join(parts)
        elif data is not None:
            body = bytes(data)
        call = Call(method, url, dict(kwargs.get("headers") or {}), body, streamed, kwargs)
        self.calls.append(call)
        if not self.queue:
            raise AssertionError(f"unexpected request {method} {url}")
        item = self.queue.pop(0)
        if callable(item) and not isinstance(item, FakeResponse):
            item = item(call)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


def error_xml(code: str, message: str = "boom") -> bytes:
    return f"<?xml version='1.0'?><Error><Code>{code}</Code><Message>{message}</Message></Error>".encode()


def s3(session, clock, **kw):
    kw.setdefault("rng", lambda: 0.0)
    return S3Store("https://s3.example.test", "bucket", "auto", KEY_ID, SECRET, "monster-ai/", session=session,
                   clock=clock, **kw)


def http_store(session, clock, base="https://cdn.example.test/art", **kw):
    kw.setdefault("rng", lambda: 0.0)
    return HttpReadStore(base, session=session, clock=clock, **kw)


def check_wire_signature(call: Call, secret: str = SECRET) -> None:
    """Re-derive the signature from the request as it went on the wire and compare."""
    parts = urlsplit(call.url)
    match = re.fullmatch(
        r"AWS4-HMAC-SHA256 Credential=([^/]+)/(\d{8})/([^/]+)/s3/aws4_request, "
        r"SignedHeaders=([a-z0-9;-]+), Signature=([0-9a-f]{64})", call.headers["Authorization"])
    assert match, call.headers["Authorization"]
    credential, _date, region, signed, signature = match.groups()
    lowered = {k.lower(): v for k, v in call.headers.items()}
    names = signed.split(";")
    assert {"host", "x-amz-content-sha256", "x-amz-date"} <= set(names) and names == sorted(names)
    assert lowered["host"] == parts.netloc
    expected = sign_request(
        method=call.method, canonical_uri=parts.path, query=parse_qsl(parts.query, keep_blank_values=True),
        headers={name: lowered[name] for name in names}, payload_hash=lowered["x-amz-content-sha256"],
        access_key_id=credential, secret_access_key=secret, region=region, amz_date=lowered["x-amz-date"])
    assert expected.endswith(f"Signature={signature}")


# --------------------------------------------------------------------------------- SigV4 known answers
AWS_ACCESS = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
AWS_DATE = "20130524T000000Z"
AWS_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sigv4_aws_documentation_get_object_range_example():
    """GET /test.txt with a Range header: every intermediate value from the AWS S3 documentation."""
    headers = {"host": "examplebucket.s3.amazonaws.com", "range": "bytes=0-9", "x-amz-content-sha256": AWS_EMPTY,
               "x-amz-date": AWS_DATE}
    block, signed = canonical_headers(headers)
    assert signed == "host;range;x-amz-content-sha256;x-amz-date"
    request_text = canonical_request("GET", "/test.txt", canonical_query_string({}), block, signed, AWS_EMPTY)
    assert request_text == (
        "GET\n/test.txt\n\nhost:examplebucket.s3.amazonaws.com\nrange:bytes=0-9\n"
        f"x-amz-content-sha256:{AWS_EMPTY}\nx-amz-date:{AWS_DATE}\n\n"
        f"host;range;x-amz-content-sha256;x-amz-date\n{AWS_EMPTY}")
    assert hashlib.sha256(request_text.encode()).hexdigest() == (
        "7344ae5b7ee6c3e7e6b0fe0640412a37625d1fbfff95c48bbb2dc43964946972")
    scope = "20130524/us-east-1/s3/aws4_request"
    to_sign = string_to_sign(AWS_DATE, scope, request_text)
    assert to_sign == ("AWS4-HMAC-SHA256\n20130524T000000Z\n20130524/us-east-1/s3/aws4_request\n"
                       "7344ae5b7ee6c3e7e6b0fe0640412a37625d1fbfff95c48bbb2dc43964946972")
    key = derive_signing_key(AWS_SECRET, "20130524", "us-east-1")
    assert osx.calculate_signature(key, to_sign) == (
        "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")
    header = sign_request(method="GET", canonical_uri="/test.txt", query={}, headers=headers,
                          payload_hash=AWS_EMPTY, access_key_id=AWS_ACCESS, secret_access_key=AWS_SECRET,
                          region="us-east-1", amz_date=AWS_DATE)
    assert header == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")


def test_sigv4_aws_documentation_put_object_example_encodes_the_key():
    body = b"Welcome to Amazon S3."
    digest = hashlib.sha256(body).hexdigest()
    assert digest == "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
    headers = {"date": "Fri, 24 May 2013 00:00:00 GMT", "host": "examplebucket.s3.amazonaws.com",
               "x-amz-content-sha256": digest, "x-amz-date": AWS_DATE, "x-amz-storage-class": "REDUCED_REDUNDANCY"}
    header = sign_request(method="PUT", canonical_uri="/" + uri_encode("test$file.text"), query={}, headers=headers,
                          payload_hash=digest, access_key_id=AWS_ACCESS, secret_access_key=AWS_SECRET,
                          region="us-east-1", amz_date=AWS_DATE)
    assert header.endswith("Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd")


@pytest.mark.parametrize(("query", "signature"), [
    ({"lifecycle": ""}, "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543"),
    ({"max-keys": "2", "prefix": "J"}, "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"),
])
def test_sigv4_aws_documentation_query_string_examples(query, signature):
    headers = {"host": "examplebucket.s3.amazonaws.com", "x-amz-content-sha256": AWS_EMPTY, "x-amz-date": AWS_DATE}
    header = sign_request(method="GET", canonical_uri="/", query=query, headers=headers, payload_hash=AWS_EMPTY,
                          access_key_id=AWS_ACCESS, secret_access_key=AWS_SECRET, region="us-east-1",
                          amz_date=AWS_DATE)
    assert header.endswith(f"Signature={signature}")


@pytest.mark.parametrize(("raw", "keep_slash", "expected"), [
    ("AZaz09-._~", False, "AZaz09-._~"),
    ("a b", False, "a%20b"),
    ("a+b=c&d", False, "a%2Bb%3Dc%26d"),
    ("photos/Jan/sample.jpg", True, "photos/Jan/sample.jpg"),
    ("photos/Jan/sample.jpg", False, "photos%2FJan%2Fsample.jpg"),
    ("test$file.text", True, "test%24file.text"),
    ("*'()!", False, "%2A%27%28%29%21"),
    ("caf\u00e9/\u4e2d", True, "caf%C3%A9/%E4%B8%AD"),
    ("%", False, "%25"),
])
def test_uri_encode_follows_sigv4_rules(raw, keep_slash, expected):
    assert uri_encode(raw, keep_slash=keep_slash) == expected


def test_canonical_query_string_is_sorted_and_encoded():
    assert canonical_query_string({"prefix": "a b/c", "list-type": "2", "continuation-token": "x+y="}) == (
        "continuation-token=x%2By%3D&list-type=2&prefix=a%20b%2Fc")
    assert canonical_query_string([("b", "2"), ("a", "z"), ("a", "1"), ("c", "")]) == "a=1&a=z&b=2&c="
    assert canonical_query_string({}) == ""


def test_canonical_headers_trim_collapse_and_sort():
    block, signed = canonical_headers({"X-Amz-Date": " 20130524T000000Z ", "Host": "h", "Content-Type": "a   b\tc"})
    assert block == "content-type:a b c\nhost:h\nx-amz-date:20130524T000000Z\n"
    assert signed == "content-type;host;x-amz-date"


# ------------------------------------------------------------------------------------------------ keys
@pytest.mark.parametrize("key", [
    "a", "models/final/latest.json", "backups/monster-20261101T020000Z.db.gz", "a_b-c/d.e~f=g@h+i", "A" * 255,
    "date=2026-11-01/part-0.gz",
])
def test_validate_key_accepts_portable_keys(key):
    assert validate_key(key) == key


@pytest.mark.parametrize("key", [
    "", "/abs", "a//b", "a/", "../x", "a/../b", "a/./b", ".", "..", "a\\b", "a\x00b", "a\nb", "a b", "a:b", "C:x",
    ".hidden", "a/.hidden", "trailing.", "con", "nul.txt", "a/COM1", "caf\u00e9", "a" * 1025, "/".join("a" * 20),
    "a%2e%2e", "a*b", "a?b", "a#b", None, 5, b"bytes",
])
def test_validate_key_rejects_unsafe_keys(key):
    with pytest.raises(ValueError, match="invalid object key"):
        validate_key(key)


@pytest.mark.parametrize("prefix", ["", "a", "a/", "a/b/", "a/b/mon", "monster-2026."])
def test_validate_prefix_accepts(prefix):
    assert validate_prefix(prefix) == prefix


@pytest.mark.parametrize("prefix", ["/", "/a", "../", "a//", "a/../b", "a\\b", "a\x00", "a b", ".h", None])
def test_validate_prefix_rejects(prefix):
    with pytest.raises(ValueError, match="invalid prefix"):
        validate_prefix(prefix)


def test_object_store_is_abstract():
    with pytest.raises(TypeError):
        ObjectStore()  # type: ignore[abstract]


# -------------------------------------------------------------------------------------- atomic writes
@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """The Windows-only retry of replace/remove never really sleeps in tests."""
    monkeypatch.setattr(osx, "_replace_sleep", lambda seconds: None)


def test_replace_path_rides_out_transient_access_denied_on_windows(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(osx, "_RETRY_REPLACE", True)
    monkeypatch.setattr(osx, "_replace_sleep", sleeps.append)
    real, attempts = os.replace, {"n": 0}

    def flaky(src, dst):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise PermissionError(13, "Access is denied")  # e.g. a virus scanner holds the file for a moment
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    source, target = tmp_path / "a", tmp_path / "b"
    source.write_text("x")
    osx.replace_path(source, target)
    assert target.read_text() == "x" and sleeps == [0.05, 0.1]


def test_replace_path_is_bounded_and_only_retries_what_can_recover(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(osx, "_RETRY_REPLACE", True)
    monkeypatch.setattr(osx, "_replace_sleep", sleeps.append)

    def denied(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(os, "replace", denied)
    with pytest.raises(PermissionError):
        osx.replace_path(tmp_path / "a", tmp_path / "b")
    assert len(sleeps) == osx._REPLACE_ATTEMPTS - 1  # gives up after a bounded number of attempts
    sleeps.clear()
    (tmp_path / "dir").mkdir()
    with pytest.raises(PermissionError):
        osx.replace_path(tmp_path / "a", tmp_path / "dir")
    assert sleeps == []  # a directory destination is a permanent error
    monkeypatch.setattr(osx, "_RETRY_REPLACE", False)
    with pytest.raises(PermissionError):
        osx.replace_path(tmp_path / "a", tmp_path / "b")
    assert sleeps == []  # POSIX semantics: never retried

    def broken(src, dst):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(osx, "_RETRY_REPLACE", True)
    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError, match="Input/output"):
        osx.replace_path(tmp_path / "a", tmp_path / "b")
    assert sleeps == []  # only sharing violations are retried


def test_remove_tree_is_best_effort_and_never_raises(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(osx, "_replace_sleep", sleeps.append)
    tree = tmp_path / "t"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "f").write_text("x")
    osx.remove_tree(tree)
    assert not tree.exists()
    osx.remove_tree(tree)  # already gone

    tree.mkdir()
    real_rmtree, calls = osx.shutil.rmtree, {"n": 0}

    def flaky(path, ignore_errors=False, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "Access is denied")
        return real_rmtree(path, ignore_errors=ignore_errors, **kwargs)

    monkeypatch.setattr(osx, "_RETRY_REPLACE", True)
    monkeypatch.setattr(osx.shutil, "rmtree", flaky)
    osx.remove_tree(tree)  # recovers on the third attempt
    assert not tree.exists() and sleeps == [0.05, 0.1]

    sleeps.clear()
    tree.mkdir()

    def stuck(path, ignore_errors=False, **kwargs):
        if not ignore_errors:
            raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(osx.shutil, "rmtree", stuck)
    osx.remove_tree(tree)  # a persistent failure is swallowed after the bounded retries
    assert len(sleeps) == osx._REPLACE_ATTEMPTS - 1
    sleeps.clear()
    monkeypatch.setattr(osx, "_RETRY_REPLACE", False)
    osx.remove_tree(tree)
    assert sleeps == []


def test_atomic_write_replaces_only_on_success(tmp_path):
    dest = tmp_path / "sub" / "f.bin"
    with atomic_write(dest) as out:
        out.write(b"one")
    assert dest.read_bytes() == b"one"
    with pytest.raises(RuntimeError), atomic_write(dest) as out:
        out.write(b"partial-write")
        raise RuntimeError("crash")
    assert dest.read_bytes() == b"one"
    assert [p.name for p in dest.parent.iterdir()] == ["f.bin"]  # no temp file left behind


# ------------------------------------------------------------------------------------- LocalDirStore
def test_local_store_roundtrip_bytes_and_files(tmp_path):
    store = LocalDirStore(tmp_path / "root")
    store.put("a/b/one.bin", b"hello")
    source = tmp_path / "src.bin"
    source.write_bytes(b"x" * 600_000)
    store.put("a/two.bin", source, content_type="application/zip")
    assert store.get("a/b/one.bin") == b"hello"
    assert store.exists("a/two.bin") and not store.exists("a/missing.bin") and not store.exists("a")
    dest = tmp_path / "out" / "copy.bin"
    assert store.get_to_file("a/two.bin", dest) == 600_000 and dest.read_bytes() == source.read_bytes()
    listing = store.list()
    assert [(o.key, o.size) for o in listing] == [("a/b/one.bin", 5), ("a/two.bin", 600_000)]
    assert listing[0].last_modified is not None and listing[0].last_modified.tzinfo is not None
    assert [o.key for o in store.list("a/b/")] == ["a/b/one.bin"]
    assert [o.key for o in store.list("a/t")] == ["a/two.bin"]
    assert store.list("zzz/") == []
    store.delete("a/b/one.bin")
    store.delete("a/b/one.bin")  # idempotent
    assert not store.exists("a/b/one.bin")
    store.close()


def test_local_store_overwrite_and_missing_key(tmp_path):
    store = LocalDirStore(tmp_path)
    store.put("k", b"1")
    store.put("k", b"22")
    assert store.get("k") == b"22"
    with pytest.raises(KeyError):
        store.get("nope")
    with pytest.raises(KeyError):
        store.get_to_file("nope", tmp_path / "d.bin")
    assert not (tmp_path / "d.bin").exists()


def test_local_store_size_limits(tmp_path):
    store = LocalDirStore(tmp_path)
    store.put("big", b"x" * 1000)
    with pytest.raises(ObjectTooLarge) as err:
        store.get("big", max_bytes=999)
    assert err.value.retryable is False and isinstance(err.value, DataUnavailable)
    dest = tmp_path / "out.bin"
    with pytest.raises(ObjectTooLarge):
        store.get_to_file("big", dest, max_bytes=10)
    assert not dest.exists() and not list(tmp_path.glob(".out.bin*"))
    assert store.get("big", max_bytes=1000) == b"x" * 1000


@pytest.mark.parametrize("key", ["../evil", "/etc/passwd", "a\\..\\b", "a\x00b", "a/../../b", "C:evil", ".."])
def test_local_store_rejects_traversal_everywhere(tmp_path, key):
    root = tmp_path / "root"
    store = LocalDirStore(root)
    for call in (lambda: store.put(key, b"x"), lambda: store.get(key), lambda: store.exists(key),
                 lambda: store.delete(key), lambda: store.get_to_file(key, tmp_path / "o")):
        with pytest.raises(ValueError, match="invalid object key"):
            call()
    assert not (tmp_path / "evil").exists() and list(root.iterdir()) == []


def test_local_store_does_not_follow_symlinks_out_of_the_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    root = tmp_path / "root"
    store = LocalDirStore(root)
    try:
        os.symlink(outside, root / "link", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    with pytest.raises(ValueError, match="outside"):
        store.get("link/secret.txt")
    with pytest.raises(ValueError, match="outside"):
        store.put("link/new.txt", b"x")
    assert not (outside / "new.txt").exists()


def test_local_store_rejects_paths_that_resolve_outside_even_without_real_symlinks(tmp_path, monkeypatch):
    store = LocalDirStore(tmp_path / "root")
    real_resolve = type(tmp_path).resolve

    def resolving_outside(self, strict=False):
        if self.name == "secret.txt":
            return tmp_path / "outside" / "secret.txt"  # what a symlink pointing out of the root would yield
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(type(tmp_path), "resolve", resolving_outside)
    with pytest.raises(ValueError, match="outside"):
        store.get("sub/secret.txt")


def test_local_store_list_skips_temp_hidden_and_odd_files(tmp_path):
    store = LocalDirStore(tmp_path)
    store.put("keep.txt", b"1")
    (tmp_path / ".keep.txt.abc.part").write_bytes(b"temp")
    (tmp_path / ".hidden").write_bytes(b"h")
    (tmp_path / ".hiddendir").mkdir()
    (tmp_path / ".hiddendir" / "f").write_bytes(b"x")
    (tmp_path / "has space.txt").write_bytes(b"x")
    assert [o.key for o in store.list()] == ["keep.txt"]


def test_local_store_put_argument_errors(tmp_path):
    store = LocalDirStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.put("k", tmp_path / "missing.bin")
    with pytest.raises(TypeError):
        store.put("k", "not bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="content type"):
        store.put("k", b"x", content_type="bad\nheader")
    store.put("k", bytearray(b"ab"))
    assert store.get("k") == b"ab"


def test_put_file_is_a_streaming_shorthand_for_every_store(tmp_path, fake_clock):
    source = tmp_path / "backup.db.gz"
    source.write_bytes(b"backup-bytes" * 100)
    local = LocalDirStore(tmp_path / "root")
    local.put_file("backups/backup.db.gz", str(source))  # str paths are accepted too
    assert local.get("backups/backup.db.gz") == source.read_bytes()
    session = FakeSession(FakeResponse(200))
    s3(session, fake_clock).put_file("backups/backup.db.gz", source, content_type="application/gzip")
    call = session.calls[0]
    assert call.streamed and call.body == source.read_bytes() and call.headers["Content-Type"] == "application/gzip"
    assert call.url.endswith("/bucket/monster-ai/backups/backup.db.gz")  # keys are relative to the store prefix


def test_local_store_write_failure_is_data_unavailable_and_keeps_old_object(tmp_path, monkeypatch):
    store = LocalDirStore(tmp_path)
    store.put("obj.bin", b"old")
    real = os.replace

    def failing(src, dst):
        if str(dst).endswith("obj.bin"):
            raise OSError(28, "No space left on device")
        return real(src, dst)

    monkeypatch.setattr(osx.os, "replace", failing)
    with pytest.raises(DataUnavailable, match="cannot write") as err:
        store.put("obj.bin", b"new")
    assert err.value.retryable is False
    monkeypatch.undo()
    assert store.get("obj.bin") == b"old" and [o.key for o in store.list()] == ["obj.bin"]
    assert not list(tmp_path.glob(".obj.bin*"))


def test_local_store_delete_of_a_directory_is_an_error(tmp_path):
    store = LocalDirStore(tmp_path)
    (tmp_path / "dir").mkdir()
    with pytest.raises(DataUnavailable, match="cannot delete"):
        store.delete("dir")


def test_local_store_get_on_unreadable_file_maps_to_data_unavailable(tmp_path, monkeypatch):
    store = LocalDirStore(tmp_path)
    store.put("k", b"1")
    real_open = open

    def deny(path, *args, **kwargs):
        if str(path).endswith("k"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(osx, "open", deny, raising=False)
    with pytest.raises(DataUnavailable, match="cannot read"):
        store.get("k")
    with pytest.raises(DataUnavailable, match="cannot copy"):
        store.get_to_file("k", tmp_path / "out.bin")


# ------------------------------------------------------------------------------------- HttpReadStore
def test_http_store_reads_with_hardened_requests(tmp_path, fake_clock):
    body = b"payload-bytes" * 10
    session = FakeSession(FakeResponse(200, body), FakeResponse(200, body), FakeResponse(200))
    store = http_store(session, fake_clock)
    assert store.get("models/final/latest.json") == body
    assert store.get_to_file("models/final/v1.zip", tmp_path / "d" / "x.zip") == len(body)
    assert store.exists("models/final/latest.json") is True
    assert [c.method for c in session.calls] == ["GET", "GET", "HEAD"]
    first = session.calls[0]
    assert first.url == "https://cdn.example.test/art/models/final/latest.json"
    assert first.kwargs["allow_redirects"] is False and first.kwargs["stream"] is True
    assert first.kwargs["timeout"] == (10.0, 60.0) and first.headers["Accept-Encoding"] == "identity"
    assert not any(name.lower() == "authorization" for name in first.headers)
    assert repr(store).startswith("HttpReadStore(") and (tmp_path / "d" / "x.zip").read_bytes() == body


def test_http_store_missing_denied_and_not_listable(tmp_path, fake_clock):
    session = FakeSession(FakeResponse(404), FakeResponse(404), FakeResponse(410), FakeResponse(403),
                          FakeResponse(403))
    store = http_store(session, fake_clock)
    with pytest.raises(KeyError):
        store.get("a")
    with pytest.raises(KeyError):
        store.get_to_file("a", tmp_path / "x")
    assert store.exists("a") is False and not (tmp_path / "x").exists()
    with pytest.raises(DataUnavailable, match="access denied") as err:
        store.get("a")
    assert err.value.retryable is False
    with pytest.raises(DataUnavailable, match="access denied"):
        store.exists("a")
    for call in (lambda: store.put("a", b"x"), lambda: store.delete("a"), lambda: store.list()):
        with pytest.raises(NotImplementedError):
            call()


@pytest.mark.parametrize("url", [
    "http://cdn.example.test/x", "ftp://cdn.example.test", "https://user:pw@cdn.example.test/x",
    "https://cdn.example.test/x?token=1", "https://cdn.example.test/x#frag", "not a url", "", "https://",
    "https://cdn.example.test:notaport/x", "https://caf\u00e9.example.test",
])
def test_http_store_rejects_unsafe_base_urls(url):
    with pytest.raises(ConfigError):
        HttpReadStore(url, session=FakeSession())


@pytest.mark.parametrize("url", ["http://localhost:8000/art", "http://127.0.0.1/art", "http://[::1]:9/art"])
def test_http_store_allows_plain_http_for_loopback_only(url):
    assert HttpReadStore(url, session=FakeSession())._base.startswith("http://")


def test_http_store_follows_limited_safe_redirects(fake_clock):
    hop = FakeResponse(302, headers={"Location": "https://other.example.test/blob"})
    session = FakeSession(hop, FakeResponse(200, b"ok"))
    assert http_store(session, fake_clock).get("a") == b"ok"
    assert session.calls[1].url == "https://other.example.test/blob" and hop.closed

    relative = FakeSession(FakeResponse(301, headers={"Location": "/elsewhere/a"}), FakeResponse(200, b"ok"))
    assert http_store(relative, fake_clock).get("a") == b"ok"
    assert relative.calls[1].url == "https://cdn.example.test/elsewhere/a"

    loop = FakeSession(*[FakeResponse(302, headers={"Location": "https://cdn.example.test/again"}) for _ in range(5)])
    with pytest.raises(DataUnavailable, match="too many redirects"):
        http_store(loop, fake_clock, max_redirects=3).get("a")
    assert len(loop.calls) == 4


@pytest.mark.parametrize("location", [
    "http://cdn.example.test/x", "https://127.0.0.1/x", "https://localhost/x", "https://10.0.0.5/x",
    "https://169.254.169.254/latest", "https://user:pw@cdn.example.test/x", "file:///etc/passwd",
    "https://[::1/unbalanced", "https://app.localhost/x",
])
def test_http_store_refuses_redirects_to_unsafe_targets(fake_clock, location):
    session = FakeSession(FakeResponse(302, headers={"Location": location}))
    with pytest.raises(DataUnavailable, match="disallowed"):
        http_store(session, fake_clock).get("a")
    assert len(session.calls) == 1


def test_http_store_redirect_without_location(fake_clock):
    with pytest.raises(DataUnavailable, match="redirect"):
        http_store(FakeSession(FakeResponse(302)), fake_clock).get("a")


def test_http_store_does_not_treat_other_3xx_as_success(fake_clock):
    with pytest.raises(DataUnavailable, match=r"unexpected redirect \(HTTP 304\)") as err:
        http_store(FakeSession(FakeResponse(304)), fake_clock).get("a")
    assert err.value.retryable is False


def test_streamed_downloads_without_content_length_are_still_bounded(tmp_path, fake_clock):
    response = FakeResponse(200, b"x" * 100)
    response.headers.pop("Content-Length")
    with pytest.raises(ObjectTooLarge, match="exceeds"):
        http_store(FakeSession(response), fake_clock).get_to_file("a", tmp_path / "o.bin", max_bytes=30)
    assert list(tmp_path.iterdir()) == []


def test_unwritable_destination_is_reported_as_data_unavailable(tmp_path, fake_clock):
    blocker = tmp_path / "dest"
    blocker.mkdir()  # a directory cannot be replaced by the downloaded file
    with pytest.raises(DataUnavailable, match="cannot write dest") as err:
        http_store(FakeSession(FakeResponse(200, b"payload")), fake_clock).get_to_file("a", blocker)
    assert err.value.retryable is False and sorted(p.name for p in tmp_path.iterdir()) == ["dest"]


def test_local_store_survives_a_file_vanishing_between_check_and_open(tmp_path, monkeypatch):
    store = LocalDirStore(tmp_path)
    store.put("k", b"1")
    real_open = open

    def vanish(path, *args, **kwargs):
        if str(path).endswith("k"):
            raise FileNotFoundError(2, "gone")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(osx, "open", vanish, raising=False)
    with pytest.raises(KeyError):
        store.get("k")
    with pytest.raises(KeyError):
        store.get_to_file("k", tmp_path / "out.bin")


def test_http_store_bounds_download_size(tmp_path, fake_clock):
    session = FakeSession(FakeResponse(200, b"x" * 50), FakeResponse(200, b"x" * 50),
                          FakeResponse(200, b"x" * 50, headers={"Content-Length": "5000"}))
    store = http_store(session, fake_clock, max_bytes=40)
    with pytest.raises(ObjectTooLarge):
        store.get("a")  # declared larger than the store-wide cap
    with pytest.raises(ObjectTooLarge):
        store.get_to_file("a", tmp_path / "o", max_bytes=10_000)  # the store cap wins over the per-call value
    with pytest.raises(ObjectTooLarge):
        store.get("a")
    chunked = FakeResponse(200, b"x" * 50)
    chunked.headers.pop("Content-Length")  # no declared size: the limit is enforced while streaming
    with pytest.raises(ObjectTooLarge, match="exceeds"):
        http_store(FakeSession(chunked), fake_clock, max_bytes=40).get("a")
    assert not list(tmp_path.glob("o*")) and not list(tmp_path.glob(".o*"))


def test_http_store_retries_transient_failures_with_the_injected_clock(fake_clock):
    session = FakeSession(FakeResponse(503), requests.exceptions.ConnectionError("reset"), FakeResponse(200, b"ok"))
    assert http_store(session, fake_clock).get("a") == b"ok"
    assert fake_clock.sleeps == [0.25, 0.5]

    dead = FakeSession(*[requests.exceptions.Timeout("slow") for _ in range(3)])
    with pytest.raises(DataUnavailable, match="failed after 3 attempt") as err:
        http_store(dead, fake_clock).get("a")
    assert err.value.retryable is True and len(dead.calls) == 3


def test_http_store_exists_head_and_decoded_content_length(fake_clock):
    session = FakeSession(FakeResponse(200, b"", headers={"Content-Length": "9"}))
    assert http_store(session, fake_clock).exists("a")
    gz = FakeResponse(200, b"x" * 20, headers={"Content-Encoding": "gzip", "Content-Length": "5"})
    assert http_store(FakeSession(gz), fake_clock).get("a") == b"x" * 20  # encoded length is not comparable


def test_artifact_source_factories(settings):
    assert osx._artifact_base_url("https://cdn.example.test/x/") == "https://cdn.example.test/x/"
    assert osx._artifact_base_url("https://cdn.example.test/x/models/final/latest.json") == (
        "https://cdn.example.test/x")
    assert osx._artifact_base_url("https://cdn.example.test/manifest.json?v=1") == "https://cdn.example.test"
    assert build_artifact_source(settings) is None
    store = build_artifact_source(dataclasses.replace(settings, model_artifact_url="https://cdn.example.test/art/"))
    assert isinstance(store, HttpReadStore) and store._base == "https://cdn.example.test/art/"
    configured = dataclasses.replace(settings, s3_bucket="bucket", s3_access_key_id=KEY_ID, s3_secret_access_key=SECRET)
    assert isinstance(build_artifact_source(configured), S3Store)


# ------------------------------------------------------------------------------------------ S3 setup
def test_s3_addressing_styles(fake_clock):
    session = FakeSession(FakeResponse(200), FakeResponse(200), FakeResponse(200), FakeResponse(200))
    s3(session, fake_clock).get("k/v.bin")  # custom endpoint: path style
    call = session.calls[0]
    assert call.url == "https://s3.example.test/bucket/monster-ai/k/v.bin" and call.headers["Host"] == "s3.example.test"

    aws = S3Store("", "my-bucket", "eu-west-1", KEY_ID, SECRET, "p", session=session, clock=fake_clock)
    aws.get("k")  # AWS: virtual hosted
    assert session.calls[1].url == "https://my-bucket.s3.eu-west-1.amazonaws.com/p/k"

    dotted = S3Store(None, "my.bucket", "us-east-1", KEY_ID, SECRET, session=session, clock=fake_clock)
    dotted.get("k")  # dotted bucket names break the wildcard certificate: path style
    assert session.calls[2].url == "https://s3.us-east-1.amazonaws.com/my.bucket/k"

    china = S3Store("", "bucket1", "cn-north-1", KEY_ID, SECRET, session=session, clock=fake_clock)
    china.get("k")
    assert urlsplit(session.calls[3].url).netloc == "bucket1.s3.cn-north-1.amazonaws.com.cn"
    for call in session.calls:
        check_wire_signature(call)


def test_s3_host_header_carries_non_default_ports(fake_clock):
    session = FakeSession(FakeResponse(200), FakeResponse(200), FakeResponse(200))
    S3Store("https://minio.example.test:9000", "bucket", "us-east-1", KEY_ID, SECRET, session=session,
            clock=fake_clock).get("k")
    S3Store("https://minio.example.test:443/", "bucket", "us-east-1", KEY_ID, SECRET, session=session,
            clock=fake_clock).get("k")
    S3Store("http://127.0.0.1:9000", "bucket", "us-east-1", KEY_ID, SECRET, session=session, clock=fake_clock).get("k")
    assert [c.headers["Host"] for c in session.calls] == ["minio.example.test:9000", "minio.example.test",
                                                           "127.0.0.1:9000"]
    for call in session.calls:
        check_wire_signature(call)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"endpoint_url": "http://s3.example.test"}, "https"),
    ({"endpoint_url": "https://s3.example.test/minio"}, "path"),
    ({"endpoint_url": "https://u:p@s3.example.test"}, "credentials"),
    ({"endpoint_url": "ftp://s3.example.test"}, "http"),
    ({"bucket": "a/b"}, "bucket"),
    ({"bucket": ""}, "bucket"),
    ({"bucket": "ab"}, "bucket"),
    ({"region": "US East"}, "region"),
    ({"access_key_id": "has space"}, "access key"),
    ({"access_key_id": "a/b"}, "access key"),
    ({"access_key_id": ""}, "access key"),
    ({"secret_access_key": ""}, "secret"),
    ({"prefix": "a/../b"}, "prefix"),
    ({"prefix": "bad prefix"}, "prefix"),
])
def test_s3_configuration_is_validated_without_echoing_values(kwargs, message):
    args = {"endpoint_url": "https://s3.example.test", "bucket": "bucket", "region": "auto",
            "access_key_id": KEY_ID, "secret_access_key": SECRET, "prefix": "", **kwargs}
    with pytest.raises(ConfigError, match=message) as err:
        S3Store(args["endpoint_url"], args["bucket"], args["region"], args["access_key_id"],
                args["secret_access_key"], args["prefix"], session=FakeSession())
    assert SECRET not in str(err.value)


def test_s3_prefix_normalisation_and_repr(fake_clock):
    session = FakeSession(FakeResponse(200), FakeResponse(200))
    S3Store("https://s3.example.test", "bucket", "auto", KEY_ID, SECRET, "/a/b//", session=session,
            clock=fake_clock).get("k")
    S3Store("https://s3.example.test", "bucket", "auto", KEY_ID, SECRET, "", session=session, clock=fake_clock).get("k")
    assert session.calls[0].url.endswith("/bucket/a/b/k") and session.calls[1].url.endswith("/bucket/k")
    store = s3(FakeSession(), fake_clock)
    assert SECRET not in repr(store) and KEY_ID not in repr(store) and "bucket" in repr(store)


def test_s3_registers_its_secret_for_log_redaction(fake_clock):
    s3(FakeSession(), fake_clock)
    assert redact(f"oops {SECRET} oops") == "oops <redacted> oops"


# ------------------------------------------------------------------------------------ S3 operations
def test_s3_signs_with_the_injected_clock_and_never_sends_the_secret(fake_clock):
    session = FakeSession(FakeResponse(200, b"data"))
    assert s3(session, fake_clock).get("models/final/latest.json") == b"data"
    call = session.calls[0]
    assert call.headers["x-amz-date"] == "20261102T030000Z"
    assert call.headers["x-amz-content-sha256"] == EMPTY_SHA256
    assert "Credential=" + KEY_ID + "/20261102/auto/s3/aws4_request" in call.headers["Authorization"]
    assert SECRET not in " ".join(call.headers.values()) and SECRET not in call.url
    assert callable(call.kwargs["auth"]) and call.kwargs["auth"]("request") == "request"  # netrc disabled
    assert call.kwargs["allow_redirects"] is False and call.kwargs["stream"] is True
    check_wire_signature(call)


def test_s3_put_bytes_signs_payload_hash_and_content_type(fake_clock):
    session = FakeSession(FakeResponse(200))
    s3(session, fake_clock).put("a/b.json", b'{"x": 1}', content_type="application/json")
    call = session.calls[0]
    assert call.method == "PUT" and call.body == b'{"x": 1}' and not call.streamed
    assert call.headers["x-amz-content-sha256"] == hashlib.sha256(b'{"x": 1}').hexdigest()
    assert call.headers["Content-Type"] == "application/json"
    assert "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date" in call.headers["Authorization"]
    check_wire_signature(call)


class SpyFile:
    """File wrapper that records how much is requested per read() (to prove nothing reads the whole file)."""

    reads: list = []

    def __init__(self, handle):
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()

    def read(self, size=-1):
        SpyFile.reads.append(size)
        return self._handle.read(size)


def test_s3_put_streams_a_file_with_a_chunked_precomputed_hash(tmp_path, fake_clock, monkeypatch):
    payload = bytes(range(256)) * 3000 + b"tail"
    source = tmp_path / "big.bin"
    source.write_bytes(payload)
    monkeypatch.setattr(osx, "HASH_CHUNK", 1000)
    real_open = open
    SpyFile.reads = []
    monkeypatch.setattr(osx, "open", lambda path, mode="r": SpyFile(real_open(path, mode)), raising=False)
    session = FakeSession(FakeResponse(200))
    s3(session, fake_clock).put("backups/big.bin", source)
    call = session.calls[0]
    assert call.streamed and call.body == payload  # the body is a file object, not a bytes copy
    assert call.headers["x-amz-content-sha256"] == hashlib.sha256(payload).hexdigest()
    hashing_reads = [size for size in SpyFile.reads if size == 1000]
    assert len(hashing_reads) >= len(payload) // 1000  # hashed piece by piece
    assert all(size in (1000, 7) for size in SpyFile.reads)  # never an unbounded read() of the whole file
    check_wire_signature(call)


def test_s3_put_empty_file_is_sent_as_an_empty_body(tmp_path, fake_clock):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    session = FakeSession(FakeResponse(200))
    s3(session, fake_clock).put("e", empty)
    call = session.calls[0]
    assert call.body == b"" and not call.streamed and call.headers["x-amz-content-sha256"] == EMPTY_SHA256


def test_s3_put_retries_reopen_the_file_and_resign(tmp_path, fake_clock):
    source = tmp_path / "f.bin"
    source.write_bytes(b"0123456789" * 50)
    session = FakeSession(FakeResponse(503), FakeResponse(200))
    s3(session, fake_clock, backoff_base=4.0).put("f", source)
    first, second = session.calls
    assert first.body == second.body == source.read_bytes()  # the second attempt re-reads from the start
    assert first.headers["x-amz-date"] != second.headers["x-amz-date"]  # re-signed after the 2 s backoff
    check_wire_signature(first)
    check_wire_signature(second)


def test_s3_put_argument_errors(tmp_path, fake_clock, monkeypatch):
    store = s3(FakeSession(), fake_clock)
    with pytest.raises(FileNotFoundError):
        store.put("k", tmp_path / "missing")
    with pytest.raises(ValueError, match="invalid object key"):
        store.put("../k", b"x")
    with pytest.raises(ValueError, match="content type"):
        store.put("k", b"x", content_type="a\r\nX-Evil: 1")
    source = tmp_path / "f"
    source.write_bytes(b"x")
    monkeypatch.setattr(osx, "MAX_SINGLE_PUT", 0)
    with pytest.raises(ValueError, match="too large"):
        store.put("k", source)


def test_s3_forbidden_is_not_retried_and_leaks_nothing(fake_clock, caplog):
    leaky = error_xml("SignatureDoesNotMatch", f"The request signature does not match {SECRET} {KEY_ID}")
    session = FakeSession(FakeResponse(403, leaky))
    with caplog.at_level(logging.DEBUG), pytest.raises(DataUnavailable) as err:
        s3(session, fake_clock).put("k", b"data")
    assert len(session.calls) == 1 and fake_clock.sleeps == []
    assert err.value.retryable is False and "SignatureDoesNotMatch" in str(err.value)
    signature = re.search(r"Signature=([0-9a-f]{64})", session.calls[0].headers["Authorization"]).group(1)
    for text in (str(err.value), caplog.text):
        assert SECRET not in text and signature not in text and KEY_ID not in text
        assert "The request signature does not match" not in text  # the provider's free text is never echoed


def test_s3_get_missing_and_missing_bucket(tmp_path, fake_clock):
    session = FakeSession(FakeResponse(404, error_xml("NoSuchKey")), FakeResponse(404),
                          FakeResponse(404, error_xml("NoSuchBucket")))
    store = s3(session, fake_clock)
    with pytest.raises(KeyError):
        store.get("a")
    with pytest.raises(KeyError):
        store.get_to_file("a", tmp_path / "x")
    with pytest.raises(DataUnavailable, match="bucket not found") as err:
        store.get("a")
    assert err.value.retryable is False and not (tmp_path / "x").exists()


def test_s3_get_to_file_streams_and_is_atomic(tmp_path, fake_clock):
    payload = b"model-bytes-" * 5000
    session = FakeSession(FakeResponse(200, payload, chunk=997))
    dest = tmp_path / "dl" / "model.bin"
    assert s3(session, fake_clock).get_to_file("m", dest) == len(payload)
    assert dest.read_bytes() == payload and [p.name for p in dest.parent.iterdir()] == ["model.bin"]


def test_s3_truncated_download_is_retried_then_fails_cleanly(tmp_path, fake_clock):
    dest = tmp_path / "model.bin"
    dest.write_bytes(b"previous good file")
    short = [FakeResponse(200, b"x" * 40, length=100) for _ in range(3)]
    with pytest.raises(DataUnavailable, match="truncated") as err:
        s3(FakeSession(*short), fake_clock, max_attempts=3).get_to_file("m", dest)
    assert err.value.retryable is True
    assert dest.read_bytes() == b"previous good file" and [p.name for p in tmp_path.iterdir()] == ["model.bin"]

    recovering = FakeSession(FakeResponse(200, b"x" * 40, length=100), FakeResponse(200, b"y" * 100))
    assert s3(recovering, fake_clock).get_to_file("m", dest) == 100 and dest.read_bytes() == b"y" * 100


def test_s3_connection_dropped_mid_body_is_retried(tmp_path, fake_clock):
    broken = FakeResponse(200, b"z" * 100, fail_after=30)
    session = FakeSession(broken, FakeResponse(200, b"z" * 100))
    assert s3(session, fake_clock).get("m") == b"z" * 100
    assert broken.closed and len(session.calls) == 2


def test_s3_download_size_limits(tmp_path, fake_clock):
    store = s3(FakeSession(FakeResponse(200, b"x" * 100), FakeResponse(200, b"x" * 100)), fake_clock)
    with pytest.raises(ObjectTooLarge):
        store.get("a", max_bytes=50)
    with pytest.raises(ObjectTooLarge):
        store.get_to_file("a", tmp_path / "o", max_bytes=50)
    assert list(tmp_path.iterdir()) == []


def test_s3_exists_and_delete(fake_clock):
    session = FakeSession(FakeResponse(200), FakeResponse(404), FakeResponse(403, error_xml("AccessDenied")),
                          FakeResponse(204), FakeResponse(404), FakeResponse(500), FakeResponse(204))
    store = s3(session, fake_clock)
    assert store.exists("a") is True and store.exists("a") is False
    with pytest.raises(DataUnavailable, match="access denied"):
        store.exists("a")
    store.delete("a")
    store.delete("a")  # a missing object is fine
    store.delete("a")  # 500 then 204: retried
    assert [c.method for c in session.calls] == ["HEAD", "HEAD", "HEAD", "DELETE", "DELETE", "DELETE", "DELETE"]
    for call in session.calls:
        check_wire_signature(call)


# ------------------------------------------------------------------------------------------ S3 listing
def listing_xml(entries, *, truncated=False, token=None, encoded=True, namespace=True):
    ns = ' xmlns="http://s3.amazonaws.com/doc/2006-03-01/"' if namespace else ""
    contents = "".join(
        f"<Contents><Key>{key}</Key><LastModified>2026-11-01T02:00:00.000Z</LastModified>"
        f'<ETag>"e"</ETag><Size>{size}</Size><StorageClass>STANDARD</StorageClass></Contents>'
        for key, size in entries)
    extra = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    kind = "<EncodingType>url</EncodingType>" if encoded else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?><ListBucketResult{ns}><Name>bucket</Name>{kind}'
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>{extra}{contents}</ListBucketResult>"
            ).encode()


def test_s3_list_paginates_strips_prefix_and_parses_entries(fake_clock):
    pages = [
        FakeResponse(200, listing_xml([("monster-ai/backups/b.db.gz", 20), ("monster-ai/backups/a%2Bb.txt", 1)],
                                      truncated=True, token="T1")),
        FakeResponse(200, listing_xml([("monster-ai/backups/c.db.gz", 30), ("monster-ai/backups/with%20space", 9),
                                       ("monster-ai/backups/", 0), ("elsewhere/x", 5)], truncated=True, token="T2")),
        FakeResponse(200, listing_xml([("monster-ai/backups/a.db.gz", 10)])),
    ]
    session = FakeSession(*pages)
    result = s3(session, fake_clock).list("backups/")
    assert [(o.key, o.size) for o in result] == [
        ("backups/a+b.txt", 1), ("backups/a.db.gz", 10), ("backups/b.db.gz", 20), ("backups/c.db.gz", 30)]
    assert result[0].last_modified.year == 2026 and result[0].last_modified.utcoffset().total_seconds() == 0
    queries = [dict(parse_qsl(urlsplit(c.url).query, keep_blank_values=True)) for c in session.calls]
    assert [q.get("continuation-token") for q in queries] == [None, "T1", "T2"]
    assert all(q["list-type"] == "2" and q["encoding-type"] == "url" and q["max-keys"] == "1000" for q in queries)
    assert all(q["prefix"] == "monster-ai/backups/" for q in queries)
    assert urlsplit(session.calls[0].url).path == "/bucket"
    for call in session.calls:
        check_wire_signature(call)


def test_s3_list_without_prefix_namespace_or_encoding(fake_clock):
    plain = listing_xml([("monster-ai/x.txt", 3)], encoded=False, namespace=False)
    session = FakeSession(FakeResponse(200, plain))
    assert [o.key for o in s3(session, fake_clock).list()] == ["x.txt"]
    assert dict(parse_qsl(urlsplit(session.calls[0].url).query))["prefix"] == "monster-ai/"
    bare = S3Store("https://s3.example.test", "bucket", "auto", KEY_ID, SECRET, "", session=FakeSession(
        FakeResponse(200, listing_xml([]))), clock=fake_clock)
    assert bare.list() == [] and "prefix" not in urlsplit(bare._session.calls[0].url).query


def test_s3_list_aws_virtual_hosted_uses_the_root_path(fake_clock):
    session = FakeSession(FakeResponse(200, listing_xml([])))
    S3Store("", "my-bucket", "us-east-1", KEY_ID, SECRET, "p", session=session, clock=fake_clock).list()
    assert urlsplit(session.calls[0].url).path == "/"
    check_wire_signature(session.calls[0])


def test_s3_list_rejects_dtds_entities_and_garbage(fake_clock):
    bomb = (b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;">]>'
            b"<ListBucketResult>&lol2;</ListBucketResult>")
    for body in (bomb, b"<ListBucketResult><unclosed>", b"<Error><Code>Odd</Code></Error>", b"\xff\xfe<x/>"):
        with pytest.raises(DataUnavailable, match="unexpected LIST response") as err:
            s3(FakeSession(FakeResponse(200, body)), fake_clock).list()
        assert err.value.retryable is False


def test_s3_list_detects_pagination_loops(fake_clock):
    page = FakeResponse(200, listing_xml([("monster-ai/a", 1)], truncated=True, token="SAME"))
    again = FakeResponse(200, listing_xml([("monster-ai/a", 1)], truncated=True, token="SAME"))
    with pytest.raises(DataUnavailable, match="repeating"):
        s3(FakeSession(page, again), fake_clock).list()


def test_s3_list_gives_up_after_too_many_pages(fake_clock, monkeypatch):
    monkeypatch.setattr(osx, "MAX_LIST_PAGES", 2)
    pages = [FakeResponse(200, listing_xml([], truncated=True, token=f"T{i}")) for i in range(2)]
    with pytest.raises(DataUnavailable, match="too many pages"):
        s3(FakeSession(*pages), fake_clock).list()


@pytest.mark.parametrize("prefix", ["../x", "/abs", "a b"])
def test_s3_list_validates_the_prefix(fake_clock, prefix):
    with pytest.raises(ValueError, match="invalid prefix"):
        s3(FakeSession(), fake_clock).list(prefix)


# ------------------------------------------------------------------------------------- retry behaviour
def test_s3_backoff_grows_exponentially_with_bounded_jitter(fake_clock):
    session = FakeSession(*[FakeResponse(503) for _ in range(6)])
    store = s3(session, fake_clock, max_attempts=6, backoff_base=1.0, backoff_max=8.0, rng=random.Random(7).random)
    with pytest.raises(DataUnavailable) as err:
        store.get("a")
    assert err.value.retryable is True and len(session.calls) == 6 and len(fake_clock.sleeps) == 5
    for attempt, delay in enumerate(fake_clock.sleeps, start=1):
        ceiling = min(8.0, 1.0 * 2 ** (attempt - 1))
        assert ceiling / 2 <= delay <= ceiling


def test_s3_backoff_extremes_and_retry_after(fake_clock):
    low = FakeSession(FakeResponse(503), FakeResponse(503), FakeResponse(200, b"ok"))
    assert s3(low, fake_clock, backoff_base=1.0, rng=lambda: 0.0).get("a") == b"ok"
    assert fake_clock.sleeps == [0.5, 1.0]
    fake_clock.sleeps.clear()
    high = FakeSession(FakeResponse(503), FakeResponse(200, b"ok"))
    assert s3(high, fake_clock, backoff_base=1.0, rng=lambda: 1.0).get("a") == b"ok"
    assert fake_clock.sleeps == [1.0]
    fake_clock.sleeps.clear()
    throttled = FakeSession(FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(503, headers={
        "Retry-After": "9999"}), FakeResponse(200, b"ok"))
    assert s3(throttled, fake_clock, backoff_max=20.0).get("a") == b"ok"
    assert fake_clock.sleeps == [3.0, 20.0]  # Retry-After is honoured but capped by backoff_max


@pytest.mark.parametrize("failure", [
    FakeResponse(500), FakeResponse(502), FakeResponse(429), FakeResponse(408),
    FakeResponse(400, error_xml("RequestTimeout")), FakeResponse(503, error_xml("SlowDown")),
    requests.exceptions.ConnectionError("connection refused"), requests.exceptions.Timeout("read timed out"),
    requests.exceptions.SSLError("eof occurred"), requests.exceptions.ChunkedEncodingError("broken"),
])
def test_s3_transient_failures_are_retried(fake_clock, failure):
    session = FakeSession(failure, FakeResponse(200, b"ok"))
    assert s3(session, fake_clock).get("a") == b"ok" and len(session.calls) == 2


@pytest.mark.parametrize("failure", [
    FakeResponse(400, error_xml("InvalidRequest")), FakeResponse(409), FakeResponse(416), FakeResponse(401),
    requests.exceptions.InvalidURL("bad"), requests.exceptions.MissingSchema("bad"),
])
def test_s3_permanent_failures_are_not_retried(fake_clock, failure):
    session = FakeSession(failure, FakeResponse(200, b"ok"))
    with pytest.raises(DataUnavailable) as err:
        s3(session, fake_clock).get("a")
    assert err.value.retryable is False and len(session.calls) == 1


def test_s3_redirect_reports_the_bucket_region_without_following_it(fake_clock):
    session = FakeSession(FakeResponse(301, error_xml("PermanentRedirect"), headers={
        "x-amz-bucket-region": "eu-central-1", "Location": "https://elsewhere.example.test/"}))
    with pytest.raises(DataUnavailable, match=r"eu-central-1") as err:
        s3(session, fake_clock).get("a")
    assert err.value.retryable is False and len(session.calls) == 1
    odd = FakeSession(FakeResponse(307, headers={"x-amz-bucket-region": "not valid!"}))
    with pytest.raises(DataUnavailable, match="S3_ENDPOINT_URL") as err2:
        s3(odd, fake_clock).get("a")
    assert "not valid" not in str(err2.value)


def test_s3_exception_text_is_scrubbed_of_credentials_and_signatures(fake_clock, caplog):
    signature = "ab" * 32
    leak = (f"secret={SECRET} X-Amz-Signature={signature} HTTPSConnectionPool: Max retries exceeded; "
            f"Authorization: AWS4-HMAC-SHA256 Credential={KEY_ID}/20261102/auto/s3/aws4_request, "
            f"SignedHeaders=host, Signature={signature} {SECRET}")
    session = FakeSession(*[requests.exceptions.ConnectionError(leak) for _ in range(2)])
    with caplog.at_level(logging.DEBUG), pytest.raises(DataUnavailable) as err:
        s3(session, fake_clock, max_attempts=2).get("a")
    for text in (str(err.value), caplog.text):
        assert SECRET not in text and KEY_ID not in text and signature not in text and "AWS4-HMAC-SHA256" not in text
    assert "connection error" in str(err.value)
    # the scrubbing also works when the registry of known secrets is empty (e.g. another test cleared it)
    store = s3(FakeSession(), fake_clock)
    clear_secrets()
    assert SECRET not in store._scrub(f"x {SECRET} y") and KEY_ID not in store._scrub(f"x {KEY_ID} y")
    short = S3Store("https://s3.example.test", "bucket", "auto", "abc", SECRET, session=FakeSession())
    assert short._scrub("key abc stays when too short to be a safe needle") == (
        "key abc stays when too short to be a safe needle")


def test_s3_error_documents_are_read_with_a_hard_cap(fake_clock):
    huge = FakeResponse(403, b"<Error><Code>AccessDenied</Code>" + b"x" * 500_000 + b"</Error>", chunk=100_000)
    with pytest.raises(DataUnavailable, match="HTTP 403") as err:
        s3(FakeSession(huge), fake_clock).get("a")
    assert "AccessDenied" not in str(err.value)  # truncated document is not parseable: no code, still an error


def test_s3_error_code_is_ignored_when_it_looks_unsafe(fake_clock):
    odd = FakeResponse(400, error_xml("Evil Code with spaces"))
    with pytest.raises(DataUnavailable, match=r"request rejected \(HTTP 400\)"):
        s3(FakeSession(odd), fake_clock).get("a")


def test_hostile_numeric_values_are_ignored_rather_than_fatal(fake_clock):
    superscript = "\u00b2"  # str.isdigit() is True for it, int() refuses it
    assert s3(FakeSession(FakeResponse(200, b"ok", headers={"Content-Length": superscript})),
              fake_clock).get("a") == b"ok"  # unknown length: no truncation check, no crash
    throttled = FakeSession(FakeResponse(429, headers={"Retry-After": superscript}), FakeResponse(200, b"ok"))
    assert s3(throttled, fake_clock).get("a") == b"ok" and fake_clock.sleeps == [0.25]
    page = listing_xml([("monster-ai/x", superscript), ("monster-ai/y", "-5")])
    assert [o.size for o in s3(FakeSession(FakeResponse(200, page)), fake_clock).list()] == [0, 0]


# -------------------------------------------------------------------------------------- misc plumbing
def test_close_only_closes_sessions_the_store_created(fake_clock):
    injected = FakeSession()
    s3(injected, fake_clock).close()
    assert injected.closed is False
    owned = S3Store("https://s3.example.test", "bucket", "auto", KEY_ID, SECRET)
    owned._session = FakeSession()
    owned.close()
    assert owned._session.closed is True


def test_default_clock_is_resolved_lazily(fake_clock):
    session = FakeSession(FakeResponse(200))
    store = S3Store("https://s3.example.test", "bucket", "auto", KEY_ID, SECRET, session=session)  # no clock passed
    store.exists("a")
    assert session.calls[0].headers["x-amz-date"] == "20261102T030000Z"


def test_build_object_store_follows_settings(settings):
    assert build_object_store(settings) is None
    partial_cfg = dataclasses.replace(settings, s3_bucket="bucket")
    assert build_object_store(partial_cfg) is None
    full = dataclasses.replace(settings, s3_bucket="bucket", s3_access_key_id=KEY_ID, s3_secret_access_key=SECRET,
                               s3_endpoint_url="https://s3.example.test", s3_region="auto", s3_prefix="p/")
    store = build_object_store(full)
    assert isinstance(store, S3Store) and store._prefix == "p/" and store._host == "s3.example.test"
    assert store._region == "auto" and store._path_style is True


def test_parse_xml_hardening():
    assert osx._parse_xml(b"<a xmlns='urn:x'><b>1</b></a>").tag.endswith("a")
    for bad in (b"<!DOCTYPE a><a/>", b"<!doctype a><a/>", b'<!ENTITY x "y"><a/>', b"<a>\x00</a>", b"\xc3(", b"<a>"):
        with pytest.raises(ValueError):
            osx._parse_xml(bad)
    # a declaration that claims another encoding cannot smuggle a DTD past the check
    with pytest.raises(ValueError):
        osx._parse_xml("<?xml version='1.0' encoding='UTF-16'?><!DOCTYPE a><a/>".encode())
    with pytest.raises(ValueError, match="too large"):
        osx._parse_xml(b"<a>" + b"x" * (osx.MAX_XML_BYTES + 1) + b"</a>")
