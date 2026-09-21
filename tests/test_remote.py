from __future__ import annotations

import json

import pytest

from sm_db.remote import fetch_asset


class _Response:
    def __init__(self, status, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, size):
        yield self._body


class _Session:
    """Serves one asset with an ETag, answering 304 to a matching If-None-Match."""

    def __init__(self, body=b"db-v1", etag='"v1"', status=200):
        self.body, self.etag, self.status = body, etag, status
        self.requests: list[dict] = []

    def get(self, url, headers=None, stream=False, timeout=None):
        self.requests.append({"url": url, "headers": dict(headers or {})})
        if self.status != 200:
            return _Response(self.status)
        if (headers or {}).get("If-None-Match") == self.etag:
            return _Response(304)
        return _Response(
            200, self.body, {"ETag": self.etag, "Last-Modified": "Mon, 21 Sep 2026"}
        )


def test_first_fetch_downloads_and_records_the_version(tmp_path):
    session = _Session()
    result = fetch_asset(tmp_path, session=session)
    assert result.updated
    assert result.path.read_bytes() == b"db-v1"
    meta = json.loads((tmp_path / "sm_frames.sqlite3.remote.json").read_text())
    assert meta["etag"] == '"v1"'
    assert session.requests[0]["url"].endswith("/archive/sm_frames.sqlite3")


def test_unchanged_archive_is_not_downloaded_again(tmp_path):
    session = _Session()
    fetch_asset(tmp_path, session=session)
    again = fetch_asset(tmp_path, session=session)
    assert not again.updated
    assert session.requests[1]["headers"] == {"If-None-Match": '"v1"'}


def test_a_new_version_replaces_the_file(tmp_path):
    fetch_asset(tmp_path, session=_Session())
    result = fetch_asset(tmp_path, session=_Session(b"db-v2", '"v2"'))
    assert result.updated
    assert result.path.read_bytes() == b"db-v2"
    assert not (tmp_path / "sm_frames.sqlite3.part").exists()


def test_force_skips_the_version_check(tmp_path):
    session = _Session()
    fetch_asset(tmp_path, session=session)
    assert fetch_asset(tmp_path, session=session, force=True).updated
    assert session.requests[1]["headers"] == {}


def test_a_failed_download_leaves_the_old_copy(tmp_path):
    fetch_asset(tmp_path, session=_Session())
    with pytest.raises(RuntimeError):
        fetch_asset(tmp_path, session=_Session(status=503), force=True)
    assert (tmp_path / "sm_frames.sqlite3").read_bytes() == b"db-v1"
