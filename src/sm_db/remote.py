"""Fetch the published frame database instead of building one.

The weekly workflow keeps the archive current and publishes it as assets of the
``archive`` release, replaced in place. Building it locally takes every orbit
file of the twelve-year archive (24 GB); fetching it is one 6 MB download. A
tool that only needs to *read* frames -- a planner, a submission UI -- should
fetch.

The download is conditional on the asset's ETag, so asking again costs a single
request that returns nothing when the archive has not changed, and the file is
replaced atomically, so a reader never sees a half-written database.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ASSETS",
    "RELEASE_URL",
    "FetchResult",
    "fetch_asset",
]

RELEASE_URL = "https://github.com/mgovorcin/sm_db/releases/download/archive"
"""Where the scheduled workflow publishes the archive."""

ASSETS = ("sm_frames.sqlite3", "sm_granules.json.gz", "sm_frames.geojson")
"""What the release carries: the database, the granule catalog, the frames."""

_CHUNK = 1 << 20


@dataclass(frozen=True)
class FetchResult:
    """What a fetch found.

    Attributes
    ----------
    path :
        The local file.
    updated :
        Whether this call downloaded a new copy.
    etag, last_modified :
        The remote asset's version, as the server reported it.
    """

    path: Path
    updated: bool
    etag: str | None
    last_modified: str | None


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".remote.json")


def fetch_asset(
    dest_dir: str | Path,
    name: str = "sm_frames.sqlite3",
    base_url: str = RELEASE_URL,
    session: Any = None,
    force: bool = False,
    timeout: float = 120.0,
) -> FetchResult:
    """Download one release asset unless the local copy is already current.

    Parameters
    ----------
    dest_dir :
        Directory to keep it in; created if missing.
    name :
        Asset name, one of `ASSETS`.
    base_url :
        Release download URL.
    session :
        A ``requests.Session``-like object; a new one by default.
    force :
        Download even when the ETag says nothing changed.
    timeout :
        Seconds per request.

    Returns
    -------
    FetchResult
    """
    import requests

    dest = Path(dest_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _sidecar(dest)
    known = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    headers = {}
    if not force and dest.exists() and known.get("etag"):
        headers["If-None-Match"] = known["etag"]

    http = session or requests.Session()
    with http.get(
        f"{base_url}/{name}", headers=headers, stream=True, timeout=timeout
    ) as response:
        if response.status_code == 304:
            return FetchResult(
                dest, False, known.get("etag"), known.get("last_modified")
            )
        response.raise_for_status()
        partial = dest.with_name(dest.name + ".part")
        with open(partial, "wb") as f:
            for chunk in response.iter_content(_CHUNK):
                f.write(chunk)
        etag = response.headers.get("ETag")
        modified = response.headers.get("Last-Modified")

    os.replace(partial, dest)
    meta_path.write_text(
        json.dumps({"etag": etag, "last_modified": modified, "url": base_url})
    )
    return FetchResult(dest, True, etag, modified)
