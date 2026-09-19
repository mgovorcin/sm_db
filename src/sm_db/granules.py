"""Stripmap granule metadata, from the ASF search API or a saved catalog.

Everything `sm_db` needs about an acquisition is available from ASF's search
API -- footprint, beam mode, track, flight direction, start and stop times -- so
frames can be defined without touching a SAFE. The one thing ASF does not serve
is the ascending node crossing, which `sm_db.anx` derives from the orbit instead.

A catalog is a plain JSON list of the records below, the same shape
`compass_batch.discover` writes, so a catalog captured by either tool is readable
by the other.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon, shape

__all__ = [
    "ASF_SEARCH_URL",
    "SM_BEAM_MODES",
    "Granule",
    "load_catalog",
    "query_asf",
    "save_catalog",
]

ASF_SEARCH_URL = "https://api.daac.asf.alaska.edu/services/search/param"

SM_BEAM_MODES = ("S1", "S2", "S3", "S4", "S5", "S6")
"""The six Sentinel-1 stripmap beams."""


@dataclass
class Granule:
    """One Sentinel-1 stripmap SLC acquisition.

    Attributes
    ----------
    name :
        Granule name, e.g. ``S1C_S3_SLC__1SDV_20260313T044134_...``.
    beam_mode :
        ``S1`` through ``S6``.
    track :
        Relative orbit number, 1-175.
    absolute_orbit :
        Absolute orbit number, used to group acquisitions sharing an ANX.
    ascending :
        True for an ascending pass.
    start, stop :
        Scene first- and last-line times, UTC.
    geometry :
        Footprint as a GeoJSON geometry mapping.
    """

    name: str
    beam_mode: str
    track: int
    absolute_orbit: int
    ascending: bool
    start: datetime
    stop: datetime
    geometry: dict[str, Any] = field(default_factory=dict)

    @property
    def footprint(self) -> Polygon:
        """Footprint as a shapely polygon in lon/lat degrees."""
        return shape(self.geometry)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["stop"] = self.stop.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Granule:
        """Rebuild a `Granule` from `to_dict` output."""
        return cls(
            name=d["name"],
            beam_mode=d["beam_mode"].upper(),
            track=int(d["track"]),
            absolute_orbit=int(d["absolute_orbit"]),
            ascending=bool(d["ascending"]),
            start=_parse_time(d["start"]),
            stop=_parse_time(d["stop"]),
            geometry=d.get("geometry") or {},
        )


def _parse_time(text: str) -> datetime:
    """Parse an ISO timestamp, dropping any timezone to keep UTC-naive times.

    Orbit files annotate times as UTC-naive, and the ANX arithmetic subtracts one
    from the other, so mixing an aware ASF timestamp in would raise.
    """
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _from_asf_feature(feature: dict[str, Any]) -> Granule:
    """Build a `Granule` from one ASF GeoJSON feature."""
    p = feature["properties"]
    return Granule(
        name=p["sceneName"],
        beam_mode=p["beamModeType"].upper(),
        track=int(p["pathNumber"]),
        absolute_orbit=int(p["orbit"]),
        ascending=p["flightDirection"].upper().startswith("ASC"),
        start=_parse_time(p["startTime"]),
        stop=_parse_time(p["stopTime"]),
        geometry=feature["geometry"],
    )


def query_asf(
    start: str,
    end: str,
    beam_modes: Iterable[str] = SM_BEAM_MODES,
    tracks: Iterable[int] | None = None,
    intersects_wkt: str | None = None,
    session: Any = None,
) -> list[Granule]:
    """Search ASF for Sentinel-1 stripmap SLCs.

    Parameters
    ----------
    start, end :
        Date range, anything ASF accepts (``2026-02-01``).
    beam_modes :
        Which stripmap beams to include.
    tracks :
        Relative orbit numbers to restrict to; `None` means all.
    intersects_wkt :
        Area of interest as WKT; `None` means global.
    session :
        Object with a ``get(url, params=...)`` method. Defaults to a new
        `requests` session. Injected so tests can run offline.

    Returns
    -------
    list of Granule
        In acquisition order.
    """
    params: dict[str, Any] = {
        "platform": "Sentinel-1",
        "processingLevel": "SLC",
        "beamMode": ",".join(beam_modes),
        "start": start,
        "end": end,
        "output": "geojson",
    }
    if tracks:
        params["relativeOrbit"] = ",".join(str(t) for t in tracks)
    if intersects_wkt:
        params["intersectsWith"] = intersects_wkt

    if session is None:
        import requests

        session = requests.Session()

    features = _get_features(session, params)
    granules = [_from_asf_feature(f) for f in features]
    return sorted(granules, key=lambda g: g.start)


def _get_features(session: Any, params: dict[str, Any], attempts: int = 3) -> list:
    """One ASF query, retried, splitting the window in half when it is refused.

    ASF answers a wide window with HTTP 400 rather than paging it: querying the
    stripmap archive a year at a time fails for the busiest years while the quiet
    ones succeed. Halving the date range until it is accepted keeps a caller from
    having to guess a safe chunk size.
    """
    import time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(ASF_SEARCH_URL, params=params)
            response.raise_for_status()
            return response.json().get("features", [])
        except Exception as exc:  # noqa: BLE001 - re-raised below if unrecoverable
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 400:
                break
            time.sleep(1.5 * (attempt + 1))

    halves = _split_window(params)
    if halves is None:
        raise RuntimeError(f"ASF refused the query and it cannot be split: {last}")

    out: list = []
    for half in halves:
        out.extend(_get_features(session, half, attempts))
    return out


def _split_window(params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Halve a query's date range, or `None` when it is already a single day."""
    start = _parse_time(params["start"])
    end = _parse_time(params["end"])
    if (end - start).days < 1:
        return None

    middle = start + (end - start) / 2
    first = dict(
        params, start=start.strftime("%Y-%m-%d"), end=middle.strftime("%Y-%m-%d")
    )
    second = dict(
        params,
        start=(middle + timedelta(days=1)).strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    return [first, second]


def save_catalog(granules: Iterable[Granule], path: str | Path) -> None:
    """Write granules to a JSON catalog, gzipped when the name says so.

    The whole stripmap archive is ~13 MB of JSON and the scheduled job commits it
    daily, so a ``.gz`` name is the sensible default: it compresses about eight to
    one and keeps the repository from growing by a catalog a day.

    Parameters
    ----------
    granules :
        Records to write.
    path :
        Destination file. A ``.gz`` suffix selects gzip.
    """
    text = json.dumps([g.to_dict() for g in granules], indent=2) + "\n"
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text)


def load_catalog(path: str | Path) -> list[Granule]:
    """Read a JSON catalog written by `save_catalog`.

    Parameters
    ----------
    path :
        Catalog file. A ``.gz`` suffix is read as gzip.

    Returns
    -------
    list of Granule
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [Granule.from_dict(d) for d in json.load(fh)]
    return [Granule.from_dict(d) for d in json.loads(path.read_text())]
