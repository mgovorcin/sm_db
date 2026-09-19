"""Fetch the precise orbits a frame definition needs.

Frames are measured from the ascending node, which `sm_db.anx` reads off the
orbit, so building or updating a database means having an EOF for every granule.
This wraps `sentineleof` rather than reimplementing the download: it already
handles the Copernicus and ASF sources, the public ``s1-orbits`` bucket needs no
credentials, and it skips files already on disk.

`sentineleof` is an optional dependency -- querying and reading an existing
database does not need it -- so it is imported only when a download is actually
requested.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from pathlib import Path

from sm_db.anx import T_ORBIT
from sm_db.granules import Granule

__all__ = ["ensure_orbits"]


def ensure_orbits(
    granules: Iterable[Granule],
    directory: str | Path,
    orbit_type: str = "precise",
) -> list[Path]:
    """Download any orbit files the granules need that are not already present.

    Parameters
    ----------
    granules :
        Acquisitions to cover.
    directory :
        Where EOFs are kept. Created if missing.
    orbit_type :
        ``"precise"`` (POEORB, available after ~20 days) or ``"restituted"``
        (RESORB, available within hours). Precise is the default because a
        database is normally built over past acquisitions.

    Returns
    -------
    list of pathlib.Path
        The files downloaded in this call; empty when everything was cached.

    Raises
    ------
    ImportError
        If `sentineleof` is not installed.
    """
    try:
        from eof.download import download_eofs
    except ImportError as exc:  # pragma: no cover - exercised by the message only
        raise ImportError(
            "Downloading orbits needs the optional `sentineleof` dependency: "
            "pip install 'sm_db[update]'"
        ) from exc

    from sm_db.frames import OrbitLookup

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Ask only for what is not already on disk. An EOF spans a whole day, so a
    # campaign's granules share a handful of files; requesting one per granule
    # would re-fetch the same file dozens of times.
    lookup = OrbitLookup(directory)
    requests = set()
    for granule in granules:
        if lookup.covers(granule):
            continue
        # The ascending node can be a full revolution before the scene, so ask
        # for an orbit covering that earlier instant, not just the acquisition.
        needed = granule.start - datetime.timedelta(seconds=T_ORBIT)
        requests.add((granule.name[:3].upper(), needed.date()))

    if not requests:
        return []

    # Midday keeps the request away from the file boundaries, which sit at
    # 22:59:42 either side of the day an EOF covers.
    times = [
        datetime.datetime.combine(day, datetime.time(12, 0))
        for _, day in sorted(requests)
    ]
    missions = [mission for mission, _ in sorted(requests)]

    return download_eofs(
        orbit_dts=times,
        missions=missions,
        save_dir=str(directory),
        orbit_type=orbit_type,
    )
