"""Ascending node crossing (ANX) times from a Sentinel-1 orbit file.

The frame index in `sm_db.tiling` is measured from the ANX, so the ANX is the
zero point of the whole scheme. It has to be derived the *same way* here as in
``s1reader.s1_reader.get_ascending_node_time_orbit``, or `sm_db` and the
stripmap reader would disagree about which frame an acquisition falls in and the
worker would find no matching row in the database.

That function detects the ascending zero-crossing of the orbit's Z coordinate
and linearly interpolates the crossing time. This module reproduces it on the
raw EOF XML, so `sm_db` stays a lightweight package: `numpy` and the standard
library, no `isce3`, no `s1reader`.

The two implementations agree to well under a millisecond, which is six orders
of magnitude inside the tolerance that matters -- a frame boundary only has to be
resolved to a fraction of the tile length (5 s by default).
"""

from __future__ import annotations

import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

__all__ = [
    "OrbitStateVectors",
    "ascending_node_time",
    "ascending_node_times",
    "read_orbit_file",
    "time_since_anx",
]

T_ORBIT = (12 * 86400.0) / 175.0
"""Nominal Sentinel-1 orbit period [s]; mirrors ``s1reader.s1_orbit.T_ORBIT``."""

PADDING_SHORT = 60.0
"""Orbit search padding [s]; mirrors ``s1reader.s1_orbit.PADDING_SHORT``."""


class OrbitStateVectors:
    """UTC times and ECEF positions from a Sentinel-1 orbit (EOF) file.

    `z` alone determines the ANX -- the south-to-north equator crossing is the
    ascending zero of Z -- while `x` and `y` are needed to place the ground track,
    which `sm_db.groundtrack` uses to give each frame its shape.

    Attributes
    ----------
    times :
        Array of `datetime.datetime`, one per state vector.
    x, y, z :
        Arrays of float, ECEF position in metres.
    """

    def __init__(
        self, times: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> None:
        self.times = times
        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        """Return the number of state vectors."""
        return len(self.times)


def read_orbit_file(path: str | Path) -> OrbitStateVectors:
    """Read the state vectors out of a Sentinel-1 EOF orbit file.

    Parameters
    ----------
    path :
        Path to a POEORB or RESORB ``.EOF`` file.

    Returns
    -------
    OrbitStateVectors

    Raises
    ------
    ValueError
        If the file contains no ``<OSV>`` elements.
    """
    tree = ET.parse(str(path))
    osv_list = tree.findall(".//OSV")
    if not osv_list:
        raise ValueError(f"No orbit state vectors found in {path}")

    def field(osv: ET.Element, name: str) -> str:
        """Return a state vector's field, or say which one the file is missing."""
        element = osv.find(name)
        if element is None or element.text is None:
            raise ValueError(f"Orbit state vector has no <{name}> in {path}")
        return element.text

    times = np.array(
        [
            datetime.datetime.fromisoformat(field(osv, "UTC").replace("UTC=", ""))
            for osv in osv_list
        ]
    )
    coords = {
        name: np.array([float(field(osv, name)) for osv in osv_list])
        for name in ("X", "Y", "Z")
    }
    return OrbitStateVectors(times, coords["X"], coords["Y"], coords["Z"])


def ascending_node_times(
    orbit: OrbitStateVectors,
    after: datetime.datetime,
    before: datetime.datetime,
) -> list[datetime.datetime]:
    """Return every ascending node crossing in a time window, in order.

    Finds each pair of state vectors that straddles ``Z = 0`` upward and linearly
    interpolates the time at which ``Z`` is zero, as
    ``s1reader.s1_reader.get_ascending_node_time_orbit`` does.

    Parameters
    ----------
    orbit :
        State vectors covering the window.
    after, before :
        Window bounds. Crossings exactly on a bound are excluded.

    Returns
    -------
    list of datetime.datetime
        Ascending in time; empty if the orbit covers no crossing in the window.
    """
    pad = datetime.timedelta(seconds=PADDING_SHORT)
    window = (orbit.times > after - pad) & (orbit.times < before + pad)
    times = orbit.times[window]
    z = orbit.z[window]

    crossings = []
    for i, (z_prev, z_next) in enumerate(zip(z, z[1:], strict=False)):
        if not z_prev < 0 <= z_next:
            continue

        # Upstream fits t(z) over a +/-3 sample neighbourhood rather than using the
        # straddling pair alone, so match that span to stay comparable with it.
        lo, hi = max(i - 3, 0), min(i + 3, len(z))
        z_near, t_near = z[lo:hi], times[lo:hi]
        t_ref = t_near[0]
        dt_near = np.array([(t - t_ref).total_seconds() for t in t_near])

        # Z rises monotonically through an ascending crossing, so t(z) is
        # single-valued here and a plain linear interpolation is exact to the
        # sampling interval.
        dt_zero = float(np.interp(0.0, z_near, dt_near))
        crossing = t_ref + datetime.timedelta(seconds=dt_zero)
        if after < crossing < before:
            crossings.append(crossing)

    return sorted(crossings)


def ascending_node_time(
    orbit: OrbitStateVectors,
    sensing_time: datetime.datetime,
    search_length: float | None = None,
) -> datetime.datetime:
    """Return the most recent ANX strictly before a sensing time.

    This is the zero point of the frame index -- see the "Which ANX" section of
    `sm_db.tiling` for why it is the true preceding crossing and not the value
    ESA annotates.

    Parameters
    ----------
    orbit :
        State vectors covering at least one revolution before `sensing_time`.
    sensing_time :
        Time of interest, normally a scene's first line.
    search_length :
        How far back to look, in seconds. Defaults to two orbital periods.

    Returns
    -------
    datetime.datetime
        The ascending node crossing time.

    Raises
    ------
    ValueError
        If no ascending crossing precedes `sensing_time` in the orbit provided,
        which means the orbit file does not cover the acquisition.
    """
    if search_length is None:
        search_length = 2 * T_ORBIT

    after = sensing_time - datetime.timedelta(seconds=search_length)
    crossings = ascending_node_times(orbit, after, sensing_time)
    if not crossings:
        raise ValueError(
            f"No ascending node crossing before {sensing_time} in the orbit provided; "
            "the orbit file probably does not cover this acquisition"
        )
    return crossings[-1]


def time_since_anx(sensing_time: datetime.datetime, anx: datetime.datetime) -> float:
    """Return seconds elapsed from an ascending node crossing to a sensing time.

    Parameters
    ----------
    sensing_time :
        Time of interest.
    anx :
        Ascending node crossing, from `ascending_node_time`.

    Returns
    -------
    float
        Seconds since the ANX.
    """
    return (sensing_time - anx).total_seconds()
