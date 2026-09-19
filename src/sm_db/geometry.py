"""Map projections and pinned bounding boxes for frames.

Two conventions live here, both carried over from
``burst_db.build_frame_db`` so that a stripmap frame database behaves like the
IW one COMPASS already consumes:

**The projection follows the frame centre**, UTM away from the poles and polar
stereographic near them.

**Bounds are snapped outward.** Mins are floored and maxes ceiled onto a fixed
30 m lattice after a 5 km margin, so a box always contains the footprint it came
from and lands on the same lattice as its neighbours.

The frame's *shape* is not computed here -- see `sm_db.groundtrack`, which builds
it from the orbit.
"""

from __future__ import annotations

import math

import utm

__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_SNAP",
    "NORTH_THRESHOLD",
    "SOUTH_THRESHOLD",
    "pick_epsg",
    "snap_bbox",
]

DEFAULT_MARGIN = 5000.0
"""Outward padding applied to every frame bbox [m]."""

DEFAULT_SNAP = 30.0
"""Lattice that frame bbox corners are snapped onto [m]."""

NORTH_THRESHOLD = 75
"""Latitude above which frames use EPSG:3413 instead of UTM [deg]."""

SOUTH_THRESHOLD = -60
"""Latitude below which frames use EPSG:3031 instead of UTM [deg]."""


def pick_epsg(lon: float, lat: float) -> int:
    """Return the projected EPSG code for a frame centred at `lon`, `lat`.

    Follows ``burst_db.build_frame_db.get_epsg_codes``: polar stereographic near
    the poles, UTM elsewhere. The zone comes from the `utm` package rather than
    from ``floor(lon / 6)`` because that package encodes the Norway and Svalbard
    exceptions, where the zone boundaries are not on the regular 6-degree grid.

    Parameters
    ----------
    lon, lat :
        Frame centre in degrees.

    Returns
    -------
    int
        An EPSG code: 3413 (north polar), 3031 (south polar), or 326xx/327xx.

    Examples
    --------
    >>> pick_epsg(-118.0, 34.0)
    32611
    >>> pick_epsg(15.0, -34.0)
    32733
    >>> pick_epsg(-45.0, 80.0)
    3413
    >>> pick_epsg(6.0, 60.0)  # Norway's widened zone 32
    32632
    """
    if lat > NORTH_THRESHOLD:
        return 3413
    if lat < SOUTH_THRESHOLD:
        return 3031

    zone = utm.from_latlon(lat, wrap_lon(lon))[2]
    return (32600 if lat >= 0 else 32700) + zone


def wrap_lon(lon: float) -> float:
    """Wrap a longitude into ``[-180, 180)``.

    Parameters
    ----------
    lon :
        Longitude in degrees.

    Returns
    -------
    float

    Examples
    --------
    >>> wrap_lon(181.0)
    -179.0
    """
    return (lon + 180.0) % 360.0 - 180.0


def snap_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    margin: float = DEFAULT_MARGIN,
    snap: float = DEFAULT_SNAP,
) -> tuple[int, int, int, int]:
    """Pad a bounding box outward and snap it onto a fixed lattice.

    Mins are floored and maxes ceiled, so the result always contains the input.
    Integers, matching the column type COMPASS reads from a burst database.

    Parameters
    ----------
    xmin, ymin, xmax, ymax :
        Bounds in projected metres.
    margin :
        Outward padding in metres.
    snap :
        Lattice spacing in metres.

    Returns
    -------
    tuple of int
        The padded, snapped bounds.

    Examples
    --------
    >>> snap_bbox(1000.0, 2000.0, 3000.0, 4000.0, margin=0.0, snap=100.0)
    (1000, 2000, 3000, 4000)
    >>> snap_bbox(1001.0, 2001.0, 2999.0, 3999.0, margin=0.0, snap=100.0)
    (1000, 2000, 3000, 4000)
    """
    return (
        int(math.floor((xmin - margin) / snap) * snap),
        int(math.floor((ymin - margin) / snap) * snap),
        int(math.ceil((xmax + margin) / snap) * snap),
        int(math.ceil((ymax + margin) / snap) * snap),
    )
