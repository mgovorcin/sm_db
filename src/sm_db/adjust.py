"""Turn a frame edited by hand into a frame the pipeline can reproduce.

Editing a frame in a GIS is the natural way to fix one that cuts an island in
half: grab the end of the quadrilateral and drag it past the coastline. But a
hand-drawn polygon is not something to pin a grid to. Its corners wander, and
its sides are wherever the mouse let go of them.

Stripmap gives a clean way to read such an edit. A frame is a slice of a
continuous swath, so it has exactly two degrees of freedom that mean anything:
where along the track it starts, and where it stops. Its east and west sides are
not a choice -- they are the edges of the swath, fixed by the beam. So an edit is
reduced to its along-track window, measured against the orbit, and the sides are
put back on the swath. Hand-drawn sides that strayed by as much as 1.4 km come
back to the swath exactly.

The window is expressed in seconds since the ascending node, the same clock the
frame index is measured on, which is what makes it hold for every repeat pass
rather than only for the acquisition it was measured against. It is then stored
as the existing ``shift`` and ``overlap`` adjustments, so an edited frame goes
through the same build path as every other frame.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

from sm_db import anx as anx_mod
from sm_db import tiling
from sm_db.granules import Granule
from sm_db.groundtrack import GroundTrack

__all__ = [
    "BOUNDARY_SNAP",
    "Window",
    "along_track_window",
    "snap_to_boundary",
    "window_to_offsets",
]

BOUNDARY_SNAP = 0.05
"""Seconds within which a drawn end is taken to mean a tile boundary.

About 350 m along track. An edit that extends a frame usually moves one end and
leaves the other on the original boundary, and a hand-placed vertex lands a few
tens of metres off it; this keeps that end exactly where it was meant to be.
"""


@dataclass(frozen=True)
class Window:
    """Where along its track a frame starts and stops.

    Attributes
    ----------
    start, stop :
        Seconds since the ascending node.
    west_deviation, east_deviation :
        How far the drawn sides sat from the swath, in metres, for reporting.
        Signed cross-track offsets; these are discarded, since the sides are put
        back on the swath.
    """

    start: float
    stop: float
    west_deviation: float = 0.0
    east_deviation: float = 0.0

    @property
    def length(self) -> float:
        """Along-track length in seconds."""
        return self.stop - self.start


def snap_to_boundary(
    t: float,
    tile_seconds: float = tiling.DEFAULT_TILE_SECONDS,
    tolerance: float = BOUNDARY_SNAP,
) -> float:
    """Pull a time onto the nearest tile boundary if it is within `tolerance`.

    Parameters
    ----------
    t :
        Seconds since the ascending node.
    tile_seconds :
        Tile length.
    tolerance :
        How close counts as meaning the boundary.

    Returns
    -------
    float
        The boundary, or `t` rounded to a hundredth of a second.

    Examples
    --------
    >>> snap_to_boundary(tiling.T_PRE + 10.02, 5.0) == tiling.T_PRE + 10.0
    True
    >>> snap_to_boundary(tiling.T_PRE + 11.30, 5.0)
    13.6
    """
    nearest = tiling.T_PRE + round((t - tiling.T_PRE) / tile_seconds) * tile_seconds
    if abs(t - nearest) <= tolerance:
        return nearest
    return round(t, 2)


def _project(polygon: Polygon, epsg: int) -> Polygon:
    """Project a lon/lat polygon into `epsg`."""
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    return transform(lambda x, y: tf.transform(x, y), polygon)


def _node_for_frame(
    granule: Granule,
    orbit: anx_mod.OrbitStateVectors,
    index: int,
    tile_seconds: float,
) -> datetime.datetime:
    """Find the ascending node a frame's index is counted from.

    Normally the node before the granule, but not always: a granule can start
    just before a node and image frames on the next track, whose index restarts
    from the later node. Whichever node puts this frame's tile inside the time
    the granule was actually imaging is the right one.
    """
    start, stop = tiling.tile_bounds(index, tile_seconds)
    middle = datetime.timedelta(seconds=(start + stop) / 2)
    slack = datetime.timedelta(seconds=60)
    nodes = anx_mod.ascending_node_times(
        orbit,
        granule.start - datetime.timedelta(seconds=2 * anx_mod.T_ORBIT),
        granule.stop + slack,
    )
    for node in reversed(nodes):
        if granule.start - slack <= node + middle <= granule.stop + slack:
            return node
    raise ValueError(
        f"No ascending node puts frame index {index} inside {granule.name}"
    )


def along_track_window(
    drawn: Polygon,
    granule: Granule,
    orbit: anx_mod.OrbitStateVectors,
    epsg: int,
    index: int,
    tile_seconds: float = tiling.DEFAULT_TILE_SECONDS,
) -> Window:
    """Read the along-track extent of a hand-drawn frame.

    Every vertex is dropped perpendicular onto the ground track; the earliest and
    latest points it lands on are where the frame starts and stops. The
    cross-track distances are reported against the swath but otherwise ignored.

    Parameters
    ----------
    drawn :
        The edited frame, in lon/lat degrees.
    granule :
        Any acquisition of that frame; it supplies the ascending node and the
        swath edges. The window is on the orbit clock, so which acquisition is
        used does not change the answer.
    orbit :
        State vectors covering the granule and the revolution before it.
    epsg :
        The frame's projected CRS.
    index :
        The frame's tile index, which fixes which ascending node it counts from.
    tile_seconds :
        Tile length, for snapping ends back onto boundaries.

    Returns
    -------
    Window
    """
    from sm_db.groundtrack import swath_offsets

    node = _node_for_frame(granule, orbit, index, tile_seconds)
    # Wide enough for an edit that reaches well past the granule's own extent.
    pad = datetime.timedelta(seconds=120)
    track = GroundTrack(orbit, epsg, granule.start - pad, granule.stop + pad)

    near, far = swath_offsets(track, _project(granule.footprint, epsg))
    abeam = [track.project(x, y) for x, y in _project(drawn, epsg).exterior.coords[:-1]]
    times = [(when - node).total_seconds() for when, _ in abeam]
    offsets = [offset for _, offset in abeam]

    return Window(
        start=snap_to_boundary(min(times), tile_seconds),
        stop=snap_to_boundary(max(times), tile_seconds),
        west_deviation=round(min(offsets) - near, 1),
        east_deviation=round(max(offsets) - far, 1),
    )


def window_to_offsets(
    index: int,
    window: Window,
    tile_seconds: float = tiling.DEFAULT_TILE_SECONDS,
) -> tuple[float, float]:
    """Express an along-track window as the ``shift`` and ``overlap`` a build takes.

    A build places a frame from ``tile.start - overlap + shift`` to
    ``tile.stop + overlap + shift``. Two unknowns and two ends, so any window has
    exactly one answer, including a lopsided one that moves a single end.

    Parameters
    ----------
    index :
        The frame's tile index.
    window :
        Where it should start and stop.
    tile_seconds :
        Tile length.

    Returns
    -------
    tuple of float
        ``(shift, overlap)`` in seconds.

    Examples
    --------
    Extending only the far end by 5 s is a shift and an overlap of 2.5 s each:

    >>> t0, t1 = tiling.tile_bounds(3, 5.0)
    >>> window_to_offsets(3, Window(t0, t1 + 5.0), 5.0)
    (2.5, 2.5)
    """
    tile_start, tile_stop = tiling.tile_bounds(index, tile_seconds)
    shift = ((window.start - tile_start) + (window.stop - tile_stop)) / 2
    overlap = ((tile_start - window.start) + (window.stop - tile_stop)) / 2
    return round(shift, 3), round(overlap, 3)
