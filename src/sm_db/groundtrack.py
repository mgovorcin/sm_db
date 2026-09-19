"""Place a frame on the ground from the orbit, not from the granule's timing.

A frame is defined by a time interval, so its footprint ought to follow from
where the satellite was during that interval. Deriving it instead by
interpolating the granule's own footprint between its start and stop times looks
simpler, but it inherits whatever precision those times have -- and ASF rounds
them to whole seconds. On a 20 s stripmap scene that is a 5 percent along-track
error, which moved frame boxes by up to 4.3 km between dates that in truth repeat
to within 40 m.

The orbit has none of that: state vectors are 10 s apart with sub-millimetre
positions. So the along-track placement comes from the orbit, and the granule
footprint is used only for what it measures well -- how far the swath sits to
either side of the ground track, a cross-track quantity that along-track timing
error does not touch.

Everything here works in the frame's own projected metres, where the geometry is
planar to well within a pixel over a frame-sized area.
"""

from __future__ import annotations

import datetime

import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon

from sm_db.anx import OrbitStateVectors

__all__ = ["GroundTrack", "swath_offsets", "tile_polygon"]

_ECEF_TO_LONLAT = Transformer.from_crs(4978, 4326, always_xy=True)


class GroundTrack:
    """The sub-satellite track over a time window, in one projected CRS.

    Parameters
    ----------
    orbit :
        State vectors covering the window.
    epsg :
        Projected CRS to work in, in metres.
    start, stop :
        Window bounds. State vectors outside it are dropped, with a little
        padding so interpolation at the bounds stays interior.
    """

    def __init__(
        self,
        orbit: OrbitStateVectors,
        epsg: int,
        start: datetime.datetime,
        stop: datetime.datetime,
    ) -> None:
        pad = datetime.timedelta(seconds=60)
        window = (orbit.times >= start - pad) & (orbit.times <= stop + pad)
        if window.sum() < 2:
            raise ValueError(
                f"Orbit covers fewer than two state vectors between {start} and {stop}"
            )

        self.epsg = epsg
        self.times = orbit.times[window]
        self._t0 = self.times[0]
        self.seconds = np.array([(t - self._t0).total_seconds() for t in self.times])

        lon, lat, _ = _ECEF_TO_LONLAT.transform(
            orbit.x[window], orbit.y[window], orbit.z[window]
        )
        to_map = Transformer.from_crs(4326, epsg, always_xy=True)
        self.x, self.y = to_map.transform(lon, lat)

    def _seconds(self, when: datetime.datetime) -> float:
        """Convert an absolute time to this track's internal seconds."""
        return (when - self._t0).total_seconds()

    def position(self, when: datetime.datetime) -> tuple[float, float]:
        """Return the sub-satellite point at `when`, in projected metres.

        Parameters
        ----------
        when :
            Time within the window.

        Returns
        -------
        tuple of float
            ``(x, y)`` in metres.
        """
        s = self._seconds(when)
        return float(np.interp(s, self.seconds, self.x)), float(
            np.interp(s, self.seconds, self.y)
        )

    def heading(self, when: datetime.datetime) -> tuple[float, float]:
        """Return the unit along-track direction at `when`.

        Parameters
        ----------
        when :
            Time within the window.

        Returns
        -------
        tuple of float
            Unit vector ``(dx, dy)`` in projected metres.
        """
        delta = datetime.timedelta(seconds=1)
        x0, y0 = self.position(when - delta)
        x1, y1 = self.position(when + delta)
        dx, dy = x1 - x0, y1 - y0
        norm = np.hypot(dx, dy)
        return dx / norm, dy / norm

    def project(self, x: float, y: float) -> tuple[datetime.datetime, float]:
        """Find where a point sits relative to the track.

        Drops a perpendicular from the point onto the track polyline.

        Parameters
        ----------
        x, y :
            Point in projected metres.

        Returns
        -------
        tuple
            ``(time, offset)`` -- when the satellite passed abeam of the point, and
            the signed cross-track distance in metres, positive to the left of
            the direction of travel.
        """
        best = (np.inf, 0.0, 0.0)
        for i in range(len(self.seconds) - 1):
            ax, ay, bx, by = self.x[i], self.y[i], self.x[i + 1], self.y[i + 1]
            dx, dy = bx - ax, by - ay
            length_sq = dx * dx + dy * dy
            f = ((x - ax) * dx + (y - ay) * dy) / length_sq
            f = min(max(f, 0.0), 1.0)
            px, py = ax + f * dx, ay + f * dy
            distance_sq = (x - px) ** 2 + (y - py) ** 2
            if distance_sq < best[0]:
                # Cross product sign: positive when the point is left of travel.
                side = np.sign(dx * (y - ay) - dy * (x - ax))
                segment_seconds = self.seconds[i] + f * (
                    self.seconds[i + 1] - self.seconds[i]
                )
                best = (distance_sq, segment_seconds, side * np.sqrt(distance_sq))

        _, seconds, offset = best
        return self._t0 + datetime.timedelta(seconds=float(seconds)), float(offset)


def swath_offsets(track: GroundTrack, footprint: Polygon) -> tuple[float, float]:
    """Measure how far a swath extends either side of the ground track.

    Parameters
    ----------
    track :
        Ground track for the acquisition, in the same CRS as `footprint`.
    footprint :
        Scene footprint, already projected to the track's CRS.

    Returns
    -------
    tuple of float
        ``(near, far)`` signed cross-track offsets in metres, the minimum and
        maximum over the footprint corners.
    """
    offsets = [track.project(x, y)[1] for x, y in footprint.exterior.coords[:-1]]
    return min(offsets), max(offsets)


def tile_polygon(
    track: GroundTrack,
    tile_start: datetime.datetime,
    tile_stop: datetime.datetime,
    near: float,
    far: float,
) -> Polygon:
    """Build a tile's quadrilateral from the ground track and swath offsets.

    The two along-track ends are the cross-track segments at `tile_start` and
    `tile_stop`, each spanning from `near` to `far` either side of the track.

    Parameters
    ----------
    track :
        Ground track covering the tile.
    tile_start, tile_stop :
        Tile bounds as absolute times.
    near, far :
        Signed cross-track offsets in metres, from `swath_offsets`.

    Returns
    -------
    shapely.geometry.Polygon
        The tile footprint in the track's projected CRS.
    """
    corners: list[tuple[float, float]] = []
    for when, offsets in ((tile_start, (near, far)), (tile_stop, (far, near))):
        px, py = track.position(when)
        hx, hy = track.heading(when)
        # Left normal of the heading, matching the sign convention in `project`.
        nx, ny = -hy, hx
        corners.extend((px + d * nx, py + d * ny) for d in offsets)
    return Polygon(corners)
