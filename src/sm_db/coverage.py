"""How much of a frame every acquisition actually fills.

A frame's pinned box is fixed, but the data landing in it is not. The
along-track extent is guaranteed -- a tile is only claimed when a scene covers it
completely -- yet across track the swath drifts between passes, and a frame's
width was measured from whichever acquisition happened to define it first. So the
area valid in *every* date, which is all a time series can use, is the
intersection of the acquisitions rather than the frame itself.

Measured by sampling rather than by intersecting polygons. That is not a
shortcut: a frame and a granule footprint are both convex quadrilaterals, so
their intersection is convex and every successive intersection should stay a
single polygon -- but in floating point, intersecting hundreds of nearly
identical quads accumulates sliver vertices (1,895 after fifty steps on a real
frame) until the result fragments into a MultiPolygon, which is geometrically
impossible for convex inputs. Snapping to a grid made it worse. Sampling is
immune to all of it, costs under a second for the densest frame in the archive,
and agrees to 0.1 percent across grid resolutions from 100 to 300.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon

__all__ = ["DEFAULT_GRID", "FrameCoverage", "frame_coverage"]

DEFAULT_GRID = 100
"""Samples per side across a frame's bounding box.

A frame is around 96 by 60 km, so this is roughly a kilometre between samples and
resolves a coverage fraction to a few tenths of a percent -- far finer than the
differences between acquisitions, which are themselves under one percent.
"""


@dataclass
class FrameCoverage:
    """What every acquisition of one frame contributes to it.

    Attributes
    ----------
    frame_id :
        The frame measured.
    per_granule :
        Fraction of the frame each acquisition covers, keyed by granule name.
    common :
        Fraction covered by *every* acquisition -- the area a stack can use
        without a nodata hole on some date.
    """

    frame_id: str
    per_granule: dict[str, float] = field(default_factory=dict)
    common: float = 0.0

    @property
    def n_acquisitions(self) -> int:
        """How many acquisitions were measured."""
        return len(self.per_granule)

    @property
    def typical(self) -> float:
        """Median coverage across the acquisitions."""
        return (
            float(np.median(list(self.per_granule.values())))
            if self.per_granule
            else 0.0
        )

    @property
    def worst(self) -> tuple[str, float] | None:
        """The acquisition covering least of the frame, or `None` if there are none."""
        if not self.per_granule:
            return None
        name = min(self.per_granule, key=lambda k: self.per_granule[k])
        return name, self.per_granule[name]

    @property
    def cost_of_worst(self) -> float:
        """How much area the poorest dates take off the typical one.

        Near zero means the bounds are effectively fixed; a large value means a
        few dates are holding the whole stack back and are worth dropping.
        """
        return round(self.typical - self.common, 4)

    def below(self, threshold: float) -> list[str]:
        """Granules covering less than `threshold` of the frame, worst first."""
        poor = [g for g, v in self.per_granule.items() if v < threshold]
        return sorted(poor, key=lambda g: self.per_granule[g])


def frame_coverage(
    frame_polygon: Polygon,
    footprints: Sequence[Polygon],
    names: Iterable[str],
    frame_id: str = "",
    grid: int = DEFAULT_GRID,
) -> FrameCoverage:
    """Measure what each acquisition contributes to one frame.

    Parameters
    ----------
    frame_polygon :
        The frame, in lon/lat degrees.
    footprints :
        Acquisition footprints in the same coordinates.
    names :
        Granule names, parallel to `footprints`.
    frame_id :
        Recorded on the result.
    grid :
        Samples per side; see `DEFAULT_GRID`.

    Returns
    -------
    FrameCoverage
    """
    out = FrameCoverage(frame_id=frame_id)
    if not footprints:
        return out

    x0, y0, x1, y1 = frame_polygon.bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, grid), np.linspace(y0, y1, grid))
    x, y = gx.ravel(), gy.ravel()

    inside = shapely.contains_xy(frame_polygon, x, y)
    if not inside.any():
        return out
    x, y = x[inside], y[inside]

    everywhere = np.ones(x.size, dtype=bool)
    for name, footprint in zip(names, footprints, strict=True):
        hit = shapely.contains_xy(footprint, x, y)
        out.per_granule[name] = round(float(hit.mean()), 4)
        everywhere &= hit

    out.common = round(float(everywhere.mean()), 4)
    return out
