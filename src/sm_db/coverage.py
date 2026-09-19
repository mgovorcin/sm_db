"""How much of a frame every acquisition actually fills.

A frame's pinned box is fixed, but the data landing in it is not. The
along-track extent is guaranteed -- a tile is only claimed when a scene covers it
completely -- yet across track the swath drifts between passes, and a frame's
width was measured from whichever acquisition happened to define it first. So the
area valid in *every* date, which is all a time series can use, is the
intersection of the acquisitions rather than the frame itself.

This measures that intersection, and answers the question it raises: when a
handful of passes are narrower than the rest, is it better to drop them and keep
the area, or keep them and lose it? The trade is real either way, so the job here
is to quantify it rather than to pick.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

from sm_db.frames import Frame
from sm_db.granules import Granule

__all__ = ["FrameCoverage", "drop_curve", "frame_coverage"]


@dataclass
class FrameCoverage:
    """What every acquisition of one frame contributes to it.

    Attributes
    ----------
    frame_id :
        The frame measured.
    n_acquisitions :
        How many acquisitions were considered.
    per_granule :
        Fraction of the frame each acquisition covers, keyed by granule name.
    common :
        Fraction of the frame covered by *every* acquisition -- the area a stack
        can use without a nodata hole on some date.
    """

    frame_id: str
    n_acquisitions: int
    per_granule: dict[str, float] = field(default_factory=dict)
    common: float = 0.0

    @property
    def worst(self) -> tuple[str, float] | None:
        """The acquisition covering least of the frame, or `None` if there are none."""
        if not self.per_granule:
            return None
        name = min(self.per_granule, key=lambda k: self.per_granule[k])
        return name, self.per_granule[name]

    def below(self, threshold: float) -> list[str]:
        """Granules covering less than `threshold` of the frame, worst first."""
        poor = [g for g, v in self.per_granule.items() if v < threshold]
        return sorted(poor, key=lambda g: self.per_granule[g])


def _project(polygon: Polygon, epsg: int) -> Polygon:
    """Project a lon/lat polygon into `epsg`."""
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    return transform(lambda x, y: tf.transform(x, y), polygon)


def frame_coverage(frame: Frame, granules: Iterable[Granule]) -> FrameCoverage:
    """Measure what each acquisition contributes to one frame.

    Parameters
    ----------
    frame :
        The frame to measure.
    granules :
        The acquisitions covering it.

    Returns
    -------
    FrameCoverage
    """
    shape = _project(frame.polygon, frame.epsg)
    area = shape.area
    out = FrameCoverage(frame_id=frame.frame_id, n_acquisitions=0)
    if area <= 0:
        return out

    common: Polygon | None = None
    for granule in granules:
        clipped = shape.intersection(_project(granule.footprint, frame.epsg))
        out.per_granule[granule.name] = round(clipped.area / area, 4)
        out.n_acquisitions += 1
        common = clipped if common is None else common.intersection(clipped)

    out.common = round(common.area / area, 4) if common is not None else 0.0
    return out


def drop_curve(coverage: FrameCoverage, max_drop: int = 5) -> list[tuple[int, float]]:
    """Show what dropping the worst acquisitions would buy.

    The common area is bounded by the narrowest pass, so removing the narrowest
    can only raise it. This lists that trade: how many acquisitions are given up,
    and the fraction of the frame the rest still share.

    It is an upper bound rather than an exact recomputation -- the true
    intersection of the survivors needs the geometry again -- but it is the right
    shape for deciding, because the binding constraint is the worst pass.

    Parameters
    ----------
    coverage :
        A measured frame.
    max_drop :
        How far down the list to go.

    Returns
    -------
    list of tuple
        ``(dropped, best_possible_common)`` for 0 to `max_drop` acquisitions.
    """
    ordered = sorted(coverage.per_granule.values())
    curve = []
    for k in range(min(max_drop, max(len(ordered) - 1, 0)) + 1):
        remaining = ordered[k:]
        curve.append((k, round(min(remaining), 4) if remaining else 0.0))
    return curve
