"""Turn stripmap acquisitions into frames with pinned grids.

This is where `sm_db.tiling` (which tile), `sm_db.anx` (from which zero point)
and `sm_db.geometry` (what shape, in which projection) meet. The output is a
`Frame` per covered tile, and a frame is the unit a CSLC is produced on: one
fixed bounding box and EPSG that every acquisition of that frame geocodes into.
"""

from __future__ import annotations

import datetime
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon

from sm_db import anx as anx_mod
from sm_db import tiling
from sm_db.geometry import DEFAULT_MARGIN, DEFAULT_SNAP, pick_epsg, snap_bbox
from sm_db.granules import Granule
from sm_db.groundtrack import GroundTrack, swath_offsets, tile_polygon

__all__ = [
    "DEFAULT_GUARD",
    "DEFAULT_INSET",
    "DEFAULT_OVERLAP",
    "DEFAULT_SHIFT",
    "Frame",
    "OrbitLookup",
    "frames_for_granule",
    "merge_frames",
]

DEFAULT_SHIFT = 0.0
"""Seconds to slide every frame along its track, positive towards the later end.

On an ascending pass that is northward, on a descending pass southward. It moves
where a frame sits on the ground without changing its index or ID, which is the
knob for walking a boundary off a target it happens to cut through.

Because the ID does not move with it, two databases built with different shifts
describe different ground under the same names: pick a value, then rebuild
everything that will be stacked together.
"""

DEFAULT_INSET = 0.0
"""Metres taken off each side of a frame across track, positive shrinking it.

Negative widens. The cross-track extent is measured from the granule footprint,
which reaches the full swath; trimming it drops the noisy near- and far-range
edges, and widening is only useful when neighbouring tracks must overlap.
"""

DEFAULT_OVERLAP = 0.0
"""Seconds of along-track overlap added to each end of a frame.

Tiles abut exactly by default, so a target that straddles a boundary -- an island,
a volcano, a city -- is split between two frames and whole in neither. Padding
every frame by a second or two at both ends makes neighbours overlap, the way IW
bursts do, so such a target lands complete inside at least one frame.

It changes only a frame's geometry and its pinned box, never its index or its ID,
so widening an existing database renumbers nothing. The first and last frame of a
scene can gain some nodata, since the pad may reach past where the scene stops.
"""

DEFAULT_GUARD = 1.0
"""Timing slack applied to each end of a scene before claiming tiles [s].

ASF reports scene start and stop rounded to whole seconds, so a scene's true
extent can be a second shorter than advertised at either end. Requiring a tile to
be covered with this much to spare keeps the set of frames an acquisition claims
insensitive to that rounding.
"""


@dataclass(frozen=True)
class Frame:
    """One stripmap frame: an ID and the grid every acquisition of it lands on.

    Attributes
    ----------
    frame_id :
        e.g. ``t095_000003_s3``.
    track :
        Relative orbit number.
    index :
        Frame index within the track.
    beam :
        Stripmap beam, ``s1`` through ``s6``.
    epsg :
        Projected EPSG code the bbox is expressed in.
    xmin, ymin, xmax, ymax :
        Pinned bounding box in projected metres.
    polygon :
        Frame footprint in lon/lat degrees.
    shift, overlap, inset :
        The geometry adjustment this frame was built with, in seconds, seconds and
        metres. Stored so a database says how each frame was framed, and so a
        rebuild reproduces it.
    fill_pct :
        Percentage of the pinned box the footprint actually covers. A frame is a
        quadrilateral rotated by the orbit heading, so its axis-aligned box is
        always larger; the remainder geocodes to nodata in every product of that
        frame, whatever the acquisition. Around 45-50 percent is normal, and a
        much lower value marks a frame whose products would be mostly empty.
    """

    frame_id: str
    track: int
    index: int
    beam: str
    epsg: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    polygon: Polygon
    fill_pct: float = 0.0
    shift: float = 0.0
    overlap: float = 0.0
    inset: float = 0.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Bounding box as ``(xmin, ymin, xmax, ymax)``."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def nodata_pct(self) -> float:
        """Percentage of the pinned box that is nodata in every product."""
        return round(100.0 - self.fill_pct, 1)


class OrbitLookup:
    """Find the orbit file covering an acquisition, from a directory of EOFs.

    Matches on the platform and the validity window encoded in the standard EOF
    filename, so no file is opened until one is needed. Parsed orbits are cached,
    because a campaign's granules usually share a handful of daily files.

    Parsed orbits are cached, but only the few most recently used: a day's EOF
    holds ~9,000 state vectors, and holding every file of a multi-year archive
    would run to many gigabytes. Granules are processed in time order, so a
    handful of entries is enough to hit the cache nearly every time.

    Parameters
    ----------
    directory :
        Directory holding ``.EOF`` files.
    cache_size :
        How many parsed orbit files to keep in memory.
    """

    def __init__(self, directory: str | Path, cache_size: int = 4) -> None:
        self.directory = Path(directory)
        self.cache_size = cache_size
        self._cache: OrderedDict[Path, anx_mod.OrbitStateVectors] = OrderedDict()

    def _candidates(
        self, platform: str
    ) -> list[tuple[datetime.datetime, datetime.datetime, Path]]:
        """Return ``(valid_start, valid_stop, path)`` for one platform's EOFs."""
        out = []
        for path in sorted(self.directory.glob(f"{platform}_OPER_AUX_*.EOF")):
            # ..._V<YYYYMMDDTHHMMSS>_<YYYYMMDDTHHMMSS>.EOF
            stem = path.stem
            try:
                window = stem.rsplit("_V", 1)[1]
                start_text, stop_text = window.split("_")
                start = datetime.datetime.strptime(start_text, "%Y%m%dT%H%M%S")
                stop = datetime.datetime.strptime(stop_text, "%Y%m%dT%H%M%S")
            except (IndexError, ValueError):
                continue
            out.append((start, stop, path))
        return out

    def covers(self, granule: Granule) -> bool:
        """Report whether an orbit for a granule is already on disk.

        Checks filenames only, so it is cheap enough to call for every granule in
        a catalog before deciding what to download.

        Parameters
        ----------
        granule :
            The acquisition to cover.

        Returns
        -------
        bool
        """
        return self._match(granule) is not None

    def _match(self, granule: Granule) -> Path | None:
        """Return the path of the EOF covering a granule, without reading it."""
        platform = granule.name[:3].upper()
        # The ANX can be up to one revolution before the scene, so the orbit has
        # to start well before it, not merely cover it.
        need_from = granule.start - datetime.timedelta(seconds=anx_mod.T_ORBIT)
        for start, stop, path in self._candidates(platform):
            if start <= need_from and stop >= granule.stop:
                return path
        return None

    def find(self, granule: Granule) -> anx_mod.OrbitStateVectors:
        """Return the state vectors covering a granule.

        Parameters
        ----------
        granule :
            The acquisition to cover.

        Returns
        -------
        OrbitStateVectors

        Raises
        ------
        FileNotFoundError
            If no EOF in the directory covers the acquisition. The ANX search
            needs a full revolution of lead-in, so an orbit that merely starts
            at the scene is not enough.
        """
        path = self._match(granule)
        if path is None:
            need_from = granule.start - datetime.timedelta(seconds=anx_mod.T_ORBIT)
            raise FileNotFoundError(
                f"No orbit file in {self.directory} covers {granule.name} "
                f"({need_from} to {granule.stop})"
            )
        if path in self._cache:
            self._cache.move_to_end(path)
        else:
            self._cache[path] = anx_mod.read_orbit_file(path)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return self._cache[path]


def frames_for_granule(
    granule: Granule,
    orbit: anx_mod.OrbitStateVectors,
    tile_seconds: float = tiling.DEFAULT_TILE_SECONDS,
    margin: float = DEFAULT_MARGIN,
    snap: float = DEFAULT_SNAP,
    guard: float = DEFAULT_GUARD,
    overlap: float = DEFAULT_OVERLAP,
    shift: float = DEFAULT_SHIFT,
    inset: float = DEFAULT_INSET,
    overrides: dict[str, dict] | None = None,
) -> list[Frame]:
    """Return the frames a granule fully covers, with their pinned grids.

    A scene that crosses the ascending node is handled in two pieces: the tiles
    before the node belong to the granule's track, those after it to the next
    track with the index restarting from the new node.

    Parameters
    ----------
    granule :
        The acquisition.
    orbit :
        State vectors covering the acquisition and the revolution before it.
    tile_seconds :
        Along-track tile length in seconds.
    margin, snap :
        Bounding box padding and lattice, in metres.
    guard :
        Timing slack in seconds. The scene is shrunk by this at each end before
        deciding which tiles it fills, because ASF rounds scene times to whole
        seconds and a tile claimed on a rounded time might not truly be covered.
    overlap :
        Seconds added to each end of every frame's geometry, so neighbouring
        frames overlap instead of merely abutting. See `DEFAULT_OVERLAP`.
    shift :
        Seconds to slide every frame along track. See `DEFAULT_SHIFT`.
    inset :
        Metres trimmed from each side across track. See `DEFAULT_INSET`.
    overrides :
        Per-frame geometry, keyed by frame ID, each a mapping with any of
        ``shift``, ``overlap`` and ``inset``. A frame listed here ignores the
        corresponding argument above. Adjusting one frame that cuts through a
        target is the normal case; the arguments are the blunt instrument that
        moves every frame at once.

    Returns
    -------
    list of Frame
        In along-track order; empty if the scene is too short to fill a tile.
    """
    scene_start, scene_stop = granule.start, granule.stop
    node = anx_mod.ascending_node_time(orbit, scene_start)

    # The next crossing bounds this track's index range. Look a little over one
    # revolution ahead so it is found even when the scene sits just before it.
    following = anx_mod.ascending_node_times(
        orbit,
        node + datetime.timedelta(seconds=1),
        node + datetime.timedelta(seconds=2 * anx_mod.T_ORBIT),
    )
    period = (following[0] - node).total_seconds() if following else None

    segments = [(granule.track, node, period)]
    if period is not None and (scene_stop - node).total_seconds() > period:
        segments.append((tiling.next_track(granule.track), following[0], None))

    frames = []
    for track, segment_node, segment_period in segments:
        # Shrink the scene by the timing guard before asking which tiles it fills,
        # so a tile is only claimed when it stays covered under the worst rounding.
        start_since = (scene_start - segment_node).total_seconds() + guard
        stop_since = (scene_stop - segment_node).total_seconds() - guard
        if stop_since <= start_since:
            continue
        tiles = tiling.tiles_covered_by(
            max(start_since, 0.0), stop_since, tile_seconds, segment_period
        )
        for tile in tiles:
            # The ID is known before the geometry is built, which is what lets a
            # single frame carry its own bounds without disturbing its neighbours.
            frame_id = tiling.format_frame_id(track, tile.index, granule.beam_mode)
            own = (overrides or {}).get(frame_id, {})
            frames.append(
                _build_frame(
                    granule,
                    orbit,
                    track,
                    tile,
                    segment_node,
                    margin,
                    snap,
                    float(own.get("overlap", overlap)),
                    float(own.get("shift", shift)),
                    float(own.get("inset", inset)),
                )
            )
    return frames


def _build_frame(
    granule: Granule,
    orbit: anx_mod.OrbitStateVectors,
    track: int,
    tile: tiling.Tile,
    node: datetime.datetime,
    margin: float,
    snap: float,
    overlap: float = DEFAULT_OVERLAP,
    shift: float = DEFAULT_SHIFT,
    inset: float = DEFAULT_INSET,
) -> Frame:
    """Assemble one `Frame` from a tile and the granule that covers it.

    The along-track extent comes from the orbit and the cross-track extent from
    the granule footprint; see `sm_db.groundtrack` for why the granule's own
    timing is not used.
    """
    epsg = pick_epsg(*_centroid_lonlat(granule.footprint))
    # The pad widens the geometry only; `tile.index` and so the frame ID are
    # untouched, which is what lets overlap be changed without renumbering.
    tile_start = node + datetime.timedelta(seconds=tile.start - overlap + shift)
    tile_stop = node + datetime.timedelta(seconds=tile.stop + overlap + shift)

    ground_track = GroundTrack(
        orbit, epsg, min(tile_start, granule.start), max(tile_stop, granule.stop)
    )
    projected_footprint = _project(granule.footprint, epsg)
    near, far = swath_offsets(ground_track, projected_footprint)
    # `near` is the left-hand offset and `far` the right-hand one, so trimming
    # moves them towards each other whichever side of the track the swath is on.
    near, far = near + inset, far - inset
    if near >= far:
        raise ValueError(
            f"inset of {inset} m leaves no swath for {granule.name}: "
            f"the frame is only {far - near + 2 * inset:.0f} m wide"
        )

    projected = tile_polygon(ground_track, tile_start, tile_stop, near, far)
    xmin, ymin, xmax, ymax = snap_bbox(*projected.bounds, margin=margin, snap=snap)

    # Measured here, where the tile is still in metres: the ratio is meaningless
    # in degrees, where a square is not square.
    box_area = (xmax - xmin) * (ymax - ymin)
    fill_pct = round(100.0 * projected.area / box_area, 1) if box_area else 0.0

    return Frame(
        frame_id=tiling.format_frame_id(track, tile.index, granule.beam_mode),
        track=track,
        index=tile.index,
        beam=granule.beam_mode.lower(),
        epsg=epsg,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        polygon=_unproject(projected, epsg),
        fill_pct=fill_pct,
        shift=shift,
        overlap=overlap,
        inset=inset,
    )


def _centroid_lonlat(footprint: Polygon) -> tuple[float, float]:
    """Return a footprint's centroid as ``(lon, lat)``."""
    c = footprint.centroid
    return c.x, c.y


def _project(polygon: Polygon, epsg: int) -> Polygon:
    """Project a lon/lat polygon into `epsg`."""
    from pyproj import Transformer
    from shapely.ops import transform

    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    return transform(lambda x, y: tf.transform(x, y), polygon)


def _unproject(polygon: Polygon, epsg: int) -> Polygon:
    """Bring a projected polygon back to lon/lat degrees."""
    from pyproj import Transformer
    from shapely.ops import transform

    tf = Transformer.from_crs(epsg, 4326, always_xy=True)
    return transform(lambda x, y: tf.transform(x, y), polygon)


def merge_frames(existing: dict[str, Frame], new: list[Frame]) -> dict[str, Frame]:
    """Add frames to a collection, keeping the definition already present.

    A frame's grid is frozen by the first acquisition that defines it. Later
    acquisitions of the same frame are expected to agree to within the orbital
    tube and are discarded rather than averaged or unioned, so that rebuilding a
    database with more dates never moves a grid that products already sit on. Use
    `sm_db.cli` ``check`` to surface any that disagree by more than a pixel.

    Parameters
    ----------
    existing :
        Frames already defined, keyed by frame ID. Not modified.
    new :
        Frames to add.

    Returns
    -------
    dict
        The merged mapping.
    """
    merged = dict(existing)
    for frame in new:
        merged.setdefault(frame.frame_id, frame)
    return merged
