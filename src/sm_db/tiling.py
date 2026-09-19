"""Deterministic along-track tiling of Sentinel-1 Stripmap acquisitions.

This module is the **specification** for the Stripmap frame ID. Anything that
needs to produce or consume one -- ``sm_db`` itself, the ``s1reader`` stripmap
reader, a planner -- must agree with the arithmetic here.

Why a new scheme
----------------
IW gets a stable burst ID for free: ESA defines a fixed along-track burst grid
locked to the ascending node, so a repeat pass lands on the same ``burst_id``.
Stripmap has no such grid. The current stripmap reader borrows the IW machinery
and bins the *mid-scene* sensing time into a 2.758273 s ESA burst interval, which
makes the ID move whenever ESA slices a datatake differently. That is a
per-acquisition label, not a frame.

Here a frame is a fixed slice of the orbit instead: quantize time-since-ascending
-node into tiles of ``tile_seconds``, and number them per relative orbit. The
tile boundaries depend only on the orbit, so every repeat pass over the same
ground lands in the same tile.

    frame index = 1 + floor((t_anx - T_PRE) / tile_seconds)
    frame ID    = t{track:03d}_{index:06d}_{beam}

``T_PRE`` is ESA's preamble constant, carried over from
``s1reader.s1_burst_id.S1BurstId`` so the two schemes share a phase origin.
Unlike ESA Eq. 9-89 we do *not* add ``(track - 1) * T_ORBIT`` to make the index
globally unique, because the track is already in the ID; the index is local to
its relative orbit and stays comfortably inside six digits (~1185 tiles per orbit
at the 5 s default).

Which ANX
---------
**The ANX is the most recent orbit-derived ascending crossing strictly before the
scene's first line.** This is a deliberate narrowing of what the stripmap reader
does today, and it has to be stated because the choice is not obvious:

* ESA's annotated ``ascendingNodeTime`` is often a full orbit stale. All four
  S1C S3 scenes on track 95 in the 2026 campaign annotate an ANX ~5935 s back,
  while the true crossing is ~10 s back.
* ``s1reader.get_ascending_node_time_orbit`` is normally called *with* that
  annotation, and then returns the orbit crossing nearest to it -- so it inherits
  the staleness, and ``S1BurstId`` compensates by subtracting a **nominal**
  ``T_ORBIT``. Because the true period differs from the nominal one, that
  round trip lands ~1 s away from the true crossing.

One second is a fifth of a tile. Rather than carry that wobble, both `sm_db` and
the stripmap reader take the nearest true crossing before the scene and no
correction, which keeps ``t_anx`` in ``[0, T_ORBIT)`` by construction and needs
no annotation at all.

Coverage rule
-------------
A scene claims a tile only when it covers that tile **completely** (see
``tiles_covered_by``). That is what makes a stack valid edge to edge: every
acquisition of a frame fills the whole frame, so no date contributes a partial
row. It also means ``tile_seconds`` must be meaningfully shorter than a scene --
a nominal 20 s S3 slice yields 3-4 full tiles at the 5 s default, but would
usually yield *zero* at 20 s.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "DEFAULT_TILE_SECONDS",
    "T_ORBIT",
    "T_PRE",
    "Tile",
    "format_frame_id",
    "frame_index",
    "next_track",
    "parse_frame_id",
    "tile_bounds",
    "tiles_covered_by",
]

# ESA Level 1 Detailed Algorithm Definition, Table 9-7. Mirrors
# s1reader.s1_burst_id.S1BurstId; kept as literals so this module stays stdlib-only.
T_PRE = 2.299849
"""Preamble time interval [s]."""

T_ORBIT = (12 * 86400.0) / 175.0
"""Nominal Sentinel-1 orbit period [s]: 175 orbits per 12-day repeat cycle."""

DEFAULT_TILE_SECONDS = 5.0
"""Default along-track tile length [s], roughly 35 km on the ground."""

_FRAME_ID_RE = re.compile(r"^t(\d{3})_(\d{6})_(s[1-6])$", re.IGNORECASE)

# Slack on the coverage test. Tile bounds and scene bounds are accumulated by
# different sums, so a scene that lines up exactly with a boundary can miss it by
# an ulp. A microsecond is ~7 mm along track -- far below anything that matters.
_EPS = 1e-6


@dataclass(frozen=True)
class Tile:
    """One along-track tile of a relative orbit.

    Attributes
    ----------
    index :
        Frame index within the track, 1-based.
    start, stop :
        Tile bounds as seconds since the ascending node.
    """

    index: int
    start: float
    stop: float

    @property
    def mid(self) -> float:
        """Tile centre, in seconds since the ascending node."""
        return 0.5 * (self.start + self.stop)


def frame_index(
    time_since_anx: float, tile_seconds: float = DEFAULT_TILE_SECONDS
) -> int:
    """Return the 1-based frame index containing a time since the ascending node.

    Parameters
    ----------
    time_since_anx :
        Seconds elapsed since the ascending node crossing.
    tile_seconds :
        Along-track tile length in seconds.

    Examples
    --------
    >>> frame_index(T_PRE, tile_seconds=5.0)
    1
    >>> frame_index(T_PRE + 5.0, tile_seconds=5.0)
    2
    """
    if tile_seconds <= 0:
        raise ValueError(f"tile_seconds must be positive, got {tile_seconds}")
    return 1 + math.floor((time_since_anx - T_PRE) / tile_seconds)


def tile_bounds(
    index: int, tile_seconds: float = DEFAULT_TILE_SECONDS
) -> tuple[float, float]:
    """Return ``(start, stop)`` of a frame index, in seconds since the ANX.

    Inverse of `frame_index`.

    Examples
    --------
    >>> tile_bounds(1, tile_seconds=5.0)
    (2.299849, 7.299849)
    """
    start = T_PRE + (index - 1) * tile_seconds
    return start, start + tile_seconds


def next_track(track: int) -> int:
    """Return the relative orbit number following `track`, wrapping 175 to 1.

    Examples
    --------
    >>> next_track(95), next_track(175)
    (96, 1)
    """
    return 1 if track == 175 else track + 1


def format_frame_id(track: int, index: int, beam: str) -> str:
    """Render a frame ID, e.g. ``t095_000123_s3``.

    The six-digit index field deliberately matches the IW burst ID layout
    (``t073_154917_iw2``) so that every downstream consumer -- the OPERA granule
    name, the internal-ID regex in ``compass_batch.planning``, ``S1BurstId.from_str``
    -- keeps working unchanged. Only the meaning of the field differs.

    Examples
    --------
    >>> format_frame_id(95, 123, "S3")
    't095_000123_s3'
    """
    beam = beam.lower()
    if not re.fullmatch(r"s[1-6]", beam):
        raise ValueError(f"Not a stripmap beam mode: {beam!r}")
    if not 0 < index < 1_000_000:
        raise ValueError(f"Frame index out of range for six digits: {index}")
    return f"t{track:03d}_{index:06d}_{beam}"


def parse_frame_id(frame_id: str) -> tuple[int, int, str]:
    """Parse a frame ID back into ``(track, index, beam)``.

    Examples
    --------
    >>> parse_frame_id("t095_000123_s3")
    (95, 123, 's3')
    """
    m = _FRAME_ID_RE.match(frame_id)
    if not m:
        raise ValueError(f"Not a stripmap frame ID: {frame_id!r}")
    return int(m.group(1)), int(m.group(2)), m.group(3).lower()


def tiles_covered_by(
    start_since_anx: float,
    stop_since_anx: float,
    tile_seconds: float = DEFAULT_TILE_SECONDS,
    orbit_period: float | None = None,
) -> list[Tile]:
    """Return every tile a scene covers **completely**.

    Partially covered tiles at the two ends of the scene are dropped, so each
    returned frame is filled edge to edge by this acquisition. That is what lets
    a whole stack share one grid with no date contributing a short row.

    Parameters
    ----------
    start_since_anx, stop_since_anx :
        Scene first- and last-line sensing times, in seconds since the ANX.
    tile_seconds :
        Along-track tile length in seconds.
    orbit_period :
        Seconds from this ANX to the next. Tiles that would run past it are
        dropped, because beyond the crossing the track number changes and the
        index restarts; `sm_db.frames.frames_for_scene` picks the remainder up
        against the following ANX. `None` (default) means do not cap, which is
        right when the caller already knows the scene does not reach the node.

    Returns
    -------
    list of Tile
        In along-track order; empty when the scene is shorter than one tile, or
        straddles the boundaries badly enough that no tile is filled.

    Raises
    ------
    ValueError
        If the scene stops before it starts.

    Examples
    --------
    A 20 s scene starting right on a tile boundary fills exactly 4 tiles:

    >>> [t.index for t in tiles_covered_by(T_PRE, T_PRE + 20.0, 5.0)]
    [1, 2, 3, 4]

    Shift it by 1 s and the two end tiles are only partial, so 3 survive:

    >>> [t.index for t in tiles_covered_by(T_PRE + 1.0, T_PRE + 21.0, 5.0)]
    [2, 3, 4]
    """
    if stop_since_anx < start_since_anx:
        raise ValueError(
            "Scene stops before it starts: "
            f"{start_since_anx} > {stop_since_anx} seconds since ANX"
        )

    limit = (
        stop_since_anx if orbit_period is None else min(stop_since_anx, orbit_period)
    )

    tiles = []
    for index in range(
        frame_index(start_since_anx, tile_seconds),
        frame_index(limit, tile_seconds) + 1,
    ):
        t0, t1 = tile_bounds(index, tile_seconds)
        if t0 >= start_since_anx - _EPS and t1 <= limit + _EPS:
            tiles.append(Tile(index, t0, t1))
    return tiles
