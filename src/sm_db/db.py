"""Read and write the stripmap frame database.

The schema is not ours to choose. COMPASS looks a grid up through
``compass.utils.helpers.burst_bboxes_from_db``, which runs::

    SELECT * FROM burst_id_map WHERE burst_id_jpl IN (...)

and reads ``burst_id_jpl, epsg, xmin, ymin, xmax, ymax``, treating the bounds as
metres in that row's own EPSG. ``compass_batch`` assumes the same table in three
more places, including the filter that decides which bursts a worker will process
and the widening of the DEM request.

So `sm_db` writes exactly that table and nothing that would confuse it. A frame
is a row; ``burst_id_jpl`` holds the frame ID. Everything specific to stripmap --
the beam, the tile length, the footprint -- goes in a separate ``frames`` table
that COMPASS never reads, alongside a ``metadata`` table recording how the file
was built.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from shapely import wkt

from sm_db.frames import Frame

__all__ = ["BURST_ID_MAP_SCHEMA", "read_frames", "write_database"]

BURST_ID_MAP_SCHEMA = """
CREATE TABLE burst_id_map (
    OGC_FID      INTEGER PRIMARY KEY,
    burst_id_jpl TEXT UNIQUE NOT NULL,
    epsg         INTEGER NOT NULL,
    xmin         INTEGER NOT NULL,
    ymin         INTEGER NOT NULL,
    xmax         INTEGER NOT NULL,
    ymax         INTEGER NOT NULL
)
"""

_FRAMES_SCHEMA = """
CREATE TABLE frames (
    burst_id_jpl TEXT PRIMARY KEY REFERENCES burst_id_map(burst_id_jpl),
    track        INTEGER NOT NULL,
    frame_index  INTEGER NOT NULL,
    beam         TEXT NOT NULL,
    fill_pct      REAL NOT NULL,
    shift_s      REAL NOT NULL DEFAULT 0,
    overlap_s    REAL NOT NULL DEFAULT 0,
    inset_m      REAL NOT NULL DEFAULT 0,
    geometry_wkt TEXT NOT NULL
)
"""

_METADATA_SCHEMA = """
CREATE TABLE metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


def write_database(
    frames: Iterable[Frame],
    path: str | Path,
    tile_seconds: float,
    margin: float,
    snap: float,
    extra: dict[str, object] | None = None,
) -> int:
    """Write frames to a sqlite database COMPASS can read.

    Overwrites `path` if it exists, so a build is reproducible rather than
    accumulating across runs.

    Parameters
    ----------
    frames :
        Frames to write.
    path :
        Destination ``.sqlite3`` file.
    tile_seconds, margin, snap :
        Build parameters, recorded in ``metadata`` so a database can be traced
        back to how it was made.
    extra :
        Further key/value pairs for ``metadata``.

    Returns
    -------
    int
        Number of frames written.
    """
    path = Path(path)
    path.unlink(missing_ok=True)

    ordered = sorted(frames, key=lambda f: (f.track, f.beam, f.index))

    with sqlite3.connect(path) as con:
        con.execute(BURST_ID_MAP_SCHEMA)
        con.execute(_FRAMES_SCHEMA)
        con.execute(_METADATA_SCHEMA)

        con.executemany(
            "INSERT INTO burst_id_map "
            "(OGC_FID, burst_id_jpl, epsg, xmin, ymin, xmax, ymax) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (i, f.frame_id, f.epsg, f.xmin, f.ymin, f.xmax, f.ymax)
                for i, f in enumerate(ordered, start=1)
            ],
        )
        con.executemany(
            "INSERT INTO frames "
            "(burst_id_jpl, track, frame_index, beam, fill_pct, "
            "shift_s, overlap_s, inset_m, geometry_wkt) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f.frame_id,
                    f.track,
                    f.index,
                    f.beam,
                    f.fill_pct,
                    f.shift,
                    f.overlap,
                    f.inset,
                    f.polygon.wkt,
                )
                for f in ordered
            ],
        )

        meta: dict[str, object] = {
            "tile_seconds": tile_seconds,
            "margin": margin,
            "snap": snap,
            "n_frames": len(ordered),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        meta.update(extra or {})
        con.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in meta.items()],
        )

    return len(ordered)


def read_frames(
    path: str | Path, frame_ids: Iterable[str] | None = None
) -> list[Frame]:
    """Read frames back out of a database.

    Parameters
    ----------
    path :
        Database file.
    frame_ids :
        Only these IDs; `None` (default) reads every frame.

    Returns
    -------
    list of Frame
        Ordered by track, beam and index.
    """
    query = """
        SELECT b.burst_id_jpl, b.epsg, b.xmin, b.ymin, b.xmax, b.ymax,
               f.track, f.frame_index, f.beam, f.fill_pct,
               f.shift_s, f.overlap_s, f.inset_m, f.geometry_wkt
        FROM burst_id_map b JOIN frames f USING (burst_id_jpl)
    """
    params: tuple[str, ...] = ()
    if frame_ids is not None:
        ids = tuple(frame_ids)
        if not ids:
            return []
        query += f" WHERE b.burst_id_jpl IN ({', '.join('?' for _ in ids)})"
        params = ids
    query += " ORDER BY f.track, f.beam, f.frame_index"

    with sqlite3.connect(path) as con:
        rows = con.execute(query, params).fetchall()

    return [
        Frame(
            frame_id=r[0],
            track=r[6],
            index=r[7],
            beam=r[8],
            epsg=r[1],
            xmin=r[2],
            ymin=r[3],
            xmax=r[4],
            ymax=r[5],
            polygon=wkt.loads(r[13]),
            fill_pct=r[9],
            shift=r[10],
            overlap=r[11],
            inset=r[12],
        )
        for r in rows
    ]
