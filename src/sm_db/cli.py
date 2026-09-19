"""Command line interface: ``sm-db``."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import click

from sm_db import db as db_mod
from sm_db import granules as granules_mod
from sm_db.frames import Frame, OrbitLookup, frames_for_granule, merge_frames
from sm_db.geometry import DEFAULT_MARGIN, DEFAULT_SNAP
from sm_db.granules import SM_BEAM_MODES
from sm_db.tiling import DEFAULT_TILE_SECONDS, parse_frame_id

DEFAULT_TOLERANCE = 250.0
"""Largest frame bbox disagreement ``check`` accepts between repeat passes [m]."""

DEFAULT_MIN_FILL = 0.0
"""Smallest share of a pinned box the footprint may cover, in percent.

Zero keeps every frame. Raise it to drop frames whose products would be mostly
nodata -- the corner cases where a tile meets its box at an awkward angle.
"""

__all__ = ["cli"]

_DB_OPTION = click.option(
    "-d",
    "--database",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="sm_frames.sqlite3",
    show_default=True,
    help="Frame database to read.",
)


@click.group()
@click.version_option()
def cli() -> None:
    """Build and query the Sentinel-1 stripmap frame database.

    A frame is a fixed along-track slice of a relative orbit. Every acquisition
    covering a frame geocodes into that frame's pinned bounding box, so products
    from different dates share one grid and stack.
    """


@cli.command()
@click.option("--start", required=True, help="Search start date, e.g. 2026-02-01.")
@click.option("--end", required=True, help="Search end date.")
@click.option(
    "--beam",
    "beams",
    multiple=True,
    type=click.Choice(SM_BEAM_MODES, case_sensitive=False),
    help="Stripmap beams to include. Repeatable; default all.",
)
@click.option(
    "--track", "tracks", multiple=True, type=int, help="Relative orbit numbers."
)
@click.option("--intersects", help="Area of interest as WKT.")
@click.option(
    "--orbit-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of POEORB/RESORB .EOF files covering the search window.",
)
@click.option(
    "--catalog",
    type=click.Path(path_type=Path),
    help="Read granules from this JSON catalog instead of querying ASF. "
    "Written out after a query when it does not exist.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="sm_frames.sqlite3",
    show_default=True,
    help="Database to write.",
)
@click.option(
    "--tile-seconds",
    default=DEFAULT_TILE_SECONDS,
    show_default=True,
    help="Along-track frame length in seconds.",
)
@click.option(
    "--margin", default=DEFAULT_MARGIN, show_default=True, help="Bbox padding [m]."
)
@click.option(
    "--snap", default=DEFAULT_SNAP, show_default=True, help="Bbox lattice [m]."
)
@click.option(
    "--shift-seconds",
    default=0.0,
    show_default=True,
    help="Slide every frame along track; positive is towards the later end.",
)
@click.option(
    "--overlap-seconds",
    default=0.0,
    show_default=True,
    help="Extend every frame at both ends so neighbours overlap.",
)
@click.option(
    "--inset-m",
    default=0.0,
    show_default=True,
    help="Trim this many metres off each side across track; negative widens.",
)
@click.option(
    "--overrides",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON of per-frame geometry, keyed by frame id, as the viewer exports it: "
    '{"t095_000003_s3": {"shift": 1.2, "overlap": 0.5, "inset": 2000}}. '
    "A frame listed here ignores the global --shift/--overlap/--inset.",
)
@click.option(
    "--min-fill",
    default=DEFAULT_MIN_FILL,
    show_default=True,
    help="Drop frames whose footprint covers less than this percent of their box.",
)
@click.option(
    "--geojson",
    type=click.Path(path_type=Path),
    help="Also write the frame footprints here, for inspection in a GIS.",
)
def build(
    start: str,
    end: str,
    beams: tuple[str, ...],
    tracks: tuple[int, ...],
    intersects: str | None,
    orbit_dir: Path,
    catalog: Path | None,
    output: Path,
    tile_seconds: float,
    margin: float,
    snap: float,
    shift_seconds: float,
    overlap_seconds: float,
    inset_m: float,
    overrides: Path | None,
    min_fill: float,
    geojson: Path | None,
) -> None:
    """Define frames from stripmap acquisitions and write the database."""
    if catalog and catalog.exists():
        granules = granules_mod.load_catalog(catalog)
        click.echo(f"Read {len(granules)} granules from {catalog}")
    else:
        granules = granules_mod.query_asf(
            start=start,
            end=end,
            beam_modes=[b.upper() for b in beams] or SM_BEAM_MODES,
            tracks=tracks or None,
            intersects_wkt=intersects,
        )
        click.echo(f"Found {len(granules)} granules from ASF")
        if catalog:
            granules_mod.save_catalog(granules, catalog)
            click.echo(f"Wrote catalog {catalog}")

    if not granules:
        raise click.ClickException("No granules matched; nothing to build.")

    per_frame = json.loads(overrides.read_text()) if overrides else {}
    if per_frame:
        click.echo(f"{len(per_frame)} frame(s) carry their own bounds")

    orbits = OrbitLookup(orbit_dir)
    defined: dict[str, Frame] = {}
    skipped: list[str] = []

    with click.progressbar(granules, label="Defining frames") as bar:
        for granule in bar:
            try:
                orbit = orbits.find(granule)
            except FileNotFoundError as exc:
                skipped.append(f"{granule.name}: {exc}")
                continue
            found = frames_for_granule(
                granule,
                orbit,
                tile_seconds=tile_seconds,
                margin=margin,
                snap=snap,
                overlap=overlap_seconds,
                shift=shift_seconds,
                inset=inset_m,
                overrides=per_frame,
            )
            if not found:
                skipped.append(f"{granule.name}: too short to fill a frame")
            found = [f for f in found if f.fill_pct >= min_fill]
            defined = merge_frames(defined, found)

    for message in skipped:
        click.echo(f"  skipped {message}", err=True)

    n = db_mod.write_database(
        defined.values(),
        output,
        tile_seconds=tile_seconds,
        margin=margin,
        snap=snap,
        extra={
            "start": start,
            "end": end,
            "n_granules": len(granules),
            "shift_seconds": shift_seconds,
            "overlap_seconds": overlap_seconds,
            "inset_m": inset_m,
        },
    )
    click.echo(f"Wrote {n} frames to {output}")

    if geojson:
        _write_geojson(defined.values(), geojson)
        click.echo(f"Wrote {geojson}")


def _write_geojson(frames: list[Frame], path: Path) -> None:
    """Write frame footprints as a GeoJSON FeatureCollection."""
    from shapely.geometry import mapping

    features = [
        {
            "type": "Feature",
            "geometry": mapping(f.polygon),
            "properties": {
                "frame_id": f.frame_id,
                "track": f.track,
                "frame_index": f.index,
                "beam": f.beam,
                "epsg": f.epsg,
                "bbox": list(f.bbox),
            },
        }
        for f in sorted(frames, key=lambda f: f.frame_id)
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n"
    )


@cli.command()
@click.argument("frame_id")
@_DB_OPTION
def lookup(frame_id: str, database: Path) -> None:
    """Print the pinned grid of one FRAME_ID."""
    parse_frame_id(frame_id)
    found = db_mod.read_frames(database, [frame_id])
    if not found:
        raise click.ClickException(f"{frame_id} is not in {database}")

    f = found[0]
    click.echo(f"frame_id : {f.frame_id}")
    click.echo(f"track    : {f.track}")
    click.echo(f"index    : {f.index}")
    click.echo(f"beam     : {f.beam}")
    click.echo(f"epsg     : {f.epsg}")
    click.echo(f"bbox     : {f.xmin} {f.ymin} {f.xmax} {f.ymax}")
    click.echo(f"size     : {f.xmax - f.xmin} x {f.ymax - f.ymin} m")
    click.echo(f"fill     : {f.fill_pct}% of the box ({f.nodata_pct}% always nodata)")
    click.echo(f"polygon  : {f.polygon.wkt}")


@cli.command("frames-for-granule")
@click.argument("granule_name")
@click.option(
    "--orbit-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of .EOF orbit files.",
)
@click.option(
    "--catalog",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Catalog holding this granule's metadata.",
)
@click.option(
    "--tile-seconds",
    default=DEFAULT_TILE_SECONDS,
    show_default=True,
    help="Frame length [s].",
)
def frames_for_granule_cmd(
    granule_name: str, orbit_dir: Path, catalog: Path, tile_seconds: float
) -> None:
    """Print the frame IDs GRANULE_NAME fully covers, one per line.

    This is the call a planner makes to turn an acquisition into the frame
    products it should yield.
    """
    matches = [g for g in granules_mod.load_catalog(catalog) if g.name == granule_name]
    if not matches:
        raise click.ClickException(f"{granule_name} is not in {catalog}")

    orbit = OrbitLookup(orbit_dir).find(matches[0])
    for frame in frames_for_granule(matches[0], orbit, tile_seconds=tile_seconds):
        click.echo(frame.frame_id)


@cli.command()
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    required=True,
    metavar="W S E N",
    help="Area of interest in degrees.",
)
@_DB_OPTION
def intersect(bbox: tuple[float, float, float, float], database: Path) -> None:
    """Print the frames intersecting an area of interest."""
    from shapely.geometry import box

    aoi = box(*bbox)
    for f in db_mod.read_frames(database):
        if f.polygon.intersects(aoi):
            click.echo(
                f"{f.frame_id}  epsg={f.epsg}  bbox={' '.join(map(str, f.bbox))}"
            )


@cli.command()
@click.option(
    "--orbit-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of .EOF orbit files.",
)
@click.option(
    "--catalog",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Catalog of granules to re-derive frames from.",
)
@click.option(
    "--tolerance",
    default=DEFAULT_TOLERANCE,
    show_default=True,
    help="Largest bbox disagreement to accept [m].",
)
@_DB_OPTION
def check(orbit_dir: Path, catalog: Path, tolerance: float, database: Path) -> None:
    """Re-derive every frame from every granule and report disagreement.

    A frame's grid is pinned by the first acquisition that defined it. Repeat
    passes reproduce it to within the orbital tube -- tens of metres, plus up to
    one snap cell of rounding. The default tolerance sits well above that and
    far below the bbox margin, so it stays quiet for real repeats while catching
    the kilometre-scale divergence that means the database no longer describes
    the data. Exits non-zero when anything is reported.
    """
    stored = {f.frame_id: f for f in db_mod.read_frames(database)}
    orbits = OrbitLookup(orbit_dir)
    problems = []

    for granule in granules_mod.load_catalog(catalog):
        try:
            orbit = orbits.find(granule)
        except FileNotFoundError as exc:
            problems.append(str(exc))
            continue

        for frame in frames_for_granule(granule, orbit):
            known = stored.get(frame.frame_id)
            if known is None:
                problems.append(
                    f"{granule.name}: {frame.frame_id} missing from database"
                )
                continue
            if known.epsg != frame.epsg:
                problems.append(
                    f"{frame.frame_id}: epsg {known.epsg} stored, "
                    f"{frame.epsg} from {granule.name}"
                )
                continue
            worst = max(abs(a - b) for a, b in zip(known.bbox, frame.bbox, strict=True))
            if worst > tolerance:
                problems.append(
                    f"{frame.frame_id}: bbox differs by {worst:.0f} m "
                    f"from {granule.name}"
                )

    for message in problems:
        click.echo(message, err=True)
    click.echo(f"{len(stored)} frames checked, {len(problems)} problem(s)")
    if problems:
        sys.exit(1)


def main() -> None:
    """Entry point for the ``sm-db`` console script."""
    cli()


@cli.command()
@click.option(
    "--catalog",
    type=click.Path(path_type=Path),
    default="catalog/sm_granules.json.gz",
    show_default=True,
    help="Granule catalog to extend in place.",
)
@click.option(
    "--orbit-dir",
    type=click.Path(path_type=Path),
    default="orbits",
    show_default=True,
    help="Where orbit files are cached; missing ones are downloaded.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="catalog/sm_frames.sqlite3",
    show_default=True,
    help="Frame database to rebuild.",
)
@click.option(
    "--geojson",
    type=click.Path(path_type=Path),
    default="catalog/sm_frames.geojson",
    show_default=True,
    help="Frame footprints for a GIS.",
)
@click.option(
    "--viewer",
    type=click.Path(path_type=Path),
    default="docs/frame-viewer.html",
    show_default=True,
    help="Map of the archive.",
)
@click.option(
    "--start",
    help="Query from this date instead of continuing from the catalog's last date.",
)
@click.option("--end", help="Query up to this date. Default: today.")
@click.option(
    "--beam",
    "beams",
    multiple=True,
    type=click.Choice(SM_BEAM_MODES, case_sensitive=False),
    help="Stripmap beams. Repeatable; default all.",
)
@click.option(
    "--track", "tracks", multiple=True, type=int, help="Relative orbit numbers."
)
@click.option("--intersects", help="Area of interest as WKT.")
@click.option(
    "--tile-seconds",
    default=DEFAULT_TILE_SECONDS,
    show_default=True,
    help="Along-track frame length in seconds.",
)
@click.option(
    "--margin", default=DEFAULT_MARGIN, show_default=True, help="Bbox padding [m]."
)
@click.option(
    "--snap", default=DEFAULT_SNAP, show_default=True, help="Bbox lattice [m]."
)
@click.option(
    "--shift-seconds",
    default=0.0,
    show_default=True,
    help="Slide every frame along track; positive is towards the later end.",
)
@click.option(
    "--overlap-seconds",
    default=0.0,
    show_default=True,
    help="Extend every frame at both ends so neighbours overlap.",
)
@click.option(
    "--inset-m",
    default=0.0,
    show_default=True,
    help="Trim this many metres off each side across track; negative widens.",
)
@click.option(
    "--overrides",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON of per-frame geometry, keyed by frame id, as the viewer exports it: "
    '{"t095_000003_s3": {"shift": 1.2, "overlap": 0.5, "inset": 2000}}. '
    "A frame listed here ignores the global --shift/--overlap/--inset.",
)
@click.option(
    "--min-fill",
    default=DEFAULT_MIN_FILL,
    show_default=True,
    help="Drop frames whose footprint covers less than this percent of their box.",
)
@click.option(
    "--no-download",
    is_flag=True,
    help="Do not fetch missing orbits; skip granules that have none.",
)
def update(
    catalog: Path,
    orbit_dir: Path,
    output: Path,
    geojson: Path,
    viewer: Path,
    start: str | None,
    end: str | None,
    beams: tuple[str, ...],
    tracks: tuple[int, ...],
    intersects: str | None,
    tile_seconds: float,
    margin: float,
    snap: float,
    shift_seconds: float,
    overlap_seconds: float,
    inset_m: float,
    overrides: Path | None,
    min_fill: float,
    no_download: bool,
) -> None:
    """Extend the catalog with new acquisitions and rebuild the database and map.

    This is what the scheduled job runs. It queries ASF from the day after the
    catalog's most recent acquisition, so a daily run costs one small query, and
    rebuilds the frame database from the whole catalog afterwards. Rebuilding is
    cheap and keeps the result independent of how the catalog was accumulated;
    grids already defined are preserved because `merge_frames` keeps the first
    definition of every frame.
    """
    from sm_db.viewer import frame_acquisitions, write_viewer

    known = granules_mod.load_catalog(catalog) if catalog.exists() else []
    by_name = {g.name: g for g in known}

    if start is None:
        start = (
            (max(g.start for g in known) + datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            if known
            else "2014-04-03"  # Sentinel-1A launch: the whole archive.
        )
    end = end or datetime.date.today().isoformat()

    per_frame = json.loads(overrides.read_text()) if overrides else {}
    if per_frame:
        click.echo(f"{len(per_frame)} frame(s) carry their own bounds")

    click.echo(f"Catalog holds {len(known)} granules; querying ASF {start} to {end}")
    found = granules_mod.query_asf(
        start=start,
        end=end,
        beam_modes=[b.upper() for b in beams] or SM_BEAM_MODES,
        tracks=tracks or None,
        intersects_wkt=intersects,
    )
    new = [g for g in found if g.name not in by_name]
    click.echo(f"{len(new)} new granule(s)")

    everything = sorted([*known, *new], key=lambda g: g.start)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    granules_mod.save_catalog(everything, catalog)

    if not everything:
        click.echo("Nothing in the catalog yet; stopping before the rebuild.")
        return

    if not no_download:
        from sm_db.orbits import ensure_orbits

        fetched = ensure_orbits(everything, orbit_dir)
        if fetched:
            click.echo(f"Downloaded {len(fetched)} orbit file(s)")

    orbits = OrbitLookup(orbit_dir)
    defined: dict[str, Frame] = {}
    covered: list[tuple] = []
    skipped = 0

    with click.progressbar(everything, label="Defining frames") as bar:
        for granule in bar:
            try:
                orbit = orbits.find(granule)
            except FileNotFoundError:
                skipped += 1
                continue
            found_frames = frames_for_granule(
                granule,
                orbit,
                tile_seconds=tile_seconds,
                margin=margin,
                snap=snap,
                overlap=overlap_seconds,
                shift=shift_seconds,
                inset=inset_m,
                overrides=per_frame,
            )
            found_frames = [f for f in found_frames if f.fill_pct >= min_fill]
            covered.append((granule, [f.frame_id for f in found_frames]))
            defined = merge_frames(defined, found_frames)

    if skipped:
        click.echo(f"  skipped {skipped} granule(s) with no orbit available", err=True)

    n = db_mod.write_database(
        defined.values(),
        output,
        tile_seconds=tile_seconds,
        margin=margin,
        snap=snap,
        extra={
            "start": start,
            "end": end,
            "n_granules": len(everything),
            "shift_seconds": shift_seconds,
            "overlap_seconds": overlap_seconds,
            "inset_m": inset_m,
        },
    )
    click.echo(f"Wrote {n} frames to {output}")

    _write_geojson(list(defined.values()), geojson)
    click.echo(f"Wrote {geojson}")

    write_viewer(
        defined.values(),
        viewer,
        acquisitions=frame_acquisitions(covered),
        subtitle=f"{n} frames from {len(everything)} acquisitions",
        tile_seconds=tile_seconds,
    )
    click.echo(f"Wrote {viewer}")


@cli.command("viewer")
@click.option(
    "--catalog",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Catalog used to count acquisitions per frame.",
)
@click.option(
    "--orbit-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Orbit files, needed only when counting from a catalog.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="frame-viewer.html",
    show_default=True,
    help="HTML file to write.",
)
@click.option(
    "--aoi",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="GeoJSON of areas of interest to ship with the page. Omit for a public "
    "copy; the page can load one locally instead.",
)
@_DB_OPTION
def viewer_cmd(
    catalog: Path | None,
    orbit_dir: Path | None,
    output: Path,
    aoi: Path | None,
    database: Path,
) -> None:
    """Render the frame database as a standalone map."""
    from sm_db.viewer import frame_acquisitions, write_viewer

    frames = db_mod.read_frames(database)
    acquisitions = None

    if catalog:
        if orbit_dir is None:
            raise click.ClickException(
                "--catalog also needs --orbit-dir to date the acquisitions"
            )
        orbits = OrbitLookup(orbit_dir)
        covered = []
        for granule in granules_mod.load_catalog(catalog):
            try:
                orbit = orbits.find(granule)
            except FileNotFoundError:
                continue
            covered.append(
                (granule, [f.frame_id for f in frames_for_granule(granule, orbit)])
            )
        acquisitions = frame_acquisitions(covered)

    import json as _json
    import sqlite3 as _sqlite3

    with _sqlite3.connect(database) as con:
        row = con.execute(
            "SELECT value FROM metadata WHERE key = 'tile_seconds'"
        ).fetchone()
    tile_seconds = _json.loads(row[0]) if row else DEFAULT_TILE_SECONDS

    write_viewer(frames, output, acquisitions=acquisitions, tile_seconds=tile_seconds)
    click.echo(f"Wrote {output}")
