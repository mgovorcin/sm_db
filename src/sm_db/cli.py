"""Command line interface: ``sm-db``."""

from __future__ import annotations

import datetime
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import click

from sm_db import db as db_mod
from sm_db import granules as granules_mod
from sm_db.coverage import DEFAULT_GRID
from sm_db.frames import Frame, OrbitLookup, frames_for_granule, merge_frames
from sm_db.geometry import DEFAULT_MARGIN, DEFAULT_SNAP
from sm_db.granules import SM_BEAM_MODES
from sm_db.remote import ASSETS, fetch_asset
from sm_db.tiling import DEFAULT_TILE_SECONDS, parse_frame_id

DEFAULT_TOLERANCE = 0.0
"""Largest bbox disagreement ``check`` accepts, or 0 to derive it from the margin.

A frame's cross-track extent is measured from whichever acquisition defined it
first, and ASF footprints differ between granules by more than the orbital tube
does, so re-deriving a frame from a later granule legitimately lands somewhere
slightly different. Across the 2014-2026 archive that spread is a median of 390 m
and a p90 of 660 m.

None of that matters while the pinned box still contains what an acquisition
would produce, which is what the 5 km margin is for. So the failure threshold is
the margin itself rather than a number picked by hand: beyond it, containment is
no longer guaranteed. Half the margin is reported as worth a look.
"""

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
    "--merge",
    "merge_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON list of frame-id groups to emit as single frames, as the viewer "
    'exports it: [["t095_000003_s3", "t095_000004_s3"]].',
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
    merge_file: Path | None,
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

    per_frame, groups = _load_adjustments(overrides, merge_file)
    dropped = _load_drops(overrides, merge_file)
    if per_frame:
        click.echo(f"{len(per_frame)} frame(s) carry their own bounds")
    if groups:
        click.echo(f"{len(groups)} group(s) merged into single frames")
    if dropped:
        click.echo(f"{len(dropped)} frame(s) dropped")

    orbits = OrbitLookup(orbit_dir)
    defined: dict[str, Frame] = {}
    covered: list[tuple] = []
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
                merges=groups,
                drops=dropped,
            )
            if not found:
                skipped.append(f"{granule.name}: too short to fill a frame")
            found = [f for f in found if f.fill_pct >= min_fill]
            covered.append((granule, [f.frame_id for f in found]))
            defined = merge_frames(defined, found)

    for message in skipped:
        click.echo(f"  skipped {message}", err=True)

    from sm_db.viewer import frame_acquisitions

    n = db_mod.write_database(
        defined.values(),
        output,
        tile_seconds=tile_seconds,
        margin=margin,
        snap=snap,
        acquisitions=frame_acquisitions(covered),
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


def _build_parameters(database: Path) -> tuple[float, float]:
    """Return the margin and snap a database was built with.

    An edited frame has to be pinned the way every other frame was, so the
    padding and lattice come from the database rather than from defaults.

    Parameters
    ----------
    database :
        Database file.

    Returns
    -------
    tuple
        ``(margin, snap)`` in metres.
    """
    import sqlite3

    with sqlite3.connect(database) as con:
        rows = dict(
            con.execute(
                "SELECT key, value FROM metadata WHERE key IN ('margin', 'snap')"
            ).fetchall()
        )
    return (
        float(json.loads(rows.get("margin", str(DEFAULT_MARGIN)))),
        float(json.loads(rows.get("snap", str(DEFAULT_SNAP)))),
    )


def _stale_frames(
    existing: dict[str, Frame],
    per_frame: dict[str, dict],
    groups: list[list[str]],
    dropped: set[str],
) -> set[str]:
    """Return the frames whose stored definition no longer matches the adjustments.

    The scheduled job passes the same adjustments file on every run, so "every
    frame an adjustment names" would re-derive the same frames each week -- and
    re-deriving needs every orbit those frames were ever built from, which a CI
    cache does not hold. Only a frame whose recorded shift, overlap or inset
    differs from what is now asked, or that should be gone but is still present,
    needs the work. The database stores each frame's adjustment for exactly this.

    Parameters
    ----------
    existing :
        Frames already in the database.
    per_frame :
        Requested geometry adjustments, keyed by frame ID.
    groups :
        Requested merges.
    dropped :
        Frames that should not exist.

    Returns
    -------
    set of str
    """
    stale = {f for f in dropped if f in existing}
    for frame_id, wanted in per_frame.items():
        frame = existing.get(frame_id)
        if frame is None:
            continue
        have = {"shift": frame.shift, "overlap": frame.overlap, "inset": frame.inset}
        if any(
            abs(float(wanted.get(key, 0.0)) - value) > 1e-3
            for key, value in have.items()
        ):
            stale.add(frame_id)
    # A merge is applied once its later members have been absorbed; while any of
    # them still stands as its own frame, the group has not been built yet.
    for group in groups:
        if any(member in existing for member in sorted(group)[1:]):
            stale.update(m for m in group if m in existing)
    return stale


def _load_drops(*paths: Path | None) -> set[str]:
    """Read the frames an adjustments file removes.

    Kept apart from `_load_adjustments` so that function's return shape, which
    callers and tests rely on, does not change.

    Parameters
    ----------
    paths :
        Adjustment files given on the command line; any may be `None`.

    Returns
    -------
    set of str
        Frame IDs never to emit.
    """
    dropped: set[str] = set()
    for path in paths:
        if path is None:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            dropped.update(data.get("drops", []))
    return dropped


def _load_adjustments(
    overrides: Path | None, merge_file: Path | None
) -> tuple[dict, list]:
    """Read per-frame bounds and merge groups, from one file or two.

    The viewer exports both in a single ``{"overrides": ..., "merges": ...}``
    document, so either flag accepts that shape; a bare mapping is still read as
    overrides alone and a bare list as merges alone.

    Parameters
    ----------
    overrides, merge_file :
        Paths given on the command line; either may be `None`.

    Returns
    -------
    tuple
        ``(overrides, merges)``.
    """
    per_frame: dict = {}
    groups: list = []
    for path in (overrides, merge_file):
        if path is None:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list):
            groups = data
        elif "overrides" in data or "merges" in data:
            per_frame = {**per_frame, **data.get("overrides", {})}
            groups = data.get("merges", groups)
        else:
            per_frame = {**per_frame, **data}
    return per_frame, groups


def _write_geojson(frames: Iterable[Frame], path: Path) -> None:
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
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=".",
    show_default=True,
    help="Directory to keep the downloaded files in.",
)
@click.option(
    "--asset",
    "assets",
    type=click.Choice(ASSETS),
    multiple=True,
    help="Asset to fetch (repeatable). Default: the frame database only.",
)
@click.option("--force", is_flag=True, help="Download even if unchanged.")
def fetch(output_dir: Path, assets: tuple[str, ...], force: bool) -> None:
    """Download the published archive instead of building it.

    The weekly workflow publishes the frame database, the granule catalog and
    the frames GeoJSON as release assets. A repeat fetch is one conditional
    request and downloads nothing unless the archive changed.
    """
    for name in assets or ("sm_frames.sqlite3",):
        result = fetch_asset(output_dir, name, force=force)
        state = "downloaded" if result.updated else "already current"
        click.echo(f"{result.path}: {state} ({result.last_modified or 'unknown date'})")


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
@click.option(
    "--overrides",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The adjustments the database was built with. Without them an adjusted "
    "frame is re-derived at its default size and reported as a disagreement.",
)
@_DB_OPTION
def check(
    orbit_dir: Path,
    catalog: Path,
    tolerance: float,
    overrides: Path | None,
    database: Path,
) -> None:
    """Re-derive every frame from every granule and report disagreement.

    A frame's grid is pinned by the first acquisition that defined it. Repeat
    passes reproduce it to within the orbital tube -- tens of metres, plus up to
    one snap cell of rounding. The default tolerance sits well above that and
    far below the bbox margin, so it stays quiet for real repeats while catching
    the kilometre-scale divergence that means the database no longer describes
    the data. Exits non-zero only on a real disagreement; granules whose orbit is
    not available are counted as unverified, not as failures.
    """
    stored = {f.frame_id: f for f in db_mod.read_frames(database)}
    per_frame, groups = _load_adjustments(overrides, None)
    dropped = _load_drops(overrides)
    margin, _ = _build_parameters(database)
    limit = tolerance or margin
    orbits = OrbitLookup(orbit_dir)
    problems = []
    spread: list[float] = []
    verified = 0
    unverifiable = 0

    for granule in granules_mod.load_catalog(catalog):
        # A granule whose orbit is not on hand cannot be checked, which is not
        # the same as a granule that disagrees. Counting the two together made a
        # CI run -- which caches only recent orbits, deliberately -- report the
        # whole archive as broken.
        if not orbits.covers(granule):
            unverifiable += 1
            continue
        orbit = orbits.find(granule)
        verified += 1

        for frame in frames_for_granule(
            granule, orbit, overrides=per_frame, merges=groups, drops=dropped
        ):
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
            spread.append(worst)
            if worst > limit:
                problems.append(
                    f"{frame.frame_id}: bbox differs by {worst:.0f} m "
                    f"from {granule.name}, past the {limit:.0f} m margin"
                )

    for message in problems:
        click.echo(message, err=True)

    click.echo(
        f"{len(stored)} frames in the database; "
        f"{verified} granule(s) verified, {len(problems)} problem(s)"
    )
    if spread:
        # Always shown, so a slow drift is visible long before it fails anything.
        ordered = sorted(spread)
        click.echo(
            "bbox spread vs the stored grid: "
            f"median {ordered[len(ordered) // 2]:.0f} m, "
            f"p90 {ordered[int(len(ordered) * 0.9)]:.0f} m, max {ordered[-1]:.0f} m "
            f"(fails past {limit:.0f} m)"
        )
        near = sum(1 for v in spread if limit / 2 < v <= limit)
        if near:
            click.echo(
                f"{near} re-derivation(s) past half the margin: not a failure, but "
                "the padding is doing more work than it should."
            )
    if unverifiable:
        click.echo(
            f"{unverifiable} granule(s) could not be checked: no orbit on hand. "
            "That is expected where orbits are fetched only for new acquisitions."
        )
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
    default="docs/index.html",
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
    "--merge",
    "merge_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON list of frame-id groups to emit as single frames, as the viewer "
    'exports it: [["t095_000003_s3", "t095_000004_s3"]].',
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
@click.option(
    "--rebuild",
    is_flag=True,
    help="Re-derive every frame instead of only the new acquisitions. Needed after "
    "a change to the tiling or the geometry; otherwise slow and unnecessary, since "
    "a frame's grid is frozen once defined.",
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
    merge_file: Path | None,
    min_fill: float,
    no_download: bool,
    rebuild: bool,
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

    per_frame, groups = _load_adjustments(overrides, merge_file)
    dropped = _load_drops(overrides, merge_file)
    if per_frame:
        click.echo(f"{len(per_frame)} frame(s) carry their own bounds")
    if groups:
        click.echo(f"{len(groups)} group(s) merged into single frames")
    if dropped:
        click.echo(f"{len(dropped)} frame(s) dropped")

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

    # A frame's grid is frozen once defined, so re-deriving the whole archive every
    # run is wasted work: it would need every orbit file ever used -- 24 GB for a
    # twelve-year archive, more than a CI cache can hold -- to arrive back at the
    # same frames. Only the acquisitions this run has not seen are processed.
    existing = (
        {f.frame_id: f for f in db_mod.read_frames(output)} if output.exists() else {}
    )
    known_acq = db_mod.read_acquisitions(output) if output.exists() else {}
    seen = {a["granule"] for entries in known_acq.values() for a in entries}

    # An adjustment changes a frame that is already defined, and an incremental
    # run would otherwise keep the old definition forever -- a grid is frozen once
    # made. So every frame an adjustment names is forgotten and re-derived from
    # the acquisitions that produced it, and nothing else is touched.
    touched = _stale_frames(existing, per_frame, groups, dropped)
    # A dropped frame only has to go: its granules were recorded against the
    # frames beside it too, and those records are still right. Re-deriving would
    # fetch an orbit for every pass it ever had, for nothing.
    gone = touched & dropped
    if gone and not rebuild:
        for f in gone:
            existing.pop(f, None)
            known_acq.pop(f, None)
        touched -= gone
        click.echo(f"{len(gone)} dropped frame(s) removed")
    if touched and not rebuild:
        redo = {a["granule"] for f in touched for a in known_acq.get(f, [])}
        for f in touched:
            existing.pop(f, None)
            known_acq.pop(f, None)
        seen -= redo
        click.echo(
            f"{len(touched)} adjusted frame(s) to re-derive "
            f"from {len(redo)} acquisition(s)"
        )

    pending = everything if rebuild else [g for g in everything if g.name not in seen]
    if existing and not rebuild:
        click.echo(
            f"{len(existing)} frames already defined; "
            f"{len(pending)} acquisition(s) to add"
        )
    if rebuild:
        existing, known_acq = {}, {}

    if not no_download and pending:
        from sm_db.orbits import ensure_orbits

        fetched = ensure_orbits(pending, orbit_dir)
        if fetched:
            click.echo(f"Downloaded {len(fetched)} orbit file(s)")

    orbits = OrbitLookup(orbit_dir)
    defined: dict[str, Frame] = dict(existing)
    covered: list[tuple] = []
    skipped = 0

    with click.progressbar(pending, label="Defining frames") as bar:
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
                merges=groups,
                drops=dropped,
            )
            found_frames = [f for f in found_frames if f.fill_pct >= min_fill]
            covered.append((granule, [f.frame_id for f in found_frames]))
            defined = merge_frames(defined, found_frames)

    if skipped:
        click.echo(f"  skipped {skipped} granule(s) with no orbit available", err=True)

    # Fold this run's acquisitions into what the database already recorded.
    # Re-deriving an adjusted frame reprocesses its granules, which also yield
    # that frame's neighbours -- already recorded -- so an acquisition is only
    # added to a frame that does not have it yet.
    observed = dict(known_acq)
    for frame_id, entries in frame_acquisitions(covered).items():
        have = {a["granule"] for a in observed.get(frame_id, [])}
        fresh = [a for a in entries if a["granule"] not in have]
        merged = observed.get(frame_id, []) + fresh
        observed[frame_id] = sorted(merged, key=lambda a: (a["date"], a["granule"]))

    n = db_mod.write_database(
        defined.values(),
        output,
        tile_seconds=tile_seconds,
        margin=margin,
        snap=snap,
        acquisitions=observed,
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
        acquisitions=observed,
        subtitle=f"{n} frames from {len(everything)} acquisitions",
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
    # The database carries what was observed, so redrawing the map is a read
    # rather than half an hour of re-deriving frames from every granule.
    acquisitions = db_mod.read_acquisitions(database) or None
    if acquisitions:
        click.echo(f"Read acquisitions for {len(acquisitions)} frames from {database}")

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

    write_viewer(frames, output, acquisitions=acquisitions)
    click.echo(f"Wrote {output}")


@cli.command()
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="sm_frames.shp",
    show_default=True,
    help="File to write. The extension picks the format: .shp, .gpkg, .geojson.",
)
@click.option(
    "--grids",
    is_flag=True,
    help="Write the pinned bounding boxes instead of the frame footprints.",
)
@_DB_OPTION
def export(output: Path, grids: bool, database: Path) -> None:
    """Write frames to a GIS file, for editing and handing back.

    Shapefile attribute names are capped at ten characters, so they are
    abbreviated on the way out and understood again by ``import-shapes``.
    Prefer GeoPackage when the choice is yours: it keeps full names and one file.
    """
    import geopandas as gpd

    from sm_db.viewer import grids_to_geojson

    frames = db_mod.read_frames(database)
    if not frames:
        raise click.ClickException(f"{database} holds no frames")

    if grids:
        shapes = [
            __import__("shapely.geometry", fromlist=["shape"]).shape(f["geometry"])
            for f in grids_to_geojson(frames)["features"]
        ]
    else:
        shapes = [f.polygon for f in frames]

    frame = gpd.GeoDataFrame(
        {
            "frame_id": [f.frame_id for f in frames],
            "track": [f.track for f in frames],
            "frame_idx": [f.index for f in frames],
            "beam": [f.beam for f in frames],
            "epsg": [f.epsg for f in frames],
            "xmin": [f.xmin for f in frames],
            "ymin": [f.ymin for f in frames],
            "xmax": [f.xmax for f in frames],
            "ymax": [f.ymax for f in frames],
            "fill_pct": [f.fill_pct for f in frames],
            "shift_s": [f.shift for f in frames],
            "overlap_s": [f.overlap for f in frames],
            "inset_m": [f.inset for f in frames],
        },
        geometry=shapes,
        crs="EPSG:4326",
    )

    # Measured coverage, when `sm-db coverage --store` has been run. Shapefile
    # field names are capped at ten characters, hence the abbreviations.
    measured = db_mod.read_coverage(database)
    if measured:
        frame["n_acq"] = [
            measured.get(f.frame_id, {}).get("n_measured") for f in frames
        ]
        frame["common"] = [measured.get(f.frame_id, {}).get("common") for f in frames]
        frame["typical"] = [measured.get(f.frame_id, {}).get("typical") for f in frames]
        frame["worst_acq"] = [measured.get(f.frame_id, {}).get("worst") for f in frames]
        click.echo(f"  including measured coverage for {len(measured)} frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(output)
    click.echo(f"Wrote {len(frame)} {'grids' if grids else 'frames'} to {output}")


@cli.command("import-shapes")
@click.argument("shapes", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--catalog",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Catalog holding the granules, to read each edit against a real pass.",
)
@click.option(
    "--orbit-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Orbit files, to measure each edit along the track.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default="sm_frame_adjustments.json",
    show_default=True,
    help="Adjustments file to write, for --overrides on the next build.",
)
@_DB_OPTION
def import_shapes(
    shapes: Path, catalog: Path, orbit_dir: Path, output: Path, database: Path
) -> None:
    """Read frames edited in a GIS back into adjustments a build reproduces.

    An edited frame keeps only what an edit can mean for a stripmap swath: where
    along the track it starts and stops. Its east and west sides go back onto the
    swath edges, so a side dragged off by hand is corrected rather than kept.
    A frame missing from the file is dropped.
    """
    import geopandas as gpd

    from sm_db.adjust import along_track_window, window_to_offsets

    edited = gpd.read_file(shapes)
    if "frame_id" not in edited.columns:
        raise click.ClickException(
            f"{shapes} has no frame_id column; export it with `sm-db export` first"
        )
    if edited.crs is not None and edited.crs.to_epsg() != 4326:
        edited = edited.to_crs(4326)

    known = {f.frame_id: f for f in db_mod.read_frames(database)}
    observed = db_mod.read_acquisitions(database)
    granules = {g.name: g for g in granules_mod.load_catalog(catalog)}
    orbits = OrbitLookup(orbit_dir)
    tile_seconds = _tile_seconds(database)

    present = set(edited["frame_id"].dropna())
    dropped = sorted(set(known) - present)
    unknown = sorted(present - set(known))

    overrides: dict[str, dict] = {}
    windows: dict[str, list[float]] = {}
    unmeasured: list[str] = []
    straightened = 0

    for _, row in edited.iterrows():
        frame = known.get(row["frame_id"])
        if frame is None or row.geometry is None:
            continue
        # Shapefiles round coordinates on the way out, hence the tolerance.
        if row.geometry.equals_exact(frame.polygon, 1e-6):
            continue
        granule = next(
            (
                granules[a["granule"]]
                for a in observed.get(frame.frame_id, [])
                if a["granule"] in granules and orbits.covers(granules[a["granule"]])
            ),
            None,
        )
        if granule is None:
            unmeasured.append(frame.frame_id)
            continue
        window = along_track_window(
            row.geometry,
            granule,
            orbits.find(granule),
            frame.epsg,
            frame.index,
            tile_seconds,
        )
        shift, overlap = window_to_offsets(frame.index, window, tile_seconds)
        overrides[frame.frame_id] = {"shift": shift, "overlap": overlap}
        windows[frame.frame_id] = [round(window.start, 3), round(window.stop, 3)]
        if max(abs(window.west_deviation), abs(window.east_deviation)) > 50:
            straightened += 1

    if unknown:
        click.echo(
            f"  ignored {len(unknown)} shape(s) with an unknown frame_id", err=True
        )
    for frame_id in unmeasured:
        click.echo(f"  could not measure {frame_id}: no orbit on hand", err=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "overrides": overrides,
                "merges": [],
                "drops": dropped,
                # Readable record of what each edit means, in seconds since the
                # ascending node. Not read back; `overrides` is the source of truth.
                "windows": windows,
            },
            indent=2,
        )
        + "\n"
    )
    click.echo(
        f"{len(overrides)} frame(s) reframed along track, "
        f"{straightened} with sides pulled back onto the swath; "
        f"{len(dropped)} dropped. Wrote {output}"
    )


def _tile_seconds(database: Path) -> float:
    """Return the tile length a database was built with."""
    import sqlite3

    with sqlite3.connect(database) as con:
        row = con.execute(
            "SELECT value FROM metadata WHERE key = 'tile_seconds'"
        ).fetchone()
    return float(json.loads(row[0])) if row else DEFAULT_TILE_SECONDS


@cli.command()
@click.option(
    "--catalog",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Catalog holding the granule footprints.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Write the full per-frame report here as JSON.",
)
@click.option(
    "--min-passes",
    default=5,
    show_default=True,
    help="Only measure frames with at least this many acquisitions.",
)
@click.option(
    "--poor-below",
    default=0.9,
    show_default=True,
    help="Call an acquisition poor when it covers less than this of its frame.",
)
@click.option(
    "--grid",
    default=DEFAULT_GRID,
    show_default=True,
    help="Samples per side across a frame; higher is finer and slower.",
)
@click.option(
    "--store",
    is_flag=True,
    help="Write the result into the database so the map and the GIS export can "
    "show it without measuring again.",
)
@_DB_OPTION
def coverage(
    catalog: Path,
    output: Path | None,
    min_passes: int,
    poor_below: float,
    grid: int,
    store: bool,
    database: Path,
) -> None:
    """Measure the area every acquisition of a frame actually shares.

    A stack can only use ground that every date covers, so the number that
    matters is the intersection, not the frame. Where a few narrow passes drag it
    down, this shows what dropping them would buy.
    """
    from shapely.geometry import shape

    from sm_db.coverage import frame_coverage

    frames = {f.frame_id: f for f in db_mod.read_frames(database)}
    observed = db_mod.read_acquisitions(database)
    if not observed:
        raise click.ClickException(
            f"{database} has no acquisitions table; rebuild it with `sm-db update`"
        )
    footprints = {g.name: shape(g.geometry) for g in granules_mod.load_catalog(catalog)}

    targets = [(fid, acq) for fid, acq in observed.items() if len(acq) >= min_passes]
    click.echo(f"Measuring {len(targets)} frames with >= {min_passes} acquisitions")

    report: dict[str, dict] = {}
    with click.progressbar(targets, label="Measuring coverage") as bar:
        for frame_id, entries in bar:
            frame = frames.get(frame_id)
            if frame is None:
                continue
            names, shapes = [], []
            for a in entries:
                if a["granule"] in footprints:
                    names.append(a["granule"])
                    shapes.append(footprints[a["granule"]])
            if len(shapes) < min_passes:
                continue
            measured = frame_coverage(
                frame.polygon, shapes, names, frame_id=frame_id, grid=grid
            )
            report[frame_id] = {
                "n_acquisitions": measured.n_acquisitions,
                "common": measured.common,
                "typical": measured.typical,
                "worst": measured.worst,
                "cost_of_worst": measured.cost_of_worst,
                "poor": measured.below(poor_below),
            }

    _summarise_coverage(report, poor_below)
    if output:
        output.write_text(json.dumps(report, indent=2) + "\n")
        click.echo(f"Wrote {output}")
    if store:
        stored = db_mod.write_coverage(database, report)
        click.echo(f"Stored coverage for {stored} frames in {database}")


def _summarise_coverage(report: dict, poor_below: float) -> None:
    """Print the shape of a coverage report: how much is lost, and to how few."""
    if not report:
        click.echo("No frames measured.")
        return

    commons = sorted(r["common"] for r in report.values())
    healthy = [f for f, r in report.items() if r["common"] >= poor_below]
    hurt = {f: r for f, r in report.items() if r["common"] < poor_below}

    def pct(x):
        return f"{100 * x:.1f}%"

    click.echo("")
    click.echo(f"frames measured      : {len(report)}")
    click.echo(f"median common area   : {pct(commons[len(commons) // 2])}")
    click.echo(f"worst common area    : {pct(commons[0])}")
    click.echo(f"frames >= {pct(poor_below)} common : {len(healthy)}")
    click.echo(f"frames below         : {len(hurt)}")

    if not hurt:
        return

    # The question is whether the loss is the frame's or a few dates'. Where the
    # typical date still fills the frame, a handful of narrow passes are holding
    # the whole stack back and dropping them buys the area back. Where even the
    # typical date falls short, the frame reaches past what one acquisition
    # covers -- a frame widened beyond a single slice -- and no amount of
    # dropping helps: it needs consecutive slices processed together.
    few_dates = {f: r for f, r in hurt.items() if r["typical"] >= poor_below}
    too_long = {f: r for f, r in hurt.items() if r["typical"] < poor_below}
    click.echo("")
    click.echo(
        f"  a few narrow dates hold the stack back : {len(few_dates):4d} frames "
        "(dropping them recovers the area)"
    )
    click.echo(
        f"  wider than one acquisition covers      : {len(too_long):4d} frames "
        "(needs consecutive slices together)"
    )
