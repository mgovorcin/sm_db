"""Render the frame database as a standalone map.

A stripmap campaign is tasked by hand, so the first questions about the archive
are spatial and temporal at once: where has anything been acquired, on which
track and beam, and *how often* -- which is what decides whether a frame can
carry a time series at all. A table answers that badly.

The output is one self-contained HTML file with the frames, their pinned grids
and their acquisition dates inlined as GeoJSON, so it can be committed, opened
from disk, or published without a server behind it. MapLibre GL comes from a CDN
and every tile source is keyless.

Two geometries per frame, and the difference matters:

* the **footprint** is the tile on the ground -- a rotated quadrilateral that
  follows the orbit, which is what a frame *is*;
* the **pinned grid** is the axis-aligned bounding box in the frame's own UTM
  zone, padded and snapped, which is what COMPASS actually geocodes into. Drawn
  north-up and always larger, it is the extent a product covers.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import box, mapping
from shapely.ops import transform

from sm_db.frames import Frame

__all__ = [
    "Acquisition",
    "acquisition_counts",
    "frame_acquisitions",
    "frames_to_geojson",
    "grids_to_geojson",
    "write_viewer",
]


class Acquisition(dict):
    """One acquisition of a frame: its date, granule and platform.

    A plain `dict` so it serializes straight into the page.
    """

    def __init__(self, date: str, granule: str, platform: str) -> None:
        super().__init__(date=date, granule=granule, platform=platform)


def frame_acquisitions(pairs: Iterable[tuple]) -> dict[str, list[dict]]:
    """Group acquisitions by the frames they cover.

    Parameters
    ----------
    pairs :
        ``(granule, frame_ids)`` for each acquisition.

    Returns
    -------
    dict
        Frame ID to its acquisitions, oldest first.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for granule, frame_ids in pairs:
        for frame_id in frame_ids:
            grouped[frame_id].append(
                Acquisition(
                    date=granule.start.strftime("%Y-%m-%d"),
                    granule=granule.name,
                    platform=granule.name[:3].upper(),
                )
            )
    return {k: sorted(v, key=lambda a: a["date"]) for k, v in grouped.items()}


def acquisition_counts(
    frame_ids_per_granule: Iterable[Iterable[str]],
) -> dict[str, int]:
    """Count how many acquisitions cover each frame.

    Parameters
    ----------
    frame_ids_per_granule :
        One iterable of frame IDs per granule.

    Returns
    -------
    dict
        Frame ID to acquisition count.
    """
    counts: dict[str, int] = defaultdict(int)
    for ids in frame_ids_per_granule:
        for frame_id in ids:
            counts[frame_id] += 1
    return dict(counts)


def _repeat_days(dates: list[str]) -> float | None:
    """Median gap between consecutive acquisition dates, in days.

    Returns
    -------
    float or None
        `None` when there is only one acquisition, so no interval exists.
    """
    if len(dates) < 2:
        return None
    stamps = [datetime.strptime(d, "%Y-%m-%d") for d in sorted(set(dates))]
    if len(stamps) < 2:
        return None
    gaps = [(b - a).days for a, b in zip(stamps, stamps[1:], strict=False)]
    return round(statistics.median(gaps), 1)


def duplicate_dates(acquisitions: list[dict]) -> list[str]:
    """Return the dates a frame was covered by more than one granule.

    Two granules covering one frame on one day is not a repeat pass -- it is the
    same pass reaching the frame twice, usually because consecutive slices of a
    datatake overlap. Left alone it would enter a stack as two entries with a
    zero temporal baseline, so it has to be visible before the stack is built.

    Parameters
    ----------
    acquisitions :
        A frame's acquisitions.

    Returns
    -------
    list of str
        The repeated dates, in order.
    """
    seen: dict[str, int] = defaultdict(int)
    for a in acquisitions:
        seen[a["date"]] += 1
    return sorted(d for d, n in seen.items() if n > 1)


def _properties(frame: Frame, acquisitions: list[dict]) -> dict:
    """Assemble the map properties for one frame."""
    # Sort here rather than trusting the caller: the page reads these as a
    # chronology, taking the ends as the first and last pass and the gaps between
    # consecutive entries as repeat intervals.
    ordered = sorted(acquisitions, key=lambda a: (a["date"], a["granule"]))
    dates = [a["date"] for a in ordered]
    repeated = duplicate_dates(ordered)
    return {
        "frame_id": frame.frame_id,
        "track": frame.track,
        "frame_index": frame.index,
        "beam": frame.beam.upper(),
        "epsg": frame.epsg,
        "bbox": list(frame.bbox),
        "width_m": frame.xmax - frame.xmin,
        "height_m": frame.ymax - frame.ymin,
        "fill_pct": frame.fill_pct,
        "nodata_pct": frame.nodata_pct,
        "n_acquisitions": len(ordered),
        "dates": dates,
        "sensors": [a["platform"] for a in ordered],
        "granules": [a["granule"] for a in ordered],
        "platforms": sorted({a["platform"] for a in ordered}),
        "first": dates[0] if dates else None,
        "last": dates[-1] if dates else None,
        # Distinct dates is what a time series can actually use.
        "n_dates": len(set(dates)),
        "n_duplicate": len(dates) - len(set(dates)),
        "duplicate_dates": repeated,
        "repeat_days": _repeat_days(dates),
    }


def frames_to_geojson(
    frames: Iterable[Frame], acquisitions: dict[str, list[dict]] | None = None
) -> dict:
    """Build a GeoJSON FeatureCollection of frame footprints.

    Parameters
    ----------
    frames :
        Frames to include.
    acquisitions :
        Acquisitions per frame ID, from `frame_acquisitions`. Frames with none
        are still included, with a zero count.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection in lon/lat degrees.
    """
    acquisitions = acquisitions or {}
    features = [
        {
            "type": "Feature",
            "geometry": mapping(f.polygon),
            "properties": _properties(f, acquisitions.get(f.frame_id, [])),
        }
        for f in sorted(frames, key=lambda f: f.frame_id)
    ]
    return {"type": "FeatureCollection", "features": features}


def grids_to_geojson(frames: Iterable[Frame]) -> dict:
    """Build a FeatureCollection of the pinned bounding boxes, in lon/lat.

    The box is axis-aligned in the frame's own projection, so reprojecting it to
    degrees gives the north-up rectangle a product actually covers -- visibly
    larger than the footprint, because it is the envelope of a rotated
    quadrilateral plus the margin.

    Parameters
    ----------
    frames :
        Frames whose grids to draw.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection in lon/lat degrees.
    """
    from pyproj import Transformer

    def to_lonlat(polygon, epsg):
        tf = Transformer.from_crs(epsg, 4326, always_xy=True)
        return transform(lambda x, y: tf.transform(x, y), polygon)

    features = []
    for frame in sorted(frames, key=lambda f: f.frame_id):
        corners = to_lonlat(
            box(frame.xmin, frame.ymin, frame.xmax, frame.ymax), frame.epsg
        )
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(corners),
                "properties": {"frame_id": frame.frame_id, "epsg": frame.epsg},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_viewer(
    frames: Iterable[Frame],
    path: str | Path,
    acquisitions: dict[str, list[dict]] | None = None,
    title: str = "Sentinel-1 Stripmap frames",
    subtitle: str = "",
    tile_seconds: float = 5.0,
    aoi: dict | None = None,
) -> Path:
    """Write the frame map to a self-contained HTML file.

    Parameters
    ----------
    frames :
        Frames to plot.
    path :
        Destination ``.html`` file.
    acquisitions :
        Acquisitions per frame ID, from `frame_acquisitions`.
    title :
        Page heading.
    subtitle :
        Line under the heading.
    tile_seconds :
        Frame length the database was built with. The page needs it to convert
        the bounds sliders, which are in kilometres, into the seconds the CLI
        flags take.
    aoi :
        GeoJSON FeatureCollection of areas of interest to ship with the page.
        `None` writes an empty one, which is what a published copy should carry;
        the page's file picker loads a local one at runtime.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    frames = list(frames)
    footprints = frames_to_geojson(frames, acquisitions)
    grids = grids_to_geojson(frames)

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle = subtitle or f"{len(frames)} frames"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _TEMPLATE.replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__BUILT__", built)
        .replace("__TILE_SECONDS__", str(tile_seconds))
        .replace(
            "__AOI__", json.dumps(aoi or {"type": "FeatureCollection", "features": []})
        )
        .replace("__FRAMES__", json.dumps(footprints))
        .replace("__GRIDS__", json.dumps(grids))
    )
    return path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link href="https://unpkg.com/maplibre-gl@5.1.0/dist/maplibre-gl.css" rel="stylesheet">
<script src="https://unpkg.com/maplibre-gl@5.1.0/dist/maplibre-gl.js"></script>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3f0;
  --surface-1: #fcfcfb;
  --surface-2: #eceae5;
  --border:    #dcd9d2;
  --text-1:    #0b0b0b;
  --text-2:    #52514e;
  --text-3:    #86847d;
  --accent:    #2a78d6;
  --grid-line: #eb6834;
  --seq-1: #86b6ef; --seq-2: #5598e7; --seq-3: #2a78d6; --seq-4: #1c5cab; --seq-5: #104281;
  --asc: #2a78d6; --desc: #eb6834;
  /* Sensors. These four hues were picked by enumerating every 4-subset of the
     categorical theme and keeping only those clearing the all-pairs CVD and
     normal-vision floors in BOTH modes; two subsets qualified. */
  --s1a: #2a78d6; --s1b: #eda100; --s1c: #e87ba4; --s1d: #008300;
  --dup: #e34948;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --surface-2: #24241f;
  --border:    #3a3a35;
  --text-1:    #ffffff;
  --text-2:    #c3c2b7;
  --text-3:    #8c8b82;
  --accent:    #3987e5;
  --grid-line: #d95926;
  --seq-1: #9ec5f4; --seq-2: #6da7ec; --seq-3: #3987e5; --seq-4: #256abf; --seq-5: #184f95;
  --asc: #3987e5; --desc: #d95926;
  --s1a: #3987e5; --s1b: #c98500; --s1c: #d55181; --s1d: #008300;
  --dup: #e66767;
}

* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  background: var(--surface-0); color: var(--text-1);
  font: 13px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
#app { display: flex; height: 100%; }

/* ---- sidebar ---- */
#sidebar {
  width: 330px; flex: 0 0 330px; display: flex; flex-direction: column;
  background: var(--surface-1); border-right: 1px solid var(--border);
}
#head { padding: 14px 16px 12px; border-bottom: 1px solid var(--border); }
#head h1 { margin: 0; font-size: 15px; font-weight: 650; letter-spacing: -0.01em; }
#head .sub { color: var(--text-2); font-size: 12px; margin-top: 3px; }
#head .built { color: var(--text-3); font-size: 11px; margin-top: 2px; }
#theme { float: right; background: none; border: 1px solid var(--border); color: var(--text-2);
  border-radius: 6px; width: 26px; height: 26px; cursor: pointer; font-size: 13px; line-height: 1; }
#theme:hover { background: var(--surface-2); color: var(--text-1); }
#scroll { overflow-y: auto; flex: 1; }

.section { border-bottom: 1px solid var(--border); }
.section-head {
  display: flex; justify-content: space-between; align-items: center; cursor: pointer;
  padding: 10px 16px; font-size: 11px; font-weight: 650; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--text-2); user-select: none;
}
.section-head:hover { color: var(--text-1); }
.chev { transition: transform 0.15s; font-size: 9px; color: var(--text-3); }
.section.collapsed .chev { transform: rotate(-90deg); }
.section.collapsed .section-body { display: none; }
.section-body { padding: 0 16px 14px; }
.note { color: var(--text-3); font-size: 11px; line-height: 1.45; margin: 0 0 9px; }
label { display: block; font-size: 11px; color: var(--text-2); margin: 10px 0 5px; }
select, input[type=date], input[type=search] {
  width: 100%; background: var(--surface-2); color: var(--text-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 5px 7px; font: inherit; font-size: 12px;
}
input[type=range] { width: 100%; accent-color: var(--accent); }
input[type=file] {
  width: 100%; font-size: 11px; color: var(--text-2);
  border: 1px dashed var(--border); border-radius: 6px; padding: 7px; background: var(--surface-2);
}
input[type=file]::file-selector-button {
  background: var(--surface-1); color: var(--text-1); border: 1px solid var(--border);
  border-radius: 5px; padding: 3px 8px; margin-right: 8px; cursor: pointer; font: inherit;
  font-size: 11px;
}
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip {
  padding: 3px 9px; border-radius: 999px; border: 1px solid var(--border);
  background: var(--surface-2); color: var(--text-2); cursor: pointer; font-size: 11px;
  font-variant-numeric: tabular-nums; user-select: none;
}
.chip.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.row { display: flex; gap: 6px; align-items: center; }
.btn {
  background: var(--surface-2); color: var(--text-2); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 9px; cursor: pointer; font: inherit; font-size: 11px;
}
.btn:hover { color: var(--text-1); }
.toggle { display: flex; align-items: center; gap: 7px; cursor: pointer; margin: 9px 0 0;
  font-size: 12px; color: var(--text-2); }
.toggle input { accent-color: var(--accent); margin: 0; }

/* ---- stats ---- */
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 4px; }
.stat { background: var(--surface-2); border-radius: 7px; padding: 8px 10px; }
.stat .v { font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.stat .k { font-size: 10px; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 1px; }

/* ---- charts ---- */
.chart { position: relative; }
.chart svg { display: block; width: 100%; overflow: visible; }
.bar { fill: var(--accent); }
.bar:hover { fill: var(--text-1); }
.axis { fill: var(--text-3); font-size: 9px; font-variant-numeric: tabular-nums; }
.gridline { stroke: var(--border); stroke-width: 1; }
.tip {
  position: absolute; pointer-events: none; z-index: 5;
  background: var(--surface-1); color: var(--text-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px;
  font-size: 11px; white-space: nowrap; box-shadow: 0 3px 12px rgb(0 0 0 / 0.16);
}
.tip[hidden] { display: none; }

/* ---- legend ---- */
.legend { display: flex; flex-direction: column; gap: 4px; margin-top: 8px; }
.legend div { display: flex; align-items: center; gap: 7px; font-size: 11px; color: var(--text-2); }
.legend i { width: 13px; height: 13px; border-radius: 3px; flex: none; }

/* ---- selected frame ---- */
#detail .empty { color: var(--text-3); font-size: 12px; }
#detail h2 { margin: 0 0 2px; font-size: 14px; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
#detail .meta { color: var(--text-2); font-size: 11px; margin-bottom: 10px; }
table.kv { width: 100%; border-collapse: collapse; font-size: 11px; font-variant-numeric: tabular-nums; }
table.kv th { text-align: left; font-weight: 500; color: var(--text-3); padding: 2px 0; width: 42%; }
table.kv td { text-align: right; color: var(--text-1); padding: 2px 0; }
.dates { max-height: 148px; overflow-y: auto; margin-top: 8px;
  border-top: 1px solid var(--border); }
.dates div { display: grid; grid-template-columns: auto 1fr auto auto; gap: 7px;
  align-items: center; padding: 3px 0;
  font-size: 11px; font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--border); }
.dates i { width: 9px; height: 9px; border-radius: 2px; }
.dates .sensor { color: var(--text-2); font-size: 10px; }
.dates .gap { color: var(--text-3); }
.dates div.dup { color: var(--dup); }
.dates div.dup .gap { color: var(--dup); }
.warn { background: color-mix(in srgb, var(--dup) 12%, transparent);
  border: 1px solid var(--dup); color: var(--text-1); border-radius: 6px;
  padding: 7px 9px; font-size: 11px; margin: 9px 0 0; line-height: 1.45; }

/* ---- frame list ---- */
.grow { display: flex; gap: 8px; align-items: center; }
.grow input[type=range] { flex: 1; }
.grow input[type=number] {
  width: 68px; flex: none; background: var(--surface-2); color: var(--text-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 6px;
  font: inherit; font-size: 12px; font-variant-numeric: tabular-nums;
}
.cliline {
  margin-top: 9px; padding: 7px 9px; background: var(--surface-2);
  border-radius: 6px; color: var(--text-2); font-size: 11px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all;
}
.flist { max-height: 210px; overflow-y: auto; border-top: 1px solid var(--border); }
.flist button {
  display: grid; grid-template-columns: 1fr auto auto; gap: 8px; align-items: center;
  width: 100%; text-align: left; background: none; border: none;
  border-bottom: 1px solid var(--border); color: var(--text-1);
  padding: 5px 2px; cursor: pointer; font: inherit; font-size: 11px;
  font-variant-numeric: tabular-nums;
}
.flist button:hover { background: var(--surface-2); }
.flist button.on { background: var(--surface-2); font-weight: 600; }
.flist .n { color: var(--text-2); }
.flist .sw { width: 9px; height: 9px; border-radius: 2px; }

/* ---- map ---- */
#map { flex: 1; position: relative; }
.maplibregl-popup-content {
  background: var(--surface-1); color: var(--text-1);
  border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
  font: 12px/1.6 system-ui, sans-serif; box-shadow: 0 4px 18px rgb(0 0 0 / 0.22);
}
.maplibregl-popup-tip { display: none; }
.maplibregl-popup-content b { font-variant-numeric: tabular-nums; }
/* ---- floating frame panel ---- */
#fpanel {
  position: absolute; top: 58px; right: 14px; width: 620px; height: 520px;
  min-width: 360px; min-height: 200px; max-width: calc(100% - 28px);
  /* `resize` needs a non-visible overflow; the body inside does the scrolling. */
  resize: both; overflow: hidden;
  display: flex; flex-direction: column; z-index: 6;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  box-shadow: 0 12px 44px rgb(0 0 0 / 0.30);
}
#fpanel[hidden] { display: none; }
#fpanel-head {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 9px 8px 9px 13px; border-bottom: 1px solid var(--border);
  cursor: move; user-select: none; flex: none;
}
#fpanel-head .t { font-size: 14px; font-weight: 650; font-variant-numeric: tabular-nums; }
#fpanel-close {
  background: none; border: none; color: var(--text-2); font-size: 20px; line-height: 1;
  padding: 0 7px 2px; cursor: pointer; border-radius: 6px;
}
#fpanel-close:hover { background: var(--surface-2); color: var(--text-1); }
#fpanel-body { flex: 1; overflow: auto; padding: 12px 14px 16px; }
#fpanel-body h3 {
  font-size: 11px; font-weight: 650; letter-spacing: .06em; text-transform: uppercase;
  color: var(--text-2); margin: 16px 0 6px;
}
#fpanel-body .meta { color: var(--text-2); font-size: 12px; margin-bottom: 12px; }
#fpanel-body svg {
  display: block; width: 100%; height: auto; background: var(--surface-0);
  border: 1px solid var(--border); border-radius: 8px;
}
#fpanel-body .ax { fill: var(--text-3); font-size: 10px; font-variant-numeric: tabular-nums; }
#fpanel-body .lg {
  display: inline-flex; align-items: center; gap: 6px; margin-right: 14px;
  font-size: 12px; color: var(--text-2);
}
#fpanel-body .lg i { width: 12px; height: 12px; border-radius: 3px; }
#fpanel-body .sw {
  display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 7px;
}
.pstats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
.pstats .stat { background: var(--surface-2); border-radius: 7px; padding: 8px 10px; }
table.acq {
  width: 100%; border-collapse: collapse; font-size: 12px;
  font-variant-numeric: tabular-nums; margin-top: 4px;
}
table.acq th {
  text-align: left; color: var(--text-3); font-weight: 500; font-size: 11px;
  padding: 4px 6px; border-bottom: 1px solid var(--border);
}
table.acq td { padding: 4px 6px; border-bottom: 1px solid var(--border); }
table.acq td.r { text-align: right; color: var(--text-2); }
table.acq td.g { color: var(--text-3); font-size: 11px; word-break: break-all; }
table.acq tr.dup td { color: var(--dup); }
#fpanel-body .grid { color: var(--text-3); font-size: 11px; margin-top: 14px; }

#search { position: absolute; top: 10px; left: 10px; z-index: 3; width: 268px; }
#search input {
  width: 100%; background: var(--surface-1); color: var(--text-1);
  border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px;
  font: inherit; font-size: 12px; box-shadow: 0 2px 10px rgb(0 0 0 / 0.14);
}
#search input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
#results {
  margin-top: 4px; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 7px; overflow: hidden; box-shadow: 0 4px 16px rgb(0 0 0 / 0.2);
  max-height: 260px; overflow-y: auto;
}
#results[hidden] { display: none; }
#results button {
  display: block; width: 100%; text-align: left; background: none; border: none;
  border-bottom: 1px solid var(--border); color: var(--text-1);
  padding: 7px 10px; cursor: pointer; font: inherit; font-size: 12px; line-height: 1.35;
}
#results button:hover { background: var(--surface-2); }
#results button small { display: block; color: var(--text-3); font-size: 11px; }
#results .msg { padding: 7px 10px; color: var(--text-3); font-size: 11px; }

#panel-toggle {
  position: absolute; top: 10px; right: 232px; z-index: 3;
  width: 30px; height: 28px; padding: 0; cursor: pointer; line-height: 0;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--surface-1); color: var(--text-3);
  border: 1px solid var(--border); border-radius: 7px;
}
#panel-toggle.on { background: var(--accent); color: #fff; border-color: var(--accent); }
#panel-toggle:hover { filter: brightness(1.08); }

#basemap {
  position: absolute; top: 10px; right: 10px; z-index: 2; display: flex; gap: 0;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 7px; overflow: hidden;
}
#basemap button {
  background: none; border: none; color: var(--text-2); padding: 5px 10px;
  cursor: pointer; font: inherit; font-size: 11px;
}
#basemap button.on { background: var(--accent); color: #fff; }
@media (max-width: 820px) {
  #app { flex-direction: column; }
  #sidebar { width: 100%; flex: 0 0 auto; max-height: 52vh; border-right: none;
    border-bottom: 1px solid var(--border); }
  #map { min-height: 48vh; }
}
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div id="head">
      <button id="theme" title="Switch theme">&#9789;</button>
      <h1>__TITLE__</h1>
      <div class="sub" id="hdr-sub">__SUBTITLE__</div>
      <div class="built">Built __BUILT__</div>
    </div>
    <div id="scroll">

      <div class="section">
        <div class="section-head" data-target="sec-summary"><span>Archive</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-summary">
          <div class="stats">
            <div class="stat"><div class="v" id="s-frames">0</div><div class="k">frames shown</div></div>
            <div class="stat"><div class="v" id="s-acq">0</div><div class="k">acquisitions</div></div>
            <div class="stat"><div class="v" id="s-ts">0</div><div class="k">5+ passes</div></div>
            <div class="stat"><div class="v" id="s-tracks">0</div><div class="k">tracks</div></div>
            <div class="stat"><div class="v" id="s-dup">0</div><div class="k">frames w/ duplicates</div></div>
            <div class="stat"><div class="v" id="s-sensors">0</div><div class="k">sensors</div></div>
          </div>
          <p class="note" id="note-ts">A frame needs repeated passes before it can carry a
            time series; most stripmap frames are imaged once.</p>
        </div>
      </div>

      <div class="section">
        <div class="section-head" data-target="sec-time"><span>Acquisitions over time</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-time">
          <p class="note">Acquisitions per day across the frames currently shown. Hover a bar for its count.</p>
          <div class="chart" id="daily"><div class="tip" id="daily-tip" hidden></div></div>
          <div class="row" style="margin-top:8px;">
            <input type="date" id="d0" aria-label="From date">
            <span style="color:var(--text-3);font-size:11px;">to</span>
            <input type="date" id="d1" aria-label="To date">
          </div>
          <button class="btn" id="d-reset" style="margin-top:6px;">All dates</button>
        </div>
      </div>

      <div class="section">
        <div class="section-head" data-target="sec-filter"><span>Filters</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-filter">
          <label>Beam</label>
          <div class="chips" id="chips-beam"></div>
          <label>Pass direction</label>
          <div class="chips" id="chips-pass"></div>
          <label>Minimum acquisitions (<span id="min-acq-v">1</span>)</label>
          <input type="range" id="min-acq" min="1" max="10" value="1">
          <label>Minimum useful area (<span id="min-fill-v">0</span>% of the box)</label>
          <input type="range" id="min-fill" min="0" max="70" step="5" value="0">
          <label>Track</label>
          <input type="search" id="q-track" placeholder="e.g. 95, or blank for all">
          <button class="btn" id="f-reset" style="margin-top:10px;">Reset filters</button>
        </div>
      </div>

      <div class="section">
        <div class="section-head" data-target="sec-style"><span>Appearance</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-style">
          <label>Colour frames by</label>
          <select id="color-by">
            <option value="n_acquisitions" selected>Acquisitions</option>
            <option value="repeat_days">Median repeat interval</option>
            <option value="pass">Pass direction</option>
            <option value="duplicates">Duplicate dates</option>
            <option value="fill_pct">Useful area (fill)</option>
          </select>
          <div class="legend" id="legend"></div>
          <label style="margin-top:12px;">Fill opacity (<span id="op-v">30</span>%)</label>
          <input type="range" id="op" min="0" max="100" value="30">
          <label class="toggle"><input type="checkbox" id="show-grid">
            <span>Show pinned grid (what a product covers)</span></label>
          <p class="note" style="margin-top:6px;">The filled quadrilateral is the frame
            footprint, following the orbit. The pinned grid is its bounding box in the
            frame's UTM zone, padded 5 km and snapped to 30 m -- north-up and larger.
            That box is what COMPASS geocodes into.</p>
        </div>
      </div>

      <div class="section">
        <div class="section-head" data-target="sec-sites"><span>Areas of interest</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-sites">
          <p class="note">Load a GeoJSON of places you care about to see which frames
            cover them. Points and polygons both work. The file is read in your
            browser and never leaves this machine.</p>
          <input type="file" id="aoi-file" accept=".geojson,.json,application/geo+json,application/json">
          <label class="toggle"><input type="checkbox" id="show-sites">
            <span>Show on the map</span></label>
          <div class="flist" id="sitelist" style="max-height:200px;margin-top:9px;"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-head" data-target="sec-geom"><span>Frame bounds</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-geom">
          <p class="note">Adjusts <b>one frame at a time</b> -- the selected one --
            because a boundary that cuts through an island is a problem for that
            frame, not for the archive. Select a frame, nudge it, then export the
            set and rebuild with <code>--overrides</code>.</p>
          <div class="cliline" id="g-target">No frame selected.</div>
          <label>Move along track (km)</label>
          <div class="grow"><input type="range" id="g-shift" min="-300" max="300" step="1" value="0">
            <input type="number" id="g-shift-v" value="0" step="1"></div>
          <label>Extend each end (km)</label>
          <div class="grow"><input type="range" id="g-grow" min="-50" max="300" step="1" value="0">
            <input type="number" id="g-grow-v" value="0" step="1"></div>
          <label>Trim each side (km)</label>
          <div class="grow"><input type="range" id="g-inset" min="-100" max="100" step="1" value="0">
            <input type="number" id="g-inset-v" value="0" step="1"></div>
          <label style="margin-top:12px;">Merge with a neighbour</label>
          <div class="row">
            <button class="btn" id="g-merge-prev">+ previous</button>
            <button class="btn" id="g-merge-next">+ next</button>
            <button class="btn" id="g-unmerge">unmerge</button>
          </div>
          <p class="note" style="margin-top:6px;">Two consecutive frames become one,
            keeping the lower id, so a target sitting on the boundary is whole in a
            single frame.</p>
          <div class="row" style="margin-top:9px;">
            <button class="btn" id="g-reset">Reset this frame</button>
            <button class="btn" id="g-export">Export adjustments</button>
          </div>
          <div class="flist" id="g-list" style="max-height:150px;margin-top:9px;"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-head" data-target="sec-list"><span>Frames</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-list">
          <p class="note">Every frame matching the filters, most-imaged first. Click one
            to select it, with or without a map.</p>
          <div class="flist" id="flist"></div>
        </div>
      </div>

      <div class="section">
        <div class="section-head" data-target="sec-detail"><span>Selected frame</span><span class="chev">&#9660;</span></div>
        <div class="section-body" id="sec-detail">
          <div id="detail"><div class="empty">Click a frame on the map.</div></div>
        </div>
      </div>

    </div>
  </aside>

  <div id="map">
    <div id="search">
      <input id="q" type="search" autocomplete="off" spellcheck="false"
             placeholder="Place, lat lon, or frame id">
      <div id="results" hidden></div>
    </div>
    <div id="fpanel" hidden></div>
    <button id="panel-toggle" class="on" title="Open the time series panel when a frame is clicked"
            aria-label="Open the time series panel when a frame is clicked">
      <svg viewBox="0 0 16 16" width="15" height="15" fill="none"
           stroke="currentColor" stroke-width="1.5" stroke-linecap="round"
           stroke-linejoin="round" aria-hidden="true">
        <rect x="1.2" y="3.2" width="9.6" height="9.6" rx="1.6"/>
        <path d="M6.2 3.2V1.9a.7.7 0 0 1 .7-.7h7.2a.7.7 0 0 1 .7.7v7.2a.7.7 0 0 1-.7.7h-1.3"/>
      </svg>
    </button>
    <div id="basemap">
      <button data-base="dark" class="on">Dark</button>
      <button data-base="light">Light</button>
      <button data-base="sat">Satellite</button>
      <button data-base="hybrid">Hybrid</button>
    </div>
  </div>
</div>

<script>
const FRAMES = __FRAMES__;
const TILE_SECONDS = __TILE_SECONDS__;
const GRIDS  = __GRIDS__;

/* ---------------------------------------------------------------- theme --- */
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
let theme = prefersDark ? "dark" : "light";
document.documentElement.dataset.theme = theme;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ------------------------------------------------------------- basemaps --- */
// Keyless raster sources only: no account, token or usage plan to keep alive,
// so a committed copy of this page still works years from now.
const BASES = {
  dark: {
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
    attribution: "Esri, HERE, Garmin, &copy; OpenStreetMap contributors", maxzoom: 16
  },
  light: {
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
    attribution: "Esri, HERE, Garmin, &copy; OpenStreetMap contributors", maxzoom: 16
  },
  sat: {
    tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
    attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics", maxzoom: 18
  },
  // Google's hybrid tiles (imagery plus roads and place labels). Keyless like the
  // rest, but unlike them it is not an open endpoint: Google's terms expect their
  // imagery to be used through the Maps API. Fine for looking at frames on your
  // own screen; swap it for "Satellite" above before putting this page anywhere
  // public.
  hybrid: {
    tiles: ["https://mt0.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "https://mt2.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "https://mt3.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"],
    attribution: "Imagery &copy; Google", maxzoom: 20
  }
};

// Imagery basemaps are dark whatever the page theme, so the theme toggle leaves
// them alone instead of yanking the user back to a road map.
const IMAGERY = new Set(["sat", "hybrid"]);
let baseName = theme === "dark" ? "dark" : "light";

function styleFor(name) {
  const b = BASES[name];
  return {
    version: 8,
    // The globe belongs in the style, not in a setProjection() call after
    // construction: as a style property it is the projection from the first
    // paint and it survives every basemap swap. Calling map.setProjection()
    // before the style has loaded throws, which kills the rest of this script.
    projection: { type: "globe" },
    sources: { base: { type: "raster", tiles: b.tiles, tileSize: 256,
                       attribution: b.attribution, maxzoom: b.maxzoom } },
    layers: [
      { id: "bg", type: "background",
        paint: { "background-color": IMAGERY.has(name) ? "#0a0a0a" : cssVar("--surface-0") } },
      { id: "base", type: "raster", source: "base" }
    ]
  };
}

function fail(message) {
  const host = document.getElementById("map");
  host.insertAdjacentHTML("afterbegin",
    `<div style="position:absolute;inset:0;display:grid;place-items:center;padding:24px;
       text-align:center;color:var(--text-2);font-size:13px;z-index:3;">
       <div><b style="color:var(--text-1)">The map could not start.</b><br>${message}<br>
       <span style="color:var(--text-3);font-size:12px;">The sidebar figures below are
       computed from the embedded data and are still correct.</span></div></div>`);
}

// MapLibre renders through WebGL and its constructor throws outright where WebGL
// is unavailable -- a remote desktop, a locked-down browser, a headless session.
// That must not take the page with it: every figure, filter and time series here
// is computed from the embedded data and stays usable without a map.
let map = null;
let mapReady = false;

if (typeof maplibregl === "undefined") {
  fail("MapLibre GL did not load from the CDN, so this page had no map library. " +
       "It needs network access the first time it is opened.");
} else {
  try {
    map = new maplibregl.Map({
      container: "map",
      style: styleFor(baseName),
      center: [0, 15],
      zoom: 1.4,
      attributionControl: { compact: true }
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");
    if (maplibregl.GlobeControl) {
      map.addControl(new maplibregl.GlobeControl(), "bottom-right");
    }
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  } catch (err) {
    map = null;
    fail("This browser could not start WebGL, which MapLibre needs to draw a map. " +
         "Use the frame list in the sidebar instead.");
  }
}

/* ---------------------------------------------------------------- state --- */
const ALL_BEAMS = [...new Set(FRAMES.features.map(f => f.properties.beam))].sort();
// A frame's pass direction follows its track geometry; ascending frames run
// south-to-north, so the footprint's last corner sits north of its first.
for (const f of FRAMES.features) {
  const ring = f.geometry.coordinates[0];
  f.properties.pass = ring[2][1] > ring[0][1] ? "Ascending" : "Descending";
}
const ALL_PASSES = ["Ascending", "Descending"];

const state = {
  beams: new Set(ALL_BEAMS),
  passes: new Set(ALL_PASSES),
  minAcq: 1,
  minFill: 0,
  track: "",
  overrides: {},          // frame_id -> {shiftKm, growKm, insetKm}
  merges: [],             // arrays of consecutive frame ids drawn as one
  showSites: false,
  autoPanel: true,
  d0: null,
  d1: null,
  colorBy: "n_acquisitions",
  opacity: 0.30,
  showGrid: false,
  selected: null
};

function inDateWindow(props) {
  if (!state.d0 && !state.d1) return true;
  return props.dates.some(d => (!state.d0 || d >= state.d0) && (!state.d1 || d <= state.d1));
}

function visible() {
  return FRAMES.features.filter(f => {
    const p = f.properties;
    if (!state.beams.has(p.beam)) return false;
    if (!state.passes.has(p.pass)) return false;
    if (p.n_acquisitions < state.minAcq) return false;
    if (p.fill_pct < state.minFill) return false;
    if (state.track && String(p.track) !== state.track.replace(/^0+/, "")) return false;
    if (!inDateWindow(p)) return false;
    return true;
  });
}

/* --------------------------------------------------------------- colour --- */
const ACQ_BINS = [
  { max: 1,        label: "1 pass",     v: "--seq-1" },
  { max: 2,        label: "2",          v: "--seq-2" },
  { max: 4,        label: "3 to 4",     v: "--seq-3" },
  { max: 9,        label: "5 to 9",     v: "--seq-4" },
  { max: Infinity, label: "10 or more", v: "--seq-5" }
];
// How much of the pinned box the footprint actually covers. The rest is nodata
// in every product of that frame, so a low value is wasted disk and compute.
const FILL_BINS = [
  { max: 30,       label: "under 30% useful", v: "--seq-1" },
  { max: 40,       label: "30 to 40%",        v: "--seq-2" },
  { max: 50,       label: "40 to 50%",        v: "--seq-3" },
  { max: 60,       label: "50 to 60%",        v: "--seq-4" },
  { max: Infinity, label: "over 60%",         v: "--seq-5" }
];
const REPEAT_BINS = [
  { max: 6,        label: "6 days or less", v: "--seq-5" },
  { max: 12,       label: "7 to 12 days",   v: "--seq-4" },
  { max: 24,       label: "13 to 24 days",  v: "--seq-3" },
  { max: 60,       label: "25 to 60 days",  v: "--seq-2" },
  { max: Infinity, label: "over 60 days",   v: "--seq-1" }
];

const SENSORS = ["S1A", "S1B", "S1C", "S1D"];
const SENSOR_VAR = { S1A: "--s1a", S1B: "--s1b", S1C: "--s1c", S1D: "--s1d" };
function sensorColor(p) { return cssVar(SENSOR_VAR[p] || "--text-3"); }

function colorOf(props) {
  if (state.colorBy === "duplicates") {
    return cssVar(props.n_duplicate ? "--dup" : "--seq-1");
  }
  if (state.colorBy === "pass") {
    return cssVar(props.pass === "Ascending" ? "--asc" : "--desc");
  }
  if (state.colorBy === "fill_pct") {
    return cssVar(FILL_BINS.find(b => props.fill_pct <= b.max).v);
  }
  if (state.colorBy === "repeat_days") {
    if (props.repeat_days == null) return cssVar("--text-3");
    return cssVar(REPEAT_BINS.find(b => props.repeat_days <= b.max).v);
  }
  return cssVar(ACQ_BINS.find(b => props.n_acquisitions <= b.max).v);
}

function renderLegend() {
  const el = document.getElementById("legend");
  let rows;
  if (state.colorBy === "duplicates") {
    rows = [[cssVar("--dup"), "has a repeated date"], [cssVar("--seq-1"), "one granule per date"]];
  } else if (state.colorBy === "pass") {
    rows = ALL_PASSES.map(p => [cssVar(p === "Ascending" ? "--asc" : "--desc"), p]);
  } else if (state.colorBy === "fill_pct") {
    rows = FILL_BINS.map(b => [cssVar(b.v), b.label]);
  } else if (state.colorBy === "repeat_days") {
    rows = REPEAT_BINS.map(b => [cssVar(b.v), b.label]);
    rows.push([cssVar("--text-3"), "single pass (no interval)"]);
  } else {
    rows = ACQ_BINS.map(b => [cssVar(b.v), b.label]);
  }
  el.innerHTML = rows
    .map(([c, t]) => `<div><i style="background:${c}"></i>${t}</div>`)
    .join("");
}

/* ------------------------------------------------- adjustable geometry --- */
// The three sliders move the frame on the ground without touching its ID, the
// same way `--shift-seconds`, `--overlap-seconds` and `--inset-m` do when the
// database is rebuilt. Doing it here first makes the choice visible before a
// rebuild commits to it.
//
// A frame ring is [nearStart, farStart, farStop, nearStop], so the along-track
// axis runs from the midpoint of the first edge to the midpoint of the last, and
// the cross-track axis lies along either end edge.
const KM_PER_DEG = 111.32;

// Areas of interest to locate on the map while judging whether the archive
// covers anything useful. The page ships with whatever `--aoi` supplied, which is
// an empty collection by default, and the file picker loads more at runtime. A
// chosen file is read in the browser with FileReader and never leaves the
// machine, so nothing site-specific has to live in this repo.
let AOI = __AOI__;

function aoiName(feature, i) {
  const p = feature.properties || {};
  return p.name || p.Name || p.NAME || p.title || p.id || `Area ${i + 1}`;
}

function aoiNote(feature) {
  const p = feature.properties || {};
  return p.note || p.description || p.Description || "";
}

// One representative point per feature, for the sidebar list and for flying to.
function aoiPoint(feature) {
  const g = feature.geometry || {};
  if (g.type === "Point") return g.coordinates;
  const coords = [];
  (function walk(c) {
    if (typeof c[0] === "number") { coords.push(c); return; }
    c.forEach(walk);
  })(g.coordinates || []);
  if (!coords.length) return null;
  let x = 0, y = 0;
  for (const c of coords) { x += c[0]; y += c[1]; }
  return [x / coords.length, y / coords.length];
}

function aoiBounds(feature) {
  const coords = [];
  (function walk(c) {
    if (typeof c[0] === "number") { coords.push(c); return; }
    c.forEach(walk);
  })((feature.geometry || {}).coordinates || []);
  if (!coords.length) return null;
  let [w, s2, e, n] = [180, 90, -180, -90];
  for (const [x, y] of coords) {
    w = Math.min(w, x); e = Math.max(e, x); s2 = Math.min(s2, y); n = Math.max(n, y);
  }
  return [w, s2, e, n];
}

function adjustRing(ring, frameId) {
  const o = state.overrides[frameId];
  if (!o) return ring;
  const { shiftKm, growKm, insetKm } = o;
  if (!shiftKm && !growKm && !insetKm) return ring;

  const [a, b, c, d] = ring;                       // near0, far0, far1, near1
  const lat = (a[1] + c[1]) / 2;
  const kx = KM_PER_DEG * Math.max(0.05, Math.cos((lat * Math.PI) / 180));
  const toDeg = (p, km) => [p[0] * km / kx, p[1] * km / KM_PER_DEG];

  const mid0 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const mid1 = [(c[0] + d[0]) / 2, (c[1] + d[1]) / 2];
  const along = unit([mid1[0] - mid0[0], mid1[1] - mid0[1]], kx);
  const across = unit([b[0] - a[0], b[1] - a[1]], kx);

  const shift = toDeg(along, shiftKm);
  const grow = toDeg(along, growKm);
  const trim = toDeg(across, insetKm);

  const move = (p, ...deltas) => {
    let [x, y] = p;
    for (const dd of deltas) { x += dd[0]; y += dd[1]; }
    return [x, y];
  };
  const neg = v => [-v[0], -v[1]];

  return [
    move(a, shift, neg(grow), trim),
    move(b, shift, neg(grow), neg(trim)),
    move(c, shift, grow, neg(trim)),
    move(d, shift, grow, trim)
  ];
}

// Unit vector in local kilometres, returned in the same (lon, lat) degree basis.
function unit(v, kx) {
  const x = v[0] * kx, y = v[1] * KM_PER_DEG;
  const n = Math.hypot(x, y) || 1;
  return [v[0] / n * (kx / kx), v[1] / n];
}

// Frames are contiguous along track, so a merged group is drawn as one
// quadrilateral running from the first member's leading edge to the last
// member's trailing edge -- the same shape `--merge` produces on a rebuild.
function neighbourId(p, step) {
  return `t${String(p.track).padStart(3, "0")}_` +
         `${String(p.frame_index + step).padStart(6, "0")}_${p.beam.toLowerCase()}`;
}

function groupOf(frameId) {
  return state.merges.find(g => g.includes(frameId)) || null;
}

function mergedFeatures(features) {
  if (!state.merges.length) return features;
  const byId = new Map(features.map(f => [f.properties.frame_id, f]));
  const used = new Set();
  const out = [];

  for (const group of state.merges) {
    const members = group.map(id => byId.get(id));
    if (members.some(m => !m)) continue;              // not all are shown
    const ordered = members.slice().sort(
      (a, b) => a.properties.frame_index - b.properties.frame_index);
    const first = adjustRing(ordered[0].geometry.coordinates[0].slice(0, 4),
                             ordered[0].properties.frame_id);
    const last = adjustRing(
      ordered[ordered.length - 1].geometry.coordinates[0].slice(0, 4),
      ordered[ordered.length - 1].properties.frame_id);
    const ring = [first[0], first[1], last[2], last[3]];
    const acq = ordered.reduce((a, m) => a + m.properties.n_acquisitions, 0);
    out.push({
      type: "Feature",
      geometry: { type: "Polygon", coordinates: [[...ring, ring[0]]] },
      properties: { ...ordered[0].properties, n_acquisitions: acq, _merged: group.length }
    });
    group.forEach(id => used.add(id));
  }
  for (const f of features) {
    if (!used.has(f.properties.frame_id)) out.push(f);
  }
  return out;
}

function adjustedFeature(f) {
  const ring = adjustRing(f.geometry.coordinates[0].slice(0, 4), f.properties.frame_id);
  return {
    ...f,
    geometry: { type: "Polygon", coordinates: [[...ring, ring[0]]] }
  };
}

function overrideFlags(o) {
  const s = km => (km / groundSpeedKmS()).toFixed(2);
  const bits = [];
  if (o.shiftKm) bits.push(`shift ${s(o.shiftKm)}s`);
  if (o.growKm) bits.push(`overlap ${s(o.growKm)}s`);
  if (o.insetKm) bits.push(`inset ${Math.round(o.insetKm * 1000)}m`);
  return bits.join(", ");
}

function updateCliLine() {
  const target = document.getElementById("g-target");
  const id = state.selected;
  target.textContent = id
    ? `Adjusting ${id}${overrideFlags(state.overrides[id] || {}) ? " -- " + overrideFlags(state.overrides[id]) : ""}`
    : "No frame selected.";

  const list = document.getElementById("g-list");
  const ids = Object.keys(state.overrides);
  const mergeRows = state.merges.map(g =>
    `<button data-ov="${g[0]}"><span>${g[0]}</span>
       <span class="n">merged with ${g.length - 1} more</span>
       <i class="sw" style="background:${cssVar("--accent")}"></i></button>`).join("");
  list.innerHTML = (mergeRows || "") + (ids.length
    ? ids.sort().map(k =>
        `<button data-ov="${k}"><span>${k}</span>
           <span class="n">${overrideFlags(state.overrides[k])}</span>
           <i class="sw" style="background:${cssVar("--grid-line")}"></i></button>`).join("")
    : (mergeRows ? "" : '<div class="note" style="margin:6px 0 0">No frames adjusted yet.</div>'));
}

// The overrides file `sm-db build --overrides` reads: seconds along track,
// metres across, keyed by frame id.
function exportOverrides() {
  const speed = groundSpeedKmS();
  const out = {};
  for (const [id, o] of Object.entries(state.overrides)) {
    const entry = {};
    if (o.shiftKm) entry.shift = +(o.shiftKm / speed).toFixed(3);
    if (o.growKm) entry.overlap = +(o.growKm / speed).toFixed(3);
    if (o.insetKm) entry.inset = Math.round(o.insetKm * 1000);
    if (Object.keys(entry).length) out[id] = entry;
  }
  const payload = { overrides: out, merges: state.merges };
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sm_frame_adjustments.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

let _speed = null;
function groundSpeedKmS() {
  if (_speed) return _speed;
  const f = FRAMES.features[0];
  if (!f) return 6.5;
  const r = f.geometry.coordinates[0];
  const mid0 = [(r[0][0] + r[1][0]) / 2, (r[0][1] + r[1][1]) / 2];
  const mid1 = [(r[2][0] + r[3][0]) / 2, (r[2][1] + r[3][1]) / 2];
  const lat = (mid0[1] + mid1[1]) / 2;
  const dx = (mid1[0] - mid0[0]) * KM_PER_DEG * Math.cos((lat * Math.PI) / 180);
  const dy = (mid1[1] - mid0[1]) * KM_PER_DEG;
  _speed = Math.hypot(dx, dy) / TILE_SECONDS;
  return _speed;
}

/* ------------------------------------------------------------------ map --- */
function paint() {
  const shown = visible();
  const ids = new Set(shown.map(f => f.properties.frame_id));

  const fc = {
    type: "FeatureCollection",
    features: mergedFeatures(shown).map(f => {
      const adj = f.properties._merged ? f : adjustedFeature(f);
      return { ...adj, properties: { ...f.properties, _color: colorOf(f.properties) } };
    })
  };
  if (map && mapReady) {
    map.getSource("frames").setData(fc);
    map.getSource("grids").setData({
      type: "FeatureCollection",
      features: state.showGrid
        ? GRIDS.features.filter(g => ids.has(g.properties.frame_id))
        : []
    });
    map.setPaintProperty("frames-fill", "fill-opacity", state.opacity);
    for (const id of ["aoi-fill", "aoi-line", "aoi-point"]) {
      map.setLayoutProperty(id, "visibility", state.showSites ? "visible" : "none");
    }
    map.getSource("aoi").setData(AOI);
    map.setFilter("frames-outline-sel",
      ["==", ["get", "frame_id"], state.selected || "__none__"]);
  }

  renderSummary(shown);
  renderDaily(shown);
  renderLegend();
  renderList(shown);
}

function addLayers() {
  map.addSource("frames", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("grids",  { type: "geojson", data: { type: "FeatureCollection", features: [] } });

  map.addLayer({
    id: "frames-fill", type: "fill", source: "frames",
    paint: { "fill-color": ["get", "_color"], "fill-opacity": state.opacity }
  });
  map.addLayer({
    id: "frames-outline", type: "line", source: "frames",
    paint: { "line-color": ["get", "_color"], "line-width": 1, "line-opacity": 0.9 }
  });
  map.addLayer({
    id: "grids-outline", type: "line", source: "grids",
    paint: { "line-color": cssVar("--grid-line"), "line-width": 1.2,
             "line-dasharray": [2, 2], "line-opacity": 0.95 }
  });
  map.addSource("aoi", { type: "geojson", data: AOI });
  map.addLayer({
    id: "aoi-fill", type: "fill", source: "aoi",
    filter: ["==", ["geometry-type"], "Polygon"],
    layout: { visibility: state.showSites ? "visible" : "none" },
    paint: { "fill-color": cssVar("--grid-line"), "fill-opacity": 0.18 }
  });
  map.addLayer({
    id: "aoi-line", type: "line", source: "aoi",
    filter: ["!=", ["geometry-type"], "Point"],
    layout: { visibility: state.showSites ? "visible" : "none" },
    paint: { "line-color": cssVar("--grid-line"), "line-width": 2 }
  });
  map.addLayer({
    id: "aoi-point", type: "circle", source: "aoi",
    filter: ["==", ["geometry-type"], "Point"],
    layout: { visibility: state.showSites ? "visible" : "none" },
    paint: {
      "circle-radius": 7,
      "circle-color": cssVar("--grid-line"),
      "circle-stroke-color": cssVar("--surface-1"),
      "circle-stroke-width": 2
    }
  });
  for (const id of ["aoi-fill", "aoi-point"]) {
    map.on("click", id, e => {
      const pr = e.features[0].properties || {};
      const name = pr.name || pr.Name || pr.title || "Area of interest";
      const note = pr.note || pr.description || "";
      new maplibregl.Popup({ closeButton: true, maxWidth: "280px" })
        .setLngLat(e.lngLat)
        .setHTML(`<b>${name}</b>${note ? "<br>" + note : ""}`)
        .addTo(map);
    });
    map.on("mouseenter", id, () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", id, () => map.getCanvas().style.cursor = "");
  }

  map.addLayer({
    id: "frames-outline-sel", type: "line", source: "frames",
    filter: ["==", ["get", "frame_id"], "__none__"],
    paint: { "line-color": cssVar("--text-1"), "line-width": 2.5 }
  });

  map.on("click", "frames-fill", e => select(e.features[0].properties.frame_id, true));
  map.on("mouseenter", "frames-fill", () => map.getCanvas().style.cursor = "pointer");
  map.on("mouseleave", "frames-fill", () => map.getCanvas().style.cursor = "");
}


function fitAll() {
  if (!map || !mapReady) return;
  const shown = visible();
  if (!shown.length) return;
  let [w, s, e, n] = [180, 90, -180, -90];
  for (const f of shown) for (const [x, y] of f.geometry.coordinates[0]) {
    w = Math.min(w, x); e = Math.max(e, x); s = Math.min(s, y); n = Math.max(n, y);
  }
  map.fitBounds([[w, s], [e, n]], { padding: 60, duration: 700, maxZoom: 9 });
}

/* ------------------------------------------------------------- summary --- */
function renderSummary(shown) {
  const acq = shown.reduce((a, f) => a + f.properties.n_acquisitions, 0);
  const ts = shown.filter(f => f.properties.n_acquisitions >= 5).length;
  const tracks = new Set(shown.map(f => f.properties.track)).size;
  document.getElementById("s-frames").textContent = shown.length.toLocaleString();
  document.getElementById("s-acq").textContent = acq.toLocaleString();
  document.getElementById("s-ts").textContent = ts.toLocaleString();
  document.getElementById("s-tracks").textContent = tracks.toLocaleString();
  const dup = shown.filter(f => f.properties.n_duplicate > 0).length;
  document.getElementById("s-dup").textContent = dup.toLocaleString();
  const sensors = new Set(shown.flatMap(f => f.properties.platforms));
  document.getElementById("s-sensors").textContent = sensors.size;
}

/* ----------------------------------------------------------- frame list --- */
// The map is one way into a frame, not the only one: it needs WebGL, and a long
// tail of one-pass frames is easier to scan as a list than to hunt for on a globe.
const LIST_CAP = 400;
function renderList(shown) {
  const host = document.getElementById("flist");
  const ordered = [...shown].sort((a, b) =>
    b.properties.n_acquisitions - a.properties.n_acquisitions ||
    a.properties.frame_id.localeCompare(b.properties.frame_id));
  const rows = ordered.slice(0, LIST_CAP).map(f => {
    const p = f.properties;
    return `<button data-id="${p.frame_id}" class="${p.frame_id === state.selected ? "on" : ""}">
      <span>${p.frame_id}</span>
      <span class="n">${p.n_acquisitions} pass${p.n_acquisitions === 1 ? "" : "es"}</span>
      <i class="sw" style="background:${colorOf(p)}"></i>
    </button>`;
  }).join("");
  const more = ordered.length > LIST_CAP
    ? `<div class="note" style="margin:8px 0 0">${ordered.length - LIST_CAP} more; narrow the filters to see them.</div>`
    : "";
  host.innerHTML = rows + more;
  host.onclick = ev => {
    const b = ev.target.closest("button");
    if (b) select(b.dataset.id, true);
  };
}

/* --------------------------------------------------------- daily chart --- */
function renderDaily(shown) {
  const host = document.getElementById("daily");
  const tip = document.getElementById("daily-tip");
  const counts = new Map();
  for (const f of shown) {
    for (const d of f.properties.dates) {
      if (state.d0 && d < state.d0) continue;
      if (state.d1 && d > state.d1) continue;
      counts.set(d, (counts.get(d) || 0) + 1);
    }
  }
  const days = [...counts.keys()].sort();
  host.querySelectorAll("svg").forEach(n => n.remove());
  if (!days.length) {
    host.insertAdjacentHTML("afterbegin",
      '<div class="note" style="margin:0">No acquisitions in this selection.</div>');
    return;
  }
  host.querySelectorAll(".note").forEach(n => n.remove());

  const W = 298, H = 88, PAD_B = 14, PAD_L = 20;
  const max = Math.max(...counts.values());
  const first = new Date(days[0]), last = new Date(days[days.length - 1]);
  const span = Math.max(1, (last - first) / 86400000);
  const plotW = W - PAD_L - 2;
  // 2px surface gap between adjacent bars, per the mark spec.
  const bw = Math.max(1.5, Math.min(9, plotW / (span + 1) - 2));

  const bars = days.map(d => {
    const x = PAD_L + ((new Date(d) - first) / 86400000) * (plotW / span);
    const h = ((H - PAD_B) * counts.get(d)) / max;
    return `<rect class="bar" x="${(x - bw / 2).toFixed(1)}" y="${(H - PAD_B - h).toFixed(1)}"
      width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="${Math.min(2, bw / 2).toFixed(1)}"
      data-d="${d}" data-n="${counts.get(d)}"></rect>`;
  }).join("");

  const fmt = s => s.slice(5).replace("-", "/");
  host.insertAdjacentHTML("afterbegin", `<svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Acquisitions per day">
    <line class="gridline" x1="${PAD_L}" y1="${H - PAD_B}" x2="${W}" y2="${H - PAD_B}"></line>
    <text class="axis" x="0" y="10">${max}</text>
    <text class="axis" x="0" y="${H - PAD_B}">0</text>
    ${bars}
    <text class="axis" x="${PAD_L}" y="${H - 3}">${fmt(days[0])}</text>
    <text class="axis" x="${W}" y="${H - 3}" text-anchor="end">${fmt(days[days.length - 1])}</text>
  </svg>`);

  const svg = host.querySelector("svg");
  if (!svg) return;
  svg.addEventListener("mousemove", ev => {
    const r = ev.target.closest(".bar");
    if (!r) { tip.hidden = true; return; }
    tip.hidden = false;
    tip.textContent = `${r.dataset.d}: ${r.dataset.n} acquisition${r.dataset.n === "1" ? "" : "s"}`;
    const b = host.getBoundingClientRect();
    tip.style.left = Math.min(ev.clientX - b.left + 10, b.width - tip.offsetWidth - 2) + "px";
    tip.style.top = (ev.clientY - b.top - 28) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.hidden = true; });
}

/* -------------------------------------------------------------- detail --- */
function select(frameId, openWindow) {
  state.selected = frameId;
  const f = FRAMES.features.find(x => x.properties.frame_id === frameId);
  const el = document.getElementById("detail");
  if (!f) { el.innerHTML = '<div class="empty">Click a frame on the map.</div>'; return; }
  const p = f.properties;

  const km = m => (m / 1000).toFixed(1);
  const rows = [
    ["Track", p.track], ["Beam", p.beam], ["Frame index", p.frame_index],
    ["Pass", p.pass], ["EPSG", p.epsg],
    ["Grid size", `${km(p.width_m)} x ${km(p.height_m)} km`],
    ["Useful area", `${p.fill_pct}% (${p.nodata_pct}% nodata)`],
    ["Acquisitions", p.n_acquisitions],
    ["Distinct dates", p.n_dates],
    ["Median repeat", p.repeat_days == null ? "n/a" : `${p.repeat_days} d`],
    ["Sensors", p.platforms.join(", ") || "n/a"]
  ];

  const warn = p.n_duplicate
    ? `<div class="warn"><b>${p.n_duplicate} duplicate acquisition${
        p.n_duplicate === 1 ? "" : "s"}.</b> ${p.duplicate_dates.join(", ")} —
       more than one granule covers this frame on that date, so a stack built from
       it would carry a zero-baseline pair. Keep one granule per date.</div>`
    : "";

  el.innerHTML = `
    <h2>${p.frame_id}</h2>
    <div class="meta">${p.first ? `${p.first} to ${p.last}` : "no acquisitions"}</div>
    <table class="kv">${rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("")}</table>
    <button class="btn" id="open-win" style="margin:8px 0 2px;">
      Open the time series panel</button>
    <div class="chart" id="ts"></div>
    ${warn}
    <div class="dates">${p.dates.map((d, i) => {
      const gap = i ? (new Date(d) - new Date(p.dates[i - 1])) / 86400000 : null;
      const dup = p.duplicate_dates.includes(d);
      return `<div class="${dup ? "dup" : ""}" title="${p.granules[i]}">
        <i style="background:${sensorColor(p.sensors[i])}"></i>
        <span>${d}</span>
        <span class="sensor">${p.sensors[i]}</span>
        <span class="gap">${gap == null ? "first" : `+${gap} d`}</span></div>`;
    }).join("")}</div>
    <div style="margin-top:8px;font-size:11px;color:var(--text-3);">
      bbox ${p.bbox.join(" ")}
    </div>`;

  syncBoundsSliders();
  renderTimeline(p);
  const openBtn = document.getElementById("open-win");
  if (openBtn) openBtn.onclick = () => openFramePanel(f);
  document.getElementById("sec-detail").parentElement.classList.remove("collapsed");
  paint();
  if (openWindow && state.autoPanel) openFramePanel(f);

  if (map && mapReady) {
    const ring = f.geometry.coordinates[0];
    let [w, s2, e2, n] = [180, 90, -180, -90];
    for (const [x, y] of ring) {
      w = Math.min(w, x); e2 = Math.max(e2, x); s2 = Math.min(s2, y); n = Math.max(n, y);
    }
    map.fitBounds([[w, s2], [e2, n]], { padding: 120, duration: 600, maxZoom: 8 });
  }
}

// A one-row strip: with a handful of passes the question is where the gaps are
// and which sensor flew them, which a dot plot on a real time axis answers and a
// bar chart does not. Colour carries the sensor, but never alone -- every dot is
// labelled in the list below, which is the table view the contrast rule requires.
function renderTimeline(p) {
  const host = document.getElementById("ts");
  if (!p.dates.length) { host.innerHTML = ""; return; }

  const W = 298, H = 46, PAD = 10, Y = 17;
  const t = p.dates.map(d => new Date(d).getTime());
  const lo = Math.min(...t), hi = Math.max(...t);
  const span = Math.max(1, hi - lo);
  const x = v => PAD + ((v - lo) / span) * (W - 2 * PAD);

  // Several granules on one date land on the same x; fan them vertically so a
  // duplicate is visible as a stack rather than hiding under its twin.
  const perDate = {};
  const dots = t.map((v, i) => {
    const d = p.dates[i];
    const k = (perDate[d] = (perDate[d] || 0) + 1) - 1;
    const dup = p.duplicate_dates.includes(d);
    const cy = Y - k * 9;
    return `<circle cx="${x(v).toFixed(1)}" cy="${cy}" r="4.5"
       fill="${sensorColor(p.sensors[i])}"
       stroke="${dup ? cssVar("--dup") : cssVar("--surface-1")}" stroke-width="2">
       <title>${d} ${p.sensors[i]}${dup ? " (duplicate date)" : ""}</title></circle>`;
  }).join("");

  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Acquisition dates by sensor">
    <line class="gridline" x1="${PAD}" y1="${Y}" x2="${W - PAD}" y2="${Y}"></line>
    ${dots}
    <text class="axis" x="${PAD}" y="${H - 2}">${p.first}</text>
    <text class="axis" x="${W - PAD}" y="${H - 2}" text-anchor="end">${p.last}</text>
  </svg>
  <div class="legend" style="margin-top:2px;">${
    [...new Set(p.sensors)].sort().map(sname =>
      `<div><i style="background:${sensorColor(sname)}"></i>${sname}</div>`).join("")
  }</div>`;
}

/* --------------------------------------------------- frame time series --- */
// Clicking a frame opens its record in a floating panel over the map: draggable
// by its header, resizable from the corner, closed with the X. A panel rather
// than a second browser window, so it cannot be swallowed by a pop-up blocker
// and stays beside the map it came from.
// Which other frames cover this same ground. Beam is part of a frame's ID, so
// every acquisition inside one frame already shares a beam -- there is nothing to
// colour there. The question that does arise is the neighbouring one: the same
// island is usually imaged by other beams and other tracks too, and each of those
// is a SEPARATE stack. A different beam is a different look angle, a different
// track a different geometry; neither can be merged into one time series. The
// panel shows them as separate rows so that is impossible to miss.
const _bboxCache = new Map();
function ringBbox(f) {
  if (_bboxCache.has(f)) return _bboxCache.get(f);
  let [w, s2, e, n] = [180, 90, -180, -90];
  for (const [x, y] of f.geometry.coordinates[0]) {
    w = Math.min(w, x); e = Math.max(e, x); s2 = Math.min(s2, y); n = Math.max(n, y);
  }
  const b = [w, s2, e, n];
  _bboxCache.set(f, b);
  return b;
}

function overlappingGroups(feature) {
  const [w, s2, e, n] = ringBbox(feature);
  const groups = new Map();
  for (const other of FRAMES.features) {
    const p2 = other.properties;
    if (!p2.n_acquisitions) continue;
    const [w2, s3, e2, n2] = ringBbox(other);
    if (e2 < w || w2 > e || n2 < s2 || s3 > n) continue;
    const key = `${p2.beam} t${String(p2.track).padStart(3, "0")}`;
    if (!groups.has(key)) {
      groups.set(key, { key, beam: p2.beam, track: p2.track, frames: 0, acq: [] });
    }
    const g = groups.get(key);
    g.frames += 1;
    for (let i = 0; i < p2.dates.length; i++) {
      g.acq.push({ date: p2.dates[i], sensor: p2.sensors[i] });
    }
  }
  return [...groups.values()].sort((a, b) => b.acq.length - a.acq.length);
}

function stacksChart(feature) {
  const groups = overlappingGroups(feature);
  const own = `${feature.properties.beam} t${String(feature.properties.track).padStart(3, "0")}`;
  if (groups.length < 2) return "";

  const all = groups.flatMap(g => g.acq.map(a => new Date(a.date).getTime()));
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = Math.max(1, hi - lo);

  const W = 760, PAD_L = 92, PAD_R = 20, ROW = 22;
  const plotW = W - PAD_L - PAD_R;
  const x = v => PAD_L + ((v - lo) / span) * plotW;
  const H = groups.length * ROW + 26;

  const rows = groups.map((g, i) => {
    const y = 12 + i * ROW;
    const mine = g.key === own;
    const dots = g.acq.map(a =>
      `<circle cx="${x(new Date(a.date).getTime()).toFixed(1)}" cy="${y}" r="4.5"
         fill="${sensorColor(a.sensor)}" stroke="var(--surface-1)" stroke-width="1.5">
         <title>${g.key} - ${a.date} - ${a.sensor}</title></circle>`).join("");
    return `<line x1="${PAD_L}" y1="${y}" x2="${W - PAD_R}" y2="${y}"
              stroke="var(--border)" stroke-width="1"/>
            <text class="ax" x="${PAD_L - 8}" y="${y + 3}" text-anchor="end"
              style="fill:${mine ? "var(--text-1)" : "var(--text-2)"};
                     font-weight:${mine ? 650 : 400}">${g.key}${mine ? " *" : ""}</text>
            ${dots}`;
  }).join("");

  const first = new Date(lo).toISOString().slice(0, 10);
  const last = new Date(hi).toISOString().slice(0, 10);
  return `<h3>Same ground, other stacks</h3>
    <p class="note">Each row is a separate time series and they cannot be combined:
      a different beam images at a different look angle, and a different track from a
      different geometry. The row marked * is this frame. Dot colour is the sensor.</p>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Acquisitions by beam and track">
      ${rows}
      <text class="ax" x="${PAD_L}" y="${H - 4}">${first}</text>
      <text class="ax" x="${W - PAD_R}" y="${H - 4}" text-anchor="end">${last}</text>
    </svg>`;
}

function timeSeriesBody(feature) {
  const p = feature.properties;
  const W = 760, PAD_L = 46, PAD_R = 20;
  const plotW = W - PAD_L - PAD_R;
  const t = p.dates.map(d => new Date(d).getTime());
  const lo = Math.min(...t), hi = Math.max(...t);
  const span = Math.max(1, hi - lo);
  const x = v => PAD_L + ((v - lo) / span) * plotW;

  // --- when each pass happened, coloured by sensor ---
  const H1 = 112, BASE = 76;
  const perDate = {};
  const stems = t.map((v, i) => {
    const d = p.dates[i];
    const k = (perDate[d] = (perDate[d] || 0) + 1) - 1;
    const cy = BASE - 12 - k * 13;
    const dup = p.duplicate_dates.includes(d);
    return `<line x1="${x(v).toFixed(1)}" y1="${BASE}" x2="${x(v).toFixed(1)}" y2="${cy}"
              stroke="var(--border)" stroke-width="2"/>
            <circle cx="${x(v).toFixed(1)}" cy="${cy}" r="5.5"
              fill="${sensorColor(p.sensors[i])}"
              stroke="${dup ? "var(--dup)" : "var(--surface-1)"}" stroke-width="2">
              <title>${d} - ${p.sensors[i]}${dup ? " (duplicate date)" : ""}</title>
            </circle>`;
  }).join("");

  const months = new Set();
  const ticks = [];
  for (const d of p.dates) {
    const m = d.slice(0, 7);
    if (months.has(m)) continue;
    months.add(m);
    const v = new Date(m + "-01").getTime();
    if (v >= lo - 86400000 && v <= hi + 86400000) {
      ticks.push(`<line x1="${x(v).toFixed(1)}" y1="14" x2="${x(v).toFixed(1)}" y2="${BASE}"
                    stroke="var(--border)" stroke-dasharray="2 3"/>
                  <text class="ax" x="${x(v).toFixed(1)}" y="${BASE + 14}" text-anchor="middle">${m}</text>`);
    }
  }

  const chart1 = `<svg viewBox="0 0 ${W} ${H1}" role="img" aria-label="Acquisition dates by sensor">
    ${ticks.join("")}
    <line x1="${PAD_L}" y1="${BASE}" x2="${W - PAD_R}" y2="${BASE}" stroke="var(--border)"/>
    ${stems}
    <text class="ax" x="${PAD_L}" y="${BASE + 30}">${p.first}</text>
    <text class="ax" x="${W - PAD_R}" y="${BASE + 30}" text-anchor="end">${p.last}</text>
  </svg>`;

  // --- the gap between consecutive passes ---
  const uniq = [...new Set(p.dates)].sort();
  const gaps = uniq.slice(1).map((d, i) => ({
    to: d, days: Math.round((new Date(d) - new Date(uniq[i])) / 86400000)
  }));
  let chart2 = `<p class="note">Only one date, so there is no interval to plot.</p>`;
  if (gaps.length) {
    const H2 = 132, B2 = 98;
    const maxG = Math.max(...gaps.map(g => g.days));
    const step = plotW / gaps.length;
    const bw = Math.max(4, Math.min(30, step - 2));   // 2px surface gap between bars
    const bars = gaps.map((g, i) => {
      const cx = PAD_L + step * (i + 0.5);
      const h = (B2 - 22) * (g.days / maxG);
      return `<rect x="${(cx - bw / 2).toFixed(1)}" y="${(B2 - h).toFixed(1)}"
                width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="4" fill="var(--accent)">
                <title>${g.to}: ${g.days} days after the previous pass</title></rect>
              <text class="ax" x="${cx.toFixed(1)}" y="${(B2 - h - 5).toFixed(1)}"
                text-anchor="middle">${g.days}</text>`;
    }).join("");
    chart2 = `<svg viewBox="0 0 ${W} ${H2}" role="img" aria-label="Days between consecutive passes">
      <line x1="${PAD_L}" y1="${B2}" x2="${W - PAD_R}" y2="${B2}" stroke="var(--border)"/>
      <text class="ax" x="6" y="22">${maxG} d</text>
      <text class="ax" x="6" y="${B2}">0</text>
      ${bars}
      <text class="ax" x="${PAD_L}" y="${B2 + 20}">interval to each later pass, in days</text>
    </svg>`;
  }

  const legend = [...new Set(p.sensors)].sort().map(sn =>
    `<span class="lg"><i style="background:${sensorColor(sn)}"></i>${sn}</span>`).join("");

  const rows = p.dates.map((d, i) => {
    const gap = i ? Math.round((new Date(d) - new Date(p.dates[i - 1])) / 86400000) : null;
    const dup = p.duplicate_dates.includes(d);
    return `<tr class="${dup ? "dup" : ""}">
      <td><i class="sw" style="background:${sensorColor(p.sensors[i])}"></i>${d}</td>
      <td>${p.sensors[i]}</td>
      <td class="r">${gap == null ? "first" : "+" + gap + " d"}</td>
      <td class="g">${p.granules[i]}</td></tr>`;
  }).join("");

  const warn = p.n_duplicate
    ? `<div class="warn"><b>${p.n_duplicate} duplicate acquisition${p.n_duplicate === 1 ? "" : "s"}</b>
       on ${p.duplicate_dates.join(", ")}. More than one granule covers this frame on that date,
       so a stack built from it would carry a zero-baseline pair. Keep one granule per date.</div>`
    : "";

  const km = m => (m / 1000).toFixed(1);
  const spanDays = Math.round((hi - lo) / 86400000);

  return `
<div class="meta">track ${p.track} &middot; beam ${p.beam} &middot; frame ${p.frame_index}
  &middot; ${p.pass} &middot; EPSG ${p.epsg}</div>
<div class="pstats">
  <div class="stat"><div class="v">${p.n_acquisitions}</div><div class="k">acquisitions</div></div>
  <div class="stat"><div class="v">${p.n_dates}</div><div class="k">distinct dates</div></div>
  <div class="stat"><div class="v">${spanDays}</div><div class="k">days spanned</div></div>
  <div class="stat"><div class="v">${p.repeat_days == null ? "n/a" : p.repeat_days}</div><div class="k">median repeat (d)</div></div>
  <div class="stat"><div class="v">${p.platforms.length}</div><div class="k">sensors</div></div>
</div>
${warn}
<h3>Acquisitions</h3>
<div>${legend}</div>
${chart1}
<h3>Interval between passes</h3>
${chart2}
${stacksChart(feature)}
<h3>All acquisitions</h3>
<table class="acq"><thead><tr><th>Date</th><th>Sensor</th><th>Interval</th><th>Granule</th></tr></thead>
<tbody>${rows}</tbody></table>
<div class="grid">Pinned grid: ${km(p.width_m)} x ${km(p.height_m)} km &middot;
  footprint covers ${p.fill_pct}% of it, so ${p.nodata_pct}% is nodata in every product
  &middot; bbox ${p.bbox.join(" ")} in EPSG ${p.epsg}</div>`;
}

function openFramePanel(feature) {
  const p = feature.properties;
  if (!p.dates.length) return;
  const panel = document.getElementById("fpanel");
  panel.innerHTML =
    `<div id="fpanel-head">
       <span class="t">${p.frame_id}</span>
       <button id="fpanel-close" title="Close" aria-label="Close">&times;</button>
     </div>
     <div id="fpanel-body">${timeSeriesBody(feature)}</div>`;
  panel.hidden = false;
  document.getElementById("fpanel-close").onclick = () => { panel.hidden = true; };
  dragBy(panel, document.getElementById("fpanel-head"));
}

// Drag from the header. Positions are pinned in pixels on first drag so the
// panel stops following its initial `right`/`top` anchoring.
function dragBy(panel, handle) {
  handle.onmousedown = ev => {
    if (ev.target.id === "fpanel-close") return;
    const box = panel.getBoundingClientRect();
    const host = panel.parentElement.getBoundingClientRect();
    const dx = ev.clientX - box.left, dy = ev.clientY - box.top;
    panel.style.right = "auto";
    const move = e => {
      panel.style.left = Math.max(0, Math.min(e.clientX - host.left - dx,
        host.width - box.width)) + "px";
      panel.style.top = Math.max(0, Math.min(e.clientY - host.top - dy,
        host.height - 40)) + "px";
    };
    const up = () => {
      document.removeEventListener("mousemove", move);
      document.removeEventListener("mouseup", up);
    };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
    ev.preventDefault();
  };
}

/* ----------------------------------------------------------- controls --- */
function chipRow(hostId, values, selected, onToggle) {
  const host = document.getElementById(hostId);
  host.innerHTML = values
    .map(v => `<span class="chip ${selected.has(v) ? "on" : ""}" data-v="${v}">${v}</span>`)
    .join("");
  host.onclick = ev => {
    const chip = ev.target.closest(".chip");
    if (!chip) return;
    onToggle(chip.dataset.v);
    chipRow(hostId, values, selected, onToggle);
    paint();
  };
}

function toggle(set, v) { set.has(v) ? set.delete(v) : set.add(v); if (!set.size) set.add(v); }

chipRow("chips-beam", ALL_BEAMS, state.beams, v => toggle(state.beams, v));
chipRow("chips-pass", ALL_PASSES, state.passes, v => toggle(state.passes, v));

document.querySelectorAll(".section-head").forEach(h => {
  h.onclick = () => h.parentElement.classList.toggle("collapsed");
});

document.getElementById("min-acq").oninput = ev => {
  state.minAcq = +ev.target.value;
  document.getElementById("min-acq-v").textContent = state.minAcq;
  paint();
};
document.getElementById("min-fill").oninput = ev => {
  state.minFill = +ev.target.value;
  document.getElementById("min-fill-v").textContent = state.minFill;
  paint();
};
function currentOverride() {
  const id = state.selected;
  if (!id) return null;
  if (!state.overrides[id]) state.overrides[id] = { shiftKm: 0, growKm: 0, insetKm: 0 };
  return state.overrides[id];
}

function syncBoundsSliders() {
  const o = state.overrides[state.selected] || { shiftKm: 0, growKm: 0, insetKm: 0 };
  for (const [id, key] of BOUND_CONTROLS) {
    const slider = document.getElementById(id);
    slider.value = Math.max(+slider.min, Math.min(+slider.max, o[key]));
    document.getElementById(id + "-v").value = o[key];
  }
  updateCliLine();
}

const BOUND_CONTROLS = [
  ["g-shift", "shiftKm"], ["g-grow", "growKm"], ["g-inset", "insetKm"]
];

function applyBound(key, value) {
  const o = currentOverride();
  if (!o) return false;
  o[key] = Number.isFinite(value) ? value : 0;
  if (!o.shiftKm && !o.growKm && !o.insetKm) delete state.overrides[state.selected];
  updateCliLine();
  paint();
  return true;
}

for (const [id, key] of BOUND_CONTROLS) {
  const slider = document.getElementById(id);
  const box = document.getElementById(id + "-v");
  slider.oninput = ev => {
    if (!applyBound(key, +ev.target.value)) { ev.target.value = 0; return; }
    box.value = ev.target.value;
  };
  // The number box accepts values beyond the slider's ends; the slider then just
  // pins to its limit while the real value is whatever was typed.
  box.oninput = ev => {
    const v = parseFloat(ev.target.value);
    if (!applyBound(key, v)) { ev.target.value = 0; return; }
    slider.value = Math.max(+slider.min, Math.min(+slider.max, v || 0));
  };
}
document.getElementById("g-reset").onclick = () => {
  if (state.selected) delete state.overrides[state.selected];
  syncBoundsSliders();
  paint();
};
function mergeWith(step) {
  const id = state.selected;
  if (!id) return;
  const f = FRAMES.features.find(x => x.properties.frame_id === id);
  if (!f) return;
  const neighbour = neighbourId(f.properties, step);
  if (!FRAMES.features.some(x => x.properties.frame_id === neighbour)) return;

  // Growing an existing group keeps it one frame rather than making two.
  const existing = groupOf(id) || groupOf(neighbour);
  if (existing) {
    for (const candidate of [id, neighbour]) {
      if (!existing.includes(candidate)) existing.push(candidate);
    }
    existing.sort();
  } else {
    state.merges.push([id, neighbour].sort());
  }
  updateCliLine();
  paint();
}

document.getElementById("g-merge-prev").onclick = () => mergeWith(-1);
document.getElementById("g-merge-next").onclick = () => mergeWith(1);
document.getElementById("g-unmerge").onclick = () => {
  const group = state.selected && groupOf(state.selected);
  if (group) state.merges.splice(state.merges.indexOf(group), 1);
  updateCliLine();
  paint();
};
document.getElementById("g-export").onclick = exportOverrides;
document.getElementById("g-list").onclick = ev => {
  const b = ev.target.closest("button");
  if (b) select(b.dataset.ov, false);
};
document.getElementById("q-track").oninput = ev => { state.track = ev.target.value.trim(); paint(); };
document.getElementById("color-by").onchange = ev => { state.colorBy = ev.target.value; paint(); };
document.getElementById("op").oninput = ev => {
  state.opacity = +ev.target.value / 100;
  document.getElementById("op-v").textContent = ev.target.value;
  paint();
};
const panelToggle = document.getElementById("panel-toggle");
panelToggle.onclick = () => {
  state.autoPanel = !state.autoPanel;
  panelToggle.classList.toggle("on", state.autoPanel);
  panelToggle.title = state.autoPanel
    ? "Open the time series panel when a frame is clicked"
    : "Frame clicks only select; the panel stays closed";
  // Turning it off should also clear a panel that is already up.
  if (!state.autoPanel) document.getElementById("fpanel").hidden = true;
};
document.getElementById("show-grid").onchange = ev => { state.showGrid = ev.target.checked; paint(); };
document.getElementById("d0").onchange = ev => { state.d0 = ev.target.value || null; paint(); };
document.getElementById("d1").onchange = ev => { state.d1 = ev.target.value || null; paint(); };
document.getElementById("d-reset").onclick = () => {
  state.d0 = state.d1 = null;
  document.getElementById("d0").value = "";
  document.getElementById("d1").value = "";
  paint();
};
document.getElementById("f-reset").onclick = () => {
  state.beams = new Set(ALL_BEAMS);
  state.passes = new Set(ALL_PASSES);
  state.minAcq = 1; state.minFill = 0; state.track = "";
  document.getElementById("min-acq").value = 1;
  document.getElementById("min-acq-v").textContent = "1";
  document.getElementById("min-fill").value = 0;
  document.getElementById("min-fill-v").textContent = "0";
  document.getElementById("q-track").value = "";
  chipRow("chips-beam", ALL_BEAMS, state.beams, v => toggle(state.beams, v));
  chipRow("chips-pass", ALL_PASSES, state.passes, v => toggle(state.passes, v));
  paint(); fitAll();
};

document.querySelectorAll("#basemap button").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll("#basemap button").forEach(x => x.classList.remove("on"));
    b.classList.add("on");
    baseName = b.dataset.base;
    if (!map) return;
    map.setStyle(styleFor(baseName), { diff: false });
    map.once("style.load", () => { addLayers(); paint(); });
  };
});

document.getElementById("theme").onclick = () => {
  theme = theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  document.getElementById("theme").innerHTML = theme === "dark" ? "&#9789;" : "&#9788;";
  if (map && !IMAGERY.has(baseName)) {
    baseName = theme === "dark" ? "dark" : "light";
    document.querySelectorAll("#basemap button").forEach(x =>
      x.classList.toggle("on", x.dataset.base === baseName));
    map.setStyle(styleFor(baseName), { diff: false });
    map.once("style.load", () => { addLayers(); paint(); });
  } else {
    paint();
  }
};
document.getElementById("theme").innerHTML = theme === "dark" ? "&#9789;" : "&#9788;";

/* ---------------------------------------------------------------- search --- */
// Three kinds of query, cheapest first: coordinates and frame ids are answered
// from what is already in the page, and only a place name costs a network call.
//
// Nominatim is queried straight from the browser because this page is a static
// file with no server to proxy through. Their usage policy is one request a
// second, so searches fire on Enter rather than per keystroke, are spaced by a
// timer, and every answer is cached for the life of the page.
const NOMINATIM = "https://nominatim.openstreetmap.org/search";
const MIN_GAP_MS = 1100;
const placeCache = new Map();
let lastQueryAt = 0;

const qEl = document.getElementById("q");
const resEl = document.getElementById("results");

function hideResults() { resEl.hidden = true; resEl.innerHTML = ""; }

function showResults(html) { resEl.innerHTML = html; resEl.hidden = false; }

function flyTo(lon, lat, zoom) {
  if (!map || !mapReady) return false;
  map.flyTo({ center: [lon, lat], zoom: zoom == null ? 8 : zoom, duration: 1200 });
  return true;
}

function fitTo(bbox, lon, lat) {
  // Nominatim gives [south, north, west, east] as strings.
  if (!map || !mapReady) return false;
  const [s2, n, w, e] = bbox.map(Number);

  // Nominatim's box is not always fittable. A point result -- a street or an
  // islet -- comes back with south == north and west == east, and fitBounds on a
  // zero-area box computes no camera and silently does nothing, which is exactly
  // what "it offers the place but never flies there" looks like. A box crossing
  // the antimeridian arrives as west > east, and a whole-region box is so large
  // that fitting it is indistinguishable from not moving. Fly to the point in all
  // three cases.
  const width = e - w, height = n - s2;
  const degenerate = !(isFinite(width) && isFinite(height)) ||
                     width < 0.002 || height < 0.002;
  if (degenerate) return flyTo(lon, lat, 11);
  if (w > e) return flyTo(lon, lat, 5);
  if (width > 120 || height > 90) return flyTo(lon, lat, 4);

  map.fitBounds([[w, s2], [e, n]], { padding: 80, duration: 1200, maxZoom: 11 });
  return true;
}

function parseCoords(text) {
  const m = text.match(
    /^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (!m) return null;
  const a = parseFloat(m[1]), b = parseFloat(m[2]);
  // "lat lon" is how people write it; fall back to "lon lat" when the first
  // number is out of latitude range and the second is not.
  if (Math.abs(a) <= 90 && Math.abs(b) <= 180) return { lat: a, lon: b };
  if (Math.abs(b) <= 90 && Math.abs(a) <= 180) return { lat: b, lon: a };
  return null;
}

function matchFrames(text) {
  const q = text.trim().toLowerCase();
  if (!q) return [];
  return FRAMES.features
    .filter(f => f.properties.frame_id.includes(q))
    .slice(0, 8);
}

function frameCenter(f) {
  const ring = f.geometry.coordinates[0];
  let x = 0, y = 0;
  for (const c of ring) { x += c[0]; y += c[1]; }
  return [x / ring.length, y / ring.length];
}

async function runSearch(text) {
  const coords = parseCoords(text);
  if (coords) {
    hideResults();
    if (!flyTo(coords.lon, coords.lat, 9)) {
      showResults('<div class="msg">No map to fly to; coordinates need a working map.</div>');
    }
    return;
  }

  const frames = matchFrames(text);
  if (frames.length) {
    showResults(frames.map(f => {
      const p = f.properties;
      return `<button data-frame="${p.frame_id}">${p.frame_id}
        <small>track ${p.track} &middot; ${p.beam} &middot; ${p.n_acquisitions} pass${
          p.n_acquisitions === 1 ? "" : "es"}</small></button>`;
    }).join(""));
    return;
  }

  const key = text.trim().toLowerCase();
  if (placeCache.has(key)) { renderPlaces(placeCache.get(key)); return; }

  const wait = MIN_GAP_MS - (Date.now() - lastQueryAt);
  if (wait > 0) await new Promise(r => setTimeout(r, wait));
  lastQueryAt = Date.now();

  showResults('<div class="msg">Searching&hellip;</div>');
  try {
    const url = `${NOMINATIM}?format=jsonv2&limit=6&q=${encodeURIComponent(text)}`;
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Nominatim returned ${response.status}`);
    const places = await response.json();
    placeCache.set(key, places);
    renderPlaces(places);
  } catch (err) {
    showResults(`<div class="msg">Place lookup failed (${err.message}).
      Coordinates and frame ids still work offline.</div>`);
  }
}

function renderPlaces(places) {
  if (!places.length) {
    showResults('<div class="msg">Nothing found.</div>');
    return;
  }
  showResults(places.map((pl, i) =>
    `<button data-place="${i}">${(pl.name || pl.display_name).replace(/</g, "&lt;")}
      <small>${pl.display_name.replace(/</g, "&lt;")}</small></button>`
  ).join(""));
  resEl.__places = places;
}

resEl.onclick = ev => {
  const b = ev.target.closest("button");
  if (!b) return;
  if (b.dataset.frame) {
    hideResults();
    qEl.value = "";
    const f = FRAMES.features.find(x => x.properties.frame_id === b.dataset.frame);
    if (f) {
      const [lon, lat] = frameCenter(f);
      flyTo(lon, lat, 7);
      select(b.dataset.frame, true);
    }
    return;
  }
  const pl = (resEl.__places || [])[+b.dataset.place];
  if (!pl) return;
  const lon = parseFloat(pl.lon), lat = parseFloat(pl.lat);
  const moved = pl.boundingbox ? fitTo(pl.boundingbox, lon, lat) : flyTo(lon, lat);
  if (!moved) {
    // Never fail silently: without a map there is nowhere to fly to.
    showResults(`<div class="msg">${pl.display_name} is at
      ${lat.toFixed(4)}, ${lon.toFixed(4)} &mdash; but this browser has no working
      map to fly to.</div>`);
    return;
  }
  hideResults();
  qEl.value = pl.name || pl.display_name;
};

qEl.addEventListener("keydown", ev => {
  if (ev.key === "Enter") { ev.preventDefault(); runSearch(qEl.value); }
  if (ev.key === "Escape") { hideResults(); qEl.blur(); }
});
// Frame ids and coordinates are local, so answer those as the user types; a
// place name waits for Enter so Nominatim is not hit on every keystroke.
qEl.addEventListener("input", () => {
  const text = qEl.value;
  if (!text.trim()) { hideResults(); return; }
  if (parseCoords(text)) { hideResults(); return; }
  const frames = matchFrames(text);
  if (frames.length) { runSearch(text); } else { hideResults(); }
});

/* ----------------------------------------------------------------- start --- */
if (map) {
  map.on("error", e => console.error("maplibre:", e && e.error && e.error.message));
  map.on("load", () => { mapReady = true; addLayers(); paint(); fitAll(); });
} else {
  paint();
}
</script>
</body>
</html>
"""
