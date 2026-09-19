# sm_db

Frame database for Sentinel-1 **Stripmap** (S1-S6), so that every acquisition of
a frame geocodes onto one identical grid and the CSLCs stack.

## The problem

IW gets a stable frame grid for free. ESA defines a fixed along-track burst grid
locked to the ascending node, so a repeat pass lands on the same `burst_id`, and
`burst_db` pins a bounding box per burst that COMPASS geocodes into.

Stripmap has neither. The stripmap reader borrows the IW machinery and bins the
*mid-scene* sensing time into a 2.758273 s ESA burst interval, so the ID moves
whenever ESA slices a datatake differently -- a per-acquisition label wearing a
frame label's clothes. And with no database to pin a box, COMPASS derives the
geogrid from each acquisition's own footprint, so no two dates share a grid.

`sm_db` supplies the missing half: a deterministic frame ID and a pinned
bounding box, written in the sqlite schema COMPASS already reads.

## How a frame is defined

A frame is a fixed slice of the orbit. Quantize time since the ascending node
into tiles and number them per relative orbit:

```
frame index = 1 + floor((t_anx - T_PRE) / tile_seconds)
frame ID    = t{track:03d}_{index:06d}_{beam}      e.g. t095_000003_s3
```

The tile boundaries depend only on the orbit, so every repeat pass over the same
ground lands in the same tile. Three points are load-bearing:

**The ID keeps the IW six-digit layout** (`t095_000003_s3`, not `t095_f00003_s3`)
so the OPERA granule name, `compass_batch.planning`'s internal-ID regex and
`S1BurstId.from_str` all keep working. Only the meaning of the middle field
changes.

**A tile is claimed only when a scene covers it completely**, with a one-second
guard because ASF rounds scene times to whole seconds. That keeps a stack valid
edge to edge -- no date contributes a partial row. It also means `tile_seconds`
must be well under a scene length: at the 5 s default a nominal 20 s S3 slice
yields 3-4 frames, where a 20 s tile would usually yield *zero*.

**The ANX is the true crossing before the scene**, taken from the orbit. ESA's
annotated `ascendingNodeTime` is often a full revolution stale, and the usual
correction subtracts a *nominal* period, landing about a second off -- a fifth of
a tile. See the "Which ANX" section of `sm_db/tiling.py`.

Frame geometry comes from the orbit too, not from interpolating the granule
footprint between its reported times: those are rounded to the second, which on a
20 s scene moved boxes by up to 4.3 km between dates that in truth repeat to
within 40 m. The footprint is used only for the cross-track swath extent, which
that rounding does not affect.

## What it writes

`burst_id_map`, the table COMPASS reads through
`compass.utils.helpers.burst_bboxes_from_db`:

| column | |
|---|---|
| `burst_id_jpl` | the frame ID |
| `epsg` | UTM, or 3413/3031 near the poles |
| `xmin, ymin, xmax, ymax` | projected metres, 5 km margin, snapped to 30 m |

plus a `frames` table (track, index, beam, footprint) and a `metadata` table
recording the build parameters. Nothing else, so `compass_batch` needs no change:
pass the file as `--burst-db` and the worker, the DEM widening and the runconfig
all pick it up.

## Install

```bash
conda env create -f environment.yml && conda activate sm-db-env
pip install -e '.[test,update]'
```

`update` pulls in `sentineleof` for downloading orbits; reading an existing
database does not need it.

## Use

```bash
# Build from a date range. Orbits must be on hand (or use `update`, below).
sm-db build --start 2026-02-25 --end 2026-04-10 --beam S3 --track 95 \
    --orbit-dir orbits --catalog catalog.json -o sm_frames.sqlite3

# Which frames does this acquisition yield? The call a planner makes.
sm-db frames-for-granule S1C_S3_SLC__1SDV_20260313T044134_... \
    --orbit-dir orbits --catalog catalog.json

sm-db lookup t095_000003_s3 -d sm_frames.sqlite3
sm-db intersect --bbox -158 1 -156 3 -d sm_frames.sqlite3

# Re-derive every frame from every granule and report disagreement.
sm-db check --orbit-dir orbits --catalog catalog.json -d sm_frames.sqlite3

# Extend the catalog, fetch orbits, rebuild the database and the map.
# This is what the scheduled job runs.
sm-db update

sm-db viewer -d sm_frames.sqlite3 --catalog catalog.json --orbit-dir orbits
```

## The tracked archive

`catalog/` holds the committed archive, refreshed daily by
[`update_sm_catalog.yml`](.github/workflows/update_sm_catalog.yml):

| file | |
|---|---|
| `sm_granules.json` | every stripmap acquisition found, with footprint and timing |
| `sm_frames.sqlite3` | the frame database COMPASS reads |
| `sm_frames.geojson` | frame footprints for a GIS |
| `docs/frame-viewer.html` | the map, coloured by acquisitions per frame |

Because a stripmap campaign is tasked by hand, coverage is patchy and the map is
the quickest way to see what can actually carry a time series. Rebuilds are safe:
`merge_frames` keeps the first definition of every frame, so adding dates never
moves a grid that products already sit on.

## Tests

```bash
pytest          # offline: the ASF query uses an injected fake, the orbit is synthetic
pre-commit install && pre-commit run -a
```

## Related

- [`burst_db`](https://github.com/opera-adt/burst_db) -- the IW equivalent, whose
  EPSG and bbox-snapping conventions this follows.
- `s1-reader` branch `feature/stripmap` -- the stripmap reader, which must stamp
  the same frame IDs this database is keyed by.
- `cslc-batch` -- runs COMPASS on AWS Batch and consumes the database.
