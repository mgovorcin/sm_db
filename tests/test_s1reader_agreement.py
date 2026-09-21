"""s1-reader carries its own copy of the tiling; it must match this one.

The reader stamps frame IDs on Stripmap scenes (``load_stripmap_burst(...,
frame_tile_seconds=...)``) without depending on sm_db, so the arithmetic lives
in both places. A frame whose ID the reader computes differently would find no
grid in the database, or the wrong one. Skipped where s1-reader is not
installed, which includes CI.
"""

from __future__ import annotations

import datetime
import random

import pytest

from sm_db import tiling

reader = pytest.importorskip("s1reader.s1_stripmap_frames")

ANX = datetime.datetime(2026, 3, 13, 4, 41, 13)


def test_constants_agree():
    assert reader.T_PRE == tiling.T_PRE
    assert reader.DEFAULT_TILE_SECONDS == tiling.DEFAULT_TILE_SECONDS


@pytest.mark.parametrize("tile_seconds", [2.5, 5.0, 10.0])
def test_same_tiles_for_random_scenes(tile_seconds):
    rng = random.Random(0)
    for _ in range(2000):
        start = rng.uniform(0.0, 5900.0)
        stop = start + rng.uniform(0.0, 60.0)
        ours = [
            (t.index, t.start, t.stop)
            for t in tiling.tiles_covered_by(start, stop, tile_seconds)
        ]
        theirs = [
            (t.index, t.start, t.stop)
            for t in reader.tiles_covered_by(95, start, stop, tile_seconds)
        ]
        assert ours == theirs, (start, stop)


def test_scene_across_the_node_restarts_on_the_next_track():
    period = 5924.0
    tiles = reader.frames_for_scene(
        175,
        ANX + datetime.timedelta(seconds=period - 12.0),
        ANX + datetime.timedelta(seconds=period + 16.0),
        ANX,
        ANX + datetime.timedelta(seconds=period),
        5.0,
    )
    before = tiling.tiles_covered_by(period - 12.0, period + 16.0, 5.0, period)
    after = tiling.tiles_covered_by(0.0, 16.0, 5.0)
    assert [(t.track, t.index) for t in tiles] == [(175, t.index) for t in before] + [
        (tiling.next_track(175), t.index) for t in after
    ]
