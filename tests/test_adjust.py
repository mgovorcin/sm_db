"""Tests for turning a hand-edited frame into a reproducible adjustment."""

from __future__ import annotations

import pytest
from shapely.affinity import scale, translate

from sm_db import tiling
from sm_db.adjust import (
    BOUNDARY_SNAP,
    Window,
    along_track_window,
    snap_to_boundary,
    window_to_offsets,
)
from sm_db.frames import frames_for_granule


class TestSnapToBoundary:
    def test_pulls_a_near_miss_onto_the_boundary(self):
        """A hand-placed vertex lands a few tens of metres off the boundary."""
        boundary = tiling.T_PRE + 10.0
        assert snap_to_boundary(boundary + BOUNDARY_SNAP / 2, 5.0) == boundary

    def test_leaves_a_deliberate_extension_alone(self):
        t = tiling.T_PRE + 11.3
        assert snap_to_boundary(t, 5.0) == round(t, 2)

    def test_works_below_a_boundary_too(self):
        boundary = tiling.T_PRE + 10.0
        assert snap_to_boundary(boundary - BOUNDARY_SNAP / 2, 5.0) == boundary


class TestWindowToOffsets:
    def test_unchanged_tile_needs_nothing(self):
        t0, t1 = tiling.tile_bounds(3, 5.0)
        assert window_to_offsets(3, Window(t0, t1), 5.0) == (0.0, 0.0)

    def test_extending_one_end_is_half_shift_half_overlap(self):
        t0, t1 = tiling.tile_bounds(3, 5.0)
        assert window_to_offsets(3, Window(t0, t1 + 5.0), 5.0) == (2.5, 2.5)

    @pytest.mark.parametrize("start_off, stop_off", [(-3, 0), (0, 7), (-2, 4), (1, -1)])
    def test_round_trips_through_the_build_formula(self, start_off, stop_off):
        """Any window must come back exactly from tile -/+ overlap + shift."""
        t0, t1 = tiling.tile_bounds(7, 5.0)
        window = Window(t0 + start_off, t1 + stop_off)
        shift, overlap = window_to_offsets(7, window, 5.0)
        assert t0 - overlap + shift == pytest.approx(window.start, abs=1e-3)
        assert t1 + overlap + shift == pytest.approx(window.stop, abs=1e-3)


class TestAlongTrackWindow:
    def _frame_and_granule(self, granule, orbit):
        frame = frames_for_granule(granule, orbit)[0]
        return frame, granule

    def test_an_untouched_frame_reads_back_as_its_own_tile(self, granule, orbit):
        frame, g = self._frame_and_granule(granule, orbit)
        window = along_track_window(frame.polygon, g, orbit, frame.epsg, frame.index)
        t0, t1 = tiling.tile_bounds(frame.index)
        assert window.start == pytest.approx(t0, abs=0.05)
        assert window.stop == pytest.approx(t1, abs=0.05)

    def test_stretching_the_frame_lengthens_the_window(self, granule, orbit):
        frame, g = self._frame_and_granule(granule, orbit)
        # Stretch along the frame's long axis, roughly north-south here.
        stretched = scale(frame.polygon, xfact=1.0, yfact=1.8, origin="centroid")
        window = along_track_window(stretched, g, orbit, frame.epsg, frame.index)
        assert window.length > tiling.DEFAULT_TILE_SECONDS

    def test_reports_a_side_dragged_off_the_swath(self, granule, orbit):
        """Sides are discarded, but a stray one is reported rather than hidden."""
        frame, g = self._frame_and_granule(granule, orbit)
        dragged = translate(frame.polygon, xoff=0.02)  # ~2 km sideways
        window = along_track_window(dragged, g, orbit, frame.epsg, frame.index)
        assert max(abs(window.west_deviation), abs(window.east_deviation)) > 500


class TestDrops:
    def test_a_dropped_frame_is_never_emitted(self, granule, orbit):
        """It must stay gone when new data arrives, or the next run revives it."""
        every = [f.frame_id for f in frames_for_granule(granule, orbit)]
        gone = every[0]
        kept = [f.frame_id for f in frames_for_granule(granule, orbit, drops={gone})]
        assert gone not in kept
        assert kept == every[1:]

    def test_no_drops_changes_nothing(self, granule, orbit):
        assert [
            f.frame_id for f in frames_for_granule(granule, orbit, drops=set())
        ] == [f.frame_id for f in frames_for_granule(granule, orbit)]
