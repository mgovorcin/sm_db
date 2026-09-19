"""Tests for the frame ID specification."""

from __future__ import annotations

import pytest

from sm_db.tiling import (
    DEFAULT_TILE_SECONDS,
    T_PRE,
    Tile,
    format_frame_id,
    frame_index,
    next_track,
    parse_frame_id,
    tile_bounds,
    tiles_covered_by,
)


class TestFrameIndex:
    def test_first_tile_starts_at_preamble(self):
        assert frame_index(T_PRE) == 1

    @pytest.mark.parametrize(
        "offset, expected", [(0.0, 1), (4.99, 1), (5.0, 2), (10.0, 3), (49.9, 10)]
    )
    def test_index_advances_one_per_tile(self, offset, expected):
        assert frame_index(T_PRE + offset, tile_seconds=5.0) == expected

    def test_index_and_bounds_are_inverses(self):
        for index in (1, 7, 1185):
            start, stop = tile_bounds(index, 5.0)
            assert frame_index(start, 5.0) == index
            # The stop belongs to the next tile: bounds are half-open.
            assert frame_index(stop, 5.0) == index + 1

    def test_rejects_non_positive_tile_length(self):
        with pytest.raises(ValueError, match="tile_seconds must be positive"):
            frame_index(10.0, tile_seconds=0.0)


class TestFrameId:
    def test_round_trip(self):
        assert parse_frame_id(format_frame_id(95, 3, "S3")) == (95, 3, "s3")

    def test_matches_iw_burst_id_layout(self):
        """Six digits, so `compass_batch.planning`'s internal-id regex accepts it."""
        assert format_frame_id(95, 3, "S3") == "t095_000003_s3"

    @pytest.mark.parametrize("beam", ["iw1", "s7", "s0", "ew1", ""])
    def test_rejects_non_stripmap_beam(self, beam):
        with pytest.raises(ValueError, match="Not a stripmap beam mode"):
            format_frame_id(95, 3, beam)

    def test_rejects_index_too_wide_for_the_field(self):
        with pytest.raises(ValueError, match="out of range for six digits"):
            format_frame_id(95, 1_000_000, "s3")

    @pytest.mark.parametrize(
        "bad", ["t095_000003_iw2", "t95_3_s3", "t095_000003", "nonsense"]
    )
    def test_parse_rejects_malformed(self, bad):
        with pytest.raises(ValueError, match="Not a stripmap frame ID"):
            parse_frame_id(bad)


class TestNextTrack:
    def test_increments(self):
        assert next_track(95) == 96

    def test_wraps_at_the_end_of_the_cycle(self):
        assert next_track(175) == 1


class TestTilesCoveredBy:
    def test_aligned_scene_fills_whole_tiles(self):
        tiles = tiles_covered_by(T_PRE, T_PRE + 20.0, 5.0)
        assert [t.index for t in tiles] == [1, 2, 3, 4]

    def test_partial_end_tiles_are_dropped(self):
        """A frame is claimed only when the scene fills it edge to edge."""
        tiles = tiles_covered_by(T_PRE + 1.0, T_PRE + 21.0, 5.0)
        assert [t.index for t in tiles] == [2, 3, 4]

    def test_scene_shorter_than_a_tile_claims_nothing(self):
        assert tiles_covered_by(T_PRE + 1.0, T_PRE + 3.0, 5.0) == []

    def test_tile_bounds_are_contiguous(self):
        tiles = tiles_covered_by(T_PRE, T_PRE + 20.0, 5.0)
        for earlier, later in zip(tiles, tiles[1:], strict=False):
            assert earlier.stop == pytest.approx(later.start)

    def test_orbit_period_caps_the_last_tile(self):
        """Past the node the track changes, so tiles must not run over it."""
        uncapped = tiles_covered_by(T_PRE, T_PRE + 20.0, 5.0)
        capped = tiles_covered_by(T_PRE, T_PRE + 20.0, 5.0, orbit_period=T_PRE + 12.0)
        assert [t.index for t in uncapped] == [1, 2, 3, 4]
        assert [t.index for t in capped] == [1, 2]

    def test_rejects_scene_that_ends_before_it_starts(self):
        with pytest.raises(ValueError, match="Scene stops before it starts"):
            tiles_covered_by(100.0, 50.0, 5.0)

    def test_tile_mid_is_the_centre(self):
        assert Tile(1, 10.0, 15.0).mid == 12.5

    def test_default_tile_is_short_enough_for_a_nominal_scene(self):
        """A 20 s stripmap slice must yield several frames at the default length.

        A tile comparable to the scene would usually straddle both ends and yield
        nothing at all, which is the trap the default guards against.
        """
        tiles = tiles_covered_by(10.5, 30.5, DEFAULT_TILE_SECONDS)
        assert len(tiles) >= 3
