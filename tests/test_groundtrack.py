"""Tests for placing a frame from the orbit."""

from __future__ import annotations

import datetime

import pytest
from pyproj import Transformer
from shapely.ops import transform

from sm_db.groundtrack import GroundTrack, swath_offsets, tile_polygon

from .conftest import ANX

EPSG = 32604


@pytest.fixture
def track(orbit):
    return GroundTrack(orbit, EPSG, ANX, ANX + datetime.timedelta(seconds=60))


def _project(polygon):
    tf = Transformer.from_crs(4326, EPSG, always_xy=True)
    return transform(lambda x, y: tf.transform(x, y), polygon)


class TestGroundTrack:
    def test_rejects_a_window_the_orbit_barely_covers(self, orbit):
        late = orbit.times[-1] + datetime.timedelta(days=1)
        with pytest.raises(ValueError, match="fewer than two state vectors"):
            GroundTrack(orbit, EPSG, late, late + datetime.timedelta(seconds=10))

    def test_position_advances_along_the_track(self, track):
        first = track.position(ANX + datetime.timedelta(seconds=10))
        later = track.position(ANX + datetime.timedelta(seconds=30))
        assert first != later

    def test_heading_is_a_unit_vector(self, track):
        dx, dy = track.heading(ANX + datetime.timedelta(seconds=20))
        assert (dx**2 + dy**2) == pytest.approx(1.0, abs=1e-6)

    def test_projecting_a_point_on_the_track_gives_back_its_time(self, track):
        when = ANX + datetime.timedelta(seconds=20)
        x, y = track.position(when)
        found, offset = track.project(x, y)
        assert abs((found - when).total_seconds()) < 1.0
        assert offset == pytest.approx(0.0, abs=100.0)

    def test_offset_sign_distinguishes_the_two_sides(self, track):
        when = ANX + datetime.timedelta(seconds=20)
        x, y = track.position(when)
        dx, dy = track.heading(when)
        nx, ny = -dy, dx  # left normal

        left = track.project(x + 50_000 * nx, y + 50_000 * ny)[1]
        right = track.project(x - 50_000 * nx, y - 50_000 * ny)[1]
        assert left > 0 > right
        assert left == pytest.approx(-right, rel=0.05)


class TestSwathOffsets:
    def test_brackets_the_footprint(self, track, granule):
        near, far = swath_offsets(track, _project(granule.footprint))
        assert near < far

    def test_width_matches_a_stripmap_swath(self, track, granule):
        """S3 images a roughly 80 km swath, so the offsets should span about that."""
        near, far = swath_offsets(track, _project(granule.footprint))
        assert 50_000 < (far - near) < 120_000


class TestTilePolygon:
    def test_is_a_closed_quadrilateral(self, track):
        polygon = tile_polygon(
            track,
            ANX + datetime.timedelta(seconds=10),
            ANX + datetime.timedelta(seconds=15),
            -40_000.0,
            40_000.0,
        )
        assert polygon.is_valid
        assert len(polygon.exterior.coords) == 5

    def test_longer_tiles_cover_more_ground(self, track):
        short = tile_polygon(
            track,
            ANX + datetime.timedelta(seconds=10),
            ANX + datetime.timedelta(seconds=15),
            -40_000.0,
            40_000.0,
        )
        long = tile_polygon(
            track,
            ANX + datetime.timedelta(seconds=10),
            ANX + datetime.timedelta(seconds=25),
            -40_000.0,
            40_000.0,
        )
        assert long.area > short.area

    def test_adjacent_tiles_abut_without_overlapping(self, track):
        """Tiles partition the track, so neighbours must touch but not overlap."""
        first = tile_polygon(
            track,
            ANX + datetime.timedelta(seconds=10),
            ANX + datetime.timedelta(seconds=15),
            -40_000.0,
            40_000.0,
        )
        second = tile_polygon(
            track,
            ANX + datetime.timedelta(seconds=15),
            ANX + datetime.timedelta(seconds=20),
            -40_000.0,
            40_000.0,
        )
        assert first.intersection(second).area == pytest.approx(0.0, abs=1.0)
        assert first.touches(second) or first.intersects(second)
