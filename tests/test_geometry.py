"""Tests for projection choice and bounding box snapping."""

from __future__ import annotations

import pytest

from sm_db.geometry import (
    DEFAULT_MARGIN,
    DEFAULT_SNAP,
    pick_epsg,
    snap_bbox,
    wrap_lon,
)


class TestPickEpsg:
    @pytest.mark.parametrize(
        "lon, lat, expected",
        [
            (-118.0, 34.0, 32611),
            (-157.0, 2.0, 32604),
            (15.0, -34.0, 32733),
            (0.5, 0.5, 32631),
        ],
    )
    def test_utm_zone(self, lon, lat, expected):
        assert pick_epsg(lon, lat) == expected

    def test_north_pole_uses_polar_stereographic(self):
        assert pick_epsg(-45.0, 80.0) == 3413

    def test_south_pole_uses_polar_stereographic(self):
        assert pick_epsg(-45.0, -70.0) == 3031

    @pytest.mark.parametrize(
        "lon, lat, expected",
        [(6.0, 60.0, 32632), (15.0, 78.0, 3413)],
    )
    def test_irregular_zones(self, lon, lat, expected):
        """Norway widens zone 32; Svalbard is past the polar threshold anyway."""
        assert pick_epsg(lon, lat) == expected

    def test_south_of_equator_uses_the_southern_band(self):
        assert pick_epsg(-157.0, -2.0) == 32704

    def test_longitude_outside_the_principal_range_is_wrapped(self):
        assert pick_epsg(-177.0, 2.0) == pick_epsg(183.0, 2.0)


class TestWrapLon:
    @pytest.mark.parametrize(
        "raw, expected", [(181.0, -179.0), (-181.0, 179.0), (0.0, 0.0), (179.0, 179.0)]
    )
    def test_wraps_into_principal_range(self, raw, expected):
        assert wrap_lon(raw) == pytest.approx(expected)


class TestSnapBbox:
    def test_already_aligned_is_unchanged(self):
        assert snap_bbox(1000.0, 2000.0, 3000.0, 4000.0, margin=0.0, snap=100.0) == (
            1000,
            2000,
            3000,
            4000,
        )

    def test_snaps_outward_never_inward(self):
        """Floor the mins and ceil the maxes, so the box always contains its input."""
        xmin, ymin, xmax, ymax = snap_bbox(
            1001.0, 2001.0, 2999.0, 3999.0, margin=0.0, snap=100.0
        )
        assert (xmin, ymin) == (1000, 2000)
        assert (xmax, ymax) == (3000, 4000)

    def test_margin_pushes_every_side_out(self):
        assert snap_bbox(1000.0, 1000.0, 2000.0, 2000.0, margin=500.0, snap=10.0) == (
            500,
            500,
            2500,
            2500,
        )

    def test_returns_integers(self):
        """COMPASS reads these columns as integers."""
        assert all(isinstance(v, int) for v in snap_bbox(1.5, 2.5, 3.5, 4.5))

    def test_defaults_match_burst_db(self):
        assert (DEFAULT_MARGIN, DEFAULT_SNAP) == (5000.0, 30.0)
