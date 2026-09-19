"""Tests for ascending node detection."""

from __future__ import annotations

import datetime

import pytest

from sm_db.anx import (
    T_ORBIT,
    ascending_node_time,
    ascending_node_times,
    read_orbit_file,
    time_since_anx,
)

from .conftest import ANX


class TestAscendingNodeTime:
    def test_finds_the_constructed_node(self, orbit):
        """The synthetic orbit crosses Z upward exactly at ANX."""
        found = ascending_node_time(orbit, ANX + datetime.timedelta(seconds=10))
        assert abs((found - ANX).total_seconds()) < 0.01

    def test_returns_the_preceding_node_not_the_following_one(self, orbit):
        """The frame index is measured from the node the scene is *after*."""
        just_before = ANX - datetime.timedelta(seconds=5)
        found = ascending_node_time(orbit, just_before)
        assert found < just_before
        # The node one revolution before ANX, not ANX itself.
        assert (ANX - found).total_seconds() == pytest.approx(T_ORBIT, abs=1.0)

    def test_time_since_anx_is_small_for_a_scene_just_after_the_node(self, orbit):
        sensing = ANX + datetime.timedelta(seconds=10)
        assert time_since_anx(
            sensing, ascending_node_time(orbit, sensing)
        ) == pytest.approx(10.0, abs=0.01)

    def test_raises_when_the_orbit_does_not_reach_back_to_a_node(self, orbit):
        too_early = orbit.times[0] + datetime.timedelta(seconds=5)
        with pytest.raises(ValueError, match="No ascending node crossing before"):
            ascending_node_time(orbit, too_early)


class TestAscendingNodeTimes:
    def test_consecutive_nodes_are_one_period_apart(self, orbit):
        nodes = ascending_node_times(
            orbit, ANX - datetime.timedelta(seconds=2 * T_ORBIT), orbit.times[-1]
        )
        assert len(nodes) >= 2
        for earlier, later in zip(nodes, nodes[1:], strict=False):
            assert (later - earlier).total_seconds() == pytest.approx(T_ORBIT, abs=1.0)

    def test_empty_outside_any_crossing(self, orbit):
        window_start = ANX + datetime.timedelta(seconds=60)
        nodes = ascending_node_times(
            orbit, window_start, window_start + datetime.timedelta(seconds=120)
        )
        assert nodes == []


class TestReadOrbitFile:
    def test_rejects_a_file_with_no_state_vectors(self, tmp_path):
        path = tmp_path / "empty.EOF"
        path.write_text("<Earth_Explorer_File></Earth_Explorer_File>")
        with pytest.raises(ValueError, match="No orbit state vectors"):
            read_orbit_file(path)

    def test_reads_positions_and_times(self, tmp_path):
        path = tmp_path / "one.EOF"
        path.write_text(
            "<Earth_Explorer_File><Data_Block><List_of_OSVs>"
            "<OSV><UTC>UTC=2026-03-13T04:41:23.000000</UTC>"
            "<X>1.0</X><Y>2.0</Y><Z>3.0</Z></OSV>"
            "</List_of_OSVs></Data_Block></Earth_Explorer_File>"
        )
        orbit = read_orbit_file(path)
        assert len(orbit) == 1
        assert orbit.times[0] == datetime.datetime(2026, 3, 13, 4, 41, 23)
        assert (orbit.x[0], orbit.y[0], orbit.z[0]) == (1.0, 2.0, 3.0)
