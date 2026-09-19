"""Tests for turning acquisitions into frames."""

from __future__ import annotations

import dataclasses
import datetime

import pytest

from sm_db.frames import Frame, OrbitLookup, frames_for_granule, merge_frames
from sm_db.tiling import parse_frame_id

from .conftest import ANX


class TestFramesForGranule:
    def test_a_twenty_second_scene_yields_several_frames(self, granule, orbit):
        frames = frames_for_granule(granule, orbit)
        assert len(frames) >= 2

    def test_frame_ids_carry_the_granule_track_and_beam(self, granule, orbit):
        for frame in frames_for_granule(granule, orbit):
            track, _, beam = parse_frame_id(frame.frame_id)
            assert (track, beam) == (95, "s3")

    def test_indices_are_consecutive_along_track(self, granule, orbit):
        indices = [f.index for f in frames_for_granule(granule, orbit)]
        assert indices == list(range(indices[0], indices[0] + len(indices)))

    def test_repeat_passes_produce_identical_frame_ids(self, granule, orbit):
        """The whole point: a later pass over the same ground stacks on the same grid.

        The repeat is offset by a fraction of a second, as the real S1C track-95
        campaign is, which must not move the frame.
        """
        first = frames_for_granule(granule, orbit)

        shifted = dataclasses.replace(
            granule,
            start=granule.start + datetime.timedelta(milliseconds=300),
            stop=granule.stop + datetime.timedelta(milliseconds=300),
        )
        second = frames_for_granule(shifted, orbit)

        assert [f.frame_id for f in first] == [f.frame_id for f in second]

    def test_repeat_passes_produce_the_same_grid(self, granule, orbit):
        first = {f.frame_id: f for f in frames_for_granule(granule, orbit)}
        shifted = dataclasses.replace(
            granule,
            start=granule.start + datetime.timedelta(milliseconds=300),
            stop=granule.stop + datetime.timedelta(milliseconds=300),
        )
        for frame in frames_for_granule(shifted, orbit):
            assert frame.epsg == first[frame.frame_id].epsg
            assert frame.bbox == first[frame.frame_id].bbox

    def test_guard_absorbs_asf_second_rounding(self, granule, orbit):
        """ASF rounds scene times to whole seconds; that must not change the frames.

        Without the guard, a tile boundary within a second of the scene edge flips
        in and out as the reported time rounds up or down.
        """
        baseline = [f.frame_id for f in frames_for_granule(granule, orbit)]
        for delta in (-0.9, -0.4, 0.4, 0.9):
            nudged = dataclasses.replace(
                granule,
                start=granule.start + datetime.timedelta(seconds=delta),
                stop=granule.stop + datetime.timedelta(seconds=delta),
            )
            assert [f.frame_id for f in frames_for_granule(nudged, orbit)] == baseline

    def test_scene_too_short_to_fill_a_tile_yields_nothing(self, granule, orbit):
        short = dataclasses.replace(
            granule, stop=granule.start + datetime.timedelta(seconds=3)
        )
        assert frames_for_granule(short, orbit) == []

    def test_bbox_contains_the_frame_polygon(self, granule, orbit):
        from pyproj import Transformer
        from shapely.ops import transform

        def to_map(polygon, epsg):
            tf = Transformer.from_crs(4326, epsg, always_xy=True)
            return transform(lambda x, y: tf.transform(x, y), polygon)

        for frame in frames_for_granule(granule, orbit):
            projected = to_map(frame.polygon, frame.epsg)
            xmin, ymin, xmax, ymax = projected.bounds
            assert frame.xmin <= xmin and frame.ymin <= ymin
            assert frame.xmax >= xmax and frame.ymax >= ymax

    def test_longer_tiles_yield_fewer_frames(self, granule, orbit):
        few = frames_for_granule(granule, orbit, tile_seconds=10.0)
        many = frames_for_granule(granule, orbit, tile_seconds=5.0)
        assert len(few) < len(many)

    def test_frames_do_not_run_past_the_next_node(self, granule, orbit):
        """Beyond the node the track changes, so the index must restart."""
        crossing = dataclasses.replace(
            granule,
            start=ANX - datetime.timedelta(seconds=10),
            stop=ANX + datetime.timedelta(seconds=10),
        )
        tracks = {
            parse_frame_id(f.frame_id)[0] for f in frames_for_granule(crossing, orbit)
        }
        assert tracks <= {95, 96}


class TestMergeFrames:
    def _frame(self, frame_id: str, xmin: int) -> Frame:
        from shapely.geometry import box

        return Frame(
            frame_id=frame_id,
            track=95,
            index=3,
            beam="s3",
            epsg=32604,
            xmin=xmin,
            ymin=0,
            xmax=xmin + 1000,
            ymax=1000,
            polygon=box(0, 0, 1, 1),
        )

    def test_adds_new_frames(self):
        merged = merge_frames({}, [self._frame("t095_000003_s3", 0)])
        assert set(merged) == {"t095_000003_s3"}

    def test_keeps_the_first_definition(self):
        """A grid is frozen once written, so adding dates never moves it."""
        first = self._frame("t095_000003_s3", 0)
        merged = merge_frames(
            {first.frame_id: first}, [self._frame("t095_000003_s3", 999)]
        )
        assert merged["t095_000003_s3"].xmin == 0

    def test_does_not_mutate_the_input(self):
        existing: dict[str, Frame] = {}
        merge_frames(existing, [self._frame("t095_000003_s3", 0)])
        assert existing == {}


class TestOrbitLookup:
    def _touch(self, directory, name):
        (directory / name).write_text(
            "<Earth_Explorer_File><Data_Block><List_of_OSVs>"
            "<OSV><UTC>UTC=2026-03-13T04:41:23.000000</UTC>"
            "<X>1.0</X><Y>2.0</Y><Z>3.0</Z></OSV>"
            "</List_of_OSVs></Data_Block></Earth_Explorer_File>"
        )

    def test_finds_the_covering_orbit(self, tmp_path, granule):
        self._touch(
            tmp_path,
            "S1C_OPER_AUX_POEORB_OPOD_20260402T070836_V20260312T225942_20260314T005942.EOF",
        )
        assert len(OrbitLookup(tmp_path).find(granule)) == 1

    def test_rejects_an_orbit_that_starts_at_the_scene(self, tmp_path, granule):
        """The node can be a full revolution back, so lead-in is required."""
        self._touch(
            tmp_path,
            "S1C_OPER_AUX_POEORB_OPOD_20260402T070836_V20260313T044000_20260314T005942.EOF",
        )
        with pytest.raises(FileNotFoundError, match="No orbit file"):
            OrbitLookup(tmp_path).find(granule)

    def test_ignores_another_platform(self, tmp_path, granule):
        self._touch(
            tmp_path,
            "S1A_OPER_AUX_POEORB_OPOD_20260402T070836_V20260312T225942_20260314T005942.EOF",
        )
        with pytest.raises(FileNotFoundError, match="No orbit file"):
            OrbitLookup(tmp_path).find(granule)

    def test_empty_directory_names_the_granule(self, tmp_path, granule):
        with pytest.raises(FileNotFoundError, match=granule.name):
            OrbitLookup(tmp_path).find(granule)


class TestFillFraction:
    def test_reported_for_every_frame(self, granule, orbit):
        for frame in frames_for_granule(granule, orbit):
            assert 0 < frame.fill_pct <= 100

    def test_footprint_is_smaller_than_its_box(self, granule, orbit):
        """The box is the envelope of a rotated quad plus a margin, so never full."""
        for frame in frames_for_granule(granule, orbit):
            assert frame.fill_pct < 100

    def test_fill_and_nodata_are_complements(self, granule, orbit):
        frame = frames_for_granule(granule, orbit)[0]
        assert frame.fill_pct + frame.nodata_pct == pytest.approx(100.0, abs=0.1)

    def test_matches_the_measured_area_ratio(self, granule, orbit):
        from pyproj import Transformer
        from shapely.ops import transform

        frame = frames_for_granule(granule, orbit)[0]
        tf = Transformer.from_crs(4326, frame.epsg, always_xy=True)
        projected = transform(lambda x, y: tf.transform(x, y), frame.polygon)
        box_area = (frame.xmax - frame.xmin) * (frame.ymax - frame.ymin)
        assert frame.fill_pct == pytest.approx(100 * projected.area / box_area, abs=0.2)

    def test_a_bigger_margin_lowers_the_fill(self, granule, orbit):
        """More padding is more nodata, by definition."""
        tight = frames_for_granule(granule, orbit, margin=0.0)[0]
        padded = frames_for_granule(granule, orbit, margin=20_000.0)[0]
        assert padded.fill_pct < tight.fill_pct


class TestOrbitCache:
    def _orbit_file(self, directory, name):
        (directory / name).write_text(
            "<Earth_Explorer_File><Data_Block><List_of_OSVs>"
            "<OSV><UTC>UTC=2026-03-13T04:41:23.000000</UTC>"
            "<X>1.0</X><Y>2.0</Y><Z>3.0</Z></OSV>"
            "</List_of_OSVs></Data_Block></Earth_Explorer_File>"
        )

    def test_caches_a_repeated_lookup(self, tmp_path, granule):
        self._orbit_file(
            tmp_path,
            "S1C_OPER_AUX_POEORB_OPOD_20260402T070836_V20260312T225942_20260314T005942.EOF",
        )
        lookup = OrbitLookup(tmp_path)
        assert lookup.find(granule) is lookup.find(granule)

    def test_does_not_grow_without_bound(self, tmp_path, granule):
        """A multi-year archive would otherwise hold gigabytes of state vectors."""
        import dataclasses
        import datetime

        for day in range(1, 9):
            self._orbit_file(
                tmp_path,
                f"S1C_OPER_AUX_POEORB_OPOD_20260402T070836_"
                f"V202603{day:02d}T000000_202604{day:02d}T000000.EOF",
            )
        lookup = OrbitLookup(tmp_path, cache_size=2)
        for day in range(1, 9):
            moved = dataclasses.replace(
                granule,
                start=granule.start + datetime.timedelta(days=day),
                stop=granule.stop + datetime.timedelta(days=day),
            )
            lookup.find(moved)
        assert len(lookup._cache) <= 2


class TestPerFrameOverrides:
    def test_moves_only_the_named_frame(self, granule, orbit):
        """A boundary cutting one island is that frame's problem, not the archive's."""
        base = {f.frame_id: f for f in frames_for_granule(granule, orbit)}
        target = sorted(base)[1]

        moved = {
            f.frame_id: f
            for f in frames_for_granule(
                granule, orbit, overrides={target: {"shift": 2.0}}
            )
        }
        assert moved[target].bbox != base[target].bbox
        for frame_id in base:
            if frame_id != target:
                assert moved[frame_id].bbox == base[frame_id].bbox

    def test_records_what_it_was_built_with(self, granule, orbit):
        target = sorted(f.frame_id for f in frames_for_granule(granule, orbit))[0]
        frames = frames_for_granule(
            granule, orbit, overrides={target: {"shift": 1.5, "inset": 3000}}
        )
        got = {f.frame_id: f for f in frames}[target]
        assert (got.shift, got.inset) == (1.5, 3000.0)

    def test_an_override_beats_the_global_setting(self, granule, orbit):
        target = sorted(f.frame_id for f in frames_for_granule(granule, orbit))[0]
        frames = frames_for_granule(
            granule, orbit, shift=5.0, overrides={target: {"shift": 0.0}}
        )
        by_id = {f.frame_id: f for f in frames}
        assert by_id[target].shift == 0.0
        assert all(f.shift == 5.0 for k, f in by_id.items() if k != target)

    def test_frame_ids_are_unchanged_by_an_override(self, granule, orbit):
        """Geometry moves; the ID must not, or products would be renamed."""
        before = sorted(f.frame_id for f in frames_for_granule(granule, orbit))
        after = sorted(
            f.frame_id
            for f in frames_for_granule(
                granule, orbit, overrides={before[0]: {"shift": 2.0, "overlap": 1.0}}
            )
        )
        assert before == after


class TestMergedFrames:
    def _ids(self, granule, orbit, **kw):
        return sorted(f.frame_id for f in frames_for_granule(granule, orbit, **kw))

    def test_two_frames_become_one(self, granule, orbit):
        plain = self._ids(granule, orbit)
        group = plain[:2]
        merged = self._ids(granule, orbit, merges=[group])
        assert group[0] in merged
        assert group[1] not in merged
        assert len(merged) == len(plain) - 1

    def test_merged_frame_spans_both(self, granule, orbit):
        plain = {f.frame_id: f for f in frames_for_granule(granule, orbit)}
        group = sorted(plain)[:2]
        merged = {
            f.frame_id: f for f in frames_for_granule(granule, orbit, merges=[group])
        }
        one = merged[group[0]]
        # The union must reach at least as far as either part did on its own.
        for part in group:
            assert one.xmin <= plain[part].xmin and one.ymin <= plain[part].ymin
            assert one.xmax >= plain[part].xmax and one.ymax >= plain[part].ymax

    def test_keeps_the_first_members_id(self, granule, orbit):
        group = self._ids(granule, orbit)[:2]
        merged = self._ids(granule, orbit, merges=[group])
        assert group[0] in merged

    def test_group_the_scene_cannot_fill_is_left_alone(self, granule, orbit):
        """A merged frame must be filled edge to edge like any other."""
        plain = self._ids(granule, orbit)
        absent = "t095_999999_s3"
        merged = self._ids(granule, orbit, merges=[[plain[0], absent]])
        assert merged == plain

    def test_non_consecutive_group_is_ignored(self, granule, orbit):
        plain = self._ids(granule, orbit)
        if len(plain) < 3:
            pytest.skip("scene yields too few frames to test a gap")
        merged = self._ids(granule, orbit, merges=[[plain[0], plain[2]]])
        assert merged == plain

    def test_no_merges_changes_nothing(self, granule, orbit):
        assert self._ids(granule, orbit, merges=[]) == self._ids(granule, orbit)


class TestNodeCrossingEdgeCase:
    def test_scene_ending_just_past_the_node_does_not_raise(self, granule, orbit):
        """Regression: a scene crossing the ANX by less than the guard.

        The segment after the crossing starts before its own node, so the start
        is clamped to zero while the stop stays negative. Comparing before
        clamping let that pair through and `tiles_covered_by` rejected it.
        """
        import dataclasses
        import datetime

        from .conftest import ANX

        crossing = dataclasses.replace(
            granule,
            start=ANX - datetime.timedelta(seconds=20),
            stop=ANX + datetime.timedelta(milliseconds=13),
        )
        frames_for_granule(crossing, orbit)  # must not raise

    def test_scene_ending_exactly_at_the_node(self, granule, orbit):
        import dataclasses
        import datetime

        from .conftest import ANX

        crossing = dataclasses.replace(
            granule, start=ANX - datetime.timedelta(seconds=25), stop=ANX
        )
        for frame in frames_for_granule(crossing, orbit):
            assert frame.track == granule.track  # nothing attributed past the node


class TestBboxOverride:
    def test_edited_bbox_wins_over_the_orbit(self, granule, orbit):
        """A GIS edit says where a frame is, not how to derive it."""
        target = sorted(f.frame_id for f in frames_for_granule(granule, orbit))[0]
        wanted = [100_000, 200_000, 180_000, 260_000]
        frames = frames_for_granule(
            granule, orbit, overrides={target: {"bbox": wanted, "epsg": 32604}}
        )
        got = {f.frame_id: f for f in frames}[target]
        assert list(got.bbox) == wanted
        assert got.epsg == 32604

    def test_polygon_follows_the_edited_box(self, granule, orbit):
        target = sorted(f.frame_id for f in frames_for_granule(granule, orbit))[0]
        frames = frames_for_granule(
            granule,
            orbit,
            overrides={
                target: {"bbox": [100_000, 200_000, 180_000, 260_000], "epsg": 32604}
            },
        )
        got = {f.frame_id: f for f in frames}[target]
        assert got.fill_pct == 100.0
        assert len(got.polygon.exterior.coords) == 5

    def test_other_frames_are_untouched(self, granule, orbit):
        base = {f.frame_id: f for f in frames_for_granule(granule, orbit)}
        target = sorted(base)[0]
        frames = frames_for_granule(
            granule,
            orbit,
            overrides={target: {"bbox": [1, 2, 3, 4], "epsg": 32604}},
        )
        for f in frames:
            if f.frame_id != target:
                assert f.bbox == base[f.frame_id].bbox
