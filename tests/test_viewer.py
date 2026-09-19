"""Tests for the frame map."""

from __future__ import annotations

import json
import re

import pytest
from shapely.geometry import box

from sm_db.frames import Frame
from sm_db.viewer import (
    acquisition_counts,
    duplicate_dates,
    frame_acquisitions,
    frames_to_geojson,
    grids_to_geojson,
    write_viewer,
)


def _frame(frame_id="t095_000003_s3", index=3, beam="s3"):
    return Frame(
        frame_id=frame_id,
        track=95,
        index=index,
        beam=beam,
        epsg=32604,
        xmin=657240,
        ymin=183120,
        xmax=753690,
        ymax=243360,
        polygon=box(-157.5, 1.8, -156.8, 2.2),
    )


class TestAcquisitionCounts:
    def test_counts_each_frame_once_per_granule(self):
        counts = acquisition_counts(
            [["a", "b"], ["b", "c"], ["b"]],
        )
        assert counts == {"a": 1, "b": 3, "c": 1}

    def test_empty_input(self):
        assert acquisition_counts([]) == {}


class TestFramesToGeojson:
    def test_is_a_feature_collection(self):
        data = frames_to_geojson([_frame()])
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1

    def test_carries_the_pinned_grid(self):
        (feature,) = frames_to_geojson([_frame()])["features"]
        props = feature["properties"]
        assert props["frame_id"] == "t095_000003_s3"
        assert props["epsg"] == 32604
        assert props["bbox"] == [657240, 183120, 753690, 243360]
        assert props["width_m"] == 753690 - 657240

    def test_counts_default_to_zero(self):
        (feature,) = frames_to_geojson([_frame()])["features"]
        assert feature["properties"]["n_acquisitions"] == 0

    def test_acquisitions_are_attached_by_frame_id(self):
        acq = {
            "t095_000003_s3": [
                {"date": "2026-03-13", "granule": "g", "platform": "S1C"}
            ]
        }
        (feature,) = frames_to_geojson([_frame()], acq)["features"]
        assert feature["properties"]["n_acquisitions"] == 1
        assert feature["properties"]["dates"] == ["2026-03-13"]

    def test_features_are_ordered_by_frame_id(self):
        data = frames_to_geojson(
            [_frame("t095_000005_s3", 5), _frame("t095_000003_s3", 3)]
        )
        assert [f["properties"]["frame_id"] for f in data["features"]] == [
            "t095_000003_s3",
            "t095_000005_s3",
        ]


class TestWriteViewer:
    @pytest.fixture
    def page(self, tmp_path):
        path = write_viewer(
            [_frame(), _frame("t096_000007_s1", 7, "s1")],
            tmp_path / "viewer.html",
            acquisitions={
                "t095_000003_s3": [
                    {"date": "2026-03-01", "granule": "a", "platform": "S1C"},
                    {"date": "2026-03-13", "granule": "b", "platform": "S1C"},
                ]
            },
        )
        return path.read_text()

    def test_data_is_inlined_as_valid_json(self, page):
        """The page has to stand alone, so the GeoJSON is embedded, not fetched."""
        match = re.search(r"const FRAMES = (\{.*?\});\n", page, re.S)
        assert match
        data = json.loads(match.group(1))
        assert len(data["features"]) == 2

    def test_no_placeholder_survives(self, page):
        assert "__FRAMES__" not in page
        assert "__GRIDS__" not in page
        assert "__TITLE__" not in page
        assert "__SUBTITLE__" not in page

    def test_basemaps_need_no_api_key(self, page):
        """Keyless sources only, so a committed copy keeps working."""
        for keyed in ("cartocdn", "api_key", "access_token", "mapbox", "{apikey}"):
            assert keyed not in page

    def test_creates_the_parent_directory(self, tmp_path):
        path = write_viewer([_frame()], tmp_path / "nested" / "deep" / "viewer.html")
        assert path.exists()

    def test_subtitle_defaults_to_a_frame_count(self, tmp_path):
        page = write_viewer([_frame()], tmp_path / "v.html").read_text()
        assert "1 frames" in page

    def test_subtitle_is_used_when_given(self, tmp_path):
        page = write_viewer(
            [_frame()], tmp_path / "v.html", subtitle="custom line"
        ).read_text()
        assert "custom line" in page


class TestFrameAcquisitions:
    def test_groups_by_frame_and_sorts_by_date(self, granule):
        import dataclasses
        import datetime

        later = dataclasses.replace(
            granule, name="S1C_later", start=granule.start + datetime.timedelta(days=12)
        )
        grouped = frame_acquisitions(
            [
                (later, ["t095_000003_s3"]),
                (granule, ["t095_000003_s3", "t095_000004_s3"]),
            ]
        )
        assert [a["date"] for a in grouped["t095_000003_s3"]] == [
            granule.start.strftime("%Y-%m-%d"),
            later.start.strftime("%Y-%m-%d"),
        ]
        assert len(grouped["t095_000004_s3"]) == 1

    def test_records_the_platform(self, granule):
        grouped = frame_acquisitions([(granule, ["t095_000003_s3"])])
        assert grouped["t095_000003_s3"][0]["platform"] == "S1C"

    def test_empty_input(self):
        assert frame_acquisitions([]) == {}


class TestRepeatInterval:
    def _props(self, dates):
        acq = {
            "t095_000003_s3": [
                {"date": d, "granule": "g", "platform": "S1C"} for d in dates
            ]
        }
        (feature,) = frames_to_geojson([_frame()], acq)["features"]
        return feature["properties"]

    def test_median_gap_between_passes(self):
        props = self._props(["2026-03-01", "2026-03-13", "2026-03-25"])
        assert props["repeat_days"] == 12.0

    def test_single_pass_has_no_interval(self):
        """One acquisition cannot define a repeat, and must not report zero."""
        assert self._props(["2026-03-01"])["repeat_days"] is None

    def test_first_and_last_span_the_record(self):
        props = self._props(["2026-03-25", "2026-03-01"])
        assert (props["first"], props["last"]) == ("2026-03-01", "2026-03-25")


class TestGridsToGeojson:
    def test_one_feature_per_frame_keyed_by_id(self):
        data = grids_to_geojson([_frame()])
        (feature,) = data["features"]
        assert feature["properties"]["frame_id"] == "t095_000003_s3"

    def test_pinned_grid_is_larger_than_the_footprint(self):
        """The bbox is the footprint's envelope plus a margin, so it must contain it."""
        frame = _frame()
        grid = grids_to_geojson([frame])["features"][0]
        from shapely.geometry import shape

        assert shape(grid["geometry"]).area > frame.polygon.area


class TestDuplicates:
    def _props(self, entries):
        acq = {
            "t095_000003_s3": [
                {"date": d, "granule": g, "platform": p} for d, g, p in entries
            ]
        }
        (feature,) = frames_to_geojson([_frame()], acq)["features"]
        return feature["properties"]

    def test_none_when_one_granule_per_date(self):
        props = self._props([("2026-03-01", "a", "S1C"), ("2026-03-13", "b", "S1C")])
        assert props["n_duplicate"] == 0
        assert props["duplicate_dates"] == []

    def test_two_granules_on_one_date_is_a_duplicate(self):
        """Overlapping slices of one datatake reach a frame twice on the same day."""
        props = self._props([("2026-03-01", "a", "S1C"), ("2026-03-01", "b", "S1C")])
        assert props["n_duplicate"] == 1
        assert props["duplicate_dates"] == ["2026-03-01"]

    def test_distinct_dates_excludes_the_duplicate(self):
        props = self._props(
            [
                ("2026-03-01", "a", "S1C"),
                ("2026-03-01", "b", "S1C"),
                ("2026-03-13", "c", "S1C"),
            ]
        )
        assert (props["n_acquisitions"], props["n_dates"]) == (3, 2)

    def test_duplicates_do_not_create_a_zero_repeat(self):
        """A repeated date must not be read as a zero-day revisit."""
        props = self._props(
            [
                ("2026-03-01", "a", "S1C"),
                ("2026-03-01", "b", "S1C"),
                ("2026-03-13", "c", "S1C"),
            ]
        )
        assert props["repeat_days"] == 12.0

    def test_duplicate_dates_helper_is_ordered(self):
        acq = [
            {"date": "2026-03-13", "granule": "c", "platform": "S1C"},
            {"date": "2026-03-13", "granule": "d", "platform": "S1C"},
            {"date": "2026-03-01", "granule": "a", "platform": "S1C"},
            {"date": "2026-03-01", "granule": "b", "platform": "S1C"},
        ]
        assert duplicate_dates(acq) == ["2026-03-01", "2026-03-13"]


class TestSensors:
    def test_sensor_runs_parallel_to_dates(self):
        """The timeline indexes `sensors` by the same position as `dates`."""
        acq = {
            "t095_000003_s3": [
                {"date": "2026-03-13", "granule": "b", "platform": "S1A"},
                {"date": "2026-03-01", "granule": "a", "platform": "S1C"},
            ]
        }
        (feature,) = frames_to_geojson([_frame()], acq)["features"]
        props = feature["properties"]
        assert props["dates"] == ["2026-03-01", "2026-03-13"]
        assert props["sensors"] == ["S1C", "S1A"]
        assert props["granules"] == ["a", "b"]

    def test_platforms_is_the_distinct_set(self):
        acq = {
            "t095_000003_s3": [
                {"date": "2026-03-01", "granule": "a", "platform": "S1C"},
                {"date": "2026-03-13", "granule": "b", "platform": "S1C"},
                {"date": "2026-03-25", "granule": "c", "platform": "S1A"},
            ]
        }
        (feature,) = frames_to_geojson([_frame()], acq)["features"]
        assert feature["properties"]["platforms"] == ["S1A", "S1C"]

    def test_page_defines_a_colour_for_every_sentinel(self, tmp_path):
        page = write_viewer([_frame()], tmp_path / "v.html").read_text()
        for token in ("--s1a", "--s1b", "--s1c", "--s1d"):
            assert token in page


class TestPageSize:
    def _acq(self, n):
        return [
            {
                "date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}",
                "granule": (
                    f"S1A_S6_SLC__1SDV_2026{i:04d}T115411_"
                    f"2026{i:04d}T115437_00_00_AA"
                ),
                "platform": "S1A",
            }
            for i in range(n)
        ]

    def test_sparse_frames_keep_granule_names(self):
        acq = {"t095_000003_s3": self._acq(5)}
        (feature,) = frames_to_geojson([_frame()], acq)["features"]
        assert len(feature["properties"]["granules"]) == 5

    def test_dense_frames_drop_them(self):
        """Granule names dominated the page: 4.4 MB of an 8.9 MB archive build."""
        acq = {"t095_000003_s3": self._acq(200)}
        (feature,) = frames_to_geojson([_frame()], acq)["features"]
        assert "granules" not in feature["properties"]
        # The chronology itself must survive; only the names go.
        assert len(feature["properties"]["dates"]) == 200
        assert len(feature["properties"]["sensors"]) == 200

    def test_coordinates_are_rounded(self):
        (feature,) = frames_to_geojson([_frame()])["features"]
        for x, y in feature["geometry"]["coordinates"][0]:
            assert round(x, 5) == x and round(y, 5) == y

    def test_grid_coordinates_are_rounded(self):
        (feature,) = grids_to_geojson([_frame()])["features"]
        for x, y in feature["geometry"]["coordinates"][0]:
            assert round(x, 5) == x and round(y, 5) == y

    def test_page_explains_the_missing_names(self, tmp_path):
        """The table still renders; the page says why the column is blank."""
        page = write_viewer(
            [_frame()],
            tmp_path / "v.html",
            acquisitions={"t095_000003_s3": self._acq(200)},
        ).read_text()
        data = json.loads(
            re.search(r"const FRAMES = (\{.*?\});\n", page, re.S).group(1)
        )
        assert "granules" not in data["features"][0]["properties"]
        assert "Granule names are left out" in page
        # The table cell falls back to a blank, never to a literal "undefined".
        assert '(p.granules || [])[i] || ""' in page
