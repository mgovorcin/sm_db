"""Tests for the ``sm-db`` command line interface."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from shapely.geometry import box

from sm_db.cli import cli
from sm_db.db import write_database
from sm_db.frames import Frame


def _frame(frame_id="t095_000003_s3", index=3):
    return Frame(
        frame_id=frame_id,
        track=95,
        index=index,
        beam="s3",
        epsg=32604,
        xmin=657240,
        ymin=183120,
        xmax=753690,
        ymax=243360,
        polygon=box(-157.5, 1.8, -156.8, 2.2),
    )


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "frames.sqlite3"
    write_database([_frame()], path, tile_seconds=5.0, margin=5000.0, snap=30.0)
    return path


class TestHelp:
    @pytest.mark.parametrize(
        "command",
        ["build", "lookup", "frames-for-granule", "intersect", "check"],
    )
    def test_every_subcommand_documents_itself(self, runner, command):
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_group_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0


class TestLookup:
    def test_prints_the_pinned_grid(self, runner, database):
        result = runner.invoke(cli, ["lookup", "t095_000003_s3", "-d", str(database)])
        assert result.exit_code == 0
        assert "32604" in result.output
        assert "657240 183120 753690 243360" in result.output

    def test_unknown_frame_is_an_error(self, runner, database):
        result = runner.invoke(cli, ["lookup", "t095_000099_s3", "-d", str(database)])
        assert result.exit_code != 0
        assert "not in" in result.output

    def test_malformed_frame_id_is_rejected_before_the_query(self, runner, database):
        result = runner.invoke(cli, ["lookup", "not-a-frame", "-d", str(database)])
        assert result.exit_code != 0


class TestIntersect:
    def test_lists_frames_over_the_area(self, runner, database):
        result = runner.invoke(
            cli, ["intersect", "--bbox", "-158", "1", "-156", "3", "-d", str(database)]
        )
        assert result.exit_code == 0
        assert "t095_000003_s3" in result.output

    def test_quiet_when_nothing_intersects(self, runner, database):
        result = runner.invoke(
            cli, ["intersect", "--bbox", "10", "10", "11", "11", "-d", str(database)]
        )
        assert result.exit_code == 0
        assert result.output.strip() == ""


class TestBuild:
    def test_empty_catalog_is_an_error(self, runner, tmp_path):
        catalog = tmp_path / "catalog.json"
        catalog.write_text("[]")
        orbits = tmp_path / "orbits"
        orbits.mkdir()

        result = runner.invoke(
            cli,
            [
                "build",
                "--start",
                "2026-02-01",
                "--end",
                "2026-03-01",
                "--orbit-dir",
                str(orbits),
                "--catalog",
                str(catalog),
                "-o",
                str(tmp_path / "out.sqlite3"),
            ],
        )
        assert result.exit_code != 0
        assert "No granules matched" in result.output

    def test_missing_orbit_is_reported_not_swallowed(self, runner, tmp_path, granule):
        from sm_db.granules import save_catalog

        catalog = tmp_path / "catalog.json"
        save_catalog([granule], catalog)
        orbits = tmp_path / "orbits"
        orbits.mkdir()

        result = runner.invoke(
            cli,
            [
                "build",
                "--start",
                "2026-02-01",
                "--end",
                "2026-04-01",
                "--orbit-dir",
                str(orbits),
                "--catalog",
                str(catalog),
                "-o",
                str(tmp_path / "out.sqlite3"),
            ],
        )
        assert "skipped" in result.output
        assert "No orbit file" in result.output


class TestGeojson:
    def test_written_with_frame_properties(self, runner, tmp_path):
        from sm_db.cli import _write_geojson

        path = tmp_path / "frames.geojson"
        _write_geojson([_frame()], path)

        data = json.loads(path.read_text())
        assert data["type"] == "FeatureCollection"
        (feature,) = data["features"]
        assert feature["properties"]["frame_id"] == "t095_000003_s3"
        assert feature["properties"]["epsg"] == 32604
        assert feature["properties"]["bbox"] == [657240, 183120, 753690, 243360]


class TestAdjustmentsFile:
    def _load(self, tmp_path, name, payload):
        import json

        from sm_db.cli import _load_adjustments

        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return _load_adjustments(path, None)

    def test_reads_the_viewers_combined_export(self, tmp_path):
        overrides, merges = self._load(
            tmp_path,
            "adj.json",
            {
                "overrides": {"t095_000003_s3": {"shift": 1.2}},
                "merges": [["t095_000004_s3", "t095_000005_s3"]],
            },
        )
        assert overrides == {"t095_000003_s3": {"shift": 1.2}}
        assert merges == [["t095_000004_s3", "t095_000005_s3"]]

    def test_bare_mapping_is_overrides(self, tmp_path):
        overrides, merges = self._load(
            tmp_path, "o.json", {"t095_000003_s3": {"inset": 2000}}
        )
        assert overrides == {"t095_000003_s3": {"inset": 2000}}
        assert merges == []

    def test_bare_list_is_merges(self, tmp_path):
        overrides, merges = self._load(tmp_path, "m.json", [["a", "b"]])
        assert overrides == {}
        assert merges == [["a", "b"]]

    def test_nothing_given(self):
        from sm_db.cli import _load_adjustments

        assert _load_adjustments(None, None) == ({}, [])


class TestIncrementalUpdate:
    """A weekly job must not re-derive twelve years of frames to add a week.

    Re-deriving needs every orbit file ever used -- 24 GB for the real archive,
    more than a CI cache holds -- to arrive back at frames that are frozen anyway.
    """

    def _db_with(self, tmp_path, granule, frame_id="t095_000003_s3"):
        from shapely.geometry import box

        path = tmp_path / "frames.sqlite3"
        write_database(
            [_frame(frame_id)],
            path,
            tile_seconds=5.0,
            margin=5000.0,
            snap=30.0,
            acquisitions={
                frame_id: [
                    {
                        "date": granule.start.strftime("%Y-%m-%d"),
                        "platform": "S1C",
                        "granule": granule.name,
                    }
                ]
            },
        )
        assert box  # keep the import meaningful for the fixture shape
        return path

    def test_already_seen_granules_are_not_reprocessed(self, tmp_path, granule):
        from sm_db.db import read_acquisitions

        path = self._db_with(tmp_path, granule)
        seen = {
            a["granule"]
            for entries in read_acquisitions(path).values()
            for a in entries
        }
        assert granule.name in seen

    def test_acquisitions_accumulate_rather_than_replace(self, tmp_path, granule):
        """The database grows across runs; it is not rebuilt from the last one."""
        import dataclasses
        import datetime

        from sm_db.db import read_acquisitions

        path = self._db_with(tmp_path, granule)
        later = dataclasses.replace(
            granule,
            name="S1C_later",
            start=granule.start + datetime.timedelta(days=12),
        )
        existing = read_acquisitions(path)
        merged = dict(existing)
        merged["t095_000003_s3"] = existing["t095_000003_s3"] + [
            {
                "date": later.start.strftime("%Y-%m-%d"),
                "platform": "S1C",
                "granule": later.name,
            }
        ]
        write_database(
            [_frame()],
            path,
            tile_seconds=5.0,
            margin=5000.0,
            snap=30.0,
            acquisitions=merged,
        )
        assert len(read_acquisitions(path)["t095_000003_s3"]) == 2


class TestCheckWithoutOrbits:
    """A missing orbit means "not checked", never "broken".

    Regression: a CI run caches orbits only for new acquisitions, so `check`
    counted 18,874 unavailable orbits as problems and failed the whole job.
    """

    def _setup(self, tmp_path, granule):
        from sm_db.granules import save_catalog

        catalog = tmp_path / "catalog.json"
        save_catalog([granule], catalog)
        orbits = tmp_path / "orbits"
        orbits.mkdir()
        database = tmp_path / "frames.sqlite3"
        write_database([_frame()], database, tile_seconds=5.0, margin=5000.0, snap=30.0)
        return catalog, orbits, database

    def test_missing_orbit_does_not_fail_the_run(self, runner, tmp_path, granule):
        catalog, orbits, database = self._setup(tmp_path, granule)
        result = runner.invoke(
            runner_cli := cli,
            [
                "check",
                "--catalog",
                str(catalog),
                "--orbit-dir",
                str(orbits),
                "-d",
                str(database),
            ],
        )
        assert runner_cli is cli
        assert result.exit_code == 0
        assert "could not be checked" in result.output

    def test_it_says_how_many_went_unverified(self, runner, tmp_path, granule):
        catalog, orbits, database = self._setup(tmp_path, granule)
        result = runner.invoke(
            cli,
            [
                "check",
                "--catalog",
                str(catalog),
                "--orbit-dir",
                str(orbits),
                "-d",
                str(database),
            ],
        )
        assert "1 granule(s) could not be checked" in result.output
        assert "0 problem(s)" in result.output


class TestLoadDrops:
    def test_reads_the_drops_list(self, tmp_path):
        import json

        from sm_db.cli import _load_drops

        path = tmp_path / "adj.json"
        path.write_text(json.dumps({"overrides": {}, "drops": ["a", "b"]}))
        assert _load_drops(path) == {"a", "b"}

    def test_absent_list_is_empty(self, tmp_path):
        import json

        from sm_db.cli import _load_drops

        path = tmp_path / "adj.json"
        path.write_text(json.dumps({"overrides": {}}))
        assert _load_drops(path) == set()

    def test_bare_list_carries_no_drops(self, tmp_path):
        """A bare list is merge groups; it must not be misread as drops."""
        import json

        from sm_db.cli import _load_drops

        path = tmp_path / "m.json"
        path.write_text(json.dumps([["a", "b"]]))
        assert _load_drops(path) == set()

    def test_nothing_given(self):
        from sm_db.cli import _load_drops

        assert _load_drops(None, None) == set()


class TestStaleFrames:
    """The weekly job passes the same adjustments every run; only changes count."""

    def _frame(self, frame_id, shift=0.0, overlap=0.0, inset=0.0):
        import dataclasses

        return dataclasses.replace(
            _frame(frame_id), shift=shift, overlap=overlap, inset=inset
        )

    def test_an_already_applied_adjustment_is_not_redone(self):
        from sm_db.cli import _stale_frames

        existing = {"t095_000003_s3": self._frame("t095_000003_s3", shift=1.2)}
        assert (
            _stale_frames(existing, {"t095_000003_s3": {"shift": 1.2}}, [], set())
            == set()
        )

    def test_a_changed_adjustment_is_redone(self):
        from sm_db.cli import _stale_frames

        existing = {"t095_000003_s3": self._frame("t095_000003_s3", shift=1.2)}
        stale = _stale_frames(existing, {"t095_000003_s3": {"shift": 2.0}}, [], set())
        assert stale == {"t095_000003_s3"}

    def test_a_drop_still_present_is_redone(self):
        from sm_db.cli import _stale_frames

        existing = {"t095_000003_s3": self._frame("t095_000003_s3")}
        assert _stale_frames(existing, {}, [], {"t095_000003_s3"}) == {"t095_000003_s3"}

    def test_a_drop_already_gone_is_ignored(self):
        from sm_db.cli import _stale_frames

        assert _stale_frames({}, {}, [], {"t095_000003_s3"}) == set()

    def test_an_unbuilt_merge_is_redone(self):
        from sm_db.cli import _stale_frames

        existing = {
            "t095_000003_s3": self._frame("t095_000003_s3"),
            "t095_000004_s3": self._frame("t095_000004_s3"),
        }
        stale = _stale_frames(
            existing, {}, [["t095_000003_s3", "t095_000004_s3"]], set()
        )
        assert stale == {"t095_000003_s3", "t095_000004_s3"}

    def test_a_built_merge_is_left_alone(self):
        from sm_db.cli import _stale_frames

        existing = {"t095_000003_s3": self._frame("t095_000003_s3")}
        stale = _stale_frames(
            existing, {}, [["t095_000003_s3", "t095_000004_s3"]], set()
        )
        assert stale == set()


class TestCoverageSummary:
    """Regression: the summary read a key the report no longer carries.

    It only surfaced once a frame fell below the threshold, which none did
    until frames were widened past a single acquisition.
    """

    def _report(self, **frames):
        return {
            fid: {
                "common": c,
                "typical": t,
                "n_acquisitions": 5,
                "cost_of_worst": t - c,
            }
            for fid, (c, t) in frames.items()
        }

    def test_frames_below_the_threshold_do_not_crash_it(self, capsys):
        from sm_db.cli import _summarise_coverage

        _summarise_coverage(self._report(a=(0.5, 0.95), b=(0.4, 0.6)), 0.9)
        out = capsys.readouterr().out
        assert "frames below" in out

    def test_separates_a_few_bad_dates_from_a_frame_too_long(self, capsys):
        from sm_db.cli import _summarise_coverage

        _summarise_coverage(self._report(a=(0.5, 0.95), b=(0.4, 0.6)), 0.9)
        out = capsys.readouterr().out
        assert "hold the stack back :    1" in out
        assert "one acquisition covers      :    1" in out
