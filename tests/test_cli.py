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
