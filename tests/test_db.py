"""Tests for the frame database, including the schema COMPASS depends on."""

from __future__ import annotations

import json
import sqlite3

import pytest
from shapely.geometry import box

from sm_db.db import read_frames, write_database
from sm_db.frames import Frame


def _frame(frame_id: str = "t095_000003_s3", index: int = 3) -> Frame:
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
def database(tmp_path):
    path = tmp_path / "frames.sqlite3"
    write_database(
        [_frame("t095_000004_s3", 4), _frame("t095_000003_s3", 3)],
        path,
        tile_seconds=5.0,
        margin=5000.0,
        snap=30.0,
    )
    return path


class TestCompassContract:
    """COMPASS reads a burst database through `helpers.burst_bboxes_from_db`.

    That function runs ``SELECT * FROM burst_id_map WHERE burst_id_jpl IN (...)``
    and reads `epsg`, `xmin`, `ymin`, `xmax`, `ymax` off each row. If any of that
    stops being true, every stripmap job silently produces nothing, because the
    worker drops bursts it cannot find in the database.
    """

    def test_query_compass_runs_returns_the_frame(self, database):
        con = sqlite3.connect(database)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM burst_id_map WHERE burst_id_jpl IN (?)",
            ("t095_000003_s3",),
        ).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row["epsg"] == 32604
        assert (row["xmin"], row["ymin"], row["xmax"], row["ymax"]) == (
            657240,
            183120,
            753690,
            243360,
        )

    def test_bounds_are_stored_as_integers(self, database):
        con = sqlite3.connect(database)
        types = con.execute(
            "SELECT typeof(xmin), typeof(ymin), typeof(xmax), typeof(ymax) "
            "FROM burst_id_map LIMIT 1"
        ).fetchone()
        assert set(types) == {"integer"}

    def test_frame_id_is_unique(self, database):
        con = sqlite3.connect(database)
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO burst_id_map "
                "(burst_id_jpl, epsg, xmin, ymin, xmax, ymax) VALUES (?,?,?,?,?,?)",
                ("t095_000003_s3", 32604, 0, 0, 1, 1),
            )


class TestWriteDatabase:
    def test_returns_the_number_written(self, tmp_path):
        n = write_database(
            [_frame()], tmp_path / "f.sqlite3", tile_seconds=5.0, margin=0.0, snap=30.0
        )
        assert n == 1

    def test_rebuilding_replaces_rather_than_appends(self, database):
        write_database([_frame()], database, tile_seconds=5.0, margin=5000.0, snap=30.0)
        con = sqlite3.connect(database)
        assert con.execute("SELECT COUNT(*) FROM burst_id_map").fetchone()[0] == 1

    def test_records_build_parameters(self, database):
        con = sqlite3.connect(database)
        meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
        assert json.loads(meta["tile_seconds"]) == 5.0
        assert json.loads(meta["margin"]) == 5000.0
        assert json.loads(meta["snap"]) == 30.0
        assert json.loads(meta["n_frames"]) == 2

    def test_records_extra_metadata(self, tmp_path):
        path = tmp_path / "f.sqlite3"
        write_database(
            [_frame()],
            path,
            tile_seconds=5.0,
            margin=0.0,
            snap=30.0,
            extra={"start": "2026-02-01"},
        )
        con = sqlite3.connect(path)
        meta = dict(con.execute("SELECT key, value FROM metadata").fetchall())
        assert json.loads(meta["start"]) == "2026-02-01"


class TestReadFrames:
    def test_round_trip(self, database):
        frames = read_frames(database)
        assert [f.frame_id for f in frames] == ["t095_000003_s3", "t095_000004_s3"]

        original = _frame()
        restored = frames[0]
        assert restored.bbox == original.bbox
        assert restored.epsg == original.epsg
        assert restored.track == original.track
        assert restored.beam == original.beam
        assert restored.polygon.equals_exact(original.polygon, 1e-9)

    def test_filters_by_id(self, database):
        frames = read_frames(database, ["t095_000004_s3"])
        assert [f.frame_id for f in frames] == ["t095_000004_s3"]

    def test_unknown_id_returns_nothing(self, database):
        assert read_frames(database, ["t111_000001_s1"]) == []

    def test_empty_id_list_returns_nothing(self, database):
        """An empty IN clause is not valid SQL, so it has to short-circuit."""
        assert read_frames(database, []) == []


class TestFillPersisted:
    def test_round_trips(self, tmp_path):
        frame = _frame()
        object.__setattr__(frame, "fill_pct", 47.3)
        path = tmp_path / "f.sqlite3"
        write_database([frame], path, tile_seconds=5.0, margin=0.0, snap=30.0)
        assert read_frames(path)[0].fill_pct == 47.3

    def test_column_exists_for_querying(self, database):
        """Stored so a frame can be filtered on emptiness without recomputing it."""
        con = sqlite3.connect(database)
        rows = con.execute(
            "SELECT burst_id_jpl FROM frames WHERE fill_pct >= 0"
        ).fetchall()
        assert len(rows) == 2
