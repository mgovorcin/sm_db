"""Tests for granule metadata and the ASF query."""

from __future__ import annotations

import datetime

import pytest

from sm_db.granules import (
    ASF_SEARCH_URL,
    SM_BEAM_MODES,
    Granule,
    load_catalog,
    query_asf,
    save_catalog,
)


class FakeResponse:
    """Stands in for a `requests` response, recording nothing but its payload."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """Records the ASF query it was asked to make and replays a canned answer."""

    def __init__(self, payload):
        self._payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None):
        self.calls.append((url, params or {}))
        return FakeResponse(self._payload)


def _feature(name, start, stop, beam="S3", track=95, direction="ASCENDING"):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        },
        "properties": {
            "sceneName": name,
            "beamModeType": beam,
            "pathNumber": track,
            "orbit": 6741,
            "flightDirection": direction,
            "startTime": start,
            "stopTime": stop,
        },
    }


class TestQueryAsf:
    def test_asks_for_every_stripmap_beam_by_default(self):
        session = FakeSession({"features": []})
        query_asf("2026-02-01", "2026-03-01", session=session)

        url, params = session.calls[0]
        assert url == ASF_SEARCH_URL
        assert params["beamMode"] == ",".join(SM_BEAM_MODES)
        assert params["processingLevel"] == "SLC"
        assert params["platform"] == "Sentinel-1"

    def test_restricts_beams_and_tracks_when_asked(self):
        session = FakeSession({"features": []})
        query_asf(
            "2026-02-01",
            "2026-03-01",
            beam_modes=["S3"],
            tracks=[95, 96],
            session=session,
        )

        _, params = session.calls[0]
        assert params["beamMode"] == "S3"
        assert params["relativeOrbit"] == "95,96"

    def test_omits_the_area_filter_when_not_given(self):
        session = FakeSession({"features": []})
        query_asf("2026-02-01", "2026-03-01", session=session)
        assert "intersectsWith" not in session.calls[0][1]

    def test_maps_asf_fields_onto_the_record(self):
        session = FakeSession(
            {
                "features": [
                    _feature("S1C_S3_A", "2026-03-13T04:41:34Z", "2026-03-13T04:41:54Z")
                ]
            }
        )
        (granule,) = query_asf("2026-02-01", "2026-04-01", session=session)

        assert granule.name == "S1C_S3_A"
        assert granule.beam_mode == "S3"
        assert granule.track == 95
        assert granule.ascending is True
        assert granule.start == datetime.datetime(2026, 3, 13, 4, 41, 34)

    def test_descending_flagged(self):
        session = FakeSession(
            {
                "features": [
                    _feature(
                        "S1C_S3_D",
                        "2026-03-13T04:41:34Z",
                        "2026-03-13T04:41:54Z",
                        direction="DESCENDING",
                    )
                ]
            }
        )
        (granule,) = query_asf("2026-02-01", "2026-04-01", session=session)
        assert granule.ascending is False

    def test_results_come_back_in_acquisition_order(self):
        session = FakeSession(
            {
                "features": [
                    _feature("later", "2026-03-25T04:41:34Z", "2026-03-25T04:41:54Z"),
                    _feature("earlier", "2026-03-13T04:41:34Z", "2026-03-13T04:41:54Z"),
                ]
            }
        )
        assert [
            g.name for g in query_asf("2026-02-01", "2026-04-01", session=session)
        ] == [
            "earlier",
            "later",
        ]

    def test_times_are_naive_utc(self):
        """Orbit files carry naive UTC, and the two get subtracted from each other."""
        session = FakeSession(
            {
                "features": [
                    _feature("S1C_S3_A", "2026-03-13T04:41:34Z", "2026-03-13T04:41:54Z")
                ]
            }
        )
        (granule,) = query_asf("2026-02-01", "2026-04-01", session=session)
        assert granule.start.tzinfo is None


class TestCatalog:
    def test_round_trip(self, tmp_path, granule):
        path = tmp_path / "catalog.json"
        save_catalog([granule], path)
        (restored,) = load_catalog(path)

        assert restored.name == granule.name
        assert restored.start == granule.start
        assert restored.stop == granule.stop
        assert restored.track == granule.track
        assert restored.ascending == granule.ascending
        assert restored.footprint.equals_exact(granule.footprint, 1e-9)

    def test_footprint_is_a_polygon(self, granule):
        assert granule.footprint.geom_type == "Polygon"

    def test_beam_mode_is_normalised_on_read(self):
        restored = Granule.from_dict(
            {
                "name": "x",
                "beam_mode": "s3",
                "track": 95,
                "absolute_orbit": 1,
                "ascending": True,
                "start": "2026-03-13T04:41:34",
                "stop": "2026-03-13T04:41:54",
                "geometry": {},
            }
        )
        assert restored.beam_mode == "S3"

    def test_missing_field_is_an_error(self):
        with pytest.raises(KeyError):
            Granule.from_dict({"name": "x"})


class TestQueryRetryAndSplit:
    class FlakySession:
        """Refuses any window wider than `max_days` with a 400, as ASF does."""

        def __init__(self, max_days, features_per_call=1):
            self.max_days = max_days
            self.features_per_call = features_per_call
            self.windows = []

        def get(self, url, params=None):
            from datetime import datetime

            p = params or {}
            self.windows.append((p["start"], p["end"]))
            span = (
                datetime.fromisoformat(p["end"]) - datetime.fromisoformat(p["start"])
            ).days
            if span > self.max_days:
                raise _HttpError(400)
            return FakeResponse(
                {
                    "features": [
                        _feature(
                            f"S1A_S3_{len(self.windows)}_{i}",
                            p["start"] + "T00:00:00Z",
                            p["start"] + "T00:00:20Z",
                        )
                        for i in range(self.features_per_call)
                    ]
                }
            )

    def test_splits_a_window_asf_refuses(self):
        """A year that 400s must come back as its halves, not as an exception."""
        session = self.FlakySession(max_days=100)
        granules = query_asf("2020-01-01", "2020-12-31", session=session)

        assert len(granules) > 1
        spans = [w for w in session.windows]
        assert ("2020-01-01", "2020-12-31") == spans[0]
        assert len(spans) > 1

    def test_no_split_when_the_first_query_succeeds(self):
        session = self.FlakySession(max_days=10_000)
        query_asf("2020-01-01", "2020-12-31", session=session)
        assert len(session.windows) == 1

    def test_gives_up_on_a_single_day_it_cannot_satisfy(self):
        session = self.FlakySession(max_days=-1)
        with pytest.raises(RuntimeError, match="cannot be split"):
            query_asf("2020-01-01", "2020-01-01", session=session)


class _HttpError(Exception):
    """Stands in for `requests.HTTPError`, carrying a response status."""

    def __init__(self, status):
        super().__init__(f"{status} Client Error")
        self.response = type("R", (), {"status_code": status})()


class TestGzippedCatalog:
    def test_round_trips_through_gzip(self, tmp_path, granule):
        path = tmp_path / "catalog.json.gz"
        save_catalog([granule], path)
        (restored,) = load_catalog(path)
        assert restored.name == granule.name
        assert restored.start == granule.start

    def test_gzip_is_actually_compressed(self, tmp_path, granule):
        """A daily commit of the whole archive is why this exists."""
        import gzip

        path = tmp_path / "catalog.json.gz"
        save_catalog([granule] * 200, path)
        assert path.stat().st_size < len(gzip.open(path, "rb").read()) / 5

    def test_plain_json_still_works(self, tmp_path, granule):
        path = tmp_path / "catalog.json"
        save_catalog([granule], path)
        assert path.read_text().lstrip().startswith("[")
        assert load_catalog(path)[0].name == granule.name
