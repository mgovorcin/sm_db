"""Shared fixtures: a synthetic circular orbit and matching granules.

The orbit is generated rather than read from a real EOF so the tests stay
offline and self-describing -- an inclined circular orbit gives an exact,
known ascending node, which is what the frame index is measured from.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from sm_db.anx import OrbitStateVectors
from sm_db.granules import Granule

EARTH_RADIUS = 6_371_000.0
ORBIT_RADIUS = EARTH_RADIUS + 700_000.0
INCLINATION = math.radians(98.18)
PERIOD = (12 * 86400.0) / 175.0

ANX = datetime.datetime(2026, 3, 13, 4, 41, 23)
"""Ascending node of the synthetic orbit: exact by construction."""

NODE_LON = -157.0
"""Longitude of the ascending node, placing the track under `granule`.

Chosen so the synthetic ground track runs beneath the fixture footprint, as a
real orbit and its own granule necessarily do. Without that the cross-track
offsets measured against the track would be meaningless.
"""


def _state_vectors(
    start_offset: float, stop_offset: float, step: float = 10.0
) -> OrbitStateVectors:
    """Sample the synthetic orbit over seconds relative to `ANX`."""
    seconds = np.arange(start_offset, stop_offset + step, step)
    angle = 2 * math.pi * seconds / PERIOD

    # Circular orbit in a plane inclined by INCLINATION, with the ascending node
    # on the +X axis, so Z crosses zero upward exactly at `ANX`.
    x = ORBIT_RADIUS * np.cos(angle)
    y = ORBIT_RADIUS * np.sin(angle) * math.cos(INCLINATION)
    z = ORBIT_RADIUS * np.sin(angle) * math.sin(INCLINATION)

    # Spin the plane about the poles to put the node at NODE_LON. Earth rotation
    # during a 20 s scene is left out: the tests only need the orbit and the
    # footprint to be mutually consistent, not physically exact.
    node = math.radians(NODE_LON)
    x, y = (
        x * math.cos(node) - y * math.sin(node),
        x * math.sin(node) + y * math.cos(node),
    )

    times = np.array([ANX + datetime.timedelta(seconds=float(s)) for s in seconds])
    return OrbitStateVectors(times, x, y, z)


@pytest.fixture
def orbit() -> OrbitStateVectors:
    """State vectors spanning one revolution either side of `ANX`."""
    return _state_vectors(-PERIOD - 600, PERIOD + 600)


@pytest.fixture
def granule() -> Granule:
    """A 20 s S3 acquisition starting 10 s after `ANX`.

    Footprint and timing follow the real S1C track-95 campaign, whose scenes sit
    just after the equator crossing, shifted in latitude onto the synthetic
    track defined by `NODE_LON`.
    """
    start = ANX + datetime.timedelta(seconds=10)
    return Granule(
        name="S1C_S3_SLC__1SDV_20260313T044134_20260313T044154_006741_00D9D7_BE10",
        beam_mode="S3",
        track=95,
        absolute_orbit=6741,
        ascending=True,
        start=start,
        stop=start + datetime.timedelta(seconds=20),
        geometry={
            "type": "Polygon",
            "coordinates": [
                [
                    [-157.3900, 0.5057],
                    [-156.6792, 0.6573],
                    [-156.9353, 1.8872],
                    [-157.6463, 1.7365],
                    [-157.3900, 0.5057],
                ]
            ],
        },
    )
