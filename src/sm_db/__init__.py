"""Sentinel-1 Stripmap frame database.

Defines a deterministic, repeat-stable frame grid for Sentinel-1 stripmap
acquisitions and writes it in the sqlite schema COMPASS already reads, so every
acquisition of a frame geocodes onto one identical output grid and the products
stack for time-series analysis.

See `sm_db.tiling` for the frame ID specification.
"""

from sm_db.frames import Frame, frames_for_granule
from sm_db.granules import Granule
from sm_db.tiling import format_frame_id, frame_index, parse_frame_id

__version__ = "0.1.0"

__all__ = [
    "Frame",
    "Granule",
    "__version__",
    "format_frame_id",
    "frame_index",
    "frames_for_granule",
    "parse_frame_id",
]
