"""3D bounding box shared by detection, tracking, verification and export.

Convention (identical to OpenPCDet's LiDAR boxes, KITTI Velodyne frame):
    x forward, y left, z up
    (x, y, z)      geometric CENTER of the box
    (dx, dy, dz)   length (along heading), width, height
    heading        yaw around +z in radians, 0 = pointing along +x
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from common.geometry import normalize_angle

# Corner order (same as OpenPCDet boxes_to_corners_3d):
#   0:+x+y-z  1:+x-y-z  2:-x-y-z  3:-x+y-z   (bottom face)
#   4:+x+y+z  5:+x-y+z  6:-x-y+z  7:-x+y+z   (top face)
# Corners 0,1,4,5 form the front face (the side the box is heading to).
_CORNER_SIGNS = np.array(
    [
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)

# Edges as pairs of corner indices (for drawing wireframes).
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


@dataclass
class BoundingBox3D:
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    heading: float
    label: str
    score: float = 1.0
    track_id: int | None = None

    def __post_init__(self) -> None:
        if min(self.dx, self.dy, self.dz) <= 0:
            raise ValueError(
                f"Box dimensions must be positive, got "
                f"({self.dx}, {self.dy}, {self.dz})"
            )
        for name in ("x", "y", "z", "dx", "dy", "dz", "heading"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Field '{name}' must be finite, got {getattr(self, name)}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
        self.heading = normalize_angle(self.heading)  # canonical range [-pi, pi)

    # ------------------------------------------------------------------ geometry
    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dz], dtype=np.float64)

    @property
    def volume(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def corners(self) -> np.ndarray:
        """(8, 3) corner coordinates, order documented at the top of this file."""
        local = _CORNER_SIGNS * (self.dimensions / 2.0)
        c, s = math.cos(self.heading), math.sin(self.heading)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return local @ rot.T + self.center

    # ------------------------------------------------------------------ helpers
    def with_track_id(self, track_id: int | None) -> BoundingBox3D:
        return replace(self, track_id=track_id)

    def to_array(self) -> np.ndarray:
        """(7,) array [x, y, z, dx, dy, dz, heading] (label/score/id dropped)."""
        return np.array(
            [self.x, self.y, self.z, self.dx, self.dy, self.dz, self.heading],
            dtype=np.float64,
        )

    @classmethod
    def from_array(
        cls,
        arr,
        label: str,
        score: float = 1.0,
        track_id: int | None = None,
    ) -> BoundingBox3D:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 7:
            raise ValueError(f"Expected 7 values, got {arr.shape[0]}")
        x, y, z, dx, dy, dz, heading = (float(v) for v in arr)
        return cls(x, y, z, dx, dy, dz, heading, label, score, track_id)