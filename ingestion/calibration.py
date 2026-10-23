"""KITTI calibration: transforms between LiDAR (Velodyne), camera and image.

Frames:
    LiDAR   x forward, y left,  z up      (what the detector outputs)
    camera  x right,   y down,  z forward (what KITTI labels use)

    X_cam = R_rect @ Tr_velo_cam @ X_velo          (homogeneous coordinates)
    uv    = P2 @ X_cam                             (pixels of the left color camera)
"""
from __future__ import annotations

import numpy as np

from ingestion.loader import parse_calib_file

IMAGE_WIDTH, IMAGE_HEIGHT = 1242, 375  # nominal KITTI image size


def _pad_to_4x4(m: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[: m.shape[0], : m.shape[1]] = m
    return out


class Calibration:
    def __init__(self, P2: np.ndarray, R_rect: np.ndarray, Tr_velo_cam: np.ndarray) -> None:
        self.P2 = np.asarray(P2, dtype=np.float64).reshape(3, 4)
        self.R_rect = np.asarray(R_rect, dtype=np.float64).reshape(3, 3)
        self.Tr_velo_cam = np.asarray(Tr_velo_cam, dtype=np.float64).reshape(3, 4)

        self.velo_to_cam_mat = _pad_to_4x4(self.R_rect) @ _pad_to_4x4(self.Tr_velo_cam)
        self.cam_to_velo_mat = np.linalg.inv(self.velo_to_cam_mat)

    @classmethod
    def from_file(cls, path) -> Calibration:
        """Load a KITTI calibration text file."""
        return cls.from_dict(parse_calib_file(path))

    @classmethod
    def from_dict(cls, calib: dict[str, np.ndarray]) -> Calibration:
        """Build from the dict returned by `parse_calib_file` (tracking or detection naming)."""
        c = {k.strip().rstrip(":"): np.asarray(v, dtype=np.float64) for k, v in calib.items()}

        def pick(*names):
            for n in names:
                if n in c:
                    return c[n]
            raise ValueError(f"Calibration is missing {names[0]} (found keys: {sorted(c)})")

        return cls(
            P2=pick("P2"),
            R_rect=pick("R_rect", "R0_rect"),
            Tr_velo_cam=pick("Tr_velo_cam", "Tr_velo_to_cam"),
        )

    # -------------------------------------------------------------- transforms
    @staticmethod
    def _apply(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        return (homo @ mat.T)[:, :3]

    def velo_to_cam(self, pts: np.ndarray) -> np.ndarray:
        return self._apply(self.velo_to_cam_mat, pts)

    def cam_to_velo(self, pts: np.ndarray) -> np.ndarray:
        return self._apply(self.cam_to_velo_mat, pts)

    def velo_to_image(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (uv (N,2) pixels, depth (N,)) of LiDAR points in the left color image."""
        cam = self.velo_to_cam(pts)
        homo = np.hstack([cam, np.ones((len(cam), 1))])
        proj = homo @ self.P2.T
        depth = proj[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = proj[:, :2] / depth[:, None]
        return uv, cam[:, 2]

    def in_image_fov(
        self,
        pts: np.ndarray,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
        margin: float = 0.15,
    ) -> np.ndarray:
        """Boolean mask: point is in front of the camera and projects inside the
        image (enlarged by `margin` * size, so objects partly visible still count)."""
        uv, depth = self.velo_to_image(pts)
        mx, my = margin * width, margin * height
        return (
            (depth > 0.1)
            & (uv[:, 0] >= -mx) & (uv[:, 0] <= width + mx)
            & (uv[:, 1] >= -my) & (uv[:, 1] <= height + my)
        )