"""Crop the point cloud to the region the detector was trained on, before inference.

OpenPCDet's KITTI models are trained with FOV_POINTS_ONLY: only points that project into the left
camera image (KittiDataset.get_fov_flag) are used. The Detector wrapper otherwise feeds the whole
x >= 0 half-space of POINT_CLOUD_RANGE, i.e. regions the model never saw in training.

Modes:
    "none"     no crop (behaviour before the crop existed)
    "camera"   keep points in front of the camera that project into the image; same rule as
               get_fov_flag (0 <= u < width, 0 <= v < height, rect depth >= 0); needs calibration
    "azimuth"  calibration-free: keep points with |atan2(y, x)| <= theta (LiDAR frame)
    "auto"     "camera" when the frame has a calibration, "azimuth" otherwise
"""
from __future__ import annotations

import math

import numpy as np

from ingestion.calibration import IMAGE_HEIGHT, IMAGE_WIDTH, Calibration

MODES = ("none", "camera", "azimuth", "auto")
# 99.5 % of the points kept by get_fov_flag on KITTI object val (3769 frames, 71.1 M points) have
# |azimuth| <= this angle; their azimuth range is -41.16 .. 40.29 deg
# (experiments.run_detector_path_check, section=azimuth: results/E0/20261108-140552_detector_path_check_azimuth_4b584e.json).
AZIMUTH_THETA_DEG: float = 39.75


def camera_mask(xyz: np.ndarray, calib: Calibration, width: int = IMAGE_WIDTH,
                height: int = IMAGE_HEIGHT) -> np.ndarray:
    """get_fov_flag's rule, including its projection: OpenPCDet's rect_to_img divides P2 @ X by the
    rectified depth z, not by the third row of the projection (z + P2[2, 3]) as
    Calibration.velo_to_image does; near the image border that moves points by up to ~1 px."""
    rect = calib.velo_to_cam(np.asarray(xyz)[:, :3])
    proj = np.hstack([rect, np.ones((len(rect), 1))]) @ calib.P2.T
    z = rect[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = proj[:, 0] / z, proj[:, 1] / z
        return (u >= 0) & (u < width) & (v >= 0) & (v < height) & (z >= 0)


def azimuth_mask(xyz: np.ndarray, theta_deg: float = AZIMUTH_THETA_DEG) -> np.ndarray:
    xyz = np.asarray(xyz)
    return np.abs(np.arctan2(xyz[:, 1], xyz[:, 0])) <= math.radians(theta_deg)


def calibration_of(calib) -> Calibration | None:
    """A Calibration from what a Frame carries (dict from parse_calib_file, Calibration or nothing)."""
    if isinstance(calib, Calibration):
        return calib
    if not calib:
        return None
    try:
        return Calibration.from_dict(calib)
    except ValueError:
        return None


def resolve_mode(mode: str, has_calibration: bool) -> str:
    """The mode actually used: "auto" becomes "camera" or "azimuth"; "camera" needs a calibration."""
    if mode not in MODES:
        raise ValueError(f"unknown FOV mode {mode!r} (use one of {', '.join(MODES)})")
    if mode == "auto":
        return "camera" if has_calibration else "azimuth"
    if mode == "camera" and not has_calibration:
        raise ValueError("FOV mode 'camera' needs a calibration; use 'azimuth' or 'none'")
    return mode


def describe(mode: str, theta_deg: float = AZIMUTH_THETA_DEG) -> str:
    """Resolved mode as stored in caches and results, e.g. "camera", "azimuth:45.0", "none"."""
    return f"azimuth:{theta_deg:g}" if mode == "azimuth" else mode


def sequence_mode(seq, mode: str, theta_deg: float = AZIMUTH_THETA_DEG) -> str:
    """describe() of the mode used for a whole sequence (its frames share one calibration)."""
    has_calibration = len(seq) > 0 and calibration_of(seq[0].calib) is not None
    return describe(resolve_mode(mode, has_calibration), theta_deg)


def add_fov_argument(parser) -> None:
    parser.add_argument("--fov", choices=MODES, default="auto",
                        help="crop the points before detection: auto = camera with a calibration, "
                             "else azimuth; none = no crop (detection/fov.py)")


def crop_points(points: np.ndarray, mode: str, calib=None, theta_deg: float = AZIMUTH_THETA_DEG,
                image_size: tuple[int, int] | None = None) -> tuple[np.ndarray, str]:
    """(points kept, resolved mode). `image_size` = (width, height), default the nominal KITTI size."""
    cal = calibration_of(calib)
    mode = resolve_mode(mode, cal is not None)
    if mode == "none":
        return points, mode
    if mode == "camera":
        w, h = image_size or (IMAGE_WIDTH, IMAGE_HEIGHT)
        return points[camera_mask(points, cal, w, h)], mode
    return points[azimuth_mask(points, theta_deg)], mode
