"""Turn KITTI label dicts (camera frame) into BoundingBox3D in the LiDAR frame.

A label dict (from `parse_tracking_labels`) has keys:
    track_id, type, bbox, dimensions = (h, w, l), location = (x, y, z), rotation_y
`location` is the BOTTOM center of the box in camera coordinates.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from common.boxes import BoundingBox3D
from ingestion.calibration import Calibration

EVAL_CLASSES = ("Car", "Pedestrian", "Cyclist")
# Real objects we do not evaluate on. A detection overlapping one of them is
# not counted as a false positive (e.g. a Van detected as Car).
IGNORE_CLASSES = ("Van", "Truck", "Tram", "Person_sitting", "Misc")


def label_to_box(label: dict, calib: Calibration, label_name: str) -> BoundingBox3D:
    h, w, l = (float(v) for v in label["dimensions"])
    loc = np.asarray(label["location"], dtype=np.float64)

    center_cam = loc - np.array([0.0, h / 2.0, 0.0])  # bottom center -> geometric center (y points down)
    x, y, z = calib.cam_to_velo(center_cam)[0]
    heading = -np.pi / 2.0 - float(label["rotation_y"])  # camera yaw -> LiDAR yaw

    return BoundingBox3D(
        x=float(x), y=float(y), z=float(z),
        dx=l, dy=w, dz=h,
        heading=float(heading),
        label=label_name,
        score=1.0,
        track_id=int(label["track_id"]),
    )


def load_frame_gt(
    labels: Sequence[dict], calib: Calibration
) -> Tuple[List[BoundingBox3D], List[BoundingBox3D]]:
    """Returns (evaluated ground truth, ignore regions) for one frame, in LiDAR frame."""
    gt, ignore = [], []
    for lab in labels:
        typ = lab["type"]
        if typ not in EVAL_CLASSES and typ not in IGNORE_CLASSES:
            continue  # DontCare etc.
        h, w, l = lab["dimensions"]
        if min(h, w, l) <= 0:
            continue
        box = label_to_box(lab, calib, typ)
        (gt if typ in EVAL_CLASSES else ignore).append(box)
    return gt, ignore


def filter_to_camera_fov(
    boxes: Sequence[BoundingBox3D], calib: Calibration
) -> List[BoundingBox3D]:
    """KITTI labels only exist where the camera sees, but the LiDAR (and our
    detector) covers 360 deg. Detections outside the image would all count as
    false positives, so they are dropped before evaluation."""
    if not boxes:
        return []
    centers = np.array([b.center for b in boxes])
    mask = calib.in_image_fov(centers)
    return [b for b, keep in zip(boxes, mask) if keep]