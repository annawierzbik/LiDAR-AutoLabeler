"""Render verified boxes as label files.

Two formats, both produced as text in memory (``{relative path: text}``); nothing is written here:

    lidar   one file per frame, ``labels/<frame>.txt``, one box per line:
                x y z dx dy dz heading Class
            in the LiDAR frame of the BoundingBox3D convention (the layout of OpenPCDet's custom
            dataset). Needs no calibration, so it works for any folder of .bin frames.
    kitti   KITTI tracking labels, one file for the sequence, ``label_02/<sequence>.txt``:
                frame track_id type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
            in the rectified camera frame (location = bottom centre of the box). Needs the
            calibration of a KITTI sequence.

Which frames are written: by default only frames that were fully reviewed (no box waiting for a
decision) and have at least one final box. A frame with boxes still to review would otherwise be
exported with missing objects, which silently teaches a detector that those objects are background.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from common.boxes import BoundingBox3D
from common.geometry import normalize_angle
from ingestion.calibration import IMAGE_HEIGHT, IMAGE_WIDTH, Calibration

OCCLUDED_UNKNOWN = 3   # KITTI: 3 = unknown (occlusion is not measured here)


class ExportError(ValueError):
    """The requested export is not possible; the message can be shown to the user as it is."""


@dataclass
class ExportSummary:
    format: str
    frames: int = 0                  # frames that got labels
    boxes: int = 0
    per_class: dict = field(default_factory=dict)
    skipped_unreviewed: int = 0      # frames with boxes still waiting for a decision
    skipped_empty: int = 0           # reviewed frames without any final box
    skipped_outside_view: int = 0    # boxes outside the camera image (kitti only)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ----------------------------------------------------------------------------- frame selection
def select_frames(session, include_unreviewed: bool = False, include_empty: bool = False):
    """(session frames to export, number skipped as unreviewed, number skipped as empty)."""
    chosen, unreviewed, empty = [], 0, 0
    for f in range(session.num_frames):
        if session.frame_has_pending(f) and not include_unreviewed:
            unreviewed += 1
        elif not session.final_boxes(f, include_pending=include_unreviewed) and not include_empty:
            empty += 1
        else:
            chosen.append(f)
    return chosen, unreviewed, empty


def _class_name(box: BoundingBox3D) -> str:
    return box.label.replace(" ", "_")


def _count(summary: ExportSummary, box: BoundingBox3D) -> None:
    summary.boxes += 1
    summary.per_class[box.label] = summary.per_class.get(box.label, 0) + 1


# ----------------------------------------------------------------------------- lidar format
def lidar_line(box: BoundingBox3D) -> str:
    return (f"{box.x:.4f} {box.y:.4f} {box.z:.4f} {box.dx:.4f} {box.dy:.4f} {box.dz:.4f} "
            f"{box.heading:.6f} {_class_name(box)}")


_LIDAR_README = """Labels exported by LiDAR-AutoLabeler (verified boxes only).

labels/<frame>.txt   one box per line:  x y z dx dy dz heading Class
                     LiDAR frame: x forward, y left, z up. (x, y, z) is the CENTRE of the box,
                     (dx, dy, dz) = length (along the heading), width, height in metres,
                     heading = yaw around +z in radians (0 = pointing along +x).
ImageSets/exported.txt   names of the frames that have a label file.
"""


def render_lidar(session, stems, include_unreviewed: bool = False, include_empty: bool = False):
    """`stems[i]` = file name (without extension) of session frame i. Returns (files, summary)."""
    frames, unreviewed, empty = select_frames(session, include_unreviewed, include_empty)
    summary = ExportSummary("lidar", skipped_unreviewed=unreviewed, skipped_empty=empty)
    files = {}
    for f in frames:
        boxes = session.final_boxes(f, include_pending=include_unreviewed)
        for b in boxes:
            _count(summary, b)
        files[f"labels/{stems[f]}.txt"] = "".join(lidar_line(b) + "\n" for b in boxes)
        summary.frames += 1
    files["ImageSets/exported.txt"] = "".join(f"{stems[f]}\n" for f in frames)
    files["README.txt"] = _LIDAR_README
    return files, summary


# ----------------------------------------------------------------------------- kitti format
@dataclass
class KittiBox:
    location: np.ndarray        # (3,) bottom centre in the rectified camera frame
    h: float
    w: float
    l: float
    rotation_y: float
    alpha: float
    bbox: tuple                 # left, top, right, bottom (pixels, clipped to the image)
    truncated: float
    in_view: bool


def to_kitti(box: BoundingBox3D, calib: Calibration, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT)) -> KittiBox:
    """Convert one LiDAR box to KITTI camera-frame values.

    The heading is carried over as a direction vector, so it stays right for any calibration
    (for the usual KITTI matrices it equals -heading - pi/2, the formula OpenPCDet uses)."""
    centre = calib.velo_to_cam(box.center)[0]
    location = centre + np.array([0.0, box.dz / 2.0, 0.0])           # camera y points down
    direction = calib.velo_to_cam_mat[:3, :3] @ np.array([math.cos(box.heading), math.sin(box.heading), 0.0])
    rotation_y = float(math.atan2(-direction[2], direction[0]))
    alpha = normalize_angle(rotation_y - math.atan2(location[0], location[2]))

    uv, depth = calib.velo_to_image(box.corners)
    front = depth > 0.1
    width, height = image_size
    bbox, truncated, in_view = (0.0, 0.0, 0.0, 0.0), 0.0, False
    if front.any():
        x1, x2 = float(uv[front, 0].min()), float(uv[front, 0].max())
        y1, y2 = float(uv[front, 1].min()), float(uv[front, 1].max())
        cx1, cy1, cx2, cy2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(width)), min(y2, float(height))
        if cx2 > cx1 and cy2 > cy1:
            in_view = True
            bbox = (cx1, cy1, cx2, cy2)
            full = (x2 - x1) * (y2 - y1)
            truncated = float(min(1.0, max(0.0, 1.0 - (cx2 - cx1) * (cy2 - cy1) / full))) if full > 0 else 0.0
    return KittiBox(location, box.dz, box.dy, box.dx, rotation_y, alpha, bbox, truncated, in_view)


def kitti_line(frame_id: int, track_id: int, box: BoundingBox3D, k: KittiBox) -> str:
    x1, y1, x2, y2 = k.bbox
    return (f"{frame_id} {track_id} {_class_name(box)} {k.truncated:.2f} {OCCLUDED_UNKNOWN} {k.alpha:.6f} "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {k.h:.6f} {k.w:.6f} {k.l:.6f} "
            f"{k.location[0]:.6f} {k.location[1]:.6f} {k.location[2]:.6f} {k.rotation_y:.6f}")


_KITTI_README = """Labels exported by LiDAR-AutoLabeler (verified boxes only), KITTI tracking format.

label_02/<sequence>.txt   frame track_id type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y
  location (x y z) = bottom centre of the box in the rectified camera frame, (h w l) in metres.
  truncated is computed from the 2D box, occluded is not measured (3 = unknown).
  Boxes outside the camera image are left out unless the export was asked to keep them (their 2D box is 0 0 0 0).
"""


def render_kitti_tracking(session, frame_ids, calib: Calibration, name: str,
                          include_unreviewed: bool = False, only_in_view: bool = True):
    """`frame_ids[i]` = KITTI frame number of session frame i. Returns (files, summary)."""
    frames, unreviewed, _ = select_frames(session, include_unreviewed, include_empty=True)
    summary = ExportSummary("kitti", skipped_unreviewed=unreviewed)
    taken = {b.track_id for f in frames for b in session.final_boxes(f, include_pending=include_unreviewed)
             if b.track_id is not None}
    next_id = max(taken, default=-1) + 1
    rows = []
    for f in frames:
        wrote = False
        for b in session.final_boxes(f, include_pending=include_unreviewed):
            k = to_kitti(b, calib)
            if only_in_view and not k.in_view:
                summary.skipped_outside_view += 1
                continue
            tid = b.track_id
            if tid is None:
                tid, next_id = next_id, next_id + 1
            rows.append((frame_ids[f], tid, kitti_line(frame_ids[f], tid, b, k)))
            _count(summary, b)
            wrote = True
        if wrote:
            summary.frames += 1
    rows.sort(key=lambda r: (r[0], r[1]))
    return {f"label_02/{name}.txt": "".join(r[2] + "\n" for r in rows), "README.txt": _KITTI_README}, summary
