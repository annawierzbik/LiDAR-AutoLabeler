import math

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from evaluation.ground_truth import filter_to_camera_fov, load_frame_gt
from ingestion.calibration import Calibration

TR = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float)
P2 = np.array([[720, 0, 620, 0], [0, 720, 187, 0], [0, 0, 1, 0]], dtype=float)
CALIB = Calibration(P2, np.eye(3), TR)


def label(typ="Car", tid=1, dims=(1.5, 1.8, 4.0), loc=(-2.0, 1.75, 10.0), ry=0.0):
    return dict(track_id=tid, type=typ, bbox=(0, 0, 0, 0), dimensions=dims, location=loc, rotation_y=ry)


def test_label_conversion_matches_hand_computation():
    gt, ignore = load_frame_gt([label()], CALIB)
    b = gt[0]
    # camera bottom-center (-2, 1.75, 10), h=1.5 -> center (-2, 1.0, 10) -> velo (10, 2, -1)
    assert (b.x, b.y, b.z) == pytest.approx((10.0, 2.0, -1.0))
    assert (b.dx, b.dy, b.dz) == pytest.approx((4.0, 1.8, 1.5))  # l, w, h
    assert b.heading == pytest.approx(-math.pi / 2)               # rotation_y = 0
    assert b.label == "Car" and b.track_id == 1 and ignore == []


def test_ground_lies_at_bottom_of_box():
    b = load_frame_gt([label()], CALIB)[0][0]
    assert b.z - b.dz / 2 == pytest.approx(-1.75)  # = camera y of bottom (down axis)


def test_heading_follows_rotation_y():
    b = load_frame_gt([label(ry=-math.pi / 2)], CALIB)[0][0]
    assert b.heading == pytest.approx(0.0)  # car pointing along LiDAR +x


def test_ignore_and_dontcare_handling():
    labs = [label("Van", 2), label("DontCare", -1, dims=(-1, -1, -1), loc=(-1000, -1000, -1000)),
            label("Pedestrian", 3, dims=(1.7, 0.6, 0.8))]
    gt, ignore = load_frame_gt(labs, CALIB)
    assert [b.label for b in gt] == ["Pedestrian"]
    assert [b.label for b in ignore] == ["Van"]


def test_fov_filter():
    inside = BoundingBox3D(10, 0, -1, 4, 2, 1.5, 0, "Car")
    behind = BoundingBox3D(-10, 0, -1, 4, 2, 1.5, 0, "Car")
    assert filter_to_camera_fov([inside, behind], CALIB) == [inside]
    assert filter_to_camera_fov([], CALIB) == []