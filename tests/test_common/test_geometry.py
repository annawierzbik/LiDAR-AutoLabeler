import math

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from common.geometry import bev_iou, iou_3d, iou_matrix, normalize_angle


def box(x=0.0, y=0.0, z=0.0, dx=4.0, dy=2.0, dz=1.5, heading=0.0, label="Car"):
    return BoundingBox3D(x, y, z, dx, dy, dz, heading, label)


@pytest.mark.parametrize(
    "angle,expected",
    [(0.0, 0.0), (6.01, 6.01 - 2 * math.pi), (-4.0, -4.0 + 2 * math.pi), (2 * math.pi, 0.0)],
)
def test_normalize_angle(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected)


def test_normalize_angle_range():
    for a in np.linspace(-20, 20, 101):
        assert -math.pi <= normalize_angle(a) < math.pi


def test_box_heading_is_normalized_on_creation():
    assert box(heading=6.44).heading == pytest.approx(6.44 - 2 * math.pi)


def test_identical_boxes_iou_one():
    assert iou_3d(box(), box()) == pytest.approx(1.0)
    assert bev_iou(box(), box()) == pytest.approx(1.0)


def test_disjoint_boxes_iou_zero():
    assert iou_3d(box(), box(x=20)) == 0.0


def test_half_overlap_along_length():
    # shifted by half the length -> intersection = 1/2, union = 3/2 -> IoU = 1/3
    assert iou_3d(box(), box(x=2.0)) == pytest.approx(1 / 3)


def test_vertical_offset_reduces_iou():
    # shifted by half the height -> same 1/3
    assert iou_3d(box(), box(z=0.75)) == pytest.approx(1 / 3)
    assert iou_3d(box(), box(z=5.0)) == 0.0


def test_rotated_square_is_octagon_overlap():
    a = box(dx=2, dy=2, dz=1)
    b = box(dx=2, dy=2, dz=1, heading=math.pi / 4)
    inter = 8 * (math.sqrt(2) - 1)  # octagon area for side-2 square rotated 45 deg
    expected = inter / (8 - inter)
    assert bev_iou(a, b) == pytest.approx(expected / 1.0, rel=1e-6)


def test_iou_is_symmetric_and_rotation_by_pi_invariant():
    a, b = box(heading=0.3), box(x=1.0, y=0.5, heading=0.9)
    assert iou_3d(a, b) == pytest.approx(iou_3d(b, a))
    assert iou_3d(box(heading=0.0), box(heading=math.pi)) == pytest.approx(1.0)


def test_iou_matrix_shape_and_values():
    m = iou_matrix([box(), box(x=50)], [box(), box(x=50), box(x=100)])
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[1, 1] == pytest.approx(1.0)
    assert m[0, 2] == 0.0


def test_iou_matrix_empty():
    assert iou_matrix([], [box()]).shape == (0, 1)