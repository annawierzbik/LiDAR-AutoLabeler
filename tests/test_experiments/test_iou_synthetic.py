"""E0: the repo's 3D/BEV IoU on boxes with known answers, and against OpenPCDet's implementations."""
import math

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from common.geometry import bev_iou, iou_3d
from experiments.lib.rotate_iou_cpu import rotate_iou_cpu_eval


def box(x=0.0, y=0.0, z=0.0, dx=4.0, dy=2.0, dz=1.5, heading=0.0):
    return BoundingBox3D(x, y, z, dx, dy, dz, heading, "Car")


def test_identical_boxes():
    assert iou_3d(box(), box()) == pytest.approx(1.0)


def test_identical_rotated_boxes():
    b = box(x=3.0, y=-2.0, heading=0.7)
    assert iou_3d(b, b) == pytest.approx(1.0)


def test_disjoint_boxes():
    assert iou_3d(box(), box(x=10.0)) == 0.0
    assert iou_3d(box(), box(z=5.0)) == 0.0          # same footprint, no vertical overlap


def test_half_shift_along_length():
    # shifted by half the length: intersection 1/2 of each box -> IoU = 0.5 / 1.5
    assert iou_3d(box(), box(x=2.0)) == pytest.approx(1 / 3)


def test_half_shift_in_height():
    # same footprint, half the height overlaps -> IoU = 0.5 / 1.5
    assert iou_3d(box(), box(z=0.75)) == pytest.approx(1 / 3)


def test_box_turned_by_180_degrees_is_the_same_box():
    assert iou_3d(box(), box(heading=math.pi)) == pytest.approx(1.0)


def test_square_rotated_45_degrees():
    # unit squares with a common center, one turned by 45 deg: the intersection is a regular
    # octagon of area 2 (sqrt 2 - 1); IoU = octagon / (2 - octagon)
    a, b = box(dx=1, dy=1, dz=1), box(dx=1, dy=1, dz=1, heading=math.pi / 4)
    octagon = 2 * (math.sqrt(2) - 1)
    assert bev_iou(a, b) == pytest.approx(octagon / (2 - octagon))
    assert iou_3d(a, b) == pytest.approx(octagon / (2 - octagon))


def test_rectangle_rotated_90_degrees():
    # 4x2 vs 2x4 with a common center: intersection 2x2 = 4, union 8 + 8 - 4 = 12
    assert bev_iou(box(), box(heading=math.pi / 2)) == pytest.approx(4 / 12)


def _random_boxes(rng, n):
    return [box(x=rng.uniform(0, 4), y=rng.uniform(0, 4), z=rng.uniform(-0.5, 0.5), dx=rng.uniform(0.5, 5),
                dy=rng.uniform(0.5, 3), dz=rng.uniform(0.5, 2), heading=rng.uniform(-math.pi, math.pi))
            for _ in range(n)]


def test_bev_iou_matches_openpcdet_eval_algorithm():
    """rotate_iou_cpu is a port of OpenPCDet eval's BEV IoU (camera BEV: x, z, l, w, ry); a LiDAR
    box (x, y, heading) maps to (x, y, -heading) under the corner convention there (float32)."""
    rng = np.random.default_rng(1)
    a, b = _random_boxes(rng, 40), _random_boxes(rng, 50)
    ref = rotate_iou_cpu_eval(np.array([[q.x, q.y, q.dx, q.dy, -q.heading] for q in a]),
                              np.array([[q.x, q.y, q.dx, q.dy, -q.heading] for q in b]), -1)
    ours = np.array([[bev_iou(p, q) for q in b] for p in a])
    assert (ref > 0).sum() > 200          # the test is about overlapping boxes
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_iou_3d_matches_openpcdet_iou3d_nms():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

    from detection.detector import Detector  # noqa: F401  (puts OpenPCDet on sys.path)

    rng = np.random.default_rng(2)
    a, b = _random_boxes(rng, 40), _random_boxes(rng, 50)
    ta = torch.tensor(np.array([q.to_array() for q in a]), dtype=torch.float32).cuda()
    tb = torch.tensor(np.array([q.to_array() for q in b]), dtype=torch.float32).cuda()
    ref = boxes_iou3d_gpu(ta, tb).cpu().numpy()
    ours = np.array([[iou_3d(p, q) for q in b] for p in a])
    assert (ref > 0).sum() > 200
    np.testing.assert_allclose(ours, ref, atol=2e-2)   # iou3d_nms differs from exact by ~1e-2 (float32)
