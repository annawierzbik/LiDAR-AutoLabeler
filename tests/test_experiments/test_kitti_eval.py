"""The rotate_iou stand-in and the OpenPCDet evaluator loaded through experiments.lib.kitti_eval."""
import math

import numpy as np
import pytest

from experiments.lib.rotate_iou_cpu import rotate_iou_cpu_eval


def test_rotate_iou_known_shapes():
    a = np.array([[0, 0, 4, 2, 0.0]], dtype=np.float32)
    b = np.array([[0, 0, 4, 2, 0.0], [2, 0, 4, 2, 0.0], [0, 0, 4, 2, math.pi / 2], [10, 10, 1, 1, 0.0]],
                 dtype=np.float32)
    iou = rotate_iou_cpu_eval(a, b, -1)
    np.testing.assert_allclose(iou[0], [1.0, 1 / 3, 4 / 12, 0.0], atol=1e-6)


def test_rotate_iou_criteria_follow_the_cuda_argument_order():
    # the CUDA kernel evaluates (query box, box): criterion 0 divides by the QUERY box area
    a = np.array([[0, 0, 4, 2, 0.0]], dtype=np.float32)     # area 8
    b = np.array([[0, 0, 2, 2, 0.0]], dtype=np.float32)     # area 4, inside a
    assert rotate_iou_cpu_eval(a, b, 2)[0, 0] == pytest.approx(4.0)
    assert rotate_iou_cpu_eval(a, b, 0)[0, 0] == pytest.approx(1.0)
    assert rotate_iou_cpu_eval(a, b, 1)[0, 0] == pytest.approx(0.5)


def test_identical_rotated_boxes_are_degenerate_as_in_the_original():
    """OpenPCDet's rotate_iou gives ~0 (9e-8 under numba 0.60 CUDA, checked in E0) for two exactly
    identical rotated boxes: all corners lie on the other box's edges. Evaluating GT against itself
    with the OpenPCDet evaluator therefore does not give AP 100. Axis-aligned copies are fine."""
    rotated = np.array([[1.0, 15.0, 4.0, 1.8, 0.1]], dtype=np.float32)
    assert rotate_iou_cpu_eval(rotated, rotated, -1)[0, 0] < 1e-6
    assert rotate_iou_cpu_eval(rotated, rotated + [0.01, 0, 0, 0, 0], -1)[0, 0] > 0.99
    axis = np.array([[1.0, 15.0, 4.0, 1.8, 0.0]], dtype=np.float32)
    assert rotate_iou_cpu_eval(axis, axis, -1)[0, 0] == pytest.approx(1.0, abs=1e-5)


def _anno(n, names, boxes_cam, score=None):
    """Minimal KITTI annos: boxes_cam rows [x, y, z, l, h, w, ry] (OpenPCDet annos dims are l, h, w)."""
    boxes_cam = np.asarray(boxes_cam, dtype=np.float64).reshape(-1, 7)
    a = {"name": np.array(names), "truncated": np.zeros(n), "occluded": np.zeros(n, dtype=np.int64),
         "alpha": np.zeros(n), "bbox": np.tile([500.0, 150.0, 600.0, 250.0], (n, 1)),
         "dimensions": boxes_cam[:, 3:6], "location": boxes_cam[:, :3], "rotation_y": boxes_cam[:, 6]}
    a["score"] = np.ones(n) if score is None else np.asarray(score, dtype=np.float64)
    return a


def test_official_eval_perfect_and_missed_predictions():
    pytest.importorskip("torch")
    from experiments.lib import kitti_eval as ke

    # 60 frames with one car each; KITTI samples 41 recall points from the TP scores, so it needs
    # at least ~40 GT objects with distinct scores to reach 100 (with 5 objects it reports 10.0)
    n = 60
    box = [[1.0, 1.6, 15.0, 4.0, 1.5, 1.8, 0.1]]
    gts = [_anno(1, ["Car"], box) for _ in range(n)]
    # 1 cm off: exact copies of rotated boxes are degenerate (see the test above)
    perfect = [_anno(1, ["Car"], [[1.01, 1.6, 15.0, 4.0, 1.5, 1.8, 0.1]], [0.3 + 0.01 * i]) for i in range(n)]
    _, res = ke.official_eval(gts, perfect, classes=("Car",))
    assert res["Car_3d/moderate_R40"] == pytest.approx(100.0)
    shifted = [_anno(1, ["Car"], [[1.0, 1.6, 17.0, 4.0, 1.5, 1.8, 0.1]], [0.3 + 0.01 * i]) for i in range(n)]
    _, res = ke.official_eval(gts, shifted, classes=("Car",))     # IoU 1/3 < 0.7
    assert res["Car_3d/moderate_R40"] == pytest.approx(0.0)
