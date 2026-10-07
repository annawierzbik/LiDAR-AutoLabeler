import numpy as np
import pytest

from ingestion.calibration import Calibration

# Idealised KITTI: cam_x = -velo_y, cam_y = -velo_z, cam_z = velo_x
TR = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0]], dtype=float)
R = np.eye(3)
P2 = np.array([[720, 0, 620, 0], [0, 720, 187, 0], [0, 0, 1, 0]], dtype=float)


@pytest.fixture
def calib():
    return Calibration(P2, R, TR)


def test_velo_to_cam_axes(calib):
    cam = calib.velo_to_cam([[10.0, 2.0, -1.0]])[0]
    assert np.allclose(cam, [-2.0, 1.0, 10.0])


def test_roundtrip(calib):
    pts = np.random.default_rng(0).uniform(-30, 30, (20, 3))
    assert np.allclose(calib.cam_to_velo(calib.velo_to_cam(pts)), pts)


def test_projection_of_point_straight_ahead(calib):
    uv, depth = calib.velo_to_image([[10.0, 0.0, 0.0]])
    assert np.allclose(uv[0], [620, 187])
    assert depth[0] == pytest.approx(10.0)


def test_fov_mask(calib):
    pts = np.array([
        [10, 0, 0],      # ahead -> in
        [10, 50, 0],     # far to the left -> out
        [-10, 0, 0],     # behind -> out
        [10, 4, 0],      # ~ left but inside image
    ], dtype=float)
    assert list(calib.in_image_fov(pts)) == [True, False, False, True]


def test_from_dict_variants():
    a = Calibration.from_dict({"P2": P2.reshape(-1), "R_rect": R.reshape(-1), "Tr_velo_cam": TR.reshape(-1)})
    b = Calibration.from_dict({"P2:": P2, "R0_rect": R, "Tr_velo_to_cam": TR})
    assert np.allclose(a.velo_to_cam_mat, b.velo_to_cam_mat)


def test_from_dict_missing_key():
    with pytest.raises(ValueError):
        Calibration.from_dict({"P2": P2})