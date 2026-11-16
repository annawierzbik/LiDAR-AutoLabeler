"""Point crop before detection (detection/fov.py)."""
from pathlib import Path

import numpy as np
import pytest

from detection.fov import (
    azimuth_mask,
    calibration_of,
    camera_mask,
    crop_points,
    describe,
    resolve_mode,
    sequence_mode,
)
from ingestion.calibration import Calibration
from ingestion.loader import parse_calib_file

KITTI = Path(__file__).resolve().parents[2] / "OpenPCDet" / "data" / "kitti" / "training"


def calib():
    # KITTI-like: camera looks along +x of the LiDAR, image 1242 x 375
    P2 = np.array([[721.5, 0, 609.6, 44.9], [0, 721.5, 172.9, 0.2], [0, 0, 1, 0.003]])
    Tr = np.array([[0, -1, 0, 0], [0, 0, -1, -0.08], [1, 0, 0, -0.27]], float)
    return Calibration(P2, np.eye(3), Tr)


def test_azimuth_mask():
    pts = np.array([[10, 0, 0], [10, 9.9, 0], [10, 10.1, 0], [10, -10.1, 0], [-10, 0, 0]], float)
    assert azimuth_mask(pts, 45.0).tolist() == [True, True, False, False, False]


def test_camera_mask_keeps_points_ahead_inside_the_image():
    c = calib()
    pts = np.array([[20, 0, 0], [20, 30, 0], [-20, 0, 0], [20, 0, 30]], float)
    assert camera_mask(pts, c).tolist() == [True, False, False, False]


def test_resolve_mode():
    assert resolve_mode("auto", True) == "camera"
    assert resolve_mode("auto", False) == "azimuth"
    assert resolve_mode("none", False) == "none"
    with pytest.raises(ValueError, match="needs a calibration"):
        resolve_mode("camera", False)
    with pytest.raises(ValueError, match="unknown FOV mode"):
        resolve_mode("everything", True)


def test_crop_points_modes():
    pts = np.array([[20, 0, 0, 0.5], [20, 30, 0, 0.5], [-20, 0, 0, 0.5]], np.float32)
    assert crop_points(pts, "none")[0].shape == (3, 4)
    kept, mode = crop_points(pts, "auto")                    # no calibration -> azimuth
    assert mode == "azimuth" and len(kept) == 1
    kept, mode = crop_points(pts, "auto", calib())
    assert mode == "camera" and len(kept) == 1 and kept.shape[1] == 4


def test_calibration_from_frame_dicts():
    assert calibration_of({}) is None
    assert calibration_of({"P2": np.zeros(12)}) is None       # incomplete -> treated as missing
    c = calib()
    assert calibration_of(c) is c


def test_describe_and_sequence_mode():
    assert describe("azimuth", 41.5) == "azimuth:41.5"
    assert describe("camera") == "camera"

    class Seq(list):
        pass

    frame = type("F", (), {"calib": {}})()
    assert sequence_mode(Seq([frame]), "auto", 41.5) == "azimuth:41.5"


@pytest.mark.skipif(not (KITTI / "image_2" / "000010.png").exists(), reason="KITTI object data not available")
def test_camera_mask_matches_openpcdet_get_fov_flag():
    pytest.importorskip("torch")
    import torch  # noqa: F401  (before pcdet)
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from pcdet.utils.calibration_kitti import Calibration as PcdetCalibration
    from PIL import Image

    import detection.detector  # noqa: F401  (puts OpenPCDet on sys.path)

    for fid in ("000001", "000010"):
        pts = np.fromfile(KITTI / "velodyne" / f"{fid}.bin", np.float32).reshape(-1, 4)
        pc = PcdetCalibration(KITTI / "calib" / f"{fid}.txt")
        w, h = Image.open(KITTI / "image_2" / f"{fid}.png").size
        ref = KittiDataset.get_fov_flag(pc.lidar_to_rect(pts[:, :3]), (h, w), pc)
        ours = camera_mask(pts, Calibration.from_dict(parse_calib_file(KITTI / "calib" / f"{fid}.txt")), w, h)
        assert 0.1 < ref.mean() < 0.5
        assert np.array_equal(ours, ref)
