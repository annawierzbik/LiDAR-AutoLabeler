import math

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from common.geometry import normalize_angle
from export import ExportError, export_session, write_files, zip_files
from export.formats import render_kitti_tracking, render_lidar, to_kitti
from ingestion.calibration import Calibration
from ingestion.folder import FolderSequence
from ingestion.loader import parse_tracking_labels
from verification.controller import ReviewController
from verification.session import AnnotationSession

# Rotation part of the usual KITTI Tr_velo_cam: x_cam = -y_velo, y_cam = -z_velo, z_cam = x_velo.
CALIB_TEXT = (
    "P2: 721.5377 0 609.5593 44.85728 0 721.5377 172.854 0.2163791 0 0 1 0.002745884\n"
    "R_rect 1 0 0 0 1 0 0 0 1\n"
    "Tr_velo_cam 0 -1 0 -0.004 0 0 -1 -0.07 1 0 0 -0.27\n"
)


@pytest.fixture
def calib(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text(CALIB_TEXT)
    return Calibration.from_file(path)


def box(x, tid, label="Car", y=0.0, heading=0.0, dims=(4.0, 1.8, 1.5)):
    return BoundingBox3D(x, y, -1.0, *dims, heading, label, 0.9, tid)


def make_session(results=None):
    results = results or [[box(10, 0), box(30, 1, "Pedestrian")], [box(11, 0), box(31, 1, "Pedestrian")], [box(12, 0)]]
    return AnnotationSession.from_tracking("0000", results)


def accept_frame(session, frame):
    c = ReviewController(session)
    c.goto(frame)
    for i in range(len(c.annotations())):
        c.select(i)
        c.accept()


STEMS = ["000000", "000001", "000002"]


# ------------------------------------------------------------------------------------ lidar
def test_lidar_writes_only_fully_reviewed_frames():
    s = make_session()
    accept_frame(s, 0)
    files, summary = render_lidar(s, STEMS)
    assert sorted(f for f in files if f.startswith("labels/")) == ["labels/000000.txt"]
    assert summary.frames == 1 and summary.boxes == 2 and summary.skipped_unreviewed == 2
    assert summary.per_class == {"Car": 1, "Pedestrian": 1}
    assert files["ImageSets/exported.txt"] == "000000\n"


def test_lidar_lines_hold_the_box_values():
    s = make_session([[box(10.5, 0, heading=1.25)]])
    accept_frame(s, 0)
    files, _ = render_lidar(s, ["000000"])
    cols = files["labels/000000.txt"].split()
    assert len(cols) == 8 and cols[7] == "Car"
    assert [float(v) for v in cols[:7]] == pytest.approx([10.5, 0.0, -1.0, 4.0, 1.8, 1.5, 1.25], abs=1e-5)


def test_lidar_flags_for_unreviewed_and_empty_frames():
    s = make_session()
    accept_frame(s, 0)
    c = ReviewController(s)
    c.goto(2)
    c.select(0)
    c.reject()                                     # frame 2: reviewed, but nothing left
    files, summary = render_lidar(s, STEMS)
    assert summary.skipped_empty == 1 and "labels/000002.txt" not in files
    files, summary = render_lidar(s, STEMS, include_empty=True)
    assert files["labels/000002.txt"] == "" and summary.frames == 2
    files, summary = render_lidar(s, STEMS, include_unreviewed=True)
    assert summary.frames == 2 and summary.skipped_unreviewed == 0     # frames 0 and 1; frame 2 has no boxes
    assert len(files["labels/000001.txt"].splitlines()) == 2           # pending boxes included on request


# ------------------------------------------------------------------------------------ kitti
def from_kitti(entry, calib):
    """Inverse of to_kitti, written independently from the label values."""
    h, w, l = entry["dimensions"]
    loc = np.array(entry["location"], dtype=float)
    loc[1] -= h / 2
    centre = calib.cam_to_velo(loc)[0]
    ry = entry["rotation_y"]
    d = calib.velo_to_cam_mat[:3, :3].T @ np.array([math.cos(ry), 0.0, -math.sin(ry)])
    return centre, (l, w, h), math.atan2(d[1], d[0])


@pytest.mark.parametrize("x, y, heading", [(12.0, 0.0, 0.0), (20.0, 3.0, 1.0), (15.0, -4.0, -2.4), (35.0, 6.0, 3.0)])
def test_kitti_round_trip_through_the_label_parser(tmp_path, calib, x, y, heading):
    b = box(x, 5, y=y, heading=heading)
    s = make_session([[b]])
    accept_frame(s, 0)
    files, summary = render_kitti_tracking(s, [0], calib, "0000")
    path = tmp_path / "0000.txt"
    path.write_text(files["label_02/0000.txt"])
    entry = parse_tracking_labels(path)[0][0]
    assert entry["track_id"] == 5 and entry["type"] == "Car"
    centre, dims, heading_back = from_kitti(entry, calib)
    assert centre == pytest.approx([x, y, -1.0], abs=1e-3)
    assert dims == pytest.approx((4.0, 1.8, 1.5), abs=1e-5)
    assert normalize_angle(heading_back - heading) == pytest.approx(0.0, abs=1e-5)
    assert summary.frames == 1 and summary.boxes == 1


def test_rotation_matches_the_openpcdet_formula(calib):
    for heading in (-3.0, -1.2, 0.0, 0.7, 2.9):
        k = to_kitti(box(15, 0, heading=heading), calib)
        assert normalize_angle(k.rotation_y - (-heading - math.pi / 2)) == pytest.approx(0.0, abs=1e-9)


def test_kitti_location_is_the_bottom_centre_and_alpha_is_relative_to_the_viewing_ray(calib):
    k = to_kitti(box(10, 0, y=2.0), calib)
    assert k.location[1] == pytest.approx(1.0 - 0.07 + 0.75, abs=1e-6)         # y_cam of the centre (-z_velo + ty) + h/2
    assert (k.h, k.w, k.l) == (1.5, 1.8, 4.0)
    assert k.alpha == pytest.approx(normalize_angle(k.rotation_y - math.atan2(k.location[0], k.location[2])))


def test_visible_box_has_a_2d_box_inside_the_image(calib):
    k = to_kitti(box(15, 0), calib)
    x1, y1, x2, y2 = k.bbox
    assert k.in_view and 0 <= x1 < x2 <= 1242 and 0 <= y1 < y2 <= 375 and k.truncated == 0.0


def test_box_cut_by_the_image_edge_is_truncated(calib):
    k = to_kitti(box(10, 0, y=-8.5), calib)
    assert k.in_view and k.bbox[2] == 1242 and 0.0 < k.truncated < 1.0


def test_boxes_outside_the_camera_view(calib):
    s = make_session([[box(-10, 0), box(15, 1)]])
    accept_frame(s, 0)
    files, summary = render_kitti_tracking(s, [0], calib, "0000")
    assert summary.boxes == 1 and summary.skipped_outside_view == 1
    files, summary = render_kitti_tracking(s, [0], calib, "0000", only_in_view=False)
    assert summary.boxes == 2 and summary.skipped_outside_view == 0
    assert any(" 0.00 0.00 0.00 0.00 " in line for line in files["label_02/0000.txt"].splitlines())


def test_missing_track_ids_get_unique_numbers(calib):
    s = make_session([[box(10, 3), box(20, None), box(25, None, y=2.0)]])
    accept_frame(s, 0)
    files, _ = render_kitti_tracking(s, [0], calib, "0000")
    ids = [int(line.split()[1]) for line in files["label_02/0000.txt"].splitlines()]
    assert sorted(ids) == [3, 4, 5]


def test_rows_are_sorted_and_use_the_kitti_frame_numbers(calib):
    s = make_session([[box(10, 2), box(20, 1)], [box(11, 1)]])
    accept_frame(s, 0)
    accept_frame(s, 1)
    files, summary = render_kitti_tracking(s, [7, 8], calib, "0000")
    rows = [line.split()[:2] for line in files["label_02/0000.txt"].splitlines()]
    assert rows == [["7", "1"], ["7", "2"], ["8", "1"]] and summary.frames == 2


# ------------------------------------------------------------------------------ export_session
def make_bin_folder(tmp_path, names, calib_text=None):
    folder = tmp_path / "seq"
    folder.mkdir()
    for n in names:
        np.zeros((5, 4), "<f4").tofile(folder / n)
    return folder


def test_export_uses_the_file_names_of_the_folder(tmp_path):
    folder = make_bin_folder(tmp_path, ["scan2.bin", "scan10.bin", "scan1.bin"])
    s = make_session()
    for f in range(3):
        accept_frame(s, f)
    result = export_session(s, FolderSequence(folder), "lidar")
    assert sorted(k for k in result.files if k.startswith("labels/")) == [
        "labels/scan1.txt", "labels/scan10.txt", "labels/scan2.txt"]
    assert result.archive_name == "0000_lidar_labels.zip"


def test_export_of_part_of_a_sequence_uses_the_matching_names(tmp_path):
    folder = make_bin_folder(tmp_path, [f"{i:06d}.bin" for i in range(6)])
    s = AnnotationSession.from_tracking("0000", [[box(10, 0)], [box(11, 0)]], frame_offset=3)
    accept_frame(s, 0)
    accept_frame(s, 1)
    result = export_session(s, FolderSequence(folder), "lidar")
    assert sorted(k for k in result.files if k.startswith("labels/")) == ["labels/000003.txt", "labels/000004.txt"]


def test_export_without_a_sequence_folder_falls_back_to_frame_numbers():
    s = AnnotationSession.from_tracking("0000", [[box(10, 0)]], frame_offset=2)
    accept_frame(s, 0)
    result = export_session(s, object(), "lidar")
    assert "labels/000002.txt" in result.files


def test_kitti_export_needs_a_calibration(tmp_path):
    s = make_session()
    with pytest.raises(ExportError, match="calibration"):
        export_session(s, FolderSequence(make_bin_folder(tmp_path, ["000000.bin"])), "kitti")
    with pytest.raises(ExportError, match="unknown export format"):
        export_session(s, object(), "coco")


def test_kitti_export_reads_the_calibration_next_to_the_data(tmp_path):
    folder = make_bin_folder(tmp_path, [f"{i:06d}.bin" for i in range(3)])

    class Loader:
        velodyne_dir = folder
        calib_path = tmp_path / "0000.txt"

    Loader.calib_path.write_text(CALIB_TEXT)
    s = make_session()
    for f in range(3):
        accept_frame(s, f)
    result = export_session(s, Loader(), "kitti")
    assert result.summary.boxes == 5 and "label_02/0000.txt" in result.files
    assert result.archive_name == "0000_kitti_labels.zip"


# ---------------------------------------------------------------------------------- writing
def test_write_files_and_zip(tmp_path):
    import io
    import zipfile

    files = {"labels/a.txt": "1 2\n", "ImageSets/exported.txt": "a\n"}
    written = write_files(files, tmp_path / "out")
    assert (tmp_path / "out" / "labels" / "a.txt").read_text() == "1 2\n" and len(written) == 2
    with zipfile.ZipFile(io.BytesIO(zip_files(files))) as z:
        assert sorted(z.namelist()) == ["ImageSets/exported.txt", "labels/a.txt"]
        assert z.read("labels/a.txt") == b"1 2\n"


def test_write_files_refuses_paths_outside_the_target(tmp_path):
    with pytest.raises(ExportError):
        write_files({"../evil.txt": "x"}, tmp_path / "out")
