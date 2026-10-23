import pytest
import numpy as np

from ingestion.loader import load_point_cloud, parse_calib_file, parse_tracking_labels


def test_load_point_cloud_success(tmp_path):
    """Test successful loading and reshaping of a binary point cloud file."""
    bin_file = tmp_path / "test_frame.bin"
    # 8 floats -> 2 points with (x, y, z, intensity) using 0.5 (exact in binary float)
    data = np.array([1.0, 2.0, 3.0, 0.5, 5.0, 6.0, 7.0, 0.25], dtype=np.float32)
    data.tofile(bin_file)

    points = load_point_cloud(bin_file)

    assert points.shape == (2, 4)
    # Using np.testing.assert_allclose or specifying dtype for comparison
    np.testing.assert_allclose(points[0], [1.0, 2.0, 3.0, 0.5], rtol=1e-5)
    np.testing.assert_allclose(points[1], [5.0, 6.0, 7.0, 0.25], rtol=1e-5)
    

def test_load_point_cloud_file_not_found():
    """Test that FileNotFoundError is raised when the binary file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_point_cloud("non_existent_file.bin")


def test_load_point_cloud_invalid_size(tmp_path):
    """Test that ValueError is raised when data size is not divisible by 4."""
    bin_file = tmp_path / "invalid_frame.bin"
    # 5 floats -> not divisible by 4
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    data.tofile(bin_file)

    with pytest.raises(ValueError) as exc_info:
        load_point_cloud(bin_file)

    assert "is not divisible by 4" in str(exc_info.value)


def test_parse_calib_file(tmp_path):
    """Test parsing KITTI calibration file with 12, 9, and arbitrary element sizes."""
    calib_file = tmp_path / "calib.txt"
    content = (
        "P2: 1 2 3 4 5 6 7 8 9 10 11 12\n"
        "R_rect: 1 0 0 0 1 0 0 0 1\n"
        "CustomKey: 42 84\n"
    )
    calib_file.write_text(content)

    calib = parse_calib_file(calib_file)

    assert "P2" in calib
    assert calib["P2"].shape == (3, 4)

    assert "R_rect" in calib
    assert calib["R_rect"].shape == (3, 3)

    assert "CustomKey" in calib
    assert calib["CustomKey"].shape == (2,)
    assert np.array_equal(calib["CustomKey"], [42.0, 84.0])


def test_parse_tracking_labels(tmp_path):
    """Test parsing KITTI tracking labels file grouped by frame IDs."""
    label_file = tmp_path / "0000.txt"
    # Format: frame track_id type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y
    content = (
        "0 1 Car 0.0 0 -10.0 10 20 30 40 1.5 0.5 2.0 10.0 20.0 30.0 1.57\n"
        "0 2 Pedestrian 0.0 1 -5.0 5 10 15 25 1.0 0.4 1.8 5.0 10.0 15.0 0.0\n"
        "1 1 Car 0.0 0 -9.0 11 21 31 41 1.5 0.5 2.0 10.1 20.1 30.1 1.58\n"
    )
    label_file.write_text(content)

    labels = parse_tracking_labels(label_file)

    assert 0 in labels
    assert 1 in labels
    assert len(labels[0]) == 2
    assert len(labels[1]) == 1

    car_entry = labels[0][0]
    assert car_entry["track_id"] == 1
    assert car_entry["type"] == "Car"
    assert car_entry["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert car_entry["dimensions"] == [1.5, 0.5, 2.0]
    assert car_entry["location"] == [10.0, 20.0, 30.0]
    assert car_entry["rotation_y"] == 1.57