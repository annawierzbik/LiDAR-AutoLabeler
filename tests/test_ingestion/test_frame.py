import numpy as np
import pytest

from ingestion.frame import Frame


def test_frame_initialization_valid_3d():
    """Test successful frame initialization with (N, 3) points."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    frame = Frame(sequence_id="seq_001", frame_id=0, points=points)

    assert frame.sequence_id == "seq_001"
    assert frame.frame_id == 0
    assert np.array_equal(frame.points, points)
    assert frame.calib == {}
    assert frame.labels_gt is None


def test_frame_initialization_valid_4d_with_intensity():
    """Test successful frame initialization with (N, 4) points (e.g., including intensity)."""
    points = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.8]])
    frame = Frame(sequence_id="seq_002", frame_id=5, points=points)

    assert frame.num_points == 2
    assert frame.points.shape == (2, 4)


def test_frame_properties():
    """Test num_points and xyz properties."""
    points = np.array([[1.0, 2.0, 3.0, 0.1], [4.0, 5.0, 6.0, 0.2], [7.0, 8.0, 9.0, 0.3]])
    frame = Frame(sequence_id="test_seq", frame_id=1, points=points)

    # Check number of points
    assert frame.num_points == 3

    # Check xyz coordinates extraction (dropping intensity)
    expected_xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    np.testing.assert_array_equal(frame.xyz, expected_xyz)
    assert frame.xyz.shape == (3, 3)


def test_frame_custom_fields():
    """Test optional calib and labels_gt fields."""
    points = np.zeros((10, 3))
    calib_data = {"cam_to_lidar": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    labels = ["car", "pedestrian"]

    frame = Frame(
        sequence_id="seq_03",
        frame_id=10,
        points=points,
        calib=calib_data,
        labels_gt=labels,
    )

    assert frame.calib == calib_data
    assert frame.labels_gt == labels


def test_frame_repr():
    """Test string representation (__repr__) of the frame."""
    points = np.zeros((5, 3))
    frame = Frame(sequence_id="alpha", frame_id=42, points=points)

    expected_repr = "Frame(sequence='alpha', frame_id=42, num_points=5)"
    assert repr(frame) == expected_repr


# --- EDGE CASES / ERROR VALIDATION ---
@pytest.mark.parametrize(
    "invalid_points",
    [
        np.array([1.0, 2.0, 3.0]),  # 1D array (insufficient dimension)
        np.zeros((10, 2)),  # 2D array with too few columns (< 3)
        np.zeros((5, 3, 2)),  # 3D array (excessive dimension)
    ],
)
def test_frame_invalid_points_shape_raises_value_error(invalid_points):
    """Test that invalid points array shapes raise a ValueError."""
    with pytest.raises(ValueError) as exc_info:
        Frame(sequence_id="seq_err", frame_id=0, points=invalid_points)

    assert "Shape of points array is expected to be" in str(exc_info.value)


def test_frame_empty_points_valid_shape():
    """Edge case: Empty frame (0 points) but with a valid shape (0, 3)."""
    points = np.zeros((0, 3))
    frame = Frame(sequence_id="empty_seq", frame_id=0, points=points)

    assert frame.num_points == 0
    assert frame.xyz.shape == (0, 3)