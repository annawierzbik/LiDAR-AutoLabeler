import pytest
import numpy as np
from pathlib import Path

from ingestion.sequence import SequenceLoader
from ingestion.frame import Frame


def create_mock_kitti_sequence(data_root: Path, split: str, sequence: str, frame_ids: list):
    """Helper function to create a mock KITTI sequence structure on disk for testing."""
    # 1. Create velodyne directory and .bin files
    velodyne_dir = data_root / split / "velodyne" / sequence
    velodyne_dir.mkdir(parents=True, exist_ok=True)

    for f_id in frame_ids:
        bin_path = velodyne_dir / f"{f_id:06d}.bin"
        # 4 floats per point: x, y, z, intensity
        data = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
        data.tofile(bin_path)

    # 2. Create calibration file
    calib_dir = data_root / split / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    calib_file = calib_dir / f"{sequence}.txt"
    calib_file.write_text("P2: 1 2 3 4 5 6 7 8 9 10 11 12\n")

    # 3. Create tracking label file
    label_dir = data_root / split / "label_02"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_file = label_dir / f"{sequence}.txt"
    # Format line: frame track_id type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y
    label_content = (
        f"{frame_ids[0]} 1 Car 0.0 0 -10.0 10 20 30 40 1.5 0.5 2.0 10.0 20.0 30.0 1.57\n"
    )
    label_file.write_text(label_content)


def test_sequence_loader_initialization_and_len(tmp_path):
    """Test successful initialization, directory scanning, and length property."""
    split = "training"
    sequence = "0000"
    create_mock_kitti_sequence(tmp_path, split, sequence, frame_ids=[0, 1, 2])

    loader = SequenceLoader(data_root=tmp_path, split=split, sequence=sequence)

    assert len(loader) == 3
    assert loader.sequence == sequence
    assert loader.split == split


def test_sequence_loader_missing_velodyne_dir_raises_error(tmp_path):
    """Test that FileNotFoundError is raised if velodyne directory does not exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        SequenceLoader(data_root=tmp_path, split="training", sequence="9999")

    assert "Directory with point clouds not found" in str(exc_info.value)


def test_sequence_loader_empty_velodyne_dir_raises_error(tmp_path):
    """Test that FileNotFoundError is raised if velodyne directory has no .bin files."""
    split = "training"
    sequence = "0001"
    velodyne_dir = tmp_path / split / "velodyne" / sequence
    velodyne_dir.mkdir(parents=True, exist_ok=True)  # Directory exists, but is empty

    with pytest.raises(FileNotFoundError) as exc_info:
        SequenceLoader(data_root=tmp_path, split=split, sequence=sequence)

    assert "does not contain .bin files" in str(exc_info.value)


def test_sequence_loader_getitem_valid(tmp_path):
    """Test retrieving frames by positive and negative indices."""
    split = "training"
    sequence = "0000"
    create_mock_kitti_sequence(tmp_path, split, sequence, frame_ids=[10, 20])

    loader = SequenceLoader(data_root=tmp_path, split=split, sequence=sequence)

    # Test positive index
    frame_0 = loader[0]
    assert frame_0.frame_id == 10
    assert frame_0.sequence_id == sequence
    assert frame_0.num_points == 1
    assert len(frame_0.labels_gt) == 1  # Frame 10 has labels in mock data

    # Test negative index (last element)
    frame_last = loader[-1]
    assert frame_last.frame_id == 20
    assert frame_last.labels_gt == []  # Frame 20 has no labels in mock data


def test_sequence_loader_getitem_out_of_bounds(tmp_path):
    """Test that IndexError is raised for out-of-bounds indices."""
    split = "training"
    sequence = "0000"
    create_mock_kitti_sequence(tmp_path, split, sequence, frame_ids=[0, 1])

    loader = SequenceLoader(data_root=tmp_path, split=split, sequence=sequence)

    with pytest.raises(IndexError):
        _ = loader[5]

    with pytest.raises(IndexError):
        _ = loader[-3]


def test_sequence_loader_iteration(tmp_path):
    """Test iterating through all frames in the sequence loader."""
    split = "training"
    sequence = "0000"
    expected_frame_ids = [5, 6, 7]
    create_mock_kitti_sequence(tmp_path, split, sequence, frame_ids=expected_frame_ids)

    loader = SequenceLoader(data_root=tmp_path, split=split, sequence=sequence)

    frames = list(loader)
    assert len(frames) == 3
    for frame, expected_id in zip(frames, expected_frame_ids):
        assert frame.frame_id == expected_id
        assert frame.sequence_id == sequence