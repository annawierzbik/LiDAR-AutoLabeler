import numpy as np
import pytest

from ingestion.folder import FolderSequence, folder_key, frame_files, validate_folder


def write_cloud(path, n=50):
    np.random.default_rng(0).uniform(-10, 10, (n, 4)).astype(np.float32).tofile(path)


def test_frames_are_sorted_naturally_and_get_index_ids(tmp_path):
    for name in ("f10.bin", "f2.bin", "f1.bin"):
        write_cloud(tmp_path / name)
    assert [p.name for p in frame_files(tmp_path)] == ["f1.bin", "f2.bin", "f10.bin"]
    seq = FolderSequence(tmp_path)
    assert len(seq) == 3 and [seq[i].frame_id for i in range(3)] == [0, 1, 2]
    assert seq[0].xyz.shape == (50, 3) and seq[0].calib == {}


def test_numeric_file_names_keep_their_number_as_frame_id(tmp_path):
    for i in (3, 4):
        write_cloud(tmp_path / f"{i:06d}.bin")
    assert [FolderSequence(tmp_path)[i].frame_id for i in range(2)] == [3, 4]


def test_validate_reports_readable_problems(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .bin"):
        validate_folder(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a folder"):
        validate_folder(tmp_path / "missing")
    (tmp_path / "bad.bin").write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError):
        validate_folder(tmp_path)


def test_parent_of_sequences_gets_a_helpful_message(tmp_path):
    (tmp_path / "0000").mkdir()
    write_cloud(tmp_path / "0000" / "000000.bin")
    with pytest.raises(ValueError, match="Choose one of them"):
        validate_folder(tmp_path)


def test_folder_key_differs_for_same_name_in_different_places(tmp_path):
    a, b = tmp_path / "x" / "0000", tmp_path / "y" / "0000"
    a.mkdir(parents=True); b.mkdir(parents=True)
    assert folder_key(a) != folder_key(b) and folder_key(a).startswith("0000_")