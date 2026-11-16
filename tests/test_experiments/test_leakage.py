import pytest

from experiments.lib import leakage


def _devkit(tmp_path, rand, mapping_lines):
    m = tmp_path / "devkit" / "mapping"
    m.mkdir(parents=True)
    (m / "train_rand.txt").write_text(",".join(str(r) for r in rand) + "\n")
    (m / "train_mapping.txt").write_text("\n".join(mapping_lines) + "\n")
    return tmp_path / "devkit"


def test_mapping_uses_1_based_rand_index(tmp_path):
    lines = ["2011_09_26 drive_A 0000000005", "2011_09_26 drive_B 0000000007", "2011_09_26 drive_A 0000000006"]
    devkit = _devkit(tmp_path, [3, 1, 2], lines)
    assert leakage.read_object_mapping(devkit) == [("drive_A", 6), ("drive_A", 5), ("drive_B", 7)]


def test_mapping_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        leakage.read_object_mapping(_devkit(tmp_path, [1, 2], ["d x 1"]))


def test_content_index_matches_only_identical_bytes(tmp_path):
    obj = tmp_path / "obj"
    obj.mkdir()
    (obj / "000000.bin").write_bytes(b"abcd")
    (obj / "000001.bin").write_bytes(b"wxyz")         # same size, different content
    (obj / "000002.bin").write_bytes(b"longer file")
    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"wxyz")
    index = leakage.ContentIndex(obj)
    assert index.find(probe) == [1]
    probe.write_bytes(b"12345")                       # no file of that size
    assert index.find(probe) == []


def _report(frames_to_obj, mapping, sets, n_frames=10):
    matches = {f: [o] for f, o in frames_to_obj.items()}
    return leakage.sequence_report("0000", n_frames, matches, mapping, sets)


def test_sequence_report_offset_and_splits():
    mapping = [("drive_A", 12), ("drive_A", 15), ("drive_A", 30), ("drive_B", 1)]
    sets = {"train": {0}, "val": {1}}
    rep = _report({2: 0, 5: 1}, mapping, sets)        # raw 12 = frame 2 + 10, raw 15 = frame 5 + 10
    assert rep["raw_drives"] == ["drive_A"] and rep["frame_offsets"] == [10]
    assert rep["consistent_mapping"]
    assert rep["num_frames_in_object_split"] == {"train": 1, "val": 1}
    # object 2 (raw 30) is in drive_A but outside raw frames 10..19, and belongs to no split
    assert rep["same_drive_object_frames_outside_sequence"] == {"none": 1}


def test_inconsistent_offset_is_reported_and_counts_as_contaminated():
    mapping = [("drive_A", 12), ("drive_A", 40)]
    rep = _report({2: 0, 5: 1}, mapping, {"train": set(), "val": {0, 1}})
    assert not rep["consistent_mapping"]
    assert leakage.is_contaminated(rep)


def test_contamination_rules():
    mapping = [("drive_A", 12), ("drive_A", 50)]
    sets = {"train": {1}, "val": {0}}
    rep = _report({2: 0}, mapping, sets)              # in-sequence frame is val, a train frame elsewhere in drive
    assert not leakage.is_contaminated(rep, "frame")
    assert leakage.is_contaminated(rep, "drive")
    with pytest.raises(ValueError):
        leakage.is_contaminated(rep, "scene")


def test_unmatched_sequence_is_clean():
    rep = _report({}, [("drive_A", 1)], {"train": {0}, "val": set()})
    assert rep["raw_drives"] == [] and not leakage.is_contaminated(rep, "drive")


def test_drive_split_overlap():
    mapping = [("A", 1), ("A", 2), ("B", 1)]
    out = leakage.drive_split_overlap(mapping, {"train": {0}, "val": {1, 2}})
    assert out["shared_drives"] == ["A"]
