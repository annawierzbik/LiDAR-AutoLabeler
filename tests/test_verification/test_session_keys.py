"""Session and cache file names include the KITTI split; files named without it still resume."""
import json
import os
from argparse import Namespace

import numpy as np
import pytest

import scripts.track_sequence
import scripts.verify
import verification.paths
from common.boxes import BoundingBox3D
from scripts.track_sequence import save_results
from scripts.verify import build_session, file_names
from verification import cache
from verification.session import AnnotationSession


def box(x, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


@pytest.fixture
def kitti(tmp_path, monkeypatch):
    """data/raw/<split>/velodyne/0000 with 4 frames for both splits; results in a temp dir."""
    root = tmp_path / "raw"
    for split in ("training", "testing"):
        vel = root / split / "velodyne" / "0000"
        vel.mkdir(parents=True)
        for i in range(4):
            np.zeros((5, 4), np.float32).tofile(vel / f"{i:06d}.bin")
    results = tmp_path / "results"
    monkeypatch.setattr(verification.paths, "RESULTS_DIR", results)
    monkeypatch.setattr(scripts.verify, "RESULTS_DIR", results)
    monkeypatch.setattr(cache, "_warned", set())
    calls = []
    monkeypatch.setattr(scripts.track_sequence, "track_sequence",
                        lambda seq, idx, th, fov="auto": calls.append(seq.split) or [[box(50 + i)] for i in idx])
    return Namespace(root=root, results=results, calls=calls)


def args(root, **kw):
    base = {"folder": None, "data_root": str(root), "split": "training", "sequence": "0000", "mode": "manual",
            "start": 0, "stop": None, "threshold": 0.5, "fps": 10.0, "fresh": False, "fov": "none",
            "use_cache_without_weights": False}
    return Namespace(**{**base, **kw})


def test_file_names_include_the_split(kitti):
    assert file_names(args(kitti.root)) == ("0000", "training_0000", "0000")
    assert file_names(args(kitti.root, split="testing")) == ("0000", "testing_0000", None)


def test_new_sessions_use_the_split_in_the_file_name_but_not_in_the_sequence(kitti):
    s, _, path = build_session(args(kitti.root))
    assert path.name == "training_0000_manual.json" and s.sequence == "0000"
    _, _, path = build_session(args(kitti.root, split="testing"))
    assert path.name == "testing_0000_manual.json"


def test_old_session_file_still_resumes_for_training(kitti):
    old = kitti.results / "sessions" / "0000_manual.json"
    s = AnnotationSession.from_tracking("0000", [[] for _ in range(4)], "manual")
    s.add_manual(1, box(3.0))
    s.save(old)
    resumed, _, path = build_session(args(kitti.root))
    assert resumed.stats()["manual"] == 1
    assert path.name == "training_0000_manual.json"                  # saved under the new name from now on
    fresh, _, _ = build_session(args(kitti.root, split="testing"))   # never for another split
    assert fresh.stats()["manual"] == 0


def test_new_file_wins_over_the_old_one(kitti):
    for name, n in (("0000_manual.json", 1), ("training_0000_manual.json", 2)):
        s = AnnotationSession.from_tracking("0000", [[] for _ in range(4)], "manual")
        for i in range(n):
            s.add_manual(i, box(3.0))
        s.save(kitti.results / "sessions" / name)
    assert build_session(args(kitti.root))[0].stats()["manual"] == 2


def test_old_cache_name_is_read_for_training_only(kitti):
    save_results(kitti.results / "0000.json", range(4), [[box(10 + i)] for i in range(4)])   # no metadata, fov none
    s, _, _ = build_session(args(kitti.root, mode="assisted"))
    assert s.annotations(0)[0].box.x == 10 and kitti.calls == []
    t, _, _ = build_session(args(kitti.root, mode="assisted", split="testing"))
    assert t.annotations(0)[0].box.x == 50 and kitti.calls == ["testing"]
    assert (kitti.results / "testing_0000.json").exists()


def test_cache_files_are_written_atomically(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    save_results(path, range(1), [[box(1)]])
    before = path.read_text()

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        save_results(path, range(1), [[box(2)]])
    assert path.read_text() == before and [p.name for p in tmp_path.iterdir()] == ["c.json"]
    assert json.loads(before)["frames"][0][0]["x"] == 1
