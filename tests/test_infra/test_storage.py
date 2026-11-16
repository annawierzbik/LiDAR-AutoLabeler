"""infra/: blob store, session store, settings, point source."""
import json
import os

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from infra.blob import BlobStore, LocalBlobStore, atomic_write_bytes
from infra.points import BlobPointSource, point_source
from infra.sessions import (
    FileSessionStore,
)
from infra.settings import PROJECT_ROOT, Settings
from verification.session import AnnotationSession


def box(x, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


# ------------------------------------------------------------------ blobs
def test_local_blob_store_round_trip(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    assert isinstance(blobs, BlobStore)
    blobs.put("cache/a.json", b"1")
    blobs.put("b.bin", b"22")
    assert blobs.get("cache/a.json") == b"1" and blobs.exists("b.bin") and not blobs.exists("c")
    assert blobs.list() == ["b.bin", "cache/a.json"] and blobs.list("cache/") == ["cache/a.json"]
    with pytest.raises(KeyError):
        blobs.get("missing")


@pytest.mark.parametrize("key", ["../x", "/abs", "a/../../b", "C:/x", "a\\b", ""])
def test_blob_keys_cannot_leave_the_root(tmp_path, key):
    with pytest.raises(ValueError):
        LocalBlobStore(tmp_path).put(key, b"x")


def test_put_is_atomic_when_the_write_fails(tmp_path, monkeypatch):
    blobs = LocalBlobStore(tmp_path)
    blobs.put("a", b"old")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        blobs.put("a", b"new")
    assert blobs.get("a") == b"old" and blobs.list() == ["a"]        # no temp file left behind


def test_atomic_write_creates_folders(tmp_path):
    atomic_write_bytes(tmp_path / "x" / "y.json", b"{}")
    assert (tmp_path / "x" / "y.json").read_bytes() == b"{}"


def test_blob_point_source_reads_frames_like_the_loader(tmp_path):
    from ingestion.folder import FolderSequence

    pts = [np.random.default_rng(i).uniform(-5, 5, (7, 4)).astype("<f4") for i in range(3)]
    for i, p in enumerate(pts):
        p.tofile(tmp_path / f"frame{i * 5}.bin")                     # natural order: 0, 5, 10
    seq = FolderSequence(tmp_path)
    src = point_source(seq)
    assert isinstance(src, BlobPointSource) and len(src) == 3
    for i in range(3):
        assert np.array_equal(src.xyz(i), seq[i].xyz)
    with pytest.raises(IndexError):
        src.xyz(3)


# --------------------------------------------------------------- sessions
# load / create / save / events / locks: tests/test_infra/test_session_store_contract.py (both stores)
@pytest.fixture
def store(tmp_path):
    return FileSessionStore(tmp_path)


def test_store_keys_are_plain_names(store, tmp_path):
    with pytest.raises(ValueError):
        store.path("../x")
    assert store.key_for(tmp_path / "abc.json") == "abc"
    with pytest.raises(ValueError):
        store.key_for(tmp_path / "sub" / "abc.json")


# --------------------------------------------------------------- settings
def test_settings_defaults_reproduce_the_old_layout(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LAL_"):
            monkeypatch.delenv(k)
    s = Settings()
    assert s.results_dir == PROJECT_ROOT / "results" and s.session_dir == PROJECT_ROOT / "results" / "sessions"
    assert (s.host, s.port, s.point_cache_frames) == ("127.0.0.1", 8000, 8)


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LAL_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("LAL_PORT", "9001")
    monkeypatch.setenv("LAL_POINT_CACHE_FRAMES", "3")
    s = Settings()
    assert s.results_dir == tmp_path and s.port == 9001 and s.point_cache_frames == 3
    monkeypatch.setenv("LAL_RESULTS_DIR", "relative/res")
    assert Settings().results_dir == PROJECT_ROOT / "relative" / "res"


def test_session_files_are_json_v2_with_their_events(store):
    s = AnnotationSession.from_tracking("0000", [[box(10)], [box(11)]])
    store.create("a", s)
    data = json.loads(store.path("a").read_text())
    assert data["version"] == 2 and data["events"] and "revision" in data


def test_list_walks_only_matching_folders(tmp_path, monkeypatch):
    blobs = LocalBlobStore(tmp_path)
    for key in ("a_1.json", "a_2.json", "b.json", "sub/a_3.json", "big/x.bin", "big/deep/y.bin"):
        blobs.put(key, b"x")
    walked = []
    real = os.scandir
    monkeypatch.setattr(os, "scandir", lambda p: walked.append(str(p)) or real(p))
    assert blobs.list("a_") == ["a_1.json", "a_2.json"]
    assert not any("big" in w or "sub" in w for w in walked)           # unrelated folders are not scanned
    assert blobs.list("sub/") == ["sub/a_3.json"] and blobs.list("big/deep/") == ["big/deep/y.bin"]


def test_blob_sequence_matches_the_loaders(tmp_path):
    from argparse import Namespace

    from infra.sequences import open_blob_sequence
    from ingestion.folder import FolderSequence
    from ingestion.sequence import SequenceLoader

    vel = tmp_path / "raw" / "training" / "velodyne" / "0003"
    vel.mkdir(parents=True)
    for i in range(3):
        np.random.default_rng(i).uniform(-5, 5, (9, 4)).astype("<f4").tofile(vel / f"{i:06d}.bin")
    calib = tmp_path / "raw" / "training" / "calib"
    calib.mkdir()
    (calib / "0003.txt").write_text("P2: " + " ".join(["1"] * 12) + "\nR_rect " + " ".join(["0"] * 9) + "\n")
    kitti = Namespace(folder=None, data_root=str(tmp_path / "raw"), split="training", sequence="0003")
    a, b = open_blob_sequence(kitti), SequenceLoader(tmp_path / "raw", "training", "0003")
    assert len(a) == len(b) == 3
    for i in range(3):
        assert np.array_equal(a[i].points, b[i].points) and a[i].frame_id == b[i].frame_id
        assert a[i].calib.keys() == b[i].calib.keys()
        assert all(np.array_equal(a[i].calib[k], b[i].calib[k]) for k in b[i].calib)
    folder = Namespace(folder=str(vel), data_root="", split="", sequence="")
    c, d = open_blob_sequence(folder), FolderSequence(vel)
    assert [c[i].frame_id for i in range(3)] == [d[i].frame_id for i in range(3)] and c[0].calib == {}
