from argparse import Namespace

import numpy as np
import pytest

import scripts.verify
import verification.paths
from common.boxes import BoundingBox3D
from ingestion.folder import folder_key
from scripts.track_sequence import save_results
from scripts.verify import build_session


def box(x, tid):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


@pytest.fixture
def clouds(tmp_path, monkeypatch):
    """A folder with 10 tiny frames; RESULTS_DIR points to a fresh temp dir (not the project's results/)."""
    folder = tmp_path / "clouds"
    folder.mkdir()
    for i in range(10):
        np.zeros((5, 4), np.float32).tofile(folder / f"{i:06d}.bin")
    results = tmp_path / "results"
    monkeypatch.setattr(verification.paths, "RESULTS_DIR", results)
    monkeypatch.setattr(scripts.verify, "RESULTS_DIR", results)
    return folder


def args(folder, **kw):
    base = dict(folder=str(folder), data_root="data/raw", split="training", sequence="0000", mode="assisted",
                start=0, stop=None, threshold=0.5, fps=10.0, fresh=False)
    return Namespace(**{**base, **kw})


def test_manual_mode_starts_empty_without_running_a_detector(clouds):
    session, _seq, path = build_session(args(clouds, mode="manual", start=2, stop=6))
    assert session.num_frames == 4 and session.frame_offset == 2 and session.stats()["total"] == 0
    assert path.name.endswith("_f2-6_manual.json")
    assert path.parent == verification.paths.RESULTS_DIR / "sessions"


def test_results_dir_does_not_depend_on_the_working_directory():
    from verification.paths import PROJECT_ROOT, RESULTS_DIR
    assert RESULTS_DIR.is_absolute() and RESULTS_DIR == PROJECT_ROOT / "results"
    assert (PROJECT_ROOT / "verification" / "paths.py").exists()


def test_assisted_mode_slices_cached_detections(clouds):
    key = folder_key(clouds)
    results = [[box(10 + i, 0)] for i in range(10)]
    save_results(verification.paths.RESULTS_DIR / f"{key}.json", range(10), results)
    session, _, _ = build_session(args(clouds, start=3, stop=7))
    assert session.num_frames == 4 and session.frame_offset == 3
    assert session.annotations(0)[0].box.x == 13 and session.annotations(3)[0].box.x == 16


def test_empty_range_is_an_error(clouds):
    with pytest.raises(ValueError, match="Empty frame range"):
        build_session(args(clouds, mode="manual", start=20, stop=30))


def test_saved_session_is_resumed(clouds):
    session, _, path = build_session(args(clouds, mode="manual", start=0, stop=5))
    session.add_manual(0, box(5, None))
    session.save(path)
    again, _, _ = build_session(args(clouds, mode="manual", start=0, stop=5))
    assert again.stats()["manual"] == 1