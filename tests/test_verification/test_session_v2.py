"""Session file v2: revision and a bounded undo/redo history survive a reload; v1 files still load."""
import json

import pytest

from common.boxes import BoundingBox3D
from verification import session as session_module
from verification.session import UNDO_LIMIT, AnnotationSession, Status


def box(x, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


def make():
    return AnnotationSession.from_tracking("0000", [[box(10)], [box(11)], [box(12)]])


def test_undo_after_reload(tmp_path):
    s = make()
    s.accept(0, 0)
    s.edit(1, 0, x=20.0)
    s.save(tmp_path / "s.json")
    r = AnnotationSession.load(tmp_path / "s.json")
    assert r.revision == s.revision == 2
    assert r.undo() == "edit" and r.annotations(1)[0].box.x == 11
    assert r.undo() == "accept" and r.annotations(0)[0].status is Status.PENDING
    assert r.undo() is None
    assert r.redo() == "accept" and r.annotations(0)[0].status is Status.ACCEPTED


def test_redo_history_survives_a_reload(tmp_path):
    s = make()
    s.reject(2, 0)
    s.undo()
    s.save(tmp_path / "s.json")
    r = AnnotationSession.load(tmp_path / "s.json")
    assert r.redo() == "reject" and r.annotations(2)[0].status is Status.REJECTED


def test_history_is_bounded(tmp_path):
    s = make()
    for i in range(UNDO_LIMIT + 7):
        s.edit(0, 0, x=10.0 + i + 1)
    assert len(s._undo) == UNDO_LIMIT
    s.save(tmp_path / "s.json")
    data = json.loads((tmp_path / "s.json").read_text())
    assert data["version"] == 2 and len(data["undo"]) == UNDO_LIMIT and data["revision"] == UNDO_LIMIT + 7
    r = AnnotationSession.load(tmp_path / "s.json")
    for _ in range(UNDO_LIMIT):
        assert r.undo() == "edit"
    assert r.undo() is None and r.annotations(0)[0].box.x == 17.0    # the 7 oldest steps are gone


def test_v1_files_load_with_revision_zero_and_no_history(tmp_path):
    s = make()
    s.accept(0, 0)
    s.save(tmp_path / "s.json")
    data = json.loads((tmp_path / "s.json").read_text())
    for k in ("revision", "undo", "redo"):
        del data[k]
    data["version"] = 1
    (tmp_path / "v1.json").write_text(json.dumps(data))
    r = AnnotationSession.load(tmp_path / "v1.json")
    assert r.revision == 0 and r.undo() is None and r.annotations(0)[0].status is Status.ACCEPTED
    r.save(tmp_path / "v1.json")
    assert json.loads((tmp_path / "v1.json").read_text())["version"] == 2


def test_unknown_versions_are_rejected(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError, match="unsupported"):
        AnnotationSession.load(tmp_path / "x.json")


def test_save_is_atomic(tmp_path, monkeypatch):
    s = make()
    s.save(tmp_path / "s.json")
    before = (tmp_path / "s.json").read_text()
    s.accept(0, 0)

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(session_module, "atomic_write_text", broken)
    with pytest.raises(OSError):
        s.save(tmp_path / "s.json")
    assert (tmp_path / "s.json").read_text() == before


def test_fast_serialisation_writes_exactly_what_asdict_wrote(tmp_path):
    """Phase 2b: Annotation.to_dict builds the dict directly; the file format must not change."""
    import dataclasses
    import random

    rng = random.Random(0)
    s = AnnotationSession.from_tracking("0000", [[BoundingBox3D(rng.uniform(-50, 50), rng.uniform(-20, 20), -1.0,
                                                                rng.uniform(0.5, 5), 1.8, 1.5, rng.uniform(-3, 3),
                                                                rng.choice(["Car", "Pedestrian"]), rng.random(),
                                                                rng.choice([None, 3, 7]))
                                                  for _ in range(5)] for _ in range(4)])
    s.accept(0, 0)
    for anns in s.frames.values():
        for a in anns:
            assert a.to_dict() == {"box": dataclasses.asdict(a.box), "status": a.status.value}
            assert json.dumps(a.to_dict()) == json.dumps({"box": dataclasses.asdict(a.box), "status": a.status.value})
    s.save(tmp_path / "s.json")
    again = AnnotationSession.load(tmp_path / "s.json")
    assert again.to_dict(include_events=False) == s.to_dict(include_events=False)
