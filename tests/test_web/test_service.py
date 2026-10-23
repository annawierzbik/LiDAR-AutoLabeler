import numpy as np
import pytest

from common.boxes import BoundingBox3D
from verification.session import AnnotationSession
from web.service import ReviewService


class FakeFrame:
    def __init__(self, n):
        rng = np.random.default_rng(n)
        self.xyz = rng.uniform(-20, 20, (50, 3))


class FakeLoader:
    def __getitem__(self, i):
        return FakeFrame(i)


def box(x, tid, label="Car"):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, label, 0.9, tid)


@pytest.fixture
def svc(tmp_path):
    results = [[box(10, 0), box(30, 1, "Pedestrian")], [box(11, 0), box(31, 1, "Pedestrian")], [box(12, 0)]]
    s = AnnotationSession.from_tracking("0000", results)
    return ReviewService(s, FakeLoader(), tmp_path / "s.json")


def statuses(svc):
    return [a["status"] for a in svc.state()["annotations"]]


def test_state_shape(svc):
    st = svc.state()
    assert st["frame"] == 0 and st["num_frames"] == 3 and st["selected"] is None
    assert [a["label"] for a in st["annotations"]] == ["Car", "Pedestrian"]
    assert st["stats"]["total"] == 5 and set(st["classes"]) == {"Car", "Pedestrian", "Cyclist"}


def test_select_and_accept_advances(svc):
    svc.act("select", index=0)
    st = svc.act("accept")
    assert statuses(svc)[0] == "accepted"
    assert st["selected"] == 1          # jumped to the next box that still needs a decision


def test_goto_next_prev(svc):
    assert svc.act("next")["frame"] == 1
    assert svc.act("goto", frame=2)["frame"] == 2
    assert svc.act("prev")["frame"] == 1
    assert svc.act("goto", frame=99)["frame"] == 2   # clamped


def test_nudge_and_resize_edit_selected(svc):
    svc.act("select", index=0)
    st = svc.act("forward")
    a = st["annotations"][0]
    assert a["x"] == pytest.approx(10.1) and a["status"] == "edited"
    st = svc.act("length_plus")
    assert st["annotations"][0]["dx"] == pytest.approx(4.1)


def test_accept_track_and_undo(svc):
    svc.act("select", index=0)
    svc.act("accept_track")
    assert svc.state()["stats"]["accepted"] == 3
    svc.act("undo")
    assert svc.state()["stats"]["accepted"] == 0


def test_add_and_copy_next(svc):
    svc.act("add")
    assert svc.state()["annotations"][-1]["status"] == "manual"
    st = svc.act("copy_next")
    assert st["frame"] == 1 and st["selected"] is not None


def test_set_class_and_unknown(svc):
    assert svc.act("set_class", label="Cyclist")["current_class"] == "Cyclist"
    with pytest.raises(ValueError):
        svc.act("set_class", label="Spaceship")
    with pytest.raises(KeyError):
        svc.act("no_such_action")


def test_reject_hidden_until_toggle(svc):
    svc.act("select", index=0); svc.act("reject")
    assert len(svc.state()["annotations"]) == 1
    assert len(svc.act("toggle_rejected")["annotations"]) == 2


def test_play_step_is_not_logged(svc):
    n = len(svc.session.events)
    svc.act("play_step")
    assert svc.state()["frame"] == 1 and len(svc.session.events) == n


def test_points_bytes_and_bounds(svc):
    data = svc.points(1)
    assert len(data) == 50 * 3 * 4
    assert np.frombuffer(data, "<f4").shape == (150,)
    with pytest.raises(IndexError):
        svc.points(3)


def test_autosave_on_change_only(svc, tmp_path):
    svc.act("next")
    assert not (tmp_path / "s.json").exists()      # navigation alone changes no annotations
    svc.act("select", index=0); svc.act("accept")
    assert (tmp_path / "s.json").exists()
    assert AnnotationSession.load(tmp_path / "s.json").stats()["accepted"] == 1


def test_failed_autosave_is_reported_in_state(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")                          # a file where the session folder should be
    s = AnnotationSession.from_tracking("0000", [[box(10, 0)]])
    svc = ReviewService(s, FakeLoader(), blocker / "s.json")
    assert svc.state()["save_error"] is None
    svc.act("select", index=0)
    st = svc.act("accept")
    assert st["save_error"] and "s.json" in st["save_error"]
    svc.save_path = tmp_path / "ok.json"             # once saving works again the message goes away
    assert svc.act("undo")["save_error"] is None


def test_closed_service_refuses_actions(svc):
    svc.close()
    with pytest.raises(LookupError, match="closed"):
        svc.act("next")
