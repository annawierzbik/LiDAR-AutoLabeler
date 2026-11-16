import numpy as np
import pytest

from common.boxes import BoundingBox3D
from infra.points import LoaderPointSource
from infra.sessions import FileSessionStore
from verification.session import AnnotationSession
from web.service import ReviewService, SessionClosed, UnknownAction


class FakeFrame:
    def __init__(self, n):
        rng = np.random.default_rng(n)
        self.xyz = rng.uniform(-20, 20, (50, 3))


class FakeLoader:
    def __getitem__(self, i):
        return FakeFrame(i)

    def __len__(self):
        return 3


def box(x, tid, label="Car"):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, label, 0.9, tid)


def make_service(tmp_path, results=None, key="s"):
    results = results or [[box(10, 0), box(30, 1, "Pedestrian")], [box(11, 0), box(31, 1, "Pedestrian")],
                          [box(12, 0)]]
    s = AnnotationSession.from_tracking("0000", results)
    return ReviewService(s, FileSessionStore(tmp_path), key, LoaderPointSource(FakeLoader()), sequence=FakeLoader())


@pytest.fixture
def svc(tmp_path):
    return make_service(tmp_path)


def statuses(state):
    return [a["status"] for a in state["annotations"]]


def test_state_of_a_frame(svc):
    st = svc.state(0)
    assert st["frame"] == 0 and st["num_frames"] == 3 and st["revision"] == 0
    assert [(a["i"], a["label"]) for a in st["annotations"]] == [(0, "Car"), (1, "Pedestrian")]
    assert st["stats"]["total"] == 5 and set(st["classes"]) == {"Car", "Pedestrian", "Cyclist"}
    assert svc.state(2)["frame"] == 2 and len(svc.state(2)["annotations"]) == 1
    assert "selected" not in st and "current_class" not in st and "show_rejected" not in st
    with pytest.raises(ValueError, match="out of range"):
        svc.state(3)


def test_actions_name_their_target(svc):
    st = svc.act("accept", frame=1, index=1)
    assert st["frame"] == 1 and statuses(st) == ["pending", "accepted"]
    assert statuses(svc.state(0)) == ["pending", "pending"]          # no other frame touched


def test_box_actions_need_frame_and_index(svc):
    with pytest.raises(ValueError, match="frame"):
        svc.act("accept", index=0)
    with pytest.raises(ValueError, match="index"):
        svc.act("accept", frame=0)
    with pytest.raises(ValueError, match="no box 5"):
        svc.act("reject", frame=0, index=5)
    with pytest.raises(UnknownAction):
        svc.act("select", frame=0, index=0)                          # selection is the browser's business


def test_nudge_and_resize(svc):
    st = svc.act("forward", frame=0, index=0)
    a = st["annotations"][0]
    assert a["x"] == pytest.approx(10.1) and a["status"] == "edited" and st["result"] == {"changed": True}
    st = svc.act("length_plus", frame=0, index=0)
    assert st["annotations"][0]["dx"] == pytest.approx(4.1)


def test_track_actions_and_undo_redo(svc):
    st = svc.act("accept_track", frame=0, track_id=0)
    assert st["result"] == {"count": 3} and st["stats"]["accepted"] == 3
    with pytest.raises(ValueError, match="track_id"):
        svc.act("reject_track", frame=0)
    assert svc.act("undo", frame=0)["stats"]["accepted"] == 0
    assert svc.act("redo", frame=0)["stats"]["accepted"] == 3


def test_add_and_copy_next_report_where_the_box_is(svc):
    st = svc.act("add", frame=0, label="Cyclist")
    assert st["result"] == {"frame": 0, "index": 2}
    assert st["annotations"][2]["status"] == "manual" and st["annotations"][2]["label"] == "Cyclist"
    st = svc.act("copy_next", frame=0, index=2)
    assert st["frame"] == 1 and st["result"]["frame"] == 1
    assert st["annotations"][st["result"]["index"]]["label"] == "Cyclist"
    assert svc.act("copy_next", frame=2, index=0)["result"] == {"copied": False}   # last frame


def test_labels_are_validated(svc):
    st = svc.act("set_label", frame=0, index=0, label="Cyclist")
    assert st["annotations"][0]["label"] == "Cyclist"
    with pytest.raises(ValueError, match="unknown class"):
        svc.act("set_label", frame=0, index=0, label="Spaceship")
    with pytest.raises(ValueError, match="unknown class"):
        svc.act("add", frame=0)


def test_rejected_boxes_stay_in_the_state(svc):
    st = svc.act("reject", frame=0, index=0)
    assert statuses(st) == ["rejected", "pending"]                    # the browser decides what to show


def test_view_is_logged_but_changes_nothing(svc, tmp_path):
    n = len(svc.session.events)
    st = svc.act("view", frame=2)
    assert st["frame"] == 2 and st["revision"] == 0
    assert svc.session.events[n].action == "frame_change" and svc.session.events[n].frame == 2
    assert not (tmp_path / "s.json").exists()                         # nothing to save


def test_next_pending(svc):
    assert svc.next_pending(0) == 1
    svc.act("accept_track", frame=0, track_id=0)
    svc.act("accept_track", frame=0, track_id=1)
    assert svc.next_pending(0) is None


def test_points_bytes_and_bounds(svc):
    data = svc.points(1)
    assert len(data) == 50 * 3 * 4
    assert np.frombuffer(data, "<f4").shape == (150,)
    with pytest.raises(IndexError):
        svc.points(3)


def test_autosave_on_change_only(svc, tmp_path):
    svc.state(1)
    assert not (tmp_path / "s.json").exists()
    svc.act("accept", frame=0, index=0)
    loaded = AnnotationSession.load(tmp_path / "s.json")
    assert loaded.stats()["accepted"] == 1 and loaded.revision == 1


def test_failed_autosave_is_reported_in_state(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")                          # a file where the session folder should be
    s = AnnotationSession.from_tracking("0000", [[box(10, 0)]])
    svc = ReviewService(s, FileSessionStore(blocker), "s", LoaderPointSource(FakeLoader()))
    assert svc.state(0)["save_error"] is None
    st = svc.act("accept", frame=0, index=0)
    assert st["save_error"] and "s" in st["save_error"]
    svc.store = FileSessionStore(tmp_path)           # once saving works again the message goes away
    assert svc.act("undo", frame=0)["save_error"] is None


def test_conflicting_writer_is_reported_not_overwritten(svc, tmp_path):
    svc.act("accept", frame=0, index=0)                              # stored revision 1
    other = make_service(tmp_path)                                   # a stale copy (revision 0)
    st = other.act("reject", frame=1, index=0)
    assert "changed elsewhere" in st["save_error"]
    assert AnnotationSession.load(tmp_path / "s.json").stats()["accepted"] == 1


def test_closed_service_refuses_actions(svc):
    svc.close()
    with pytest.raises(SessionClosed, match="closed"):
        svc.act("view", frame=0)


def test_frame_marks_follow_review_progress(svc):
    assert svc.state(0)["frame_marks"] == [1, 1, 1]
    svc.act("accept_track", frame=0, track_id=0)
    st = svc.act("accept_track", frame=0, track_id=1)
    assert st["frame_marks"] == [2, 2, 2]
