import math

import pytest

from common.boxes import BoundingBox3D
from verification.session import AnnotationSession, Status


def box(x=10.0, tid=None, label="Car"):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, label, 0.9, tid)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def results():
    # track 0 (car) in frames 0-2, track 1 (pedestrian) in frames 1-2
    return [
        [box(10, 0)],
        [box(11, 0), box(30, 1, "Pedestrian")],
        [box(12, 0), box(31, 1, "Pedestrian")],
    ]


@pytest.fixture
def s(results):
    return AnnotationSession.from_tracking("0000", results, clock=FakeClock())


def statuses(s, frame):
    return [a.status for a in s.annotations(frame)]


def test_from_tracking_everything_pending(s):
    assert s.num_frames == 3
    assert statuses(s, 1) == [Status.PENDING, Status.PENDING]
    st = s.stats()
    assert st["total"] == 5 and st["pending"] == 5 and st["reviewed_fraction"] == 0.0


def test_manual_mode_starts_empty(results):
    m = AnnotationSession.from_tracking("0000", results, mode="manual")
    assert m.stats()["total"] == 0 and m.num_frames == 3


def test_invalid_mode():
    with pytest.raises(ValueError):
        AnnotationSession("0000", 1, mode="auto")


def test_accept_and_reject(s):
    s.accept(1, 0)
    s.reject(1, 1)
    assert statuses(s, 1) == [Status.ACCEPTED, Status.REJECTED]
    assert s.stats()["reviewed"] == 2


def test_accept_track_only_touches_pending_boxes_of_that_track(s):
    s.reject(1, 0)                      # car in frame 1 already decided
    n = s.accept_track(0)
    assert n == 2                       # frames 0 and 2
    assert statuses(s, 0) == [Status.ACCEPTED]
    assert statuses(s, 1) == [Status.REJECTED, Status.PENDING]
    assert statuses(s, 2) == [Status.ACCEPTED, Status.PENDING]


def test_accept_track_from_frame(s):
    assert s.accept_track(0, from_frame=1) == 2
    assert statuses(s, 0) == [Status.PENDING]


def test_reject_track(s):
    assert s.reject_track(1) == 2
    assert statuses(s, 2)[1] is Status.REJECTED


def test_edit_marks_edited_and_keeps_track_id(s):
    s.edit(0, 0, x=10.5, heading=0.3)
    a = s.annotations(0)[0]
    assert a.status is Status.EDITED and a.box.x == 10.5 and a.box.track_id == 0
    assert a.box.heading == pytest.approx(0.3)


def test_edit_heading_is_normalised(s):
    s.edit(0, 0, heading=7.0)
    assert s.annotations(0)[0].box.heading == pytest.approx(7.0 - 2 * math.pi)


def test_invalid_edit_leaves_session_untouched(s):
    with pytest.raises(ValueError):
        s.edit(0, 0, dx=-1.0)
    with pytest.raises(ValueError):
        s.edit(0, 0, color="red")
    assert s.annotations(0)[0].status is Status.PENDING and s.annotations(0)[0].box.dx == 4.0
    assert s.undo() is None  # nothing was recorded


def test_accept_does_not_downgrade_edited(s):
    s.edit(0, 0, x=11.0)
    s.accept(0, 0)
    assert s.annotations(0)[0].status is Status.EDITED


def test_add_manual_assigns_fresh_track_id(s):
    idx = s.add_manual(0, box(50))
    a = s.annotations(0)[idx]
    assert a.status is Status.MANUAL and a.box.track_id == 2  # 0 and 1 are taken
    idx2 = s.add_manual(0, box(60))
    assert s.annotations(0)[idx2].box.track_id == 3


def test_rejecting_manual_box_deletes_it(s):
    idx = s.add_manual(0, box(50))
    s.reject(0, idx)
    assert len(s.annotations(0)) == 1


def test_copy_to_frame(s):
    m = AnnotationSession("0000", 3, mode="manual")
    i = m.add_manual(0, box(10))
    j = m.copy_to_frame(0, i, 1)
    a, b = m.annotations(0)[i], m.annotations(1)[j]
    assert b.box.track_id == a.box.track_id and b.status is Status.MANUAL
    with pytest.raises(ValueError):
        m.copy_to_frame(0, i, 1)  # already has that track


def test_undo_redo_single_and_track_operations(s):
    s.accept(0, 0)
    s.accept_track(1)
    assert s.undo() == "accept_track"
    assert statuses(s, 1) == [Status.PENDING, Status.PENDING]
    assert s.undo() == "accept"
    assert s.stats()["reviewed"] == 0
    assert s.undo() is None
    assert s.redo() == "accept"
    assert statuses(s, 0) == [Status.ACCEPTED]
    assert s.redo() == "accept_track"
    assert statuses(s, 2) == [Status.PENDING, Status.ACCEPTED]


def test_new_action_clears_redo(s):
    s.accept(0, 0)
    s.undo()
    s.reject(0, 0)
    assert s.redo() is None


def test_undo_add_and_edit(s):
    s.edit(0, 0, x=99.0)
    idx = s.add_manual(0, box(50))
    s.undo()
    assert len(s.annotations(0)) == 1
    s.undo()
    assert s.annotations(0)[0].box.x == 10.0 and s.annotations(0)[0].status is Status.PENDING


def test_next_pending_frame_wraps(s):
    s.accept(0, 0)
    assert s.next_pending_frame(0) == 1
    s.accept_track(0)
    s.accept_track(1)
    assert s.next_pending_frame(0) is None
    s.undo()
    assert s.next_pending_frame(2) is not None


def test_final_boxes(s):
    s.accept(0, 0)
    s.reject(1, 0)
    s.edit(2, 0, x=13.0)
    assert [b.x for b in s.final_boxes(0)] == [10.0]
    assert s.final_boxes(1) == []
    assert [b.x for b in s.final_boxes(2)] == [13.0]
    assert len(s.final_boxes(1, include_pending=True)) == 1


def test_frame_out_of_range(s):
    with pytest.raises(IndexError):
        s.annotations(3)
    with pytest.raises(IndexError):
        s.accept(0, 5)


def test_event_log_and_active_time():
    clock = FakeClock()
    s = AnnotationSession.from_tracking("0000", [[box(10, 0)], [box(11, 0)]], clock=clock)
    clock.advance(5); s.accept(0, 0)
    clock.advance(10); s.accept(1, 0)
    clock.advance(600); s.undo()          # long pause is not counted
    assert [e.action for e in s.events] == ["session_start", "accept", "accept", "undo"]
    assert s.active_seconds(max_gap=30) == pytest.approx(15.0)
    assert s.events[1].detail["track_id"] == 0 and s.events[1].frame == 0


def test_failed_edit_is_not_logged(s):
    n = len(s.events)
    with pytest.raises(ValueError):
        s.edit(0, 0, dz=0)
    assert len(s.events) == n


def test_save_and_load_roundtrip(s, tmp_path):
    s.accept(0, 0)
    s.edit(1, 0, x=12.5)
    s.add_manual(2, box(70))
    path = tmp_path / "sub" / "session.json"
    s.save(path)

    t = AnnotationSession.load(path)
    assert t.sequence == "0000" and t.num_frames == 3
    for f in range(3):
        assert t.annotations(f) == s.annotations(f)
    assert t.new_track_id() == s.new_track_id()
    assert [e.action for e in t.events][:-1] == [e.action for e in s.events]
    assert t.events[-1].action == "session_resume"
    assert not list(path.parent.glob("*.tmp"))


def test_old_file_with_participant_still_loads(tmp_path):
    """A session saved before the "participant" field was dropped (exact old format)."""
    p = tmp_path / "0000_assisted_P01.json"
    p.write_text(
        '{"version": 1, "sequence": "0000", "mode": "assisted", "participant": "P01",'
        ' "frame_offset": 0, "num_frames": 2, "next_track_id": 1,'
        ' "frames": {"0": [{"box": {"x": 10.0, "y": 0.0, "z": -1.0, "dx": 4.0, "dy": 1.8, "dz": 1.5,'
        ' "heading": 0.0, "label": "Car", "score": 0.9, "track_id": 0}, "status": "accepted"}]},'
        ' "events": [{"t": 1000.0, "action": "session_start", "frame": null,'
        ' "detail": {"mode": "assisted", "participant": "P01"}}]}')
    t = AnnotationSession.load(p)
    assert t.num_frames == 2 and t.stats()["accepted"] == 1
    assert not hasattr(t, "participant")


def test_load_rejects_unknown_version(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"version": 99}')
    with pytest.raises(ValueError):
        AnnotationSession.load(p)


def test_revision_counts_real_changes_only(s):
    r0 = s.revision
    s.accept(0, 0)
    assert s.revision == r0 + 1
    s.accept(0, 0)                      # already accepted: no change
    assert s.revision == r0 + 1
    with pytest.raises(ValueError):
        s.edit(0, 0, dx=-1)
    assert s.revision == r0 + 1
    s.undo(); s.redo()
    assert s.revision == r0 + 3