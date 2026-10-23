import pytest

from common.boxes import BoundingBox3D
from verification.controller import GROUND_Z, ReviewController
from verification.session import AnnotationSession, Status


def box(x, tid, label="Car"):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, label, 0.9, tid)


@pytest.fixture
def c():
    results = [[box(10, 0), box(30, 1, "Pedestrian"), box(50, 2)], [box(11, 0), box(31, 1, "Pedestrian")], []]
    return ReviewController(AnnotationSession.from_tracking("0000", results))


def status(c, i, frame=None):
    return c.session.annotations(c.frame if frame is None else frame)[i].status


def test_goto_clamps_and_clears_selection(c):
    c.select(1)
    assert c.goto(1) and c.selected is None
    assert not c.goto(1)
    c.goto(99)
    assert c.frame == 2
    c.goto(-5)
    assert c.frame == 0


def test_select_step_cycles(c):
    c.select_step(1); assert c.selected == 0
    c.select_step(1); c.select_step(1); assert c.selected == 2
    c.select_step(1); assert c.selected == 0
    c.select_step(-1); assert c.selected == 2


def test_select_step_on_empty_frame(c):
    c.goto(2)
    c.select_step(1)
    assert c.selected is None


def test_accept_advances_to_next_pending(c):
    c.select(0)
    c.accept()
    assert status(c, 0) is Status.ACCEPTED and c.selected == 1
    c.accept()
    assert c.selected == 2
    c.accept()                    # nothing pending left -> selection stays
    assert c.selected == 2 and c.session.stats()["pending"] == 2  # the 2 boxes of frame 1


def test_reject_hides_box_unless_requested(c):
    c.select(0); c.reject()
    assert [i for i, _ in c.visible()] == [1, 2]
    c.show_rejected = True
    assert [i for i, _ in c.visible()] == [0, 1, 2]


def test_select_step_skips_hidden_rejected(c):
    c.select(1); c.reject()
    c.select(0); c.select_step(1)
    assert c.selected == 2


def test_nothing_selected_means_no_review_action(c):
    assert not c.accept() and not c.reject() and not c.nudge(dx=1) and not c.set_field("x", 1.0)
    assert c.accept_track() == 0


def test_accept_track_whole_and_from_here(c):
    c.goto(1); c.select(0)
    assert c.accept_track(from_here=True) == 1
    assert status(c, 0, frame=0) is Status.PENDING
    assert c.accept_track() == 1  # the remaining pending box of track 0 (frame 0)


def test_reject_track(c):
    c.select(1)
    assert c.reject_track() == 2


def test_nudge_and_rotate(c):
    c.select(0)
    assert c.nudge(dx=0.1, dy=-0.2, dz=0.05, dheading=0.1)
    b = c.selected_annotation().box
    assert (b.x, b.y, b.z) == pytest.approx((10.1, -0.2, -0.95))
    assert b.heading == pytest.approx(0.1)
    assert c.selected_annotation().status is Status.EDITED


def test_resize_keeps_bottom_on_the_ground_and_has_a_minimum(c):
    c.select(0)
    before = c.selected_annotation().box
    bottom = before.z - before.dz / 2
    assert c.resize(dlength=0.5, dwidth=-0.2, dheight=0.3)
    b = c.selected_annotation().box
    assert (b.dx, b.dy, b.dz) == pytest.approx((4.5, 1.6, 1.8))
    assert b.z - b.dz / 2 == pytest.approx(bottom)
    assert c.selected_annotation().status is Status.EDITED
    assert not c.resize(dwidth=-1.5)               # would go below the minimum -> ignored
    assert c.selected_annotation().box.dy == pytest.approx(1.6)


def test_resize_without_selection(c):
    assert not c.resize(dlength=0.1)


def test_invalid_field_is_rejected_without_change(c):
    c.select(0)
    assert not c.set_field("dx", -2.0)
    assert c.selected_annotation().box.dx == 4.0 and c.selected_annotation().status is Status.PENDING


def test_set_class_relabels_selected_and_sets_default(c):
    c.select(0)
    c.set_class("Cyclist")
    assert c.selected_annotation().box.label == "Cyclist" and c.current_class == "Cyclist"
    with pytest.raises(ValueError):
        c.set_class("Bus")


def test_add_box_stands_on_ground(c):
    c.goto(2)
    idx = c.add_box(label="Pedestrian")
    b = c.selected_annotation().box
    assert idx == 0 and c.selected == 0
    assert b.z - b.dz / 2 == pytest.approx(GROUND_Z)
    assert (b.x, b.y) == (8.0, 0.0) and b.label == "Pedestrian"   # in front of the car
    assert c.selected_annotation().status is Status.MANUAL
    c.add_box()                                  # without a label: the current class
    assert c.selected_annotation().box.label == "Car"


def test_copy_to_next_frame_moves_there(c):
    c.goto(2); c.add_box(); tid = c.selected_annotation().box.track_id
    c.goto(1)  # frame 1 has no such track
    c.goto(2)
    c.select(0)
    assert not c.copy_selected_to_next_frame()   # no frame after the last one
    c.goto(1); c.add_box()
    assert c.copy_selected_to_next_frame()
    assert c.frame == 2 and c.selected is not None


def test_remove_manual_box_clamps_selection(c):
    c.goto(2); c.add_box()
    c.reject()
    assert c.annotations() == [] and c.selected is None


def test_undo_redo_keep_selection_valid(c):
    c.goto(2); c.add_box()
    assert c.undo() == "add"
    assert c.selected is None and c.annotations() == []
    assert c.redo() == "add"


def test_next_pending_frame(c):
    c.select(0); c.accept(); c.accept(); c.accept()   # frame 0 done
    assert c.next_pending_frame() and c.frame == 1
    c.goto(1); c.select(0); c.accept(); c.accept()
    assert not c.next_pending_frame()


def test_progress_text(c):
    c.select(0); c.accept()
    assert c.progress_text().startswith("frame 1/3") and "1/5" in c.progress_text()


def test_select_next_pending(c):
    assert c.select_next_pending() and c.selected == 0
    c.accept()                          # moves on to the next pending one
    assert c.selected == 1
    c.goto(2)
    c.selected = None
    assert not c.select_next_pending() and c.selected is None


def test_playback_navigation_is_not_logged(c):
    n = len(c.session.events)
    c.goto(1, log=False)
    assert len(c.session.events) == n
    c.goto(2)
    assert c.session.events[-1].action == "frame_change"