import math

import pytest

from common.boxes import BoundingBox3D
from verification.interpolation import MAX_GAP, fill_plan, interpolate_box
from verification.session import AnnotationSession, Status


def box(x, tid=0, heading=0.0, score=0.9):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, heading, "Car", score, tid)


def session(per_frame):
    """per_frame: list of lists of boxes (track 0 present where listed)."""
    return AnnotationSession.from_tracking("0000", per_frame)


def statuses(s, track=0):
    return [next((a.status.value for a in s.annotations(f) if a.box.track_id == track), None)
            for f in range(s.num_frames)]


# ---------------------------------------------------------------- pure functions
def test_interpolate_box_midpoint():
    a, b = box(10.0, score=0.9), box(14.0, score=0.6)
    m = interpolate_box(a, b, 0.5)
    assert m.x == pytest.approx(12.0) and m.track_id == 0 and m.score == 0.6 and m.label == "Car"


def test_interpolate_heading_takes_the_short_way_round():
    a, b = box(0, heading=math.radians(170)), box(0, heading=math.radians(-170))
    m = interpolate_box(a, b, 0.5)
    assert abs(abs(m.heading) - math.pi) < 1e-9   # 180 degrees, not 0


def test_fill_plan_respects_max_gap():
    keys = {0: box(0), 4: box(4), 4 + MAX_GAP + 2: box(10)}
    plan = fill_plan(keys)
    assert sorted(plan) == [1, 2, 3]               # hole of 3 filled, hole of 4 left alone
    assert plan[2].x == pytest.approx(2.0)


# ---------------------------------------------------------------------- session
def test_holes_are_filled_when_the_session_is_created():
    s = session([[box(0)], [], [], [box(3)]])
    assert statuses(s) == ["pending", "interpolated", "interpolated", "pending"]
    assert s.annotations(1)[0].box.x == pytest.approx(1.0)
    assert s.annotations(2)[0].box.x == pytest.approx(2.0)
    assert not s._undo                              # initial fill is not an undo step


def test_long_holes_stay_empty():
    s = session([[box(0)], [], [], [], [], [box(5)]])
    assert statuses(s) == ["pending", None, None, None, None, "pending"]


def test_interpolated_boxes_need_review_and_are_not_exported():
    s = session([[box(0)], [], [box(2)]])
    st = s.stats()
    assert st["interpolated"] == 1 and st["total"] == 3 and st["reviewed"] == 0
    assert s.frame_has_pending(1) and s.next_pending_frame(2) == 0
    assert s.final_boxes(1) == []
    s.accept(1, 0)
    assert s.final_boxes(1)[0].x == pytest.approx(1.0) and s.stats()["reviewed"] == 1


def test_accept_track_covers_interpolated_boxes():
    s = session([[box(0)], [], [box(2)]])
    assert s.accept_track(0) == 3
    assert statuses(s) == ["accepted"] * 3


def test_editing_a_key_box_moves_the_interpolated_ones():
    s = session([[box(0)], [], [box(2)]])
    s.edit(2, 0, x=4.0)
    assert s.annotations(1)[0].box.x == pytest.approx(2.0)
    assert s.annotations(1)[0].status is Status.INTERPOLATED


def test_editing_an_interpolated_box_makes_it_a_key_box():
    s = session([[box(0)], [], [], [box(3)]])
    s.edit(1, 0, y=2.0)                             # frame 1 becomes EDITED, frame 2 re-interpolates
    assert statuses(s) == ["pending", "edited", "interpolated", "pending"]
    assert s.annotations(2)[0].box.y == pytest.approx(1.0)


def test_undo_restores_the_previous_interpolation():
    s = session([[box(0)], [], [box(2)]])
    s.edit(2, 0, x=4.0)
    s.undo()
    assert s.annotations(1)[0].box.x == pytest.approx(1.0) and s.annotations(2)[0].box.x == 2.0
    s.redo()
    assert s.annotations(1)[0].box.x == pytest.approx(2.0)


def test_adding_boxes_by_hand_fills_between_them():
    s = AnnotationSession("0000", 8)
    s.add_manual(1, box(0, tid=None))
    tid = s.annotations(1)[0].box.track_id
    s.add_manual(5, box(8, tid=tid), track_id=tid)
    assert statuses(s, tid) == [None, "manual", "interpolated", "interpolated", "interpolated", "manual", None, None]
    assert s.annotations(3)[0].box.x == pytest.approx(4.0)
    s.undo()                                        # one undo removes the box and the filled frames
    assert statuses(s, tid) == [None, "manual"] + [None] * 6


def test_rejected_interpolated_box_is_not_recreated():
    s = session([[box(0)], [], [box(2)]])
    s.reject(1, 0)
    s.edit(2, 0, x=3.0)                             # triggers a refill
    assert statuses(s) == ["pending", "rejected", "edited"]


def test_rejecting_a_key_box_enlarges_the_hole():
    s = session([[box(0)], [], [box(2)], [box(3)]])
    s.reject(2, 0)
    assert statuses(s) == ["pending", "interpolated", "rejected", "pending"]


def test_fill_gaps_command_is_one_undo_step():
    s = session([[box(0)], [], [box(2)]])
    s.accept(1, 0); s.reject(1, 0)                  # nothing to fill any more (rejected box blocks the slot)
    assert s.fill_gaps() == 0
    s2 = session([[box(0)], [box(1)], [box(2)]])
    s2.frames[1] = []                               # simulate a hole created without going through edit()
    assert s2.fill_gaps() == 1 and s2.undo() == "fill_gaps" and statuses(s2)[1] is None


def test_save_load_keeps_interpolated_status(tmp_path):
    s = session([[box(0)], [], [box(2)]])
    s.save(tmp_path / "s.json")
    assert AnnotationSession.load(tmp_path / "s.json").annotations(1)[0].status is Status.INTERPOLATED