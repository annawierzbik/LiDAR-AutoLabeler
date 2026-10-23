from common.boxes import BoundingBox3D
from tracking.tracker import Tracker


def car(x, y=0.0, label="Car", score=0.9):
    return BoundingBox3D(x, y, -1.0, 4.0, 1.8, 1.5, 0.0, label, score)


def run(tracker, frames):
    return [tracker.update(f) for f in frames]


def test_single_object_keeps_id():
    t = Tracker()
    out = run(t, [[car(10 + 0.8 * i)] for i in range(10)])
    ids = {b.track_id for frame in out for b in frame}
    assert len(ids) == 1
    assert all(len(f) == 1 for f in out)


def test_two_objects_get_distinct_stable_ids():
    t = Tracker()
    out = run(t, [[car(10 + i * 0.5, 0), car(30 - i * 0.5, 8)] for i in range(10)])
    last = {round(b.y): b.track_id for b in out[-1]}
    first = {round(b.y): b.track_id for b in out[0]}
    assert last == first and len(set(last.values())) == 2


def test_different_class_never_matches():
    t = Tracker(min_hits=1)
    t.update([car(10, label="Car")])
    out = t.update([car(10, label="Pedestrian")])
    assert out[0].track_id != 0  # new track created for the pedestrian


def test_survives_short_gap_and_keeps_id():
    t = Tracker(max_age=3)
    frames = [[car(10 + 1.0 * i)] for i in range(6)]
    frames += [[], []]  # detector misses two frames
    frames += [[car(10 + 1.0 * i)] for i in range(8, 12)]
    out = run(t, frames)
    ids = {b.track_id for f in out for b in f}
    assert ids == {0}
    assert out[6] == [] and out[7] == []  # nothing reported while unseen


def test_long_gap_creates_new_id():
    t = Tracker(max_age=2)
    frames = [[car(10)]] * 5 + [[]] * 5 + [[car(10)]] * 5
    out = run(t, frames)
    assert out[4][0].track_id != out[-1][0].track_id


def test_one_frame_false_positive_is_filtered():
    t = Tracker(min_hits=3)
    frames = [[car(10 + i)] for i in range(10)]
    frames[6] = frames[6] + [car(60, 20)]  # ghost box appears for a single frame
    out = run(t, frames)
    assert all(len(f) == 1 for f in out[4:])


def test_ids_stable_when_objects_cross_paths_closely():
    t = Tracker()
    out = run(t, [[car(10 + 1.0 * i, 0.0), car(30 - 1.0 * i, 3.5)] for i in range(12)])
    for f in out[3:]:
        by_y = {round(b.y): b.track_id for b in f}
        assert len(set(by_y.values())) == 2
    assert {round(b.y): b.track_id for b in out[3]} == {round(b.y): b.track_id for b in out[-1]}


def test_detections_not_mutated_and_reset():
    t = Tracker(min_hits=1)
    d = car(10)
    t.update([d])
    assert d.track_id is None
    t.reset()
    assert t.update([car(10)])[0].track_id == 0


def test_empty_input():
    assert Tracker().update([]) == []