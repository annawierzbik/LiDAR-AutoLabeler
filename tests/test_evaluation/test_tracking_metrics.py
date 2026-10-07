import pytest

from common.boxes import BoundingBox3D
from evaluation.tracking_metrics import TrackingEvaluator


def box(x, tid, label="Car"):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, label, 1.0, tid)


def run(gt_frames, trk_frames, ignore=()):
    ev = TrackingEvaluator()
    for g, t in zip(gt_frames, trk_frames):
        ev.update(g, list(ignore), t)
    return ev.summary()


def moving(n, tid, x0=10.0, v=0.5):
    return [box(x0 + v * i, tid) for i in range(n)]


def test_perfect_tracking():
    gt = [[b] for b in moving(10, 7)]
    trk = [[b] for b in moving(10, 3)]  # ids differ from GT ids but are consistent
    s = run(gt, trk)
    assert s["MOTA"] == pytest.approx(1.0) and s["id_switches"] == 0
    assert s["MOTP_mean_iou"] == pytest.approx(1.0)
    assert s["mostly_tracked"] == 1 and s["mostly_lost"] == 0


def test_id_switch_is_counted_once():
    gt = [[b] for b in moving(10, 1)]
    trk = [[box(10 + 0.5 * i, 5 if i < 5 else 6)] for i in range(10)]
    s = run(gt, trk)
    assert s["id_switches"] == 1
    assert s["MOTA"] == pytest.approx(1 - 1 / 10)


def test_missing_track_counts_fn_and_fragmentation():
    gt = [[b] for b in moving(10, 1)]
    trk = [[b] for b in moving(10, 4)]
    trk[4], trk[5] = [], []
    s = run(gt, trk)
    assert s["false_negatives"] == 2 and s["fragmentations"] == 1
    assert s["id_switches"] == 0  # same track id after the gap


def test_false_positive_counted():
    gt = [[b] for b in moving(5, 1)]
    trk = [[b, box(80, 9)] for b in moving(5, 4)]
    s = run(gt, trk)
    assert s["false_positives"] == 5
    assert s["MOTA"] == pytest.approx(1 - 5 / 5)


def test_false_positive_on_ignore_region_not_counted():
    van = BoundingBox3D(80, 0, -1, 4, 1.8, 1.5, 0, "Van")
    gt = [[b] for b in moving(3, 1)]
    trk = [[b, box(80, 9)] for b in moving(3, 4)]
    assert run(gt, trk, ignore=[van])["false_positives"] == 0


def test_mostly_lost():
    gt = [[b] for b in moving(10, 1)]
    trk = [[] for _ in range(10)]
    s = run(gt, trk)
    assert s["mostly_lost"] == 1 and s["MOTA"] == pytest.approx(0.0)


def test_class_mismatch_is_not_matched():
    gt = [[box(10, 1, "Car")]]
    trk = [[box(10, 2, "Pedestrian")]]
    s = run(gt, trk)
    assert s["false_negatives"] == 1 and s["false_positives"] == 1


def test_empty_sequence():
    assert TrackingEvaluator().summary()["MOTA"] == 0.0


def test_pooling_sums_counts_and_recomputes_ratios():
    from evaluation.tracking_metrics import pool_tracking_summaries

    a = run([[b] for b in moving(10, 1)], [[b] for b in moving(10, 1)])              # perfect, 10 objects
    b = run([[b] for b in moving(10, 1)], [[] for _ in range(10)])                    # all missed, 10 objects
    pooled = pool_tracking_summaries([a, b])
    assert pooled["gt_objects"] == 20 and pooled["false_negatives"] == 10
    assert pooled["MOTA"] == pytest.approx(0.5)
    assert pooled["recall"] == pytest.approx(0.5)
    assert pooled["MOTP_mean_iou"] == pytest.approx(1.0)
    assert pooled["gt_tracks"] == 2