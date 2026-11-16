"""E3: effort model counts on hand-built sequences with known IoU."""
import pytest

from common.boxes import BoundingBox3D
from common.geometry import iou_3d
from experiments.lib.effort import (
    DEFAULT_THRESHOLDS,
    effort_counts,
    pool_effort,
    validate_thresholds,
)

PED = (0.8, 0.6, 1.7)
CAR = (4.0, 1.8, 1.5)


def box(x, label="Pedestrian", tid=1, dims=PED):
    return BoundingBox3D(x, 0.0, 0.0, *dims, 0.0, label, track_id=tid)


def shift_for_iou(iou, length):
    """x shift of an equal axis-aligned box giving this IoU: (L - d) / (L + d) = iou."""
    return length * (1 - iou) / (1 + iou)


def test_shift_helper_gives_the_iou():
    for v in (0.4, 0.6):
        assert iou_3d(box(0.0), box(shift_for_iou(v, PED[0]))) == pytest.approx(v)


def test_pedestrian_iou_04_is_matched_and_fixed():
    gt = [[box(0.0, tid=1)]]
    pred = [[box(shift_for_iou(0.4, PED[0]), tid=7)]]
    c = effort_counts(gt, pred)
    assert (c["n_delete"], c["n_add_tracks"], c["n_missing_boxes"]) == (0, 0, 0)
    assert c["n_fix_boxes"] == 1


def test_pedestrian_iou_06_is_matched_not_fixed():
    gt = [[box(0.0, tid=1)]]
    pred = [[box(shift_for_iou(0.6, PED[0]), tid=7)]]
    c = effort_counts(gt, pred)
    assert (c["n_delete"], c["n_add_tracks"], c["n_missing_boxes"], c["n_fix_boxes"]) == (0, 0, 0, 0)


def test_car_iou_04_is_not_matched():
    gt = [[box(0.0, "Car", 1, CAR)]]
    pred = [[box(shift_for_iou(0.4, CAR[0]), "Car", 7, CAR)]]
    c = effort_counts(gt, pred)
    assert (c["n_delete"], c["n_add_tracks"], c["n_fix_boxes"]) == (1, 1, 0)


def test_car_iou_06_is_matched_and_fixed():
    gt = [[box(0.0, "Car", 1, CAR)]]
    pred = [[box(shift_for_iou(0.6, CAR[0]), "Car", 7, CAR)]]
    c = effort_counts(gt, pred)
    assert (c["n_delete"], c["n_add_tracks"], c["n_fix_boxes"]) == (0, 0, 1)


def test_split_track_is_one_id_switch_not_a_delete():
    gt = [[box(0.0, tid=1)] for _ in range(6)]
    pred = [[box(0.0, tid=7 if k < 3 else 8)] for k in range(6)]
    c = effort_counts(gt, pred)
    assert c["n_id_switches"] == 1
    assert (c["n_delete"], c["n_add_tracks"], c["n_missing_boxes"], c["n_fix_boxes"]) == (0, 0, 0, 0)


def test_interleaved_pieces_count_every_change():
    gt = [[box(0.0, tid=1)] for _ in range(4)]
    pred = [[box(0.0, tid=(7, 8, 7, 8)[k])] for k in range(4)]
    assert effort_counts(gt, pred)["n_id_switches"] == 3


def test_false_track_missing_track_and_gap():
    gt = [[box(0.0, tid=1), box(20.0, tid=2)], [box(0.0, tid=1), box(20.0, tid=2)], [box(0.0, tid=1)]]
    pred = [[box(0.0, tid=7), box(-30.0, tid=9)], [], [box(0.0, tid=7)]]
    c = effort_counts(gt, pred)
    assert c["n_delete"] == 1            # track 9 matches nothing
    assert c["n_add_tracks"] == 1        # GT track 2 has no predicted track
    assert c["n_missing_boxes"] == 1     # GT track 1, frame 1
    assert c["n_manual_boxes"] == 5 and c["n_gt_tracks"] == 2
    assert c["effort_ratio"] == pytest.approx(3 / 5)


def test_classes_must_agree():
    gt = [[box(0.0, "Cyclist", 1)]]
    pred = [[box(0.0, "Pedestrian", 7)]]
    c = effort_counts(gt, pred)
    assert (c["n_delete"], c["n_add_tracks"]) == (1, 1)


def test_weights_and_pooling():
    gt = [[box(0.0, tid=1)]]
    pred = [[box(shift_for_iou(0.4, PED[0]), tid=7), box(30.0, tid=8)]]
    c = effort_counts(gt, pred, weights={"n_fix_boxes": 2.0, "n_delete": 0.5})
    assert c["effort_cost"] == pytest.approx(2.0 + 0.5)
    p = pool_effort([c, c])
    assert p["n_manual_boxes"] == 2 and p["effort_ratio"] == pytest.approx(2.5)


@pytest.mark.parametrize("tau_fix", [0.25, 0.2])
def test_tau_fix_not_above_tau_match_raises(tau_fix):
    with pytest.raises(ValueError, match="tau_match < tau_fix"):
        validate_thresholds({"Pedestrian": {"tau_match": 0.25, "tau_fix": tau_fix}})


def test_effort_counts_rejects_bad_thresholds():
    bad = {**DEFAULT_THRESHOLDS, "Car": {"tau_match": 0.7, "tau_fix": 0.7}}
    with pytest.raises(ValueError, match="Car"):
        effort_counts([[]], [[]], thresholds=bad)


def test_missing_class_threshold_raises():
    with pytest.raises(ValueError, match="Cyclist"):
        effort_counts([[box(0.0, "Cyclist")]], [[]], thresholds={"Car": DEFAULT_THRESHOLDS["Car"]})


def test_defaults():
    assert validate_thresholds(DEFAULT_THRESHOLDS) == {
        "Car": {"tau_match": 0.5, "tau_fix": 0.7},
        "Pedestrian": {"tau_match": 0.25, "tau_fix": 0.5},
        "Cyclist": {"tau_match": 0.25, "tau_fix": 0.5},
    }
