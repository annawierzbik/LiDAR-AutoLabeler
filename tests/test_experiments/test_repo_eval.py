import numpy as np
import pytest

from common.boxes import BoundingBox3D
from experiments.lib import repo_eval as re_
from ingestion.calibration import Calibration

# KITTI object calibration of frame 000008 (rounded), enough for FOV filtering in tests
CALIB = Calibration(
    P2=np.array([[721.5, 0, 609.6, 44.9], [0, 721.5, 172.9, 0.2], [0, 0, 1, 0.003]]),
    R_rect=np.eye(3),
    Tr_velo_cam=np.array([[0, -1, 0, 0], [0, 0, -1, -0.08], [1, 0, 0, -0.27]]),
)


def car(x, y=0.0, label="Car", score=1.0):
    return BoundingBox3D(x, y, -0.8, 4.0, 1.8, 1.5, 0.0, label, score)


def gt(label, x, y=0.0, height_px=60.0, occluded=0, truncated=0.0):
    return re_.GTObject(car(x, y, label), label, np.array([100, 100, 150, 100 + height_px]), truncated, occluded)


def frame(gts, preds):
    return re_.EvalFrame("f", gts, [re_.Prediction(p, np.array([100, 100, 150, 160.0])) for p in preds], CALIB,
                         {"labels": []})


def test_parse_object_labels(tmp_path):
    p = tmp_path / "000000.txt"
    p.write_text("Car 0.88 3 -0.69 0.00 192.37 402.31 374.00 1.60 1.57 3.23 -2.70 1.74 3.68 -1.29\n"
                 "DontCare -1 -1 -10 800.38 163.67 825.45 184.07 -1 -1 -1 -1000 -1000 -1000 -10\n")
    labels = re_.parse_object_labels(p)
    assert labels[0]["type"] == "Car" and labels[0]["truncated"] == 0.88 and labels[0]["occluded"] == 3
    assert labels[0]["dimensions"] == [1.60, 1.57, 3.23] and labels[0]["track_id"] == -1
    objs = re_.gt_objects(labels, CALIB)
    assert objs[0].box is not None and objs[1].box is None     # DontCare has no 3D box


def test_perfect_detection_gives_ap_100():
    frames = [frame([gt("Car", 10)], [car(10)])]
    assert re_.evaluate(frames, re_.REPO_DEFAULT)["Car"]["AP40"] == pytest.approx(1.0)


def test_detection_on_van_is_ignored_by_repo_rules_but_not_without_ignore():
    frames = [frame([gt("Car", 10), gt("Van", 20)], [car(10, score=0.9), car(20, score=0.95)])]
    assert re_.evaluate(frames, re_.REPO_DEFAULT)["Car"]["AP40"] == pytest.approx(1.0)
    none = re_.evaluate(frames, {**re_.REPO_DEFAULT, "ignore": "none"})["Car"]
    assert none["AP40"] < 1.0                     # the Van detection ranks first and is a FP


def test_extra_ignore_class():
    frames = [frame([gt("Pedestrian", 10), gt("Person", 15)],
                    [car(10, label="Pedestrian", score=0.5), car(15, label="Pedestrian", score=0.9)])]
    default = re_.evaluate(frames, re_.REPO_DEFAULT)["Pedestrian"]["AP40"]
    ignored = re_.evaluate(frames, {**re_.REPO_DEFAULT, "extra_ignore": ["Person"]})["Pedestrian"]["AP40"]
    assert default < 1.0 and ignored == pytest.approx(1.0)


def test_difficulty_moves_hard_gt_to_ignore():
    hard = gt("Car", 20, occluded=2)              # too occluded for easy and moderate
    frames = [frame([gt("Car", 10), hard], [car(10)])]
    no_filter = re_.evaluate(frames, re_.REPO_DEFAULT)["Car"]
    moderate = re_.evaluate(frames, {**re_.REPO_DEFAULT, "difficulty": "moderate"})["Car"]
    assert no_filter["num_gt"] == 2 and moderate["num_gt"] == 1
    assert moderate["AP40"] == pytest.approx(1.0)


def test_small_detections_are_dropped_with_difficulty():
    f = frame([gt("Car", 10)], [car(10, score=0.5), car(30, score=0.99)])
    f.predictions[1].bbox2d = np.array([0, 0, 10, 20.0])     # 20 px < 25 px (moderate)
    assert re_.evaluate([f], {**re_.REPO_DEFAULT, "difficulty": "moderate"})["Car"]["AP40"] == pytest.approx(1.0)
    assert re_.evaluate([f], re_.REPO_DEFAULT)["Car"]["AP40"] < 1.0


def test_dontcare_2d():
    f = frame([gt("Car", 10), re_.GTObject(None, "DontCare", np.array([0, 0, 50, 50.0]), -1, -1)],
              [car(10, score=0.5), car(30, score=0.99)])
    f.predictions[1].bbox2d = np.array([10, 10, 40, 40.0])   # inside the DontCare box
    assert re_.evaluate([f], {**re_.REPO_DEFAULT, "dontcare_2d": 0.5})["Car"]["AP40"] == pytest.approx(1.0)
    assert re_.evaluate([f], re_.REPO_DEFAULT)["Car"]["AP40"] < 1.0


def test_variant_machinery_reproduces_runner_path():
    """evaluate(REPO_DEFAULT) splits the work per class; it must equal one DetectionEvaluator
    fed by load_frame_gt as in evaluation/runner.py."""
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(30):
        labels, preds = [], []
        for _ in range(rng.integers(1, 6)):
            cls = rng.choice(["Car", "Pedestrian", "Cyclist", "Van", "Misc", "DontCare"])
            x, y = rng.uniform(5, 40), rng.uniform(-8, 8)
            # camera frame label (h, w, l), bottom center: x_cam = -y_velo, y_cam = -z_velo + h/2, z_cam = x_velo
            labels.append({"track_id": -1, "type": str(cls), "bbox": [0, 0, 50, 80],
                           "dimensions": [1.5, 1.8, 4.0] if cls != "DontCare" else [-1, -1, -1],
                           "location": [-y, 1.55, x + 0.27], "rotation_y": float(rng.uniform(-3, 3))})
            if cls != "DontCare" and rng.random() < 0.8:
                label = "Car" if cls in ("Car", "Van") else ("Pedestrian" if cls == "Misc" else str(cls))
                preds.append(BoundingBox3D(x + rng.normal(0, 0.3), y + rng.normal(0, 0.3), -0.8, 4.0, 1.8, 1.5,
                                           float(rng.uniform(-3, 3)), label, float(rng.uniform(0.1, 1.0))))
            if rng.random() < 0.3:
                preds.append(car(rng.uniform(5, 40), rng.uniform(-8, 8), score=float(rng.uniform(0.1, 1))))
        frames.append(re_.EvalFrame("f", re_.gt_objects(labels, CALIB), [re_.Prediction(p) for p in preds],
                                    CALIB, {"labels": labels}))
    reference = re_.evaluate_repo_default(frames)
    machinery = re_.evaluate(frames, re_.REPO_DEFAULT)
    for cls in reference:
        assert machinery[cls]["AP40"] == pytest.approx(reference[cls]["AP40"])
        assert machinery[cls]["num_gt"] == reference[cls]["num_gt"]


def test_kitti_sampled_ap_on_perfect_records():
    from experiments.lib.kitti_eval import load_eval_module
    get_thresholds = load_eval_module().get_thresholds
    frames = [frame([gt("Car", 10)], [car(10, score=0.3 + 0.01 * i)]) for i in range(60)]
    evs = {}
    re_.evaluate(frames, re_.REPO_DEFAULT, evaluators=evs)
    assert re_.kitti_sampled_ap(evs["Car"], "Car", get_thresholds) == pytest.approx(1.0)


def test_the_iou_cache_never_answers_for_freed_boxes():
    """The IoU cache is keyed by id(box); CPython reuses the id of a freed box for a new one. A far
    pair (IoU 0) evaluated and freed must not make a perfect pair made right after it score AP 0
    (the cause of a flaky test_perfect_detection_gives_ap_100)."""
    for _ in range(200):
        far = [frame([gt("Car", 10)], [car(40)])]
        re_.evaluate(far, re_.REPO_DEFAULT)
        del far
        good = [frame([gt("Car", 10)], [car(10)])]
        assert re_.evaluate(good, re_.REPO_DEFAULT)["Car"]["AP40"] == pytest.approx(1.0)
