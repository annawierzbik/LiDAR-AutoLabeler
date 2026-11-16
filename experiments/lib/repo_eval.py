"""The repo's detection metric (evaluation/detection_metrics.py) on KITTI data, with switchable
evaluation rules, so it can be compared with OpenPCDet's evaluator rule by rule.

The repo code is called, not copied: GT boxes come from evaluation.ground_truth.label_to_box, the
FOV filter from filter_to_camera_fov, matching and AP from DetectionEvaluator. A variant only
decides which boxes are handed to it:

    ignore         "repo"  -> evaluation.ground_truth.IGNORE_CLASSES, for every class (repo default)
                   "kitti" -> Van for Car, Person_sitting for Pedestrian (as OpenPCDet's clean_data)
                   "none"
    extra_ignore   more class names added to the ignore set of every class (e.g. ["Person"])
    ignore_iou     IoU with an ignore box at which a detection is neither TP nor FP;
                   "class" = the class's matching threshold (what OpenPCDet does)
    fov_filter     drop detections whose center is outside the camera image (repo default)
    dontcare_2d    null, or a fraction: drop detections whose 2D box lies in a DontCare box by more
                   than this share of its own area (OpenPCDet does this for the 2D bbox metric only)
    difficulty     null | easy | moderate | hard: GT of the class that fails the KITTI limits
                   becomes an ignore box; detections lower than the 2D height limit are dropped
    iou            "3d" | "bev"

Known remaining difference to OpenPCDet even with all rules aligned: the matching order (repo:
detections by score, each takes the best-IoU free GT, IoU >= threshold; OpenPCDet: per GT, the
best-IoU free detection, IoU > threshold), the recall sampling (repo: exact precision envelope at
40 recall points; OpenPCDet: precision at 41 score thresholds), and OpenPCDet's handling of a GT
matched only by a too-small detection (neither TP nor FN; here the detection is dropped, the GT is FN).
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import evaluation.detection_metrics as dm
from common.boxes import BoundingBox3D
from common.geometry import bev_iou, iou_3d
from evaluation.ground_truth import (
    EVAL_CLASSES,
    IGNORE_CLASSES,
    filter_to_camera_fov,
    label_to_box,
    load_frame_gt,
)
from ingestion.calibration import Calibration

# KITTI difficulty limits, as in OpenPCDet kitti_object_eval_python/eval.py: clean_data
MIN_HEIGHT = {"easy": 40, "moderate": 25, "hard": 25}
MAX_OCCLUSION = {"easy": 0, "moderate": 1, "hard": 2}
MAX_TRUNCATION = {"easy": 0.15, "moderate": 0.3, "hard": 0.5}
KITTI_IGNORE = {"Car": ("Van",), "Pedestrian": ("Person_sitting",), "Cyclist": ()}
REPO_DEFAULT = {"ignore": "repo", "ignore_iou": 0.3, "fov_filter": True, "dontcare_2d": None,
                "difficulty": None, "iou": "3d"}


@dataclass
class GTObject:
    box: BoundingBox3D | None      # None for DontCare (no 3D box)
    label: str
    bbox2d: np.ndarray             # left, top, right, bottom (pixels)
    truncated: float
    occluded: int


@dataclass
class Prediction:
    box: BoundingBox3D
    bbox2d: np.ndarray | None = None


@dataclass
class EvalFrame:
    frame_id: str
    gt: list[GTObject]
    predictions: list[Prediction]
    calib: Calibration
    meta: dict = field(default_factory=dict)


def parse_object_labels(path: Path) -> list[dict]:
    """KITTI object label_2 file -> dicts in the shape of ingestion.loader.parse_tracking_labels
    (track_id = -1) plus truncated, occluded, alpha."""
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        out.append({
            "track_id": -1, "type": p[0], "truncated": float(p[1]), "occluded": int(float(p[2])),
            "alpha": float(p[3]), "bbox": [float(v) for v in p[4:8]],
            "dimensions": [float(v) for v in p[8:11]], "location": [float(v) for v in p[11:14]],
            "rotation_y": float(p[14]),
        })
    return out


def gt_objects(labels: Sequence[dict], calib: Calibration) -> list[GTObject]:
    objs = []
    for lab in labels:
        typ = lab["type"]
        box = None
        if typ != "DontCare" and min(lab["dimensions"]) > 0:
            box = label_to_box(lab, calib, typ)
        objs.append(GTObject(box, typ, np.asarray(lab["bbox"], dtype=np.float64),
                             float(lab.get("truncated", 0.0)), int(lab.get("occluded", 0))))
    return objs


def predictions_from_anno(anno: dict) -> list[Prediction]:
    """OpenPCDet det_anno (boxes_lidar, name, score, bbox) -> Predictions in the LiDAR frame."""
    return [Prediction(BoundingBox3D.from_array(b, label=str(n), score=float(min(max(s, 0.0), 1.0))),
                       np.asarray(bb, dtype=np.float64))
            for b, n, s, bb in zip(anno["boxes_lidar"], anno["name"], anno["score"], anno["bbox"])]


def _passes_difficulty(o: GTObject, difficulty: str) -> bool:
    height = o.bbox2d[3] - o.bbox2d[1]
    return (o.occluded <= MAX_OCCLUSION[difficulty] and o.truncated <= MAX_TRUNCATION[difficulty]
            and height > MIN_HEIGHT[difficulty])


def _inside_dontcare(bbox: np.ndarray, dontcare: list[np.ndarray], min_fraction: float) -> bool:
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area <= 0:
        return False
    for dc in dontcare:
        iw = min(bbox[2], dc[2]) - max(bbox[0], dc[0])
        ih = min(bbox[3], dc[3]) - max(bbox[1], dc[1])
        if iw > 0 and ih > 0 and iw * ih / area > min_fraction:
            return True
    return False


class _PairCache:
    """iou_matrix replacement that memoizes pairwise IoU by box identity. Boxes are reused across
    variants, so each pair is computed once; values come from the repo's iou_3d / bev_iou."""

    def __init__(self) -> None:
        self.fns = {"3d": iou_3d, "bev": bev_iou}
        self.cache: dict[tuple[str, int, int], float] = {}
        self.kind = "3d"

    def __call__(self, boxes_a, boxes_b) -> np.ndarray:
        fn, m = self.fns[self.kind], np.zeros((len(boxes_a), len(boxes_b)))
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                key = (self.kind, id(a), id(b))
                v = self.cache.get(key)
                if v is None:
                    # same cheap reject as common.geometry.iou_matrix
                    if np.linalg.norm(a.center[:2] - b.center[:2]) > (np.hypot(a.dx, a.dy) + np.hypot(b.dx, b.dy)) / 2:
                        v = 0.0
                    else:
                        v = fn(a, b)
                    self.cache[key] = v
                m[i, j] = v
        return m


_PAIRS = _PairCache()


@contextlib.contextmanager
def _iou_kind(kind: str) -> Iterator[None]:
    """Make DetectionEvaluator use the cached 3D or BEV IoU for the duration of the block."""
    original = dm.iou_matrix
    _PAIRS.kind = kind
    dm.iou_matrix = _PAIRS
    try:
        yield
    finally:
        dm.iou_matrix = original


def kitti_sampled_ap(evaluator: dm.DetectionEvaluator, cls: str, get_thresholds) -> float:
    """AP of the evaluator's (score, is_tp) records, but with OpenPCDet's recall sampling
    (eval.py: get_thresholds over the TP scores, precision at those 41 thresholds, envelope,
    mean of points 1..40) instead of DetectionEvaluator.average_precision. Separates the effect of
    recall sampling from that of matching. Returns a fraction like average_precision."""
    records, n_gt = evaluator.records[cls], evaluator.n_gt[cls]
    if not records or n_gt == 0:
        return 0.0
    scores = np.array([s for s, _ in records])
    is_tp = np.array([t for _, t in records])
    thresholds = get_thresholds(scores[is_tp].copy(), n_gt)
    prec = np.zeros(41)
    for i, t in enumerate(thresholds):
        kept = scores >= t
        prec[i] = is_tp[kept].sum() / kept.sum()
    for i in range(len(thresholds)):
        prec[i] = prec[i:].max()
    return float(prec[1:].sum() / 40)


def evaluate(frames: Sequence[EvalFrame], variant: dict, classes=EVAL_CLASSES,
             evaluators: dict | None = None) -> dict:
    """DetectionEvaluator.summary() per class under one variant (keys documented at the top).
    Pass a dict as `evaluators` to receive the per-class DetectionEvaluator objects."""
    ignore_mode = variant.get("ignore", "repo")
    extra = tuple(variant.get("extra_ignore", ()))
    difficulty = variant.get("difficulty")
    dontcare = variant.get("dontcare_2d")
    thresholds = dict(dm.DEFAULT_IOU_THRESHOLDS)
    out = {}
    with _iou_kind(variant.get("iou", "3d")):
        for cls in classes:
            ignore_iou = thresholds[cls] if variant.get("ignore_iou", 0.3) == "class" else float(variant.get("ignore_iou", 0.3))
            ev = dm.DetectionEvaluator(classes=(cls,), ignore_iou=ignore_iou)
            ignore_classes = {"repo": IGNORE_CLASSES, "kitti": KITTI_IGNORE[cls], "none": ()}[ignore_mode] + extra
            for fr in frames:
                gt, ign = [], []
                for o in fr.gt:
                    if o.box is None:
                        continue
                    if o.label == cls:
                        (gt if difficulty is None or _passes_difficulty(o, difficulty) else ign).append(o.box)
                    elif o.label in ignore_classes:
                        ign.append(o.box)
                preds = [p for p in fr.predictions if p.box.label == cls]
                if variant.get("fov_filter", True):
                    kept = {id(b) for b in filter_to_camera_fov([p.box for p in preds], fr.calib)}
                    preds = [p for p in preds if id(p.box) in kept]
                if difficulty is not None:
                    preds = [p for p in preds if p.bbox2d is None or p.bbox2d[3] - p.bbox2d[1] >= MIN_HEIGHT[difficulty]]
                if dontcare is not None:
                    dcs = [o.bbox2d for o in fr.gt if o.label == "DontCare"]
                    preds = [p for p in preds if p.bbox2d is None or not _inside_dontcare(p.bbox2d, dcs, float(dontcare))]
                ev.update(gt, ign, [p.box for p in preds])
            out[cls] = ev.summary()[cls]
            if evaluators is not None:
                evaluators[cls] = ev
    return out


def evaluate_repo_default(frames: Sequence[EvalFrame], classes=EVAL_CLASSES) -> dict:
    """Exactly what evaluation/runner.py does per frame (load_frame_gt on the raw label dicts in
    frame.meta["labels"] + FOV filter + one evaluator, uncached IoU), as a reference for the
    variant machinery: evaluate(frames, REPO_DEFAULT) must give the same numbers."""
    ev = dm.DetectionEvaluator(classes=classes)
    for fr in frames:
        gt, ign = load_frame_gt(fr.meta["labels"], fr.calib)
        ev.update(gt, ign, filter_to_camera_fov([p.box for p in fr.predictions], fr.calib))
    s = ev.summary()
    return {c: s[c] for c in classes}
