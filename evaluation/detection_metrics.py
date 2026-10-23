"""Detection metrics: AP (KITTI-style, 40 recall points) and precision/recall.

Usage:
    ev = DetectionEvaluator()
    for each frame:
        ev.update(gt_boxes, ignore_boxes, detected_boxes)
    ev.summary(score_threshold=0.5)

Simplification vs. the official KITTI benchmark: no difficulty filtering
(easy/moderate/hard), all labeled objects count. Numbers are therefore lower than
published ones and should only be compared with each other (before/after fine-tuning).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from common.boxes import BoundingBox3D
from common.geometry import iou_matrix

DEFAULT_IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}  # as in KITTI 3D
CLASSES = ("Car", "Pedestrian", "Cyclist")


class DetectionEvaluator:
    def __init__(
        self,
        classes: Sequence[str] = CLASSES,
        iou_thresholds: dict[str, float] | None = None,
        ignore_iou: float = 0.3,
    ) -> None:
        self.classes = tuple(classes)
        self.iou_thresholds = dict(DEFAULT_IOU_THRESHOLDS if iou_thresholds is None else iou_thresholds)
        self.ignore_iou = ignore_iou
        self.n_gt: dict[str, int] = defaultdict(int)
        self.records: dict[str, list[tuple[float, bool]]] = defaultdict(list)  # (score, is_tp)
        self.tp_ious: dict[str, list[float]] = defaultdict(list)

    def update(
        self,
        gt: Sequence[BoundingBox3D],
        ignore: Sequence[BoundingBox3D],
        detections: Sequence[BoundingBox3D],
    ) -> None:
        for cls in self.classes:
            g = [b for b in gt if b.label == cls]
            d = sorted((b for b in detections if b.label == cls), key=lambda b: -b.score)
            self.n_gt[cls] += len(g)
            if not d:
                continue

            thr = self.iou_thresholds[cls]
            iou_gt = iou_matrix(d, g) if g else np.zeros((len(d), 0))
            iou_ign = iou_matrix(d, list(ignore)) if ignore else np.zeros((len(d), 0))
            used = set()

            for i, det in enumerate(d):
                best_j, best_iou = -1, thr
                for j in range(len(g)):
                    if j not in used and iou_gt[i, j] >= best_iou:
                        best_j, best_iou = j, iou_gt[i, j]
                if best_j >= 0:
                    used.add(best_j)
                    self.records[cls].append((det.score, True))
                    self.tp_ious[cls].append(float(best_iou))
                elif iou_ign.shape[1] and iou_ign[i].max() >= self.ignore_iou:
                    continue  # overlaps an ignored object -> neither TP nor FP
                else:
                    self.records[cls].append((det.score, False))

    # --------------------------------------------------------------------- metrics
    def average_precision(self, cls: str) -> float:
        n_gt = self.n_gt[cls]
        rec = sorted(self.records[cls], key=lambda r: -r[0])
        if n_gt == 0 or not rec:
            return 0.0
        tp = np.cumsum([1 if is_tp else 0 for _, is_tp in rec])
        fp = np.cumsum([0 if is_tp else 1 for _, is_tp in rec])
        recall = tp / n_gt
        precision = tp / (tp + fp)
        ap = 0.0
        for r in np.linspace(1 / 40, 1.0, 40):  # R40, as in the current KITTI benchmark
            mask = recall >= r
            ap += precision[mask].max() if mask.any() else 0.0
        return float(ap / 40)

    def precision_recall(self, cls: str, score_threshold: float) -> tuple[float, float]:
        kept = [(s, t) for s, t in self.records[cls] if s >= score_threshold]
        tp = sum(1 for _, t in kept if t)
        fp = len(kept) - tp
        precision = tp / (tp + fp) if kept else 0.0
        recall = tp / self.n_gt[cls] if self.n_gt[cls] else 0.0
        return precision, recall

    def summary(self, score_threshold: float = 0.5) -> dict:
        out = {}
        for cls in self.classes:
            p, r = self.precision_recall(cls, score_threshold)
            ious = self.tp_ious[cls]
            out[cls] = {
                "AP40": self.average_precision(cls),
                "precision": p,
                "recall": r,
                "num_gt": self.n_gt[cls],
                "mean_iou_of_matches": float(np.mean(ious)) if ious else 0.0,
            }
        present = [c for c in self.classes if self.n_gt[c] > 0]  # classes absent from GT would drag the mean to 0
        out["mAP40"] = float(np.mean([out[c]["AP40"] for c in present])) if present else 0.0
        out["score_threshold"] = score_threshold
        return out