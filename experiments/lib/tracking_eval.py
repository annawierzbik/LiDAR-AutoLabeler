"""Detector + the system's tracker on KITTI tracking sequences, scored with the repo's metrics.

Same per-frame steps as evaluation/runner.py: evaluate_sequence (detection AP on all detections
above the raw threshold after the camera-FOV filter; tracker fed with detections >= `threshold`;
TrackingEvaluator on the FOV-filtered tracks), but with the detector passed in, so any model can be
used, and detections cached per model in the runner's JSON format.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from common.boxes import BoundingBox3D
from evaluation.detection_metrics import DetectionEvaluator
from evaluation.ground_truth import filter_to_camera_fov, load_frame_gt
from evaluation.tracking_metrics import TrackingEvaluator, pool_tracking_summaries
from experiments.lib.paths import KITTI_TRACKING_ROOT, cache_dir
from ingestion.calibration import Calibration
from ingestion.loader import parse_tracking_labels
from ingestion.sequence import SequenceLoader
from tracking.tracker import Tracker

RAW_THRESHOLD = 0.1   # as evaluation/runner.py: AP needs the whole score range


def detection_cache(experiment_id: str, model: str, fov: str, seq: str) -> Path:
    """Cache file of one model's detections on one sequence with the resolved FOV crop `fov`
    (detection.fov.describe). Runs before the crop existed used fov "none" and the path without
    the fov folder, which is kept for them."""
    root = cache_dir(experiment_id) / model
    if fov != "none":
        root = root / ("fov_" + fov.replace(":", "_"))
    return root / f"{seq}.json"


def detect_sequence(detector, seq: str, cache: Path, split: str = "training") -> tuple[list, float | None]:
    """(detections per frame, seconds spent detecting or None when read from the cache)."""
    if cache.exists():
        data = json.loads(cache.read_text())
        return [[BoundingBox3D(**b) for b in frame] for frame in data["frames"]], None
    loader = SequenceLoader(KITTI_TRACKING_ROOT, split, seq)
    t0 = time.perf_counter()
    dets = [detector.predict(loader[i]) for i in range(len(loader))]
    seconds = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"start": 0, "stop": len(loader), "frames": [[asdict(b) for b in f] for f in dets]}))
    return dets, seconds


def evaluate_sequence(seq: str, detections: list, threshold: float, tracker_kwargs: dict,
                      match_iou: float, det_eval: DetectionEvaluator, split: str = "training") -> dict:
    root = KITTI_TRACKING_ROOT / split
    calib = Calibration.from_file(root / "calib" / f"{seq}.txt")
    labels = parse_tracking_labels(root / "label_02" / f"{seq}.txt")
    seq_det = DetectionEvaluator()
    trk_eval, tracker = TrackingEvaluator(match_iou=match_iou), Tracker(**tracker_kwargs)
    for i, dets in enumerate(detections):
        gt, ignore = load_frame_gt(labels.get(i, []), calib)
        in_fov = filter_to_camera_fov(dets, calib)
        det_eval.update(gt, ignore, in_fov)
        seq_det.update(gt, ignore, in_fov)
        tracked = tracker.update([d for d in dets if d.score >= threshold])
        trk_eval.update(gt, ignore, filter_to_camera_fov(tracked, calib))
    return {"tracking": trk_eval.summary(), "detection": seq_det.summary(score_threshold=threshold)}


def evaluate_sequences(seqs: list[str], detections: dict[str, list], threshold: float, tracker_kwargs: dict,
                       match_iou: float) -> dict:
    det_eval = DetectionEvaluator()
    per_seq = {s: evaluate_sequence(s, detections[s], threshold, tracker_kwargs, match_iou, det_eval) for s in seqs}
    return {
        "per_sequence": per_seq,
        "pooled_tracking": pool_tracking_summaries([v["tracking"] for v in per_seq.values()]),
        "pooled_detection": det_eval.summary(score_threshold=threshold),
    }
