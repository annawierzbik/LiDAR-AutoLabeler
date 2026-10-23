"""Shared evaluation pipeline used by the evaluate_* scripts.

evaluate_sequence() runs (or loads cached) detections for one sequence, feeds them to a
DetectionEvaluator (shared across sequences, so AP is pooled) and runs the tracker with its
own TrackingEvaluator (track ids are per-sequence, so it is created fresh each time).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from common.boxes import BoundingBox3D
from evaluation.detection_metrics import DetectionEvaluator
from evaluation.ground_truth import filter_to_camera_fov, load_frame_gt
from evaluation.tracking_metrics import TrackingEvaluator
from ingestion.calibration import Calibration
from ingestion.loader import parse_tracking_labels
from ingestion.sequence import SequenceLoader
from tracking.tracker import Tracker
from verification.paths import RESULTS_DIR

RAW_THRESHOLD = 0.1  # low on purpose: AP needs the whole score range
_detector = None     # created once, reused for all sequences


def _get_detector():
    global _detector
    if _detector is None:
        from detection.detector import Detector  # heavy import, only when needed
        _detector = Detector(score_threshold=RAW_THRESHOLD)
    return _detector


def run_detector(seq: SequenceLoader, indices: range) -> List[List[BoundingBox3D]]:
    detector = _get_detector()
    out, t0 = [], time.time()
    for n, i in enumerate(indices, 1):
        out.append(detector.predict(seq[i]))
        if n % 20 == 0 or n == len(indices):
            print(f"    detected {n}/{len(indices)} ({(time.time() - t0) / n:.2f} s/frame)")
    return out


def load_or_detect(seq, indices: range, cache: Path, use_cache: bool) -> List[List[BoundingBox3D]]:
    if use_cache and cache.exists():
        data = json.loads(cache.read_text())
        if data["start"] == indices.start and data["stop"] == indices.stop:
            print(f"    loaded cached detections ({cache.name})")
            return [[BoundingBox3D(**b) for b in frame] for frame in data["frames"]]
        print("    cache covers a different frame range -> recomputing")
    dets = run_detector(seq, indices)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "start": indices.start, "stop": indices.stop,
        "frames": [[asdict(b) for b in frame] for frame in dets],
    }))
    return dets


def parse_sequence_spec(spec: str, available: List[str]) -> List[str]:
    """'all' | '0,3,5' | '0-5' | '0-2,7' -> zero-padded ids that exist in `available`."""
    if spec.strip().lower() == "all":
        return list(available)
    wanted: List[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            wanted += [f"{n:04d}" for n in range(int(a), int(b) + 1)]
        else:
            wanted.append(f"{int(part):04d}")
    missing = [w for w in wanted if w not in available]
    if missing:
        print(f"Warning: sequences not found and skipped: {missing}")
    return [w for w in wanted if w in available]


def available_sequences(data_root: Path, split: str) -> List[str]:
    vel = Path(data_root) / split / "velodyne"
    return sorted(p.name for p in vel.iterdir() if p.is_dir()) if vel.exists() else []


def evaluate_sequence(
    sequence: str,
    data_root: Path,
    split: str,
    threshold: float,
    use_cache: bool,
    det_eval: DetectionEvaluator,
    start: int = 0,
    stop: Optional[int] = None,
) -> Optional[dict]:
    """Returns the tracking summary of this sequence (None if it has no labels).
    Detection results are accumulated into `det_eval`."""
    root = Path(data_root) / split
    label_path = root / "label_02" / f"{sequence}.txt"
    if not label_path.exists():
        print(f"  {sequence}: no labels at {label_path}, skipped")
        return None

    seq = SequenceLoader(Path(data_root), split, sequence)
    calib = Calibration.from_file(root / "calib" / f"{sequence}.txt")
    labels = parse_tracking_labels(label_path)

    indices = range(start, min(stop if stop is not None else len(seq), len(seq)))
    detections = load_or_detect(seq, indices, RESULTS_DIR / f"{sequence}_detections.json", use_cache)

    trk_eval, tracker = TrackingEvaluator(), Tracker()
    for i, dets in zip(indices, detections):
        gt, ignore = load_frame_gt(labels.get(i, []), calib)
        det_eval.update(gt, ignore, filter_to_camera_fov(dets, calib))
        tracked = tracker.update([d for d in dets if d.score >= threshold])
        trk_eval.update(gt, ignore, filter_to_camera_fov(tracked, calib))
    return trk_eval.summary()