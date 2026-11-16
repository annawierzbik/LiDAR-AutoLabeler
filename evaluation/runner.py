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

from common.boxes import BoundingBox3D
from detection.fov import sequence_mode
from evaluation.detection_metrics import DetectionEvaluator
from evaluation.ground_truth import filter_to_camera_fov, load_frame_gt
from evaluation.tracking_metrics import TrackingEvaluator
from ingestion.calibration import Calibration
from ingestion.loader import parse_tracking_labels
from ingestion.sequence import SequenceLoader
from tracking.tracker import Tracker
from verification import cache
from verification.paths import RESULTS_DIR

RAW_THRESHOLD = 0.1  # low on purpose: AP needs the whole score range
_detectors: dict = {}  # created once per FOV mode, reused for all sequences


def _get_detector(fov: str = "auto"):
    if fov not in _detectors:
        from detection.detector import Detector  # heavy import, only when needed
        _detectors[fov] = Detector(score_threshold=RAW_THRESHOLD, fov=fov)
    return _detectors[fov]


def run_detector(seq: SequenceLoader, indices: range, fov: str = "auto") -> list[list[BoundingBox3D]]:
    detector = _get_detector(fov)
    out, t0 = [], time.time()
    for n, i in enumerate(indices, 1):
        out.append(detector.predict(seq[i]))
        if n % 20 == 0 or n == len(indices):
            print(f"    detected {n}/{len(indices)} ({(time.time() - t0) / n:.2f} s/frame)")
    return out


def load_or_detect(seq, indices: range, path: Path, use_cache: bool, split: str | None = "training",
                   fov: str = "auto", allow_missing_weights: bool = False) -> list[list[BoundingBox3D]]:
    """Detections of `indices`, from `path` (or its _c<key> variant, see verification/cache.py)
    when it holds exactly this frame range and was made with the same detector settings
    (including the resolved FOV crop and the KITTI split)."""
    meta = cache.make_meta(RAW_THRESHOLD, fov=sequence_mode(seq, fov), split=split)
    if use_cache:
        found = cache.choose([cache.keyed_path(path, meta), path], meta,
                             usable=lambda d: d["start"] == indices.start and d["stop"] == indices.stop,
                             allow_missing_weights=allow_missing_weights)
        if found is not None:
            print(f"    loaded cached detections ({found[0].name})")
            return [[BoundingBox3D(**b) for b in frame] for frame in found[1]["frames"]]
        if path.exists():
            print("    no cached detections for this frame range and these settings -> recomputing")
    dets = run_detector(seq, indices, fov)
    out = cache.output_path(path, meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "start": indices.start, "stop": indices.stop,
        "frames": [[asdict(b) for b in frame] for frame in dets],
        "meta": meta,
    }))
    return dets


def parse_sequence_spec(spec: str, available: list[str]) -> list[str]:
    """'all' | '0,3,5' | '0-5' | '0-2,7' -> zero-padded ids that exist in `available`."""
    if spec.strip().lower() == "all":
        return list(available)
    wanted: list[str] = []
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


def available_sequences(data_root: Path, split: str) -> list[str]:
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
    stop: int | None = None,
    fov: str = "auto",
) -> dict | None:
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
    detections = load_or_detect(seq, indices, RESULTS_DIR / f"{sequence}_detections.json", use_cache,
                                split, fov)

    trk_eval, tracker = TrackingEvaluator(), Tracker()
    for i, dets in zip(indices, detections):
        gt, ignore = load_frame_gt(labels.get(i, []), calib)
        det_eval.update(gt, ignore, filter_to_camera_fov(dets, calib))
        tracked = tracker.update([d for d in dets if d.score >= threshold])
        trk_eval.update(gt, ignore, filter_to_camera_fov(tracked, calib))
    return trk_eval.summary()