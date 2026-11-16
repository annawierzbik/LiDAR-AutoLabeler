"""Run PointPillars on one KITTI frame and print the detections.

    python -m scripts.detect_frame --sequence 0000 --frame 0 --threshold 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from detection.detector import Detector  # imports torch first
from detection.fov import add_fov_argument
from ingestion.sequence import SequenceLoader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    add_fov_argument(p)
    args = p.parse_args()

    seq = SequenceLoader(Path(args.data_root), args.split, args.sequence)
    frame = seq[args.frame]

    detector = Detector(score_threshold=args.threshold, fov=args.fov)
    boxes = detector.predict(frame)

    print(f"Frame {frame.frame_id}: {frame.num_points} points, {len(boxes)} detections")
    for b in boxes:
        print(f"  {b.label:<10} score={b.score:.2f} center=({b.x:.1f}, {b.y:.1f}, {b.z:.2f}) "
              f"size=({b.dx:.2f}, {b.dy:.2f}, {b.dz:.2f}) heading={b.heading:.2f}")


if __name__ == "__main__":
    main()
