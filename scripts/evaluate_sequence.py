"""Evaluate detection and tracking on ONE KITTI tracking sequence against ground truth.

    python -m scripts.evaluate_sequence --sequence 0000
    python -m scripts.evaluate_sequence --sequence 0000 --stop 60 --threshold 0.5

For many sequences at once use scripts.evaluate_all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.detection_metrics import DetectionEvaluator
from evaluation.report import print_detection, print_tracking
from evaluation.runner import evaluate_sequence


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.5, help="score threshold fed to the tracker")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    det_eval = DetectionEvaluator()
    trk = evaluate_sequence(args.sequence, Path(args.data_root), args.split, args.threshold,
                            not args.no_cache, det_eval, args.start, args.stop)
    if trk is None:
        return

    det = det_eval.summary(score_threshold=args.threshold)
    print_detection(det)
    print_tracking(trk)

    out = Path(f"results/{args.sequence}_metrics.json")
    out.write_text(json.dumps({"detection": det, "tracking": trk}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()