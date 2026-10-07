"""Evaluate detection and tracking on MANY sequences and report pooled results.

    python -m scripts.evaluate_all --sequences all
    python -m scripts.evaluate_all --sequences 0-9
    python -m scripts.evaluate_all --sequences 0,3,7 --threshold 0.4

Detection AP is computed over the pooled detections of all selected sequences.
Tracking counts (FN, FP, ID switches, ...) are summed over sequences and the metrics are
recomputed from the totals. Per-sequence detections are cached in results/<seq>_detections.json.
Results are saved to results/<name>_metrics.json (use --name to keep different runs apart,
e.g. --name before_finetuning / after_finetuning).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation.detection_metrics import DetectionEvaluator
from evaluation.report import print_detection, print_per_sequence, print_tracking
from evaluation.runner import available_sequences, evaluate_sequence, parse_sequence_spec
from evaluation.tracking_metrics import pool_tracking_summaries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequences", default="all", help="'all', '0-9', or '0,3,7'")
    p.add_argument("--stop", type=int, default=None, help="max frames per sequence (quick tests)")
    p.add_argument("--threshold", type=float, default=0.5, help="score threshold fed to the tracker")
    p.add_argument("--name", default="all", help="name of this run, used in the output file name")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    sequences = parse_sequence_spec(args.sequences, available_sequences(data_root, args.split))
    if not sequences:
        raise SystemExit("No sequences to evaluate.")
    print(f"Evaluating {len(sequences)} sequences: {', '.join(sequences)}")

    det_eval = DetectionEvaluator()
    per_seq, t0 = {}, time.time()
    for k, seq_id in enumerate(sequences, 1):
        print(f"[{k}/{len(sequences)}] sequence {seq_id}")
        summary = evaluate_sequence(seq_id, data_root, args.split, args.threshold,
                                    not args.no_cache, det_eval, stop=args.stop)
        if summary is not None:
            per_seq[seq_id] = summary
    if not per_seq:
        raise SystemExit("No sequence had labels.")

    pooled = pool_tracking_summaries(list(per_seq.values()))
    det = det_eval.summary(score_threshold=args.threshold)

    print_per_sequence(per_seq)
    print_detection(det)
    print_tracking(pooled, title=f"TRACKING, POOLED OVER {len(per_seq)} SEQUENCES")
    print(f"\nTotal time: {time.time() - t0:.0f} s")

    out = Path(f"results/{args.name}_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"sequences": list(per_seq), "threshold": args.threshold,
         "detection": det, "tracking": pooled, "per_sequence": per_seq}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()