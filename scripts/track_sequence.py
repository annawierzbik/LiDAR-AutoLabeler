"""Detect + track a whole KITTI sequence and write the result to the cache file.

    python -m scripts.track_sequence --sequence 0000
    python -m scripts.track_sequence --sequence 0000 --start 0 --stop 60

Detection is the slow part, so tracked results are cached to a JSON file
(results/<split>_<sequence>.json by default). Next runs load the cache instantly;
use --no-cache to recompute. The run prints track statistics; the boxes are reviewed in the
browser (python -m scripts.web).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from detection.fov import add_fov_argument, sequence_mode
from ingestion.sequence import SequenceLoader
from pipeline.detect_track import current_meta, parse_results, print_stats, save_results, track_sequence
from verification import cache
from verification.paths import RESULTS_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None, help="exclusive; default = end of sequence")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cache", default=None, help="default: results/<split>_<sequence>.json")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--use-cache-without-weights", action="store_true",
                   help="use a matching cache even though the checkpoint is missing on disk")
    add_fov_argument(p)
    args = p.parse_args()

    seq = SequenceLoader(Path(args.data_root), args.split, args.sequence)
    path = Path(args.cache) if args.cache else RESULTS_DIR / f"{args.split}_{args.sequence}.json"
    meta = current_meta(args.threshold, sequence_mode(seq, args.fov), args.split)
    candidates = [cache.keyed_path(path, meta), path]
    if not args.cache and args.split == "training":       # written before the split was in the name
        old = RESULTS_DIR / f"{args.sequence}.json"
        candidates += [cache.keyed_path(old, meta), old]

    found = None if args.no_cache else cache.choose(candidates, meta,
                                                    allow_missing_weights=args.use_cache_without_weights)
    if found is not None:
        print(f"Loading cached results from {found[0]}")
        indices, results = parse_results(found[1])
    else:
        stop = min(args.stop if args.stop is not None else len(seq), len(seq))
        indices = range(args.start, stop)
        print(f"Detecting + tracking frames {indices.start}..{indices.stop - 1}")
        results = track_sequence(seq, indices, args.threshold, args.fov)
        out = cache.output_path(path, meta)
        save_results(out, indices, results, meta)
        print(f"Saved to {out}")

    print_stats(results)


if __name__ == "__main__":
    main()
