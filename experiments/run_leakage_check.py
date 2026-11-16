"""E0: overlap of KITTI tracking sequences with the object train split, and the dev/test split.

    python -m experiments.run_leakage_check --config experiments/configs/E0_leakage.yaml

Writes splits/tracking_overlap_report.json, splits/kitti_tracking_dev_test.json and one result
file in results/E0/. Refuses to overwrite an existing split file unless --overwrite is given:
the split must stay fixed once tuning has started.
"""
from __future__ import annotations

import json
import time

from experiments.lib import leakage
from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import KITTI_OBJECT_ROOT, KITTI_TRACKING_ROOT, SPLITS_DIR
from experiments.lib.results_io import finish_run, start_run
from experiments.lib.splits import (
    TRACKING_SPLIT_FILE,
    fraction_summary,
    make_dev_test_split,
    save_split,
)

OVERLAP_REPORT = SPLITS_DIR / "tracking_overlap_report.json"


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing split file")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    if TRACKING_SPLIT_FILE.exists() and not args.overwrite:
        raise SystemExit(f"{TRACKING_SPLIT_FILE} exists; pass --overwrite to regenerate it on purpose")

    run = start_run("E0", cfg, seed=cfg["seed"], tag="leakage", dataset="kitti_tracking+kitti_object",
                    split="training")
    t0 = time.perf_counter()
    mapping = leakage.read_object_mapping(KITTI_OBJECT_ROOT / "devkit_object")
    image_sets = leakage.read_image_sets(KITTI_OBJECT_ROOT / "ImageSets")
    index = leakage.ContentIndex(KITTI_OBJECT_ROOT / "training" / "velodyne")

    vel_root = KITTI_TRACKING_ROOT / cfg["tracking_split"] / "velodyne"
    reports = []
    for seq_dir in sorted(p for p in vel_root.iterdir() if p.is_dir()):
        n_frames = len(list(seq_dir.glob("*.bin")))
        matches = leakage.match_sequence(seq_dir, index)
        rep = leakage.sequence_report(seq_dir.name, n_frames, matches, mapping, image_sets)
        rep["contaminated"] = leakage.is_contaminated(rep, cfg["contamination_rule"], cfg["detector_train_split"])
        reports.append(rep)
        print(f"  {seq_dir.name}: {n_frames:4d} frames, {len(matches):3d} in object set "
              f"{rep['num_frames_in_object_split']}, drives {rep['raw_drives']} -> "
              f"{'CONTAMINATED' if rep['contaminated'] else 'clean'}")

    clean = [r["sequence"] for r in reports if not r["contaminated"]]
    frames = {r["sequence"]: r["num_frames"] for r in reports}
    split = make_dev_test_split(clean, cfg["test_fraction"], cfg["split_seed"])
    split_doc = {
        "description": "KITTI tracking training sequences without frames of the detector's training "
                       "split, divided by whole sequences. Tune on dev only; test needs --final.",
        "contamination_rule": cfg["contamination_rule"],
        "detector_train_split": cfg["detector_train_split"],
        "seed": cfg["split_seed"],
        "test_fraction": cfg["test_fraction"],
        "excluded": [r["sequence"] for r in reports if r["contaminated"]],
        "dev": split["dev"],
        "test": split["test"],
        "balance": fraction_summary(split, frames),
        "created_by_run": run["run_id"],
    }
    overlap_doc = {
        "method": leakage.__doc__.strip(),
        "object_split_drive_overlap": leakage.drive_split_overlap(mapping, image_sets),
        "md5_computations": index.hashed,
        "sequences": reports,
    }
    OVERLAP_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OVERLAP_REPORT.write_text(json.dumps(overlap_doc, indent=2), encoding="utf-8")
    save_split(split_doc)

    summary = {r["sequence"]: {k: r[k] for k in ("num_frames", "num_matched_frames", "raw_drives",
                                                 "frame_offsets", "consistent_mapping",
                                                 "num_frames_in_object_split",
                                                 "same_drive_object_frames_outside_sequence", "contaminated")}
               for r in reports}
    path = finish_run(run, metrics={"sequences": summary, "split": split_doc},
                      timing={"total_s": time.perf_counter() - t0},
                      notes=f"overlap report: {OVERLAP_REPORT}; split: {TRACKING_SPLIT_FILE}")
    print(f"excluded {split_doc['excluded']}\ndev {split['dev']}\ntest {split['test']}\nresult: {path}")


if __name__ == "__main__":
    main()
