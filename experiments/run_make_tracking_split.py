"""Freeze a dev/test split of KITTI tracking training sequences (whole sequences, seeded).

    python -m experiments.run_make_tracking_split --config experiments/configs/E0_split_v2.yaml

Unlike run_leakage_check (v1: only sequences without detector-training frames), this split takes the
sequences listed in the config, e.g. all 21 for a detector retrained without their drives. The file
is written once; it is never overwritten (delete it by hand if a split really has to be redone).
Label statistics per side are stored next to the split so its balance can be judged; they are
counts of ground truth only, no detector or tracker output is used.
"""
from __future__ import annotations

from collections import Counter

from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import KITTI_TRACKING_ROOT, SPLITS_DIR
from experiments.lib.results_io import finish_run, start_run
from experiments.lib.splits import fraction_summary, make_dev_test_split, save_split
from ingestion.loader import parse_tracking_labels


def label_counts(seq: str, split: str) -> tuple[Counter, Counter]:
    """(boxes per class, tracks per class) of one sequence."""
    labels = parse_tracking_labels(KITTI_TRACKING_ROOT / split / "label_02" / f"{seq}.txt")
    boxes, tracks = Counter(), set()
    for frame_labels in labels.values():
        for lab in frame_labels:
            boxes[lab["type"]] += 1
            tracks.add((lab["type"], lab["track_id"]))
    return boxes, Counter(t for t, _ in tracks)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    out = SPLITS_DIR / cfg["file"]
    if out.exists():
        raise SystemExit(f"{out} exists and is frozen")
    vel = KITTI_TRACKING_ROOT / cfg["tracking_split"] / "velodyne"
    available = sorted(p.name for p in vel.iterdir() if p.is_dir())
    seqs = available if cfg["sequences"] == "all" else [s for s in cfg["sequences"] if s in available]

    run = start_run("E0", cfg, seed=cfg["split_seed"], tag=f"split_{cfg['name']}", dataset="kitti_tracking",
                    split=cfg["tracking_split"])
    split = make_dev_test_split(seqs, cfg["test_fraction"], cfg["split_seed"])
    frames = {s: len(list((vel / s).glob("*.bin"))) for s in seqs}
    stats = {}
    for side in ("dev", "test"):
        boxes, tracks = Counter(), Counter()
        for s in split[side]:
            b, t = label_counts(s, cfg["tracking_split"])
            boxes.update(b)
            tracks.update(t)
        stats[side] = {"boxes": dict(sorted(boxes.items())), "tracks": dict(sorted(tracks.items()))}
    doc = {
        "name": cfg["name"],
        "description": cfg["description"],
        "seed": cfg["split_seed"],
        "test_fraction": cfg["test_fraction"],
        "sequences": seqs,
        "dev": split["dev"],
        "test": split["test"],
        "balance": fraction_summary(split, frames),
        "label_counts": stats,
        "frozen": True,
        "created_by_run": run["run_id"],
    }
    save_split(doc, out)
    path = finish_run(run, metrics=doc, notes=f"split file: {out}")
    print(f"dev  {split['dev']}\ntest {split['test']}\n{doc['balance']}\n{stats}\nresult: {path}")


if __name__ == "__main__":
    main()
