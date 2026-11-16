"""E0: a detector training split without the raw drives of the excluded KITTI tracking sequences.

    python -m experiments.run_clean_train_split --config experiments/configs/E0_clean_train_split.yaml

Reads splits/tracking_overlap_report.json (run_leakage_check) and writes, next to the originals
(which are not modified):
    OpenPCDet/data/kitti/ImageSets/<name>.txt             train frames outside those drives
    OpenPCDet/data/kitti/kitti_infos_<name>.pkl           the matching subset of kitti_infos_train.pkl
    OpenPCDet/data/kitti/kitti_dbinfos_<name>.pkl         gt_sampling database without objects from
                                                          removed frames (the augmentation pastes
                                                          database objects into training scenes, so
                                                          an unfiltered database would leak them back)
The database entries point to the existing gt_database/*.bin files; no point files are copied.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter

from experiments.lib import leakage
from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import KITTI_OBJECT_ROOT, SPLITS_DIR
from experiments.lib.results_io import finish_run, start_run


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    name = cfg["clean_split_name"]
    out_ids = KITTI_OBJECT_ROOT / "ImageSets" / f"{name}.txt"
    out_infos = KITTI_OBJECT_ROOT / f"kitti_infos_{name}.pkl"
    out_db = KITTI_OBJECT_ROOT / f"kitti_dbinfos_{name}.pkl"
    if out_ids.exists() and not args.overwrite:
        raise SystemExit(f"{out_ids} exists; pass --overwrite to regenerate it")

    run = start_run("E0", cfg, seed=cfg["seed"], tag="clean_train_split", dataset="kitti_object", split=name)
    report = json.loads((SPLITS_DIR / "tracking_overlap_report.json").read_text(encoding="utf-8"))
    excluded = [s for s in report["sequences"] if s["contaminated"]]
    drives = sorted({d for s in excluded for d in s["raw_drives"]})

    mapping = leakage.read_object_mapping(KITTI_OBJECT_ROOT / "devkit_object")
    train_ids = [ln.strip() for ln in (KITTI_OBJECT_ROOT / "ImageSets" / "train.txt").read_text().splitlines() if ln.strip()]
    removed = [i for i in train_ids if mapping[int(i)][0] in drives]
    kept = [i for i in train_ids if mapping[int(i)][0] not in drives]
    removed_set = set(removed)
    per_drive = Counter(mapping[int(i)][0] for i in removed)

    with open(KITTI_OBJECT_ROOT / "kitti_infos_train.pkl", "rb") as f:
        infos = pickle.load(f)
    infos_clean = [inf for inf in infos if inf["point_cloud"]["lidar_idx"] not in removed_set]
    with open(KITTI_OBJECT_ROOT / "kitti_dbinfos_train.pkl", "rb") as f:
        db = pickle.load(f)
    db_clean = {cls: [e for e in entries if f"{int(e['image_idx']):06d}" not in removed_set] for cls, entries in db.items()}

    def objects(info_list):
        c = Counter()
        for inf in info_list:
            c.update(n for n in inf["annos"]["name"] if n != "DontCare")
        return dict(sorted(c.items()))

    if len(infos_clean) != len(kept):
        raise RuntimeError(f"{len(infos_clean)} infos vs {len(kept)} ids")
    out_ids.write_text("\n".join(kept) + "\n")
    with open(out_infos, "wb") as f:
        pickle.dump(infos_clean, f)
    with open(out_db, "wb") as f:
        pickle.dump(db_clean, f)

    metrics = {
        "excluded_sequences": [s["sequence"] for s in excluded],
        "excluded_drives": drives,
        "train_frames": len(train_ids),
        "removed_frames": len(removed),
        "kept_frames": len(kept),
        "removed_per_drive": dict(sorted(per_drive.items())),
        "val_frames_in_excluded_drives": sum(1 for i in (KITTI_OBJECT_ROOT / "ImageSets" / "val.txt").read_text().split()
                                             if mapping[int(i)][0] in drives),
        "objects_train": objects(infos),
        "objects_clean": objects(infos_clean),
        "gt_database_entries_train": {k: len(v) for k, v in db.items()},
        "gt_database_entries_clean": {k: len(v) for k, v in db_clean.items()},
        "files": [str(out_ids), str(out_infos), str(out_db)],
    }
    path = finish_run(run, metrics=metrics, notes="train split without the raw drives of the excluded tracking sequences")
    print(json.dumps({k: v for k, v in metrics.items() if k != "files"}, indent=1))
    print(f"result: {path}")


if __name__ == "__main__":
    main()
