"""E0: environment and data report (versions, GPU, git state, weights per config, dataset file counts).

    python -m experiments.run_env_report --config experiments/configs/E0_env.yaml
"""
from __future__ import annotations

from pathlib import Path

from experiments.lib.config import base_parser, load_config
from experiments.lib.envinfo import available_weights
from experiments.lib.paths import KITTI_OBJECT_ROOT, KITTI_TRACKING_ROOT, PROJECT_ROOT
from experiments.lib.results_io import finish_run, start_run


def _count(folder: Path, pattern: str) -> int | None:
    return sum(1 for _ in folder.glob(pattern)) if folder.exists() else None


def dataset_report() -> dict:
    obj = {split: {sub: _count(KITTI_OBJECT_ROOT / split / sub, "*.*")
                   for sub in ("velodyne", "calib", "label_2", "image_2")}
           for split in ("training", "testing")}
    image_sets = {}
    for name in ("train", "val", "test", "trainval"):
        p = KITTI_OBJECT_ROOT / "ImageSets" / f"{name}.txt"
        image_sets[name] = len(p.read_text().split()) if p.exists() else None
    infos = {p.name: p.stat().st_size for p in sorted(KITTI_OBJECT_ROOT.glob("kitti_*infos_*.pkl"))}
    trk = {}
    for split in ("training", "testing"):
        vel = KITTI_TRACKING_ROOT / split / "velodyne"
        trk[split] = {p.name: _count(p, "*.bin") for p in sorted(vel.iterdir()) if p.is_dir()} if vel.exists() else None
    mapping = KITTI_OBJECT_ROOT / "devkit_object" / "mapping"
    return {
        "kitti_object_root": str(KITTI_OBJECT_ROOT),
        "kitti_object_files": obj,
        "image_sets": image_sets,
        "openpcdet_infos_bytes": infos,
        "gt_database_files": _count(KITTI_OBJECT_ROOT / "gt_database", "*.bin"),
        "devkit_mapping_files": sorted(p.name for p in mapping.glob("*.txt")) if mapping.exists() else [],
        "kitti_tracking_root": str(KITTI_TRACKING_ROOT),
        "kitti_tracking_frames": trk,
    }


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    run = start_run("E0", cfg, seed=cfg["seed"], tag="env")
    metrics = {"weights_per_config": available_weights(), "datasets": dataset_report()}
    weights = {k: v for k, v in metrics["weights_per_config"].items() if v}
    path = finish_run(run, metrics=metrics, notes=f"configs with local weights: {weights}")
    env = run["environment"]
    print(f"python {env['python']}, {env['platform']}")
    print("libraries:", env["libraries"])
    print("cuda:", env["cuda"])
    print("git:", run["git"]["commit"], "dirty" if run["git"]["dirty"] else "clean",
          "| OpenPCDet:", run["openpcdet_git"]["commit"], "dirty" if run["openpcdet_git"]["dirty"] else "clean")
    print("weights:", weights)
    print(f"result: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
