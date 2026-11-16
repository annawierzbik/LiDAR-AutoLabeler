"""Locations of datasets and experiment outputs (all absolute, independent of the working directory)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENPCDET_ROOT = PROJECT_ROOT / "OpenPCDet"

KITTI_OBJECT_ROOT = OPENPCDET_ROOT / "data" / "kitti"        # training/, testing/, ImageSets/, devkit_object/
KITTI_TRACKING_ROOT = PROJECT_ROOT / "data" / "raw"           # training/{velodyne,calib,label_02}/<seq>
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

CONFIGS_DIR = PROJECT_ROOT / "experiments" / "configs"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTS_ROOT = PROJECT_ROOT / "results"                       # results/<experiment_id>/<run_id>.json
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"


def experiment_dir(experiment_id: str) -> Path:
    return RESULTS_ROOT / experiment_id


def cache_dir(experiment_id: str) -> Path:
    """Large intermediate files (detections, hashes) that are not results themselves."""
    return RESULTS_ROOT / experiment_id / "cache"
