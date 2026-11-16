"""One JSON file per run: results/<experiment_id>/<run_id>.json with full metadata.

    run = start_run("E0", config, seed=0)       # seeds everything, collects git/env metadata
    ...
    path = finish_run(run, metrics={...}, timing={...}, notes="...")
"""
from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from pathlib import Path

import numpy as np

from experiments.lib.envinfo import environment
from experiments.lib.paths import RESULTS_ROOT
from experiments.lib.seeds import set_seeds

REQUIRED_KEYS = ("experiment_id", "run_id", "timestamp", "git", "openpcdet_git", "config", "seeds",
                 "dataset", "split", "model", "weights", "detector_fov", "metrics", "timing", "notes")
# detector_fov: which points the detector saw. OPENPCDET_FOV = OpenPCDet's KittiDataset (tools/test.py,
# training) with FOV_POINTS_ONLY; otherwise detection.fov.describe() of the Detector wrapper's crop
# ("camera", "azimuth:<deg>", "none"), a dict when parts of a run differ, None = no detector inference.
OPENPCDET_FOV = "openpcdet_fov_points_only"


def new_run_id(tag: str = "") -> str:
    """Sortable and unique: 20261028-164100_<tag>_<6 hex>."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return "_".join(p for p in (stamp, tag, uuid.uuid4().hex[:6]) if p)


def _jsonable(obj):
    """numpy scalars/arrays, Paths and non-finite floats -> plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def start_run(experiment_id: str, config: dict, seed: int = 0, tag: str = "", *,
              dataset: str | None = None, split: str | None = None, model: str | None = None,
              weights: str | Path | None = None, deterministic: bool = False,
              detector_fov: str | dict | None = None) -> dict:
    """Seeds the RNGs and returns the run record that `finish_run` completes and writes."""
    env = environment()
    return {
        "experiment_id": experiment_id,
        "run_id": new_run_id(tag),
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": env.pop("git"),
        "openpcdet_git": env.pop("openpcdet_git"),
        "environment": env,
        "config": config,
        "seeds": set_seeds(seed, deterministic),
        "dataset": dataset,
        "split": split,
        "model": model,
        "weights": str(weights) if weights else None,
        "detector_fov": detector_fov,
        "metrics": {},
        "timing": {},
        "notes": "",
    }


def finish_run(run: dict, metrics: dict | None = None, timing: dict | None = None, notes: str = "",
               root: Path = RESULTS_ROOT, **extra) -> Path:
    run = dict(run)
    run["metrics"] = metrics or {}
    run["timing"] = timing or {}
    run["notes"] = notes
    run.update(extra)
    missing = [k for k in REQUIRED_KEYS if k not in run]
    if missing:
        raise ValueError(f"result record is missing {missing}")
    path = Path(root) / run["experiment_id"] / f"{run['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(run), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_results(experiment_id: str, root: Path = RESULTS_ROOT) -> list[dict]:
    """All run records of one experiment, oldest first (run ids sort by time)."""
    folder = Path(root) / experiment_id
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
