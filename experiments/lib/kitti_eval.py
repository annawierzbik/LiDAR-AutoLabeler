"""OpenPCDet's KITTI object evaluator (kitti_object_eval_python/eval.py), usable in this environment.

eval.py imports `rotate_iou_gpu_eval` from rotate_iou.py, whose numba.cuda kernels fail to compile
at import time with the installed numba (0.68: "Signature mismatch"), so tools/test.py crashes in
its evaluation step. `load_eval_module()` registers a stand-in rotate_iou module before eval.py is
imported; the stand-in is a statement-by-statement CPU port of the same algorithm
(experiments/lib/rotate_iou_cpu.py). Everything else in eval.py (matching, difficulty filtering,
DontCare, 41-point R40) runs unchanged from the OpenPCDet checkout.

E0 compared the full result dict with the unmodified eval.py + rotate_iou.py run under numba 0.60
in a separate environment (see results/E0, run tag "evalshim").
"""
from __future__ import annotations

import importlib
import pickle
import sys
import types
from pathlib import Path

import numpy as np

from experiments.lib.paths import KITTI_OBJECT_ROOT, OPENPCDET_ROOT
from experiments.lib.rotate_iou_cpu import rotate_iou_cpu_eval

if str(OPENPCDET_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPCDET_ROOT))

EVAL_PKG = "pcdet.datasets.kitti.kitti_object_eval_python"
DIFFICULTIES = ("easy", "moderate", "hard")
CLASSES = ("Car", "Pedestrian", "Cyclist")


def load_eval_module():
    """pcdet's kitti_object_eval_python.eval, with the rotate_iou stand-in."""
    name = f"{EVAL_PKG}.rotate_iou"
    if name not in sys.modules:
        import torch  # noqa: F401  (before pcdet: Windows DLL loading of the compiled ops)
        importlib.import_module(EVAL_PKG)          # the package __init__ has no CUDA code
        stub = types.ModuleType(name)
        stub.rotate_iou_gpu_eval = rotate_iou_cpu_eval
        stub.__doc__ = "stand-in registered by experiments.lib.kitti_eval"
        sys.modules[name] = stub
    return importlib.import_module(f"{EVAL_PKG}.eval")


def load_infos(split: str, root: Path = KITTI_OBJECT_ROOT) -> list[dict]:
    with open(Path(root) / f"kitti_infos_{split}.pkl", "rb") as f:
        return pickle.load(f)


def load_predictions(result_pkl: Path) -> list[dict]:
    """det_annos written by tools/test.py (one dict per frame, KITTI camera-frame fields +
    boxes_lidar, score, frame_id)."""
    with open(result_pkl, "rb") as f:
        return pickle.load(f)


def align_to_infos(det_annos: list[dict], infos: list[dict]) -> tuple[list[dict], list[dict]]:
    """(gt_annos, det_annos) in the same frame order, as KittiDataset.evaluation builds them."""
    by_id = {d["frame_id"]: d for d in det_annos}
    ids = [info["point_cloud"]["lidar_idx"] for info in infos]
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} frames have no prediction, e.g. {missing[:3]}")
    gt = [{k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in info["annos"].items()} for info in infos]
    return gt, [by_id[i] for i in ids]


def official_eval(gt_annos: list[dict], det_annos: list[dict], classes=CLASSES):
    """(printable result string, {"<Class>_<3d|bev|image|aos>/<difficulty>[_R40]": AP in %}).
    eval.py modifies nothing in place that callers rely on, but it is given copies anyway."""
    kitti_eval = load_eval_module()
    gt = [dict(a) for a in gt_annos]
    dt = [dict(a) for a in det_annos]
    text, result = kitti_eval.get_official_eval_result(gt, dt, list(classes))
    return text, {k: float(v) for k, v in result.items()}


def r40_table(result: dict, classes=CLASSES, metrics=("3d", "bev")) -> dict:
    """{cls: {metric: {difficulty: AP_R40 in %}}} picked out of official_eval()'s dict."""
    return {c: {m: {d: result[f"{c}_{m}/{d}_R40"] for d in DIFFICULTIES} for m in metrics} for c in classes}
