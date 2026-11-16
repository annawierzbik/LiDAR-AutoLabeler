"""E0: repo detection metric vs OpenPCDet's KITTI evaluator on identical predictions (KITTI val),
plus sensitivity of the repo metric to its evaluation rules.

    # 1. predictions of the reference pipeline (once, ~2 min on an RTX 4070; its own evaluation
    #    step crashes with numba 0.68 after result.pkl is written, see experiments/lib/kitti_eval.py)
    cd OpenPCDet/tools
    python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../../checkpoints/pointpillar_7728.pth --batch_size 4 --workers 0 --extra_tag e0_crosscheck
    # 2. this script, from the repo root
    python -m experiments.run_metric_crosscheck --config experiments/configs/E0_metric_crosscheck.yaml

Sections (each lands under "metrics" in the result file):
    openpcdet        OpenPCDet R40/R11 APs (3D, BEV) and, if given, the comparison with the
                     unmodified evaluator (reference_eval_dict, see experiments/tools/reference_kitti_eval.py)
    gt_conversion    repo label_to_box vs OpenPCDet gt_boxes_lidar on every val object
    iou              repo iou_3d / bev_iou vs OpenPCDet eval.py d3/bev overlaps on real GT x prediction pairs
    repo_default     repo metric exactly as evaluation/runner.py computes it, and the variant machinery's copy
    variants         repo metric under each rule variant of the config
    aligned          repo metric with OpenPCDet's rules per difficulty vs OpenPCDet R40 (the cross-check)
    decomposition    the aligned difference split into GT conversion / recall sampling / small detections
    tracking_dev     optional: repo metric on the dev tracking sequences with / without "Person" ignored
"""
from __future__ import annotations

import pickle
import time

import numpy as np

from common.boxes import BoundingBox3D
from experiments.lib import kitti_eval as ke
from experiments.lib import repo_eval as re_
from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import (
    KITTI_OBJECT_ROOT,
    KITTI_TRACKING_ROOT,
    PROJECT_ROOT,
    cache_dir,
)
from experiments.lib.results_io import OPENPCDET_FOV, finish_run, start_run
from experiments.lib.splits import check_split_access, load_split
from ingestion.calibration import Calibration


def _angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def build_val_frames(infos: list[dict], det_annos: list[dict]) -> list[re_.EvalFrame]:
    frames = []
    root = KITTI_OBJECT_ROOT / "training"
    for info, det in zip(infos, det_annos):
        fid = info["point_cloud"]["lidar_idx"]
        calib = Calibration.from_file(root / "calib" / f"{fid}.txt")
        labels = re_.parse_object_labels(root / "label_2" / f"{fid}.txt")
        frames.append(re_.EvalFrame(fid, re_.gt_objects(labels, calib), re_.predictions_from_anno(det), calib,
                                    {"labels": labels}))
    return frames


def gt_conversion_check(frames, infos) -> dict:
    """OpenPCDet stores non-DontCare objects first in gt_boxes_lidar, in label order."""
    d_center, d_dim, d_head, n = [], [], [], 0
    for fr, info in zip(frames, infos):
        ref = info["annos"]["gt_boxes_lidar"]
        ours = [o.box for o in fr.gt if o.label != "DontCare"]
        if len(ours) != len(ref):
            raise ValueError(f"{fr.frame_id}: {len(ours)} repo boxes vs {len(ref)} OpenPCDet boxes")
        for b, r in zip(ours, ref):
            d_center.append(np.abs(b.center - r[:3]).max())
            d_dim.append(np.abs(b.dimensions - r[3:6]).max())
            d_head.append(_angle_diff(b.heading, r[6]))
            n += 1
    return {"num_objects": n, "max_abs_center_m": float(max(d_center)), "max_abs_dims_m": float(max(d_dim)),
            "max_abs_heading_rad": float(max(d_head))}


def iou_check(frames, gt_annos, det_annos, n_frames: int) -> dict:
    """repo IoU vs eval.py's overlaps on the same (GT, prediction) pairs of the first n_frames."""
    kitti_eval = ke.load_eval_module()
    out = {}
    for kind in ("3d", "bev"):
        diffs, n_pairs, n_overlap = [], 0, 0
        for fr, g, d in zip(frames[:n_frames], gt_annos, det_annos):
            k = int((g["name"] != "DontCare").sum())
            if k == 0 or len(d["name"]) == 0:
                continue
            if kind == "3d":
                gb = np.c_[g["location"][:k], g["dimensions"][:k], g["rotation_y"][:k]]
                db = np.c_[d["location"], d["dimensions"], d["rotation_y"]]
                ref = kitti_eval.d3_box_overlap(gb, db).astype(np.float64)
            else:
                gb = np.c_[g["location"][:k, [0, 2]], g["dimensions"][:k, [0, 2]], g["rotation_y"][:k]]
                db = np.c_[d["location"][:, [0, 2]], d["dimensions"][:, [0, 2]], d["rotation_y"]]
                ref = kitti_eval.bev_box_overlap(gb, db).astype(np.float64)
            fn = re_.iou_3d if kind == "3d" else re_.bev_iou
            gts = [o.box for o in fr.gt if o.label != "DontCare"]
            ours = np.array([[fn(a, p.box) for p in fr.predictions] for a in gts])
            diff = np.abs(ours - ref)
            diffs.append(diff.ravel())
            n_pairs += diff.size
            n_overlap += int(((ours > 0) | (ref > 0)).sum())
        allv = np.concatenate(diffs)
        out[kind] = {"frames": n_frames, "pairs": n_pairs, "overlapping_pairs": n_overlap,
                     "max_abs_diff": float(allv.max()), "pairs_diff_gt_1e-3": int((allv > 1e-3).sum()),
                     "pairs_diff_gt_1e-2": int((allv > 1e-2).sum())}
    return out


def with_openpcdet_gt(frames, infos) -> list[re_.EvalFrame]:
    """Copies of `frames` whose GT boxes are OpenPCDet's gt_boxes_lidar instead of label_to_box."""
    out = []
    for fr, info in zip(frames, infos):
        ref = iter(info["annos"]["gt_boxes_lidar"])
        gts = [re_.GTObject(BoundingBox3D.from_array(next(ref), label=o.label) if o.label != "DontCare" else None,
                            o.label, o.bbox2d, o.truncated, o.occluded) for o in fr.gt]
        out.append(re_.EvalFrame(fr.frame_id, gts, fr.predictions, fr.calib, fr.meta))
    return out


def decomposition(frames, infos, gt_annos, det_annos, cfg, openpcdet_r40) -> dict:
    """Where the aligned difference comes from, one cause at a time (3D, per difficulty):
    a) GT boxes from OpenPCDet's conversion instead of the repo's label_to_box,
    b) a) + OpenPCDet's recall sampling applied to the repo's own TP/FP records,
    c) OpenPCDet evaluator with detections below the 25 px height limit removed beforehand
       (tests whether the different handling of small detections matters; moderate/hard only)."""
    kitti_eval = ke.load_eval_module()
    opcdet_frames = with_openpcdet_gt(frames, infos)
    out = {"repo_gt_openpcdet": {}, "repo_gt_openpcdet_kitti_sampling": {}}
    for diff in ke.DIFFICULTIES:
        evs: dict = {}
        res = re_.evaluate(opcdet_frames, {**re_.REPO_DEFAULT, **cfg["aligned_rules"], "difficulty": diff}, evaluators=evs)
        out["repo_gt_openpcdet"][diff] = {c: 100.0 * res[c]["AP40"] for c in ke.CLASSES}
        out["repo_gt_openpcdet_kitti_sampling"][diff] = {
            c: 100.0 * re_.kitti_sampled_ap(evs[c], c, kitti_eval.get_thresholds) for c in ke.CLASSES}
    small = []
    for d in det_annos:
        keep = (d["bbox"][:, 3] - d["bbox"][:, 1]) >= re_.MIN_HEIGHT["moderate"]
        small.append({k: v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v for k, v in d.items()})
    _, res = ke.official_eval(gt_annos, small)
    out["openpcdet_without_small_detections"] = {
        d: {c: res[f"{c}_3d/{d}_R40"] for c in ke.CLASSES} for d in ("moderate", "hard")}
    out["openpcdet_R40_3d"] = {d: {c: openpcdet_r40[c]["3d"][d] for c in ke.CLASSES} for d in ke.DIFFICULTIES}
    return out


def ap_only(summary: dict) -> dict:
    return {c: {"AP40": 100.0 * v["AP40"], "num_gt": v["num_gt"]} for c, v in summary.items()}


def tracking_dev_section(cfg: dict) -> dict:
    """Repo metric on the dev tracking sequences, detections from the system's own Detector."""
    from evaluation.runner import load_or_detect
    from ingestion.loader import parse_tracking_labels
    from ingestion.sequence import SequenceLoader

    access = check_split_access("dev")
    seqs = load_split()["dev"]
    root = KITTI_TRACKING_ROOT / "training"
    frames = []
    for seq in seqs:
        loader = SequenceLoader(KITTI_TRACKING_ROOT, "training", seq)
        calib = Calibration.from_file(root / "calib" / f"{seq}.txt")
        labels = parse_tracking_labels(root / "label_02" / f"{seq}.txt")
        indices = range(len(loader))
        dets = load_or_detect(loader, indices, cache_dir("E0") / f"tracking_{seq}_detections.json", True,
                              split="training", fov=cfg["detector_fov"])
        for i, boxes in zip(indices, dets):
            lab = labels.get(i, [])
            frames.append(re_.EvalFrame(f"{seq}/{i:06d}", re_.gt_objects(lab, calib),
                                        [re_.Prediction(b) for b in boxes], calib, {"labels": lab}))
    counts: dict[str, int] = {}
    for fr in frames:
        for o in fr.gt:
            counts[o.label] = counts.get(o.label, 0) + 1
    res = {name: ap_only(re_.evaluate(frames, {**re_.REPO_DEFAULT, **v})) for name, v in cfg["variants"].items()}
    return {"split_access": access, "sequences": seqs, "num_frames": len(frames), "label_counts": counts,
            "variants": res}


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    result_pkl = PROJECT_ROOT / cfg["openpcdet_result"]
    run = start_run("E0", cfg, seed=cfg["seed"], tag="metric_crosscheck", dataset="kitti_object",
                    split=cfg["split"], model="pointpillar", weights=cfg["weights"],
                    detector_fov={"val": OPENPCDET_FOV,
                                  "tracking_dev": cfg.get("tracking_dev", {}).get("detector_fov", "auto")})
    timing, metrics = {}, {}

    t = time.perf_counter()
    infos = ke.load_infos(cfg["split"])
    gt_annos, det_annos = ke.align_to_infos(ke.load_predictions(result_pkl), infos)
    text, official = ke.official_eval(gt_annos, det_annos)
    metrics["openpcdet"] = {"R40": ke.r40_table(official), "all": official, "text": text,
                            "num_frames": len(infos)}
    if cfg.get("reference_eval_dict"):
        with open(PROJECT_ROOT / cfg["reference_eval_dict"], "rb") as f:
            reference = pickle.load(f)
        diffs = {k: abs(official[k] - float(v)) for k, v in reference.items()}
        metrics["openpcdet"]["vs_unmodified_evaluator"] = {
            "keys": len(diffs), "max_abs_diff_pp": max(diffs.values()),
            "nonzero": sorted(k for k, v in diffs.items() if v > 0)}
    timing["openpcdet_eval_s"] = time.perf_counter() - t
    print("OpenPCDet R40 3D:", {c: v["3d"] for c, v in metrics["openpcdet"]["R40"].items()})

    t = time.perf_counter()
    frames = build_val_frames(infos, det_annos)
    metrics["gt_conversion"] = gt_conversion_check(frames, infos)
    metrics["iou"] = iou_check(frames, gt_annos, det_annos, cfg["iou_check_frames"])
    timing["checks_s"] = time.perf_counter() - t
    print("gt conversion:", metrics["gt_conversion"], "\niou:", metrics["iou"])

    t = time.perf_counter()
    reference_default = ap_only(re_.evaluate_repo_default(frames))
    machinery_default = ap_only(re_.evaluate(frames, re_.REPO_DEFAULT))
    metrics["repo_default"] = {"runner_path": reference_default, "variant_machinery": machinery_default,
                               "identical": reference_default == machinery_default}
    timing["repo_default_s"] = time.perf_counter() - t
    print("repo default:", reference_default, "identical:", metrics["repo_default"]["identical"])

    t = time.perf_counter()
    metrics["variants"] = {}
    for name, v in cfg["variants"].items():
        metrics["variants"][name] = {"rules": {**re_.REPO_DEFAULT, **v},
                                     "AP": ap_only(re_.evaluate(frames, {**re_.REPO_DEFAULT, **v}))}
        aps = ", ".join(f"{c} {r['AP40']:.2f}" for c, r in metrics["variants"][name]["AP"].items())
        print(f"  {name}: {aps}")
    timing["variants_s"] = time.perf_counter() - t

    t = time.perf_counter()
    aligned = {}
    for kind in ("3d", "bev"):
        for diff in ke.DIFFICULTIES:
            rules = {**re_.REPO_DEFAULT, **cfg["aligned_rules"], "difficulty": diff, "iou": kind}
            ours = ap_only(re_.evaluate(frames, rules))
            aligned[f"{kind}/{diff}"] = {
                c: {"repo_AP40": ours[c]["AP40"], "openpcdet_R40": metrics["openpcdet"]["R40"][c][kind][diff],
                    "diff_pp": ours[c]["AP40"] - metrics["openpcdet"]["R40"][c][kind][diff],
                    "num_gt_repo": ours[c]["num_gt"]}
                for c in ke.CLASSES}
            print(f"  aligned {kind}/{diff}: " + ", ".join(f"{c} {v['repo_AP40']:.2f} vs {v['openpcdet_R40']:.2f}"
                                                         for c, v in aligned[f"{kind}/{diff}"].items()))
    metrics["aligned"] = {"rules": cfg["aligned_rules"], "results": aligned,
                          "max_abs_diff_pp": max(abs(v["diff_pp"]) for r in aligned.values() for v in r.values())}
    timing["aligned_s"] = time.perf_counter() - t

    t = time.perf_counter()
    metrics["decomposition"] = decomposition(frames, infos, gt_annos, det_annos, cfg, metrics["openpcdet"]["R40"])
    timing["decomposition_s"] = time.perf_counter() - t

    if cfg.get("tracking_dev", {}).get("enabled"):
        t = time.perf_counter()
        metrics["tracking_dev"] = tracking_dev_section(cfg["tracking_dev"])
        timing["tracking_dev_s"] = time.perf_counter() - t
        print("tracking dev:", metrics["tracking_dev"]["variants"])

    path = finish_run(run, metrics=metrics, timing=timing,
                      notes="Predictions from OpenPCDet tools/test.py (result.pkl). OpenPCDet evaluator run "
                            "with the CPU rotate_iou stand-in (experiments/lib/kitti_eval.py).")
    print(f"result: {path}")


if __name__ == "__main__":
    main()
