"""E0: which points the detector sees, and what that does to KITTI val detection.

    python -m experiments.run_detector_path_check --config experiments/configs/E0_detector_path_check.yaml --set section=azimuth
    python -m experiments.run_detector_path_check --config experiments/configs/E0_detector_path_check.yaml --set section=model model=zoo

section=azimuth (no model): azimuth atan2(y, x) of the points OpenPCDet's get_fov_flag keeps
    (FOV_POINTS_ONLY, real image size of every frame) over all KITTI val frames: min/max and the
    |azimuth| quantiles; the 99.5 % quantile is the default theta of detection.fov's "azimuth" mode.
    Also the share of GT boxes (Car/Pedestrian/Cyclist) whose center lies outside |azimuth| <= theta.
section=model (one model per process: the Detector writes pcdet's global config):
    path "test_py"   result.pkl of tools/test.py (experiments.run_eval_detector_val; KittiDataset with
                     FOV_POINTS_ONLY)
    path "wrapper_<mode>"  detection.detector.Detector on the raw .bin with the crop <mode>
                     (none / camera with the frame's real image size / azimuth), score threshold as
                     test.py, converted to KITTI annos with OpenPCDet's generate_prediction_dicts
    path "wrapper_none_box_fov"  wrapper_none with the boxes whose center is outside get_fov_flag dropped
    per path: boxes per frame (score >= 0.1 / 0.5), share of boxes outside the camera FOV (box center,
    get_fov_flag) and AP R40 (OpenPCDet evaluator, experiments.lib.kitti_eval).
"""
from __future__ import annotations

import json
import time

import numpy as np

from experiments.lib import kitti_eval as ke
from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import KITTI_OBJECT_ROOT, PROJECT_ROOT
from experiments.lib.results_io import OPENPCDET_FOV, finish_run, start_run

TRAINING = KITTI_OBJECT_ROOT / "training"


def _pcdet():
    import torch  # noqa: F401  (before pcdet: Windows DLL loading)
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from pcdet.utils.calibration_kitti import Calibration

    import detection.detector  # noqa: F401  (puts OpenPCDet on sys.path)
    return KittiDataset, Calibration


def _frames(infos):
    for info in infos:
        fid = info["point_cloud"]["lidar_idx"]
        pts = np.fromfile(TRAINING / "velodyne" / f"{fid}.bin", dtype=np.float32).reshape(-1, 4)
        yield info, fid, pts


def _gt_boxes(info, classes):
    a = info["annos"]
    boxes = a["gt_boxes_lidar"]
    names = a["name"][: len(boxes)]                     # DontCare rows come last and have no lidar box
    keep = np.isin(names, classes)
    return boxes[keep], names[keep]


def gt_outside_azimuth(infos, theta_deg: float, classes) -> dict:
    out = {c: [0, 0] for c in classes}
    for info in infos:
        boxes, names = _gt_boxes(info, classes)
        az = np.degrees(np.abs(np.arctan2(boxes[:, 1], boxes[:, 0])))
        for c in classes:
            sel = names == c
            out[c][0] += int(sel.sum())
            out[c][1] += int((az[sel] > theta_deg).sum())
    total = [sum(v[0] for v in out.values()), sum(v[1] for v in out.values())]
    return {"theta_deg": theta_deg,
            **{c: {"gt": n, "outside": k, "share": k / n if n else 0.0} for c, (n, k) in out.items()},
            "all": {"gt": total[0], "outside": total[1], "share": total[1] / total[0] if total[0] else 0.0}}


def azimuth_section(cfg, infos) -> dict:
    KittiDataset, Calibration = _pcdet()
    edges = np.linspace(0.0, 180.0, int(180 / cfg["hist_bin_deg"]) + 1)
    hist = np.zeros(len(edges) - 1, np.int64)
    lo, hi, n = 180.0, -180.0, 0
    gt_cam = {c: [0, 0] for c in ke.CLASSES}            # GT centers outside get_fov_flag, for comparison
    for info, fid, pts in _frames(infos):
        calib = Calibration(TRAINING / "calib" / f"{fid}.txt")
        shape = info["image"]["image_shape"]
        keep = KittiDataset.get_fov_flag(calib.lidar_to_rect(pts[:, :3]), shape, calib)
        az = np.degrees(np.arctan2(pts[keep, 1], pts[keep, 0]))
        if len(az):
            lo, hi, n = min(lo, float(az.min())), max(hi, float(az.max())), n + len(az)
            hist += np.histogram(np.abs(az), edges)[0]
        boxes, names = _gt_boxes(info, ke.CLASSES)
        if len(boxes):
            inside = KittiDataset.get_fov_flag(calib.lidar_to_rect(boxes[:, :3]), shape, calib)
            for c in ke.CLASSES:
                gt_cam[c][0] += int((names == c).sum())
                gt_cam[c][1] += int(((names == c) & ~inside).sum())
    cdf = np.cumsum(hist) / n
    quant = {f"{q:g}": float(edges[1:][np.searchsorted(cdf, q / 100)]) for q in cfg["quantiles"]}
    theta = quant[f"{cfg['theta_quantile']:g}"]
    tot = [sum(v[0] for v in gt_cam.values()), sum(v[1] for v in gt_cam.values())]
    return {"points_kept_by_get_fov_flag": n, "azimuth_min_deg": lo, "azimuth_max_deg": hi,
            "abs_azimuth_quantiles_deg": quant, "hist_bin_deg": cfg["hist_bin_deg"],
            "theta_quantile": cfg["theta_quantile"], "theta_deg": theta,
            "gt_outside_theta": gt_outside_azimuth(infos, theta, ke.CLASSES),
            "gt_center_outside_camera_fov": {
                **{c: {"gt": a, "outside": b, "share": b / a if a else 0.0} for c, (a, b) in gt_cam.items()},
                "all": {"gt": tot[0], "outside": tot[1], "share": tot[1] / tot[0] if tot[0] else 0.0}}}


def _box_stats(annos, infos, calibs, KittiDataset) -> dict:
    out = {}
    for th in (0.1, 0.5):
        n, outside = [], 0
        for a, info in zip(annos, infos):
            boxes = a["boxes_lidar"][a["score"] >= th]
            n.append(len(boxes))
            if len(boxes):
                calib = calibs[a["frame_id"]]
                rect = calib.lidar_to_rect(boxes[:, :3])
                outside += int((~KittiDataset.get_fov_flag(rect, info["image"]["image_shape"], calib)).sum())
        n = np.array(n)
        out[f"score>={th:g}"] = {"boxes": int(n.sum()), "per_frame_mean": float(n.mean()),
                                 "per_frame_median": float(np.median(n)), "outside_fov": outside,
                                 "outside_fov_share": outside / n.sum() if n.sum() else 0.0}
    return out


def _box_fov_only(annos, infos, calibs, KittiDataset):
    res = []
    for a, info in zip(annos, infos):
        n = len(a["boxes_lidar"])
        if n == 0:
            res.append(a)
            continue
        calib = calibs[a["frame_id"]]
        flag = KittiDataset.get_fov_flag(calib.lidar_to_rect(a["boxes_lidar"][:, :3]),
                                         info["image"]["image_shape"], calib)
        res.append({k: (v[flag] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in a.items()})
    return res


def _ap(annos, infos) -> dict:
    gt, det = ke.align_to_infos(annos, infos)
    _, res = ke.official_eval(gt, det)
    return ke.r40_table(res)


def model_section(cfg, infos) -> tuple[dict, dict, dict]:
    import torch

    from detection.detector import Detector
    from detection.fov import AZIMUTH_THETA_DEG, describe
    from ingestion.frame import Frame
    from ingestion.loader import parse_calib_file

    KittiDataset, Calibration = _pcdet()
    model = cfg["models"][cfg["model"]]
    calibs = {i["point_cloud"]["lidar_idx"]: Calibration(TRAINING / "calib" / f"{i['point_cloud']['lidar_idx']}.txt")
              for i in infos}
    val = json.loads((PROJECT_ROOT / model["val_result"]).read_text(encoding="utf-8"))
    paths = {"test_py": ke.align_to_infos(ke.load_predictions(val["metrics"]["result_pkl"]), infos)[1]}
    timing, modes = {}, {}
    theta = cfg.get("azimuth_theta_deg") or AZIMUTH_THETA_DEG

    for mode in cfg["modes"]:
        det = Detector(cfg_file=PROJECT_ROOT / model["cfg"], checkpoint=PROJECT_ROOT / model["weights"],
                       score_threshold=cfg["score_threshold"], fov=mode, fov_theta_deg=theta)
        names, annos, t0 = det.class_names, [], time.perf_counter()
        for info, fid, pts in _frames(infos):
            frame = Frame("val", int(fid), pts, calib=parse_calib_file(TRAINING / "calib" / f"{fid}.txt"))
            h, w = info["image"]["image_shape"]
            boxes = det.predict(frame, image_size=(int(w), int(h)))
            pred = {"pred_boxes": torch.tensor(np.array([b.to_array() for b in boxes]).reshape(-1, 7), dtype=torch.float32),
                    "pred_scores": torch.tensor([b.score for b in boxes], dtype=torch.float32),
                    "pred_labels": torch.tensor([names.index(b.label) + 1 for b in boxes], dtype=torch.long)}
            batch = {"calib": [calibs[fid]], "image_shape": torch.tensor(np.array([info["image"]["image_shape"]])),
                     "frame_id": np.array([fid])}
            annos += KittiDataset.generate_prediction_dicts(batch, [pred], names)
        timing[f"wrapper_{mode}_s"] = time.perf_counter() - t0
        modes[mode] = describe(det.last_fov, theta)
        paths[f"wrapper_{mode}"] = annos
        del det
        torch.cuda.empty_cache()
    if "none" in cfg["modes"]:
        paths["wrapper_none_box_fov"] = _box_fov_only(paths["wrapper_none"], infos, calibs, KittiDataset)

    t0 = time.perf_counter()
    metrics = {p: {"boxes": _box_stats(a, infos, calibs, KittiDataset), "AP_R40": _ap(a, infos)}
               for p, a in paths.items()}
    timing["evaluation_s"] = time.perf_counter() - t0
    metrics["gt_outside_azimuth"] = gt_outside_azimuth(infos, theta, ke.CLASSES)
    fov = {"test_py": OPENPCDET_FOV, **{f"wrapper_{m}": d for m, d in modes.items()},
           "wrapper_none_box_fov": "none + box-center filter (get_fov_flag)"}
    return metrics, timing, fov


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    infos = ke.load_infos("val")
    if cfg.get("max_frames"):
        infos = infos[: cfg["max_frames"]]
    if cfg["section"] == "azimuth":
        run = start_run("E0", cfg, seed=cfg["seed"], tag="detector_path_check_azimuth", dataset="kitti_object",
                        split="val", detector_fov=OPENPCDET_FOV)
        t0 = time.perf_counter()
        metrics = azimuth_section(cfg["azimuth"], infos)
        path = finish_run(run, metrics=metrics, timing={"total_s": time.perf_counter() - t0},
                          notes="points kept by OpenPCDet's get_fov_flag; no model")
        print(f"azimuth min/max {metrics['azimuth_min_deg']:.2f}/{metrics['azimuth_max_deg']:.2f} deg, "
              f"theta({cfg['azimuth']['theta_quantile']}%) = {metrics['theta_deg']:.2f} deg")
    else:
        model = cfg["models"][cfg["model"]]
        run = start_run("E0", cfg, seed=cfg["seed"], tag=f"detector_path_check_{cfg['model']}",
                        dataset="kitti_object", split="val", model=cfg["model"],
                        weights=PROJECT_ROOT / model["weights"])
        metrics, timing, fov = model_section(cfg, infos)
        run["detector_fov"] = fov
        path = finish_run(run, metrics=metrics, timing=timing,
                          notes="same KITTI val frames through tools/test.py and through detection.detector.Detector")
        for p, m in metrics.items():
            if p != "gt_outside_azimuth":
                print(f"{p:22s} boxes/frame@0.5 {m['boxes']['score>=0.5']['per_frame_mean']:.2f} "
                      f"outside {m['boxes']['score>=0.5']['outside_fov_share'] * 100:.1f}% | AP3D mod "
                      + " ".join(f"{c} {m['AP_R40'][c]['3d']['moderate']:.2f}" for c in ke.CLASSES))
    print(f"result: {path}")


if __name__ == "__main__":
    main()
