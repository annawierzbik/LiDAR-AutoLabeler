"""Detection + tracking of one detector on a set of KITTI tracking sequences (system defaults, no tuning).

    python -m experiments.run_model_tracking_eval --config experiments/configs/E0_leakage_tracking.yaml --set model=clean_666

One model per process: detection.detector.Detector writes into pcdet's global config, so models must
not share a process. Detections are cached in results/<experiment_id>/cache/<model>/<seq>.json.
The tracker runs with the system's default parameters and is not tuned here, so evaluating sequences
of the v2 test split does not use them for tuning; the result file records this as `split_access`.
"""
from __future__ import annotations

import json

from detection.fov import sequence_mode
from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import KITTI_TRACKING_ROOT, PROJECT_ROOT, SPLITS_DIR
from experiments.lib.results_io import finish_run, start_run
from experiments.lib.tracking_eval import (
    RAW_THRESHOLD,
    detect_sequence,
    detection_cache,
    evaluate_sequences,
)
from ingestion.sequence import SequenceLoader


def resolve_sequences(spec) -> list[str]:
    """A list, or "<split file>:<dev|test|all|excluded>" (e.g. "kitti_tracking_dev_test.json:excluded")."""
    if isinstance(spec, list):
        return [f"{int(s):04d}" for s in spec]
    file, _, part = spec.partition(":")
    doc = json.loads((SPLITS_DIR / file).read_text(encoding="utf-8"))
    if part == "all":
        return sorted(doc.get("sequences") or (doc["dev"] + doc["test"] + doc.get("excluded", [])))
    return sorted(doc[part])


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    model = cfg["models"][cfg["model"]]
    seqs = resolve_sequences(cfg["sequences"])
    fov = cfg["detector_fov"]
    resolved = {s: sequence_mode(SequenceLoader(KITTI_TRACKING_ROOT, "training", s), fov) for s in seqs}
    run = start_run(cfg["experiment_id"], cfg, seed=cfg["seed"], tag=f"tracking_{cfg['model']}",
                    dataset="kitti_tracking", split=str(cfg["sequences"]), model=cfg["model"],
                    weights=PROJECT_ROOT / model["weights"],
                    detector_fov=next(iter(set(resolved.values()))) if len(set(resolved.values())) == 1 else resolved)

    detector = None
    detections, det_seconds = {}, {}
    for s in seqs:
        cache = detection_cache(cfg["experiment_id"], cfg["model"], resolved[s], s)
        if detector is None and not cache.exists():
            from detection.detector import Detector  # heavy import, only when detecting
            detector = Detector(cfg_file=PROJECT_ROOT / model["cfg"], checkpoint=PROJECT_ROOT / model["weights"],
                                score_threshold=RAW_THRESHOLD, fov=fov)
        detections[s], det_seconds[s] = detect_sequence(detector, s, cache)
        print(f"  {s}: {len(detections[s])} frames" + (f", detected in {det_seconds[s]:.0f} s" if det_seconds[s] else " (cache)"))

    trk = cfg["tracking"]
    res = evaluate_sequences(seqs, detections, trk["threshold"], trk.get("tracker", {}), trk["match_iou"])
    metrics = {
        "split_access": {"purpose": cfg["purpose"], "tuned": False,
                         "parameters": "system defaults (tracking section of the config)"},
        "sequences": seqs,
        **res,
    }
    p, d = res["pooled_tracking"], res["pooled_detection"]
    print(f"pooled MOTA {p['MOTA']:.4f} IDS {p['id_switches']} | AP40 " +
          ", ".join(f"{c} {d[c]['AP40'] * 100:.2f}" for c in ("Car", "Pedestrian", "Cyclist")))
    path = finish_run(run, metrics=metrics, timing={"detection_s_per_sequence": det_seconds},
                      notes="tracker and score threshold: system defaults, not tuned")
    print(f"result: {path}")


if __name__ == "__main__":
    main()
