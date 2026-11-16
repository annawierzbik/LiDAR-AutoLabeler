"""KITTI val detection AP of one detector: OpenPCDet's tools/test.py for the predictions, OpenPCDet's
evaluator (experiments.lib.kitti_eval) for the numbers.

    python -m experiments.run_eval_detector_val --config experiments/configs/E0_leakage_val.yaml --set model=zoo

test.py writes result.pkl and then crashes in its own evaluation step (numba 0.68); the script uses
result.pkl if test.py got that far. Weights must be loadable with torch.load(weights_only=True)
(model-zoo files, or the export written by run_train_detector).
"""
from __future__ import annotations

import subprocess
import sys
import time

from experiments.lib import kitti_eval as ke
from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import OPENPCDET_ROOT, PROJECT_ROOT
from experiments.lib.results_io import OPENPCDET_FOV, finish_run, start_run

TOOLS = OPENPCDET_ROOT / "tools"


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    model = cfg["models"][cfg["model"]]
    cfg_file = PROJECT_ROOT / model["cfg"]
    cfg_rel = cfg_file.relative_to(TOOLS).as_posix()       # test.py names its output dir after it
    tag = f"eval_{cfg['model']}"
    out_dir = OPENPCDET_ROOT / "output" / cfg_file.parent.relative_to(TOOLS / "cfgs") / cfg_file.stem / tag
    run = start_run(cfg["experiment_id"], cfg, seed=cfg["seed"], tag=f"val_{cfg['model']}", dataset="kitti_object",
                    split="val", model=cfg["model"], weights=PROJECT_ROOT / model["weights"],
                    detector_fov=OPENPCDET_FOV)

    cmd = [sys.executable, "test.py", "--cfg_file", cfg_rel, "--ckpt", str(PROJECT_ROOT / model["weights"]),
           "--batch_size", str(cfg["batch_size"]), "--workers", "0", "--extra_tag", tag]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"test_{run['run_id']}.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        code = subprocess.call(cmd, cwd=TOOLS, stdout=log, stderr=subprocess.STDOUT)
    infer_s = time.perf_counter() - t0
    results = sorted(out_dir.glob("eval/**/val/default/result.pkl"), key=lambda p: p.stat().st_mtime)
    if not results or results[-1].stat().st_mtime < time.time() - infer_s - 5:
        raise SystemExit(f"test.py (exit {code}) wrote no new result.pkl; see {log_path}")

    t0 = time.perf_counter()
    gt_annos, det_annos = ke.align_to_infos(ke.load_predictions(results[-1]), ke.load_infos("val"))
    text, official = ke.official_eval(gt_annos, det_annos)
    eval_s = time.perf_counter() - t0
    r40 = ke.r40_table(official)
    metrics = {"R40": r40, "all": official, "text": text, "result_pkl": str(results[-1]), "test_py_exit": code}
    path = finish_run(run, metrics=metrics, timing={"test_py_s": infer_s, "evaluation_s": eval_s},
                      notes=f"test.py log: {log_path}")
    print("R40 3D moderate: " + ", ".join(f"{c} {r40[c]['3d']['moderate']:.2f}" for c in ke.CLASSES))
    print(f"result: {path}")


if __name__ == "__main__":
    main()
