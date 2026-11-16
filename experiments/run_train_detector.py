"""Train an OpenPCDet detector with OpenPCDet's own tools/train.py, on a config derived from a model-zoo
config with only the keys listed under `overrides` changed.

    python -m experiments.run_train_detector --config experiments/configs/train_pointpillar_clean.yaml
    python -m experiments.run_train_detector --config ... --set train.epochs=1 tag=timing_bs4

The derived config is written to OpenPCDet/tools/cfgs/<cfg_group>/<name>.yaml (train.py names its
output directory after the config path: OpenPCDet/output/<cfg_group>/<name>/<tag>/) and copied into
the result file. stdout/stderr of train.py go to <output dir>/run_<run_id>.log; GPU memory is sampled
with nvidia-smi every `gpu_poll_s` seconds into <output dir>/gpu_<run_id>.csv.

train.py is started through experiments/tools/train_with_seed.py so that `seed` is honoured (train.py
itself always uses 666). train.py resumes from the newest checkpoint in its output directory, so an
existing ckpt/ folder stops this script unless --resume is given. After training, the last checkpoint
is exported without optimizer state as <output dir>/<name>_<tag>.pth (the format of the model-zoo
files, loadable with torch.load(weights_only=True) and detection.detector.Detector), and the per-epoch
training loss is read from train.py's log.
"""
from __future__ import annotations

import copy
import itertools
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from experiments.lib.config import base_parser, load_config
from experiments.lib.paths import OPENPCDET_ROOT
from experiments.lib.results_io import OPENPCDET_FOV, finish_run, start_run

TOOLS = OPENPCDET_ROOT / "tools"
LAUNCHER = Path(__file__).resolve().parent / "tools" / "train_with_seed.py"
LOSS_LINE = re.compile(r"Train:\s+(\d+)/\d+ .*?\[\s*(\d+)/(\d+).*?Loss: ([\d.]+) \(([\d.]+)\).*?LR: ([\d.e+-]+)")


def set_path(cfg: dict, dotted: str, value) -> None:
    """'DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST.0.USE_ROAD_PLANE' -> value (ints index lists)."""
    *parents, leaf = dotted.split(".")
    node = cfg
    for p in parents:
        node = node[int(p)] if isinstance(node, list) else node[p]
    if isinstance(node, list):
        node[int(leaf)] = value
    else:
        node[leaf] = value


def derive_config(base_rel: str, overrides: dict) -> dict:
    cfg = yaml.safe_load((TOOLS / base_rel).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        set_path(cfg, key, value)
    return cfg


def gpu_memory_peak(csv_path: Path) -> int | None:
    values = [int(m.group(1)) for m in re.finditer(r"(\d+) MiB", csv_path.read_text())] if csv_path.exists() else []
    return max(values) if values else None


def loss_curve(train_log: Path) -> dict:
    """Per epoch: mean of the logged loss values, the running average at the last logged iteration
    and the learning rate there (train.py logs every 50 iterations)."""
    per_epoch: dict[int, list] = {}
    for m in LOSS_LINE.finditer(train_log.read_text(encoding="utf-8", errors="replace")):
        per_epoch.setdefault(int(m.group(1)), []).append((float(m.group(4)), float(m.group(5)), float(m.group(6))))
    return {e: {"mean_logged_loss": sum(v[0] for v in vals) / len(vals), "running_avg_end": vals[-1][1],
                "lr_end": vals[-1][2], "n_logged": len(vals)} for e, vals in sorted(per_epoch.items())}


def export_weights(ckpt: Path, out: Path) -> Path:
    """Our own checkpoint (trusted) -> {epoch, it, model_state}, like the model-zoo files."""
    import torch
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch.save({"epoch": c["epoch"], "it": c["it"], "model_state": c["model_state"]}, out)
    return out


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--resume", action="store_true", help="let train.py continue from existing checkpoints")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    tr = cfg["train"]
    derived = derive_config(cfg["base_config"], cfg.get("overrides", {}))
    cfg_rel = f"cfgs/{cfg['cfg_group']}/{cfg['name']}.yaml"
    (TOOLS / cfg_rel).parent.mkdir(parents=True, exist_ok=True)
    (TOOLS / cfg_rel).write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")

    out_dir = OPENPCDET_ROOT / "output" / cfg["cfg_group"] / cfg["name"] / cfg["tag"]
    if any((out_dir / "ckpt").glob("*.pth")) and not args.resume:
        raise SystemExit(f"{out_dir / 'ckpt'} has checkpoints; train.py would resume. Use another tag or --resume")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = start_run(cfg["experiment_id"], cfg, seed=cfg["seed"], tag=f"train_{cfg['tag']}",
                    dataset="kitti_object", split=derived["DATA_CONFIG"]["DATA_SPLIT"]["train"],
                    model=cfg["name"], detector_fov=OPENPCDET_FOV)
    cmd = [sys.executable, str(LAUNCHER), str(cfg["seed"]), "--cfg_file", cfg_rel, "--extra_tag", cfg["tag"],
           "--batch_size", str(tr["batch_size"]), "--workers", str(tr["workers"]),
           "--epochs", str(tr["epochs"]), "--ckpt_save_interval", str(tr.get("ckpt_save_interval", 1)),
           "--max_ckpt_save_num", str(tr.get("max_ckpt_save_num", 30))]
    cmd.append("--fix_random_seed")
    log_path = out_dir / f"run_{run['run_id']}.log"
    gpu_path = out_dir / f"gpu_{run['run_id']}.csv"
    print("cmd:", " ".join(cmd), "\nlog:", log_path)

    with open(gpu_path, "w") as gpu_log:
        poller = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu",
                                   "--format=csv", "-l", str(cfg.get("gpu_poll_s", 2))], stdout=gpu_log)
        t0, start_wall = time.perf_counter(), time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=TOOLS, stdout=log, stderr=subprocess.STDOUT)
        wall = time.perf_counter() - t0
        poller.terminate()

    ckpts = sorted((out_dir / "ckpt").glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    mtimes = [p.stat().st_mtime for p in ckpts]
    trained = bool(ckpts) and ckpts[-1].stem == f"checkpoint_epoch_{tr['epochs']}"
    logs = sorted(out_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    weights = export_weights(ckpts[-1], out_dir / f"{cfg['name']}_{cfg['tag']}.pth") if trained else None
    metrics = {
        "weights": str(weights) if weights else None,
        "loss_curve": loss_curve(logs[-1]) if logs else {},
        "return_code": code,
        # train.py ends with an evaluation of the last checkpoint that crashes under numba 0.68
        # (see experiments/lib/kitti_eval.py); training itself succeeded if the last checkpoint exists
        "training_completed": trained,
        "epoch_durations_s": [b - a for a, b in itertools.pairwise(mtimes)],
        "first_checkpoint_after_start_s": (mtimes[0] - start_wall) if mtimes else None,
        "derived_config": derived,
        "derived_config_path": str(TOOLS / cfg_rel),
        "output_dir": str(out_dir),
        "peak_gpu_memory_mib_total_device": gpu_memory_peak(gpu_path),
        "checkpoints": {p.name: p.stat().st_size for p in ckpts},
    }
    path = finish_run(run, metrics=metrics, timing={"wall_s": wall},
                      notes=f"train.py log: {log_path}; GPU memory log: {gpu_path}")
    print(f"return code {code}, training completed {trained}, wall {wall:.0f} s, "
          f"epochs {[round(d) for d in metrics['epoch_durations_s']]} s, "
          f"peak GPU {metrics['peak_gpu_memory_mib_total_device']} MiB, checkpoints {metrics['checkpoints']}")
    print(f"result: {path}")
    if not trained:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    main()
