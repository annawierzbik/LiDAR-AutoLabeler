# Experiment framework (thesis chapter "Ewaluacja")

Every experiment is a terminal command with a YAML config in `experiments/configs/`. Each run writes
one JSON file `results/<experiment_id>/<run_id>.json` with: experiment_id, run_id, timestamp, git
commit + dirty flag (repo and OpenPCDet), environment (library versions, GPU), full config, seeds,
dataset, split, model, weights, metrics, timing, notes. Notebooks in `notebooks/` only read these
files and write thesis tables (`outputs/tables/*.tex`, booktabs) and figures
(`outputs/figures/*.pdf|png`), labels in Polish by default (`LANG = "pl"`).

All commands are run from the repository root in PowerShell with the venv active
(`.\venv\Scripts\Activate.ps1`). Extra packages: `pip install -r experiments/requirements.txt "numpy==1.26.4"`.

## Data locations

| Data | Path |
|---|---|
| KITTI object (training/testing, ImageSets, devkit_object, OpenPCDet infos) | `OpenPCDet/data/kitti/` |
| KITTI tracking (training/testing: velodyne, calib, label_02) | `data/raw/` |
| Weights | `checkpoints/` (only `pointpillar_7728.pth` at the moment) |

OpenPCDet infos were generated once with
`cd OpenPCDet; python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`
(2.5 min; ~42 MB of pkl files + 169 MB gt_database).

## Leakage protocol

* KITTI detection AP: only the official val split (3769 frames) for models trained on train (3712).
* KITTI tracking: `experiments.run_leakage_check` maps every tracking frame to the object dataset
  by file content (byte-identical .bin) and the object frames to raw drives with
  `devkit_object/mapping`. Sequences containing frames of the detector's train split are excluded;
  the rest is split by whole sequences into dev/test (`splits/kitti_tracking_dev_test.json`,
  report in `splits/tracking_overlap_report.json`). The split file is not overwritten without `--overwrite`.
* Tuning only on dev. The test split is reachable only through
  `experiments.lib.splits.check_split_access("test", final=True, frozen_from=<dev run id>)`;
  scripts expose it as `--final --frozen-from <run_id>` and log it in the result file.

## E0 (Stage B)

```powershell
python -m experiments.run_env_report --config experiments/configs/E0_env.yaml
python -m experiments.run_leakage_check --config experiments/configs/E0_leakage.yaml
# reference predictions of the OpenPCDet pipeline (its own evaluation step then crashes, see below)
cd OpenPCDet/tools
python test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../../checkpoints/pointpillar_7728.pth --batch_size 4 --workers 0 --extra_tag e0_crosscheck
cd ../..
# optional: unmodified OpenPCDet evaluator in a separate numba 0.60 venv (see the script's docstring)
python -m experiments.run_metric_crosscheck --config experiments/configs/E0_metric_crosscheck.yaml
python -m pytest tests/test_experiments
```

Runtime on the RTX 4070 (measured in Stage B): env report < 10 s, leakage check ~1 min (hashes
~9.7k files), test.py ~2 min, metric cross-check ~4–6 min (OpenPCDet evaluator on CPU ~100 s per call;
detection of the 3 dev tracking sequences ~1 min, cached afterwards).

**OpenPCDet evaluator and numba.** `kitti_object_eval_python/rotate_iou.py` compiles numba.cuda
kernels at import time; with numba 0.68 this fails ("Signature mismatch"), so `tools/test.py`
crashes in evaluation and no OpenPCDet AP can be computed as installed. `experiments/lib/kitti_eval.py`
loads `eval.py` unchanged with a CPU port of `rotate_iou` (`experiments/lib/rotate_iou_cpu.py`).
E0 checked it against the unmodified evaluator under numba 0.60: all 36 AP values identical.

## E3 effort model (prepared before Stage C)

`experiments/lib/effort.py` (tests: `tests/test_experiments/test_effort.py`); matching rules in its
docstring. Per-class thresholds, defaults: Car tau_match 0.5 / tau_fix 0.7, Pedestrian and Cyclist
0.25 / 0.5 (tau_fix = official KITTI 3D IoU threshold). To do in E3, on the v2 dev split only:
sensitivity of the chosen operating point (score threshold) to tau_match in {0.25, 0.5} (all
classes) x tau_fix shifted by {-0.1, 0, +0.1} from the class defaults; combinations with
tau_fix <= tau_match are invalid and are listed as skipped, not run. Table in
`notebooks/03_threshold_and_effort.ipynb`.

## Overview

| Experiment | Script | Config | Results | Notebook | Thesis output |
|---|---|---|---|---|---|
| E0 env | `run_env_report.py` | `E0_env.yaml` | `results/E0/*_env_*.json` | `00_sanity_checks.ipynb` | – |
| E0 leakage + split | `run_leakage_check.py` | `E0_leakage.yaml` | `results/E0/*_leakage_*.json`, `splits/*.json` | `00_sanity_checks.ipynb` | `tab:e0_tracking_overlap` |
| E0 metric cross-check | `run_metric_crosscheck.py` | `E0_metric_crosscheck.yaml` | `results/E0/*_metric_crosscheck_*.json` | `00_sanity_checks.ipynb` | `tab:e0_metric_crosscheck`, `tab:e0_crosscheck_decomposition`, `tab:e0_metric_sensitivity_*`, `fig e0_metric_crosscheck_diff` |
| E1–E8 | Stages C–G | | | | |

## Layout

```
experiments/
  configs/        YAML per experiment
  lib/            shared code (paths, results_io, seeds, envinfo, config, timing, splits, leakage,
                  kitti_eval + rotate_iou_cpu, repo_eval, report, tracking_eval, effort)
  tools/          one-off helpers (reference_kitti_eval.py)
  run_*.py        entry points (python -m experiments.run_...)
splits/           dev/test split + overlap report (versioned)
results/E*/       result JSON files (results/ is git-ignored)
outputs/          figures/, tables/ (git-ignored)
notebooks/        analysis only
tests/test_experiments/
```
