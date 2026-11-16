r"""Run the UNMODIFIED OpenPCDet KITTI evaluator (eval.py + rotate_iou.py with numba.cuda) on a
tools/test.py result.pkl, in a separate minimal environment where numba.cuda still compiles it.
Used once in E0 to validate experiments/lib/kitti_eval.py; needs only numpy and numba.

    py -3.11 -m venv $env:TEMP\numba060
    & $env:TEMP\numba060\Scripts\python -m pip install numpy==1.26.4 numba==0.60.0
    & $env:TEMP\numba060\Scripts\python -I experiments\tools\reference_kitti_eval.py `
        OpenPCDet\pcdet\datasets\kitti\kitti_object_eval_python `
        OpenPCDet\data\kitti\kitti_infos_val.pkl `
        OpenPCDet\output\kitti_models\pointpillar\e0_crosscheck\eval\epoch_7728\val\default\result.pkl `
        results\E0\cache\reference_eval_dict.pkl

The two OpenPCDet files are copied byte for byte into a temporary package (eval.py uses a relative
import), so nothing else from pcdet (torch, compiled ops) is imported.
"""
from __future__ import annotations

import copy
import pickle
import shutil
import sys
import tempfile
from pathlib import Path


def main(eval_dir: str, infos_pkl: str, result_pkl: str, out_pkl: str) -> None:
    with open(infos_pkl, "rb") as f:
        infos = pickle.load(f)
    with open(result_pkl, "rb") as f:
        dets = {d["frame_id"]: d for d in pickle.load(f)}
    gt = [copy.deepcopy(i["annos"]) for i in infos]
    dt = [copy.deepcopy(dets[i["point_cloud"]["lidar_idx"]]) for i in infos]

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "kitti_eval_unmodified"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        for name in ("eval.py", "rotate_iou.py"):
            shutil.copyfile(Path(eval_dir) / name, pkg / name)
        sys.path.insert(0, tmp)
        from kitti_eval_unmodified import eval as kitti_eval
        text, result = kitti_eval.get_official_eval_result(gt, dt, ["Car", "Pedestrian", "Cyclist"])

    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump({k: float(v) for k, v in result.items()}, f)
    print(text)
    print(f"saved {len(result)} values to {out_pkl}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
