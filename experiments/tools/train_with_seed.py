"""Run OpenPCDet's tools/train.py with a seed other than its hard-coded 666.

    (cwd = OpenPCDet/tools)  python <repo>/experiments/tools/train_with_seed.py <seed> <train.py arguments> --fix_random_seed

train.py's --fix_random_seed always seeds with 666: common_utils.set_random_seed(666 + rank) and
build_dataloader(seed=666) (worker_init_fn seeds every worker with seed + worker_id). This launcher
replaces only those two values at runtime, in this process, and then runs train.py unchanged.
With seed 666 it behaves exactly like train.py.

It also makes torch.load default to weights_only=False (torch >= 2.6 defaults to True). train.py
resumes through load_params_with_optimizer, whose torch.load of a checkpoint with optimizer state
fails under weights_only=True; train.py swallows that in a bare except and silently starts from
epoch 0. Only checkpoints written by these training runs are loaded, so this is safe here.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "OpenPCDet" / "tools"


def main() -> None:
    seed = int(sys.argv[1])
    sys.argv = [str(TOOLS / "train.py"), *sys.argv[2:]]
    sys.path.insert(0, str(TOOLS))

    import torch  # noqa: I001  (before pcdet: Windows DLL loading)
    import pcdet.datasets
    from pcdet.utils import common_utils

    original_seed = common_utils.set_random_seed
    original_loader = pcdet.datasets.build_dataloader

    def set_random_seed(value):
        original_seed(seed + (value - 666))          # keeps train.py's "+ LOCAL_RANK"

    def build_dataloader(*args, **kwargs):
        if kwargs.get("seed") is not None:
            kwargs["seed"] = seed
        return original_loader(*args, **kwargs)

    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load
    common_utils.set_random_seed = set_random_seed
    pcdet.datasets.build_dataloader = build_dataloader   # train.py imports it by name afterwards
    print(f"[train_with_seed] seed {seed}", flush=True)
    runpy.run_path(str(TOOLS / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
