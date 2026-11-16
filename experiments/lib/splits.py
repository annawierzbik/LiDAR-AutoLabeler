"""Dev/test split of the KITTI tracking sequences and the guard that keeps tuning away from test.

Protocol: parameters are tuned on `dev`; `test` is evaluated once, with parameters frozen from a
dev run. A script reaches the test split only through `check_split_access(..., final=True,
frozen_from=<dev run id>)`, and the dev run must exist and must have been run on dev.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from experiments.lib.paths import RESULTS_ROOT, SPLITS_DIR

TRACKING_SPLIT_FILE = SPLITS_DIR / "kitti_tracking_dev_test.json"


class SplitAccessError(PermissionError):
    """A script tried to use the test split without the --final protocol."""


def make_dev_test_split(sequences: list[str], test_fraction: float, seed: int) -> dict[str, list[str]]:
    """Whole sequences, seeded and independent of the input order. At least one sequence per side
    when there are two or more."""
    seqs = sorted(set(sequences))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = random.Random(seed)
    shuffled = seqs[:]
    rng.shuffle(shuffled)
    n_test = min(max(1, round(len(seqs) * test_fraction)), len(seqs) - 1) if len(seqs) > 1 else 0
    return {"dev": sorted(shuffled[n_test:]), "test": sorted(shuffled[:n_test])}


def save_split(split: dict, path: Path = TRACKING_SPLIT_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return path


def load_split(path: Path = TRACKING_SPLIT_FILE) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} does not exist; create it with experiments.run_leakage_check")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _find_run(run_id: str, results_root: Path) -> dict | None:
    for p in Path(results_root).glob(f"*/{run_id}.json"):
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def check_split_access(split_name: str, final: bool = False, frozen_from: str | None = None,
                       results_root: Path = RESULTS_ROOT) -> dict:
    """Raise SplitAccessError unless using `split_name` follows the protocol. Returns the
    provenance to log in the result file: {"split", "final", "frozen_from"}."""
    if split_name != "test":
        if final:
            raise SplitAccessError("--final is only meaningful for the test split")
        return {"split": split_name, "final": False, "frozen_from": None}
    if not final:
        raise SplitAccessError("the test split is only evaluated with --final (parameters frozen on dev)")
    if not frozen_from:
        raise SplitAccessError("--final needs --frozen-from <dev run id> naming where the parameters come from")
    source = _find_run(frozen_from, results_root)
    if source is None:
        raise SplitAccessError(f"run '{frozen_from}' not found under {results_root}")
    if source.get("split") != "dev":
        raise SplitAccessError(f"run '{frozen_from}' was on split '{source.get('split')}', not 'dev'")
    return {"split": "test", "final": True, "frozen_from": frozen_from}


def fraction_summary(split: dict, frames_per_sequence: dict[str, int]) -> dict:
    """Number of sequences and frames per side, to judge how balanced the split is."""
    total = sum(frames_per_sequence[s] for side in ("dev", "test") for s in split[side]) or math.nan
    return {side: {"sequences": len(split[side]),
                   "frames": sum(frames_per_sequence[s] for s in split[side]),
                   "frame_fraction": sum(frames_per_sequence[s] for s in split[side]) / total}
            for side in ("dev", "test")}
