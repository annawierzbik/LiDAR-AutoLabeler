"""Seed every random number generator an experiment can touch and report what was set."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seeds(seed: int, deterministic: bool = False) -> dict:
    """Seed python, numpy and (if importable) torch + CUDA. Returns the values for the result file.

    `deterministic=True` additionally asks cuDNN for deterministic kernels (slower); it does not make
    every CUDA op deterministic (e.g. scatter-add in spconv), so it is logged, not promised.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # only affects subprocesses started after this point
    info = {"seed": seed, "python": seed, "numpy": seed, "torch": None, "cuda": None,
            "cudnn_deterministic": None}
    try:
        import torch
    except ImportError:
        return info
    torch.manual_seed(seed)
    info["torch"] = seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        info["cuda"] = seed
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    info["cudnn_deterministic"] = deterministic
    return info
