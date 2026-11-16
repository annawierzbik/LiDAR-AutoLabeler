"""Wall-clock timing for GPU code: warm-up, torch.cuda.synchronize around each call, median/IQR/n."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import numpy as np


def _sync() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize(times_s: Iterable[float]) -> dict:
    t = np.asarray(list(times_s), dtype=np.float64) * 1000.0
    if t.size == 0:
        return {"n": 0, "median_ms": None, "iqr_ms": None, "q1_ms": None, "q3_ms": None,
                "mean_ms": None, "min_ms": None, "max_ms": None}
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return {"n": int(t.size), "median_ms": float(med), "iqr_ms": float(q3 - q1), "q1_ms": float(q1),
            "q3_ms": float(q3), "mean_ms": float(t.mean()), "min_ms": float(t.min()), "max_ms": float(t.max())}


def time_calls(fn: Callable, inputs: Iterable, warmup: int = 10, sync: bool = True) -> tuple[list, dict]:
    """Calls fn(x) for every x; the first `warmup` calls are run but not timed.
    Returns (outputs of all calls, summarize(timed durations))."""
    outputs, durations = [], []
    for i, x in enumerate(inputs):
        if sync:
            _sync()
        t0 = time.perf_counter()
        outputs.append(fn(x))
        if sync:
            _sync()
        if i >= warmup:
            durations.append(time.perf_counter() - t0)
    stats = summarize(durations)
    stats["warmup"] = warmup
    stats["cuda_synchronized"] = sync
    return outputs, stats
