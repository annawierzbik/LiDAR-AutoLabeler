"""Detectors for detection jobs (docs/adr/0015).

GPU worker processes (Celery): a process loads one detector configuration (LAL_WORKER_DETECTOR_*:
config, checkpoint, score threshold) once, on its first detection job, and keeps it. A job states
the configuration it needs (the cache metadata of its run); if that is not the one loaded, the job
fails with DetectorMismatch naming both, instead of silently running another model (`get_detector`).

Thread mode (LAL_JOB_QUEUE=thread, inside the API process): as before the job queue existed, every
job makes a detector with its own configuration and releases it afterwards (`make_for_job`,
`release`), so the API process holds no model between jobs.
"""
from __future__ import annotations

import importlib
import threading

import detection.paths
from infra.settings import get_settings
from verification import cache

COMPARED = ("detector_config_hash", "checkpoint_hash", "score_threshold")

_lock = threading.Lock()
_loaded = None     # the detector, once made


class DetectorMismatch(Exception):
    pass


def worker_config() -> dict:
    """What this process loads: paths, fingerprints and score threshold (the cache-metadata fields)."""
    s = get_settings()
    cfg = s.worker_detector_config or detection.paths.DEFAULT_CFG
    ckpt = s.worker_detector_checkpoint or detection.paths.DEFAULT_CKPT
    return {"detector_config": str(cfg), "detector_config_hash": cache.config_hash(cfg),
            "checkpoint": str(ckpt), "checkpoint_hash": cache.file_hash(ckpt),
            "score_threshold": float(s.worker_score_threshold), "factory": s.worker_detector_factory}


def check(required: dict) -> None:
    mine = worker_config()
    diff = [f"{k}: job needs {required.get(k)!r}, worker has {mine[k]!r}" for k in COMPARED
            if required.get(k) != mine[k]]
    if diff:
        raise DetectorMismatch(
            "this worker has another detector loaded (" + "; ".join(diff) + "). Start a worker with the "
            "job's configuration (LAL_WORKER_DETECTOR_CONFIG, LAL_WORKER_DETECTOR_CHECKPOINT, "
            "LAL_WORKER_SCORE_THRESHOLD) for this job.")


def get_detector(required: dict, fov: str):
    """The process's detector, made on first use; DetectorMismatch if `required` is another one.
    `fov` is the point crop for this job (a per-call setting, not part of the model)."""
    global _loaded
    check(required)
    with _lock:
        if _loaded is None:
            mine = worker_config()
            _loaded = _factory()(cfg_file=mine["detector_config"], checkpoint=mine["checkpoint"],
                                 score_threshold=mine["score_threshold"], fov=fov)
        _loaded.fov = fov
        return _loaded


def _factory():
    module, _, name = get_settings().worker_detector_factory.partition(":")
    return getattr(importlib.import_module(module), name)


def make_for_job(required: dict, fov: str):
    """A detector with exactly the job's configuration, for one job (thread mode)."""
    return _factory()(cfg_file=required["detector_config"], checkpoint=required["checkpoint"],
                      score_threshold=required["score_threshold"], fov=fov)


def release() -> None:
    """Give the GPU memory of a dropped per-job detector back (thread mode). Call it after the last
    reference to the detector is gone."""
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def loaded() -> bool:
    return _loaded is not None


def reset() -> None:
    """Forget the loaded detector (tests)."""
    global _loaded
    with _lock:
        _loaded = None
