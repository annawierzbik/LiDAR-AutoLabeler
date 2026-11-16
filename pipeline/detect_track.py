"""Detect + track a sequence and cache the result as JSON (with settings metadata, see
verification/cache.py).

    track_sequence(seq, indices, threshold, fov)   -> tracked boxes per frame
    save_results / load_results / parse_results    cache files: {start, stop, frames, meta}
"""
from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common.boxes import BoundingBox3D
from infra.blob import LocalBlobStore
from ingestion.sequence import SequenceLoader
from tracking.tracker import Tracker
from verification import cache


class Cancelled(Exception):
    """track_sequence stopped because should_stop() said so; nothing was produced."""


def track_sequence(seq: SequenceLoader, indices: range, threshold: float, fov: str = "auto", detector=None,
                   should_stop: Callable[[], bool] | None = None,
                   progress: Callable[[float], None] | None = None) -> list[list[BoundingBox3D]]:
    """Detect + track `indices` of `seq`. `detector`: an already loaded detector (a worker's), else
    one is made for this run. `should_stop` is asked before every frame (cooperative cancellation:
    raises Cancelled, the partial result is dropped); `progress(fraction)` is called after every frame."""
    if detector is None:
        from detection.detector import (
            Detector,  # heavy import (torch + pcdet), only when needed
        )

        detector = Detector(score_threshold=threshold, fov=fov)
    tracker = Tracker()
    results = []
    t0 = time.time()
    for n, i in enumerate(indices, 1):
        if should_stop is not None and should_stop():
            raise Cancelled(f"stopped after {n - 1} of {len(indices)} frames")
        results.append(tracker.update(detector.predict(seq[i])))
        if progress is not None:
            progress(n / len(indices))
        if n % 10 == 0 or n == len(indices):
            print(f"  {n}/{len(indices)} frames ({(time.time() - t0) / n:.2f} s/frame)")
    return results


def current_meta(threshold: float, fov: str, split: str | None) -> dict:
    """Settings of track_sequence(): default detector, `threshold`, resolved FOV crop
    (detection.fov.sequence_mode), default Tracker(); `split` = KITTI split or None for a folder."""
    return cache.make_meta(threshold, cache.tracker_params(Tracker()), fov=fov, split=split)


def save_results(path: Path, indices: range, results: list[list[BoundingBox3D]], meta: dict | None = None) -> None:
    data = {
        "start": indices.start, "stop": indices.stop,
        "frames": [[asdict(b) for b in boxes] for boxes in results],
    }
    if meta is not None:
        data["meta"] = meta
    LocalBlobStore(path.parent).put(path.name, json.dumps(data).encode("utf-8"))   # atomic


def write_results(blobs, key: str, indices: range, results: list[list[BoundingBox3D]], meta: dict | None) -> None:
    """save_results into a BlobStore (atomic put)."""
    data = {"start": indices.start, "stop": indices.stop, "frames": [[asdict(b) for b in boxes] for boxes in results]}
    if meta is not None:
        data["meta"] = meta
    blobs.put(key, json.dumps(data).encode("utf-8"))


def parse_results(data: dict):
    indices = range(data["start"], data["stop"])
    results = [[BoundingBox3D(**b) for b in boxes] for boxes in data["frames"]]
    return indices, results


def load_results(path: Path):
    return parse_results(json.loads(path.read_text()))


def print_stats(results: list[list[BoundingBox3D]]) -> None:
    lengths: Counter = Counter()
    labels: dict[int, str] = {}
    for boxes in results:
        for b in boxes:
            lengths[b.track_id] += 1
            labels[b.track_id] = b.label
    if not lengths:
        print("No tracks.")
        return
    per_class = Counter(labels.values())
    print(f"\nTracks: {len(lengths)}  ({dict(per_class)})")
    print(f"Mean track length: {np.mean(list(lengths.values())):.1f} frames, "
          f"longest: {max(lengths.values())}, single-frame tracks: "
          f"{sum(1 for v in lengths.values() if v == 1)}")
