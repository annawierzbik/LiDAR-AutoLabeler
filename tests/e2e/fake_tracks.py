"""Fake detection + tracking for the end-to-end tests (no GPU, model or KITTI data).

Fake tracks per frame i: track 0, a Car at x = 10 + i in every frame; track 1, a Pedestrian at
x = 20 in frames 0..2. LAL_E2E_TRACK_DELAY slows it down (seconds per frame, to watch progress).
"""
from __future__ import annotations

import os
import time

from common.boxes import BoundingBox3D


def fake_track(seq, indices, threshold, fov="auto", detector=None, should_stop=None, progress=None):
    from pipeline.detect_track import Cancelled

    delay = float(os.environ.get("LAL_E2E_TRACK_DELAY", "0"))
    out = []
    for n, i in enumerate(indices, 1):
        if should_stop is not None and should_stop():
            raise Cancelled(f"stopped after {n - 1} of {len(indices)} frames")
        time.sleep(delay)
        if progress is not None:
            progress(n / len(indices))
        boxes = [BoundingBox3D(10.0 + i, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, 0)]
        if i < 3:
            boxes.append(BoundingBox3D(20.0, 3.0, -0.9, 0.8, 0.6, 1.7, 0.0, "Pedestrian", 0.8, 1))
        out.append(boxes)
    return out
