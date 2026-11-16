"""Test doubles used across processes (e.g. by a Celery worker started from a test):

    LAL_WORKER_DETECTOR_FACTORY=tests.fakes:FakeDetector

FakeDetector takes the arguments of detection.detector.Detector and returns one Car per frame,
moving with the frame id. Environment knobs:
    LAL_FAKE_DETECTOR_DELAY         seconds per frame (time to cancel or kill a job in the middle)
    LAL_FAKE_DETECTOR_LOAD_SECONDS  sleep while "loading the model" (in the constructor)
    LAL_FAKE_DETECTOR_SLOW_FRAME    frame id that takes LAL_FAKE_DETECTOR_SLOW_SECONDS, spent in a
                                    busy Python loop (holds the GIL, like CPU-bound model code)
"""
from __future__ import annotations

import os
import time
import weakref

from common.boxes import BoundingBox3D

LOADS = 0                      # FakeDetector objects made in this process
ALIVE = weakref.WeakSet()      # those not garbage collected yet


class FakeDetector:
    def __init__(self, cfg_file=None, checkpoint=None, score_threshold=0.5, fov="auto", **_) -> None:
        global LOADS
        LOADS += 1
        self.score_threshold, self.fov = score_threshold, fov
        self.delay = float(os.environ.get("LAL_FAKE_DETECTOR_DELAY", "0"))
        self.calls = 0
        self.slow_frame = int(os.environ.get("LAL_FAKE_DETECTOR_SLOW_FRAME", "-1"))
        self.slow_seconds = float(os.environ.get("LAL_FAKE_DETECTOR_SLOW_SECONDS", "0"))
        ALIVE.add(self)
        time.sleep(float(os.environ.get("LAL_FAKE_DETECTOR_LOAD_SECONDS", "0")))

    def predict(self, frame, image_size=None):
        if self.delay:
            time.sleep(self.delay)
        if frame.frame_id == self.slow_frame:
            end = time.perf_counter() + self.slow_seconds
            while time.perf_counter() < end:                          # busy, holds the GIL
                sum(range(1000))
        self.calls += 1
        return [BoundingBox3D(10.0 + 0.1 * frame.frame_id, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9)]
