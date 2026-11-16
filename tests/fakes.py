"""Test doubles used across processes (e.g. by a Celery worker started from a test):

    LAL_WORKER_DETECTOR_FACTORY=tests.fakes:FakeDetector

FakeDetector takes the arguments of detection.detector.Detector and returns one Car per frame,
moving with the frame id. LAL_FAKE_DETECTOR_DELAY (seconds per frame) makes it slow enough to
cancel or kill a job in the middle.
"""
from __future__ import annotations

import os
import time

from common.boxes import BoundingBox3D

LOADS = 0      # FakeDetector objects made in this process


class FakeDetector:
    def __init__(self, cfg_file=None, checkpoint=None, score_threshold=0.5, fov="auto", **_) -> None:
        global LOADS
        LOADS += 1
        self.score_threshold, self.fov = score_threshold, fov
        self.delay = float(os.environ.get("LAL_FAKE_DETECTOR_DELAY", "0"))
        self.calls = 0

    def predict(self, frame, image_size=None):
        if self.delay:
            time.sleep(self.delay)
        self.calls += 1
        return [BoundingBox3D(10.0 + 0.1 * frame.frame_id, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9)]
