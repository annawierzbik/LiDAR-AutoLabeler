"""The real web application for the end-to-end tests, with a fake detector.

    LAL_RESULTS_DIR=<tmp> python -m tests.e2e.app_server <port>

Everything is the production code path (pipeline.session_builder.build_session, session files, cache files,
SessionManager, FastAPI app) except detection + tracking: `pipeline.detect_track.track_sequence`
is replaced by FAKE_TRACKS, so no GPU, model or KITTI data is needed.

Fake tracks per frame i: track 0, a Car at x = 10 + i in every frame; track 1, a Pedestrian at
x = 20 in frames 0..2. LAL_E2E_TRACK_DELAY slows it down (seconds per frame).
"""
from __future__ import annotations

import os
import sys
import time

import uvicorn

import pipeline.detect_track
from common.boxes import BoundingBox3D

DELAY = float(os.environ.get("LAL_E2E_TRACK_DELAY", "0"))   # seconds per frame (to watch progress)


def fake_track(seq, indices, threshold, fov="auto", detector=None, should_stop=None, progress=None):
    from pipeline.detect_track import Cancelled

    out = []
    for n, i in enumerate(indices, 1):
        if should_stop is not None and should_stop():
            raise Cancelled(f"stopped after {n - 1} of {len(indices)} frames")
        time.sleep(DELAY)
        if progress is not None:
            progress(n / len(indices))
        boxes = [BoundingBox3D(10.0 + i, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, 0)]
        if i < 3:
            boxes.append(BoundingBox3D(20.0, 3.0, -0.9, 0.8, 0.6, 1.7, 0.0, "Pedestrian", 0.8, 1))
        out.append(boxes)
    return out


def main() -> None:
    pipeline.detect_track.track_sequence = fake_track

    from web.manager import SessionManager
    from web.server import create_app

    uvicorn.run(create_app(SessionManager()), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")


if __name__ == "__main__":
    main()
