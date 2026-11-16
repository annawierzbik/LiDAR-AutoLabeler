"""The real web application for the end-to-end tests, with a fake detector.

    LAL_RESULTS_DIR=<tmp> python -m tests.e2e.app_server <port>

Everything is the production code path (pipeline.session_builder.build_session, session files, cache files,
SessionManager, FastAPI app) except detection + tracking: `pipeline.detect_track.track_sequence`
is replaced by FAKE_TRACKS, so no GPU, model or KITTI data is needed.

Fake tracks per frame i: track 0, a Car at x = 10 + i in every frame; track 1, a Pedestrian at
x = 20 in frames 0..2.
"""
from __future__ import annotations

import sys

import uvicorn

import pipeline.detect_track
from common.boxes import BoundingBox3D


def fake_track(seq, indices, threshold, fov="auto"):
    out = []
    for i in indices:
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
