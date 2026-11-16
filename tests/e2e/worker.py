"""A real Celery worker (solo pool, queues gpu and cpu) for the end-to-end tests, with detection +
tracking replaced by tests/e2e/fake_tracks.py and the fake detector (LAL_WORKER_DETECTOR_FACTORY).

    python -m tests.e2e.worker <name>
"""
from __future__ import annotations

import sys

import pipeline.detect_track
from tests.e2e.fake_tracks import fake_track


def main() -> None:
    pipeline.detect_track.track_sequence = fake_track

    from infra.celery_app import app

    app.worker_main(["worker", "-Q", "gpu,cpu", "-P", "solo", "-n", f"{sys.argv[1]}@e2e", "--without-gossip",
                     "--without-mingle", "--without-heartbeat", "-l", "warning"])


if __name__ == "__main__":
    main()
