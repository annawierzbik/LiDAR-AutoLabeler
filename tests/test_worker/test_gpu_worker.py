"""The real PointPillars model through a real Celery worker (markers `gpu`, `worker`, `postgres`).

Run only when no experiment is using the GPU:
    pytest -m "gpu and worker" tests/test_worker/test_gpu_worker.py
Needs CUDA, OpenPCDet, checkpoints/pointpillar_7728.pth, data/raw/training (KITTI tracking 0000) and
the services of docker-compose.db.yml.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from tests.test_worker.test_celery_worker import (
    PROJECT_ROOT,
    REDIS_URL,
    Worker,
    wait_for,
)

pytestmark = [pytest.mark.gpu, pytest.mark.worker, pytest.mark.postgres]

DATA = PROJECT_ROOT / "data" / "raw"


def test_real_detector_through_a_gpu_worker(tmp_path, pg_url, pg_engine):
    import redis
    from celery import Celery

    import detection.paths
    from infra.celery_queue import CeleryTaskQueue
    from infra.pg_jobs import PgJobStore
    from infra.tasks import detect_track_payload
    from pipeline.session_builder import make_args

    if not (DATA / "training" / "velodyne" / "0000").is_dir() or not detection.paths.DEFAULT_CKPT.exists():
        pytest.skip("KITTI tracking data or the checkpoint is missing")
    redis.Redis.from_url(REDIS_URL).flushdb()
    env = {**os.environ, "LAL_JOB_QUEUE": "celery", "LAL_REDIS_URL": REDIS_URL, "LAL_DATABASE_URL": pg_url,
           "LAL_RESULTS_DIR": str(tmp_path / "results"), "LAL_JOB_HEARTBEAT_SECONDS": "1",
           "LAL_JOB_STALE_SECONDS": "120", "PYTHONUNBUFFERED": "1"}
    worker = Worker("gpu1", env, tmp_path / "gpu1.log").start()
    try:
        sender = Celery("lal-test", broker=REDIS_URL)
        queue = CeleryTaskQueue(PgJobStore(pg_engine, stale_seconds=120),
                                send=lambda job_id, q: sender.send_task("lal.run_job", args=[job_id], queue=q))
        args = make_args(data_root=str(DATA), split="training", sequence="0000", stop=20)
        t0 = time.time()
        job = queue.submit("detect_track", detect_track_payload(args), "0000")
        st = wait_for(queue, job, lambda s: s.state in ("done", "failed"), timeout=600, logs=tmp_path)
        assert st.state == "done", st.message
        data = json.loads((tmp_path / "results" / st.result["cache_key"]).read_text())
        assert len(data["frames"]) == 20 and sum(len(f) for f in data["frames"]) > 0
        assert data["meta"]["checkpoint_hash"] and data["meta"]["fov"] == "camera"
        print(f"20 frames in {time.time() - t0:.1f} s (model load included)")
    finally:
        worker.stop()
