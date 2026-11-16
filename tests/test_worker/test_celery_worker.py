"""Real Celery worker processes + Redis + PostgreSQL, fake detector (markers `worker`, `postgres`).

    docker compose -f docker-compose.db.yml up -d
    pytest -m worker

Each test starts worker processes the way they are started on Windows (solo pool) and drives them
through CeleryTaskQueue, as the API does.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.worker, pytest.mark.postgres]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REDIS_URL = os.environ.get("LAL_TEST_REDIS_URL", "redis://127.0.0.1:56379/7")


class Worker:
    def __init__(self, name: str, env: dict, log: Path) -> None:
        self.name, self.env, self.log = name, env, log
        self.proc: subprocess.Popen | None = None
        self._out = None

    def start(self) -> Worker:
        self._out = self.log.open("ab")                   # closed in kill() / stop()
        out = self._out
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "celery", "-A", "infra.celery_app", "worker", "-Q", "gpu,cpu", "-P", "solo",
             "-n", f"{self.name}@test", "--without-gossip", "--without-mingle", "--without-heartbeat", "-l", "info"],
            cwd=PROJECT_ROOT, env=self.env, stdout=out, stderr=subprocess.STDOUT)
        return self

    def kill(self) -> None:
        """Like a crash: no shutdown, no acknowledgement."""
        self.proc.kill()
        self.proc.wait(10)
        self._out.close()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._out is not None and not self._out.closed:
            self._out.close()


@pytest.fixture
def env(tmp_path, pg_url, pg_engine):
    import redis

    redis.Redis.from_url(REDIS_URL).flushdb()
    folder = tmp_path / "drive"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(60):
        rng.uniform(-20, 20, (300, 4)).astype("<f4").tofile(folder / f"{i:06d}.bin")
    worker_env = {**os.environ, "LAL_JOB_QUEUE": "celery", "LAL_REDIS_URL": REDIS_URL, "LAL_DATABASE_URL": pg_url,
                  "LAL_RESULTS_DIR": str(tmp_path / "results"), "LAL_WORKER_DETECTOR_FACTORY": "tests.fakes:FakeDetector",
                  "LAL_FAKE_DETECTOR_DELAY": "0.08", "LAL_JOB_HEARTBEAT_SECONDS": "0.3",
                  "LAL_JOB_STALE_SECONDS": "2", "LAL_CELERY_VISIBILITY_TIMEOUT": "4", "PYTHONUNBUFFERED": "1"}
    workers: list[Worker] = []

    def start_worker(name: str) -> Worker:
        w = Worker(name, worker_env, tmp_path / f"{name}.log").start()
        workers.append(w)
        return w

    from celery import Celery

    from infra.celery_queue import CeleryTaskQueue
    from infra.pg_jobs import PgJobStore

    sender = Celery("lal-test", broker=REDIS_URL)
    queue = CeleryTaskQueue(PgJobStore(pg_engine, stale_seconds=2, max_attempts=3),
                            send=lambda job_id, q: sender.send_task("lal.run_job", args=[job_id], queue=q))
    yield type("Env", (), {"queue": queue, "folder": folder, "results": tmp_path / "results",
                           "start_worker": staticmethod(start_worker), "logs": tmp_path,
                           "worker_env": worker_env})
    for w in workers:
        w.stop()


def payload(folder, **kw):
    from infra.tasks import detect_track_payload
    from web.manager import make_args

    return detect_track_payload(make_args(folder=str(folder), **kw))


def wait_for(queue, job_id, check, timeout=60.0, logs=None):
    deadline = time.time() + timeout
    while True:
        st = queue.status(job_id)
        if check(st):
            return st
        if time.time() > deadline:
            tails = "".join(p.read_text(errors="replace")[-3000:] for p in sorted(logs.glob("*.log"))) if logs else ""
            raise AssertionError(f"timed out waiting; last status {st}\n{tails}")
        time.sleep(0.1)


def cache_files(results):
    return sorted(p.name for p in results.glob("*.json")) if results.exists() else []


def test_submit_progress_and_success(env):
    env.start_worker("w1")
    job = env.queue.submit("detect_track", payload(env.folder, stop=30), "drive")
    st = wait_for(env.queue, job, lambda s: s.state == "running" and 0 < s.progress < 1, logs=env.logs)
    assert st.attempts == 1
    st = wait_for(env.queue, job, lambda s: s.state in ("done", "failed"), logs=env.logs)
    assert st.state == "done", st.message
    assert st.result["frames"] == 30 and cache_files(env.results) == [st.result["cache_key"]]
    again = env.queue.submit("detect_track", payload(env.folder, stop=30), "drive")
    assert again == job                                              # duplicate submit: the same job


def test_cancel_while_running(env):
    env.start_worker("w1")
    job = env.queue.submit("detect_track", payload(env.folder), "drive")
    wait_for(env.queue, job, lambda s: s.state == "running" and s.progress > 0.1, logs=env.logs)
    assert env.queue.cancel(job) is True
    st = wait_for(env.queue, job, lambda s: s.state in ("cancelled", "done", "failed"), logs=env.logs)
    assert st.state == "cancelled" and "stopped after" in st.message, st
    assert st.progress < 1.0 and cache_files(env.results) == []      # the partial result was dropped


def test_a_killed_worker_does_not_leave_the_job_running(env):
    first = env.start_worker("w1")
    job = env.queue.submit("detect_track", payload(env.folder, stop=40), "drive")
    wait_for(env.queue, job, lambda s: s.state == "running" and s.progress > 0.2, logs=env.logs)
    first.kill()                                                     # mid-job, nothing acknowledged
    time.sleep(2.5)                                                  # longer than LAL_JOB_STALE_SECONDS
    st = env.queue.status(job)                                       # notices the lost worker
    assert st.state in ("queued", "running"), st
    env.start_worker("w2")
    st = wait_for(env.queue, job, lambda s: s.state in ("done", "failed"), timeout=90, logs=env.logs)
    assert st.state == "done", st.message
    assert st.attempts == 2                                          # run again, once
    assert cache_files(env.results) == [st.result["cache_key"]]      # one complete cache entry


def test_a_job_for_another_detector_configuration_fails_clearly(env):
    env.start_worker("w1")                                           # loads score threshold 0.5
    job = env.queue.submit("detect_track", payload(env.folder, stop=5, threshold=0.7), "drive")
    st = wait_for(env.queue, job, lambda s: s.state in ("done", "failed"), logs=env.logs)
    assert st.state == "failed" and "another detector loaded" in st.message and "0.7" in st.message
    assert cache_files(env.results) == []


def test_heartbeat_continues_while_the_model_loads_and_during_a_slow_frame(env):
    """A healthy job must never be taken for a lost one: the model load (4 s) and one frame (4 s)
    each take twice LAL_JOB_STALE_SECONDS (2 s), yet the heartbeat keeps going."""
    from sqlalchemy import text

    env.worker_env.update({"LAL_FAKE_DETECTOR_LOAD_SECONDS": "4", "LAL_FAKE_DETECTOR_SLOW_FRAME": "5",
                           "LAL_FAKE_DETECTOR_SLOW_SECONDS": "4", "LAL_FAKE_DETECTOR_DELAY": "0.02"})
    env.start_worker("w1")
    job = env.queue.submit("detect_track", payload(env.folder, stop=12), "drive")
    engine = env.queue.store.engine
    seen_states, beats = [], []
    deadline = time.time() + 90
    while True:
        st = env.queue.status(job)                                   # also runs the stale check
        seen_states.append((st.state, st.attempts, round(st.progress, 2), st.message))
        with engine.connect() as c:
            hb = c.execute(text("SELECT heartbeat_at FROM jobs WHERE id = :i"), {"i": job}).scalar()
        if hb is not None:
            beats.append((time.time(), hb.timestamp(), st.message, st.progress))
        if st.state in ("done", "failed", "cancelled") or time.time() > deadline:
            break
        time.sleep(0.2)
    assert st.state == "done", (st, seen_states[-5:])
    assert all(s[0] in ("queued", "running", "done") for s in seen_states)
    running = [i for i, s in enumerate(seen_states) if s[0] == "running"]
    assert running and all(s[0] != "queued" for s in seen_states[running[0]:])   # never requeued
    assert st.attempts == 1
    loading = [b for b in beats if b[2] == "loading the detector"]
    assert loading and max(b[1] for b in loading) - min(b[1] for b in loading) > 2.5   # beats during the load
    stalled = [b for b in beats if b[2] == "detecting" and 0.4 < b[3] < 0.6]          # around frame 5
    assert stalled and max(b[1] for b in stalled) - min(b[1] for b in stalled) > 2.5   # beats during the slow frame
