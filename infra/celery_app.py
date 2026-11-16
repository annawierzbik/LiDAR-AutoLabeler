"""Celery application and worker side of the job queue (LAL_JOB_QUEUE=celery, docs/adr/0015).

Workers (Windows: the "solo" pool, one job at a time per process):
    celery -A infra.celery_app worker -Q gpu -P solo -n gpu@%h        # detection + tracking, one model
    celery -A infra.celery_app worker -Q cpu -P solo -n cpu@%h        # export (start several for more)
Environment: LAL_REDIS_URL (broker), LAL_DATABASE_URL (jobs table, sessions), LAL_RESULTS_DIR, and
for gpu workers the detector configuration LAL_WORKER_DETECTOR_* (infra/worker_detector.py).

The broker message carries only the job id; payload, state, progress and errors are rows of the
`jobs` table. Messages are acknowledged late (after the task) and a worker takes one at a time, so a
worker that dies mid-job leaves the message to be delivered again (Redis visibility timeout); a
started job's heartbeat lets anyone see that it stopped (PgJobStore.recover_stale), and a worker
that starts requeues such jobs. Running a job twice is harmless (PgJobStore.start lets one run win;
the tasks are idempotent on the detection cache).
"""
from __future__ import annotations

import logging
import socket
import threading

from celery import Celery
from celery.signals import worker_ready

from infra.settings import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

app = Celery("lal", broker=settings.redis_url or "redis://127.0.0.1:56379/0")
app.conf.update(
    task_acks_late=True,                     # acknowledge after the task: a dead worker's job is delivered again
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,            # take one job at a time
    broker_transport_options={"visibility_timeout": settings.celery_visibility_timeout},
    broker_connection_retry_on_startup=True,
    task_default_queue="cpu",
    task_ignore_result=True,                 # results live in the jobs table
    worker_hijack_root_logger=False,
)

_store = None


def job_store():
    global _store
    if _store is None:
        from infra.db import make_engine
        from infra.pg_jobs import PgJobStore

        if not settings.database_url:
            raise RuntimeError("LAL_JOB_QUEUE=celery needs LAL_DATABASE_URL (the jobs table)")
        _store = PgJobStore(make_engine(settings.database_url), stale_seconds=settings.job_stale_seconds,
                            max_attempts=settings.job_max_attempts)
    return _store


def dispatch(job_id: str, queue: str) -> None:
    app.send_task("lal.run_job", args=[job_id], queue=queue)


class _WorkerContext:
    """Progress and cancellation for one running job. A daemon thread writes the heartbeat every
    LAL_JOB_HEARTBEAT_SECONDS (also while the model loads); progress is written with it, so the
    table gets one write per interval, not one per frame. should_stop() reads the flag at the same
    rate."""

    def __init__(self, store, job_id: str) -> None:
        self.store, self.job_id = store, job_id
        self.interval = settings.job_heartbeat_seconds
        self._progress, self._message = 0.0, None
        self._stop_seen = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()

    def _beat(self) -> None:
        while not self._done.wait(self.interval):
            try:
                self.store.heartbeat(self.job_id, self._progress, self._message)
                self._stop_seen = self._stop_seen or self.store.cancel_requested(self.job_id)
            except Exception:
                log.exception("heartbeat of job %s failed", self.job_id)

    def progress(self, fraction: float, message: str = "") -> None:
        self._progress = max(0.0, min(1.0, float(fraction)))
        if message:
            self._message = message

    def should_stop(self) -> bool:
        return self._stop_seen

    def close(self) -> None:
        self._done.set()
        self._thread.join(self.interval + 5)


def run_job(job_id: str, worker: str | None = None) -> str:
    """Run one job in this process; returns its final state (or "skipped" if it was not queued)."""
    from infra.tasks import TASKS, JobCancelled
    from infra.worker_detector import DetectorMismatch

    store = job_store()
    payload = store.start(job_id, worker or f"{socket.gethostname()}")
    if payload is None:
        return "skipped"
    kind = store.status(job_id).kind
    ctx = _WorkerContext(store, job_id)
    try:
        result = TASKS[kind][1](payload, ctx)
        ctx.close()
        store.finish(job_id, result)
        return "done"
    except JobCancelled as exc:
        ctx.close()
        store.cancelled(job_id, str(exc) or "cancelled")
        return "cancelled"
    except DetectorMismatch as exc:
        ctx.close()
        store.fail(job_id, str(exc))
        return "failed"
    except Exception as exc:
        ctx.close()
        log.exception("job %s failed", job_id)
        store.fail(job_id, f"{type(exc).__name__}: {exc}")
        return "failed"


@app.task(name="lal.run_job", bind=True)
def run_job_task(self, job_id: str) -> str:
    return run_job(job_id, self.request.hostname)


@worker_ready.connect
def _recover_on_start(sender=None, **_) -> None:
    """A starting worker requeues jobs whose worker died and sends queued jobs again (a lost broker
    message must not leave a job queued forever; a duplicate message is harmless)."""
    try:
        store = job_store()
        store.recover_stale()
        for job_id, queue in store.queued():
            dispatch(job_id, queue)
    except Exception:
        log.exception("recovering stale jobs failed")
