"""Background jobs.

TaskQueue: submit a named task (infra.tasks.TASKS) with a JSON payload; status with progress;
cooperative cancel. ThreadTaskQueue (the default, LAL_JOB_QUEUE=thread; used by the tests, no script uses a
queue) runs tasks in threads of this process; CeleryTaskQueue (infra/celery_queue.py) sends them to worker processes
(always used by the web API, ADR 0003).
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)

State = Literal["queued", "running", "cancelling", "done", "failed", "cancelled"]
FINAL = ("done", "failed", "cancelled")


@dataclass
class JobStatus:
    id: str
    title: str
    state: State = "queued"
    message: str = ""
    result: Any = None
    error: BaseException | None = None
    kind: str = ""
    progress: float = 0.0
    attempts: int = 0


# ------------------------------------------------------------------- task queues
@runtime_checkable
class TaskQueue(Protocol):
    def submit(self, kind: str, payload: dict, title: str = "",
               reuse_done: Callable[[dict | None], bool] | None = None) -> str:
        """Queue task `kind` (infra.tasks.TASKS) with a JSON payload; returns the job id. A payload
        identical to a queued, running or finished job's returns that job (no second run); a
        finished job only if `reuse_done(result)` is true (default: always)."""

    def status(self, job_id: str) -> JobStatus:
        """KeyError for unknown ids."""

    def cancel(self, job_id: str) -> bool:
        """Ask the job to stop at its next check; True unless it has already finished."""


class _ThreadContext:
    keeps_detector = False      # thread mode: a detector per job, released afterwards

    def __init__(self, job: JobStatus, stop: threading.Event, queue: ThreadTaskQueue) -> None:
        self.job, self.stop, self.queue = job, stop, queue

    def submit(self, kind: str, payload: dict, title: str = "", reuse_done=None) -> str:
        return self.queue.submit(kind, payload, title, reuse_done)

    def progress(self, fraction: float, message: str = "") -> None:
        self.job.progress = max(0.0, min(1.0, float(fraction)))
        if message:
            self.job.message = message

    def should_stop(self) -> bool:
        return self.stop.is_set()


class ThreadTaskQueue:
    """Tasks in daemon threads of this process (or inline with background=False). Progress and
    cancellation in memory; nothing survives a restart (the detection cache does)."""

    def __init__(self, background: bool = True, tasks: dict | None = None) -> None:
        from infra.tasks import TASKS

        self.background = background
        self.tasks = tasks if tasks is not None else TASKS
        self._jobs: dict[str, JobStatus] = {}
        self._stops: dict[str, threading.Event] = {}
        self._dedupe: dict[str, str] = {}
        self._mutex = threading.Lock()

    def submit(self, kind: str, payload: dict, title: str = "",
               reuse_done: Callable[[dict | None], bool] | None = None) -> str:
        from infra.tasks import dedupe_key

        if kind not in self.tasks:
            raise KeyError(f"unknown task {kind!r}")
        dkey = dedupe_key(kind, payload)
        with self._mutex:
            old = self._dedupe.get(dkey)
            if old is not None and self._jobs[old].state not in ("failed", "cancelled"):
                done = self._jobs[old].state == "done"
                if not (done and reuse_done is not None and not reuse_done(self._jobs[old].result)):
                    return old
            job_id = uuid.uuid4().hex[:12]
            job = self._jobs[job_id] = JobStatus(job_id, title, kind=kind)
            self._stops[job_id] = threading.Event()
            self._dedupe[dkey] = job_id
        payload = {**payload, "job_id": job_id}
        if self.background:
            threading.Thread(target=self._run, args=(job, payload), daemon=True).start()
        else:
            self._run(job, payload)
        return job_id

    def _run(self, job: JobStatus, payload: dict) -> None:
        from infra.tasks import JobCancelled, TaskFailed

        stop = self._stops[job.id]
        with self._mutex:
            if stop.is_set():
                job.state = "cancelled"
                return
            job.state, job.attempts = "running", job.attempts + 1
        try:
            job.result = self.tasks[job.kind][1](payload, _ThreadContext(job, stop, self))
            job.state, job.progress = "done", 1.0
        except JobCancelled as exc:
            job.state, job.message = "cancelled", str(exc) or "cancelled"
        except TaskFailed as exc:
            job.error, job.message, job.state = exc, str(exc), "failed"
        except Exception as exc:
            job.error, job.message, job.state = exc, f"{type(exc).__name__}: {exc}", "failed"
            log.exception("Job %s (%s) failed", job.id, job.kind)

    def status(self, job_id: str) -> JobStatus:
        return self._jobs[job_id]

    def cancel(self, job_id: str) -> bool:
        with self._mutex:
            job = self._jobs[job_id]
            if job.state in FINAL:
                return False
            self._stops[job_id].set()
            if job.state == "queued":
                job.state = "cancelled"
            elif job.state == "running":
                job.state = "cancelling"
            return True
