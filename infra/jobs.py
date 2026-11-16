"""Background jobs: submit a function, poll its status, cancel it.

ThreadJobQueue runs every job in its own daemon thread (as the web server did before), or inline
with background=False (tests). Cancelling only works before a job has started: a running Python
thread cannot be stopped, so `cancel` then returns False.
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)

State = Literal["queued", "running", "done", "failed", "cancelled"]


@dataclass
class JobStatus:
    id: str
    title: str
    state: State = "queued"
    message: str = ""
    result: Any = None
    error: BaseException | None = None


@runtime_checkable
class JobQueue(Protocol):
    def submit(self, fn: Callable[[], Any], title: str = "", job_id: str | None = None) -> str:
        """Start `fn` in the background; returns the job id (`job_id` if given)."""

    def status(self, job_id: str) -> JobStatus:
        """KeyError for unknown ids."""

    def cancel(self, job_id: str) -> bool:
        """True if the job will not run (or already did not); False if it is running or finished."""


class ThreadJobQueue:
    def __init__(self, background: bool = True) -> None:
        self.background = background
        self._jobs: dict[str, JobStatus] = {}
        self._mutex = threading.Lock()

    def submit(self, fn: Callable[[], Any], title: str = "", job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex[:8]
        with self._mutex:
            if job_id in self._jobs:
                raise ValueError(f"job id {job_id} already used")
            self._jobs[job_id] = JobStatus(job_id, title)
        self._start(job_id, fn)
        return job_id

    def _start(self, job_id: str, fn: Callable[[], Any]) -> None:
        if self.background:
            threading.Thread(target=self._run, args=(job_id, fn), daemon=True).start()
        else:
            self._run(job_id, fn)

    def _run(self, job_id: str, fn: Callable[[], Any]) -> None:
        job = self._jobs[job_id]
        with self._mutex:
            if job.state == "cancelled":
                return
            job.state = "running"
        try:
            job.result = fn()
            job.state = "done"
        except Exception as exc:
            job.error, job.message, job.state = exc, str(exc), "failed"
            log.exception("Job %s (%s) failed", job_id, job.title)

    def status(self, job_id: str) -> JobStatus:
        return self._jobs[job_id]

    def cancel(self, job_id: str) -> bool:
        with self._mutex:
            job = self._jobs[job_id]
            if job.state == "queued":
                job.state = "cancelled"
            return job.state == "cancelled"
