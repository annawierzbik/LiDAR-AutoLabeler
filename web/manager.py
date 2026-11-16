"""Everything the start screen and the API need, without HTTP: opening sequences and serving sessions.

Opening a folder submits a *job* to the JobQueue that prepares the session in the SessionStore
(resume it, or make a new one from cached detections or by running the detector, which can take
minutes) and takes the session's lock for the job id (`sid`, used by the browser tab).

Between requests the manager keeps only a small handle per sid (session key, sequence, point
source); no ReviewService and no session document stays in memory (docs/adr/0011). Every request
refreshes the lock (a heartbeat), loads the document and builds a ReviewService for that request.

Locks expire (LAL_LOCK_TTL_SECONDS) unless the browser sends heartbeats, so a tab that was closed
without "Finish" frees its session after a while. Opening a session that another live tab holds is
refused with a message naming that tab's sid; a tab whose lock was taken over gets 409.

Detection + tracking (when a new session needs it) and exports run as tasks of the TaskQueue
(LAL_JOB_QUEUE: threads of this process, or Celery workers; docs/adr/0015). The opening job waits
for the detection task, shows its progress, and can be cancelled.
"""
from __future__ import annotations

import argparse
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from infra.jobs import FINAL, JobQueue, TaskQueue, ThreadJobQueue
from infra.points import PointCache, PointSource, point_source
from infra.sessions import SessionLocked, SessionNotFound, SessionStore
from infra.settings import get_settings, make_session_store, make_task_queue
from infra.tasks import JobCancelled, detect_track_payload
from web.service import ReviewService, point_bytes

STATES = {"queued": "loading", "running": "loading", "done": "ready", "failed": "error", "cancelled": "error"}


class SessionAlreadyOpen(Exception):
    pass


@dataclass
class SessionHandle:
    """What a request needs besides the session document."""
    key: str
    sequence: object
    points: PointSource
    num_frames: int          # fixed for the life of a session
    frame_offset: int
    args: dict               # session-builder arguments (to reopen the sequence in a worker)


class Job:
    """What the start screen shows about one sid; status and result come from the JobQueue."""

    def __init__(self, sid: str, title: str, queue: JobQueue, tasks: TaskQueue) -> None:
        self.sid, self.title, self._queue, self._tasks = sid, title, queue, tasks
        self.task_id: str | None = None        # the detection task this opening waits for, if any
        self.cancel_requested = False

    @property
    def status(self) -> str:
        return STATES[self._queue.status(self.sid).state]

    @property
    def message(self) -> str:
        st = self._queue.status(self.sid)
        return "cancelled" if st.state == "cancelled" else st.message

    @property
    def progress(self) -> dict | None:
        """Progress of the detection task while the session is loading."""
        if self.task_id is None or self.status != "loading":
            return None
        t = self._tasks.status(self.task_id)
        return {"state": t.state, "fraction": t.progress, "message": t.message, "attempts": t.attempts}

    @property
    def handle(self) -> SessionHandle | None:
        st = self._queue.status(self.sid)
        return st.result if st.state == "done" else None

    def view(self) -> dict:
        return {"sid": self.sid, "status": self.status, "message": self.message, "title": self.title,
                "progress": self.progress}


def default_builder(args, store: SessionStore, detect=None):
    from pipeline.session_builder import prepare
    return prepare(args, store, detect)


def make_args(**overrides) -> argparse.Namespace:
    """Arguments of pipeline.session_builder with their defaults, plus `overrides`."""
    from pipeline.session_builder import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class SessionManager:
    def __init__(self, builder: Callable | None = None, background: bool = True,
                 queue: JobQueue | None = None, store: SessionStore | None = None,
                 point_cache: PointCache | None = None, tasks: TaskQueue | None = None,
                 poll_seconds: float = 0.25) -> None:
        """`builder(args, store, detect)` -> (sequence, session key), making sure the session exists
        in the store (pipeline.session_builder.prepare; `detect(args)` runs detection as a task).
        `background=False` runs the opening jobs inline (tests)."""
        self.builder = builder or default_builder
        self.queue = queue or ThreadJobQueue(background=background)
        self.store = store or make_session_store()
        self.tasks = tasks or make_task_queue()
        self.poll_seconds = poll_seconds
        self.point_cache = point_cache or PointCache(get_settings().point_cache_frames * 8)
        self.jobs: dict[str, Job] = {}
        self.finished: set[str] = set()   # sids of finished sessions (a few bytes each, kept for 409s)

    # ------------------------------------------------------------------ jobs
    def start_job(self, args, title: str) -> str:
        sid = uuid.uuid4().hex[:8]
        job = self.jobs[sid] = Job(sid, title, self.queue, self.tasks)
        self.queue.submit(lambda: self._build(job, args), title, job_id=sid)
        self._reap(sid)
        return sid

    def _build(self, job: Job, args) -> SessionHandle:
        seq, key = self.builder(args, self.store, lambda a: self._detect(job, a))
        try:
            self.store.lock(key, owner=job.sid)
        except SessionLocked as exc:
            raise SessionAlreadyOpen(f"This session is already open in another tab (session id {exc.owner}). "
                                     "Continue there, or finish it first.") from None
        s = self.store.load(key)
        return SessionHandle(key, seq, point_source(seq), s.num_frames, s.frame_offset, vars(args).copy())

    def _detect(self, job: Job, args) -> None:
        """Run detection + tracking as a task and wait for it (the opening job's thread waits; the
        work happens in the TaskQueue: a worker process with LAL_JOB_QUEUE=celery)."""
        if job.cancel_requested:
            raise JobCancelled("cancelled")
        job.task_id = self.tasks.submit("detect_track", detect_track_payload(args), title=job.title)
        while True:
            st = self.tasks.status(job.task_id)
            if st.state in FINAL:
                break
            time.sleep(self.poll_seconds)
        if st.state == "cancelled":
            raise JobCancelled("cancelled")
        if st.state == "failed":
            raise RuntimeError(f"detection failed: {st.message}")

    def cancel(self, sid: str) -> bool:
        """Stop opening a session (its detection task stops at the next frame)."""
        job = self.job(sid)
        if job.status != "loading":
            return False
        job.cancel_requested = True
        if job.task_id is not None:
            self.tasks.cancel(job.task_id)
        return True

    def _reap(self, sid: str) -> None:
        """Forget a job that failed for another reason than a refused second opening (its
        traceback was logged by the queue); the start screen then gets 404, as before."""
        job = self.jobs.get(sid)
        if job is None:
            return
        st = self.queue.status(sid)
        if st.state == "failed" and not isinstance(st.error, SessionAlreadyOpen | JobCancelled):
            del self.jobs[sid]

    def job(self, sid: str) -> Job:
        self._reap(sid)
        return self.jobs[sid]

    def handle(self, sid: str) -> SessionHandle:
        """KeyError for unknown ids, LookupError while loading, when refused or when finished
        (a tab still open on a finished session gets 409, not 404)."""
        if sid in self.finished:
            raise LookupError(f"session {sid} was finished and is closed")
        job = self.job(sid)
        handle = job.handle
        if handle is None:
            raise LookupError(job.message or f"session {sid} is not ready ({job.status})")
        return handle

    def heartbeat(self, sid: str) -> SessionHandle:
        """Refresh this tab's lock; LookupError when another tab has taken the session over."""
        handle = self.handle(sid)
        try:
            self.store.lock(handle.key, owner=sid)
        except SessionLocked as exc:
            raise LookupError(f"this session is now open in another tab (session id {exc.owner}); "
                              "reload to continue there") from None
        return handle

    def service(self, sid: str) -> ReviewService:
        """A ReviewService for one request: lock refreshed, document loaded from the store."""
        handle = self.heartbeat(sid)
        try:
            session = self.store.load(handle.key)
        except SessionNotFound:
            raise LookupError(f"session {handle.key} no longer exists") from None
        return ReviewService(session, self.store, handle.key, handle.points, sequence=handle.sequence,
                             point_cache=self.point_cache)

    def points(self, sid: str, frame: int) -> bytes:
        """Point cloud of a session frame without loading the session document."""
        handle = self.handle(sid)
        if not 0 <= frame < handle.num_frames:
            raise IndexError(frame)
        return point_bytes(handle.points, self.point_cache, handle.key, handle.frame_offset + frame)

    def finish(self, sid: str) -> None:
        handle = self.handle(sid)
        self.store.finish(handle.key, sid)
        del self.jobs[sid]
        self.finished.add(sid)

    # ---------------------------------------------------------------- folder
    def open_folder(self, path: str) -> str:
        from ingestion.folder import validate_folder
        folder = Path(path.strip().strip('"'))
        frames = validate_folder(folder)               # readable errors for folders that cannot be used
        return self.start_job(make_args(folder=str(folder)), f"{folder.name} ({frames} frames)")

    # ---------------------------------------------------------------- export
    def export_job(self, sid: str, fmt: str, include_unreviewed: bool = False, include_empty: bool = False) -> str:
        """Labels of the session as a zip, made by an export task (cpu queue); returns the task id."""
        handle = self.heartbeat(sid)
        revision = self.store.load(handle.key).revision
        payload = {"session_key": handle.key, "revision": revision, "sequence": handle.args, "format": fmt,
                   "include_unreviewed": include_unreviewed, "include_empty": include_empty}
        return self.tasks.submit("export", payload, title=f"export {handle.key} ({fmt})")

    def task(self, task_id: str):
        return self.tasks.status(task_id)

    def task_file(self, task_id: str) -> tuple[str, bytes]:
        """(file name, bytes) of a finished export task; LookupError while it is not done."""
        from infra.tasks import export_blobs

        st = self.tasks.status(task_id)
        if st.kind != "export" or st.state != "done":
            raise LookupError(f"task {task_id} has no file ({st.kind}, {st.state})")
        return st.result["archive_name"], export_blobs().get(st.result["blob"])
