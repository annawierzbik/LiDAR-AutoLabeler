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
"""
from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from infra.jobs import JobQueue, ThreadJobQueue
from infra.points import PointCache, PointSource, point_source
from infra.sessions import SessionLocked, SessionNotFound, SessionStore
from infra.settings import get_settings, make_session_store
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


class Job:
    """What the start screen shows about one sid; status and result come from the JobQueue."""

    def __init__(self, sid: str, title: str, queue: JobQueue) -> None:
        self.sid, self.title, self._queue = sid, title, queue

    @property
    def status(self) -> str:
        return STATES[self._queue.status(self.sid).state]

    @property
    def message(self) -> str:
        st = self._queue.status(self.sid)
        return "cancelled" if st.state == "cancelled" else st.message

    @property
    def handle(self) -> SessionHandle | None:
        st = self._queue.status(self.sid)
        return st.result if st.state == "done" else None

    def view(self) -> dict:
        return {"sid": self.sid, "status": self.status, "message": self.message, "title": self.title}


def default_builder(args, store: SessionStore):
    from pipeline.session_builder import prepare
    return prepare(args, store)


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
                 point_cache: PointCache | None = None) -> None:
        """`builder(args, store)` -> (sequence, session key), making sure the session exists in the
        store (pipeline.session_builder.prepare). `background=False` runs jobs inline (tests)."""
        self.builder = builder or default_builder
        self.queue = queue or ThreadJobQueue(background=background)
        self.store = store or make_session_store()
        self.point_cache = point_cache or PointCache(get_settings().point_cache_frames * 8)
        self.jobs: dict[str, Job] = {}
        self.finished: set[str] = set()   # sids of finished sessions (a few bytes each, kept for 409s)

    # ------------------------------------------------------------------ jobs
    def start_job(self, args, title: str) -> str:
        sid = uuid.uuid4().hex[:8]
        job = self.jobs[sid] = Job(sid, title, self.queue)
        self.queue.submit(lambda: self._build(job, args), title, job_id=sid)
        self._reap(sid)
        return sid

    def _build(self, job: Job, args) -> SessionHandle:
        seq, key = self.builder(args, self.store)
        try:
            self.store.lock(key, owner=job.sid)
        except SessionLocked as exc:
            raise SessionAlreadyOpen(f"This session is already open in another tab (session id {exc.owner}). "
                                     "Continue there, or finish it first.") from None
        s = self.store.load(key)
        return SessionHandle(key, seq, point_source(seq), s.num_frames, s.frame_offset)

    def _reap(self, sid: str) -> None:
        """Forget a job that failed for another reason than a refused second opening (its
        traceback was logged by the queue); the start screen then gets 404, as before."""
        job = self.jobs.get(sid)
        if job is None:
            return
        st = self.queue.status(sid)
        if st.state == "failed" and not isinstance(st.error, SessionAlreadyOpen):
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
