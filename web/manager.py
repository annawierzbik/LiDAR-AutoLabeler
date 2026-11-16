"""Everything the start screen needs, without HTTP: opening folders and the open sessions.

Opening a folder submits a *job* to the JobQueue that builds the session (loading cached
detections or, when none exist, running the detector can take minutes) and turns it into a
`ReviewService`. Several sessions can be open at the same time, each addressed by a short id
(`sid`, the job id). A job is removed as soon as it fails (the traceback goes to the log) or its
session is finished. A session is opened by one sid at a time: the SessionStore lock; a second
opening of the same session file gets status "error" with a message naming the open sid.

The manager depends only on the protocols in infra/ (SessionStore, JobQueue, BlobStore via
infra.points); the defaults are the local implementations.
"""
from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable
from pathlib import Path

from infra.jobs import JobQueue, ThreadJobQueue
from infra.points import point_source
from infra.sessions import FileSessionStore, SessionLocked, SessionStore
from infra.settings import get_settings
from web.service import ReviewService

STATES = {"queued": "loading", "running": "loading", "done": "ready", "failed": "error", "cancelled": "error"}


class SessionAlreadyOpen(Exception):
    pass


class Job:
    """What the start screen shows about one sid; status and result come from the JobQueue."""

    def __init__(self, sid: str, title: str, queue: JobQueue) -> None:
        self.sid, self.title, self._queue = sid, title, queue
        self.key: str | None = None

    @property
    def status(self) -> str:
        return STATES[self._queue.status(self.sid).state]

    @property
    def message(self) -> str:
        st = self._queue.status(self.sid)
        return "cancelled" if st.state == "cancelled" else st.message

    @property
    def service(self) -> ReviewService | None:
        st = self._queue.status(self.sid)
        return st.result if st.state == "done" else None

    def view(self) -> dict:
        return {"sid": self.sid, "status": self.status, "message": self.message, "title": self.title}


def default_builder(args):
    from scripts.verify import build_session
    return build_session(args)


def make_args(**overrides) -> argparse.Namespace:
    """Arguments of scripts.verify with their defaults, plus `overrides`."""
    from scripts.verify import add_arguments
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class SessionManager:
    def __init__(self, builder: Callable | None = None, background: bool = True,
                 queue: JobQueue | None = None, store: SessionStore | None = None) -> None:
        """`builder(args)` -> (session, sequence, session file path) as scripts.verify.build_session;
        the session file must lie in the store's folder. `background=False` runs jobs inline (tests)."""
        self.builder = builder or default_builder
        self.queue = queue or ThreadJobQueue(background=background)
        self.store = store or FileSessionStore(get_settings().session_dir)
        self.jobs: dict[str, Job] = {}
        self.finished: set[str] = set()   # sids of finished sessions (a few bytes each, kept for 409s)

    # ------------------------------------------------------------------ jobs
    def start_job(self, args, title: str) -> str:
        sid = uuid.uuid4().hex[:8]
        job = self.jobs[sid] = Job(sid, title, self.queue)
        self.queue.submit(lambda: self._build(job, args), title, job_id=sid)
        self._reap(sid)
        return sid

    def _build(self, job: Job, args) -> ReviewService:
        session, seq, path = self.builder(args)
        key = self.store.key_for(Path(path))
        try:
            self.store.lock(key, owner=job.sid)
        except SessionLocked as exc:
            raise SessionAlreadyOpen(f"This session is already open in another tab (session id {exc.owner}). "
                                     "Continue there, or finish it first.") from None
        job.key = key
        return ReviewService(session, self.store, key, point_source(seq), sequence=seq)

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

    def service(self, sid: str) -> ReviewService:
        """KeyError for unknown ids, LookupError while loading, when refused or when finished
        (a tab still open on a finished session gets 409, not 404)."""
        if sid in self.finished:
            raise LookupError(f"session {sid} was finished and is closed")
        job = self.job(sid)
        service = job.service
        if service is None:
            raise LookupError(job.message or f"session {sid} is not ready ({job.status})")
        if service.closed:
            raise LookupError(f"session {sid} was finished and is closed")
        return service

    def finish(self, sid: str) -> None:
        job = self.job(sid)
        service = job.service
        if service is not None:
            service.close()
            self.store.unlock(job.key, sid)
        del self.jobs[sid]
        self.finished.add(sid)

    # ---------------------------------------------------------------- folder
    def open_folder(self, path: str) -> str:
        from ingestion.folder import validate_folder
        folder = Path(path.strip().strip('"'))
        frames = validate_folder(folder)               # readable errors for folders that cannot be used
        return self.start_job(make_args(folder=str(folder)), f"{folder.name} ({frames} frames)")
