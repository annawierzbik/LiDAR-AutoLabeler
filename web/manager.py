"""Everything the start screen needs, without HTTP: opening folders and the open sessions.

A *job* builds a session in a background thread (loading cached detections or, when none exist,
running the detector can take minutes) and turns into a `ReviewService` when ready. Several
sessions can be open at the same time, each addressed by a short id (`sid`). A job is removed
as soon as it fails (the traceback goes to the log) or its session is finished.
"""
from __future__ import annotations

import argparse
import logging
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, Optional

from web.service import ReviewService

log = logging.getLogger(__name__)


class Job:
    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.status = "loading"      # loading -> ready
        self.message = ""
        self.service: Optional[ReviewService] = None
        self.title = ""

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
    def __init__(self, builder: Optional[Callable] = None, background: bool = True) -> None:
        self.builder = builder or default_builder
        self.background = background            # tests build synchronously
        self.jobs: Dict[str, Job] = {}
        self.by_path: Dict[str, ReviewService] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ jobs
    def start_job(self, args, title: str) -> str:
        job = Job(uuid.uuid4().hex[:8])
        job.title = title
        self.jobs[job.sid] = job
        if self.background:
            threading.Thread(target=self._build, args=(job, args), daemon=True).start()
        else:
            self._build(job, args)
        return job.sid

    def _build(self, job: Job, args) -> None:
        try:
            session, seq, path = self.builder(args)
            with self._lock:
                key = str(path)
                service = self.by_path.get(key)       # same session already open: share it
                if service is None or service.closed:
                    service = ReviewService(session, seq, path)
                    self.by_path[key] = service
            job.service, job.status = service, "ready"
        except Exception:
            log.exception("Could not open session %s (%s)", job.sid, job.title)
            self.jobs.pop(job.sid, None)

    def job(self, sid: str) -> Job:
        return self.jobs[sid]

    def service(self, sid: str) -> ReviewService:
        """KeyError for unknown ids, LookupError while loading or when the session was closed."""
        job = self.jobs[sid]
        if job.service is None:
            raise LookupError(f"session {sid} is not ready ({job.status})")
        if job.service.closed:
            raise LookupError(f"session {sid} was finished and is closed")
        return job.service

    def finish(self, sid: str) -> None:
        job = self.jobs[sid]
        if job.service is not None:
            job.service.close()
            with self._lock:
                self.by_path = {k: v for k, v in self.by_path.items() if v is not job.service}
        del self.jobs[sid]

    # ---------------------------------------------------------------- folder
    def open_folder(self, path: str) -> str:
        from ingestion.folder import validate_folder
        folder = Path(path.strip().strip('"'))
        frames = validate_folder(folder)               # readable errors for folders that cannot be used
        return self.start_job(make_args(folder=str(folder)), f"{folder.name} ({frames} frames)")
