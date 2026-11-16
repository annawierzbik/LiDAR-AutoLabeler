"""Job state in the PostgreSQL `jobs` table, shared by the API and the Celery workers (docs/adr/0015).

States: queued -> running -> done | failed | cancelled; running -> cancelling -> cancelled.
- `create` dedupes: a queued, running, cancelling or done job with the same dedupe key is returned
  instead; a partial unique index (migration 0005) keeps concurrent requests to one active job.
- `start` is one conditional UPDATE (queued -> running, attempts + 1): if the broker hands the same
  job to two workers, only one runs it; the other sees no row and returns.
- A running job writes a heartbeat; `recover_stale` turns a running job whose heartbeat is older
  than `stale_seconds` (its worker died) back into queued, or into failed after `max_attempts`
  runs, so no job stays "running" forever. The caller sends requeued jobs to the broker again.
- Cancellation is a flag the worker reads between frames; a queued job is cancelled at once.
"""
from __future__ import annotations

import datetime as dt
import time
import uuid
from collections.abc import Callable

from sqlalchemy import and_, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from infra.db import JobRow
from infra.jobs import FINAL, JobStatus

# the predicate of ux_jobs_active_cache_key, as literal SQL so PostgreSQL can match the index to it
ACTIVE = text("state IN ('queued', 'running')")


class PgJobStore:
    def __init__(self, engine: Engine, stale_seconds: float = 60.0, max_attempts: int = 3,
                 clock: Callable[[], float] = time.time) -> None:
        self.engine = engine
        self.stale_seconds, self.max_attempts, self._clock = stale_seconds, max_attempts, clock

    def _now(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self._clock(), dt.UTC)

    # --------------------------------------------------------------- API side
    def create(self, kind: str, queue: str, payload: dict, title: str, dedupe: str) -> tuple[str, bool]:
        """(job id, True if a new job was made). A queued, running, cancelling or done job with the
        same dedupe key is returned instead of a new one. The partial unique index
        ux_jobs_active_cache_key (migration 0005) allows one queued or running job per key, so of
        two concurrent requests one inserts and the other gets that job."""
        for _ in range(3):
            with self.engine.begin() as c:
                old = self._existing(c, dedupe)
                if old is not None:
                    return old, False
                job_id = uuid.uuid4().hex[:12]
                made = c.execute(
                    pg_insert(JobRow)
                    .values(id=job_id, kind=kind, queue=queue, title=title, state="queued", progress=0.0,
                            message="", cache_key=dedupe, owner_id=payload.get("owner_id"),
                            payload={**payload, "job_id": job_id})
                    .on_conflict_do_nothing(index_elements=[JobRow.cache_key], index_where=ACTIVE)
                    .returning(JobRow.id)).scalar()
                if made is not None:
                    return job_id, True
            # another request inserted the active job between our select and insert; read it in a
            # new transaction (READ COMMITTED sees it now). If it finished as failed / cancelled
            # meanwhile, try again.
        raise RuntimeError(f"could not create or find a job for key {dedupe}")

    @staticmethod
    def _existing(c, dedupe: str) -> str | None:
        return c.execute(select(JobRow.id).where(JobRow.cache_key == dedupe,
                                                 JobRow.state.notin_(("failed", "cancelled")))
                         .order_by(JobRow.created_at.desc()).limit(1)).scalar()

    VIEW = (JobRow.id, JobRow.kind, JobRow.title, JobRow.state, JobRow.progress, JobRow.message, JobRow.result,
            JobRow.attempts, JobRow.owner_id, JobRow.created_at, JobRow.finished_at)

    @staticmethod
    def _view(r) -> dict:
        return {"id": r.id, "kind": r.kind, "title": r.title, "state": r.state, "progress": r.progress,
                "message": r.message, "result": r.result, "attempts": r.attempts, "owner_id": r.owner_id,
                "created_at": r.created_at.isoformat(),
                "finished_at": r.finished_at.isoformat() if r.finished_at else None}

    def view(self, job_id: str, owner_id: int | None = None) -> dict:
        """The job as the API shows it; KeyError if it does not exist or (with `owner_id`) is not
        that user's."""
        q = select(*self.VIEW).where(JobRow.id == job_id)
        if owner_id is not None:
            q = q.where(JobRow.owner_id == owner_id)
        with self.engine.connect() as c:
            r = c.execute(q).first()
        if r is None:
            raise KeyError(job_id)
        return self._view(r)

    def page(self, owner_id: int | None, limit: int, offset: int) -> tuple[list[dict], int]:
        """Newest first; `owner_id` None: every user's jobs."""
        cond = [] if owner_id is None else [JobRow.owner_id == owner_id]
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(JobRow).where(*cond)).scalar()
            rows = c.execute(select(*self.VIEW).where(*cond).order_by(JobRow.created_at.desc(), JobRow.id)
                             .limit(limit).offset(offset)).all()
        return [self._view(r) for r in rows], total

    def status(self, job_id: str) -> JobStatus:
        with self.engine.connect() as c:
            r = c.execute(select(JobRow.id, JobRow.title, JobRow.state, JobRow.message, JobRow.result, JobRow.kind,
                                 JobRow.progress, JobRow.attempts).where(JobRow.id == job_id)).first()
        if r is None:
            raise KeyError(job_id)
        return JobStatus(r.id, r.title, r.state, r.message, r.result, None, r.kind, r.progress, r.attempts)

    def request_cancel(self, job_id: str) -> bool:
        with self.engine.begin() as c:
            row = c.execute(select(JobRow.state).where(JobRow.id == job_id)).first()
            if row is None:
                raise KeyError(job_id)
            if row.state in FINAL:
                return False
            new = "cancelled" if row.state == "queued" else "cancelling"
            c.execute(update(JobRow).where(JobRow.id == job_id)
                      .values(cancel_requested=True, state=new, message="cancel requested",
                              finished_at=self._now() if new == "cancelled" else None))
            return True

    def recover_stale(self, job_id: str | None = None) -> list[tuple[str, str]]:
        """Jobs whose worker stopped sending heartbeats: requeued (returned as (id, queue) for the
        broker) or, after max_attempts runs or when cancelling, finished."""
        limit = self._now() - dt.timedelta(seconds=self.stale_seconds)
        cond = and_(JobRow.state.in_(("running", "cancelling")), JobRow.heartbeat_at < limit)
        if job_id is not None:
            cond = and_(cond, JobRow.id == job_id)
        requeued = []
        with self.engine.begin() as c:
            for r in c.execute(select(JobRow.id, JobRow.state, JobRow.attempts, JobRow.queue, JobRow.worker)
                               .where(cond).with_for_update(skip_locked=True)).all():
                lost = f"worker {r.worker} stopped responding"
                if r.state == "cancelling":
                    values = {"state": "cancelled", "message": f"cancelled ({lost})", "finished_at": self._now()}
                elif r.attempts >= self.max_attempts:
                    values = {"state": "failed", "message": f"{lost}; gave up after {r.attempts} attempts",
                              "finished_at": self._now()}
                else:
                    values = {"state": "queued", "message": f"{lost}; waiting for another attempt"}
                    requeued.append((r.id, r.queue))
                c.execute(update(JobRow).where(JobRow.id == r.id).values(**values))
        return requeued

    def queued(self) -> list[tuple[str, str]]:
        with self.engine.connect() as c:
            return [tuple(r) for r in c.execute(select(JobRow.id, JobRow.queue).where(JobRow.state == "queued"))]

    # ------------------------------------------------------------ worker side
    def start(self, job_id: str, worker: str) -> dict | None:
        """Claim a queued job: its payload, or None if it is not queued any more (running elsewhere,
        finished, or cancelled before it started)."""
        now = self._now()
        with self.engine.begin() as c:
            c.execute(update(JobRow).where(JobRow.id == job_id, JobRow.state == "queued", JobRow.cancel_requested)
                      .values(state="cancelled", message="cancelled before it started", finished_at=now))
            row = c.execute(update(JobRow).where(JobRow.id == job_id, JobRow.state == "queued")
                            .values(state="running", attempts=JobRow.attempts + 1, worker=worker, heartbeat_at=now,
                                    started_at=now, message="started")
                            .returning(JobRow.payload)).first()
        return dict(row.payload) if row is not None else None

    def heartbeat(self, job_id: str, progress: float | None = None, message: str | None = None) -> None:
        values = {"heartbeat_at": self._now()}
        if progress is not None:
            values["progress"] = progress
        if message is not None:
            values["message"] = message
        with self.engine.begin() as c:
            c.execute(update(JobRow).where(JobRow.id == job_id, JobRow.state.in_(("running", "cancelling")))
                      .values(**values))

    def cancel_requested(self, job_id: str) -> bool:
        with self.engine.connect() as c:
            return bool(c.execute(select(JobRow.cancel_requested).where(JobRow.id == job_id)).scalar())

    def _end(self, job_id: str, **values) -> None:
        with self.engine.begin() as c:
            c.execute(update(JobRow).where(JobRow.id == job_id,
                                           or_(JobRow.state == "running", JobRow.state == "cancelling"))
                      .values(finished_at=self._now(), **values))

    def finish(self, job_id: str, result: dict) -> None:
        self._end(job_id, state="done", progress=1.0, result=result, message="done")

    def fail(self, job_id: str, message: str) -> None:
        self._end(job_id, state="failed", message=message)

    def cancelled(self, job_id: str, message: str = "cancelled") -> None:
        self._end(job_id, state="cancelled", message=message)
