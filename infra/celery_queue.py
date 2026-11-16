"""TaskQueue that sends jobs to Celery workers; state in the `jobs` table (docs/adr/0015)."""
from __future__ import annotations

from collections.abc import Callable

from infra.jobs import JobStatus
from infra.pg_jobs import PgJobStore
from infra.tasks import TASKS, dedupe_key


class CeleryTaskQueue:
    def __init__(self, store: PgJobStore, send: Callable[[str, str], None] | None = None) -> None:
        if send is None:
            from infra.celery_app import dispatch as send
        self.store, self.send = store, send

    def submit(self, kind: str, payload: dict, title: str = "") -> str:
        if kind not in TASKS:
            raise KeyError(f"unknown task {kind!r}")
        queue = TASKS[kind][0]
        job_id, created = self.store.create(kind, queue, payload, title, dedupe_key(kind, payload))
        if created:
            self.send(job_id, queue)
        return job_id

    def status(self, job_id: str) -> JobStatus:
        """Also notices a lost worker: a stale running job is requeued (and sent again) or failed."""
        for jid, queue in self.store.recover_stale(job_id):
            self.send(jid, queue)
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> bool:
        return self.store.request_cancel(job_id)
