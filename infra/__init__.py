"""Storage, job and settings abstractions behind the web layer (see docs/adr/).

SessionStore: FileSessionStore (default) and PgSessionStore (LAL_SESSION_STORE=postgres).
BlobStore: LocalBlobStore. TaskQueue: ThreadTaskQueue and CeleryTaskQueue (infra/jobs.py,
infra/celery_queue.py). Settings: LAL_* environment variables.
"""
