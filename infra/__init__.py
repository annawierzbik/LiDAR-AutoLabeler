"""Storage, job and settings abstractions behind the web layer (see docs/adr/).

SessionStore: FileSessionStore (default) and PgSessionStore (LAL_SESSION_STORE=postgres).
BlobStore: LocalBlobStore and S3BlobStore (LAL_BLOB_STORE=s3; the areas are in infra/storage.py).
TaskQueue: ThreadTaskQueue and CeleryTaskQueue (infra/jobs.py, infra/celery_queue.py).
Settings: LAL_* environment variables.
"""
