# 0015 Job queue and workers (Celery + Redis)

Status: accepted (phase 3)

## Decision
- **Tasks** (`infra/tasks.py`) are plain functions `fn(payload, ctx) -> result`, independent of the
  queue: `detect_track` (queue `gpu`: detection + tracking into the detection cache) and `export`
  (queue `cpu`: label zip into the export BlobStore). `ctx.progress(fraction)` and
  `ctx.should_stop()` are their only link to the queue.
- **TaskQueue** (`submit(kind, payload)`, `status`, `cancel`), chosen with `LAL_JOB_QUEUE`:
  - `thread` (default): `ThreadTaskQueue`, tasks in threads of the API process, state in memory.
  - `celery`: `CeleryTaskQueue`; the broker message carries only the job id; payload, state,
    progress, attempts, result and error are rows of the `jobs` table (`infra/pg_jobs.py`, migration
    0003). Needs `LAL_REDIS_URL` and `LAL_DATABASE_URL`. Redis runs in `docker-compose.db.yml`.
- **Opening a session** still runs as a short local job of the web manager; when the session needs
  detection, that job submits `detect_track` and waits for it, so the start page shows the task's
  progress and a Cancel button (`POST /api/jobs/<sid>/cancel`). Exports can run as tasks
  (`POST /api/s/<sid>/export_jobs`, `GET /api/tasks/<id>`, `GET /api/tasks/<id>/download`); the
  annotation page keeps its direct download for now.
- **One detector per worker process** (`infra/worker_detector.py`): a process loads the configuration
  of `LAL_WORKER_DETECTOR_*` (config, checkpoint, score threshold, factory) once, on its first job.
  A job whose cache metadata needs another configuration fails with a message naming both; it never
  runs on the wrong model. The FOV crop is a per-job setting of the loaded detector.
- **Cooperative cancellation** (plan of ADR 0014): `should_stop()` is checked before every frame; a
  stopped run raises and writes nothing (the cache entry is written once, atomically, at the end).
  Queued jobs are cancelled at once; running ones go `cancelling` -> `cancelled`.
- **Progress, throttled**: a worker thread writes heartbeat + progress + checks the cancel flag every
  `LAL_JOB_HEARTBEAT_SECONDS` (default 2 s), not once per frame.
- **Worker death**: messages are acknowledged late (`task_acks_late`, `task_reject_on_worker_lost`,
  prefetch 1). A running job whose heartbeat is older than `LAL_JOB_STALE_SECONDS` is requeued and
  sent again (noticed on `status` and when a worker starts), or failed after `LAL_JOB_MAX_ATTEMPTS`
  runs. Running a job twice is harmless: `PgJobStore.start` lets one claim win, and `detect_track`
  first looks in the detection cache (a finished run is never repeated).
- **Duplicate submits** return the existing queued, running or done job with the same kind and
  payload (`dedupe_key`; an export's payload includes the session revision).
- **Storage**: tasks read point clouds and calibration through BlobStores (`infra/sequences.py`) and
  read / write the detection cache through `BlobStore` (`verification.cache.choose_blob`,
  `output_key`); the path-based helpers remain as thin adapters for `evaluation/`.

## Windows (local development)
```
celery -A infra.celery_app worker -Q gpu -P solo -n gpu@%h
celery -A infra.celery_app worker -Q cpu -P solo -n cpu@%h
```
The `solo` pool runs one job at a time in the worker process itself (prefork does not work on
Windows); start more cpu workers for more parallel exports. Linux containers follow in phase 5.

## Consequences
- With `LAL_JOB_QUEUE=thread` the API process holds the detector after its first detection (it was
  made and dropped per run before); with `celery` only gpu workers do.
- The web manager's opening job waits in an API thread; if the API restarts meanwhile, the
  detection still finishes in the worker and a new opening finds it in the cache.
