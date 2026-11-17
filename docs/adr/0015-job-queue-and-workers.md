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
- **Opening a session** (phase 4, ADR 0017; split after the review of phase 4) starts on the
  **cpu** queue (`open_session`): it resumes an existing session, builds a manual one, or builds an
  assisted one from the detection cache. Only when real detection is needed does it submit
  `open_session_detect` (**gpu** queue, detection + tracking in that job) and finish with
  `{"next_job": id}`; the tab follows the chain (`api/routes_tabs.py: _job`), and Cancel cancels
  the job the chain is at. So an opening that needs no detection never waits behind detection
  jobs. Whether the cache has a result depends on the model, which only GPU workers have on disk:
  a worker that has the weights publishes its fingerprint (`detector_config_hash`,
  `checkpoint_hash`; Redis key `lal:detector-fingerprint`) with a TTL
  (`LAL_DETECTOR_FINGERPRINT_TTL_SECONDS`, 90 s) that a daemon thread of the worker refreshes every
  third of it, like a heartbeat, and again at every gpu opening
  (`infra.worker_detector.start_fingerprint_heartbeat`, `publish_fingerprint`); the cpu step
  compares the cache with it. A stopped or reconfigured worker's fingerprint expires, so the cpu
  step no longer builds from that model's cache. Without a published fingerprint (no GPU worker
  alive, Redis unreachable) the cpu step does not judge the cache at all and hands over to the
  gpu queue, where the worker decides itself. Every assisted session records which model made its
  system boxes (`AnnotationSession.detector`: `detector_config_hash`, `checkpoint_hash`,
  `score_threshold` of the cache entry it was built from; stored in the session metadata). `open_session_manual` remains as
  an alias of the cpu step for jobs queued before. Exports run only as tasks
  (`POST /api/v1/tabs/<sid>/exports`, `GET /api/v1/jobs/<id>`, `GET /api/v1/jobs/<id>/download`,
  ADR 0018). Before phase 4 an opening was a thread of the API process that waited for a
  `detect_track` job.
- **One detector per GPU worker process** (`infra/worker_detector.py`): a process loads the configuration
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
- **Duplicate submits** return the existing queued, running, cancelling or done job with the same
  kind and payload (`dedupe_key`, stored in `jobs.cache_key`; an export's payload includes the
  session revision). Since migration 0005 the database enforces it for active jobs: the partial
  unique index `ux_jobs_active_cache_key` on `cache_key WHERE state IN ('queued', 'running')`.
  `PgJobStore.create` inserts with `ON CONFLICT DO NOTHING` on that index, so of two concurrent
  requests one inserts and the other reads the job it made (test:
  `tests/test_infra/test_pg_jobs.py::test_two_concurrent_requests_get_one_job`). Migration 0005
  marks older duplicates that already exist as failed, keeping the newest. A done job serves a
  new request only if the submitter's `reuse_done(result)` accepts it: an export while its zip is
  still there, an opening never (review after phase 4). An opening's payload names no tab, so two
  tabs opening the same sequence at the same time share one job (ADR 0017).
- **Storage**: tasks read point clouds and calibration through BlobStores (`infra/sequences.py`) and
  read / write the detection cache through `BlobStore` (`verification.cache.choose_blob`,
  `output_key`); the path-based helpers remain as thin adapters for `evaluation/`.

## Windows (local development)
```
celery -A infra.celery_app worker -Q gpu -P solo -n gpu@%h
celery -A infra.celery_app worker -Q cpu -P solo -n cpu@%h
celery -A infra.celery_app beat          # one scheduler: periodic tasks (lal.prune_exports)
```
Periodic tasks are housekeeping, not jobs: no `jobs` row; a message older than its interval
expires, so a scheduler running without workers does not pile them up. Exactly one beat process
may run (two would send every task twice; harmless for pruning, but wasted).
The `solo` pool runs one job at a time in the worker process itself (prefork does not work on
Windows); start more cpu workers for more parallel exports. Linux containers follow in phase 5.

## Consequences
- With `LAL_JOB_QUEUE=thread` every detection job makes a detector with its own configuration
  (including its score threshold) and releases it afterwards, as before the queue existed; only GPU
  worker processes keep a model loaded (and only that one configuration).
- Since phase 4 the web API always uses `celery` (ADR 0003); `thread` serves the CLI and tests. An
  API restart does not affect an opening: the job runs on the worker and the page keeps polling
  the tab (ADR 0017).
