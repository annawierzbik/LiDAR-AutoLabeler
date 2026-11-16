# 0004 Storage and job protocols with local implementations

Status: accepted (phase 1). `SessionManager` was removed in phase 4 (ADR 0017); the lock owner
is now `u<user id>/<client id>`. `JobQueue` / `ThreadJobQueue` were removed after phase 4
(unused since openings became tasks); the task queues of ADR 0015 replace them.

## Decision
`infra/` defines typing protocols and local implementations that reproduce the single-process
behaviour:

| Protocol | Methods | Local implementation |
|---|---|---|
| `SessionStore` | `load`, `save(key, session, expected_revision)`, `list`, `lock`, `unlock`, `locked_by` | `FileSessionStore`: one JSON file per session (`results/sessions/<key>.json`), locks in memory |
| `BlobStore` | `get`, `put` (atomic), `exists`, `list` | `LocalBlobStore`: files under a root; keys are relative POSIX paths, `..`/absolute keys rejected |
| `JobQueue` | `submit(fn, title, job_id)`, `status`, `cancel` | `ThreadJobQueue`: a daemon thread per job, or inline for tests |
| `PointSource` | `__len__`, `xyz(index)` | `BlobPointSource` over a `LocalBlobStore` on the velodyne folder |

- `ReviewService` and `SessionManager` use only these protocols. The second-tab refusal is now the
  SessionStore lock (owner = the tab's sid); phase 2 replaces it with an expiring lock.
- `save` raises `RevisionConflict` when the stored revision differs from `expected_revision`
  (optimistic locking, also in the file store); the service reports it as `save_error`.
- `ThreadJobQueue.cancel` succeeds only before a job starts; a running thread cannot be stopped.
- Detection/tracking cache files are written through `LocalBlobStore.put` (atomic). Reading them
  still uses the path-based `verification/cache.py`; the workers of phase 3 move that behind the
  BlobStore. `evaluation/` (not to be touched) still writes its cache non-atomically.
