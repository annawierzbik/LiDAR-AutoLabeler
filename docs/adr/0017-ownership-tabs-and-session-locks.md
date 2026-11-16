# 0017 Ownership, browser tabs in the database, session build on the worker, locks per (user, client)

Status: accepted (phase 4)

## Context
Before phase 4 the API process built sessions in a thread and kept, per open browser tab, a handle
(session key, sequence, point source) in memory, and the lock owner was that tab's id. A restart
lost the handles (the page had to be opened again and had to wait for the lock to expire), and two
API replicas could not serve the same tab.

## Decision
- **Ownership.** `sequences`, `sessions` and `jobs` have an `owner_id`; a tab belongs to the user
  who opened it. A user sees and reaches only their own; an admin sees all. A request for another
  user's resource answers **404**, exactly like an id that does not exist, so ids of other users'
  resources cannot be confirmed (tested for every endpoint in `tests/test_api/test_ownership.py`).
  Session keys start with `u<owner id>_`, so every user has their own session of a sequence.
- **Opening = a job.** `POST /api/v1/tabs {sequence_id, mode}` stores a row of `tabs` (id = the
  `sid` of `/annotate?sid=`, user, client id, sequence, job) and submits `open_session` (gpu queue)
  or `open_session_manual` (cpu queue). The worker resumes the session or builds it (from the
  detection cache, or by running detection + tracking inside the same job, with progress and
  cooperative cancellation) and returns `{session_key, num_frames, frame_offset}` as the job
  result. The API only reads job state from the database; `GET /api/v1/tabs/<sid>` shows
  loading (with progress) | ready | error | finished.
- **Every request** reads the tab row and the job result (two small queries), refreshes the lock,
  loads the session document (ADR 0011) and serves the request. The process keeps nothing about
  tabs or sessions between requests; only the point-cloud LRU is per process, and it is a cache.
- **Lock owner = (user, client).** The page makes a random client id once per browser tab
  (`sessionStorage`, survives reloads, not a credential) and sends it as `X-Client-Id`. The lock
  owner is `u<user id>/<client id>`. The same owner gets the lock again at once, also after an API
  restart, since the lock check (`PgSessionStore.lock`) accepts its own owner; another tab or user
  gets 409 `locked` until the holder finishes or stops sending heartbeats for
  `LAL_LOCK_TTL_SECONDS`. When the opening job is done, the tab takes the lock on its next status
  poll, so a second tab on the same session sees the "open in another tab" message on the start
  page. An admin may read another user's tab, but the lock still protects a live session.
- **Finish** releases the lock, marks the session finished and the tab `finished`; later requests
  of that tab get 409 `finished`.

## Consequences
- Two tabs of the same user that open the same sequence at the same time may both run detection
  (each opening is its own job); the detection cache and `SessionExists` handling keep the result
  consistent.
- Sessions stored before phase 4 have no owner and no `u<id>_` prefix: admins see them in
  `GET /api/v1/sessions`, but no tab opens them (they could be assigned with an UPDATE of
  `owner_id` and key; not automated).
- Tab rows are never deleted yet; they are small (one per opening).
