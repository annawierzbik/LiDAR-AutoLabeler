# 0014 Cooperative cancellation (plan for phase 3)

Status: implemented in phase 3 (see ADR 0015). The flag is `jobs.cancel_requested`, read by the
worker's heartbeat thread; `track_sequence` takes `should_stop` and `progress` callbacks.

## Context
`ThreadJobQueue.cancel` only works before a job starts; a running detection cannot be stopped, and
the same holds for a Celery task unless it is killed (which leaves partial results and a model
process in an unknown state).

## Plan
- Long jobs check a cancellation flag between units of work: detection + tracking between frames
  (`pipeline.detect_track.track_sequence` gets an optional `should_stop()` callable and a progress
  callback), export between frames.
- The flag lives where every worker can read it: `jobs.state = 'cancelling'` (set by
  `JobQueue.cancel`), read through a cheap query every N frames (or a Redis key in phase 3).
- On a stop request the job finishes its current frame, writes nothing to the detection cache (a
  partial run is not a valid cache entry), sets `state = 'cancelled'` and returns.
- `cancel` returns immediately; the state shows `cancelling` until the worker has stopped.
- Tests: a fake detector that blocks per frame; cancel mid-run; assert the state sequence, that the
  cache file was not written and that a resubmit computes from scratch.
- Hard kill (Celery `revoke(terminate=True)`) stays a last resort for a worker that does not react
  within a timeout; it is recorded as `failed` with the reason.
