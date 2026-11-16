# 0011 Session document loaded per request; cache proposal (not implemented)

Status: accepted (phase 2); the cache is a proposal awaiting a decision

## Decision
- No `ReviewService` and no session document stays in memory between requests. Per sid the manager
  keeps only a handle: session key, sequence (frame names, calibration), point source, number of
  frames and frame offset (fixed for the life of a session; phase 4 moves these to the database).
- Every request: refresh the lock (heartbeat) -> `store.load(key)` -> one `ReviewService` -> query or
  action -> `store.save(key, session, expected_revision=<loaded revision>)` if the annotations
  changed, otherwise only `append_events`.
- Point clouds are served from the handle and a process-wide LRU (`infra.points.PointCache`) without
  loading the document.

## Measurement (note H): load + one edit + save, 400 frames x 10 tracks = 4000 boxes
Median of 20 requests on this machine (RTX 4070 workstation, Windows 11, PostgreSQL 16.4 in Docker):

| session | document | file store | PostgreSQL |
|---|---|---|---|
| typical: 50 single-box undo steps, 2000 events | 0.80 MB | 109 ms | 122 ms |
| worst: 50 whole-track undo steps, 5000 events | 7.82 MB | 1084 ms | 1347 ms |

Breakdown (typical / worst): `to_dict` 30 / 319 ms, `json.dumps` 11 / 99 ms, `json.loads` 7 / 99 ms,
`from_dict` 7 / 90 ms; the rest is I/O. The undo/redo history is 11 % / 91 % of the document
(a whole-track step stores every frame of the track as it was).

Both are above the 50 ms target. Proposal, for a decision before implementing anything:
1. **Cache parsed sessions per process keyed by (session key, revision)**: check the stored revision
   with a cheap query (`SELECT revision`), reuse the parsed session when it matches. Saves the load
   part only (~15 ms typical, ~190 ms worst); the save still costs > 50 ms. Not enough alone.
2. **History out of the document**: undo/redo steps as rows (append on change, delete on undo
   beyond the limit) storing only the changed boxes instead of whole frames. Shrinks the worst
   document ~10x; the typical case barely changes.
3. **Annotations per frame** (one row or JSONB value per frame): a save writes only the frames the
   action changed (the transaction already knows them), so a save costs in proportion to the change,
   not to the session. Largest gain, largest change.
4. **Cheaper serialisation**: `to_dict` uses `dataclasses.asdict` per box (30 ms for 4000 boxes);
   a hand-written conversion is a small, local improvement.

Recommendation: 4 now (small), then 3 + 2 together if the typical case must reach 50 ms; 1 only on
top of them, because without them it cannot reach the target.
