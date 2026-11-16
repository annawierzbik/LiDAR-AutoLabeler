# 0011 Session document loaded per request; sessions stored in parts in PostgreSQL

Status: accepted (phase 2, storage format changed in phase 2b). No cache (decision of phase 2b).

## Decision (phase 2)
- No `ReviewService` and no session document stays in memory between requests. Per sid the manager
  keeps only a handle: session key, sequence (frame names, calibration), point source, number of
  frames and frame offset (fixed for the life of a session; phase 4 moves these to the database).
- Every request: refresh the lock (heartbeat) -> `store.load(key)` -> one `ReviewService` -> query or
  action -> `store.save(key, session, expected_revision=<loaded revision>)` if the annotations
  changed, otherwise only `append_events`.
- Point clouds are served from the handle and a process-wide LRU (`infra.points.PointCache`) without
  loading the document.

## Problem found in phase 2 (note H)
One request (load + one edit + save) on a 400-frame session (10 tracks, 4000 boxes) took 117 ms
(typical history) and 1.37 s (50 whole-track undo steps) on PostgreSQL. Breakdown of the phase-2
request (median of 20, ms): typical / worst

| load (SELECT) | deserialise | edit | state() | serialise | write |
|---|---|---|---|---|---|
| 13.4 / 104.6 | 19.8 / 373.1 | 0.5 / 1.0 | 0.8 / 1.0 | 43.9 / 511.2 | 36.2 / 327.3 |

Loading was not the main cost: serialising and writing the whole document (91 % of it the undo
history in the worst case) were. `state()` costs about 1 ms, so it was not changed.

## Decision (phase 2b)
1. **Faster serialisation** (`Annotation.to_dict` builds the dict directly instead of
   `dataclasses.asdict`; status lookup by table). The JSON written is identical, so the file format
   does not change (`FileSessionStore` keeps the whole document; only this step applies to it).
2. **PostgreSQL: annotations per frame.** `session_frames` has one row per frame; a save upserts
   only the frames the action changed (`AnnotationSession.dirty_frames`, filled by the
   transaction / undo code). Revision check, frame writes, history changes and events are one
   database transaction.
3. **PostgreSQL: history in its own rows.** `session_history` has one row per undo or redo step,
   holding only the frames that step changed. A save applies the request's history operations
   (`AnnotationSession.history_ops`: push, pop, clear) and deletes all but the newest 50 rows per
   stack. `load` reads only the newest step of each stack, enough for the one action of a request,
   so its cost does not depend on the length of the history.
4. Migration `0002` moves existing documents into the rows (and back on downgrade); data are kept
   both ways (test: `test_migration_0002_moves_documents_into_rows_and_back`).

## Measurements before and after
`python -m scripts.bench_session_store` (same benchmark for both columns; raw CSV in
`docs/benchmarks/`: `session_store_phase2_before.csv` = commit 2ba5bc8,
`session_store_phase2b_after.csv`). Median of 20 requests, ms, "request" = load + edit + save.
Machine: Windows 11 workstation, PostgreSQL 16.4 in Docker Desktop on the same machine.

| store | session | before (phase 2) | after (phase 2b) | target |
|---|---|---|---|---|
| PostgreSQL | typical (50 single-box steps) | 117.2 | **40.3** | <= 50 |
| PostgreSQL | 10 whole-track steps | 1337.8 | **43.5** | |
| PostgreSQL | 25 whole-track steps | 1364.7 | **44.7** | |
| PostgreSQL | 50 whole-track steps (worst) | 1368.3 | **44.9** | <= 150 |
| file | typical | 87.0 | 61.1 | (no target) |
| file | 50 whole-track steps | 1053.0 | 735.1 | (no target) |

After phase 2b the PostgreSQL request costs the same for 10, 25 and 50 history steps (43.5 /
44.7 / 44.9 ms); stage medians for the worst case: load 34.2, edit 0.7, state 0.9, save 9.3 ms.
The file store still writes the whole document, so its cost still follows the document size.

## Consequences
- A session loaded from PostgreSQL holds only the newest undo and redo step; undoing several steps
  takes one request each (as the browser does).
- `AnnotationSession` tracks what changed since it was loaded (`dirty_frames`, `history_ops`);
  every change of annotations must keep going through `_transaction` or `_swap`.
