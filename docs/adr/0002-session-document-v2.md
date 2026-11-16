# 0002 Session document v2: persisted revision and bounded undo history

Status: accepted (phase 1)

## Context
The session file (v1) stored annotations, the event log and the next track id. The revision
counter and the undo/redo stacks were memory-only: the revision restarted at 0 after every load and
the undo history was lost on restart. Optimistic locking (phase 2) needs a persisted revision, and
a stateless API (phase 4) must be able to undo after loading the session in another request or
replica.

## Decision
- File format version 2 adds `revision` and `undo` / `redo`: the last `UNDO_LIMIT = 50` steps each,
  a step being `{label, frames: {frame: [annotation]}}` (the frames as they were before the step).
- The in-memory stacks are bounded to the same 50 steps, so memory and file agree.
- The loader accepts v1 (revision 0, empty history) and v2; saving always writes v2.
- Writes stay atomic (temporary file in the same folder, fsync, `os.replace`;
  `infra.blob.atomic_write_text`).

## Consequences
- Undo works after a reload, also in the desktop app.
- Files grow with the history: at most 50 snapshots of the changed frames (a track action snapshots
  every frame it touched).
- Older versions of the program cannot read v2 files (they check `version == 1`).
