# 0012 Session locks with expiry and heartbeat; per-process locks in the file store

Status: accepted (phase 2); replaces the second-tab refusal of phase 1

## Decision
- A session is used by one browser tab at a time: `store.lock(key, owner=<sid>)`. Every request of
  that tab refreshes the lock, and the page sends `POST /api/s/<sid>/heartbeat` every 30 s.
- The lock expires `LAL_LOCK_TTL_SECONDS` (default 120 s) after the last refresh. A tab closed
  without "Finish" therefore frees its session after at most two minutes, instead of until the
  server restarts.
- Opening a session whose lock another live tab holds is refused with a message naming that tab's
  sid. A tab whose lock expired and was taken by another tab gets 409 ("now open in another tab").
- `finish` releases the lock (PostgreSQL also sets `status = finished`).
- **PostgreSQL**: the lock is the row's `locked_by` / `locked_at`, taken or refreshed with one
  conditional `UPDATE`, so it holds across processes and server restarts.
- **File store**: the lock lives in the memory of the one server process (a dict guarded by a
  mutex, with the same expiry rule). It is correct only while a single process serves the files,
  which is the file store's use (local, single user). Two processes on the same `results/sessions`
  would not see each other's locks; saves would still be protected by the revision check
  (read-check-write under a per-process mutex, not across processes). OS file locks were not used:
  they behave differently on Windows and Linux and do not expire.

## Consequences
- After a server restart the file store has forgotten all locks; PostgreSQL keeps them, so the
  same session can be opened again only after the old lock expired (up to the TTL).
- Open issue for phase 4: after a restart (or on another replica) the old tab's sid is unknown to
  the new process (the sid -> handle map is in memory), so that tab gets 404 and must reopen the
  sequence. Phase 4 stores the handles (sequence, key) in the database.
