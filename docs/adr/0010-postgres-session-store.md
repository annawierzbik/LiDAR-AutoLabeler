# 0010 PostgreSQL session store

Status: accepted (phase 2)

## Decision
- Schema (SQLAlchemy 2 models in `infra/db.py`, Alembic migrations in `migrations/`):
  `users`, `sequences`, `jobs` (created now, used from phase 3/4), `sessions`, `session_events`.
  A test checks that the migrations and the models agree and that the downgrade works.
- `sessions`: `key` (= the session file name without `.json`, same keys as the file store),
  `document` JSONB (`AnnotationSession.to_dict(include_events=False)`: annotations, revision,
  undo/redo history), `revision`, `status` (open | finished), `locked_by`, `locked_at`, `owner_id`,
  `sequence_id`, timestamps.
- `session_events`: one row per event (`t`, `action`, `frame`, `detail` JSONB), only ever inserted.
  The event log is no longer part of the document in the database, so a save does not rewrite it.
- `PgSessionStore.save` is optimistic: `UPDATE sessions SET ... WHERE key = :key AND revision =
  :expected`; no row updated -> `RevisionConflict`, which the API answers with 409. New sessions
  are made with `create` (`SessionExists` if the key is taken); `save` of an unknown key is an error
  in both stores.
- The store is chosen with `LAL_SESSION_STORE` (`file`, the default, or `postgres` with
  `LAL_DATABASE_URL`). `FileSessionStore` keeps working unchanged for local use.
- `scripts/import_sessions.py` imports `results/sessions/*.json` (v1 and v2) once: document, history
  and event log; existing sessions are skipped unless `--replace`.
- Local database and test database: `docker-compose.db.yml` (PostgreSQL 16.4 only, host port 55432).
  Database tests are marked `postgres`, deselected by default, and create and drop their own
  throw-away database (`tests/conftest.py`).

## Consequences
- Several API processes can share sessions (needed in phase 4); the file store stays single-process.
- The JSONB document is still written whole on every change (see 0011 for the cost).
