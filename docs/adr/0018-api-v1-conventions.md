# 0018 API v1 conventions: paths, errors, pagination, OpenAPI, health, exports, statelessness

Status: accepted (phase 4)

## Decision
- **Paths.** Every data endpoint is under `/api/v1` (`api/app.py` and `api/routes_*.py`); the pages
  `/`, `/login`, `/annotate` and `/static` stay at the root. The old `/api/...` endpoints, the
  in-process `web/manager.py` and `web/server.py` are removed.
- **Errors.** Every non-2xx answer has the body
  `{"error": {"code": "...", "message": "...", "details": ...}}` (`api/errors.py`). `code` is
  stable for programs (`not_found`, `unauthorized`, `invalid_credentials`, `csrf_failed`,
  `forbidden`, `outside_data_root`, `validation_error`, `conflict`, `not_ready`, `locked`,
  `finished`, `revision_conflict`, `unknown_action`, `payload_too_large`, `gone`,
  `internal_error`); `message` is for people; validation errors list `{field, message}` in
  `details`. Unexpected exceptions are logged and answered 500 without internals.
- **Pagination.** List endpoints (`/sequences`, `/jobs`, `/sessions`, `/users`) take `limit`
  (1..`LAL_PAGE_MAX`, default 50) and `offset` and answer `{"items", "total", "limit", "offset"}`,
  newest first (users: by id).
- **OpenAPI** at `/api/v1/openapi.json`, UI at `/api/v1/docs`; endpoints carry descriptions, the
  error responses and the `ErrorResponse` schema.
- **Health.** `GET /healthz` answers 200 while the process runs (no dependency checked);
  `GET /readyz` checks the database (`SELECT 1`), Redis (`PING`) and that the results and upload
  blob folders are writable, and answers 200 `ready` or 503 `not_ready` with the result per check.
- **Exports** go through the cpu queue only: `POST /api/v1/tabs/<sid>/exports` returns a job id
  (202), `GET /api/v1/jobs/<id>` shows progress and, when done, the summary;
  `GET /api/v1/jobs/<id>/download` returns the stored zip. The zip is a blob named
  `<expiry epoch>/<job id>/<archive>`; after `LAL_EXPORT_TTL_HOURS` (24) the download answers 410
  `gone`, and every export run deletes expired zips. A format that cannot be produced fails the
  job with the exporter's message. The direct in-request export and its preview are removed.
- **Statelessness.** The API keeps no per-session or per-tab state between requests (ADR 0017);
  users, sequences, tabs, sessions and jobs are database rows, the queue is Redis, frames and
  exports are blobs. `tests/e2e/test_annotate_ui.py::test_two_api_processes_behind_a_round_robin_gateway`
  runs the whole UI scenario (login, opening, review, undo / redo, export, finish) with every
  request going to the next of two API processes.

## Consequences
- The API refuses to start without `LAL_DATABASE_URL`, `LAL_REDIS_URL`, `LAL_JWT_SECRET`,
  `LAL_SESSION_STORE=postgres` and `LAL_JOB_QUEUE=celery` (ADR 0003).
- A client that relied on the old paths or on `{"detail": ...}` errors must change.
