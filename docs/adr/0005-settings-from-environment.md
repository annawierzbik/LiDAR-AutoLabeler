# 0005 Settings from environment variables

Status: accepted (phase 1)

## Decision
`infra/settings.py` uses pydantic-settings; every setting comes from an environment variable with
the prefix `LAL_`. Defaults reproduce the behaviour before settings existed:

| Variable | Default | Used by |
|---|---|---|
| `LAL_RESULTS_DIR` | `<project>/results` (relative values are relative to the project) | `verification.paths.RESULTS_DIR`, FileSessionStore (`<results>/sessions`) |
| `LAL_HOST`, `LAL_PORT` | `127.0.0.1`, `8000` | `scripts.web` defaults (command-line flags still win) |
| `LAL_POINT_CACHE_FRAMES` | `8` | point clouds kept per open session |
| `LAL_SESSION_STORE`, `LAL_BLOB_STORE`, `LAL_JOB_QUEUE` | `file`, `local`, `thread` | only value allowed so far; later phases add backends |
| `LAL_SESSION_STORE` | `file` | `postgres` selects PgSessionStore (phase 2, ADR 0010) |
| `LAL_DATABASE_URL` | none | SQLAlchemy URL for `postgres`, e.g. `postgresql+psycopg://lal:...@127.0.0.1:55432/lal` |
| `LAL_LOCK_TTL_SECONDS` | `120` | session lock expiry without a heartbeat (ADR 0012) |

Sessions always live in `<results>/sessions`: a separate sessions setting could make the session
builder (`verification.paths.session_file`) and the web store disagree.
