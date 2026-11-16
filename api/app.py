"""The FastAPI application: pages, /api/v1, /healthz, /readyz (docs/adr/0016-0018).

    GET  /, /login, /annotate?sid=   pages (data only through /api/v1, after login)
    GET  /static/...                 vendored three.js and font, client.js
    GET  /api/v1/docs                OpenAPI UI (schema: /api/v1/openapi.json)

The API holds no per-session state between requests: users, sequences, tabs, sessions and jobs are
rows in PostgreSQL, the queue is Redis (Celery), frames and exports are in BlobStores. Any number
of API processes can serve the same users behind a load balancer. It therefore needs
LAL_DATABASE_URL, LAL_REDIS_URL, LAL_JWT_SECRET, LAL_SESSION_STORE=postgres and LAL_JOB_QUEUE=celery
(the workers read the same settings).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import errors
from api.deps import Services
from infra.settings import Settings, get_settings

STATIC = Path(__file__).resolve().parents[1] / "web" / "static"


def check_settings(s: Settings, need_redis: bool = True) -> None:
    missing = [name for name, ok in (("LAL_DATABASE_URL", s.database_url), ("LAL_JWT_SECRET", s.jwt_secret),
                                     ("LAL_REDIS_URL", s.redis_url or not need_redis)) if not ok]
    if s.session_store != "postgres":
        missing.append("LAL_SESSION_STORE=postgres")
    if s.job_queue != "celery":
        missing.append("LAL_JOB_QUEUE=celery")
    if missing:
        raise RuntimeError("the web API needs " + ", ".join(missing) + " (docs/adr/0003)")


def make_services(settings: Settings | None = None, send: Callable[[str, str], None] | None = None,
                  ping_redis: Callable[[], None] | None = None) -> Services:
    """`send(job_id, queue)` replaces the Celery dispatch and `ping_redis` the Redis check (tests)."""
    from infra.auth import Tokens, UserStore
    from infra.celery_queue import CeleryTaskQueue
    from infra.db import make_engine
    from infra.pg_jobs import PgJobStore
    from infra.pg_sessions import PgSessionStore
    from infra.points import PointCache
    from infra.sequence_store import SequenceStore
    from infra.tabs import TabStore

    s = settings or get_settings()
    check_settings(s, need_redis=send is None)
    engine = make_engine(s.database_url)
    jobs = PgJobStore(engine, stale_seconds=s.job_stale_seconds, max_attempts=s.job_max_attempts)
    if ping_redis is None:
        def ping_redis():
            import redis

            redis.Redis.from_url(s.redis_url, socket_timeout=2, socket_connect_timeout=2).ping()
    return Services(settings=s, engine=engine, users=UserStore(engine),
                    tokens=Tokens(s.jwt_secret.get_secret_value(), s.access_token_minutes, s.refresh_token_days),
                    sessions=PgSessionStore(engine, lock_ttl=s.lock_ttl_seconds), jobs=jobs,
                    queue=CeleryTaskQueue(jobs, send=send), sequences=SequenceStore(engine), tabs=TabStore(engine),
                    ping_redis=ping_redis, point_cache=PointCache(s.point_cache_frames * 8))


def create_app(svc: Services | None = None) -> FastAPI:
    from api import routes_auth, routes_jobs, routes_sequences, routes_tabs

    app = FastAPI(title="LiDAR-AutoLabeler", version="1", openapi_url="/api/v1/openapi.json",
                  docs_url="/api/v1/docs", redoc_url=None,
                  description="Review and correct 3D boxes proposed for LiDAR sequences. Log in with "
                              "POST /api/v1/auth/login; the cookies it sets authenticate the other calls. "
                              "POST requests need the header X-CSRF-Token (the value of the lal_csrf cookie). "
                              "Errors: {\"error\": {\"code\", \"message\", \"details\"}}. Lists: "
                              "{\"items\", \"total\", \"limit\", \"offset\"}.")
    app.state.services = svc or make_services()
    errors.install(app)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    for r in (routes_auth.router, routes_sequences.router, routes_tabs.router, routes_jobs.router,
              routes_jobs.health):
        app.include_router(r)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC / "home.html")

    @app.get("/login", include_in_schema=False)
    def login_page():
        return FileResponse(STATIC / "login.html")

    @app.get("/annotate", include_in_schema=False)
    def annotate():
        return FileResponse(STATIC / "annotate.html")

    return app


def app_factory() -> FastAPI:
    """For `uvicorn --factory api.app:app_factory`."""
    return create_app()
