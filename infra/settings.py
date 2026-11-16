"""Runtime settings, read from environment variables (prefix LAL_, e.g. LAL_RESULTS_DIR).

The defaults reproduce the behaviour before the settings existed: results under <project>/results,
sessions in results/sessions, the web server on 127.0.0.1:8000, 8 point clouds cached per session.
The web API (api/app.py) additionally needs LAL_DATABASE_URL, LAL_REDIS_URL, LAL_JWT_SECRET,
LAL_SESSION_STORE=postgres and LAL_JOB_QUEUE=celery (docs/adr/0003, 0016-0018).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAL_", extra="ignore")

    results_dir: Path = PROJECT_ROOT / "results"   # sessions in <results_dir>/sessions
    host: str = "127.0.0.1"
    port: int = 8000
    point_cache_frames: int = 8               # point clouds kept in memory per open session
    session_store: Literal["file", "postgres"] = "file"
    database_url: str | None = None           # e.g. postgresql+psycopg://lal:<password>@localhost:55432/lal
    lock_ttl_seconds: float = 120.0           # a session lock expires without a heartbeat for this long
    # blob areas uploads / cache / exports (infra/storage.py, docs/adr/0020): local folders or one S3 bucket
    blob_store: Literal["local", "s3"] = "local"
    s3_endpoint_url: str | None = None        # e.g. http://minio:9000 (None: AWS)
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: SecretStr | None = None
    s3_region: str = "us-east-1"
    # background jobs (detection + tracking on the gpu queue, export on the cpu queue), docs/adr/0015
    job_queue: Literal["thread", "celery"] = "thread"   # celery needs LAL_REDIS_URL and LAL_DATABASE_URL
    redis_url: str | None = None              # e.g. redis://127.0.0.1:56379/0
    celery_visibility_timeout: float = 3600.0  # seconds before Redis hands an unacknowledged job to another worker
    job_heartbeat_seconds: float = 2.0        # progress / heartbeat written at most this often
    job_stale_seconds: float = 60.0           # a running job without a heartbeat this long is considered lost
    job_max_attempts: int = 3                 # runs of a job before a lost worker makes it fail
    # the one detector configuration a worker process loads (once)
    worker_detector_factory: str = "detection.detector:Detector"
    worker_detector_config: Path | None = None       # default: detection.paths.DEFAULT_CFG
    worker_detector_checkpoint: Path | None = None   # default: detection.paths.DEFAULT_CKPT
    worker_score_threshold: float = 0.5
    # API v1 (docs/adr/0016-0018)
    jwt_secret: SecretStr | None = None       # required by the web API; the same value on every replica
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    cookie_secure: bool = True                # set LAL_COOKIE_SECURE=false only for plain-http local work
    data_root: Path | None = None             # admins may register folders below it
    upload_root: Path | None = None           # default: <results_dir>/uploads
    upload_max_bytes: int = 2 * 1024**3       # zip upload limit (compressed); 2 GiB
    upload_max_unpacked_bytes: int = 8 * 1024**3
    export_ttl_hours: float = 24.0            # stored export zips are deleted after this
    export_prune_minutes: float = 15.0        # the scheduler (celery beat) deletes expired exports this often
    page_max: int = 200                       # largest page of a list endpoint
    assisted_mode: bool = True                # false without a GPU worker (cpu-only): assisted openings are refused

    @field_validator("results_dir")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        """Relative paths are relative to the project, not to the working directory."""
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @property
    def session_dir(self) -> Path:
        return self.results_dir / "sessions"

    @property
    def uploads_dir(self) -> Path:
        return self.upload_root or self.results_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def make_session_store(settings: Settings | None = None):
    """The SessionStore chosen by LAL_SESSION_STORE: "file" (default) or "postgres" (needs LAL_DATABASE_URL)."""
    settings = settings or get_settings()
    if settings.session_store == "postgres":
        if not settings.database_url:
            raise ValueError("LAL_SESSION_STORE=postgres needs LAL_DATABASE_URL")
        from infra.db import make_engine
        from infra.pg_sessions import PgSessionStore

        return PgSessionStore(make_engine(settings.database_url), lock_ttl=settings.lock_ttl_seconds)
    from infra.sessions import FileSessionStore

    return FileSessionStore(settings.session_dir, lock_ttl=settings.lock_ttl_seconds)


def make_task_queue(settings: Settings | None = None):
    """The TaskQueue chosen by LAL_JOB_QUEUE: "thread" (default, in this process) or "celery"
    (worker processes; needs LAL_REDIS_URL and LAL_DATABASE_URL)."""
    settings = settings or get_settings()
    if settings.job_queue == "celery":
        if not (settings.redis_url and settings.database_url):
            raise ValueError("LAL_JOB_QUEUE=celery needs LAL_REDIS_URL and LAL_DATABASE_URL")
        from infra.celery_queue import CeleryTaskQueue
        from infra.db import make_engine
        from infra.pg_jobs import PgJobStore

        return CeleryTaskQueue(PgJobStore(make_engine(settings.database_url), stale_seconds=settings.job_stale_seconds,
                                          max_attempts=settings.job_max_attempts))
    from infra.jobs import ThreadTaskQueue

    return ThreadTaskQueue()
