"""Runtime settings, read from environment variables (prefix LAL_, e.g. LAL_RESULTS_DIR).

The defaults reproduce the behaviour before the settings existed: results under <project>/results,
sessions in results/sessions, the web server on 127.0.0.1:8000, 8 point clouds cached per session.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
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
    blob_store: Literal["local"] = "local"
    job_queue: Literal["thread"] = "thread"

    @field_validator("results_dir")
    @classmethod
    def _absolute(cls, v: Path) -> Path:
        """Relative paths are relative to the project, not to the working directory."""
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @property
    def session_dir(self) -> Path:
        return self.results_dir / "sessions"


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
