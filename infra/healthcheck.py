"""Container healthchecks (docker-compose.yml), without curl in the images.

    python -m infra.healthcheck http http://127.0.0.1:8000/readyz   # exit 0 if the URL answers 2xx
    python -m infra.healthcheck worker                              # the worker's database and broker answer

A Celery worker with the solo pool answers `inspect ping` only between jobs, so a long detection
would look unhealthy; this checks what the worker needs instead (PostgreSQL, Redis, blob store).
"""
from __future__ import annotations

import sys
import urllib.request


def http(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 0 if 200 <= r.status < 300 else 1
    except Exception as exc:  # noqa: BLE001
        print(f"{url}: {exc}", file=sys.stderr)
        return 1


def worker() -> int:
    import redis
    from sqlalchemy import text

    from infra import storage
    from infra.db import make_engine
    from infra.settings import get_settings

    s = get_settings()
    try:
        engine = make_engine(s.database_url)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        engine.dispose()
        redis.Redis.from_url(s.redis_url, socket_timeout=3, socket_connect_timeout=3).ping()
        storage.check(storage.results_store())
    except Exception as exc:  # noqa: BLE001
        print(f"not healthy: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["http"] and len(argv) == 2:
        return http(argv[1])
    if argv == ["worker"]:
        return worker()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
