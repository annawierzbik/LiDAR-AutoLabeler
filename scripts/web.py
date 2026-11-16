"""Run the LiDAR AutoLabeler web API and pages (one API process).

    python -m scripts.web                      # then open http://127.0.0.1:8000 and log in
    python -m scripts.web --host 0.0.0.0       # reachable from other computers

Needs PostgreSQL, Redis and at least one worker of each queue (docs/adr/0003, 0015):
    docker compose -f docker-compose.db.yml up -d
    python -m infra.migrate
    python -m scripts.create_user admin --role admin
    celery -A infra.celery_app worker -Q gpu -P solo -n gpu@%h
    celery -A infra.celery_app worker -Q cpu -P solo -n cpu@%h
and the settings LAL_DATABASE_URL, LAL_REDIS_URL, LAL_JWT_SECRET, LAL_SESSION_STORE=postgres,
LAL_JOB_QUEUE=celery (the workers need the same values). The login cookies are marked Secure, so
over plain http (local work without TLS) set LAL_COOKIE_SECURE=false. Several API processes can
run behind a load balancer: they keep no state between requests.
"""
from __future__ import annotations

import argparse
import webbrowser


def main() -> None:
    from infra.settings import get_settings

    settings = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=settings.host, help="default: LAL_HOST or 127.0.0.1")
    p.add_argument("--port", type=int, default=settings.port, help="default: LAL_PORT or 8000")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    import uvicorn

    from api.app import create_app

    app = create_app()                       # fails here with a clear message when a setting is missing
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"Open {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
