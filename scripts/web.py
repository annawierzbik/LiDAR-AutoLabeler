"""Run LiDAR AutoLabeler in the browser.

    python -m scripts.web                      # then open http://127.0.0.1:8000
    python -m scripts.web --host 0.0.0.0       # reachable from other computers (server)

WARNING: the server has no authentication. Anyone who can reach it can open ANY folder on the
server's disk by typing its path, read its point clouds and write results. Only use
--host 0.0.0.0 (or any non-local address) on a network you trust.

The start screen annotates a folder of .bin frames; sessions are autosaved to results/sessions/.
Run it from the project root so data/ and results/ are found.
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")

NETWORK_WARNING = (
    "WARNING: listening on {host} - this server has NO authentication and opens any folder\n"
    "         path on this computer's disk for anyone who can reach it. Use only on a trusted network."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    import uvicorn

    from web.manager import SessionManager
    from web.server import create_app

    manager = SessionManager()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    if args.host not in LOCAL_HOSTS:
        print(NETWORK_WARNING.format(host=args.host))
    print(f"Open {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        uvicorn.run(create_app(manager), host=args.host, port=args.port, log_level="warning")
    finally:
        for job in manager.jobs.values():
            if job.service is not None:
                job.service.save()


if __name__ == "__main__":
    main()
