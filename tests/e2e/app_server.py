"""The real web API (api.app) for the end-to-end tests, configured by LAL_* environment variables.

    python -m tests.e2e.app_server <port>

Detection runs in the worker processes (tests/e2e/worker.py), not here.
"""
from __future__ import annotations

import sys

import uvicorn


def main() -> None:
    from api.app import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")


if __name__ == "__main__":
    main()
