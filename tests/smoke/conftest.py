"""The smoke test runs against a deployed stack (docker-compose.yml + tests/smoke/docker-compose.smoke.yml)
that scripts/smoke_stack.py brings up; it is never started by plain pytest (marker `smoke`).

Environment set by that script:
    LAL_SMOKE_URL             https://localhost:<port> (nginx, self-signed certificate)
    LAL_SMOKE_HTTP_URL        http://localhost:<port>  (must redirect to HTTPS)
    LAL_SMOKE_ADMIN_PASSWORD  password of the user "admin" it created
    LAL_SMOKE_COMPOSE         JSON list: the docker compose command of the stack (project, files, env file)
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set: run python scripts/smoke_stack.py")
    return value


@pytest.fixture(scope="session")
def url() -> str:
    return env("LAL_SMOKE_URL").rstrip("/")


@pytest.fixture(scope="session")
def compose():
    base = json.loads(env("LAL_SMOKE_COMPOSE"))

    def run(*args: str, input: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run([*base, *args], input=input, capture_output=True, text=True, check=check, timeout=600)

    return run


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """The certificate is self-signed (scripts/make_dev_cert.py)."""
    return {**browser_context_args, "ignore_https_errors": True}


@pytest.fixture
def console_errors(page, context, url):
    """Like tests/e2e: console errors (also CSP violations) fail the test, and the page may not
    load anything from another origin."""
    errors: list[str] = []

    def watch(p):
        p.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        p.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    watch(page)
    context.on("page", watch)

    def same_origin(route):
        if route.request.url.startswith(url + "/"):
            route.continue_()
        else:
            errors.append(f"external request: {route.request.url}")
            route.abort()

    context.route("**/*", same_origin)
    yield errors
    assert errors == [], "browser console errors:\n" + "\n".join(errors)
