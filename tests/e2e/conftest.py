"""Fixtures for the browser tests: the whole stack in processes (API, a Celery worker with fake
detection, PostgreSQL and Redis of docker-compose.db.yml), users, a synthetic sequence, logging in,
and a check that fails the test on any browser console error.

    docker compose -f docker-compose.db.yml up -d
    pytest -m e2e
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_FRAMES = 6
PASSWORD = "e2e password 1234"
REDIS_URL = os.environ.get("LAL_TEST_E2E_REDIS_URL", "redis://127.0.0.1:56379/6")
SECRET = "e2e-secret-0123456789abcdef0123456789abcdef"


def write_sequence(folder: Path) -> Path:
    """NUM_FRAMES small random point clouds (x, y, z, intensity as float32)."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(NUM_FRAMES):
        rng.uniform(-30, 30, (2000, 4)).astype("<f4").tofile(folder / f"{i:06d}.bin")
    return folder


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proc:
    """A child process with its output in a log file."""

    def __init__(self, args: list[str], env: dict, log: Path) -> None:
        self.args, self.env, self.log = args, env, log
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        with self.log.open("ab") as out:
            self.proc = subprocess.Popen([sys.executable, *self.args], cwd=PROJECT_ROOT, env=self.env, stdout=out,
                                         stderr=subprocess.STDOUT)

    def wait_http(self, url: str, timeout: float = 40) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=1)
                return
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.args} exited:\n{self.log.read_text(errors='replace')[-3000:]}") from None
                time.sleep(0.2)
        raise RuntimeError(f"{self.args} did not start")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)


class AppServer(Proc):
    """One API process (tests/e2e/app_server.py); `restart` starts a new process on the same port."""

    def __init__(self, env: dict, log: Path) -> None:
        self.port = _free_port()
        super().__init__(["-m", "tests.e2e.app_server", str(self.port)], env, log)
        self.url = f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        super().start()
        self.wait_http(self.url + "/healthz")

    def restart(self) -> None:
        self.stop()
        self.start()


class Stack:
    def __init__(self, tmp: Path, pg_url: str, pg_engine, apis: int = 1, proxy: bool = False,
                 env: dict | None = None) -> None:
        self.tmp, self.pg_engine = tmp, pg_engine
        self.results_dir = tmp / "results"
        self.data_root = tmp / "data"
        self.env = {**os.environ, "LAL_DATABASE_URL": pg_url, "LAL_REDIS_URL": REDIS_URL, "LAL_JWT_SECRET": SECRET,
                    "LAL_SESSION_STORE": "postgres", "LAL_JOB_QUEUE": "celery",
                    "LAL_RESULTS_DIR": str(self.results_dir), "LAL_DATA_ROOT": str(self.data_root),
                    "LAL_COOKIE_SECURE": "false", "LAL_WORKER_DETECTOR_FACTORY": "tests.fakes:FakeDetector",
                    "LAL_JOB_HEARTBEAT_SECONDS": "0.3", "PYTHONUNBUFFERED": "1", **(env or {})}
        self.worker = Proc(["-m", "tests.e2e.worker", "w1"], self.env, tmp / "worker.log")
        self.apis = [AppServer(self.env, tmp / f"api{i}.log") for i in range(apis)]
        self.proxy: Proc | None = None
        self.proxy_url = ""
        if proxy:
            port = _free_port()
            self.proxy = Proc(["-m", "tests.e2e.rr_proxy", str(port), *[a.url for a in self.apis]], self.env,
                              tmp / "proxy.log")
            self.proxy_url = f"http://127.0.0.1:{port}"

    @property
    def url(self) -> str:
        return self.proxy_url or self.apis[0].url

    def start(self) -> Stack:
        import redis

        redis.Redis.from_url(REDIS_URL).flushdb()
        self.worker.start()
        for a in self.apis:
            a.start()
        if self.proxy is not None:
            self.proxy.start()
            self.proxy.wait_http(self.proxy_url + "/healthz")
        return self

    def stop(self) -> None:
        for p in [self.proxy, *self.apis, self.worker]:
            if p is not None:
                p.stop()

    def proxy_hits(self) -> list[int]:
        import json

        return json.loads(urllib.request.urlopen(self.proxy_url + "/__proxy_stats").read())["hits"]

    # ------------------------------------------------------------- data
    def user(self, name: str, role: str = "annotator") -> int:
        from infra.auth import UserStore

        return UserStore(self.pg_engine).create(name, PASSWORD, role).id

    def sequence(self, owner: int, name: str = "drive") -> int:
        from infra.sequence_store import SequenceStore

        folder = write_sequence(self.data_root / f"{name}_{owner}")
        return SequenceStore(self.pg_engine).add(name, "folder", str(folder), {"folder": str(folder)}, NUM_FRAMES,
                                                 owner)["id"]


@pytest.fixture
def make_stack(tmp_path, pg_url, pg_engine):
    stacks: list[Stack] = []

    def make(**kw) -> Stack:
        s = Stack(tmp_path, pg_url, pg_engine, **kw).start()
        stacks.append(s)
        return s

    yield make
    for s in stacks:
        s.stop()


@pytest.fixture
def stack(make_stack):
    """The stack with user anna (annotator), who owns the sequence "drive"."""
    s = make_stack()
    s.anna = s.user("anna")
    s.drive = s.sequence(s.anna)
    return s


def login(page, stack: Stack, username: str = "anna") -> None:
    page.goto(stack.url + "/login")
    page.fill("#username", username)
    page.fill("#password", PASSWORD)
    page.click("#login")
    page.wait_for_url(stack.url + "/")


class ConsoleErrors:
    """Browser console errors and uncaught exceptions of every page of the test. A test that
    provokes an error on purpose (e.g. an HTTP 409, which Chromium logs as a failed resource)
    declares it with `expect(substring, count)`; anything else fails the test."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.expected: dict[str, int] = {}

    def watch(self, page) -> None:
        page.on("console", lambda m: self.errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))

    def expect(self, substring: str, count: int = 1) -> None:
        self.expected[substring] = self.expected.get(substring, 0) + count

    def unexpected(self) -> list[str]:
        """Errors nobody expected, plus a line for every expected error that did not happen
        (which also proves that the console is really being watched)."""
        left, out = dict(self.expected), []
        for e in self.errors:
            hit = next((k for k, n in left.items() if n > 0 and k in e), None)
            if hit is None:
                out.append(e)
            else:
                left[hit] -= 1
        return out + [f"expected console error containing {k!r} did not happen" for k, n in left.items() if n > 0]

    def guard(self, context) -> None:
        """Watch every page of a browser context and block requests outside 127.0.0.1."""
        context.on("page", self.watch)

        def offline(route):                      # the page must not need the internet (vendored assets)
            if route.request.url.startswith("http://127.0.0.1:"):
                route.continue_()
            else:
                self.errors.append(f"external request: {route.request.url}")
                route.abort()

        context.route("**/*", offline)


@pytest.fixture
def console_errors(page, context):
    errors = ConsoleErrors()
    errors.watch(page)
    errors.guard(context)
    yield errors
    bad = errors.unexpected()
    assert bad == [], "browser console errors:\n" + "\n".join(bad)
