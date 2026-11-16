"""Fixtures for the browser tests: a synthetic sequence, the app server in a subprocess, and a
check that fails the test on any browser console error."""
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


@pytest.fixture
def sequence_dir(tmp_path) -> Path:
    """A folder with NUM_FRAMES small random point clouds (x, y, z, intensity as float32)."""
    folder = tmp_path / "drive"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(NUM_FRAMES):
        rng.uniform(-30, 30, (2000, 4)).astype("<f4").tofile(folder / f"{i:06d}.bin")
    return folder


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AppServer:
    """The web app (tests/e2e/app_server.py) in its own process; `restart` keeps the results folder."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.proc: subprocess.Popen | None = None
        self.url = ""

    def start(self) -> None:
        port = _free_port()
        env = {**os.environ, "LAL_RESULTS_DIR": str(self.results_dir)}
        self.proc = subprocess.Popen([sys.executable, "-m", "tests.e2e.app_server", str(port)],
                                     cwd=PROJECT_ROOT, env=env)
        self.url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(self.url + "/", timeout=1)
                return
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError("the app server exited") from None
                time.sleep(0.2)
        raise RuntimeError("the app server did not start")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(10)

    def restart(self) -> None:
        self.stop()
        self.start()


@pytest.fixture
def server(tmp_path):
    srv = AppServer(tmp_path / "results")
    srv.start()
    yield srv
    srv.stop()


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


@pytest.fixture
def console_errors(page, context):
    errors = ConsoleErrors()
    errors.watch(page)
    context.on("page", errors.watch)            # pages opened later (a second tab)
    yield errors
    bad = errors.unexpected()
    assert bad == [], "browser console errors:\n" + "\n".join(bad)
