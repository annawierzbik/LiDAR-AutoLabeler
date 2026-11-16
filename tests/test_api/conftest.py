"""Fixtures for the API v1 tests (marker `postgres`): the real app on a throw-away database, with
jobs run in this process by the real worker code (infra.celery_app.run_job) instead of being sent
to Redis, and the fake detector. `Api.drain()` runs the queued jobs."""
from __future__ import annotations

import io
import zipfile
from argparse import Namespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import pipeline.session_builder
import tests.fakes
import verification.paths
from infra import worker_detector
from infra.settings import get_settings
from verification import cache

SECRET = "test-secret-0123456789abcdef0123456789"
PASSWORD = "correct horse battery"
FRAMES = 5


def write_frames(folder, n=FRAMES, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        rng.uniform(-20, 20, (300, 4)).astype("<f4").tofile(folder / f"{i:06d}.bin")
    return folder


def zip_of(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


def frames_zip(n=FRAMES, folder="drive/") -> bytes:
    rng = np.random.default_rng(1)
    return zip_of({f"{folder}{i:06d}.bin": rng.uniform(-20, 20, (300, 4)).astype("<f4").tobytes() for i in range(n)})


class Client:
    """A logged-in browser: cookies, the CSRF header and one client id (a browser tab)."""

    def __init__(self, app, username: str, password: str = PASSWORD, client_id: str = "tab-aaaaaaaa") -> None:
        self.http = TestClient(app)
        self.client_id = client_id
        r = self.http.post("/api/v1/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        self.user = r.json()

    def headers(self, extra=None) -> dict:
        return {"X-CSRF-Token": self.http.cookies.get("lal_csrf", ""), "X-Client-Id": self.client_id, **(extra or {})}

    def get(self, path, **kw):
        return self.http.get(path, headers=self.headers(kw.pop("headers", None)), **kw)

    def post(self, path, json=None, **kw):
        return self.http.post(path, json=json, headers=self.headers(kw.pop("headers", None)), **kw)

    def delete(self, path, **kw):
        return self.http.delete(path, headers=self.headers(kw.pop("headers", None)), **kw)

    def upload(self, data: bytes, name="drive"):
        return self.post(f"/api/v1/sequences/upload?name={name}", content=data,
                         headers={"Content-Type": "application/zip"})


class Api:
    def __init__(self, env, svc, app, sent) -> None:
        self.env, self.svc, self.app, self.sent = env, svc, app, sent

    def drain(self) -> list[str]:
        """Run every queued job (as a worker would); returns the final states."""
        from infra.celery_app import run_job

        states = []
        while self.sent:
            states.append(run_job(self.sent.pop(0), "test-worker", store=self.svc.jobs))
        return states

    def user(self, name: str, role: str = "annotator"):
        return self.svc.users.create(name, PASSWORD, role)

    def login(self, name: str, client_id: str = "tab-aaaaaaaa") -> Client:
        return Client(self.app, name, client_id=client_id)

    def restart(self) -> Api:
        """A new app (new process state: services, point cache) on the same database and files."""
        return make_api(self.env)

    def open_tab(self, c: Client, seq_id: int, mode="manual", **kw) -> str:
        r = c.post("/api/v1/tabs", {"sequence_id": seq_id, "mode": mode, **kw})
        assert r.status_code == 201, r.text
        self.drain()
        st = c.get(f"/api/v1/tabs/{r.json()['sid']}").json()
        assert st["status"] == "ready", st
        return r.json()["sid"]


def make_api(env) -> Api:
    from api.app import create_app, make_services

    sent: list[str] = []
    svc = make_services(get_settings(), send=lambda job_id, queue: sent.append(job_id), ping_redis=env.ping)
    return Api(env, svc, create_app(svc), sent)


@pytest.fixture
def api(tmp_path, monkeypatch, pg_url, pg_engine):
    results = tmp_path / "results"
    data_root = write_frames(tmp_path / "data" / "drive_a").parent
    for k, v in {"LAL_DATABASE_URL": pg_url, "LAL_JWT_SECRET": SECRET, "LAL_SESSION_STORE": "postgres",
                 "LAL_JOB_QUEUE": "celery", "LAL_REDIS_URL": "redis://unused", "LAL_RESULTS_DIR": str(results),
                 "LAL_DATA_ROOT": str(data_root), "LAL_COOKIE_SECURE": "false", "LAL_JOB_HEARTBEAT_SECONDS": "0.2",
                 "LAL_WORKER_DETECTOR_FACTORY": "tests.fakes:FakeDetector"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(verification.paths, "RESULTS_DIR", results)
    monkeypatch.setattr(pipeline.session_builder, "RESULTS_DIR", results)
    monkeypatch.setattr(cache, "_warned", set())
    get_settings.cache_clear()
    worker_detector.reset()
    tests.fakes.LOADS = 0
    env = Namespace(tmp=tmp_path, results=results, data_root=data_root, ping=lambda: None)
    a = make_api(env)
    yield a
    a.svc.engine.dispose()
    worker_detector.reset()
    get_settings.cache_clear()
