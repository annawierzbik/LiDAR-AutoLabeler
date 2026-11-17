"""The deployed stack through nginx at https://localhost (marker `smoke`, run by scripts/smoke_stack.py).

Runs in file order on one stack: gateway (TLS, redirect, headers, body limits), the browser scenario
of two users (upload, assisted opening with the fake detector on the gpu queue, review, undo / redo,
export through the cpu queue, finish, the second user sees nothing of the first), scaling worker-cpu
to 3, and last the login rate limit (it blocks logins from this address for a minute).
"""
from __future__ import annotations

import os
import re
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.smoke

FRAMES = 6
PASSWORD = "smoke test password 1"
STATE: dict = {}                                   # shared by the tests of this file, in order


def client(url: str) -> httpx.Client:
    return httpx.Client(base_url=url, verify=False, timeout=30, follow_redirects=False)


def api_login(url: str, username: str, password: str) -> httpx.Client:
    c = client(url)
    r = c.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    c.headers["X-CSRF-Token"] = c.cookies["lal_csrf"]
    c.headers["X-Client-Id"] = "smoke-api-client"
    return c


def wait_until(check, timeout=30.0, step=0.2):
    """Poll `check()` until it returns without AssertionError / KeyError / IndexError."""
    deadline = time.time() + timeout
    while True:
        try:
            return check()
        except (AssertionError, KeyError, IndexError):
            if time.time() > deadline:
                raise
            time.sleep(step)


def until(predicate, timeout=30.0, step=0.2, what=""):
    """Poll `predicate()` until it is true."""
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError(f"timed out waiting for {what or predicate}")
        time.sleep(step)


# --------------------------------------------------------------------- gateway
def test_tls_redirect_health_and_security_headers(url):
    http_url = os.environ["LAL_SMOKE_HTTP_URL"].rstrip("/")
    r = httpx.get(http_url + "/annotate?sid=x", follow_redirects=False, timeout=10)
    assert r.status_code == 301 and r.headers["location"] == url + "/annotate?sid=x"
    with client(url) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        ready = c.get("/readyz").json()
        assert ready["status"] == "ready" and set(ready["checks"]) == {
            "database", "redis", "results_blobs", "upload_blobs", "export_blobs"}, ready
        page = c.get("/login")
        h = page.headers
        assert h["strict-transport-security"] == "max-age=0"              # LAL_HSTS_MAX_AGE=0 on localhost
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "same-origin" and "nginx/" not in h.get("server", "")   # no version
        assert re.search(r"script-src 'self' 'sha256-[A-Za-z0-9+/=]+'", h["content-security-policy"])
        assert c.get("/api/v1/openapi.json").json()["info"]["title"] == "LiDAR-AutoLabeler"


def test_body_limits_of_the_gateway(url):
    limit = int(os.environ.get("LAL_SMOKE_UPLOAD_MAX_BYTES", "1000000"))
    with client(url) as c:
        r = c.post("/api/v1/sequences/upload?name=x", content=b"\0" * (limit + 1),
                   headers={"Content-Type": "application/zip"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        assert f"larger than {limit} bytes" in r.json()["error"]["message"]
        r = c.post("/api/v1/tabs", content=b"{" + b" " * (2 * 1024 * 1024) + b"}",
                   headers={"Content-Type": "application/json"})
        assert r.status_code == 413 and "1 MiB" in r.json()["error"]["message"]   # outside the upload: 1 MiB


# ---------------------------------------------------------------- the browser
def ui_login(page, url, username, password=PASSWORD):
    page.goto(url + "/login")
    page.fill("#username", username)
    page.fill("#password", password)
    page.click("#login")
    page.wait_for_url(url + "/")


def test_two_users_upload_review_export_and_finish(page, browser, url, console_errors, tmp_path):
    admin = api_login(url, "admin", os.environ["LAL_SMOKE_ADMIN_PASSWORD"])
    for name in ("anna", "bob"):
        r = admin.post("/api/v1/users", json={"username": name, "password": PASSWORD, "role": "annotator"})
        assert r.status_code == 201, r.text

    archive = tmp_path / "street.zip"
    real = os.environ.get("LAL_SMOKE_FRAMES_DIR")                     # --gpu: real KITTI frames for the model
    with zipfile.ZipFile(archive, "w") as z:
        if real:
            for f in sorted(Path(real).glob("*.bin"))[:FRAMES]:
                z.write(f, f"street/{f.name}")
        else:
            rng = np.random.default_rng(5)
            for i in range(FRAMES):
                z.writestr(f"street/scan_{i}.bin", rng.uniform(-30, 30, (800, 4)).astype("<f4").tobytes())
        first = min(z.namelist()).split("/")[-1][:-4]
    ui_login(page, url, "anna")
    page.set_input_files("#upFile", str(archive))
    page.click("#upload")
    expect(page.locator("#upMsg")).to_have_text(f"Uploaded street: {FRAMES} frames.", timeout=30000)
    page.select_option("#mode", "assisted")
    page.get_by_role("button", name="Open street").click()            # detection runs on the gpu queue
    page.wait_for_url(re.compile(r"/annotate\?sid="), timeout=60000)
    expect(page.locator("#sub")).to_have_text(re.compile(rf"^Frame 1 of {FRAMES},"), timeout=30000)
    sid = page.url.split("sid=")[1]
    cid = page.evaluate("() => sessionStorage.getItem('lal:client')")

    def statuses(frame=0):
        r = page.request.get(f"{url}/api/v1/tabs/{sid}/state?frame={frame}", headers={"X-Client-Id": cid})
        assert r.ok, r.text()
        return [a["status"] for a in r.json()["annotations"]]

    assert statuses() and set(statuses()) == {"pending"}                 # proposals of the fake detector
    n = len(statuses())
    page.keyboard.press("Tab")
    page.keyboard.press("a")
    until(lambda: statuses()[0] == "accepted", what="accept")
    page.keyboard.press("b")                                             # add a box ...
    until(lambda: len(statuses()) == n + 1, what="add")
    page.keyboard.press("z")                                             # ... undo, redo
    until(lambda: len(statuses()) == n, what="undo")
    page.keyboard.press("y")
    until(lambda: len(statuses()) == n + 1, what="redo")

    page.click("#exp summary")
    page.check("#expUnrev")
    page.click("#expRun")
    expect(page.locator("#expInfo")).to_contain_text(f"{FRAMES} frames with", timeout=30000)
    with page.expect_download() as dl:
        page.click("#expGo")
    names = zipfile.ZipFile(dl.value.path()).namelist()
    assert any(n.endswith(f"labels/{first}.txt") for n in names), names   # the names of the archive

    seq_id = page.request.get(f"{url}/api/v1/sequences").json()["items"][0]["id"]
    page.once("dialog", lambda d: d.accept())
    page.click("#finish")
    page.wait_for_url(url + "/")
    STATE.update(anna_sequence=seq_id, anna_tab=sid)

    bob = api_login(url, "bob", PASSWORD)                                # the second user sees nothing of anna's
    assert bob.get("/api/v1/sequences").json()["total"] == 0
    assert bob.get("/api/v1/sessions").json()["total"] == 0
    assert bob.get(f"/api/v1/sequences/{seq_id}").status_code == 404
    assert bob.get(f"/api/v1/tabs/{sid}").status_code == 404
    other = browser.new_context(ignore_https_errors=True)
    try:
        p2 = other.new_page()
        ui_login(p2, url, "bob")
        expect(p2.locator("#noSeqs")).to_be_visible()
    finally:
        other.close()


# --------------------------------------------------------------------- scaling
def test_worker_cpu_scales_to_3_and_exports_still_run(url, compose):
    compose("up", "-d", "--no-recreate", "--scale", "worker-cpu=3", "worker-cpu")

    def three_healthy():
        out = compose("ps", "worker-cpu", "--format", "{{.Name}} {{.Health}}").stdout.split("\n")
        rows = [r for r in out if r.strip()]
        assert len(rows) == 3 and all(r.endswith(" healthy") for r in rows), rows
        return rows

    rows = wait_until(three_healthy, timeout=180, step=3)
    replies = compose("exec", "-T", "worker-cpu", "celery", "-A", "infra.celery_app", "inspect", "ping", "--timeout", "5",
                      check=False).stdout
    cpu = sorted(set(re.findall(r"(cpu@[0-9a-f]+): OK", replies)))
    assert len(cpu) == 3, replies                                          # three cpu workers on the broker
    anna = api_login(url, "anna", PASSWORD)
    seq = STATE["anna_sequence"]
    sid = anna.post("/api/v1/tabs", json={"sequence_id": seq, "mode": "manual"}).json()["sid"]
    until(lambda: anna.get(f"/api/v1/tabs/{sid}").json()["status"] == "ready", timeout=60, what="manual opening")
    jobs = [anna.post(f"/api/v1/tabs/{sid}/exports", json={"format": "lidar", "include_empty": e,
                                                             "include_unreviewed": u}).json()["job_id"]
            for e in (False, True) for u in (False, True)]
    for job in jobs:
        until(lambda job=job: anna.get(f"/api/v1/jobs/{job}").json()["state"] == "done", timeout=60, what=job)
    print(f"worker-cpu replicas: {rows}; workers that answered ping: {cpu}")


# ----------------------------------------------------------------- rate limit (last)
def test_login_attempts_are_rate_limited_by_the_gateway(url):
    codes = []
    with client(url) as c:
        for _ in range(12):
            r = c.post("/api/v1/auth/login", json={"username": "anna", "password": "wrong password"})
            codes.append(r.status_code)
            if r.status_code == 429:
                assert r.json()["error"]["code"] == "rate_limited" and r.headers.get("retry-after") == "60"
                break
    assert codes[0] == 401 and codes[-1] == 429, codes
    print(f"login answers until the limit: {codes}")

