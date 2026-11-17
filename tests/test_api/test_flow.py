"""API v1 end to end in one process: upload, open a tab (session built by the job), review,
undo, export through the cpu queue, finish; locks across a restart; health; errors; pagination."""
import io
import time
import zipfile

import pytest

from tests.test_api.conftest import FRAMES, frames_zip

pytestmark = pytest.mark.postgres


@pytest.fixture
def anna(api):
    api.user("anna")
    return api.login("anna")


def upload(c, data=None, name="drive"):
    r = c.upload(data or frames_zip(), name)
    assert r.status_code == 201, r.text
    return r.json()


def test_health(api, caplog):
    assert api.app and TestClientGet(api, "/healthz") == (200, {"status": "ok"})
    status, body = TestClientGet(api, "/readyz")
    assert status == 200 and body["status"] == "ready"
    assert set(body["checks"]) == {"database", "redis", "results_blobs", "upload_blobs", "export_blobs"}

    def down():
        raise ConnectionError("no route to redis")
    api.env.ping = down
    with caplog.at_level("WARNING", logger="api.routes_jobs"):
        status, body = TestClientGet(api.restart(), "/readyz")
    assert status == 503 and body["checks"]["redis"] == {"ok": False}
    assert body["checks"]["database"] == {"ok": True}
    assert "no route to redis" not in str(body)                       # no exception text for the client ...
    assert "no route to redis" in caplog.text and "readiness check redis failed" in caplog.text   # ... only in the log


def TestClientGet(api, path):
    from fastapi.testclient import TestClient

    r = TestClient(api.app).get(path)
    return r.status_code, r.json()


def test_manual_session_review_undo_export_finish(api, anna):
    seq = upload(anna)
    assert seq["num_frames"] == FRAMES and seq["source"] == "upload"
    r = anna.post("/api/v1/tabs", {"sequence_id": seq["id"], "mode": "manual"})
    assert r.status_code == 201
    sid, job_id = r.json()["sid"], r.json()["job_id"]
    st = anna.get(f"/api/v1/tabs/{sid}").json()
    assert st["status"] == "loading" and st["progress"]["state"] == "queued"
    assert anna.get(f"/api/v1/tabs/{sid}/state").status_code == 409
    assert api.drain() == ["done"]
    assert anna.get(f"/api/v1/tabs/{sid}").json()["status"] == "ready"

    s = anna.get(f"/api/v1/tabs/{sid}/state?frame=0").json()
    assert s["num_frames"] == FRAMES and s["annotations"] == [] and s["mode"] == "manual"
    added = anna.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 1, "label": "Car"}).json()
    assert added["result"] == {"frame": 1, "index": 0}
    assert anna.post(f"/api/v1/tabs/{sid}/action", {"name": "undo", "frame": 1}).json()["annotations"] == []
    redo = anna.post(f"/api/v1/tabs/{sid}/action", {"name": "redo", "frame": 1}).json()
    assert len(redo["annotations"]) == 1 and redo["result"]["step"]
    pts = anna.get(f"/api/v1/tabs/{sid}/points/2")
    assert pts.status_code == 200 and len(pts.content) == 300 * 3 * 4
    assert anna.get(f"/api/v1/tabs/{sid}/points/{FRAMES}").status_code == 404
    assert anna.post(f"/api/v1/tabs/{sid}/heartbeat").json()["ok"] is True

    ex = anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "lidar"})
    assert ex.status_code == 202
    eid = ex.json()["job_id"]
    assert anna.get(f"/api/v1/jobs/{eid}/download").status_code == 409            # not run yet
    assert api.drain() == ["done"]
    job = anna.get(f"/api/v1/jobs/{eid}").json()
    assert job["state"] == "done" and job["result"]["summary"]["boxes"] == 1
    dl = anna.get(f"/api/v1/jobs/{eid}/download")
    assert dl.status_code == 200 and dl.headers["content-type"] == "application/zip"
    assert zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
    assert anna.get(f"/api/v1/jobs/{job_id}/download").status_code == 409          # not an export

    assert anna.post(f"/api/v1/tabs/{sid}/finish").json() == {"ok": True}
    assert anna.get(f"/api/v1/tabs/{sid}").json()["status"] == "finished"
    r = anna.post(f"/api/v1/tabs/{sid}/action", {"name": "accept", "frame": 1, "index": 0})
    assert r.status_code == 409 and r.json()["error"]["code"] == "finished"


def test_reopening_resumes_the_users_session(api, anna):
    seq = upload(anna)
    sid = api.open_tab(anna, seq["id"])
    anna.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 0, "label": "Car"})
    anna.post(f"/api/v1/tabs/{sid}/finish")
    again = api.open_tab(anna, seq["id"])
    assert len(anna.get(f"/api/v1/tabs/{again}/state?frame=0").json()["annotations"]) == 1


def test_assisted_session_runs_detection_in_the_job(api, anna):
    seq = upload(anna)
    r = anna.post("/api/v1/tabs", {"sequence_id": seq["id"], "mode": "assisted"})
    job = anna.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["kind"] == "open_session"
    assert api.drain() == ["done"]
    sid = r.json()["sid"]
    assert anna.get(f"/api/v1/tabs/{sid}").json()["status"] == "ready"
    s = anna.get(f"/api/v1/tabs/{sid}/state?frame=0").json()
    assert s["mode"] == "assisted" and s["stats"]["total"] == FRAMES    # the fake detector: one car per frame


def test_cancel_while_loading(api, anna):
    seq = upload(anna)
    sid = anna.post("/api/v1/tabs", {"sequence_id": seq["id"], "mode": "assisted"}).json()["sid"]
    assert anna.post(f"/api/v1/tabs/{sid}/cancel").json() == {"cancelled": True}
    api.drain()
    st = anna.get(f"/api/v1/tabs/{sid}").json()
    assert st["status"] == "error" and st["message"] == "cancelled"


def test_lock_same_owner_after_restart_other_tab_refused(api, anna):
    seq = upload(anna)
    sid = api.open_tab(anna, seq["id"])
    # the API restarts: a new process, the page (same tab, same client id) keeps working at once
    api2 = api.restart()
    same = api2.login("anna", client_id=anna.client_id)
    assert same.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 0, "label": "Car"}).status_code == 200
    # another browser tab of the same user, on the same session, is refused while the lock is held
    other = api2.login("anna", client_id="tab-bbbbbbbb")
    r = other.get(f"/api/v1/tabs/{sid}/state")
    assert r.status_code == 409 and r.json()["error"]["code"] == "locked"
    second = other.post("/api/v1/tabs", {"sequence_id": seq["id"], "mode": "manual"}).json()["sid"]
    api2.drain()
    st = other.get(f"/api/v1/tabs/{second}").json()
    assert st["status"] == "error" and "open in another tab" in st["message"]
    same.post(f"/api/v1/tabs/{sid}/finish")
    assert other.get(f"/api/v1/tabs/{second}").json()["status"] == "ready"


def test_revision_conflict_is_409_with_its_code(api, anna):
    seq = upload(anna)
    sid = api.open_tab(anna, seq["id"])
    anna.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 0, "label": "Car"})
    key = api.svc.tabs.get(sid, None)
    job = api.svc.jobs.view(key["job_id"])
    store = api.svc.sessions
    s = store.load(job["result"]["session_key"])
    real_load = store.load
    kept = {}

    def stale_load(k):                       # another writer saves between our load and our save
        loaded = real_load(k)
        other = real_load(k)
        other.reject(0, 0)
        store.save(k, other, expected_revision=loaded.revision)
        kept.update(revision=other.revision, n=len(other.annotations(0)))
        return loaded
    store.load = stale_load
    r = anna.post(f"/api/v1/tabs/{sid}/action", {"name": "forward", "frame": 0, "index": 0})
    store.load = real_load
    assert s is not None and r.status_code == 409 and r.json()["error"]["code"] == "revision_conflict"
    st = anna.get(f"/api/v1/tabs/{sid}/state?frame=0").json()
    assert st["revision"] == kept["revision"] and len(st["annotations"]) == kept["n"]   # the other write was kept


def test_points_do_not_need_the_session_document(api, anna, monkeypatch):
    seq = upload(anna)
    sid = api.open_tab(anna, seq["id"])

    def no_load(key):
        raise AssertionError("points must not load the session document")

    monkeypatch.setattr(api.svc.sessions, "load", no_load)
    r = anna.get(f"/api/v1/tabs/{sid}/points/0")
    assert r.status_code == 200 and len(r.content) == 300 * 3 * 4


def test_export_expires(api, anna, monkeypatch):
    from infra.settings import get_settings
    from infra.tasks import export_blobs

    seq = upload(anna)
    sid = api.open_tab(anna, seq["id"])
    monkeypatch.setenv("LAL_EXPORT_TTL_HOURS", "0")
    get_settings.cache_clear()
    eid = anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "lidar", "include_empty": True}).json()["job_id"]
    api.drain()
    job = anna.get(f"/api/v1/jobs/{eid}").json()
    assert job["state"] == "done", job
    assert job["result"]["expires_at"] <= time.time()
    r = anna.get(f"/api/v1/jobs/{eid}/download")
    assert r.status_code == 410 and r.json()["error"]["code"] == "gone"
    # the next export run deletes the expired zip
    anna.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 0, "label": "Car"})
    anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "lidar", "include_empty": True})
    api.drain()
    assert job["result"]["blob"] not in export_blobs().list()


def test_export_error_message_is_shown_as_it_is(api, anna):
    seq = upload(anna)
    sid = api.open_tab(anna, seq["id"])
    eid = anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "kitti"}).json()["job_id"]
    api.drain()
    job = anna.get(f"/api/v1/jobs/{eid}").json()
    assert job["state"] == "failed" and not job["message"].startswith("ExportError")
    assert anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "nope"}).status_code == 400


def test_error_model_and_pagination(api, anna):
    r = anna.get("/api/v1/nothing-here")
    assert r.status_code == 404 and set(r.json()["error"]) == {"code", "message", "details"}
    r = anna.post("/api/v1/tabs", {"sequence_id": "x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert r.json()["error"]["details"][0]["field"] == "body.sequence_id"
    for i in range(3):
        upload(anna, name=f"s{i}")
    page = anna.get("/api/v1/sequences?limit=2&offset=1").json()
    assert page["total"] == 3 and page["limit"] == 2 and page["offset"] == 1
    assert [s["name"] for s in page["items"]] == ["s1", "s0"]
    r = anna.get("/api/v1/sequences?limit=100000")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert anna.get("/api/v1/tabs/nope", headers={"X-Client-Id": "x"}).status_code == 400


def test_openapi_lists_the_v1_endpoints(api):
    from fastapi.testclient import TestClient

    spec = TestClient(api.app).get("/api/v1/openapi.json").json()
    paths = set(spec["paths"])
    assert {"/api/v1/auth/login", "/api/v1/tabs/{sid}/action", "/api/v1/jobs/{job_id}/download", "/healthz",
            "/readyz", "/api/v1/sequences/upload", "/api/v1/sequences/register"} <= paths
    assert not any(p.startswith("/api/") and not p.startswith("/api/v1/") for p in paths)
    assert "ErrorResponse" in spec["components"]["schemas"]


def test_the_export_result_records_the_revision_actually_exported(api, anna):
    seq = upload(anna)
    sid = api.open_tab(anna, seq["id"])
    anna.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 0, "label": "Car"})
    requested = anna.get(f"/api/v1/tabs/{sid}/state?frame=0").json()["revision"]
    eid = anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "lidar"}).json()["job_id"]
    anna.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 1, "label": "Car"})   # while it waits
    now = anna.get(f"/api/v1/tabs/{sid}/state?frame=0").json()["revision"]
    assert now > requested
    assert api.drain() == ["done"]
    result = anna.get(f"/api/v1/jobs/{eid}").json()["result"]
    assert result["requested_revision"] == requested and result["revision"] == now
    assert result["summary"]["boxes"] == 2                              # the exported state is that revision
