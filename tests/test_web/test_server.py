import pytest
from fastapi.testclient import TestClient

from tests.test_web.test_manager import Builder, make_folder
from web.manager import SessionManager
from web.server import create_app


@pytest.fixture
def env(tmp_path):
    mgr = SessionManager(builder=Builder(tmp_path), background=False)
    return TestClient(create_app(mgr)), mgr, tmp_path


@pytest.fixture
def sid(env):
    client, _, tmp = env
    r = client.post("/api/folder/open", json={"path": str(make_folder(tmp))})
    assert r.status_code == 200
    return r.json()["sid"]


def test_pages_are_served(env):
    client = env[0]
    assert "Annotate a point cloud sequence" in client.get("/").text
    assert "annotate" in client.get("/annotate").text.lower()


def test_open_folder_errors_are_400_with_a_message(env):
    client = env[0]
    r = client.post("/api/folder/open", json={"path": "/definitely/not/here"})
    assert r.status_code == 400 and "not a folder" in r.json()["detail"]


def test_job_state_and_unknown_ids(env, sid):
    client = env[0]
    assert client.get(f"/api/jobs/{sid}").json()["status"] == "ready"
    assert client.get("/api/jobs/zzz").status_code == 404
    assert client.get("/api/s/zzz/state").status_code == 404


def test_state_points_and_actions(env, sid):
    client = env[0]
    assert client.get(f"/api/s/{sid}/state").json()["num_frames"] == 2
    r = client.get(f"/api/s/{sid}/points/0")
    assert r.status_code == 200 and len(r.content) == 50 * 12
    assert client.get(f"/api/s/{sid}/points/9").status_code == 404
    post = lambda **kw: client.post(f"/api/s/{sid}/action", json=kw)
    assert post(name="select", index=0).json()["selected"] == 0
    assert post(name="accept").json()["stats"]["accepted"] == 1
    assert post(name="goto", frame=1).json()["frame"] == 1
    assert post(name="select").json()["selected"] is None
    assert post(name="bogus").status_code == 400
    assert post(name="set_class", label="X").status_code == 400


def test_unknown_action_and_missing_field_have_different_messages(env, sid):
    client = env[0]
    post = lambda **kw: client.post(f"/api/s/{sid}/action", json=kw)
    r = post(name="bogus")
    assert r.status_code == 400 and r.json()["detail"] == "unknown action 'bogus'"
    r = post(name="goto")
    assert r.status_code == 400 and "frame" in r.json()["detail"] and "unknown" not in r.json()["detail"]
    r = post(name="set_class")
    assert r.status_code == 400 and "label" in r.json()["detail"]


def test_finish_removes_the_job_and_stops_actions(env, sid):
    client = env[0]
    assert client.post(f"/api/s/{sid}/finish").status_code == 200
    assert client.get(f"/api/jobs/{sid}").status_code == 404
    assert client.post(f"/api/s/{sid}/action", json={"name": "add"}).status_code == 404


def test_other_tab_on_a_finished_session_gets_409(env, sid):
    client, _, tmp = env
    other = client.post("/api/folder/open", json={"path": str(tmp / "clouds")}).json()["sid"]
    client.post(f"/api/s/{sid}/finish")
    assert client.post(f"/api/s/{other}/action", json={"name": "add"}).status_code == 409
    assert client.get(f"/api/s/{other}/state").status_code == 409
