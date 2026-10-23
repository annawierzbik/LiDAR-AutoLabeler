import io
import zipfile

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
    return client.post("/api/folder/open", json={"path": str(make_folder(tmp))}).json()["sid"]


def test_export_preview_and_download(env, sid):
    client = env[0]
    preview = lambda **kw: client.get(f"/api/s/{sid}/export/preview", params=kw).json()
    first = preview(format="lidar")
    assert first["available"] and first["summary"]["frames"] == 0 and first["summary"]["skipped_unreviewed"] == 2

    client.post(f"/api/s/{sid}/action", json={"name": "select", "index": 0})
    client.post(f"/api/s/{sid}/action", json={"name": "accept"})
    second = preview(format="lidar")["summary"]
    assert second["frames"] == 1 and second["boxes"] == 1 and second["per_class"] == {"Car": 1}

    r = client.get(f"/api/s/{sid}/export", params={"format": "lidar"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers["content-disposition"] and ".zip" in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert "labels/000000.txt" in z.namelist() and len(z.read("labels/000000.txt").split()) == 8


def test_export_errors(env, sid):
    client = env[0]
    kitti = client.get(f"/api/s/{sid}/export/preview", params={"format": "kitti"}).json()
    assert kitti["available"] is False and "calibration" in kitti["reason"]
    assert client.get(f"/api/s/{sid}/export", params={"format": "kitti"}).status_code == 400
    assert client.get(f"/api/s/{sid}/export", params={"format": "coco"}).status_code == 400
    assert client.get("/api/s/zzz/export", params={"format": "lidar"}).status_code == 404
