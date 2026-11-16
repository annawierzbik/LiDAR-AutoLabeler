"""Uploads addressed by a blob prefix (docs/adr/0020): frame names survive into exports, assisted
mode can be switched off, and the whole flow on the S3 store (MinIO) touches no local folder."""
import io
import os
import zipfile

import numpy as np
import pytest

from tests.test_api.conftest import zip_of

pytestmark = pytest.mark.postgres

NAMES = ["scan_2", "scan_10", "scan_1"]                               # natural order: 1, 2, 10


def frames(names=NAMES):
    rng = np.random.default_rng(3)
    return zip_of({f"d/{n}.bin": rng.uniform(-20, 20, (300, 4)).astype("<f4").tobytes() for n in names})


@pytest.fixture
def anna(api):
    api.user("anna")
    return api.login("anna")


def export_names(api, c, sid) -> list[str]:
    c.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 0, "label": "Car"})
    eid = c.post(f"/api/v1/tabs/{sid}/exports", {"format": "lidar", "include_empty": True,
                                                  "include_unreviewed": True}).json()["job_id"]
    assert api.drain() == ["done"]
    dl = c.get(f"/api/v1/jobs/{eid}/download")
    assert dl.status_code == 200, dl.text
    return sorted(os.path.basename(n) for n in zipfile.ZipFile(io.BytesIO(dl.content)).namelist()
                  if n.startswith("labels/"))


def test_an_upload_exports_the_frame_names_of_its_archive_like_a_folder_does(api, anna, tmp_path):
    seq = anna.upload(frames()).json()
    names = export_names(api, anna, api.open_tab(anna, seq["id"]))
    assert names == ["scan_1.txt", "scan_10.txt", "scan_2.txt"]          # not 000000.txt, ...
    folder = api.env.data_root / "same"                                # the same frames as a registered folder
    folder.mkdir()
    with zipfile.ZipFile(io.BytesIO(frames())) as z:
        for n in NAMES:
            (folder / f"{n}.bin").write_bytes(z.read(f"d/{n}.bin"))
    api.user("boss", "admin")
    admin = api.login("boss", "tab-cccccccc")
    reg = admin.post("/api/v1/sequences/register", {"path": "same", "owner_id": anna.user["id"]}).json()
    assert export_names(api, anna, api.open_tab(anna, reg["id"])) == names


def test_assisted_mode_can_be_switched_off(api, anna, monkeypatch):
    from infra.settings import get_settings

    monkeypatch.setenv("LAL_ASSISTED_MODE", "false")
    get_settings.cache_clear()
    a = api.restart()
    c = a.login("anna")
    seq = c.upload(frames()).json()
    r = c.post("/api/v1/tabs", {"sequence_id": seq["id"], "mode": "assisted"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "assisted_unavailable"
    assert a.sent == []                                                # no job was queued
    assert a.open_tab(c, seq["id"], "manual")


@pytest.mark.s3
def test_the_whole_flow_on_s3_uses_no_local_folder(api, monkeypatch, s3_bucket, tmp_path):
    """Upload, assisted opening (fake detector), points, export, download and delete with
    LAL_BLOB_STORE=s3: every blob is in the bucket; the results and upload folders stay empty."""
    from infra.settings import get_settings

    env, bucket, client = s3_bucket
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    a = api.restart()
    a.user("anna")
    c = a.login("anna")
    ready = c.get("/readyz").json()
    assert ready["status"] == "ready" and ready["checks"]["upload_blobs"] == {"ok": True}

    def keys():
        return sorted(o["Key"] for p in client.get_paginator("list_objects_v2").paginate(Bucket=bucket)
                      for o in p.get("Contents", []))

    seq = c.upload(frames()).json()
    prefix = a.svc.sequences.get(seq["id"], None)["spec"]["blob_prefix"]
    assert keys() == [f"uploads/{prefix}/{n}.bin" for n in ("scan_1", "scan_10", "scan_2")]
    sid = a.open_tab(c, seq["id"], "assisted")
    assert f"cache/{prefix}.json" in keys()                             # detection cache in the bucket
    pts = c.get(f"/api/v1/tabs/{sid}/points/1")
    assert pts.status_code == 200 and len(pts.content) == 300 * 3 * 4
    assert export_names(a, c, sid) == ["scan_1.txt", "scan_10.txt", "scan_2.txt"]
    assert any(k.startswith("exports/") for k in keys())
    r = c.delete(f"/api/v1/sequences/{seq['id']}")
    assert r.status_code == 409                                         # it has a session
    a.user("boss", "admin")
    r = a.login("boss", "tab-cccccccc").delete(f"/api/v1/sequences/{seq['id']}?force=true")
    assert r.status_code == 200 and r.json()["frames_deleted"] == 3 and r.json()["cache_deleted"] == [f"{prefix}.json"]
    assert [k for k in keys() if not k.startswith("exports/")] == []
    results = a.env.results
    assert not results.exists() or [p for p in results.rglob("*") if p.is_file()] == []
