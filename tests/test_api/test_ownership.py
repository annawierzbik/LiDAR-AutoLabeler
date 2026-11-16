"""Ownership: a user sees only their own sequences, tabs, sessions and jobs; an admin sees all.
Every endpoint that takes the id of someone else's resource answers 404 (not 403), so ids of other
users' resources cannot even be confirmed."""
import pytest

from tests.test_api.conftest import frames_zip

pytestmark = pytest.mark.postgres


@pytest.fixture
def world(api):
    """anna's sequence, open tab (with a box) and export job; bob and an admin."""
    api.user("anna")
    api.user("bob")
    api.user("boss", "admin")
    anna, bob, admin = api.login("anna"), api.login("bob", "tab-bbbbbbbb"), api.login("boss", "tab-cccccccc")
    seq = anna.upload(frames_zip()).json()
    sid = api.open_tab(anna, seq["id"])
    anna.post(f"/api/v1/tabs/{sid}/action", {"name": "add", "frame": 0, "label": "Car"})
    export = anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "lidar"}).json()["job_id"]
    api.drain()
    open_job = api.svc.tabs.get(sid, None)["job_id"]
    return {"api": api, "anna": anna, "bob": bob, "admin": admin, "seq": seq["id"], "sid": sid,
            "export": export, "open_job": open_job}


def endpoints(w) -> list[tuple[str, str, dict | None]]:
    seq, sid, export, job = w["seq"], w["sid"], w["export"], w["open_job"]
    return [
        ("get", f"/api/v1/sequences/{seq}", None),
        ("post", "/api/v1/tabs", {"sequence_id": seq, "mode": "manual"}),
        ("get", f"/api/v1/tabs/{sid}", None),
        ("post", f"/api/v1/tabs/{sid}/cancel", {}),
        ("get", f"/api/v1/tabs/{sid}/state?frame=0", None),
        ("get", f"/api/v1/tabs/{sid}/next_pending?frame=0", None),
        ("get", f"/api/v1/tabs/{sid}/points/0", None),
        ("post", f"/api/v1/tabs/{sid}/action", {"name": "accept", "frame": 0, "index": 0}),
        ("post", f"/api/v1/tabs/{sid}/heartbeat", {}),
        ("post", f"/api/v1/tabs/{sid}/exports", {"format": "lidar"}),
        ("post", f"/api/v1/tabs/{sid}/finish", {}),
        ("get", f"/api/v1/jobs/{export}", None),
        ("get", f"/api/v1/jobs/{export}/download", None),
        ("post", f"/api/v1/jobs/{export}/cancel", {}),
        ("get", f"/api/v1/jobs/{job}", None),
        ("post", f"/api/v1/jobs/{job}/cancel", {}),
    ]


def call(client, method, path, body):
    return client.get(path) if method == "get" else client.post(path, body)


def test_every_endpoint_hides_other_users_resources(world):
    bob = world["bob"]
    for method, path, body in endpoints(world):
        r = call(bob, method, path, body)
        assert r.status_code == 404, (method, path, r.status_code, r.text)
        assert r.json()["error"]["code"] == "not_found", path
    # nothing changed for anna: her tab still works, her box is still there
    anna = world["anna"]
    st = anna.get(f"/api/v1/tabs/{world['sid']}/state?frame=0").json()
    assert len(st["annotations"]) == 1 and anna.get(f"/api/v1/jobs/{world['export']}/download").status_code == 200


def test_the_owner_reaches_every_endpoint(world):
    anna = world["anna"]
    for method, path, body in endpoints(world):
        r = call(anna, method, path, body)
        assert r.status_code != 404, (method, path, r.text)


def test_lists_show_only_the_users_own_items(world):
    bob, anna, admin = world["bob"], world["anna"], world["admin"]
    for path in ("/api/v1/sequences", "/api/v1/jobs", "/api/v1/sessions"):
        assert bob.get(path).json() == {"items": [], "total": 0, "limit": 50, "offset": 0}, path
        mine = anna.get(path).json()
        assert mine["total"] >= 1 and all(i["owner_id"] == anna.user["id"] for i in mine["items"]), path
        assert admin.get(path).json()["total"] == mine["total"], path
    bob.upload(frames_zip(), "bobs")
    assert [s["name"] for s in bob.get("/api/v1/sequences").json()["items"]] == ["bobs"]
    assert {s["name"] for s in admin.get("/api/v1/sequences").json()["items"]} == {"drive", "bobs"}
    assert "bobs" not in str(anna.get("/api/v1/sequences").json())


def test_admin_sees_everything(world):
    admin = world["admin"]
    assert admin.get(f"/api/v1/sequences/{world['seq']}").status_code == 200
    assert admin.get(f"/api/v1/jobs/{world['export']}").json()["owner_id"] == world["anna"].user["id"]
    assert admin.get(f"/api/v1/jobs/{world['export']}/download").status_code == 200
    # anna's tab holds the lock; the admin may look at the tab but not take over the live session
    r = admin.get(f"/api/v1/tabs/{world['sid']}/state")
    assert r.status_code == 409 and r.json()["error"]["code"] == "locked"


def test_users_have_separate_sessions_of_the_same_folder(world):
    api, bob = world["api"], world["bob"]
    folder_seq = world["admin"].post("/api/v1/sequences/register",
                                     {"path": "drive_a", "owner_id": bob.user["id"]}).json()
    sid = api.open_tab(bob, folder_seq["id"])
    assert bob.get(f"/api/v1/tabs/{sid}/state?frame=0").json()["annotations"] == []
    keys = {i["key"] for i in world["admin"].get("/api/v1/sessions").json()["items"]}
    assert len(keys) == 2 and all(k.startswith(("u1_", "u2_")) for k in keys)
