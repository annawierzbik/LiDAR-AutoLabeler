"""Sequence sources: zip upload (limits, safe extraction, validate_folder) and admin registration
of folders below LAL_DATA_ROOT."""
import os
import stat
import zipfile

import numpy as np
import pytest

from tests.test_api.conftest import FRAMES, frames_zip, write_frames, zip_of

pytestmark = pytest.mark.postgres

FRAME = np.zeros((10, 4), "<f4").tobytes()


@pytest.fixture
def anna(api):
    api.user("anna")
    return api.login("anna")


@pytest.fixture
def admin(api):
    api.user("boss", "admin")
    return api.login("boss", "tab-cccccccc")


def uploads(api):
    folder = api.svc.settings.uploads_dir
    return sorted(p.name for p in folder.iterdir() if not p.name.startswith(".")) if folder.exists() else []


def assert_rejected(r, status, text):
    assert r.status_code == status, r.text
    assert text in r.json()["error"]["message"]


def test_upload_writes_the_frames_into_the_blob_store(api, anna):
    import glob
    import tempfile

    temp = set(glob.glob(os.path.join(tempfile.gettempdir(), "lal-upload-*")))
    r = anna.upload(frames_zip(), "my drive")
    assert r.status_code == 201
    seq = r.json()
    assert seq["name"] == "my drive" and seq["num_frames"] == FRAMES and seq["source"] == "upload"
    [prefix] = uploads(api)
    assert sorted(os.listdir(api.svc.settings.uploads_dir / prefix)) == [f"{i:06d}.bin" for i in range(FRAMES)]
    spec = api.svc.sequences.get(seq["id"], None)["spec"]
    assert spec == {"blob_prefix": prefix, "frames": [f"{i:06d}.bin" for i in range(FRAMES)]}   # a prefix, not a path
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "lal-upload-*"))) == temp  # the archive is not kept


def test_upload_of_frames_at_the_top_level_and_other_files_ignored(api, anna):
    data = zip_of({"0001.bin": FRAME, "0002.bin": FRAME, "readme.txt": b"hi", "__MACOSX/._0001.bin": b"x"})
    assert anna.upload(data).json()["num_frames"] == 2


@pytest.mark.parametrize("name", ["../evil.bin", "a/../../evil.bin", "/abs.bin", "C:/x/evil.bin", "a\\evil.bin"])
def test_unsafe_names_are_refused(api, anna, name):
    buf = zip_of({"ok/0001.bin": FRAME})
    import io

    b = io.BytesIO(buf)
    with zipfile.ZipFile(b, "a") as z:
        info = zipfile.ZipInfo("placeholder")
        info.filename = name                       # bypass ZipFile's own normalisation on write
        z.writestr(info, FRAME)
    r = anna.upload(b.getvalue())
    if "\\" in name and os.sep == "\\":
        # zipfile on Windows reads a backslash as a folder separator: a\evil.bin is a/evil.bin, a
        # second folder next to ok/ (refused for that); elsewhere the backslash itself is refused
        assert_rejected(r, 400, "more than one folder")
    else:
        assert_rejected(r, 400, "unsafe file name")
    assert uploads(api) == []
    assert not (api.env.tmp / "evil.bin").exists() and not (api.svc.settings.uploads_dir.parent / "evil.bin").exists()


def test_symlinks_are_refused(api, anna):
    import io

    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("d/0001.bin", FRAME)
        link = zipfile.ZipInfo("d/0002.bin")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(link, "/etc/passwd")
    assert_rejected(anna.upload(b.getvalue()), 400, "symbolic link")
    assert uploads(api) == []


def test_frames_in_several_folders_or_none_are_refused(api, anna):
    assert_rejected(anna.upload(zip_of({"a/1.bin": FRAME, "b/2.bin": FRAME})), 400, "more than one folder")
    assert_rejected(anna.upload(zip_of({"a/readme.txt": b"x"})), 400, "no .bin")
    assert_rejected(anna.upload(b"this is not a zip"), 400, "not a zip")
    assert_rejected(anna.upload(zip_of({"a/1.bin": FRAME, "b/1.bin": FRAME})), 400, "appears twice")


def test_invalid_point_clouds_fail_validation_and_leave_nothing(api, anna):
    r = anna.upload(zip_of({"d/0001.bin": b"\x00" * 10}))
    assert r.status_code == 400
    assert uploads(api) == []


def test_size_limits(api, anna, monkeypatch):
    from infra.settings import get_settings

    monkeypatch.setenv("LAL_UPLOAD_MAX_BYTES", "2000")
    monkeypatch.setenv("LAL_UPLOAD_MAX_UNPACKED_BYTES", "20000")
    get_settings.cache_clear()
    a = api.restart()
    c = a.login("anna")
    big = os.urandom(5000)                         # does not compress
    r = c.upload(zip_of({"d/1.bin": big[:4992]}))
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
    bomb = zip_of({"d/1.bin": b"\x00" * 400_000})  # small archive, large content
    assert len(bomb) < 2000
    assert_rejected(c.upload(bomb), 413, "unpack")
    assert uploads(a) == []


def test_a_member_that_unpacks_to_more_than_its_header_says_is_stopped(tmp_path):
    from infra.blob import LocalBlobStore
    from infra.sequence_store import TooLarge, extract_frames

    archive = tmp_path / "lie.zip"
    archive.write_bytes(zip_of({"d/1.bin": b"\x00" * 100_000}))
    data = bytearray(archive.read_bytes())
    # set the declared uncompressed size (local header and central directory) to 16 bytes
    for sig, offset in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):
        i = data.index(sig)
        data[i + offset:i + offset + 4] = (16).to_bytes(4, "little")
    archive.write_bytes(bytes(data))
    blobs = LocalBlobStore(tmp_path / "up")
    with pytest.raises((TooLarge, zipfile.BadZipFile)):
        extract_frames(archive, blobs, "seq-x", max_unpacked=50_000)
    assert blobs.list() == []


def test_register_a_folder_below_the_data_root(api, admin, anna):
    r = admin.post("/api/v1/sequences/register", {"path": "drive_a"})
    assert r.status_code == 201 and r.json()["num_frames"] == FRAMES and r.json()["source"] == "folder"
    absolute = str(api.env.data_root / "drive_a")
    r = admin.post("/api/v1/sequences/register", {"path": absolute, "name": "A", "owner_id": anna.user["id"]})
    assert r.status_code == 201 and r.json()["owner_id"] == anna.user["id"]
    assert [s["name"] for s in anna.get("/api/v1/sequences").json()["items"]] == ["A"]


def test_register_refuses_paths_outside_the_root(api, admin, tmp_path):
    outside = write_frames(tmp_path / "outside")
    for path in (str(outside), "../outside", "drive_a/../../outside"):
        r = admin.post("/api/v1/sequences/register", {"path": path})
        assert r.status_code == 403 and r.json()["error"]["code"] == "outside_data_root", path
    link = api.env.data_root / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create symbolic links here")
    assert admin.post("/api/v1/sequences/register", {"path": "link"}).status_code == 403


def test_register_is_for_admins_and_needs_a_data_root(api, admin, anna, monkeypatch):
    from infra.settings import get_settings

    r = anna.post("/api/v1/sequences/register", {"path": "drive_a"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    assert admin.post("/api/v1/sequences/register", {"path": "missing"}).status_code == 400
    assert admin.post("/api/v1/sequences/register", {"path": "drive_a", "owner_id": 999}).status_code == 404
    monkeypatch.delenv("LAL_DATA_ROOT")
    get_settings.cache_clear()
    a = api.restart()
    r = a.login("boss").post("/api/v1/sequences/register", {"path": "drive_a"})
    assert r.status_code == 403 and "LAL_DATA_ROOT" in r.json()["error"]["message"]


def test_a_session_from_before_phase_4_opens_after_it_is_assigned(api, admin, anna):
    """An unowned session of a registered folder, given to anna (scripts.assign_session_owner),
    is what anna's tab opens: her box is there."""
    from common.boxes import BoundingBox3D
    from ingestion.folder import folder_key
    from pipeline.session_builder import key_for_owner
    from verification.session import AnnotationSession

    label = folder_key(api.env.data_root / "drive_a")
    old = AnnotationSession.from_tracking(label, [[] for _ in range(FRAMES)], "manual")
    old.add_manual(1, BoundingBox3D(7.5, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 1.0, 0))
    api.svc.sessions.create(f"{label}_manual", old)
    api.svc.sessions.assign_owner(f"{label}_manual", anna.user["id"], key_for_owner(f"{label}_manual", label,
                                                                                    anna.user["id"]))
    seq = admin.post("/api/v1/sequences/register", {"path": "drive_a", "owner_id": anna.user["id"]}).json()
    sid = api.open_tab(anna, seq["id"], "manual")
    boxes = anna.get(f"/api/v1/tabs/{sid}/state?frame=1").json()["annotations"]
    assert [b["x"] for b in boxes] == [7.5]


# ------------------------------------------------------------------ delete
def _open_and_cache(api, client, seq_id):
    """A tab opened in assisted mode: a session and a detection-cache entry of the sequence."""
    return api.open_tab(client, seq_id, "assisted")


def test_the_owner_deletes_an_upload_without_sessions(api, anna):
    seq = anna.upload(frames_zip()).json()
    prefix = uploads(api)[0]
    r = anna.delete(f"/api/v1/sequences/{seq['id']}")
    assert r.status_code == 200, r.text
    assert r.json() == {"id": seq["id"], "sessions_deleted": [], "frames_deleted": FRAMES, "cache_deleted": []}
    assert uploads(api) == [] and prefix
    assert anna.get(f"/api/v1/sequences/{seq['id']}").status_code == 404
    assert anna.delete(f"/api/v1/sequences/{seq['id']}").status_code == 404


def test_delete_is_refused_while_sessions_exist_unless_an_admin_forces_it(api, anna, admin):
    from pipeline.session_builder import results_blobs

    seq = anna.upload(frames_zip()).json()
    sid = _open_and_cache(api, anna, seq["id"])
    cached = [k for k in results_blobs().list() if k.startswith("seq-")]
    assert len(cached) == 1                                            # the detection cache of the upload
    r = anna.delete(f"/api/v1/sequences/{seq['id']}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "in_use"
    key = r.json()["error"]["details"]["sessions"][0]
    r = anna.delete(f"/api/v1/sequences/{seq['id']}?force=true")
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"   # force is for admins
    assert anna.get(f"/api/v1/tabs/{sid}/state?frame=0").status_code == 200    # nothing was deleted
    r = admin.delete(f"/api/v1/sequences/{seq['id']}")
    assert r.status_code == 409                                        # an admin needs force too
    r = admin.delete(f"/api/v1/sequences/{seq['id']}?force=true")
    assert r.status_code == 200, r.text
    assert r.json()["sessions_deleted"] == [key] and r.json()["cache_deleted"] == cached
    assert r.json()["frames_deleted"] == FRAMES and uploads(api) == []
    assert not api.svc.sessions.exists(key) and [k for k in results_blobs().list() if k.startswith("seq-")] == []
    assert anna.get(f"/api/v1/tabs/{sid}").status_code == 404           # the tab went with it


def test_delete_is_refused_while_the_sequence_is_being_opened(api, anna, admin):
    seq = anna.upload(frames_zip()).json()
    anna.post("/api/v1/tabs", {"sequence_id": seq["id"], "mode": "manual"})       # queued, not run yet
    r = admin.delete(f"/api/v1/sequences/{seq['id']}?force=true")
    assert r.status_code == 409 and list(r.json()["error"]["details"]) == ["jobs"]
    api.drain()
    assert admin.delete(f"/api/v1/sequences/{seq['id']}?force=true").status_code == 200


def test_registered_folders_are_not_deleted(api, admin):
    seq = admin.post("/api/v1/sequences/register", {"path": "drive_a"}).json()
    r = admin.delete(f"/api/v1/sequences/{seq['id']}?force=true")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_an_upload"
    assert (api.env.data_root / "drive_a").is_dir()



@pytest.mark.parametrize("source", ["upload", "folder"])
def test_frames_are_listed_once_when_the_sequence_is_made(api, anna, monkeypatch, source):
    """The ordered frame names are stored with the sequence (spec["frames"]); opening (cpu and gpu
    steps), points, state and export use them: after creation nothing lists the frames again, and a
    file added to a registered folder later does not change the sequence."""
    import infra.sequences
    import ingestion.folder

    api.user("boss", "admin")
    admin = api.login("boss", "tab-cccccccc")
    if source == "upload":
        seq = anna.upload(frames_zip()).json()
    else:
        seq = admin.post("/api/v1/sequences/register", {"path": "drive_a", "owner_id": anna.user["id"]}).json()
        (api.env.data_root / "drive_a" / "zzz_added_later.bin").write_bytes(FRAME)
    assert len(api.svc.sequences.get(seq["id"], None)["spec"]["frames"]) == FRAMES

    def no_listing(*a, **k):
        raise AssertionError("frames listed after the sequence was made")

    monkeypatch.setattr(infra.sequences, "upload_frame_keys", no_listing)
    monkeypatch.setattr(ingestion.folder, "frame_files", no_listing)
    monkeypatch.setattr("export.session_export.frame_files", no_listing)
    sid = api.open_tab(anna, seq["id"], "assisted")                     # cpu step, then detection on gpu
    st = anna.get(f"/api/v1/tabs/{sid}/state?frame=0").json()
    assert st["num_frames"] == FRAMES
    assert anna.get(f"/api/v1/tabs/{sid}/points/{FRAMES - 1}").status_code == 200
    assert anna.get(f"/api/v1/tabs/{sid}/points/{FRAMES}").status_code == 404   # the added file is not a frame
    eid = anna.post(f"/api/v1/tabs/{sid}/exports", {"format": "lidar", "include_unreviewed": True}).json()["job_id"]
    assert api.drain() == ["done"]
    assert anna.get(f"/api/v1/jobs/{eid}/download").status_code == 200



@pytest.mark.parametrize("extraction", ["thread_pool", "on_the_event_loop"])
def test_the_api_keeps_serving_while_an_upload_is_unpacked(api, anna, monkeypatch, extraction):
    """One uvicorn server (one event loop): while an upload's extraction is held, /healthz and a
    logged-in request still answer. The second case runs the extraction on the event loop, as a
    plain call would, and shows that the test notices a blocked loop."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    import api.routes_sequences as rs
    from tests.test_api.conftest import PASSWORD

    held, release = threading.Event(), threading.Event()
    real = rs.extract_frames

    def slow_extract(*args, **kwargs):
        held.set()
        release.wait(20)                                              # a large archive being unpacked
        return real(*args, **kwargs)

    monkeypatch.setattr(rs, "extract_frames", slow_extract)
    if extraction == "on_the_event_loop":
        async def inline(fn, *args, **kwargs):
            return fn(*args, **kwargs)
        monkeypatch.setattr(rs, "run_in_threadpool", inline)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="warning"))
    serving = threading.Thread(target=server.run, daemon=True)
    serving.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while not server.started:
        assert time.time() < deadline
        time.sleep(0.05)
    try:
        c = httpx.Client(base_url=base, timeout=30)
        c.post("/api/v1/auth/login", json={"username": "anna", "password": PASSWORD}).raise_for_status()
        headers = {"X-CSRF-Token": c.cookies["lal_csrf"], "Content-Type": "application/zip"}
        result = {}
        uploading = threading.Thread(target=lambda: result.update(
            r=c.post("/api/v1/sequences/upload?name=big", content=frames_zip(), headers=headers)))
        uploading.start()
        assert held.wait(10)                                          # the extraction is running now
        try:
            health = httpx.get(base + "/healthz", timeout=2)
            seqs = httpx.get(base + "/api/v1/sequences", cookies=c.cookies, timeout=2)
            answered = (health.status_code, seqs.status_code)
        except httpx.TimeoutException:
            answered = "timeout"
        release.set()
        uploading.join(30)
        assert result["r"].status_code == 201, result["r"].text      # the upload itself completes
        if extraction == "thread_pool":
            assert answered == (200, 200)
        else:
            assert answered == "timeout"                              # a blocked loop answers nothing
    finally:
        release.set()
        server.should_exit = True
        serving.join(10)
