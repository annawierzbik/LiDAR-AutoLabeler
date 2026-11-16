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
    r = anna.upload(frames_zip(), "my drive")
    assert r.status_code == 201
    seq = r.json()
    assert seq["name"] == "my drive" and seq["num_frames"] == FRAMES and seq["source"] == "upload"
    [prefix] = uploads(api)
    assert sorted(os.listdir(api.svc.settings.uploads_dir / prefix)) == [f"{i:06d}.bin" for i in range(FRAMES)]
    assert not os.listdir(api.svc.settings.uploads_dir / ".incoming")      # the archive is not kept


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
