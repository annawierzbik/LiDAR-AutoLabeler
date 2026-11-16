"""infra/: blob store, session store, job queue, settings, point source."""
import json
import os
import threading

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from infra.blob import BlobStore, LocalBlobStore, atomic_write_bytes
from infra.jobs import JobQueue, ThreadJobQueue
from infra.points import BlobPointSource, point_source
from infra.sessions import (
    FileSessionStore,
    RevisionConflict,
    SessionLocked,
    SessionNotFound,
    SessionStore,
)
from infra.settings import PROJECT_ROOT, Settings
from verification.session import AnnotationSession


def box(x, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


# ------------------------------------------------------------------ blobs
def test_local_blob_store_round_trip(tmp_path):
    blobs = LocalBlobStore(tmp_path)
    assert isinstance(blobs, BlobStore)
    blobs.put("cache/a.json", b"1")
    blobs.put("b.bin", b"22")
    assert blobs.get("cache/a.json") == b"1" and blobs.exists("b.bin") and not blobs.exists("c")
    assert blobs.list() == ["b.bin", "cache/a.json"] and blobs.list("cache/") == ["cache/a.json"]
    with pytest.raises(KeyError):
        blobs.get("missing")


@pytest.mark.parametrize("key", ["../x", "/abs", "a/../../b", "C:/x", "a\\b", ""])
def test_blob_keys_cannot_leave_the_root(tmp_path, key):
    with pytest.raises(ValueError):
        LocalBlobStore(tmp_path).put(key, b"x")


def test_put_is_atomic_when_the_write_fails(tmp_path, monkeypatch):
    blobs = LocalBlobStore(tmp_path)
    blobs.put("a", b"old")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        blobs.put("a", b"new")
    assert blobs.get("a") == b"old" and blobs.list() == ["a"]        # no temp file left behind


def test_atomic_write_creates_folders(tmp_path):
    atomic_write_bytes(tmp_path / "x" / "y.json", b"{}")
    assert (tmp_path / "x" / "y.json").read_bytes() == b"{}"


def test_blob_point_source_reads_frames_like_the_loader(tmp_path):
    from ingestion.folder import FolderSequence

    pts = [np.random.default_rng(i).uniform(-5, 5, (7, 4)).astype("<f4") for i in range(3)]
    for i, p in enumerate(pts):
        p.tofile(tmp_path / f"frame{i * 5}.bin")                     # natural order: 0, 5, 10
    seq = FolderSequence(tmp_path)
    src = point_source(seq)
    assert isinstance(src, BlobPointSource) and len(src) == 3
    for i in range(3):
        assert np.array_equal(src.xyz(i), seq[i].xyz)
    with pytest.raises(IndexError):
        src.xyz(3)


# --------------------------------------------------------------- sessions
@pytest.fixture
def store(tmp_path):
    return FileSessionStore(tmp_path)


def session():
    return AnnotationSession.from_tracking("0000", [[box(10)], [box(11)]])


def test_file_session_store_save_load_list(store):
    assert isinstance(store, SessionStore)
    s = session()
    store.save("a", s, expected_revision=0)
    assert store.list() == ["a"] and store.load("a").num_frames == 2
    with pytest.raises(SessionNotFound):
        store.load("b")


def test_save_with_a_stale_revision_is_a_conflict(store):
    s = session()
    store.save("a", s, expected_revision=0)
    s.accept(0, 0)
    store.save("a", s, expected_revision=0)                          # stored 0 -> now 1
    stale = store.load("a")
    stale.revision = 0
    with pytest.raises(RevisionConflict) as err:
        store.save("a", stale, expected_revision=0)
    assert err.value.stored == 1 and err.value.expected == 0
    store.save("a", stale, expected_revision=None)                   # explicit overwrite


def test_lock_is_exclusive_per_owner(store):
    store.lock("a", "tab1")
    store.lock("a", "tab1")                                          # re-entrant for the same owner
    with pytest.raises(SessionLocked) as err:
        store.lock("a", "tab2")
    assert err.value.owner == "tab1" and store.locked_by("a") == "tab1"
    store.unlock("a", "tab2")                                        # not the owner: nothing happens
    assert store.locked_by("a") == "tab1"
    store.unlock("a", "tab1")
    store.lock("a", "tab2")


def test_store_keys_are_plain_names(store, tmp_path):
    with pytest.raises(ValueError):
        store.path("../x")
    assert store.key_for(tmp_path / "abc.json") == "abc"
    with pytest.raises(ValueError):
        store.key_for(tmp_path / "sub" / "abc.json")


# ------------------------------------------------------------------- jobs
def test_thread_queue_runs_and_reports():
    q = ThreadJobQueue(background=False)
    assert isinstance(q, JobQueue)
    ok = q.submit(lambda: 42, "answer")
    assert q.status(ok).state == "done" and q.status(ok).result == 42
    bad = q.submit(lambda: 1 / 0, "broken", job_id="x1")
    assert bad == "x1" and q.status(bad).state == "failed" and "division" in q.status(bad).message
    with pytest.raises(ValueError):
        q.submit(lambda: 0, job_id="x1")
    with pytest.raises(KeyError):
        q.status("nope")


def test_cancel_before_start_and_not_while_running():
    q = ThreadJobQueue(background=True)
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return "done"

    running = q.submit(slow)
    started.wait(5)
    assert q.cancel(running) is False                                # a running thread cannot be stopped
    release.set()

    class Held(ThreadJobQueue):                                      # jobs wait until released
        def _start(self, job_id, fn):
            self.waiting = (job_id, fn)

    q2 = Held()
    jid = q2.submit(lambda: 1)
    assert q2.status(jid).state == "queued"
    assert q2.cancel(jid) is True and q2.status(jid).state == "cancelled"
    q2._run(*q2.waiting)
    assert q2.status(jid).state == "cancelled" and q2.status(jid).result is None   # it is not run afterwards


# --------------------------------------------------------------- settings
def test_settings_defaults_reproduce_the_old_layout(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LAL_"):
            monkeypatch.delenv(k)
    s = Settings()
    assert s.results_dir == PROJECT_ROOT / "results" and s.session_dir == PROJECT_ROOT / "results" / "sessions"
    assert (s.host, s.port, s.point_cache_frames) == ("127.0.0.1", 8000, 8)


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LAL_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("LAL_PORT", "9001")
    monkeypatch.setenv("LAL_POINT_CACHE_FRAMES", "3")
    s = Settings()
    assert s.results_dir == tmp_path and s.port == 9001 and s.point_cache_frames == 3
    monkeypatch.setenv("LAL_RESULTS_DIR", "relative/res")
    assert Settings().results_dir == PROJECT_ROOT / "relative" / "res"


def test_session_files_are_json(store):
    store.save("a", session(), expected_revision=0)
    assert json.loads(store.path("a").read_text())["version"] == 2
