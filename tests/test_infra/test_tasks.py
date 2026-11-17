"""Tasks (infra/tasks.py) on the ThreadTaskQueue with the fake detector: no GPU, no broker."""
import io
import json
import time
import zipfile
from argparse import Namespace

import numpy as np
import pytest

import pipeline.session_builder
import tests.fakes
import verification.paths
from infra import worker_detector
from infra.jobs import FINAL, ThreadTaskQueue
from infra.settings import get_settings
from infra.tasks import detect_track_payload
from verification import cache

FRAMES = 12


@pytest.fixture
def env(tmp_path, monkeypatch):
    folder = tmp_path / "drive"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(FRAMES):
        rng.uniform(-20, 20, (300, 4)).astype("<f4").tofile(folder / f"{i:06d}.bin")
    results = tmp_path / "results"
    monkeypatch.setattr(verification.paths, "RESULTS_DIR", results)
    monkeypatch.setattr(pipeline.session_builder, "RESULTS_DIR", results)
    monkeypatch.setenv("LAL_WORKER_DETECTOR_FACTORY", "tests.fakes:FakeDetector")
    monkeypatch.setattr(cache, "_warned", set())
    get_settings.cache_clear()
    worker_detector.reset()
    tests.fakes.LOADS = 0
    yield Namespace(folder=folder, results=results)
    worker_detector.reset()
    get_settings.cache_clear()


def payload(folder, **kw):
    from pipeline.session_builder import make_args

    return detect_track_payload(make_args(folder=str(folder), **kw))


def wait(q, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while q.status(job_id).state not in FINAL:
        assert time.time() < deadline, q.status(job_id)
        time.sleep(0.02)
    return q.status(job_id)


def cache_files(results):
    return sorted(p.name for p in results.glob("*.json")) if results.exists() else []


def test_detect_track_writes_the_cache_with_progress(env):
    q = ThreadTaskQueue(background=False)
    seen = []
    q.tasks = {**q.tasks, "detect_track": (q.tasks["detect_track"][0],
                                           lambda p, ctx: _recording(p, ctx, seen))}
    st = q.status(q.submit("detect_track", payload(env.folder)))
    assert st.state == "done" and st.progress == 1.0 and st.result["frames"] == FRAMES
    assert st.result["cached"] is False and cache_files(env.results) == [st.result["cache_key"]]
    assert seen[0] == 0.0 and seen[-1] == 1.0 and len(seen) > 3 and seen == sorted(seen)
    data = json.loads((env.results / st.result["cache_key"]).read_text())
    assert data["meta"]["split"] is None and len(data["frames"]) == FRAMES


def _recording(p, ctx, seen):
    from infra.tasks import detect_track

    class Ctx:
        def progress(self, f, m=""):
            seen.append(f)
            ctx.progress(f, m)

        def should_stop(self):
            return ctx.should_stop()

    return detect_track(p, Ctx())


def test_a_second_run_uses_the_cache_and_loads_no_model(env):
    q = ThreadTaskQueue(background=False)
    q.status(q.submit("detect_track", payload(env.folder)))
    assert tests.fakes.LOADS == 1
    q2 = ThreadTaskQueue(background=False)                          # e.g. after a restart
    st = q2.status(q2.submit("detect_track", payload(env.folder)))
    assert st.state == "done" and st.result["cached"] is True
    assert tests.fakes.LOADS == 1                                    # no second model
    assert len(cache_files(env.results)) == 1


def test_duplicate_submit_returns_the_same_job(env):
    q = ThreadTaskQueue(background=False)
    a = q.submit("detect_track", payload(env.folder))
    assert q.submit("detect_track", payload(env.folder)) == a
    assert q.submit("detect_track", payload(env.folder, threshold=0.5, start=2)) != a


def test_thread_mode_makes_and_releases_a_detector_per_job(env):
    import gc

    q = ThreadTaskQueue(background=False)
    q.submit("detect_track", payload(env.folder, start=0, stop=5))
    q.submit("detect_track", payload(env.folder, start=5, stop=9))
    gc.collect()
    assert tests.fakes.LOADS == 2 and len(tests.fakes.ALIVE) == 0   # made per job, nothing kept
    assert not worker_detector.loaded()


def test_thread_mode_uses_the_jobs_own_threshold(env):
    q = ThreadTaskQueue(background=False)
    st = q.status(q.submit("detect_track", payload(env.folder, threshold=0.7)))
    assert st.state == "done", st.message                           # as before phase 3


class WorkerCtx:
    """The context of a GPU worker process (keeps one detector)."""
    keeps_detector = True

    def progress(self, fraction, message=""):
        pass

    def should_stop(self):
        return False


def test_a_worker_loads_one_detector_and_keeps_it(env):
    from infra.tasks import detect_track

    detect_track(payload(env.folder, start=0, stop=5), WorkerCtx())
    detect_track(payload(env.folder, start=5, stop=9), WorkerCtx())
    assert tests.fakes.LOADS == 1 and worker_detector.loaded()


def test_a_worker_refuses_another_detector_configuration(env):
    from infra.tasks import detect_track
    from infra.worker_detector import DetectorMismatch

    with pytest.raises(DetectorMismatch, match="another detector loaded.*score_threshold"):
        detect_track(payload(env.folder, threshold=0.7), WorkerCtx())          # the worker has 0.5
    assert tests.fakes.LOADS == 0 and cache_files(env.results) == []


def test_cancel_while_running_discards_the_partial_result(env, monkeypatch):
    monkeypatch.setenv("LAL_FAKE_DETECTOR_DELAY", "0.05")
    q = ThreadTaskQueue(background=True)
    job = q.submit("detect_track", payload(env.folder))
    deadline = time.time() + 10
    while q.status(job).progress < 0.2:
        assert time.time() < deadline
        time.sleep(0.01)
    assert q.cancel(job) is True and q.status(job).state in ("cancelling", "cancelled")
    st = wait(q, job)
    assert st.state == "cancelled" and "stopped after" in st.message
    assert cache_files(env.results) == []                           # nothing half-written
    assert q.cancel(job) is False


def test_cancel_before_start(env):
    class NeverStarts(ThreadTaskQueue):
        def _run(self, job, payload):                               # the job stays queued
            pass

    q = NeverStarts(background=False, tasks={"slow": ("cpu", lambda p, ctx: {})})
    jid = q.submit("slow", {})
    assert q.status(jid).state == "queued"
    assert q.cancel(jid) is True and q.status(jid).state == "cancelled"
    assert q.cancel(jid) is False


def test_export_task_makes_a_zip(env, tmp_path):
    from common.boxes import BoundingBox3D
    from infra.sessions import FileSessionStore
    from verification.session import AnnotationSession

    store = FileSessionStore(env.results / "sessions")
    s = AnnotationSession.from_tracking("drive", [[BoundingBox3D(10, 0, -1, 4, 1.8, 1.5, 0, "Car", 0.9, 0)]
                                                  for _ in range(FRAMES)])
    s.accept_track(0)
    store.create("k", s)
    from pipeline.session_builder import make_args

    q = ThreadTaskQueue(background=False)
    pl = {"session_key": "k", "revision": 1, "sequence": vars(make_args(folder=str(env.folder))),
          "format": "lidar", "include_unreviewed": False, "include_empty": False}
    from unittest import mock

    with mock.patch("infra.settings.make_session_store", return_value=store):
        st = q.status(q.submit("export", pl))
    assert st.state == "done" and st.result["summary"]["frames"] == FRAMES
    assert st.result["blob"].split("/")[0] == str(st.result["expires_at"])       # <expiry>/<job id>/<name>
    data = (env.results / "exports" / st.result["blob"]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert len([n for n in z.namelist() if n.startswith("labels/")]) == FRAMES


def open_payload(env, owner=3, mode="assisted"):
    from infra.tasks import open_session_payload

    return open_session_payload({"folder": str(env.folder)}, owner, 1, mode)


def test_open_session_builds_the_owners_session(env):
    from unittest import mock

    from infra.sessions import FileSessionStore

    store = FileSessionStore(env.results / "sessions")
    q = ThreadTaskQueue(background=False)
    with mock.patch("infra.settings.make_session_store", return_value=store):
        st = q.status(q.submit("open_session", open_payload(env)))
    assert st.state == "done", st.message
    assert st.result["session_key"].startswith("u3_") and st.result["num_frames"] == FRAMES
    assert store.load(st.result["session_key"]).stats()["total"] == FRAMES      # one fake car per frame
    assert len(cache_files(env.results)) == 1 and tests.fakes.LOADS == 1
    with mock.patch("infra.settings.make_session_store", return_value=store):   # another user: own session
        other = q.status(q.submit("open_session", open_payload(env, owner=4)))
    assert other.result["session_key"].startswith("u4_") and tests.fakes.LOADS == 1   # detection cached


def test_open_session_manual_needs_no_detector(env):
    from unittest import mock

    from infra.sessions import FileSessionStore
    from infra.tasks import TASKS, open_session_kind

    store = FileSessionStore(env.results / "sessions")
    assert open_session_kind("manual") == "open_session_manual" and TASKS["open_session_manual"][0] == "cpu"
    q = ThreadTaskQueue(background=False)
    with mock.patch("infra.settings.make_session_store", return_value=store):
        st = q.status(q.submit("open_session_manual", open_payload(env, mode="manual")))
    assert st.state == "done" and store.load(st.result["session_key"]).stats()["total"] == 0
    assert tests.fakes.LOADS == 0 and cache_files(env.results) == []


def test_open_session_of_a_bad_folder_fails_with_a_readable_message(env, tmp_path):
    from unittest import mock

    from infra.sessions import FileSessionStore
    from infra.tasks import open_session_payload

    empty = tmp_path / "empty"
    empty.mkdir()
    q = ThreadTaskQueue(background=False)
    with mock.patch("infra.settings.make_session_store", return_value=FileSessionStore(env.results / "s")):
        st = q.status(q.submit("open_session", open_session_payload({"folder": str(empty)}, 1, 1)))
    assert st.state == "failed" and st.message.startswith("No .bin point cloud files")


def test_open_session_cancel_discards_the_partial_detection(env, monkeypatch):
    from unittest import mock

    from infra.sessions import FileSessionStore

    monkeypatch.setenv("LAL_FAKE_DETECTOR_DELAY", "0.05")
    q = ThreadTaskQueue(background=True)
    with mock.patch("infra.settings.make_session_store", return_value=FileSessionStore(env.results / "sessions")):
        jid = q.submit("open_session", open_payload(env))
        deadline = time.time() + 10
        while q.status(jid).progress < 0.1:
            assert time.time() < deadline
            time.sleep(0.01)
        assert q.cancel(jid) is True
        st = wait(q, jid)
    assert st.state == "cancelled" and cache_files(env.results) == []


def test_expired_exports_are_pruned(tmp_path):
    from infra.blob import LocalBlobStore
    from infra.tasks import export_key, prune_exports

    blobs = LocalBlobStore(tmp_path)
    old, new = export_key(1000.9, "j1", "a.zip"), export_key(5000, "j2", "b.zip")
    assert old == "1000/j1/a.zip"
    for k in (old, new):
        blobs.put(k, b"zip")
    assert prune_exports(blobs, now=2000) == [old]
    assert blobs.list() == [new]


def test_the_scheduler_prunes_exports_periodically(tmp_path, monkeypatch):
    """celery beat sends lal.prune_exports every LAL_EXPORT_PRUNE_MINUTES to the cpu queue; the task
    deletes the expired zips (a real beat + worker: tests/test_worker/test_celery_worker.py)."""
    import infra.celery_app as ca
    import infra.tasks
    from infra.blob import LocalBlobStore
    from infra.tasks import export_key

    entry = ca.app.conf.beat_schedule["prune-exports"]
    minutes = ca.settings.export_prune_minutes
    assert entry["task"] == "lal.prune_exports" and entry["schedule"] == minutes * 60
    assert entry["options"] == {"queue": "cpu", "expires": minutes * 60}
    blobs = LocalBlobStore(tmp_path)
    old, new = export_key(time.time() - 1, "j1", "a.zip"), export_key(time.time() + 3600, "j2", "b.zip")
    for k in (old, new):
        blobs.put(k, b"zip")
    monkeypatch.setattr(infra.tasks, "export_blobs", lambda: blobs)
    assert ca.prune_exports_task() == 1 and blobs.list() == [new]


def test_thread_queue_reuses_a_done_job_only_if_reuse_done_accepts_it():
    from infra.jobs import ThreadTaskQueue

    q = ThreadTaskQueue(background=False, tasks={"t": ("cpu", lambda payload, ctx: {"n": 1})})
    a = q.submit("t", {"x": 1})
    assert q.submit("t", {"x": 1}, reuse_done=lambda r: r["n"] == 1) == a
    b = q.submit("t", {"x": 1}, reuse_done=lambda r: False)
    assert b != a and q.status(b).state == "done"
