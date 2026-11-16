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
    from web.manager import make_args

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
    from web.manager import make_args

    q = ThreadTaskQueue(background=False)
    pl = {"session_key": "k", "revision": 1, "sequence": vars(make_args(folder=str(env.folder))),
          "format": "lidar", "include_unreviewed": False, "include_empty": False}
    from unittest import mock

    with mock.patch("infra.settings.make_session_store", return_value=store):
        st = q.status(q.submit("export", pl))
    assert st.state == "done" and st.result["summary"]["frames"] == FRAMES
    data = (env.results / "exports" / st.result["blob"]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert len([n for n in z.namelist() if n.startswith("labels/")]) == FRAMES


def test_manager_opens_a_folder_through_the_task_queue(env, tmp_path):
    from infra.sessions import FileSessionStore
    from web.manager import SessionManager

    q = ThreadTaskQueue(background=True)
    mgr = SessionManager(background=False, store=FileSessionStore(env.results / "sessions"), tasks=q,
                         poll_seconds=0.01)
    sid = mgr.open_folder(str(env.folder))
    assert mgr.job(sid).status == "ready" and mgr.job(sid).task_id is not None
    assert q.status(mgr.job(sid).task_id).state == "done"
    st = mgr.service(sid).state(0)
    assert st["num_frames"] == FRAMES and st["stats"]["total"] > 0


def test_manager_cancels_the_detection_of_an_opening(env, monkeypatch):
    from infra.sessions import FileSessionStore
    from web.manager import SessionManager

    monkeypatch.setenv("LAL_FAKE_DETECTOR_DELAY", "0.05")
    q = ThreadTaskQueue(background=True)
    mgr = SessionManager(background=True, store=FileSessionStore(env.results / "sessions"), tasks=q,
                         poll_seconds=0.01)
    sid = mgr.open_folder(str(env.folder))
    deadline = time.time() + 10
    while not (mgr.job(sid).progress and mgr.job(sid).progress["fraction"] > 0.1):
        assert time.time() < deadline
        time.sleep(0.01)
    assert mgr.job(sid).view()["progress"]["state"] == "running"
    assert mgr.cancel(sid) is True
    while mgr.job(sid).status == "loading":
        assert time.time() < deadline
        time.sleep(0.01)
    assert mgr.job(sid).status == "error" and mgr.job(sid).message == "cancelled"
    assert cache_files(env.results) == []
