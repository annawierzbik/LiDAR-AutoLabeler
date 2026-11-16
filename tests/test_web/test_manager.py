import logging

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from infra.sessions import FileSessionStore
from ingestion.folder import folder_key
from tests.test_web.test_service import FakeLoader
from verification.session import AnnotationSession
from web.manager import SessionManager


def box(x=10.0, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


class Builder:
    """Stands in for pipeline.session_builder.prepare: no detector, no data. Makes the session in
    the store unless it exists (then it is resumed), returns (sequence, key)."""

    def __init__(self, tmp_path, fail=None):
        self.tmp, self.calls, self.fail = tmp_path, [], fail

    def __call__(self, args, store, detect=None):
        self.calls.append(args)
        if self.fail:
            raise RuntimeError(self.fail)
        key = f"{folder_key(args.folder) if args.folder else args.sequence}_{args.mode}_{args.start}"
        if not store.exists(key):
            store.create(key, AnnotationSession.from_tracking("0000", [[box()], [box(11)]], args.mode))
        return FakeLoader(), key


@pytest.fixture
def builder(tmp_path):
    return Builder(tmp_path)


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def manager(tmp_path, builder, clock=None):
    return SessionManager(builder=builder, background=False,
                          store=FileSessionStore(tmp_path, lock_ttl=120, clock=clock or Clock()))


@pytest.fixture
def mgr(tmp_path, builder):
    return manager(tmp_path, builder)


def make_folder(tmp_path):
    folder = tmp_path / "clouds"
    folder.mkdir()
    np.random.default_rng(0).uniform(-5, 5, (20, 4)).astype("<f4").tofile(folder / "000000.bin")
    return folder


def test_open_folder_builds_a_ready_session(mgr, builder, tmp_path):
    sid = mgr.open_folder(str(make_folder(tmp_path)))
    assert mgr.job(sid).status == "ready"
    assert builder.calls[0].folder.endswith("clouds")
    assert mgr.service(sid).state()["num_frames"] == 2


def test_open_folder_rejects_bad_paths_readably(mgr, tmp_path):
    with pytest.raises(FileNotFoundError):
        mgr.open_folder(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        mgr.open_folder(str(empty))


def test_failed_build_is_logged_with_traceback_and_removed(tmp_path, caplog):
    mgr = manager(tmp_path, Builder(tmp_path, fail="no GPU"))
    with caplog.at_level(logging.ERROR):
        sid = mgr.open_folder(str(make_folder(tmp_path)))
    assert sid not in mgr.jobs
    assert "no GPU" in caplog.text and "Traceback" in caplog.text
    with pytest.raises(LookupError):
        mgr.service(sid)


def test_second_tab_on_an_open_session_is_refused(mgr, tmp_path):
    folder = str(make_folder(tmp_path))
    a, b = mgr.open_folder(folder), mgr.open_folder(folder)
    assert mgr.job(b).status == "error"
    assert "already open in another tab" in mgr.job(b).message and a in mgr.job(b).message
    with pytest.raises(LookupError):
        mgr.service(b)
    assert mgr.service(a).act("add", frame=0, label="Car")["stats"]["manual"] == 1   # the first tab keeps working


def test_session_can_be_opened_again_after_finish(mgr, tmp_path):
    folder = str(make_folder(tmp_path))
    a = mgr.open_folder(folder)
    mgr.finish(a)
    b = mgr.open_folder(folder)
    assert mgr.job(b).status == "ready" and mgr.service(b).act("add", frame=0, label="Car")["stats"]["manual"] == 1


def test_other_sessions_are_not_refused(mgr, tmp_path):
    a = mgr.open_folder(str(make_folder(tmp_path)))
    other = tmp_path / "x"
    other.mkdir()
    b = mgr.open_folder(str(make_folder(other)))
    assert mgr.job(a).status == mgr.job(b).status == "ready"


def test_finish_saves_and_removes_the_job(mgr, tmp_path):
    sid = mgr.open_folder(str(make_folder(tmp_path)))
    svc = mgr.service(sid)
    svc.act("accept", frame=0, index=0)
    mgr.finish(sid)
    assert sid not in mgr.jobs
    assert mgr.store.load(svc.key).stats()["accepted"] == 1
    assert mgr.store.locked_by(svc.key) is None


def test_finished_session_accepts_no_more_requests(mgr, tmp_path):
    sid = mgr.open_folder(str(make_folder(tmp_path)))
    key = mgr.service(sid).key
    mgr.finish(sid)
    saved = mgr.store.path(key).read_text()
    for call in (lambda: mgr.service(sid), lambda: mgr.heartbeat(sid), lambda: mgr.points(sid, 0)):
        with pytest.raises(LookupError, match="finished"):
            call()                                  # not KeyError: the API answers 409, not 404
    assert mgr.store.path(key).read_text() == saved


def test_no_review_service_is_kept_between_requests(mgr, tmp_path):
    sid = mgr.open_folder(str(make_folder(tmp_path)))
    a, b = mgr.service(sid), mgr.service(sid)
    assert a is not b and a.session is not b.session
    a.act("accept", frame=0, index=0)
    assert mgr.service(sid).state(0)["stats"]["accepted"] == 1    # every request reads the store


def test_lock_expires_without_heartbeats_and_the_old_tab_gets_409(tmp_path):
    clock = Clock()
    mgr = manager(tmp_path, Builder(tmp_path), clock)
    folder = str(make_folder(tmp_path))
    a = mgr.open_folder(folder)
    clock.t += 100
    mgr.heartbeat(a)                                # keeps it
    clock.t += 100
    assert mgr.job(mgr.open_folder(folder)).status == "error"     # still held (100 s since the heartbeat)
    clock.t += 30                                   # 130 s without a heartbeat: the tab was closed
    b = mgr.open_folder(folder)
    assert mgr.job(b).status == "ready"
    with pytest.raises(LookupError, match="open in another tab"):
        mgr.service(a)                              # the old tab learns it lost the session
    mgr.service(b).act("add", frame=0, label="Car")


def test_reopening_a_finished_folder_gives_a_fresh_service(mgr, tmp_path):
    folder = str(make_folder(tmp_path))
    a = mgr.open_folder(folder)
    old = mgr.service(a)
    mgr.finish(a)
    c = mgr.open_folder(folder)
    assert mgr.service(c) is not old
    mgr.service(c).act("add", frame=0, label="Car")   # and the fresh one works
