import logging

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from tests.test_web.test_service import FakeLoader
from verification.session import AnnotationSession
from web.manager import SessionManager


def box(x=10.0, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


class Builder:
    """Stands in for scripts.verify.build_session: no detector, no data."""

    def __init__(self, tmp_path, fail=None):
        self.tmp, self.calls, self.fail = tmp_path, [], fail

    def __call__(self, args):
        self.calls.append(args)
        if self.fail:
            raise RuntimeError(self.fail)
        name = f"{args.folder or args.sequence}_{args.mode}_{args.start}"
        return AnnotationSession.from_tracking("0000", [[box()], [box(11)]], args.mode), \
            FakeLoader(), self.tmp / f"{name}.json"


@pytest.fixture
def builder(tmp_path):
    return Builder(tmp_path)


@pytest.fixture
def mgr(tmp_path, builder):
    return SessionManager(builder=builder, background=False)


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
    mgr = SessionManager(builder=Builder(tmp_path, fail="no GPU"), background=False)
    with caplog.at_level(logging.ERROR):
        sid = mgr.open_folder(str(make_folder(tmp_path)))
    assert sid not in mgr.jobs
    assert "no GPU" in caplog.text and "Traceback" in caplog.text
    with pytest.raises(LookupError):
        mgr.service(sid)


def test_same_session_is_shared_not_duplicated(mgr, tmp_path):
    folder = str(make_folder(tmp_path))
    a, b = mgr.open_folder(folder), mgr.open_folder(folder)
    assert a != b and mgr.service(a) is mgr.service(b)


def test_finish_saves_and_removes_the_job(mgr, tmp_path):
    sid = mgr.open_folder(str(make_folder(tmp_path)))
    svc = mgr.service(sid)
    svc.act("select", index=0); svc.act("accept")
    mgr.finish(sid)
    assert sid not in mgr.jobs
    assert svc.save_path.exists()


def test_finished_session_accepts_no_more_actions(mgr, tmp_path):
    sid = mgr.open_folder(str(make_folder(tmp_path)))
    svc = mgr.service(sid)
    mgr.finish(sid)
    saved = svc.save_path.read_text()
    with pytest.raises(LookupError):
        mgr.service(sid)
    with pytest.raises(LookupError):
        svc.act("add")                              # a stale reference cannot change it either
    assert svc.save_path.read_text() == saved


def test_finish_closes_the_session_for_every_tab(mgr, tmp_path):
    folder = str(make_folder(tmp_path))
    a, b = mgr.open_folder(folder), mgr.open_folder(folder)
    mgr.finish(a)
    with pytest.raises(LookupError, match="closed"):
        mgr.service(b)


def test_reopening_a_finished_folder_gives_a_fresh_service(mgr, tmp_path):
    folder = str(make_folder(tmp_path))
    a = mgr.open_folder(folder)
    old = mgr.service(a)
    mgr.finish(a)
    c = mgr.open_folder(folder)
    assert mgr.service(c) is not old
    mgr.service(c).act("add")                       # and the fresh one works
