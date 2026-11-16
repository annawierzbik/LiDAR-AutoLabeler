import numpy as np
import pytest

from common.boxes import BoundingBox3D
from infra.points import LoaderPointSource, PointCache
from infra.sessions import FileSessionStore, RevisionConflict
from verification.session import AnnotationSession
from web.service import ReviewService, UnknownAction


class FakeFrame:
    def __init__(self, n):
        rng = np.random.default_rng(n)
        self.xyz = rng.uniform(-20, 20, (50, 3))


class FakeLoader:
    def __getitem__(self, i):
        return FakeFrame(i)

    def __len__(self):
        return 3


def box(x, tid, label="Car"):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, label, 0.9, tid)


RESULTS = [[box(10, 0), box(30, 1, "Pedestrian")], [box(11, 0), box(31, 1, "Pedestrian")], [box(12, 0)]]


class Env:
    """A store with one session "s"; `svc()` is the ReviewService of one request (loaded fresh)."""

    def __init__(self, tmp_path, results=RESULTS, clock=None):
        self.store = FileSessionStore(tmp_path)
        kwargs = {"clock": clock} if clock else {}
        self.store.create("s", AnnotationSession.from_tracking("0000", results, **kwargs))
        self.cache = PointCache(8)

    def svc(self):
        return ReviewService(self.store.load("s"), self.store, "s", LoaderPointSource(FakeLoader()),
                             sequence=FakeLoader(), point_cache=self.cache)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def statuses(state):
    return [a["status"] for a in state["annotations"]]


def test_state_of_a_frame(env):
    st = env.svc().state(0)
    assert st["frame"] == 0 and st["num_frames"] == 3 and st["revision"] == 0
    assert [(a["i"], a["label"]) for a in st["annotations"]] == [(0, "Car"), (1, "Pedestrian")]
    assert st["stats"]["total"] == 5 and set(st["classes"]) == {"Car", "Pedestrian", "Cyclist"}
    assert len(env.svc().state(2)["annotations"]) == 1
    assert "selected" not in st and "current_class" not in st and "show_rejected" not in st
    with pytest.raises(ValueError, match="out of range"):
        env.svc().state(3)


def test_actions_name_their_target_and_are_stored(env):
    st = env.svc().act("accept", frame=1, index=1)
    assert st["frame"] == 1 and statuses(st) == ["pending", "accepted"]
    assert statuses(env.svc().state(1)) == ["pending", "accepted"]   # a later request sees it
    assert statuses(env.svc().state(0)) == ["pending", "pending"]
    assert env.store.load("s").revision == 1


def test_box_actions_need_frame_and_index(env):
    svc = env.svc()
    with pytest.raises(ValueError, match="frame"):
        svc.act("accept", index=0)
    with pytest.raises(ValueError, match="index"):
        svc.act("accept", frame=0)
    with pytest.raises(ValueError, match="no box 5"):
        svc.act("reject", frame=0, index=5)
    with pytest.raises(UnknownAction):
        svc.act("select", frame=0, index=0)                          # selection is the browser's business


def test_nudge_and_resize(env):
    st = env.svc().act("forward", frame=0, index=0)
    a = st["annotations"][0]
    assert a["x"] == pytest.approx(10.1) and a["status"] == "edited" and st["result"] == {"changed": True}
    st = env.svc().act("length_plus", frame=0, index=0)
    assert st["annotations"][0]["dx"] == pytest.approx(4.1)


def test_track_actions_and_undo_redo_across_requests(env):
    st = env.svc().act("accept_track", frame=0, track_id=0)
    assert st["result"] == {"count": 3} and st["stats"]["accepted"] == 3
    with pytest.raises(ValueError, match="track_id"):
        env.svc().act("reject_track", frame=0)
    assert env.svc().act("undo", frame=0)["stats"]["accepted"] == 0   # history is in the stored document
    assert env.svc().act("redo", frame=0)["stats"]["accepted"] == 3


def test_add_and_copy_next_report_where_the_box_is(env):
    st = env.svc().act("add", frame=0, label="Cyclist")
    assert st["result"] == {"frame": 0, "index": 2}
    assert st["annotations"][2]["status"] == "manual" and st["annotations"][2]["label"] == "Cyclist"
    st = env.svc().act("copy_next", frame=0, index=2)
    assert st["frame"] == 1 and st["result"]["frame"] == 1
    assert st["annotations"][st["result"]["index"]]["label"] == "Cyclist"
    assert env.svc().act("copy_next", frame=2, index=0)["result"] == {"copied": False}   # last frame


def test_labels_are_validated(env):
    st = env.svc().act("set_label", frame=0, index=0, label="Cyclist")
    assert st["annotations"][0]["label"] == "Cyclist"
    with pytest.raises(ValueError, match="unknown class"):
        env.svc().act("set_label", frame=0, index=0, label="Spaceship")
    with pytest.raises(ValueError, match="unknown class"):
        env.svc().act("add", frame=0)


def test_rejected_boxes_stay_in_the_state(env):
    st = env.svc().act("reject", frame=0, index=0)
    assert statuses(st) == ["rejected", "pending"]                    # the browser decides what to show


@pytest.mark.parametrize("name", ["view", "playback_start", "playback_tick", "playback_stop"])
def test_log_only_actions_append_an_event_and_keep_the_revision(env, name):
    n = len(env.store.events("s"))
    st = env.svc().act(name, frame=2)
    assert st["frame"] == 2 and st["revision"] == 0 and env.store.load("s").revision == 0
    event = env.store.events("s")[n]
    assert event.action == ("frame_change" if name == "view" else name) and event.frame == 2


def test_log_only_actions_do_not_conflict_with_an_edit(env):
    stale = env.svc()                                                # loaded at revision 0
    env.svc().act("accept", frame=0, index=0)                        # stored revision 1
    stale.act("view", frame=1)                                       # append-only: no conflict
    assert [e.action for e in env.store.events("s")][-2:] == ["accept", "frame_change"]


def test_next_pending(env):
    assert env.svc().next_pending(0) == 1
    env.svc().act("accept_track", frame=0, track_id=0)
    env.svc().act("accept_track", frame=0, track_id=1)
    assert env.svc().next_pending(0) is None


def test_points_bytes_and_bounds(env):
    data = env.svc().points(1)
    assert len(data) == 50 * 3 * 4
    assert np.frombuffer(data, "<f4").shape == (150,)
    assert env.cache.get(("s", 1)) == data                           # shared between requests
    with pytest.raises(IndexError):
        env.svc().points(3)


def test_queries_store_nothing(env):
    before = env.store.path("s").read_text()
    svc = env.svc()
    svc.state(1)
    svc.next_pending(0)
    assert env.store.path("s").read_text() == before


def test_failed_save_is_reported_in_state(env, monkeypatch):
    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(env.store, "save", broken)
    st = env.svc().act("accept", frame=0, index=0)
    assert "disk full" in st["save_error"]


def test_concurrent_writer_is_a_conflict_not_an_overwrite(env):
    stale = env.svc()                                                # loaded at revision 0
    env.svc().act("accept", frame=0, index=0)                        # stored revision 1
    with pytest.raises(RevisionConflict):
        stale.act("reject", frame=1, index=0)
    assert statuses(env.svc().state(0))[0] == "accepted" and statuses(env.svc().state(1))[0] == "pending"


def test_frame_marks_follow_review_progress(env):
    assert env.svc().state(0)["frame_marks"] == [1, 1, 1]
    env.svc().act("accept_track", frame=0, track_id=0)
    st = env.svc().act("accept_track", frame=0, track_id=1)
    assert st["frame_marks"] == [2, 2, 2]


def test_playback_counts_as_active_time(tmp_path):
    """Deliberate change (docs/adr/0013): playback logs a start, a tick every 10 s and a stop, so
    60 s of watching count as ~60 s of active time. The frames shown are not logged one by one."""
    now = [1000.0]

    def clock():
        return now[0]

    e = Env(tmp_path, results=[[box(10 + i * 0.1, 0)] for i in range(241)], clock=clock)

    def request():
        svc = ReviewService(e.store.load("s"), e.store, "s", LoaderPointSource(FakeLoader()))
        svc.session._clock = clock                                   # the store loads with the real clock
        return svc

    def at(t, name, frame):
        now[0] = 1000.0 + t
        request().act(name, frame=frame)

    n = len(e.store.events("s"))
    at(0.0, "playback_start", 0)
    for step in range(1, 241):                                       # the browser: a frame every 250 ms ...
        now[0] = 1000.0 + step * 0.25
        request().state(step)                                        # ... read without logging
        if step % 40 == 0 and step < 240:                            # ... and a tick every 10 s
            at(step * 0.25, "playback_tick", step)
    at(60.0, "playback_stop", 240)
    events = e.store.events("s")[n:]
    assert [ev.action for ev in events] == ["playback_start"] + ["playback_tick"] * 5 + ["playback_stop"]
    assert e.store.load("s").revision == 0
    probe = AnnotationSession("x", 1)
    probe.events = events
    assert probe.active_seconds() == pytest.approx(60.0)
