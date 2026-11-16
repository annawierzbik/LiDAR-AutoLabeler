"""The SessionStore contract, run against FileSessionStore and (marker `postgres`) PgSessionStore."""
import pytest

from common.boxes import BoundingBox3D
from infra.sessions import (
    FileSessionStore,
    RevisionConflict,
    SessionExists,
    SessionLocked,
    SessionNotFound,
    SessionStore,
)
from verification.session import AnnotationSession, Event


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(params=["file", pytest.param("postgres", marks=pytest.mark.postgres)])
def store_and_clock(request, tmp_path):
    clock = Clock()
    if request.param == "file":
        return FileSessionStore(tmp_path, lock_ttl=120, clock=clock), clock
    from infra.pg_sessions import PgSessionStore

    return PgSessionStore(request.getfixturevalue("pg_engine"), lock_ttl=120, clock=clock), clock


@pytest.fixture
def store(store_and_clock):
    return store_and_clock[0]


def box(x, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


def session():
    return AnnotationSession.from_tracking("0000", [[box(10)], [box(11)], [box(12)]])


def test_is_a_session_store(store):
    assert isinstance(store, SessionStore)


def test_create_load_list(store):
    s = session()
    store.create("a", s)
    assert store.exists("a") and not store.exists("b") and store.list() == ["a"]
    loaded = store.load("a")
    assert loaded.num_frames == 3 and loaded.revision == 0 and loaded.events == []
    assert [e.action for e in store.events("a")] == [e.action for e in s.events]
    with pytest.raises(SessionExists):
        store.create("a", session())
    with pytest.raises(SessionNotFound):
        store.load("b")


def test_save_round_trip_with_history(store):
    store.create("a", session())
    s = store.load("a")
    s.accept(0, 0)
    s.edit(1, 0, x=20.0)
    store.save("a", s, expected_revision=0)
    r = store.load("a")
    assert r.revision == 2 and r.annotations(1)[0].box.x == 20.0
    assert r.undo() == "edit" and r.undo() == "accept"


def test_save_with_a_stale_revision_is_a_conflict(store):
    store.create("a", session())
    first, second = store.load("a"), store.load("a")
    first.accept(0, 0)
    store.save("a", first, expected_revision=0)
    second.reject(1, 0)
    with pytest.raises(RevisionConflict) as err:
        store.save("a", second, expected_revision=0)
    assert err.value.stored == 1 and err.value.expected == 0
    assert store.load("a").annotations(1)[0].status.value == "pending"     # not overwritten


def test_events_are_appended_without_a_revision_change(store):
    store.create("a", session())
    n = len(store.events("a"))
    store.append_events("a", [Event(5.0, "frame_change", 2), Event(6.0, "playback_tick", None, {"x": 1})])
    assert store.load("a").revision == 0
    evs = store.events("a")
    assert [(e.action, e.frame) for e in evs[n:]] == [("frame_change", 2), ("playback_tick", None)]
    assert evs[-1].detail == {"x": 1}
    s = store.load("a")
    s.accept(0, 0)                                       # logs "accept"
    store.save("a", s, expected_revision=0)              # events of the request are appended once
    assert [e.action for e in store.events("a")][n + 2:] == ["accept"]


def test_lock_is_exclusive_and_expires(store_and_clock):
    store, clock = store_and_clock
    store.create("a", session())
    store.lock("a", "tab1")
    with pytest.raises(SessionLocked) as err:
        store.lock("a", "tab2")
    assert err.value.owner == "tab1" and store.locked_by("a") == "tab1"
    clock.t += 100
    store.lock("a", "tab1")                              # heartbeat refreshes the lock
    clock.t += 100
    with pytest.raises(SessionLocked):
        store.lock("a", "tab2")                          # 100 s since the heartbeat: still held
    clock.t += 30
    assert store.locked_by("a") is None                  # 130 s without a heartbeat: expired
    store.lock("a", "tab2")
    assert store.locked_by("a") == "tab2"


def test_unlock_and_finish_release_only_for_the_owner(store):
    store.create("a", session())
    store.lock("a", "tab1")
    store.unlock("a", "tab2")
    assert store.locked_by("a") == "tab1"
    store.finish("a", "tab1")
    assert store.locked_by("a") is None
    store.lock("a", "tab2")


def test_unknown_session(store):
    with pytest.raises(SessionNotFound):
        store.load("zz")
    with pytest.raises(SessionNotFound):
        store.save("zz", session(), expected_revision=0)       # new sessions are made with create()
    with pytest.raises(SessionNotFound):
        store.append_events("zz", [Event(1.0, "frame_change", 0)])
