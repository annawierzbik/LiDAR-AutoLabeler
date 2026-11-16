"""Randomised consistency of the session stores.

A reference session lives in memory with its whole history. Every step applies one random
operation (accept, reject, track accept / reject, edit, add, copy, fill gaps, undo, redo) the way
the API does: load the session from the store, apply the operation, save with the loaded revision.
After every step the session is loaded again and compared with the reference: frames, the newest
undo and redo step, revision and next track id. Fixed seeds, a few hundred steps each.
"""
import random

import pytest

from common.boxes import BoundingBox3D
from infra.sessions import FileSessionStore
from verification.session import EDITABLE_FIELDS, AnnotationSession

SEEDS = (1, 2, 3)
STEPS = 300
FRAMES, TRACKS = 12, 3
LABELS = ("Car", "Pedestrian", "Cyclist")


@pytest.fixture(params=["file", pytest.param("postgres", marks=pytest.mark.postgres)])
def store(request, tmp_path):
    if request.param == "file":
        return FileSessionStore(tmp_path)
    from infra.pg_sessions import PgSessionStore

    return PgSessionStore(request.getfixturevalue("pg_engine"))


def initial() -> AnnotationSession:
    rng = random.Random(0)
    results = []
    for f in range(FRAMES):
        frame = []
        for k in range(TRACKS):
            if rng.random() < 0.75:                                 # holes: interpolation has work to do
                frame.append(BoundingBox3D(5.0 + 4 * k + 0.3 * f, 2.0 * k, -1.0, 4.0, 1.8, 1.5, 0.1 * k,
                                           LABELS[k], 0.5 + 0.1 * k, k))
        results.append(frame)
    return AnnotationSession.from_tracking("0000", results)


def random_box(rng):
    return BoundingBox3D(rng.uniform(-20, 40), rng.uniform(-10, 10), -1.0, rng.uniform(0.5, 5),
                         rng.uniform(0.5, 2), rng.uniform(1, 2), rng.uniform(-3, 3), rng.choice(LABELS), 1.0)


def pick_box(rng, s):
    frames = [f for f in range(s.num_frames) if s.annotations(f)]
    if not frames:
        return None
    f = rng.choice(frames)
    return f, rng.randrange(len(s.annotations(f)))


def random_operation(rng, s):
    """(name, function applying it to a session) with arguments drawn from the session `s`."""
    kind = rng.choice(["accept", "reject", "accept_track", "reject_track", "edit", "edit", "add", "copy",
                       "fill_gaps", "undo", "undo", "redo"])
    if kind in ("undo", "redo", "fill_gaps"):
        return kind, lambda x: getattr(x, kind)()
    if kind == "add":
        f, box = rng.randrange(s.num_frames), random_box(rng)
        return kind, lambda x: x.add_manual(f, box)
    target = pick_box(rng, s)
    if target is None:
        return "undo", lambda x: x.undo()
    f, i = target
    if kind in ("accept", "reject"):
        return kind, lambda x: getattr(x, kind)(f, i)
    if kind in ("accept_track", "reject_track"):
        tid, start = s.annotations(f)[i].box.track_id, rng.choice([0, f])
        if tid is None:
            return "undo", lambda x: x.undo()
        return kind, lambda x: getattr(x, kind)(tid, start)
    if kind == "copy":
        to = rng.randrange(s.num_frames)
        return kind, lambda x: x.copy_to_frame(f, i, to)
    field = rng.choice(EDITABLE_FIELDS)
    value = rng.choice(LABELS) if field == "label" else (
        rng.uniform(0.3, 6) if field in ("dx", "dy", "dz") else rng.uniform(-30, 30))
    return f"edit {field}", lambda x: x.edit(f, i, **{field: value})


def assert_same(stored: AnnotationSession, ref: AnnotationSession, where: str) -> None:
    assert stored.revision == ref.revision, where
    assert stored._next_track_id == ref._next_track_id, where
    assert stored.frames == ref.frames, where
    for name in ("_undo", "_redo"):
        mine, theirs = getattr(stored, name), getattr(ref, name)
        assert bool(mine) == bool(theirs), f"{where}: {name} empty in one of them"
        if theirs:
            assert mine[-1] == theirs[-1], f"{where}: newest {name} step differs"


@pytest.mark.parametrize("seed", SEEDS)
def test_random_operations_round_trip_through_the_store(store, seed):
    rng = random.Random(seed)
    ref = initial()
    store.create("s", AnnotationSession.from_dict(ref.to_dict()))
    applied = {}
    for step in range(STEPS):
        name, op = random_operation(rng, ref)
        loaded = store.load("s")
        rev = loaded.revision
        try:
            op(loaded)                                               # the API's copy, from the store
        except (ValueError, IndexError):                             # e.g. a copy onto a frame that has the track
            continue
        op(ref)                                                      # the same on the reference
        if loaded.revision != rev:
            store.save("s", loaded, expected_revision=rev)
        applied[name.split()[0]] = applied.get(name.split()[0], 0) + 1
        assert_same(store.load("s"), ref, f"seed {seed} step {step} ({name})")
    assert {"accept", "reject", "accept_track", "reject_track", "edit", "add", "copy", "undo", "redo"} <= set(applied)
