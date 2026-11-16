"""PostgreSQL: migrations, JSON session import, and the web layer on PgSessionStore (marker `postgres`)."""
import json

import numpy as np
import pytest

from common.boxes import BoundingBox3D
from infra.pg_sessions import PgSessionStore
from infra.sessions import RevisionConflict
from verification.session import AnnotationSession

pytestmark = pytest.mark.postgres


def box(x, tid=0):
    return BoundingBox3D(x, 0.0, -1.0, 4.0, 1.8, 1.5, 0.0, "Car", 0.9, tid)


def test_migrations_match_the_models_and_downgrade_cleanly(pg_url):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine, inspect

    from infra.db import Base
    from infra.migrate import downgrade, upgrade

    engine = create_engine(pg_url)
    with engine.connect() as c:
        diff = compare_metadata(MigrationContext.configure(c), Base.metadata)
    assert diff == [], diff
    downgrade(pg_url, "base")
    assert set(inspect(create_engine(pg_url)).get_table_names()) <= {"alembic_version"}
    upgrade(pg_url)
    assert {"users", "sequences", "jobs", "sessions", "session_events"} <= set(inspect(engine).get_table_names())


def test_import_json_sessions(pg_engine, tmp_path):
    from scripts.import_sessions import import_folder

    s = AnnotationSession.from_tracking("0000", [[box(10)], [box(11)]])
    s.accept(0, 0)
    s.save(tmp_path / "a_assisted.json")
    v1 = json.loads((tmp_path / "a_assisted.json").read_text())
    for k in ("revision", "undo", "redo"):
        del v1[k]
    v1["version"] = 1
    (tmp_path / "old_assisted.json").write_text(json.dumps(v1))
    (tmp_path / "broken.json").write_text("{not json")

    store = PgSessionStore(pg_engine)
    out = import_folder(tmp_path, store)
    assert out["imported"] == ["a_assisted", "old_assisted"] and list(out["failed"]) == ["broken"]
    a = store.load("a_assisted")
    assert a.revision == 1 and a.undo() == "accept"                 # v2: history imported
    assert [e.action for e in store.events("a_assisted")] == [e.action for e in s.events]
    assert store.load("old_assisted").revision == 0                  # v1: revision 0
    again = import_folder(tmp_path, store)
    assert again["imported"] == [] and again["skipped"] == ["a_assisted", "old_assisted"]
    assert import_folder(tmp_path, store, replace=True)["imported"] == ["a_assisted", "old_assisted"]
    assert len(store.events("a_assisted")) == len(s.events)          # replaced, not doubled


@pytest.fixture
def web(pg_engine, tmp_path):
    from fastapi.testclient import TestClient

    from tests.test_web.test_manager import Builder, make_folder
    from web.manager import SessionManager
    from web.server import create_app

    mgr = SessionManager(builder=Builder(tmp_path), background=False, store=PgSessionStore(pg_engine))
    client = TestClient(create_app(mgr))
    sid = client.post("/api/folder/open", json={"path": str(make_folder(tmp_path))}).json()["sid"]
    return client, mgr, sid, tmp_path


def test_web_flow_on_postgres(web):
    client, mgr, sid, tmp = web
    post = lambda **kw: client.post(f"/api/s/{sid}/action", json=kw)
    assert post(name="accept", frame=0, index=0).json()["stats"]["accepted"] == 1
    assert post(name="view", frame=1).json()["revision"] == 1         # looking changes nothing
    assert post(name="undo", frame=0).json()["stats"]["accepted"] == 0
    assert client.post(f"/api/s/{sid}/heartbeat").json()["ok"] is True
    other = client.post("/api/folder/open", json={"path": str(tmp / "clouds")}).json()["sid"]
    assert client.get(f"/api/jobs/{other}").json()["status"] == "error"
    assert client.post(f"/api/s/{sid}/finish").status_code == 200
    key = next(iter(mgr.store.list()))
    assert mgr.store.status(key) == "finished" and mgr.store.locked_by(key) is None
    assert post(name="accept", frame=0, index=0).status_code == 409


def test_revision_conflict_is_409(web, monkeypatch):
    client, mgr, sid, _ = web
    real_load = mgr.store.load

    def load_then_someone_else_saves(key):
        s = real_load(key)
        other = real_load(key)
        other.reject(1, 0)
        mgr.store.save(key, other, expected_revision=other.revision - 1)
        return s                                                       # our copy is now stale

    monkeypatch.setattr(mgr.store, "load", load_then_someone_else_saves)
    r = client.post(f"/api/s/{sid}/action", json={"name": "accept", "frame": 0, "index": 0})
    assert r.status_code == 409 and "changed elsewhere" in r.json()["detail"]
    monkeypatch.setattr(mgr.store, "load", real_load)
    st = client.get(f"/api/s/{sid}/state", params={"frame": 1}).json()
    assert st["annotations"][0]["status"] == "rejected"                # the other write was kept
    assert client.get(f"/api/s/{sid}/state").json()["annotations"][0]["status"] == "pending"


def test_concurrent_saves_one_wins(pg_engine):
    store = PgSessionStore(pg_engine)
    store.create("k", AnnotationSession.from_tracking("0000", [[box(10)], [box(11)]]))
    a, b = store.load("k"), store.load("k")
    a.accept(0, 0)
    b.reject(0, 0)
    store.save("k", a, expected_revision=0)
    with pytest.raises(RevisionConflict):
        store.save("k", b, expected_revision=0)
    assert store.load("k").annotations(0)[0].status.value == "accepted"


def test_points_do_not_need_the_document(web, monkeypatch):
    client, mgr, sid, _ = web

    def no_load(key):
        raise AssertionError("points must not load the session document")

    monkeypatch.setattr(mgr.store, "load", no_load)
    r = client.get(f"/api/s/{sid}/points/0")
    assert r.status_code == 200 and np.frombuffer(r.content, "<f4").size == 150


@pytest.fixture
def fresh_db_url():
    """An empty throw-away database (for migration tests that start below head)."""
    import os
    import uuid

    from sqlalchemy import create_engine, text

    from tests.conftest import DEFAULT_ADMIN_URL

    admin_url = os.environ.get("LAL_TEST_DATABASE_URL", DEFAULT_ADMIN_URL)
    name = f"lal_mig_{uuid.uuid4().hex[:8]}"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    yield admin_url.rsplit("/", 1)[0] + "/" + name
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


def test_migration_0002_moves_documents_into_rows_and_back(fresh_db_url):
    from sqlalchemy import create_engine, text

    from infra.migrate import downgrade, upgrade

    upgrade(fresh_db_url, "0001")
    s = AnnotationSession.from_tracking("0000", [[box(10), box(20, 1)], [box(11)], [], [box(13)], []])
    s.accept(0, 0)
    s.accept_track(1)
    s.edit(3, 0, x=99.0)
    s.undo()                                                         # one step on the redo stack
    phase2_doc = s.to_dict(include_events=False)                     # the whole document of phase 2
    engine = create_engine(fresh_db_url)
    with engine.begin() as c:
        c.execute(text("INSERT INTO sessions (key, document, revision, status) "
                       "VALUES ('k', CAST(:d AS jsonb), :r, 'open')"), {"d": json.dumps(phase2_doc), "r": s.revision})

    upgrade(fresh_db_url, "0002")
    with engine.connect() as c:
        meta = c.execute(text("SELECT document FROM sessions WHERE key = 'k'")).scalar_one()
        frames = c.execute(text("SELECT frame FROM session_frames WHERE session_key = 'k' ORDER BY frame")).scalars().all()
        stacks = c.execute(text("SELECT stack, label FROM session_history WHERE session_key = 'k' ORDER BY id")).all()
    assert set(meta) == {"version", "sequence", "mode", "frame_offset", "num_frames", "next_track_id"}
    assert frames == [0, 1, 2, 3]       # frame 2: the interpolated box of track 0; empty frame 4 has no row
    assert [tuple(r) for r in stacks] == [("undo", "accept"), ("undo", "accept_track"), ("redo", "edit")]
    loaded = PgSessionStore(engine).load("k")
    assert loaded.to_dict(include_events=False)["frames"] == phase2_doc["frames"]
    assert loaded.revision == s.revision and loaded.redo() == "edit"

    downgrade(fresh_db_url, "0001")
    with engine.connect() as c:
        back = c.execute(text("SELECT document FROM sessions WHERE key = 'k'")).scalar_one()
        tables = set(c.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).scalars())
    assert back == phase2_doc                                         # the whole document again
    assert "session_frames" not in tables and "session_history" not in tables
    engine.dispose()


def test_an_action_writes_only_the_frames_it_changed(pg_engine):
    from sqlalchemy import event

    store = PgSessionStore(pg_engine)
    store.create("k", AnnotationSession.from_tracking("0000", [[box(10 + i)] for i in range(50)]))
    s = store.load("k")
    s.accept(7, 0)
    frame_rows = []

    def count(conn, cursor, statement, params, context, executemany):
        if "session_frames" in statement and statement.lstrip().upper().startswith("INSERT"):
            frame_rows.append(len(params) if executemany else 1)

    event.listen(pg_engine, "before_cursor_execute", count)
    try:
        store.save("k", s, expected_revision=0)
    finally:
        event.remove(pg_engine, "before_cursor_execute", count)
    assert sum(frame_rows) == 1                                       # frame 7 only
    assert store.load("k").annotations(7)[0].status.value == "accepted"


def test_import_writes_frames_and_history_rows(pg_engine, tmp_path):
    from sqlalchemy import text

    from scripts.import_sessions import import_folder

    s = AnnotationSession.from_tracking("0000", [[box(10)], [box(11)], []])
    s.accept_track(0)
    s.save(tmp_path / "x_assisted.json")
    import_folder(tmp_path, PgSessionStore(pg_engine))
    with pg_engine.connect() as c:
        frames = c.execute(text("SELECT frame FROM session_frames WHERE session_key = 'x_assisted' "
                                "ORDER BY frame")).scalars().all()
        history = c.execute(text("SELECT stack, label FROM session_history WHERE session_key = 'x_assisted'")).all()
    assert frames == [0, 1] and [tuple(h) for h in history] == [("undo", "accept_track")]
