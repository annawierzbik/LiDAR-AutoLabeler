"""PgJobStore: job states in the jobs table (marker `postgres`)."""
import pytest

from infra.celery_queue import CeleryTaskQueue
from infra.pg_jobs import PgJobStore

pytestmark = pytest.mark.postgres


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def jobs(pg_engine):
    clock = Clock()
    return PgJobStore(pg_engine, stale_seconds=30, max_attempts=2, clock=clock), clock


def test_create_dedupes_until_the_job_failed(jobs):
    store, _ = jobs
    a, new = store.create("detect_track", "gpu", {"x": 1}, "t", "key1")
    assert new and store.status(a).state == "queued" and store.status(a).kind == "detect_track"
    assert store.create("detect_track", "gpu", {"x": 1}, "t", "key1") == (a, False)
    store.start(a, "w1")
    store.fail(a, "boom")
    b, new = store.create("detect_track", "gpu", {"x": 1}, "t", "key1")
    assert new and b != a                                           # a failed job may be run again


def test_only_one_worker_starts_a_job(jobs):
    store, _ = jobs
    job, _ = store.create("export", "cpu", {"y": 2}, "", "k")
    payload = store.start(job, "w1")
    assert payload == {"y": 2, "job_id": job}
    assert store.start(job, "w2") is None                           # a duplicate delivery runs nothing
    st = store.status(job)
    assert st.state == "running" and st.attempts == 1


def test_progress_heartbeat_and_finish(jobs):
    store, _ = jobs
    job, _ = store.create("export", "cpu", {}, "", "k")
    store.start(job, "w1")
    store.heartbeat(job, 0.4, "rendering")
    st = store.status(job)
    assert (st.progress, st.message) == (0.4, "rendering")
    store.finish(job, {"blob": "x"})
    st = store.status(job)
    assert st.state == "done" and st.progress == 1.0 and st.result == {"blob": "x"}
    store.heartbeat(job, 0.1)                                        # a late heartbeat changes nothing
    assert store.status(job).progress == 1.0


def test_cancel_queued_and_running(jobs):
    store, _ = jobs
    queued, _ = store.create("export", "cpu", {}, "", "k1")
    assert store.request_cancel(queued) and store.status(queued).state == "cancelled"
    assert store.start(queued, "w") is None
    running, _ = store.create("export", "cpu", {}, "", "k2")
    store.start(running, "w")
    assert store.request_cancel(running) and store.status(running).state == "cancelling"
    assert store.cancel_requested(running)
    store.cancelled(running, "stopped after 3 of 9 frames")
    assert store.status(running).state == "cancelled" and not store.request_cancel(running)


def test_a_lost_worker_requeues_then_fails_the_job(jobs):
    store, clock = jobs
    job, _ = store.create("detect_track", "gpu", {}, "", "k")
    store.start(job, "w1")
    clock.t += 10
    assert store.recover_stale() == []                              # heartbeat 10 s old: alive
    clock.t += 25
    assert store.recover_stale() == [(job, "gpu")]                  # 35 s: lost -> queued again
    st = store.status(job)
    assert st.state == "queued" and "w1 stopped responding" in st.message
    store.start(job, "w2")                                          # second attempt
    clock.t += 40
    assert store.recover_stale() == []                              # max_attempts reached
    st = store.status(job)
    assert st.state == "failed" and "gave up after 2 attempts" in st.message


def test_a_lost_worker_of_a_cancelling_job_ends_it(jobs):
    store, clock = jobs
    job, _ = store.create("detect_track", "gpu", {}, "", "k")
    store.start(job, "w1")
    store.request_cancel(job)
    clock.t += 60
    store.recover_stale()
    assert store.status(job).state == "cancelled"


def test_celery_queue_sends_new_jobs_and_resends_lost_ones(jobs):
    store, clock = jobs
    sent = []
    q = CeleryTaskQueue(store, send=lambda job_id, queue: sent.append((job_id, queue)))
    from infra.tasks import detect_track_payload
    from pipeline.session_builder import make_args

    p = detect_track_payload(make_args(folder="D:/x"))
    job = q.submit("detect_track", p, "x")
    assert sent == [(job, "gpu")] and q.submit("detect_track", p, "x") == job and len(sent) == 1
    exp = q.submit("export", {"session_key": "k", "revision": 1}, "e")
    assert sent[-1] == (exp, "cpu")
    store.start(job, "w1")
    clock.t += 31
    assert q.status(job).state == "queued" and sent[-1] == (job, "gpu")   # noticed on status, sent again
    assert q.cancel(job) and q.status(job).state == "cancelled"


def test_two_concurrent_requests_get_one_job(jobs, pg_engine):
    """Both requests look for an active job before either inserts (the race of a check-then-insert);
    the partial unique index (migration 0005) lets one insert win and the other gets that job."""
    import threading

    from sqlalchemy import func, select

    from infra.db import JobRow

    store, _ = jobs
    both_looked = threading.Barrier(2, timeout=10)
    real = PgJobStore._existing
    calls = []

    def existing(c, dedupe, reuse_done=None):
        found = real(c, dedupe, reuse_done)
        calls.append(found)
        if len(calls) <= 2:
            both_looked.wait()                                       # neither has inserted yet
        return found

    results = []
    store._existing = existing
    threads = [threading.Thread(target=lambda: results.append(store.create("export", "cpu", {"z": 3}, "", "kc")))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    assert calls[:2] == [None, None]                                 # both saw no job: a real race
    assert sorted(new for _, new in results) == [False, True]
    assert results[0][0] == results[1][0]                             # the second request got the first job
    with pg_engine.connect() as c:
        assert c.execute(select(func.count()).select_from(JobRow).where(JobRow.cache_key == "kc")).scalar() == 1


def test_the_database_refuses_a_second_active_job(jobs, pg_engine):
    import sqlalchemy
    from sqlalchemy import insert

    from infra.db import JobRow

    store, _ = jobs
    a, _ = store.create("export", "cpu", {}, "", "kd")
    with pytest.raises(sqlalchemy.exc.IntegrityError), pg_engine.begin() as c:
        c.execute(insert(JobRow).values(id="other", kind="export", queue="cpu", state="running", cache_key="kd"))
    store.start(a, "w")
    store.finish(a, {})
    with pg_engine.begin() as c:                                     # a finished job does not count
        c.execute(insert(JobRow).values(id="later", kind="export", queue="cpu", state="queued", cache_key="kd"))


def test_migration_0005_keeps_the_newest_of_duplicate_active_jobs(pg_url, pg_engine):
    import datetime as dt

    from sqlalchemy import insert, select

    from infra.db import JobRow
    from infra.migrate import downgrade, upgrade

    downgrade(pg_url, "0004")
    try:
        t0 = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        with pg_engine.begin() as c:
            for i, state in enumerate(("queued", "running", "queued")):
                c.execute(insert(JobRow).values(id=f"d{i}", kind="export", queue="cpu", state=state, cache_key="dup",
                                                created_at=t0 + dt.timedelta(minutes=i)))
    finally:
        upgrade(pg_url)
    with pg_engine.connect() as c:
        rows = dict(c.execute(select(JobRow.id, JobRow.state).where(JobRow.cache_key == "dup")).all())
        msg = c.execute(select(JobRow.message).where(JobRow.id == "d0")).scalar()
    assert rows == {"d0": "failed", "d1": "failed", "d2": "queued"} and "superseded by job d2" in msg


def test_a_done_job_is_reused_only_while_reuse_done_accepts_its_result(jobs):
    store, _ = jobs
    a, _ = store.create("export", "cpu", {"r": 1}, "", "kr")
    store.start(a, "w")
    store.finish(a, {"blob": "x", "ok": True})
    assert store.create("export", "cpu", {"r": 1}, "", "kr", reuse_done=lambda r: r["ok"]) == (a, False)
    b, new = store.create("export", "cpu", {"r": 1}, "", "kr", reuse_done=lambda r: False)
    assert new and b != a                                             # the result can no longer serve
    assert store.create("export", "cpu", {"r": 1}, "", "kr", reuse_done=lambda r: False) == (b, False)  # b is queued
