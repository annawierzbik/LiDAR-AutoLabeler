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
    from web.manager import make_args

    p = detect_track_payload(make_args(folder="D:/x"))
    job = q.submit("detect_track", p, "x")
    assert sent == [(job, "gpu")] and q.submit("detect_track", p, "x") == job and len(sent) == 1
    exp = q.submit("export", {"session_key": "k", "revision": 1}, "e")
    assert sent[-1] == (exp, "cpu")
    store.start(job, "w1")
    clock.t += 31
    assert q.status(job).state == "queued" and sent[-1] == (job, "gpu")   # noticed on status, sent again
    assert q.cancel(job) and q.status(job).state == "cancelled"
