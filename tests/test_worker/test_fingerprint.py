"""The GPU model fingerprint in Redis (marker `worker`: the Redis of docker-compose.db.yml): it has
a TTL, a heartbeat keeps it alive, and it disappears when the heartbeat stops."""
from __future__ import annotations

import time

import pytest

from tests.test_worker.test_celery_worker import REDIS_URL

pytestmark = [pytest.mark.worker]


@pytest.fixture
def celery_mode(monkeypatch):
    import redis

    import infra.worker_detector as wd
    from infra.settings import get_settings

    if wd.worker_config()["checkpoint_hash"] is None:
        pytest.skip("needs the weights on disk (their hash is the fingerprint)")
    monkeypatch.setenv("LAL_JOB_QUEUE", "celery")
    monkeypatch.setenv("LAL_REDIS_URL", REDIS_URL)
    get_settings.cache_clear()
    r = redis.Redis.from_url(REDIS_URL)
    r.delete(wd.FINGERPRINT_KEY)
    yield wd, r
    r.delete(wd.FINGERPRINT_KEY)
    get_settings.cache_clear()


def test_the_fingerprint_expires_without_a_heartbeat(celery_mode):
    wd, r = celery_mode
    fp = wd.publish_fingerprint(ttl=0.5)
    assert wd.published_fingerprint() == fp and 0 < r.pttl(wd.FINGERPRINT_KEY) <= 500
    time.sleep(0.7)
    assert wd.published_fingerprint() is None and not r.exists(wd.FINGERPRINT_KEY)


def test_a_heartbeat_keeps_it_until_the_worker_stops(celery_mode):
    wd, _ = celery_mode
    beat = wd.start_fingerprint_heartbeat(ttl=0.6)
    try:
        time.sleep(1.8)                                                   # three TTLs
        assert wd.published_fingerprint() is not None                     # refreshed every 0.2 s
    finally:
        beat.stop.set()                                                   # the worker stops
        beat.join(2)
    time.sleep(0.8)
    assert wd.published_fingerprint() is None
