"""Shared fixtures. Database tests (marker `postgres`) use the PostgreSQL of docker-compose.db.yml:

    docker compose -f docker-compose.db.yml up -d
    pytest -m postgres

LAL_TEST_DATABASE_URL points at a database the tests may use to create and drop their own
throw-away database (default: the compose service).
"""
from __future__ import annotations

import os
import uuid

import pytest

DEFAULT_ADMIN_URL = "postgresql+psycopg://lal:lal_local@127.0.0.1:55432/lal"


@pytest.fixture(scope="session")
def pg_url():
    """URL of a fresh database at the newest migration, dropped after the test session."""
    from sqlalchemy import create_engine, text

    from infra.migrate import upgrade

    admin_url = os.environ.get("LAL_TEST_DATABASE_URL", DEFAULT_ADMIN_URL)
    name = f"lal_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # noqa: BLE001  (any connection problem: say how to start the database)
        pytest.fail(f"PostgreSQL not reachable at {admin_url} ({exc}); "
                    "start it with: docker compose -f docker-compose.db.yml up -d")
    url = admin_url.rsplit("/", 1)[0] + "/" + name
    upgrade(url)
    yield url
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def pg_engine(pg_url):
    """Engine on the test database; every table is emptied after the test."""
    from sqlalchemy import text

    from infra.db import make_engine

    engine = make_engine(pg_url)
    yield engine
    with engine.begin() as c:
        c.execute(text("TRUNCATE tabs, session_events, sessions, jobs, sequences, users RESTART IDENTITY CASCADE"))
    engine.dispose()


def gpu_requested(markexpr: str) -> bool:
    """True when the -m expression names the gpu marker positively (not only as "not gpu")."""
    import re

    return re.search(r"(?<!not )\bgpu\b", markexpr or "") is not None


def pytest_collection_modifyitems(config, items):
    """GPU tests run only when asked for by name (`-m gpu`, `-m "gpu and worker"`, ...). Another
    selection such as `-m worker` must not start the real model on a GPU that experiments may use."""
    if gpu_requested(config.getoption("markexpr")):
        return
    skip = pytest.mark.skip(reason="uses the GPU: select it explicitly with -m gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


S3_TEST = {"endpoint": os.environ.get("LAL_TEST_S3_ENDPOINT", "http://127.0.0.1:59000"),
           "access": os.environ.get("LAL_TEST_S3_ACCESS_KEY", "lal-local"),
           "secret": os.environ.get("LAL_TEST_S3_SECRET_KEY", "lal-local-minio")}


@pytest.fixture
def s3_bucket():
    """A fresh bucket on the MinIO of docker-compose.db.yml (marker `s3`), emptied and removed after
    the test; yields (settings overrides for LAL_*, bucket name)."""
    from infra.s3 import make_client

    client = make_client(S3_TEST["endpoint"], S3_TEST["access"], S3_TEST["secret"], "us-east-1")
    name = f"lal-test-{uuid.uuid4().hex[:10]}"
    client.create_bucket(Bucket=name)
    env = {"LAL_BLOB_STORE": "s3", "LAL_S3_ENDPOINT_URL": S3_TEST["endpoint"], "LAL_S3_BUCKET": name,
           "LAL_S3_ACCESS_KEY": S3_TEST["access"], "LAL_S3_SECRET_KEY": S3_TEST["secret"]}
    yield env, name, client
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=name):
        for obj in page.get("Contents", []):
            client.delete_object(Bucket=name, Key=obj["Key"])
    client.delete_bucket(Bucket=name)
