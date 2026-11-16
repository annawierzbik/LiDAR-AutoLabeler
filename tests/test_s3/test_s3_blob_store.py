"""S3BlobStore against the MinIO of docker-compose.db.yml (marker `s3`): the BlobStore contract of
LocalBlobStore (tests/test_infra/test_storage.py), and areas kept apart by their prefixes."""
import pytest

from infra.blob import BlobStore
from infra.s3 import S3BlobStore

pytestmark = pytest.mark.s3


@pytest.fixture
def store(s3_bucket):
    _, name, client = s3_bucket
    return S3BlobStore(client, name, "cache/")


def test_round_trip_list_and_delete(store):
    assert isinstance(store, BlobStore)
    store.put("cache/a.json", b"1")
    store.put("b.bin", b"22")
    store.put("b.bin", b"33")                                          # replaced
    assert store.get("cache/a.json") == b"1" and store.get("b.bin") == b"33"
    assert store.exists("b.bin") and not store.exists("c")
    assert store.list() == ["b.bin", "cache/a.json"] and store.list("cache/") == ["cache/a.json"]
    with pytest.raises(KeyError):
        store.get("missing")
    store.delete("b.bin")
    store.delete("never-there")                                        # no error
    assert store.list() == ["cache/a.json"]


@pytest.mark.parametrize("key", ["../x", "/abs", "a/../../b", "C:/x", "a\\b", ""])
def test_keys_cannot_leave_the_area(store, key):
    with pytest.raises(ValueError):
        store.put(key, b"x")


def test_areas_are_separate_prefixes_of_one_bucket(s3_bucket):
    _, name, client = s3_bucket
    uploads, cache = S3BlobStore(client, name, "uploads/"), S3BlobStore(client, name, "cache/")
    uploads.put("seq-1/0001.bin", b"x")
    cache.put("seq-1.json", b"{}")
    assert uploads.list() == ["seq-1/0001.bin"] and cache.list() == ["seq-1.json"]
    keys = sorted(o["Key"] for o in client.list_objects_v2(Bucket=name)["Contents"])
    assert keys == ["cache/seq-1.json", "uploads/seq-1/0001.bin"]


def test_list_pages_and_delete_many(store):
    keys = [f"p/{i:05d}.bin" for i in range(1205)]                     # more than one page (1000) of S3
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda k: store.put(k, b"x"), keys))
    assert store.list("p/") == keys
    store.delete_many(keys)
    assert store.list() == []


def test_ping_and_ensure_bucket(s3_bucket):
    import uuid

    from botocore.exceptions import ClientError

    _, name, client = s3_bucket
    S3BlobStore(client, name).ping()
    other = S3BlobStore(client, f"lal-test-{uuid.uuid4().hex[:10]}")
    with pytest.raises(ClientError):
        other.ping()
    assert other.ensure_bucket() is True and other.ensure_bucket() is False
    other.ping()
    client.delete_bucket(Bucket=other.bucket)
