"""The three blob areas of the application and where they live (docs/adr/0020).

    uploads   frames of uploaded sequences, under seq-<random>/      (API writes, everyone reads)
    cache     detection + tracking results, <file key>.json          (workers write and read)
    exports   label zips, <expiry>/<job id>/<archive>                (cpu workers write, API reads)

LAL_BLOB_STORE=local (default): folders on this machine (LAL_UPLOAD_ROOT or <results>/uploads,
<results>, <results>/exports), as before. LAL_BLOB_STORE=s3: one bucket (LAL_S3_BUCKET) of an
S3-compatible server (LAL_S3_ENDPOINT_URL, LAL_S3_ACCESS_KEY, LAL_S3_SECRET_KEY), each area under
its own key prefix; API and workers then share no filesystem (except the read-only data root of
registered folders).

The detection cache and exports in local mode are rooted at pipeline.session_builder.RESULTS_DIR,
read when the store is made (tests point it at a temporary folder).
"""
from __future__ import annotations

from pathlib import Path

from infra.blob import BlobStore, LocalBlobStore
from infra.settings import Settings, get_settings

AREAS = ("uploads", "cache", "exports")


def s3_store(area: str, settings: Settings | None = None):
    from infra.s3 import S3BlobStore

    s = settings or get_settings()
    if area not in AREAS:
        raise ValueError(f"unknown blob area {area!r}")
    missing = [n for n in ("s3_bucket", "s3_access_key", "s3_secret_key") if not getattr(s, n)]
    if missing:
        raise ValueError("LAL_BLOB_STORE=s3 needs " + ", ".join(f"LAL_{n.upper()}" for n in missing))
    client = _client(s.s3_endpoint_url, s.s3_access_key, s.s3_secret_key.get_secret_value(), s.s3_region)
    return S3BlobStore(client, s.s3_bucket, f"{area}/")


_clients: dict = {}


def _client(endpoint, access, secret, region):
    """One boto3 client per configuration and process (clients are thread-safe and pool connections)."""
    from infra.s3 import make_client

    key = (endpoint, access, secret, region)
    if key not in _clients:
        _clients[key] = make_client(endpoint, access, secret, region)
    return _clients[key]


def uploads_store(settings: Settings | None = None) -> BlobStore:
    s = settings or get_settings()
    return s3_store("uploads", s) if s.blob_store == "s3" else LocalBlobStore(s.uploads_dir)


def results_store(settings: Settings | None = None) -> BlobStore:
    s = settings or get_settings()
    if s.blob_store == "s3":
        return s3_store("cache", s)
    from pipeline import session_builder

    return LocalBlobStore(session_builder.RESULTS_DIR)


def exports_store(settings: Settings | None = None) -> BlobStore:
    s = settings or get_settings()
    if s.blob_store == "s3":
        return s3_store("exports", s)
    from pipeline import session_builder

    return LocalBlobStore(session_builder.RESULTS_DIR / "exports")


def check(store: BlobStore) -> None:
    """Raises unless the store can be used: S3 bucket reachable; local folder writable."""
    if hasattr(store, "ping"):
        store.ping()
        return
    root = Path(store.root)
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".ready"
    probe.write_bytes(b"ok")
    probe.unlink()


def delete_keys(store: BlobStore, keys: list[str]) -> None:
    if hasattr(store, "delete_many"):
        store.delete_many(keys)
    else:
        for key in keys:
            store.delete(key)


def main(argv: list[str] | None = None) -> int:
    """python -m infra.storage --ensure-bucket: create LAL_S3_BUCKET if it does not exist (the
    one-shot migration service runs it); nothing to do for LAL_BLOB_STORE=local."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ensure-bucket", action="store_true")
    args = p.parse_args(argv)
    s = get_settings()
    if not args.ensure_bucket:
        p.error("nothing to do")
    if s.blob_store != "s3":
        print("LAL_BLOB_STORE is not s3: no bucket to create")
        return 0
    created = s3_store("uploads", s).ensure_bucket()
    print(f"bucket {s.s3_bucket} {'created' if created else 'exists'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
