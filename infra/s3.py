"""BlobStore on an S3-compatible object store (MinIO in the deployment, docs/adr/0020).

One bucket holds every area under its own key prefix (`uploads/`, `cache/`, `exports/`,
infra/storage.py); an S3BlobStore is one area: its keys are relative to that prefix, exactly like
LocalBlobStore keys are relative to its root. A PUT of one object is atomic in S3 (a reader sees
the old or the whole new object), which is the guarantee `put` promises.
"""
from __future__ import annotations

from pathlib import PurePosixPath


def check_key(key: str) -> str:
    """The same key rules as LocalBlobStore: relative POSIX path, no "..", no drive or backslash."""
    p = PurePosixPath(key)
    if p.is_absolute() or ".." in p.parts or not p.parts or ":" in key or "\\" in key:
        raise ValueError(f"invalid blob key {key!r}")
    return key


def make_client(endpoint_url: str | None, access_key: str | None, secret_key: str | None, region: str):
    import boto3
    from botocore.config import Config

    return boto3.client("s3", endpoint_url=endpoint_url, aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key, region_name=region,
                        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                      retries={"max_attempts": 3, "mode": "standard"}))


class S3BlobStore:
    def __init__(self, client, bucket: str, prefix: str = "") -> None:
        if prefix and not prefix.endswith("/"):
            raise ValueError("the prefix of an S3BlobStore ends with '/'")
        self.client, self.bucket, self.prefix = client, bucket, prefix

    def _key(self, key: str) -> str:
        return self.prefix + check_key(key)

    @staticmethod
    def _missing(exc) -> bool:
        return exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound")

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except ClientError as exc:
            if self._missing(exc):
                raise KeyError(key) from None
            raise

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as exc:
            if self._missing(exc):
                return False
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))     # no error if absent

    def delete_many(self, keys: list[str]) -> None:
        """Delete up to any number of keys, 1000 per request (S3 DeleteObjects)."""
        for i in range(0, len(keys), 1000):
            chunk = [{"Key": self._key(k)} for k in keys[i:i + 1000]]
            out = self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": chunk, "Quiet": True})
            if out.get("Errors"):
                first = out["Errors"][0]
                raise OSError(f"could not delete {first.get('Key')}: {first.get('Message')}")

    def list(self, prefix: str = "") -> list[str]:
        out = []
        pages = self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=self.prefix + prefix)
        for page in pages:
            out.extend(obj["Key"][len(self.prefix):] for obj in page.get("Contents", []))
        return sorted(out)

    def ping(self) -> None:
        """Raises unless the bucket is reachable with these credentials (readiness check)."""
        self.client.head_bucket(Bucket=self.bucket)

    def ensure_bucket(self) -> bool:
        """Create the bucket if it does not exist; True if it was created."""
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
            return False
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket", "NotFound"):
                raise
        self.client.create_bucket(Bucket=self.bucket)
        return True
