# 0022 MinIO as the object store: source-only, archived, pinned; the app depends on S3 only

Status: accepted (after phase 5)

## Context
Phase 5 needs an S3-compatible object store that runs in the same compose project as the rest
(ADR 0019, 0020). MinIO was chosen: one container, path-style S3 API, a health endpoint, and it is
the S3 server most widely run on-premises, so it is the likely store of a lab deployment.

What changed in 2025 (checked on 2026-11-17 through the GitHub API):
- `docker pull minio/minio:<tag>` answers "repository does not exist"; quay.io/minio/minio has no
  tags. The community edition is distributed as source only.
- The repository `github.com/minio/minio` is **archived** (read-only). Its last release is
  `RELEASE.2025-10-15T17-29-55Z` (commit `9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a`).
- That release is the one that fixes **CVE-2025-62506** (GHSA-jjjj-jwhf-8rgr, severity high,
  CVSS 8.1: privilege escalation via a session-policy bypass in service accounts and STS). The
  advisory lists "all" earlier versions as vulnerable and this release as patched; the fix is PR
  #21642 (merge commit `c1a49490c78e9c3ebcad86ba0662319138ace190`), one commit before the tag.

## Decision
- `deploy/minio/Dockerfile` builds that release from source and **pins tag and commit**: the
  build fails unless the tag points at `9e49d5e7…` (tested by building with a wrong commit). The
  commit is recorded in the image label `org.opencontainers.image.revision`.
- The application talks to the store **only through the S3 protocol** (`infra/s3.py`,
  `S3BlobStore` on boto3 with path-style addressing): GetObject, PutObject, HeadObject,
  DeleteObject(s), ListObjectsV2 (paginated), HeadBucket, CreateBucket. No MinIO API, no MinIO
  client library, no MinIO-only feature.
- A **compatibility test** keeps that true: the S3BlobStore contract tests (`tests/test_s3`,
  marker `s3`) run against MinIO and against a second implementation, SeaweedFS 3.80, started by
  the test-only override `tests/s3compat/docker-compose.s3compat.yml`
  (`docker compose -f docker-compose.db.yml -f tests/s3compat/docker-compose.s3compat.yml up -d`).
  SeaweedFS was chosen over Garage because it needs no setup step after start (Garage needs a
  layout, a key and a bucket made with its admin CLI). A missing server fails the tests; it is
  not skipped.

## Risk
- No more security fixes from upstream: a future MinIO vulnerability will not be patched in this
  release. The store must stay on the internal network (it publishes no port in
  docker-compose.yml) and the app uses one set of credentials.
- Building from source depends on GitHub and the Go toolchain being available at build time.

## Consequences
- Replacing MinIO is a deployment change only: another S3 server (SeaweedFS, Garage, Ceph RGW,
  AWS S3) with `LAL_S3_ENDPOINT_URL`, `LAL_S3_BUCKET` and keys. Before switching, run the contract
  tests against it (add it to `S3_SERVERS` in `tests/conftest.py`). Moving the data is a bucket
  copy (`aws s3 sync` or the target's tool).
- Revisit when a maintained fork or another store becomes the lab standard.
