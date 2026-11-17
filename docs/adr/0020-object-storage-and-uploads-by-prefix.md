# 0020 Object storage: three blob areas, S3BlobStore, uploads addressed by prefix

Status: accepted (phase 5)

## Context
Until phase 4 an uploaded sequence was a folder on the API's disk (`spec = {"folder": path}`), the
detection cache and the export zips were folders under `results/`, and API and workers had to
share those folders. Containers on several hosts cannot.

## Decision
- **Three blob areas** (`infra/storage.py`): `uploads` (frames of uploaded sequences,
  `seq-<random>/<frame>.bin`), `cache` (detection + tracking results, `<file key>.json`) and
  `exports` (`<expiry>/<job id>/<archive>`). Code gets them only through `uploads_store()`,
  `results_store()` (also `pipeline.session_builder.results_blobs()`) and `exports_store()`.
- `LAL_BLOB_STORE=local` (default, development and CLI): `LocalBlobStore` folders as before.
  `LAL_BLOB_STORE=s3`: one bucket (`LAL_S3_BUCKET`) of an S3-compatible server
  (`LAL_S3_ENDPOINT_URL`, `LAL_S3_ACCESS_KEY`, `LAL_S3_SECRET_KEY`, `LAL_S3_REGION`), each area
  under its own key prefix (`uploads/`, `cache/`, `exports/`). `infra/s3.py` `S3BlobStore`
  (boto3, path-style addressing for MinIO) keeps the BlobStore contract: the same key rules as
  `LocalBlobStore`, `get` raises KeyError for a missing key, `put` is one PUT (atomic in S3),
  `list` pages through every result; `delete_many` uses DeleteObjects (1000 keys per request).
  One boto3 client per process and configuration.
- **Uploads are addressed by prefix**: a new upload's spec is `{"blob_prefix": "seq-<random>"}`.
  The session builder takes it as `blob_prefix` (a hidden argument): the prefix is the sequence
  label and file key (it is unique), the sequence has no split (like a folder), and
  `open_sequence` / `open_blob_sequence` return a `BlobSequence` over the upload's keys in the
  natural order of `ingestion.folder.frame_files`. The API serves points from the same keys (listed
  once per process; an upload never changes). `validate_upload` repeats the checks of
  `validate_folder` on blobs. The zip itself is only a temporary file of the API process.
  Extraction, validation and the frame listing run in the thread pool (`run_in_threadpool`), so
  the event loop keeps serving other requests during a large upload (test with a real uvicorn
  server: `test_the_api_keeps_serving_while_an_upload_is_unpacked`); only the streaming of the body
  into the temporary file (one write per received chunk) stays on the loop.
- **Frame list stored with the sequence** (review after phase 4): when a sequence is made (upload
  or registration) its frame file names are listed once, in frame order, and stored in its spec
  (`"frames"`). The session builder takes them as the hidden argument `frames`, so the API (points,
  state), the cpu and gpu opening steps, detection and export all use that list and never list a
  folder or the bucket again; a file added to a registered folder later is not part of the
  sequence. Sequences stored without the list (older rows) are listed once per process.
- **Export frame names**: `export/session_export.py: sequence_info` takes the names from the
  loader's `frame_names` when it has them (a BlobSequence) and otherwise from its velodyne folder,
  as before; rendering is unchanged (decided with the user). Tested: an upload and the same frames
  as a registered folder export the same file names.
- **What is still a filesystem**: registered folders below `LAL_DATA_ROOT` (`{"folder": path}`),
  mounted read-only at the same path into API and workers; the model weights (GPU worker only);
  temporary files of one process. Sessions, users, tabs and jobs are PostgreSQL rows.
- **Readiness** (`/readyz`) checks the three areas: a local folder is writable, or the bucket
  answers HeadBucket.
- **Without a GPU worker** (the cpu-only profile, ADR 0019) `LAL_ASSISTED_MODE=false` makes
  `POST /api/v1/tabs` refuse `mode=assisted` with 409 `assisted_unavailable` (decided with the
  user), instead of a job that waits in the gpu queue forever; the start page shows the message.

## Consequences
- Uploads made before phase 5 keep `{"folder": path}` and open wherever that folder exists (local
  work); they are not moved into the bucket (their session keys contain a hash of the path).
- The detection cache and exports must be in the same store for every process: all API and
  worker containers get the same `LAL_BLOB_STORE` and bucket settings.
- Tests: `tests/test_s3` (the BlobStore contract on MinIO) and
  `tests/test_api/test_blob_storage.py` (marker `s3`: the whole flow on S3 leaves the local
  results and upload folders empty).
