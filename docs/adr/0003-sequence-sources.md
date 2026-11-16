# 0003 Sequence sources: zip upload and registered folders under a data root

Status: accepted, implemented in phase 4 (`infra/sequence_store.py`, `api/routes_sequences.py`)

## Context
Before phase 4 the browser sent any folder path on the server's disk (`POST /api/folder/open`), and
the server read it. With authentication, containers and several users this is unsafe (any readable
folder is exposed) and does not work when the API and the workers do not share a disk.

## Decision
A sequence is a row of the `sequences` table (owner, name, frame count, `spec` = the session-builder
arguments that open it). It is created in one of two ways:
1. **Upload** of a zip of `.bin` files (`POST /api/v1/sequences/upload?name=`, body
   `application/zip`, any logged-in user, who owns it). The body is streamed to a temporary file
   and refused above `LAL_UPLOAD_MAX_BYTES`. The archive is refused when a member name is unsafe
   (absolute, drive letter, `..`, backslash), is a symbolic link or encrypted, when the `.bin`
   members are in more than one folder of the archive or share a name, or when they unpack to more
   than `LAL_UPLOAD_MAX_UNPACKED_BYTES` (declared sizes are checked first; the bytes actually
   decompressed are counted too, so a member whose header lies is stopped). Only `.bin` members
   are written, under their base name, into the uploads BlobStore at `seq-<random>/`. The folder
   then has to pass `ingestion.folder.validate_folder`; otherwise everything written is removed.
2. **Registration** (admins only, `POST /api/v1/sequences/register`) of a folder below
   `LAL_DATA_ROOT`, relative to it or absolute, for a given user. The path is resolved (symbolic
   links followed) and refused with 403 unless it lies inside the resolved root. Without
   `LAL_DATA_ROOT` registration is disabled.

**The web API always runs with PostgreSQL, Redis and at least one worker of each queue** (decision
of phase 4). The free-path folder open is removed from the web UI entirely, also locally: local
folders are registered by an admin below `LAL_DATA_ROOT`. `FileSessionStore` and
`ThreadTaskQueue` remain for the CLI scripts and the tests.

## Consequences
- The uploads BlobStore is a `LocalBlobStore` on `LAL_UPLOAD_ROOT` (default
  `<results_dir>/uploads`), so an uploaded sequence is also a folder (`spec = {"folder": ...}`)
  that the existing loaders and workers open. API and workers must share that volume and the data
  root; with object storage (phase 5) the upload `spec` has to name a blob prefix instead, read
  through `infra.sequences.BlobSequence` (already used by the detection task).
- Uploads are not deduplicated: the same zip uploaded twice is two sequences.
- Nothing deletes uploaded sequences yet (no delete endpoint).
