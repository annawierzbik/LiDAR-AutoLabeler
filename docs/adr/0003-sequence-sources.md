# 0003 Sequence sources: zip upload and registered folders under a data root

Status: accepted, to be implemented in phase 4

## Context
Today the browser sends any folder path on the server's disk (`POST /api/folder/open`), and the
server reads it. With authentication, containers and several users this is unsafe (any readable
folder is exposed) and does not work when the API and the workers do not share a disk.

## Decision
A sequence is created in one of two ways:
1. **Upload** of a zip of `.bin` files. The zip is size-limited (setting), extracted safely (no
   absolute paths, no `..`, no links, member and total size limits, only `.bin` members kept),
   validated with the existing `ingestion.folder.validate_folder` rules, and stored in the
   BlobStore under the sequence's id.
2. **Registration** (admin only) of a folder under a configured data root that is mounted into the
   containers. The path is resolved and rejected unless it lies inside that root (also after
   resolving symlinks).

The free-path folder open stays available only to the local single-process mode (`scripts.web`
without the container settings) so the local and CLI workflow is unchanged.

## Consequences
- Workers read point clouds through the BlobStore (uploads) or the mounted data root (registered).
- The start screen needs an upload control; the path field is shown only in local mode.
