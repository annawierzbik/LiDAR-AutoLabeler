"""A sequence of point clouds read through BlobStores, for workers (docs/adr/0015).

The same frames as SequenceLoader (KITTI layout) / FolderSequence (any folder of .bin files), but
the point clouds and the calibration are read with BlobStore.get, so a worker needs no path logic
of its own and the storage can change (phase 5: object storage) without touching the tasks.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import numpy as np

from infra.blob import BlobStore, LocalBlobStore
from ingestion.frame import Frame
from ingestion.loader import parse_calib_text


def natural_key(name: str):
    """The order of ingestion.folder.frame_files (frame2 before frame10), for blob keys."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", PurePosixPath(name).name)]


def upload_frame_keys(blobs: BlobStore, prefix: str) -> list[str]:
    """The .bin keys of an upload (directly under `prefix`/), in frame order."""
    keys = [k for k in blobs.list(prefix + "/") if "/" not in k[len(prefix) + 1:] and k.lower().endswith(".bin")]
    return sorted(keys, key=natural_key)


def validate_upload(blobs: BlobStore, prefix: str) -> int:
    """validate_folder for an upload in a BlobStore: the number of frames, or an error whose message
    can be shown to the user (no frames; the first frame is not x, y, z, intensity or is empty)."""
    keys = upload_frame_keys(blobs, prefix)
    if not keys:
        raise FileNotFoundError("No .bin point cloud files in the upload")
    first = BlobSequence(prefix, blobs, keys)[0]                    # ValueError if not a multiple of 4 values
    if len(first.points) == 0:
        raise ValueError(f"{PurePosixPath(keys[0]).name} contains no points")
    return len(keys)


class BlobSequence:
    def __init__(self, name: str, frames: BlobStore, keys: list[str], calib: dict | None = None) -> None:
        self.sequence, self.frames, self.keys = name, frames, list(keys)
        self._calib = calib or {}

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def frame_names(self) -> list[str]:
        """File names of the frames without extension, in order (export/session_export.py)."""
        return [PurePosixPath(k).stem for k in self.keys]

    def __getitem__(self, index: int) -> Frame:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of bounds [0, {len(self)})")
        key = self.keys[index]
        data = np.frombuffer(self.frames.get(key), dtype="<f4")
        if data.size % 4:
            raise ValueError(f"{key}: {data.size} values is not a multiple of 4 (x, y, z, intensity)")
        stem = Path(key).stem
        return Frame(sequence_id=self.sequence, frame_id=int(stem) if stem.isdigit() else index,
                     points=data.reshape(-1, 4).copy(), calib=self._calib, labels_gt=[])


def open_blob_sequence(args) -> BlobSequence:
    """The sequence of session-builder arguments (an upload's blob prefix, --folder, or
    --data-root/--split/--sequence)."""
    from ingestion.folder import frame_files

    prefix = getattr(args, "blob_prefix", None)
    if prefix:
        from infra.storage import uploads_store

        blobs = uploads_store()
        keys = upload_frame_keys(blobs, prefix)
        if not keys:
            raise FileNotFoundError(f"the upload {prefix} has no frames (was it deleted?)")
        return BlobSequence(prefix, blobs, keys)
    if args.folder:
        folder = Path(args.folder)
        return BlobSequence(folder.name, LocalBlobStore(folder), [p.name for p in frame_files(folder)])
    root = Path(args.data_root) / args.split
    velodyne = root / "velodyne" / args.sequence
    if not velodyne.is_dir():
        raise FileNotFoundError(f"Directory with point clouds not found: {velodyne}")
    calib_blobs = LocalBlobStore(root / "calib")
    key = f"{args.sequence}.txt"
    calib = parse_calib_text(calib_blobs.get(key).decode("utf-8")) if calib_blobs.exists(key) else {}
    return BlobSequence(args.sequence, LocalBlobStore(velodyne), [p.name for p in frame_files(velodyne)], calib)
