"""Point clouds of a sequence by frame index, read through a BlobStore.

BlobPointSource reads the .bin files of a folder through a LocalBlobStore rooted at it, in the
order the ingestion loaders use (natural sort, ingestion.folder.frame_files).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from infra.blob import BlobStore, LocalBlobStore


@runtime_checkable
class PointSource(Protocol):
    def __len__(self) -> int: ...

    def xyz(self, index: int) -> np.ndarray:
        """(N, 3) x, y, z of frame `index` of the sequence; IndexError when out of range."""


class BlobPointSource:
    def __init__(self, blobs: BlobStore, keys: list[str]) -> None:
        self.blobs, self.keys = blobs, list(keys)

    def __len__(self) -> int:
        return len(self.keys)

    def xyz(self, index: int) -> np.ndarray:
        if not 0 <= index < len(self.keys):
            raise IndexError(index)
        data = np.frombuffer(self.blobs.get(self.keys[index]), dtype="<f4")
        if data.size % 4:
            raise ValueError(f"{self.keys[index]}: {data.size} values is not a multiple of 4 (x, y, z, intensity)")
        return data.reshape(-1, 4)[:, :3]


class LoaderPointSource:
    """Adapter for objects that only offer loader[i].xyz (e.g. test doubles)."""

    def __init__(self, loader) -> None:
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def xyz(self, index: int) -> np.ndarray:
        return np.asarray(self.loader[index].xyz)


def point_source(sequence) -> PointSource:
    """Points of a SequenceLoader / FolderSequence through a LocalBlobStore on its velodyne folder."""
    folder = getattr(sequence, "velodyne_dir", None)
    if folder is None:
        return LoaderPointSource(sequence)
    from ingestion.folder import frame_files

    return BlobPointSource(LocalBlobStore(folder), [p.name for p in frame_files(folder)])
