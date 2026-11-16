"""Point clouds of a sequence by frame index, read through a BlobStore.

BlobPointSource reads the .bin files of a folder through a LocalBlobStore rooted at it, in the
order the ingestion loaders use (natural sort, ingestion.folder.frame_files).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
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


class PointCache:
    """Thread-safe LRU of encoded point clouds, shared by all requests of one process (a cache only:
    any process can rebuild every entry from the point source)."""

    def __init__(self, max_items: int = 64) -> None:
        self.max_items = max_items
        self._items: OrderedDict = OrderedDict()
        self._mutex = threading.Lock()

    def get(self, key):
        with self._mutex:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key, value) -> None:
        with self._mutex:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
