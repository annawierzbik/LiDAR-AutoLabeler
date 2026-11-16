"""Blob storage: named byte strings (point clouds, detection caches, exports).

Keys are relative POSIX paths such as "cache/training_0000.json". LocalBlobStore maps them to files
under a root folder; `put` is atomic (temporary file in the same folder, then os.replace), so a
reader never sees a half-written blob and a crash leaves the previous version in place.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    def get(self, key: str) -> bytes:
        """The blob's bytes; KeyError if there is none."""

    def put(self, key: str, data: bytes) -> None:
        """Create or replace the blob atomically."""

    def exists(self, key: str) -> bool: ...

    def list(self, prefix: str = "") -> list[str]:
        """Sorted keys starting with `prefix`."""


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` so that `path` holds either the old or the complete new content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


class LocalBlobStore:
    def __init__(self, root) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        p = PurePosixPath(key)
        if p.is_absolute() or ".." in p.parts or not p.parts or ":" in key or "\\" in key:
            raise ValueError(f"invalid blob key {key!r}")
        return self.root.joinpath(*p.parts)

    def get(self, key: str) -> bytes:
        path = self.path(key)
        if not path.is_file():
            raise KeyError(key)
        return path.read_bytes()

    def put(self, key: str, data: bytes) -> None:
        atomic_write_bytes(self.path(key), data)

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        """Walks only the folders that can hold keys starting with `prefix` (results/ also holds
        large experiment folders that a cache lookup must not scan)."""
        if not self.root.is_dir():
            return []
        out = []

        def walk(folder: Path, rel: str) -> None:
            for entry in os.scandir(folder):
                key = rel + entry.name
                if entry.is_dir():
                    sub = key + "/"
                    if sub.startswith(prefix) or prefix.startswith(sub):
                        walk(Path(entry.path), sub)
                elif key.startswith(prefix) and not entry.name.endswith(".tmp"):
                    out.append(key)

        walk(self.root, "")
        return sorted(out)
