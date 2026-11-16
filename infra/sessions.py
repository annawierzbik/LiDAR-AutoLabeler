"""Session storage: load and save annotation sessions by key, with optimistic locking and an
exclusive per-session lock.

`save(key, session, expected_revision)` fails with RevisionConflict when the stored revision is not
the one the caller loaded, so two writers cannot silently overwrite each other. `lock` gives one
owner (e.g. a browser tab's session id) exclusive use of a session.

FileSessionStore keeps one JSON file per session (<root>/<key>.json, written atomically by
AnnotationSession.save) and its locks in memory, i.e. for one process.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from verification.session import AnnotationSession


class SessionNotFound(KeyError):
    pass


class RevisionConflict(Exception):
    def __init__(self, key: str, expected: int | None, stored: int | None) -> None:
        super().__init__(f"session {key} was changed elsewhere (stored revision {stored}, "
                         f"this copy is based on {expected})")
        self.key, self.expected, self.stored = key, expected, stored


class SessionLocked(Exception):
    def __init__(self, key: str, owner: str) -> None:
        super().__init__(f"session {key} is locked by {owner}")
        self.key, self.owner = key, owner


@runtime_checkable
class SessionStore(Protocol):
    def load(self, key: str) -> AnnotationSession:
        """SessionNotFound if there is no such session."""

    def save(self, key: str, session: AnnotationSession, expected_revision: int | None) -> None:
        """Store `session`. `expected_revision` = the stored revision the caller's copy is based on
        (None: no check). RevisionConflict when the stored revision differs."""

    def list(self) -> list[str]: ...

    def lock(self, key: str, owner: str) -> None:
        """Exclusive lock; SessionLocked naming the current owner if someone else holds it.
        Locking again by the same owner is allowed."""

    def unlock(self, key: str, owner: str) -> None:
        """Release the lock if `owner` holds it (otherwise nothing happens)."""

    def locked_by(self, key: str) -> str | None: ...


class FileSessionStore:
    def __init__(self, root) -> None:
        self.root = Path(root)
        self._locks: dict[str, str] = {}
        self._mutex = threading.Lock()

    def path(self, key: str) -> Path:
        if not key or "/" in key or "\\" in key or key in (".", ".."):
            raise ValueError(f"invalid session key {key!r}")
        return self.root / f"{key}.json"

    def key_for(self, path) -> str:
        """Key of a session file inside this store's folder."""
        path = Path(path)
        if path.parent.resolve() != self.root.resolve() or path.suffix != ".json":
            raise ValueError(f"{path} is not a session file of {self.root}")
        return path.stem

    def load(self, key: str) -> AnnotationSession:
        path = self.path(key)
        if not path.is_file():
            raise SessionNotFound(key)
        return AnnotationSession.load(path)

    def stored_revision(self, key: str) -> int | None:
        path = self.path(key)
        if not path.is_file():
            return None
        return int(json.loads(path.read_text()).get("revision", 0))   # v1 files have none: 0

    def save(self, key: str, session: AnnotationSession, expected_revision: int | None) -> None:
        with self._mutex:
            stored = self.stored_revision(key)
            if expected_revision is not None and stored is not None and stored != expected_revision:
                raise RevisionConflict(key, expected_revision, stored)
            session.save(self.path(key))

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json")) if self.root.is_dir() else []

    def lock(self, key: str, owner: str) -> None:
        with self._mutex:
            holder = self._locks.get(key)
            if holder is not None and holder != owner:
                raise SessionLocked(key, holder)
            self._locks[key] = owner

    def unlock(self, key: str, owner: str) -> None:
        with self._mutex:
            if self._locks.get(key) == owner:
                del self._locks[key]

    def locked_by(self, key: str) -> str | None:
        return self._locks.get(key)
