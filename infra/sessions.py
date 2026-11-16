"""Session storage: annotation sessions by key, with optimistic locking, an append-only event log
and an expiring exclusive lock.

Contract (FileSessionStore here, PgSessionStore in infra/pg_sessions.py):
- `load(key)` returns the session document; its `events` list is empty (the log is read with
  `events(key)`), so a session loaded for one request only holds the events of that request.
- `create(key, session)` stores a new session and its events; SessionExists if the key is taken.
- `save(key, session, expected_revision)` stores the document and appends `session.events`.
  RevisionConflict when the stored revision is not `expected_revision` (None: no check).
- `append_events(key, events)` only appends to the log: the revision does not change, so looking
  at a frame or playing back never conflicts with an edit (docs/adr/0013).
- `lock(key, owner)` gives one owner (a browser tab's session id) the session for `lock_ttl`
  seconds; calling it again refreshes the lock (heartbeat). Someone else gets SessionLocked until
  the lock expires. `finish(key, owner)` releases it and marks the session finished.

FileSessionStore: one JSON file per session (<root>/<key>.json, written atomically, events in the
file). Its locks live in the memory of one process (docs/adr/0012).
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

from infra.blob import atomic_write_text
from verification.session import AnnotationSession, Event

DEFAULT_LOCK_TTL = 120.0   # seconds; the browser sends a heartbeat every 30 s


class SessionNotFound(KeyError):
    pass


class SessionExists(Exception):
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
    lock_ttl: float

    def exists(self, key: str) -> bool: ...

    def load(self, key: str) -> AnnotationSession:
        """SessionNotFound if there is no such session. The returned events list is empty."""

    def create(self, key: str, session: AnnotationSession) -> None: ...

    def save(self, key: str, session: AnnotationSession, expected_revision: int | None) -> None: ...

    def append_events(self, key: str, events: Sequence[Event]) -> None: ...

    def events(self, key: str) -> list[Event]: ...

    def list(self) -> list[str]: ...

    def lock(self, key: str, owner: str) -> None: ...

    def unlock(self, key: str, owner: str) -> None: ...

    def locked_by(self, key: str) -> str | None:
        """The owner of a lock that has not expired, else None."""

    def finish(self, key: str, owner: str) -> None: ...


class FileSessionStore:
    def __init__(self, root, lock_ttl: float = DEFAULT_LOCK_TTL, clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root)
        self.lock_ttl, self._clock = lock_ttl, clock
        self._locks: dict[str, tuple[str, float]] = {}       # key -> (owner, time of the last lock call)
        self._mutex = threading.RLock()

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

    # ------------------------------------------------------------- documents
    def _read(self, key: str) -> dict:
        path = self.path(key)
        if not path.is_file():
            raise SessionNotFound(key)
        return json.loads(path.read_text())

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def load(self, key: str) -> AnnotationSession:
        s = AnnotationSession.from_dict(self._read(key))
        s.events = []
        return s

    def create(self, key: str, session: AnnotationSession) -> None:
        with self._mutex:
            if self.exists(key):
                raise SessionExists(key)
            session.save(self.path(key))

    def save(self, key: str, session: AnnotationSession, expected_revision: int | None) -> None:
        with self._mutex:
            stored = self._read(key)                     # SessionNotFound: use create() for new sessions
            if expected_revision is not None and stored.get("revision", 0) != expected_revision:
                raise RevisionConflict(key, expected_revision, stored.get("revision", 0))
            data = session.to_dict(include_events=False)
            data["events"] = stored.get("events", []) + [asdict(e) for e in session.events]
            atomic_write_text(self.path(key), json.dumps(data))

    def append_events(self, key: str, events: Sequence[Event]) -> None:
        if not events:
            return
        with self._mutex:
            data = self._read(key)
            data["events"] = data.get("events", []) + [asdict(e) for e in events]
            atomic_write_text(self.path(key), json.dumps(data))

    def events(self, key: str) -> list[Event]:
        return [Event(**e) for e in self._read(key).get("events", [])]

    def stored_revision(self, key: str) -> int | None:
        return int(self._read(key).get("revision", 0)) if self.exists(key) else None

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json")) if self.root.is_dir() else []

    # ----------------------------------------------------------------- locks
    def lock(self, key: str, owner: str) -> None:
        with self._mutex:
            now = self._clock()
            held = self._locks.get(key)
            if held is not None and held[0] != owner and now - held[1] < self.lock_ttl:
                raise SessionLocked(key, held[0])
            self._locks[key] = (owner, now)

    def unlock(self, key: str, owner: str) -> None:
        with self._mutex:
            if self._locks.get(key, (None,))[0] == owner:
                del self._locks[key]

    def locked_by(self, key: str) -> str | None:
        held = self._locks.get(key)
        return held[0] if held is not None and self._clock() - held[1] < self.lock_ttl else None

    def finish(self, key: str, owner: str) -> None:
        self.unlock(key, owner)
