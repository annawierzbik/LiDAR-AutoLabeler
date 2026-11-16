"""SessionStore on PostgreSQL (tables `sessions` and `session_events`, infra/db.py).

- The document (annotations, revision, undo/redo history) is one JSONB value per session.
- `save` is optimistic: UPDATE ... WHERE key = :key AND revision = :expected. No row updated means
  someone else saved first -> RevisionConflict (the API answers 409).
- Events are rows of `session_events`, only ever inserted; they never touch the revision.
- The lock is `locked_by` / `locked_at`: taken or refreshed (heartbeat) with one conditional
  UPDATE, free again when `locked_at` is older than `lock_ttl` seconds.
"""
from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable, Sequence

from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from infra.db import SessionEventRow, SessionRow
from infra.sessions import (
    DEFAULT_LOCK_TTL,
    RevisionConflict,
    SessionExists,
    SessionLocked,
    SessionNotFound,
)
from verification.session import AnnotationSession, Event


class PgSessionStore:
    def __init__(self, engine: Engine, lock_ttl: float = DEFAULT_LOCK_TTL,
                 clock: Callable[[], float] = time.time) -> None:
        self.engine = engine
        self.lock_ttl, self._clock = lock_ttl, clock

    def _now(self) -> dt.datetime:
        return dt.datetime.fromtimestamp(self._clock(), dt.UTC)

    @staticmethod
    def _event_rows(key: str, events: Sequence[Event]) -> list[dict]:
        return [{"session_key": key, "t": e.t, "action": e.action, "frame": e.frame, "detail": e.detail}
                for e in events]

    # ------------------------------------------------------------- documents
    def exists(self, key: str) -> bool:
        with self.engine.connect() as c:
            return c.execute(select(SessionRow.key).where(SessionRow.key == key)).first() is not None

    def load(self, key: str) -> AnnotationSession:
        with self.engine.connect() as c:
            row = c.execute(select(SessionRow.document, SessionRow.revision).where(SessionRow.key == key)).first()
        if row is None:
            raise SessionNotFound(key)
        s = AnnotationSession.from_dict({**row.document, "revision": row.revision})
        s.events = []
        return s

    def create(self, key: str, session: AnnotationSession) -> None:
        try:
            with self.engine.begin() as c:
                c.execute(insert(SessionRow).values(key=key, document=session.to_dict(include_events=False),
                                                    revision=session.revision, status="open"))
                if session.events:
                    c.execute(insert(SessionEventRow), self._event_rows(key, session.events))
        except IntegrityError:
            raise SessionExists(key) from None

    def save(self, key: str, session: AnnotationSession, expected_revision: int | None) -> None:
        stmt = update(SessionRow).where(SessionRow.key == key)
        if expected_revision is not None:
            stmt = stmt.where(SessionRow.revision == expected_revision)
        with self.engine.begin() as c:
            done = c.execute(stmt.values(document=session.to_dict(include_events=False), revision=session.revision,
                                         updated_at=self._now()).returning(SessionRow.key)).first()
            if done is None:
                stored = c.execute(select(SessionRow.revision).where(SessionRow.key == key)).scalar()
                if stored is None:
                    raise SessionNotFound(key)
                raise RevisionConflict(key, expected_revision, stored)
            if session.events:
                c.execute(insert(SessionEventRow), self._event_rows(key, session.events))

    def append_events(self, key: str, events: Sequence[Event]) -> None:
        if not events:
            return
        try:
            with self.engine.begin() as c:
                c.execute(insert(SessionEventRow), self._event_rows(key, events))
        except IntegrityError:
            raise SessionNotFound(key) from None

    def events(self, key: str) -> list[Event]:
        with self.engine.connect() as c:
            rows = c.execute(select(SessionEventRow.t, SessionEventRow.action, SessionEventRow.frame,
                                    SessionEventRow.detail)
                             .where(SessionEventRow.session_key == key).order_by(SessionEventRow.id)).all()
        return [Event(t, action, frame, dict(detail or {})) for t, action, frame, detail in rows]

    def stored_revision(self, key: str) -> int | None:
        with self.engine.connect() as c:
            return c.execute(select(SessionRow.revision).where(SessionRow.key == key)).scalar()

    def list(self) -> list[str]:
        with self.engine.connect() as c:
            return list(c.execute(select(SessionRow.key).order_by(SessionRow.key)).scalars())

    def delete(self, key: str) -> None:
        with self.engine.begin() as c:
            c.execute(delete(SessionRow).where(SessionRow.key == key))

    # ----------------------------------------------------------------- locks
    def lock(self, key: str, owner: str) -> None:
        now = self._now()
        expired = now - dt.timedelta(seconds=self.lock_ttl)
        with self.engine.begin() as c:
            got = c.execute(
                update(SessionRow)
                .where(SessionRow.key == key)
                .where((SessionRow.locked_by.is_(None)) | (SessionRow.locked_by == owner)
                       | (SessionRow.locked_at < expired))
                .values(locked_by=owner, locked_at=now, status="open")
                .returning(SessionRow.key)).first()
            if got is None:
                holder = c.execute(select(SessionRow.locked_by).where(SessionRow.key == key)).first()
                if holder is None:
                    raise SessionNotFound(key)
                raise SessionLocked(key, holder.locked_by)

    def unlock(self, key: str, owner: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(SessionRow).where(SessionRow.key == key, SessionRow.locked_by == owner)
                      .values(locked_by=None, locked_at=None))

    def locked_by(self, key: str) -> str | None:
        with self.engine.connect() as c:
            row = c.execute(select(SessionRow.locked_by, SessionRow.locked_at).where(SessionRow.key == key)).first()
        if row is None or row.locked_by is None:
            return None
        return row.locked_by if self._now() - row.locked_at < dt.timedelta(seconds=self.lock_ttl) else None

    def finish(self, key: str, owner: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(SessionRow).where(SessionRow.key == key, SessionRow.locked_by == owner)
                      .values(locked_by=None, locked_at=None, status="finished"))

    def status(self, key: str) -> str | None:
        with self.engine.connect() as c:
            return c.execute(select(SessionRow.status).where(SessionRow.key == key)).scalar()

