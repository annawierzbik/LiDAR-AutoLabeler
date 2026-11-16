"""SessionStore on PostgreSQL (infra/db.py), storing a session in parts (docs/adr/0011).

- `sessions.document`: metadata only; `session_frames`: one row per frame (written for non-empty
  frames at creation); `session_history`: one row per undo / redo step holding only the frames that
  step changed, the newest UNDO_LIMIT per stack; `session_events`: the append-only event log.
- `load` reads the metadata, every frame row and only the newest step of each stack (one action per
  request never needs more), so its cost does not depend on the length of the history.
- `save` is optimistic and writes only what the request changed, in one transaction:
  UPDATE sessions ... WHERE key = :key AND revision = :expected (no row -> RevisionConflict, HTTP 409),
  upsert of the frames in `session.dirty_frames`, the history operations of the request
  (`session.history_ops`), trimming each stack to UNDO_LIMIT rows, new events.
- Events never touch the revision. The lock is `locked_by` / `locked_at`: taken or refreshed
  (heartbeat) with one conditional UPDATE, free again when `locked_at` is older than `lock_ttl`.
"""
from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable, Sequence

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy import literal as sa_literal
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from infra.db import SessionEventRow, SessionFrameRow, SessionHistoryRow, SessionRow
from infra.sessions import (
    DEFAULT_LOCK_TTL,
    RevisionConflict,
    SessionExists,
    SessionLocked,
    SessionNotFound,
    check_revision_argument,
)
from verification.session import UNDO_LIMIT, AnnotationSession, Event, step_to_dict


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

    @staticmethod
    def _frame_rows(key: str, session: AnnotationSession, frames) -> list[dict]:
        return [{"session_key": key, "frame": f, "annotations": [a.to_dict() for a in session.frames[f]]}
                for f in sorted(frames)]

    @staticmethod
    def _history_row(key: str, stack: str, step) -> dict:
        d = step_to_dict(step)
        return {"session_key": key, "stack": stack, "label": d["label"], "frames": d["frames"]}

    # ------------------------------------------------------------- documents
    def exists(self, key: str) -> bool:
        with self.engine.connect() as c:
            return c.execute(select(SessionRow.key).where(SessionRow.key == key)).first() is not None

    def load(self, key: str) -> AnnotationSession:
        with self.engine.connect() as c:
            row = c.execute(select(SessionRow.document, SessionRow.revision).where(SessionRow.key == key)).first()
            if row is None:
                raise SessionNotFound(key)
            frames = c.execute(select(SessionFrameRow.frame, SessionFrameRow.annotations)
                               .where(SessionFrameRow.session_key == key)).all()
            tops = {stack: self._newest_steps(c, key, stack, 1) for stack in ("undo", "redo")}
        s = AnnotationSession.from_parts(row.document, dict(frames), tops["undo"], tops["redo"], row.revision)
        s.events = []
        s.clear_changes()
        return s

    @staticmethod
    def _newest_steps(c: Connection, key: str, stack: str, n: int) -> list[dict]:
        rows = c.execute(select(SessionHistoryRow.label, SessionHistoryRow.frames)
                         .where(SessionHistoryRow.session_key == key, SessionHistoryRow.stack == stack)
                         .order_by(SessionHistoryRow.id.desc()).limit(n)).all()
        return [{"label": label, "frames": frames} for label, frames in reversed(rows)]

    def _write_all(self, c: Connection, key: str, session: AnnotationSession) -> None:
        """Every frame, the whole history (as held by `session`) and its events."""
        rows = self._frame_rows(key, session, [f for f, anns in session.frames.items() if anns])
        if rows:
            c.execute(insert(SessionFrameRow), rows)
        history = [self._history_row(key, stack, st) for stack, steps in (("undo", session._undo),
                                                                          ("redo", session._redo)) for st in steps]
        if history:
            c.execute(insert(SessionHistoryRow), history)
        if session.events:
            c.execute(insert(SessionEventRow), self._event_rows(key, session.events))

    def create(self, key: str, session: AnnotationSession) -> None:
        try:
            with self.engine.begin() as c:
                c.execute(insert(SessionRow).values(key=key, document=session.metadata(),
                                                    revision=session.revision, status="open"))
                self._write_all(c, key, session)
        except IntegrityError:
            raise SessionExists(key) from None
        session.clear_changes()

    def replace(self, key: str, session: AnnotationSession) -> None:
        """Start the session over with `session` (frames and history replaced, event log kept)."""
        with self.engine.begin() as c:
            done = c.execute(update(SessionRow).where(SessionRow.key == key)
                             .values(document=session.metadata(), revision=session.revision, status="open",
                                     updated_at=self._now()).returning(SessionRow.key)).first()
            if done is None:
                raise SessionNotFound(key)
            c.execute(delete(SessionFrameRow).where(SessionFrameRow.session_key == key))
            c.execute(delete(SessionHistoryRow).where(SessionHistoryRow.session_key == key))
            self._write_all(c, key, session)
        session.clear_changes()

    def save(self, key: str, session: AnnotationSession, expected_revision: int) -> None:
        check_revision_argument(expected_revision)
        stmt = update(SessionRow).where(SessionRow.key == key, SessionRow.revision == expected_revision)
        with self.engine.begin() as c:
            done = c.execute(stmt.values(document=session.metadata(), revision=session.revision,
                                         updated_at=self._now()).returning(SessionRow.key)).first()
            if done is None:
                stored = c.execute(select(SessionRow.revision).where(SessionRow.key == key)).scalar()
                if stored is None:
                    raise SessionNotFound(key)
                raise RevisionConflict(key, expected_revision, stored)
            if session.dirty_frames:
                up = pg_insert(SessionFrameRow)
                c.execute(up.on_conflict_do_update(index_elements=[SessionFrameRow.session_key, SessionFrameRow.frame],
                                                   set_={"annotations": up.excluded.annotations}),
                          self._frame_rows(key, session, session.dirty_frames))
            self._apply_history(c, key, session.history_ops)
            if session.events:
                c.execute(insert(SessionEventRow), self._event_rows(key, session.events))
        session.clear_changes()

    def _apply_history(self, c: Connection, key: str, ops: list[tuple]) -> None:
        pushed = set()
        for op in ops:
            stack = op[1]
            rows = (SessionHistoryRow.session_key == key, SessionHistoryRow.stack == stack)
            if op[0] == "push":
                c.execute(insert(SessionHistoryRow), [self._history_row(key, stack, op[2])])
                pushed.add(stack)
            elif op[0] == "pop":
                newest = select(SessionHistoryRow.id).where(*rows).order_by(SessionHistoryRow.id.desc()).limit(1)
                c.execute(delete(SessionHistoryRow).where(SessionHistoryRow.id == newest.scalar_subquery()))
            else:                                                  # clear
                c.execute(delete(SessionHistoryRow).where(*rows))
        for stack in pushed:                                       # keep the newest UNDO_LIMIT steps
            rows = (SessionHistoryRow.session_key == key, SessionHistoryRow.stack == stack)
            oldest_kept = (select(SessionHistoryRow.id).where(*rows).order_by(SessionHistoryRow.id.desc())
                           .offset(UNDO_LIMIT - 1).limit(1).scalar_subquery())
            c.execute(delete(SessionHistoryRow).where(*rows, SessionHistoryRow.id < oldest_kept))

    def history_length(self, key: str) -> dict[str, int]:
        with self.engine.connect() as c:
            return {stack: len(self._newest_steps(c, key, stack, UNDO_LIMIT + 10)) for stack in ("undo", "redo")}

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

    def set_owner(self, key: str, owner_id: int, sequence_id: int | None) -> None:
        with self.engine.begin() as c:
            c.execute(update(SessionRow).where(SessionRow.key == key)
                      .values(owner_id=owner_id, sequence_id=sequence_id))

    def assign_owner(self, key: str, owner_id: int, new_key: str) -> None:
        """Give a session without an owner (stored before phase 4) to `owner_id` under `new_key`
        (pipeline.session_builder.key_for_owner), in one transaction: the session row is copied to
        the new key, its frames, history and events are moved, the old row is deleted.
        SessionNotFound; ValueError if it already has an owner or is locked; SessionExists if
        `new_key` is taken."""
        cols = [col.name for col in SessionRow.__table__.columns if col.name not in ("key", "owner_id")]
        try:
            with self.engine.begin() as c:
                row = c.execute(select(SessionRow.owner_id, SessionRow.locked_by, SessionRow.locked_at)
                                .where(SessionRow.key == key).with_for_update()).first()
                if row is None:
                    raise SessionNotFound(key)
                if row.owner_id is not None:
                    raise ValueError(f"session {key} already belongs to user {row.owner_id}")
                if row.locked_by and row.locked_at and row.locked_at > self._now() - dt.timedelta(seconds=self.lock_ttl):
                    raise ValueError(f"session {key} is locked by {row.locked_by}")
                src = select(*[sa_literal(new_key).label("key"), sa_literal(owner_id).label("owner_id")],
                             *[SessionRow.__table__.c[n] for n in cols]).where(SessionRow.key == key)
                c.execute(insert(SessionRow).from_select(["key", "owner_id", *cols], src))
                for table in (SessionFrameRow, SessionHistoryRow, SessionEventRow):
                    c.execute(update(table).where(table.session_key == key).values(session_key=new_key))
                c.execute(delete(SessionRow).where(SessionRow.key == key))
        except IntegrityError:
            raise SessionExists(new_key) from None

    def unowned(self) -> list[tuple[str, str]]:
        """(key, sequence label) of every session without an owner."""
        with self.engine.connect() as c:
            return [(r.key, (r.document or {}).get("sequence", "")) for r in
                    c.execute(select(SessionRow.key, SessionRow.document).where(SessionRow.owner_id.is_(None))
                              .order_by(SessionRow.key))]

    def owner(self, key: str) -> int | None:
        """The owner's user id; SessionNotFound if there is no such session."""
        with self.engine.connect() as c:
            row = c.execute(select(SessionRow.owner_id).where(SessionRow.key == key)).first()
        if row is None:
            raise SessionNotFound(key)
        return row.owner_id

    def page(self, owner_id: int | None, limit: int, offset: int) -> tuple[list[dict], int]:
        """Session metadata, most recently changed first; `owner_id` None: every user's sessions."""
        cond = [] if owner_id is None else [SessionRow.owner_id == owner_id]
        cols = (SessionRow.key, SessionRow.revision, SessionRow.status, SessionRow.owner_id, SessionRow.sequence_id,
                SessionRow.document, SessionRow.updated_at)
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(SessionRow).where(*cond)).scalar()
            rows = c.execute(select(*cols).where(*cond).order_by(SessionRow.updated_at.desc(), SessionRow.key)
                             .limit(limit).offset(offset)).all()
        return [{"key": r.key, "revision": r.revision, "status": r.status, "owner_id": r.owner_id,
                 "sequence_id": r.sequence_id, "sequence": r.document.get("sequence"), "mode": r.document.get("mode"),
                 "updated_at": r.updated_at.isoformat()} for r in rows], total

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

