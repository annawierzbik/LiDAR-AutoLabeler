"""PostgreSQL schema (SQLAlchemy 2 declarative models) and engine factory.

The migrations in migrations/ create exactly these tables; tests check that the two agree.
users, sequences and jobs are created now and used from phase 3/4 on.

A session is stored in parts so that an action writes only what it changed (docs/adr/0011):
`sessions.document` holds the small metadata (sequence, mode, frame offset and count, next track
id), `session_frames` one row per non-empty frame, `session_history` one row per undo or redo step
(only the frames that step changed), `session_events` the append-only event log.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20), default="annotator")          # annotator | admin
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SequenceRow(Base):
    __tablename__ = "sequences"
    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(20))                            # folder | upload
    location: Mapped[str] = mapped_column(Text)                                # folder path or blob prefix
    split: Mapped[str | None] = mapped_column(String(20))
    num_frames: Mapped[int] = mapped_column(Integer)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class JobRow(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(Text, default="")
    # queued | running | cancelling | done | failed | cancelled
    state: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[dict | None] = mapped_column(JSONB)
    cache_key: Mapped[str | None] = mapped_column(String(128), index=True)     # dedupe key (infra.tasks.dedupe_key)
    session_key: Mapped[str | None] = mapped_column(Text)
    queue: Mapped[str] = mapped_column(String(20), server_default="cpu")       # gpu | cpu
    payload: Mapped[dict] = mapped_column(JSONB, server_default="{}")
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, server_default=false())
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    worker: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                    onupdate=func.now())


class SessionRow(Base):
    __tablename__ = "sessions"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    document: Mapped[dict] = mapped_column(JSONB)                              # metadata only, see PgSessionStore
    revision: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="open")           # open | finished
    locked_by: Mapped[str | None] = mapped_column(String(64))
    locked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    sequence_id: Mapped[int | None] = mapped_column(ForeignKey("sequences.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                    onupdate=func.now())


class SessionFrameRow(Base):
    __tablename__ = "session_frames"
    session_key: Mapped[str] = mapped_column(ForeignKey("sessions.key", ondelete="CASCADE"), primary_key=True)
    frame: Mapped[int] = mapped_column(Integer, primary_key=True)
    annotations: Mapped[list] = mapped_column(JSONB)                           # [Annotation.to_dict(), ...]


class SessionHistoryRow(Base):
    __tablename__ = "session_history"
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)  # order within a stack
    session_key: Mapped[str] = mapped_column(ForeignKey("sessions.key", ondelete="CASCADE"))
    stack: Mapped[str] = mapped_column(String(4))                              # undo | redo
    label: Mapped[str] = mapped_column(String(40))
    frames: Mapped[dict] = mapped_column(JSONB)                                # {frame: [annotation]} before the step

    __table_args__ = (Index("ix_session_history_session_key_stack_id", "session_key", "stack", "id"),)


class SessionEventRow(Base):
    __tablename__ = "session_events"
    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    session_key: Mapped[str] = mapped_column(ForeignKey("sessions.key", ondelete="CASCADE"))
    t: Mapped[float] = mapped_column(Double)                                    # seconds since the epoch (session clock)
    action: Mapped[str] = mapped_column(String(40))
    frame: Mapped[int | None] = mapped_column(Integer)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (Index("ix_session_events_session_key_id", "session_key", "id"),)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)
