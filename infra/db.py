"""PostgreSQL schema (SQLAlchemy 2 declarative models) and engine factory.

The migrations in migrations/ create exactly these tables; tests check that the two agree.
users, sequences and jobs are created now and used from phase 3/4 on.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger,
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
    state: Mapped[str] = mapped_column(String(20), default="queued")          # queued | running | done | failed | cancelled
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[dict | None] = mapped_column(JSONB)
    cache_key: Mapped[str | None] = mapped_column(String(128), index=True)     # idempotency (phase 3)
    session_key: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                    onupdate=func.now())


class SessionRow(Base):
    __tablename__ = "sessions"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    document: Mapped[dict] = mapped_column(JSONB)                              # AnnotationSession.to_dict(include_events=False)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="open")           # open | finished
    locked_by: Mapped[str | None] = mapped_column(String(64))
    locked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    sequence_id: Mapped[int | None] = mapped_column(ForeignKey("sequences.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                    onupdate=func.now())


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
