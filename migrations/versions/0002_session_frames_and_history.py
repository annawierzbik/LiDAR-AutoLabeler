"""sessions in parts: session_frames and session_history (data migrated both ways)

Upgrade: every `sessions.document` (the whole phase-2 document) is split into one
`session_frames` row per non-empty frame and one `session_history` row per undo/redo step
(oldest first, so `id` keeps the stack order); the document keeps only the metadata.
Downgrade: the whole documents are rebuilt from the rows (the newest 50 steps of each stack) and the
two tables are dropped.

Revision ID: 0002
Revises: 0001
"""
import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

PARTS = ("frames", "undo", "redo", "revision")   # moved out of the document (revision: the column)
UNDO_LIMIT = 50


def upgrade() -> None:
    op.create_table(
        "session_frames",
        sa.Column("session_key", sa.Text(), sa.ForeignKey("sessions.key", ondelete="CASCADE"), primary_key=True),
        sa.Column("frame", sa.Integer(), primary_key=True),
        sa.Column("annotations", postgresql.JSONB(), nullable=False),
    )
    op.create_table(
        "session_history",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("session_key", sa.Text(), sa.ForeignKey("sessions.key", ondelete="CASCADE"), nullable=False),
        sa.Column("stack", sa.String(4), nullable=False),
        sa.Column("label", sa.String(40), nullable=False),
        sa.Column("frames", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_session_history_session_key_stack_id", "session_history", ["session_key", "stack", "id"])

    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT key, document FROM sessions ORDER BY key")).all()
    for key, doc in rows:
        frames = [{"k": key, "f": int(f), "a": json.dumps(anns)} for f, anns in doc.get("frames", {}).items() if anns]
        if frames:
            bind.execute(sa.text("INSERT INTO session_frames (session_key, frame, annotations) "
                                 "VALUES (:k, :f, CAST(:a AS jsonb))"), frames)
        for stack in ("undo", "redo"):
            steps = [{"k": key, "s": stack, "l": st["label"], "fr": json.dumps(st["frames"])}
                     for st in doc.get(stack, [])]
            if steps:
                bind.execute(sa.text("INSERT INTO session_history (session_key, stack, label, frames) "
                                     "VALUES (:k, :s, :l, CAST(:fr AS jsonb))"), steps)
        meta = {k: v for k, v in doc.items() if k not in PARTS}
        bind.execute(sa.text("UPDATE sessions SET document = CAST(:d AS jsonb) WHERE key = :k"),
                     {"d": json.dumps(meta), "k": key})


def downgrade() -> None:
    bind = op.get_bind()
    keys = [r[0] for r in bind.execute(sa.text("SELECT key FROM sessions ORDER BY key")).all()]
    for key in keys:
        doc, revision = bind.execute(sa.text("SELECT document, revision FROM sessions WHERE key = :k"),
                                     {"k": key}).one()
        frames = bind.execute(sa.text("SELECT frame, annotations FROM session_frames WHERE session_key = :k "
                                      "ORDER BY frame"), {"k": key}).all()
        doc = {**doc, "revision": revision, "frames": {str(f): anns for f, anns in frames}}
        for stack in ("undo", "redo"):
            steps = bind.execute(sa.text("SELECT label, frames FROM session_history WHERE session_key = :k "
                                         "AND stack = :s ORDER BY id DESC LIMIT :n"),
                                 {"k": key, "s": stack, "n": UNDO_LIMIT}).all()
            doc[stack] = [{"label": label, "frames": fr} for label, fr in reversed(steps)]
        bind.execute(sa.text("UPDATE sessions SET document = CAST(:d AS jsonb) WHERE key = :k"),
                     {"d": json.dumps(doc), "k": key})
    op.drop_index("ix_session_history_session_key_stack_id", table_name="session_history")
    op.drop_table("session_history")
    op.drop_table("session_frames")
