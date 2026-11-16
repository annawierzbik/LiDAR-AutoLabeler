"""initial schema: users, sequences, jobs, sessions, session_events

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column("username", sa.String(100), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text()),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "sequences",
        sa.Column("id", sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("location", sa.Text(), nullable=False),
        sa.Column("split", sa.String(20)),
        sa.Column("num_frames", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("result", postgresql.JSONB()),
        sa.Column("cache_key", sa.String(128)),
        sa.Column("session_key", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_jobs_cache_key", "jobs", ["cache_key"])
    op.create_table(
        "sessions",
        sa.Column("key", sa.Text(), primary_key=True),
        sa.Column("document", postgresql.JSONB(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("locked_by", sa.String(64)),
        sa.Column("locked_at", sa.DateTime(timezone=True)),
        sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("sequence_id", sa.Integer(), sa.ForeignKey("sequences.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "session_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("session_key", sa.Text(), sa.ForeignKey("sessions.key", ondelete="CASCADE"), nullable=False),
        sa.Column("t", sa.Double(), nullable=False),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("frame", sa.Integer()),
        sa.Column("detail", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_session_events_session_key_id", "session_events", ["session_key", "id"])


def downgrade() -> None:
    op.drop_index("ix_session_events_session_key_id", table_name="session_events")
    op.drop_table("session_events")
    op.drop_table("sessions")
    op.drop_index("ix_jobs_cache_key", table_name="jobs")
    op.drop_table("jobs")
    op.drop_table("sequences")
    op.drop_table("users")
