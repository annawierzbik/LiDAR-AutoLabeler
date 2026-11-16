"""jobs: columns for the worker queue (payload, attempts, cancellation, heartbeat)

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("queue", sa.String(20), nullable=False, server_default="cpu"))
    op.add_column("jobs", sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"))
    op.add_column("jobs", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("jobs", sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("jobs", sa.Column("heartbeat_at", sa.DateTime(timezone=True)))
    op.add_column("jobs", sa.Column("worker", sa.Text()))
    op.add_column("jobs", sa.Column("started_at", sa.DateTime(timezone=True)))
    op.add_column("jobs", sa.Column("finished_at", sa.DateTime(timezone=True)))
    op.create_index("ix_jobs_state", "jobs", ["state"])


def downgrade() -> None:
    op.drop_index("ix_jobs_state", table_name="jobs")
    for column in ("finished_at", "started_at", "worker", "heartbeat_at", "cancel_requested", "attempts",
                   "payload", "queue"):
        op.drop_column("jobs", column)
