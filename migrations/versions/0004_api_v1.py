"""API v1: token revocation, sequence specs, job owners, tabs (browser tab -> session binding)

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("sequences", sa.Column("spec", postgresql.JSONB(), nullable=False, server_default="{}"))
    op.create_index("ix_sequences_owner_id", "sequences", ["owner_id"])
    op.add_column("jobs", sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id")))
    op.create_index("ix_jobs_owner_id", "jobs", ["owner_id"])
    op.create_index("ix_sessions_owner_id", "sessions", ["owner_id"])
    op.create_table(
        "tabs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("client_id", sa.String(64), nullable=False),
        sa.Column("sequence_id", sa.Integer(), sa.ForeignKey("sequences.id"), nullable=False),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("jobs.id")),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_tabs_user_id", "tabs", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_tabs_user_id", table_name="tabs")
    op.drop_table("tabs")
    op.drop_index("ix_sessions_owner_id", table_name="sessions")
    op.drop_index("ix_jobs_owner_id", table_name="jobs")
    op.drop_column("jobs", "owner_id")
    op.drop_index("ix_sequences_owner_id", table_name="sequences")
    op.drop_column("sequences", "spec")
    op.drop_column("users", "token_version")
