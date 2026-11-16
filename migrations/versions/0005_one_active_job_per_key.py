"""At most one queued or running job per dedupe key (jobs.cache_key), enforced by a partial unique index

Rows that already break the rule (two active jobs with the same key, possible before this
migration because the check and the insert were separate statements) keep the newest one active;
the older ones are marked failed with a message naming the job that remains.

Revision ID: 0005
Revises: 0004
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

ACTIVE = "state IN ('queued', 'running')"


def upgrade() -> None:
    op.execute(f"""
        UPDATE jobs AS old
           SET state = 'failed', finished_at = now(),
               message = 'superseded by job ' || keep.id || ' with the same key (migration 0005)'
          FROM (SELECT DISTINCT ON (cache_key) id, cache_key FROM jobs
                 WHERE cache_key IS NOT NULL AND {ACTIVE}
                 ORDER BY cache_key, created_at DESC, id) AS keep
         WHERE old.cache_key = keep.cache_key AND old.id <> keep.id AND old.{ACTIVE}
    """)
    op.execute(f"CREATE UNIQUE INDEX ux_jobs_active_cache_key ON jobs (cache_key) WHERE {ACTIVE}")


def downgrade() -> None:
    op.drop_index("ux_jobs_active_cache_key", table_name="jobs")
