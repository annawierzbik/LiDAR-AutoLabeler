"""Alembic environment: the database URL is the `sqlalchemy.url` option when set by the caller
(tests, infra.migrate) or LAL_DATABASE_URL."""
from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from infra.db import Base
from infra.settings import Settings

config = context.config
url = config.get_main_option("sqlalchemy.url") or Settings().database_url
if not url:
    raise RuntimeError("no database URL: set LAL_DATABASE_URL")

if context.is_offline_mode():
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    with create_engine(url).connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()
