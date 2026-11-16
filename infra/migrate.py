"""Apply the Alembic migrations to a database.

    python -m infra.migrate                      # LAL_DATABASE_URL, to the newest revision
    python -m infra.migrate --url postgresql+psycopg://... --revision 0001
"""
from __future__ import annotations

import argparse
from pathlib import Path

from alembic import command
from alembic.config import Config

from infra.settings import PROJECT_ROOT, Settings


def alembic_config(url: str) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(Path(PROJECT_ROOT) / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str) -> None:
    command.downgrade(alembic_config(url), revision)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=None, help="default: LAL_DATABASE_URL")
    p.add_argument("--revision", default="head")
    args = p.parse_args()
    url = args.url or Settings().database_url
    if not url:
        raise SystemExit("no database URL: pass --url or set LAL_DATABASE_URL")
    upgrade(url, args.revision)
    print(f"database at revision {args.revision}")


if __name__ == "__main__":
    main()
