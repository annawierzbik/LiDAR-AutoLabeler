"""Create a user of the web API (the first admin, or any other).

    python -m scripts.create_user admin --role admin            # asks for the password twice
    echo <password> | python -m scripts.create_user anna --password-stdin

Needs LAL_DATABASE_URL (the database at the newest migration: python -m infra.migrate).
Further users can also be created by an admin through POST /api/v1/users.
"""
from __future__ import annotations

import argparse
import getpass
import sys


def main(argv: list[str] | None = None) -> int:
    from infra.auth import ROLES, UserExists, UserStore
    from infra.db import make_engine
    from infra.settings import get_settings

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("username")
    p.add_argument("--role", choices=ROLES, default="annotator")
    p.add_argument("--password-stdin", action="store_true", help="read the password from the first line of stdin")
    args = p.parse_args(argv)

    url = get_settings().database_url
    if not url:
        print("LAL_DATABASE_URL is not set", file=sys.stderr)
        return 2
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass("Password: ")
        if getpass.getpass("Repeat: ") != password:
            print("the passwords differ", file=sys.stderr)
            return 1
    engine = make_engine(url)
    try:
        user = UserStore(engine).create(args.username, password, args.role)
    except (UserExists, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print(f"created user {user.username} (id {user.id}, role {user.role})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
