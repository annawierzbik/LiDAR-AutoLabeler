"""Give sessions stored before phase 4 (no owner) to a user, so a browser tab of theirs opens them.

    python -m scripts.assign_session_owner --list                    # sessions without an owner
    python -m scripts.assign_session_owner anna --all --dry-run      # show the new keys only
    python -m scripts.assign_session_owner anna --all
    python -m scripts.assign_session_owner anna training_0000_assisted 0000_f0-20_manual

Each session gets the key a tab of that user computes for the same sequence, range and mode
(u<id>_<key>; a legacy KITTI key also gains its split, pipeline.session_builder.key_for_owner) and
the owner; frames, history and events move with it in one transaction (PgSessionStore.assign_owner).
A tab opens it when the user opens the same sequence: for a folder, a sequence registered at the
same path; for KITTI, the same split and sequence. Sessions that already have an owner, are locked,
or whose new key is taken are reported and left as they are. Needs LAL_DATABASE_URL.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from infra.auth import UserStore
    from infra.db import make_engine
    from infra.pg_sessions import PgSessionStore
    from infra.sessions import SessionExists, SessionNotFound
    from infra.settings import get_settings
    from pipeline.session_builder import key_for_owner

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("username", nargs="?")
    p.add_argument("keys", nargs="*", help="session keys (without an owner) to assign")
    p.add_argument("--all", action="store_true", help="every session without an owner")
    p.add_argument("--list", action="store_true", help="list the sessions without an owner and stop")
    p.add_argument("--dry-run", action="store_true", help="print the new keys, change nothing")
    args = p.parse_args(argv)

    url = get_settings().database_url
    if not url:
        print("LAL_DATABASE_URL is not set", file=sys.stderr)
        return 2
    engine = make_engine(url)
    try:
        store = PgSessionStore(engine, lock_ttl=get_settings().lock_ttl_seconds)
        unowned = dict(store.unowned())
        if args.list:
            for key, label in unowned.items():
                print(f"{key}\t(sequence {label})")
            print(f"{len(unowned)} session(s) without an owner")
            return 0
        if not args.username or (not args.all and not args.keys) or (args.all and args.keys):
            p.error("give a user name and either --all or session keys")
        user = UserStore(engine).by_name(args.username)
        if user is None:
            print(f"no user {args.username!r}", file=sys.stderr)
            return 2
        keys = list(unowned) if args.all else args.keys
        failed = 0
        for key in keys:
            if key not in unowned:
                print(f"skipped {key}: no such session without an owner", file=sys.stderr)
                failed += 1
                continue
            new = key_for_owner(key, unowned[key], user.id)
            if args.dry_run:
                print(f"{key} -> {new}")
                continue
            try:
                store.assign_owner(key, user.id, new)
            except (ValueError, SessionExists, SessionNotFound) as exc:
                print(f"skipped {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
                failed += 1
                continue
            print(f"{key} -> {new} (user {user.username}, id {user.id})")
        return 1 if failed else 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
