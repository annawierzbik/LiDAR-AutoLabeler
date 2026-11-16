"""Import session files (results/sessions/*.json, v1 or v2) into the PostgreSQL session store.

    python -m scripts.import_sessions                          # LAL_DATABASE_URL, results/sessions
    python -m scripts.import_sessions --folder D:\\old\\sessions --url postgresql+psycopg://...
    python -m scripts.import_sessions --replace                # overwrite sessions that already exist
    python -m scripts.import_sessions --owner anna             # the sessions belong to user anna

Each file becomes one row of `sessions` (key = file name without .json, same as FileSessionStore,
metadata only), its frames rows of `session_frames`, its undo/redo history rows of
`session_history` (at most the newest 50 steps per stack) and its event log rows of
`session_events`. Sessions that already exist are skipped unless
--replace is given (then the old row and its events are deleted first). The files are not changed.

Without --owner the sessions have no owner: admins see them, but no browser tab opens them. With
--owner the key becomes the one a tab of that user opens for the same sequence, range and mode
(u<id>_<key>, a legacy KITTI key also gains its split: pipeline.session_builder.key_for_owner).
Sessions imported earlier without an owner are assigned with scripts.assign_session_owner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from infra.db import make_engine
from infra.pg_sessions import PgSessionStore
from infra.settings import Settings
from verification.session import AnnotationSession


def import_folder(folder: Path, store: PgSessionStore, replace: bool = False, owner_id: int | None = None) -> dict:
    """{"imported": [...], "skipped": [...], "failed": {key: reason}}; keys as stored."""
    from pipeline.session_builder import key_for_owner

    out: dict = {"imported": [], "skipped": [], "failed": {}}
    for path in sorted(Path(folder).glob("*.json")):
        key = path.stem
        try:
            session = AnnotationSession.from_dict(json.loads(path.read_text()))   # keeps its events
        except (OSError, ValueError, KeyError, TypeError) as exc:
            out["failed"][key] = f"{type(exc).__name__}: {exc}"
            continue
        if owner_id is not None:
            key = key_for_owner(key, session.sequence, owner_id)
        if store.exists(key):
            if not replace:
                out["skipped"].append(key)
                continue
            store.delete(key)
        store.create(key, session)
        if owner_id is not None:
            store.set_owner(key, owner_id, None)        # the sequence is linked when a tab opens it
        out["imported"].append(key)
    return out


def main(argv=None) -> int:
    settings = Settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default=str(settings.session_dir), help="default: results/sessions")
    p.add_argument("--url", default=settings.database_url, help="default: LAL_DATABASE_URL")
    p.add_argument("--replace", action="store_true", help="overwrite sessions that already exist")
    p.add_argument("--owner", default=None, help="user name the imported sessions belong to")
    args = p.parse_args(argv)
    if not args.url:
        print("No database URL: pass --url or set LAL_DATABASE_URL")
        return 2
    engine = make_engine(args.url)
    owner_id = None
    if args.owner is not None:
        from infra.auth import UserStore

        user = UserStore(engine).by_name(args.owner)
        if user is None:
            print(f"no user {args.owner!r}")
            return 2
        owner_id = user.id
    result = import_folder(Path(args.folder), PgSessionStore(engine), args.replace, owner_id)
    print(f"imported {len(result['imported'])}, skipped {len(result['skipped'])} (already there), "
          f"failed {len(result['failed'])}")
    for key, reason in result["failed"].items():
        print(f"  failed {key}: {reason}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
