"""Benchmark of one API request on a session store (docs/adr/0011): load, one edit, save.

    python -m scripts.bench_session_store                       # PostgreSQL of docker-compose.db.yml + file store
    python -m scripts.bench_session_store --url postgresql+psycopg://... --runs 20 --out results/benchmarks/x.csv

Sessions: 400 frames x 10 tracks (4000 boxes), with an undo history of
    typical      50 single-box edits
    track-10/25/50   10, 25, 50 whole-track accepts/rejects (each step holds every frame of the track)
Every request: store.load(key) -> session.edit(one box) -> ReviewService-style state(frame) ->
store.save(key, session, expected_revision). The stages are timed separately; "request" is
load + edit + save (the measure of note H), "with_state" adds building the response.
A throw-away database is created and dropped for the PostgreSQL runs. Raw results go to a CSV
file (one row per request); the summary printed at the end is the median per variant.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from common.boxes import BoundingBox3D
from infra.settings import PROJECT_ROOT
from verification.session import AnnotationSession

FRAMES, TRACKS = 400, 10
VARIANTS = {"typical": ("edit", 50), "track-10": ("track", 10), "track-25": ("track", 25), "track-50": ("track", 50)}
STAGES = ("load", "edit", "state", "save")


def build(kind: str, steps: int) -> AnnotationSession:
    results = [[BoundingBox3D(5.0 + 3 * k + 0.05 * f, 2.0 * k - 10, -1.0, 4.0, 1.8, 1.5, 0.1, "Car", 0.8, k)
                for k in range(TRACKS)] for f in range(FRAMES)]
    s = AnnotationSession.from_tracking("0000", results)
    for i in range(steps):
        if kind == "edit":
            s.edit(i % FRAMES, 0, x=20.0 + i * 0.01)
        else:
            (s.accept_track if i % 2 == 0 else s.reject_track)(i % TRACKS)
    return s


def state(s: AnnotationSession, frame: int) -> None:
    """What ReviewService.state computes for the response."""
    s.stats()
    [1 if s.frame_has_pending(f) else (2 if s.annotations(f) else 0) for f in range(s.num_frames)]
    [a.to_dict() for a in s.annotations(frame)]


def run(store, store_name: str, runs: int, rows: list[dict]) -> None:
    for variant, (kind, steps) in VARIANTS.items():
        key = f"bench_{variant}"
        store.create(key, build(kind, steps))
        for i in range(runs):
            t = [time.perf_counter()]
            s = store.load(key)
            rev = s.revision
            t.append(time.perf_counter())
            s.edit((i * 37) % FRAMES, 1, x=30.0 + i * 0.01)
            t.append(time.perf_counter())
            state(s, (i * 37) % FRAMES)
            t.append(time.perf_counter())
            store.save(key, s, expected_revision=rev)
            t.append(time.perf_counter())
            ms = {name: (t[k + 1] - t[k]) * 1000 for k, name in enumerate(STAGES)}
            rows.append({"store": store_name, "variant": variant, "undo_steps": steps, "run": i, **ms,
                         "request": ms["load"] + ms["edit"] + ms["save"], "with_state": sum(ms.values())})


def main(argv=None) -> int:
    from infra.sessions import FileSessionStore

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="postgresql+psycopg://lal:lal_local@127.0.0.1:55432/lal",
                   help="a database where a throw-away benchmark database may be created")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--no-file", action="store_true", help="skip the file store")
    p.add_argument("--out", default=None, help="CSV file (default: results/benchmarks/session_store_<time>.csv)")
    args = p.parse_args(argv)

    rows: list[dict] = []
    if not args.no_file:
        run(FileSessionStore(tempfile.mkdtemp()), "file", args.runs, rows)

    from sqlalchemy import create_engine, text

    from infra.db import make_engine
    from infra.migrate import upgrade
    from infra.pg_sessions import PgSessionStore

    name = f"lal_bench_{uuid.uuid4().hex[:8]}"
    admin = create_engine(args.url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        url = args.url.rsplit("/", 1)[0] + "/" + name
        upgrade(url)
        engine = make_engine(url)
        run(PgSessionStore(engine), "postgres", args.runs, rows)
        engine.dispose()
    finally:
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))

    out = Path(args.out) if args.out else (PROJECT_ROOT / "results" / "benchmarks" /
                                           f"session_store_{dt.datetime.now():%Y%m%d-%H%M%S}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"median of {args.runs} requests, ms (raw: {out})")
    print(f"{'store':9s}{'variant':10s}" + "".join(f"{c:>11s}" for c in (*STAGES, "request", "with_state")))
    for store_name in dict.fromkeys(r["store"] for r in rows):
        for variant in VARIANTS:
            sel = [r for r in rows if r["store"] == store_name and r["variant"] == variant]
            print(f"{store_name:9s}{variant:10s}" + "".join(
                f"{statistics.median(r[c] for r in sel):11.1f}" for c in (*STAGES, "request", "with_state")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
