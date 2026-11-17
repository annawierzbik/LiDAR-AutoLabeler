"""Load (resume) or create the annotation session of a sequence: the web server's session builder.

A KITTI sequence (--data-root/--split/--sequence) or any folder of .bin frames (--folder).
Boxes come from the detection + tracking cache in results/ (pipeline.detect_track); when there is
no usable cache, detection + tracking run first. Sessions live in results/sessions/ and are resumed
automatically (fresh=True starts over). In manual mode no detector runs.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from detection.fov import add_fov_argument, sequence_mode
from ingestion.sequence import SequenceLoader
from verification import cache
from verification.paths import RESULTS_DIR, session_file  # noqa: F401  (RESULTS_DIR: read by infra.storage)
from verification.session import AnnotationSession


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--folder", default=None,
                   help="any folder with KITTI-style .bin frames (overrides --data-root/--split/--sequence)")
    p.add_argument("--blob-prefix", default=None, help=argparse.SUPPRESS)   # an upload in the uploads blob area (API)
    p.add_argument("--mode", choices=["assisted", "manual"], default="assisted")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None, help="exclusive; default = whole sequence")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fresh", action="store_true", help="ignore a saved session and start over")
    p.add_argument("--use-cache-without-weights", action="store_true",
                   help="use matching cached detections even though the checkpoint is missing on disk")
    add_fov_argument(p)


def make_args(**overrides) -> argparse.Namespace:
    """The arguments of add_arguments with their defaults, plus `overrides`."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def file_names(args) -> tuple[str, str, str | None]:
    """(sequence label, file key, legacy file key) of the sequence in `args`.

    The file key names the session and cache files: a folder's folder_key, or <split>_<sequence>
    for a KITTI sequence (training/0000 and testing/0000 must not share files). Files written
    before the split was part of the key are named <sequence> only; they are still read for the
    split "training" (the default then), which is the legacy key. The sequence label is what the
    session stores as its sequence (used for export file names) and does not change. An upload
    (`blob_prefix`, seq-<random>) is its own label and key: the prefix is unique."""
    prefix = getattr(args, "blob_prefix", None)
    if prefix:
        return prefix, prefix, None
    if args.folder:
        from ingestion.folder import folder_key

        key = folder_key(args.folder)   # unique per folder
        return key, key, None
    return args.sequence, f"{args.split}_{args.sequence}", args.sequence if args.split == "training" else None


def cache_key_name(name: str, indices: range, num_frames: int) -> str:
    """results/<name>.json for a whole sequence, <name>_f<start>-<stop>.json for a part of it."""
    whole = indices.start == 0 and indices.stop == num_frames
    return f"{name}.json" if whole else f"{name}_f{indices.start}-{indices.stop}.json"


def results_blobs():
    """The detection + tracking cache (keys = file names): a LocalBlobStore on results/, or the
    cache area of the S3 bucket (LAL_BLOB_STORE=s3, infra/storage.py)."""
    from infra.storage import results_store

    return results_store()


def _cache_candidates(blobs, name: str) -> list[str]:
    return [f"{name}.json"] + [k for k in blobs.list(f"{name}_")
                               if "/" not in k and k.endswith(".json") and cache.is_cache_name(k[:-5], name)]


def _find_cached(name: str, want: range, meta: dict, allow_missing_weights: bool = False,
                 legacy_name: str | None = None):
    """Cached detections that cover the frames in `want` and were made with the settings `meta`:
    (indices, results) sliced to `want`, or None. Looks at results/<name>.json (whole run) and
    results/<name>_f<start>-<stop>.json (partial runs), each optionally with a _c<key> suffix
    (verification/cache.py: files with other settings are not reused, files without settings are,
    with a warning), then at the same names with `legacy_name`."""
    from pipeline.detect_track import parse_results

    blobs = results_blobs()
    candidates = _cache_candidates(blobs, name) + (_cache_candidates(blobs, legacy_name) if legacy_name else [])
    found = cache.choose_blob(blobs, candidates, meta,
                              usable=lambda d: d["start"] <= want.start and want.stop <= d["stop"],
                              allow_missing_weights=allow_missing_weights)
    if found is None:
        return None
    indices, results = parse_results(found[1])
    offset = want.start - indices.start
    return want, results[offset: offset + len(want)]


def has_split(args) -> bool:
    """KITTI sequences have a split; folders and uploads do not (it is not part of their cache meta)."""
    return not (args.folder or getattr(args, "blob_prefix", None))


def open_sequence(args):
    """The SequenceLoader (KITTI), FolderSequence (--folder) or BlobSequence (an upload) of `args`."""
    if getattr(args, "blob_prefix", None):
        from infra.sequences import open_blob_sequence

        return open_blob_sequence(args)
    if args.folder:
        from ingestion.folder import FolderSequence

        return FolderSequence(args.folder)
    return SequenceLoader(Path(args.data_root), args.split, args.sequence)


def session_keys(args) -> tuple[str, str | None]:
    """(key, legacy key) of the session of `args` in a SessionStore: the session file name without
    .json (<file key>[_f<a>-<b>]_<mode>); the legacy key is the name used before the split was part
    of it (KITTI training sequences only, docs/adr/0006).

    With `args.owner` (a user id, API v1) the key starts with u<owner>_, so every user has their
    own session of a sequence (docs/adr/0017); such keys have no legacy key."""
    _, name, legacy = file_names(args)
    key = session_file(name, args.mode, args.start, args.stop).stem
    owner = getattr(args, "owner", None)
    if owner is not None:
        return owned_key(owner, key), None
    return key, (session_file(legacy, args.mode, args.start, args.stop).stem if legacy else None)


def owned_key(owner: int, key: str) -> str:
    """The key of a user's session: u<owner>_<key without owner> (docs/adr/0017)."""
    return f"u{int(owner)}_{key}"


_FOLDER_KEY = re.compile(r"_[0-9a-f]{6}$")


def key_for_owner(key: str, sequence_label: str, owner: int) -> str:
    """The key a session stored before phase 4 (no owner) gets when it is assigned to `owner`:
    the key a tab of that user computes for the same sequence, range and mode.

    Keys of KITTI training sequences written before the split was part of the name (legacy keys,
    <sequence>[_f<a>-<b>]_<mode>, docs/adr/0006) get the split: training_<sequence>...; an owner
    key has no legacy form. A folder's key (<name>_<6 hex>) and a <split>_<sequence> key are kept."""
    legacy = (key.startswith(sequence_label + "_") and not _FOLDER_KEY.search(sequence_label)
              and not key.startswith(("training_", "testing_")))
    return owned_key(owner, f"training_{key}" if legacy else key)


def new_session(args, seq, detect=None, detector: dict | None = None) -> AnnotationSession:
    """A new session from cached detections, or after running detection + tracking (assisted), or
    empty (manual: no detector at all, so it works without a GPU). --start/--stop select a part.

    `detect(args)`: runs detection + tracking elsewhere (a job of the TaskQueue, infra/tasks.py)
    and returns when its result is in the detection cache; without it the detector runs here.
    `detector`: the fingerprint ({"detector_config_hash", "checkpoint_hash"}) of the model the GPU
    workers load, used instead of this process's files to decide whether the cache has a result
    (a process without the weights, e.g. a cpu worker; infra.worker_detector.published_fingerprint)."""
    label, name, legacy = file_names(args)
    stop = min(args.stop if args.stop is not None else len(seq), len(seq))
    want = range(args.start, stop)
    if len(want) == 0:
        raise ValueError(f"Empty frame range {args.start}..{args.stop} (the sequence has {len(seq)} frames)")

    if args.mode == "manual":
        indices, results = want, [[] for _ in want]
    else:
        from pipeline.detect_track import current_meta

        split = args.split if has_split(args) else None
        meta = current_meta(args.threshold, sequence_mode(seq, args.fov), split)
        if detector:
            meta.update({k: detector[k] for k in ("detector_config_hash", "checkpoint_hash")})
        found = _find_cached(name, want, meta, args.use_cache_without_weights, legacy)
        if found is None and detect is not None:
            detect(args)
            found = _find_cached(name, want, meta, args.use_cache_without_weights, legacy)
            if found is None:
                raise RuntimeError("detection finished, but its result is not in the detection cache")
        if found is not None:
            indices, results = found
        else:
            from pipeline.detect_track import track_sequence, write_results

            indices = want
            print(f"Detecting + tracking frames {indices.start}..{indices.stop - 1}")
            results = track_sequence(seq, indices, args.threshold, args.fov)
            blobs = results_blobs()
            key = cache_key_name(name, indices, len(seq))
            write_results(blobs, cache.output_key(blobs, key, meta), indices, results, meta)
    session = AnnotationSession.from_tracking(
        label, results, args.mode, frame_offset=indices.start)
    print(f"New {args.mode} session with {session.stats()['total']} system boxes in {len(indices)} frames")
    return session


def prepare(args, store, detect=None, detector: dict | None = None) -> tuple[object, str]:
    """Make sure the session of `args` exists in `store`; returns (sequence, key).

    An existing session is resumed ("session_resume" is logged), one stored under its legacy key is
    copied to the new key (with its event log), otherwise a new one is made (new_session). With
    args.fresh an existing session is replaced by a new one."""
    from infra.sessions import SessionExists

    seq = open_sequence(args)
    key, legacy = session_keys(args)
    if store.exists(key) and not args.fresh:
        store.append_events(key, [_resume_event()])
        return seq, key
    if legacy and store.exists(legacy) and not args.fresh:
        old = store.load(legacy)
        old.events = store.events(legacy) + [_resume_event()]
        store.create(key, old)
        print(f"Resumed session {legacy} under its new name {key}")
        return seq, key
    session = new_session(args, seq, detect, detector)
    if store.exists(key):                                    # fresh: start over
        store.replace(key, session)
    else:
        try:
            store.create(key, session)
        except SessionExists:                                # another tab made it meanwhile: use that one
            pass
    return seq, key


def _resume_event():
    import time

    from verification.session import Event

    return Event(time.time(), "session_resume")


def build_session(args):
    """Load (resume) or create the session for `args` from the session files. Returns (session,
    sequence loader, session path). The web server uses `prepare` with its SessionStore instead."""
    seq = open_sequence(args)
    _, name, legacy = file_names(args)
    session_path = session_file(name, args.mode, args.start, args.stop)

    if not args.fresh:
        legacy_path = session_file(legacy, args.mode, args.start, args.stop) if legacy else None
        source = session_path if session_path.exists() else (
            legacy_path if legacy_path is not None and legacy_path.exists() else None)
        if source is not None:
            session = AnnotationSession.load(source)
            moved = "" if source == session_path else f" (from the old file name; now saved as {session_path.name})"
            print(f"Resumed session {source}{moved} ({session.stats()['reviewed']} boxes reviewed)")
            return session, seq, session_path
    return new_session(args, seq), seq, session_path
