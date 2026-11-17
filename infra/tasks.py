"""Background tasks, independent of the queue that runs them (ThreadTaskQueue or Celery workers).

Each task is `fn(payload: dict, ctx: TaskContext) -> dict` (JSON-able result):
    detect_track         gpu queue: detection + tracking of a sequence into the detection cache
    open_session         gpu queue: build (or resume) a user's session of a sequence; detection +
                         tracking first when the cache has no result (API v1, docs/adr/0017)
    open_session_manual  cpu queue: the same for manual mode (no detector)
    export               cpu queue: label files of a session as a zip in the export BlobStore,
                         deleted after LAL_EXPORT_TTL_HOURS

Tasks are idempotent (docs/adr/0015): detect_track first looks for a cache entry with the run's
settings and returns it without loading a model; a run that is stopped (cancelled, or its worker
died) writes nothing, because the cache entry is written once, atomically, at the end.
"""
from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from collections.abc import Callable
from typing import Protocol

from infra.blob import BlobStore

GPU, CPU = "gpu", "cpu"


class TaskContext(Protocol):
    keeps_detector: bool
    """True in a GPU worker process (one detector, loaded once); False in thread mode (a detector
    per job, released afterwards)."""

    def progress(self, fraction: float, message: str = "") -> None:
        """Report progress (0..1); implementations throttle the writes."""

    def should_stop(self) -> bool:
        """True once the job was asked to stop (cooperative cancellation)."""


class JobCancelled(Exception):
    pass


class TaskFailed(Exception):
    """A failure whose message is meant for the user as it is (no exception class name)."""


def results_blobs() -> BlobStore:
    """The detection cache (results/, the same store the session builder reads)."""
    from pipeline import session_builder

    return session_builder.results_blobs()


def export_blobs() -> BlobStore:
    from infra.storage import exports_store

    return exports_store()


# ------------------------------------------------------------------ detect_track
def detect_track_payload(args) -> dict:
    """The JSON payload of a detect_track job for session-builder arguments."""
    out = {k: getattr(args, k) for k in ("folder", "data_root", "split", "sequence", "start", "stop", "threshold",
                                        "fov", "use_cache_without_weights")}
    if getattr(args, "blob_prefix", None):
        out["blob_prefix"] = args.blob_prefix
    return out


def detect_track(payload: dict, ctx: TaskContext) -> dict:
    from detection.fov import sequence_mode
    from infra.sequences import open_blob_sequence
    from infra.worker_detector import get_detector, make_for_job, release
    from pipeline.detect_track import (
        Cancelled,
        current_meta,
        track_sequence,
        write_results,
    )
    from pipeline.session_builder import (
        _find_cached,
        cache_key_name,
        file_names,
        has_split,
    )
    from verification import cache

    args = Namespace(**payload)
    seq = open_blob_sequence(args)
    _, name, legacy = file_names(args)
    stop = min(args.stop if args.stop is not None else len(seq), len(seq))
    want = range(args.start, stop)
    if len(want) == 0:
        raise ValueError(f"Empty frame range {args.start}..{args.stop} (the sequence has {len(seq)} frames)")
    meta = current_meta(args.threshold, sequence_mode(seq, args.fov), args.split if has_split(args) else None)
    if _find_cached(name, want, meta, args.use_cache_without_weights, legacy) is not None:
        ctx.progress(1.0, "found in the detection cache")
        return {"cached": True, "frames": len(want)}
    ctx.progress(0.0, "loading the detector")
    keep = getattr(ctx, "keeps_detector", False)
    detector = get_detector(meta, args.fov) if keep else make_for_job(meta, args.fov)
    try:
        results = track_sequence(seq, want, args.threshold, args.fov, detector=detector,
                                 should_stop=ctx.should_stop, progress=lambda f: ctx.progress(f, "detecting"))
    except Cancelled as exc:
        raise JobCancelled(str(exc)) from None
    finally:
        if not keep:
            detector = None                                         # the only reference
            release()
    blobs = results_blobs()
    key = cache.output_key(blobs, cache_key_name(name, want, len(seq)), meta)
    write_results(blobs, key, want, results, meta)
    return {"cached": False, "frames": len(want), "cache_key": key}


# ------------------------------------------------------------------ open_session
def open_session_payload(spec: dict, owner_id: int, sequence_id: int, tab_id: str, mode: str = "assisted",
                         start: int = 0, stop: int | None = None) -> dict:
    """Payload of an open_session(_manual) job: session-builder arguments of the sequence `spec`."""
    from pipeline.session_builder import make_args

    args = make_args(**spec, mode=mode, start=start, stop=stop)
    return {"args": vars(args), "owner_id": owner_id, "sequence_id": sequence_id, "tab_id": tab_id}


def open_session_kind(mode: str) -> str:
    return "open_session_manual" if mode == "manual" else "open_session"


def open_session(payload: dict, ctx: TaskContext) -> dict:
    """Make sure the owner's session of the sequence exists (resume, cached detections, or
    detection + tracking right here, in this job) and return what a request needs to serve it."""
    from infra.settings import make_session_store
    from pipeline.session_builder import prepare

    args = Namespace(**payload["args"], owner=payload["owner_id"])
    store = make_session_store()

    def detect(a) -> None:
        detect_track(detect_track_payload(a), ctx)

    if ctx.should_stop():
        raise JobCancelled("cancelled")
    try:
        _, key = prepare(args, store, detect)
    except (FileNotFoundError, ValueError) as exc:
        raise TaskFailed(str(exc)) from None
    if hasattr(store, "set_owner"):
        store.set_owner(key, payload["owner_id"], payload["sequence_id"])
    s = store.load(key)
    ctx.progress(1.0, "ready")
    return {"session_key": key, "num_frames": s.num_frames, "frame_offset": s.frame_offset,
            "sequence": s.sequence}


# ------------------------------------------------------------------------ export
def export_key(expires_at: float, job_id: str, archive_name: str) -> str:
    """Export blobs are named <expiry, epoch seconds>/<job id>/<archive>, so expired ones are found
    and deleted without a database."""
    return f"{int(expires_at)}/{job_id}/{archive_name}"


def prune_exports(blobs: BlobStore, now: float) -> list[str]:
    """Delete the export blobs whose expiry has passed; returns their keys."""
    gone = []
    for key in blobs.list():
        head = key.split("/", 1)[0]
        if head.isdigit() and int(head) <= now:
            blobs.delete(key)
            gone.append(key)
    return gone


EXPORT_REUSE_MARGIN = 60.0     # a reused export must stay downloadable at least this long (seconds)


def export_reusable(result: dict | None, now: float | None = None) -> bool:
    """Whether a finished export can serve a new request for the same session state: its zip has
    not expired (with EXPORT_REUSE_MARGIN to spare) and was not pruned. Otherwise the request
    makes a new export job (review after phase 4)."""
    import time

    if not result or "blob" not in result:
        return False
    now = time.time() if now is None else now
    return result.get("expires_at", 0) > now + EXPORT_REUSE_MARGIN and export_blobs().exists(result["blob"])


def export(payload: dict, ctx: TaskContext) -> dict:
    """payload: session_key, revision (so an export of an older state is not reused), sequence
    (session-builder arguments), format, include_unreviewed, include_empty, job_id.
    The session is loaded when the job runs: the result's `revision` is the revision actually
    exported (it is newer than `requested_revision` if the session changed while the job waited).
    The zip is kept for LAL_EXPORT_TTL_HOURS; every export run deletes the expired ones."""
    import time

    from export import ExportError, export_session, zip_files
    from infra.settings import get_settings, make_session_store
    from pipeline.session_builder import open_sequence

    blobs = export_blobs()
    now = time.time()
    prune_exports(blobs, now)
    session = make_session_store().load(payload["session_key"])
    seq = open_sequence(Namespace(**payload["sequence"]))
    ctx.progress(0.2, "rendering labels")
    try:
        result = export_session(session, seq, payload["format"], payload.get("include_unreviewed", False),
                                payload.get("include_empty", False))
    except ExportError as exc:
        raise TaskFailed(str(exc)) from None
    if ctx.should_stop():
        raise JobCancelled("export cancelled")
    data = zip_files(result.files)
    expires_at = now + get_settings().export_ttl_hours * 3600
    key = export_key(expires_at, payload["job_id"], result.archive_name)
    blobs.put(key, data)
    return {"blob": key, "archive_name": result.archive_name, "bytes": len(data), "expires_at": int(expires_at),
            "revision": session.revision, "requested_revision": payload.get("revision"),
            "summary": result.summary.to_dict()}


TASKS: dict[str, tuple[str, Callable[[dict, TaskContext], dict]]] = {
    "detect_track": (GPU, detect_track),
    "open_session": (GPU, open_session),
    "open_session_manual": (CPU, open_session),
    "export": (CPU, export),
}


def dedupe_key(kind: str, payload: dict) -> str:
    """Jobs with the same kind and payload are the same job: a duplicate submit returns the first."""
    body = {k: v for k, v in payload.items() if k != "job_id"}
    return hashlib.sha1(json.dumps([kind, body], sort_keys=True, default=str).encode()).hexdigest()[:24]
