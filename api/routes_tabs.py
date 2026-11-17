"""Browser tabs on sessions (docs/adr/0017).

A tab is opened on a sequence: the API stores the tab row and submits the job that builds the
session on a worker (open_session on the gpu queue, open_session_manual on the cpu queue). The
start page polls the tab; once the job is done the tab takes the session's lock for its owner
`u<user id>/<client id>` (X-Client-Id, kept per browser tab in sessionStorage). Every request of
the annotation page reads the tab row and the job result, refreshes that lock and loads the
session document; nothing is kept in the API process between requests (docs/adr/0011). The same
owner gets the lock back at once (also after a server restart); another tab or user gets 409
until the lock is released (Finish) or has expired (LAL_LOCK_TTL_SECONDS without a heartbeat).
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from api.deps import ClientId, CurrentUser, Page, Services, Svc, owner_filter
from api.errors import ApiError, not_found, responses
from infra.auth import Principal
from infra.sessions import RevisionConflict, SessionLocked, SessionNotFound
from infra.tasks import (
    export_reusable,
    open_session_kind,
    open_session_payload,
    opening_reusable,
)
from web.service import ReviewService, UnknownAction, point_bytes

router = APIRouter(prefix="/api/v1", tags=["tabs"])

LOADING = ("queued", "running", "cancelling")


class OpenTab(BaseModel):
    sequence_id: int
    mode: str = Field("assisted", pattern="^(assisted|manual)$")
    start: int = Field(0, ge=0)
    stop: int | None = Field(None, ge=1, description="exclusive; default: the whole sequence")


class Action(BaseModel):
    name: str
    frame: int | None = None
    index: int | None = None
    track_id: int | None = None
    label: str | None = None


class ExportRequest(BaseModel):
    format: str = "lidar"
    include_unreviewed: bool = False
    include_empty: bool = False


@dataclass
class Ready:
    tab: dict
    key: str
    num_frames: int
    frame_offset: int
    spec: dict                  # the sequence's session-builder arguments (blob_prefix or folder)


def lock_owner(user: Principal, client: str) -> str:
    return f"u{user.id}/{client}"


def _tab(svc: Services, sid: str, user: Principal) -> dict:
    try:
        return svc.tabs.get(sid, owner_filter(user))
    except KeyError:
        raise not_found("tab", sid) from None


def _job(svc: Services, tab: dict):
    """The tab's opening job: the last of its chain (an opening on the cpu queue that needed
    detection names the gpu job it submitted as `next_job`)."""
    st = svc.queue.status(tab["job_id"])
    for _ in range(4):
        nxt = (st.result or {}).get("next_job") if st.state == "done" else None
        if not nxt:
            break
        st = svc.queue.status(nxt)
    return st


def _ready(svc: Services, sid: str, user: Principal) -> Ready:
    """The tab's session once its job is done; 409 while loading, after an error or when finished."""
    tab = _tab(svc, sid, user)
    if tab["status"] == "finished":
        raise ApiError(409, f"session {sid} was finished and is closed", code="finished")
    job = _job(svc, tab)
    if job.state != "done":
        msg = job.message if job.state in ("failed", "cancelled") else "the session is still loading"
        raise ApiError(409, msg or f"the session is not ready ({job.state})", code="not_ready")
    seq = svc.sequences.get(tab["sequence_id"], None)
    r = job.result
    return Ready(tab, r["session_key"], r["num_frames"], r["frame_offset"], seq["spec"])


def _lock(svc: Services, ready: Ready, user: Principal, client: str) -> None:
    try:
        svc.sessions.lock(ready.key, lock_owner(user, client))
    except SessionLocked:
        raise ApiError(409, "this session is open in another tab; continue there, or finish it first "
                            "(an abandoned tab frees it after a few minutes)", code="locked") from None
    except SessionNotFound:
        raise ApiError(409, f"session {ready.key} no longer exists", code="not_ready") from None


_LISTED: dict[str, list[str]] = {}      # sequences stored without "frames" (before the review of phase 4)


def _points(spec: dict):
    """The point clouds of a sequence spec, by the ordered frame names stored with the sequence
    (`spec["frames"]`); nothing is listed per request. Sequences stored without them are listed
    once per process."""
    from infra.blob import LocalBlobStore
    from infra.points import BlobPointSource
    from ingestion.folder import frame_files

    prefix, names = spec.get("blob_prefix"), spec.get("frames")
    if prefix:
        from infra.sequences import upload_frame_keys
        from infra.storage import uploads_store

        blobs = uploads_store()
        if names is None:
            names = _LISTED.get(prefix) or _LISTED.setdefault(
                prefix, [k[len(prefix) + 1:] for k in upload_frame_keys(blobs, prefix)])
        return BlobPointSource(blobs, [f"{prefix}/{n}" for n in names])
    folder = spec["folder"]
    if names is None:
        names = _LISTED.get(folder) or _LISTED.setdefault(folder, [p.name for p in frame_files(folder)])
    return BlobPointSource(LocalBlobStore(folder), list(names))


def review(svc: Services, sid: str, user: Principal, client: str) -> ReviewService:
    """A ReviewService for one request: tab and job read from the database, lock refreshed, document
    loaded."""
    ready = _ready(svc, sid, user)
    _lock(svc, ready, user, client)
    try:
        session = svc.sessions.load(ready.key)
    except SessionNotFound:
        raise ApiError(409, f"session {ready.key} no longer exists", code="not_ready") from None
    return ReviewService(session, svc.sessions, ready.key, _points(ready.spec), point_cache=svc.point_cache)


def _view(svc: Services, tab: dict, user: Principal, client: str) -> dict:
    out = {"sid": tab["id"], "title": tab["title"], "sequence_id": tab["sequence_id"], "job_id": tab["job_id"],
           "status": "loading", "message": "", "progress": None}
    if tab["status"] == "finished":
        return {**out, "status": "finished", "message": "finished"}
    job = _job(svc, tab)
    if job.state in LOADING:
        return {**out, "progress": {"state": job.state, "fraction": job.progress, "message": job.message,
                                    "attempts": job.attempts}}
    if job.state != "done":
        return {**out, "status": "error", "message": "cancelled" if job.state == "cancelled" else job.message}
    try:
        _lock(svc, _ready(svc, tab["id"], user), user, client)
    except ApiError as exc:
        return {**out, "status": "error", "message": exc.message}
    return {**out, "status": "ready", "num_frames": job.result["num_frames"]}


# ------------------------------------------------------------------ endpoints
@router.post("/tabs", status_code=201, responses=responses(404, 409))
def open_tab(body: OpenTab, user: CurrentUser, client: ClientId,
             svc: Svc):
    """Open a sequence in this browser tab: submits the job that builds (or resumes) the user's
    session. Poll GET /tabs/{sid} until its status is ready."""
    try:
        seq = svc.sequences.get(body.sequence_id, owner_filter(user))
    except KeyError:
        raise not_found("sequence", body.sequence_id) from None
    if body.stop is not None and body.stop <= body.start:
        raise ApiError(400, "stop must be greater than start")
    if body.mode == "assisted" and not svc.settings.assisted_mode:
        raise ApiError(409, "assisted mode is not available here (no GPU worker; LAL_ASSISTED_MODE=false); "
                            "open the sequence in manual mode", code="assisted_unavailable")
    sid = svc.tabs.new_id()
    owner = seq["owner_id"]                     # an admin opening a user's sequence works on that user's session
    payload = open_session_payload(seq["spec"], owner, seq["id"], body.mode, body.start, body.stop)
    title = f"{seq['name']} ({seq['num_frames']} frames)"
    job_id = svc.queue.submit(open_session_kind(body.mode), payload, title=title, reuse_done=opening_reusable)
    svc.tabs.create(sid, user.id, client, seq["id"], job_id, title)
    return {"sid": sid, "job_id": job_id}


@router.get("/tabs/{sid}", responses=responses(404))
def tab_status(sid: str, user: CurrentUser, client: ClientId,
               svc: Svc):
    """loading (with the job's progress) | ready | error | finished. When the session is ready the
    tab takes its lock; a session open in another tab gives status error with a message."""
    return _view(svc, _tab(svc, sid, user), user, client)


@router.post("/tabs/{sid}/cancel", responses=responses(404))
def cancel_tab(sid: str, user: CurrentUser, svc: Svc):
    """Stop opening the session (detection stops at the next frame)."""
    tab = _tab(svc, sid, user)
    return {"cancelled": svc.queue.cancel(_job(svc, tab).id)}


@router.get("/tabs/{sid}/state", responses=responses(400, 404, 409))
def state(sid: str, user: CurrentUser, client: ClientId, svc: Svc, frame: int = 0):
    """Boxes of `frame`, statistics, timeline marks, revision."""
    try:
        return review(svc, sid, user, client).state(frame)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None


@router.get("/tabs/{sid}/next_pending", responses=responses(400, 404, 409))
def next_pending(sid: str, frame: int, user: CurrentUser, client: ClientId,
                 svc: Svc):
    try:
        return {"frame": review(svc, sid, user, client).next_pending(frame)}
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None


@router.get("/tabs/{sid}/points/{frame}", responses=responses(404, 409),
            response_class=Response, openapi_extra={"responses": {"200": {"content": {"application/octet-stream": {}}}}})
def points(sid: str, frame: int, user: CurrentUser, svc: Svc):
    """Point cloud of a session frame: little-endian float32 x y z per point."""
    ready = _ready(svc, sid, user)
    if not 0 <= frame < ready.num_frames:
        raise not_found("frame", frame)
    data = point_bytes(_points(ready.spec), svc.point_cache, ready.key, ready.frame_offset + frame)
    return Response(data, media_type="application/octet-stream")


@router.post("/tabs/{sid}/action", responses=responses(400, 404, 409))
def action(sid: str, a: Action, user: CurrentUser, client: ClientId,
           svc: Svc):
    """One named action on an explicit target (docs/adr/0001); answers with the frame's state and
    "result". 409 code "revision_conflict" when the session was changed elsewhere meanwhile."""
    try:
        return review(svc, sid, user, client).act(a.name, frame=a.frame, index=a.index, track_id=a.track_id,
                                                  label=a.label)
    except UnknownAction:
        raise ApiError(400, f"unknown action '{a.name}'", code="unknown_action") from None
    except RevisionConflict as exc:
        raise ApiError(409, str(exc), code="revision_conflict") from None
    except (ValueError, TypeError) as exc:
        raise ApiError(400, str(exc)) from None


@router.post("/tabs/{sid}/heartbeat", responses=responses(404, 409))
def heartbeat(sid: str, user: CurrentUser, client: ClientId,
              svc: Svc):
    """Keep this tab's lock (the page sends one every 30 s)."""
    _lock(svc, _ready(svc, sid, user), user, client)
    return {"ok": True, "lock_ttl": svc.sessions.lock_ttl}


@router.post("/tabs/{sid}/finish", responses=responses(404, 409))
def finish(sid: str, user: CurrentUser, client: ClientId,
           svc: Svc):
    """Release the session and close the tab."""
    ready = _ready(svc, sid, user)
    _lock(svc, ready, user, client)
    svc.sessions.finish(ready.key, lock_owner(user, client))
    svc.tabs.finish(sid)
    return {"ok": True}


@router.post("/tabs/{sid}/exports", status_code=202, responses=responses(400, 404, 409))
def export(sid: str, r: ExportRequest, user: CurrentUser, client: ClientId,
           svc: Svc):
    """Submit an export of the session's labels (cpu queue). Poll GET /jobs/{job_id}; download the
    zip from GET /jobs/{job_id}/download until it expires (LAL_EXPORT_TTL_HOURS)."""
    from export import FORMATS
    from pipeline.session_builder import make_args

    if r.format not in FORMATS:
        raise ApiError(400, f"unknown export format '{r.format}' (use one of: {', '.join(FORMATS)})")
    ready = _ready(svc, sid, user)
    _lock(svc, ready, user, client)
    seq = svc.sequences.get(ready.tab["sequence_id"], None)
    payload = {"session_key": ready.key, "revision": svc.sessions.stored_revision(ready.key),
               "owner_id": seq["owner_id"], "sequence": vars(make_args(**seq["spec"])), "format": r.format,
               "include_unreviewed": r.include_unreviewed, "include_empty": r.include_empty}
    job_id = svc.queue.submit("export", payload, title=f"export {seq['name']} ({r.format})",
                              reuse_done=export_reusable)
    return {"job_id": job_id}


@router.get("/sessions", tags=["sessions"], responses=responses())
def list_sessions(page: Page, user: CurrentUser,
                  svc: Svc):
    """The user's sessions (admins: everyone's), most recently changed first."""
    return page.wrap(*svc.sessions.page(owner_filter(user), page.limit, page.offset))
