"""FastAPI app: start screen, annotation page and the JSON API behind them.

    GET  /                          start screen (open a folder)
    GET  /annotate?sid=<id>         the annotation page of one open session
    GET  /static/...                vendored assets (three.js 0.160.0, Schibsted Grotesk), no CDN
    POST /api/folder/open           {"path": "..."}              -> {"sid"}
    GET  /api/jobs/<sid>            loading | ready | error  (404 once the job failed or was finished)
    GET  /api/s/<sid>/state?frame=  boxes of that frame, stats, timeline marks, save_error, revision
    GET  /api/s/<sid>/next_pending?frame=   {"frame": next frame with boxes to review, or null}
    GET  /api/s/<sid>/points/<i>    point cloud of frame i, raw float32 xyz
    POST /api/s/<sid>/action        {"name": "accept", "frame": 3, "index": 1} ... -> state + "result"
    POST /api/s/<sid>/heartbeat     keep this tab's lock on the session (the page sends one every 30 s)
    POST /api/s/<sid>/finish        release the session and mark it finished

Actions name their target explicitly (docs/adr/0001-explicit-target-actions.md); the server keeps
no current frame or selection, and loads the session document for every request (docs/adr/0011).
409: the session is still loading, was finished, was taken over by another tab, or was changed
by someone else since it was loaded (revision conflict: reload the frame and try again).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from export import FORMATS, ExportError, zip_files
from infra.sessions import RevisionConflict
from web.manager import SessionManager
from web.service import UnknownAction

STATIC = Path(__file__).parent / "static"


class Action(BaseModel):
    name: str
    frame: int | None = None
    index: int | None = None
    track_id: int | None = None
    label: str | None = None


class FolderRequest(BaseModel):
    path: str


def create_app(manager: SessionManager) -> FastAPI:
    app = FastAPI(title="LiDAR-AutoLabeler")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")   # vendored three.js and font

    def service(sid: str):
        try:
            return manager.service(sid)
        except KeyError:
            raise HTTPException(404, f"unknown session {sid}")
        except LookupError as exc:
            raise HTTPException(409, str(exc))

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def home():
        return FileResponse(STATIC / "home.html")

    @app.get("/annotate")
    def annotate():
        return FileResponse(STATIC / "annotate.html")

    # ------------------------------------------------------------- start screen
    @app.post("/api/folder/open")
    def open_folder(r: FolderRequest):
        try:
            return {"sid": manager.open_folder(r.path)}
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/jobs/{sid}")
    def job(sid: str):
        try:
            return manager.job(sid).view()
        except KeyError:
            raise HTTPException(404, f"unknown session {sid}")

    # -------------------------------------------------------------- one session
    @app.get("/api/s/{sid}/state")
    def state(sid: str, frame: int = 0):
        try:
            return service(sid).state(frame)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/s/{sid}/next_pending")
    def next_pending(sid: str, frame: int):
        try:
            return {"frame": service(sid).next_pending(frame)}
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/s/{sid}/points/{frame}")
    def points(sid: str, frame: int):
        try:
            return Response(manager.points(sid, frame), media_type="application/octet-stream")
        except KeyError:
            raise HTTPException(404, f"unknown session {sid}")
        except IndexError:
            raise HTTPException(404, f"no frame {frame}")
        except LookupError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/s/{sid}/heartbeat")
    def heartbeat(sid: str):
        try:
            manager.heartbeat(sid)
        except KeyError:
            raise HTTPException(404, f"unknown session {sid}")
        except LookupError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True, "lock_ttl": manager.store.lock_ttl}

    @app.post("/api/s/{sid}/action")
    def action(sid: str, a: Action):
        try:
            return service(sid).act(a.name, frame=a.frame, index=a.index, track_id=a.track_id, label=a.label)
        except UnknownAction:
            raise HTTPException(400, f"unknown action '{a.name}'")
        except RevisionConflict as exc:
            raise HTTPException(409, str(exc))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))

    def exported(sid: str, fmt: str, include_unreviewed: bool, include_empty: bool):
        if fmt not in FORMATS:
            raise HTTPException(400, f"unknown export format '{fmt}' (use one of: {', '.join(FORMATS)})")
        return service(sid).export(fmt, include_unreviewed, include_empty)

    @app.get("/api/s/{sid}/export/preview")
    def export_preview(sid: str, fmt: str = Query("lidar", alias="format"),
                       include_unreviewed: bool = False, include_empty: bool = False):
        try:
            result = exported(sid, fmt, include_unreviewed, include_empty)
        except ExportError as exc:
            return {"available": False, "reason": str(exc), "summary": None}
        return {"available": True, "reason": None, "summary": result.summary.to_dict()}

    @app.get("/api/s/{sid}/export")
    def export_download(sid: str, fmt: str = Query("lidar", alias="format"),
                        include_unreviewed: bool = False, include_empty: bool = False):
        try:
            result = exported(sid, fmt, include_unreviewed, include_empty)
        except ExportError as exc:
            raise HTTPException(400, str(exc)) from exc
        return Response(zip_files(result.files), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{result.archive_name}"'})

    @app.post("/api/s/{sid}/finish")
    def finish(sid: str):
        try:
            manager.finish(sid)
        except KeyError:
            raise HTTPException(404, f"unknown session {sid}")
        except LookupError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True}

    return app
