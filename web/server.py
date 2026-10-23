"""FastAPI app: start screen, annotation page and the JSON API behind them.

    GET  /                          start screen (open a folder)
    GET  /annotate?sid=<id>         the annotation page of one open session
    POST /api/folder/open           {"path": "..."}              -> {"sid"}
    GET  /api/jobs/<sid>            loading | ready  (404 once the job failed or was finished)
    GET  /api/s/<sid>/state         frame, selection, stats, visible boxes, save_error
    GET  /api/s/<sid>/points/<i>    point cloud of frame i, raw float32 xyz
    POST /api/s/<sid>/action        {"name": "accept"} / {"name": "goto", "frame": 12} ...
    POST /api/s/<sid>/finish        save and close the session (409 for other tabs afterwards)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from web.manager import SessionManager
from web.service import SessionClosed, UnknownAction

STATIC = Path(__file__).parent / "static"


class Action(BaseModel):
    name: str
    frame: Optional[int] = None
    index: Optional[int] = None
    label: Optional[str] = None


class FolderRequest(BaseModel):
    path: str


def create_app(manager: SessionManager) -> FastAPI:
    app = FastAPI(title="LiDAR-AutoLabeler")

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
    def state(sid: str):
        return service(sid).state()

    @app.get("/api/s/{sid}/points/{frame}")
    def points(sid: str, frame: int):
        try:
            return Response(service(sid).points(frame), media_type="application/octet-stream")
        except IndexError:
            raise HTTPException(404, f"no frame {frame}")

    @app.post("/api/s/{sid}/action")
    def action(sid: str, a: Action):
        args = {k: v for k, v in (("frame", a.frame), ("index", a.index), ("label", a.label)) if v is not None}
        if a.name == "select" and a.index is None:
            args["index"] = None
        try:
            return service(sid).act(a.name, **args)
        except UnknownAction:
            raise HTTPException(400, f"unknown action '{a.name}'")
        except SessionClosed as exc:
            raise HTTPException(409, str(exc))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/s/{sid}/finish")
    def finish(sid: str):
        service(sid)
        manager.finish(sid)
        return {"ok": True}

    return app
