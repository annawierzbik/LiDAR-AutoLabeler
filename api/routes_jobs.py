"""Jobs (state from the `jobs` table) and the download of a stored export; health checks."""
from __future__ import annotations

import time

from fastapi import APIRouter
from fastapi.responses import Response

from api.deps import CurrentUser, Page, Services, Svc, owner_filter
from api.errors import ApiError, not_found, responses
from infra.auth import Principal

router = APIRouter(prefix="/api/v1", tags=["jobs"])
health = APIRouter(tags=["health"])


def _job(svc: Services, job_id: str, user: Principal) -> dict:
    """The job if it is the user's (admins: anyone's), else 404. Reading it also notices a lost
    worker (the job is requeued or failed, PgJobStore.recover_stale)."""
    try:
        svc.jobs.view(job_id, owner_filter(user))
    except KeyError:
        raise not_found("job", job_id) from None
    svc.queue.status(job_id)
    return svc.jobs.view(job_id)


@router.get("/jobs", responses=responses())
def list_jobs(page: Page, user: CurrentUser,
              svc: Svc):
    """The user's jobs (admins: everyone's), newest first."""
    return page.wrap(*svc.jobs.page(owner_filter(user), page.limit, page.offset))


@router.get("/jobs/{job_id}", responses=responses(404))
def get_job(job_id: str, user: CurrentUser, svc: Svc):
    """State (queued, running, cancelling, done, failed, cancelled), progress 0..1, message, result."""
    return _job(svc, job_id, user)


@router.post("/jobs/{job_id}/cancel", responses=responses(404))
def cancel_job(job_id: str, user: CurrentUser, svc: Svc):
    _job(svc, job_id, user)
    return {"cancelled": svc.queue.cancel(job_id)}


@router.get("/jobs/{job_id}/download", response_class=Response, responses={
    **responses(404, 409, 410), 200: {"content": {"application/zip": {}}, "description": "the export zip"}})
def download(job_id: str, user: CurrentUser, svc: Svc):
    """The zip of a finished export job, while it has not expired."""
    from infra.tasks import export_blobs

    job = _job(svc, job_id, user)
    if job["kind"] != "export":
        raise ApiError(409, f"job {job_id} is not an export")
    if job["state"] != "done":
        raise ApiError(409, f"the export is not finished ({job['state']})", code="not_ready")
    result = job["result"]
    if result["expires_at"] <= time.time():
        raise ApiError(410, "the export has expired; export again")
    try:
        data = export_blobs().get(result["blob"])
    except KeyError:
        raise ApiError(410, "the export has expired; export again") from None
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{result["archive_name"]}"'})


# ---------------------------------------------------------------------- health
@health.get("/healthz")
def healthz():
    """The process is alive (no dependencies checked)."""
    return {"status": "ok"}


@health.get("/readyz", responses={503: {"description": "a dependency is unreachable (see checks)"}})
def readyz(svc: Svc):
    """Database, Redis and the blob stores are reachable."""
    import os

    from sqlalchemy import text

    from pipeline.session_builder import RESULTS_DIR

    def database():
        with svc.engine.connect() as c:
            c.execute(text("SELECT 1"))

    def writable(folder):
        def check():
            folder.mkdir(parents=True, exist_ok=True)
            if not os.access(folder, os.W_OK):
                raise OSError(f"{folder} is not writable")
        return check

    checks = {}
    for name, fn in (("database", database), ("redis", svc.ping_redis), ("results_blobs", writable(RESULTS_DIR)),
                     ("upload_blobs", writable(svc.settings.uploads_dir))):
        try:
            fn()
            checks[name] = {"ok": True}
        except Exception as exc:  # noqa: BLE001  (any failure means not ready; say which)
            checks[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    ok = all(c["ok"] for c in checks.values())
    from fastapi.responses import JSONResponse

    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)
