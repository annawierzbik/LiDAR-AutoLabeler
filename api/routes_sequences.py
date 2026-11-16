"""Sequences: zip upload (any user) and folder registration below LAL_DATA_ROOT (admins), docs/adr/0018."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from api.deps import AdminUser, CurrentUser, Page, Svc, owner_filter
from api.errors import ApiError, not_found, responses
from infra.blob import LocalBlobStore
from infra.sequence_store import (
    TooLarge,
    UploadRejected,
    extract_frames,
    new_upload_prefix,
    resolve_registered,
)

router = APIRouter(prefix="/api/v1", tags=["sequences"])


class Register(BaseModel):
    path: str = Field(description="folder below LAL_DATA_ROOT: relative to it, or absolute")
    name: str | None = Field(None, max_length=200)
    owner_id: int | None = Field(None, description="user the sequence belongs to (default: the admin)")


class SequenceView(BaseModel):
    id: int
    name: str
    source: str
    num_frames: int
    owner_id: int | None
    created_at: str


def _validate(folder: Path) -> int:
    from ingestion.folder import validate_folder

    try:
        return validate_folder(folder)
    except (OSError, ValueError) as exc:
        raise ApiError(400, str(exc)) from None


@router.get("/sequences", responses=responses())
def list_sequences(page: Page, user: CurrentUser,
                   svc: Svc):
    """The user's sequences (admins: everyone's), newest first."""
    return page.wrap(*svc.sequences.page(owner_filter(user), page.limit, page.offset))


@router.get("/sequences/{seq_id}", response_model=SequenceView, responses=responses(404))
def get_sequence(seq_id: int, user: CurrentUser, svc: Svc):
    try:
        seq = svc.sequences.get(seq_id, owner_filter(user))
    except KeyError:
        raise not_found("sequence", seq_id) from None
    seq.pop("spec")
    return seq


@router.post("/sequences/upload", status_code=201, response_model=SequenceView, responses=responses(400, 403, 413),
             openapi_extra={"requestBody": {"required": True, "content": {"application/zip": {
                 "schema": {"type": "string", "format": "binary"}}}}})
async def upload_sequence(request: Request, name: Annotated[str, Query(min_length=1, max_length=200)],
                          user: CurrentUser, svc: Svc):
    """Upload a zip of .bin frames (the request body, Content-Type application/zip). Limits:
    LAL_UPLOAD_MAX_BYTES for the archive, LAL_UPLOAD_MAX_UNPACKED_BYTES unpacked. The frames must
    be in one folder of the archive; names are checked (no paths outside it, no links)."""
    s = svc.settings
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > s.upload_max_bytes:
        raise ApiError(413, f"the archive is larger than {s.upload_max_bytes} bytes")
    uploads = s.uploads_dir
    incoming = uploads / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=incoming)
    prefix = new_upload_prefix()
    blobs = LocalBlobStore(uploads)
    try:
        size = 0
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > s.upload_max_bytes:
                    raise ApiError(413, f"the archive is larger than {s.upload_max_bytes} bytes")
                f.write(chunk)
        try:
            extract_frames(Path(tmp), blobs, prefix, s.upload_max_unpacked_bytes)
        except TooLarge as exc:
            raise ApiError(413, str(exc)) from None
        except UploadRejected as exc:
            raise ApiError(400, str(exc)) from None
        folder = uploads / prefix
        try:
            frames = _validate(folder)
        except ApiError:
            for key in blobs.list(prefix + "/"):
                blobs.delete(key)
            folder.rmdir()
            raise
    finally:
        Path(tmp).unlink(missing_ok=True)
    return svc.sequences.add(name.strip(), "upload", prefix, {"folder": str(folder)}, frames, user.id)


@router.post("/sequences/register", status_code=201, response_model=SequenceView, responses=responses(400, 403, 404))
def register_sequence(body: Register, admin: AdminUser, svc: Svc):
    """Register a folder of .bin frames below LAL_DATA_ROOT (admins only). Paths outside the root,
    also through symbolic links, are refused."""
    try:
        folder = resolve_registered(body.path, svc.settings.data_root)
    except PermissionError as exc:
        raise ApiError(403, str(exc), code="outside_data_root") from None
    frames = _validate(folder)
    owner = body.owner_id if body.owner_id is not None else admin.id
    if svc.users.get(owner) is None:
        raise not_found("user", owner)
    return svc.sequences.add(body.name or folder.name, "folder", str(folder), {"folder": str(folder)}, frames, owner)
