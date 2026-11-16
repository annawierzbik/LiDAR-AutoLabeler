"""Sequences of the API v1 (docs/adr/0018): the `sequences` table and the two ways to add one.

- Upload: a zip of .bin frames. The archive is checked before anything is written: size limits
  (compressed, by the API; unpacked, here), only plain file members with safe relative names (no
  absolute paths, drive letters, "..", backslashes, symlinks or encryption), every .bin file in one
  folder of the archive, no duplicate names. The frames are written into the uploads BlobStore
  under `seq-<random>/` while counting the bytes actually decompressed (a member that unpacks to
  more than its header says is stopped at the limit). Then the folder must pass
  ingestion.folder.validate_folder, as a folder opened on disk would; otherwise it is removed.
- Registration (admins only): a folder below LAL_DATA_ROOT, by its path relative to that root or
  absolute. The resolved path (symlinks followed) must be inside the resolved root.

A sequence's `spec` holds the session-builder arguments that open it ({"folder": <path>}); workers
open it from there (the uploads BlobStore is a LocalBlobStore, so the frames are a folder too).
"""
from __future__ import annotations

import stat
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from infra.blob import LocalBlobStore
from infra.db import SequenceRow

MAX_MEMBERS = 100_000
CHUNK = 1 << 20


class UploadRejected(ValueError):
    """The archive cannot be used; the message says why (shown to the user)."""


class TooLarge(UploadRejected):
    pass


def _safe_name(name: str) -> PurePosixPath:
    if "\\" in name or ":" in name or name.startswith("/") or "\x00" in name:
        raise UploadRejected(f"unsafe file name in the archive: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise UploadRejected(f"unsafe file name in the archive: {name!r}")
    return p


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def frame_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """The .bin members of a checked archive (all in one folder of it)."""
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS:
        raise UploadRejected(f"the archive has {len(infos)} entries (at most {MAX_MEMBERS})")
    frames, folders, names = [], set(), set()
    for info in infos:
        p = _safe_name(info.filename)
        if _is_symlink(info):
            raise UploadRejected(f"the archive contains a symbolic link: {info.filename!r}")
        if info.flag_bits & 0x1:
            raise UploadRejected("encrypted archives are not supported")
        if info.is_dir() or p.suffix.lower() != ".bin" or p.name.startswith("."):
            continue
        if p.parts and p.parts[0] == "__MACOSX":
            continue
        folders.add(str(p.parent))
        if p.name in names:
            raise UploadRejected(f"the frame name {p.name!r} appears twice in the archive")
        names.add(p.name)
        frames.append(info)
    if not frames:
        raise UploadRejected("the archive contains no .bin point cloud files")
    if len(folders) > 1:
        shown = ", ".join(sorted(folders)[:5])
        raise UploadRejected(f"the .bin files are in more than one folder of the archive ({shown}); "
                             "upload one sequence per archive")
    return frames


def extract_frames(archive: Path, blobs: LocalBlobStore, prefix: str, max_unpacked: int) -> int:
    """Write the frames of `archive` to `blobs` under `prefix`; returns the number of frames.
    Removes what was written when the archive is rejected."""
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile:
        raise UploadRejected("the file is not a zip archive") from None
    written: list[str] = []
    try:
        with zf:
            members = frame_members(zf)
            declared = sum(i.file_size for i in members)
            if declared > max_unpacked:
                raise TooLarge(f"the frames unpack to {declared} bytes (at most {max_unpacked})")
            total = 0
            for info in members:
                parts, size = [], 0
                with zf.open(info) as f:
                    while chunk := f.read(CHUNK):
                        size += len(chunk)
                        total += len(chunk)
                        if total > max_unpacked or size > info.file_size:
                            raise TooLarge(f"the frames unpack to more than {max_unpacked} bytes")
                        parts.append(chunk)
                key = f"{prefix}/{PurePosixPath(info.filename).name}"
                blobs.put(key, b"".join(parts))
                written.append(key)
    except BaseException:
        for key in written:
            blobs.delete(key)
        _remove_empty(blobs.root / prefix)
        raise
    return len(written)


def _remove_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass


def resolve_registered(path: str, root: Path | None) -> Path:
    """The folder `path` (relative to `root`, or absolute) if it lies inside `root`."""
    if root is None:
        raise PermissionError("registering folders is disabled: LAL_DATA_ROOT is not set")
    root = Path(root).resolve()
    raw = Path(path.strip().strip('"'))
    folder = (raw if raw.is_absolute() else root / raw).resolve()
    if folder != root and root not in folder.parents:
        raise PermissionError(f"{path} is outside the data root")
    return folder


# ----------------------------------------------------------------------- table
def _view(r) -> dict:
    return {"id": r.id, "name": r.name, "source": r.source, "num_frames": r.num_frames, "owner_id": r.owner_id,
            "created_at": r.created_at.isoformat()}


COLUMNS = (SequenceRow.id, SequenceRow.name, SequenceRow.source, SequenceRow.num_frames, SequenceRow.owner_id,
           SequenceRow.created_at, SequenceRow.spec, SequenceRow.location)


class SequenceStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def add(self, name: str, source: str, location: str, spec: dict, num_frames: int, owner_id: int) -> dict:
        with self.engine.begin() as c:
            row = c.execute(insert(SequenceRow).values(name=name, source=source, location=location, spec=spec,
                                                       num_frames=num_frames, owner_id=owner_id)
                            .returning(*COLUMNS)).first()
        return _view(row)

    def get(self, seq_id: int, owner_id: int | None) -> dict:
        """The sequence (with its spec); KeyError if it does not exist or is not the owner's
        (`owner_id` None: any owner, for admins)."""
        q = select(*COLUMNS).where(SequenceRow.id == seq_id)
        if owner_id is not None:
            q = q.where(SequenceRow.owner_id == owner_id)
        with self.engine.connect() as c:
            row = c.execute(q).first()
        if row is None:
            raise KeyError(seq_id)
        return {**_view(row), "spec": dict(row.spec)}

    def page(self, owner_id: int | None, limit: int, offset: int) -> tuple[list[dict], int]:
        cond = [] if owner_id is None else [SequenceRow.owner_id == owner_id]
        with self.engine.connect() as c:
            total = c.execute(select(func.count()).select_from(SequenceRow).where(*cond)).scalar()
            rows = c.execute(select(*COLUMNS).where(*cond).order_by(SequenceRow.id.desc())
                             .limit(limit).offset(offset)).all()
        return [_view(r) for r in rows], total


def new_upload_prefix() -> str:
    return f"seq-{uuid.uuid4().hex[:16]}"
