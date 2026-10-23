"""Export the final boxes of an annotation session: glue between a session, its point cloud
source (for frame names and calibration) and the formats in export.formats."""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from export.formats import (
    ExportError,
    ExportSummary,
    render_kitti_tracking,
    render_lidar,
)
from ingestion.calibration import Calibration
from ingestion.folder import frame_files

FORMATS = ("lidar", "kitti")


@dataclass
class ExportResult:
    files: dict[str, str]
    summary: ExportSummary
    archive_name: str


def sequence_info(session, loader) -> tuple[list[str], Calibration | None]:
    """File names (without extension) of the session's frames and the calibration, if there is one.

    Without a readable velodyne folder the names are the frame numbers, zero padded to six digits."""
    offset, n = session.frame_offset, session.num_frames
    stems = [f"{offset + i:06d}" for i in range(n)]
    velodyne = getattr(loader, "velodyne_dir", None)
    if velodyne is not None and Path(velodyne).is_dir():
        files = frame_files(velodyne)
        if len(files) >= offset + n:
            stems = [p.stem for p in files[offset: offset + n]]
    calib = None
    calib_path = getattr(loader, "calib_path", None)
    if calib_path is not None and Path(calib_path).is_file():
        try:
            calib = Calibration.from_file(calib_path)
        except ValueError:
            calib = None
    return stems, calib


def export_session(session, loader, fmt: str, include_unreviewed: bool = False,
                   include_empty: bool = False) -> ExportResult:
    """Render the labels of `session`. Raises ExportError when the format cannot be produced."""
    if fmt not in FORMATS:
        raise ExportError(f"unknown export format '{fmt}' (use one of: {', '.join(FORMATS)})")
    stems, calib = sequence_info(session, loader)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session.sequence) or "sequence"
    if fmt == "lidar":
        files, summary = render_lidar(session, stems, include_unreviewed, include_empty)
    else:
        if calib is None:
            raise ExportError("KITTI tracking labels need the calibration file of a KITTI sequence, "
                              "and this sequence has none. Use the LiDAR format instead.")
        if not all(s.isdigit() for s in stems):
            raise ExportError("KITTI tracking labels need numeric frame names (000000.bin, 000001.bin, ...).")
        files, summary = render_kitti_tracking(session, [int(s) for s in stems], calib, safe, include_unreviewed)
    return ExportResult(files, summary, f"{safe}_{fmt}_labels.zip")


def write_files(files: dict[str, str], out_dir) -> list[Path]:
    """Write the rendered files under `out_dir` (created if needed). Returns the written paths."""
    out_dir = Path(out_dir).resolve()
    written = []
    for rel, text in files.items():
        target = (out_dir / rel).resolve()
        if out_dir not in target.parents:
            raise ExportError(f"refusing to write outside {out_dir}: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        written.append(target)
    return written


def zip_files(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, text in files.items():
            z.writestr(rel, text)
    return buf.getvalue()
