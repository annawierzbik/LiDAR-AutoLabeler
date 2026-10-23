"""Open an arbitrary folder of KITTI-style .bin point clouds (x, y, z, intensity as float32).

SequenceLoader expects the full KITTI tree (<root>/<split>/velodyne/<sequence>) and numeric file
names. FolderSequence takes any folder: calibration and ground truth are simply absent, and files
are ordered the way a human would sort them (frame2 before frame10).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List

from .frame import Frame
from .loader import load_point_cloud
from .sequence import SequenceLoader


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def frame_files(folder) -> List[Path]:
    return sorted(Path(folder).glob("*.bin"), key=_natural_key)


def validate_folder(folder) -> int:
    """Check that `folder` can be opened as a sequence. Returns the number of frames, or raises
    an error whose message can be shown to the user as it is."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    files = frame_files(folder)
    if not files:
        subfolders = sorted(d.name for d in folder.iterdir() if d.is_dir() and any(d.glob("*.bin")))
        if subfolders:
            shown = ", ".join(subfolders[:5]) + (" ..." if len(subfolders) > 5 else "")
            raise ValueError(f"No .bin files directly in this folder, but its subfolders have some "
                             f"({shown}). Choose one of them.")
        raise FileNotFoundError(f"No .bin point cloud files in {folder}")
    points = load_point_cloud(files[0])  # raises ValueError for files that are not x, y, z, intensity
    if len(points) == 0:
        raise ValueError(f"{files[0].name} contains no points")
    return len(files)


def folder_key(folder) -> str:
    """Short unique name for a folder: used for cache and session file names, so two different
    folders that are both called "0000" never overwrite each other's results."""
    folder = Path(folder).resolve()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", folder.name) or "sequence"
    return f"{safe}_{hashlib.sha1(str(folder).encode()).hexdigest()[:6]}"


class FolderSequence(SequenceLoader):
    def __init__(self, folder) -> None:
        folder = Path(folder)
        validate_folder(folder)
        self.data_root, self.split, self.sequence = folder.parent, "", folder.name
        self.velodyne_dir, self.calib_path, self.label_path = folder, None, None
        self._frame_files = frame_files(folder)
        self._calib, self._labels_by_frame = {}, {}

    def _load_frame(self, index):
        path = self._frame_files[index]
        frame_id = int(path.stem) if path.stem.isdigit() else index
        return Frame(sequence_id=self.sequence, frame_id=frame_id, points=load_point_cloud(path),
                     calib={}, labels_gt=[])