"""Which KITTI tracking frames were seen by a detector trained on the KITTI object train split.

Two mappings are combined:

* object -> raw: devkit_object/mapping. Line i (0-based) of train_rand.txt holds the 1-based line
  of train_mapping.txt for object image i; that line is "<date> <drive> <raw frame>".
* tracking -> object: by content. A tracking frame and an object frame taken from the same raw
  recording are byte-identical .bin files, so they are matched by (file size, md5). Frames that
  are not byte-identical are simply not matched; nothing is matched approximately.

A tracking sequence is placed in a raw drive through its matched frames; the frame offset
(raw frame - tracking frame) must be the same for all of them, otherwise the report says so.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path


def read_object_mapping(devkit_dir: Path) -> list[tuple[str, int]]:
    """[(raw drive, raw frame)] indexed by object training image id (0-based)."""
    mapping_dir = Path(devkit_dir) / "mapping"
    rand = [int(v) for v in (mapping_dir / "train_rand.txt").read_text().replace("\n", "").split(",") if v.strip()]
    lines = [ln.split() for ln in (mapping_dir / "train_mapping.txt").read_text().splitlines() if ln.strip()]
    if len(rand) != len(lines):
        raise ValueError(f"train_rand has {len(rand)} entries but train_mapping has {len(lines)} lines")
    return [(lines[r - 1][1], int(lines[r - 1][2])) for r in rand]


def read_image_sets(image_sets_dir: Path, names=("train", "val")) -> dict[str, set[int]]:
    return {n: {int(v) for v in (Path(image_sets_dir) / f"{n}.txt").read_text().split()} for n in names}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class ContentIndex:
    """Finds the object frame with exactly the same bytes as a given file.
    Files are bucketed by size first, so only same-size candidates are ever hashed."""

    def __init__(self, velodyne_dir: Path) -> None:
        self.by_size: dict[int, list[Path]] = defaultdict(list)
        for p in sorted(Path(velodyne_dir).glob("*.bin")):
            self.by_size[p.stat().st_size].append(p)
        self._hash: dict[Path, str] = {}
        self.hashed = 0   # number of md5 computations, for the report

    def _md5(self, p: Path) -> str:
        if p not in self._hash:
            self._hash[p] = md5(p)
            self.hashed += 1
        return self._hash[p]

    def find(self, path: Path) -> list[int]:
        """Object ids of all byte-identical files (normally zero or one)."""
        candidates = self.by_size.get(path.stat().st_size, [])
        if not candidates:
            return []
        target = md5(path)
        self.hashed += 1
        return [int(c.stem) for c in candidates if self._md5(c) == target]


def match_sequence(seq_dir: Path, index: ContentIndex) -> dict[int, list[int]]:
    """{tracking frame: [object ids]} for the frames of one tracking sequence that have a twin."""
    out = {}
    for p in sorted(Path(seq_dir).glob("*.bin")):
        ids = index.find(p)
        if ids:
            out[int(p.stem)] = ids
    return out


def sequence_report(seq: str, n_frames: int, matches: dict[int, list[int]],
                    mapping: list[tuple[str, int]], image_sets: dict[str, set[int]]) -> dict:
    """Overlap summary of one tracking sequence (see module docstring)."""
    pairs = [(f, o) for f, ids in sorted(matches.items()) for o in ids]
    drives = sorted({mapping[o][0] for _, o in pairs})
    offsets = sorted({mapping[o][1] - f for f, o in pairs})
    split_of = {o: name for name, ids in image_sets.items() for o in ids}

    frames_by_split: dict[str, list[int]] = defaultdict(list)
    for f, o in pairs:
        frames_by_split[split_of.get(o, "none")].append(f)

    # object frames of the same raw drive that are NOT inside this tracking sequence
    # (temporal neighbours: a different frame of the same scene, recorded seconds apart)
    drive_outside: dict[str, list[int]] = defaultdict(list)
    if len(drives) == 1 and len(offsets) == 1:
        drive, off = drives[0], offsets[0]
        first, last = off, off + n_frames - 1
        for o, (d, raw) in enumerate(mapping):
            if d == drive and not first <= raw <= last:
                drive_outside[split_of.get(o, "none")].append(raw)

    return {
        "sequence": seq,
        "num_frames": n_frames,
        "num_matched_frames": len(matches),
        "raw_drives": drives,
        "frame_offsets": offsets,
        "consistent_mapping": len(drives) == 1 and len(offsets) == 1,
        "matched": [{"tracking_frame": f, "object_id": f"{o:06d}", "object_split": split_of.get(o, "none"),
                     "raw_frame": mapping[o][1]} for f, o in pairs],
        "num_frames_in_object_split": {k: len(v) for k, v in sorted(frames_by_split.items())},
        "same_drive_object_frames_outside_sequence": {k: len(v) for k, v in sorted(drive_outside.items())},
    }


def is_contaminated(report: dict, rule: str = "frame", train_split: str = "train") -> bool:
    """rule "frame": a frame of the sequence is in the detector's training split.
    rule "drive": additionally, any training frame comes from the same raw drive.
    A sequence with an inconsistent mapping is treated as contaminated (cannot be cleared)."""
    if not report["consistent_mapping"] and report["num_matched_frames"]:
        return True
    in_seq = report["num_frames_in_object_split"].get(train_split, 0) > 0
    if rule == "frame":
        return in_seq
    if rule == "drive":
        return in_seq or report["same_drive_object_frames_outside_sequence"].get(train_split, 0) > 0
    raise ValueError(f"unknown contamination rule '{rule}' (use 'frame' or 'drive')")


def drive_split_overlap(mapping: list[tuple[str, int]], image_sets: dict[str, set[int]]) -> dict:
    """Do the object train and val splits share raw drives? (They are meant not to.)"""
    drives = {name: {mapping[o][0] for o in ids} for name, ids in image_sets.items()}
    shared = sorted(drives.get("train", set()) & drives.get("val", set()))
    return {"num_drives": {k: len(v) for k, v in drives.items()}, "shared_drives": shared}
