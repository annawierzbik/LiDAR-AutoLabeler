"""State of one annotation/verification session (no GUI code in here).

Two modes share the same data model:
    assisted  every box found by detector + tracker starts as PENDING; the human
              accepts, rejects or edits it
    manual    nothing is pre-filled; the human draws every box (no detector needed)

Every change goes through a transaction, so undo/redo and the timestamped event log
(used to measure annotation time) work for all operations.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from common.boxes import BoundingBox3D
from verification.interpolation import MAX_GAP, fill_plan

SESSION_VERSION = 1
EDITABLE_FIELDS = ("x", "y", "z", "dx", "dy", "dz", "heading", "label")


class Status(str, Enum):
    PENDING = "pending"    # produced by the system, not looked at yet
    ACCEPTED = "accepted"  # system box confirmed unchanged
    REJECTED = "rejected"  # system box thrown away
    EDITED = "edited"      # system box corrected by the human
    MANUAL = "manual"      # box created by the human
    INTERPOLATED = "interpolated"  # filled in between two key boxes of a track, waiting for a decision


INCLUDED = (Status.ACCEPTED, Status.EDITED, Status.MANUAL)  # these end up in the exported labels
NEEDS_REVIEW = (Status.PENDING, Status.INTERPOLATED)         # the human still has to look at these
KEY_STATUSES = (Status.PENDING, Status.ACCEPTED, Status.EDITED, Status.MANUAL)  # interpolation anchors


@dataclass(frozen=True)
class Annotation:
    box: BoundingBox3D
    status: Status = Status.PENDING

    def with_status(self, status: Status) -> "Annotation":
        return replace(self, status=status)

    def to_dict(self) -> dict:
        return {"box": asdict(self.box), "status": self.status.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(BoundingBox3D(**d["box"]), Status(d["status"]))


@dataclass
class Event:
    t: float
    action: str
    frame: Optional[int] = None
    detail: dict = field(default_factory=dict)


class AnnotationSession:
    def __init__(
        self,
        sequence: str,
        num_frames: int,
        mode: str = "assisted",
        frame_offset: int = 0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if mode not in ("assisted", "manual"):
            raise ValueError("mode must be 'assisted' or 'manual'")
        self.sequence = sequence
        self.num_frames = num_frames
        self.mode = mode
        self.frame_offset = frame_offset  # session frame i == sequence frame i + offset
        self._clock = clock
        self.frames: Dict[int, List[Annotation]] = {i: [] for i in range(num_frames)}
        self.events: List[Event] = []
        self._undo: List[Tuple[str, Dict[int, List[Annotation]]]] = []
        self._redo: List[Tuple[str, Dict[int, List[Annotation]]]] = []
        self._next_track_id = 0
        self.max_gap = MAX_GAP  # longest hole in a track that gets filled automatically
        self.revision = 0  # bumped by every change of the annotations (edit, undo, ...) -> autosave trigger

    @classmethod
    def from_tracking(
        cls,
        sequence: str,
        results: Sequence[Sequence[BoundingBox3D]],
        mode: str = "assisted",
        frame_offset: int = 0,
        clock: Callable[[], float] = time.time,
    ) -> "AnnotationSession":
        """`results[i]` = tracked boxes (with track_id) of session frame i. In manual mode
        they are ignored: the session starts empty."""
        s = cls(sequence, len(results), mode, frame_offset, clock)
        if mode == "assisted":
            for i, boxes in enumerate(results):
                s.frames[i] = [Annotation(b) for b in boxes]
                for b in boxes:
                    if b.track_id is not None:
                        s._next_track_id = max(s._next_track_id, b.track_id + 1)
            for tid in s._track_ids():
                s._fill_track(tid)   # initial state, not an undoable user action
        s.log("session_start", mode=mode)
        return s

    # ------------------------------------------------------------------ reading
    def annotations(self, frame: int) -> List[Annotation]:
        return list(self.frames[self._check(frame)])

    def final_boxes(self, frame: int, include_pending: bool = False) -> List[BoundingBox3D]:
        keep = INCLUDED + ((Status.PENDING,) if include_pending else ())
        return [a.box for a in self.frames[self._check(frame)] if a.status in keep]

    def stats(self) -> dict:
        counts = {s.value: 0 for s in Status}
        for anns in self.frames.values():
            for a in anns:
                counts[a.status.value] += 1
        total = sum(counts.values())
        reviewed = total - counts["pending"] - counts["interpolated"]
        return {**counts, "total": total, "reviewed": reviewed,
                "reviewed_fraction": reviewed / total if total else 1.0}

    def frame_has_pending(self, frame: int) -> bool:
        return any(a.status in NEEDS_REVIEW for a in self.frames[self._check(frame)])

    def next_pending_frame(self, frame: int) -> Optional[int]:
        """First frame after `frame` (wrapping around) that still has pending boxes."""
        order = list(range(frame + 1, self.num_frames)) + list(range(0, frame + 1))
        return next((f for f in order if self.frame_has_pending(f)), None)

    def new_track_id(self) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    # ------------------------------------------------------------ transactions
    def _check(self, frame: int) -> int:
        if not 0 <= frame < self.num_frames:
            raise IndexError(f"frame {frame} out of range 0..{self.num_frames - 1}")
        return frame

    def _item(self, frame: int, idx: int) -> Annotation:
        anns = self.frames[self._check(frame)]
        if not 0 <= idx < len(anns):
            raise IndexError(f"no annotation {idx} in frame {frame}")
        return anns[idx]

    @contextlib.contextmanager
    def _transaction(self, label: str, frames: Iterable[int], refill_track: Optional[int] = None,
                     **detail) -> Iterator[None]:
        """Undo/redo + event-log wrapper. With `refill_track` the interpolated boxes of that track
        are recomputed after the change, inside the same undo step."""
        frames = sorted(set(frames)) if refill_track is None else list(range(self.num_frames))
        before = {f: list(self.frames[f]) for f in frames}
        yield
        if refill_track is not None:
            self._fill_track(refill_track)
        changed = [f for f in frames if self.frames[f] != before[f]]
        if changed:
            self._undo.append((label, {f: before[f] for f in changed}))
            self._redo.clear()
            self.revision += 1
            self.log(label, frame=changed[0] if len(changed) == 1 else None, **detail)

    # ---------------------------------------------------------- gap filling
    def _track_ids(self) -> List[int]:
        return sorted({a.box.track_id for anns in self.frames.values() for a in anns
                       if a.box.track_id is not None})

    def _fill_track(self, track_id: int) -> None:
        """Recompute the INTERPOLATED boxes of one track (in place, not undoable by itself)."""
        keys: Dict[int, BoundingBox3D] = {}
        blocked = set()  # frames where the track already has a real (or rejected) box
        for f, anns in self.frames.items():
            for a in anns:
                if a.box.track_id != track_id or a.status is Status.INTERPOLATED:
                    continue
                blocked.add(f)
                if a.status in KEY_STATUSES:
                    keys.setdefault(f, a.box)
        plan = fill_plan(keys, self.max_gap)
        for f, anns in self.frames.items():
            new = [a for a in anns if not (a.box.track_id == track_id and a.status is Status.INTERPOLATED)]
            if f in plan and f not in blocked:
                new.append(Annotation(plan[f], Status.INTERPOLATED))
            if new != anns:
                self.frames[f] = new

    def fill_gaps(self) -> int:
        """Fill every hole (up to `max_gap` frames) of every track. Returns how many boxes were added."""
        before = sum(a.status is Status.INTERPOLATED for anns in self.frames.values() for a in anns)
        with self._transaction("fill_gaps", range(self.num_frames)):
            for tid in self._track_ids():
                self._fill_track(tid)
        return sum(a.status is Status.INTERPOLATED for anns in self.frames.values() for a in anns) - before

    # --------------------------------------------------------------- operations
    def accept(self, frame: int, idx: int) -> None:
        a = self._item(frame, idx)
        with self._transaction("accept", [frame], index=idx, track_id=a.box.track_id):
            if a.status in (Status.PENDING, Status.INTERPOLATED, Status.REJECTED):
                self.frames[frame][idx] = a.with_status(Status.ACCEPTED)

    def reject(self, frame: int, idx: int) -> None:
        a = self._item(frame, idx)
        with self._transaction("reject", [frame], refill_track=a.box.track_id,
                               index=idx, track_id=a.box.track_id):
            if a.status is Status.MANUAL:
                del self.frames[frame][idx]            # a rejected human box is simply gone
            else:
                self.frames[frame][idx] = a.with_status(Status.REJECTED)

    def _set_track(self, track_id: int, status: Status, from_frame: int, label: str) -> int:
        targets = [(f, i) for f in range(from_frame, self.num_frames)
                   for i, a in enumerate(self.frames[f])
                   if a.box.track_id == track_id and a.status in NEEDS_REVIEW]
        with self._transaction(label, {f for f, _ in targets}, track_id=track_id,
                               from_frame=from_frame, count=len(targets)):
            for f, i in targets:
                self.frames[f][i] = self.frames[f][i].with_status(status)
        return len(targets)

    def accept_track(self, track_id: int, from_frame: int = 0) -> int:
        """Accept every still-pending box of this track from `from_frame` on. Returns how many."""
        return self._set_track(track_id, Status.ACCEPTED, from_frame, "accept_track")

    def reject_track(self, track_id: int, from_frame: int = 0) -> int:
        return self._set_track(track_id, Status.REJECTED, from_frame, "reject_track")

    def edit(self, frame: int, idx: int, **fields) -> None:
        """Change box fields (x, y, z, dx, dy, dz, heading, label). Raises ValueError if the
        result is not a valid box; the session is left untouched in that case."""
        unknown = set(fields) - set(EDITABLE_FIELDS)
        if unknown:
            raise ValueError(f"cannot edit {sorted(unknown)}")
        a = self._item(frame, idx)
        new_box = replace(a.box, **fields)  # validates in BoundingBox3D.__post_init__
        status = Status.MANUAL if a.status is Status.MANUAL else Status.EDITED
        with self._transaction("edit", [frame], refill_track=a.box.track_id, index=idx,
                               track_id=a.box.track_id, fields=sorted(fields)):
            self.frames[frame][idx] = Annotation(new_box, status)

    def add_manual(self, frame: int, box: BoundingBox3D, track_id: Optional[int] = None) -> int:
        """Add a human-drawn box. Gets a fresh track id unless one is given. Returns its index."""
        tid = track_id if track_id is not None else (
            box.track_id if box.track_id is not None else self.new_track_id())
        self._next_track_id = max(self._next_track_id, tid + 1)
        with self._transaction("add", [self._check(frame)], refill_track=tid, track_id=tid,
                               label_name=box.label):
            self.frames[frame].append(Annotation(box.with_track_id(tid), Status.MANUAL))
        return len(self.frames[frame]) - 1

    def copy_to_frame(self, frame: int, idx: int, target: int) -> int:
        """Copy a box (same track id) into another frame as a human box; returns the new index.
        Speeds up manual annotation of objects that barely move between frames."""
        a = self._item(frame, idx)
        tid = a.box.track_id
        if tid is not None and any(x.box.track_id == tid and x.status is not Status.REJECTED
                                   for x in self.frames[self._check(target)]):
            raise ValueError(f"track {tid} already has a box in frame {target}")
        return self.add_manual(target, a.box, track_id=tid)

    # -------------------------------------------------------------- undo / redo
    def _swap(self, source: list, destination: list, label: str) -> Optional[str]:
        if not source:
            return None
        name, saved = source.pop()
        current = {f: list(self.frames[f]) for f in saved}
        for f, anns in saved.items():
            self.frames[f] = list(anns)
        destination.append((name, current))
        self.revision += 1
        self.log(label, frame=min(saved) if len(saved) == 1 else None, undone=name)
        return name

    def undo(self) -> Optional[str]:
        return self._swap(self._undo, self._redo, "undo")

    def redo(self) -> Optional[str]:
        return self._swap(self._redo, self._undo, "redo")

    # ------------------------------------------------------------------ logging
    def log(self, action: str, frame: Optional[int] = None, **detail) -> None:
        self.events.append(Event(self._clock(), action, frame, detail))

    def active_seconds(self, max_gap: float = 30.0) -> float:
        """Time spent working: sum of gaps between consecutive events, ignoring gaps longer than
        `max_gap` seconds (breaks, coffee)."""
        ts = [e.t for e in self.events]
        return sum(b - a for a, b in zip(ts, ts[1:]) if b - a <= max_gap)

    # --------------------------------------------------------------------- I/O
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": SESSION_VERSION,
            "sequence": self.sequence, "mode": self.mode,
            "frame_offset": self.frame_offset, "num_frames": self.num_frames,
            "next_track_id": self._next_track_id,
            "frames": {str(f): [a.to_dict() for a in anns] for f, anns in self.frames.items() if anns},
            "events": [asdict(e) for e in self.events],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)  # atomic: a crash while saving cannot corrupt the previous file

    @classmethod
    def load(cls, path, clock: Callable[[], float] = time.time) -> "AnnotationSession":
        data = json.loads(Path(path).read_text())
        if data.get("version") != SESSION_VERSION:
            raise ValueError(f"unsupported session version {data.get('version')}")
        # files written before the "participant" field was dropped still have it; it is ignored
        s = cls(data["sequence"], data["num_frames"], data["mode"], data["frame_offset"], clock)
        s._next_track_id = data["next_track_id"]
        for f, anns in data["frames"].items():
            s.frames[int(f)] = [Annotation.from_dict(a) for a in anns]
        s.events = [Event(**e) for e in data["events"]]
        s.log("session_resume")
        return s
