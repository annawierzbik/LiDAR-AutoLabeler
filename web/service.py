"""Web-facing wrapper around one annotation session (no HTTP and no drawing in here).

Stateless actions (docs/adr/0001-explicit-target-actions.md): every action names its target
(frame, and box index or track id where needed). The view state (current frame, selected box,
class for new boxes, whether rejected boxes are shown) lives in the browser; nothing here depends
on a server-side current frame or selection.

`ReviewService` exposes the annotations of a frame as plain JSON-able dicts, the point cloud of a
frame as raw float32 bytes, `next_pending(frame)`, and `act(name, frame, ...)` which performs one
named action and autosaves through the SessionStore when the annotations changed.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from export import ExportResult, export_session
from infra.points import PointSource
from infra.sessions import RevisionConflict, SessionStore
from infra.settings import get_settings
from verification.controller import CLASSES, ReviewController
from verification.edit_steps import (
    MOVE_STEP,
    ROT_STEP,
    SIZE_STEP,
    Z_STEP,
)
from verification.session import AnnotationSession

NUDGES = {
    "forward": {"dx": MOVE_STEP}, "back": {"dx": -MOVE_STEP},
    "left": {"dy": MOVE_STEP}, "right": {"dy": -MOVE_STEP},
    "up": {"dz": Z_STEP}, "down": {"dz": -Z_STEP},
    "rotate_left": {"dheading": ROT_STEP}, "rotate_right": {"dheading": -ROT_STEP},
}
RESIZES = {
    "length_minus": {"dlength": -SIZE_STEP}, "length_plus": {"dlength": SIZE_STEP},
    "width_minus": {"dwidth": -SIZE_STEP}, "width_plus": {"dwidth": SIZE_STEP},
    "height_minus": {"dheight": -SIZE_STEP}, "height_plus": {"dheight": SIZE_STEP},
}
BOX_ACTIONS = ("accept", "reject", "copy_next", "set_label", *NUDGES, *RESIZES)   # need frame + index
TRACK_ACTIONS = ("accept_track", "reject_track")                                   # need track_id
ACTIONS = ("view", "add", "undo", "redo", *BOX_ACTIONS, *TRACK_ACTIONS)


class UnknownAction(KeyError):
    """`act` was called with a name that is not an action."""


class SessionClosed(LookupError):
    """The session was finished; it no longer accepts actions."""


class ReviewService:
    def __init__(self, session: AnnotationSession, store: SessionStore, key: str, points: PointSource,
                 sequence=None, cache_frames: int | None = None) -> None:
        """`sequence` (a SequenceLoader / FolderSequence) is only used by `export` for frame
        names and calibration."""
        self.session = session
        self.store, self.key = store, key
        self.points_source = points
        self.sequence = sequence
        self._lock = threading.RLock()   # re-entrant: act() calls state() while holding it
        self._saved_revision = session.revision
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._cache_frames = cache_frames or get_settings().point_cache_frames
        self.save_error: str | None = None  # last autosave failure, shown in the browser
        self.closed = False

    # ------------------------------------------------------------------ queries
    def _frame(self, frame) -> int:
        if frame is None:
            raise ValueError("this action needs the field 'frame'")
        frame = int(frame)
        if not 0 <= frame < self.session.num_frames:
            raise ValueError(f"frame {frame} out of range 0..{self.session.num_frames - 1}")
        return frame

    def state(self, frame: int = 0) -> dict:
        """Everything the page shows for `frame`: all its boxes (rejected ones included, with their
        index `i`), review statistics and the per-frame marks of the timeline."""
        with self._lock:
            frame = self._frame(frame)
            s = self.session
            return {
                "sequence": s.sequence,
                "mode": s.mode,
                "frame": frame,
                "num_frames": s.num_frames,
                "revision": s.revision,
                "classes": list(CLASSES),
                "stats": s.stats(),
                "frame_marks": self._frame_marks(),
                "save_error": self.save_error,
                "annotations": [
                    {"i": i, "x": a.box.x, "y": a.box.y, "z": a.box.z, "dx": a.box.dx, "dy": a.box.dy,
                     "dz": a.box.dz, "heading": a.box.heading, "label": a.box.label, "score": a.box.score,
                     "track_id": a.box.track_id, "status": a.status.value}
                    for i, a in enumerate(s.annotations(frame))
                ],
            }

    def _frame_marks(self) -> list[int]:
        """One value per frame for the timeline: 0 = no boxes, 1 = boxes still to review, 2 = all reviewed."""
        s = self.session
        return [1 if s.frame_has_pending(f) else (2 if s.annotations(f) else 0) for f in range(s.num_frames)]

    def next_pending(self, frame: int) -> int | None:
        """First frame after `frame` (wrapping around) that still has boxes to review."""
        with self._lock:
            return self.session.next_pending_frame(self._frame(frame))

    def export(self, fmt: str, include_unreviewed: bool = False, include_empty: bool = False) -> ExportResult:
        """Verified boxes as label files. Raises export.ExportError if the format is not possible here."""
        with self._lock:
            return export_session(self.session, self.sequence, fmt, include_unreviewed, include_empty)

    def points(self, frame: int) -> bytes:
        """Raw little-endian float32 (N, 3) x y z of one session frame."""
        with self._lock:
            if not 0 <= frame < self.session.num_frames:
                raise IndexError(frame)
            if frame in self._cache:
                self._cache.move_to_end(frame)
                return self._cache[frame]
            xyz = np.asarray(self.points_source.xyz(self.session.frame_offset + frame), dtype="<f4")
            data = np.ascontiguousarray(xyz[:, :3]).tobytes()
            self._cache[frame] = data
            while len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
            return data

    # ------------------------------------------------------------------ actions
    def act(self, name: str, frame: int | None = None, index: int | None = None,
            track_id: int | None = None, label: str | None = None) -> dict:
        """Run one named action and return the state of the frame it concerns, plus "result".

            view                 frame              log that the human looked at `frame` (no change)
            accept, reject       frame, index
            accept_track, reject_track   track_id   every box of the track that still needs a decision
            undo, redo           frame (whose state is returned)
            add                  frame, label       a new box of class `label`; result {"frame", "index"}
            copy_next            frame, index       copy into frame + 1; result {"frame", "index"} (or
                                                    {"copied": false} on the last frame / track already there)
            set_label            frame, index, label
            forward ... rotate_right, length_minus ... height_plus   frame, index (fixed steps)

        Raises UnknownAction for unknown names, ValueError for missing or invalid arguments and
        SessionClosed after `close()`."""
        with self._lock:
            if self.closed:
                raise SessionClosed("this session was finished and is closed")
            if name not in ACTIONS:
                raise UnknownAction(name)
            frame = self._frame(frame)
            if name in BOX_ACTIONS:
                index = self._index(frame, index)
            result = self._run(name, frame, index, track_id, label)
            self._autosave()
            return {**self.state(result.get("frame", frame) if result else frame), "result": result}

    def _index(self, frame: int, index) -> int:
        if index is None:
            raise ValueError("this action needs the field 'index'")
        index = int(index)
        n = len(self.session.annotations(frame))
        if not 0 <= index < n:
            raise ValueError(f"no box {index} in frame {frame} (it has {n})")
        return index

    def _at(self, frame: int, index: int | None = None) -> ReviewController:
        """A controller pointed at `frame` / `index`, to reuse its editing rules."""
        c = ReviewController(self.session)
        c.frame, c.selected = frame, index
        return c

    def _run(self, name, frame, index, track_id, label) -> dict | None:
        s = self.session
        if name == "view":
            s.log("frame_change", frame=frame)
        elif name in ("accept", "reject"):
            getattr(s, name)(frame, index)
        elif name in TRACK_ACTIONS:
            if track_id is None:
                raise ValueError("this action needs the field 'track_id'")
            return {"count": getattr(s, name)(int(track_id))}
        elif name in ("undo", "redo"):
            return {"step": getattr(s, name)()}
        elif name == "add":
            return {"frame": frame, "index": self._at(frame).add_box(self._label(label))}
        elif name == "copy_next":
            c = self._at(frame, index)
            if not c.copy_selected_to_next_frame():
                return {"copied": False}
            return {"frame": c.frame, "index": c.selected}
        elif name == "set_label":
            s.edit(frame, index, label=self._label(label))
        elif name in NUDGES:
            return {"changed": self._at(frame, index).nudge(**NUDGES[name])}
        else:
            return {"changed": self._at(frame, index).resize(**RESIZES[name])}
        return None

    @staticmethod
    def _label(label) -> str:
        if label not in CLASSES:
            raise ValueError(f"unknown class {label!r} (use one of {', '.join(CLASSES)})")
        return label

    def _autosave(self) -> None:
        if self.session.revision == self._saved_revision:
            return
        try:
            self.store.save(self.key, self.session, expected_revision=self._saved_revision)
            self._saved_revision = self.session.revision
            self.save_error = None
        except (OSError, RevisionConflict) as exc:
            self.save_error = f"Could not save the session {self.key}: {exc}"
            print(f"WARNING: {self.save_error}")

    def save(self) -> None:
        with self._lock:
            self.store.save(self.key, self.session, expected_revision=self._saved_revision)
            self._saved_revision = self.session.revision

    def close(self) -> None:
        """Save one last time and refuse every further action."""
        with self._lock:
            self.save()
            self.closed = True
