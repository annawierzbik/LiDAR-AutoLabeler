"""Web-facing operations on one annotation session (no HTTP and no drawing in here).

Stateless actions (docs/adr/0001-explicit-target-actions.md): every action names its target
(frame, and box index or track id where needed). The view state (current frame, selected box,
class for new boxes, whether rejected boxes are shown) lives in the browser.

A `ReviewService` lives for one request (docs/adr/0011): the manager loads the session document
from the SessionStore, the service runs one query or action and stores the result:
- a change of the annotations is saved with `expected_revision` = the revision that was loaded,
  so a concurrent writer is detected (RevisionConflict, HTTP 409) instead of being overwritten;
- events that change nothing (looking at a frame, playback start / stop / heartbeat) are only
  appended to the event log and never change the revision (docs/adr/0013).
"""
from __future__ import annotations

import numpy as np

from export import ExportResult, export_session
from infra.points import PointCache, PointSource
from infra.sessions import SessionStore
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
LOG_ONLY = ("view", "playback_start", "playback_stop", "playback_tick")           # append-only events
BOX_ACTIONS = ("accept", "reject", "copy_next", "set_label", *NUDGES, *RESIZES)   # need frame + index
TRACK_ACTIONS = ("accept_track", "reject_track")                                   # need track_id
ACTIONS = (*LOG_ONLY, "add", "undo", "redo", *BOX_ACTIONS, *TRACK_ACTIONS)


class UnknownAction(KeyError):
    """`act` was called with a name that is not an action."""


def point_bytes(points: PointSource, cache: PointCache | None, cache_key, index: int) -> bytes:
    """Raw little-endian float32 (N, 3) x y z of frame `index` of a point source."""
    if cache is not None:
        hit = cache.get((cache_key, index))
        if hit is not None:
            return hit
    xyz = np.asarray(points.xyz(index), dtype="<f4")
    data = np.ascontiguousarray(xyz[:, :3]).tobytes()
    if cache is not None:
        cache.put((cache_key, index), data)
    return data


class ReviewService:
    def __init__(self, session: AnnotationSession, store: SessionStore, key: str, points: PointSource,
                 sequence=None, point_cache: PointCache | None = None) -> None:
        """`session` as loaded from `store` for this request. `sequence` (a SequenceLoader /
        FolderSequence) is only used by `export` for frame names and calibration."""
        self.session = session
        self.store, self.key = store, key
        self.points_source = points
        self.sequence = sequence
        self.point_cache = point_cache
        self.loaded_revision = session.revision
        self.save_error: str | None = None   # shown in the browser when storing failed (disk full, ...)

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
        return self.session.next_pending_frame(self._frame(frame))

    def export(self, fmt: str, include_unreviewed: bool = False, include_empty: bool = False) -> ExportResult:
        """Verified boxes as label files. Raises export.ExportError if the format is not possible here."""
        return export_session(self.session, self.sequence, fmt, include_unreviewed, include_empty)

    def points(self, frame: int) -> bytes:
        """Raw little-endian float32 (N, 3) x y z of one session frame."""
        if not 0 <= frame < self.session.num_frames:
            raise IndexError(frame)
        return point_bytes(self.points_source, self.point_cache, self.key, self.session.frame_offset + frame)

    # ------------------------------------------------------------------ actions
    def act(self, name: str, frame: int | None = None, index: int | None = None,
            track_id: int | None = None, label: str | None = None) -> dict:
        """Run one named action and return the state of the frame it concerns, plus "result".

            view                 frame              log that the human looked at `frame` (no change)
            playback_start, playback_stop, playback_tick   frame   log playback (no change); the
                                                    browser sends a tick every 10 s while playing
            accept, reject       frame, index
            accept_track, reject_track   track_id   every box of the track that still needs a decision
            undo, redo           frame (whose state is returned)
            add                  frame, label       a new box of class `label`; result {"frame", "index"}
            copy_next            frame, index       copy into frame + 1; result {"frame", "index"} (or
                                                    {"copied": false} on the last frame / track already there)
            set_label            frame, index, label
            forward ... rotate_right, length_minus ... height_plus   frame, index (fixed steps)

        Raises UnknownAction for unknown names, ValueError for missing or invalid arguments and
        infra.sessions.RevisionConflict when the session was changed by someone else meanwhile."""
        if name not in ACTIONS:
            raise UnknownAction(name)
        frame = self._frame(frame)
        if name in BOX_ACTIONS:
            index = self._index(frame, index)
        result = self._run(name, frame, index, track_id, label)
        self._store()
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
        elif name in LOG_ONLY:
            s.log(name, frame=frame)
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

    def _store(self) -> None:
        """Save a changed document with the loaded revision as the expected one, or only append the
        events of this request. OSError -> save_error; RevisionConflict propagates (HTTP 409)."""
        s = self.session
        try:
            if s.revision != self.loaded_revision:
                self.store.save(self.key, s, expected_revision=self.loaded_revision)
                self.loaded_revision = s.revision
            elif s.events:
                self.store.append_events(self.key, s.events)
            s.events = []
            self.save_error = None
        except OSError as exc:
            self.save_error = f"Could not save the session {self.key}: {exc}"
            print(f"WARNING: {self.save_error}")
