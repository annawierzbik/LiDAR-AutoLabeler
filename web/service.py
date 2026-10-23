"""Web-facing wrapper around the review logic (no HTTP and no drawing in here).

`ReviewService` owns one `AnnotationSession` + `ReviewController` and exposes three things the
browser needs: the current state as plain JSON-able dicts, the point cloud of a frame as raw
float32 bytes, and `act(name, ...)` which performs one named user action (the same actions the
Open3D window has on its keys) and autosaves when the annotations changed.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from verification.controller import CLASSES, ReviewController
from verification.session import AnnotationSession
from export import ExportResult, export_session

MOVE_STEP, Z_STEP, ROT_STEP, SIZE_STEP = 0.10, 0.05, math.radians(2.0), 0.10  # same steps as the desktop UI
CACHE_FRAMES = 8


class UnknownAction(KeyError):
    """`act` was called with a name that is not an action."""


class SessionClosed(LookupError):
    """The session was finished; it no longer accepts actions."""


def _actions() -> dict[str, Callable[[ReviewController], object]]:
    return {
        "next": lambda c: c.next_frame(),
        "prev": lambda c: c.prev_frame(),
        "next_pending": lambda c: c.next_pending_frame(),
        "select_next": lambda c: c.select_step(1),
        "select_prev": lambda c: c.select_step(-1),
        "accept": lambda c: c.accept(),
        "reject": lambda c: c.reject(),
        "accept_track": lambda c: c.accept_track(),
        "reject_track": lambda c: c.reject_track(),
        "undo": lambda c: c.undo(),
        "redo": lambda c: c.redo(),
        "add": lambda c: c.add_box(),
        "copy_next": lambda c: c.copy_selected_to_next_frame(),
        "toggle_rejected": lambda c: setattr(c, "show_rejected", not c.show_rejected),
        "forward": lambda c: c.nudge(dx=MOVE_STEP),
        "back": lambda c: c.nudge(dx=-MOVE_STEP),
        "left": lambda c: c.nudge(dy=MOVE_STEP),
        "right": lambda c: c.nudge(dy=-MOVE_STEP),
        "up": lambda c: c.nudge(dz=Z_STEP),
        "down": lambda c: c.nudge(dz=-Z_STEP),
        "rotate_left": lambda c: c.nudge(dheading=ROT_STEP),
        "rotate_right": lambda c: c.nudge(dheading=-ROT_STEP),
        "length_minus": lambda c: c.resize(dlength=-SIZE_STEP),
        "length_plus": lambda c: c.resize(dlength=SIZE_STEP),
        "width_minus": lambda c: c.resize(dwidth=-SIZE_STEP),
        "width_plus": lambda c: c.resize(dwidth=SIZE_STEP),
        "height_minus": lambda c: c.resize(dheight=-SIZE_STEP),
        "height_plus": lambda c: c.resize(dheight=SIZE_STEP),
        # automatic playback: the frame change is not written to the event log
        "play_step": lambda c: c.goto(c.frame + 1, log=False),
    }


def _required(name: str, args: dict, field: str):
    if args.get(field) is None:
        raise ValueError(f"action '{name}' needs the field '{field}'")
    return args[field]


class ReviewService:
    def __init__(self, session: AnnotationSession, loader, save_path: Path | None = None) -> None:
        self.session = session
        self.loader = loader
        self.save_path = Path(save_path) if save_path else None
        self.ctrl = ReviewController(session)
        self._actions = _actions()
        self._lock = threading.RLock()   # re-entrant: act() calls state() while holding it
        self._saved_revision = session.revision
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self.save_error: str | None = None  # last autosave failure, shown in the browser
        self.closed = False

    # ------------------------------------------------------------------ queries
    def state(self) -> dict:
        with self._lock:
            c = self.ctrl
            st = self.session.stats()
            return {
                "sequence": self.session.sequence,
                "mode": self.session.mode,
                "frame": c.frame,
                "num_frames": c.num_frames,
                "selected": c.selected,
                "current_class": c.current_class,
                "classes": list(CLASSES),
                "show_rejected": c.show_rejected,
                "stats": st,
                "frame_marks": self._frame_marks(),
                "save_error": self.save_error,
                "annotations": [
                    {"i": i, "x": a.box.x, "y": a.box.y, "z": a.box.z, "dx": a.box.dx, "dy": a.box.dy,
                     "dz": a.box.dz, "heading": a.box.heading, "label": a.box.label, "score": a.box.score,
                     "track_id": a.box.track_id, "status": a.status.value}
                    for i, a in c.visible()
                ],
            }
    
    def _frame_marks(self) -> list[int]:
        """One value per frame for the timeline: 0 = no boxes, 1 = boxes still to review, 2 = all reviewed."""
        s = self.session
        return [1 if s.frame_has_pending(f) else (2 if s.annotations(f) else 0) for f in range(s.num_frames)]

    def export(self, fmt: str, include_unreviewed: bool = False, include_empty: bool = False) -> ExportResult:
        """Verified boxes as label files. Raises export.ExportError if the format is not possible here."""
        with self._lock:
            return export_session(self.session, self.loader, fmt, include_unreviewed, include_empty)

    def points(self, frame: int) -> bytes:
        """Raw little-endian float32 (N, 3) x y z of one session frame."""
        with self._lock:
            if not 0 <= frame < self.session.num_frames:
                raise IndexError(frame)
            if frame in self._cache:
                self._cache.move_to_end(frame)
                return self._cache[frame]
            xyz = np.asarray(self.loader[self.session.frame_offset + frame].xyz, dtype="<f4")
            data = np.ascontiguousarray(xyz[:, :3]).tobytes()
            self._cache[frame] = data
            while len(self._cache) > CACHE_FRAMES:
                self._cache.popitem(last=False)
            return data

    # ------------------------------------------------------------------ actions
    def act(self, name: str, **args) -> dict:
        """Run one named action and return the new state. Raises UnknownAction for unknown names,
        ValueError for missing or invalid arguments and SessionClosed after `close()`."""
        with self._lock:
            if self.closed:
                raise SessionClosed("this session was finished and is closed")
            c = self.ctrl
            if name == "goto":
                c.goto(int(_required(name, args, "frame")))
            elif name == "select":
                c.select(None if args.get("index") is None else int(args["index"]))
            elif name == "set_class":
                c.set_class(str(_required(name, args, "label")))
            elif name in self._actions:
                self._actions[name](c)
            else:
                raise UnknownAction(name)
            self._autosave()
            return self.state()

    def _autosave(self) -> None:
        if self.save_path is None or self.session.revision == self._saved_revision:
            return
        try:
            self.session.save(self.save_path)
            self._saved_revision = self.session.revision
            self.save_error = None
        except OSError as exc:
            self.save_error = f"Could not save the session to {self.save_path}: {exc}"
            print(f"WARNING: {self.save_error}")

    def save(self) -> None:
        if self.save_path is not None:
            self.session.save(self.save_path)
            self._saved_revision = self.session.revision

    def close(self) -> None:
        """Save one last time and refuse every further action (also from other open tabs)."""
        with self._lock:
            self.save()
            self.closed = True


