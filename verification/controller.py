"""What the verification UI *does*, independent of how it is drawn.

The GUI only translates clicks and key presses into calls on this class and then draws
`controller.visible()`. Everything that decides behaviour (selection, auto-advance after
accept/reject, nudging, adding boxes, ...) lives here so it can be unit tested without a window.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from common.boxes import BoundingBox3D
from verification.session import NEEDS_REVIEW, Annotation, AnnotationSession, Status

CLASSES = ("Car", "Pedestrian", "Cyclist")
# KITTI mean sizes (length, width, height) in metres
DEFAULT_DIMS = {"Car": (3.9, 1.6, 1.56), "Pedestrian": (0.8, 0.6, 1.73), "Cyclist": (1.76, 0.6, 1.73)}
MIN_SIZE = 0.2  # smallest box side (m) reachable with the resize keys
GROUND_Z = -1.73  # LiDAR sits about 1.73 m above the road in KITTI
DEFAULT_SPAWN = (8.0, 0.0)  # where a new box appears: on the road a few metres ahead


class ReviewController:
    def __init__(self, session: AnnotationSession, ground_z: float = GROUND_Z) -> None:
        self.session = session
        self.ground_z = ground_z
        self.frame = 0
        self.selected: Optional[int] = None
        self.current_class = "Car"
        self.show_rejected = False

    # ------------------------------------------------------------------ queries
    @property
    def num_frames(self) -> int:
        return self.session.num_frames

    def annotations(self) -> List[Annotation]:
        return self.session.annotations(self.frame)

    def visible(self) -> List[Tuple[int, Annotation]]:
        """(index, annotation) pairs to draw in the current frame."""
        return [(i, a) for i, a in enumerate(self.annotations())
                if self.show_rejected or a.status is not Status.REJECTED]

    def selected_annotation(self) -> Optional[Annotation]:
        anns = self.annotations()
        return anns[self.selected] if self.selected is not None and self.selected < len(anns) else None

    def _clamp_selection(self) -> None:
        n = len(self.annotations())
        if self.selected is not None and not 0 <= self.selected < n:
            self.selected = n - 1 if n else None

    # --------------------------------------------------------------- navigation
    def goto(self, frame: int, log: bool = True) -> bool:
        """Change frame (clamped). `log=False` for automatic playback, so the event log only
        contains frame changes made by the human."""
        frame = max(0, min(self.num_frames - 1, frame))
        if frame == self.frame:
            return False
        self.frame = frame
        self.selected = None
        if log:
            self.session.log("frame_change", frame=frame)
        return True

    def next_frame(self) -> bool:
        return self.goto(self.frame + 1)

    def prev_frame(self) -> bool:
        return self.goto(self.frame - 1)

    def next_pending_frame(self) -> bool:
        target = self.session.next_pending_frame(self.frame)
        return target is not None and self.goto(target)

    # ---------------------------------------------------------------- selection
    def select(self, idx: Optional[int]) -> None:
        self.selected = idx if idx is None or 0 <= idx < len(self.annotations()) else None

    def select_step(self, step: int = 1) -> None:
        """Cycle through the visible objects of this frame."""
        indices = [i for i, _ in self.visible()]
        if not indices:
            self.selected = None
        elif self.selected not in indices:
            self.selected = indices[0] if step > 0 else indices[-1]
        else:
            self.selected = indices[(indices.index(self.selected) + step) % len(indices)]

    def select_next_pending(self) -> bool:
        """Select the first object of this frame that still needs a decision."""
        before = self.selected
        self.selected = None
        self._advance_after_review()
        if self.selected is None:
            self.selected = before
            return False
        return True

    def _advance_after_review(self) -> None:
        """After accept/reject jump to the next object that still needs a decision."""
        anns = self.annotations()
        n = len(anns)
        start = self.selected if self.selected is not None else -1
        for step in range(1, n + 1):
            i = (start + step) % n
            if anns[i].status in NEEDS_REVIEW:
                self.selected = i
                return

    # ------------------------------------------------------------------- review
    def accept(self) -> bool:
        if self.selected is None:
            return False
        self.session.accept(self.frame, self.selected)
        self._advance_after_review()
        return True

    def reject(self) -> bool:
        if self.selected is None:
            return False
        was_manual = self.selected_annotation().status is Status.MANUAL
        self.session.reject(self.frame, self.selected)
        if was_manual:
            self._clamp_selection()
        self._advance_after_review()
        return True

    def accept_track(self, from_here: bool = False) -> int:
        a = self.selected_annotation()
        if a is None or a.box.track_id is None:
            return 0
        return self.session.accept_track(a.box.track_id, self.frame if from_here else 0)

    def reject_track(self, from_here: bool = False) -> int:
        a = self.selected_annotation()
        if a is None or a.box.track_id is None:
            return 0
        n = self.session.reject_track(a.box.track_id, self.frame if from_here else 0)
        return n

    def undo(self) -> Optional[str]:
        name = self.session.undo()
        self._clamp_selection()
        return name

    def redo(self) -> Optional[str]:
        name = self.session.redo()
        self._clamp_selection()
        return name

    # ------------------------------------------------------------------ editing
    def set_field(self, name: str, value) -> bool:
        """Set one box field of the selected box. Returns False for invalid values."""
        if self.selected is None:
            return False
        try:
            self.session.edit(self.frame, self.selected, **{name: value})
        except ValueError:
            return False
        return True

    def nudge(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, dheading: float = 0.0) -> bool:
        a = self.selected_annotation()
        if a is None:
            return False
        b = a.box
        return self.set_field_many(x=b.x + dx, y=b.y + dy, z=b.z + dz, heading=b.heading + dheading)

    def resize(self, dlength: float = 0.0, dwidth: float = 0.0, dheight: float = 0.0) -> bool:
        """Grow/shrink the selected box. Height changes keep the bottom where it is (box stays on the ground)."""
        a = self.selected_annotation()
        if a is None:
            return False
        b = a.box
        new = (b.dx + dlength, b.dy + dwidth, b.dz + dheight)
        if min(new) < MIN_SIZE:
            return False
        return self.set_field_many(dx=new[0], dy=new[1], dz=new[2], z=b.z + dheight / 2)

    def set_field_many(self, **fields) -> bool:
        if self.selected is None:
            return False
        try:
            self.session.edit(self.frame, self.selected, **fields)
        except ValueError:
            return False
        return True

    def set_class(self, label: str) -> None:
        """Class used for new boxes; also relabels the selected box."""
        if label not in CLASSES:
            raise ValueError(f"unknown class {label}")
        self.current_class = label
        if self.selected is not None:
            self.set_field("label", label)

    # ------------------------------------------------------------ adding boxes
    def add_box(self, label: Optional[str] = None) -> int:
        """Add a human box standing on the ground at DEFAULT_SPAWN."""
        label = label or self.current_class
        l, w, h = DEFAULT_DIMS[label]
        x, y = DEFAULT_SPAWN
        box = BoundingBox3D(float(x), float(y), self.ground_z + h / 2, l, w, h, 0.0, label, 1.0)
        self.selected = self.session.add_manual(self.frame, box)
        return self.selected

    def copy_selected_to_next_frame(self) -> bool:
        """Copy the selected box into the next frame (same track id) and move there."""
        if self.selected is None or self.frame + 1 >= self.num_frames:
            return False
        try:
            new_idx = self.session.copy_to_frame(self.frame, self.selected, self.frame + 1)
        except ValueError:
            return False
        self.goto(self.frame + 1)
        self.selected = new_idx
        return True

    # ------------------------------------------------------------------ summary
    def progress_text(self) -> str:
        st = self.session.stats()
        return (f"frame {self.frame + 1}/{self.num_frames}   reviewed {st['reviewed']}/{st['total']} "
                f"({st['reviewed_fraction'] * 100:.0f}%)")