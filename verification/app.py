"""Open3D window for verifying / drawing 3D boxes.

The window only draws things and forwards button clicks and key presses to ReviewController
(verification/controller.py), where all behaviour lives and is unit tested.

    Mouse   left drag = rotate, wheel = zoom, Ctrl+drag = pan  (boxes are selected from the list)
    Keys    see HELP_LINES below (also shown in the side panel)
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from common.boxes import EDGES
from verification.controller import CLASSES, ReviewController
from verification.session import AnnotationSession, Status

STATUS_COLORS = {
    Status.PENDING: (1.0, 0.65, 0.0),    # orange: waiting for a decision
    Status.ACCEPTED: (0.25, 1.0, 0.25),  # green
    Status.REJECTED: (0.55, 0.15, 0.15), # dark red (only shown on request)
    Status.EDITED: (0.2, 0.9, 1.0),      # cyan
    Status.MANUAL: (1.0, 0.3, 1.0),      # magenta: drawn by the human
    Status.INTERPOLATED: (0.75, 0.55, 1.0),  # light purple: filled in between two boxes of a track
}
SELECTED_COLOR = (1.0, 1.0, 1.0)
FRONT_DIAGONALS = [(0, 5), (1, 4)]  # cross on the front face shows where the box is heading

MOVE_STEP, Z_STEP, ROT_STEP, SIZE_STEP = 0.10, 0.05, math.radians(2.0), 0.10

HELP_LINES = [
    "Left/Right  previous/next frame",
    "Space  play/pause    N  next frame with pending boxes",
    "Tab or [ ]  select next/previous object",
    "A or Enter  accept    R or Del  reject",
    "T  accept whole track    X  reject whole track",
    "I J K L  move box (forward/left/back/right)",
    "PgUp/PgDn  box up/down    , .  rotate",
    "1/2  length -/+    3/4  width -/+    5/6  height -/+",
    "B  add box    C  copy box to next frame",
    "Z  undo    Y  redo",
]


# height -> color: dark slate blue (road level) -> light grey (tall things), kept dim so the boxes stand out
RAMP_T = [0.0, 0.25, 0.5, 0.75, 1.0]
RAMP_RGB = np.array([
    (0.12, 0.16, 0.32), (0.20, 0.28, 0.45), (0.35, 0.45, 0.55), (0.55, 0.62, 0.68), (0.80, 0.82, 0.85),
])


def height_colors(xyz: np.ndarray) -> np.ndarray:
    t = np.clip((xyz[:, 2] + 2.0) / 3.5, 0.0, 1.0)  # z from -2.0 m (road) to 1.5 m
    return np.stack([np.interp(t, RAMP_T, RAMP_RGB[:, k]) for k in range(3)], axis=1)


class VerificationApp:
    def __init__(self, session: AnnotationSession, loader, save_path, fps: float = 10.0) -> None:
        self.session = session
        self.ctrl = ReviewController(session)
        self.loader = loader  # SequenceLoader: loader[i].xyz -> (N, 3) points
        self.save_path = Path(save_path)
        self.fps = fps

        self.playing = False
        self._last_tick = 0.0
        self._updating = False      # True while we set widget values from code
        self._saved_revision = session.revision
        self._box_names: List[str] = []
        self._labels: list = []
        self._has_points = False
        self.closed = True          # becomes False in show(), True again when the window is closed

    # ================================================================== window
    def run(self) -> None:
        """Stand-alone use: start the GUI loop with this window only."""
        gui.Application.instance.initialize()
        self.show()
        gui.Application.instance.run()

    def show(self) -> None:
        """Open the window inside an already initialized application (e.g. from the launcher)."""
        self.closed = False
        self.window = gui.Application.instance.create_window(
            f"Verification - sequence {self.session.sequence} ({self.session.mode})", 1650, 950)
        self._build()
        self._refresh_points()
        self._refresh_boxes()
        self._refresh_panel()

    def _build(self) -> None:
        w = self.window
        em = w.theme.font_size

        # ---- 3D view
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(w.renderer)
        scene = self.scene_widget.scene
        scene.set_background([0.0, 0.0, 0.0, 1.0])
        scene.show_axes(False)
        bounds = o3d.geometry.AxisAlignedBoundingBox([-10, -40, -3], [80, 40, 5])
        self.scene_widget.setup_camera(60.0, bounds, [15.0, 0.0, 0.0])
        self.scene_widget.look_at([15.0, 0.0, 0.0], [-18.0, 0.0, 22.0], [0.0, 0.0, 1.0])
        self.scene_widget.set_on_key(self._on_key)

        # ---- side panel
        self.panel = gui.Vert(0.4 * em, gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em))
        p = self.panel

        self.progress_label = gui.Label("")
        p.add_child(self.progress_label)

        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, max(1, self.ctrl.num_frames - 1))
        self.slider.set_on_value_changed(self._on_slider)
        p.add_child(self.slider)

        nav = gui.Horiz(0.3 * em)
        self.play_button = self._button("Play", self._toggle_play)
        nav.add_child(self._button("<", lambda: self._navigate(self.ctrl.prev_frame)))
        nav.add_child(self.play_button)
        nav.add_child(self._button(">", lambda: self._navigate(self.ctrl.next_frame)))
        nav.add_child(self._button("Next pending", lambda: self._navigate(self.ctrl.next_pending_frame)))
        p.add_child(nav)

        fps_row = gui.Horiz(0.3 * em)
        fps_row.add_child(gui.Label("Playback fps"))
        self.fps_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.fps_edit.set_limits(0.5, 30.0)
        self.fps_edit.set_value(self.fps)
        self.fps_edit.set_on_value_changed(self._on_fps)
        fps_row.add_child(self.fps_edit)
        p.add_child(fps_row)

        p.add_child(gui.Label("Objects in this frame"))
        self.listview = gui.ListView()
        self.listview.set_max_visible_items(9)
        self.listview.set_on_selection_changed(self._on_list)
        p.add_child(self.listview)

        row = gui.Horiz(0.3 * em)
        row.add_child(self._button("Accept (A)", lambda: self._act(self._accept)))
        row.add_child(self._button("Reject (R)", lambda: self._act(self._reject)))
        p.add_child(row)
        row = gui.Horiz(0.3 * em)
        row.add_child(self._button("Accept track (T)", lambda: self._act(self.ctrl.accept_track)))
        row.add_child(self._button("Reject track (X)", lambda: self._act(self.ctrl.reject_track)))
        p.add_child(row)
        row = gui.Horiz(0.3 * em)
        row.add_child(self._button("Undo (Z)", lambda: self._act(self.ctrl.undo)))
        row.add_child(self._button("Redo (Y)", lambda: self._act(self.ctrl.redo)))
        p.add_child(row)

        row = gui.Horiz(0.3 * em)
        row.add_child(gui.Label("Class"))
        self.class_combo = gui.Combobox()
        for name in CLASSES:
            self.class_combo.add_item(name)
        self.class_combo.set_on_selection_changed(self._on_class)
        row.add_child(self.class_combo)
        p.add_child(row)
        row = gui.Horiz(0.3 * em)
        row.add_child(self._button("Add box (B)", lambda: self._act(self.ctrl.add_box)))
        row.add_child(self._button("Copy to next (C)", lambda: self._act(self._copy_next)))
        p.add_child(row)

        self.show_rejected = gui.Checkbox("Show rejected")
        self.show_rejected.set_on_checked(self._on_show_rejected)
        self.show_labels = gui.Checkbox("Show labels")
        self.show_labels.checked = True
        self.show_labels.set_on_checked(lambda _: self._refresh_boxes())
        p.add_child(self.show_rejected)
        p.add_child(self.show_labels)

        edit = gui.CollapsableVert("Edit selected box", 0.3 * em, gui.Margins(0.5 * em, 0.3 * em, 0, 0.3 * em))
        self.fields = {}
        for name in ("x", "y", "z", "dx", "dy", "dz", "heading"):
            line = gui.Horiz(0.3 * em)
            line.add_child(gui.Label("heading (deg)" if name == "heading" else name))
            edit_widget = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            edit_widget.decimal_precision = 2
            edit_widget.set_on_value_changed(lambda v, n=name: self._on_field(n, v))
            self.fields[name] = edit_widget
            line.add_child(edit_widget)
            edit.add_child(line)
        p.add_child(edit)

        keys = gui.CollapsableVert("Keys", 0.2 * em, gui.Margins(0.5 * em, 0.3 * em, 0, 0.3 * em))
        for line in HELP_LINES:
            keys.add_child(gui.Label(line))
        keys.set_is_open(False)
        p.add_child(keys)

        legend = gui.Horiz(0.4 * em)
        for status in (Status.PENDING, Status.INTERPOLATED, Status.ACCEPTED, Status.EDITED, Status.MANUAL):
            lab = gui.Label(status.value)
            lab.text_color = gui.Color(*STATUS_COLORS[status])
            legend.add_child(lab)
        p.add_child(legend)

        w.add_child(self.scene_widget)
        w.add_child(self.panel)
        w.set_on_layout(self._on_layout)
        w.set_on_tick_event(self._on_tick)
        w.set_on_close(self._on_close)

    def _button(self, text: str, callback) -> "gui.Button":
        b = gui.Button(text)
        b.set_on_clicked(callback)
        return b

    def _on_layout(self, ctx) -> None:
        r = self.window.content_rect
        panel_w = int(24 * self.window.theme.font_size)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)

    # =============================================================== rendering
    def _refresh_points(self) -> None:
        frame = self.loader[self.session.frame_offset + self.ctrl.frame]
        xyz = np.asarray(frame.xyz, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(height_colors(xyz))

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0 * self.window.scaling

        scene = self.scene_widget.scene
        if self._has_points:
            scene.remove_geometry("points")
        scene.add_geometry("points", pcd, mat)
        self._has_points = True

    def _refresh_boxes(self) -> None:
        sw, scene = self.scene_widget, self.scene_widget.scene
        for name in self._box_names:
            scene.remove_geometry(name)
        for lab in self._labels:
            sw.remove_3d_label(lab)
        self._box_names, self._labels = [], []

        edges = list(EDGES) + FRONT_DIAGONALS
        for i, ann in self.ctrl.visible():
            selected = i == self.ctrl.selected
            color = SELECTED_COLOR if selected else STATUS_COLORS[ann.status]

            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(ann.box.corners)
            lines.lines = o3d.utility.Vector2iVector(edges)
            lines.colors = o3d.utility.Vector3dVector([color] * len(edges))
            mat = rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = (6.0 if selected else 3.0) * self.window.scaling
            name = f"box{i}"
            scene.add_geometry(name, lines, mat)
            self._box_names.append(name)

            if self.show_labels.checked:
                b = ann.box
                suffix = {Status.PENDING: f" {b.score:.2f}", Status.INTERPOLATED: " ~"}.get(ann.status, "")
                text = f"#{b.track_id} {b.label}{suffix}"
                label = sw.add_3d_label(b.center + np.array([0.0, 0.0, b.dz / 2 + 0.3]), text)
                label.color = gui.Color(*color)
                self._labels.append(label)

    def _refresh_panel(self) -> None:
        self._updating = True
        try:
            c = self.ctrl
            self.progress_label.text = c.progress_text()
            self.slider.int_value = c.frame

            items = []
            for i, ann in c.visible():
                b = ann.box
                tid = "-" if b.track_id is None else b.track_id
                items.append(f"{i:>2}  #{tid} {b.label} {b.score:.2f}  [{ann.status.value}]")
            self.listview.set_items(items)
            visible_idx = [i for i, _ in c.visible()]
            self.listview.selected_index = visible_idx.index(c.selected) if c.selected in visible_idx else -1

            sel = c.selected_annotation()
            for name, widget in self.fields.items():
                widget.enabled = sel is not None
                if sel is not None:
                    value = getattr(sel.box, name)
                    widget.set_value(math.degrees(value) if name == "heading" else value)
            shown_class = sel.box.label if sel is not None else c.current_class
            self.class_combo.selected_index = CLASSES.index(shown_class) if shown_class in CLASSES else 0
            self.show_rejected.checked = c.show_rejected
        finally:
            self._updating = False
        self.window.post_redraw()

    def _refresh_all(self) -> None:
        self._refresh_points()
        self._refresh_boxes()
        self._refresh_panel()

    def _after_action(self) -> None:
        """Redraw boxes + panel and autosave if the annotations changed."""
        self._refresh_boxes()
        self._refresh_panel()
        if self.session.revision != self._saved_revision:
            self._save()

    def _save(self) -> None:
        try:
            self.session.save(self.save_path)
            self._saved_revision = self.session.revision
        except OSError as exc:
            print(f"WARNING: could not save session to {self.save_path}: {exc}")

    # ================================================================= actions
    def _act(self, action) -> None:
        action()
        self._after_action()

    def _accept(self) -> None:
        if self.ctrl.selected is None:
            self.ctrl.select_next_pending()
        else:
            self.ctrl.accept()

    def _reject(self) -> None:
        if self.ctrl.selected is None:
            self.ctrl.select_next_pending()
        else:
            self.ctrl.reject()

    def _copy_next(self) -> None:
        if self.ctrl.copy_selected_to_next_frame():
            self._refresh_points()

    def _navigate(self, move) -> None:
        if move():
            self._refresh_all()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.text = "Pause" if self.playing else "Play"
        self._last_tick = time.time()

    # ================================================================ callbacks
    def _on_slider(self, value) -> None:
        if not self._updating and self.ctrl.goto(int(value)):
            self._refresh_all()

    def _on_fps(self, value) -> None:
        self.fps = float(value)

    def _on_list(self, text: str, is_double_click: bool) -> None:
        if self._updating:
            return
        visible_idx = [i for i, _ in self.ctrl.visible()]
        k = self.listview.selected_index
        self.ctrl.select(visible_idx[k] if 0 <= k < len(visible_idx) else None)
        self._refresh_boxes()
        self._refresh_panel()

    def _on_class(self, text: str, index: int) -> None:
        if self._updating:
            return
        self.ctrl.set_class(text)
        self._after_action()

    def _on_show_rejected(self, checked: bool) -> None:
        if self._updating:
            return
        self.ctrl.show_rejected = checked
        sel = self.ctrl.selected_annotation()
        if sel is not None and sel.status is Status.REJECTED and not checked:
            self.ctrl.select(None)  # the selected box just became invisible
        self._refresh_boxes()
        self._refresh_panel()

    def _on_field(self, name: str, value: float) -> None:
        if self._updating:
            return
        if name == "heading":
            value = math.radians(value)
        self.ctrl.set_field(name, value)  # invalid values (e.g. size <= 0) are ignored
        self._after_action()

    def _on_tick(self) -> bool:
        if not self.playing or time.time() - self._last_tick < 1.0 / self.fps:
            return False
        self._last_tick = time.time()
        if not self.ctrl.goto(self.ctrl.frame + 1, log=False):
            self._toggle_play()  # reached the last frame
            return True
        self._refresh_all()
        return True

    def _on_close(self) -> bool:
        self.session.log("session_end")
        self._save()
        self.closed = True
        return True

    # ------------------------------------------------------------------ keys
    def _on_key(self, event) -> "gui.Widget.EventCallbackResult":
        handled, ignored = gui.Widget.EventCallbackResult.HANDLED, gui.Widget.EventCallbackResult.IGNORED
        if event.type != gui.KeyEvent.Type.DOWN:
            return ignored
        k, c, K = event.key, self.ctrl, gui.KeyName
        if ord("a") <= k <= ord("z"):
            k -= 32  # Open3D reports letters as lowercase codes (A == 97); compare with ord("A") below

        if k == K.RIGHT:
            self._navigate(c.next_frame)
        elif k == K.LEFT:
            self._navigate(c.prev_frame)
        elif k == K.SPACE:
            self._toggle_play()
        elif k == ord("N"):
            self._navigate(c.next_pending_frame)
        elif k in (K.TAB, K.RIGHT_BRACKET):
            c.select_step(1); self._after_action()
        elif k == K.LEFT_BRACKET:
            c.select_step(-1); self._after_action()
        elif k in (ord("A"), K.ENTER):
            self._act(self._accept)
        elif k in (ord("R"), K.DELETE, K.BACKSPACE):
            self._act(self._reject)
        elif k == ord("T"):
            self._act(c.accept_track)
        elif k == ord("X"):
            self._act(c.reject_track)
        elif k == ord("I"):
            self._act(lambda: c.nudge(dx=MOVE_STEP))
        elif k == ord("K"):
            self._act(lambda: c.nudge(dx=-MOVE_STEP))
        elif k == ord("J"):
            self._act(lambda: c.nudge(dy=MOVE_STEP))
        elif k == ord("L"):
            self._act(lambda: c.nudge(dy=-MOVE_STEP))
        elif k == K.PAGE_UP:
            self._act(lambda: c.nudge(dz=Z_STEP))
        elif k == K.PAGE_DOWN:
            self._act(lambda: c.nudge(dz=-Z_STEP))
        elif k == K.COMMA:
            self._act(lambda: c.nudge(dheading=ROT_STEP))
        elif k == K.PERIOD:
            self._act(lambda: c.nudge(dheading=-ROT_STEP))
        elif k == ord("1"):
            self._act(lambda: c.resize(dlength=-SIZE_STEP))
        elif k == ord("2"):
            self._act(lambda: c.resize(dlength=SIZE_STEP))
        elif k == ord("3"):
            self._act(lambda: c.resize(dwidth=-SIZE_STEP))
        elif k == ord("4"):
            self._act(lambda: c.resize(dwidth=SIZE_STEP))
        elif k == ord("5"):
            self._act(lambda: c.resize(dheight=-SIZE_STEP))
        elif k == ord("6"):
            self._act(lambda: c.resize(dheight=SIZE_STEP))
        elif k == ord("B"):
            self._act(c.add_box)
        elif k == ord("C"):
            self._act(self._copy_next)
        elif k == ord("Z"):
            self._act(c.undo)
        elif k == ord("Y"):
            self._act(c.redo)
        else:
            return ignored
        return handled