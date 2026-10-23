"""Exercises VerificationApp's own logic (key handling, refresh code) with the
window replaced by mocks. Not a rendering test: that must be checked by looking at the window."""
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("open3d")
from open3d.visualization import gui

from common.boxes import BoundingBox3D
from verification.app import VerificationApp
from verification.session import AnnotationSession, Status

K = gui.KeyName


class FakeLoader:
    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        return SimpleNamespace(xyz=rng.uniform(-20, 20, (200, 3)))


def car(x, y, tid):
    return BoundingBox3D(x, y, -0.95, 3.9, 1.6, 1.56, 0.0, "Car", 0.9, tid)


@pytest.fixture
def app(tmp_path):
    results = [[car(15, 0, 0), car(30, 8, 1)], [car(16, 0, 0)], []]
    session = AnnotationSession.from_tracking("0000", results)
    a = VerificationApp(session, FakeLoader(), tmp_path / "s.json")
    a.window = MagicMock(scaling=1.0)
    a.scene_widget = MagicMock()
    a.scene_widget.frame = SimpleNamespace(x=0, y=0, width=1000, height=800)
    a.progress_label, a.slider, a.listview = MagicMock(), MagicMock(), MagicMock()
    a.class_combo, a.show_rejected, a.play_button = MagicMock(), MagicMock(), MagicMock()
    a.show_rejected.checked = False
    a.show_labels = MagicMock(checked=True)
    a.fields = {n: MagicMock() for n in ("x", "y", "z", "dx", "dy", "dz", "heading")}
    a.fps_edit = MagicMock()
    return a


def key(k):
    return SimpleNamespace(type=gui.KeyEvent.Type.DOWN, key=k)


def press(app, k):
    return app._on_key(key(k))


def test_refresh_draws_a_box_and_label_per_visible_annotation(app):
    app._refresh_points()
    app._refresh_boxes()
    assert len(app._box_names) == 2 and app.scene_widget.add_3d_label.call_count == 2
    app.ctrl.select(0); app.ctrl.reject()
    app._refresh_boxes()
    assert len(app._box_names) == 1  # rejected boxes are hidden


def test_review_keys(app):
    press(app, ord("A"))                       # nothing selected -> selects first pending
    assert app.ctrl.selected == 0
    press(app, K.ENTER)                        # accepts it and moves on
    assert app.session.annotations(0)[0].status is Status.ACCEPTED and app.ctrl.selected == 1
    press(app, ord("R"))
    assert app.session.annotations(0)[1].status is Status.REJECTED
    press(app, ord("Z")); press(app, ord("Z"))
    assert app.session.stats()["reviewed"] == 0
    press(app, ord("Y"))
    assert app.session.stats()["reviewed"] == 1


def test_autosave_on_change_only(app, tmp_path):
    assert not (tmp_path / "s.json").exists()
    press(app, K.RIGHT)                        # navigation: nothing to save
    assert not (tmp_path / "s.json").exists()
    press(app, K.LEFT)
    press(app, K.TAB); press(app, ord("A"))
    saved = AnnotationSession.load(tmp_path / "s.json")
    assert saved.stats()["accepted"] == 1


def test_navigation_and_playback(app):
    press(app, K.RIGHT); assert app.ctrl.frame == 1
    press(app, K.LEFT); assert app.ctrl.frame == 0
    logged = sum(e.action == "frame_change" for e in app.session.events)   # the two key presses
    press(app, K.SPACE); assert app.playing
    app._last_tick = 0.0
    assert app._on_tick() and app.ctrl.frame == 1
    app._last_tick = 0.0; app._on_tick()
    app._last_tick = 0.0; app._on_tick()      # last frame reached -> playback stops
    assert app.ctrl.frame == 2 and not app.playing
    assert sum(e.action == "frame_change" for e in app.session.events) == logged   # playback is not logged


def test_move_rotate_height_keys(app):
    press(app, K.TAB)
    press(app, ord("I")); press(app, ord("J")); press(app, K.PAGE_UP); press(app, K.COMMA)
    b = app.ctrl.selected_annotation().box
    assert (b.x, b.y, b.z) == pytest.approx((15.1, 0.1, -0.9))
    assert b.heading == pytest.approx(math.radians(2))


def test_resize_keys(app):
    press(app, K.TAB)
    for k in "2", "4", "6", "6":
        press(app, ord(k))
    press(app, ord("1"))
    b = app.ctrl.selected_annotation().box
    assert (b.dx, b.dy, b.dz) == pytest.approx((3.9, 1.7, 1.76))


def test_unhandled_keys_and_key_up_are_ignored(app):
    ignored = gui.Widget.EventCallbackResult.IGNORED
    assert press(app, ord("Q")) == ignored
    assert app._on_key(SimpleNamespace(type=gui.KeyEvent.Type.UP, key=ord("A"))) == ignored


def test_field_edit_and_invalid_value(app):
    app.ctrl.select(0)
    app._on_field("heading", 90.0)
    assert app.ctrl.selected_annotation().box.heading == pytest.approx(math.pi / 2)
    app._on_field("dx", -1.0)                  # invalid -> ignored
    assert app.ctrl.selected_annotation().box.dx == 3.9


def test_widget_callbacks_ignored_while_updating(app):
    app.ctrl.select(0)
    app._updating = True
    app._on_field("x", 99.0)
    app._on_slider(2)
    assert app.ctrl.selected_annotation().box.x == 15 and app.ctrl.frame == 0


def test_close_saves_and_logs(app, tmp_path):
    assert app._on_close()
    assert AnnotationSession.load(tmp_path / "s.json").events[-2].action == "session_end"


def test_letter_keys_use_the_codes_open3d_really_sends(app):
    """Open3D reports letters as lowercase ASCII codes (KeyName.A == 97), not ord('A') == 65."""
    assert int(K.A) == ord("a") and int(K.I) == ord("i")
    press(app, int(K.TAB))
    press(app, int(K.I))                       # move forward
    assert app.ctrl.selected_annotation().box.x == pytest.approx(15.1)
    press(app, int(K.A))                       # accept
    assert app.session.stats()["reviewed"] == 1