from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("open3d")

from verification.launcher import Launcher  # noqa: E402


def make(**kw):
    launcher = Launcher(**kw)
    launcher.window = MagicMock()
    launcher.status = MagicMock()
    launcher._dialog = MagicMock()
    return launcher


def test_chosen_folder_is_opened_and_status_shown():
    opener = MagicMock(return_value="Opening x")
    launcher = make(on_open_folder=opener)
    launcher._folder_chosen("C:/clouds/x")
    opener.assert_called_once_with(Path("C:/clouds/x"))
    assert launcher.status.text == "Opening x"
    launcher.window.close_dialog.assert_called_once()


def test_error_when_opening_a_folder_is_shown_not_raised():
    launcher = make(on_open_folder=MagicMock(side_effect=FileNotFoundError("No .bin point cloud files")))
    launcher._folder_chosen("C:/empty")
    title, text = launcher._dialog.call_args.args
    assert "FileNotFoundError" in text and "No .bin" in text


def test_tick_shows_a_message_only_when_there_is_one():
    ticks = MagicMock(side_effect=[None, "The window for x closed with an error (code 1)."])
    launcher = make(on_tick=ticks)
    assert launcher._on_tick() is False
    launcher._dialog.assert_not_called()
    assert launcher._on_tick() is True
    assert "code 1" in launcher._dialog.call_args.args[1]
