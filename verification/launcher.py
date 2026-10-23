"""Start screen of the desktop application.

Only the screen lives here. What the button does is passed in, so it can be built and changed
separately (see scripts/app.py):

    on_open_folder(path)   annotate the folder chosen in the dialog; returns a status text
    on_tick()              called regularly; may return a message that is then shown in a dialog
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from open3d.visualization import gui


class Launcher:
    def __init__(
        self,
        on_open_folder: Callable[[Path], str | None] | None = None,
        on_tick: Callable[[], str | None] | None = None,
    ) -> None:
        self.on_open_folder = on_open_folder
        self.on_tick = on_tick
        self._start_dir = str(Path.cwd())  # the file dialog changes the working directory; remember the start

    def run(self) -> None:
        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("LiDAR AutoLabeler", 700, 360)
        self._build()
        app.run()

    # ================================================================== layout
    def _build(self) -> None:
        w = self.window
        em = w.theme.font_size
        self.root = gui.Vert(0, gui.Margins(0, 0, 0, 0))

        self.home = gui.Vert(1.0 * em, gui.Margins(2 * em, 2 * em, 2 * em, 2 * em))
        self.home.add_child(gui.Label("LiDAR AutoLabeler"))
        self.home.add_child(gui.Label("Automatic 3D box annotation with human verification"))
        self.home.add_child(gui.Label(""))
        folder = gui.Button("Annotate a point cloud sequence (choose a folder)...")
        folder.set_on_clicked(self._choose_folder)
        self.home.add_child(folder)
        self.home.add_child(gui.Label(""))
        self.status = gui.Label("")
        self.home.add_child(self.status)

        self.root.add_child(self.home)
        w.add_child(self.root)
        w.set_on_layout(self._on_layout)
        w.set_on_tick_event(self._on_tick)

    def _on_layout(self, ctx) -> None:
        self.root.frame = self.window.content_rect

    def _on_tick(self) -> bool:
        message = self.on_tick() if self.on_tick else None
        if message:
            self._dialog("Verification window", message)
        return bool(message)

    # ================================================================== folders
    def _choose_folder(self) -> None:
        if self.on_open_folder is None:
            self._dialog("Not available", "Opening folders is not available.")
            return
        w = self.window
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Choose the folder with .bin frames", w.theme)
        dlg.set_path(self._start_dir)
        dlg.set_on_cancel(w.close_dialog)
        dlg.set_on_done(self._folder_chosen)
        w.show_dialog(dlg)

    def _folder_chosen(self, path: str) -> None:
        self.window.close_dialog()
        try:
            self.status.text = self.on_open_folder(Path(path)) or ""
        except Exception as exc:  # show the problem instead of killing the whole application
            self._dialog("Could not open", f"{type(exc).__name__}: {exc}")

    def _dialog(self, title: str, text: str) -> None:
        w = self.window
        em = w.theme.font_size
        dlg = gui.Dialog(title)
        body = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        for line in text.split("\n"):
            body.add_child(gui.Label(line))
        ok = gui.Button("OK")
        ok.set_on_clicked(w.close_dialog)
        body.add_child(ok)
        dlg.add_child(body)
        w.show_dialog(dlg)
