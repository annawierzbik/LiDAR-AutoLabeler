"""Start screen of LiDAR AutoLabeler.

    python -m scripts.app

"Annotate a point cloud sequence" opens a folder of .bin frames in the verification window.
Every verification window is a separate process (scripts.verify), so a problem in one window
can never take the start screen down.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _spawn(args) -> subprocess.Popen:
    # cwd is set explicitly: Open3D's folder dialog changes the working directory of this process,
    # and the child must still find "scripts" and the relative data/ and results/ folders
    return subprocess.Popen([sys.executable, "-m", "scripts.verify", *args], cwd=PROJECT_ROOT)


def main() -> None:
    from ingestion.folder import validate_folder
    from verification.launcher import Launcher

    current = {"proc": None, "what": ""}

    def folder_busy() -> bool:
        proc = current["proc"]
        return proc is not None and proc.poll() is None

    def open_folder(path: Path) -> str | None:
        if folder_busy():
            return "A verification window is already open (close it first)."
        frames = validate_folder(path)   # raises a readable error for folders that cannot be used
        current["what"] = path.name
        current["proc"] = _spawn(["--folder", str(path)])
        return (f"Opening {path.name} ({frames} frames)... if detection has to run first it takes "
                "a while (progress in the terminal).")

    def poll() -> str | None:
        proc = current["proc"]
        if proc is None or proc.poll() is None:
            return None
        current["proc"] = None
        if proc.returncode != 0:
            return (f"The window for {current['what']} closed with an error (code {proc.returncode}).\n"
                    "Details are in the terminal.")
        return None

    Launcher(on_open_folder=open_folder, on_tick=poll).run()


if __name__ == "__main__":
    main()
