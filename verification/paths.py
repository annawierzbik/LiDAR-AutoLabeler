"""File names of saved sessions, cached detections and metrics (verification, evaluation, scripts)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"  # absolute: results land here whatever the working directory


def range_tag(name: str, start: int, stop: Optional[int]) -> str:
    """`name` for a whole sequence, `name_f<start>-<stop>` for a part of it."""
    return name if start == 0 and stop is None else f"{name}_f{start}-{stop}"


def session_file(name: str, mode: str, start: int = 0, stop: Optional[int] = None) -> Path:
    return RESULTS_DIR / "sessions" / f"{range_tag(name, start, stop)}_{mode}.json"
