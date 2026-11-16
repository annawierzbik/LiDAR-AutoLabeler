"""File names of saved sessions, cached detections and metrics (verification, evaluation, scripts)."""
from __future__ import annotations

from infra.settings import PROJECT_ROOT, get_settings  # noqa: F401  (PROJECT_ROOT re-exported)

RESULTS_DIR = get_settings().results_dir  # absolute: results land here whatever the working directory


def range_tag(name: str, start: int, stop: int | None) -> str:
    """`name` for a whole sequence, `name_f<start>-<stop>` for a part of it."""
    return name if start == 0 and stop is None else f"{name}_f{start}-{stop}"


def session_file(name: str, mode: str, start: int = 0, stop: int | None = None):
    return RESULTS_DIR / "sessions" / f"{range_tag(name, start, stop)}_{mode}.json"
