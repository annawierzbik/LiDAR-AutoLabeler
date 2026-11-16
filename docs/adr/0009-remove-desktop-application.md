# 0009 Remove the desktop application

Status: accepted (before phase 2)

## Context
The project had two review UIs: an Open3D desktop window (`scripts.app` launcher +
`scripts.verify` window, `verification/app.py`, `verification/launcher.py`) and the browser
(`scripts.web`). The multi-service target keeps only the browser; maintaining the Open3D window,
its key handling and its tests in parallel costs effort and keeps open3d as a dependency.

## Decision
Removed:
- `verification/app.py` (Open3D verification window), `verification/launcher.py` (start screen),
  `scripts/app.py`, `scripts/verify.py` (its desktop entry point);
- the Open3D viewers of `scripts/track_sequence.py` (`Player`) and `scripts/detect_frame.py`.
  Both keep their computation and output: `track_sequence` still writes the detection + tracking
  cache file and prints track statistics, `detect_frame` prints the detections;
- `tests/test_verification/test_app_logic.py`, `test_launcher.py`, the desktop part of
  `test_edit_steps.py`;
- `open3d` from `requirements.txt` (no import of it remains in the application).

Moved (not deleted), since the web server needs them:
- `scripts/verify.py: add_arguments, file_names, build_session` -> `pipeline/session_builder.py`;
- `scripts/track_sequence.py: track_sequence, current_meta, save_results, parse_results,
  load_results, print_stats` -> `pipeline/detect_track.py`. Code bodies unchanged; the `--fps`
  argument (desktop playback) is gone.

Kept: `verification/controller.py` (`ReviewController`): the web service reuses its editing rules
(nudge, resize, add on the ground, copy to the next frame), and `tests/test_export` uses it.

## Consequences
- The only review UI is the browser; `python -m scripts.app` and `python -m scripts.verify` no
  longer exist.
- `experiments/lib/envinfo.py` still asks for open3d's version when it writes an environment
  report (`importlib.import_module` in a try/except): on an installation without open3d it records
  `None`. It is a version probe, not a dependency; experiments/ was not changed.
- Detection results can still be inspected without a UI through the cache files and the CLI output.
