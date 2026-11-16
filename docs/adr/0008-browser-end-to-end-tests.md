# 0008 Browser end-to-end tests

Status: accepted (phase 1)

## Decision
- `tests/e2e` drives the real page in Chromium with Playwright (`requirements-dev.txt`, not part of
  `requirements.txt`). Marked `e2e`, deselected by default; run with `pytest -m e2e`.
- Each test starts the real web app in a subprocess on a free port (`tests/e2e/app_server.py`):
  production code everywhere except detection + tracking, which return fixed fake tracks. The
  sequence is a synthetic folder of 6 random point clouds; results go to a temporary
  `LAL_RESULTS_DIR`. No GPU, model or KITTI data.
- Assertions use the API state and the visible DOM, not pixels.
- Any browser console error or uncaught exception fails the test. A test that provokes one on
  purpose (an HTTP 409) declares it; a declared error that does not happen also fails.
- A tab left open on a finished session now gets HTTP 409 ("finished") for its requests instead
  of 404: the manager remembers finished session ids.

## Not covered
Picking a box by clicking it in the 3D view, orbit / pan / zoom, the view buttons, and whether
boxes and points are drawn correctly. Selection is tested through the keyboard and the box list.
Every request outside 127.0.0.1 is blocked and fails the test, so the tests also prove that the
page works offline (three.js and the font are vendored in `web/static/vendor`).
