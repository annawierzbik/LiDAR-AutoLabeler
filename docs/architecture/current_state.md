# Current state (before the multi-service migration)

> Historical snapshot of 2026-11-16. Later changes: phase 1 (ADRs 0001-0008) and the removal of the
> desktop application (ADR 0009); the Open3D windows described below no longer exist.

Snapshot of `web/`, `verification/`, `export/` and `scripts/` as of commit `ab07700` plus the
uncommitted working tree of 2026-11-16 (cache metadata, FOV crop, shared edit steps, second-tab
refusal). Read-only report: no code was changed for it. Line numbers refer to that working tree.

## 1. Process model

| Entry point | Process | UI | How detection runs |
|---|---|---|---|
| `python -m scripts.web` | one uvicorn process, one FastAPI app (`web/server.py`) | browser (`web/static/*.html`, three.js from jsDelivr CDN) | inside the API process, in a background thread per job |
| `python -m scripts.app` | launcher process (Open3D start screen) + **one child process per opened folder** (`subprocess.Popen` of `scripts.verify`, `scripts/app.py:18-21`) | Open3D desktop window | inside the child process, before the window opens |
| `python -m scripts.verify` | one process | Open3D desktop window (`verification/app.py`) | synchronously, before the window opens |
| `scripts.track_sequence`, `detect_frame`, `evaluate_*`, `export_labels`, `verify_dataset` | one process each, CLI | none / Open3D viewer | synchronously |

There is no database, no queue and no authentication. Every process reads and writes the local
disk directly; the browser talks only to the FastAPI process.

## 2. State

### 2.1 In memory (lost when the process ends)

**Web server (`scripts.web`)**

| Where | What | Lifetime / notes |
|---|---|---|
| `SessionManager.jobs` (`web/manager.py:54`) | `sid -> Job{sid, status: loading/ready/error, message, title, service}` | until `finish`; a failed build removes the job (`:89`); a refused second tab stays with status `error` |
| `SessionManager.by_path` (`web/manager.py:55`) | session file path -> `ReviewService` | one live service per session file; used to refuse a second tab |
| `SessionManager._lock` (`:56`) | `threading.Lock` around `by_path` | `jobs` is mutated without the lock (`start_job`, `finish`) |
| `ReviewService` (`web/service.py:74-86`) | `session` (`AnnotationSession`), `ctrl` (`ReviewController`), `loader`, `save_path`, `_saved_revision`, `_cache`, `save_error`, `closed`, `_lock` (`RLock`) | one per open session |
| `ReviewService._cache` (`:83`) | LRU of the last 8 frames' point clouds as raw float32 bytes (`CACHE_FRAMES = 8`) | per service |
| `ReviewController` (`verification/controller.py:20-27`) | **UI state on the server**: `frame`, `selected`, `current_class`, `show_rejected` | shared by every client of that service |
| `AnnotationSession` (`verification/session.py:162-184`) | `frames` (annotations), `events` (timestamped log), `_undo` / `_redo` stacks, `_next_track_id`, `max_gap`, `revision` | `frames`, `events`, `next_track_id` are saved; **`_undo`, `_redo` and `revision` are not** (revision restarts at 0 on load, undo history is lost on restart) |
| loader (`SequenceLoader` / `FolderSequence`, `ingestion/`) | file list, parsed calibration, parsed GT labels | per service |
| `detection.detector` / pcdet | a `Detector` (model on the GPU) is created **per job** inside `track_sequence` (`scripts/track_sequence.py:42`) and dropped afterwards; `cfg_from_yaml_file` writes **pcdet's process-global `cfg`** (`detection/detector.py:99-100`) | two concurrent jobs with different configs would race on the global config |
| `verification.cache._warned` (`verification/cache.py:40`) | set of warnings already printed | process lifetime |

**Desktop (`scripts.verify`, `verification/app.py`)**: the same `AnnotationSession` +
`ReviewController` + loader, plus GUI state (`playing`, `_last_tick`, fps, widget handles). One
session per process.

**CLI / evaluation**: `evaluation.runner._detectors` (one `Detector` per FOV mode, process
lifetime). Out of scope for the migration (`evaluation/` is not to be touched).

**Browser (`localStorage`)**: `recentFolders`, `lastFolder` (`home.html:101-128`), `pointSize`,
`labels` (`annotate.html:286, 399-402`). Per browser, view preferences only. The session id is
only in the URL (`/annotate?sid=...`).

### 2.2 On local disk

All project-relative paths resolve against `verification.paths.RESULTS_DIR = <project>/results`
(absolute, independent of the working directory).

| Path | Written by | Read by | Format / notes |
|---|---|---|---|
| input point clouds: `data/raw/<split>/velodyne/<seq>/*.bin` (KITTI tree) or **any folder path on the server's disk** | – (read-only) | `SequenceLoader`, `FolderSequence`, `web` folder open (`web/manager.py` `open_folder`) | float32 x, y, z, intensity; calibration `data/raw/<split>/calib/<seq>.txt`, GT `label_02/<seq>.txt` |
| `results/sessions/<name>[_f<a>-<b>]_<mode>.json` | `AnnotationSession.save` (`verification/session.py:299-312`), autosave after every changing action (`web/service.py` `_autosave`, `verification/app.py` `_save`), final save on finish / window close / server shutdown (`scripts/web.py:49-52`) | `scripts.verify.build_session` (resume), `scripts.export_labels` | JSON v1: `version, sequence, mode, frame_offset, num_frames, next_track_id, frames{frame: [annotation]}, events[]`. **Atomic**: write `<file>.tmp`, then `replace` |
| `results/<name>.json`, `results/<name>_f<a>-<b>.json`, and `..._c<key>.json` | `scripts.track_sequence.save_results` (from `verify.build_session` and `track_sequence` CLI) | `scripts.verify._find_cached`, `track_sequence` CLI | detection + tracking cache: `{start, stop, frames[[box]], meta}`; `meta` = settings (`verification/cache.py`); not atomic (plain `write_text`) |
| `results/<seq>_detections.json` (+ `_c<key>`) | `evaluation.runner.load_or_detect` | same | detection-only cache (evaluation) |
| `results/<seq>_metrics.json`, `results/<name>_metrics.json` | `scripts.evaluate_sequence`, `scripts.evaluate_all` | humans | metrics |
| `results/exports/<sequence>_<format>/` | `scripts.export_labels` (`export.write_files`) | humans | label files; the **web export writes nothing to disk** (zip built in memory and streamed) |
| `checkpoints/pointpillar_7728.pth`, `OpenPCDet/tools/cfgs/**` | – | `Detector` | weights and configs |
| `results/E*/`, `results/E0/cache/` | `experiments/` | `experiments/`, notebooks | out of scope |

Names: `<name>` is the KITTI sequence id (`0000`) or `folder_key(folder)` = `<folder name>_<sha1 of
the absolute path>[:6]` (`ingestion/folder.py:46-51`). The session file name does not contain the
split, so `training/0000` and `testing/0000` share a session file.

## 3. Background threads and other concurrency

| Where | What | Synchronisation |
|---|---|---|
| `web/manager.py:64` | `threading.Thread(target=self._build, daemon=True)` per opened folder: builds the session (resume from disk, or read the cache, or **run detection + tracking on the GPU**) | `_lock` only around `by_path`; `background=False` runs it inline (tests) |
| uvicorn | sync FastAPI endpoints run in the AnyIO thread pool, so requests of different clients run concurrently | `ReviewService._lock` (`RLock`) serialises `state`, `points`, `act`, `export`, `close` of one service |
| `scripts/app.py:18-21` | child **process** per verification window; the launcher polls it on Open3D's tick event | at most one window at a time (`folder_busy`) |
| Open3D GUI (`verification/app.py`, `launcher.py`) | single GUI thread; playback via `set_on_tick_event`, no extra threads | – |
| `scripts/track_sequence.py` player | Open3D animation callback | – |

No thread pool, timer or scheduler beyond these. A detection job cannot be cancelled; it runs to
the end (or fails, then it is logged and removed). Several jobs can run detection at the same time
(one model per job on the same GPU).

## 4. HTTP endpoints (`web/server.py`)

All JSON errors use FastAPI's default `{"detail": "<message>"}`; request-body validation errors
are FastAPI's `422 {"detail": [...]}`. No authentication, no CORS configuration, no versioning,
no pagination. Default bind `127.0.0.1:8000` (`scripts/web.py`).

| Method, path | Request | Success response | Errors |
|---|---|---|---|
| `GET /` | – | `home.html` | – |
| `GET /annotate?sid=` | – (the page reads `sid` itself) | `annotate.html` | – |
| `POST /api/folder/open` | `{"path": str}` (server-side folder path) | `200 {"sid": str}` (job started) | `400` folder missing / no `.bin` / unreadable first frame |
| `GET /api/jobs/{sid}` | – | `{"sid", "status": "loading"\|"ready"\|"error", "message", "title"}` | `404` unknown or failed-and-removed job |
| `GET /api/s/{sid}/state` | – | state object (below) | `404` unknown sid; `409` not ready, refused, or finished |
| `GET /api/s/{sid}/points/{frame}` | – | `application/octet-stream`, little-endian float32 `(N, 3)` x y z | `404` unknown sid / frame out of range; `409` as above |
| `POST /api/s/{sid}/action` | `{"name": str, "frame"?: int, "index"?: int, "label"?: str}` | state object | `400` unknown action, missing/invalid field; `404`; `409` finished |
| `GET /api/s/{sid}/export/preview` | query `format=lidar\|kitti`, `include_unreviewed`, `include_empty` | `{"available": bool, "reason": str\|null, "summary": ExportSummary\|null}` | `400` unknown format; `404`; `409` |
| `GET /api/s/{sid}/export` | same query | `application/zip`, `Content-Disposition: attachment; filename="<seq>_<format>_labels.zip"` | `400` unknown format or format not possible (`ExportError`); `404`; `409` |
| `POST /api/s/{sid}/finish` | – | `{"ok": true}` (saved, service closed, job removed) | `404`; `409` |

**State object** (`ReviewService.state`, `web/service.py:88-110`): `sequence, mode, frame,
num_frames, selected, current_class, classes, show_rejected, stats{pending, accepted, rejected,
edited, manual, interpolated, total, reviewed, reviewed_fraction}, frame_marks[num_frames]
(0 no boxes, 1 to review, 2 reviewed), save_error, annotations[{i, x, y, z, dx, dy, dz, heading,
label, score, track_id, status}]` (only the visible boxes of the current frame).

**ExportSummary** (`export/formats.py:36-47`): `format, frames, boxes, per_class{class: n},
skipped_unreviewed, skipped_empty, skipped_outside_view`.

**Action names** (`web/service.py:33-65` + `act`): `goto(frame)`, `select(index|null)`,
`set_class(label)`, `next`, `prev`, `next_pending`, `select_next`, `select_prev`, `accept`,
`reject`, `accept_track`, `reject_track`, `undo`, `redo`, `add`, `copy_next`, `toggle_rejected`,
`forward`, `back`, `left`, `right`, `up`, `down`, `rotate_left`, `rotate_right`, `length_minus`,
`length_plus`, `width_minus`, `width_plus`, `height_minus`, `height_plus`, `play_step`. Every
action operates on the **server-side** current frame and selection of the service.

## 5. Facts that matter for the migration

1. **The API is stateful by design.** Frame, selection, class and the undo/redo history live in
   the API process (`ReviewController`, `AnnotationSession._undo/_redo`). Actions are relative to
   that state (`accept` = accept the server's selected box). A stateless API needs either the
   client to send frame/selection with every action, or that state persisted per user/session.
2. **`revision` is not persisted**; it restarts at 0 after every load. Optimistic locking needs a
   persisted revision.
3. **Undo/redo history is not persisted** (only the annotations and the event log are).
4. **Sequences are server-side folder paths**, not uploads. Nothing in the code uploads point
   clouds; the browser only names a path on the server's disk.
5. **Detection runs inside the API process** and loads the model once per job; pcdet's config is
   process-global.
6. Cache files and session files share `results/`; the session file is written atomically, cache
   files are not.
7. Tests: system tests use a fake builder for the web layer (`tests/test_web/test_manager.py`);
   GPU tests are skipped by a CUDA check at module level (`tests/test_detection/test_detector.py`),
   not by a pytest marker; `tests/test_detection/test_fov.py` has one test that needs torch + OpenPCDet
   + KITTI data.
