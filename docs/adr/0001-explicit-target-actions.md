# 0001 Explicit-target actions; view state in the browser

Status: accepted (phase 1)

## Context
`ReviewService` wrapped a `ReviewController` whose current frame, selected box, class for new boxes
and "show rejected" flag lived in the API process. Actions were relative to that state (`accept` =
accept the server's selected box), so every tab of a session shared one cursor, and the API could
not run as several stateless replicas (phase 4).

## Decision
- Every web action names its target: `frame`, plus `index` (box actions), `track_id` (track
  actions) or `label` (`add`, `set_label`). Nothing on the server depends on a current frame or
  selection.
- View state (frame, selection, class for new boxes, show rejected) lives in `annotate.html`.
  Selection, frame stepping and playback are browser-only; playback reads `GET state?frame=`.
- `GET /api/s/<sid>/state?frame=N` returns all boxes of frame N (rejected ones included, each
  with its index `i`); the browser filters. `GET /api/s/<sid>/next_pending?frame=N` replaces the
  `next_pending` action.
- Action `view(frame)` only writes a `frame_change` event, so the event log used to measure
  annotation time keeps recording human navigation; playback does not log (as before).
- Responses are the state of the frame the action concerns plus `result` (e.g. `{frame, index}`
  of a new or copied box, `{count}` of a track action).
- The desktop window keeps `ReviewController` unchanged; the web service reuses its editing
  rules (nudge, resize, add, copy) by pointing a throw-away controller at the named box.

## Consequences
- The action API changed shape (no `select`, `goto`, `next`, `prev`, `select_next`,
  `select_prev`, `toggle_rejected`, `set_class`, `play_step`; new `view`, `set_label`).
- "Accept with nothing selected selects the first box to review" and the auto-advance after a
  decision are implemented in the browser (they were in the controller before).
- Box indices are per frame and can shift (rejecting a manual box removes it); the browser
  re-reads the frame from every response.
