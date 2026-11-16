# 0013 Append-only view and playback events; playback counts as active time

Status: accepted (phase 2). Deliberate change of the active-time measure.

## Decision
- Events that change no annotation are appended to the event log and never change the revision:
  `view` (logged as `frame_change`), `playback_start`, `playback_stop`, `playback_tick`. They
  therefore never conflict with an edit (see 0010, optimistic locking).
- The page logs `playback_start` when playback starts, `playback_tick` every 10 s while it runs,
  and `playback_stop` when it stops (by the user or at the last frame). The frames shown during
  playback are still not logged one by one.
- `AnnotationSession.active_seconds` is unchanged: the sum of the gaps between consecutive events,
  ignoring gaps longer than 30 s.

## Effect on the measure
| 60 s of playback, no other action | events | active time |
|---|---|---|
| before (up to commit 07c8da4) | 0 | 0 s |
| now | start + 5 ticks + stop | 60 s |

Watching the sequence is part of reviewing it, so it now counts, as long as the heartbeat keeps the
gaps under 30 s. Active times of sessions recorded before this change are not comparable with later
ones when playback was used; reports that compare them must say so or exclude playback periods
(the events make playback periods identifiable from now on).
