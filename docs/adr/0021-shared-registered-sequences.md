# 0021 Sharing a registered sequence among several annotators (draft)

Status: proposed — not implemented; needs a decision (review after phase 4, item 8)

## Context
A sequence has one `owner_id`; only that user (and admins) can see it and open it, and
`POST /api/v1/tabs` builds the session of the *sequence's owner* (an admin who opens a user's
sequence works on that user's session, ADR 0017). Several annotators working on the same
registered folder today need one registration per annotator: separate rows that only share the
detection cache (its key is the folder's key, so detection still runs once). Nothing shows that
those rows are the same data, and an admin must repeat every registration.

## Options
1. **Keep one row per annotator** (today). No change; the cost is repeated registrations and no
   overview.
2. **Registered sequences visible to every annotator.** Simple, but no control over who works on
   what, and every annotator sees every folder.
3. **Access grants (proposed).** A table `sequence_access(sequence_id, user_id, granted_by,
   created_at)`, primary key `(sequence_id, user_id)`. An admin grants or revokes access to a
   registered sequence (`POST` / `DELETE /api/v1/sequences/<id>/access/<user_id>`). Uploads stay
   private to the uploader (no grants), unless decided otherwise.

## What option 3 changes in the ownership checks
- **Sequence visibility**: owner, or a grant, or admin (`SequenceStore.get` / `page` join the
  grants). Others still get 404, as for any resource they cannot see.
- **Whose session**: `POST /api/v1/tabs` builds the session of the *requesting user*
  (`owner = user.id`), not of the sequence owner; session keys stay `u<user id>_...`, so every
  annotator has their own session of the shared sequence. An admin opening a sequence gets their
  own session too; looking at an annotator's work stays possible through that annotator's
  sessions and tabs (read-only by the lock, ADR 0017). If admins must keep working *on behalf of* a
  user, that becomes an explicit parameter instead of the implicit rule of today.
- **Sessions, tabs and jobs** stay strictly per user (`owner_id`): a grant on the sequence gives
  no access to another annotator's session, tab or export (still 404). Locks stay per
  (user, client).
- **Revoking** a grant hides the sequence from that user; their session stays (admin-visible)
  and opens again if access is granted again. Deleting a registered sequence (unregistering,
  admin only) is refused while any user's sessions exist, as for uploads (`force` deletes them).
- **Detection** runs once per folder and model: the cache key does not depend on the user, and
  concurrent openings of different users are not merged (they have different payloads), so two
  annotators opening a new folder at the same moment may still both run detection once each.
- **Tests**: `tests/test_api/test_ownership.py` gains a granted user (sees the sequence, opens
  their own session, gets 404 for the other annotator's tab, job, session and export) and a
  revoked user (404 for the sequence).

## Open questions for the decision
- Grants on uploads too, or registered folders only?
- Should an admin's opening work on their own session (proposed) or on a chosen user's?
- Is a combined view of several annotators' sessions of one sequence needed (comparison,
  consensus)? That is a separate decision; this draft only lets them work side by side.
