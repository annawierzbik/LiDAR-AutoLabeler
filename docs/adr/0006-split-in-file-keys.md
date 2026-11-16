# 0006 KITTI split in session and cache file names

Status: accepted (phase 1)

## Context
Session and cache files of a KITTI sequence were named after the sequence id only (`0000`), so
`training/0000` and `testing/0000` shared them.

## Decision
- File key = `<split>_<sequence>` for KITTI sequences (`training_0000_assisted.json`,
  `training_0000.json`); folders keep their `folder_key`.
- The session's `sequence` field stays the plain sequence id: export file names
  (`label_02/0000.txt`) and archive names must not change.
- Fallback: when the new file does not exist and the split is `training` (the only split the old
  code was used with by default), the old name (`0000_...`) is read. A resumed session is saved
  under the new name from then on; the old file is left untouched. A newer file under the new name
  wins over an old one.
- Cache lookups try the new names first, then the old ones (training only); old cache files without
  metadata are still subject to the rules of `verification/cache.py`.
