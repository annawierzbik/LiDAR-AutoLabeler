"""Open the verification UI for one KITTI sequence.

    python -m scripts.verify --sequence 0000                       # assisted: review system boxes
    python -m scripts.verify --sequence 0000 --mode manual         # manual: draw every box yourself
    python -m scripts.verify --folder D:\\clouds\\drive1             # any folder of KITTI-style .bin frames

Boxes come from results/<seq>.json (written by scripts.track_sequence). If it does not exist,
detection + tracking run first. Progress is autosaved to
results/sessions/<seq>_<mode>.json and resumed automatically
(use --fresh to start over).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from detection.fov import add_fov_argument, sequence_mode
from ingestion.sequence import SequenceLoader
from verification import cache
from verification.paths import RESULTS_DIR, session_file
from verification.session import AnnotationSession


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--folder", default=None,
                   help="any folder with KITTI-style .bin frames (overrides --data-root/--split/--sequence)")
    p.add_argument("--mode", choices=["assisted", "manual"], default="assisted")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None, help="exclusive; default = whole sequence")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--fresh", action="store_true", help="ignore a saved session and start over")
    p.add_argument("--use-cache-without-weights", action="store_true",
                   help="use matching cached detections even though the checkpoint is missing on disk")
    add_fov_argument(p)


def file_names(args) -> tuple[str, str, str | None]:
    """(sequence label, file key, legacy file key) of the sequence in `args`.

    The file key names the session and cache files: a folder's folder_key, or <split>_<sequence>
    for a KITTI sequence (training/0000 and testing/0000 must not share files). Files written
    before the split was part of the key are named <sequence> only; they are still read for the
    split "training" (the default then), which is the legacy key. The sequence label is what the
    session stores as its sequence (used for export file names) and does not change."""
    if args.folder:
        from ingestion.folder import folder_key

        key = folder_key(args.folder)   # unique per folder
        return key, key, None
    return args.sequence, f"{args.split}_{args.sequence}", args.sequence if args.split == "training" else None


def _cache_candidates(name: str) -> list[Path]:
    folder = RESULTS_DIR
    return [folder / f"{name}.json"] + sorted(
        p for p in folder.glob(f"{name}_*.json") if cache.is_cache_name(p.stem, name))


def _find_cached(name: str, want: range, meta: dict, allow_missing_weights: bool = False,
                 legacy_name: str | None = None):
    """Cached detections that cover the frames in `want` and were made with the settings `meta`:
    (indices, results) sliced to `want`, or None. Looks at results/<name>.json (whole run) and
    results/<name>_f<start>-<stop>.json (partial runs), each optionally with a _c<key> suffix
    (verification/cache.py: files with other settings are not reused, files without settings are,
    with a warning), then at the same names with `legacy_name`."""
    from scripts.track_sequence import parse_results

    candidates = _cache_candidates(name) + (_cache_candidates(legacy_name) if legacy_name else [])
    found = cache.choose(candidates, meta,
                         usable=lambda d: d["start"] <= want.start and want.stop <= d["stop"],
                         allow_missing_weights=allow_missing_weights)
    if found is None:
        return None
    indices, results = parse_results(found[1])
    offset = want.start - indices.start
    return want, results[offset: offset + len(want)]


def build_session(args):
    """Load (resume) or create the session for `args`. Returns (session, sequence loader, session path).

    --start/--stop select a part of the sequence. In manual mode no detector is run at all,
    so you can start drawing immediately (and it works without a GPU)."""
    if args.folder:
        from ingestion.folder import FolderSequence

        seq = FolderSequence(args.folder)
    else:
        seq = SequenceLoader(Path(args.data_root), args.split, args.sequence)
    label, name, legacy = file_names(args)
    session_path = session_file(name, args.mode, args.start, args.stop)

    if not args.fresh:
        legacy_path = session_file(legacy, args.mode, args.start, args.stop) if legacy else None
        source = session_path if session_path.exists() else (
            legacy_path if legacy_path is not None and legacy_path.exists() else None)
        if source is not None:
            session = AnnotationSession.load(source)
            moved = "" if source == session_path else f" (from the old file name; now saved as {session_path.name})"
            print(f"Resumed session {source}{moved} ({session.stats()['reviewed']} boxes reviewed)")
            return session, seq, session_path

    stop = min(args.stop if args.stop is not None else len(seq), len(seq))
    want = range(args.start, stop)
    if len(want) == 0:
        raise ValueError(f"Empty frame range {args.start}..{args.stop} (the sequence has {len(seq)} frames)")

    if args.mode == "manual":
        indices, results = want, [[] for _ in want]
    else:
        from scripts.track_sequence import current_meta

        split = None if args.folder else args.split
        meta = current_meta(args.threshold, sequence_mode(seq, args.fov), split)
        found = _find_cached(name, want, meta, args.use_cache_without_weights, legacy)
        if found is not None:
            indices, results = found
        else:
            from scripts.track_sequence import save_results, track_sequence

            indices = want
            print(f"Detecting + tracking frames {indices.start}..{indices.stop - 1}")
            results = track_sequence(seq, indices, args.threshold, args.fov)
            whole = indices.start == 0 and indices.stop == len(seq)
            path = RESULTS_DIR / (f"{name}.json" if whole else f"{name}_f{indices.start}-{indices.stop}.json")
            save_results(cache.output_path(path, meta), indices, results, meta)
    session = AnnotationSession.from_tracking(
        label, results, args.mode, frame_offset=indices.start)
    print(f"New {args.mode} session with {session.stats()['total']} system boxes in {len(indices)} frames")
    return session, seq, session_path


def main() -> None:
    p = argparse.ArgumentParser()
    add_arguments(p)
    args = p.parse_args()
    session, seq, session_path = build_session(args)

    from verification.app import VerificationApp  # imports open3d GUI only when needed

    VerificationApp(session, seq, session_path, fps=args.fps).run()

    st = session.stats()
    print(f"\nSession closed. Boxes: {st}")
    print(f"Active time: {session.active_seconds() / 60:.1f} min   saved to {session_path}")


if __name__ == "__main__":
    main()
