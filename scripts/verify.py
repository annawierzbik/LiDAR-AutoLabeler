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

from ingestion.sequence import SequenceLoader
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


def _find_cached(name: str, want: range):
    """Cached detections that cover the frames in `want`: (indices, results) sliced to `want`, or None.
    Looks at results/<name>.json (whole run) and results/<name>_f<start>-<stop>.json (partial runs)."""
    from scripts.track_sequence import load_results

    folder = RESULTS_DIR
    candidates = [folder / f"{name}.json"] + sorted(folder.glob(f"{name}_f*-*.json"))
    for path in candidates:
        if not path.exists():
            continue
        indices, results = load_results(path)
        if indices.start <= want.start and want.stop <= indices.stop:
            offset = want.start - indices.start
            return want, results[offset: offset + len(want)]
    return None


def build_session(args):
    """Load (resume) or create the session for `args`. Returns (session, sequence loader, session path).

    --start/--stop select a part of the sequence. In manual mode no detector is run at all,
    so you can start drawing immediately (and it works without a GPU)."""
    if args.folder:
        from ingestion.folder import FolderSequence, folder_key

        seq = FolderSequence(args.folder)
        name = folder_key(args.folder)   # unique per folder, used for cache + session file names
    else:
        seq = SequenceLoader(Path(args.data_root), args.split, args.sequence)
        name = args.sequence
    session_path = session_file(name, args.mode, args.start, args.stop)

    if session_path.exists() and not args.fresh:
        session = AnnotationSession.load(session_path)
        print(f"Resumed session {session_path} ({session.stats()['reviewed']} boxes reviewed)")
        return session, seq, session_path

    stop = min(args.stop if args.stop is not None else len(seq), len(seq))
    want = range(args.start, stop)
    if len(want) == 0:
        raise ValueError(f"Empty frame range {args.start}..{args.stop} (the sequence has {len(seq)} frames)")

    if args.mode == "manual":
        indices, results = want, [[] for _ in want]
    else:
        found = _find_cached(name, want)
        if found is not None:
            indices, results = found
        else:
            from scripts.track_sequence import save_results, track_sequence

            indices = want
            print(f"Detecting + tracking frames {indices.start}..{indices.stop - 1}")
            results = track_sequence(seq, indices, args.threshold)
            whole = indices.start == 0 and indices.stop == len(seq)
            cache = RESULTS_DIR / (f"{name}.json" if whole else f"{name}_f{indices.start}-{indices.stop}.json")
            save_results(cache, indices, results)
    session = AnnotationSession.from_tracking(
        name, results, args.mode, frame_offset=indices.start)
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
