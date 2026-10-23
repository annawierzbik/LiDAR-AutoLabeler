"""Write the verified labels of a saved session to a folder.

    python -m scripts.export_labels --session results/sessions/<file>.json --folder D:\\clouds\\drive1
    python -m scripts.export_labels --session results/sessions/<file>.json --sequence 0000 --format kitti

The sequence arguments only tell the exporter where the point clouds (frame names) and the
calibration are; the boxes come from the session file. In the browser the same export is the
"Export labels" section of the annotation page.

Formats (see export/formats.py):
    lidar   labels/<frame>.txt per frame: x y z dx dy dz heading Class (LiDAR frame, no calibration needed)
    kitti   label_02/<sequence>.txt in KITTI tracking format (needs the calibration of a KITTI sequence)
By default only frames without boxes waiting for a decision are written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from export import FORMATS, ExportError, export_session, write_files
from verification.session import AnnotationSession

EXPORTS_DIR = Path(__file__).resolve().parents[1] / "results" / "exports"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True, help="saved session file (results/sessions/*.json)")
    p.add_argument("--folder", default=None, help="folder with the .bin frames (overrides --data-root/--split/--sequence)")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--format", choices=FORMATS, default="lidar")
    p.add_argument("--out", default=None, help="output folder (default: <project>/results/exports/<sequence>_<format>)")
    p.add_argument("--include-unreviewed", action="store_true",
                   help="also write frames that still have boxes to review, including those boxes")
    p.add_argument("--include-empty", action="store_true",
                   help="also write empty label files for frames without boxes (lidar format)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    session = AnnotationSession.load(args.session)
    if args.folder:
        from ingestion.folder import FolderSequence

        loader = FolderSequence(args.folder)
    else:
        from ingestion.sequence import SequenceLoader

        loader = SequenceLoader(Path(args.data_root), args.split, args.sequence)

    try:
        result = export_session(session, loader, args.format, args.include_unreviewed, args.include_empty)
    except ExportError as exc:
        print(f"Cannot export: {exc}")
        return 1
    out = Path(args.out) if args.out else EXPORTS_DIR / f"{session.sequence}_{args.format}"
    write_files(result.files, out)

    s = result.summary
    classes = ", ".join(f"{n} {c}" for c, n in sorted(s.per_class.items())) or "none"
    print(f"Wrote {s.frames} frames, {s.boxes} boxes ({classes}) to {out}")
    for count, why in ((s.skipped_unreviewed, "frames still have boxes to review"),
                       (s.skipped_empty, "reviewed frames have no boxes"),
                       (s.skipped_outside_view, "boxes lie outside the camera view")):
        if count:
            print(f"  left out: {count} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
