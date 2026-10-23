"""Console tables for evaluation results."""
from __future__ import annotations


def print_detection(summary: dict) -> None:
    print(f"\n=== DETECTION (score >= {summary['score_threshold']} for P/R, AP over all scores) ===")
    print(f"{'class':<12}{'AP40':>8}{'precision':>11}{'recall':>9}{'mean IoU':>10}{'#GT':>7}")
    for cls in ("Car", "Pedestrian", "Cyclist"):
        s = summary[cls]
        if s["num_gt"] == 0:
            print(f"{cls:<12}{'n/a':>8}{'n/a':>11}{'n/a':>9}{'n/a':>10}{0:>7}")
            continue
        print(f"{cls:<12}{s['AP40']*100:>7.1f}%{s['precision']*100:>10.1f}%"
              f"{s['recall']*100:>8.1f}%{s['mean_iou_of_matches']:>10.2f}{s['num_gt']:>7}")
    print(f"{'mAP40':<12}{summary['mAP40']*100:>7.1f}%")


def print_tracking(s: dict, title: str = "TRACKING (CLEAR MOT)") -> None:
    print(f"\n=== {title} ===")
    print(f"frames {s['frames']} | GT objects {s['gt_objects']} | GT tracks {s['gt_tracks']}")
    print(f"MOTA {s['MOTA']*100:.1f}%   MOTP (mean IoU) {s['MOTP_mean_iou']:.2f}")
    print(f"recall {s['recall']*100:.1f}%   precision {s['precision']*100:.1f}%")
    print(f"FN {s['false_negatives']}   FP {s['false_positives']}   "
          f"ID switches {s['id_switches']}   fragmentations {s['fragmentations']}")
    print(f"mostly tracked {s['mostly_tracked']}/{s['gt_tracks']}   "
          f"mostly lost {s['mostly_lost']}/{s['gt_tracks']}")


def print_per_sequence(rows: dict[str, dict]) -> None:
    print("\n=== PER SEQUENCE (tracking) ===")
    print(f"{'seq':<6}{'frames':>7}{'GT obj':>8}{'GT trk':>8}{'MOTA':>8}{'recall':>8}{'prec':>7}{'IDsw':>6}{'frag':>6}")
    for seq, s in rows.items():
        print(f"{seq:<6}{s['frames']:>7}{s['gt_objects']:>8}{s['gt_tracks']:>8}"
              f"{s['MOTA']*100:>7.1f}%{s['recall']*100:>7.1f}%{s['precision']*100:>6.1f}%"
              f"{s['id_switches']:>6}{s['fragmentations']:>6}")