"""Annotator-effort model for E3: how many correction operations turn predicted tracks into the GT.

A proxy, not time: every operation has a configurable weight (default 1), and the ratio
sum(weight * count) / n_manual_boxes compares that to drawing every GT box from scratch.

Thresholds (per class, config key `effort.thresholds`):
    tau_fix   the official KITTI 3D IoU threshold of the class (Car 0.7, Pedestrian/Cyclist 0.5).
              A matched box with IoU < tau_fix would not count as correct in KITTI evaluation,
              so the annotator has to fix it.
    tau_match a looser threshold used only to associate boxes with GT tracks (Car 0.5,
              Pedestrian/Cyclist 0.25). A box with tau_match <= IoU < tau_fix is the right object
              in a slightly wrong place: one fix, not a delete plus a new box.
tau_fix > tau_match is required for every class; with tau_fix <= tau_match no matched box could
ever be below tau_fix and n_fix_boxes would be 0 by construction.

Matching, per sequence:
1. Boxes, per frame: predicted and GT boxes of the same class are matched one-to-one, greedily
   by 3D IoU (highest first), keeping pairs with IoU >= tau_match of that class.
2. Tracks: each predicted track is assigned to the GT track with which it has the most frame
   matches from step 1 (ties: higher IoU sum, then lower GT track id). The assignment is
   independent per predicted track, so several predicted tracks may map to the same GT track
   (many-to-one); it is not one-to-one. A predicted track with no frame match is a false track.
3. Counts:
   n_delete         predicted tracks assigned to no GT track (one operation per track);
   n_add_tracks     GT tracks no predicted track is assigned to (the whole track is drawn);
   n_missing_boxes  boxes of the other GT tracks that are not matched in step 1 by a box of a
                    predicted track assigned to them (gaps; the box is drawn);
   n_fix_boxes      boxes of those GT tracks matched by an assigned track with IoU < tau_fix;
   n_id_switches    per GT track, in frame order, the number of times the assigned predicted
                    track covering it changes (a track split into two pieces is 1 switch, and
                    neither piece is a delete);
   n_manual_boxes, n_gt_tracks  all GT boxes and tracks (the cost of labelling from scratch).
Predicted boxes of an assigned track in frames where they match nothing are not counted.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from common.boxes import BoundingBox3D
from common.geometry import iou_matrix

DEFAULT_THRESHOLDS = {
    "Car": {"tau_match": 0.5, "tau_fix": 0.7},
    "Pedestrian": {"tau_match": 0.25, "tau_fix": 0.5},
    "Cyclist": {"tau_match": 0.25, "tau_fix": 0.5},
}
OPERATIONS = ("n_delete", "n_add_tracks", "n_missing_boxes", "n_fix_boxes", "n_id_switches")
DEFAULT_WEIGHTS = dict.fromkeys(OPERATIONS, 1.0)


def validate_thresholds(thresholds: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Checked copy of per-class {tau_match, tau_fix}; ValueError unless 0 < tau_match < tau_fix <= 1."""
    out = {}
    for cls, t in thresholds.items():
        if set(t) != {"tau_match", "tau_fix"}:
            raise ValueError(f"{cls}: expected keys tau_match and tau_fix, got {sorted(t)}")
        m, f = float(t["tau_match"]), float(t["tau_fix"])
        if not 0.0 < m < f <= 1.0:
            raise ValueError(f"{cls}: need 0 < tau_match < tau_fix <= 1, got tau_match={m}, tau_fix={f}")
        out[cls] = {"tau_match": m, "tau_fix": f}
    return out


def _match_frame(gt: Sequence[BoundingBox3D], pred: Sequence[BoundingBox3D], th) -> list[tuple[int, int, float]]:
    """One-to-one (gt index, pred index, IoU) pairs of equal class with IoU >= tau_match."""
    if not gt or not pred:
        return []
    iou = iou_matrix(gt, pred)
    cand = sorted(((iou[i, j], i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)
                   if g.label == p.label and iou[i, j] >= th[g.label]["tau_match"]), reverse=True)
    pairs, used_g, used_p = [], set(), set()
    for v, i, j in cand:
        if i not in used_g and j not in used_p:
            pairs.append((i, j, float(v)))
            used_g.add(i)
            used_p.add(j)
    return pairs


def effort_counts(
    gt_frames: Sequence[Sequence[BoundingBox3D]],
    pred_frames: Sequence[Sequence[BoundingBox3D]],
    thresholds: Mapping[str, Mapping[str, float]] | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict:
    """Effort counts for one sequence. Boxes carry track_id; frames are aligned by index."""
    if len(gt_frames) != len(pred_frames):
        raise ValueError(f"{len(gt_frames)} GT frames but {len(pred_frames)} predicted frames")
    th = validate_thresholds(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    unknown = {b.label for f in gt_frames for b in f} - set(th)
    if unknown:
        raise ValueError(f"no tau_match/tau_fix for GT classes {sorted(unknown)}")

    gt_boxes = defaultdict(list)                         # gt tid -> [frame]
    pred_tracks = set()
    frame_matches = []                                   # per frame: {gt tid: (pred tid, iou, cls)}
    overlap = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))   # pred tid -> gt tid -> [n, iou sum]
    for k, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
        for g in gt:
            gt_boxes[g.track_id].append(k)
        pred_tracks.update(p.track_id for p in pred)
        fm = {}
        for i, j, v in _match_frame(gt, pred, th):
            g, p = gt[i], pred[j]
            fm[g.track_id] = (p.track_id, v, g.label)
            overlap[p.track_id][g.track_id][0] += 1
            overlap[p.track_id][g.track_id][1] += v
        frame_matches.append(fm)

    assigned = {pt: min(ov.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))[0]
                for pt, ov in overlap.items()}
    covered_gt = set(assigned.values())

    n_missing = n_fix = n_idsw = 0
    for gid in covered_gt:
        last = None
        for k in gt_boxes[gid]:
            m = frame_matches[k].get(gid)
            if m is None or assigned.get(m[0]) != gid:
                n_missing += 1
                continue
            pt, v, cls = m
            if v < th[cls]["tau_fix"]:
                n_fix += 1
            if last is not None and pt != last:
                n_idsw += 1
            last = pt

    counts = {
        "n_delete": len(pred_tracks - set(assigned)),
        "n_add_tracks": len(set(gt_boxes) - covered_gt),
        "n_missing_boxes": n_missing,
        "n_fix_boxes": n_fix,
        "n_id_switches": n_idsw,
        "n_manual_boxes": sum(len(v) for v in gt_boxes.values()),
        "n_gt_tracks": len(gt_boxes),
        "n_pred_tracks": len(pred_tracks),
    }
    return {**counts, **_ratio(counts, w), "weights": w, "thresholds": th}


def _ratio(counts: Mapping[str, float], w: Mapping[str, float]) -> dict:
    cost = sum(w[op] * counts[op] for op in OPERATIONS)
    return {"effort_cost": cost,
            "effort_ratio": cost / counts["n_manual_boxes"] if counts["n_manual_boxes"] else 0.0}


def pool_effort(per_sequence: Sequence[Mapping]) -> dict:
    """Sum the counts of several sequences and recompute the ratio (weights must be equal)."""
    if not per_sequence:
        raise ValueError("no sequences")
    w = per_sequence[0]["weights"]
    if any(s["weights"] != w for s in per_sequence):
        raise ValueError("sequences were scored with different weights")
    keys = (*OPERATIONS, "n_manual_boxes", "n_gt_tracks", "n_pred_tracks")
    t = {k: sum(s[k] for s in per_sequence) for k in keys}
    return {**t, **_ratio(t, w), "weights": w}
