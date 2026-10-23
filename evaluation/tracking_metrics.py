"""Tracking metrics (CLEAR MOT): MOTA, MOTP, ID switches, fragmentation, MT/ML.

Usage:
    ev = TrackingEvaluator()
    for each frame, in order:
        ev.update(gt_boxes, ignore_boxes, tracked_boxes)   # tracked boxes carry track_id
    ev.summary()

Matching follows CLEAR MOT: a GT object stays matched to the track it had in the
previous frame while their IoU stays above the threshold; remaining pairs are matched
greedily by IoU. A change of the matched track id for one GT object is an ID switch.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from common.boxes import BoundingBox3D
from common.geometry import iou_matrix


class TrackingEvaluator:
    def __init__(self, match_iou: float = 0.3, ignore_iou: float = 0.3) -> None:
        self.match_iou = match_iou
        self.ignore_iou = ignore_iou
        self.frame = 0
        self.num_gt = self.fn = self.fp = self.idsw = self.frag = self.num_matches = 0
        self.iou_sum = 0.0
        self.prev_match: dict[int, int] = {}          # gt id -> track id in previous frame
        self.last_track_of: dict[int, int] = {}       # gt id -> last track id ever assigned
        self.last_frame_tracked: dict[int, int] = {}  # gt id -> last frame with a match
        self.gt_lifetime: dict[int, int] = defaultdict(int)
        self.gt_tracked: dict[int, int] = defaultdict(int)

    def update(
        self,
        gt: Sequence[BoundingBox3D],
        ignore: Sequence[BoundingBox3D],
        tracks: Sequence[BoundingBox3D],
    ) -> None:
        self.frame += 1
        gt, tracks = list(gt), list(tracks)
        self.num_gt += len(gt)
        for g in gt:
            self.gt_lifetime[g.track_id] += 1

        iou = iou_matrix(gt, tracks) if gt and tracks else np.zeros((len(gt), len(tracks)))
        for i, g in enumerate(gt):  # matches only between equal classes
            for j, t in enumerate(tracks):
                if g.label != t.label:
                    iou[i, j] = 0.0

        matches: dict[int, int] = {}  # gt index -> track index
        used_tracks = set()

        # 1. keep matches from the previous frame while still valid
        for i, g in enumerate(gt):
            prev_tid = self.prev_match.get(g.track_id)
            if prev_tid is None:
                continue
            for j, t in enumerate(tracks):
                if t.track_id == prev_tid and j not in used_tracks and iou[i, j] >= self.match_iou:
                    matches[i] = j
                    used_tracks.add(j)
                    break

        # 2. greedy by IoU for the rest
        candidates = sorted(
            ((iou[i, j], i, j)
             for i in range(len(gt)) if i not in matches
             for j in range(len(tracks)) if j not in used_tracks and iou[i, j] >= self.match_iou),
            reverse=True,
        )
        for _, i, j in candidates:
            if i not in matches and j not in used_tracks:
                matches[i] = j
                used_tracks.add(j)

        # 3. bookkeeping
        new_prev: dict[int, int] = {}
        for i, j in matches.items():
            gid, tid = gt[i].track_id, tracks[j].track_id
            if gid in self.last_track_of and self.last_track_of[gid] != tid:
                self.idsw += 1
            if gid in self.last_frame_tracked and self.last_frame_tracked[gid] < self.frame - 1:
                self.frag += 1
            self.last_track_of[gid] = tid
            self.last_frame_tracked[gid] = self.frame
            self.gt_tracked[gid] += 1
            new_prev[gid] = tid
            self.iou_sum += float(iou[i, j])
            self.num_matches += 1
        self.prev_match = new_prev

        self.fn += len(gt) - len(matches)
        unmatched = [j for j in range(len(tracks)) if j not in used_tracks]
        if unmatched and ignore:
            iou_ign = iou_matrix([tracks[j] for j in unmatched], list(ignore))
            unmatched = [j for k, j in enumerate(unmatched) if iou_ign[k].max() < self.ignore_iou]
        self.fp += len(unmatched)

    def summary(self) -> dict:
        mota = 1.0 - (self.fn + self.fp + self.idsw) / self.num_gt if self.num_gt else 0.0
        ratios = [self.gt_tracked[g] / n for g, n in self.gt_lifetime.items()]
        return {
            "frames": self.frame,
            "gt_objects": self.num_gt,
            "gt_tracks": len(self.gt_lifetime),
            "num_matches": self.num_matches,
            "iou_sum": self.iou_sum,
            "MOTA": mota,
            "MOTP_mean_iou": self.iou_sum / self.num_matches if self.num_matches else 0.0,
            "recall": self.num_matches / self.num_gt if self.num_gt else 0.0,
            "precision": (self.num_matches / (self.num_matches + self.fp)
                          if (self.num_matches + self.fp) else 0.0),
            "false_negatives": self.fn,
            "false_positives": self.fp,
            "id_switches": self.idsw,
            "fragmentations": self.frag,
            "mostly_tracked": sum(1 for r in ratios if r >= 0.8),
            "mostly_lost": sum(1 for r in ratios if r <= 0.2),
        }

def pool_tracking_summaries(summaries: Sequence[dict]) -> dict:
    """Combine per-sequence summaries into one (sums raw counts, recomputes the ratios).

    Averaging per-sequence MOTA values would over-weight short sequences, so the
    counts are added up first and the metrics are computed from the totals.
    """
    keys = ("frames", "gt_objects", "gt_tracks", "num_matches", "iou_sum", "false_negatives",
            "false_positives", "id_switches", "fragmentations", "mostly_tracked", "mostly_lost")
    t = {k: sum(s[k] for s in summaries) for k in keys}
    gt, m, fp = t["gt_objects"], t["num_matches"], t["false_positives"]
    t["MOTA"] = 1.0 - (t["false_negatives"] + fp + t["id_switches"]) / gt if gt else 0.0
    t["MOTP_mean_iou"] = t["iou_sum"] / m if m else 0.0
    t["recall"] = m / gt if gt else 0.0
    t["precision"] = m / (m + fp) if (m + fp) else 0.0
    return t