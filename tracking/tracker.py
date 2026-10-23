"""Multi-object tracker: Kalman prediction + Hungarian assignment on 3D IoU.

    tracker = Tracker()
    for frame in sequence:
        boxes = detector.predict(frame)
        tracked = tracker.update(boxes)      # same boxes, with stable track_id set
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.boxes import BoundingBox3D
from common.geometry import iou_matrix
from tracking.kalman import BoxKalmanFilter


@dataclass
class Track:
    track_id: int
    label: str
    kf: BoxKalmanFilter
    dims: np.ndarray  # smoothed (dx, dy, dz)
    score: float
    hits: int = 1
    misses: int = 0  # consecutive frames without a match

    @classmethod
    def from_box(cls, box: BoundingBox3D, track_id: int) -> "Track":
        return cls(
            track_id=track_id,
            label=box.label,
            kf=BoxKalmanFilter(np.array([box.x, box.y, box.z, box.heading])),
            dims=box.dimensions.copy(),
            score=box.score,
        )

    def to_box(self) -> BoundingBox3D:
        x, y, z = self.kf.position
        dx, dy, dz = self.dims
        return BoundingBox3D(
            x=float(x), y=float(y), z=float(z),
            dx=float(dx), dy=float(dy), dz=float(dz),
            heading=self.kf.heading,
            label=self.label,
            score=float(min(max(self.score, 0.0), 1.0)),
            track_id=self.track_id,
        )

    def predict(self) -> BoundingBox3D:
        self.kf.predict()
        return self.to_box()

    def update(self, box: BoundingBox3D, dim_alpha: float = 0.3) -> None:
        self.kf.update(np.array([box.x, box.y, box.z, box.heading]))
        self.dims = (1 - dim_alpha) * self.dims + dim_alpha * box.dimensions
        self.score = box.score
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(
        self,
        iou_threshold: float = 0.1,
        max_age: int = 3,
        min_hits: int = 3,
    ) -> None:
        """
        iou_threshold: minimum 3D IoU between a prediction and a detection to match.
        max_age:       frames a track survives without detections before deletion.
        min_hits:      detections needed before a track is reported (filters
                       one-frame false positives). During the first `min_hits`
                       frames of the sequence all tracks are reported.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.reset()

    def reset(self) -> None:
        self.tracks: List[Track] = []
        self._next_id = 0
        self.frame_count = 0

    def update(self, detections: List[BoundingBox3D]) -> List[BoundingBox3D]:
        """Process one frame; returns confirmed boxes with `track_id` set."""
        self.frame_count += 1

        # 1. predict all existing tracks to the current frame
        predicted = [t.predict() for t in self.tracks]

        # 2. associate (only boxes of the same class may match)
        matches, unmatched_tracks, unmatched_dets = self._associate(predicted, detections)

        # 3. update matched, age unmatched, create new
        for ti, di in matches:
            self.tracks[ti].update(detections[di])
        for ti in unmatched_tracks:
            self.tracks[ti].misses += 1
        for di in unmatched_dets:
            self.tracks.append(Track.from_box(detections[di], self._next_id))
            self._next_id += 1

        # 4. drop dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # 5. report tracks that were matched this frame and are confirmed
        out = []
        for t in self.tracks:
            confirmed = t.hits >= self.min_hits or self.frame_count <= self.min_hits
            if t.misses == 0 and confirmed:
                out.append(t.to_box())
        return out

    def _associate(self, predicted, detections):
        if not predicted or not detections:
            return [], list(range(len(predicted))), list(range(len(detections)))

        iou = iou_matrix(predicted, detections)
        for i, t in enumerate(self.tracks):  # forbid cross-class matches
            for j, d in enumerate(detections):
                if t.label != d.label:
                    iou[i, j] = 0.0

        rows, cols = linear_sum_assignment(-iou)  # maximise total IoU
        matches, matched_t, matched_d = [], set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.iou_threshold:
                matches.append((r, c))
                matched_t.add(r)
                matched_d.add(c)

        unmatched_tracks = [i for i in range(len(predicted)) if i not in matched_t]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_d]
        return matches, unmatched_tracks, unmatched_dets