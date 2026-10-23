"""Filling gaps in a track by interpolating between its key boxes (no session/GUI code here).

A *key box* is a box someone actually placed: detected by the system, accepted, edited or drawn by
the human. The frames between two key boxes of the same track that have no box get an
interpolated one, as long as the hole is at most MAX_GAP frames long (same as the tracker's
max_age: for longer holes a straight line is a guess, so the human decides).
"""
from __future__ import annotations

from collections.abc import Mapping

from common.boxes import BoundingBox3D
from common.geometry import normalize_angle
import itertools

MAX_GAP = 3


def interpolate_box(a: BoundingBox3D, b: BoundingBox3D, t: float) -> BoundingBox3D:
    """Box at fraction `t` (0 = a, 1 = b). Heading goes the short way round; the score is the
    lower of the two so a doubtful neighbour makes the filled box doubtful as well."""
    lerp = lambda p, q: p + (q - p) * t
    return BoundingBox3D(
        lerp(a.x, b.x), lerp(a.y, b.y), lerp(a.z, b.z),
        lerp(a.dx, b.dx), lerp(a.dy, b.dy), lerp(a.dz, b.dz),
        normalize_angle(a.heading + normalize_angle(b.heading - a.heading) * t),
        a.label, min(a.score, b.score), a.track_id,
    )


def fill_plan(key_boxes: Mapping[int, BoundingBox3D], max_gap: int = MAX_GAP) -> dict[int, BoundingBox3D]:
    """`key_boxes`: frame -> key box of ONE track. Returns frame -> interpolated box for every
    empty frame lying in a hole of 1..max_gap frames between two key boxes."""
    frames = sorted(key_boxes)
    plan: dict[int, BoundingBox3D] = {}
    for f1, f2 in itertools.pairwise(frames):
        gap = f2 - f1 - 1
        if not 1 <= gap <= max_gap:
            continue
        for f in range(f1 + 1, f2):
            plan[f] = interpolate_box(key_boxes[f1], key_boxes[f2], (f - f1) / (f2 - f1))
    return plan