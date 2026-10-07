"""Geometry helpers shared by tracking, evaluation and verification.

All boxes follow the BoundingBox3D convention (LiDAR frame, z up, center-based).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Sequence

import numpy as np

if TYPE_CHECKING:  # avoid a circular import at runtime (boxes imports this module)
    from common.boxes import BoundingBox3D


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _ccw(poly: np.ndarray) -> np.ndarray:
    """Return the polygon with counter-clockwise vertex order."""
    x, y = poly[:, 0], poly[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if signed_area >= 0 else poly[::-1]


def _clip_polygon(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip `subject` by the convex CCW polygon `clipper`."""
    output = [p for p in subject]
    for i in range(len(clipper)):
        if not output:
            break
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a

        def inside(p) -> bool:
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12

        def intersect(p, q):
            d = q - p
            denom = edge[0] * d[1] - edge[1] * d[0]
            if abs(denom) < 1e-12:
                return q
            t = (edge[0] * (a[1] - p[1]) - edge[1] * (a[0] - p[0])) / denom
            return p + t * d

        inputs, output = output, []
        prev = inputs[-1]
        for cur in inputs:
            if inside(cur):
                if not inside(prev):
                    output.append(intersect(prev, cur))
                output.append(cur)
            elif inside(prev):
                output.append(intersect(prev, cur))
            prev = cur
    return np.array(output) if output else np.empty((0, 2))


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def bev_polygon(box: "BoundingBox3D") -> np.ndarray:
    """(4, 2) bird's-eye-view footprint, counter-clockwise."""
    return _ccw(box.corners[:4, :2])


def bev_iou(a: "BoundingBox3D", b: "BoundingBox3D") -> float:
    """IoU of the rotated footprints seen from above."""
    pa, pb = bev_polygon(a), bev_polygon(b)
    inter = _polygon_area(_clip_polygon(pa, pb))
    union = _polygon_area(pa) + _polygon_area(pb) - inter
    return inter / union if union > 0 else 0.0


def iou_3d(a: "BoundingBox3D", b: "BoundingBox3D") -> float:
    """Exact 3D IoU for boxes rotated only around the vertical axis."""
    pa, pb = bev_polygon(a), bev_polygon(b)
    inter_area = _polygon_area(_clip_polygon(pa, pb))
    z_overlap = max(
        0.0,
        min(a.z + a.dz / 2, b.z + b.dz / 2) - max(a.z - a.dz / 2, b.z - b.dz / 2),
    )
    inter = inter_area * z_overlap
    union = a.volume + b.volume - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(
    boxes_a: Sequence["BoundingBox3D"], boxes_b: Sequence["BoundingBox3D"]
) -> np.ndarray:
    """(len(a), len(b)) matrix of 3D IoU values."""
    m = np.zeros((len(boxes_a), len(boxes_b)))
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            # cheap reject: centers further apart than the diagonals
            if np.linalg.norm(a.center[:2] - b.center[:2]) > (
                math.hypot(a.dx, a.dy) + math.hypot(b.dx, b.dy)
            ) / 2:
                continue
            m[i, j] = iou_3d(a, b)
    return m