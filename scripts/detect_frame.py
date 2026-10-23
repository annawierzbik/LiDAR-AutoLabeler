"""Run PointPillars on one KITTI frame and preview the result in Open3D.

    python -m scripts.detect_frame --sequence 0000 --frame 0 --threshold 0.5

Colors: Car = green, Pedestrian = cyan, Cyclist = yellow. Points are colored by height.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from common.boxes import EDGES, BoundingBox3D
from detection.detector import Detector  # imports torch first
from ingestion.sequence import SequenceLoader

CLASS_COLORS = {
    "Car": [0.0, 1.0, 0.0],
    "Pedestrian": [0.0, 1.0, 1.0],
    "Cyclist": [1.0, 1.0, 0.0],
}


def height_colors(xyz: np.ndarray) -> np.ndarray:
    z = xyz[:, 2]
    t = np.clip((z - (-2.5)) / (1.5 - (-2.5)), 0.0, 1.0)  # approx LiDAR z range
    # blue (low) -> purple -> orange/white (high)
    return np.stack([0.2 + 0.8 * t, 0.3 + 0.5 * t, 1.0 - 0.7 * t], axis=1)


def box_lineset(box: BoundingBox3D) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(box.corners)
    ls.lines = o3d.utility.Vector2iVector(EDGES)
    color = CLASS_COLORS.get(box.label, [1.0, 0.0, 0.0])
    ls.colors = o3d.utility.Vector3dVector([color] * len(EDGES))
    return ls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    seq = SequenceLoader(Path(args.data_root), args.split, args.sequence)
    frame = seq[args.frame]

    detector = Detector(score_threshold=args.threshold)
    boxes = detector.predict(frame)

    print(f"Frame {frame.frame_id}: {frame.num_points} points, {len(boxes)} detections")
    for b in boxes:
        print(f"  {b.label:<10} score={b.score:.2f} center=({b.x:.1f}, {b.y:.1f}, {b.z:.2f}) "
              f"size=({b.dx:.2f}, {b.dy:.2f}, {b.dz:.2f}) heading={b.heading:.2f}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame.xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(height_colors(frame.xyz))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"{args.sequence}/{frame.frame_id}", width=1280, height=800)
    vis.add_geometry(pcd)
    for b in boxes:
        vis.add_geometry(box_lineset(b))
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()