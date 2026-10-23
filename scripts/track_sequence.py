"""Detect + track a whole KITTI sequence and play it back in Open3D.

    python -m scripts.track_sequence --sequence 0000
    python -m scripts.track_sequence --sequence 0000 --start 0 --stop 60 --fps 10

Detection is the slow part, so tracked results are cached to a JSON file
(results/<sequence>.json by default). Next runs load the cache instantly;
use --no-cache to recompute.

Playback keys:  SPACE pause/resume   LEFT/RIGHT step (when paused)   R restart   Q quit
Each track keeps one color for its whole life, so ID switches are visible as color flips.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common.boxes import EDGES, BoundingBox3D
from ingestion.sequence import SequenceLoader
from tracking.tracker import Tracker
from verification.paths import RESULTS_DIR

KEY_SPACE, KEY_R, KEY_RIGHT, KEY_LEFT = 32, 82, 262, 263


# ----------------------------------------------------------------- computation
def track_sequence(seq: SequenceLoader, indices: range, threshold: float) -> list[list[BoundingBox3D]]:
    from detection.detector import (
        Detector,  # heavy import (torch + pcdet), only when needed
    )

    detector = Detector(score_threshold=threshold)
    tracker = Tracker()
    results = []
    t0 = time.time()
    for n, i in enumerate(indices, 1):
        results.append(tracker.update(detector.predict(seq[i])))
        if n % 10 == 0 or n == len(indices):
            print(f"  {n}/{len(indices)} frames ({(time.time() - t0) / n:.2f} s/frame)")
    return results


def save_results(path: Path, indices: range, results: list[list[BoundingBox3D]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "start": indices.start, "stop": indices.stop,
        "frames": [[asdict(b) for b in boxes] for boxes in results],
    }
    path.write_text(json.dumps(data))


def load_results(path: Path):
    data = json.loads(path.read_text())
    indices = range(data["start"], data["stop"])
    results = [[BoundingBox3D(**b) for b in boxes] for boxes in data["frames"]]
    return indices, results


def print_stats(results: list[list[BoundingBox3D]]) -> None:
    lengths: Counter = Counter()
    labels: dict[int, str] = {}
    for boxes in results:
        for b in boxes:
            lengths[b.track_id] += 1
            labels[b.track_id] = b.label
    if not lengths:
        print("No tracks.")
        return
    per_class = Counter(labels.values())
    print(f"\nTracks: {len(lengths)}  ({dict(per_class)})")
    print(f"Mean track length: {np.mean(list(lengths.values())):.1f} frames, "
          f"longest: {max(lengths.values())}, single-frame tracks: "
          f"{sum(1 for v in lengths.values() if v == 1)}")


# ---------------------------------------------------------------- visualization
def id_color(track_id: int) -> list[float]:
    """Stable, well separated color per track id (golden-ratio hue walk)."""
    hue = (track_id * 0.61803398875) % 1.0
    return list(colorsys.hsv_to_rgb(hue, 0.85, 1.0))


def height_colors(xyz: np.ndarray) -> np.ndarray:
    t = np.clip((xyz[:, 2] + 2.5) / 4.0, 0.0, 1.0)
    return np.stack([0.2 + 0.8 * t, 0.3 + 0.5 * t, 1.0 - 0.7 * t], axis=1)


class Player:
    """Plays pre-computed tracking results frame by frame."""

    MOVE_STEP = 1.5  # meters per key press for the fly camera

    def __init__(self, seq: SequenceLoader, indices: range, results, fps: float) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.seq, self.indices, self.results = seq, indices, results
        self.fps = fps
        self.pos = 0
        self.playing = True
        self.last_tick = 0.0
        self.box_sets: list = []

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()

    def run(self) -> None:
        vis = self.vis
        vis.create_window("Tracking playback", width=1400, height=850)
        self._load_frame(first=True)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.0, 0.0, 0.0])
        opt.point_size = 2.0
        ctl = vis.get_view_control()
        ctl.set_lookat([15.0, 0.0, 0.0])
        ctl.set_front([-1.0, 0.0, 0.6])
        ctl.set_up([0.0, 0.0, 1.0])
        ctl.set_zoom(0.12)

        # playback
        vis.register_key_callback(KEY_SPACE, lambda v: self._toggle())
        vis.register_key_callback(KEY_R, lambda v: self._seek(0))
        vis.register_key_callback(KEY_RIGHT, lambda v: self._step(1))
        vis.register_key_callback(KEY_LEFT, lambda v: self._step(-1))
        for key in (61, 334):  # '=' and keypad '+'
            vis.register_key_callback(key, lambda v: self._change_speed(1.25))
        for key in (45, 333):  # '-' and keypad '-'
            vis.register_key_callback(key, lambda v: self._change_speed(1 / 1.25))

        # fly camera: (forward, right, up) in meters, camera-local
        moves = {
            ord("W"): (1, 0, 0), ord("S"): (-1, 0, 0),
            ord("D"): (0, 1, 0), ord("A"): (0, -1, 0),
            ord("E"): (0, 0, 1), ord("C"): (0, 0, -1),
        }
        for key, (f, r, u) in moves.items():
            vis.register_key_callback(key, lambda v, f=f, r=r, u=u: self._move(f, r, u))

        vis.register_animation_callback(self._tick)
        vis.run()
        vis.destroy_window()

    # ---- controls
    def _toggle(self) -> bool:
        self.playing = not self.playing
        return False

    def _change_speed(self, factor: float) -> bool:
        self.fps = min(max(self.fps * factor, 0.5), 30.0)
        print(f"\nspeed: {self.fps:.1f} fps")
        return False

    def _move(self, forward: float, right: float, up: float) -> bool:
        s = self.MOVE_STEP
        self.vis.get_view_control().camera_local_translate(forward * s, right * s, up * s)
        return False

    def _seek(self, pos: int) -> bool:
        self.pos = pos % len(self.results)
        self._load_frame()
        return True

    def _step(self, delta: int) -> bool:
        if not self.playing:
            self._seek(self.pos + delta)
        return True

    def _tick(self, vis) -> bool:
        now = time.time()
        if self.playing and now - self.last_tick >= 1.0 / self.fps:
            self.last_tick = now
            self._seek(self.pos + 1)
        return False

    # ---- rendering
    def _load_frame(self, first: bool = False) -> None:
        o3d, vis = self.o3d, self.vis
        frame = self.seq[self.indices[self.pos]]
        self.pcd.points = o3d.utility.Vector3dVector(frame.xyz.astype(np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(height_colors(frame.xyz))
        if first:
            vis.add_geometry(self.pcd)
        else:
            vis.update_geometry(self.pcd)

        for ls in self.box_sets:
            vis.remove_geometry(ls, reset_bounding_box=False)
        self.box_sets = []
        for b in self.results[self.pos]:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(b.corners)
            ls.lines = o3d.utility.Vector2iVector(EDGES)
            ls.colors = o3d.utility.Vector3dVector([id_color(b.track_id)] * len(EDGES))
            vis.add_geometry(ls, reset_bounding_box=False)
            self.box_sets.append(ls)
        print(f"\rframe {frame.frame_id}  objects: "
              + ", ".join(f"#{b.track_id} {b.label}" for b in self.results[self.pos]) + " " * 20,
              end="", flush=True)


# ------------------------------------------------------------------------ main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--split", default="training")
    p.add_argument("--sequence", default="0000")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None, help="exclusive; default = end of sequence")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--cache", default=None, help="default: results/<sequence>.json")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    seq = SequenceLoader(Path(args.data_root), args.split, args.sequence)
    cache = Path(args.cache) if args.cache else RESULTS_DIR / f"{args.sequence}.json"

    if cache.exists() and not args.no_cache:
        print(f"Loading cached results from {cache}")
        indices, results = load_results(cache)
    else:
        stop = min(args.stop if args.stop is not None else len(seq), len(seq))
        indices = range(args.start, stop)
        print(f"Detecting + tracking frames {indices.start}..{indices.stop - 1}")
        results = track_sequence(seq, indices, args.threshold)
        save_results(cache, indices, results)
        print(f"Saved to {cache}")

    print_stats(results)
    print("\nControls: SPACE pause | LEFT/RIGHT step | R restart | =/- speed | WASD+E/C fly | mouse orbit | Q quit")
    Player(seq, indices, results, args.fps).run()


if __name__ == "__main__":
    main()