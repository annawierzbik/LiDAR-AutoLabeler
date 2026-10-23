"""OpenPCDet PointPillars wrapped behind a minimal interface.

    detector = Detector()
    boxes = detector.predict(frame)      # List[BoundingBox3D] in LiDAR frame

IMPORTANT: ``torch`` must be imported before the compiled pcdet ops (Windows DLL
loading), which is why it is the first third-party import below.
"""
from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from common.boxes import BoundingBox3D
from ingestion.frame import Frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = PROJECT_ROOT / "OpenPCDet"
DEFAULT_CFG = OPENPCDET_ROOT / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "pointpillar_7728.pth"

if str(OPENPCDET_ROOT) not in sys.path:  # allows use without `pip install -e`
    sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


@contextlib.contextmanager
def _chdir(path: Path) -> Iterator[None]:
    """OpenPCDet resolves _BASE_CONFIG_ relative to the cwd (tools/)."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class _InMemoryDataset(DatasetTemplate):
    """Feeds one in-memory point cloud through OpenPCDet's own preprocessing
    (range crop, voxelisation) so inference matches the reference pipeline."""

    def __init__(self, dataset_cfg, class_names, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=None,
            logger=logger,
        )
        self._points: np.ndarray | None = None

    def set_points(self, points: np.ndarray) -> None:
        self._points = points

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index):
        return self.prepare_data(data_dict={"points": self._points, "frame_id": 0})


class Detector:
    """PointPillars (Car / Pedestrian / Cyclist) trained on KITTI 3D detection."""

    def __init__(
        self,
        cfg_file: Path = DEFAULT_CFG,
        checkpoint: Path = DEFAULT_CKPT,
        score_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        cfg_file, checkpoint = Path(cfg_file), Path(checkpoint)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config not found: {cfg_file}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

        self.device = device
        self.score_threshold = score_threshold
        self.logger = common_utils.create_logger()

        with _chdir(cfg_file.parents[2]):  # .../tools
            cfg_from_yaml_file(str(cfg_file), cfg)
        cfg.MODEL.POST_PROCESSING.SCORE_THRESH = score_threshold
        self.class_names: list[str] = list(cfg.CLASS_NAMES)

        self.dataset = _InMemoryDataset(cfg.DATA_CONFIG, self.class_names, self.logger)
        self.model = build_network(
            model_cfg=cfg.MODEL, num_class=len(self.class_names), dataset=self.dataset
        )
        self.model.load_params_from_file(
            filename=str(checkpoint), logger=self.logger, to_cpu=(device == "cpu")
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, frame: Frame) -> list[BoundingBox3D]:
        """Detect objects in one frame. Boxes are in the LiDAR frame."""
        points = np.ascontiguousarray(frame.points[:, :4], dtype=np.float32)
        if points.shape[1] < 4:
            raise ValueError("PointPillars needs (x, y, z, intensity) points")

        self.dataset.set_points(points)
        data_dict = self.dataset.collate_batch([self.dataset[0]])
        if self.device == "cuda":
            load_data_to_gpu(data_dict)
        pred_dicts, _ = self.model.forward(data_dict)
        pred = pred_dicts[0]

        boxes = pred["pred_boxes"].cpu().numpy()
        scores = pred["pred_scores"].cpu().numpy()
        labels = pred["pred_labels"].cpu().numpy()  # 1-based class ids

        return [
            BoundingBox3D.from_array(
                b,
                label=self.class_names[int(l) - 1],
                score=float(min(max(s, 0.0), 1.0)),
            )
            for b, s, l in zip(boxes, scores, labels)
            if s >= self.score_threshold
        ]