"""Default detector config and weights (no torch import, so cheap to use anywhere)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = PROJECT_ROOT / "OpenPCDet"
DEFAULT_CFG = OPENPCDET_ROOT / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "pointpillar_7728.pth"
