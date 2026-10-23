"""Integration tests: skipped automatically without torch/pcdet/GPU/checkpoint/data."""
from pathlib import Path

import pytest

pytest.importorskip("torch")
torch = __import__("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from detection.detector import DEFAULT_CKPT

if not DEFAULT_CKPT.exists():
    pytest.skip("checkpoint missing", allow_module_level=True)

from common.boxes import BoundingBox3D
from detection.detector import Detector
from ingestion.sequence import SequenceLoader

DATA = Path("data/raw")


@pytest.fixture(scope="module")
def detector():
    return Detector(score_threshold=0.5)


@pytest.fixture(scope="module")
def frame():
    if not (DATA / "training" / "velodyne" / "0000").exists():
        pytest.skip("KITTI data missing")
    return SequenceLoader(DATA, "training", "0000")[0]


def test_predict_returns_boxes(detector, frame):
    boxes = detector.predict(frame)
    assert isinstance(boxes, list)
    assert all(isinstance(b, BoundingBox3D) for b in boxes)
    assert len(boxes) > 0


def test_labels_and_scores_valid(detector, frame):
    for b in detector.predict(frame):
        assert b.label in detector.class_names
        assert 0.5 <= b.score <= 1.0


def test_threshold_monotonic(frame):
    low = Detector(score_threshold=0.3).predict(frame)
    high = Detector(score_threshold=0.8).predict(frame)
    assert len(high) <= len(low)


def test_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        Detector(checkpoint=Path("nope.pth"))