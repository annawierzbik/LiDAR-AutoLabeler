"""Integration tests with the real model. Marked `gpu`: deselected by default (pytest.ini), run with

    pytest -m gpu

They need CUDA, OpenPCDet, the model-zoo checkpoint and the KITTI tracking data under data/raw.
torch / pcdet are imported inside the fixtures, so collecting this module stays cheap.
"""
from pathlib import Path

import pytest

from common.boxes import BoundingBox3D

pytestmark = pytest.mark.gpu

DATA = Path("data/raw")


@pytest.fixture(scope="module")
def Detector():
    from detection.detector import DEFAULT_CKPT, Detector

    if not DEFAULT_CKPT.exists():
        pytest.skip("checkpoint missing")
    return Detector


@pytest.fixture(scope="module")
def detector(Detector):
    return Detector(score_threshold=0.5)


@pytest.fixture(scope="module")
def frame():
    from ingestion.sequence import SequenceLoader

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


def test_threshold_monotonic(Detector, frame):
    low = Detector(score_threshold=0.3).predict(frame)
    high = Detector(score_threshold=0.8).predict(frame)
    assert len(high) <= len(low)


def test_missing_checkpoint_raises(Detector):
    with pytest.raises(FileNotFoundError):
        Detector(checkpoint=Path("nope.pth"))
