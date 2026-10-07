import math

import numpy as np
import pytest

from common.boxes import EDGES, BoundingBox3D


def make(**kw):
    base = dict(x=1.0, y=2.0, z=-1.0, dx=4.0, dy=2.0, dz=1.5, heading=0.0, label="Car")
    base.update(kw)
    return BoundingBox3D(**base)


def test_defaults():
    b = make()
    assert b.score == 1.0 and b.track_id is None


@pytest.mark.parametrize("field", ["dx", "dy", "dz"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_dimension_rejected(field, value):
    with pytest.raises(ValueError):
        make(**{field: value})


@pytest.mark.parametrize("field", ["x", "y", "z", "heading"])
def test_non_finite_rejected(field):
    with pytest.raises(ValueError):
        make(**{field: float("nan")})


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_score_range(score):
    with pytest.raises(ValueError):
        make(score=score)


def test_corners_axis_aligned():
    c = make(x=0, y=0, z=0, dx=4, dy=2, dz=1).corners
    assert c.shape == (8, 3)
    assert np.allclose(c.min(axis=0), [-2, -1, -0.5])
    assert np.allclose(c.max(axis=0), [2, 1, 0.5])
    assert np.allclose(c.mean(axis=0), [0, 0, 0])


def test_corners_rotated_90deg_swaps_extents():
    c = make(x=0, y=0, z=0, dx=4, dy=2, dz=1, heading=math.pi / 2).corners
    assert np.allclose(c.max(axis=0), [1, 2, 0.5], atol=1e-9)


def test_front_face_follows_heading():
    b = make(x=0, y=0, z=0, heading=math.pi / 2)
    front = b.corners[[0, 1, 4, 5]].mean(axis=0)
    assert np.allclose(front, [0, 2, 0], atol=1e-9)  # +y when heading = 90 deg


def test_corners_preserve_edge_lengths():
    b = make(heading=0.7)
    c = b.corners
    assert np.isclose(np.linalg.norm(c[0] - c[3]), b.dx)
    assert np.isclose(np.linalg.norm(c[0] - c[1]), b.dy)
    assert np.isclose(np.linalg.norm(c[0] - c[4]), b.dz)
    assert len(EDGES) == 12


def test_array_roundtrip():
    b = make(heading=0.3)
    b2 = BoundingBox3D.from_array(b.to_array(), label="Car")
    assert np.allclose(b.to_array(), b2.to_array())


def test_from_array_wrong_size():
    with pytest.raises(ValueError):
        BoundingBox3D.from_array([1, 2, 3], label="Car")


def test_with_track_id_does_not_mutate():
    b = make()
    b2 = b.with_track_id(5)
    assert b.track_id is None and b2.track_id == 5


def test_volume():
    assert make(dx=2, dy=3, dz=4).volume == 24