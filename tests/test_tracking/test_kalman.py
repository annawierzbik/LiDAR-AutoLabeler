import math

import numpy as np
import pytest

from tracking.kalman import BoxKalmanFilter


def test_estimates_constant_velocity():
    kf = BoxKalmanFilter(np.array([0.0, 0.0, 0.0, 0.0]))
    for t in range(1, 30):
        kf.predict()
        kf.update(np.array([1.0 * t, 0.5 * t, 0.0, 0.0]))
    assert kf.velocity[0] == pytest.approx(1.0, abs=0.1)
    assert kf.velocity[1] == pytest.approx(0.5, abs=0.1)


def test_predict_extrapolates_motion():
    kf = BoxKalmanFilter(np.array([0.0, 0.0, 0.0, 0.0]))
    for t in range(1, 30):
        kf.predict()
        kf.update(np.array([1.0 * t, 0.0, 0.0, 0.0]))
    before = kf.position[0]
    kf.predict()
    assert kf.position[0] == pytest.approx(before + 1.0, abs=0.15)


def test_smooths_noise():
    rng = np.random.default_rng(0)
    kf = BoxKalmanFilter(np.array([0.0, 0.0, 0.0, 0.0]))
    raw_err, kf_err = [], []
    for t in range(1, 80):
        truth = np.array([1.0 * t, 0.0, 0.0, 0.0])
        meas = truth + rng.normal(0, 0.3, 4)
        kf.predict()
        kf.update(meas)
        if t > 20:
            raw_err.append(abs(meas[0] - truth[0]))
            kf_err.append(abs(kf.position[0] - truth[0]))
    assert np.mean(kf_err) < np.mean(raw_err)


def test_heading_wraparound_is_handled():
    kf = BoxKalmanFilter(np.array([0.0, 0.0, 0.0, math.pi - 0.05]))
    kf.predict()
    kf.update(np.array([0.0, 0.0, 0.0, -math.pi + 0.05]))  # +0.1 rad across the seam
    assert abs(abs(kf.heading) - math.pi) < 0.1  # stays near +-pi, no swing through 0
    assert -math.pi <= kf.heading < math.pi