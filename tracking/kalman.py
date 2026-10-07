"""Constant-velocity Kalman filter for one 3D box.

State  s = [x, y, z, heading, vx, vy, vz]
Measurement z = [x, y, z, heading]
Time is measured in frames by default (dt = 1); KITTI runs at 10 Hz.
"""
from __future__ import annotations

import numpy as np

from common.geometry import normalize_angle


class BoxKalmanFilter:
    def __init__(
        self,
        initial: np.ndarray,
        dt: float = 1.0,
        pos_noise: float = 0.5,
        heading_noise: float = 0.3,
        process_pos: float = 0.1,
        process_vel: float = 0.5,
        process_heading: float = 0.05,
    ) -> None:
        """`initial` is [x, y, z, heading]."""
        initial = np.asarray(initial, dtype=np.float64).reshape(4)
        self.x = np.zeros(7)
        self.x[:4] = initial

        self.F = np.eye(7)
        for i in range(3):
            self.F[i, 4 + i] = dt

        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)

        self.P = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0])  # unknown velocity
        self.Q = np.diag(
            [process_pos] * 3 + [process_heading] + [process_vel] * 3
        ).astype(np.float64)
        self.R = np.diag([pos_noise] * 3 + [heading_noise])

    # ----------------------------------------------------------------- accessors
    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def heading(self) -> float:
        return float(self.x[3])

    @property
    def velocity(self) -> np.ndarray:
        return self.x[4:].copy()

    # --------------------------------------------------------------------- steps
    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.x[3] = normalize_angle(self.x[3])
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, measurement: np.ndarray) -> None:
        z = np.asarray(measurement, dtype=np.float64).reshape(4).copy()
        # Use the angle closest to the prediction so +-pi wrap-around is harmless.
        z[3] = self.x[3] + normalize_angle(z[3] - self.x[3])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = normalize_angle(self.x[3])
        self.P = (np.eye(7) - K @ self.H) @ self.P