"""Registered non-learning state estimators used with the shared frozen PD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FullStateEstimator:
    name = "full_state_pd"

    def reset(self) -> None:
        return None

    def estimate(self, clean, measured, mask, dt):
        del measured, mask, dt
        return np.asarray(clean[:4], dtype=np.float32).copy()


class ZeroOrderHoldEstimator:
    name = "zoh_pd"

    def __init__(self) -> None:
        self.state = np.zeros(4, dtype=np.float64)
        self.initialized = np.zeros(4, dtype=bool)

    def reset(self) -> None:
        self.state[:] = 0.0
        self.initialized[:] = False

    def estimate(self, clean, measured, mask, dt):
        del clean, dt
        observed = np.asarray(measured[:4], dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        self.state[valid] = observed[valid]
        self.initialized[valid] = True
        return self.state.astype(np.float32).copy()


@dataclass(frozen=True)
class KalmanConfig:
    process_variance: float = 1e-3
    measurement_variance: float = 2e-3
    initial_variance: float = 1.0


class KalmanEstimator:
    """Causal constant-velocity Kalman filter with partial observations."""

    name = "kalman_pd"

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        if min(
            self.config.process_variance,
            self.config.measurement_variance,
            self.config.initial_variance,
        ) <= 0:
            raise ValueError("Kalman variances must be positive")
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)

    def reset(self) -> None:
        self.state[:] = 0.0
        self.covariance[:] = np.eye(4) * self.config.initial_variance

    @staticmethod
    def _wrap(value: np.ndarray) -> np.ndarray:
        return (value + np.pi) % (2 * np.pi) - np.pi

    def estimate(self, clean, measured, mask, dt):
        del clean
        elapsed = float(dt)
        transition = np.asarray(
            [
                [1.0, 0.0, elapsed, 0.0],
                [0.0, 1.0, 0.0, elapsed],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        q = self.config.process_variance
        process = np.diag([q * elapsed**3, q * elapsed**3, q * elapsed, q * elapsed])
        self.state = transition @ self.state
        self.state[:2] = self._wrap(self.state[:2])
        self.covariance = transition @ self.covariance @ transition.T + process
        valid_indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if len(valid_indices):
            observation = np.asarray(measured[:4], dtype=np.float64)[valid_indices]
            measurement = np.eye(4)[valid_indices]
            innovation = observation - measurement @ self.state
            for row, index in enumerate(valid_indices):
                if index < 2:
                    innovation[row] = self._wrap(np.asarray([innovation[row]]))[0]
            noise = np.eye(len(valid_indices)) * self.config.measurement_variance
            residual_covariance = measurement @ self.covariance @ measurement.T + noise
            gain = np.linalg.solve(
                residual_covariance, measurement @ self.covariance
            ).T
            self.state = self.state + gain @ innovation
            self.state[:2] = self._wrap(self.state[:2])
            identity = np.eye(4)
            self.covariance = (
                (identity - gain @ measurement)
                @ self.covariance
                @ (identity - gain @ measurement).T
                + gain @ noise @ gain.T
            )
        return self.state.astype(np.float32).copy()


def build_nonlearned_baseline(name: str, config: dict | None = None):
    normalized = name.lower()
    if normalized == "full_state_pd":
        return FullStateEstimator()
    if normalized == "zoh_pd":
        return ZeroOrderHoldEstimator()
    if normalized == "kalman_pd":
        return KalmanEstimator(KalmanConfig(**dict(config or {})))
    raise ValueError(f"Unknown non-learning baseline: {name}")
