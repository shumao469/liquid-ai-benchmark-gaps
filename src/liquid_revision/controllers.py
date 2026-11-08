"""Frozen, non-learning controllers used to create and benchmark trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PDObserverConfig:
    kp: float = 3.5
    kd: float = 1.4
    mass: float = 0.20
    observation_alpha: float = 0.85
    velocity_decay: float = 0.995
    action_limit: float = 1.0

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, Any], *, action_limit: float = 1.0
    ) -> "PDObserverConfig":
        values = dict(mapping)
        values.setdefault("action_limit", action_limit)
        config = cls(**values)
        if config.kp <= 0 or config.kd < 0 or config.mass <= 0:
            raise ValueError("PD gains and mass must be physically valid")
        if not 0.0 <= config.observation_alpha <= 1.0:
            raise ValueError("observation_alpha must be in [0, 1]")
        if not 0.0 <= config.velocity_decay <= 1.0:
            raise ValueError("velocity_decay must be in [0, 1]")
        if config.action_limit <= 0:
            raise ValueError("action_limit must be positive")
        return config


class FrozenPDObserver:
    """A fixed-gain state observer and PD controller with no parameter updates."""

    def __init__(self, config: PDObserverConfig) -> None:
        self.config = config
        self._position = np.zeros(2, dtype=np.float64)
        self._velocity = np.zeros(2, dtype=np.float64)
        self._initialized = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {"type": "frozen_pd_observer", "trainable": False, **asdict(self.config)}

    def reset(self, initial_clean_observation: np.ndarray | None = None) -> None:
        self._position[:] = 0.0
        self._velocity[:] = 0.0
        self._initialized = False
        if initial_clean_observation is not None:
            observation = np.asarray(initial_clean_observation, dtype=np.float64)
            if observation.shape != (10,):
                raise ValueError("initial_clean_observation must have shape (10,)")
            self._position[:] = observation[:2]
            self._velocity[:] = observation[2:4]
            self._initialized = True

    def act(
        self,
        measured_observation: np.ndarray,
        sensor_mask: np.ndarray,
        observed_dt: float,
    ) -> np.ndarray:
        estimated_state = self.estimate(
            measured_observation, sensor_mask, observed_dt
        )
        return self.control(estimated_state, measured_observation)

    def estimate(
        self,
        measured_observation: np.ndarray,
        sensor_mask: np.ndarray,
        observed_dt: float,
    ) -> np.ndarray:
        obs = np.asarray(measured_observation, dtype=np.float64)
        mask = np.asarray(sensor_mask, dtype=bool)
        if obs.shape != (10,) or mask.shape != (4,):
            raise ValueError("Invalid observer input shapes")
        if observed_dt <= 0 or not np.isfinite(observed_dt):
            raise ValueError("observed_dt must be finite and positive")

        if not self._initialized:
            self._position[mask[:2]] = obs[:2][mask[:2]]
            self._velocity[mask[2:4]] = obs[2:4][mask[2:4]]
            self._initialized = True

        self._position += self._velocity * observed_dt
        self._velocity *= self.config.velocity_decay
        alpha = self.config.observation_alpha
        for index in range(2):
            if mask[index]:
                self._position[index] = (
                    alpha * obs[index] + (1.0 - alpha) * self._position[index]
                )
            if mask[index + 2]:
                self._velocity[index] = (
                    alpha * obs[index + 2] + (1.0 - alpha) * self._velocity[index]
                )

        return np.concatenate([self._position, self._velocity]).astype(np.float32)

    def control(
        self, estimated_state: np.ndarray, measured_observation: np.ndarray
    ) -> np.ndarray:
        state = np.asarray(estimated_state, dtype=np.float64)
        obs = np.asarray(measured_observation, dtype=np.float64)
        if state.shape != (4,) or obs.shape != (10,):
            raise ValueError("Point-mass state/observation shapes must be (4,) and (10,)")
        position = state[:2]
        velocity = state[2:]

        reference_position = obs[4:6]
        reference_velocity = obs[6:8]
        reference_acceleration = obs[8:10]
        action = (
            self.config.kp * (reference_position - position)
            + self.config.kd * (reference_velocity - velocity)
            + self.config.mass * reference_acceleration
        )
        return np.clip(
            action, -self.config.action_limit, self.config.action_limit
        ).astype(np.float32)


@dataclass(frozen=True)
class ReacherPDObserverConfig:
    joint_kp: float = 10.0
    joint_kd: float = 1.0
    actuator_gear: float = 200.0
    observation_alpha: float = 0.85
    velocity_decay: float = 0.995
    action_limit: float = 1.0

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, Any], *, action_limit: float = 1.0
    ) -> "ReacherPDObserverConfig":
        values = dict(mapping)
        # Point-mass gains are not silently reinterpreted as joint-space gains.
        for point_mass_key in ("kp", "kd", "mass"):
            values.pop(point_mass_key, None)
        values.setdefault("action_limit", action_limit)
        config = cls(**values)
        if config.joint_kp <= 0 or config.joint_kd < 0:
            raise ValueError("Reacher PD gains must be non-negative and kp positive")
        if config.actuator_gear <= 0:
            raise ValueError("actuator_gear must be positive")
        if not 0.0 <= config.observation_alpha <= 1.0:
            raise ValueError("observation_alpha must be in [0, 1]")
        if not 0.0 <= config.velocity_decay <= 1.0:
            raise ValueError("velocity_decay must be in [0, 1]")
        if config.action_limit <= 0:
            raise ValueError("action_limit must be positive")
        return config


class FrozenReacherPDObserver:
    """Fixed state observer plus shared inverse-kinematics joint-space PD.

    Every learned temporal model replaces only :meth:`estimate`.  Its estimated
    ``[q1, q2, dq1, dq2]`` is then passed to the same :meth:`control` method,
    isolating temporal representation from controller design.
    """

    def __init__(self, config: ReacherPDObserverConfig) -> None:
        self.config = config
        self._position = np.zeros(2, dtype=np.float64)
        self._velocity = np.zeros(2, dtype=np.float64)
        self._initialized = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "type": "frozen_reacher_ik_pd_observer",
            "trainable": False,
            **asdict(self.config),
        }

    @staticmethod
    def _wrap(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def reset(self, initial_clean_observation: np.ndarray | None = None) -> None:
        self._position[:] = 0.0
        self._velocity[:] = 0.0
        self._initialized = False
        if initial_clean_observation is not None:
            observation = np.asarray(initial_clean_observation, dtype=np.float64)
            if observation.shape != (8,):
                raise ValueError("initial_clean_observation must have shape (8,)")
            self._position[:] = observation[:2]
            self._velocity[:] = observation[2:4]
            self._initialized = True

    def estimate(
        self,
        measured_observation: np.ndarray,
        sensor_mask: np.ndarray,
        observed_dt: float,
    ) -> np.ndarray:
        obs = np.asarray(measured_observation, dtype=np.float64)
        mask = np.asarray(sensor_mask, dtype=bool)
        if obs.shape != (8,) or mask.shape != (4,):
            raise ValueError("Invalid Reacher observer input shapes")
        if observed_dt <= 0 or not np.isfinite(observed_dt):
            raise ValueError("observed_dt must be finite and positive")
        if not self._initialized:
            self._position[mask[:2]] = obs[:2][mask[:2]]
            self._velocity[mask[2:4]] = obs[2:4][mask[2:4]]
            self._initialized = True

        self._position = self._wrap(self._position + self._velocity * observed_dt)
        self._velocity *= self.config.velocity_decay
        alpha = self.config.observation_alpha
        for index in range(2):
            if mask[index]:
                innovation = self._wrap(
                    np.asarray([obs[index] - self._position[index]])
                )[0]
                self._position[index] = self._wrap(
                    np.asarray([self._position[index] + alpha * innovation])
                )[0]
            if mask[index + 2]:
                self._velocity[index] = (
                    alpha * obs[index + 2] + (1.0 - alpha) * self._velocity[index]
                )
        return np.concatenate([self._position, self._velocity]).astype(np.float32)

    def control(
        self, estimated_state: np.ndarray, measured_observation: np.ndarray
    ) -> np.ndarray:
        state = np.asarray(estimated_state, dtype=np.float64)
        obs = np.asarray(measured_observation, dtype=np.float64)
        if state.shape != (4,) or obs.shape != (8,):
            raise ValueError("Reacher state/observation shapes must be (4,) and (8,)")
        desired_position = obs[4:6]
        desired_velocity = obs[6:8]
        position_error = self._wrap(desired_position - state[:2])
        joint_torque = (
            self.config.joint_kp * position_error
            + self.config.joint_kd * (desired_velocity - state[2:])
        )
        # Reacher-v5's XML maps normalized actuator control to joint torque with
        # a gear ratio of 200.  Gains are specified in joint-torque units; this
        # conversion prevents environment-specific actuator scaling from being
        # mistaken for controller saturation.
        action = joint_torque / self.config.actuator_gear
        return np.clip(
            action, -self.config.action_limit, self.config.action_limit
        ).astype(np.float32)

    def act(
        self,
        measured_observation: np.ndarray,
        sensor_mask: np.ndarray,
        observed_dt: float,
    ) -> np.ndarray:
        estimated_state = self.estimate(
            measured_observation, sensor_mask, observed_dt
        )
        return self.control(estimated_state, measured_observation)

