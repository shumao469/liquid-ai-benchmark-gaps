"""Moving-target Gymnasium Reacher-v5 wrapper for confirmatory evaluation."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.reacher_v5 import ReacherEnv


class MovingTargetReacherEnv(ReacherEnv):
    """Reacher-v5 with a seeded moving target and fixed-duration episodes.

    The plant, actuator model, and two-link morphology are the official
    Gymnasium Reacher-v5 implementation.  The only task change is that its two
    target slide joints are advanced along a pre-generated Lissajous reference.
    The clean observation is
    ``[q1,q2,dq1,dq2,q1*,q2*,dq1*,dq2*]``.  The visual target is
    positioned at the forward kinematics of ``q*``, so joint-space tracking
    and the reported end-effector error refer to the same moving target.

    Sensor perturbations are deliberately applied outside this class, allowing
    every observer to see the same frozen schedule while the reference remains
    available to the shared controller.
    """

    sensor_dim = 4
    state_dim = 4
    observation_dim = 8

    def __init__(
        self,
        *,
        horizon_steps: int = 500,
        frame_skip: int = 2,
        trajectory_family: str = "lissajous",
        amplitude_range: tuple[float, float] = (0.15, 0.50),
        frequency_range_hz: tuple[float, float] = (0.05, 0.25),
        action_limit: float = 1.0,
        velocity_failure_limit: float = 80.0,
        joint_position_failure_limit: float = 3.0,
        render_mode: str | None = None,
    ) -> None:
        if horizon_steps <= 0 or frame_skip <= 0:
            raise ValueError("horizon_steps and frame_skip must be positive")
        if trajectory_family != "lissajous":
            raise ValueError("Only the pre-registered lissajous family is supported")
        if len(amplitude_range) != 2 or not 0 < amplitude_range[0] <= amplitude_range[1]:
            raise ValueError("amplitude_range must contain ordered positive limits")
        if len(frequency_range_hz) != 2 or not 0 < frequency_range_hz[0] <= frequency_range_hz[1]:
            raise ValueError("frequency_range_hz must contain ordered positive limits")
        if not 0 < action_limit <= 1.0:
            raise ValueError("action_limit must be in (0, 1] for Reacher-v5")
        if velocity_failure_limit <= 0:
            raise ValueError("velocity_failure_limit must be positive")
        if not 0 < joint_position_failure_limit <= math.pi:
            raise ValueError("joint_position_failure_limit must be in (0, pi]")

        self.horizon_steps = int(horizon_steps)
        self.trajectory_family = trajectory_family
        self.amplitude_range = tuple(map(float, amplitude_range))
        self.frequency_range_hz = tuple(map(float, frequency_range_hz))
        self.action_limit = float(action_limit)
        self.velocity_failure_limit = float(velocity_failure_limit)
        self.joint_position_failure_limit = float(joint_position_failure_limit)
        self._step_index = 0
        self._large_error_steps = 0
        self._saturation_steps = 0
        self._external_force = np.zeros(2, dtype=np.float64)
        self._trajectory: dict[str, np.ndarray] | None = None

        super().__init__(
            frame_skip=frame_skip,
            reward_dist_weight=1.0,
            reward_control_weight=0.01,
            render_mode=render_mode,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-self.action_limit,
            high=self.action_limit,
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def control_dt(self) -> float:
        return float(self.dt)

    @property
    def trajectory_parameters(self) -> dict[str, list[float]]:
        if self._trajectory is None:
            raise RuntimeError("Environment must be reset before reading trajectory parameters")
        return {key: value.tolist() for key, value in self._trajectory.items()}

    def _sample_trajectory(self) -> dict[str, np.ndarray]:
        amplitude = self.np_random.uniform(
            self.amplitude_range[0], self.amplitude_range[1], size=2
        )
        frequency = self.np_random.uniform(
            self.frequency_range_hz[0], self.frequency_range_hz[1], size=2
        )
        phase = self.np_random.uniform(-math.pi, math.pi, size=2)
        phase[1] += math.pi / 2.0
        return {
            "amplitude": amplitude.astype(np.float64),
            "frequency_hz": frequency.astype(np.float64),
            "phase": phase.astype(np.float64),
        }

    def reference(self, time_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._trajectory is None:
            raise RuntimeError("Environment must be reset before evaluating the reference")
        amplitude = self._trajectory["amplitude"]
        omega = 2.0 * math.pi * self._trajectory["frequency_hz"]
        angle = omega * float(time_seconds) + self._trajectory["phase"]
        position = amplitude * np.sin(angle)
        velocity = amplitude * omega * np.cos(angle)
        acceleration = -amplitude * (omega**2) * np.sin(angle)
        return position, velocity, acceleration

    @staticmethod
    def _wrapped_angles(qpos: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(qpos), np.cos(qpos))

    def _get_obs(self) -> np.ndarray:
        if self._trajectory is None:
            # ReacherEnv construction does not normally query observations, but
            # retaining this guard makes upstream lifecycle changes fail clearly.
            return np.zeros(8, dtype=np.float64)
        desired_position, desired_velocity, _ = self.reference(
            self._step_index * self.control_dt
        )
        return np.concatenate(
            [
                self._wrapped_angles(np.asarray(self.data.qpos[:2], dtype=np.float64)),
                np.asarray(self.data.qvel[:2], dtype=np.float64),
                desired_position,
                desired_velocity,
            ]
        )

    @staticmethod
    def _target_kinematics(
        desired_position: np.ndarray, desired_velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        l1, l2 = 0.10, 0.11
        q1, q2 = desired_position
        jacobian = np.asarray(
            [
                [
                    -l1 * np.sin(q1) - l2 * np.sin(q1 + q2),
                    -l2 * np.sin(q1 + q2),
                ],
                [
                    l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
                    l2 * np.cos(q1 + q2),
                ],
            ],
            dtype=np.float64,
        )
        target_position = np.asarray(
            [
                l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
                l1 * np.sin(q1) + l2 * np.sin(q1 + q2),
            ],
            dtype=np.float64,
        )
        return target_position, jacobian @ desired_velocity

    def reset_model(self) -> np.ndarray:
        self._step_index = 0
        self._large_error_steps = 0
        self._saturation_steps = 0
        self._external_force[:] = 0.0
        self._trajectory = self._sample_trajectory()
        desired_position, desired_velocity, _ = self.reference(0.0)
        target_position, target_velocity = self._target_kinematics(
            desired_position, desired_velocity
        )
        qpos = self.init_qpos.copy()
        qpos[:2] += self.np_random.uniform(-0.1, 0.1, size=2)
        qpos[-2:] = target_position
        qvel = self.init_qvel.copy()
        qvel[:2] += self.np_random.uniform(-0.005, 0.005, size=2)
        qvel[-2:] = target_velocity
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _advance_target(self) -> None:
        desired_position, desired_velocity, _ = self.reference(
            self._step_index * self.control_dt
        )
        target_position, target_velocity = self._target_kinematics(
            desired_position, desired_velocity
        )
        self.data.qpos[-2:] = target_position
        self.data.qvel[-2:] = target_velocity
        mujoco.mj_forward(self.model, self.data)

    def set_external_force(
        self, generalized_torque: np.ndarray | list[float] | tuple[float, float]
    ) -> None:
        torque = np.asarray(generalized_torque, dtype=np.float64)
        if torque.shape != (2,) or not np.all(np.isfinite(torque)):
            raise ValueError("External generalized torque must be a finite length-2 vector")
        self._external_force = torque.copy()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite length-2 vector")
        clipped = np.clip(action, -self.action_limit, self.action_limit)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[:2] = self._external_force
        self.do_simulation(clipped, self.frame_skip)
        self.data.qfrc_applied[:] = 0.0
        self._external_force[:] = 0.0
        self._step_index += 1
        self._advance_target()

        observation = self._get_obs()
        fingertip = self.get_body_com("fingertip")[:2]
        target = self.get_body_com("target")[:2]
        tracking_error = float(np.linalg.norm(fingertip - target))
        control_effort = float(np.dot(clipped, clipped) * self.control_dt)
        reward = -(tracking_error**2 + 0.01 * control_effort)
        self._large_error_steps = (
            self._large_error_steps + 1 if tracking_error > 0.10 else 0
        )
        saturated = bool(
            np.any(np.abs(clipped) >= self.action_limit - 1e-7)
        )
        self._saturation_steps = self._saturation_steps + 1 if saturated else 0
        finite = bool(np.all(np.isfinite(observation)))
        speed_safe = bool(np.max(np.abs(observation[2:4])) <= self.velocity_failure_limit)
        joint_safe = bool(
            np.max(np.abs(observation[:2])) <= self.joint_position_failure_limit
        )
        prolonged_error = self._large_error_steps >= 25
        prolonged_saturation = self._saturation_steps >= 50
        terminated = not (
            finite
            and speed_safe
            and joint_safe
            and not prolonged_error
            and not prolonged_saturation
        )
        truncated = self._step_index >= self.horizon_steps
        info: dict[str, Any] = {
            "tracking_error": tracking_error,
            "squared_tracking_error": tracking_error**2,
            "control_effort": control_effort,
            "reference_position": observation[4:6].copy(),
            "step_index": self._step_index,
            "failure": bool(terminated),
            "failure_reasons": {
                "nonfinite": not finite,
                "velocity_limit": not speed_safe,
                "joint_position_limit": not joint_safe,
                "error_over_0p10m_for_25_steps": prolonged_error,
                "saturation_for_50_steps": prolonged_saturation,
            },
            "actuator_saturated": saturated,
            "environment_id": "Reacher-v5-moving-target",
        }
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info
