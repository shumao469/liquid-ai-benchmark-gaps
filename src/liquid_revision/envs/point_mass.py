"""A deliberately minimal closed-loop MuJoCo trajectory-tracking plant."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.mujoco import MujocoEnv
    from gymnasium.utils import EzPickle
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised by dependency smoke test
    raise ImportError(
        "PointMassTrackingEnv requires gymnasium[mujoco] and mujoco. "
        "Install the project dependencies before importing liquid_revision.envs."
    ) from exc


class PointMassTrackingEnv(MujocoEnv, EzPickle):
    """Track a seeded moving reference with a force-controlled 2-DoF point mass.

    The observation is clean simulator state and reference information:
    ``[q_x, q_y, v_x, v_y, r_x, r_y, rdot_x, rdot_y, rddot_x, rddot_y]``.
    Sensor corruption is kept outside the plant so that every policy receives the
    exact same pre-generated perturbation schedule.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }
    sensor_dim = 4
    state_dim = 4
    observation_dim = 10

    def __init__(
        self,
        *,
        horizon_steps: int = 500,
        frame_skip: int = 2,
        trajectory_family: str = "lissajous",
        amplitude_range: tuple[float, float] = (0.45, 1.10),
        frequency_range_hz: tuple[float, float] = (0.05, 0.25),
        action_limit: float = 1.0,
        position_limit: float = 1.95,
        render_mode: str | None = None,
    ) -> None:
        if horizon_steps <= 0 or frame_skip <= 0:
            raise ValueError("horizon_steps and frame_skip must be positive")
        if trajectory_family != "lissajous":
            raise ValueError("Only the pre-registered lissajous family is supported")
        self.horizon_steps = int(horizon_steps)
        self.trajectory_family = trajectory_family
        self.amplitude_range = tuple(map(float, amplitude_range))
        self.frequency_range_hz = tuple(map(float, frequency_range_hz))
        self.action_limit = float(action_limit)
        self.position_limit = float(position_limit)
        self._step_index = 0
        self._external_force = np.zeros(2, dtype=np.float64)
        self._trajectory: dict[str, np.ndarray] | None = None

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
        )
        model_path = Path(__file__).resolve().parents[1] / "assets" / "point_mass_2d.xml"
        EzPickle.__init__(
            self,
            horizon_steps=horizon_steps,
            frame_skip=frame_skip,
            trajectory_family=trajectory_family,
            amplitude_range=amplitude_range,
            frequency_range_hz=frequency_range_hz,
            action_limit=action_limit,
            position_limit=position_limit,
            render_mode=render_mode,
        )
        MujocoEnv.__init__(
            self,
            str(model_path),
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config={"distance": 4.5, "elevation": -90.0},
        )
        self.action_space = spaces.Box(
            low=-self.action_limit,
            high=self.action_limit,
            shape=(2,),
            dtype=np.float32,
        )
        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(round(1.0 / self.dt))

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

    def _update_target_site(self) -> None:
        position, _, _ = self.reference(self._step_index * self.control_dt)
        self.model.site_pos[self._target_site_id, 0:2] = position
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        position, velocity, acceleration = self.reference(
            self._step_index * self.control_dt
        )
        return np.concatenate(
            [
                np.asarray(self.data.qpos[:2], dtype=np.float64),
                np.asarray(self.data.qvel[:2], dtype=np.float64),
                position,
                velocity,
                acceleration,
            ]
        )

    def set_external_force(self, force_xy: np.ndarray | list[float] | tuple[float, float]) -> None:
        force = np.asarray(force_xy, dtype=np.float64)
        if force.shape != (2,) or not np.all(np.isfinite(force)):
            raise ValueError("External force must be a finite length-2 vector")
        self._external_force = force.copy()

    def reset_model(self) -> np.ndarray:
        self._step_index = 0
        self._external_force[:] = 0.0
        self._trajectory = self._sample_trajectory()
        qpos = self.np_random.normal(0.0, 0.025, size=2)
        qvel = self.np_random.normal(0.0, 0.010, size=2)
        self.set_state(qpos, qvel)
        self._update_target_site()
        return self._get_obs()

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
        self._update_target_site()

        observation = self._get_obs()
        tracking_error = float(np.linalg.norm(observation[:2] - observation[4:6]))
        control_effort = float(np.dot(clipped, clipped) * self.control_dt)
        reward = -(tracking_error**2 + 0.01 * control_effort)
        finite = bool(np.all(np.isfinite(observation)))
        within_bounds = bool(np.max(np.abs(observation[:2])) <= self.position_limit)
        terminated = not (finite and within_bounds)
        truncated = self._step_index >= self.horizon_steps
        info: dict[str, Any] = {
            "tracking_error": tracking_error,
            "squared_tracking_error": tracking_error**2,
            "control_effort": control_effort,
            "reference_position": observation[4:6].copy(),
            "step_index": self._step_index,
            "failure": bool(terminated),
        }
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

