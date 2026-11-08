from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from liquid_revision.data.closed_loop import (
    generate_closed_loop_dataset,
    load_closed_loop_dataset,
)
from liquid_revision.envs import MovingTargetReacherEnv, PointMassTrackingEnv
from liquid_revision.evaluation import evaluate_observer_episode


def test_tracking_environment_steps() -> None:
    env = PointMassTrackingEnv(horizon_steps=4)
    try:
        observation, _ = env.reset(seed=11)
        assert observation.shape == (10,)
        env.set_external_force([0.1, -0.1])
        next_observation, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert next_observation.shape == (10,)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["step_index"] == 1
    finally:
        env.close()


def test_moving_target_reacher_steps_and_reports_identity() -> None:
    env = MovingTargetReacherEnv(horizon_steps=4)
    try:
        observation, _ = env.reset(seed=12)
        assert observation.shape == (8,)
        first_target = observation[4:6].copy()
        env.set_external_force([0.01, -0.01])
        next_observation, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert next_observation.shape == (8,)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["environment_id"] == "Reacher-v5-moving-target"
        assert not np.array_equal(first_target, next_observation[4:6])
    finally:
        env.close()


def test_short_dataset_generation_is_atomic(tmp_path: Path) -> None:
    config = {
        "experiment": {"seed": 1},
        "environment": {
            "kind": "point_mass_smoke",
            "horizon_steps": 6,
            "frame_skip": 2,
        },
        "controller": {"kp": 3.5, "kd": 1.4, "mass": 0.2},
        "dataset": {
            "episodes": 1,
            "first_episode_seed": 50,
            "perturbation": {"iid_dropout_probability": 0.1},
        },
    }
    output = tmp_path / "dataset"
    manifest = generate_closed_loop_dataset(config, output)
    arrays, loaded_manifest = load_closed_loop_dataset(output)
    assert manifest["dataset_sha256"] == loaded_manifest["dataset_sha256"]
    assert arrays["observer_input"].shape == (6, 21)
    assert arrays["target_state"].shape == (6, 4)
    assert arrays["expert_estimated_state"].shape == (6, 4)
    assert arrays["target_action"].shape == (6, 2)
    assert loaded_manifest["supervision"]["task"] == "causal_state_estimation"


def test_reacher_dataset_is_observer_supervision_not_action_policy(tmp_path: Path) -> None:
    config = {
        "experiment": {"seed": 2},
        "environment": {
            "kind": "reacher_v5",
            "horizon_steps": 5,
            "frame_skip": 2,
            "amplitude_range": [0.15, 0.20],
            "frequency_range_hz": [0.05, 0.08],
        },
        "controller": {"joint_kp": 10.0, "joint_kd": 1.0},
        "dataset": {
            "episodes": 1,
            "first_episode_seed": 60,
            "plant_perturbations_during_rollout": False,
            "perturbation": {"iid_dropout_probability": 0.2},
        },
    }
    output = tmp_path / "reacher_dataset"
    generate_closed_loop_dataset(config, output)
    arrays, manifest = load_closed_loop_dataset(output)
    assert arrays["observer_input"].shape == (5, 19)
    assert arrays["target_state"].shape == (5, 4)
    assert manifest["environment"]["reported_as"] == "Reacher-v5 moving-target"
    assert manifest["supervision"]["rollout_driver"] == "full_state_pd"
    assert manifest["supervision"]["sensor_corruption_changes_training_target"] is False


def test_causal_observer_is_embedded_in_shared_pd_loop() -> None:
    torch = pytest.importorskip("torch")

    class MeasuredStateObserver(torch.nn.Module):
        def forward_sequence(self, x, dt, state=None):
            del dt
            return x[..., :4], state

    config = {
        "environment": {
            "kind": "reacher_v5",
            "horizon_steps": 4,
            "amplitude_range": [0.15, 0.20],
            "frequency_range_hz": [0.05, 0.08],
        },
        "controller": {
            "joint_kp": 10.0,
            "joint_kd": 1.0,
            "actuator_gear": 200.0,
        },
        "evaluation": {"perturbation": {}},
    }
    report = evaluate_observer_episode(
        MeasuredStateObserver(),
        config,
        episode_seed=4,
        schedule_seed=5,
        max_steps=4,
    )
    assert report["steps"] == 4
    assert report["state_rmse"] == pytest.approx(0.0)
    assert report["weights_updated"] is False

