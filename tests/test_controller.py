from __future__ import annotations

import numpy as np

from liquid_revision.controllers import (
    FrozenPDObserver,
    FrozenReacherPDObserver,
    PDObserverConfig,
    ReacherPDObserverConfig,
)


def test_frozen_controller_is_bounded_and_repeatable() -> None:
    config = PDObserverConfig(action_limit=0.7)
    obs = np.array([0, 0, 0, 0, 1, -1, 0, 0, 0, 0], dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)
    first = FrozenPDObserver(config)
    second = FrozenPDObserver(config)
    first.reset(obs)
    second.reset(obs)
    action_a = first.act(obs, mask, 0.02)
    action_b = second.act(obs, mask, 0.02)
    np.testing.assert_array_equal(action_a, action_b)
    assert np.max(np.abs(action_a)) <= 0.7
    assert first.metadata["trainable"] is False


def test_observer_predicts_through_missing_measurement() -> None:
    controller = FrozenPDObserver(PDObserverConfig())
    initial = np.zeros(10, dtype=np.float32)
    initial[2] = 1.0
    controller.reset(initial)
    missing = initial.copy()
    missing[:4] = 0
    action = controller.act(missing, np.zeros(4), 0.1)
    assert action.shape == (2,)
    assert np.isfinite(action).all()


def test_reacher_estimator_and_shared_pd_contract() -> None:
    controller = FrozenReacherPDObserver(ReacherPDObserverConfig())
    observation = np.array(
        [0.1, -0.2, 0.0, 0.0, 0.12, 0.02, 0.0, 0.0],
        dtype=np.float32,
    )
    controller.reset()
    estimate = controller.estimate(observation, np.ones(4), 0.02)
    action = controller.control(estimate, observation)
    assert estimate.shape == (4,)
    assert action.shape == (2,)
    assert np.isfinite(estimate).all()
    assert np.max(np.abs(action)) <= 1.0
    assert controller.metadata["actuator_gear"] == 200.0

