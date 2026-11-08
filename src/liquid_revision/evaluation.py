"""Closed-loop observer-in-PD evaluation primitives (no training)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data.closed_loop import _make_controller, _make_environment
from .perturbations import (
    ObservationPerturber,
    PerturbationSpec,
    build_observer_input,
    make_perturbation_schedule,
)


def evaluate_observer_episode(
    model,
    config: dict[str, Any],
    *,
    episode_seed: int,
    schedule_seed: int,
    max_steps: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one causal observer inside the shared frozen PD feedback loop.

    No true simulator state enters the model or controller.  True state is read
    only after the action is selected to compute observer-error diagnostics.
    """
    import torch

    environment_config = dict(config.get("environment", {}))
    controller_config = dict(config.get("controller", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    perturbation_mapping = evaluation_config.get(
        "perturbation", dict(config.get("dataset", {})).get("perturbation")
    )
    environment_kind = str(environment_config.get("kind", "reacher_v5"))
    env = _make_environment(environment_config)
    controller = _make_controller(
        environment_kind,
        controller_config,
        float(environment_config.get("action_limit", 1.0)),
    )
    spec = PerturbationSpec.from_mapping(perturbation_mapping)
    schedule = make_perturbation_schedule(
        horizon_steps=env.horizon_steps,
        nominal_dt=env.control_dt,
        spec=spec,
        seed=int(schedule_seed),
    )
    perturber = ObservationPerturber(schedule)
    controller.reset()
    model = model.to(device)
    model.eval()

    state = None
    previous_control = np.zeros(2, dtype=np.float32)
    estimated_states: list[np.ndarray] = []
    true_states: list[np.ndarray] = []
    tracking_errors: list[float] = []
    control_efforts: list[float] = []
    saturations: list[bool] = []
    terminated_by_failure = False

    try:
        clean_observation, _ = env.reset(seed=int(episode_seed))
        step_limit = env.horizon_steps if max_steps is None else min(
            env.horizon_steps, int(max_steps)
        )
        if step_limit <= 0:
            raise ValueError("max_steps must be positive when provided")

        with torch.no_grad():
            for step in range(step_limit):
                measured, mask, channel_age, observed_dt = perturber.observe(
                    clean_observation, step
                )
                observer_input = build_observer_input(
                    measured,
                    mask,
                    channel_age,
                    observed_dt,
                    previous_control,
                )
                x = torch.as_tensor(
                    observer_input, dtype=torch.float32, device=device
                )[None, None, :]
                dt = torch.as_tensor(
                    [[observed_dt]], dtype=torch.float32, device=device
                )
                output, state = model.forward_sequence(x, dt, state)
                estimated_state = (
                    output[0, -1].detach().cpu().numpy().astype(np.float32)
                )
                if estimated_state.shape != (env.state_dim,):
                    raise ValueError(
                        f"Observer returned {estimated_state.shape}; "
                        f"expected ({env.state_dim},)"
                    )
                if not np.all(np.isfinite(estimated_state)):
                    raise FloatingPointError("Observer returned non-finite state")

                action = controller.control(estimated_state, measured)
                applied_action = np.clip(
                    action * schedule.actuator_gain[step],
                    -env.action_limit,
                    env.action_limit,
                ).astype(np.float32)
                env.set_external_force(schedule.external_force[step])

                true_states.append(
                    np.asarray(clean_observation[: env.state_dim], dtype=np.float32)
                )
                estimated_states.append(estimated_state)
                next_observation, _, terminated, truncated, info = env.step(
                    applied_action
                )
                tracking_errors.append(float(info["tracking_error"]))
                control_efforts.append(float(info["control_effort"]))
                saturations.append(
                    bool(np.any(np.abs(applied_action) >= env.action_limit - 1e-7))
                )
                actuator_gear = float(
                    getattr(controller.config, "actuator_gear", 1.0)
                )
                previous_control = (applied_action * actuator_gear).astype(
                    np.float32
                )
                clean_observation = next_observation
                if terminated or truncated:
                    terminated_by_failure = bool(info["failure"])
                    break
    finally:
        env.close()

    estimates = np.stack(estimated_states)
    targets = np.stack(true_states)
    state_error = estimates - targets
    return {
        "environment_kind": environment_kind,
        "episode_seed": int(episode_seed),
        "schedule_seed": int(schedule_seed),
        "steps": int(len(tracking_errors)),
        "terminated_by_failure": terminated_by_failure,
        "state_rmse": float(np.sqrt(np.mean(np.square(state_error)))),
        "tracking_rmse": float(
            np.sqrt(np.mean(np.square(np.asarray(tracking_errors))))
        ),
        "control_effort": float(np.sum(control_efforts)),
        "saturation_fraction": float(np.mean(saturations)),
        "weights_updated": False,
    }
