"""Paired closed-loop perturbation surfaces and auditable aggregate metrics."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.closed_loop import _make_controller, _make_environment
from ..io_utils import (
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
    source_tree_manifest,
)
from ..perturbations import (
    ObservationPerturber,
    PerturbationSpec,
    build_observer_input,
    make_perturbation_schedule,
    packet_loss_run_lengths,
)
from .baselines import build_nonlearned_baseline
from .manifests import load_trajectory_manifest, verify_split_datasets
from .training import load_trained_observer


def build_response_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = dict(config.get("evaluation", {}))
    base = dict(evaluation.get("base_perturbation", {}))
    conditions: list[dict[str, Any]] = []
    families = (
        (
            "iid_dropout",
            "iid_dropout_probability",
            [float(v) for v in evaluation.get("iid_dropout_probabilities", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])],
        ),
        (
            "burst",
            "burst_length_steps",
            [int(v) for v in evaluation.get("burst_lengths", [0, 1, 2, 5, 10, 20])],
        ),
        (
            "delay",
            "delay_steps",
            [int(v) for v in evaluation.get("delay_steps", [0, 1, 2, 4, 8])],
        ),
    )
    for family, key, severities in families:
        if not severities or float(severities[0]) != 0.0:
            raise ValueError(f"{family} severity grid must start at zero")
        if any(float(right) <= float(left) for left, right in zip(severities, severities[1:])):
            raise ValueError(f"{family} severities must be strictly increasing")
        for severity in severities:
            spec = dict(base)
            spec.update(
                {
                    "iid_dropout_probability": 0.0,
                    "burst_start_probability": 0.0,
                    "burst_stationary_probability": 0.0,
                    "burst_length_steps": 0,
                    "controlled_single_burst": False,
                    "delay_steps": 0,
                }
            )
            spec[key] = severity
            if family == "burst" and int(severity) > 0:
                spec["burst_stationary_probability"] = float(
                    evaluation.get("burst_stationary_probability", 0.30)
                )
            PerturbationSpec.from_mapping(spec)
            conditions.append(
                {
                    "condition_id": f"{family}-{float(severity):g}",
                    "family": family,
                    "severity": float(severity),
                    "spec": spec,
                }
            )
    controlled_length = int(evaluation.get("controlled_single_burst_length_steps", 10))
    if controlled_length > 0:
        spec = dict(base)
        spec.update(
            {
                "iid_dropout_probability": 0.0,
                "burst_start_probability": 0.0,
                "burst_stationary_probability": 0.0,
                "burst_length_steps": controlled_length,
                "controlled_single_burst": True,
                "delay_steps": 0,
            }
        )
        PerturbationSpec.from_mapping(spec)
        conditions.append(
            {
                "condition_id": f"controlled_single_burst-{controlled_length}",
                "family": "controlled_single_burst",
                "severity": float(controlled_length),
                "spec": spec,
            }
        )
    return conditions


def _paired_schedule_seed(base_seed: int, condition_id: str) -> int:
    digest = hashlib.sha256(condition_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return int((int(base_seed) + offset) % (2**31 - 1))


def _audit_burst_realization(
    config: dict[str, Any],
    conditions: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    environment = _make_environment(dict(config.get("environment", {})))
    try:
        horizon = int(environment.horizon_steps)
        nominal_dt = float(environment.control_dt)
    finally:
        environment.close()
    evaluation = dict(config.get("evaluation", {}))
    target_fraction = float(evaluation.get("burst_stationary_probability", 0.30))
    fraction_tolerance = float(
        evaluation.get("burst_missing_fraction_absolute_tolerance", 0.01)
    )
    mean_tolerance = float(evaluation.get("burst_mean_relative_tolerance", 0.10))
    rows = []
    for condition in conditions:
        if condition["family"] != "burst" or float(condition["severity"]) <= 0:
            continue
        masked_steps = 0
        run_steps = 0
        run_count = 0
        for episode in test_rows:
            schedule = make_perturbation_schedule(
                horizon_steps=horizon,
                nominal_dt=nominal_dt,
                spec=PerturbationSpec.from_mapping(condition["spec"]),
                seed=_paired_schedule_seed(
                    int(episode["schedule_seed"]), condition["condition_id"]
                ),
            )
            runs = packet_loss_run_lengths(schedule.sensor_mask)
            masked_steps += int(np.count_nonzero(~schedule.sensor_mask[:, 0]))
            run_steps += int(sum(runs))
            run_count += len(runs)
        realized_fraction = masked_steps / max(1, len(test_rows) * horizon)
        realized_mean = run_steps / max(1, run_count)
        requested_mean = float(condition["severity"])
        fraction_pass = abs(realized_fraction - target_fraction) <= fraction_tolerance
        mean_pass = (
            abs(realized_mean - requested_mean) / requested_mean <= mean_tolerance
        )
        rows.append(
            {
                "condition_id": condition["condition_id"],
                "requested_stationary_missing_fraction": target_fraction,
                "realized_stationary_missing_fraction": realized_fraction,
                "missing_fraction_absolute_tolerance": fraction_tolerance,
                "requested_mean_run_steps": requested_mean,
                "realized_mean_run_steps": realized_mean,
                "mean_run_relative_tolerance": mean_tolerance,
                "run_count": run_count,
                "fraction_gate_passed": fraction_pass,
                "mean_run_gate_passed": mean_pass,
                "condition_gate_passed": bool(fraction_pass and mean_pass),
            }
        )
    passed = bool(rows) and all(row["condition_gate_passed"] for row in rows)
    report = {
        "generator": "two_state_markov_geometric_runs",
        "episodes": len(test_rows),
        "conditions": rows,
        "gate_passed": passed,
        "gate_enforced": bool(evaluation.get("enforce_burst_realization_gate", False)),
    }
    if report["gate_enforced"] and not passed:
        raise RuntimeError(f"Burst realization gate failed: {report}")
    return report


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _estimate_state_for_control(
    estimator: dict[str, Any],
    *,
    clean: np.ndarray,
    measured: np.ndarray,
    mask: np.ndarray,
    age: np.ndarray,
    observed_dt: float,
    previous_control: np.ndarray,
    recurrent_state: Any,
    device: str,
) -> tuple[np.ndarray, Any, np.ndarray | None]:
    """Single audited estimator path used by the paper-facing rollout.

    Learned observers receive exactly the 19-D causal deployment input and
    never the clean simulator state.  Only the explicitly labelled full-state
    non-learning oracle is allowed to consume ``clean``.
    """
    learned_model = estimator.get("model")
    if learned_model is not None:
        observer_input = build_observer_input(
            measured, mask, age, observed_dt, previous_control
        )
        if observer_input.shape != (19,):
            raise RuntimeError(
                f"Learned observer input schema changed: {observer_input.shape}"
            )
        x = torch.as_tensor(
            observer_input, dtype=torch.float32, device=device
        )[None, None, :]
        dt = torch.as_tensor([[observed_dt]], dtype=torch.float32, device=device)
        output, next_state = learned_model.forward_sequence(x, dt, recurrent_state)
        estimate = output[0, -1].detach().cpu().numpy().astype(np.float32)
        return estimate, next_state, observer_input
    baseline = estimator.get("baseline")
    if baseline is None:
        raise ValueError("Estimator has neither learned model nor registered baseline")
    estimate = baseline.estimate(clean, measured, mask, observed_dt)
    return np.asarray(estimate, dtype=np.float32), recurrent_state, None


def _recovery_steps(
    errors: np.ndarray,
    disrupted: np.ndarray,
    *,
    scheduled_disrupted: np.ndarray | None = None,
    floor_m: float,
    baseline_multiplier: float,
    baseline_window_steps: int,
    consecutive: int,
    horizon_steps: int,
) -> tuple[float, bool, int, int, float]:
    """Return per-episode restricted recovery time with censoring metadata.

    Each contiguous packet-loss run is one event.  Its threshold uses only the
    same episode's pre-event tracking-error window.  An unrecovered event retains
    the remaining registered episode duration as a right-censored observation;
    it is never converted to NaN and silently discarded.
    """
    if floor_m <= 0 or baseline_multiplier <= 0:
        raise ValueError("Recovery threshold parameters must be positive")
    if baseline_window_steps <= 0 or consecutive <= 0 or horizon_steps <= 0:
        raise ValueError("Recovery window, run length, and horizon must be positive")
    event_mask = (
        np.asarray(scheduled_disrupted, dtype=bool)
        if scheduled_disrupted is not None
        else np.asarray(disrupted, dtype=bool)
    )
    if event_mask.ndim != 1:
        raise ValueError("Recovery disruption schedule must be one-dimensional")
    if not event_mask.any():
        return math.nan, False, 0, 0, math.nan
    starts = np.flatnonzero(event_mask & ~np.r_[False, event_mask[:-1]])
    ends = np.flatnonzero(event_mask & ~np.r_[event_mask[1:], False])
    if len(starts) != len(ends):
        raise AssertionError("Disruption run boundaries are inconsistent")
    restricted_durations: list[int] = []
    thresholds: list[float] = []
    censored_events = 0
    for start, end in zip(starts, ends):
        baseline_values = errors[
            max(0, int(start) - baseline_window_steps) : min(int(start), len(errors))
        ]
        baseline = float(np.median(baseline_values)) if len(baseline_values) else 0.0
        threshold = max(float(floor_m), float(baseline_multiplier) * baseline)
        thresholds.append(threshold)
        found = None
        for start in range(int(end) + 1, len(errors) - consecutive + 1):
            if np.all(errors[start : start + consecutive] <= threshold):
                found = start - int(end)
                break
        if found is None:
            censored_events += 1
            restricted_durations.append(max(0, int(horizon_steps) - 1 - int(end)))
        else:
            restricted_durations.append(int(found))
    return (
        float(np.mean(restricted_durations)),
        bool(censored_events),
        len(restricted_durations),
        censored_events,
        float(np.mean(thresholds)),
    )


def _rollout(
    estimator: dict[str, Any],
    config: dict[str, Any],
    condition: dict[str, Any],
    *,
    episode_seed: int,
    schedule_seed: int,
    device: str,
) -> dict[str, Any]:
    environment_config = dict(config.get("environment", {}))
    controller_config = dict(config.get("controller", {}))
    environment_kind = str(environment_config.get("kind", "reacher_v5"))
    if environment_kind != "reacher_v5":
        raise ValueError("Paper-facing response surfaces require reacher_v5")
    env = _make_environment(environment_config)
    controller = _make_controller(
        environment_kind,
        controller_config,
        float(environment_config.get("action_limit", 1.0)),
    )
    spec = PerturbationSpec.from_mapping(condition["spec"])
    schedule = make_perturbation_schedule(
        horizon_steps=env.horizon_steps,
        nominal_dt=env.control_dt,
        spec=spec,
        seed=int(schedule_seed),
    )
    perturber = ObservationPerturber(schedule)
    controller.reset()
    previous_control = np.zeros(2, dtype=np.float32)
    learned_model = estimator.get("model")
    baseline = estimator.get("baseline")
    recurrent_state = None
    if learned_model is not None:
        learned_model.eval()
    else:
        baseline.reset()
    tracking_errors: list[float] = []
    state_squared_errors: list[float] = []
    joint_position_squared_errors: list[float] = []
    joint_velocity_squared_errors: list[float] = []
    control_effort: list[float] = []
    saturated: list[bool] = []
    disrupted: list[bool] = []
    latency_ms: list[float] = []
    step_log: list[dict[str, Any]] = []
    ode_solver_function_evaluations = 0
    ltc_internal_unfolds = 0
    failure = False
    failure_reason = "none"
    try:
        clean, _ = env.reset(seed=int(episode_seed))
        with torch.no_grad():
            for step in range(env.horizon_steps):
                measured, mask, age, observed_dt = perturber.observe(clean, step)
                step_ode_nfe = 0
                step_ltc_unfolds = 0
                if estimator.get("model_name") in {"odernn", "ctrnn"}:
                    core_model = getattr(learned_model, "model", learned_model)
                    max_step = float(getattr(core_model, "max_step", 0.02))
                    step_ode_nfe = 4 * max(1, int(math.ceil(observed_dt / max_step)))
                elif estimator.get("model_name") == "ltc":
                    step_ltc_unfolds = int(
                        config.get("models", {}).get("liquid", {}).get("ode_unfolds", 6)
                    )
                ode_solver_function_evaluations += step_ode_nfe
                ltc_internal_unfolds += step_ltc_unfolds
                _synchronize(device)
                started = time.perf_counter_ns()
                estimate, recurrent_state, _ = _estimate_state_for_control(
                    estimator,
                    clean=clean,
                    measured=measured,
                    mask=mask,
                    age=age,
                    observed_dt=observed_dt,
                    previous_control=previous_control,
                    recurrent_state=recurrent_state,
                    device=device,
                )
                _synchronize(device)
                latency_ms.append((time.perf_counter_ns() - started) / 1e6)
                if estimate.shape != (4,) or not np.all(np.isfinite(estimate)):
                    # A model-level numerical failure is a registered episode
                    # outcome.  Preserve the current plant/observation state in
                    # the immutable step log instead of aborting the surface.
                    current_error = float(
                        np.linalg.norm(
                            env.get_body_com("fingertip")[:2]
                            - env.get_body_com("target")[:2]
                        )
                    )
                    tracking_errors.append(current_error)
                    control_effort.append(0.0)
                    saturated.append(False)
                    disrupted.append(bool(np.any(mask < 0.5)))
                    state_squared_errors.append(math.nan)
                    joint_position_squared_errors.append(math.nan)
                    joint_velocity_squared_errors.append(math.nan)
                    failure = True
                    failure_reason = "estimator_nonfinite_or_invalid_shape"
                    step_log.append(
                        {
                            "step": int(step),
                            "clean_state_and_reference": np.asarray(clean[:8], dtype=np.float32),
                            # No plant integration is performed after an
                            # observer-level numerical failure.  Retaining the
                            # current clean state in the post-step slot and an
                            # explicit advancement flag lets the shard verifier
                            # distinguish this registered terminal event from a
                            # genuine MuJoCo transition.
                            "post_step_clean_state_and_reference": np.asarray(
                                clean[:8], dtype=np.float32
                            ),
                            "plant_advanced": False,
                            "measured_state_and_reference": np.asarray(measured[:8], dtype=np.float32),
                            "sensor_mask": np.asarray(mask, dtype=np.float32),
                            "channel_age": np.asarray(age, dtype=np.float32),
                            "predicted_state": np.full(4, np.nan, dtype=np.float32),
                            "applied_action": np.full(2, np.nan, dtype=np.float32),
                            "applied_joint_torque": np.full(2, np.nan, dtype=np.float32),
                            "observed_timestamp_dt": float(observed_dt),
                            "nominal_plant_dt": float(env.control_dt),
                            "delay_steps": int(schedule.delay_steps),
                            "delayed_source_step": int(max(0, step - schedule.delay_steps)),
                            "tracking_error": current_error,
                            "control_effort": 0.0,
                            "actuator_saturated": False,
                            "failure": True,
                            "failure_reason": failure_reason,
                            "latency_ms": float(latency_ms[-1]),
                            "ode_solver_function_evaluations": int(step_ode_nfe),
                            "ltc_internal_unfolds": int(step_ltc_unfolds),
                        }
                    )
                    break
                action = controller.control(estimate, measured)
                applied = np.clip(
                    action * schedule.actuator_gain[step],
                    -env.action_limit,
                    env.action_limit,
                ).astype(np.float32)
                env.set_external_force(schedule.external_force[step])
                state_squared_errors.append(
                    float(np.mean(np.square(estimate - np.asarray(clean[:4]))))
                )
                joint_position_squared_errors.append(
                    float(np.mean(np.square(estimate[:2] - np.asarray(clean[:2]))))
                )
                joint_velocity_squared_errors.append(
                    float(np.mean(np.square(estimate[2:] - np.asarray(clean[2:4]))))
                )
                next_clean, _, terminated, truncated, info = env.step(applied)
                tracking_errors.append(float(info["tracking_error"]))
                control_effort.append(float(info["control_effort"]))
                saturated.append(bool(info["actuator_saturated"]))
                disrupted.append(bool(np.any(mask < 0.5)))
                active_failure_reasons = [
                    name
                    for name, active in info.get("failure_reasons", {}).items()
                    if active
                ]
                step_failure_reason = ",".join(active_failure_reasons) or "none"
                step_log.append(
                    {
                        "step": int(step),
                        "clean_state_and_reference": np.asarray(clean[:8], dtype=np.float32),
                        "post_step_clean_state_and_reference": np.asarray(
                            next_clean[:8], dtype=np.float32
                        ),
                        "plant_advanced": True,
                        "measured_state_and_reference": np.asarray(measured[:8], dtype=np.float32),
                        "sensor_mask": np.asarray(mask, dtype=np.float32),
                        "channel_age": np.asarray(age, dtype=np.float32),
                        "predicted_state": np.asarray(estimate, dtype=np.float32),
                        "applied_action": np.asarray(applied, dtype=np.float32),
                        "applied_joint_torque": np.asarray(
                            applied
                            * float(getattr(controller.config, "actuator_gear", 1.0)),
                            dtype=np.float32,
                        ),
                        "observed_timestamp_dt": float(observed_dt),
                        "nominal_plant_dt": float(env.control_dt),
                        "delay_steps": int(schedule.delay_steps),
                        "delayed_source_step": int(max(0, step - schedule.delay_steps)),
                        "tracking_error": float(info["tracking_error"]),
                        "control_effort": float(info["control_effort"]),
                        "actuator_saturated": bool(info["actuator_saturated"]),
                        "failure": bool(info["failure"]),
                        "failure_reason": step_failure_reason,
                        "latency_ms": float(latency_ms[-1]),
                        "ode_solver_function_evaluations": int(step_ode_nfe),
                        "ltc_internal_unfolds": int(step_ltc_unfolds),
                    }
                )
                previous_control = (
                    applied * float(getattr(controller.config, "actuator_gear", 1.0))
                ).astype(np.float32)
                clean = next_clean
                if terminated or truncated:
                    failure = bool(info["failure"])
                    failure_reason = step_failure_reason
                    break
    finally:
        env.close()
    errors = np.asarray(tracking_errors, dtype=np.float64)
    if errors.size == 0 or not np.all(np.isfinite(errors)):
        raise ValueError("Rollout produced an empty or non-finite tracking-error path")
    failure_penalty_m = float(
        config.get("evaluation", {}).get("failure_tracking_penalty_m", 1.0)
    )
    if not np.isfinite(failure_penalty_m) or failure_penalty_m <= 0.0:
        raise ValueError("failure_tracking_penalty_m must be finite and positive")
    penalty_steps = max(0, int(env.horizon_steps) - len(errors)) if failure else 0
    scored_errors = (
        np.concatenate(
            [errors, np.full(penalty_steps, failure_penalty_m, dtype=np.float64)]
        )
        if penalty_steps
        else errors
    )
    if condition["family"] == "controlled_single_burst":
        (
            recovery,
            recovery_censored,
            recovery_event_count,
            recovery_censored_events,
            recovery_threshold_mean,
        ) = _recovery_steps(
            errors,
            np.asarray(disrupted, dtype=bool),
            scheduled_disrupted=np.asarray(
                ~schedule.sensor_mask[:, 0], dtype=bool
            ),
            floor_m=float(
                config.get("evaluation", {}).get("recovery_floor_m", 0.02)
            ),
            baseline_multiplier=float(
                config.get("evaluation", {}).get(
                    "recovery_baseline_multiplier", 1.0
                )
            ),
            baseline_window_steps=int(
                config.get("evaluation", {}).get(
                    "recovery_baseline_window_steps", 50
                )
            ),
            consecutive=int(
                config.get("evaluation", {}).get("recovery_consecutive_steps", 10)
            ),
            horizon_steps=env.horizon_steps,
        )
    else:
        recovery = math.nan
        recovery_censored = False
        recovery_event_count = 0
        recovery_censored_events = 0
        recovery_threshold_mean = math.nan
    control_applied = np.asarray(
        [bool(row["plant_advanced"]) for row in step_log], dtype=bool
    )
    applied_control_steps = int(np.count_nonzero(control_applied))
    applied_saturation = np.asarray(saturated, dtype=bool)[control_applied]
    return {
        "episode_seed": int(episode_seed),
        "schedule_seed": int(schedule_seed),
        "steps": len(errors),
        "tracking_scored_steps": len(scored_errors),
        "tracking_penalty_steps": int(penalty_steps),
        "tracking_failure_penalty_m": failure_penalty_m,
        "tracking_estimand": (
            "fixed-horizon RMSE/p95; after a registered failure, each unobserved "
            "remaining step is scored at the preregistered fixed penalty"
        ),
        "tracking_rmse": float(np.sqrt(np.mean(np.square(scored_errors)))),
        "tracking_p95": float(np.quantile(scored_errors, 0.95)),
        "failure": float(failure),
        "failure_reason": failure_reason,
        "recovery_steps": recovery,
        "recovery_seconds": (
            float(recovery * env.control_dt) if np.isfinite(recovery) else math.nan
        ),
        "recovery_censored": bool(recovery_censored),
        "recovery_event_count": int(recovery_event_count),
        "recovery_censored_events": int(recovery_censored_events),
        "recovery_threshold_mean_m": recovery_threshold_mean,
        "recovery_definition": (
            "controlled single burst; threshold=max(pre-burst 50-step median,"
            "0.02m); 10 consecutive steps; restricted duration with "
            "unrecovered episodes right-censored at registered remainder"
            if recovery_event_count
            else "not applicable outside the controlled_single_burst condition"
        ),
        "burst_masked_steps": int(np.count_nonzero(~schedule.sensor_mask[:, 0])),
        "burst_run_count": len(packet_loss_run_lengths(schedule.sensor_mask)),
        "burst_run_steps_total": int(
            sum(packet_loss_run_lengths(schedule.sensor_mask))
        ),
        "control_effort": float(np.sum(control_effort)),
        "control_effort_definition": (
            "sum of squared normalized MuJoCo actuator commands multiplied by "
            "the fixed plant control interval (command^2*s)"
        ),
        "saturation_fraction": float(
            np.mean(applied_saturation) if applied_control_steps else 0.0
        ),
        "control_observed_steps": applied_control_steps,
        "control_observed_fraction": float(
            applied_control_steps / env.horizon_steps
        ),
        "plant_advanced_steps": applied_control_steps,
        "plant_completion_fraction": float(
            applied_control_steps / env.horizon_steps
        ),
        "control_endpoint_interpretation": (
            "observed-step-only after an early failure; interpret effort and "
            "saturation jointly with completion/failure"
        ),
        "state_rmse": float(
            np.sqrt(np.nanmean(state_squared_errors))
            if np.any(np.isfinite(state_squared_errors))
            else math.nan
        ),
        "joint_position_rmse": float(
            np.sqrt(np.nanmean(joint_position_squared_errors))
            if np.any(np.isfinite(joint_position_squared_errors))
            else math.nan
        ),
        "joint_velocity_rmse": float(
            np.sqrt(np.nanmean(joint_velocity_squared_errors))
            if np.any(np.isfinite(joint_velocity_squared_errors))
            else math.nan
        ),
        "state_accuracy_finite_steps": int(
            np.count_nonzero(np.isfinite(state_squared_errors))
        ),
        "state_accuracy_finite_fraction": float(
            np.count_nonzero(np.isfinite(state_squared_errors)) / env.horizon_steps
        ),
        "state_accuracy_definition": (
            "conditional on finite observer outputs before registered failure; "
            "joint position is radians and joint velocity is radians/second"
        ),
        "latency_median_ms": float(np.median(latency_ms)),
        "latency_p95_ms": float(np.quantile(latency_ms, 0.95)),
        "masked_fraction": float(1.0 - schedule.sensor_mask.mean()),
        "ode_solver_function_evaluations": int(
            ode_solver_function_evaluations
        ),
        "ltc_internal_unfolds": int(ltc_internal_unfolds),
        "weights_updated": False,
        "_step_log": step_log,
    }


def _bootstrap_mean_ci(
    rows: list[dict[str, Any]], metric: str, *, seed: int, replicates: int
) -> tuple[float, float]:
    values_by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            values_by_seed[int(row["training_seed"])].append(value)
    if not values_by_seed:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    training_seeds = sorted(values_by_seed)
    draws = []
    for _ in range(replicates):
        sampled_seeds = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        per_seed = []
        for sampled in sampled_seeds:
            candidates = values_by_seed[int(sampled)]
            sampled_episodes = rng.choice(candidates, size=len(candidates), replace=True)
            per_seed.append(float(np.mean(sampled_episodes)))
        draws.append(float(np.mean(per_seed)))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summarize(rows: list[dict[str, Any]], config: dict[str, Any]):
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["condition_id"])].append(row)
    metrics = (
        "tracking_rmse",
        "tracking_p95",
        "failure",
        "recovery_steps",
        "recovery_seconds",
        "control_effort",
        "saturation_fraction",
        "control_observed_fraction",
        "plant_completion_fraction",
        "state_rmse",
        "joint_position_rmse",
        "joint_velocity_rmse",
        "state_accuracy_finite_fraction",
        "latency_median_ms",
        "latency_p95_ms",
        "ode_solver_function_evaluations",
        "ltc_internal_unfolds",
    )
    replicates = int(config.get("evaluation", {}).get("bootstrap_replicates", 2000))
    summary = []
    for (model_name, condition_id), group in sorted(grouped.items()):
        episode_identities: set[tuple[str, str]] = set()
        for item in group:
            trajectory_id = item.get("trajectory_id")
            if trajectory_id is not None and str(trajectory_id).strip():
                episode_identities.add(("trajectory_id", str(trajectory_id)))
                continue
            episode_id = item.get("episode_id")
            if episode_id is None or not str(episode_id).strip():
                raise ValueError(
                    "Condition summary rows require trajectory_id or episode_id"
                )
            episode_identities.add(("episode_id", str(episode_id)))
        row = {
            "model_name": model_name,
            "condition_id": condition_id,
            "family": group[0]["family"],
            "severity": group[0]["severity"],
            "model_seed_count": len(
                {int(item["training_seed"]) for item in group}
            ),
            "unique_test_episodes": len(episode_identities),
            "row_count": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group], dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else math.nan
            low, high = _bootstrap_mean_ci(
                group,
                metric,
                seed=int.from_bytes(hashlib.sha256(f"{model_name}|{condition_id}|{metric}".encode()).digest()[:4], "big"),
                replicates=replicates,
            )
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary.append(row)
    return summary


def _curve_naurc(severities: np.ndarray, means: np.ndarray) -> float:
    severities = np.asarray(severities, dtype=float)
    means = np.asarray(means, dtype=float)
    if len(severities) < 2 or severities.shape != means.shape:
        raise ValueError("nAURC requires paired severity and mean vectors")
    if not np.all(np.isfinite(severities)) or not np.all(np.isfinite(means)):
        raise ValueError("nAURC input contains a non-finite value")
    if not np.all(np.diff(severities) > 0.0):
        raise ValueError("nAURC severity grid must be strictly increasing")
    if means[0] <= 0.0:
        raise ValueError("nAURC nominal denominator must be finite and positive")
    value = float(
        np.trapezoid(means / means[0], severities)
        / (severities[-1] - severities[0])
    )
    if not np.isfinite(value):
        raise ValueError("nAURC calculation produced a non-finite value")
    return value


def _curve_matrices(
    raw: list[dict[str, Any]],
    model_name: str,
    family: str,
    metric: str = "tracking_rmse",
) -> tuple[np.ndarray, dict[int, tuple[list[str], np.ndarray]]]:
    rows = [
        row
        for row in raw
        if row["model_name"] == model_name and row["family"] == family
    ]
    severities = np.asarray(sorted({float(row["severity"]) for row in rows}))
    if len(severities) < 2 or not len(rows):
        return severities, {}
    output: dict[int, tuple[list[str], np.ndarray]] = {}
    for training_seed in sorted({int(row["training_seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["training_seed"]) == training_seed]
        episode_sets = [
            {
                str(row["trajectory_id"])
                for row in seed_rows
                if float(row["severity"]) == severity
            }
            for severity in severities
        ]
        if not episode_sets or any(values != episode_sets[0] for values in episode_sets):
            raise ValueError(
                f"Unpaired episode set across severities for {model_name}, seed "
                f"{training_seed}, family {family}"
            )
        episodes = sorted(episode_sets[0])
        lookup = {
            (float(row["severity"]), str(row["trajectory_id"])): float(
                row[metric]
            )
            for row in seed_rows
        }
        if len(lookup) != len(seed_rows):
            raise ValueError("Duplicate model/seed/severity/trajectory evaluation row")
        matrix = np.asarray(
            [[lookup[(float(severity), episode)] for episode in episodes] for severity in severities],
            dtype=float,
        )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"Non-finite {metric} surface for {model_name}, seed "
                f"{training_seed}, family {family}"
            )
        output[training_seed] = (episodes, matrix)
    return severities, output


def _naurc_by_seed(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pairs = sorted({(row["model_name"], row["family"]) for row in raw})
    for model_name, family in pairs:
        severities, matrices = _curve_matrices(raw, model_name, family)
        for training_seed, (episodes, matrix) in matrices.items():
            value = _curve_naurc(severities, matrix.mean(axis=1))
            output.append(
                {
                    "model_name": model_name,
                    "training_seed": training_seed,
                    "family": family,
                    "episodes": len(episodes),
                    "severity_levels": len(severities),
                    "tracking_naurc": value,
                    "normalization": "same-model same-training-seed severity-zero episode mean",
                    "definition": "per-seed paired-episode normalized trapezoidal response area; lower is better",
                }
            )
    return output


def _bootstrap_model_naurc(
    raw: list[dict[str, Any]],
    model_name: str,
    family: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    severities, matrices = _curve_matrices(raw, model_name, family)
    if not matrices:
        return math.nan, math.nan
    training_seeds = sorted(matrices)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        values = []
        for sampled_seed in sampled_seeds:
            _, matrix = matrices[int(sampled_seed)]
            episode_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            values.append(
                _curve_naurc(severities, matrix[:, episode_indices].mean(axis=1))
            )
        draws[replicate] = float(np.mean(values))
    if not np.all(np.isfinite(draws)):
        raise ValueError("nAURC bootstrap produced a non-finite draw")
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _summarize_naurc(
    raw: list[dict[str, Any]],
    by_seed: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    evaluation = dict(config.get("evaluation", {}))
    replicates = int(evaluation.get("bootstrap_replicates", 10_000))
    analysis_seed = int(evaluation.get("analysis_seed", 20261105))
    output = []
    for model_name, family in sorted(
        {(row["model_name"], row["family"]) for row in by_seed}
    ):
        group = [
            row
            for row in by_seed
            if row["model_name"] == model_name and row["family"] == family
        ]
        values = np.asarray([float(row["tracking_naurc"]) for row in group])
        if not np.all(np.isfinite(values)):
            raise ValueError("Per-seed nAURC summary contains a non-finite value")
        digest = hashlib.sha256(f"{model_name}|{family}".encode()).digest()
        low, high = _bootstrap_model_naurc(
            raw,
            model_name,
            family,
            replicates=replicates,
            seed=(analysis_seed + int.from_bytes(digest[:4], "big")) % (2**32),
        )
        output.append(
            {
                "model_name": model_name,
                "family": family,
                "training_seeds": len(group),
                "episodes_per_seed": int(group[0]["episodes"]),
                "tracking_naurc_mean": float(np.mean(values)),
                "tracking_naurc_ci95_low": low,
                "tracking_naurc_ci95_high": high,
                "bootstrap_replicates": replicates,
                "inferential_role": (
                    "primary endpoint" if family == "iid_dropout" else "descriptive secondary"
                ),
            }
        )
    return output


PRIMARY_CONTRASTS = (
    (
        "adaptive_time_scale_ltc_minus_ctrnn",
        "ltc",
        "ctrnn",
        "non-trainable state-invariant tau control versus input/state-dependent liquid dynamics; operational architecture contrast, not a single-parameter causal effect",
    ),
    (
        "solver_free_cfc_minus_ltc",
        "cfc",
        "ltc",
        "ODE-unfolded LTC versus solver-free CfC under matched training/evaluation and registered parameter tolerance",
    ),
    (
        "structured_ncp_minus_degree_matched_random",
        "ncp_cfc",
        "random_sparse_cfc",
        "same synchronous graph-CfC update, motor outputs, node and edge counts, and exact per-node in/out degrees; global swaps break NCP pathway constraints",
    ),
)

SECONDARY_BASELINE_CONTRASTS = (
    ("ltc_minus_lstm", "ltc", "lstm", "secondary learned-model comparison"),
    ("ltc_minus_odernn", "ltc", "odernn", "secondary learned-model comparison"),
    ("cfc_minus_lstm", "cfc", "lstm", "secondary learned-model comparison"),
    ("cfc_minus_odernn", "cfc", "odernn", "secondary learned-model comparison"),
)


def _exact_paired_seed_sign_flip_p(differences: np.ndarray) -> float:
    """Two-sided exact randomization p-value over paired training-seed effects."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Exact seed sign-flip test requires finite paired effects")
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 2 ** int(values.size)
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        if statistic >= observed - 1e-15:
            extreme += 1
    return float(extreme / total)


def _paired_bootstrap_contrast(
    raw: list[dict[str, Any]],
    *,
    contrast: str,
    first_model: str,
    second_model: str,
    scope: str,
    family: str,
    replicates: int,
    analysis_seed: int,
    failure_noninferiority_margin: float,
) -> dict[str, Any] | None:
    first_severities, first = _curve_matrices(raw, first_model, family)
    second_severities, second = _curve_matrices(raw, second_model, family)
    if not first or not second:
        return None
    if not np.array_equal(first_severities, second_severities):
        raise ValueError(f"Severity grids differ for contrast {contrast}")
    paired_seeds = sorted(set(first) & set(second))
    if set(first) != set(second):
        raise ValueError(f"Training seeds are not paired for contrast {contrast}")
    failure_first_severities, failure_first = _curve_matrices(
        raw, first_model, family, metric="failure"
    )
    failure_second_severities, failure_second = _curve_matrices(
        raw, second_model, family, metric="failure"
    )
    if (
        not np.array_equal(first_severities, failure_first_severities)
        or not np.array_equal(first_severities, failure_second_severities)
        or set(failure_first) != set(first)
        or set(failure_second) != set(second)
    ):
        raise ValueError(f"Failure surface is not paired for contrast {contrast}")
    paired: dict[int, np.ndarray] = {}
    paired_failures: dict[int, np.ndarray] = {}
    observed_first = []
    observed_second = []
    observed_failure_first = []
    observed_failure_second = []
    for training_seed in paired_seeds:
        first_episodes, first_matrix = first[training_seed]
        second_episodes, second_matrix = second[training_seed]
        if first_episodes != second_episodes:
            raise ValueError(f"Episodes are not paired for contrast {contrast}")
        failure_first_episodes, failure_first_matrix = failure_first[training_seed]
        failure_second_episodes, failure_second_matrix = failure_second[training_seed]
        if (
            failure_first_episodes != first_episodes
            or failure_second_episodes != first_episodes
            or failure_first_matrix.shape != first_matrix.shape
            or failure_second_matrix.shape != first_matrix.shape
        ):
            raise ValueError(f"Failure episodes are not paired for contrast {contrast}")
        paired[training_seed] = np.stack([first_matrix, second_matrix], axis=0)
        paired_failures[training_seed] = np.stack(
            [failure_first_matrix, failure_second_matrix], axis=0
        )
        observed_first.append(
            _curve_naurc(first_severities, first_matrix.mean(axis=1))
        )
        observed_second.append(
            _curve_naurc(first_severities, second_matrix.mean(axis=1))
        )
        observed_failure_first.append(float(failure_first_matrix.mean()))
        observed_failure_second.append(float(failure_second_matrix.mean()))
    paired_seed_differences = np.asarray(observed_first) - np.asarray(observed_second)
    estimate = float(np.mean(paired_seed_differences))
    exact_seed_p = _exact_paired_seed_sign_flip_p(paired_seed_differences)
    digest = hashlib.sha256(contrast.encode()).digest()
    rng = np.random.default_rng(
        (int(analysis_seed) + int.from_bytes(digest[:4], "big")) % (2**32)
    )
    draws = np.empty(replicates, dtype=float)
    failure_draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(paired_seeds, size=len(paired_seeds), replace=True)
        differences = []
        failure_differences = []
        for sampled_seed in sampled_seeds:
            matrix = paired[int(sampled_seed)]
            failure_matrix = paired_failures[int(sampled_seed)]
            episode_indices = rng.integers(0, matrix.shape[2], size=matrix.shape[2])
            means = matrix[:, :, episode_indices].mean(axis=2)
            differences.append(
                _curve_naurc(first_severities, means[0])
                - _curve_naurc(first_severities, means[1])
            )
            sampled_failure = failure_matrix[:, :, episode_indices]
            failure_differences.append(
                float(sampled_failure[0].mean() - sampled_failure[1].mean())
            )
        draws[replicate] = float(np.mean(differences))
        failure_draws[replicate] = float(np.mean(failure_differences))
    if not np.all(np.isfinite(draws)) or not np.all(np.isfinite(failure_draws)):
        raise ValueError(f"Non-finite paired bootstrap draw for {contrast}")
    nonpositive = int(np.count_nonzero(draws <= 0.0))
    nonnegative = int(np.count_nonzero(draws >= 0.0))
    sign_p = min(
        1.0,
        2.0 * (min(nonpositive, nonnegative) + 1.0) / (replicates + 1.0),
    )
    failure_difference = float(
        np.mean(
            np.asarray(observed_failure_first)
            - np.asarray(observed_failure_second)
        )
    )
    failure_ci_high = float(np.quantile(failure_draws, 0.975))
    failure_noninferiority_met = bool(
        failure_ci_high <= float(failure_noninferiority_margin)
    )
    return {
        "contrast": contrast,
        "family": family,
        "first_model": first_model,
        "second_model": second_model,
        "first_tracking_naurc_mean": float(np.mean(observed_first)),
        "second_tracking_naurc_mean": float(np.mean(observed_second)),
        "estimate_first_minus_second": estimate,
        "relative_change_vs_second": (
            estimate / float(np.mean(observed_second))
            if float(np.mean(observed_second)) != 0.0
            else math.nan
        ),
        "practical_effect_threshold": "at least 5% nAURC reduction",
        "practical_effect_met": bool(
            float(np.mean(observed_second)) > 0
            and estimate / float(np.mean(observed_second)) <= -0.05
        ),
        "first_failure_rate_mean": float(np.mean(observed_failure_first)),
        "second_failure_rate_mean": float(np.mean(observed_failure_second)),
        "failure_rate_difference_first_minus_second": failure_difference,
        "failure_rate_difference_ci95_low": float(
            np.quantile(failure_draws, 0.025)
        ),
        "failure_rate_difference_ci95_high": failure_ci_high,
        "failure_noninferiority_margin_absolute": float(
            failure_noninferiority_margin
        ),
        "failure_noninferiority_met": failure_noninferiority_met,
        "superiority_requires_failure_noninferiority": True,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "paired_seed_exact_sign_flip_p_raw": exact_seed_p,
        "bootstrap_sign_probability_two_sided_descriptive": sign_p,
        # Backward-compatible field name retained only as a descriptive value;
        # Holm correction below uses the exact training-seed randomization p.
        "bootstrap_sign_p_raw": sign_p,
        "bootstrap_replicates": replicates,
        "paired_training_seeds": len(paired_seeds),
        "paired_episodes_per_seed": int(next(iter(paired.values())).shape[2]),
        "pairing": "seed then episode; both models and all severity levels retained",
        "interpretation_scope": scope,
    }


def _holm(rows: list[dict[str, Any]], family_name: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    p_field = "paired_seed_exact_sign_flip_p_raw"
    if any(p_field not in row for row in rows):
        raise ValueError("Holm family lacks exact paired-seed p-values")
    order = sorted(range(len(rows)), key=lambda index: rows[index][p_field])
    adjusted = [math.nan] * len(rows)
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        candidate = min(
            1.0,
            (total - rank) * float(rows[index][p_field]),
        )
        running = max(running, candidate)
        adjusted[index] = running
    output = []
    for index, row in enumerate(rows):
        superiority_eligible = bool(
            adjusted[index] < 0.05
            and row.get("practical_effect_met", False)
            and row.get("failure_noninferiority_met", False)
        )
        output.append(
            {
                **row,
                "multiplicity_family": family_name,
                "holm_input_p": p_field,
                "holm_adjusted_p": adjusted[index],
                "holm_reject_at_0_05": bool(adjusted[index] < 0.05),
                "superiority_claim_eligible": superiority_eligible,
                "inferential_conclusion": (
                    "superiority eligible under statistical, practical-effect, and failure non-inferiority gates"
                    if superiority_eligible
                    else "no superiority claim; at least one registered gate was not met"
                ),
            }
        )
    return output


def _mechanism_contrasts(
    raw: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}))
    replicates = int(evaluation.get("bootstrap_replicates", 10_000))
    analysis_seed = int(evaluation.get("analysis_seed", 20261105))
    failure_noninferiority_margin = float(
        evaluation.get("failure_noninferiority_margin_absolute", 0.02)
    )
    if (
        not np.isfinite(failure_noninferiority_margin)
        or failure_noninferiority_margin < 0.0
    ):
        raise ValueError("Invalid failure non-inferiority margin")

    def build(definitions):
        output = []
        for contrast, first_model, second_model, scope in definitions:
            result = _paired_bootstrap_contrast(
                raw,
                contrast=contrast,
                first_model=first_model,
                second_model=second_model,
                scope=scope,
                family="iid_dropout",
                replicates=replicates,
                analysis_seed=analysis_seed,
                failure_noninferiority_margin=failure_noninferiority_margin,
            )
            if result is not None:
                output.append(result)
        return output

    primary = _holm(build(PRIMARY_CONTRASTS), "primary_packet_dropout_nAURC")
    secondary = _holm(
        build(SECONDARY_BASELINE_CONTRASTS),
        "secondary_learned_baseline_packet_dropout_nAURC",
    )
    by_seed = _naurc_by_seed(raw)
    lookup = {
        (row["model_name"], int(row["training_seed"]), row["family"]): float(
            row["tracking_naurc"]
        )
        for row in by_seed
    }
    descriptive = []
    for contrast, first_model, second_model, scope in PRIMARY_CONTRASTS:
        for family in ("burst", "delay"):
            common_seeds = sorted(
                {
                    seed
                    for model, seed, row_family in lookup
                    if model == first_model and row_family == family
                }
                & {
                    seed
                    for model, seed, row_family in lookup
                    if model == second_model and row_family == family
                }
            )
            if common_seeds:
                descriptive.append(
                    {
                        "contrast": contrast,
                        "family": family,
                        "first_model": first_model,
                        "second_model": second_model,
                        "estimate_first_minus_second": float(
                            np.mean(
                                [
                                    lookup[(first_model, seed, family)]
                                    - lookup[(second_model, seed, family)]
                                    for seed in common_seeds
                                ]
                            )
                        ),
                        "inferential_role": "descriptive secondary; no p-value or Holm test",
                        "interpretation_scope": scope,
                    }
                )
    return {
        "schema_version": 2,
        "analysis_seed": analysis_seed,
        "bootstrap_replicates": replicates,
        "primary_packet_dropout": primary,
        "secondary_learned_baseline_packet_dropout": secondary,
        "descriptive_burst_and_delay": descriptive,
        "sign_convention": "first model minus second model; lower nAURC is better",
    }


def _paired_scalar_endpoint_contrast(
    raw: list[dict[str, Any]],
    *,
    contrast: str,
    first_model: str,
    second_model: str,
    condition_family: str,
    metric: str,
    replicates: int,
    analysis_seed: int,
    practical_threshold: float | None,
) -> dict[str, Any] | None:
    selected = [row for row in raw if row["family"] == condition_family]
    models = {}
    for model_name in (first_model, second_model):
        model_rows = [row for row in selected if row["model_name"] == model_name]
        if not model_rows:
            return None
        per_seed = {}
        severities = sorted({float(row["severity"]) for row in model_rows})
        for seed in sorted({int(row["training_seed"]) for row in model_rows}):
            seed_rows = [row for row in model_rows if int(row["training_seed"]) == seed]
            episodes = sorted({str(row["trajectory_id"]) for row in seed_rows})
            matrix = np.empty((len(severities), len(episodes)), dtype=float)
            for severity_index, severity in enumerate(severities):
                lookup = {
                    str(row["trajectory_id"]): float(row[metric])
                    for row in seed_rows
                    if float(row["severity"]) == severity
                }
                if set(lookup) != set(episodes):
                    raise ValueError(f"Incomplete paired {metric} surface for {contrast}")
                matrix[severity_index] = [lookup[episode] for episode in episodes]
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"Non-finite paired {metric} endpoint for {contrast}")
            per_seed[seed] = (episodes, matrix)
        models[model_name] = (severities, per_seed)
    first_severities, first = models[first_model]
    second_severities, second = models[second_model]
    if first_severities != second_severities or set(first) != set(second):
        raise ValueError(f"Unpaired scalar endpoint grid for {contrast}")
    seeds = sorted(first)
    paired = {}
    first_means = []
    second_means = []
    for seed in seeds:
        first_episodes, first_matrix = first[seed]
        second_episodes, second_matrix = second[seed]
        if first_episodes != second_episodes:
            raise ValueError(f"Episode mismatch for {contrast}")
        paired[seed] = np.stack([first_matrix, second_matrix], axis=0)
        first_means.append(float(np.mean(first_matrix)))
        second_means.append(float(np.mean(second_matrix)))
    estimate = float(np.mean(np.asarray(first_means) - np.asarray(second_means)))
    paired_seed_differences = np.asarray(first_means) - np.asarray(second_means)
    exact_seed_p = _exact_paired_seed_sign_flip_p(paired_seed_differences)
    digest = hashlib.sha256(f"{contrast}|{metric}".encode()).digest()
    rng = np.random.default_rng(
        (int(analysis_seed) + int.from_bytes(digest[:4], "big")) % (2**32)
    )
    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        differences = []
        for sampled_seed in sampled_seeds:
            matrix = paired[int(sampled_seed)]
            episode_indices = rng.integers(0, matrix.shape[2], size=matrix.shape[2])
            sampled = matrix[:, :, episode_indices]
            differences.append(float(sampled[0].mean() - sampled[1].mean()))
        draws[replicate] = float(np.mean(differences))
    nonpositive = int(np.count_nonzero(draws <= 0.0))
    nonnegative = int(np.count_nonzero(draws >= 0.0))
    output = {
        "contrast": contrast,
        "condition_family": condition_family,
        "metric": metric,
        "first_model": first_model,
        "second_model": second_model,
        "first_mean": float(np.mean(first_means)),
        "second_mean": float(np.mean(second_means)),
        "estimate_first_minus_second": estimate,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_sign_p_raw": min(
            1.0,
            2.0 * (min(nonpositive, nonnegative) + 1.0) / (replicates + 1.0),
        ),
        "paired_seed_exact_sign_flip_p_raw": exact_seed_p,
        "bootstrap_replicates": replicates,
        "pairing": "seed then episode; both models and every registered severity retained",
        "paired_training_seeds": len(seeds),
        "paired_episodes_per_seed": int(next(iter(paired.values())).shape[2]),
        "practical_effect_threshold_first_minus_second": practical_threshold,
        "practical_effect_met": (
            bool(estimate <= practical_threshold)
            if practical_threshold is not None
            else None
        ),
    }
    return output


def _mechanical_endpoint_contrasts(
    raw: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}))
    replicates = int(evaluation.get("bootstrap_replicates", 10_000))
    seed = int(evaluation.get("analysis_seed", 20261105))
    failure_rows = []
    recovery_rows = []
    for name, first, second, _ in PRIMARY_CONTRASTS:
        failure = _paired_scalar_endpoint_contrast(
            raw,
            contrast=f"{name}__failure_rate",
            first_model=first,
            second_model=second,
            condition_family="iid_dropout",
            metric="failure",
            replicates=replicates,
            analysis_seed=seed,
            practical_threshold=-0.05,
        )
        recovery = _paired_scalar_endpoint_contrast(
            raw,
            contrast=f"{name}__restricted_recovery_seconds",
            first_model=first,
            second_model=second,
            condition_family="controlled_single_burst",
            metric="recovery_seconds",
            replicates=replicates,
            analysis_seed=seed,
            practical_threshold=None,
        )
        if failure is not None:
            failure_rows.append(failure)
        if recovery is not None:
            recovery_rows.append(recovery)
    return {
        "secondary_failure_family": _holm(
            failure_rows, "secondary_mechanism_failure_rate_three_contrasts"
        ),
        "secondary_recovery_family": _holm(
            recovery_rows, "secondary_mechanism_restricted_recovery_seconds_three_contrasts"
        ),
        "role": "secondary mechanical endpoints; primary inference remains IID packet-drop nAURC",
        "recovery_censoring": (
            "unrecovered controlled-burst episodes retain the registered remaining "
            "duration in seconds (steps multiplied by fixed plant dt)"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _safe_component(value: Any) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))


def _write_step_shard(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write one estimator/condition shard once, then return its audit identity."""
    if not rows:
        raise ValueError("Cannot write an empty per-step shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite per-step shard: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                trajectory_id=np.asarray([row["trajectory_id"] for row in rows], dtype="U32"),
                episode_seed=np.asarray([row["episode_seed"] for row in rows], dtype=np.int64),
                schedule_seed=np.asarray([row["schedule_seed"] for row in rows], dtype=np.int64),
                step=np.asarray([row["step"] for row in rows], dtype=np.int32),
                clean_state_and_reference=np.stack([row["clean_state_and_reference"] for row in rows]),
                post_step_clean_state_and_reference=np.stack(
                    [row["post_step_clean_state_and_reference"] for row in rows]
                ),
                plant_advanced=np.asarray(
                    [row["plant_advanced"] for row in rows], dtype=bool
                ),
                measured_state_and_reference=np.stack([row["measured_state_and_reference"] for row in rows]),
                sensor_mask=np.stack([row["sensor_mask"] for row in rows]),
                channel_age=np.stack([row["channel_age"] for row in rows]),
                predicted_state=np.stack([row["predicted_state"] for row in rows]),
                applied_action=np.stack([row["applied_action"] for row in rows]),
                applied_joint_torque=np.stack([row["applied_joint_torque"] for row in rows]),
                observed_timestamp_dt=np.asarray([row["observed_timestamp_dt"] for row in rows], dtype=np.float32),
                nominal_plant_dt=np.asarray([row["nominal_plant_dt"] for row in rows], dtype=np.float32),
                delay_steps=np.asarray([row["delay_steps"] for row in rows], dtype=np.int16),
                delayed_source_step=np.asarray([row["delayed_source_step"] for row in rows], dtype=np.int32),
                tracking_error=np.asarray([row["tracking_error"] for row in rows], dtype=np.float32),
                control_effort=np.asarray([row["control_effort"] for row in rows], dtype=np.float32),
                actuator_saturated=np.asarray([row["actuator_saturated"] for row in rows], dtype=bool),
                failure=np.asarray([row["failure"] for row in rows], dtype=bool),
                failure_reason=np.asarray([row["failure_reason"] for row in rows], dtype="U96"),
                latency_ms=np.asarray([row["latency_ms"] for row in rows], dtype=np.float64),
                ode_solver_function_evaluations=np.asarray(
                    [row["ode_solver_function_evaluations"] for row in rows],
                    dtype=np.int16,
                ),
                ltc_internal_unfolds=np.asarray(
                    [row["ltc_internal_unfolds"] for row in rows], dtype=np.int16
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    episode_ids = {row["trajectory_id"] for row in rows}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "step_rows": len(rows),
        "episode_count": len(episode_ids),
        "format": "compressed_npz_no_pickle",
    }


_STEP_SHARD_FIELDS = {
    "trajectory_id",
    "episode_seed",
    "schedule_seed",
    "step",
    "clean_state_and_reference",
    "post_step_clean_state_and_reference",
    "plant_advanced",
    "measured_state_and_reference",
    "sensor_mask",
    "channel_age",
    "predicted_state",
    "applied_action",
    "applied_joint_torque",
    "observed_timestamp_dt",
    "nominal_plant_dt",
    "delay_steps",
    "delayed_source_step",
    "tracking_error",
    "control_effort",
    "actuator_saturated",
    "failure",
    "failure_reason",
    "latency_ms",
    "ode_solver_function_evaluations",
    "ltc_internal_unfolds",
}


def _validate_evaluation_shard(
    shard_directory: Path,
    *,
    config: dict[str, Any],
    run_identity_sha256: str,
    condition: dict[str, Any],
    estimator: dict[str, Any],
    test_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Deeply verify CSV/NPZ contents, not only rehashable wrapper metadata."""
    manifest_path = shard_directory / "manifest.json"
    episode_file = shard_directory / "episode_metrics.csv"
    step_file = shard_directory / "step_log.npz"
    if not all(path.is_file() for path in (manifest_path, episode_file, step_file)):
        raise ValueError(f"Incomplete evaluation shard: {shard_directory.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in manifest.items()
        if key != "shard_manifest_sha256"
    }
    if canonical_json_hash(core) != manifest.get("shard_manifest_sha256"):
        raise ValueError(f"Shard manifest hash mismatch: {shard_directory.name}")
    expected_manifest = {
        "run_identity_sha256": run_identity_sha256,
        "condition_id": condition["condition_id"],
        "model_name": estimator["model_name"],
        "training_seed": int(estimator["training_seed"]),
        "episode_count": len(test_rows),
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ValueError(f"Shard identity mismatch for {key}: {shard_directory.name}")
    if sha256_file(episode_file) != manifest.get("episode_metrics_sha256"):
        raise ValueError(f"Episode shard hash mismatch: {shard_directory.name}")
    if sha256_file(step_file) != manifest.get("step_log_sha256"):
        raise ValueError(f"Step shard hash mismatch: {shard_directory.name}")
    with episode_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_by_id = {str(row["trajectory_id"]): row for row in test_rows}
    if len(rows) != len(test_rows) or {
        str(row.get("trajectory_id")) for row in rows
    } != set(expected_by_id):
        raise ValueError(f"Episode identities differ from test commitment: {shard_directory.name}")
    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in row_by_id:
            raise ValueError(f"Duplicate episode row: {trajectory_id}")
        expected_episode = expected_by_id[trajectory_id]
        expected_schedule_seed = _paired_schedule_seed(
            int(expected_episode["schedule_seed"]), condition["condition_id"]
        )
        checks = {
            "model_name": str(estimator["model_name"]),
            "training_seed": int(estimator["training_seed"]),
            "condition_id": str(condition["condition_id"]),
            "family": str(condition["family"]),
            "episode_seed": int(expected_episode["episode_seed"]),
            "schedule_seed": int(expected_schedule_seed),
        }
        for key, value in checks.items():
            observed = int(row[key]) if isinstance(value, int) else str(row[key])
            if observed != value:
                raise ValueError(f"Episode shard mismatch for {key}: {trajectory_id}")
        if not math.isclose(
            float(row["severity"]), float(condition["severity"]), rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError(f"Episode shard severity mismatch: {trajectory_id}")
        if estimator.get("checkpoint_sha256") in (None, ""):
            if row.get("checkpoint_sha256") not in (None, "", "None"):
                raise ValueError(f"Baseline shard has checkpoint identity: {trajectory_id}")
        elif row.get("checkpoint_sha256") != estimator["checkpoint_sha256"]:
            raise ValueError(f"Episode checkpoint mismatch: {trajectory_id}")
        for metric in (
            "tracking_rmse",
            "tracking_p95",
            "failure",
            "control_effort",
            "saturation_fraction",
        ):
            if not np.isfinite(float(row[metric])):
                raise ValueError(f"Non-finite episode metric {metric}: {trajectory_id}")
        row_by_id[trajectory_id] = row
    with np.load(step_file, allow_pickle=False) as arrays:
        if set(arrays.files) != _STEP_SHARD_FIELDS:
            raise ValueError(f"Step shard schema mismatch: {shard_directory.name}")
        lengths = {name: int(np.asarray(arrays[name]).shape[0]) for name in arrays.files}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Step shard arrays have unequal lengths: {shard_directory.name}")
        step_rows = next(iter(lengths.values()))
        if step_rows != int(manifest.get("step_rows", -1)):
            raise ValueError(f"Step shard row count mismatch: {shard_directory.name}")
        trajectory_ids = np.asarray(arrays["trajectory_id"]).astype(str)
        if set(trajectory_ids.tolist()) != set(expected_by_id):
            raise ValueError(f"Step shard episode set mismatch: {shard_directory.name}")
        episode_seeds = np.asarray(arrays["episode_seed"], dtype=np.int64)
        schedule_seeds = np.asarray(arrays["schedule_seed"], dtype=np.int64)
        steps = np.asarray(arrays["step"], dtype=np.int64)
        clean_values = np.asarray(arrays["clean_state_and_reference"], dtype=np.float64)
        post_clean_values = np.asarray(
            arrays["post_step_clean_state_and_reference"], dtype=np.float64
        )
        plant_advanced = np.asarray(arrays["plant_advanced"], dtype=bool)
        measured_values = np.asarray(
            arrays["measured_state_and_reference"], dtype=np.float64
        )
        masks = np.asarray(arrays["sensor_mask"], dtype=np.float64)
        ages = np.asarray(arrays["channel_age"], dtype=np.float64)
        predicted = np.asarray(arrays["predicted_state"], dtype=np.float64)
        applied_actions = np.asarray(arrays["applied_action"], dtype=np.float64)
        applied_torques = np.asarray(
            arrays["applied_joint_torque"], dtype=np.float64
        )
        observed_dts = np.asarray(arrays["observed_timestamp_dt"], dtype=np.float64)
        nominal_dts = np.asarray(arrays["nominal_plant_dt"], dtype=np.float64)
        tracking_errors = np.asarray(arrays["tracking_error"], dtype=np.float64)
        effort = np.asarray(arrays["control_effort"], dtype=np.float64)
        saturation = np.asarray(arrays["actuator_saturated"], dtype=bool)
        failures = np.asarray(arrays["failure"], dtype=bool)
        failure_reasons = np.asarray(arrays["failure_reason"]).astype(str)
        latencies = np.asarray(arrays["latency_ms"], dtype=np.float64)
        ode_nfe = np.asarray(
            arrays["ode_solver_function_evaluations"], dtype=np.int64
        )
        ltc_unfolds = np.asarray(arrays["ltc_internal_unfolds"], dtype=np.int64)
        for name, values in (
            ("clean", clean_values),
            ("post_clean", post_clean_values),
            ("measured", measured_values),
            ("mask", masks),
            ("age", ages),
            ("observed_dt", observed_dts),
            ("nominal_dt", nominal_dts),
            ("tracking_error", tracking_errors),
            ("control_effort", effort),
            ("latency", latencies),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite {name} in step shard")
        if np.any(observed_dts <= 0.0) or np.any(nominal_dts <= 0.0):
            raise ValueError("Step shard contains a nonpositive interval")
        for trajectory_id, expected_episode in expected_by_id.items():
            selected = np.flatnonzero(trajectory_ids == trajectory_id)
            expected_steps = int(float(row_by_id[trajectory_id]["steps"]))
            if len(selected) != expected_steps or not np.array_equal(
                steps[selected], np.arange(expected_steps, dtype=np.int64)
            ):
                raise ValueError(f"Step sequence mismatch: {trajectory_id}")
            expected_schedule_seed = _paired_schedule_seed(
                int(expected_episode["schedule_seed"]), condition["condition_id"]
            )
            if (
                not np.all(episode_seeds[selected] == int(expected_episode["episode_seed"]))
                or not np.all(schedule_seeds[selected] == expected_schedule_seed)
            ):
                raise ValueError(f"Step seed identity mismatch: {trajectory_id}")
            row = row_by_id[trajectory_id]
            replay_environment = _make_environment(dict(config.get("environment", {})))
            try:
                current_clean, _ = replay_environment.reset(
                    seed=int(expected_episode["episode_seed"])
                )
                horizon = int(replay_environment.horizon_steps)
                nominal_dt = float(replay_environment.control_dt)
                if not np.allclose(
                    nominal_dts[selected], nominal_dt, rtol=0.0, atol=1e-7
                ):
                    raise ValueError(f"Nominal plant-dt mismatch: {trajectory_id}")
                schedule = make_perturbation_schedule(
                    horizon_steps=horizon,
                    nominal_dt=nominal_dt,
                    spec=PerturbationSpec.from_mapping(condition["spec"]),
                    seed=int(expected_schedule_seed),
                )
                replay = ObservationPerturber(schedule)
                controller = _make_controller(
                    str(config.get("environment", {}).get("kind", "reacher_v5")),
                    dict(config.get("controller", {})),
                    float(config.get("environment", {}).get("action_limit", 1.0)),
                )
                controller.reset()
                replay_recurrent_state = None
                replay_previous_control = np.zeros(2, dtype=np.float32)
                replay_model = estimator.get("model")
                if replay_model is not None:
                    replay_model.eval()
                    try:
                        replay_device = str(next(replay_model.parameters()).device)
                    except StopIteration:
                        replay_device = "cpu"
                else:
                    replay_device = "cpu"
                    replay_baseline = estimator.get("baseline")
                    if replay_baseline is None:
                        raise ValueError("Evaluation shard estimator is unavailable")
                    replay_baseline.reset()
                final_failure_reason = "none"
                for local_step, position in enumerate(selected):
                    if not np.allclose(
                        clean_values[position], current_clean, rtol=0.0, atol=1e-6
                    ):
                        raise ValueError(
                            f"Pre-step clean plant replay mismatch: {trajectory_id}"
                        )
                    replay_measured, replay_mask, replay_age, replay_dt = replay.observe(
                        np.asarray(current_clean, dtype=np.float32), local_step
                    )
                    if not np.allclose(
                        measured_values[position], replay_measured, rtol=0.0, atol=1e-6
                    ):
                        raise ValueError(f"Measured replay mismatch: {trajectory_id}")
                    if not np.array_equal(masks[position], replay_mask):
                        raise ValueError(f"Sensor-mask replay mismatch: {trajectory_id}")
                    if not np.allclose(
                        ages[position], replay_age, rtol=0.0, atol=1e-6
                    ):
                        raise ValueError(f"Channel-age replay mismatch: {trajectory_id}")
                    if not math.isclose(
                        float(observed_dts[position]),
                        float(replay_dt),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    ):
                        raise ValueError(f"Observed-dt replay mismatch: {trajectory_id}")

                    with torch.no_grad():
                        (
                            replay_prediction,
                            replay_recurrent_state,
                            _,
                        ) = _estimate_state_for_control(
                            estimator,
                            clean=np.asarray(current_clean, dtype=np.float32),
                            measured=np.asarray(replay_measured, dtype=np.float32),
                            mask=np.asarray(replay_mask, dtype=np.float32),
                            age=np.asarray(replay_age, dtype=np.float32),
                            observed_dt=float(replay_dt),
                            previous_control=replay_previous_control,
                            recurrent_state=replay_recurrent_state,
                            device=replay_device,
                        )
                    replay_prediction_is_valid = bool(
                        np.asarray(replay_prediction).shape == (4,)
                        and np.all(np.isfinite(replay_prediction))
                    )
                    prediction_is_finite = bool(np.all(np.isfinite(predicted[position])))
                    if replay_prediction_is_valid != prediction_is_finite:
                        raise ValueError(
                            f"Observer terminal-status replay mismatch: {trajectory_id}"
                        )
                    if replay_prediction_is_valid and not np.allclose(
                        predicted[position], replay_prediction, rtol=1e-6, atol=1e-6
                    ):
                        raise ValueError(
                            f"Observer prediction replay mismatch: {trajectory_id}"
                        )
                    if prediction_is_finite:
                        if not bool(plant_advanced[position]):
                            raise ValueError(
                                f"Finite estimator output without plant advance: {trajectory_id}"
                            )
                        expected_action = controller.control(
                            predicted[position].astype(np.float32),
                            replay_measured.astype(np.float32),
                        )
                        expected_action = np.clip(
                            expected_action * schedule.actuator_gain[local_step],
                            -float(replay_environment.action_limit),
                            float(replay_environment.action_limit),
                        ).astype(np.float32)
                        if not np.allclose(
                            applied_actions[position], expected_action, rtol=0.0, atol=1e-6
                        ):
                            raise ValueError(f"Applied-action replay mismatch: {trajectory_id}")
                        expected_torque = expected_action * float(
                            config.get("controller", {}).get("actuator_gear", 1.0)
                        )
                        if not np.allclose(
                            applied_torques[position], expected_torque, rtol=0.0, atol=1e-4
                        ):
                            raise ValueError(f"Applied-torque replay mismatch: {trajectory_id}")
                        replay_environment.set_external_force(
                            schedule.external_force[local_step]
                        )
                        next_clean, _, terminated, truncated, info = replay_environment.step(
                            expected_action
                        )
                        if not np.allclose(
                            post_clean_values[position], next_clean, rtol=0.0, atol=1e-6
                        ):
                            raise ValueError(
                                f"Post-step clean plant replay mismatch: {trajectory_id}"
                            )
                        active_reasons = [
                            name
                            for name, active in info.get("failure_reasons", {}).items()
                            if active
                        ]
                        derived_failure_reason = ",".join(active_reasons) or "none"
                        derived_failure = bool(info["failure"])
                        physical_metrics = {
                            "tracking_error": float(info["tracking_error"]),
                            "control_effort": float(info["control_effort"]),
                        }
                        for metric_name, derived_value in physical_metrics.items():
                            observed_values = (
                                tracking_errors
                                if metric_name == "tracking_error"
                                else effort
                            )
                            if not math.isclose(
                                float(observed_values[position]),
                                derived_value,
                                rel_tol=1e-6,
                                abs_tol=1e-7,
                            ):
                                raise ValueError(
                                    f"Physical {metric_name} replay mismatch: {trajectory_id}"
                                )
                        if bool(saturation[position]) != bool(
                            info["actuator_saturated"]
                        ):
                            raise ValueError(
                                f"Actuator-saturation replay mismatch: {trajectory_id}"
                            )
                        if bool(failures[position]) != derived_failure:
                            raise ValueError(f"Plant-failure replay mismatch: {trajectory_id}")
                        if str(failure_reasons[position]) != derived_failure_reason:
                            raise ValueError(
                                f"Plant failure-reason replay mismatch: {trajectory_id}"
                            )
                        current_clean = next_clean
                        replay_previous_control = np.asarray(
                            expected_torque, dtype=np.float32
                        )
                        final_failure_reason = derived_failure_reason
                        if (terminated or truncated) and local_step != len(selected) - 1:
                            raise ValueError(
                                f"Step rows continue after plant termination: {trajectory_id}"
                            )
                        if (
                            local_step == len(selected) - 1
                            and not (terminated or truncated)
                        ):
                            raise ValueError(
                                f"Step shard stops before a registered terminal event: {trajectory_id}"
                            )
                    else:
                        if local_step != len(selected) - 1:
                            raise ValueError(
                                f"Rows continue after estimator failure: {trajectory_id}"
                            )
                        if bool(plant_advanced[position]):
                            raise ValueError(
                                f"Estimator failure incorrectly advances plant: {trajectory_id}"
                            )
                        if not bool(failures[position]) or str(
                            failure_reasons[position]
                        ) != "estimator_nonfinite_or_invalid_shape":
                            raise ValueError(
                                f"Invalid estimator-failure provenance: {trajectory_id}"
                            )
                        if not np.allclose(
                            post_clean_values[position], current_clean, rtol=0.0, atol=1e-6
                        ):
                            raise ValueError(
                                f"Estimator-failure post-state mismatch: {trajectory_id}"
                            )
                        if np.any(np.isfinite(applied_actions[position])) or np.any(
                            np.isfinite(applied_torques[position])
                        ):
                            raise ValueError(
                                f"Estimator failure contains an applied action: {trajectory_id}"
                            )
                        current_error = float(
                            np.linalg.norm(
                                replay_environment.get_body_com("fingertip")[:2]
                                - replay_environment.get_body_com("target")[:2]
                            )
                        )
                        if not math.isclose(
                            float(tracking_errors[position]),
                            current_error,
                            rel_tol=1e-6,
                            abs_tol=1e-7,
                        ):
                            raise ValueError(
                                f"Estimator-failure tracking replay mismatch: {trajectory_id}"
                            )
                        if float(effort[position]) != 0.0 or bool(saturation[position]):
                            raise ValueError(
                                f"Estimator failure has physical control effort: {trajectory_id}"
                            )
                        final_failure_reason = "estimator_nonfinite_or_invalid_shape"
                if str(row["failure_reason"]) != final_failure_reason:
                    raise ValueError(f"Episode failure-reason mismatch: {trajectory_id}")
            finally:
                replay_environment.close()
            episode_errors = tracking_errors[selected]
            episode_failed = bool(np.any(failures[selected]))
            penalty_m = float(row["tracking_failure_penalty_m"])
            penalty_steps = horizon - len(selected) if episode_failed else 0
            if int(float(row["tracking_penalty_steps"])) != penalty_steps:
                raise ValueError(f"Failure penalty length mismatch: {trajectory_id}")
            scored = (
                np.concatenate(
                    [episode_errors, np.full(penalty_steps, penalty_m, dtype=float)]
                )
                if penalty_steps
                else episode_errors
            )
            applied_control = np.asarray(plant_advanced[selected], dtype=bool)
            applied_control_steps = int(np.count_nonzero(applied_control))
            derived = {
                "tracking_scored_steps": float(len(scored)),
                "tracking_rmse": float(np.sqrt(np.mean(np.square(scored)))),
                "tracking_p95": float(np.quantile(scored, 0.95)),
                "failure": float(episode_failed),
                "control_effort": float(np.sum(effort[selected])),
                "saturation_fraction": float(
                    np.mean(saturation[selected][applied_control])
                    if applied_control_steps
                    else 0.0
                ),
                "control_observed_steps": float(applied_control_steps),
                "control_observed_fraction": float(
                    applied_control_steps / horizon
                ),
                "plant_advanced_steps": float(applied_control_steps),
                "plant_completion_fraction": float(
                    applied_control_steps / horizon
                ),
                "latency_median_ms": float(np.median(latencies[selected])),
                "latency_p95_ms": float(np.quantile(latencies[selected], 0.95)),
                "masked_fraction": float(1.0 - schedule.sensor_mask.mean()),
                "burst_masked_steps": float(
                    np.count_nonzero(~schedule.sensor_mask[:, 0])
                ),
                "burst_run_count": float(
                    len(packet_loss_run_lengths(schedule.sensor_mask))
                ),
                "burst_run_steps_total": float(
                    sum(packet_loss_run_lengths(schedule.sensor_mask))
                ),
                "ode_solver_function_evaluations": float(np.sum(ode_nfe[selected])),
                "ltc_internal_unfolds": float(np.sum(ltc_unfolds[selected])),
            }
            finite_state = np.all(np.isfinite(predicted[selected]), axis=1)
            derived["state_accuracy_finite_steps"] = float(
                np.count_nonzero(finite_state)
            )
            derived["state_accuracy_finite_fraction"] = float(
                np.count_nonzero(finite_state) / horizon
            )
            derived_state_rmse = (
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                predicted[selected][finite_state]
                                - clean_values[selected, :4][finite_state]
                            )
                        )
                    )
                )
                if np.any(finite_state)
                else math.nan
            )
            if np.any(finite_state):
                state_delta = (
                    predicted[selected][finite_state]
                    - clean_values[selected, :4][finite_state]
                )
                derived_joint_position_rmse = float(
                    np.sqrt(np.mean(np.square(state_delta[:, :2])))
                )
                derived_joint_velocity_rmse = float(
                    np.sqrt(np.mean(np.square(state_delta[:, 2:4])))
                )
            else:
                derived_joint_position_rmse = math.nan
                derived_joint_velocity_rmse = math.nan
            for key, value in (
                ("state_rmse", derived_state_rmse),
                ("joint_position_rmse", derived_joint_position_rmse),
                ("joint_velocity_rmse", derived_joint_velocity_rmse),
            ):
                observed = float(row[key])
                if np.isfinite(value) != np.isfinite(observed) or (
                    np.isfinite(value)
                    and not math.isclose(
                        value,
                        observed,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                ):
                    raise ValueError(f"Derived {key} mismatch: {trajectory_id}")
            if condition["family"] == "controlled_single_burst":
                recovery = _recovery_steps(
                    episode_errors,
                    np.any(masks[selected] < 0.5, axis=1),
                    scheduled_disrupted=np.asarray(
                        ~schedule.sensor_mask[:, 0], dtype=bool
                    ),
                    floor_m=float(config.get("evaluation", {}).get("recovery_floor_m", 0.02)),
                    baseline_multiplier=float(config.get("evaluation", {}).get("recovery_baseline_multiplier", 1.0)),
                    baseline_window_steps=int(config.get("evaluation", {}).get("recovery_baseline_window_steps", 50)),
                    consecutive=int(config.get("evaluation", {}).get("recovery_consecutive_steps", 10)),
                    horizon_steps=horizon,
                )
                for key, value in (
                    ("recovery_steps", recovery[0]),
                    ("recovery_seconds", recovery[0] * float(nominal_dts[selected][0])),
                    ("recovery_event_count", recovery[2]),
                    ("recovery_censored_events", recovery[3]),
                    ("recovery_threshold_mean_m", recovery[4]),
                ):
                    derived[key] = float(value)
            for key, value in derived.items():
                if not math.isclose(
                    float(row[key]), float(value), rel_tol=1e-6, abs_tol=1e-6
                ):
                    raise ValueError(
                        f"Derived episode metric mismatch for {key}: {trajectory_id}"
                    )
    step_identity = {
        "path": str(step_file),
        "sha256": manifest["step_log_sha256"],
        "step_rows": int(manifest["step_rows"]),
        "episode_count": int(manifest["episode_count"]),
        "format": "compressed_npz_no_pickle",
        "condition_id": condition["condition_id"],
        "model_name": estimator["model_name"],
        "training_seed": int(estimator["training_seed"]),
    }
    return rows, step_identity, manifest


def _profile_batch1_latency(
    estimators: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    device: str,
) -> list[dict[str, Any]]:
    """Dedicated synchronized observer-only latency endpoint.

    Inputs are preallocated on the target device; tensor construction, host to
    device transfer, PD control and MuJoCo stepping are deliberately excluded
    and the endpoint is reported separately from end-to-end plant throughput.
    """
    evaluation = dict(config.get("evaluation", {}))
    warmup = int(evaluation.get("latency_warmup_calls", 1_000))
    calls = int(evaluation.get("latency_measured_calls", 10_000))
    if warmup < 0 or calls <= 0:
        raise ValueError("Latency warm-up/call counts are invalid")
    rows = []
    for estimator in estimators:
        model = estimator.get("model")
        baseline = estimator.get("baseline")
        durations = np.empty(calls, dtype=np.float64)
        if model is not None:
            x = torch.zeros((1, 1, int(config.get("models", {}).get("input_dim", 19))), dtype=torch.float32, device=device)
            dt = torch.full((1, 1), 0.02, dtype=torch.float32, device=device)
            state = None
            with torch.no_grad():
                for _ in range(warmup):
                    _, state = model.forward_sequence(x, dt, state)
                _synchronize(device)
                for index in range(calls):
                    started = time.perf_counter_ns()
                    _, state = model.forward_sequence(x, dt, state)
                    _synchronize(device)
                    durations[index] = (time.perf_counter_ns() - started) / 1e6
        else:
            clean = np.zeros(8, dtype=np.float32)
            measured = clean.copy()
            mask = np.ones(4, dtype=np.float32)
            baseline.reset()
            for _ in range(warmup):
                baseline.estimate(clean, measured, mask, 0.02)
            for index in range(calls):
                started = time.perf_counter_ns()
                baseline.estimate(clean, measured, mask, 0.02)
                durations[index] = (time.perf_counter_ns() - started) / 1e6
        rows.append(
            {
                "model_name": estimator["model_name"],
                "training_seed": int(estimator["training_seed"]),
                "warmup_calls": warmup,
                "measured_calls": calls,
                "latency_median_ms": float(np.median(durations)),
                "latency_p95_ms": float(np.quantile(durations, 0.95)),
                "deadline_20ms_miss_fraction": float(np.mean(durations > 20.0)),
                "scope": "observer-only batch-1 recurrent step; preallocated input; synchronized; excludes transfer, PD and plant",
            }
        )
    return rows


def _verify_completed_evaluation(
    destination: Path,
    state: dict[str, Any],
    *,
    estimators: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify every published file and immutable raw shard before reuse."""
    manifest_path = destination / "evaluation_manifest.json"
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != state.get("evaluation_manifest_sha256")
    ):
        raise ValueError("Completed evaluation manifest identity mismatch")
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_core = {
        key: value
        for key, value in report.items()
        if key != "evaluation_release_sha256"
    }
    if canonical_json_hash(report_core) != report.get("evaluation_release_sha256"):
        raise ValueError("Completed evaluation release-root hash mismatch")
    for name, expected_sha256 in report.get("files", {}).items():
        path = destination / name
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"Completed evaluation file hash mismatch: {name}")
    step_manifest_path = destination / "step_log_manifest.json"
    step_manifest = json.loads(step_manifest_path.read_text(encoding="utf-8"))
    step_core = {
        key: value
        for key, value in step_manifest.items()
        if key != "step_log_manifest_sha256"
    }
    if canonical_json_hash(step_core) != step_manifest.get(
        "step_log_manifest_sha256"
    ):
        raise ValueError("Completed step-log manifest hash mismatch")
    for shard in step_manifest.get("shards", []):
        path = Path(shard["path"])
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Completed raw step-shard hash mismatch: {path}")
    run_identity = state.get("run_identity", {})
    conditions = run_identity.get("conditions")
    test_rows = run_identity.get("test_rows")
    estimator_identities = run_identity.get("estimator_identities")
    if not all(isinstance(value, list) for value in (conditions, test_rows, estimator_identities)):
        raise ValueError("Completed evaluation lacks deep-resume commitments")
    expected_shards: set[str] = set()
    estimator_lookup = {
        (str(estimator["model_name"]), int(estimator["training_seed"])): estimator
        for estimator in estimators
    }
    for condition in conditions:
        for estimator_identity in estimator_identities:
            identity = (
                str(estimator_identity["model_name"]),
                int(estimator_identity["training_seed"]),
            )
            estimator = estimator_lookup.get(identity)
            if estimator is None:
                raise ValueError(f"Completed evaluation estimator is unavailable: {identity}")
            shard_name = (
                f"{_safe_component(condition['condition_id'])}__"
                f"{_safe_component(estimator['model_name'])}__"
                f"seed_{int(estimator['training_seed'])}"
            )
            expected_shards.add(shard_name)
            _validate_evaluation_shard(
                destination / "shards" / shard_name,
                config=run_identity["config"],
                run_identity_sha256=state["run_identity_sha256"],
                condition=condition,
                estimator=estimator,
                test_rows=test_rows,
            )
    observed_shards = {
        path.name for path in (destination / "shards").iterdir() if path.is_dir()
    }
    if observed_shards != expected_shards:
        raise ValueError("Completed evaluation shard set is incomplete or unexpected")
    return report


def evaluate_response_surface(
    *,
    config: dict[str, Any],
    data_root: str | Path,
    checkpoints: list[str | Path],
    output_dir: str | Path,
    device: str = "cpu",
    resume: bool = False,
    expected_master_release_sha256: str | None = None,
    expected_test_release_sha256: str | None = None,
    test_claim_custody_receipt_path: str | Path | None = None,
    expected_test_claim_custody_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    if destination.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite evaluation: {destination}")
    data_directory = Path(data_root).resolve()
    verify_split_datasets(data_directory, expected_config=config)
    manifest = load_trajectory_manifest(
        data_directory / "trajectory_manifest.json", expected_config=config
    )
    current_config_sha256 = canonical_json_hash(config)
    if manifest.get("source_config_sha256") != current_config_sha256:
        raise ValueError(
            "Current evaluation config differs from the frozen trajectory source"
        )
    require_complete = bool(
        config.get("evaluation", {}).get(
            "require_complete_registered_checkpoint_set", False
        )
    )
    maximum = config.get("evaluation", {}).get("max_test_episodes")
    if require_complete and maximum is not None:
        raise ValueError(
            "Confirmatory evaluation forbids max_test_episodes development override"
        )
    expected_test_episodes = int(config.get("splits", {}).get("test", {}).get("episodes", 0))
    conditions = build_response_conditions(config)
    from .formal import (
        apply_and_validate_precision_policy,
        claim_test_release,
        current_run_accelerator_identity,
        is_formal,
        validate_checkpoint_release,
        validate_test_release,
    )

    checkpoint_release = validate_checkpoint_release(
        config,
        manifest,
        checkpoints,
        expected_master_release_sha256=expected_master_release_sha256,
    )
    test_release = validate_test_release(
        config,
        checkpoint_release,
        expected_master_release_sha256=expected_master_release_sha256,
        expected_test_release_sha256=expected_test_release_sha256,
        evaluation_output_dir=destination,
        test_claim_custody_receipt_path=test_claim_custody_receipt_path,
        expected_test_claim_custody_receipt_sha256=(
            expected_test_claim_custody_receipt_sha256
        ),
    )
    estimators: list[dict[str, Any]] = []
    apply_and_validate_precision_policy(config)
    evaluation_runtime = current_run_accelerator_identity(
        device,
        require_physical_mapping=is_formal(config),
        config=config,
    )
    for name in config.get("evaluation", {}).get(
        "nonlearned_baselines", ["full_state_pd", "zoh_pd", "kalman_pd"]
    ):
        estimators.append(
            {
                "model_name": str(name),
                "training_seed": -1,
                "baseline": build_nonlearned_baseline(
                    str(name), config.get("evaluation", {}).get("kalman") if name == "kalman_pd" else None
                ),
                "checkpoint_sha256": None,
            }
        )
    learned_identities: set[tuple[str, int]] = set()
    registry_hashes: set[str] = set()
    formal_identity_values: dict[str, set[str]] = defaultdict(set)
    for checkpoint_path in checkpoints:
        model, metadata = load_trained_observer(
            checkpoint_path,
            device=device,
            expected_config=config,
            expected_trajectory_manifest_sha256=manifest["manifest_sha256"],
        )
        learned_identity = (metadata["registered_name"], int(metadata["seed"]))
        if learned_identity in learned_identities:
            raise ValueError(f"Duplicate checkpoint identity: {learned_identity}")
        learned_identities.add(learned_identity)
        registry_hashes.add(str(metadata["parameter_registry_sha256"]))
        for key in (
            "environment_sha256",
            "controller_selection_sha256",
            "tuning_selection_sha256",
        ):
            value = metadata.get(key)
            if value is not None:
                formal_identity_values[key].add(str(value))
        estimators.append(
            {
                "model_name": metadata["registered_name"],
                "training_seed": int(metadata["seed"]),
                "model": model,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
    if len(registry_hashes) > 1:
        raise ValueError("Evaluation checkpoints use different parameter registries")
    if any(len(values) > 1 for values in formal_identity_values.values()):
        raise ValueError("Evaluation checkpoints use inconsistent formal artifacts")
    if require_complete:
        expected_identities = {
            (str(model_name), int(seed))
            for model_name in config.get("matching", {}).get("registered_models", [])
            for seed in config.get("training", {}).get("seeds", [])
        }
        if learned_identities != expected_identities:
            missing = sorted(expected_identities - learned_identities)
            unexpected = sorted(learned_identities - expected_identities)
            raise ValueError(
                "Confirmatory checkpoint set is incomplete or unexpected; "
                f"missing={missing}, unexpected={unexpected}"
            )
    # Test trajectory IDs and perturbation schedules are materialized only
    # after the exact checkpoint set and, in formal mode, both release
    # artifacts have passed.  This ordering prevents partial training runs from
    # probing any registered test condition.
    test_rows = list(manifest["splits"]["test"])
    if maximum is not None:
        test_rows = test_rows[: int(maximum)]
    if require_complete and len(test_rows) != expected_test_episodes:
        raise ValueError("Confirmatory test episode count differs from frozen config")
    run_identity = {
        "schema_version": 1,
        "config_sha256": current_config_sha256,
        "config": config,
        "trajectory_manifest_sha256": manifest["manifest_sha256"],
        "conditions_sha256": canonical_json_hash(conditions),
        "conditions": conditions,
        "test_rows_sha256": canonical_json_hash(test_rows),
        "test_rows": test_rows,
        "estimator_identities": sorted(
            [
                {
                    "model_name": estimator["model_name"],
                    "training_seed": int(estimator["training_seed"]),
                    "checkpoint_sha256": estimator["checkpoint_sha256"],
                }
                for estimator in estimators
            ],
            key=lambda row: (row["model_name"], row["training_seed"]),
        ),
        "evaluation_runtime": evaluation_runtime,
        "formal_artifact_identities": {
            key: next(iter(values))
            for key, values in sorted(formal_identity_values.items())
        },
        "checkpoint_release_sha256": (
            checkpoint_release["checkpoint_release_sha256"]
            if checkpoint_release is not None
            else None
        ),
        "test_release_sha256": (
            test_release["test_release_sha256"]
            if test_release is not None
            else None
        ),
        "test_authorization_design_sha256": (
            test_release["test_authorization_design_sha256"]
            if test_release is not None
            else None
        ),
        "test_claim_custody_receipt_sha256": (
            test_release["test_claim_custody_receipt_sha256"]
            if test_release is not None
            else None
        ),
        "test_claim_custody_receipt_file_sha256": (
            test_release["test_claim_custody_receipt_file_sha256"]
            if test_release is not None
            else None
        ),
    }
    run_identity_sha256 = canonical_json_hash(run_identity)
    test_release_claim = claim_test_release(
        config,
        test_release,
        output_dir=destination,
        run_identity_sha256=run_identity_sha256,
        resume=resume,
    )
    claim_recovery_after_preoutput_crash = False
    if resume:
        state_path = destination / "evaluation_state.json"
        recovered_preoutput_crash = False
        if not state_path.is_file():
            # The external one-shot claim is written before the local output.
            # A crash in that narrow window is recoverable only by an explicit
            # resume with the same externally bound output/run identity.
            if destination.exists() or test_release_claim is None:
                raise FileNotFoundError(
                    "Evaluation resume found no recoverable state; only a formal "
                    "externally claimed pre-output crash may create it"
                )
            destination.mkdir(parents=True, exist_ok=False)
            atomic_write_json(
                state_path,
                {
                    "status": "running",
                    "run_identity": run_identity,
                    "run_identity_sha256": run_identity_sha256,
                    "completed_shards": [],
                    "claim_recovery_after_preoutput_crash": True,
                },
            )
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
            recovered_preoutput_crash = True
            claim_recovery_after_preoutput_crash = True
        else:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        if prior_state.get("run_identity_sha256") != run_identity_sha256:
            raise ValueError("Evaluation resume identity mismatch")
        if prior_state.get("status") == "complete":
            return _verify_completed_evaluation(
                destination, prior_state, estimators=estimators
            )
        if prior_state.get("status") != "running":
            raise ValueError("Evaluation resume state is malformed")
        if not recovered_preoutput_crash:
            raise RuntimeError(
                "A formal evaluation that created local running state may not "
                "resume: the one-shot test authorization is consumed. Only the "
                "externally claimed pre-output crash window is recoverable"
            )
    else:
        destination.mkdir(parents=True, exist_ok=False)
        atomic_write_json(
            destination / "evaluation_state.json",
            {
                "status": "running",
                "run_identity": run_identity,
                "run_identity_sha256": run_identity_sha256,
                "completed_shards": [],
                "claim_recovery_after_preoutput_crash": False,
            },
        )
    # This is the first operation that materializes registered perturbation
    # schedules, after the exact checkpoint release, one-time test claim and
    # fail-closed resume gate. Once local running state exists, an interruption
    # consumes the authorization rather than rerunning any latency or episode.
    burst_realization_audit = _audit_burst_realization(
        config, conditions, test_rows
    )
    (destination / "shards").mkdir(exist_ok=True)
    latency_rows = _profile_batch1_latency(
        estimators, config, device=device
    )
    raw: list[dict[str, Any]] = []
    step_shards: list[dict[str, Any]] = []
    completed_shards: list[dict[str, Any]] = []
    for condition in conditions:
        for estimator in estimators:
            shard_name = (
                f"{_safe_component(condition['condition_id'])}__"
                f"{_safe_component(estimator['model_name'])}__"
                f"seed_{int(estimator['training_seed'])}"
            )
            shard_directory = destination / "shards" / shard_name
            if shard_directory.exists():
                raise ValueError(
                    "An evaluation shard already exists before its single write; "
                    "refusing to adopt, overwrite or rerun test work"
                )
            shard_steps: list[dict[str, Any]] = []
            shard_episode_rows: list[dict[str, Any]] = []
            for episode in test_rows:
                schedule_seed = _paired_schedule_seed(
                    int(episode["schedule_seed"]), condition["condition_id"]
                )
                result = _rollout(
                    estimator,
                    config,
                    condition,
                    episode_seed=int(episode["episode_seed"]),
                    schedule_seed=schedule_seed,
                    device=device,
                )
                steps = result.pop("_step_log")
                for step_row in steps:
                    step_row.update(
                        {
                            "trajectory_id": episode["trajectory_id"],
                            "episode_seed": int(episode["episode_seed"]),
                            "schedule_seed": int(schedule_seed),
                        }
                    )
                shard_steps.extend(steps)
                episode_result = {
                        **result,
                        "trajectory_id": episode["trajectory_id"],
                        "model_name": estimator["model_name"],
                        "training_seed": estimator["training_seed"],
                        "checkpoint_sha256": estimator["checkpoint_sha256"],
                        "condition_id": condition["condition_id"],
                        "family": condition["family"],
                        "severity": condition["severity"],
                    }
                raw.append(episode_result)
                shard_episode_rows.append(episode_result)
            staging = destination / "shards" / f".{shard_name}.staging-{uuid.uuid4().hex}"
            staging.mkdir(parents=False, exist_ok=False)
            episode_file = staging / "episode_metrics.csv"
            step_file = staging / "step_log.npz"
            _write_csv(episode_file, shard_episode_rows)
            step_identity = _write_step_shard(step_file, shard_steps)
            shard_core = {
                "schema_version": 1,
                "run_identity_sha256": run_identity_sha256,
                "condition_id": condition["condition_id"],
                "model_name": estimator["model_name"],
                "training_seed": int(estimator["training_seed"]),
                "episode_count": len(shard_episode_rows),
                "step_rows": int(step_identity["step_rows"]),
                "episode_metrics_sha256": sha256_file(episode_file),
                "step_log_sha256": step_identity["sha256"],
            }
            shard_manifest = {
                **shard_core,
                "shard_manifest_sha256": canonical_json_hash(shard_core),
            }
            atomic_write_json(staging / "manifest.json", shard_manifest)
            os.replace(staging, shard_directory)
            step_shards.append(
                {
                    **step_identity,
                    "path": str(shard_directory / "step_log.npz"),
                    "condition_id": condition["condition_id"],
                    "model_name": estimator["model_name"],
                    "training_seed": int(estimator["training_seed"]),
                }
            )
            completed_shards.append(
                {
                    "shard_name": shard_name,
                    "shard_manifest_sha256": shard_manifest[
                        "shard_manifest_sha256"
                    ],
                }
            )
            atomic_write_json(
                destination / "evaluation_state.json",
                {
                    "status": "running",
                    "run_identity": run_identity,
                    "run_identity_sha256": run_identity_sha256,
                    "completed_shards": completed_shards,
                    "claim_recovery_after_preoutput_crash": (
                        claim_recovery_after_preoutput_crash
                    ),
                },
            )
    controlled_rows = [
        row for row in raw if row["family"] == "controlled_single_burst"
    ]
    if controlled_rows and any(
        not np.isfinite(float(row["recovery_steps"]))
        or not np.isfinite(float(row["recovery_seconds"]))
        or int(row["recovery_event_count"]) != 1
        for row in controlled_rows
    ):
        raise RuntimeError(
            "Controlled-burst recovery completeness gate failed: every episode "
            "must retain one finite observed-or-right-censored endpoint"
        )
    summary = _summarize(raw, config)
    naurc_by_seed = _naurc_by_seed(raw)
    naurc = _summarize_naurc(raw, naurc_by_seed, config)
    contrasts = _mechanism_contrasts(raw, config)
    mechanical_contrasts = _mechanical_endpoint_contrasts(raw, config)
    expected_rows = len(estimators) * len(conditions) * len(test_rows)
    if len(raw) != expected_rows:
        raise RuntimeError(
            f"Evaluation completeness gate failed: {len(raw)} != {expected_rows}"
        )
    _write_csv(destination / "episode_metrics.csv", raw)
    _write_csv(destination / "condition_summary.csv", summary)
    _write_csv(destination / "tracking_naurc_by_seed.csv", naurc_by_seed)
    _write_csv(destination / "tracking_naurc.csv", naurc)
    _write_csv(destination / "batch1_latency.csv", latency_rows)
    atomic_write_json(destination / "mechanism_contrasts.json", contrasts)
    atomic_write_json(
        destination / "mechanical_endpoint_contrasts.json",
        mechanical_contrasts,
    )
    step_log_core = {
        "schema_version": 1,
        "immutable_write_once_shards": True,
        "shards": step_shards,
        "shard_count": len(step_shards),
        "total_step_rows": int(sum(row["step_rows"] for row in step_shards)),
        "expected_episode_rows": expected_rows,
    }
    step_log_manifest = {
        **step_log_core,
        "step_log_manifest_sha256": canonical_json_hash(step_log_core),
    }
    atomic_write_json(destination / "step_log_manifest.json", step_log_manifest)
    report = {
        "status": "complete",
        "config_sha256": current_config_sha256,
        "trajectory_manifest_sha256": manifest["manifest_sha256"],
        "parameter_registry_sha256": (
            next(iter(registry_hashes)) if registry_hashes else None
        ),
        "formal_artifact_identities": run_identity["formal_artifact_identities"],
        "checkpoint_release_sha256": run_identity[
            "checkpoint_release_sha256"
        ],
        "test_release_sha256": run_identity["test_release_sha256"],
        "test_authorization_design_sha256": run_identity[
            "test_authorization_design_sha256"
        ],
        "test_claim_custody_receipt_sha256": run_identity[
            "test_claim_custody_receipt_sha256"
        ],
        "test_claim_custody_receipt_file_sha256": run_identity[
            "test_claim_custody_receipt_file_sha256"
        ],
        "test_release_state_sha256": (
            test_release_claim["test_release_state_sha256"]
            if test_release_claim is not None
            else None
        ),
        "claim_recovery_after_preoutput_crash": (
            claim_recovery_after_preoutput_crash
        ),
        "evaluation_runtime": evaluation_runtime,
        "checkpoint_identity_gate_passed": True,
        "complete_registered_checkpoint_set_required": require_complete,
        "complete_registered_checkpoint_set_passed": (
            True if require_complete else None
        ),
        "evaluation_row_completeness_gate_passed": len(raw) == expected_rows,
        "controlled_burst_recovery_finite_gate_passed": bool(controlled_rows),
        "models": sorted({row["model_name"] for row in raw}),
        "training_seeds_by_model": {
            name: sorted({int(row["training_seed"]) for row in raw if row["model_name"] == name})
            for name in sorted({row["model_name"] for row in raw})
        },
        "test_episodes": len(test_rows),
        "conditions": conditions,
        "burst_realization_audit": burst_realization_audit,
        "episode_rows": len(raw),
        "formal_batch1_latency_gate_passed": all(
            int(row["measured_calls"])
            == int(config.get("evaluation", {}).get("latency_measured_calls", 10_000))
            for row in latency_rows
        ),
        "per_step_log_gate_passed": (
            len(step_shards) == len(estimators) * len(conditions)
            and step_log_manifest["total_step_rows"]
            == sum(int(row["steps"]) for row in raw)
        ),
        "step_log_manifest_sha256": step_log_manifest[
            "step_log_manifest_sha256"
        ],
        "fixed_checkpoints_no_retraining": True,
        "paired_episode_and_schedule_seeds": True,
        "bootstrap_hierarchy": "training seed then episode",
        "bootstrap_pairing": "models and all severity levels retained within every resample",
        "bootstrap_replicates": int(config.get("evaluation", {}).get("bootstrap_replicates", 10_000)),
        "implementation": source_tree_manifest(Path(__file__).resolve().parents[1]),
        "files": {
            name: sha256_file(destination / name)
            for name in (
                "episode_metrics.csv",
                "condition_summary.csv",
                "tracking_naurc_by_seed.csv",
                "tracking_naurc.csv",
                "batch1_latency.csv",
                "mechanism_contrasts.json",
                "mechanical_endpoint_contrasts.json",
                "step_log_manifest.json",
            )
        },
    }
    report["evaluation_release_sha256"] = canonical_json_hash(report)
    atomic_write_json(destination / "evaluation_manifest.json", report)
    atomic_write_json(
        destination / "evaluation_state.json",
        {
            "status": "complete",
            "run_identity": run_identity,
            "run_identity_sha256": run_identity_sha256,
            "completed_shards": completed_shards,
            "claim_recovery_after_preoutput_crash": (
                claim_recovery_after_preoutput_crash
            ),
            "evaluation_manifest_sha256": sha256_file(
                destination / "evaluation_manifest.json"
            ),
        },
    )
    return report
