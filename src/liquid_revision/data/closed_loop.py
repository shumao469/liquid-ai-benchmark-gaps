"""Deterministic demonstration generation for the MuJoCo tracking task."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..controllers import (
    FrozenPDObserver,
    FrozenReacherPDObserver,
    PDObserverConfig,
    ReacherPDObserverConfig,
)
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
)


def _environment_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kind = str(config.get("kind", "reacher_v5"))
    common = {
        "horizon_steps": int(config.get("horizon_steps", 500)),
        "frame_skip": int(config.get("frame_skip", 2)),
        "trajectory_family": str(config.get("trajectory_family", "lissajous")),
        "action_limit": float(config.get("action_limit", 1.0)),
        "render_mode": None,
    }
    if kind == "reacher_v5":
        return {
            **common,
            "amplitude_range": tuple(config.get("amplitude_range", (0.15, 0.50))),
            "frequency_range_hz": tuple(config.get("frequency_range_hz", (0.05, 0.25))),
            "velocity_failure_limit": float(
                config.get("velocity_failure_limit", 80.0)
            ),
            "joint_position_failure_limit": float(
                config.get("joint_position_failure_limit", 3.0)
            ),
        }
    if kind == "point_mass_smoke":
        return {
            **common,
            "amplitude_range": tuple(config.get("amplitude_range", (0.45, 1.10))),
            "frequency_range_hz": tuple(config.get("frequency_range_hz", (0.05, 0.25))),
            "position_limit": float(config.get("position_limit", 1.95)),
        }
    raise ValueError("environment.kind must be 'reacher_v5' or 'point_mass_smoke'")


def _make_environment(config: dict[str, Any]):
    from ..envs import MovingTargetReacherEnv, PointMassTrackingEnv

    kind = str(config.get("kind", "reacher_v5"))
    kwargs = _environment_kwargs(config)
    if kind == "reacher_v5":
        return MovingTargetReacherEnv(**kwargs)
    return PointMassTrackingEnv(**kwargs)


def _make_controller(
    environment_kind: str, controller_config: dict[str, Any], action_limit: float
):
    if environment_kind == "reacher_v5":
        return FrozenReacherPDObserver(
            ReacherPDObserverConfig.from_mapping(
                controller_config, action_limit=action_limit
            )
        )
    return FrozenPDObserver(
        PDObserverConfig.from_mapping(controller_config, action_limit=action_limit)
    )


def generate_closed_loop_dataset(
    config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Generate demonstrations atomically; refuse to overwrite prior evidence."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"

    experiment = dict(config.get("experiment", {}))
    environment_config = dict(config.get("environment", {}))
    dataset_config = dict(config.get("dataset", {}))
    controller_config = dict(config.get("controller", {}))
    episode_count = int(dataset_config.get("episodes", 1))
    first_seed = int(dataset_config.get("first_episode_seed", experiment.get("seed", 0)))
    configured_episode_seeds = dataset_config.get("episode_seeds")
    configured_schedule_seeds = dataset_config.get("schedule_seeds")
    configured_episode_perturbations = dataset_config.get("episode_perturbations")
    configured_trajectory_ids = dataset_config.get("trajectory_ids")
    configured_source_rows = dataset_config.get("source_trajectory_rows")
    source_split = dataset_config.get("source_split")
    if configured_episode_seeds is not None:
        episode_seeds = [int(value) for value in configured_episode_seeds]
        if len(episode_seeds) != episode_count:
            raise ValueError("dataset.episode_seeds length must equal dataset.episodes")
    else:
        episode_seeds = [first_seed + index for index in range(episode_count)]
    if configured_schedule_seeds is not None:
        schedule_seeds = [int(value) for value in configured_schedule_seeds]
        if len(schedule_seeds) != episode_count:
            raise ValueError("dataset.schedule_seeds length must equal dataset.episodes")
    else:
        schedule_seeds = [seed * 1009 + 17 for seed in episode_seeds]
    if configured_trajectory_ids is not None:
        trajectory_ids = [str(value) for value in configured_trajectory_ids]
        if len(trajectory_ids) != episode_count or len(set(trajectory_ids)) != episode_count:
            raise ValueError(
                "dataset.trajectory_ids must contain one unique ID per episode"
            )
    else:
        trajectory_ids = [f"episode-{index:06d}" for index in range(episode_count)]
    if configured_episode_perturbations is not None:
        episode_perturbations = [
            PerturbationSpec.from_mapping(value)
            for value in configured_episode_perturbations
        ]
        if len(episode_perturbations) != episode_count:
            raise ValueError(
                "dataset.episode_perturbations length must equal dataset.episodes"
            )
    else:
        common_spec = PerturbationSpec.from_mapping(dataset_config.get("perturbation"))
        episode_perturbations = [common_spec] * episode_count
    canonical_episode_specs = [asdict(spec) for spec in episode_perturbations]
    source_trajectory_rows: list[dict[str, Any]] | None = None
    if configured_source_rows is not None:
        source_trajectory_rows = json.loads(json.dumps(configured_source_rows))
        if len(source_trajectory_rows) != episode_count:
            raise ValueError(
                "dataset.source_trajectory_rows length must equal dataset.episodes"
            )
        for index, row in enumerate(source_trajectory_rows):
            if str(row.get("trajectory_id")) != trajectory_ids[index]:
                raise ValueError("Source trajectory ID order differs from dataset order")
            if int(row.get("episode_seed", -1)) != episode_seeds[index]:
                raise ValueError("Source episode seed order differs from dataset order")
            if int(row.get("schedule_seed", -1)) != schedule_seeds[index]:
                raise ValueError("Source schedule seed order differs from dataset order")
            row_spec = row.get("training_corruption", {}).get(
                "spec", dataset_config.get("perturbation", {})
            )
            if asdict(PerturbationSpec.from_mapping(row_spec)) != canonical_episode_specs[index]:
                raise ValueError("Source perturbation spec differs from dataset order")
    plant_perturbations_during_rollout = bool(
        dataset_config.get("plant_perturbations_during_rollout", False)
    )
    if episode_count <= 0:
        raise ValueError("dataset.episodes must be positive")

    environment_kind = str(environment_config.get("kind", "reacher_v5"))
    controller = _make_controller(
        environment_kind,
        controller_config,
        float(environment_config.get("action_limit", 1.0)),
    )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        env = _make_environment(environment_config)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    observer_inputs: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    expert_estimated_states: list[np.ndarray] = []
    target_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    clean_observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    channel_ages: list[np.ndarray] = []
    previous_controls: list[np.ndarray] = []
    observed_dts: list[float] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    rewards: list[float] = []
    tracking_errors: list[float] = []
    actuator_gains: list[float] = []
    external_forces: list[np.ndarray] = []
    failures: list[bool] = []
    episode_offsets = [0]
    episode_metadata: list[dict[str, Any]] = []

    try:
        for episode_index in range(episode_count):
            episode_seed = episode_seeds[episode_index]
            spec = episode_perturbations[episode_index]
            clean_obs, reset_info = env.reset(seed=episode_seed)
            del reset_info
            schedule_seed = schedule_seeds[episode_index]
            schedule = make_perturbation_schedule(
                horizon_steps=env.horizon_steps,
                nominal_dt=env.control_dt,
                spec=spec,
                seed=schedule_seed,
            )
            perturber = ObservationPerturber(schedule)
            perturber.reset()
            # The training demonstrator is the registered full-state PD loop.
            # The perturbed stream is stored only as causal observer input.
            controller.reset()
            previous_control = np.zeros(2, dtype=np.float32)
            steps_written = 0

            for step in range(env.horizon_steps):
                measured, mask, channel_age, observed_dt = perturber.observe(
                    clean_obs, step
                )
                # Training trajectories are generated by a full-state PD loop.
                # Corruption creates the observer input but never changes the
                # simulator state target or the demonstrator action.
                expert_estimated_state = np.asarray(
                    clean_obs[: env.state_dim], dtype=np.float32
                ).copy()
                target_action = controller.control(
                    expert_estimated_state, clean_obs
                )
                effective_gain = (
                    float(schedule.actuator_gain[step])
                    if plant_perturbations_during_rollout
                    else 1.0
                )
                effective_force = (
                    schedule.external_force[step]
                    if plant_perturbations_during_rollout
                    else np.zeros(2, dtype=np.float32)
                )
                applied_action = np.clip(
                    target_action * effective_gain,
                    -env.action_limit,
                    env.action_limit,
                ).astype(np.float32)
                env.set_external_force(effective_force)

                observer_inputs.append(
                    build_observer_input(
                        measured,
                        mask,
                        channel_age,
                        observed_dt,
                        previous_control,
                    )
                )
                target_states.append(
                    np.asarray(clean_obs[: env.state_dim], dtype=np.float32).copy()
                )
                expert_estimated_states.append(expert_estimated_state.copy())
                target_actions.append(target_action.copy())
                applied_actions.append(applied_action.copy())
                clean_observations.append(np.asarray(clean_obs, dtype=np.float32).copy())
                masks.append(mask.copy())
                channel_ages.append(channel_age.copy())
                previous_controls.append(previous_control.copy())
                observed_dts.append(observed_dt)
                episode_ids.append(episode_index)
                step_ids.append(step)
                actuator_gains.append(effective_gain)
                external_forces.append(effective_force.copy())

                next_obs, reward, terminated, truncated, info = env.step(applied_action)
                rewards.append(float(reward))
                tracking_errors.append(float(info["tracking_error"]))
                failures.append(bool(info["failure"]))
                clean_obs = next_obs
                actuator_gear = float(
                    getattr(controller.config, "actuator_gear", 1.0)
                )
                previous_control = (applied_action * actuator_gear).astype(
                    np.float32
                )
                steps_written += 1
                if terminated or truncated:
                    break

            episode_offsets.append(episode_offsets[-1] + steps_written)
            episode_metadata.append(
                {
                    "episode_index": episode_index,
                    "trajectory_id": trajectory_ids[episode_index],
                    "episode_seed": episode_seed,
                    "schedule": schedule.metadata(),
                    "trajectory": env.trajectory_parameters,
                    "steps": steps_written,
                    "terminated_by_failure": bool(failures[-1]) if steps_written else False,
                }
            )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        env.close()

    arrays = {
        "observer_input": np.stack(observer_inputs).astype(np.float32),
        "target_state": np.stack(target_states).astype(np.float32),
        "expert_estimated_state": np.stack(expert_estimated_states).astype(np.float32),
        "target_action": np.stack(target_actions).astype(np.float32),
        "applied_action": np.stack(applied_actions).astype(np.float32),
        "clean_observation": np.stack(clean_observations).astype(np.float32),
        "sensor_mask": np.stack(masks).astype(np.float32),
        "channel_age": np.stack(channel_ages).astype(np.float32),
        "previous_control": np.stack(previous_controls).astype(np.float32),
        "observed_timestamp_dt": np.asarray(observed_dts, dtype=np.float32)[:, None],
        "episode_id": np.asarray(episode_ids, dtype=np.int32),
        "step_id": np.asarray(step_ids, dtype=np.int32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "tracking_error": np.asarray(tracking_errors, dtype=np.float32),
        "actuator_gain": np.asarray(actuator_gains, dtype=np.float32),
        "external_force": np.stack(external_forces).astype(np.float32),
        "failure": np.asarray(failures, dtype=bool),
        "episode_offsets": np.asarray(episode_offsets, dtype=np.int64),
    }

    try:
        dataset_path = staging / "demonstrations.npz"
        with dataset_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        core = {
            "schema_version": 2,
            "config_sha256": canonical_json_hash(config),
            "implementation": source_tree_manifest(Path(__file__).resolve().parents[1]),
            "dataset_file": dataset_path.name,
            "dataset_sha256": sha256_file(dataset_path),
            "transitions": int(len(observer_inputs)),
            "episodes": episode_count,
            "source_split": str(source_split) if source_split is not None else None,
            "source_trajectory_manifest_sha256": dataset_config.get(
                "source_trajectory_manifest_sha256"
            ),
            "trajectory_ids": trajectory_ids,
            "episode_seeds": episode_seeds,
            "schedule_seeds": schedule_seeds,
            "episode_perturbation_specs": canonical_episode_specs,
            "source_trajectory_rows": source_trajectory_rows,
            "source_trajectory_rows_sha256": (
                canonical_json_hash(source_trajectory_rows)
                if source_trajectory_rows is not None
                else None
            ),
            "supervision": {
                "task": "causal_state_estimation",
                "input": "observer_input",
                "target": "target_state",
                "state_order": ["q1", "q2", "dq1", "dq2"],
                "observer_input_order": (
                    [
                        "measured_q1",
                        "measured_q2",
                        "measured_dq1",
                        "measured_dq2",
                        "valid_q1",
                        "valid_q2",
                        "valid_dq1",
                        "valid_dq2",
                        "age_q1",
                        "age_q2",
                        "age_dq1",
                        "age_dq2",
                        "desired_q1",
                        "desired_q2",
                        "desired_dq1",
                        "desired_dq2",
                        "previous_joint_torque_1",
                        "previous_joint_torque_2",
                        "observed_timestamp_dt",
                    ]
                    if environment_kind == "reacher_v5"
                    else [
                        "measured_x",
                        "measured_y",
                        "measured_dx",
                        "measured_dy",
                        "valid_x",
                        "valid_y",
                        "valid_dx",
                        "valid_dy",
                        "age_x",
                        "age_y",
                        "age_dx",
                        "age_dy",
                        "reference_x",
                        "reference_y",
                        "reference_dx",
                        "reference_dy",
                        "reference_ddx",
                        "reference_ddy",
                        "previous_force_x",
                        "previous_force_y",
                        "observed_timestamp_dt",
                    ]
                ),
                "controller_in_evaluation": "shared_frozen_pd",
                "rollout_driver": "full_state_pd",
                "plant_perturbations_during_rollout": plant_perturbations_during_rollout,
                "sensor_corruption_changes_training_target": False,
            },
            "array_schema": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
            "controller": controller.metadata,
            "environment": {
                "kind": environment_kind,
                "reported_as": (
                    "Reacher-v5 moving-target"
                    if environment_kind == "reacher_v5"
                    else "custom point-mass smoke test (not Reacher-v5)"
                ),
                "control_dt": float(env.control_dt),
                **_environment_kwargs(environment_config),
            },
            "episode_metadata": episode_metadata,
        }
        core["environment"].pop("render_mode", None)
        manifest = {
            **core,
            "manifest_sha256": canonical_json_hash(core),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_closed_loop_dataset(
    dataset_dir: str | Path, *, verify_hash: bool = True
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(dataset_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_core = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_sha256", "created_at_utc"}
    }
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Unsupported closed-loop dataset manifest schema: {manifest_path}")
    if canonical_json_hash(manifest_core) != manifest.get("manifest_sha256"):
        raise ValueError(f"Dataset manifest SHA-256 mismatch: {manifest_path}")
    dataset_name = str(manifest.get("dataset_file", ""))
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError(f"Dataset manifest contains an unsafe dataset path: {dataset_name}")
    dataset_path = root / dataset_name
    if verify_hash and sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError(f"Dataset SHA-256 mismatch: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    declared_schema = manifest.get("array_schema")
    if not isinstance(declared_schema, dict) or set(arrays) != set(declared_schema):
        raise ValueError("Dataset arrays differ from the frozen array schema")
    for name, value in arrays.items():
        expected = declared_schema[name]
        if list(value.shape) != expected.get("shape") or str(value.dtype) != expected.get(
            "dtype"
        ):
            raise ValueError(f"Dataset array schema mismatch for {name}")

    required = {
        "observer_input",
        "target_state",
        "expert_estimated_state",
        "target_action",
        "applied_action",
        "clean_observation",
        "sensor_mask",
        "channel_age",
        "previous_control",
        "observed_timestamp_dt",
        "episode_id",
        "step_id",
        "reward",
        "tracking_error",
        "actuator_gain",
        "external_force",
        "failure",
        "episode_offsets",
    }
    if set(arrays) != required:
        raise ValueError("Dataset array names differ from the registered schema")
    transitions = int(manifest.get("transitions", -1))
    episodes = int(manifest.get("episodes", -1))
    if transitions <= 0 or episodes <= 0:
        raise ValueError("Dataset manifest has invalid transition or episode counts")
    for name, value in arrays.items():
        if name == "episode_offsets":
            continue
        if value.shape[0] != transitions:
            raise ValueError(f"Dataset transition count mismatch for {name}")
    offsets = arrays["episode_offsets"]
    if (
        offsets.shape != (episodes + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != transitions
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("Dataset episode offsets are invalid")
    metadata = manifest.get("episode_metadata")
    if not isinstance(metadata, list) or len(metadata) != episodes:
        raise ValueError("Dataset episode metadata count mismatch")
    for episode_index, row in enumerate(metadata):
        start, stop = int(offsets[episode_index]), int(offsets[episode_index + 1])
        if (
            int(row.get("episode_index", -1)) != episode_index
            or int(row.get("steps", -1)) != stop - start
            or not np.all(arrays["episode_id"][start:stop] == episode_index)
            or not np.array_equal(
                arrays["step_id"][start:stop], np.arange(stop - start, dtype=np.int32)
            )
        ):
            raise ValueError(f"Dataset episode indexing mismatch for episode {episode_index}")
    return arrays, manifest

