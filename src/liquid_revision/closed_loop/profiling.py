"""Measured cost profiles for the frozen closed-loop protocol.

These profiles are deliberately separate from the confirmatory result files.  They
are capacity checks used to decide whether the preregistered evaluation workload is
feasible before any long run is launched.
"""

from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file
from .baselines import build_nonlearned_baseline
from .benchmark import _rollout, build_response_conditions
from .parameter_matching import (
    build_registered_model,
    load_parameter_registry,
    parameter_count_audit,
)
from .training import load_trained_observer


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime_device(device: torch.device) -> dict[str, Any]:
    return {
        "requested": str(device),
        "resolved": str(device),
        "cuda_visible_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty profile")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def profile_optimizer_steps(
    *,
    config: dict[str, Any],
    registry_path: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    steps: int = 100,
    warmup_steps: int = 5,
) -> dict[str, Any]:
    """Time matched synthetic optimizer steps for every registered observer."""
    if steps <= 0 or warmup_steps < 0:
        raise ValueError("steps must be positive and warmup_steps non-negative")
    destination = Path(output_path).resolve()
    csv_path = destination.with_suffix(".csv")
    if destination.exists() or csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite profile: {destination}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    registry = load_parameter_registry(registry_path)
    training = dict(config.get("training", {}))
    batch_size = int(training.get("batch_size", 32))
    window_steps = int(training.get("window_steps", 64))
    input_dim = int(config.get("models", {}).get("input_dim", 19))
    state_dim = int(config.get("models", {}).get("state_dim", 4))
    learning_rate = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 1e-5))
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    episodes = int(config.get("splits", {}).get("train", {}).get("episodes", 0))
    horizon = int(config.get("environment", {}).get("horizon_steps", 0))
    calculated_steps_per_epoch = (
        (episodes * ((horizon + window_steps - 1) // window_steps) + batch_size - 1)
        // batch_size
    )
    declared_steps_per_epoch = int(
        training.get("optimizer_steps_per_epoch", calculated_steps_per_epoch)
    )
    if declared_steps_per_epoch != calculated_steps_per_epoch:
        raise RuntimeError(
            "Profile detected a frozen-budget mismatch: calculated "
            f"{calculated_steps_per_epoch} != declared {declared_steps_per_epoch}"
        )
    planned_steps = int(training.get("epochs", 100)) * calculated_steps_per_epoch
    if int(training.get("planned_optimizer_steps", planned_steps)) != planned_steps:
        raise RuntimeError("Profile detected a declared planned-step mismatch")
    generator = torch.Generator(device="cpu").manual_seed(
        int(config.get("experiment", {}).get("seed", 20261102))
    )
    x_cpu = torch.randn(
        batch_size, window_steps, input_dim, generator=generator, dtype=torch.float32
    )
    y_cpu = torch.randn(
        batch_size, window_steps, state_dim, generator=generator, dtype=torch.float32
    )
    dt_cpu = torch.full((batch_size, window_steps), 0.02, dtype=torch.float32)
    x = x_cpu.to(torch_device)
    y = y_cpu.to(torch_device)
    dt = dt_cpu.to(torch_device)
    rows: list[dict[str, Any]] = []
    for model_index, registered_name in enumerate(registry["entries"]):
        torch.manual_seed(1701 + model_index)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(1701 + model_index)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch_device)
        model = build_registered_model(registered_name, config, registry).to(torch_device)
        count_audit = parameter_count_audit(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        def one_step() -> float:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, dt)
            loss = torch.mean((prediction - y) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite profiling loss for {registered_name}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            return float(loss.detach().cpu())

        for _ in range(warmup_steps):
            one_step()
        _synchronize(torch_device)
        started = time.perf_counter()
        final_loss = 0.0
        for _ in range(steps):
            final_loss = one_step()
        _synchronize(torch_device)
        elapsed = time.perf_counter() - started
        seconds_per_step = elapsed / steps
        rows.append(
            {
                "registered_name": registered_name,
                "model_nominal_trainable_parameters": int(
                    model.trainable_parameters
                ),
                "nominal_allocated_trainable_parameters": count_audit[
                    "nominal_allocated_trainable_parameters"
                ],
                "effective_active_trainable_parameters": count_audit[
                    "effective_active_trainable_parameters"
                ],
                "batch_size": batch_size,
                "window_steps": window_steps,
                "measured_optimizer_steps": steps,
                "warmup_optimizer_steps": warmup_steps,
                "elapsed_seconds": elapsed,
                "seconds_per_optimizer_step": seconds_per_step,
                "optimizer_steps_per_second": 1.0 / seconds_per_step,
                "projected_planned_optimizer_steps": planned_steps,
                "calculated_optimizer_steps_per_epoch": calculated_steps_per_epoch,
                "projected_training_hours_per_seed": seconds_per_step
                * planned_steps
                / 3600.0,
                "peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(torch_device))
                    if torch_device.type == "cuda"
                    else 0
                ),
                "final_synthetic_loss": final_loss,
            }
        )
        del optimizer, model
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    _write_csv(csv_path, rows)
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "capacity profile only; not manuscript performance evidence",
        "config_sha256": canonical_json_hash(config),
        "parameter_registry_sha256": registry["registry_sha256"],
        "device": _runtime_device(torch_device),
        "rows": rows,
        "files": {"csv": sha256_file(csv_path)},
    }
    atomic_write_json(destination, report)
    return report


def profile_rollout_throughput(
    *,
    config: dict[str, Any],
    checkpoints: list[str | Path],
    output_path: str | Path,
    device: str = "cpu",
    episodes: int = 2,
) -> dict[str, Any]:
    """Measure end-to-end Reacher plant steps/s for learned and fixed baselines."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    destination = Path(output_path).resolve()
    csv_path = destination.with_suffix(".csv")
    if destination.exists() or csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite profile: {destination}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    conditions = build_response_conditions(config)
    clean_condition = next(
        condition
        for condition in conditions
        if condition["family"] == "iid_dropout" and condition["severity"] == 0.0
    )
    profiling = dict(config.get("profiling", {}))
    episode_seed_start = int(profiling.get("episode_seed_start", 900_000_000))
    schedule_seed_start = int(profiling.get("schedule_seed_start", 1_000_000_000))
    profile_episode_seeds = list(range(episode_seed_start, episode_seed_start + episodes))
    profile_schedule_seeds = list(
        range(schedule_seed_start, schedule_seed_start + episodes)
    )
    for split_name, split_spec in config.get("splits", {}).items():
        split_spec = dict(split_spec)
        split_count = int(split_spec.get("episodes", 0))
        frozen_episode_seeds = set(
            range(
                int(split_spec.get("episode_seed_start", 0)),
                int(split_spec.get("episode_seed_start", 0)) + split_count,
            )
        )
        frozen_schedule_seeds = set(
            range(
                int(split_spec.get("schedule_seed_start", 0)),
                int(split_spec.get("schedule_seed_start", 0)) + split_count,
            )
        )
        if set(profile_episode_seeds) & frozen_episode_seeds or set(
            profile_schedule_seeds
        ) & frozen_schedule_seeds:
            raise ValueError(
                f"Dedicated profiling seeds overlap the frozen {split_name} split"
            )
    profile_seed_identity = {
        "source": "dedicated_synthetic_profile_namespace_never_trajectory_manifest",
        "episode_seeds": profile_episode_seeds,
        "schedule_seeds": profile_schedule_seeds,
        "validation_or_synthetic_only": True,
        "test_episodes_used": 0,
    }
    horizon = int(config.get("environment", {}).get("horizon_steps", 500))
    test_episodes = int(
        config.get("splits", {}).get("test", {}).get("episodes", 200)
    )
    condition_count = len(conditions)
    estimators: list[dict[str, Any]] = []
    for name in config.get("evaluation", {}).get(
        "nonlearned_baselines", ["full_state_pd", "zoh_pd", "kalman_pd"]
    ):
        estimators.append(
            {
                "profile_id": str(name),
                "model_name": str(name),
                "training_seed": -1,
                "baseline": build_nonlearned_baseline(
                    str(name),
                    config.get("evaluation", {}).get("kalman")
                    if str(name) == "kalman_pd"
                    else None,
                ),
                "checkpoint_sha256": None,
            }
        )
    for checkpoint_path in checkpoints:
        model, metadata = load_trained_observer(
            checkpoint_path,
            device=str(torch_device),
            expected_config=config,
        )
        estimators.append(
            {
                "profile_id": f"{metadata['registered_name']}:seed_{metadata['seed']}",
                "model_name": metadata["registered_name"],
                "training_seed": int(metadata["seed"]),
                "model": model,
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )
    rows: list[dict[str, Any]] = []
    for estimator in estimators:
        _synchronize(torch_device)
        started = time.perf_counter()
        measured_steps = 0
        for offset in range(episodes):
            result = _rollout(
                estimator,
                config,
                clean_condition,
                episode_seed=profile_episode_seeds[offset],
                schedule_seed=profile_schedule_seeds[offset],
                device=str(torch_device),
            )
            measured_steps += int(result["steps"])
        _synchronize(torch_device)
        elapsed = time.perf_counter() - started
        steps_per_second = measured_steps / elapsed
        projected_steps = test_episodes * condition_count * horizon
        rows.append(
            {
                "profile_id": estimator["profile_id"],
                "model_name": estimator["model_name"],
                "training_seed": estimator["training_seed"],
                "checkpoint_sha256": estimator["checkpoint_sha256"],
                "profile_episodes": episodes,
                "horizon_steps": horizon,
                "measured_plant_steps": measured_steps,
                "elapsed_seconds": elapsed,
                "plant_steps_per_second": steps_per_second,
                "confirmatory_test_episodes": test_episodes,
                "response_conditions": condition_count,
                "projected_plant_steps_per_estimator": projected_steps,
                "projected_hours_per_full_surface": projected_steps
                / steps_per_second
                / 3600.0,
            }
        )
    _write_csv(csv_path, rows)
    learned = [row for row in rows if int(row["training_seed"]) >= 0]
    baselines = [row for row in rows if int(row["training_seed"]) < 0]
    projected_seconds = sum(
        float(row["projected_hours_per_full_surface"]) * 3600.0 for row in rows
    )
    registered_models = [
        str(name)
        for name in config.get("matching", {}).get("registered_models", [])
    ]
    confirmatory_seed_count = len(config.get("training", {}).get("seeds", []))
    representative_hours: dict[str, float] = {}
    for name in registered_models:
        candidates = [
            float(row["projected_hours_per_full_surface"])
            for row in learned
            if row["model_name"] == name
        ]
        if candidates:
            representative_hours[name] = sum(candidates) / len(candidates)
    formal_projection_complete = set(representative_hours) == set(registered_models)
    formal_total_hours = None
    if formal_projection_complete:
        formal_total_hours = sum(
            representative_hours[name] * confirmatory_seed_count
            for name in registered_models
        ) + sum(float(row["projected_hours_per_full_surface"]) for row in baselines)
    formal_estimator_surfaces = (
        len(registered_models) * confirmatory_seed_count + len(baselines)
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "capacity profile only; not manuscript performance evidence",
        "config_sha256": canonical_json_hash(config),
        "device": _runtime_device(torch_device),
        "profile_condition": clean_condition,
        "profile_episodes": episodes,
        "profile_seed_source": profile_seed_identity["source"],
        "profile_seed_namespace_sha256": canonical_json_hash(profile_seed_identity),
        "profile_episode_seeds": profile_episode_seeds,
        "profile_schedule_seeds": profile_schedule_seeds,
        "validation_or_synthetic_only": True,
        "test_episodes_used": 0,
        "rows": rows,
        "projected_serial_total_plant_steps_for_supplied_estimators": sum(
            int(row["projected_plant_steps_per_estimator"]) for row in rows
        ),
        "projected_serial_total_hours_for_supplied_estimators": projected_seconds
        / 3600.0,
        "confirmatory_training_seed_count": confirmatory_seed_count,
        "confirmatory_estimator_surfaces": formal_estimator_surfaces,
        "projected_confirmatory_plant_steps": formal_estimator_surfaces
        * test_episodes
        * condition_count
        * horizon,
        "formal_projection_complete": formal_projection_complete,
        "projected_confirmatory_serial_hours": formal_total_hours,
        "missing_learned_models_for_formal_projection": sorted(
            set(registered_models) - set(representative_hours)
        ),
        "learned_estimators_profiled": len(learned),
        "nonlearned_estimators_profiled": len(baselines),
        "projection_note": (
            "Linear serial projection from measured end-to-end clean-condition "
            "throughput; perturbation severity may change realized throughput."
        ),
        "files": {"csv": sha256_file(csv_path)},
    }
    atomic_write_json(destination, report)
    return report
