"""Validation-only fixed-budget calibration on an independent tuning seed.

The calibration deliberately has no code path to test-set metrics or corruption
curves.  It may inform the shared optimizer-step budget, but it is never an
estimate of confirmatory performance.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from liquid_revision.config import load_config
from liquid_revision.io_utils import atomic_write_json, sha256_file
from liquid_revision.repro import seed_everything

from .models import build_offline_model, resolve_hidden_dim
from .training import (
    TASKS,
    _atomic_torch_save,
    _criterion,
    _loss_for_task,
    _tensor_batch,
    _timing_summary,
    compose_model_features,
    deterministic_batch_indices,
    evaluate_arrays,
    load_all_datasets,
    utc_now,
)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    best_step: int,
    best_metric: float,
    metadata: dict[str, Any],
    validation_history: list[dict[str, Any]],
    step_times: list[float],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_step": int(optimizer_step),
        "best_step": int(best_step),
        "best_metric": float(best_metric),
        "metadata": metadata,
        "validation_history": validation_history,
        "step_times": step_times,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }


def _plateau_diagnostics(
    history: list[dict[str, Any]], calibration: dict[str, Any]
) -> dict[str, Any]:
    window = int(calibration.get("plateau_window_validations", 5))
    slope_threshold = float(
        calibration.get("slope_threshold_metric_per_1000_steps", 0.005)
    )
    gain_threshold = float(calibration.get("recent_best_gain_threshold", 0.005))
    minimum_margin = int(calibration.get("minimum_margin_after_best_validations", 2))
    if len(history) < 2 * window:
        return {
            "status": "insufficient_validation_points",
            "plateau": False,
            "required_points": 2 * window,
            "available_points": len(history),
        }
    steps = np.asarray([item["optimizer_step"] for item in history], dtype=np.float64)
    values = np.asarray([item["selection_metric"] for item in history], dtype=np.float64)
    recent_steps = steps[-window:]
    recent_values = values[-window:]
    slope_per_1000 = float(np.polyfit(recent_steps, recent_values, 1)[0] * 1000.0)
    previous_best = float(np.max(values[:-window]))
    recent_best = float(np.max(recent_values))
    recent_best_gain = recent_best - previous_best
    best_index = int(np.argmax(values))
    best_step = int(steps[best_index])
    margin_validations = int((steps[-1] - best_step) / (steps[1] - steps[0]))
    plateau = bool(
        abs(slope_per_1000) <= slope_threshold
        and recent_best_gain <= gain_threshold
        and margin_validations >= minimum_margin
    )
    clearly_not_plateau = bool(
        slope_per_1000 > slope_threshold
        or recent_best_gain > gain_threshold
        or margin_validations < minimum_margin
    )
    return {
        "status": "plateau" if plateau else "not_plateau",
        "plateau": plateau,
        "clearly_not_plateau": clearly_not_plateau,
        "window_validations": window,
        "recent_slope_metric_per_1000_steps": slope_per_1000,
        "slope_threshold_metric_per_1000_steps": slope_threshold,
        "previous_best": previous_best,
        "recent_best": recent_best,
        "recent_best_gain": recent_best_gain,
        "recent_best_gain_threshold": gain_threshold,
        "global_best_step": best_step,
        "validations_after_global_best": margin_validations,
        "minimum_margin_after_best_validations": minimum_margin,
        "recent_metric_range": float(np.ptp(recent_values)),
    }


def calibrate_one(
    *,
    dataset_name: str,
    model_name: str,
    seed: int,
    data: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    task = TASKS[dataset_name]
    run_id = f"budget-{dataset_name}-{model_name}-seed{seed}-{config_hash[:10]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    manifest_path = run_dir / "run_manifest.json"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            return existing

    experiment = dict(config.get("experiment", {}))
    if bool(experiment.get("test_metrics_permitted", False)):
        raise ValueError("budget calibration requires test_metrics_permitted=false")
    training = dict(config.get("training", {}))
    calibration = dict(config.get("budget_calibration", {}))
    if str(calibration.get("test_evaluation", "forbidden")) != "forbidden":
        raise ValueError("test evaluation must remain forbidden")
    if str(calibration.get("corruption_evaluation", "forbidden")) != "forbidden":
        raise ValueError("corruption evaluation must remain forbidden")
    if int(calibration.get("tuning_seed", seed)) != int(seed):
        raise ValueError("calibration seed does not match the independent tuning seed")

    seed_state = seed_everything(
        seed, deterministic_torch=bool(experiment.get("deterministic_torch", True))
    )
    device = torch.device(str(experiment.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA budget calibration requested but unavailable")
    channels = list(
        dict(config.get("features", {})).get(
            "channels", ["values", "mask", "age", "dt"]
        )
    )
    train_arrays = data["train"]
    validation_arrays = data["val"]
    example = compose_model_features(
        train_arrays.values[:1],
        train_arrays.mask[:1],
        train_arrays.age[:1],
        train_arrays.dt[:1],
        channels=channels,
    )
    model_config = dict(config.get("models", {}))
    hidden_dim = resolve_hidden_dim(model_config, dataset_name, model_name)
    model = build_offline_model(
        model_name,
        input_dim=int(example.shape[-1]),
        hidden_dim=hidden_dim,
        output_dim=task.output_dim,
        ode_max_step=float(model_config.get("ode_max_step", 1.0)),
        ltc_ode_unfolds=int(model_config.get("ltc_ode_unfolds", 6)),
        cfc_mode=str(model_config.get("cfc_mode", "default")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    criterion = _criterion(task, train_arrays.labels, device)
    max_steps = int(training.get("optimizer_steps", 2000))
    validation_interval = int(training.get("validation_interval", 100))
    if max_steps > 2000 or validation_interval != 100:
        raise ValueError("registered calibration is capped at 2000 steps with 100-step validation")
    batch_size = int(training.get("batch_size", 64))
    evaluation_batch_size = int(training.get("evaluation_batch_size", 128))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    metadata = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "seed": int(seed),
        "config_sha256": config_hash,
        "split_manifest_sha256": data.get("manifest_sha256"),
        "input_dim": int(example.shape[-1]),
        "hidden_dim": hidden_dim,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "device": str(device),
        "seed_state": seed_state,
        "max_optimizer_steps": max_steps,
        "validation_interval": validation_interval,
        "selection_metric": task.selection_metric,
        "test_metrics_computed": False,
        "corruption_metrics_computed": False,
    }
    start_step = 0
    best_step = -1
    best_metric = -float("inf")
    validation_history: list[dict[str, Any]] = []
    step_times: list[float] = []
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["metadata"] != metadata:
            raise ValueError("budget-calibration resume metadata mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["optimizer_step"])
        best_step = int(checkpoint["best_step"])
        best_metric = float(checkpoint["best_metric"])
        validation_history = list(checkpoint["validation_history"])
        step_times = [float(value) for value in checkpoint["step_times"]]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {
        **metadata,
        "status": "running",
        "started_at_utc": previous.get("started_at_utc", utc_now()),
        "attempt_started_at_utc": utc_now(),
        "attempt_number": int(previous.get("attempt_number", 0)) + 1,
        "resumed_from_step": start_step,
        "last_completed_step": start_step,
    }
    atomic_write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        for zero_based_step in range(start_step, max_steps):
            step_started = time.perf_counter()
            model.train()
            indices = deterministic_batch_indices(
                len(train_arrays.labels), batch_size, seed, zero_based_step
            )
            features, dt, labels, lengths = _tensor_batch(
                train_arrays, indices, device=device, feature_channels=channels
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, dt, lengths)
            loss = _loss_for_task(task, logits, labels, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite calibration training loss")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                .detach()
                .cpu()
            )
            optimizer.step()
            step_times.append(float(time.perf_counter() - step_started))
            completed_step = zero_based_step + 1
            if completed_step % validation_interval == 0 or completed_step == max_steps:
                validation_started = time.perf_counter()
                metrics, _ = evaluate_arrays(
                    model,
                    validation_arrays,
                    task=task,
                    device=device,
                    batch_size=evaluation_batch_size,
                    feature_channels=channels,
                    criterion=criterion,
                )
                selection_metric = float(metrics[task.selection_metric])
                record = {
                    "optimizer_step": completed_step,
                    "selection_metric": selection_metric,
                    "metrics": metrics,
                    "training_loss_at_validation": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": gradient_norm,
                    "validation_wall_time_seconds": float(
                        time.perf_counter() - validation_started
                    ),
                    "median_step_seconds_to_date": float(np.median(step_times)),
                    "steps_per_second_from_median": float(1.0 / np.median(step_times)),
                }
                validation_history.append(record)
                if selection_metric > best_metric:
                    best_metric = selection_metric
                    best_step = completed_step
                    _atomic_torch_save(
                        best_path,
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            optimizer_step=completed_step,
                            best_step=best_step,
                            best_metric=best_metric,
                            metadata=metadata,
                            validation_history=validation_history,
                            step_times=step_times,
                        ),
                    )
                _atomic_torch_save(
                    latest_path,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        optimizer_step=completed_step,
                        best_step=best_step,
                        best_metric=best_metric,
                        metadata=metadata,
                        validation_history=validation_history,
                        step_times=step_times,
                    ),
                )
                manifest.update(
                    {
                        "last_completed_step": completed_step,
                        "best_step": best_step,
                        "best_validation_metric": best_metric,
                    }
                )
                atomic_write_json(manifest_path, manifest)
        diagnostics = _plateau_diagnostics(validation_history, calibration)
        trailing = step_times[-min(500, len(step_times)) :]
        result: dict[str, Any] = {
            **metadata,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "wall_time_seconds_this_attempt": float(time.perf_counter() - started),
            "best_step": best_step,
            "best_validation_metric": best_metric,
            "final_validation_metric": validation_history[-1]["selection_metric"],
            "validation_history": validation_history,
            "plateau_diagnostics": diagnostics,
            "optimizer_step_timing_all": _timing_summary(step_times),
            "optimizer_step_timing_last_500": _timing_summary(trailing),
            "best_checkpoint": str(best_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path),
            "latest_checkpoint": str(latest_path.resolve()),
            "latest_checkpoint_sha256": sha256_file(latest_path),
            "test_metrics_computed": False,
            "corruption_metrics_computed": False,
        }
        atomic_write_json(result_path, result)
        manifest.update(
            {
                "status": "completed",
                "completed_at_utc": result["completed_at_utc"],
                "last_completed_step": max_steps,
                "result": str(result_path.resolve()),
            }
        )
        atomic_write_json(manifest_path, manifest)
        return result
    except BaseException as exc:
        terminal = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        manifest.update(
            {
                "status": terminal,
                f"{terminal}_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise


def run_calibration(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, config_hash = load_config(config_path)
    experiment = dict(config.get("experiment", {}))
    calibration = dict(config.get("budget_calibration", {}))
    seeds = [int(value) for value in experiment.get("seeds", [17])]
    if seeds != [17]:
        raise ValueError("budget calibration is registered only for independent seed 17")
    selected_datasets = [str(value) for value in experiment.get("datasets", ["har"])]
    datasets = load_all_datasets(
        config, config_path, selected_datasets=selected_datasets
    )
    model_names = [str(value) for value in dict(config.get("models", {})).get("names", [])]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "budget_calibration_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_hash:
            raise ValueError("existing calibration output has a different config hash")
    attempts = list(previous.get("attempts", []))
    if previous and previous.get("status") not in {"running", "completed"}:
        previous_attempt = {
            "attempt_number": len(attempts) + 1,
            "status": previous.get("status"),
            "started_at_utc": previous.get("started_at_utc"),
            "interrupted_at_utc": previous.get("interrupted_at_utc"),
            "interruption": previous.get("interruption"),
        }
        if not attempts or attempts[-1].get("status") != previous_attempt["status"]:
            attempts.append(previous_attempt)
    current_attempt = {
        "attempt_number": len(attempts) + 1,
        "status": "running",
        "started_at_utc": utc_now(),
        "resuming_from_previous_output": bool(previous),
    }
    aggregate: dict[str, Any] = {
        "status": "running",
        "confirmatory_evidence": False,
        "test_metrics_computed": False,
        "corruption_metrics_computed": False,
        "config": str(config_path),
        "config_sha256": config_hash,
        "started_at_utc": previous.get("started_at_utc", utc_now()),
        "attempts": attempts,
        "current_attempt": current_attempt,
        "previous_interruption": previous.get("interruption"),
        "runs": list(previous.get("runs", [])),
    }
    atomic_write_json(manifest_path, aggregate)
    try:
        results: list[dict[str, Any]] = []
        for dataset_name in selected_datasets:
            for model_name in model_names:
                result = calibrate_one(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    seed=17,
                    data=datasets[dataset_name],
                    config=config,
                    config_hash=config_hash,
                    output_root=output / "runs",
                )
                results.append(result)
                aggregate["runs"] = [
                    {
                        "run_id": item["run_id"],
                        "status": item["status"],
                        "best_step": item["best_step"],
                        "plateau": item["plateau_diagnostics"]["plateau"],
                    }
                    for item in results
                ]
                atomic_write_json(manifest_path, aggregate)
        all_plateau = all(
            bool(item["plateau_diagnostics"]["plateau"]) for item in results
        )
        candidates = [int(value) for value in calibration.get(
            "candidate_fixed_budgets", [5000, 10000, 20000]
        )]
        decision = {
            "rule": calibration.get("freeze_rule", "all_models_must_plateau"),
            "candidate_fixed_budgets": candidates,
            "all_models_plateau_by_2000": all_plateau,
            "frozen_budget": min(candidates) if all_plateau else None,
            "decision": (
                "freeze_smallest_candidate"
                if all_plateau
                else "defer_budget_freeze_due_to_nonplateau_validation_curves"
            ),
            "test_metrics_used": False,
        }
        aggregate.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "attempts": [
                    *attempts,
                    {
                        **current_attempt,
                        "status": "completed",
                        "completed_at_utc": utc_now(),
                    },
                ],
                "current_attempt": None,
                "runs": [
                    {
                        "run_id": item["run_id"],
                        "status": item["status"],
                        "result": str(
                            (output / "runs" / item["run_id"] / "result.json").resolve()
                        ),
                        "best_step": item["best_step"],
                        "plateau_diagnostics": item["plateau_diagnostics"],
                        "optimizer_step_timing_last_500": item[
                            "optimizer_step_timing_last_500"
                        ],
                    }
                    for item in results
                ],
                "budget_decision": decision,
            }
        )
        atomic_write_json(manifest_path, aggregate)
        return aggregate
    except BaseException as exc:
        terminal = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        aggregate.update(
            {
                "status": terminal,
                f"{terminal}_at_utc": utc_now(),
                "attempts": [
                    *attempts,
                    {
                        **current_attempt,
                        "status": terminal,
                        f"{terminal}_at_utc": utc_now(),
                    },
                ],
                "current_attempt": None,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(manifest_path, aggregate)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_calibration(args.config, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
