"""Resumable fixed-checkpoint training and robustness evaluation."""

from __future__ import annotations

import json
import hashlib
import os
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from liquid_revision.config import load_config
from liquid_revision.io_utils import atomic_write_json, canonical_json_hash, sha256_file
from liquid_revision.prepare_data import verify_prepared_data
from liquid_revision.repro import seed_everything

from .corruption import CorruptionSpec, apply_corruption, default_corruption_specs
from .compute_accounting import sequence_compute_accounting
from .metrics import (
    accuracy_macro_f1,
    binary_classification_metrics,
    count_trainable_parameters,
    measure_batch1_latency,
)
from .models import OfflineSequenceClassifier, build_offline_model, resolve_hidden_dim
from .formal import (
    create_formal_freeze,
    current_run_accelerator_identity,
    formal_enabled,
    publish_checkpoint_release,
    verify_checkpoint_release,
    verify_completed_run,
    write_completion_artifact,
)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    output_dim: int
    selection_metric: str
    packetwise_corruption: bool


TASKS = {
    "har": TaskSpec("har", 6, "macro_f1", True),
    "physionet2012": TaskSpec("physionet2012", 1, "auroc", False),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compose_model_features(
    values: np.ndarray,
    mask: np.ndarray,
    age: np.ndarray,
    dt: np.ndarray,
    *,
    channels: list[str] | tuple[str, ...] = ("values", "mask", "age", "dt"),
) -> np.ndarray:
    """Build the shared model input without fitting any test-set statistic."""

    values_array = np.asarray(values, dtype=np.float32)
    mask_array = np.asarray(mask, dtype=np.float32)
    age_array = np.asarray(age, dtype=np.float32)
    dt_array = np.asarray(dt, dtype=np.float32)
    if values_array.ndim != 3 or mask_array.shape != values_array.shape:
        raise ValueError("values and mask must share shape [sample,time,feature]")
    if age_array.shape != values_array.shape:
        raise ValueError("age must match values")
    if dt_array.ndim == 3 and dt_array.shape[-1] == 1:
        dt_array = dt_array[..., 0]
    if dt_array.shape != values_array.shape[:2]:
        raise ValueError("dt must have shape [sample,time]")
    parts: list[np.ndarray] = []
    for channel in channels:
        if channel == "values":
            parts.append(values_array)
        elif channel == "mask":
            parts.append(mask_array)
        elif channel == "age":
            # log1p is a fixed transform, not a statistic learned from evaluation data.
            parts.append(np.log1p(np.maximum(age_array, 0.0)).astype(np.float32))
        elif channel == "dt":
            parts.append(np.log1p(dt_array)[..., None].astype(np.float32))
        else:
            raise ValueError(f"Unknown feature channel: {channel}")
    if not parts:
        raise ValueError("at least one model input channel is required")
    result = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("composed model features contain nonfinite values")
    return result


def deterministic_batch_indices(
    sample_count: int, batch_size: int, seed: int, optimizer_step: int
) -> np.ndarray:
    """Map an optimizer step to a reproducible shuffled epoch and minibatch."""

    if sample_count <= 0 or batch_size <= 0 or optimizer_step < 0:
        raise ValueError("invalid deterministic batch request")
    batches_per_epoch = int(np.ceil(sample_count / batch_size))
    epoch = optimizer_step // batches_per_epoch
    batch_index = optimizer_step % batches_per_epoch
    generator = np.random.default_rng(int(seed) + int(epoch) * 1_000_003)
    permutation = generator.permutation(sample_count)
    start = batch_index * batch_size
    return permutation[start : min(start + batch_size, sample_count)]


def _tensor_batch(
    arrays: Any,
    indices: np.ndarray,
    *,
    device: torch.device,
    feature_channels: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = compose_model_features(
        arrays.values[indices],
        arrays.mask[indices],
        arrays.age[indices],
        arrays.dt[indices],
        channels=feature_channels,
    )
    values = torch.from_numpy(features).to(device)
    dt = torch.from_numpy(np.asarray(arrays.dt[indices], dtype=np.float32)).to(device)
    labels = torch.from_numpy(np.asarray(arrays.labels[indices])).to(device)
    lengths = torch.from_numpy(np.asarray(arrays.lengths[indices], dtype=np.int64)).to(device)
    return values, dt, labels, lengths


def _loss_for_task(
    task: TaskSpec, logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module
) -> torch.Tensor:
    if task.name == "har":
        return criterion(logits, labels.long())
    return criterion(logits[:, 0], labels.float())


def _compute_task_metrics(
    task: TaskSpec, labels: np.ndarray, logits: np.ndarray
) -> dict[str, Any]:
    if task.name == "har":
        return accuracy_macro_f1(labels, np.argmax(logits, axis=1), num_classes=6)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -40, 40)))
    return binary_classification_metrics(labels, probabilities)


def evaluate_arrays(
    model: OfflineSequenceClassifier,
    arrays: Any,
    *,
    task: TaskSpec,
    device: torch.device,
    batch_size: int,
    feature_channels: list[str],
    criterion: nn.Module | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(arrays.labels), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays.labels)))
            features, dt, labels, lengths = _tensor_batch(
                arrays, indices, device=device, feature_channels=feature_channels
            )
            logits = model(features, dt, lengths)
            if criterion is not None:
                losses.append(float(_loss_for_task(task, logits, labels, criterion).cpu()))
            all_logits.append(logits.detach().cpu().numpy())
    concatenated = np.concatenate(all_logits, axis=0)
    metrics = _compute_task_metrics(task, np.asarray(arrays.labels), concatenated)
    if losses:
        metrics["loss"] = float(np.mean(losses))
    return metrics, concatenated


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _criterion(task: TaskSpec, train_labels: np.ndarray, device: torch.device) -> nn.Module:
    if task.name == "har":
        return nn.CrossEntropyLoss()
    labels = np.asarray(train_labels, dtype=np.int64)
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("PhysioNet training split must contain both outcome classes")
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_metric: float,
    best_step: int,
    run_metadata: dict[str, Any],
    validation_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_step": int(step),
        "best_metric": float(best_metric),
        "best_step": int(best_step),
        "run_metadata": run_metadata,
        "validation_history": validation_history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _model_state_sha256(model: nn.Module) -> str:
    """Hash an in-memory state dict to prove evaluation did not mutate it."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "min_seconds": float(array.min()),
        "max_seconds": float(array.max()),
    }


def _run_identifier(dataset: str, model: str, seed: int, config_hash: str) -> str:
    return f"{dataset}-{model}-seed{seed}-{config_hash[:10]}"


def train_fixed_budget(
    *,
    dataset_name: str,
    model_name: str,
    seed: int,
    data: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
    output_root: Path,
    formal_context: Mapping[str, Any] | None = None,
    stop_request_path: Path | None = None,
) -> dict[str, Any]:
    """Train and validation-select one checkpoint without touching test arrays."""

    task = TASKS[dataset_name]
    run_id = _run_identifier(dataset_name, model_name, seed, config_hash)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    result_path = run_dir / "result.json"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    if result_path.is_file():
        identity = {
            "run_id": run_id,
            "dataset": dataset_name,
            "model": model_name,
            "seed": int(seed),
        }
        verified = verify_completed_run(
            run_dir=run_dir,
            identity=identity,
            config_hash=config_hash,
            formal_freeze_sha256=(
                None
                if formal_context is None
                else str(formal_context["formal_freeze_sha256"])
            ),
        )
        return dict(verified["result"])

    experiment = dict(config.get("experiment", {}))
    training = dict(config.get("training", {}))
    model_config = dict(config.get("models", {}))
    feature_channels = list(
        dict(config.get("features", {})).get(
            "channels", ["values", "mask", "age", "dt"]
        )
    )
    deterministic = bool(experiment.get("deterministic_torch", True))
    seed_state = seed_everything(seed, deterministic_torch=deterministic)
    requested_device = str(experiment.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for offline training but is unavailable")
    device = torch.device(requested_device)
    accelerator_identity = current_run_accelerator_identity(device)
    accelerator_identity.update(
        {
            "training_precision": "float32",
            "autocast_enabled": False,
            "cpu_intraop_threads": torch.get_num_threads(),
        }
    )
    if (
        formal_context is not None
        and device.type == "cuda"
        and accelerator_identity.get("selected_physical_gpu") is None
    ):
        raise RuntimeError(
            "formal run cannot map the selected CUDA device to physical UUID/PCI identity"
        )

    train_arrays = data["train"]
    validation_arrays = data["val"]
    example_features = compose_model_features(
        train_arrays.values[:1],
        train_arrays.mask[:1],
        train_arrays.age[:1],
        train_arrays.dt[:1],
        channels=feature_channels,
    )
    hidden_dim = resolve_hidden_dim(model_config, dataset_name, model_name)
    model = build_offline_model(
        model_name,
        input_dim=example_features.shape[-1],
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
    max_steps = int(training.get("optimizer_steps", 20_000))
    validation_interval = int(training.get("validation_interval", 500))
    batch_size = int(training.get("batch_size", 64))
    evaluation_batch_size = int(training.get("evaluation_batch_size", batch_size))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    if max_steps <= 0 or validation_interval <= 0 or batch_size <= 0:
        raise ValueError("optimizer_steps, validation_interval, and batch_size must be positive")

    run_metadata = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": model_name,
        "seed": int(seed),
        "config_sha256": config_hash,
        "split_manifest_sha256": data.get("manifest_sha256"),
        "feature_channels": feature_channels,
        "input_dim": int(example_features.shape[-1]),
        "hidden_dim": hidden_dim,
        "trainable_parameters": count_trainable_parameters(model),
        "device": str(device),
        "accelerator_identity": accelerator_identity,
        "seed_state": seed_state,
        "fixed_budget_optimizer_steps": max_steps,
        "selection_metric": task.selection_metric,
        "formal_freeze_sha256": (
            None
            if formal_context is None
            else str(formal_context["formal_freeze_sha256"])
        ),
        "source_tree_sha256": (
            None
            if formal_context is None
            else str(formal_context["source_tree"]["aggregate_sha256"])
        ),
        "environment_sha256": (
            None
            if formal_context is None
            else str(formal_context["environment"]["sha256"])
        ),
        "prepared_manifest_sha256": (
            None
            if formal_context is None
            else str(formal_context["prepared_data"]["manifest_sha256"])
        ),
        "parameter_registry_sha256": (
            None
            if formal_context is None
            else str(formal_context["parameter_registry_sha256"])
        ),
        "split_manifest_digest": data.get("manifest_digest"),
        "temporal_semantics": data.get("temporal_semantics"),
        "compute_accounting": {
            "train": sequence_compute_accounting(
                model, train_arrays.dt, train_arrays.lengths
            ),
            "validation": sequence_compute_accounting(
                model, validation_arrays.dt, validation_arrays.lengths
            ),
        },
    }
    start_step = 0
    best_metric = -float("inf")
    best_step = -1
    validation_history: list[dict[str, Any]] = []
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["run_metadata"] != run_metadata:
            raise ValueError(f"Resume metadata mismatch for {run_id}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["optimizer_step"])
        best_metric = float(checkpoint["best_metric"])
        best_step = int(checkpoint["best_step"])
        validation_history = list(checkpoint.get("validation_history", []))
        recorded_steps = [int(item["optimizer_step"]) for item in validation_history]
        if recorded_steps != sorted(set(recorded_steps)):
            raise ValueError(f"Resume validation history is duplicated or unordered for {run_id}")
        if recorded_steps and recorded_steps[-1] > start_step:
            raise ValueError(f"Resume validation history extends past checkpoint for {run_id}")
        # map_location=device also moves the CPU RNG byte tensor to CUDA; the
        # CPU generator explicitly requires a CPU ByteTensor on resume.
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )

    previous_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = {
        **run_metadata,
        "status": "running",
        "started_at_utc": previous_manifest.get("started_at_utc", utc_now()),
        "attempt_started_at_utc": utc_now(),
        "attempt_number": int(previous_manifest.get("attempt_number", 0)) + 1,
        "resumed_from_step": start_step,
        "last_completed_step": start_step,
        "restored_validation_records": len(validation_history),
    }
    atomic_write_json(manifest_path, manifest)
    started = time.perf_counter()
    step_times: list[float] = []
    validation_wall_time_seconds = 0.0
    try:
        for zero_based_step in range(start_step, max_steps):
            step_started = time.perf_counter()
            model.train()
            indices = deterministic_batch_indices(
                len(train_arrays.labels), batch_size, seed, zero_based_step
            )
            features, dt, labels, lengths = _tensor_batch(
                train_arrays,
                indices,
                device=device,
                feature_channels=feature_channels,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, dt, lengths)
            loss = _loss_for_task(task, logits, labels, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite training loss at optimizer step {zero_based_step + 1}"
                )
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip).detach().cpu()
            )
            optimizer.step()
            step_times.append(float(time.perf_counter() - step_started))
            completed_step = zero_based_step + 1

            validation_due = (
                completed_step % validation_interval == 0 or completed_step == max_steps
            )
            if (
                stop_request_path is not None
                and stop_request_path.is_file()
                and not validation_due
            ):
                # A thermal pause preserves the exact optimizer/RNG state.  It
                # does not insert an extra validation or alter checkpoint choice.
                _atomic_torch_save(
                    latest_path,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        step=completed_step,
                        best_metric=best_metric,
                        best_step=best_step,
                        run_metadata=run_metadata,
                        validation_history=validation_history,
                    ),
                )
                manifest["last_completed_step"] = completed_step
                manifest["stop_request"] = str(stop_request_path.resolve())
                atomic_write_json(manifest_path, manifest)
                raise SystemExit("checkpointed pause requested by GPU supervisor")

            if validation_due:
                validation_started = time.perf_counter()
                metrics, _ = evaluate_arrays(
                    model,
                    validation_arrays,
                    task=task,
                    device=device,
                    batch_size=evaluation_batch_size,
                    feature_channels=feature_channels,
                    criterion=criterion,
                )
                validation_wall_time_seconds += float(
                    time.perf_counter() - validation_started
                )
                selection_value = float(metrics[task.selection_metric])
                validation_record = {
                    "optimizer_step": completed_step,
                    "training_loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": gradient_norm,
                    "metrics": metrics,
                }
                validation_history.append(validation_record)
                if selection_value > best_metric:
                    best_metric = selection_value
                    best_step = completed_step
                    _atomic_torch_save(
                        best_path,
                        _checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            step=completed_step,
                            best_metric=best_metric,
                            best_step=best_step,
                            run_metadata=run_metadata,
                            validation_history=validation_history,
                        ),
                    )
                _atomic_torch_save(
                    latest_path,
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        step=completed_step,
                        best_metric=best_metric,
                        best_step=best_step,
                        run_metadata=run_metadata,
                        validation_history=validation_history,
                    ),
                )
                manifest["last_completed_step"] = completed_step
                manifest["best_step"] = best_step
                manifest["best_validation_metric"] = best_metric
                atomic_write_json(manifest_path, manifest)
                if (
                    stop_request_path is not None
                    and stop_request_path.is_file()
                    and completed_step < max_steps
                ):
                    raise SystemExit("checkpointed pause requested by GPU supervisor")

        if not best_path.is_file():
            raise RuntimeError("training completed without a selected validation checkpoint")
        result = {
            **run_metadata,
            "status": "checkpoint_frozen",
            "completed_at_utc": utc_now(),
            "wall_time_seconds": float(time.perf_counter() - started),
            "best_step": best_step,
            "best_validation_metric": best_metric,
            "best_checkpoint": str(best_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path),
            "validation_history": validation_history,
            "optimizer_step_timing": _timing_summary(step_times),
            "validation_wall_time_seconds_this_attempt": validation_wall_time_seconds,
            "optimizer_steps_planned": max_steps,
            "optimizer_steps_completed": max_steps,
            "test_metrics_computed": False,
        }
        atomic_write_json(result_path, result)
        completion = write_completion_artifact(
            result_path=result_path,
            best_path=best_path,
            latest_path=latest_path,
            result=result,
        )
        manifest.update(
            {
                "status": "checkpoint_frozen",
                "completed_at_utc": result["completed_at_utc"],
                "last_completed_step": max_steps,
                "result": str(result_path.resolve()),
                "completion": str((run_dir / "completion.json").resolve()),
                "completion_sha256": completion["completion_sha256"],
                "test_metrics_computed": False,
            }
        )
        atomic_write_json(manifest_path, manifest)
        return result
    except BaseException as exc:
        terminal_status = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        manifest.update(
            {
                "status": terminal_status,
                f"{terminal_status}_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "validation_records_persisted": len(validation_history),
                "optimizer_step_timing_this_attempt": _timing_summary(step_times),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise


def _replace_arrays(arrays: Any, corrupted: Any) -> Any:
    """Create the same dataclass with corruption-updated values/mask/age."""

    return type(arrays)(
        values=corrupted.values,
        mask=corrupted.mask,
        age=corrupted.age,
        dt=arrays.dt,
        labels=arrays.labels,
        ids=arrays.ids,
        group_ids=getattr(arrays, "group_ids", arrays.ids),
        lengths=arrays.lengths,
    )


def evaluate_fixed_checkpoint(
    *,
    model: OfflineSequenceClassifier,
    arrays: Any,
    task: TaskSpec,
    dataset_name: str,
    feature_channels: list[str],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    run_id: str | None = None,
    test_authorization: Mapping[str, Any] | None = None,
    temporal_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one immutable checkpoint over both registered corruption curves."""

    if formal_enabled(config):
        authorized = bool(test_authorization and test_authorization.get("authorized"))
        release = (test_authorization or {}).get("release", {})
        release_core = {
            key: value
            for key, value in release.items()
            if key not in {"checkpoint_release_sha256", "created_at_utc"}
        }
        release_hash_valid = bool(release) and (
            release.get("checkpoint_release_sha256") == canonical_json_hash(release_core)
            and (test_authorization or {}).get("checkpoint_release_sha256")
            == release.get("checkpoint_release_sha256")
        )
        released_ids = {
            str(entry.get("run_id"))
            for entry in release.get("entries", [])
        }
        if (
            not authorized
            or not release_hash_valid
            or run_id is None
            or run_id not in released_ids
        ):
            raise PermissionError(
                "formal test evaluation requires a verified complete checkpoint release"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    corruption_config = dict(config.get("corruption", {}))
    if bool(corruption_config.get("retrain_per_severity", False)):
        raise ValueError(
            "retrain_per_severity must be false: the registered robustness protocol "
            "evaluates one immutable validation-selected checkpoint"
        )
    base_seed = int(corruption_config.get("base_seed", 20261102))
    evaluation_batch_size = int(
        dict(config.get("training", {})).get("evaluation_batch_size", 128)
    )
    records: list[dict[str, Any]] = []
    state_hash_before = _model_state_sha256(model)
    for spec in default_corruption_specs(
        base_seed=base_seed,
        dataset=dataset_name,
        packetwise=task.packetwise_corruption,
        mcar_levels=corruption_config.get("mcar_levels"),
        temporal_block_levels=corruption_config.get("temporal_block_levels"),
    ):
        condition_started = time.perf_counter()
        corrupted = apply_corruption(arrays.values, arrays.mask, arrays.dt, spec)
        condition_arrays = _replace_arrays(arrays, corrupted)
        metrics, logits = evaluate_arrays(
            model,
            condition_arrays,
            task=task,
            device=device,
            batch_size=evaluation_batch_size,
            feature_channels=feature_channels,
        )
        condition_id = f"{spec.kind}-{spec.severity:.1f}"
        prediction_path = output_dir / f"predictions-{condition_id}.npz"
        _atomic_npz(
            prediction_path,
            ids=np.asarray(arrays.ids),
            group_ids=np.asarray(getattr(arrays, "group_ids", arrays.ids)),
            **(
                {"subject_id": np.asarray(getattr(arrays, "group_ids", arrays.ids))}
                if dataset_name == "har"
                else {"patient_id": np.asarray(getattr(arrays, "group_ids", arrays.ids))}
            ),
            labels=np.asarray(arrays.labels),
            logits=logits,
            final_mask=corrupted.mask,
        )
        records.append(
            {
                "condition_id": condition_id,
                "spec": asdict(spec),
                "corruption": corrupted.metadata,
                "metrics": metrics,
                "evaluation_wall_time_seconds": float(
                    time.perf_counter() - condition_started
                ),
                "predictions": str(prediction_path.resolve()),
                "predictions_sha256": sha256_file(prediction_path),
            }
        )

    latency_config = dict(config.get("latency", {}))
    nominal_features = compose_model_features(
        arrays.values[:1],
        arrays.mask[:1],
        arrays.age[:1],
        arrays.dt[:1],
        channels=feature_channels,
    )
    latency_values = torch.from_numpy(nominal_features).to(device)
    latency_dt = torch.from_numpy(np.asarray(arrays.dt[:1], dtype=np.float32)).to(device)
    latency_lengths = torch.from_numpy(
        np.asarray(arrays.lengths[:1], dtype=np.int64)
    ).to(device)
    model.eval()
    latency = measure_batch1_latency(
        lambda: model(latency_values, latency_dt, latency_lengths),
        device=device,
        warmup=int(latency_config.get("warmup", 1000)),
        repeats=int(latency_config.get("repeats", 10000)),
    )
    sequence_length = int(latency_values.shape[1])
    latency.update(
        {
            "measurement_scope": "full_padded_sequence_forward",
            "sequence_length_updates": sequence_length,
            "median_total_sequence_ms": latency["median_ms"],
            "p95_total_sequence_ms": latency["p95_ms"],
            "full_sequence_deadline_miss_fraction_20ms": latency[
                "deadline_miss_fraction_20ms"
            ],
            "deadline_applies_to_full_sequence_not_single_update": True,
            "amortized_median_ms_per_update": float(
                latency["median_ms"] / sequence_length
            ),
            "amortized_p95_ms_per_update": float(latency["p95_ms"] / sequence_length),
            "per_update_values_are_amortized_not_independently_timed": True,
            "compute_accounting": sequence_compute_accounting(
                model,
                np.asarray(arrays.dt[:1]),
                np.asarray(arrays.lengths[:1]),
            ),
        }
    )
    state_hash_after = _model_state_sha256(model)
    if state_hash_after != state_hash_before:
        raise AssertionError("Fixed-checkpoint evaluation mutated the model state")
    summary = {
        "fixed_checkpoint": True,
        "retrained_per_severity": False,
        "optimizer_steps_during_robustness_evaluation": 0,
        "model_state_sha256_before": state_hash_before,
        "model_state_sha256_after": state_hash_after,
        "trainable_parameters": count_trainable_parameters(model),
        "resampling_unit": "subject" if dataset_name == "har" else "patient",
        "run_id": run_id,
        "checkpoint_release_sha256": (
            None
            if test_authorization is None
            else test_authorization.get("checkpoint_release_sha256")
        ),
        "temporal_semantics": (
            None if temporal_semantics is None else dict(temporal_semantics)
        ),
        "compute_accounting": sequence_compute_accounting(
            model, np.asarray(arrays.dt), np.asarray(arrays.lengths)
        ),
        "conditions": records,
        "latency": latency,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def load_all_datasets(
    config: dict[str, Any],
    config_path: Path,
    *,
    selected_datasets: list[str] | tuple[str, ...] | None = None,
    include_test: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load requested datasets using one frozen manifest per task."""

    from .data import (
        load_har_dataset,
        load_har_official,
        load_physionet_dataset,
        load_physionet_raw,
    )
    from .splits import (
        create_har_split_manifest,
        create_physionet_split_manifest,
        load_split_manifest,
        validate_har_manifest_identity,
        validate_physionet_manifest_identity,
        write_split_manifest,
    )

    data_config = dict(config.get("data", {}))
    root = Path(data_config.get("root", "revision_data"))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    # Rehash the prepared source files at the start of every launched benchmark.
    # A cached extraction marker is not accepted as evidence of completeness.
    verification = verify_prepared_data(root)
    prepared_manifest_path = root / "manifest.json"
    prepared_manifest_sha256 = sha256_file(prepared_manifest_path)
    manifest_dir = Path(data_config.get("manifest_dir", "manifests/offline"))
    if not manifest_dir.is_absolute():
        manifest_dir = (config_path.parent / manifest_dir).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    selected = set(selected_datasets or ("har", "physionet2012"))
    unknown = selected - {"har", "physionet2012"}
    if unknown:
        raise ValueError(f"Unknown requested offline datasets: {sorted(unknown)}")
    loaded: dict[str, dict[str, Any]] = {}
    if "har" in selected:
        har_config = dict(data_config.get("har", {}))
        har_root = root / "har" / "UCI HAR Dataset"
        har_manifest_path = manifest_dir / "har_split.json"
        official = load_har_official(har_root)
        validation_fraction = float(
            har_config.get("validation_subject_fraction", 0.20)
        )
        split_seed = int(har_config.get("split_seed", 20261102))
        if not har_manifest_path.is_file():
            manifest = create_har_split_manifest(
                official["official_train"].ids,
                official["train_subjects"],
                official["official_test"].ids,
                official["test_subjects"],
                validation_fraction=validation_fraction,
                split_seed=split_seed,
            )
            write_split_manifest(har_manifest_path, manifest)
        observed_har_manifest = load_split_manifest(har_manifest_path)
        validate_har_manifest_identity(
            observed_har_manifest,
            train_ids=official["official_train"].ids,
            train_subjects=official["train_subjects"],
            test_ids=official["official_test"].ids,
            test_subjects=official["test_subjects"],
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        har = load_har_dataset(
            har_root,
            observed_har_manifest,
            include_test=include_test,
            preloaded=official,
        )
        har["manifest_sha256"] = sha256_file(har_manifest_path)
        har["manifest_digest"] = observed_har_manifest.digest
        har["manifest_path"] = str(har_manifest_path.resolve())
        har["prepared_data_verification"] = verification
        har["prepared_manifest_sha256"] = prepared_manifest_sha256
        loaded["har"] = har

    if "physionet2012" in selected:
        physio_config = dict(data_config.get("physionet2012", {}))
        physionet_root = root / "physionet2012"
        physionet_manifest_path = manifest_dir / "physionet2012_split.json"
        n_hours = int(physio_config.get("horizon_hours", 48))
        if formal_enabled(config) and n_hours != 48:
            raise ValueError(
                "formal PhysioNet protocol requires a fixed 48-hour hourly grid"
            )
        expected_records = int(physio_config.get("expected_records", 4000))
        train_fraction = float(physio_config.get("train_fraction", 0.70))
        validation_fraction = float(physio_config.get("validation_fraction", 0.15))
        test_fraction = float(physio_config.get("test_fraction", 0.15))
        split_seed = int(physio_config.get("split_seed", 20261102))
        raw_physionet = load_physionet_raw(
            physionet_root, expected_records=expected_records, n_hours=n_hours
        )
        if not physionet_manifest_path.is_file():
            manifest = create_physionet_split_manifest(
                raw_physionet["all"].ids,
                raw_physionet["all"].labels,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                split_seed=split_seed,
            )
            write_split_manifest(physionet_manifest_path, manifest)
        observed_physionet_manifest = load_split_manifest(physionet_manifest_path)
        validate_physionet_manifest_identity(
            observed_physionet_manifest,
            record_ids=raw_physionet["all"].ids,
            outcomes=raw_physionet["all"].labels,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
        )
        physionet = load_physionet_dataset(
            physionet_root,
            observed_physionet_manifest,
            expected_records=expected_records,
            n_hours=n_hours,
            include_test=include_test,
            preloaded=raw_physionet,
        )
        physionet["manifest_sha256"] = sha256_file(physionet_manifest_path)
        physionet["manifest_digest"] = observed_physionet_manifest.digest
        physionet["manifest_path"] = str(physionet_manifest_path.resolve())
        physionet["prepared_data_verification"] = verification
        physionet["prepared_manifest_sha256"] = prepared_manifest_sha256
        loaded["physionet2012"] = physionet
    return loaded


def run_offline_benchmark(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Train and freeze all checkpoints; never compute test predictions here."""

    config_path = Path(config_path).resolve()
    config, config_hash = load_config(config_path)
    experiment = dict(config.get("experiment", {}))
    seeds = [int(item) for item in experiment.get("seeds", [101, 202, 303, 404, 505])]
    models = [str(item) for item in dict(config.get("models", {})).get(
        "names", ["lstm", "ode-rnn", "ltc", "cfc"]
    )]
    selected_datasets = [
        str(item) for item in experiment.get("datasets", ["har", "physionet2012"])
    ]
    datasets = load_all_datasets(
        config,
        config_path,
        selected_datasets=selected_datasets,
        include_test=False,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    formal_context: Mapping[str, Any] | None = None
    if formal_enabled(config):
        formal_context = create_formal_freeze(
            config=config,
            config_path=config_path,
            config_hash=config_hash,
            datasets=datasets,
            output_root=output,
        )
    aggregate_path = output / "offline_benchmark_manifest.json"
    previous: dict[str, Any] = {}
    if aggregate_path.is_file():
        previous = json.loads(aggregate_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_hash:
            raise ValueError(
                "Existing offline benchmark output belongs to a different config hash"
            )
    attempts = list(previous.get("attempts", []))
    attempt = {
        "attempt_number": len(attempts) + 1,
        "started_at_utc": utc_now(),
        "status": "running",
    }
    run_records = {
        str(item["run_id"]): dict(item) for item in previous.get("runs", [])
    }
    aggregate: dict[str, Any] = {
        "status": "running",
        "config": str(config_path),
        "config_sha256": config_hash,
        "started_at_utc": previous.get("started_at_utc", utc_now()),
        "attempts": attempts,
        "current_attempt": attempt,
        "runs": list(run_records.values()),
        "test_metrics_computed": False,
        "formal_freeze_sha256": (
            None
            if formal_context is None
            else formal_context["formal_freeze_sha256"]
        ),
    }
    atomic_write_json(aggregate_path, aggregate)
    try:
        for dataset_name in selected_datasets:
            if dataset_name not in datasets or dataset_name not in TASKS:
                raise ValueError(f"Unknown offline dataset: {dataset_name}")
            for model_name in models:
                for seed in seeds:
                    result = train_fixed_budget(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        seed=seed,
                        data=datasets[dataset_name],
                        config=config,
                        config_hash=config_hash,
                        output_root=output / "runs",
                        formal_context=formal_context,
                    )
                    run_records[result["run_id"]] = {
                        "run_id": result["run_id"],
                        "status": result["status"],
                        "result": str(
                            (output / "runs" / result["run_id"] / "result.json").resolve()
                        ),
                    }
                    aggregate["runs"] = list(run_records.values())
                    atomic_write_json(aggregate_path, aggregate)
    except BaseException as exc:
        terminal_status = (
            "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        attempt.update(
            {
                "status": terminal_status,
                f"{terminal_status}_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        attempts.append(attempt)
        aggregate.update(
            {
                "status": terminal_status,
                "attempts": attempts,
                "current_attempt": None,
                "runs": list(run_records.values()),
                f"{terminal_status}_at_utc": attempt[f"{terminal_status}_at_utc"],
            }
        )
        atomic_write_json(aggregate_path, aggregate)
        raise
    release: Mapping[str, Any] | None = None
    if formal_context is not None:
        release = publish_checkpoint_release(
            config=config,
            config_hash=config_hash,
            output_root=output,
            formal_freeze=formal_context,
        )
    final_status = (
        "checkpoint_release_frozen" if release is not None else "checkpoint_set_frozen"
    )
    attempt.update({"status": final_status, "completed_at_utc": utc_now()})
    attempts.append(attempt)
    aggregate.update(
        {
            "status": final_status,
            "completed_at_utc": attempt["completed_at_utc"],
            "attempts": attempts,
            "current_attempt": None,
            "runs": list(run_records.values()),
            "run_count": len(run_records),
            "expected_run_count": len(selected_datasets) * len(models) * len(seeds),
            "checkpoint_release": (
                None
                if release is None
                else str((output / "checkpoint_release.json").resolve())
            ),
            "checkpoint_release_sha256": (
                None if release is None else release["checkpoint_release_sha256"]
            ),
            "test_metrics_computed": False,
        }
    )
    atomic_write_json(aggregate_path, aggregate)
    return aggregate


def run_offline_test_release(
    config_path: str | Path,
    training_output_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the single authorized test sweep after the complete release verifies.

    The release and every completed artifact are rehashed before datasets are
    loaded with ``include_test=True``.  A durable one-shot claim is then written
    before inference so a failed or partial sweep cannot silently be repeated.
    """

    from .analysis import hierarchical_paired_analysis

    config_path = Path(config_path).resolve()
    config, config_hash = load_config(config_path)
    if not formal_enabled(config):
        raise PermissionError("the unified test-release command requires formal.enabled=true")
    training_output = Path(training_output_dir).resolve()
    # This is intentionally the first substantive operation: no test arrays or
    # metrics are accessible until freeze, release, and every artifact rehash.
    authorization = verify_checkpoint_release(
        config=config,
        config_path=config_path,
        config_hash=config_hash,
        output_root=training_output,
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"test release output must be a fresh path: {output}")
    state_path = training_output / "test_release_state.json"
    if state_path.exists():
        raise PermissionError(
            "a test-release claim already exists; repeat access requires explicit adjudication"
        )
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "authorized_running",
        "started_at_utc": utc_now(),
        "config_sha256": config_hash,
        "formal_freeze_sha256": authorization["formal_freeze_sha256"],
        "checkpoint_release_sha256": authorization["checkpoint_release_sha256"],
        "output_dir": str(output),
        "test_metrics_computed": False,
    }
    atomic_write_json(state_path, state)
    atomic_write_json(output / "test_authorization.json", state)
    try:
        experiment = dict(config.get("experiment", {}))
        selected_datasets = [
            str(value)
            for value in experiment.get("datasets", ["har", "physionet2012"])
        ]
        datasets = load_all_datasets(
            config,
            config_path,
            selected_datasets=selected_datasets,
            include_test=True,
        )
        model_config = dict(config.get("models", {}))
        requested_device = str(
            experiment.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for formal test inference but is unavailable")
        device = torch.device(requested_device)
        records: list[dict[str, Any]] = []
        run_summaries: list[dict[str, Any]] = []
        for entry in authorization["release"]["entries"]:
            run_id = str(entry["run_id"])
            dataset_name = str(entry["dataset"])
            model_name = str(entry["model"])
            seed = int(entry["seed"])
            result = json.loads(Path(entry["result_path"]).read_text(encoding="utf-8"))
            model = build_offline_model(
                model_name,
                input_dim=int(result["input_dim"]),
                hidden_dim=int(result["hidden_dim"]),
                output_dim=TASKS[dataset_name].output_dim,
                ode_max_step=float(model_config.get("ode_max_step", 1.0)),
                ltc_ode_unfolds=int(model_config.get("ltc_ode_unfolds", 6)),
                cfc_mode=str(model_config.get("cfc_mode", "default")),
            ).to(device)
            checkpoint = torch.load(
                Path(entry["best_checkpoint_path"]),
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state"])
            summary = evaluate_fixed_checkpoint(
                model=model,
                arrays=datasets[dataset_name]["test"],
                task=TASKS[dataset_name],
                dataset_name=dataset_name,
                feature_channels=list(result["feature_channels"]),
                config=config,
                device=device,
                output_dir=output / "runs" / run_id,
                run_id=run_id,
                test_authorization=authorization,
                temporal_semantics=datasets[dataset_name].get("temporal_semantics"),
            )
            summary_path = output / "runs" / run_id / "summary.json"
            run_summaries.append(
                {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "summary": str(summary_path.resolve()),
                    "summary_sha256": sha256_file(summary_path),
                }
            )
            for condition in summary["conditions"]:
                records.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "condition_kind": condition["spec"]["kind"],
                        "severity": float(condition["spec"]["severity"]),
                        "predictions": condition["predictions"],
                        "predictions_sha256": condition["predictions_sha256"],
                    }
                )
        formal_config = dict(config.get("formal", {}))
        analysis_path = output / "paired_hierarchical_analysis.json"
        hierarchical_paired_analysis(
            records,
            bootstrap_replicates=int(formal_config.get("bootstrap_replicates", 10_000)),
            analysis_seed=int(formal_config.get("analysis_seed", 20_261_105)),
            comparisons=formal_config.get("comparisons"),
            output_path=analysis_path,
        )
        manifest = {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "config_sha256": config_hash,
            "formal_freeze_sha256": authorization["formal_freeze_sha256"],
            "checkpoint_release_sha256": authorization[
                "checkpoint_release_sha256"
            ],
            "fixed_checkpoint_test_sweep_count": 1,
            "test_metrics_computed": True,
            "runs": run_summaries,
            "run_count": len(run_summaries),
            "analysis": str(analysis_path.resolve()),
            "analysis_sha256": sha256_file(analysis_path),
        }
        atomic_write_json(output / "offline_test_release_manifest.json", manifest)
        state.update(
            {
                "status": "completed",
                "completed_at_utc": manifest["completed_at_utc"],
                "test_metrics_computed": True,
                "manifest": str(
                    (output / "offline_test_release_manifest.json").resolve()
                ),
            }
        )
        atomic_write_json(state_path, state)
        return manifest
    except BaseException as exc:
        state.update(
            {
                "status": "failed_or_interrupted_locked",
                "failed_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "test_metrics_may_have_been_partially_computed": True,
            }
        )
        atomic_write_json(state_path, state)
        raise

