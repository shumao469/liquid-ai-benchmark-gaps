"""Resumable observer training with fixed validation checkpoint selection."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data.closed_loop import load_closed_loop_dataset
from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file
from ..models.liquid import load_graph_safe_state_dict
from .manifests import load_trajectory_manifest, verify_split_datasets
from .parameter_matching import (
    build_parameter_registry,
    build_registered_model,
    load_parameter_registry,
)


@dataclass(frozen=True)
class Normalizer:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> "Normalizer":
        return cls(
            input_mean=np.asarray(mapping["input_mean"], dtype=np.float32),
            input_std=np.asarray(mapping["input_std"], dtype=np.float32),
            target_mean=np.asarray(mapping["target_mean"], dtype=np.float32),
            target_std=np.asarray(mapping["target_std"], dtype=np.float32),
        )


def fit_normalizer(arrays: dict[str, np.ndarray]) -> Normalizer:
    x = np.asarray(arrays["observer_input"], dtype=np.float32)
    y = np.asarray(arrays["target_state"], dtype=np.float32)
    input_std = x.std(axis=0)
    target_std = y.std(axis=0)
    return Normalizer(
        input_mean=x.mean(axis=0).astype(np.float32),
        input_std=np.maximum(input_std, 1e-6).astype(np.float32),
        target_mean=y.mean(axis=0).astype(np.float32),
        target_std=np.maximum(target_std, 1e-6).astype(np.float32),
    )


class SequenceWindows:
    """Deterministic non-overlapping windows that never cross episode boundaries."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        normalizer: Normalizer,
        window_steps: int,
        burn_in_steps: int = 0,
    ) -> None:
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        raw_x = np.asarray(arrays["observer_input"], dtype=np.float32)
        self.x = (raw_x - normalizer.input_mean) / normalizer.input_std
        self.y = (
            np.asarray(arrays["target_state"], dtype=np.float32)
            - normalizer.target_mean
        ) / normalizer.target_std
        self.dt = np.asarray(
            arrays["observed_timestamp_dt"], dtype=np.float32
        ).reshape(-1)
        offsets = np.asarray(arrays["episode_offsets"], dtype=np.int64)
        if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(self.x):
            raise ValueError("Invalid episode_offsets")
        # A packet may be absent before any value has ever been observed.  The
        # raw ZOH fallback is zero, but its normalized representation is defined
        # as zero (training mean) rather than as a potentially extreme z-score.
        # Once a channel has been observed, the held raw value is normalized in
        # the ordinary way.
        if raw_x.shape[1] >= 8:
            masks = raw_x[:, 4:8] > 0.5
            for episode_start, episode_stop in zip(offsets[:-1], offsets[1:]):
                initialized = np.zeros(4, dtype=bool)
                for step in range(int(episode_start), int(episode_stop)):
                    missing_without_history = ~initialized & ~masks[step]
                    self.x[step, :4][missing_without_history] = 0.0
                    initialized |= masks[step]
        self.window_steps = int(window_steps)
        self.burn_in_steps = int(burn_in_steps)
        if not 0 <= self.burn_in_steps < self.window_steps:
            raise ValueError("burn_in_steps must be in [0, window_steps)")
        self.slices: list[tuple[int, int]] = []
        for start, stop in zip(offsets[:-1], offsets[1:]):
            for left in range(int(start), int(stop), self.window_steps):
                self.slices.append((left, min(left + self.window_steps, int(stop))))
        if not self.slices:
            raise ValueError("Dataset contains no windows")
        self.actual_sequence_steps = int(
            sum(int(stop) - int(start) for start, stop in self.slices)
        )
        self.executed_padded_steps = int(len(self.slices) * self.window_steps)
        self.padding_steps = int(
            self.executed_padded_steps - self.actual_sequence_steps
        )

    def __len__(self) -> int:
        return len(self.slices)

    def batch(self, indices: list[int], device: torch.device):
        batch = len(indices)
        x = np.zeros((batch, self.window_steps, self.x.shape[1]), dtype=np.float32)
        y = np.zeros((batch, self.window_steps, self.y.shape[1]), dtype=np.float32)
        dt = np.ones((batch, self.window_steps), dtype=np.float32)
        valid = np.zeros((batch, self.window_steps), dtype=np.float32)
        for row, index in enumerate(indices):
            start, stop = self.slices[index]
            length = stop - start
            x[row, :length] = self.x[start:stop]
            y[row, :length] = self.y[start:stop]
            dt[row, :length] = self.dt[start:stop]
            if length < self.window_steps:
                dt[row, length:] = self.dt[stop - 1]
            valid[row, :length] = 1.0
            valid[row, : min(length, self.burn_in_steps)] = 0.0
        return (
            torch.as_tensor(x, device=device),
            torch.as_tensor(dt, device=device),
            torch.as_tensor(y, device=device),
            torch.as_tensor(valid, device=device),
        )


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    squared = torch.square(prediction - target).mean(dim=-1)
    return (squared * valid).sum() / valid.sum().clamp_min(1.0)


def _sequence_compute_accounting(
    windows: SequenceWindows,
    *,
    registered_name: str,
    config: dict[str, Any],
) -> dict[str, int]:
    """Count recurrent updates and registered solver work for one full epoch."""
    valid_nfe = 0
    padded_nfe = 0
    valid_unfolds = 0
    padded_unfolds = 0
    if registered_name in {"odernn", "ctrnn"}:
        model_key = "ode" if registered_name == "odernn" else "ctrnn"
        max_step = float(config.get("models", {}).get(model_key, {}).get("max_step", 0.02))
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError(f"Invalid {registered_name} RK4 max_step")
        for start, stop in windows.slices:
            actual_dt = windows.dt[int(start) : int(stop)]
            valid_nfe += int(
                np.sum(4 * np.maximum(1, np.ceil(actual_dt / max_step)).astype(int))
            )
            padding = windows.window_steps - (int(stop) - int(start))
            if padding:
                last_dt = float(windows.dt[int(stop) - 1])
                padded_nfe += int(
                    padding * 4 * max(1, int(math.ceil(last_dt / max_step)))
                )
    elif registered_name == "ltc":
        unfolds = int(
            config.get("models", {}).get("liquid", {}).get("ode_unfolds", 6)
        )
        valid_unfolds = int(windows.actual_sequence_steps * unfolds)
        padded_unfolds = int(windows.padding_steps * unfolds)
    return {
        "valid_recurrent_updates": int(windows.actual_sequence_steps),
        "padded_recurrent_updates": int(windows.padding_steps),
        "executed_recurrent_updates": int(windows.executed_padded_steps),
        "valid_rk4_nfe": int(valid_nfe),
        "padded_rk4_nfe": int(padded_nfe),
        "executed_rk4_nfe": int(valid_nfe + padded_nfe),
        "valid_ltc_internal_unfolds": int(valid_unfolds),
        "padded_ltc_internal_unfolds": int(padded_unfolds),
        "executed_ltc_internal_unfolds": int(valid_unfolds + padded_unfolds),
    }


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Confirmatory reproducibility is fail-closed: an unsupported
    # nondeterministic kernel must stop the run rather than emit a warning.
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save_new(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a write-once torch payload.

    Epoch validation checkpoints are evidence, not rolling state.  A hard link
    from a fully fsynced temporary file gives us an atomic, no-overwrite commit
    on the same volume (``os.link`` fails if the destination already exists).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _epoch_checkpoint_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("epoch_*.pt")) if directory.is_dir() else []
    expected_names = [f"epoch_{index:04d}.pt" for index in range(len(paths))]
    if [path.name for path in paths] != expected_names:
        raise ValueError("Epoch checkpoint sequence is not contiguous from epoch zero")
    return paths


def _epoch_checkpoint_ledger(
    checkpoint_directory: Path,
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Deep-read immutable epoch files and construct their deterministic ledger."""
    rows: list[dict[str, Any]] = []
    previous_payload: dict[str, Any] | None = None
    previous_checkpoint_sha256: str | None = None
    previous_entry_sha256: str | None = None
    for expected_epoch, path in enumerate(_epoch_checkpoint_paths(checkpoint_directory)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in expected_identity.items():
            if payload.get(key) != value:
                raise ValueError(
                    f"Epoch checkpoint identity mismatch for {key}: {path}"
                )
        history = payload.get("history")
        checkpoint_sha256 = sha256_file(path)
        required_payload = {
            "model_state_dict",
            "optimizer_state_dict",
            "rng_state",
            "normalizer",
            "data_sha256",
            "steps_per_epoch",
            "previous_epoch_checkpoint_sha256",
            "history_sha256",
            "selection_rule",
        }
        if (
            int(payload.get("epoch", -1)) != expected_epoch
            or not isinstance(history, list)
            or len(history) != expected_epoch + 1
            or int(history[-1].get("epoch", -1)) != expected_epoch
            or not math.isfinite(float(history[-1].get("val_loss", math.nan)))
            or not math.isfinite(float(history[-1].get("train_loss", math.nan)))
            or int(payload.get("optimizer_steps_completed", -1))
            != int(history[-1].get("optimizer_steps_completed", -2))
            or int(payload.get("optimizer_steps_completed", -1))
            != (expected_epoch + 1) * int(payload.get("steps_per_epoch", -2))
            or payload.get("history_sha256") != canonical_json_hash(history)
            or payload.get("previous_epoch_checkpoint_sha256")
            != previous_checkpoint_sha256
            or payload.get("selection_rule")
            != "minimum_finite_validation_loss_then_earliest_epoch"
            or required_payload - set(payload)
            or not isinstance(payload.get("model_state_dict"), dict)
            or not payload["model_state_dict"]
            or not isinstance(payload.get("optimizer_state_dict"), dict)
            or {"state", "param_groups"} - set(payload["optimizer_state_dict"])
            or not isinstance(payload.get("rng_state"), dict)
            or not isinstance(payload.get("normalizer"), dict)
        ):
            raise ValueError(f"Malformed immutable epoch checkpoint: {path}")
        if previous_payload is not None and history[:-1] != previous_payload["history"]:
            raise ValueError(f"Epoch checkpoint history fork detected: {path}")
        entry_core = {
            "epoch": expected_epoch,
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "previous_checkpoint_sha256": previous_checkpoint_sha256,
            "previous_entry_sha256": previous_entry_sha256,
            "validation_loss": float(history[-1]["val_loss"]),
            "training_loss": float(history[-1]["train_loss"]),
            "optimizer_steps_completed": int(payload["optimizer_steps_completed"]),
            "history_sha256": canonical_json_hash(history),
        }
        entry = {**entry_core, "entry_sha256": canonical_json_hash(entry_core)}
        rows.append(entry)
        previous_payload = payload
        previous_checkpoint_sha256 = checkpoint_sha256
        previous_entry_sha256 = entry["entry_sha256"]
    core = {
        "schema_version": 1,
        "producer": "liquid_revision.closed_loop.training",
        "write_once_epoch_checkpoints": True,
        "identity": expected_identity,
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }
    return {**core, "checkpoint_ledger_sha256": canonical_json_hash(core)}


def _validate_existing_ledger_prefix(
    ledger_path: Path, rebuilt: dict[str, Any]
) -> None:
    """Require an existing mutable index to be an exact prefix, never a fork."""
    if not ledger_path.is_file():
        return
    existing = json.loads(ledger_path.read_text(encoding="utf-8"))
    existing_core = {
        key: value
        for key, value in existing.items()
        if key != "checkpoint_ledger_sha256"
    }
    if canonical_json_hash(existing_core) != existing.get(
        "checkpoint_ledger_sha256"
    ):
        raise ValueError("Existing checkpoint ledger self-hash mismatch")
    existing_rows = existing.get("checkpoints")
    rebuilt_rows = rebuilt.get("checkpoints")
    if (
        existing.get("identity") != rebuilt.get("identity")
        or not isinstance(existing_rows, list)
        or len(existing_rows) > len(rebuilt_rows)
        or existing_rows != rebuilt_rows[: len(existing_rows)]
        or int(existing.get("checkpoint_count", -1)) != len(existing_rows)
    ):
        raise ValueError("Existing checkpoint ledger is not an exact immutable prefix")


def _write_epoch_checkpoint_ledger(
    ledger_path: Path,
    checkpoint_directory: Path,
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    ledger = _epoch_checkpoint_ledger(
        checkpoint_directory, expected_identity=expected_identity
    )
    _validate_existing_ledger_prefix(ledger_path, ledger)
    atomic_write_json(ledger_path, ledger)
    return ledger


def _repair_incomplete_epoch_transaction(
    *,
    checkpoint_directory: Path,
    ledger_path: Path,
    latest_path: Path,
    best_path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Publish a fully written pending epoch after a process interruption.

    The immutable epoch file is the commit record.  If a process died before
    updating ``latest.pt``/``best.pt``/the ledger, resume deterministically
    completes those derived rolling files without re-running or skipping the
    epoch.
    """
    ledger = _epoch_checkpoint_ledger(
        checkpoint_directory, expected_identity=expected_identity
    )
    rows = ledger["checkpoints"]
    if not rows:
        return None
    _validate_existing_ledger_prefix(ledger_path, ledger)
    newest_path = Path(rows[-1]["checkpoint_path"])
    newest = torch.load(newest_path, map_location="cpu", weights_only=False)
    latest_epoch = -1
    if latest_path.is_file():
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        for key, value in expected_identity.items():
            if latest.get(key) != value:
                raise ValueError(f"Latest checkpoint identity mismatch for {key}")
        latest_epoch = int(latest.get("epoch", -1))
        if latest_epoch > int(newest["epoch"]):
            raise ValueError("Rolling latest checkpoint is ahead of immutable ledger")
    if latest_epoch < int(newest["epoch"]):
        _atomic_torch_save(latest_path, newest)
    finite_losses = [float(row["validation_loss"]) for row in rows]
    best_epoch = min(range(len(finite_losses)), key=lambda index: (finite_losses[index], index))
    if not 0 <= best_epoch < len(rows):
        raise ValueError("Pending epoch checkpoint records an invalid best epoch")
    selected_path = Path(rows[best_epoch]["checkpoint_path"])
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    if (
        not best_path.is_file()
        or int(
            torch.load(best_path, map_location="cpu", weights_only=False).get(
                "epoch", -1
            )
        )
        != best_epoch
    ):
        _atomic_torch_save(best_path, selected)
    atomic_write_json(ledger_path, ledger)
    atomic_write_json(latest_path.parent / "history.json", newest["history"])
    return newest


def _batches(length: int, batch_size: int, seed: int, shuffle: bool) -> list[list[int]]:
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(length, generator=generator).tolist()
    else:
        order = list(range(length))
    return [order[index : index + batch_size] for index in range(0, length, batch_size)]


def _run_epoch(
    model: nn.Module,
    windows: SequenceWindows,
    *,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    order_seed: int,
) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    batch_plan = _batches(len(windows), batch_size, order_seed, training)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for indices in batch_plan:
            x, dt, y, valid = windows.batch(indices, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x, dt)
            loss = _masked_mse(prediction, y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite observer loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
            count += len(indices)
    return total / max(1, count), len(batch_plan)


def _train_registered_model_impl(
    *,
    registered_name: str,
    seed: int,
    config: dict[str, Any],
    data_root: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    resume: bool = False,
    run_phase: str = "confirmatory",
    optimizer_override: dict[str, float] | None = None,
    expected_pretraining_master_sha256: str | None = None,
    pretraining_audit_receipt_path: str | Path | None = None,
    expected_pretraining_audit_receipt_sha256: str | None = None,
    expected_pretraining_audit_receipt_file_sha256: str | None = None,
    run_registry_path: str | Path | None = None,
    run_attempt_id: str | None = None,
    run_reservation_token_path: str | Path | None = None,
    expected_run_reservation_token_sha256: str | None = None,
    tuning_candidate_identity: dict[str, Any] | None = None,
    tuning_lease_validation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one registered seed and select its checkpoint on the fixed val split."""
    data_directory = Path(data_root).resolve()
    verify_split_datasets(data_directory, expected_config=config)
    split_manifest = load_trajectory_manifest(
        data_directory / "trajectory_manifest.json", expected_config=config
    )
    registry = load_parameter_registry(registry_path, expected_config=config)
    current_config_sha256 = canonical_json_hash(config)
    if split_manifest.get("source_config_sha256") != current_config_sha256:
        raise ValueError(
            "Current config hash differs from the frozen trajectory manifest source"
        )
    if registry.get("source_config_sha256") != current_config_sha256:
        raise ValueError(
            "Current config hash differs from the frozen parameter-registry source"
        )
    if not registry.get("implementation", {}).get("aggregate_sha256"):
        raise ValueError("Parameter registry lacks a frozen implementation hash")
    if run_phase not in {"confirmatory", "validation_tuning"}:
        raise ValueError(f"Unknown training phase: {run_phase}")
    from .formal import (
        apply_and_validate_precision_policy,
        current_run_accelerator_identity,
        formal_artifact_identities,
        is_formal,
        tuning_design_identity,
        validate_tuning_selection,
        validate_tuning_runtime_identity,
        validate_tuning_namespace,
        validate_tuning_lease_authorization,
        validate_pretraining_master_release,
        validate_environment_freeze,
    )

    if is_formal(config) and run_phase == "validation_tuning":
        if resume:
            raise PermissionError(
                "Formal validation-tuning candidates are one-shot; in-place resume "
                "is forbidden after authorization"
            )
        required_tuning_identity = {
            "tuning_namespace_sha256",
            "tuning_namespace_file_sha256",
            "tuning_namespace_instance_id",
            "tuning_attempt_id",
            "tuning_slot_index",
            "tuning_candidate_id",
            "tuning_candidate_output_dir",
            "tuning_candidate_result_path",
            "tuning_gpu_identity",
            "tuning_trusted_nvidia_smi",
            "tuning_lease_authorization_path",
            "tuning_lease_authorization_sha256",
            "tuning_lease_authorization_file_sha256",
            "tuning_lease_custody_receipt_sha256",
            "tuning_lease_custody_receipt_file_sha256",
            "tuning_lease_session_id",
            "tuning_lease_session_index",
            "tuning_lease_session_acquisition_sha256",
            "tuning_lease_global_active_event_sha256",
            "tuning_idle_gate_evidence_sha256",
            "tuning_authorized_worker_pid",
            "tuning_worker_capability_nonce_sha256",
            "tuning_worker_ready_sha256",
            "tuning_output_reservation_sha256",
            "tuning_output_reservation_file_sha256",
            "tuning_output_directory_device",
            "tuning_output_directory_inode",
        }
        if not isinstance(tuning_candidate_identity, dict) or (
            required_tuning_identity - set(tuning_candidate_identity)
        ):
            raise PermissionError(
                "Formal validation tuning requires an externally anchored namespace slot"
            )
        required_lease_context = {
            "tuning_namespace_path",
            "expected_tuning_namespace_sha256",
            "expected_tuning_namespace_file_sha256",
            "tuning_lease_authorization_path",
            "expected_tuning_lease_authorization_sha256",
            "expected_tuning_lease_authorization_file_sha256",
            "tuning_lease_custody_receipt_path",
            "expected_tuning_lease_custody_receipt_sha256",
            "expected_tuning_lease_custody_receipt_file_sha256",
            "tuning_attempt_id",
            "expected_worker_pid",
            "expected_worker_capability_nonce_sha256",
            "tuning_output_reservation_path",
            "expected_tuning_output_reservation_sha256",
            "expected_tuning_output_reservation_file_sha256",
        }
        if not isinstance(tuning_lease_validation_context, dict) or (
            required_lease_context - set(tuning_lease_validation_context)
        ):
            raise PermissionError(
                "Formal validation tuning requires an active lease authorization context"
            )

        def _assert_active_tuning_lease() -> dict[str, Any]:
            assert tuning_candidate_identity is not None
            assert tuning_lease_validation_context is not None
            namespace = validate_tuning_namespace(
                config,
                registry,
                split_manifest,
                namespace_path=tuning_lease_validation_context[
                    "tuning_namespace_path"
                ],
                expected_namespace_sha256=tuning_lease_validation_context[
                    "expected_tuning_namespace_sha256"
                ],
                expected_namespace_file_sha256=tuning_lease_validation_context[
                    "expected_tuning_namespace_file_sha256"
                ],
            )
            matching = [
                slot
                for slot in namespace["candidate_slots"]
                if slot["attempt_id"]
                == tuning_lease_validation_context["tuning_attempt_id"]
                and int(slot["slot_index"])
                == int(tuning_candidate_identity["tuning_slot_index"])
            ]
            if len(matching) != 1:
                raise PermissionError(
                    "Active tuning lease context is not a unique namespace slot"
                )
            slot = matching[0]
            namespace_expected = {
                "tuning_namespace_sha256": namespace[
                    "tuning_namespace_sha256"
                ],
                "tuning_namespace_file_sha256": tuning_lease_validation_context[
                    "expected_tuning_namespace_file_sha256"
                ],
                "tuning_namespace_instance_id": namespace[
                    "namespace_instance_id"
                ],
                "tuning_attempt_id": slot["attempt_id"],
                "tuning_slot_index": int(slot["slot_index"]),
                "tuning_candidate_id": slot["candidate_id"],
                "tuning_candidate_output_dir": slot["output_dir"],
                "tuning_candidate_result_path": slot["result_path"],
                "tuning_gpu_identity": namespace["gpu_identity"],
                "tuning_trusted_nvidia_smi": namespace[
                    "trusted_nvidia_smi"
                ],
            }
            for key, value in namespace_expected.items():
                if tuning_candidate_identity.get(key) != value:
                    raise PermissionError(
                        f"Tuning candidate namespace identity mismatch for {key}"
                    )
            if (
                registered_name != slot["registered_name"]
                or int(seed) != int(slot["seed"])
                or Path(output_dir).resolve() != Path(slot["output_dir"])
                or optimizer_override is None
                or float(optimizer_override.get("learning_rate", float("nan")))
                != float(slot["learning_rate"])
                or float(optimizer_override.get("weight_decay", float("nan")))
                != float(slot["weight_decay"])
            ):
                raise PermissionError(
                    "Tuning candidate call differs from its authorized slot"
                )
            token = validate_tuning_lease_authorization(
                config,
                namespace,
                slot,
                authorization_path=tuning_lease_validation_context[
                    "tuning_lease_authorization_path"
                ],
                expected_authorization_sha256=tuning_lease_validation_context[
                    "expected_tuning_lease_authorization_sha256"
                ],
                expected_authorization_file_sha256=tuning_lease_validation_context[
                    "expected_tuning_lease_authorization_file_sha256"
                ],
                custody_receipt_path=tuning_lease_validation_context[
                    "tuning_lease_custody_receipt_path"
                ],
                expected_custody_receipt_sha256=tuning_lease_validation_context[
                    "expected_tuning_lease_custody_receipt_sha256"
                ],
                expected_custody_receipt_file_sha256=tuning_lease_validation_context[
                    "expected_tuning_lease_custody_receipt_file_sha256"
                ],
                namespace_file_sha256=tuning_lease_validation_context[
                    "expected_tuning_namespace_file_sha256"
                ],
                require_active=True,
                expected_worker_pid=int(
                    tuning_lease_validation_context["expected_worker_pid"]
                ),
                expected_worker_capability_nonce_sha256=(
                    tuning_lease_validation_context[
                        "expected_worker_capability_nonce_sha256"
                    ]
                ),
            )
            proof_expected = {
                "tuning_lease_authorization_path": slot[
                    "authorization_token_path"
                ],
                "tuning_lease_authorization_sha256": token[
                    "tuning_lease_authorization_sha256"
                ],
                "tuning_lease_authorization_file_sha256": token[
                    "_validated_file_sha256"
                ],
                "tuning_lease_custody_receipt_sha256": token[
                    "tuning_lease_custody_receipt_sha256"
                ],
                "tuning_lease_custody_receipt_file_sha256": token[
                    "tuning_lease_custody_receipt_file_sha256"
                ],
                "tuning_lease_session_id": token["lease_session_id"],
                "tuning_lease_session_index": int(token["lease_session_index"]),
                "tuning_lease_session_acquisition_sha256": token[
                    "lease_session_acquisition_sha256"
                ],
                "tuning_lease_global_active_event_sha256": token[
                    "global_active_event_sha256"
                ],
                "tuning_idle_gate_evidence_sha256": token[
                    "idle_gate_evidence_sha256"
                ],
                "tuning_authorized_worker_pid": token[
                    "authorized_worker_pid"
                ],
                "tuning_worker_capability_nonce_sha256": token[
                    "worker_capability_nonce_sha256"
                ],
                "tuning_worker_ready_sha256": token["worker_ready_sha256"],
                "tuning_output_reservation_sha256": token[
                    "output_reservation_sha256"
                ],
                "tuning_output_reservation_file_sha256": token[
                    "output_reservation_file_sha256"
                ],
                "tuning_output_directory_device": token[
                    "output_directory_device"
                ],
                "tuning_output_directory_inode": token[
                    "output_directory_inode"
                ],
            }
            for key, value in proof_expected.items():
                if tuning_candidate_identity.get(key) != value:
                    raise PermissionError(
                        f"Tuning candidate lease identity mismatch for {key}"
                    )
            return token

        _assert_active_tuning_lease()
    else:
        _assert_active_tuning_lease = None

    tuning_artifact = validate_tuning_selection(
        config, registry, split_manifest
    ) if is_formal(config) and run_phase == "confirmatory" else None
    frozen_identities = formal_artifact_identities(
        config,
        registry=registry,
        trajectory_manifest=split_manifest,
        require_tuning=run_phase == "confirmatory",
    )
    pre_tuning_identities = {
        key: value
        for key, value in frozen_identities.items()
        if key != "tuning_selection_sha256"
    }
    if split_manifest.get("formal_artifact_identities") != pre_tuning_identities:
        raise ValueError(
            "Trajectory manifest formal provenance/controller identity mismatch"
        )
    if registry.get("formal_artifact_identities") != pre_tuning_identities:
        raise ValueError(
            "Parameter registry formal provenance/controller identity mismatch"
        )
    if is_formal(config) and run_phase == "confirmatory":
        pretraining_master, pretraining_audit = validate_pretraining_master_release(
            config,
            data_root=data_directory,
            expected_master_sha256=expected_pretraining_master_sha256,
            audit_receipt_path=(
                pretraining_audit_receipt_path
                or config.get("formal", {}).get("pretraining_audit_receipt")
            ),
            expected_audit_receipt_sha256=(
                expected_pretraining_audit_receipt_sha256
            ),
            expected_audit_receipt_file_sha256=(
                expected_pretraining_audit_receipt_file_sha256
            ),
            expected_identities={
                "run_registry_path": str(Path(run_registry_path).resolve())
                if run_registry_path
                else None,
                "environment_sha256": frozen_identities[
                    "environment_sha256"
                ],
                "controller_selection_sha256": frozen_identities[
                    "controller_selection_sha256"
                ],
                "trajectory_manifest_sha256": split_manifest[
                    "manifest_sha256"
                ],
                "parameter_registry_sha256": registry["registry_sha256"],
                "implementation_sha256": registry["implementation"][
                    "aggregate_sha256"
                ],
                "tuning_selection_sha256": frozen_identities[
                    "tuning_selection_sha256"
                ],
            },
        )
        frozen_identities = {
            **frozen_identities,
            "pretraining_master_sha256": pretraining_master[
                "pretraining_master_sha256"
            ],
            "pretraining_audit_receipt_sha256": pretraining_audit[
                "pretraining_audit_receipt_sha256"
            ],
            "pretraining_audit_receipt_file_sha256": pretraining_audit[
                "_validated_file_sha256"
            ],
        }
    destination = Path(output_dir).resolve()
    if is_formal(config) and run_phase == "validation_tuning":
        assert tuning_candidate_identity is not None
        if str(destination) != str(
            tuning_candidate_identity["tuning_candidate_output_dir"]
        ):
            raise ValueError("Tuning output directory differs from its namespace slot")
    reservation_identity: dict[str, Any] = {}
    if is_formal(config) and run_phase == "confirmatory":
        if not all(
            (
                run_registry_path,
                run_attempt_id,
                run_reservation_token_path,
                expected_run_reservation_token_sha256,
            )
        ):
            raise PermissionError(
                "Formal confirmatory training requires a registry-issued reservation token"
            )
        from .runner import validate_run_reservation

        reservation = validate_run_reservation(
            run_registry_path,
            attempt_id=str(run_attempt_id),
            token_path=run_reservation_token_path,
            expected_token_file_sha256=str(
                expected_run_reservation_token_sha256
            ),
            model=registered_name,
            seed=int(seed),
            output_dir=destination,
            allowed_statuses={"reserved", "running"},
        )
        token = reservation["token"]
        if (
            token.get("pretraining_master_sha256")
            != frozen_identities["pretraining_master_sha256"]
            or token.get("pretraining_audit_receipt_sha256")
            != frozen_identities["pretraining_audit_receipt_sha256"]
            or token.get("pretraining_audit_receipt_file_sha256")
            != frozen_identities["pretraining_audit_receipt_file_sha256"]
            or token.get("source_config_sha256") != current_config_sha256
            or token.get("run_registry_instance_id")
            != pretraining_master["run_registry_instance_id"]
            or token.get("initial_registry_sha256")
            != pretraining_master["initial_registry_sha256"]
            or token.get("run_registry_custody_sha256")
            != canonical_json_hash(pretraining_audit["run_registry_custody"])
        ):
            raise ValueError("Run reservation differs from the frozen pretraining chain")
        reservation_identity = {
            "formal_run_registry_path": str(Path(run_registry_path).resolve()),
            "formal_run_registry_namespace_sha256": token[
                "run_registry_namespace_sha256"
            ],
            "formal_run_registry_instance_id": token[
                "run_registry_instance_id"
            ],
            "formal_initial_run_registry_sha256": token[
                "initial_registry_sha256"
            ],
            "formal_run_registry_custody_sha256": token[
                "run_registry_custody_sha256"
            ],
            "formal_run_attempt_id": str(run_attempt_id),
            "formal_run_reservation_token_path": str(
                Path(run_reservation_token_path).resolve()
            ),
            "formal_run_reservation_token_sha256": token[
                "reservation_token_sha256"
            ],
            "formal_run_reservation_token_file_sha256": str(
                expected_run_reservation_token_sha256
            ),
            "formal_run_reservation_nonce": token["reservation_nonce"],
        }
    latest_path = destination / "latest.pt"
    best_path = destination / "best.pt"
    epoch_checkpoint_directory = destination / "epoch_checkpoints"
    checkpoint_ledger_path = destination / "checkpoint_ledger.json"
    reserved_tuning_directory = (
        is_formal(config)
        and run_phase == "validation_tuning"
        and isinstance(tuning_candidate_identity, dict)
        and destination.is_dir()
        and int(tuning_candidate_identity.get("tuning_output_directory_device", -1))
        == int(destination.stat().st_dev)
        and int(tuning_candidate_identity.get("tuning_output_directory_inode", -1))
        == int(destination.stat().st_ino)
        and not any(destination.iterdir())
    )
    if destination.exists() and not resume and not reserved_tuning_directory:
        raise FileExistsError(f"Refusing to overwrite training run: {destination}")
    if resume and (destination / "run_state.json").is_file():
        pre_resume_state = json.loads(
            (destination / "run_state.json").read_text(encoding="utf-8")
        )
        if pre_resume_state.get("status") == "failed":
            raise ValueError(
                "A failed training attempt cannot be resumed in place; start a "
                "new output directory with explicit authorized rerun linkage"
            )
    if (
        resume
        and not latest_path.is_file()
        and not _epoch_checkpoint_paths(epoch_checkpoint_directory)
    ):
        raise FileNotFoundError(
            "Resume requested but neither rolling nor immutable epoch checkpoint exists"
        )
    destination.mkdir(parents=True, exist_ok=True)
    session_started = time.perf_counter()

    train_arrays, train_manifest = load_closed_loop_dataset(data_directory / "train")
    val_arrays, val_manifest = load_closed_loop_dataset(data_directory / "val")
    normalizer = fit_normalizer(train_arrays)
    training = dict(config.get("training", {}))
    window_steps = int(training.get("window_steps", 64))
    burn_in_steps = int(training.get("burn_in_steps", 0))
    batch_size = int(training.get("batch_size", 32))
    tuning_config = dict(config.get("tuning", {}))
    epochs = int(
        tuning_config.get("epochs", 20)
        if run_phase == "validation_tuning"
        else training.get("epochs", 100)
    )
    patience = int(training.get("patience", 12))
    early_stopping_enabled = (
        False
        if run_phase == "validation_tuning"
        else bool(training.get("early_stopping_enabled", False))
    )
    if run_phase == "validation_tuning":
        if int(seed) != int(tuning_config.get("seed", 17)):
            raise ValueError("Validation-tuning run must use the frozen tuning seed")
        if optimizer_override is None:
            raise ValueError("Validation-tuning run requires an optimizer candidate")
        learning_rate = float(optimizer_override["learning_rate"])
        weight_decay = float(optimizer_override["weight_decay"])
        registered_pairs = {
            (float(lr), float(wd))
            for lr in tuning_config.get("learning_rates", [3e-4, 1e-3])
            for wd in tuning_config.get("weight_decays", [0.0, 1e-4])
        }
        if (learning_rate, weight_decay) not in registered_pairs:
            raise ValueError("Validation-tuning optimizer is outside the frozen grid")
        optimizer_selection = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        optimizer_selection_source = "registered_validation_tuning_candidate"
    elif tuning_artifact is not None:
        optimizer_selection = tuning_artifact["selected"][registered_name]
        learning_rate = float(optimizer_selection["learning_rate"])
        weight_decay = float(optimizer_selection["weight_decay"])
        optimizer_selection_source = "frozen_seed17_tuning_selection"
    else:
        learning_rate = float(training.get("learning_rate", 1e-3))
        weight_decay = float(training.get("weight_decay", 0.0))
        optimizer_selection = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        optimizer_selection_source = "development_config_bypass"
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    if min(window_steps, batch_size, epochs) <= 0 or learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    if early_stopping_enabled and patience <= 0:
        raise ValueError("patience must be positive when early stopping is enabled")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    deterministic = bool(config.get("experiment", {}).get("deterministic_torch", True))
    _seed_everything(int(seed), deterministic)
    apply_and_validate_precision_policy(config)
    runtime_device = current_run_accelerator_identity(
        torch_device,
        require_physical_mapping=is_formal(config),
        config=config,
    )
    if is_formal(config) and run_phase == "validation_tuning":
        assert tuning_candidate_identity is not None
        expected_gpu = tuning_candidate_identity["tuning_gpu_identity"]
        validate_tuning_runtime_identity(runtime_device, expected_gpu)
        if runtime_device.get("trusted_nvidia_smi") != tuning_candidate_identity[
            "tuning_trusted_nvidia_smi"
        ]:
            raise ValueError(
                "Validation-tuning runtime trusted nvidia-smi identity mismatch"
            )
    if is_formal(config) and run_phase == "confirmatory":
        environment = validate_environment_freeze(config)
        physical = runtime_device.get("selected_physical_gpu")
        inventory = environment.get("physical_gpu_inventory", [])
        known_physical = any(
            row.get("uuid") == (physical or {}).get("uuid")
            and row.get("pci_bus_id") == (physical or {}).get("pci_bus_id")
            for row in inventory
        )
        if (
            not known_physical
            or runtime_device.get("precision_policy")
            != pretraining_master.get("precision_policy")
        ):
            raise ValueError(
                "Confirmatory runtime GPU is outside the frozen physical inventory "
                "or its precision policy differs from preflight"
            )
    model = build_registered_model(registered_name, config, registry).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_windows = SequenceWindows(
        train_arrays, normalizer, window_steps, burn_in_steps
    )
    val_windows = SequenceWindows(val_arrays, normalizer, window_steps, burn_in_steps)
    train_compute_per_epoch = _sequence_compute_accounting(
        train_windows, registered_name=registered_name, config=config
    )
    val_compute_per_epoch = _sequence_compute_accounting(
        val_windows, registered_name=registered_name, config=config
    )
    compute_accounting_per_epoch = {
        "model_name": registered_name,
        "train": train_compute_per_epoch,
        "validation": val_compute_per_epoch,
        "scope": (
            "forward solver/update work; valid sequence steps and actually "
            "executed dense padding are reported separately"
        ),
    }
    steps_per_epoch = math.ceil(len(train_windows) / batch_size)
    planned_optimizer_steps = int(epochs * steps_per_epoch)
    declared_steps_per_epoch = (
        tuning_config.get("optimizer_steps_per_epoch")
        if run_phase == "validation_tuning"
        else training.get("optimizer_steps_per_epoch")
    )
    if (
        declared_steps_per_epoch is not None
        and steps_per_epoch != int(declared_steps_per_epoch)
    ):
        raise RuntimeError(
            "Frozen optimizer-budget gate failed before training: actual "
            f"steps_per_epoch {steps_per_epoch} != declared "
            f"{int(declared_steps_per_epoch)}"
        )
    declared_planned_steps = (
        tuning_config.get("planned_optimizer_steps")
        if run_phase == "validation_tuning"
        else training.get("planned_optimizer_steps")
    )
    if (
        declared_planned_steps is not None
        and planned_optimizer_steps != int(declared_planned_steps)
    ):
        raise RuntimeError(
            "Frozen optimizer-budget gate failed before training: actual planned "
            f"steps {planned_optimizer_steps} != declared "
            f"{int(declared_planned_steps)}"
        )

    resume_expected = {
        "run_phase": run_phase,
        "registered_name": registered_name,
        "seed": int(seed),
        "trajectory_manifest_sha256": split_manifest["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "config_sha256": current_config_sha256,
        **frozen_identities,
        **reservation_identity,
    }
    if run_phase == "validation_tuning":
        resume_expected["tuning_design_sha256"] = tuning_design_identity(
            config, registry, split_manifest
        )["tuning_design_sha256"]
        if tuning_candidate_identity:
            resume_expected.update(tuning_candidate_identity)

    # A completed run is write-once.  Resume verifies the full release identity,
    # optimizer budget, precision policy, and checkpoint hashes, then returns the
    # existing state without touching any file or timestamp.
    run_state_path = destination / "run_state.json"
    if resume and run_state_path.is_file():
        completed_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        if completed_state.get("status") == "failed":
            raise ValueError(
                "A failed training attempt cannot be resumed in place; start a "
                "new output directory and retain the failed-attempt linkage"
            )
        if completed_state.get("status") == "complete":
            for key, value in resume_expected.items():
                if completed_state.get(key) != value:
                    raise ValueError(f"Completed-resume identity mismatch for {key}")
            if (
                int(completed_state.get("optimizer_steps_completed", -1))
                != planned_optimizer_steps
                or int(completed_state.get("optimizer_steps_planned", -2))
                != planned_optimizer_steps
                or completed_state.get("fixed_budget_gate_passed") is not True
            ):
                raise ValueError("Completed-resume fixed optimizer budget mismatch")
            if not best_path.is_file() or not latest_path.is_file():
                raise FileNotFoundError("Completed run is missing a checkpoint")
            if sha256_file(best_path) != completed_state.get("best_checkpoint_sha256"):
                raise ValueError("Completed-resume best checkpoint hash mismatch")
            if sha256_file(latest_path) != completed_state.get("latest_checkpoint_sha256"):
                raise ValueError("Completed-resume latest checkpoint hash mismatch")
            ledger = _epoch_checkpoint_ledger(
                epoch_checkpoint_directory, expected_identity=resume_expected
            )
            stored_ledger = json.loads(
                checkpoint_ledger_path.read_text(encoding="utf-8")
            ) if checkpoint_ledger_path.is_file() else None
            if (
                len(ledger["checkpoints"])
                != int(completed_state.get("epochs_completed", -1))
                or not checkpoint_ledger_path.is_file()
                or stored_ledger != ledger
                or sha256_file(checkpoint_ledger_path)
                != completed_state.get("checkpoint_ledger_file_sha256")
                or ledger.get("checkpoint_ledger_sha256")
                != completed_state.get("checkpoint_ledger_sha256")
            ):
                raise ValueError("Completed-resume immutable epoch ledger mismatch")
            checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            for key, value in resume_expected.items():
                if checkpoint.get(key) != value:
                    raise ValueError(
                        f"Completed-resume checkpoint identity mismatch for {key}"
                    )
            if (
                int(checkpoint.get("training_run_optimizer_steps_completed", -1))
                != planned_optimizer_steps
                or int(checkpoint.get("training_run_optimizer_steps_planned", -2))
                != planned_optimizer_steps
                or not checkpoint.get("selection_finalized_after_training_run")
            ):
                raise ValueError("Completed-resume checkpoint budget/finalization mismatch")
            if (
                int(checkpoint.get("best_epoch", -1))
                != int(completed_state.get("best_epoch", -2))
                or not math.isclose(
                    float(checkpoint.get("best_val_loss", math.nan)),
                    float(completed_state.get("best_val_loss", math.nan)),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("Completed-resume selected epoch/loss mismatch")
            selected_epoch_path = epoch_checkpoint_directory / (
                f"epoch_{int(completed_state['best_epoch']):04d}.pt"
            )
            if (
                not selected_epoch_path.is_file()
                or checkpoint.get("selected_epoch_checkpoint_path")
                != str(selected_epoch_path.resolve())
                or checkpoint.get("selected_epoch_checkpoint_sha256")
                != sha256_file(selected_epoch_path)
                or completed_state.get("selected_epoch_checkpoint_sha256")
                != sha256_file(selected_epoch_path)
            ):
                raise ValueError("Completed-resume selected epoch binding mismatch")
            prior_runtime = completed_state.get("runtime_device", {})
            if canonical_json_hash(prior_runtime.get("precision_policy")) != canonical_json_hash(
                runtime_device.get("precision_policy")
            ):
                raise ValueError("Completed-resume precision policy mismatch")
            prior_gpu = prior_runtime.get("selected_physical_gpu") or {}
            current_gpu = runtime_device.get("selected_physical_gpu") or {}
            if (prior_gpu.get("uuid"), prior_gpu.get("pci_bus_id")) != (
                current_gpu.get("uuid"),
                current_gpu.get("pci_bus_id"),
            ):
                raise ValueError("Completed-resume physical GPU identity mismatch")
            return completed_state

    if resume:
        _repair_incomplete_epoch_transaction(
            checkpoint_directory=epoch_checkpoint_directory,
            ledger_path=checkpoint_ledger_path,
            latest_path=latest_path,
            best_path=best_path,
            expected_identity=resume_expected,
        )
        if not latest_path.is_file():
            raise FileNotFoundError(
                "Resume repair did not produce a rolling latest checkpoint"
            )

    start_epoch = 0
    best_val = math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    optimizer_steps_completed = 0
    elapsed_seconds_before_resume = 0.0
    peak_allocated_bytes_before_resume = 0
    if resume:
        checkpoint = torch.load(latest_path, map_location=torch_device, weights_only=False)
        for key, value in resume_expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Resume identity mismatch for {key}")
        previous_gpu = checkpoint.get("runtime_device", {}).get(
            "selected_physical_gpu"
        )
        current_gpu = runtime_device.get("selected_physical_gpu")
        if (previous_gpu or {}).get("uuid") != (current_gpu or {}).get("uuid"):
            raise ValueError("Resume selected-physical-GPU identity mismatch")
        if canonical_json_hash(
            checkpoint.get("runtime_device", {}).get("precision_policy")
        ) != canonical_json_hash(runtime_device.get("precision_policy")):
            raise ValueError("Resume precision policy mismatch")
        load_graph_safe_state_dict(
            model, checkpoint["model_state_dict"], strict=True
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _restore_rng(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        optimizer_steps_completed = int(checkpoint["optimizer_steps_completed"])
        elapsed_seconds_before_resume = float(
            checkpoint.get("elapsed_training_seconds", 0.0)
        )
        peak_allocated_bytes_before_resume = int(
            checkpoint.get("peak_allocated_bytes", 0)
        )

    identity = {
        "run_phase": run_phase,
        "registered_name": registered_name,
        "seed": int(seed),
        "trajectory_manifest_sha256": split_manifest["manifest_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "config_sha256": current_config_sha256,
        "parameter_registry_source_config_sha256": registry[
            "source_config_sha256"
        ],
        "parameter_registry_implementation_sha256": registry["implementation"][
            "aggregate_sha256"
        ],
        **frozen_identities,
        **reservation_identity,
    }
    if run_phase == "validation_tuning":
        design = tuning_design_identity(config, registry, split_manifest)
        identity["tuning_design_sha256"] = design["tuning_design_sha256"]
        if tuning_candidate_identity:
            identity.update(tuning_candidate_identity)
    stopped_early = False
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        train_loss, optimizer_steps = _run_epoch(
            model,
            train_windows,
            batch_size=batch_size,
            device=torch_device,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
            order_seed=int(seed) * 1_000_003 + epoch,
        )
        val_loss, _ = _run_epoch(
            model,
            val_windows,
            batch_size=batch_size,
            device=torch_device,
            optimizer=None,
            gradient_clip=gradient_clip,
            order_seed=0,
        )
        optimizer_steps_completed += optimizer_steps
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "optimizer_steps_completed": optimizer_steps_completed,
                "epoch_wall_seconds": time.perf_counter() - epoch_started,
                "compute_accounting": compute_accounting_per_epoch,
            }
        )
        # Fixed-budget formal/tuning selection is the exact finite validation
        # argmin (ties remain at the earlier epoch).  ``minimum_delta`` is only
        # an early-stopping patience threshold on development runs.
        prior_best_val = best_val
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
        patience_improved = val_loss < prior_best_val - float(
            training.get("minimum_delta", 1e-7)
        )
        if patience_improved:
            stale_epochs = 0
        else:
            stale_epochs += 1
        previous_epoch_sha256 = (
            None
            if epoch == 0
            else sha256_file(
                epoch_checkpoint_directory / f"epoch_{epoch - 1:04d}.pt"
            )
        )
        payload = {
            **identity,
            "schema_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": _rng_state(),
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "optimizer_steps_completed": optimizer_steps_completed,
            "optimizer_steps_planned": planned_optimizer_steps,
            "steps_per_epoch": steps_per_epoch,
            "declared_optimizer_steps_per_epoch": declared_steps_per_epoch,
            "declared_planned_optimizer_steps": declared_planned_steps,
            "fixed_optimizer_budget": not early_stopping_enabled,
            "history": history,
            "history_sha256": canonical_json_hash(history),
            "epoch_validation_loss": float(val_loss),
            "epoch_checkpoint_schema_version": 1,
            "previous_epoch_checkpoint_sha256": previous_epoch_sha256,
            "selection_rule": "minimum_finite_validation_loss_then_earliest_epoch",
            "normalizer": normalizer.to_dict(),
            "config": config,
            "parameter_entry": registry["entries"][registered_name],
            "parameter_registry_implementation": registry.get("implementation"),
            "topology_audit": getattr(model, "topology_audit", {}),
            "elapsed_time_adapter": getattr(model, "elapsed_time_adapter", None),
            "data_sha256": {
                "train": train_manifest["dataset_sha256"],
                "val": val_manifest["dataset_sha256"],
            },
            "validation_split": "val",
            "checkpoint_selection": "minimum fixed-validation normalized state MSE",
            "runtime_device": runtime_device,
            "optimizer_hyperparameters": optimizer_selection,
            "optimizer_selection_source": optimizer_selection_source,
            "compute_accounting_per_epoch": compute_accounting_per_epoch,
            "elapsed_training_seconds": (
                elapsed_seconds_before_resume + time.perf_counter() - session_started
            ),
            "peak_allocated_bytes": max(
                peak_allocated_bytes_before_resume,
                int(torch.cuda.max_memory_allocated(torch_device))
                if torch_device.type == "cuda"
                else 0,
            ),
        }
        epoch_checkpoint_path = (
            epoch_checkpoint_directory / f"epoch_{epoch:04d}.pt"
        )
        if _assert_active_tuning_lease is not None:
            _assert_active_tuning_lease()
        _atomic_torch_save_new(epoch_checkpoint_path, payload)
        # The immutable epoch file is the commit point.  First extend the
        # validated ledger, then update rolling caches.  Resume repairs any
        # interruption after the commit without re-running this epoch.
        checkpoint_ledger = _write_epoch_checkpoint_ledger(
            checkpoint_ledger_path,
            epoch_checkpoint_directory,
            expected_identity=resume_expected,
        )
        if improved:
            _atomic_torch_save(best_path, payload)
        _atomic_torch_save(latest_path, payload)
        atomic_write_json(destination / "history.json", history)
        atomic_write_json(
            destination / "run_state.json",
            {
                **identity,
                "status": "running",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "stale_epochs": stale_epochs,
                "optimizer_steps_completed": optimizer_steps_completed,
                "optimizer_steps_planned": planned_optimizer_steps,
                "steps_per_epoch": steps_per_epoch,
                "declared_optimizer_steps_per_epoch": declared_steps_per_epoch,
                "declared_planned_optimizer_steps": declared_planned_steps,
                "fixed_optimizer_budget": not early_stopping_enabled,
                "latest_checkpoint_sha256": sha256_file(latest_path),
                "best_checkpoint_sha256": sha256_file(best_path),
                "checkpoint_ledger": str(checkpoint_ledger_path.resolve()),
                "checkpoint_ledger_file_sha256": sha256_file(
                    checkpoint_ledger_path
                ),
                "checkpoint_ledger_sha256": checkpoint_ledger[
                    "checkpoint_ledger_sha256"
                ],
                "runtime_device": runtime_device,
                "compute_accounting_per_epoch": compute_accounting_per_epoch,
                "elapsed_training_seconds": payload["elapsed_training_seconds"],
                "peak_allocated_bytes": payload["peak_allocated_bytes"],
            },
        )
        if early_stopping_enabled and stale_epochs >= patience:
            stopped_early = True
            break

    if not best_path.is_file():
        raise RuntimeError("Training ended without a selected checkpoint")
    checkpoint_ledger = _epoch_checkpoint_ledger(
        epoch_checkpoint_directory, expected_identity=resume_expected
    )
    if len(checkpoint_ledger["checkpoints"]) != len(history):
        raise RuntimeError("Immutable epoch checkpoint count differs from history")
    if _assert_active_tuning_lease is not None:
        _assert_active_tuning_lease()
    atomic_write_json(checkpoint_ledger_path, checkpoint_ledger)
    selected_epoch_checkpoint_path = (
        epoch_checkpoint_directory / f"epoch_{best_epoch:04d}.pt"
    )
    if (
        not selected_epoch_checkpoint_path.is_file()
        or checkpoint_ledger["checkpoints"][best_epoch]["checkpoint_sha256"]
        != sha256_file(selected_epoch_checkpoint_path)
    ):
        raise RuntimeError("Selected immutable epoch checkpoint binding failed")
    if not early_stopping_enabled and optimizer_steps_completed != planned_optimizer_steps:
        raise RuntimeError(
            "Fixed-budget gate failed: completed "
            f"{optimizer_steps_completed} != planned {planned_optimizer_steps}"
        )
    # Finalize selection metadata only after the full registered training budget
    # has completed.  The model weights remain those of the best validation
    # epoch; this makes it impossible to mistake an early best epoch for an
    # early-terminated confirmatory run.
    selected_checkpoint = torch.load(
        best_path, map_location="cpu", weights_only=False
    )
    selected_checkpoint.update(
        {
            "training_run_optimizer_steps_completed": optimizer_steps_completed,
            "training_run_optimizer_steps_planned": planned_optimizer_steps,
            "training_run_epochs_completed": len(history),
            "training_run_stopped_early": stopped_early,
            "training_run_history": history,
            "training_run_fixed_budget_gate_passed": (
                None
                if early_stopping_enabled
                else optimizer_steps_completed == planned_optimizer_steps
            ),
            "selection_finalized_after_training_run": True,
            "checkpoint_ledger_sha256": checkpoint_ledger[
                "checkpoint_ledger_sha256"
            ],
            "checkpoint_ledger_file_sha256": sha256_file(
                checkpoint_ledger_path
            ),
            "selected_epoch_checkpoint_path": str(
                selected_epoch_checkpoint_path.resolve()
            ),
            "selected_epoch_checkpoint_sha256": sha256_file(
                selected_epoch_checkpoint_path
            ),
            "compute_accounting_training_run": {
                split: {
                    key: int(value) * len(history)
                    for key, value in values.items()
                }
                for split, values in (
                    ("train", train_compute_per_epoch),
                    ("validation", val_compute_per_epoch),
                )
            },
        }
    )
    if _assert_active_tuning_lease is not None:
        _assert_active_tuning_lease()
    _atomic_torch_save(best_path, selected_checkpoint)
    final = {
        **identity,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "early_stopping_enabled": early_stopping_enabled,
        "optimizer_steps_completed": optimizer_steps_completed,
        "optimizer_steps_planned": planned_optimizer_steps,
        "steps_per_epoch": steps_per_epoch,
        "declared_optimizer_steps_per_epoch": declared_steps_per_epoch,
        "declared_planned_optimizer_steps": declared_planned_steps,
        "fixed_budget_gate_passed": (
            None
            if early_stopping_enabled
            else optimizer_steps_completed == planned_optimizer_steps
        ),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "checkpoint_ledger": str(checkpoint_ledger_path.resolve()),
        "checkpoint_ledger_file_sha256": sha256_file(checkpoint_ledger_path),
        "checkpoint_ledger_sha256": checkpoint_ledger[
            "checkpoint_ledger_sha256"
        ],
        "selected_epoch_checkpoint_path": str(
            selected_epoch_checkpoint_path.resolve()
        ),
        "selected_epoch_checkpoint_sha256": sha256_file(
            selected_epoch_checkpoint_path
        ),
        "runtime_device": runtime_device,
        "compute_accounting_per_epoch": compute_accounting_per_epoch,
        "compute_accounting_training_run": {
            split: {
                key: int(value) * len(history)
                for key, value in values.items()
            }
            for split, values in (
                ("train", train_compute_per_epoch),
                ("validation", val_compute_per_epoch),
            )
        },
        "elapsed_training_seconds": (
            elapsed_seconds_before_resume + time.perf_counter() - session_started
        ),
        "peak_allocated_bytes": max(
            peak_allocated_bytes_before_resume,
            int(torch.cuda.max_memory_allocated(torch_device))
            if torch_device.type == "cuda"
            else 0,
        ),
    }
    if _assert_active_tuning_lease is not None:
        _assert_active_tuning_lease()
    atomic_write_json(destination / "run_state.json", final)
    return final


def _record_failed_training_attempt(
    *,
    destination: Path,
    registered_name: str,
    seed: int,
    run_phase: str,
    resume: bool,
    error: Exception,
) -> None:
    """Append an immutable, self-hashed failure ledger and close running state."""
    failure_directory = destination / "failed_attempts"
    failure_directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(failure_directory.glob("attempt_*.json"))
    attempt_number = len(existing) + 1
    failure_path = failure_directory / f"attempt_{attempt_number:04d}.json"
    if failure_path.exists():
        raise FileExistsError(f"Failure ledger collision: {failure_path}")
    run_state_path = destination / "run_state.json"
    prior_state: dict[str, Any] = {}
    if run_state_path.is_file():
        prior_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    prior_attempt = existing[-1] if existing else None
    core = {
        "schema_version": 1,
        "producer": "liquid_revision.closed_loop.training",
        "attempt_number": attempt_number,
        "registered_name": str(registered_name),
        "seed": int(seed),
        "run_phase": str(run_phase),
        "resume_requested": bool(resume),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "optimizer_steps_completed": int(
            prior_state.get("optimizer_steps_completed", 0)
        ),
        "optimizer_steps_planned": prior_state.get("optimizer_steps_planned"),
        "prior_run_status": prior_state.get("status"),
        "prior_run_state_sha256": (
            sha256_file(run_state_path) if run_state_path.is_file() else None
        ),
        "prior_failed_attempt_sha256": (
            sha256_file(prior_attempt) if prior_attempt is not None else None
        ),
        "rerun_linkage": (
            str(prior_attempt.resolve()) if prior_attempt is not None else None
        ),
    }
    artifact = {**core, "failed_attempt_sha256": canonical_json_hash(core)}
    atomic_write_json(failure_path, artifact)
    # Never mutate an already-complete write-once run.  All other attempts must
    # leave an explicit failed state instead of a stale `running` marker.
    if prior_state.get("status") != "complete":
        failed_state = {
            **prior_state,
            "status": "failed",
            "registered_name": str(registered_name),
            "seed": int(seed),
            "run_phase": str(run_phase),
            "failed_attempt": str(failure_path.resolve()),
            "failed_attempt_sha256": artifact["failed_attempt_sha256"],
            "failure_error_type": type(error).__name__,
            "failure_error_message": str(error),
        }
        atomic_write_json(run_state_path, failed_state)


def train_registered_model(
    *,
    registered_name: str,
    seed: int,
    config: dict[str, Any],
    data_root: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    resume: bool = False,
    run_phase: str = "confirmatory",
    optimizer_override: dict[str, float] | None = None,
    expected_pretraining_master_sha256: str | None = None,
    pretraining_audit_receipt_path: str | Path | None = None,
    expected_pretraining_audit_receipt_sha256: str | None = None,
    expected_pretraining_audit_receipt_file_sha256: str | None = None,
    run_registry_path: str | Path | None = None,
    run_attempt_id: str | None = None,
    run_reservation_token_path: str | Path | None = None,
    expected_run_reservation_token_sha256: str | None = None,
    tuning_candidate_identity: dict[str, Any] | None = None,
    tuning_lease_validation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with an immutable failure-attempt ledger around the core runner."""
    destination = Path(output_dir).resolve()
    existed_before = destination.exists()
    try:
        return _train_registered_model_impl(
            registered_name=registered_name,
            seed=seed,
            config=config,
            data_root=data_root,
            registry_path=registry_path,
            output_dir=destination,
            device=device,
            resume=resume,
            run_phase=run_phase,
            optimizer_override=optimizer_override,
            expected_pretraining_master_sha256=expected_pretraining_master_sha256,
            pretraining_audit_receipt_path=pretraining_audit_receipt_path,
            expected_pretraining_audit_receipt_sha256=(
                expected_pretraining_audit_receipt_sha256
            ),
            expected_pretraining_audit_receipt_file_sha256=(
                expected_pretraining_audit_receipt_file_sha256
            ),
            run_registry_path=run_registry_path,
            run_attempt_id=run_attempt_id,
            run_reservation_token_path=run_reservation_token_path,
            expected_run_reservation_token_sha256=(
                expected_run_reservation_token_sha256
            ),
            tuning_candidate_identity=tuning_candidate_identity,
            tuning_lease_validation_context=tuning_lease_validation_context,
        )
    except Exception as error:
        # A non-resume request against an existing directory was never admitted
        # as a new attempt and must not alter user-owned prior results.
        prior_failed = False
        state_path = destination / "run_state.json"
        if state_path.is_file():
            try:
                prior_failed = (
                    json.loads(state_path.read_text(encoding="utf-8")).get("status")
                    == "failed"
                )
            except (OSError, ValueError):
                prior_failed = False
        if not (existed_before and not resume) and not (resume and prior_failed):
            _record_failed_training_attempt(
                destination=destination,
                registered_name=registered_name,
                seed=seed,
                run_phase=run_phase,
                resume=resume,
                error=error,
            )
        raise


def train_tuning_candidate(
    *,
    registered_name: str,
    config: dict[str, Any],
    data_root: str | Path,
    registry_path: str | Path,
    output_dir: str | Path,
    learning_rate: float,
    weight_decay: float,
    device: str = "cpu",
    resume: bool = False,
    tuning_namespace_path: str | Path | None = None,
    expected_tuning_namespace_sha256: str | None = None,
    expected_tuning_namespace_file_sha256: str | None = None,
    tuning_attempt_id: str | None = None,
    tuning_lease_authorization_path: str | Path | None = None,
    expected_tuning_lease_authorization_sha256: str | None = None,
    expected_tuning_lease_authorization_file_sha256: str | None = None,
    tuning_lease_custody_receipt_path: str | Path | None = None,
    expected_tuning_lease_custody_receipt_sha256: str | None = None,
    expected_tuning_lease_custody_receipt_file_sha256: str | None = None,
    tuning_authorized_worker_pid: int | None = None,
    tuning_worker_capability_nonce: str | None = None,
    tuning_output_reservation_path: str | Path | None = None,
    expected_tuning_output_reservation_sha256: str | None = None,
    expected_tuning_output_reservation_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one immutable validation-only seed-17 optimizer candidate."""
    tuning_seed = int(config.get("tuning", {}).get("seed", 17))
    destination = Path(output_dir).resolve()
    artifact_path = destination / "tuning_candidate_result.json"
    tuning_candidate_identity: dict[str, Any] | None = None
    tuning_lease_validation_context: dict[str, Any] | None = None
    from .formal import (
        is_formal,
        validate_tuning_lease_authorization,
        validate_tuning_namespace,
    )

    if is_formal(config):
        if resume:
            raise PermissionError(
                "Formal validation-tuning candidates cannot resume an authorized slot"
            )
        if not all(
            (
                tuning_namespace_path,
                expected_tuning_namespace_sha256,
                expected_tuning_namespace_file_sha256,
                tuning_attempt_id,
                tuning_lease_authorization_path,
                expected_tuning_lease_authorization_sha256,
                expected_tuning_lease_authorization_file_sha256,
                tuning_lease_custody_receipt_path,
                expected_tuning_lease_custody_receipt_sha256,
                expected_tuning_lease_custody_receipt_file_sha256,
                tuning_authorized_worker_pid,
                tuning_worker_capability_nonce,
                tuning_output_reservation_path,
                expected_tuning_output_reservation_sha256,
                expected_tuning_output_reservation_file_sha256,
            )
        ):
            raise PermissionError(
                "Formal tuning candidate requires namespace path, external anchors, "
                "registered attempt ID, custody, and active lease authorization"
            )
        registry = load_parameter_registry(registry_path, expected_config=config)
        trajectory = load_trajectory_manifest(
            Path(data_root).resolve() / "trajectory_manifest.json",
            expected_config=config,
        )
        namespace = validate_tuning_namespace(
            config,
            registry,
            trajectory,
            namespace_path=str(tuning_namespace_path),
            expected_namespace_sha256=str(expected_tuning_namespace_sha256),
            expected_namespace_file_sha256=str(
                expected_tuning_namespace_file_sha256
            ),
        )
        matching_slots = [
            slot
            for slot in namespace["candidate_slots"]
            if slot["registered_name"] == registered_name
            and float(slot["learning_rate"]) == float(learning_rate)
            and float(slot["weight_decay"]) == float(weight_decay)
            and slot["attempt_id"] == str(tuning_attempt_id)
        ]
        if len(matching_slots) != 1:
            raise ValueError("Candidate optimizer/attempt is not a unique namespace slot")
        slot = matching_slots[0]
        if (
            str(destination) != slot["output_dir"]
            or str(artifact_path.resolve()) != slot["result_path"]
        ):
            raise ValueError("Candidate output/result path differs from namespace slot")
        worker_nonce_sha256 = canonical_json_hash(
            str(tuning_worker_capability_nonce)
        )
        if int(tuning_authorized_worker_pid) != os.getpid():
            raise PermissionError(
                "Tuning authorization is bound to a different worker PID"
            )
        reservation_path = Path(str(tuning_output_reservation_path)).resolve()
        if reservation_path != Path(slot["output_reservation_path"]).resolve():
            raise PermissionError("Tuning output reservation path differs from slot")
        if (
            not reservation_path.is_file()
            or sha256_file(reservation_path)
            != str(expected_tuning_output_reservation_file_sha256)
        ):
            raise PermissionError("Tuning output reservation file anchor mismatch")
        token = validate_tuning_lease_authorization(
            config,
            namespace,
            slot,
            authorization_path=str(tuning_lease_authorization_path),
            expected_authorization_sha256=str(
                expected_tuning_lease_authorization_sha256
            ),
            expected_authorization_file_sha256=str(
                expected_tuning_lease_authorization_file_sha256
            ),
            custody_receipt_path=str(tuning_lease_custody_receipt_path),
            expected_custody_receipt_sha256=str(
                expected_tuning_lease_custody_receipt_sha256
            ),
            expected_custody_receipt_file_sha256=str(
                expected_tuning_lease_custody_receipt_file_sha256
            ),
            namespace_file_sha256=str(expected_tuning_namespace_file_sha256),
            require_active=True,
            expected_worker_pid=os.getpid(),
            expected_worker_capability_nonce_sha256=worker_nonce_sha256,
        )
        if (
            token.get("output_reservation_sha256")
            != str(expected_tuning_output_reservation_sha256)
            or token.get("output_reservation_file_sha256")
            != str(expected_tuning_output_reservation_file_sha256)
        ):
            raise PermissionError("Tuning output reservation anchors are stale")
        if not destination.is_dir():
            raise PermissionError("Reserved tuning output directory is absent")
        destination_stat = destination.stat()
        if (
            int(token.get("output_directory_device", -1))
            != int(destination_stat.st_dev)
            or int(token.get("output_directory_inode", -1))
            != int(destination_stat.st_ino)
            or any(destination.iterdir())
        ):
            raise PermissionError(
                "Reserved tuning output directory was replaced or pre-populated"
            )
        tuning_candidate_identity = {
            "tuning_namespace_sha256": namespace["tuning_namespace_sha256"],
            "tuning_namespace_file_sha256": str(
                expected_tuning_namespace_file_sha256
            ),
            "tuning_namespace_instance_id": namespace["namespace_instance_id"],
            "tuning_attempt_id": slot["attempt_id"],
            "tuning_slot_index": int(slot["slot_index"]),
            "tuning_candidate_id": slot["candidate_id"],
            "tuning_candidate_output_dir": slot["output_dir"],
            "tuning_candidate_result_path": slot["result_path"],
            "tuning_gpu_identity": namespace["gpu_identity"],
            "tuning_trusted_nvidia_smi": namespace[
                "trusted_nvidia_smi"
            ],
            "tuning_lease_authorization_path": slot[
                "authorization_token_path"
            ],
            "tuning_lease_authorization_sha256": token[
                "tuning_lease_authorization_sha256"
            ],
            "tuning_lease_authorization_file_sha256": token[
                "_validated_file_sha256"
            ],
            "tuning_lease_custody_receipt_sha256": token[
                "tuning_lease_custody_receipt_sha256"
            ],
            "tuning_lease_custody_receipt_file_sha256": token[
                "tuning_lease_custody_receipt_file_sha256"
            ],
            "tuning_lease_session_id": token["lease_session_id"],
            "tuning_lease_session_index": int(token["lease_session_index"]),
            "tuning_lease_session_acquisition_sha256": token[
                "lease_session_acquisition_sha256"
            ],
            "tuning_lease_global_active_event_sha256": token[
                "global_active_event_sha256"
            ],
            "tuning_idle_gate_evidence_sha256": token[
                "idle_gate_evidence_sha256"
            ],
            "tuning_authorized_worker_pid": token["authorized_worker_pid"],
            "tuning_worker_capability_nonce_sha256": token[
                "worker_capability_nonce_sha256"
            ],
            "tuning_worker_ready_sha256": token["worker_ready_sha256"],
            "tuning_output_reservation_sha256": token[
                "output_reservation_sha256"
            ],
            "tuning_output_reservation_file_sha256": token[
                "output_reservation_file_sha256"
            ],
            "tuning_output_directory_device": token[
                "output_directory_device"
            ],
            "tuning_output_directory_inode": token[
                "output_directory_inode"
            ],
        }
        tuning_lease_validation_context = {
            "tuning_namespace_path": str(tuning_namespace_path),
            "expected_tuning_namespace_sha256": str(
                expected_tuning_namespace_sha256
            ),
            "expected_tuning_namespace_file_sha256": str(
                expected_tuning_namespace_file_sha256
            ),
            "tuning_lease_authorization_path": str(
                tuning_lease_authorization_path
            ),
            "expected_tuning_lease_authorization_sha256": str(
                expected_tuning_lease_authorization_sha256
            ),
            "expected_tuning_lease_authorization_file_sha256": str(
                expected_tuning_lease_authorization_file_sha256
            ),
            "tuning_lease_custody_receipt_path": str(
                tuning_lease_custody_receipt_path
            ),
            "expected_tuning_lease_custody_receipt_sha256": str(
                expected_tuning_lease_custody_receipt_sha256
            ),
            "expected_tuning_lease_custody_receipt_file_sha256": str(
                expected_tuning_lease_custody_receipt_file_sha256
            ),
            "tuning_attempt_id": str(tuning_attempt_id),
            "expected_worker_pid": os.getpid(),
            "expected_worker_capability_nonce_sha256": worker_nonce_sha256,
            "tuning_output_reservation_path": str(reservation_path),
            "expected_tuning_output_reservation_sha256": str(
                expected_tuning_output_reservation_sha256
            ),
            "expected_tuning_output_reservation_file_sha256": str(
                expected_tuning_output_reservation_file_sha256
            ),
        }
    result = train_registered_model(
        registered_name=registered_name,
        seed=tuning_seed,
        config=config,
        data_root=data_root,
        registry_path=registry_path,
        output_dir=output_dir,
        device=device,
        resume=resume,
        run_phase="validation_tuning",
        optimizer_override={
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
        },
        tuning_candidate_identity=tuning_candidate_identity,
        tuning_lease_validation_context=tuning_lease_validation_context,
    )
    if resume and artifact_path.is_file():
        existing = json.loads(artifact_path.read_text(encoding="utf-8"))
        existing_core = {
            key: value
            for key, value in existing.items()
            if key != "tuning_candidate_sha256"
        }
        if canonical_json_hash(existing_core) != existing.get(
            "tuning_candidate_sha256"
        ):
            raise ValueError("Completed tuning-candidate artifact hash mismatch")
        expected_existing = {
            "registered_name": registered_name,
            "seed": tuning_seed,
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "checkpoint_sha256": result["best_checkpoint_sha256"],
            "optimizer_steps_completed": result["optimizer_steps_completed"],
            "optimizer_steps_planned": result["optimizer_steps_planned"],
            "validation_only": True,
            "test_episodes_used": 0,
        }
        if tuning_candidate_identity:
            expected_existing.update(tuning_candidate_identity)
        for key, value in expected_existing.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"Completed tuning-candidate identity mismatch for {key}"
                )
        return {**result, "tuning_candidate_result": str(artifact_path)}
    if tuning_lease_validation_context is not None:
        validate_tuning_lease_authorization(
            config,
            namespace,
            slot,
            authorization_path=tuning_lease_validation_context[
                "tuning_lease_authorization_path"
            ],
            expected_authorization_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_authorization_sha256"
            ],
            expected_authorization_file_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_authorization_file_sha256"
            ],
            custody_receipt_path=tuning_lease_validation_context[
                "tuning_lease_custody_receipt_path"
            ],
            expected_custody_receipt_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_custody_receipt_sha256"
            ],
            expected_custody_receipt_file_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_custody_receipt_file_sha256"
            ],
            namespace_file_sha256=tuning_lease_validation_context[
                "expected_tuning_namespace_file_sha256"
            ],
            require_active=True,
            expected_worker_pid=int(
                tuning_lease_validation_context["expected_worker_pid"]
            ),
            expected_worker_capability_nonce_sha256=(
                tuning_lease_validation_context[
                    "expected_worker_capability_nonce_sha256"
                ]
            ),
        )
    checkpoint = torch.load(
        result["best_checkpoint"], map_location="cpu", weights_only=False
    )
    core = {
        "schema_version": 1,
        "producer": "liquid_revision.closed_loop.training.train_tuning_candidate",
        "registered_name": registered_name,
        "seed": tuning_seed,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "validation_loss": float(result["best_val_loss"]),
        "checkpoint_path": str(Path(result["best_checkpoint"]).resolve()),
        "checkpoint_sha256": result["best_checkpoint_sha256"],
        "config_sha256": checkpoint["config_sha256"],
        "tuning_design_sha256": checkpoint["tuning_design_sha256"],
        "trajectory_manifest_sha256": checkpoint["trajectory_manifest_sha256"],
        "parameter_registry_sha256": checkpoint["parameter_registry_sha256"],
        "implementation_sha256": checkpoint[
            "parameter_registry_implementation_sha256"
        ],
        "environment_sha256": checkpoint.get("environment_sha256"),
        "controller_selection_sha256": checkpoint.get(
            "controller_selection_sha256"
        ),
        **(tuning_candidate_identity or {}),
        "optimizer_steps_completed": result["optimizer_steps_completed"],
        "optimizer_steps_planned": result["optimizer_steps_planned"],
        "validation_only": True,
        "test_episodes_used": 0,
    }
    artifact = {
        **core,
        "tuning_candidate_sha256": canonical_json_hash(core),
    }
    if tuning_lease_validation_context is not None:
        validate_tuning_lease_authorization(
            config,
            namespace,
            slot,
            authorization_path=tuning_lease_validation_context[
                "tuning_lease_authorization_path"
            ],
            expected_authorization_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_authorization_sha256"
            ],
            expected_authorization_file_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_authorization_file_sha256"
            ],
            custody_receipt_path=tuning_lease_validation_context[
                "tuning_lease_custody_receipt_path"
            ],
            expected_custody_receipt_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_custody_receipt_sha256"
            ],
            expected_custody_receipt_file_sha256=tuning_lease_validation_context[
                "expected_tuning_lease_custody_receipt_file_sha256"
            ],
            namespace_file_sha256=tuning_lease_validation_context[
                "expected_tuning_namespace_file_sha256"
            ],
            require_active=True,
            expected_worker_pid=int(
                tuning_lease_validation_context["expected_worker_pid"]
            ),
            expected_worker_capability_nonce_sha256=(
                tuning_lease_validation_context[
                    "expected_worker_capability_nonce_sha256"
                ]
            ),
        )
    atomic_write_json(artifact_path, artifact)
    return {**result, "tuning_candidate_result": str(artifact_path)}


class NormalizedObserver(nn.Module):
    """Apply train-only normalization around a frozen registered observer."""

    def __init__(self, model: nn.Module, normalizer: Normalizer) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", torch.as_tensor(normalizer.input_mean))
        self.register_buffer("input_std", torch.as_tensor(normalizer.input_std))
        self.register_buffer("target_mean", torch.as_tensor(normalizer.target_mean))
        self.register_buffer("target_std", torch.as_tensor(normalizer.target_std))

    def forward_sequence(self, x, dt, state=None):
        if isinstance(state, dict):
            core_state = state["core_state"]
            initialized = state["input_initialized"].to(device=x.device)
        else:
            core_state = state
            initialized = torch.zeros(
                (x.shape[0], 4), dtype=torch.bool, device=x.device
            )
        normalized = (x - self.input_mean) / self.input_std
        normalized = normalized.clone()
        if x.shape[-1] >= 8:
            masks = x[..., 4:8] > 0.5
            for step in range(x.shape[1]):
                missing_without_history = ~initialized & ~masks[:, step]
                normalized[:, step, :4] = torch.where(
                    missing_without_history,
                    torch.zeros_like(normalized[:, step, :4]),
                    normalized[:, step, :4],
                )
                initialized = initialized | masks[:, step]
        output, next_core_state = self.model.forward_sequence(
            normalized, dt, core_state
        )
        next_state = {
            "core_state": next_core_state,
            "input_initialized": initialized,
        }
        return output * self.target_std + self.target_mean, next_state


def _validate_checkpoint_tuning_binding(
    checkpoint: dict[str, Any], tuning: dict[str, Any]
) -> None:
    """Bind a confirmatory checkpoint to the frozen per-model optimizer winner."""
    model_name = str(checkpoint.get("registered_name"))
    selected = tuning.get("selected", {}).get(model_name)
    if not isinstance(selected, dict):
        raise ValueError(f"Tuning artifact lacks selected optimizer for {model_name}")
    optimizer = checkpoint.get("optimizer_hyperparameters", {})
    for key in ("learning_rate", "weight_decay"):
        if float(optimizer.get(key, float("nan"))) != float(selected[key]):
            raise ValueError(f"Checkpoint optimizer differs from tuning winner for {key}")
    expected = {
        "trajectory_manifest_sha256": tuning.get("trajectory_manifest_sha256"),
        "parameter_registry_sha256": tuning.get("parameter_registry_sha256"),
        "parameter_registry_implementation_sha256": tuning.get(
            "implementation_sha256"
        ),
        "environment_sha256": tuning.get("environment_sha256"),
        "controller_selection_sha256": tuning.get(
            "controller_selection_sha256"
        ),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint/tuning frozen identity mismatch for {key}")
    matching_candidates = [
        row
        for row in tuning.get("candidates", [])
        if row.get("registered_name") == model_name
        and float(row.get("learning_rate", float("nan")))
        == float(selected["learning_rate"])
        and float(row.get("weight_decay", float("nan")))
        == float(selected["weight_decay"])
        and row.get("checkpoint_sha256") == selected.get("checkpoint_sha256")
    ]
    if len(matching_candidates) != 1:
        raise ValueError("Tuning winner lacks unique candidate provenance")


_REGISTRY_NONIDENTITY_FIELDS = frozenset({"registry_sha256", "created_at_utc"})
# No topology-audit field is runtime-generated today.  Keeping an explicit
# allowlist prevents a future ad-hoc annotation from silently becoming trusted
# checkpoint input; additions require a reviewed schema change.
_TOPOLOGY_RUNTIME_ANNOTATION_KEYS: frozenset[str] = frozenset()


def _registry_identity_core(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in registry.items()
        if key not in _REGISTRY_NONIDENTITY_FIELDS
    }


def _authoritative_observer_registry(
    config: dict[str, Any],
    external_registry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Rebuild registry identity independently of checkpoint metadata."""
    rebuilt = build_parameter_registry(config)
    if external_registry is None:
        return rebuilt
    if not isinstance(external_registry, dict):
        raise ValueError("Expected parameter registry must be a mapping")
    external_core = _registry_identity_core(external_registry)
    if canonical_json_hash(external_core) != external_registry.get(
        "registry_sha256"
    ):
        raise ValueError("Expected parameter registry self-hash mismatch")
    if external_core != _registry_identity_core(rebuilt):
        raise ValueError(
            "Expected parameter registry differs from deterministic config rebuild"
        )
    return external_registry


def _validate_checkpoint_model_metadata(
    checkpoint: dict[str, Any], model: nn.Module
) -> None:
    expected_audit = getattr(model, "topology_audit", {})
    checkpoint_audit = checkpoint.get("topology_audit")
    if not isinstance(expected_audit, dict) or not isinstance(checkpoint_audit, dict):
        raise ValueError("Checkpoint topology audit must be a mapping")
    expected_static = {
        key: value
        for key, value in expected_audit.items()
        if key not in _TOPOLOGY_RUNTIME_ANNOTATION_KEYS
    }
    checkpoint_static = {
        key: value
        for key, value in checkpoint_audit.items()
        if key not in _TOPOLOGY_RUNTIME_ANNOTATION_KEYS
    }
    if checkpoint_static != expected_static:
        raise ValueError(
            "Checkpoint topology audit differs from independently rebuilt model"
        )
    if checkpoint.get("elapsed_time_adapter") != getattr(
        model, "elapsed_time_adapter", None
    ):
        raise ValueError(
            "Checkpoint elapsed-time adapter differs from independently rebuilt model"
        )


def load_trained_observer(
    checkpoint_path: str | Path,
    device: str = "cpu",
    *,
    expected_config: dict[str, Any] | None = None,
    expected_registry: dict[str, Any] | None = None,
    expected_trajectory_manifest_sha256: str | None = None,
):
    checkpoint = torch.load(
        Path(checkpoint_path).resolve(), map_location=device, weights_only=False
    )
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint lacks its immutable source config")
    if expected_config is None and expected_registry is None:
        raise ValueError(
            "Observer loading requires expected_config or an expected parameter registry"
        )
    checkpoint_config_sha256 = canonical_json_hash(checkpoint_config)
    if checkpoint.get("config_sha256") != checkpoint_config_sha256:
        raise ValueError("Checkpoint config payload/hash mismatch")
    if (
        checkpoint.get("parameter_registry_source_config_sha256")
        != checkpoint_config_sha256
    ):
        raise ValueError("Checkpoint parameter-registry source/config mismatch")
    if expected_config is not None:
        expected_config_sha256 = canonical_json_hash(expected_config)
        if checkpoint_config_sha256 != expected_config_sha256:
            raise ValueError("Checkpoint config differs from current evaluation config")
    authoritative_config = (
        expected_config if expected_config is not None else checkpoint_config
    )
    registry = _authoritative_observer_registry(
        authoritative_config, expected_registry
    )
    registered_name = checkpoint.get("registered_name")
    if registered_name not in registry.get("entries", {}):
        raise ValueError("Checkpoint model is absent from authoritative registry")
    if checkpoint.get("parameter_registry_sha256") != registry.get(
        "registry_sha256"
    ):
        raise ValueError("Checkpoint parameter-registry SHA differs from authority")
    if checkpoint.get("parameter_entry") != registry["entries"][registered_name]:
        raise ValueError("Checkpoint parameter entry differs from authority")
    if checkpoint.get("parameter_registry_implementation") != registry.get(
        "implementation"
    ):
        raise ValueError(
            "Checkpoint parameter-registry implementation differs from authority"
        )
    if checkpoint.get("parameter_registry_implementation_sha256") != registry.get(
        "implementation", {}
    ).get("aggregate_sha256"):
        raise ValueError(
            "Checkpoint parameter-registry implementation SHA differs from authority"
        )
    from .formal import (
        _load_hashed,
        formal_artifact_identities,
        is_formal,
    )

    if is_formal(authoritative_config):
        current_formal = formal_artifact_identities(authoritative_config)
        tuning_path = authoritative_config.get("tuning", {}).get("artifact")
        if not tuning_path:
            raise ValueError("Formal checkpoint config lacks tuning artifact")
        tuning = _load_hashed(tuning_path, "tuning_selection_sha256")
        current_formal["tuning_selection_sha256"] = tuning[
            "tuning_selection_sha256"
        ]
        for key, value in current_formal.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Checkpoint formal identity mismatch for {key}")
        if tuning.get("source_config_sha256") != checkpoint_config_sha256:
            raise ValueError("Formal tuning artifact source config mismatch")
        _validate_checkpoint_tuning_binding(checkpoint, tuning)
        if (
            expected_trajectory_manifest_sha256 is not None
            and tuning.get("trajectory_manifest_sha256")
            != expected_trajectory_manifest_sha256
        ):
            raise ValueError("Formal tuning artifact trajectory identity mismatch")
        if (
            checkpoint.get("optimizer_selection_source")
            != "frozen_seed17_tuning_selection"
        ):
            raise ValueError("Formal checkpoint bypassed frozen optimizer selection")
    if (
        expected_trajectory_manifest_sha256 is not None
        and checkpoint.get("trajectory_manifest_sha256")
        != expected_trajectory_manifest_sha256
    ):
        raise ValueError("Checkpoint trajectory identity differs from evaluation data")
    registered_models = set(
        authoritative_config.get("matching", {}).get("registered_models", [])
    )
    if checkpoint.get("registered_name") not in registered_models:
        raise ValueError("Checkpoint model is not registered by its source config")
    registered_seeds = {
        int(value)
        for value in authoritative_config.get("training", {}).get("seeds", [])
    }
    if registered_seeds and int(checkpoint.get("seed")) not in registered_seeds:
        raise ValueError("Checkpoint seed is not registered by its source config")
    if not checkpoint.get("selection_finalized_after_training_run", False):
        raise ValueError("Checkpoint selection was not finalized after training")
    early_stopping_enabled = bool(
        authoritative_config.get("training", {}).get(
            "early_stopping_enabled", False
        )
    )
    if not early_stopping_enabled:
        completed = checkpoint.get("training_run_optimizer_steps_completed")
        planned = checkpoint.get("training_run_optimizer_steps_planned")
        declared = authoritative_config.get("training", {}).get(
            "planned_optimizer_steps"
        )
        if (
            checkpoint.get("training_run_fixed_budget_gate_passed") is not True
            or completed != planned
            or (declared is not None and completed != declared)
        ):
            raise ValueError("Checkpoint failed the confirmatory fixed-budget gate")
    model = build_registered_model(
        registered_name, authoritative_config, registry
    )
    _validate_checkpoint_model_metadata(checkpoint, model)
    load_graph_safe_state_dict(model, checkpoint["model_state_dict"], strict=True)
    wrapped = NormalizedObserver(model, Normalizer.from_dict(checkpoint["normalizer"]))
    wrapped.to(device).eval()
    return wrapped, checkpoint
