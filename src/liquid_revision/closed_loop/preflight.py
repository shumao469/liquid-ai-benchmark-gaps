"""Fail-closed preflight evidence for confirmatory closed-loop runs.

The preflight deliberately runs before any confirmatory model/seed reservation.
It may inspect the committed test *identifiers* in the frozen trajectory manifest,
but it never materializes a test trajectory, evaluates a test condition, or loads a
confirmatory checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import load_config
from ..data.closed_loop import _make_controller, load_closed_loop_dataset
from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file
from ..models.liquid import load_graph_safe_state_dict
from ..perturbations import packet_loss_run_lengths
from .baselines import build_nonlearned_baseline
from .benchmark import (
    _estimate_state_for_control,
    _paired_schedule_seed,
    _rollout,
    _validate_evaluation_shard,
    _write_csv,
    _write_step_shard,
    build_response_conditions,
)
from .formal import (
    apply_and_validate_precision_policy,
    current_run_accelerator_identity,
    is_development_only,
    is_formal,
    validate_controller_selection,
    validate_environment_freeze,
    validate_tuning_selection,
)
from .manifests import load_trajectory_manifest, verify_split_datasets
from .parameter_matching import (
    REGISTERED_MODELS,
    _closed_loop_implementation_manifest,
    build_registered_model,
    load_parameter_registry,
)
from .training import (
    NormalizedObserver,
    Normalizer,
    _atomic_torch_save_new,
    fit_normalizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_AUDITS = (
    "cuda_smoke",
    "validation_perturbation",
    "tiny_overfit_finite_gradients",
    "dt_path",
    "information_flow",
    "one_seed_raw_log_pilot",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        (PROJECT_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _receipt_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"preflight_receipt_sha256", "created_at_utc"}
    }


def _validate_audit_bundle(audits: Mapping[str, Any]) -> None:
    if set(audits) != set(REQUIRED_AUDITS):
        missing = sorted(set(REQUIRED_AUDITS) - set(audits))
        extra = sorted(set(audits) - set(REQUIRED_AUDITS))
        raise ValueError(f"Preflight audit set mismatch; missing={missing}, extra={extra}")
    for name in REQUIRED_AUDITS:
        row = audits[name]
        if not isinstance(row, Mapping) or row.get("passed") is not True:
            raise ValueError(f"Preflight audit did not pass: {name}")
    model_rows = audits["tiny_overfit_finite_gradients"].get("models")
    dt_rows = audits["dt_path"].get("models")
    expected = set(REGISTERED_MODELS)
    if not isinstance(model_rows, Mapping) or set(model_rows) != expected:
        raise ValueError("Tiny-overfit audit must contain exactly all seven models")
    if not isinstance(dt_rows, Mapping) or set(dt_rows) != expected:
        raise ValueError("dt-path audit must contain exactly all seven models")
    _validate_raw_pilot_artifacts(audits["one_seed_raw_log_pilot"])


def write_preflight_receipt(
    output_path: str | Path,
    *,
    identities: Mapping[str, Any],
    audits: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a complete receipt once; partial/failed evidence is never promoted."""
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite preflight receipt: {destination}")
    required_identities = {
        "config_sha256",
        "source_implementation_sha256",
        "environment_sha256",
        "controller_selection_sha256",
        "trajectory_manifest_sha256",
        "data_completion_sha256",
        "parameter_registry_sha256",
        "tuning_selection_sha256",
        "mujoco_xml_sha256",
        "selected_physical_gpu",
        "precision_policy",
    }
    missing = sorted(required_identities - set(identities))
    if missing:
        raise ValueError(f"Preflight identity bundle is incomplete: {missing}")
    if not identities.get("selected_physical_gpu"):
        raise ValueError("Preflight lacks a selected physical GPU UUID/PCI identity")
    if not isinstance(identities.get("precision_policy"), Mapping):
        raise ValueError("Preflight lacks a precision policy")
    _validate_audit_bundle(audits)
    core = {
        "schema_version": 1,
        "status": "complete",
        "phase": "pre_confirmatory_preflight",
        "validation_only": True,
        "test_episodes_used": 0,
        "test_episodes_materialized": 0,
        "test_conditions_evaluated": 0,
        "all_required_gates_passed": True,
        # Stable top-level component identities are intentionally duplicated
        # from the complete identity bundle so the independently implemented
        # pretraining master can bind them without interpreting nested data.
        "source_config_sha256": identities["config_sha256"],
        "source_implementation_sha256": identities[
            "source_implementation_sha256"
        ],
        "environment_sha256": identities["environment_sha256"],
        "mujoco_xml_sha256": identities["mujoco_xml_sha256"],
        "controller_selection_sha256": identities[
            "controller_selection_sha256"
        ],
        "trajectory_manifest_sha256": identities[
            "trajectory_manifest_sha256"
        ],
        "data_completion_sha256": identities["data_completion_sha256"],
        "parameter_registry_sha256": identities[
            "parameter_registry_sha256"
        ],
        "tuning_selection_sha256": identities["tuning_selection_sha256"],
        "identities": dict(identities),
        "audits": dict(audits),
    }
    receipt = {
        **core,
        "preflight_receipt_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, receipt)
    return receipt


def load_preflight_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if canonical_json_hash(_receipt_core(payload)) != payload.get(
        "preflight_receipt_sha256"
    ):
        raise ValueError(f"Preflight receipt hash mismatch: {source}")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("all_required_gates_passed") is not True
        or payload.get("validation_only") is not True
        or int(payload.get("test_episodes_used", -1)) != 0
        or int(payload.get("test_episodes_materialized", -1)) != 0
        or int(payload.get("test_conditions_evaluated", -1)) != 0
    ):
        raise ValueError("Preflight receipt is not a completed validation-only gate")
    identities = payload.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("Preflight receipt lacks component identities")
    top_level_bindings = {
        "source_config_sha256": "config_sha256",
        "source_implementation_sha256": "source_implementation_sha256",
        "environment_sha256": "environment_sha256",
        "mujoco_xml_sha256": "mujoco_xml_sha256",
        "controller_selection_sha256": "controller_selection_sha256",
        "trajectory_manifest_sha256": "trajectory_manifest_sha256",
        "data_completion_sha256": "data_completion_sha256",
        "parameter_registry_sha256": "parameter_registry_sha256",
        "tuning_selection_sha256": "tuning_selection_sha256",
    }
    for top_level, nested in top_level_bindings.items():
        if payload.get(top_level) != identities.get(nested):
            raise ValueError(f"Preflight top-level/nested mismatch for {top_level}")
    _validate_audit_bundle(payload.get("audits", {}))
    return payload


def _cuda_smoke(config: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal preflight requires an available CUDA device")
    precision = apply_and_validate_precision_policy(config)
    accelerator = current_run_accelerator_identity(
        resolved, require_physical_mapping=True, config=config
    )
    if str(accelerator["selected_physical_gpu"].get("physical_index")) != "0":
        raise RuntimeError("Closed-loop formal preflight is authorized only on physical GPU 0")
    generator = torch.Generator(device="cpu").manual_seed(8675309)
    left = torch.randn(64, 64, generator=generator).to(resolved).requires_grad_(True)
    right = torch.randn(64, 64, generator=generator).to(resolved)
    loss = (left @ right).square().mean()
    loss.backward()
    torch.cuda.synchronize(resolved)
    if not torch.isfinite(loss) or left.grad is None or not torch.all(
        torch.isfinite(left.grad)
    ):
        raise FloatingPointError("CUDA smoke produced a non-finite value or gradient")
    return {
        "passed": True,
        "device": accelerator,
        "precision_policy": precision,
        "smoke_loss": float(loss.detach().cpu()),
        "finite_gradient": True,
    }


def _validation_perturbation_audit(
    trajectory: dict[str, Any],
    validation_manifest: dict[str, Any],
    validation_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = trajectory["splits"]["val"]
    expected_specs = [row["training_corruption"]["spec"] for row in rows]
    observed_specs = validation_manifest.get("episode_perturbation_specs")
    if observed_specs != expected_specs:
        raise ValueError("Validation perturbation specs differ from the frozen rows")
    kinds = Counter(str(row["training_corruption"]["kind"]) for row in rows)
    if set(kinds) != {"clean", "iid", "burst"} or any(
        int(kinds[name]) <= 0 for name in ("clean", "iid", "burst")
    ):
        raise ValueError("Validation split must contain clean, IID, and burst episodes")
    if validation_manifest.get("source_split") != "val":
        raise ValueError("Validation dataset has the wrong source split")
    offsets = np.asarray(validation_arrays.get("episode_offsets"), dtype=np.int64)
    masks = np.asarray(validation_arrays.get("sensor_mask"), dtype=np.float64)
    if (
        offsets.ndim != 1
        or len(offsets) != len(rows) + 1
        or offsets[0] != 0
        or offsets[-1] != len(masks)
        or masks.ndim != 2
        or masks.shape[1] != 4
    ):
        raise ValueError("Validation perturbation arrays have an invalid schema")
    # Registered dropout is a complete proprioceptive packet.  Audit this
    # invariant before reducing the masks to one packet indicator.
    if not np.all(masks == masks[:, :1]):
        raise ValueError("Validation dropout is not a complete four-channel packet")
    fraction_tolerance = 0.01
    mean_run_relative_tolerance = 0.10
    fraction_rows: list[dict[str, Any]] = []
    burst_runs_by_length: dict[int, list[int]] = {}
    for kind in ("clean", "iid", "burst"):
        requested_missing_steps = 0.0
        realized_missing_steps = 0
        total_steps = 0
        episode_count = 0
        for index, row in enumerate(rows):
            if str(row["training_corruption"]["kind"]) != kind:
                continue
            start, stop = int(offsets[index]), int(offsets[index + 1])
            packet_mask = masks[start:stop, 0] > 0.5
            steps = int(stop - start)
            requested_fraction = float(
                row["training_corruption"]["target_missing_fraction"]
            )
            requested_missing_steps += requested_fraction * steps
            realized_missing_steps += int(np.count_nonzero(~packet_mask))
            total_steps += steps
            episode_count += 1
            if kind == "burst":
                requested_length = int(
                    row["training_corruption"]["spec"]["burst_length_steps"]
                )
                burst_runs_by_length.setdefault(requested_length, []).extend(
                    packet_loss_run_lengths(
                        np.repeat(packet_mask[:, None], 4, axis=1)
                    )
                )
        if episode_count <= 0 or total_steps <= 0:
            raise ValueError(f"Validation perturbation audit lacks {kind} episodes")
        requested_fraction = requested_missing_steps / total_steps
        realized_fraction = realized_missing_steps / total_steps
        passed = abs(realized_fraction - requested_fraction) <= fraction_tolerance
        fraction_rows.append(
            {
                "kind": kind,
                "episode_count": episode_count,
                "step_count": total_steps,
                "requested_missing_fraction": requested_fraction,
                "realized_missing_fraction": realized_fraction,
                "absolute_error": abs(realized_fraction - requested_fraction),
                "absolute_tolerance": fraction_tolerance,
                "passed": bool(passed),
            }
        )
    burst_rows: list[dict[str, Any]] = []
    for requested_length, runs in sorted(burst_runs_by_length.items()):
        if not runs:
            raise ValueError(
                f"Validation burst audit observed no runs for L={requested_length}"
            )
        realized = float(np.mean(runs))
        relative_error = abs(realized - requested_length) / requested_length
        burst_rows.append(
            {
                "requested_mean_run_steps": requested_length,
                "realized_mean_run_steps": realized,
                "run_count": len(runs),
                "relative_error": relative_error,
                "relative_tolerance": mean_run_relative_tolerance,
                "passed": bool(relative_error <= mean_run_relative_tolerance),
            }
        )
    if not burst_rows or not all(row["passed"] for row in fraction_rows + burst_rows):
        raise RuntimeError(
            "Validation perturbation realization gate failed: "
            f"fractions={fraction_rows}, burst_runs={burst_rows}"
        )
    return {
        "passed": True,
        "validation_episode_count": len(rows),
        "kind_counts": dict(sorted(kinds.items())),
        "perturbation_specs_sha256": canonical_json_hash(expected_specs),
        "packet_dropout_four_channels_together": True,
        "missing_fraction_absolute_tolerance": fraction_tolerance,
        "burst_mean_relative_tolerance": mean_run_relative_tolerance,
        "missing_fraction_audit": fraction_rows,
        "burst_run_length_audit": burst_rows,
        "plant_perturbations_during_rollout": bool(
            validation_manifest.get("supervision", {}).get(
                "plant_perturbations_during_rollout", False
            )
        ),
    }


def _tiny_batch(data_root: Path, config: dict[str, Any], device: Any):
    import torch

    arrays, _ = load_closed_loop_dataset(data_root / "train")
    offsets = np.asarray(arrays["episode_offsets"], dtype=np.int64)
    sequence_steps = int(config.get("preflight", {}).get("tiny_sequence_steps", 12))
    batch_size = int(config.get("preflight", {}).get("tiny_batch_size", 2))
    if sequence_steps < 2 or batch_size < 1 or len(offsets) <= batch_size:
        raise ValueError("Invalid preflight tiny-batch dimensions")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    dt_rows: list[np.ndarray] = []
    for episode in range(batch_size):
        start, stop = int(offsets[episode]), int(offsets[episode + 1])
        if stop - start < sequence_steps:
            raise ValueError("Training episode is too short for the tiny-overfit audit")
        x_rows.append(arrays["observer_input"][start : start + sequence_steps])
        y_rows.append(arrays["target_state"][start : start + sequence_steps])
        dt_rows.append(
            arrays["observed_timestamp_dt"][start : start + sequence_steps].reshape(-1)
        )
    x_np = np.stack(x_rows).astype(np.float32)
    y_np = np.stack(y_rows).astype(np.float32)
    dt_np = np.stack(dt_rows).astype(np.float32)
    x_mean = x_np.mean(axis=(0, 1), keepdims=True)
    x_std = np.maximum(x_np.std(axis=(0, 1), keepdims=True), 1e-5)
    y_mean = y_np.mean(axis=(0, 1), keepdims=True)
    y_std = np.maximum(y_np.std(axis=(0, 1), keepdims=True), 1e-5)
    x_np = (x_np - x_mean) / x_std
    y_np = (y_np - y_mean) / y_std
    if not np.isfinite(x_np).all() or not np.isfinite(y_np).all() or np.any(dt_np <= 0):
        raise ValueError("Tiny-overfit batch is non-finite or has non-positive dt")
    return (
        torch.as_tensor(x_np, device=device),
        torch.as_tensor(y_np, device=device),
        torch.as_tensor(dt_np, device=device),
        {
            "batch_size": batch_size,
            "sequence_steps": sequence_steps,
            "source_episode_indices": list(range(batch_size)),
            "source_split": "train",
        },
    )


def _model_path_audits(
    config: dict[str, Any],
    registry: dict[str, Any],
    data_root: Path,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    registered = tuple(config.get("matching", {}).get("registered_models", ()))
    if set(registered) != set(REGISTERED_MODELS) or len(registered) != len(
        REGISTERED_MODELS
    ):
        raise ValueError("Formal preflight requires the exact seven-model registry")
    resolved = torch.device(device)
    x, target, dt, source = _tiny_batch(data_root, config, resolved)
    steps = int(config.get("preflight", {}).get("tiny_overfit_steps", 16))
    learning_rate = float(
        config.get("preflight", {}).get("tiny_overfit_learning_rate", 3e-3)
    )
    if steps <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid tiny-overfit preflight settings")
    tiny_rows: dict[str, Any] = {}
    dt_rows: dict[str, Any] = {}
    time_aware = set(REGISTERED_MODELS) - {"lstm"}
    for index, model_name in enumerate(REGISTERED_MODELS):
        torch.manual_seed(91000 + index)
        torch.cuda.manual_seed_all(91000 + index)
        model = build_registered_model(model_name, config, registry).to(resolved)
        model.eval()
        with torch.no_grad():
            nominal = model(x, dt)
            perturbed = model(x, dt * 1.75)
        delta = float(torch.max(torch.abs(nominal - perturbed)).detach().cpu())
        if not math.isfinite(delta):
            raise FloatingPointError(f"Non-finite dt-path result for {model_name}")
        expects_dt = model_name in time_aware
        if expects_dt and delta <= 1e-9:
            raise RuntimeError(f"Registered time-aware model ignores dt: {model_name}")
        if not expects_dt and delta != 0.0:
            raise RuntimeError(f"Registered discrete model unexpectedly consumes dt: {model_name}")
        dt_rows[model_name] = {
            "expects_dt_dependence": expects_dt,
            "max_abs_output_change": delta,
            "passed": True,
        }

        model.train(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        losses: list[float] = []
        gradient_norms: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, dt)
            loss = torch.mean((prediction - target) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite tiny-overfit loss for {model_name}")
            loss.backward()
            squared = torch.zeros((), device=resolved)
            gradient_count = 0
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                gradient_count += int(parameter.grad.numel())
                if not torch.all(torch.isfinite(parameter.grad)):
                    raise FloatingPointError(
                        f"Non-finite tiny-overfit gradient for {model_name}"
                    )
                squared = squared + parameter.grad.square().sum()
            gradient_norm = float(torch.sqrt(squared).detach().cpu())
            if gradient_count == 0 or not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError(f"Invalid gradient path for {model_name}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(gradient_norm)
        initial = losses[0]
        best = min(losses[1:]) if len(losses) > 1 else losses[0]
        if not best < initial:
            raise RuntimeError(f"Tiny-overfit loss did not improve for {model_name}")
        tiny_rows[model_name] = {
            "passed": True,
            "optimizer_steps": steps,
            "initial_loss": initial,
            "best_post_initial_loss": best,
            "final_loss": losses[-1],
            "minimum_gradient_norm": min(gradient_norms),
            "maximum_gradient_norm": max(gradient_norms),
            "all_losses_finite": True,
            "all_gradients_finite": True,
        }
        del optimizer, model
        torch.cuda.empty_cache()
    return (
        {"passed": True, "source": source, "models": tiny_rows},
        {
            "passed": True,
            "dt_multiplier": 1.75,
            "time_aware_models": sorted(time_aware),
            "dt_invariant_models": ["lstm"],
            "models": dt_rows,
        },
    )


def _information_flow_audit(
    config: dict[str, Any],
    data_root: Path,
    trajectory: dict[str, Any],
    controller: dict[str, Any],
    tuning: dict[str, Any],
) -> dict[str, Any]:
    import torch

    complete = json.loads((data_root / "COMPLETE.json").read_text(encoding="utf-8"))
    if (data_root / "test").exists():
        raise ValueError("Preflight found a materialized test dataset")
    if (
        complete.get("materialized_splits") != ["train", "val"]
        or complete.get("test_rows_committed_not_materialized") is not True
    ):
        raise ValueError("Data completion marker violates the pre-test boundary")
    if (
        controller.get("validation_only") is not True
        or int(controller.get("test_episodes_used", -1)) != 0
        or tuning.get("validation_only") is not True
        or int(tuning.get("test_episodes_used", -1)) != 0
    ):
        raise ValueError("Controller/tuning evidence crossed the test boundary")
    split_ids = {
        split: {str(row["trajectory_id"]) for row in rows}
        for split, rows in trajectory["splits"].items()
    }
    if any(
        split_ids[left] & split_ids[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Trajectory IDs leak across frozen splits")

    class _SpyObserver(torch.nn.Module):
        """Deterministic probe of the exact estimator dispatch used by _rollout."""

        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.last_input: torch.Tensor | None = None

        def forward_sequence(self, x, dt, state=None):
            del dt
            self.last_input = x.detach().clone()
            return x[..., :4], state

    clean_a = np.asarray(
        [0.10, -0.20, 0.30, -0.40, 0.15, 0.25, 0.01, -0.02],
        dtype=np.float32,
    )
    clean_b = clean_a.copy()
    clean_b[:4] += np.asarray([1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    measured = np.asarray(
        [0.11, -0.19, 0.29, -0.39, 0.15, 0.25, 0.01, -0.02],
        dtype=np.float32,
    )
    mask = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    age = np.asarray([0.0, 0.02, 0.0, 0.04], dtype=np.float32)
    previous_torque = np.asarray([0.2, -0.3], dtype=np.float32)
    spy = _SpyObserver()
    learned = {"model_name": "preflight_spy", "model": spy}
    first, _, observer_input_a = _estimate_state_for_control(
        learned,
        clean=clean_a,
        measured=measured,
        mask=mask,
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    second, _, observer_input_b = _estimate_state_for_control(
        learned,
        clean=clean_b,
        measured=measured,
        mask=mask,
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    if (
        observer_input_a is None
        or observer_input_a.shape != (19,)
        or not np.array_equal(observer_input_a, observer_input_b)
        or not np.array_equal(first, second)
        or spy.last_input is None
        or tuple(spy.last_input.shape) != (1, 1, 19)
    ):
        raise RuntimeError("Learned rollout path leaks the clean simulator state")
    oracle = build_nonlearned_baseline("full_state_pd")
    oracle_a, _, _ = _estimate_state_for_control(
        {"model_name": "full_state_pd", "baseline": oracle},
        clean=clean_a,
        measured=measured,
        mask=mask,
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    corrupted_measured = measured.copy()
    corrupted_measured[:4] = -999.0
    oracle_corrupt, _, _ = _estimate_state_for_control(
        {"model_name": "full_state_pd", "baseline": oracle},
        clean=clean_a,
        measured=corrupted_measured,
        mask=np.zeros(4, dtype=np.float32),
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    oracle_b, _, _ = _estimate_state_for_control(
        {"model_name": "full_state_pd", "baseline": oracle},
        clean=clean_b,
        measured=measured,
        mask=mask,
        age=age,
        observed_dt=0.021,
        previous_control=previous_torque,
        recurrent_state=None,
        device="cpu",
    )
    if not np.array_equal(oracle_a, oracle_corrupt) or np.array_equal(
        oracle_a, oracle_b
    ):
        raise RuntimeError("Full-state oracle information-flow semantics changed")
    shared_controller = _make_controller(
        "reacher_v5",
        dict(config.get("controller", {})),
        float(config.get("environment", {}).get("action_limit", 1.0)),
    )
    action_a = shared_controller.control(first, measured)
    action_sensor_corrupt = shared_controller.control(first, corrupted_measured)
    if not np.array_equal(action_a, action_sensor_corrupt):
        raise RuntimeError("Controller consumed corrupted sensor channels directly")
    return {
        "passed": True,
        "materialized_splits": ["train", "val"],
        "test_rows_committed_not_materialized": True,
        "test_dataset_path_absent": True,
        "controller_test_episodes_used": 0,
        "tuning_test_episodes_used": 0,
        "preflight_test_conditions_evaluated": 0,
        "split_id_hashes": {
            name: canonical_json_hash(sorted(values))
            for name, values in split_ids.items()
        },
        "runtime_probe": {
            "audited_callable": (
                "benchmark._estimate_state_for_control, the sole estimator dispatch "
                "inside benchmark._rollout"
            ),
            "learned_input_dimension": 19,
            "learned_clean_state_substitution_invariant": True,
            "learned_controller_sensor_corruption_invariant": True,
            "full_state_oracle_sensor_corruption_invariant": True,
            "full_state_oracle_clean_state_sensitive": True,
        },
    }


def _raw_log_capacity_projection(
    config: dict[str, Any], registry: dict[str, Any], log_root: Path
) -> dict[str, Any]:
    conditions = build_response_conditions(config)
    test_episodes = int(config.get("splits", {}).get("test", {}).get("episodes", 0))
    horizon = int(config.get("environment", {}).get("horizon_steps", 0))
    learned_estimators = len(registry["entries"])
    baseline_estimators = len(
        config.get("evaluation", {}).get("nonlearned_baselines", [])
    )
    if not conditions or test_episodes <= 0 or horizon <= 0:
        raise ValueError("Raw-log projection has invalid condition/episode/horizon counts")
    rows = (
        len(conditions)
        * test_episodes
        * horizon
        * (learned_estimators + baseline_estimators)
    )
    preflight = dict(config.get("preflight", {}))
    bytes_per_step = int(preflight.get("raw_log_bytes_per_step_upper_bound", 1536))
    reserve_multiplier = float(preflight.get("raw_log_free_space_multiplier", 1.25))
    if bytes_per_step < 512 or reserve_multiplier < 1.0:
        raise ValueError("Raw-log projection safety factors are too small")
    projected = int(rows * bytes_per_step)
    required_free = int(math.ceil(projected * reserve_multiplier))
    log_root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(log_root).free)
    if free < required_free:
        raise OSError(
            f"Insufficient space for one-seed raw logs: {free} < {required_free} bytes"
        )
    return {
        "passed": True,
        "scope": "confirmatory raw-log disk-capacity upper bound",
        "condition_count": len(conditions),
        "test_episode_count": test_episodes,
        "horizon_steps_upper_bound": horizon,
        "learned_estimator_count": learned_estimators,
        "nonlearned_baseline_count": baseline_estimators,
        "projected_step_rows_upper_bound": rows,
        "bytes_per_step_upper_bound": bytes_per_step,
        "projected_bytes_upper_bound": projected,
        "free_space_multiplier": reserve_multiplier,
        "required_free_bytes": required_free,
        "observed_free_bytes": free,
        "target_volume": str(log_root.resolve().anchor),
        "test_data_read": False,
    }


def _pilot_manifest_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"pilot_manifest_sha256", "created_at_utc"}
    }


def _validate_raw_pilot_artifacts(audit: Mapping[str, Any]) -> None:
    """Rehash every validation-only pilot payload bound into the receipt."""
    if audit.get("passed") is not True or audit.get("validation_only") is not True:
        raise ValueError("Raw-log pilot is not completed validation-only evidence")
    if int(audit.get("test_episodes_used", -1)) != 0:
        raise ValueError("Raw-log pilot touched a registered test episode")
    manifest_path = Path(str(audit.get("pilot_manifest_path", ""))).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != audit.get(
        "pilot_manifest_file_sha256"
    ):
        raise ValueError("Raw-log pilot manifest file identity mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if canonical_json_hash(_pilot_manifest_core(manifest)) != manifest.get(
        "pilot_manifest_sha256"
    ):
        raise ValueError("Raw-log pilot manifest self-hash mismatch")
    if audit.get("pilot_manifest_sha256") not in {
        None,
        manifest.get("pilot_manifest_sha256"),
    }:
        raise ValueError("Raw-log pilot audit/manifest canonical hash mismatch")
    if (
        manifest.get("phase")
        != "validation_only_preflight_raw_log_pilot"
        or manifest.get("validation_only") is not True
        or int(manifest.get("test_episodes_used", -1)) != 0
        or int(manifest.get("test_conditions_evaluated", -1)) != 0
        or not isinstance(manifest.get("run_identity"), Mapping)
        or canonical_json_hash(manifest["run_identity"])
        != manifest.get("run_identity_sha256")
    ):
        raise ValueError("Raw-log pilot manifest is not validation-only evidence")
    expected_estimators = set(REGISTERED_MODELS) | set(
        manifest.get("nonlearned_baselines", [])
    )
    rows = manifest.get("estimators")
    pilot_seed = int(manifest.get("run_identity", {}).get("raw_log_pilot_seed", -1))
    expected_identities = {
        *((str(name), pilot_seed) for name in REGISTERED_MODELS),
        *((str(name), -1) for name in manifest.get("nonlearned_baselines", [])),
    }
    observed_identities = (
        [
            (str(row.get("model_name")), int(row.get("training_seed", -999)))
            for row in rows
        ]
        if isinstance(rows, list)
        else []
    )
    if (
        not isinstance(rows, list)
        or pilot_seed < 0
        or set(observed_identities) != expected_identities
        or len(observed_identities) != len(expected_identities)
        or {name for name, _ in observed_identities} != expected_estimators
    ):
        raise ValueError("Raw-log pilot estimator coverage is incomplete")
    for row in rows:
        if (
            row.get("deep_parse_passed") is not True
            or not str(row.get("trajectory_id", "")).strip()
            or int(row.get("steps", 0)) <= 0
        ):
            raise ValueError("Raw-log pilot estimator row is incomplete")
        for path_key, sha_key in (
            ("shard_manifest_path", "shard_manifest_file_sha256"),
            ("episode_metrics_path", "episode_metrics_sha256"),
            ("step_log_path", "step_log_sha256"),
        ):
            path = Path(str(row.get(path_key, ""))).resolve()
            if manifest_path.parent not in path.parents:
                raise ValueError("Raw-log pilot payload escaped its frozen root")
            if not path.is_file() or sha256_file(path) != row.get(sha_key):
                raise ValueError(
                    f"Raw-log pilot payload identity mismatch for {row.get('model_name')}"
                )
        checkpoint_path = row.get("pilot_checkpoint_path")
        if str(row.get("model_name")) in REGISTERED_MODELS:
            if checkpoint_path is None:
                raise ValueError("Learned raw-log pilot lacks its checkpoint")
            path = Path(str(checkpoint_path)).resolve()
            if manifest_path.parent not in path.parents:
                raise ValueError("Raw-log pilot checkpoint escaped its frozen root")
            if not path.is_file() or sha256_file(path) != row.get(
                "pilot_checkpoint_sha256"
            ):
                raise ValueError("Raw-log pilot checkpoint identity mismatch")
        elif checkpoint_path is not None or row.get("pilot_checkpoint_sha256") is not None:
            raise ValueError("Nonlearned raw-log pilot unexpectedly has a checkpoint")


def _deep_validate_raw_pilot(
    *,
    audit: Mapping[str, Any],
    config: dict[str, Any],
    registry: dict[str, Any],
    trajectory: dict[str, Any],
    device: str,
) -> None:
    """Reload every pilot estimator and replay every validation-only shard."""
    import torch

    _validate_raw_pilot_artifacts(audit)
    manifest_path = Path(str(audit["pilot_manifest_path"])).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_identity = manifest["run_identity"]
    expected_baselines = [
        str(value)
        for value in config.get("evaluation", {}).get(
            "nonlearned_baselines", []
        )
    ]
    if (
        run_identity.get("phase")
        != "validation_only_preflight_raw_log_pilot"
        or run_identity.get("config_sha256") != canonical_json_hash(config)
        or run_identity.get("trajectory_manifest_sha256")
        != trajectory["manifest_sha256"]
        or int(run_identity.get("test_episodes_used", -1)) != 0
        or manifest.get("registered_models") != list(REGISTERED_MODELS)
        or manifest.get("nonlearned_baselines") != expected_baselines
    ):
        raise ValueError("Raw-log pilot/current frozen chain identity mismatch")
    validation_matches = [
        row
        for row in trajectory["splits"]["val"]
        if row["trajectory_id"] == run_identity.get("validation_trajectory_id")
    ]
    if len(validation_matches) != 1:
        raise ValueError("Raw-log pilot trajectory is not uniquely in validation")
    episode = validation_matches[0]
    condition = run_identity.get("condition")
    registered_conditions = {
        row["condition_id"]: row for row in build_response_conditions(config)
    }
    if (
        not isinstance(condition, Mapping)
        or condition.get("condition_id") != "iid_dropout-0"
        or registered_conditions.get("iid_dropout-0") != dict(condition)
    ):
        raise ValueError("Raw-log pilot condition is not the frozen nominal condition")
    estimator_identities = {
        (str(row["model_name"]), int(row["training_seed"])): row
        for row in run_identity.get("estimator_identities", [])
    }
    if len(estimator_identities) != len(manifest["estimators"]):
        raise ValueError("Raw-log pilot estimator identity set is duplicated")
    pilot_seed = int(run_identity.get("raw_log_pilot_seed", -1))
    expected_estimator_identities = {
        *((str(name), pilot_seed) for name in REGISTERED_MODELS),
        *((str(name), -1) for name in expected_baselines),
    }
    if (
        pilot_seed < 0
        or set(estimator_identities) != expected_estimator_identities
        or len(manifest["estimators"]) != len(expected_estimator_identities)
    ):
        raise ValueError("Raw-log pilot model/seed rows are not exact")
    for row in manifest["estimators"]:
        name = str(row["model_name"])
        seed = int(row["training_seed"])
        frozen_identity = estimator_identities.get((name, seed))
        if (
            frozen_identity is None
            or frozen_identity.get("checkpoint_sha256")
            != row.get("pilot_checkpoint_sha256")
            or row.get("trajectory_id") != episode["trajectory_id"]
            or int(row.get("episode_seed", -1)) != int(episode["episode_seed"])
            or int(row.get("schedule_seed", -1))
            != _paired_schedule_seed(
                int(episode["schedule_seed"]), str(condition["condition_id"])
            )
        ):
            raise ValueError("Raw-log pilot row identity mismatch")
        if name in REGISTERED_MODELS:
            checkpoint_path = Path(row["pilot_checkpoint_path"]).resolve()
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            expected_checkpoint = {
                "phase": "validation_only_preflight_raw_log_pilot",
                "registered_name": name,
                "seed": pilot_seed,
                "config_sha256": canonical_json_hash(config),
                "parameter_registry_sha256": registry["registry_sha256"],
                "test_episodes_used": 0,
            }
            for key, value in expected_checkpoint.items():
                if checkpoint.get(key) != value:
                    raise ValueError(
                        f"Raw-log pilot checkpoint mismatch for {name}/{key}"
                    )
            normalizer = Normalizer.from_dict(checkpoint["normalizer"])
            core = build_registered_model(name, config, registry)
            model = NormalizedObserver(core, normalizer).to(torch.device(device))
            load_graph_safe_state_dict(
                model, checkpoint["model_state_dict"], strict=True
            )
            model.eval()
            estimator = {
                "model_name": name,
                "training_seed": seed,
                "model": model,
                "checkpoint_sha256": row["pilot_checkpoint_sha256"],
            }
        else:
            if name not in expected_baselines:
                raise ValueError("Raw-log pilot contains an unregistered baseline")
            estimator = {
                "model_name": name,
                "training_seed": seed,
                "baseline": build_nonlearned_baseline(
                    name,
                    config.get("evaluation", {}).get("kalman")
                    if name == "kalman_pd"
                    else None,
                ),
                "checkpoint_sha256": None,
            }
        shard_directory = Path(row["shard_manifest_path"]).resolve().parent
        _validate_evaluation_shard(
            shard_directory,
            config=config,
            run_identity_sha256=manifest["run_identity_sha256"],
            condition=dict(condition),
            estimator=estimator,
            test_rows=[episode],
        )


def _one_seed_raw_log_pilot(
    config: dict[str, Any],
    registry: dict[str, Any],
    trajectory: dict[str, Any],
    data_root: Path,
    log_root: Path,
    device: str,
) -> dict[str, Any]:
    """Run and deeply parse one validation episode for all core estimators."""
    import torch

    capacity = _raw_log_capacity_projection(config, registry, log_root)
    pilot_root = (log_root / "preflight_validation_raw_pilot").resolve()
    if pilot_root.exists():
        raise FileExistsError(f"Refusing to overwrite raw-log pilot: {pilot_root}")
    pilot_root.mkdir(parents=True)
    (pilot_root / "models").mkdir()
    (pilot_root / "shards").mkdir()
    validation_rows = list(trajectory["splits"]["val"])
    if not validation_rows:
        raise ValueError("Raw-log pilot lacks a validation episode")
    episode = validation_rows[0]
    nominal_conditions = [
        row
        for row in build_response_conditions(config)
        if row["condition_id"] == "iid_dropout-0"
    ]
    if len(nominal_conditions) != 1:
        raise ValueError("Raw-log pilot lacks the unique nominal registered condition")
    condition = nominal_conditions[0]
    pilot_seed = int(config.get("preflight", {}).get("raw_log_pilot_seed", 919191))
    train_arrays, _ = load_closed_loop_dataset(data_root / "train")
    normalizer = fit_normalizer(train_arrays)
    estimators: list[dict[str, Any]] = []
    for index, model_name in enumerate(REGISTERED_MODELS):
        torch.manual_seed(pilot_seed + index)
        torch.cuda.manual_seed_all(pilot_seed + index)
        core = build_registered_model(model_name, config, registry)
        model = NormalizedObserver(core, normalizer).to(torch.device(device))
        model.eval()
        checkpoint_path = pilot_root / "models" / f"{model_name}.pt"
        _atomic_torch_save_new(
            checkpoint_path,
            {
                "phase": "validation_only_preflight_raw_log_pilot",
                "registered_name": model_name,
                "seed": pilot_seed,
                "model_state_dict": model.state_dict(),
                "normalizer": normalizer.to_dict(),
                "config_sha256": canonical_json_hash(config),
                "parameter_registry_sha256": registry["registry_sha256"],
                "test_episodes_used": 0,
            },
        )
        estimators.append(
            {
                "model_name": model_name,
                "training_seed": pilot_seed,
                "model": model,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "pilot_checkpoint_path": str(checkpoint_path),
            }
        )
    for baseline_name in config.get("evaluation", {}).get(
        "nonlearned_baselines", []
    ):
        estimators.append(
            {
                "model_name": str(baseline_name),
                "training_seed": -1,
                "baseline": build_nonlearned_baseline(
                    str(baseline_name),
                    config.get("evaluation", {}).get("kalman")
                    if baseline_name == "kalman_pd"
                    else None,
                ),
                "checkpoint_sha256": None,
                "pilot_checkpoint_path": None,
            }
        )
    run_identity = {
        "phase": "validation_only_preflight_raw_log_pilot",
        "config_sha256": canonical_json_hash(config),
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "validation_trajectory_id": episode["trajectory_id"],
        "condition": condition,
        "raw_log_pilot_seed": pilot_seed,
        "estimator_identities": [
            {
                "model_name": row["model_name"],
                "training_seed": int(row["training_seed"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for row in estimators
        ],
        "test_episodes_used": 0,
    }
    run_identity_sha256 = canonical_json_hash(run_identity)
    schedule_seed = _paired_schedule_seed(
        int(episode["schedule_seed"]), condition["condition_id"]
    )
    pilot_rows: list[dict[str, Any]] = []
    for estimator in estimators:
        result = _rollout(
            estimator,
            config,
            condition,
            episode_seed=int(episode["episode_seed"]),
            schedule_seed=schedule_seed,
            device=device,
        )
        step_rows = result.pop("_step_log")
        for step_row in step_rows:
            step_row.update(
                {
                    "trajectory_id": episode["trajectory_id"],
                    "episode_seed": int(episode["episode_seed"]),
                    "schedule_seed": int(schedule_seed),
                }
            )
        episode_result = {
            **result,
            "trajectory_id": episode["trajectory_id"],
            "model_name": estimator["model_name"],
            "training_seed": int(estimator["training_seed"]),
            "checkpoint_sha256": estimator["checkpoint_sha256"],
            "condition_id": condition["condition_id"],
            "family": condition["family"],
            "severity": condition["severity"],
        }
        shard_directory = pilot_root / "shards" / str(estimator["model_name"])
        shard_directory.mkdir()
        episode_path = shard_directory / "episode_metrics.csv"
        step_path = shard_directory / "step_log.npz"
        _write_csv(episode_path, [episode_result])
        step_identity = _write_step_shard(step_path, step_rows)
        shard_core = {
            "schema_version": 1,
            "run_identity_sha256": run_identity_sha256,
            "condition_id": condition["condition_id"],
            "model_name": estimator["model_name"],
            "training_seed": int(estimator["training_seed"]),
            "episode_count": 1,
            "step_rows": int(step_identity["step_rows"]),
            "episode_metrics_sha256": sha256_file(episode_path),
            "step_log_sha256": step_identity["sha256"],
        }
        shard_manifest = {
            **shard_core,
            "shard_manifest_sha256": canonical_json_hash(shard_core),
        }
        shard_manifest_path = shard_directory / "manifest.json"
        atomic_write_json(shard_manifest_path, shard_manifest)
        _validate_evaluation_shard(
            shard_directory,
            config=config,
            run_identity_sha256=run_identity_sha256,
            condition=condition,
            estimator=estimator,
            test_rows=[episode],
        )
        checkpoint_path = estimator.get("pilot_checkpoint_path")
        pilot_rows.append(
            {
                "model_name": estimator["model_name"],
                "training_seed": int(estimator["training_seed"]),
                "trajectory_id": episode["trajectory_id"],
                "episode_seed": int(episode["episode_seed"]),
                "schedule_seed": int(schedule_seed),
                "steps": int(result["steps"]),
                "failure": float(result["failure"]),
                "deep_parse_passed": True,
                "shard_manifest_path": str(shard_manifest_path),
                "shard_manifest_file_sha256": sha256_file(shard_manifest_path),
                "episode_metrics_path": str(episode_path),
                "episode_metrics_sha256": sha256_file(episode_path),
                "step_log_path": str(step_path),
                "step_log_sha256": sha256_file(step_path),
                "pilot_checkpoint_path": checkpoint_path,
                "pilot_checkpoint_sha256": (
                    sha256_file(checkpoint_path) if checkpoint_path else None
                ),
            }
        )
    nonlearned = [
        str(value)
        for value in config.get("evaluation", {}).get("nonlearned_baselines", [])
    ]
    manifest_core = {
        "schema_version": 1,
        "phase": "validation_only_preflight_raw_log_pilot",
        "validation_only": True,
        "test_episodes_used": 0,
        "test_conditions_evaluated": 0,
        "run_identity": run_identity,
        "run_identity_sha256": run_identity_sha256,
        "registered_models": list(REGISTERED_MODELS),
        "nonlearned_baselines": nonlearned,
        "estimators": pilot_rows,
    }
    pilot_manifest = {
        **manifest_core,
        "pilot_manifest_sha256": canonical_json_hash(manifest_core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    pilot_manifest_path = pilot_root / "pilot_manifest.json"
    atomic_write_json(pilot_manifest_path, pilot_manifest)
    audit = {
        "passed": True,
        "validation_only": True,
        "test_episodes_used": 0,
        "validation_episode_count": 1,
        "validation_trajectory_id": episode["trajectory_id"],
        "raw_log_pilot_seed": pilot_seed,
        "condition_id": condition["condition_id"],
        "registered_model_count": len(REGISTERED_MODELS),
        "nonlearned_baseline_count": len(nonlearned),
        "all_shards_deep_parsed": True,
        "pilot_manifest_path": str(pilot_manifest_path),
        "pilot_manifest_file_sha256": sha256_file(pilot_manifest_path),
        "pilot_manifest_sha256": pilot_manifest["pilot_manifest_sha256"],
        "capacity_projection": capacity,
    }
    _validate_raw_pilot_artifacts(audit)
    return audit


def create_preflight_receipt(
    *,
    config: dict[str, Any],
    data_root: str | Path,
    registry_path: str | Path,
    output_path: str | Path,
    device: str = "cuda:0",
    raw_log_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run every required gate and emit one immutable receipt on success."""
    if not is_formal(config) or is_development_only(config):
        raise ValueError("Preflight accepts only a non-development formal config")
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite preflight receipt: {destination}")
    data_directory = Path(data_root).resolve()
    registry_source = Path(registry_path).resolve()

    environment = validate_environment_freeze(config)
    controller = validate_controller_selection(config)
    registry = load_parameter_registry(registry_source, expected_config=config)
    verified_data = verify_split_datasets(
        data_directory, expected_config=config, full_replay=False
    )
    trajectory = load_trajectory_manifest(
        data_directory / "trajectory_manifest.json", expected_config=config
    )
    tuning = validate_tuning_selection(config, registry, trajectory)
    if environment is None or controller is None or tuning is None:
        raise ValueError("Formal preflight lacks a required frozen artifact")
    source = _closed_loop_implementation_manifest()
    if source["aggregate_sha256"] != registry["implementation"]["aggregate_sha256"]:
        raise ValueError("Registry and current source implementation differ")

    cuda = _cuda_smoke(config, device)
    validation_arrays, validation_manifest = load_closed_loop_dataset(
        data_directory / "val"
    )
    validation_audit = _validation_perturbation_audit(
        trajectory, validation_manifest, validation_arrays
    )
    tiny, dt_path = _model_path_audits(
        config, registry, data_directory, device
    )
    information_flow = _information_flow_audit(
        config, data_directory, trajectory, controller, tuning
    )
    log_root = (
        Path(raw_log_root).resolve()
        if raw_log_root is not None
        else destination.parent.resolve()
    )
    raw_log = _one_seed_raw_log_pilot(
        config, registry, trajectory, data_directory, log_root, device
    )

    accelerator = cuda["device"]
    identities = {
        "config_sha256": canonical_json_hash(config),
        "source_implementation_sha256": source["aggregate_sha256"],
        "environment_sha256": environment["environment_sha256"],
        "mujoco_xml_sha256": environment["mujoco_xml"]["sha256"],
        "mujoco_xml_path": environment["mujoco_xml"]["path"],
        "controller_selection_sha256": controller[
            "controller_selection_sha256"
        ],
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "trajectory_manifest_file_sha256": sha256_file(
            data_directory / "trajectory_manifest.json"
        ),
        "data_completion_sha256": verified_data["completion_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "parameter_registry_file_sha256": sha256_file(registry_source),
        "tuning_selection_sha256": tuning["tuning_selection_sha256"],
        "selected_physical_gpu": accelerator["selected_physical_gpu"],
        "cuda_visible_device_token": accelerator.get("cuda_visible_device_token"),
        "precision_policy": accelerator["precision_policy"],
    }
    audits = {
        "cuda_smoke": cuda,
        "validation_perturbation": validation_audit,
        "tiny_overfit_finite_gradients": tiny,
        "dt_path": dt_path,
        "information_flow": information_flow,
        "one_seed_raw_log_pilot": raw_log,
    }
    return write_preflight_receipt(
        destination, identities=identities, audits=audits
    )


def validate_preflight_receipt(
    *,
    config: dict[str, Any],
    data_root: str | Path,
    registry_path: str | Path,
    receipt_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Rebind an existing receipt to current immutable inputs and selected GPU."""
    receipt = load_preflight_receipt(receipt_path)
    data_directory = Path(data_root).resolve()
    registry = load_parameter_registry(registry_path, expected_config=config)
    verified = verify_split_datasets(
        data_directory, expected_config=config, full_replay=False
    )
    trajectory = load_trajectory_manifest(
        data_directory / "trajectory_manifest.json", expected_config=config
    )
    environment = validate_environment_freeze(config)
    controller = validate_controller_selection(config)
    tuning = validate_tuning_selection(config, registry, trajectory)
    source = _closed_loop_implementation_manifest()
    accelerator = current_run_accelerator_identity(
        device, require_physical_mapping=True, config=config
    )
    if str(accelerator["selected_physical_gpu"].get("physical_index")) != "0":
        raise RuntimeError("Closed-loop formal runs are authorized only on physical GPU 0")
    expected = {
        "config_sha256": canonical_json_hash(config),
        "source_implementation_sha256": source["aggregate_sha256"],
        "environment_sha256": environment["environment_sha256"],
        "mujoco_xml_sha256": environment["mujoco_xml"]["sha256"],
        "controller_selection_sha256": controller[
            "controller_selection_sha256"
        ],
        "trajectory_manifest_sha256": trajectory["manifest_sha256"],
        "data_completion_sha256": verified["completion_sha256"],
        "parameter_registry_sha256": registry["registry_sha256"],
        "tuning_selection_sha256": tuning["tuning_selection_sha256"],
        "selected_physical_gpu": accelerator["selected_physical_gpu"],
        "precision_policy": accelerator["precision_policy"],
    }
    observed = receipt["identities"]
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"Preflight receipt identity mismatch for {key}")
    _deep_validate_raw_pilot(
        audit=receipt["audits"]["one_seed_raw_log_pilot"],
        config=config,
        registry=registry,
        trajectory=trajectory,
        device=device,
    )
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-log-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config, _ = load_config(args.config)
    receipt = create_preflight_receipt(
        config=config,
        data_root=args.data_root,
        registry_path=args.registry,
        output_path=args.output,
        device=args.device,
        raw_log_root=args.raw_log_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
