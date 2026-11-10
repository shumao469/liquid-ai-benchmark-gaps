"""Portable, validation-only hyperparameter tuning for the reproduction path.

This module deliberately does not participate in the private formal-release
workflow under :mod:`liquid_revision.closed_loop`.  It creates clearly scoped
reproduction artifacts, never opens a test split, and delegates one candidate
at a time to ``reproduction.training.fit_observer(phase="tune")``.

Each candidate runs in a fresh Python process.  The supervisor maps physical
GPU indices or UUIDs through ``CUDA_VISIBLE_DEVICES`` and caps the number of
workers assigned to each GPU.  Candidate specifications and completion
receipts are content-hashed, so an interrupted invocation can be resumed
without silently accepting a different grid, configuration, registry, budget,
readout, or dataset identity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

from liquid_revision.config import load_config
from liquid_revision.closed_loop.parameter_matching import load_parameter_registry
from liquid_revision.io_utils import atomic_write_json, canonical_json_hash, sha256_file


SCHEMA_VERSION = "liquid_revision.reproduction.tuning.v1"
SELECTION_SCHEMA_VERSION = "liquid_revision.reproduction.tuning_selection.v1"
SCOPE = "validation_only_seed17"
TUNING_SEED = 17
UNIT_MODELS = (
    "lstm",
    "odernn",
    "ctrnn",
    "ltc",
    "cfc",
    "ncp_cfc",
    "random_sparse_cfc",
)
SCALED_MODELS = ("cfc", "ncp_cfc", "random_sparse_cfc")
LEARNING_RATES = (3.0e-4, 1.0e-3)
WEIGHT_DECAYS = (0.0, 1.0e-4)
HASH_LENGTH = 64


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model: str
    learning_rate: float
    weight_decay: float
    learning_rate_index: int
    weight_decay_index: int
    directory: Path
    spec_path: Path


@dataclass
class RunningCandidate:
    candidate: Candidate
    gpu_slot: str
    process: subprocess.Popen[str]
    stdout_handle: TextIO
    stderr_handle: TextIO


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid mapping: {key}")
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != HASH_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_registry_identity(path: str | Path, config: dict[str, Any]) -> str:
    registry_path = Path(path).resolve()
    # Reuse the authoritative registry reader.  Its semantic hash intentionally
    # excludes non-identity metadata such as ``created_at_utc`` and its
    # deterministic rebuild checks model widths/topologies against the config.
    registry = load_parameter_registry(registry_path, expected_config=config)
    claimed = registry.get("registry_sha256")
    if not _is_sha256(claimed):
        raise ValueError("Parameter registry lacks a valid registry_sha256")
    if registry.get("source_config_sha256") != canonical_json_hash(config):
        raise ValueError("Parameter registry was not built from the supplied config")
    return str(claimed)


def _expected_models(config: dict[str, Any], readout: str) -> tuple[str, ...]:
    matching = _require_mapping(config, "matching")
    registered = tuple(str(value) for value in matching.get("registered_models", []))
    if registered != UNIT_MODELS:
        raise ValueError(
            "matching.registered_models must preserve the frozen seven-model order: "
            + ", ".join(UNIT_MODELS)
        )
    return UNIT_MODELS if readout == "unit" else SCALED_MODELS


def _validate_tuning_design(
    config: dict[str, Any], *, readout: str
) -> tuple[tuple[str, ...], int, int | None]:
    if readout not in {"unit", "scaled"}:
        raise ValueError("readout must be 'unit' or 'scaled'")
    formal = _require_mapping(config, "formal")
    experiment = _require_mapping(config, "experiment")
    if bool(formal.get("required", False)) or not bool(
        experiment.get("development_only", False)
    ):
        raise ValueError(
            "Portable tuning requires an exploratory reproduction config with "
            "formal.required=false and experiment.development_only=true; it cannot "
            "bypass the private formal-release workflow"
        )
    tuning = _require_mapping(config, "tuning")
    seed = int(tuning.get("seed", -1))
    if seed != TUNING_SEED:
        raise ValueError(f"Tuning seed must remain frozen at {TUNING_SEED}")

    learning_rates = tuple(float(value) for value in tuning.get("learning_rates", []))
    weight_decays = tuple(float(value) for value in tuning.get("weight_decays", []))
    if learning_rates != LEARNING_RATES:
        raise ValueError(f"Frozen learning-rate grid must be {list(LEARNING_RATES)}")
    if weight_decays != WEIGHT_DECAYS:
        raise ValueError(f"Frozen weight-decay grid must be {list(WEIGHT_DECAYS)}")

    epochs = int(tuning.get("epochs", 0))
    if epochs <= 0:
        raise ValueError("tuning.epochs must be positive")
    per_epoch_raw = tuning.get("optimizer_steps_per_epoch")
    planned_raw = tuning.get("planned_optimizer_steps")
    per_epoch = int(per_epoch_raw) if per_epoch_raw is not None else None
    planned = int(planned_raw) if planned_raw is not None else None
    if per_epoch is not None and per_epoch <= 0:
        raise ValueError("tuning.optimizer_steps_per_epoch must be positive")
    if planned is not None and planned <= 0:
        raise ValueError("tuning.planned_optimizer_steps must be positive")
    if planned is not None and per_epoch is None:
        raise ValueError(
            "tuning.planned_optimizer_steps requires optimizer_steps_per_epoch"
        )
    if planned is not None and planned != epochs * int(per_epoch):
        raise ValueError(
            "Frozen tuning optimizer budget is inconsistent: planned_optimizer_steps "
            "must equal epochs * optimizer_steps_per_epoch"
        )
    expected_steps = planned if planned is not None else (
        epochs * per_epoch if per_epoch is not None else None
    )
    return _expected_models(config, readout), epochs, expected_steps


def _candidate_id(model: str, learning_rate_index: int, weight_decay_index: int) -> str:
    return f"{model}__lr{learning_rate_index:02d}__wd{weight_decay_index:02d}"


def _candidate_core(
    *,
    candidate: Candidate,
    config_sha256: str,
    registry_sha256: str,
    readout: str,
    expected_epochs: int,
    expected_optimizer_steps: int | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "scope": SCOPE,
        "phase": "tune",
        "formal_release_artifact": False,
        "candidate_id": candidate.candidate_id,
        "model": candidate.model,
        "seed": TUNING_SEED,
        "readout": readout,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "learning_rate_index": candidate.learning_rate_index,
        "weight_decay_index": candidate.weight_decay_index,
        "expected_epochs": expected_epochs,
        "expected_optimizer_steps": expected_optimizer_steps,
        "config_sha256": config_sha256,
        "registry_sha256": registry_sha256,
    }


def _write_once_json(path: Path, payload: dict[str, Any], *, hash_field: str) -> None:
    if path.is_file():
        existing = _load_json(path)
        if existing != payload:
            raise ValueError(f"Refusing to overwrite a different immutable artifact: {path}")
        return
    if payload.get(hash_field) != canonical_json_hash(
        {key: value for key, value in payload.items() if key != hash_field}
    ):
        raise ValueError(f"Invalid {hash_field} before write: {path}")
    atomic_write_json(path, payload)


def _build_candidates(
    *, output_root: Path, models: Sequence[str]
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for model in models:
        for learning_rate_index, learning_rate in enumerate(LEARNING_RATES):
            for weight_decay_index, weight_decay in enumerate(WEIGHT_DECAYS):
                identifier = _candidate_id(
                    model, learning_rate_index, weight_decay_index
                )
                directory = output_root / "candidates" / model / identifier
                candidates.append(
                    Candidate(
                        candidate_id=identifier,
                        model=model,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        learning_rate_index=learning_rate_index,
                        weight_decay_index=weight_decay_index,
                        directory=directory,
                        spec_path=directory / "candidate_spec.json",
                    )
                )
    return candidates


def build_plan(
    *,
    config_path: str | Path,
    data_root: str | Path,
    registry_path: str | Path,
    output_root: str | Path,
    readout: str,
) -> tuple[dict[str, Any], dict[str, Any], list[Candidate]]:
    """Validate immutable inputs and return a portable tuning plan."""

    config_file = Path(config_path).resolve()
    data_directory = Path(data_root).resolve()
    registry_file = Path(registry_path).resolve()
    destination = Path(output_root).resolve() / readout
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    if not data_directory.is_dir():
        raise FileNotFoundError(data_directory)
    if not registry_file.is_file():
        raise FileNotFoundError(registry_file)

    config, config_sha256 = load_config(config_file)
    models, expected_epochs, expected_optimizer_steps = _validate_tuning_design(
        config, readout=readout
    )
    registry_sha256 = _load_registry_identity(registry_file, config)
    candidates = _build_candidates(output_root=destination, models=models)
    expected_count = 28 if readout == "unit" else 12
    if len(candidates) != expected_count:
        raise AssertionError(
            f"Internal grid error: expected {expected_count}, got {len(candidates)}"
        )

    core = {
        "schema": SCHEMA_VERSION,
        "scope": SCOPE,
        "phase": "tune",
        "formal_release_artifact": False,
        "readout": readout,
        "seed": TUNING_SEED,
        "config_path": str(config_file),
        "config_sha256": config_sha256,
        "registry_path": str(registry_file),
        "registry_sha256": registry_sha256,
        "data_root": str(data_directory),
        "output_root": str(destination),
        "grid": {
            "models": list(models),
            "learning_rates": list(LEARNING_RATES),
            "weight_decays": list(WEIGHT_DECAYS),
            "candidate_count": len(candidates),
        },
        "budget": {
            "epochs": expected_epochs,
            "optimizer_steps": expected_optimizer_steps,
        },
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
    }
    plan = {**core, "tuning_plan_sha256": canonical_json_hash(core)}
    context = {
        "config": config,
        "config_path": config_file,
        "config_sha256": config_sha256,
        "data_root": data_directory,
        "registry_path": registry_file,
        "registry_sha256": registry_sha256,
        "output_root": destination,
        "readout": readout,
        "expected_epochs": expected_epochs,
        "expected_optimizer_steps": expected_optimizer_steps,
    }
    return plan, context, candidates


def _candidate_spec(candidate: Candidate, context: dict[str, Any]) -> dict[str, Any]:
    core = _candidate_core(
        candidate=candidate,
        config_sha256=str(context["config_sha256"]),
        registry_sha256=str(context["registry_sha256"]),
        readout=str(context["readout"]),
        expected_epochs=int(context["expected_epochs"]),
        expected_optimizer_steps=context["expected_optimizer_steps"],
    )
    return {**core, "candidate_spec_sha256": canonical_json_hash(core)}


def _coerce_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Candidate state has invalid {field}") from error
    if not math.isfinite(result):
        raise ValueError(f"Candidate state has non-finite {field}")
    return result


def _validate_state_identity(
    state: dict[str, Any], spec: dict[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    expected_equal = {
        "scope": SCOPE,
        "phase": "tune",
        "model": spec["model"],
        "seed": TUNING_SEED,
        "readout": spec["readout"],
        "config_sha256": spec["config_sha256"],
        "registry_sha256": spec["registry_sha256"],
    }
    for field, expected in expected_equal.items():
        if state.get(field) != expected:
            raise ValueError(
                f"Candidate state identity mismatch for {field}: "
                f"expected {expected!r}, got {state.get(field)!r}"
            )

    learning_rate = _coerce_float(state.get("learning_rate"), field="learning_rate")
    weight_decay = _coerce_float(state.get("weight_decay"), field="weight_decay")
    if learning_rate != float(spec["learning_rate"]):
        raise ValueError("Candidate state learning_rate mismatch")
    if weight_decay != float(spec["weight_decay"]):
        raise ValueError("Candidate state weight_decay mismatch")

    optimizer = state.get("optimizer_hyperparameters")
    if not isinstance(optimizer, dict):
        raise ValueError("Candidate state lacks optimizer_hyperparameters")
    if _coerce_float(optimizer.get("learning_rate"), field="optimizer learning_rate") != learning_rate:
        raise ValueError("Optimizer learning rate differs from candidate identity")
    if _coerce_float(optimizer.get("weight_decay"), field="optimizer weight_decay") != weight_decay:
        raise ValueError("Optimizer weight decay differs from candidate identity")

    if int(state.get("epochs_completed", -1)) != int(spec["expected_epochs"]):
        raise ValueError("Candidate did not complete the frozen epoch budget")
    expected_steps = spec.get("expected_optimizer_steps")
    completed_steps = int(state.get("optimizer_steps_completed", -1))
    if expected_steps is not None and completed_steps != int(expected_steps):
        raise ValueError("Candidate did not complete the frozen optimizer-step budget")
    if completed_steps <= 0:
        raise ValueError("Candidate optimizer-step count must be positive")

    best_val_loss = _coerce_float(state.get("best_val_loss"), field="best_val_loss")
    if best_val_loss < 0.0:
        raise ValueError("Candidate best_val_loss must be non-negative")
    checkpoint_sha256 = state.get("best_checkpoint_sha256")
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("Candidate state lacks a valid best_checkpoint_sha256")
    if not checkpoint_path.is_file():
        raise ValueError(f"Candidate best checkpoint is missing: {checkpoint_path}")
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("Candidate best checkpoint byte hash mismatch")

    train_hash = state.get("train_data_sha256")
    val_hash = state.get("val_data_sha256")
    if not _is_sha256(train_hash) or not _is_sha256(val_hash):
        raise ValueError("Candidate state lacks valid train/validation dataset hashes")
    return {
        "best_val_loss": best_val_loss,
        "best_checkpoint_sha256": str(checkpoint_sha256),
        "epochs_completed": int(state["epochs_completed"]),
        "optimizer_steps_completed": completed_steps,
        "train_data_sha256": str(train_hash),
        "val_data_sha256": str(val_hash),
    }


def _result_core(
    *,
    candidate: Candidate,
    spec: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    validated = _validate_state_identity(state, spec, candidate.directory / "best.pt")
    return {
        "schema": SCHEMA_VERSION,
        "scope": SCOPE,
        "phase": "tune",
        "formal_release_artifact": False,
        "candidate_id": candidate.candidate_id,
        "candidate_spec_sha256": spec["candidate_spec_sha256"],
        "model": candidate.model,
        "seed": TUNING_SEED,
        "readout": spec["readout"],
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "learning_rate_index": candidate.learning_rate_index,
        "weight_decay_index": candidate.weight_decay_index,
        "config_sha256": spec["config_sha256"],
        "registry_sha256": spec["registry_sha256"],
        "run_state_sha256": sha256_file(state_path),
        **validated,
    }


def _load_valid_result(candidate: Candidate, spec: dict[str, Any]) -> dict[str, Any] | None:
    result_path = candidate.directory / "candidate_result.json"
    if not result_path.is_file():
        return None
    result = _load_json(result_path)
    claimed = result.get("candidate_result_sha256")
    core = {
        key: value for key, value in result.items() if key != "candidate_result_sha256"
    }
    if not _is_sha256(claimed) or canonical_json_hash(core) != claimed:
        raise ValueError(f"Candidate completion receipt hash mismatch: {result_path}")
    state_path = candidate.directory / "run_state.json"
    if not state_path.is_file() or sha256_file(state_path) != result.get(
        "run_state_sha256"
    ):
        raise ValueError(f"Candidate run state changed after completion: {state_path}")
    expected = _result_core(
        candidate=candidate,
        spec=spec,
        state=_load_json(state_path),
        state_path=state_path,
    )
    if core != expected:
        raise ValueError(f"Candidate completion receipt identity mismatch: {result_path}")
    return result


def _parse_gpu_slots(value: str) -> tuple[str, ...]:
    slots = tuple(token.strip() for token in value.split(",") if token.strip())
    if not slots:
        raise ValueError("At least one GPU token is required")
    if len(set(slots)) != len(slots):
        raise ValueError("GPU tokens must be unique")
    cpu_tokens = [token for token in slots if token.lower() == "cpu"]
    if cpu_tokens and len(slots) != 1:
        raise ValueError("The CPU lane cannot be mixed with CUDA lanes")
    return slots


def _worker_command(
    *, candidate: Candidate, context: dict[str, Any], device: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "liquid_revision.reproduction.tuning",
        "_worker",
        "--spec",
        str(candidate.spec_path),
        "--config",
        str(context["config_path"]),
        "--data-root",
        str(context["data_root"]),
        "--registry",
        str(context["registry_path"]),
        "--output-dir",
        str(candidate.directory),
        "--device",
        device,
    ]


def _child_environment(gpu_slot: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(TUNING_SEED)
    source_root = str(Path(__file__).resolve().parents[2])
    prior_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not prior_python_path
        else source_root + os.pathsep + prior_python_path
    )
    if gpu_slot.lower() == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = gpu_slot
    return environment


def _run_worker(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    output_dir = Path(args.output_dir).resolve()
    if spec_path.parent != output_dir:
        raise ValueError("Worker spec must be inside its exclusive output directory")
    spec = _load_json(spec_path)
    spec_core = {
        key: value for key, value in spec.items() if key != "candidate_spec_sha256"
    }
    if canonical_json_hash(spec_core) != spec.get("candidate_spec_sha256"):
        raise ValueError("Worker candidate specification hash mismatch")

    config, config_sha256 = load_config(args.config)
    if config_sha256 != spec.get("config_sha256"):
        raise ValueError("Worker config differs from the immutable candidate spec")
    registry_sha256 = _load_registry_identity(args.registry, config)
    if registry_sha256 != spec.get("registry_sha256"):
        raise ValueError("Worker registry differs from the immutable candidate spec")

    candidate = Candidate(
        candidate_id=str(spec["candidate_id"]),
        model=str(spec["model"]),
        learning_rate=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
        learning_rate_index=int(spec["learning_rate_index"]),
        weight_decay_index=int(spec["weight_decay_index"]),
        directory=output_dir,
        spec_path=spec_path,
    )
    existing = _load_valid_result(candidate, spec)
    if existing is not None:
        return 0

    # Imported only inside the isolated worker so planning/status commands do
    # not initialize Torch, CUDA, MuJoCo, or dataset loaders.
    from liquid_revision.reproduction.training import fit_observer

    state = fit_observer(
        config=config,
        data_root=Path(args.data_root).resolve(),
        registry_path=Path(args.registry).resolve(),
        output_dir=output_dir,
        model=candidate.model,
        seed=TUNING_SEED,
        learning_rate=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
        phase="tune",
        readout=str(spec["readout"]),
        device=str(args.device),
    )
    if not isinstance(state, dict):
        raise TypeError("fit_observer must return a run-state mapping")
    state_path = output_dir / "run_state.json"
    if not state_path.is_file():
        raise ValueError("fit_observer returned without writing run_state.json")
    persisted_state = _load_json(state_path)
    if persisted_state != state:
        raise ValueError("Returned and persisted run states differ")
    core = _result_core(
        candidate=candidate, spec=spec, state=state, state_path=state_path
    )
    result = {**core, "candidate_result_sha256": canonical_json_hash(core)}
    _write_once_json(
        output_dir / "candidate_result.json",
        result,
        hash_field="candidate_result_sha256",
    )
    return 0


def _status_payload(
    plan: dict[str, Any], context: dict[str, Any], candidates: Sequence[Candidate]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for candidate in candidates:
        spec = _candidate_spec(candidate, context)
        if candidate.spec_path.is_file():
            existing_spec = _load_json(candidate.spec_path)
            if existing_spec != spec:
                raise ValueError(
                    f"Existing candidate spec conflicts with the plan: {candidate.spec_path}"
                )
            result = _load_valid_result(candidate, spec)
        else:
            result = None
        if result is None:
            incomplete.append(candidate.candidate_id)
        else:
            completed.append(result)
    payload = {
        "schema": SCHEMA_VERSION,
        "scope": SCOPE,
        "readout": context["readout"],
        "tuning_plan_sha256": plan["tuning_plan_sha256"],
        "candidate_count": len(candidates),
        "completed_count": len(completed),
        "incomplete_count": len(incomplete),
        "incomplete_candidate_ids": incomplete,
        "selection_path": str(Path(context["output_root"]) / "tuning_selection.json"),
    }
    return payload, completed


def _selection_from_results(
    *, plan: dict[str, Any], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected_count = int(plan["grid"]["candidate_count"])
    if len(results) != expected_count:
        raise ValueError(
            f"Selection requires all {expected_count} candidates; got {len(results)}"
        )
    config_hashes = {str(row["config_sha256"]) for row in results}
    registry_hashes = {str(row["registry_sha256"]) for row in results}
    train_hashes = {str(row["train_data_sha256"]) for row in results}
    val_hashes = {str(row["val_data_sha256"]) for row in results}
    if config_hashes != {str(plan["config_sha256"])}:
        raise ValueError("Candidate config identities disagree")
    if registry_hashes != {str(plan["registry_sha256"])}:
        raise ValueError("Candidate registry identities disagree")
    if len(train_hashes) != 1 or len(val_hashes) != 1:
        raise ValueError("Candidates did not use one shared train/validation dataset identity")

    by_model: dict[str, list[dict[str, Any]]] = {
        model: [] for model in plan["grid"]["models"]
    }
    for row in results:
        model = str(row["model"])
        if model not in by_model:
            raise ValueError(f"Unexpected model in tuning result: {model}")
        by_model[model].append(row)
    selected: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    for model in plan["grid"]["models"]:
        rows = by_model[model]
        if len(rows) != 4:
            raise ValueError(f"Model {model} does not have exactly four grid candidates")
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["best_val_loss"]),
                float(row["weight_decay"]),
                float(row["learning_rate"]),
            ),
        )
        winner = ordered[0]
        selected[model] = {
            "candidate_id": winner["candidate_id"],
            "learning_rate": winner["learning_rate"],
            "weight_decay": winner["weight_decay"],
            "best_val_loss": winner["best_val_loss"],
            "best_checkpoint_sha256": winner["best_checkpoint_sha256"],
            "run_state_sha256": winner["run_state_sha256"],
            "candidate_result_sha256": winner["candidate_result_sha256"],
        }
        candidate_rows.extend(
            {
                "candidate_id": row["candidate_id"],
                "model": row["model"],
                "learning_rate": row["learning_rate"],
                "weight_decay": row["weight_decay"],
                "learning_rate_index": row["learning_rate_index"],
                "weight_decay_index": row["weight_decay_index"],
                "best_val_loss": row["best_val_loss"],
                "best_checkpoint_sha256": row["best_checkpoint_sha256"],
                "run_state_sha256": row["run_state_sha256"],
                "candidate_result_sha256": row["candidate_result_sha256"],
            }
            for row in sorted(
                rows,
                key=lambda item: (
                    int(item["learning_rate_index"]),
                    int(item["weight_decay_index"]),
                ),
            )
        )

    core = {
        "schema": SELECTION_SCHEMA_VERSION,
        "scope": SCOPE,
        "phase": "tune",
        "formal_release_artifact": False,
        "complete": True,
        "seed": TUNING_SEED,
        "readout": plan["readout"],
        "config_sha256": plan["config_sha256"],
        "registry_sha256": plan["registry_sha256"],
        "data_sha256": {
            "train": next(iter(train_hashes)),
            "val": next(iter(val_hashes)),
        },
        "tuning_plan_sha256": plan["tuning_plan_sha256"],
        "grid": plan["grid"],
        "budget": plan["budget"],
        "selection_rule": (
            "minimum validation loss; ties by lower weight decay then lower "
            "learning rate"
        ),
        "selected": selected,
        "candidates": candidate_rows,
    }
    return {**core, "tuning_selection_sha256": canonical_json_hash(core)}


def _launch_candidates(
    *,
    context: dict[str, Any],
    candidates: Sequence[Candidate],
    gpu_slots: Sequence[str],
    workers_per_gpu: int,
    poll_seconds: float,
    fail_fast: bool,
) -> None:
    if workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu must be positive")
    if workers_per_gpu > 8:
        raise ValueError("workers_per_gpu cannot exceed the safety cap of 8")
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")

    pending = list(candidates)
    running: list[RunningCandidate] = []
    failures: list[tuple[str, int]] = []
    while pending or running:
        launched = False
        for gpu_slot in gpu_slots:
            occupied = sum(item.gpu_slot == gpu_slot for item in running)
            while occupied < workers_per_gpu and pending:
                candidate = pending.pop(0)
                candidate.directory.mkdir(parents=True, exist_ok=True)
                stdout_handle = (candidate.directory / "worker.stdout.log").open(
                    "a", encoding="utf-8", newline="\n"
                )
                stderr_handle = (candidate.directory / "worker.stderr.log").open(
                    "a", encoding="utf-8", newline="\n"
                )
                device = "cpu" if gpu_slot.lower() == "cpu" else "cuda:0"
                try:
                    process = subprocess.Popen(
                        _worker_command(
                            candidate=candidate, context=context, device=device
                        ),
                        env=_child_environment(gpu_slot),
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                    )
                except Exception:
                    stdout_handle.close()
                    stderr_handle.close()
                    raise
                running.append(
                    RunningCandidate(
                        candidate=candidate,
                        gpu_slot=gpu_slot,
                        process=process,
                        stdout_handle=stdout_handle,
                        stderr_handle=stderr_handle,
                    )
                )
                occupied += 1
                launched = True

        finished: list[RunningCandidate] = []
        for item in running:
            return_code = item.process.poll()
            if return_code is None:
                continue
            item.stdout_handle.close()
            item.stderr_handle.close()
            finished.append(item)
            if return_code != 0:
                failures.append((item.candidate.candidate_id, int(return_code)))
        for item in finished:
            running.remove(item)

        if failures and fail_fast:
            for item in running:
                item.process.terminate()
            for item in running:
                try:
                    item.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    item.process.kill()
                finally:
                    item.stdout_handle.close()
                    item.stderr_handle.close()
            break
        if running and not finished and not launched:
            time.sleep(poll_seconds)

    if failures:
        summary = ", ".join(f"{name} (exit {code})" for name, code in failures)
        raise RuntimeError(
            "One or more isolated tuning candidates failed; rerun the same command "
            f"to resume after inspection: {summary}"
        )


def _prepare_specs(
    context: dict[str, Any], candidates: Iterable[Candidate]
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    pending: list[Candidate] = []
    completed: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate.directory.mkdir(parents=True, exist_ok=True)
        spec = _candidate_spec(candidate, context)
        _write_once_json(
            candidate.spec_path, spec, hash_field="candidate_spec_sha256"
        )
        result = _load_valid_result(candidate, spec)
        if result is None:
            pending.append(candidate)
        else:
            completed.append(result)
    return pending, completed


def _run_supervisor(args: argparse.Namespace) -> int:
    plan, context, candidates = build_plan(
        config_path=args.config,
        data_root=args.data_root,
        registry_path=args.registry,
        output_root=args.output_root,
        readout=args.readout,
    )
    output_root = Path(context["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    _write_once_json(
        output_root / "tuning_plan.json", plan, hash_field="tuning_plan_sha256"
    )
    pending, _ = _prepare_specs(context, candidates)
    gpu_slots = _parse_gpu_slots(args.gpus)
    if pending:
        _launch_candidates(
            context=context,
            candidates=pending,
            gpu_slots=gpu_slots,
            workers_per_gpu=int(args.workers_per_gpu),
            poll_seconds=float(args.poll_seconds),
            fail_fast=bool(args.fail_fast),
        )

    status, results = _status_payload(plan, context, candidates)
    if status["incomplete_count"]:
        raise RuntimeError(
            f"Tuning ended with {status['incomplete_count']} incomplete candidates"
        )
    selection = _selection_from_results(plan=plan, results=results)
    _write_once_json(
        output_root / "tuning_selection.json",
        selection,
        hash_field="tuning_selection_sha256",
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _run_plan_or_status(args: argparse.Namespace, *, status_only: bool) -> int:
    plan, context, candidates = build_plan(
        config_path=args.config,
        data_root=args.data_root,
        registry_path=args.registry,
        output_root=args.output_root,
        readout=args.readout,
    )
    if status_only:
        payload, _ = _status_payload(plan, context, candidates)
    else:
        payload = plan
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--readout", choices=("unit", "scaled"), default="unit")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m liquid_revision.reproduction.tuning",
        description=(
            "Run the frozen seed-17 validation-only grid in isolated, resumable "
            "candidate processes. This command does not create formal-release evidence."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan", help="validate inputs and print the immutable grid without writing"
    )
    _add_common_arguments(plan_parser)

    status_parser = subparsers.add_parser(
        "status", help="validate and report existing candidate completion state"
    )
    _add_common_arguments(status_parser)

    run_parser = subparsers.add_parser(
        "run", help="run or resume all candidates and emit validation-only selection"
    )
    _add_common_arguments(run_parser)
    run_parser.add_argument(
        "--gpus",
        default="0,1",
        help=(
            "comma-separated physical CUDA indices/UUIDs mapped one per worker lane; "
            "use 'cpu' for a single CPU-only verification lane"
        ),
    )
    run_parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="maximum simultaneous isolated candidates per GPU (default: 1)",
    )
    run_parser.add_argument("--poll-seconds", type=float, default=0.25)
    run_parser.add_argument("--fail-fast", action="store_true")

    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--spec", required=True, type=Path)
    worker_parser.add_argument("--config", required=True, type=Path)
    worker_parser.add_argument("--data-root", required=True, type=Path)
    worker_parser.add_argument("--registry", required=True, type=Path)
    worker_parser.add_argument("--output-dir", required=True, type=Path)
    worker_parser.add_argument("--device", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "plan":
        return _run_plan_or_status(args, status_only=False)
    if args.command == "status":
        return _run_plan_or_status(args, status_only=True)
    if args.command == "run":
        return _run_supervisor(args)
    if args.command == "_worker":
        return _run_worker(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
