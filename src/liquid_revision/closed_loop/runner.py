"""Detached, fail-closed orchestration and the formal model/seed run ledger.

This module intentionally is not registered in the general CLI.  The detached
PowerShell launcher invokes it explicitly, making test evaluation opt-in and
requiring two independently supplied release anchors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..config import load_config
from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file
from .formal import (
    validate_checkpoint_release,
    validate_controller_selection,
    validate_environment_freeze,
    validate_pretraining_master_release,
    validate_test_release,
)
from .manifests import load_trajectory_manifest
from .preflight import load_preflight_receipt, validate_preflight_receipt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_FAILURE_CODES = {
    "nonfinite_loss",
    "nonfinite_gradient",
    "cuda_out_of_memory",
    "determinism_violation",
    "budget_gate",
    "model_exception",
}
INFRASTRUCTURE_FAILURE_CODES = {
    "cuda_driver",
    "gpu_unavailable",
    "filesystem_io",
    "host_preemption",
    "process_launch",
}

FULL_GPU_UUID_PATTERN = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
TRAINING_GPU_PLAN_ENV_PATH = "LIQUID_FORMAL_TRAINING_GPU_PLAN_PATH"
TRAINING_GPU_PLAN_ENV_FILE_SHA256 = (
    "LIQUID_FORMAL_TRAINING_GPU_PLAN_FILE_SHA256"
)


def _canonical_gpu_uuid(value: Any) -> str:
    text = str(value).strip()
    if FULL_GPU_UUID_PATTERN.fullmatch(text) is None:
        raise ValueError("Training GPU assignment requires a complete GPU UUID")
    return "GPU-" + text[4:].lower()


def parse_training_gpu_assignments(
    values: Iterable[str] | None, *, seeds: Iterable[int]
) -> dict[str, str]:
    """Parse a complete runtime-only seed=GPU-UUID assignment."""
    seed_values = [int(seed) for seed in seeds]
    expected_seeds = set(seed_values)
    if not seed_values or len(expected_seeds) != len(seed_values):
        raise ValueError("Registered training seeds must be non-empty and unique")
    assignments: dict[int, str] = {}
    for raw in values or []:
        if "=" not in str(raw):
            raise ValueError("Training GPU assignment must be SEED=GPU-UUID")
        raw_seed, raw_uuid = str(raw).split("=", 1)
        if re.fullmatch(r"0|[1-9][0-9]*", raw_seed.strip()) is None:
            raise ValueError("Training GPU assignment seed is not canonical")
        seed = int(raw_seed)
        if seed in assignments:
            raise ValueError(f"Duplicate training GPU assignment for seed {seed}")
        assignments[seed] = _canonical_gpu_uuid(raw_uuid)
    if set(assignments) != expected_seeds:
        raise ValueError(
            "Training GPU assignments must cover the exact registered seed set"
        )
    unique_uuids = set(assignments.values())
    if len(unique_uuids) != 2:
        raise ValueError("Formal dual-GPU training requires exactly two GPU UUIDs")
    return {
        str(seed): assignments[seed]
        for seed in sorted(assignments)
    }


def thermal_dispatch_gate(
    gpu_temperatures_c: Mapping[str, float],
    *,
    was_paused: bool,
    stop_c: float = 82.0,
    resume_c: float = 70.0,
) -> tuple[bool, bool]:
    """Return (dispatch_allowed, paused) using a global hysteresis gate.

    This gate never terminates a worker or changes a device setting.  It only
    controls whether the parent scheduler may launch another child.
    """
    stop = float(stop_c)
    resume = float(resume_c)
    if not math.isfinite(stop) or not math.isfinite(resume) or resume >= stop:
        raise ValueError("Thermal guard requires finite resume_c < stop_c")
    if not gpu_temperatures_c:
        raise ValueError("Thermal admission requires every assigned GPU temperature")
    temperatures = [float(value) for value in gpu_temperatures_c.values()]
    if any(not math.isfinite(value) for value in temperatures):
        raise ValueError("GPU temperature sample must be finite")
    if was_paused:
        paused = max(temperatures) > resume
    else:
        paused = max(temperatures) >= stop
    return (not paused, paused)


def _training_gpu_plan_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _without(
        payload,
        "training_gpu_plan_sha256",
        "created_at_utc",
        "_validated_path",
        "_validated_file_sha256",
    )


def load_training_gpu_plan(
    path: str | Path, *, expected_file_sha256: str | None = None
) -> dict[str, Any]:
    source = Path(path).resolve()
    if expected_file_sha256 is not None:
        if len(str(expected_file_sha256)) != 64:
            raise PermissionError("Training GPU plan file anchor must be SHA-256")
        int(str(expected_file_sha256), 16)
        if sha256_file(source) != str(expected_file_sha256):
            raise PermissionError("Training GPU plan differs from its file anchor")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("phase") != "pre_confirmatory_training_gpu_plan"
        or canonical_json_hash(_training_gpu_plan_core(payload))
        != payload.get("training_gpu_plan_sha256")
    ):
        raise ValueError(f"Training GPU plan hash/schema mismatch: {source}")
    mapping = payload.get("training_gpu_by_seed")
    seeds = payload.get("registered_seeds")
    if not isinstance(mapping, dict) or not isinstance(seeds, list):
        raise ValueError("Training GPU plan has an invalid assignment schema")
    if (
        not seeds
        or any(type(seed) is not int for seed in seeds)
        or seeds != sorted(seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("Training GPU plan registered seeds are not canonical")
    canonical_mapping = parse_training_gpu_assignments(
        [f"{seed}={value}" for seed, value in mapping.items()],
        seeds=[int(seed) for seed in seeds],
    )
    if mapping != canonical_mapping:
        raise ValueError("Training GPU plan assignment is not canonical")
    if int(payload.get("max_parallel_training_runs", -1)) != 2:
        raise ValueError("Formal dual-GPU scheduler requires max parallel runs = 2")
    thermal = payload.get("thermal_guard")
    if not isinstance(thermal, dict) or set(thermal) != {"stop_c", "resume_c"}:
        raise ValueError("Training GPU plan thermal guard schema is invalid")
    thermal_dispatch_gate(
        {uuid_value: 0.0 for uuid_value in set(mapping.values())},
        was_paused=False,
        stop_c=float(thermal["stop_c"]),
        resume_c=float(thermal["resume_c"]),
    )
    payload["_validated_path"] = str(source)
    payload["_validated_file_sha256"] = sha256_file(source)
    return payload


def create_training_gpu_plan(
    path: str | Path,
    *,
    config_sha256: str,
    preflight_receipt_sha256: str,
    run_registry_path: str | Path,
    models: Iterable[str],
    seeds: Iterable[int],
    assignments: Mapping[str, str],
    max_parallel_training_runs: int,
    thermal_stop_c: float,
    thermal_resume_c: float,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Create or validate the write-once runtime GPU scheduling plan."""
    destination = Path(path).resolve()
    seed_values = sorted(int(seed) for seed in seeds)
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("Training GPU plan requires unique registered seeds")
    canonical_assignments = parse_training_gpu_assignments(
        [f"{seed}={value}" for seed, value in assignments.items()],
        seeds=seed_values,
    )
    if int(max_parallel_training_runs) != 2:
        raise ValueError("Formal dual-GPU scheduler requires max parallel runs = 2")
    inventory = environment.get("physical_gpu_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Frozen environment lacks a physical GPU inventory")
    inventory_by_uuid = {
        _canonical_gpu_uuid(row.get("uuid")): row for row in inventory
        if isinstance(row, Mapping)
    }
    assigned = set(canonical_assignments.values())
    if not assigned.issubset(set(inventory_by_uuid)):
        raise ValueError("Training GPU assignment is outside the frozen inventory")
    thermal_dispatch_gate(
        {uuid_value: 0.0 for uuid_value in assigned},
        was_paused=False,
        stop_c=thermal_stop_c,
        resume_c=thermal_resume_c,
    )
    expected_identities = sorted(
        f"{model}::seed_{seed}" for model in models for seed in seed_values
    )
    if not expected_identities or len(expected_identities) != len(
        set(expected_identities)
    ):
        raise ValueError("Training GPU plan requires unique model/seed identities")
    expected = {
        "schema_version": 1,
        "phase": "pre_confirmatory_training_gpu_plan",
        "source_config_sha256": str(config_sha256),
        "preflight_receipt_sha256": str(preflight_receipt_sha256),
        "run_registry_path": str(Path(run_registry_path).resolve()),
        "registered_seeds": seed_values,
        "expected_run_identities": expected_identities,
        "training_gpu_by_seed": canonical_assignments,
        "assigned_gpu_inventory": [
            dict(inventory_by_uuid[gpu_uuid]) for gpu_uuid in sorted(assigned)
        ],
        "physical_gpu_inventory_sha256": canonical_json_hash(inventory),
        "environment_sha256": environment.get("environment_sha256"),
        "max_parallel_training_runs": 2,
        "thermal_guard": {
            "stop_c": float(thermal_stop_c),
            "resume_c": float(thermal_resume_c),
        },
        "test_episodes_used": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(destination):
        if destination.exists():
            existing = load_training_gpu_plan(destination)
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise ValueError(f"Existing training GPU plan differs for {key}")
            return existing
        core = {**expected, "plan_nonce": uuid.uuid4().hex}
        payload = {
            **core,
            "training_gpu_plan_sha256": canonical_json_hash(core),
            "created_at_utc": _utc_now(),
        }
        atomic_write_json(destination, payload)
        return load_training_gpu_plan(destination)


def _training_gpu_plan_from_environment() -> dict[str, Any] | None:
    raw_path = os.environ.get(TRAINING_GPU_PLAN_ENV_PATH)
    raw_hash = os.environ.get(TRAINING_GPU_PLAN_ENV_FILE_SHA256)
    if raw_path is None and raw_hash is None:
        return None
    if not raw_path or not raw_hash:
        raise PermissionError("Training GPU plan environment binding is incomplete")
    return load_training_gpu_plan(raw_path, expected_file_sha256=raw_hash)


def _training_gpu_namespace_fields(
    plan: Mapping[str, Any] | None, *, plan_path: str | Path | None = None
) -> dict[str, Any]:
    if plan is None:
        return {}
    source = Path(plan_path or plan.get("_validated_path", "")).resolve()
    if not source.is_file():
        raise FileNotFoundError("Training GPU plan path is absent")
    return {
        "training_gpu_plan_path": str(source),
        "training_gpu_plan_sha256": plan["training_gpu_plan_sha256"],
        "training_gpu_plan_file_sha256": sha256_file(source),
        "training_gpu_by_seed": dict(plan["training_gpu_by_seed"]),
        "max_parallel_training_runs": int(plan["max_parallel_training_runs"]),
        "thermal_guard": dict(plan["thermal_guard"]),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_is_alive(process_id: int) -> bool:
    if process_id <= 0:
        return False
    try:
        os.kill(process_id, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        (PROJECT_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _without(payload: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    excluded = set(keys)
    return {key: value for key, value in payload.items() if key not in excluded}


def _registry_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_hash(_without(payload, "registry_sha256", "updated_at_utc"))


def _state_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_hash(
        _without(payload, "pipeline_state_sha256", "updated_at_utc")
    )


def _journal_entry_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _without(payload, "journal_entry_sha256", "created_at_utc")


def _verify_append_only_custody(
    directory: str | Path, *, run_registry_instance_id: str
) -> dict[str, Any]:
    """Fail closed unless the worker can create, but cannot mutate/delete, files.

    The directory must be provisioned by the independent auditor.  This probe is
    deliberately operational rather than declarative: a normal writable folder
    fails.  The immutable probe is retained as evidence in the external journal.
    An administrator who can replace the ACL is outside this local process's
    threat boundary and therefore still requires auditor custody/monitoring.
    """

    journal_directory = Path(directory).resolve()
    if not journal_directory.is_dir():
        raise FileNotFoundError(
            "Auditor-managed append-only journal directory is not provisioned"
        )
    probe_path = journal_directory / (
        f"{run_registry_instance_id}.acl_probe_{uuid.uuid4().hex}.json"
    )
    probe_core = {
        "schema_version": 1,
        "purpose": "verify_create_only_no_modify_no_delete_acl",
        "run_registry_instance_id": str(run_registry_instance_id),
        "probe_nonce": uuid.uuid4().hex,
    }
    probe = {
        **probe_core,
        "probe_sha256": canonical_json_hash(probe_core),
        "created_at_utc": _utc_now(),
    }
    encoded = (
        json.dumps(probe, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(probe_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())

    mutation_blocked = False
    try:
        descriptor = os.open(probe_path, os.O_WRONLY | os.O_APPEND)
    except PermissionError:
        mutation_blocked = True
    else:
        os.close(descriptor)

    deletion_blocked = False
    try:
        probe_path.unlink()
    except PermissionError:
        deletion_blocked = True

    if not mutation_blocked or not deletion_blocked:
        raise PermissionError(
            "Run-registry journal custody is not append-only: the training "
            "worker can modify or delete an existing auditor-managed object"
        )
    return {
        "journal_directory": str(journal_directory),
        "acl_policy": "create_only_no_modify_no_delete_existing_objects",
        "probe_path": str(probe_path),
        "probe_file_sha256": sha256_file(probe_path),
        "probe_sha256": probe["probe_sha256"],
        "mutation_blocked": True,
        "deletion_blocked": True,
        "verified_at_utc": _utc_now(),
    }


def _append_registry_journal_entry(
    *,
    registry_path: Path,
    payload: Mapping[str, Any],
    previous_registry_sha256: str,
) -> dict[str, Any]:
    custody = payload.get("run_registry_custody")
    if not isinstance(custody, Mapping) or custody.get("mode") != (
        "auditor_managed_append_only"
    ):
        raise PermissionError(
            "Formal registry mutation lacks auditor-managed append-only custody"
        )
    journal_directory = Path(str(custody.get("journal_directory", ""))).resolve()
    if not journal_directory.is_dir():
        raise FileNotFoundError("Run-registry external journal directory is absent")
    revision = int(payload.get("revision", -1))
    instance_id = str(payload.get("run_registry_instance_id", ""))
    if revision < 1 or not instance_id:
        raise ValueError("Run-registry journal entry lacks revision/instance")
    destination = journal_directory / (
        f"{instance_id}.revision_{revision:06d}.json"
    )
    core = {
        "schema_version": 1,
        "run_registry_instance_id": instance_id,
        "run_registry_namespace_sha256": payload[
            "run_registry_namespace_sha256"
        ],
        "run_registry_path": str(registry_path.resolve()),
        "revision": revision,
        "previous_registry_sha256": str(previous_registry_sha256),
        "registry_sha256": str(payload["registry_sha256"]),
        "registry_status": str(payload.get("status")),
        "slots_sha256": canonical_json_hash(payload.get("slots", {})),
        "transition_event": payload.get("last_transition_event"),
        "pretraining_master_sha256": payload.get(
            "pretraining_master_sha256"
        ),
        "pretraining_audit_receipt_sha256": payload.get(
            "pretraining_audit_receipt_sha256"
        ),
    }
    entry = {
        **core,
        "journal_entry_sha256": canonical_json_hash(core),
        "created_at_utc": _utc_now(),
    }
    encoded = (
        json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError(
            f"Run-registry journal revision already exists: {destination}"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "revision": revision,
        "path": str(destination),
        "journal_entry_sha256": entry["journal_entry_sha256"],
        "file_sha256": sha256_file(destination),
        "registry_sha256": payload["registry_sha256"],
    }


def _validate_registry_journal(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    custody = payload.get("run_registry_custody")
    if not isinstance(custody, Mapping) or custody.get("mode") != (
        "auditor_managed_append_only"
    ):
        return None
    journal_directory = Path(str(custody.get("journal_directory", ""))).resolve()
    instance_id = str(payload.get("run_registry_instance_id", ""))
    revision = int(payload.get("revision", -1))
    initial = str(payload.get("initial_registry_canonical_sha256", ""))
    if not journal_directory.is_dir() or len(initial) != 64 or revision < 1:
        raise ValueError("Activated registry lacks its external journal identity")
    paths = sorted(journal_directory.glob(f"{instance_id}.revision_*.json"))
    if len(paths) != revision:
        raise ValueError(
            "Run-registry external journal has a missing or extra revision"
        )
    previous = initial
    identities: list[dict[str, Any]] = []
    for expected_revision, path in enumerate(paths, start=1):
        expected_name = f"{instance_id}.revision_{expected_revision:06d}.json"
        if path.name != expected_name:
            raise ValueError("Run-registry external journal has a revision gap")
        entry = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json_hash(_journal_entry_core(entry)) != entry.get(
            "journal_entry_sha256"
        ):
            raise ValueError("Run-registry external journal entry hash mismatch")
        expected = {
            "run_registry_instance_id": instance_id,
            "run_registry_namespace_sha256": payload[
                "run_registry_namespace_sha256"
            ],
            "run_registry_path": str(Path(payload["run_registry_path"]).resolve())
            if payload.get("run_registry_path")
            else None,
            "revision": expected_revision,
            "previous_registry_sha256": previous,
        }
        for key, value in expected.items():
            if value is not None and entry.get(key) != value:
                raise ValueError(
                    f"Run-registry external journal mismatch for {key}"
                )
        previous = str(entry.get("registry_sha256", ""))
        identities.append(
            {
                "revision": expected_revision,
                "journal_entry_sha256": entry["journal_entry_sha256"],
                "file_sha256": sha256_file(path),
                "registry_sha256": previous,
            }
        )
    if previous != payload.get("registry_sha256"):
        raise ValueError("Run-registry external journal head is stale or divergent")
    return {
        "journal_directory": str(journal_directory),
        "entry_count": revision,
        "journal_head_registry_sha256": previous,
        "journal_entries_sha256": canonical_json_hash(identities),
        "entries": identities,
    }


def _run_namespace_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _without(payload, "run_registry_namespace_sha256", "created_at_utc")


def create_run_registry_namespace(
    path: str | Path,
    *,
    run_registry_path: str | Path,
    config_sha256: str,
    preflight_receipt_sha256: str,
    models: Iterable[str],
    seeds: Iterable[int],
    training_gpu_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the write-once registry namespace frozen before training.

    The mutable attempt ledger is intentionally created only after the
    pretraining release is externally anchored.  This artifact commits that
    release to one exact ledger path, one random namespace, and the complete
    model/seed slot set so a second ledger cannot later be substituted without
    changing the externally published pretraining root.
    """
    destination = Path(path).resolve()
    identities = sorted(
        f"{model}::seed_{int(seed)}" for model in models for seed in seeds
    )
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Formal run-registry namespace has invalid identities")
    plan = dict(training_gpu_plan) if training_gpu_plan is not None else (
        _training_gpu_plan_from_environment()
    )
    if plan is not None:
        plan_path = plan.get("_validated_path")
        if not plan_path:
            raise ValueError("Training GPU plan lacks its validated source path")
        plan = load_training_gpu_plan(plan_path)
        if (
            plan.get("source_config_sha256") != str(config_sha256)
            or plan.get("preflight_receipt_sha256")
            != str(preflight_receipt_sha256)
            or plan.get("run_registry_path")
            != str(Path(run_registry_path).resolve())
            or plan.get("expected_run_identities") != identities
        ):
            raise ValueError("Training GPU plan differs from the run namespace")
    expected = {
        "schema_version": 1,
        "phase": "pre_confirmatory_run_namespace",
        "source_config_sha256": str(config_sha256),
        "preflight_receipt_sha256": str(preflight_receipt_sha256),
        "run_registry_path": str(Path(run_registry_path).resolve()),
        "expected_run_identities": identities,
        "test_episodes_used": 0,
        **_training_gpu_namespace_fields(
            plan, plan_path=(plan.get("_validated_path") if plan else None)
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(destination):
        if destination.exists():
            payload = validate_run_registry_namespace(
                destination,
                run_registry_path=run_registry_path,
                config_sha256=config_sha256,
                preflight_receipt_sha256=preflight_receipt_sha256,
                models=models,
                seeds=seeds,
                training_gpu_plan=plan,
            )
            return payload
        if Path(run_registry_path).resolve().exists():
            raise FileExistsError(
                "Run registry exists before its namespace was frozen"
            )
        core = {
            **expected,
            "namespace_nonce": uuid.uuid4().hex,
        }
        payload = {
            **core,
            "run_registry_namespace_sha256": canonical_json_hash(core),
            "created_at_utc": _utc_now(),
        }
        atomic_write_json(destination, payload)
        return payload


def validate_run_registry_namespace(
    path: str | Path,
    *,
    run_registry_path: str | Path,
    config_sha256: str,
    preflight_receipt_sha256: str,
    models: Iterable[str],
    seeds: Iterable[int],
    expected_namespace_sha256: str | None = None,
    training_gpu_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if canonical_json_hash(_run_namespace_core(payload)) != payload.get(
        "run_registry_namespace_sha256"
    ):
        raise ValueError(f"Run-registry namespace hash mismatch: {source}")
    identities = sorted(
        f"{model}::seed_{int(seed)}" for model in models for seed in seeds
    )
    expected = {
        "schema_version": 1,
        "phase": "pre_confirmatory_run_namespace",
        "source_config_sha256": str(config_sha256),
        "preflight_receipt_sha256": str(preflight_receipt_sha256),
        "run_registry_path": str(Path(run_registry_path).resolve()),
        "expected_run_identities": identities,
        "test_episodes_used": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Run-registry namespace mismatch for {key}")
    if not str(payload.get("namespace_nonce", "")).strip():
        raise ValueError("Run-registry namespace lacks its unique nonce")
    plan_fields = {
        "training_gpu_plan_path",
        "training_gpu_plan_sha256",
        "training_gpu_plan_file_sha256",
        "training_gpu_by_seed",
        "max_parallel_training_runs",
        "thermal_guard",
    }
    present_plan_fields = plan_fields.intersection(payload)
    if present_plan_fields and present_plan_fields != plan_fields:
        raise ValueError("Run-registry namespace has an incomplete GPU-plan binding")
    bound_plan: dict[str, Any] | None = None
    if present_plan_fields:
        bound_plan = load_training_gpu_plan(
            payload["training_gpu_plan_path"],
            expected_file_sha256=payload["training_gpu_plan_file_sha256"],
        )
        bound_fields = _training_gpu_namespace_fields(
            bound_plan, plan_path=payload["training_gpu_plan_path"]
        )
        for key, value in bound_fields.items():
            if payload.get(key) != value:
                raise ValueError(f"Run-registry GPU-plan mismatch for {key}")
        if (
            bound_plan.get("source_config_sha256") != str(config_sha256)
            or bound_plan.get("preflight_receipt_sha256")
            != str(preflight_receipt_sha256)
            or bound_plan.get("run_registry_path")
            != str(Path(run_registry_path).resolve())
            or bound_plan.get("expected_run_identities") != identities
        ):
            raise ValueError("Run-registry GPU plan differs from namespace inputs")
    if training_gpu_plan is not None:
        supplied_path = training_gpu_plan.get("_validated_path")
        if not supplied_path:
            raise ValueError("Supplied training GPU plan lacks a validated path")
        supplied = load_training_gpu_plan(supplied_path)
        if bound_plan is None or (
            supplied["training_gpu_plan_sha256"]
            != bound_plan["training_gpu_plan_sha256"]
            or supplied["_validated_file_sha256"]
            != bound_plan["_validated_file_sha256"]
        ):
            raise ValueError("Run-registry namespace differs from the supplied GPU plan")
    if (
        expected_namespace_sha256 is not None
        and payload["run_registry_namespace_sha256"]
        != str(expected_namespace_sha256)
    ):
        raise PermissionError(
            "Run-registry namespace differs from the externally anchored release"
        )
    return payload


def _namespace_gpu_bindings(namespace: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "training_gpu_plan_path",
        "training_gpu_plan_sha256",
        "training_gpu_plan_file_sha256",
        "training_gpu_by_seed",
        "max_parallel_training_runs",
        "thermal_guard",
    )
    present = [key for key in keys if key in namespace]
    if present and len(present) != len(keys):
        raise ValueError("Run namespace has a partial training GPU binding")
    return {key: namespace[key] for key in keys} if present else {}


@contextmanager
def _exclusive_lock(target: Path, timeout_seconds: float = 30.0):
    lock = target.with_name(f".{target.name}.lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()} utc={_utc_now()}\n".encode("utf-8"))
            os.fsync(descriptor)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for ledger lock: {lock}")
            time.sleep(0.1)
    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def _load_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or _registry_hash(payload) != payload.get(
        "registry_sha256"
    ):
        raise ValueError(f"Formal run registry hash/schema mismatch: {source}")
    expected = payload.get("expected_run_identities")
    if not isinstance(expected, list) or len(expected) != len(set(expected)):
        raise ValueError("Formal run registry has an invalid expected identity set")
    slots = payload.get("slots")
    if not isinstance(slots, dict) or not set(slots).issubset(set(expected)):
        raise ValueError("Formal run registry contains an unexpected slot")
    if not str(payload.get("run_registry_namespace_path", "")).strip() or len(
        str(payload.get("run_registry_namespace_sha256", ""))
    ) != 64:
        raise ValueError("Formal run registry lacks its frozen namespace binding")
    if not str(payload.get("run_registry_instance_id", "")).strip() or len(
        str(payload.get("initial_registry_sha256", ""))
    ) != 64:
        raise ValueError("Formal run registry lacks its frozen initial instance")
    gpu_keys = {
        "training_gpu_plan_path",
        "training_gpu_plan_sha256",
        "training_gpu_plan_file_sha256",
        "training_gpu_by_seed",
        "max_parallel_training_runs",
        "thermal_guard",
    }
    present_gpu_keys = gpu_keys.intersection(payload)
    if present_gpu_keys and present_gpu_keys != gpu_keys:
        raise ValueError("Formal run registry has a partial GPU-plan binding")
    if present_gpu_keys:
        plan = load_training_gpu_plan(
            payload["training_gpu_plan_path"],
            expected_file_sha256=payload["training_gpu_plan_file_sha256"],
        )
        expected_bindings = _training_gpu_namespace_fields(
            plan, plan_path=payload["training_gpu_plan_path"]
        )
        for key, value in expected_bindings.items():
            if payload.get(key) != value:
                raise ValueError(f"Formal run registry GPU-plan mismatch for {key}")
        paired_by_seed: dict[int, str] = {}
        for attempts in slots.values():
            if not isinstance(attempts, list):
                raise ValueError("Formal run registry slot attempts must be a list")
            for attempt in attempts:
                seed = int(attempt.get("seed", -1))
                assigned = _canonical_gpu_uuid(attempt.get("assigned_gpu_uuid"))
                if payload["training_gpu_by_seed"].get(str(seed)) != assigned:
                    raise ValueError("Run attempt differs from frozen seed/GPU assignment")
                expected_assignment_sha256 = canonical_json_hash(
                    {
                        "source_config_sha256": payload["source_config_sha256"],
                        "run_registry_namespace_sha256": payload[
                            "run_registry_namespace_sha256"
                        ],
                        "seed": seed,
                        "assigned_gpu_uuid": assigned,
                    }
                )
                if (
                    attempt.get("training_gpu_assignment_sha256")
                    != expected_assignment_sha256
                ):
                    raise ValueError("Run attempt GPU-assignment hash mismatch")
                prior = paired_by_seed.setdefault(seed, assigned)
                if prior != assigned:
                    raise ValueError("Paired model runs for one seed use different GPUs")
    if payload.get("status") in {"active", "complete_frozen"}:
        _validate_registry_journal(payload)
    return payload


def _save_registry(path: Path, payload: dict[str, Any]) -> None:
    previous_registry_sha256: str | None = None
    previous_revision: int | None = None
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if _registry_hash(previous) != previous.get("registry_sha256"):
            raise ValueError("Refusing to replace a corrupt run registry")
        previous_registry_sha256 = str(previous["registry_sha256"])
        previous_revision = int(previous.get("revision", -1))
    payload["updated_at_utc"] = _utc_now()
    payload["registry_sha256"] = _registry_hash(payload)
    custody = payload.get("run_registry_custody")
    if isinstance(custody, Mapping) and custody.get("mode") == (
        "auditor_managed_append_only"
    ):
        if previous_registry_sha256 is None or previous_revision is None:
            raise ValueError("Audited registry mutation lacks a prior registry root")
        if int(payload.get("revision", -1)) != previous_revision + 1:
            raise ValueError("Audited registry revision is not strictly monotonic")
        _append_registry_journal_entry(
            registry_path=path,
            payload=payload,
            previous_registry_sha256=previous_registry_sha256,
        )
    atomic_write_json(path, payload)
    if isinstance(custody, Mapping) and custody.get("mode") == (
        "auditor_managed_append_only"
    ):
        _validate_registry_journal(payload)


def create_pretraining_run_registry(
    path: str | Path,
    *,
    namespace_path: str | Path,
    expected_namespace_sha256: str,
    config_sha256: str,
    preflight_receipt_sha256: str,
    models: Iterable[str],
    seeds: Iterable[int],
) -> dict[str, Any]:
    """Create the exact empty registry instance committed before training."""
    destination = Path(path).resolve()
    namespace = validate_run_registry_namespace(
        namespace_path,
        run_registry_path=destination,
        config_sha256=config_sha256,
        preflight_receipt_sha256=preflight_receipt_sha256,
        models=models,
        seeds=seeds,
        expected_namespace_sha256=expected_namespace_sha256,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(destination):
        if destination.exists():
            existing = _load_registry(destination)
            expected = {
                "status": "pretraining_frozen",
                "run_registry_namespace_path": str(Path(namespace_path).resolve()),
                "run_registry_path": str(destination),
                "run_registry_namespace_sha256": namespace[
                    "run_registry_namespace_sha256"
                ],
                "source_config_sha256": str(config_sha256),
                "preflight_receipt_sha256": str(preflight_receipt_sha256),
                "expected_run_identities": namespace[
                    "expected_run_identities"
                ],
                "slots": {},
                "revision": 0,
                **_namespace_gpu_bindings(namespace),
            }
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise ValueError(
                        f"Existing pretraining run registry differs for {key}"
                    )
            return existing
        initial_core = {
            "schema_version": 1,
            "status": "pretraining_frozen",
            "run_registry_instance_id": uuid.uuid4().hex,
            "run_registry_namespace_path": str(Path(namespace_path).resolve()),
            "run_registry_path": str(destination),
            "run_registry_namespace_sha256": namespace[
                "run_registry_namespace_sha256"
            ],
            "source_config_sha256": str(config_sha256),
            "preflight_receipt_sha256": str(preflight_receipt_sha256),
            "expected_run_identities": namespace["expected_run_identities"],
            "slots": {},
            "revision": 0,
            **_namespace_gpu_bindings(namespace),
        }
        payload = {
            **initial_core,
            "initial_registry_sha256": canonical_json_hash(initial_core),
            "created_at_utc": _utc_now(),
        }
        _save_registry(destination, payload)
        return payload


def activate_pretraining_run_registry(
    path: str | Path,
    *,
    pretraining_master: Mapping[str, Any],
    pretraining_audit_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Activate only the externally anchored, auditor-custodied instance."""
    destination = Path(path).resolve()
    with _exclusive_lock(destination):
        registry = _load_registry(destination)
        expected = {
            "run_registry_instance_id": pretraining_master[
                "run_registry_instance_id"
            ],
            "initial_registry_sha256": pretraining_master[
                "initial_registry_sha256"
            ],
            "run_registry_namespace_sha256": pretraining_master[
                "run_registry_namespace_sha256"
            ],
            "source_config_sha256": pretraining_master[
                "source_config_sha256"
            ],
            "preflight_receipt_sha256": pretraining_master[
                "preflight_receipt_sha256"
            ],
            "expected_run_identities": pretraining_master[
                "expected_run_identities"
            ],
        }
        for key, value in expected.items():
            if registry.get(key) != value:
                raise ValueError(f"Pretraining run registry differs for {key}")
        custody = pretraining_audit_receipt.get("run_registry_custody")
        if (
            not isinstance(custody, Mapping)
            or custody.get("mode") != "auditor_managed_append_only"
            or custody.get("run_registry_instance_id")
            != registry["run_registry_instance_id"]
            or custody.get("initial_registry_sha256")
            != registry["initial_registry_sha256"]
            or custody.get("acl_policy")
            != "create_only_no_modify_no_delete_existing_objects"
            or not Path(str(custody.get("journal_directory", ""))).is_absolute()
            or not str(custody.get("external_attestation_id", "")).strip()
        ):
            raise PermissionError(
                "Formal registry activation requires an externally anchored "
                "auditor-managed append-only custody attestation"
            )
        if registry.get("status") == "pretraining_frozen":
            custody_evidence = _verify_append_only_custody(
                custody["journal_directory"],
                run_registry_instance_id=registry["run_registry_instance_id"],
            )
            registry.update(
                {
                    "status": "active",
                    "pretraining_master_sha256": pretraining_master[
                        "pretraining_master_sha256"
                    ],
                    "pretraining_audit_receipt_sha256": pretraining_audit_receipt[
                        "pretraining_audit_receipt_sha256"
                    ],
                    "pretraining_audit_receipt_file_sha256": pretraining_audit_receipt[
                        "_validated_file_sha256"
                    ],
                    "run_registry_custody": dict(custody),
                    "run_registry_custody_runtime_evidence": custody_evidence,
                    "initial_registry_canonical_sha256": pretraining_master[
                        "initial_registry_canonical_sha256"
                    ],
                    "activated_at_utc": _utc_now(),
                    "last_transition_event": {
                        "event": "activate_registry",
                        "run_identity": None,
                        "attempt_id": None,
                    },
                    "revision": 1,
                }
            )
            _save_registry(destination, registry)
        elif registry.get("status") not in {"active", "complete_frozen"}:
            raise ValueError("Run registry has an invalid activation state")
        for key, value in {
            "pretraining_master_sha256": pretraining_master[
                "pretraining_master_sha256"
            ],
            "pretraining_audit_receipt_sha256": pretraining_audit_receipt[
                "pretraining_audit_receipt_sha256"
            ],
            "pretraining_audit_receipt_file_sha256": pretraining_audit_receipt[
                "_validated_file_sha256"
            ],
            "run_registry_custody": dict(custody),
        }.items():
            if registry.get(key) != value:
                raise ValueError(f"Activated run registry differs for {key}")
        return registry


def validate_pretraining_run_registry_binding(
    path: str | Path,
    *,
    pretraining_master: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the current ledger as a descendant of the anchored instance."""
    registry = _load_registry(path)
    expected = {
        "run_registry_instance_id": pretraining_master[
            "run_registry_instance_id"
        ],
        "initial_registry_sha256": pretraining_master[
            "initial_registry_sha256"
        ],
        "run_registry_namespace_sha256": pretraining_master[
            "run_registry_namespace_sha256"
        ],
        "source_config_sha256": pretraining_master["source_config_sha256"],
        "preflight_receipt_sha256": pretraining_master[
            "preflight_receipt_sha256"
        ],
        "expected_run_identities": pretraining_master[
            "expected_run_identities"
        ],
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise ValueError(f"Run registry/pretraining root mismatch for {key}")
    if registry.get("status") == "pretraining_frozen":
        if registry.get("slots") != {} or int(registry.get("revision", -1)) != 0:
            raise ValueError("Pretraining run registry is not the frozen empty instance")
        if sha256_file(path) != pretraining_master.get(
            "initial_registry_file_sha256"
        ) or registry.get("registry_sha256") != pretraining_master.get(
            "initial_registry_canonical_sha256"
        ):
            raise ValueError("Initial run registry file differs from the anchored root")
    elif registry.get("status") not in {"active", "complete_frozen"}:
        raise ValueError("Run registry has an invalid lifecycle state")
    return registry


def initialize_run_registry(
    path: str | Path,
    *,
    namespace_path: str | Path,
    expected_namespace_sha256: str,
    config_sha256: str,
    preflight_receipt_sha256: str,
    pretraining_master_sha256: str,
    pretraining_audit_receipt_sha256: str,
    models: Iterable[str],
    seeds: Iterable[int],
) -> dict[str, Any]:
    destination = Path(path).resolve()
    namespace = validate_run_registry_namespace(
        namespace_path,
        run_registry_path=destination,
        config_sha256=config_sha256,
        preflight_receipt_sha256=preflight_receipt_sha256,
        models=models,
        seeds=seeds,
        expected_namespace_sha256=expected_namespace_sha256,
    )
    identities = list(namespace["expected_run_identities"])
    if not identities:
        raise ValueError("Formal run registry cannot be empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(destination):
        if destination.exists():
            existing = _load_registry(destination)
            expected = {
                "run_registry_namespace_path": str(Path(namespace_path).resolve()),
                "run_registry_namespace_sha256": namespace[
                    "run_registry_namespace_sha256"
                ],
                "source_config_sha256": config_sha256,
                "preflight_receipt_sha256": preflight_receipt_sha256,
                "pretraining_master_sha256": pretraining_master_sha256,
                "pretraining_audit_receipt_sha256": pretraining_audit_receipt_sha256,
                "expected_run_identities": identities,
                **_namespace_gpu_bindings(namespace),
            }
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise ValueError(f"Existing run registry differs for {key}")
            return existing
        payload = {
            "schema_version": 1,
            "status": "active",
            "run_registry_instance_id": uuid.uuid4().hex,
            "run_registry_namespace_path": str(Path(namespace_path).resolve()),
            "run_registry_namespace_sha256": namespace[
                "run_registry_namespace_sha256"
            ],
            "source_config_sha256": str(config_sha256),
            "preflight_receipt_sha256": str(preflight_receipt_sha256),
            "pretraining_master_sha256": str(pretraining_master_sha256),
            "pretraining_audit_receipt_sha256": str(
                pretraining_audit_receipt_sha256
            ),
            "run_registry_custody": {
                "mode": "development_unit_test_not_formal",
                "run_registry_instance_id": None,
                "initial_registry_sha256": None,
                "external_attestation_id": "not-formal",
            },
            "expected_run_identities": identities,
            "slots": {},
            "revision": 0,
            "created_at_utc": _utc_now(),
            **_namespace_gpu_bindings(namespace),
        }
        initial_core = _without(payload, "created_at_utc")
        payload["initial_registry_sha256"] = canonical_json_hash(initial_core)
        payload["run_registry_custody"].update(
            {
                "run_registry_instance_id": payload["run_registry_instance_id"],
                "initial_registry_sha256": payload["initial_registry_sha256"],
            }
        )
        _save_registry(destination, payload)
        return payload


def _load_failure_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    core = _without(payload, "failure_receipt_sha256", "created_at_utc")
    if canonical_json_hash(core) != payload.get("failure_receipt_sha256"):
        raise ValueError(f"Failure receipt hash mismatch: {source}")
    if payload.get("failure_class") not in {"model", "infrastructure"}:
        raise ValueError("Failure receipt has an invalid failure class")
    log_path = Path(payload.get("log_path", "")).resolve()
    if (
        not log_path.is_file()
        or sha256_file(log_path) != payload.get("log_sha256")
    ):
        raise ValueError("Failure receipt subprocess-log hash mismatch")
    return payload


def _load_retry_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
    identity: str,
    prior_failure: dict[str, Any],
    preflight_receipt_sha256: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if len(str(expected_file_sha256)) != 64:
        raise PermissionError("Retry authorization requires an external SHA-256 anchor")
    int(str(expected_file_sha256), 16)
    if sha256_file(source) != str(expected_file_sha256):
        raise PermissionError("Retry authorization differs from its external SHA-256 anchor")
    payload = json.loads(source.read_text(encoding="utf-8"))
    core = _without(payload, "authorization_sha256", "created_at_utc")
    if canonical_json_hash(core) != payload.get("authorization_sha256"):
        raise ValueError(f"Retry authorization hash mismatch: {source}")
    expected = {
        "schema_version": 1,
        "action": "authorize_infrastructure_retry",
        "run_identity": identity,
        "prior_failure_receipt_sha256": prior_failure[
            "failure_receipt_sha256"
        ],
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "allow_new_attempt": True,
        "issued_out_of_band": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Retry authorization mismatch for {key}")
    if not str(payload.get("authorization_id", "")).strip() or not str(
        payload.get("authorized_by", "")
    ).strip():
        raise ValueError("Retry authorization lacks an external ID/authorizer")
    return payload


def _reservation_token_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"reservation_token_sha256", "created_at_utc"}
    }


def _load_reservation_token(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if canonical_json_hash(_reservation_token_core(payload)) != payload.get(
        "reservation_token_sha256"
    ):
        raise ValueError(f"Run reservation token hash mismatch: {source}")
    if payload.get("schema_version") != 1 or payload.get("action") != "authorize_single_training_attempt":
        raise ValueError("Run reservation token schema/action mismatch")
    return payload


def validate_run_reservation(
    registry_path: str | Path,
    *,
    attempt_id: str,
    token_path: str | Path,
    expected_token_file_sha256: str,
    model: str,
    seed: int,
    output_dir: str | Path,
    allowed_statuses: set[str] | None = None,
    assigned_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Bind a formal worker to one write-once registry-issued attempt token."""
    if len(str(expected_token_file_sha256)) != 64:
        raise PermissionError("Run reservation requires an external token SHA-256")
    int(str(expected_token_file_sha256), 16)
    token_source = Path(token_path).resolve()
    if sha256_file(token_source) != str(expected_token_file_sha256):
        raise PermissionError("Run reservation token differs from its supplied anchor")
    registry = _load_registry(registry_path)
    attempt = _find_attempt(registry, str(attempt_id))
    statuses = allowed_statuses or {"reserved", "running"}
    if attempt.get("status") not in statuses:
        raise RuntimeError(
            f"Run reservation status is not admissible: {attempt.get('status')}"
        )
    token = _load_reservation_token(token_source)
    attempt_gpu = attempt.get("assigned_gpu_uuid")
    expected_gpu = (
        _canonical_gpu_uuid(assigned_gpu_uuid)
        if assigned_gpu_uuid is not None
        else (_canonical_gpu_uuid(attempt_gpu) if attempt_gpu is not None else None)
    )
    expected = {
        "run_identity": f"{model}::seed_{int(seed)}",
        "model": str(model),
        "seed": int(seed),
        "attempt_id": str(attempt_id),
        "output_dir": str(Path(output_dir).resolve()),
        "run_registry_namespace_sha256": registry[
            "run_registry_namespace_sha256"
        ],
        "run_registry_instance_id": registry["run_registry_instance_id"],
        "initial_registry_sha256": registry["initial_registry_sha256"],
        "run_registry_custody_sha256": (
            canonical_json_hash(registry["run_registry_custody"])
            if isinstance(registry.get("run_registry_custody"), Mapping)
            else None
        ),
        "source_config_sha256": registry["source_config_sha256"],
        "preflight_receipt_sha256": registry["preflight_receipt_sha256"],
        "pretraining_master_sha256": registry["pretraining_master_sha256"],
        "pretraining_audit_receipt_sha256": registry[
            "pretraining_audit_receipt_sha256"
        ],
        "assigned_gpu_uuid": expected_gpu,
        "training_gpu_assignment_sha256": (
            canonical_json_hash(
                {
                    "source_config_sha256": registry["source_config_sha256"],
                    "run_registry_namespace_sha256": registry[
                        "run_registry_namespace_sha256"
                    ],
                    "seed": int(seed),
                    "assigned_gpu_uuid": expected_gpu,
                }
            )
            if expected_gpu is not None
            else None
        ),
    }
    for key, value in expected.items():
        if token.get(key) != value or attempt.get(key) not in (None, value):
            raise ValueError(f"Run reservation identity mismatch for {key}")
    if (
        Path(attempt.get("reservation_token_path", "")).resolve() != token_source
        or attempt.get("reservation_token_sha256")
        != token.get("reservation_token_sha256")
        or attempt.get("reservation_token_file_sha256")
        != str(expected_token_file_sha256)
    ):
        raise ValueError("Run registry/token binding mismatch")
    return {
        "registry_path": str(Path(registry_path).resolve()),
        "registry_sha256": registry["registry_sha256"],
        "attempt": dict(attempt),
        "token": token,
        "reservation_token_path": str(token_source),
        "reservation_token_file_sha256": str(expected_token_file_sha256),
    }


def _worker_launch_receipt_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _without(payload, "worker_launch_receipt_sha256", "created_at_utc")


def _validate_worker_launch_receipt(
    path: str | Path,
    *,
    expected_file_sha256: str | None,
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    source = Path(path).resolve()
    if len(str(expected_file_sha256 or "")) != 64:
        raise PermissionError("Worker launch receipt requires a SHA-256 file anchor")
    int(str(expected_file_sha256), 16)
    if not source.is_file() or sha256_file(source) != str(expected_file_sha256):
        raise PermissionError("Worker launch receipt differs from its file anchor")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("action") != "launch_formal_training_child"
        or canonical_json_hash(_worker_launch_receipt_core(payload))
        != payload.get("worker_launch_receipt_sha256")
    ):
        raise ValueError("Worker launch receipt hash/schema mismatch")
    expected = {
        "run_identity": attempt.get("run_identity"),
        "attempt_id": attempt.get("attempt_id"),
        "attempt_number": int(attempt.get("attempt_number", -1)),
        "assigned_gpu_uuid": attempt.get("assigned_gpu_uuid"),
        "reservation_token_sha256": attempt.get("reservation_token_sha256"),
        "reservation_token_file_sha256": attempt.get(
            "reservation_token_file_sha256"
        ),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Worker launch receipt mismatch for {key}")
    if int(payload.get("child_process_id", -1)) <= 0 or not re.fullmatch(
        r"[0-9a-f]{32}", str(payload.get("worker_nonce", ""))
    ):
        raise ValueError("Worker launch receipt has an invalid PID/nonce")
    return payload


def create_worker_launch_receipt(
    registry_path: str | Path,
    *,
    reservation: Mapping[str, Any],
    command: Iterable[str],
    child_process_id: int,
    worker_nonce: str,
    child_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Write one immutable child-start receipt before marking it running."""
    if int(child_process_id) <= 0 or re.fullmatch(r"[0-9a-f]{32}", worker_nonce) is None:
        raise ValueError("Worker launch requires a positive PID and random nonce")
    registry = _load_registry(registry_path)
    attempt = _find_attempt(registry, str(reservation["attempt_id"]))
    if attempt.get("status") not in {"reserved", "running"}:
        raise RuntimeError("Worker launch receipt requires an open attempt")
    if attempt.get("assigned_gpu_uuid") is None:
        raise ValueError("Formal worker launch requires an assigned physical GPU")
    if (
        attempt.get("prelaunch_pending") is not True
        or attempt.get("worker_nonce") != worker_nonce
    ):
        raise ValueError("Worker launch differs from its frozen prelaunch nonce")
    launch_number = len(attempt.get("worker_launches", [])) + 1
    command_list = [str(value) for value in command]
    environment_identity = {
        "CUDA_VISIBLE_DEVICES": child_environment.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": child_environment.get("CUDA_DEVICE_ORDER"),
        "LIQUID_FORMAL_WORKER_NONCE": child_environment.get(
            "LIQUID_FORMAL_WORKER_NONCE"
        ),
    }
    if environment_identity != {
        "CUDA_VISIBLE_DEVICES": attempt["assigned_gpu_uuid"],
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "LIQUID_FORMAL_WORKER_NONCE": worker_nonce,
    }:
        raise ValueError("Formal child environment differs from its assigned GPU")
    core = {
        "schema_version": 1,
        "action": "launch_formal_training_child",
        "run_registry_namespace_sha256": registry[
            "run_registry_namespace_sha256"
        ],
        "run_registry_instance_id": registry["run_registry_instance_id"],
        "run_identity": attempt["run_identity"],
        "attempt_id": attempt["attempt_id"],
        "attempt_number": int(attempt["attempt_number"]),
        "launch_number": launch_number,
        "assigned_gpu_uuid": attempt["assigned_gpu_uuid"],
        "training_gpu_assignment_sha256": attempt[
            "training_gpu_assignment_sha256"
        ],
        "reservation_token_sha256": attempt["reservation_token_sha256"],
        "reservation_token_file_sha256": attempt[
            "reservation_token_file_sha256"
        ],
        "child_process_id": int(child_process_id),
        "worker_nonce": worker_nonce,
        "child_environment": environment_identity,
        "command": command_list,
        "command_sha256": canonical_json_hash(command_list),
    }
    receipt = {
        **core,
        "worker_launch_receipt_sha256": canonical_json_hash(core),
        "created_at_utc": _utc_now(),
    }
    directory = Path(registry_path).resolve().parent / "worker_launch_receipts"
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", attempt["attempt_id"])
    destination = directory / f"{safe}.launch_{launch_number:03d}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite worker receipt: {destination}")
    atomic_write_json(destination, receipt)
    receipt["_validated_path"] = str(destination)
    receipt["_validated_file_sha256"] = sha256_file(destination)
    return receipt


def reserve_run(
    registry_path: str | Path,
    *,
    model: str,
    seed: int,
    owner_id: str,
    output_dir: str | Path,
    assigned_gpu_uuid: str | None = None,
    retry_authorization_path: str | Path | None = None,
    retry_authorization_expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Reserve exactly one slot, or resume the same owner's open attempt."""
    path = Path(registry_path).resolve()
    identity = f"{model}::seed_{int(seed)}"
    assigned_gpu = (
        _canonical_gpu_uuid(assigned_gpu_uuid)
        if assigned_gpu_uuid is not None
        else None
    )
    with _exclusive_lock(path):
        registry = _load_registry(path)
        if registry.get("status") != "active":
            raise RuntimeError("Formal run registry is frozen and accepts no attempts")
        if identity not in registry["expected_run_identities"]:
            raise KeyError(f"Run identity is not registered: {identity}")
        frozen_mapping = registry.get("training_gpu_by_seed")
        if frozen_mapping is not None:
            if assigned_gpu is None:
                raise PermissionError("GPU-plan-bound registry requires an assigned GPU")
            if frozen_mapping.get(str(int(seed))) != assigned_gpu:
                raise ValueError("Assigned GPU differs from the frozen seed mapping")
        elif assigned_gpu is not None and (
            registry.get("run_registry_custody", {}).get("mode")
            == "auditor_managed_append_only"
        ):
            raise PermissionError("Formal assigned GPU lacks a frozen runtime plan")
        attempts = registry["slots"].setdefault(identity, [])
        if attempts:
            prior = attempts[-1]
            status = prior.get("status")
            if status == "complete":
                raise RuntimeError(f"Run identity is already complete: {identity}")
            if status in {"reserved", "running"}:
                if prior.get("owner_id") != owner_id:
                    raise RuntimeError(f"Run identity is owned by another launcher: {identity}")
                if status == "reserved" and prior.get("prelaunch_pending") is True:
                    raise RuntimeError(
                        "Formal attempt has an indeterminate prelaunch; reconcile it "
                        "before any resume to avoid a double worker"
                    )
                expected_output = (
                    Path(output_dir).resolve()
                    / f"attempt_{int(prior['attempt_number']):03d}"
                )
                if Path(prior.get("output_dir", "")).resolve() != expected_output:
                    raise ValueError("Open attempt output directory differs from reservation")
                if assigned_gpu is not None and prior.get("assigned_gpu_uuid") != assigned_gpu:
                    raise ValueError("Resume attempted on a different physical GPU")
                validate_run_reservation(
                    path,
                    attempt_id=prior["attempt_id"],
                    token_path=prior["reservation_token_path"],
                    expected_token_file_sha256=prior[
                        "reservation_token_file_sha256"
                    ],
                    model=model,
                    seed=seed,
                    output_dir=expected_output,
                    allowed_statuses={"reserved", "running"},
                    assigned_gpu_uuid=assigned_gpu,
                )
                if assigned_gpu is not None and status == "running":
                    worker_pid = int(prior.get("child_process_id", -1))
                    worker_nonce = str(prior.get("worker_nonce", ""))
                    receipt_path = prior.get("worker_launch_receipt_path")
                    if not worker_nonce or not receipt_path:
                        raise ValueError("Running formal attempt lacks worker identity evidence")
                    _validate_worker_launch_receipt(
                        receipt_path,
                        expected_file_sha256=prior.get(
                            "worker_launch_receipt_file_sha256"
                        ),
                        attempt=prior,
                    )
                    if _pid_is_alive(worker_pid):
                        raise RuntimeError(
                            f"Formal training child is still live for {identity}: {worker_pid}"
                        )
                    recovery = {
                        "detected_at_utc": _utc_now(),
                        "child_process_id": worker_pid,
                        "worker_nonce": worker_nonce,
                        "worker_launch_receipt_path": prior.get(
                            "worker_launch_receipt_path"
                        ),
                        "worker_launch_receipt_sha256": prior.get(
                            "worker_launch_receipt_sha256"
                        ),
                        "worker_launch_receipt_file_sha256": prior.get(
                            "worker_launch_receipt_file_sha256"
                        ),
                    }
                    recoveries = list(prior.get("dead_worker_recoveries", []))
                    recoveries.append(recovery)
                    prior.update(
                        {
                            "status": "reserved",
                            "prelaunch_pending": False,
                            "dead_worker_recoveries": recoveries,
                            "resumed_after_dead_child_at_utc": recovery[
                                "detected_at_utc"
                            ],
                        }
                    )
                    for field in (
                        "child_process_id",
                        "worker_nonce",
                        "worker_launch_receipt_path",
                        "worker_launch_receipt_sha256",
                        "worker_launch_receipt_file_sha256",
                        "launch_parent_process_id",
                        "prelaunch_at_utc",
                    ):
                        prior.pop(field, None)
                    registry["last_transition_event"] = {
                        "event": "prepare_dead_worker_relaunch",
                        "run_identity": identity,
                        "attempt_id": prior["attempt_id"],
                        "attempt_number": int(prior["attempt_number"]),
                        "dead_child_process_id": worker_pid,
                        "prior_worker_nonce": worker_nonce,
                    }
                    registry["revision"] = int(registry.get("revision", 0)) + 1
                    _save_registry(path, registry)
                return {**prior, "resumed_existing_attempt": True}
            if status != "failed":
                raise ValueError(f"Unknown prior attempt status for {identity}: {status}")
            if prior.get("failure_class") != "infrastructure":
                raise RuntimeError(
                    f"A model-class failure cannot receive a new attempt: {identity}"
                )
            if retry_authorization_path is None or retry_authorization_expected_sha256 is None:
                raise PermissionError(
                    f"Infrastructure retry requires external authorization with a SHA-256 anchor: {identity}"
                )
            failure = _load_failure_receipt(prior["failure_receipt_path"])
            if failure.get("failure_receipt_sha256") != prior.get(
                "failure_receipt_sha256"
            ):
                raise ValueError("Registry/failure-receipt identity mismatch")
            authorization = _load_retry_authorization(
                retry_authorization_path,
                expected_file_sha256=retry_authorization_expected_sha256,
                identity=identity,
                prior_failure=failure,
                preflight_receipt_sha256=registry[
                    "preflight_receipt_sha256"
                ],
            )
            authorization_identity = {
                "path": str(Path(retry_authorization_path).resolve()),
                "sha256": authorization["authorization_sha256"],
                "file_sha256": sha256_file(retry_authorization_path),
                "expected_external_file_sha256": str(
                    retry_authorization_expected_sha256
                ),
                "prior_failure_receipt_sha256": failure[
                    "failure_receipt_sha256"
                ],
            }
        else:
            authorization_identity = None
        attempt_number = len(attempts) + 1
        attempt_output = Path(output_dir).resolve() / f"attempt_{attempt_number:03d}"
        attempt = {
            "run_identity": identity,
            "model": model,
            "seed": int(seed),
            "attempt_number": attempt_number,
            "attempt_id": f"{identity}::attempt_{attempt_number:03d}",
            "owner_id": str(owner_id),
            "status": "reserved",
            "output_dir": str(attempt_output),
            "retry_authorization": authorization_identity,
            "reserved_at_utc": _utc_now(),
        }
        if assigned_gpu is not None:
            assignment_sha256 = canonical_json_hash(
                {
                    "source_config_sha256": registry["source_config_sha256"],
                    "run_registry_namespace_sha256": registry[
                        "run_registry_namespace_sha256"
                    ],
                    "seed": int(seed),
                    "assigned_gpu_uuid": assigned_gpu,
                }
            )
            attempt.update(
                {
                    "assigned_gpu_uuid": assigned_gpu,
                    "training_gpu_assignment_sha256": assignment_sha256,
                }
            )
        token_directory = path.parent / "reservation_tokens"
        token_directory.mkdir(parents=True, exist_ok=True)
        safe_attempt = re.sub(r"[^A-Za-z0-9_.-]+", "_", attempt["attempt_id"])
        token_path = token_directory / f"{safe_attempt}.json"
        if token_path.exists():
            raise FileExistsError(f"Refusing to overwrite reservation token: {token_path}")
        token_core = {
            "schema_version": 1,
            "action": "authorize_single_training_attempt",
            "reservation_nonce": uuid.uuid4().hex,
            "run_identity": identity,
            "model": str(model),
            "seed": int(seed),
            "attempt_number": attempt_number,
            "attempt_id": attempt["attempt_id"],
            "owner_id": str(owner_id),
            "output_dir": str(attempt_output),
            "run_registry_namespace_sha256": registry[
                "run_registry_namespace_sha256"
            ],
            "run_registry_instance_id": registry["run_registry_instance_id"],
            "initial_registry_sha256": registry["initial_registry_sha256"],
            "run_registry_custody_sha256": (
                canonical_json_hash(registry["run_registry_custody"])
                if isinstance(registry.get("run_registry_custody"), Mapping)
                else None
            ),
            "source_config_sha256": registry["source_config_sha256"],
            "preflight_receipt_sha256": registry["preflight_receipt_sha256"],
            "pretraining_master_sha256": registry["pretraining_master_sha256"],
            "pretraining_audit_receipt_sha256": registry[
                "pretraining_audit_receipt_sha256"
            ],
            "pretraining_audit_receipt_file_sha256": registry.get(
                "pretraining_audit_receipt_file_sha256"
            ),
            "retry_authorization": authorization_identity,
        }
        if assigned_gpu is not None:
            token_core.update(
                {
                    "assigned_gpu_uuid": assigned_gpu,
                    "training_gpu_assignment_sha256": assignment_sha256,
                }
            )
        token = {
            **token_core,
            "reservation_token_sha256": canonical_json_hash(token_core),
            "created_at_utc": _utc_now(),
        }
        atomic_write_json(token_path, token)
        attempt.update(
            {
                "reservation_token_path": str(token_path),
                "reservation_token_sha256": token["reservation_token_sha256"],
                "reservation_token_file_sha256": sha256_file(token_path),
            }
        )
        attempts.append(attempt)
        registry["last_transition_event"] = {
            "event": "reserve_training_attempt",
            "run_identity": identity,
            "attempt_id": attempt["attempt_id"],
            "attempt_number": attempt_number,
            "retry_authorized": authorization_identity is not None,
        }
        registry["revision"] = int(registry.get("revision", 0)) + 1
        _save_registry(path, registry)
        return {**attempt, "resumed_existing_attempt": False}


def mark_run_prelaunch(
    registry_path: str | Path,
    *,
    attempt_id: str,
    owner_id: str,
    log_path: str | Path,
    worker_nonce: str,
) -> dict[str, Any]:
    """Close the parent-crash window before Popen without claiming a child PID."""
    if re.fullmatch(r"[0-9a-f]{32}", worker_nonce) is None:
        raise ValueError("Formal prelaunch requires a random worker nonce")
    path = Path(registry_path).resolve()
    with _exclusive_lock(path):
        registry = _load_registry(path)
        attempt = _find_attempt(registry, attempt_id)
        if (
            registry.get("status") != "active"
            or attempt.get("owner_id") != owner_id
            or attempt.get("status") != "reserved"
            or attempt.get("assigned_gpu_uuid") is None
            or attempt.get("prelaunch_pending") is True
        ):
            raise RuntimeError(f"Attempt cannot enter formal prelaunch: {attempt_id}")
        attempt.update(
            {
                "prelaunch_pending": True,
                "worker_nonce": worker_nonce,
                "log_path": str(Path(log_path).resolve()),
                "launch_parent_process_id": os.getpid(),
                "prelaunch_at_utc": _utc_now(),
            }
        )
        registry["last_transition_event"] = {
            "event": "mark_attempt_prelaunch",
            "run_identity": attempt["run_identity"],
            "attempt_id": attempt_id,
            "attempt_number": int(attempt["attempt_number"]),
            "worker_nonce": worker_nonce,
        }
        registry["revision"] = int(registry.get("revision", 0)) + 1
        _save_registry(path, registry)
        return dict(attempt)


def mark_run_running(
    registry_path: str | Path,
    *,
    attempt_id: str,
    owner_id: str,
    log_path: str | Path,
    child_process_id: int | None = None,
    worker_nonce: str | None = None,
    worker_launch_receipt_path: str | Path | None = None,
    worker_launch_receipt_file_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(registry_path).resolve()
    with _exclusive_lock(path):
        registry = _load_registry(path)
        if registry.get("status") != "active":
            raise RuntimeError("Formal run registry is frozen")
        attempt = _find_attempt(registry, attempt_id)
        if attempt.get("owner_id") != owner_id:
            raise PermissionError(f"Attempt owner mismatch: {attempt_id}")
        if attempt.get("status") not in {"reserved", "running"}:
            raise RuntimeError(f"Attempt cannot be started: {attempt_id}")
        assigned_gpu = attempt.get("assigned_gpu_uuid")
        if assigned_gpu is None:
            if any(
                value is not None
                for value in (
                    child_process_id,
                    worker_nonce,
                    worker_launch_receipt_path,
                    worker_launch_receipt_file_sha256,
                )
            ):
                raise ValueError("Legacy unassigned run cannot carry formal worker evidence")
            updates = {
                "status": "running",
                "started_at_utc": _utc_now(),
                "log_path": str(Path(log_path).resolve()),
                "process_id": os.getpid(),
            }
        else:
            if (
                child_process_id is None
                or worker_nonce is None
                or worker_launch_receipt_path is None
                or worker_launch_receipt_file_sha256 is None
            ):
                raise PermissionError("Formal run requires child PID/nonce/launch receipt")
            if attempt.get("status") == "running" and _pid_is_alive(
                int(attempt.get("child_process_id", -1))
            ):
                raise RuntimeError("Refusing to attach a second worker to a live attempt")
            receipt = _validate_worker_launch_receipt(
                worker_launch_receipt_path,
                expected_file_sha256=worker_launch_receipt_file_sha256,
                attempt=attempt,
            )
            if (
                receipt["child_process_id"] != int(child_process_id)
                or receipt["worker_nonce"] != worker_nonce
                or receipt["assigned_gpu_uuid"] != assigned_gpu
            ):
                raise ValueError("Worker launch receipt differs from running transition")
            launch_row = {
                "launch_number": int(receipt["launch_number"]),
                "child_process_id": int(child_process_id),
                "worker_nonce": worker_nonce,
                "worker_launch_receipt_path": str(
                    Path(worker_launch_receipt_path).resolve()
                ),
                "worker_launch_receipt_sha256": receipt[
                    "worker_launch_receipt_sha256"
                ],
                "worker_launch_receipt_file_sha256": str(
                    worker_launch_receipt_file_sha256
                ),
                "started_at_utc": _utc_now(),
            }
            launches = list(attempt.get("worker_launches", []))
            if int(receipt["launch_number"]) != len(launches) + 1:
                raise ValueError("Worker launch sequence is not contiguous")
            launches.append(launch_row)
            updates = {
                "status": "running",
                "started_at_utc": launch_row["started_at_utc"],
                "log_path": str(Path(log_path).resolve()),
                "process_id": os.getpid(),
                "child_process_id": int(child_process_id),
                "worker_nonce": worker_nonce,
                "worker_launch_receipt_path": launch_row[
                    "worker_launch_receipt_path"
                ],
                "worker_launch_receipt_sha256": launch_row[
                    "worker_launch_receipt_sha256"
                ],
                "worker_launch_receipt_file_sha256": launch_row[
                    "worker_launch_receipt_file_sha256"
                ],
                "worker_launches": launches,
                "prelaunch_pending": False,
            }
        attempt.update(updates)
        registry["last_transition_event"] = {
            "event": "mark_attempt_running",
            "run_identity": attempt["run_identity"],
            "attempt_id": attempt_id,
            "attempt_number": int(attempt["attempt_number"]),
            "child_process_id": attempt.get("child_process_id"),
            "worker_nonce": attempt.get("worker_nonce"),
        }
        registry["revision"] = int(registry.get("revision", 0)) + 1
        _save_registry(path, registry)
        return dict(attempt)


def _find_attempt(registry: dict[str, Any], attempt_id: str):
    matches = [
        attempt
        for attempts in registry["slots"].values()
        for attempt in attempts
        if attempt.get("attempt_id") == attempt_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Attempt ID is absent or duplicated: {attempt_id}")
    return matches[0]


def _transition_attempt(
    registry_path: str | Path,
    *,
    attempt_id: str,
    owner_id: str,
    expected: set[str],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(registry_path).resolve()
    with _exclusive_lock(path):
        registry = _load_registry(path)
        if registry.get("status") != "active":
            raise RuntimeError("Formal run registry is frozen")
        attempt = _find_attempt(registry, attempt_id)
        if attempt.get("owner_id") != owner_id:
            raise PermissionError(f"Attempt owner mismatch: {attempt_id}")
        if attempt.get("status") not in expected:
            raise RuntimeError(
                f"Invalid attempt transition {attempt.get('status')} -> "
                f"{updates.get('status')}: {attempt_id}"
            )
        attempt.update(dict(updates))
        registry["last_transition_event"] = {
            "event": f"mark_attempt_{updates.get('status')}",
            "run_identity": attempt["run_identity"],
            "attempt_id": attempt_id,
            "attempt_number": int(attempt["attempt_number"]),
        }
        registry["revision"] = int(registry.get("revision", 0)) + 1
        _save_registry(path, registry)
        return dict(attempt)


def mark_run_complete(
    registry_path: str | Path,
    *,
    attempt_id: str,
    owner_id: str,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    import torch

    checkpoint = Path(checkpoint_path).resolve()
    run_state = checkpoint.parent / "run_state.json"
    if not checkpoint.is_file() or not run_state.is_file():
        raise FileNotFoundError("Completion requires a checkpoint and run_state.json")
    state = json.loads(run_state.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or state.get(
        "best_checkpoint_sha256"
    ) != sha256_file(checkpoint):
        raise ValueError("Training run state does not bind the completed checkpoint")
    registry = _load_registry(registry_path)
    attempt = _find_attempt(registry, attempt_id)
    reserved_output = Path(attempt["output_dir"]).resolve()
    if (
        checkpoint != reserved_output / "best.pt"
        or run_state != reserved_output / "run_state.json"
        or Path(state.get("best_checkpoint", "")).resolve() != checkpoint
    ):
        raise ValueError(
            "Completed checkpoint/run_state are outside the reserved output directory"
        )
    reservation = validate_run_reservation(
        registry_path,
        attempt_id=attempt_id,
        token_path=attempt["reservation_token_path"],
        expected_token_file_sha256=attempt[
            "reservation_token_file_sha256"
        ],
        model=attempt["model"],
        seed=int(attempt["seed"]),
        output_dir=attempt["output_dir"],
        allowed_statuses={"running"},
    )
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    assigned_gpu = attempt.get("assigned_gpu_uuid")
    if assigned_gpu is not None:
        checkpoint_gpu = (
            checkpoint_payload.get("runtime_device", {})
            .get("selected_physical_gpu", {})
            .get("uuid")
        )
        state_gpu = (
            state.get("runtime_device", {})
            .get("selected_physical_gpu", {})
            .get("uuid")
        )
        if (
            _canonical_gpu_uuid(checkpoint_gpu) != assigned_gpu
            or _canonical_gpu_uuid(state_gpu) != assigned_gpu
        ):
            raise ValueError(
                "Completed checkpoint runtime physical GPU differs from reservation"
            )
    expected_binding = {
        "formal_run_registry_namespace_sha256": registry[
            "run_registry_namespace_sha256"
        ],
        "formal_run_registry_instance_id": registry[
            "run_registry_instance_id"
        ],
        "formal_initial_run_registry_sha256": registry[
            "initial_registry_sha256"
        ],
        "formal_run_registry_custody_sha256": canonical_json_hash(
            registry["run_registry_custody"]
        ),
        "formal_run_attempt_id": attempt_id,
        "formal_run_registry_path": str(Path(registry_path).resolve()),
        "formal_run_reservation_token_path": reservation[
            "reservation_token_path"
        ],
        "formal_run_reservation_token_sha256": reservation["token"][
            "reservation_token_sha256"
        ],
        "formal_run_reservation_token_file_sha256": reservation[
            "reservation_token_file_sha256"
        ],
        "formal_run_reservation_nonce": reservation["token"][
            "reservation_nonce"
        ],
    }
    for key, value in expected_binding.items():
        if checkpoint_payload.get(key) != value or state.get(key) != value:
            raise ValueError(f"Completed checkpoint/reservation mismatch for {key}")
    return _transition_attempt(
        registry_path,
        attempt_id=attempt_id,
        owner_id=owner_id,
        expected={"running"},
        updates={
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "run_state_path": str(run_state),
            "run_state_sha256": sha256_file(run_state),
            "reservation_token_path": reservation["reservation_token_path"],
            "reservation_token_sha256": reservation["token"][
                "reservation_token_sha256"
            ],
            "reservation_token_file_sha256": reservation[
                "reservation_token_file_sha256"
            ],
            "assigned_gpu_uuid": assigned_gpu,
            "runtime_physical_gpu_uuid": assigned_gpu,
        },
    )


def mark_run_failed(
    registry_path: str | Path,
    *,
    attempt_id: str,
    owner_id: str,
    failure_class: str,
    failure_code: str,
    message: str,
    return_code: int | None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    if failure_class not in {"model", "infrastructure"}:
        raise ValueError("failure_class must be model or infrastructure")
    allowed = (
        MODEL_FAILURE_CODES
        if failure_class == "model"
        else INFRASTRUCTURE_FAILURE_CODES
    )
    if failure_code not in allowed:
        raise ValueError(f"Invalid {failure_class} failure code: {failure_code}")
    path = Path(registry_path).resolve()
    with _exclusive_lock(path):
        registry = _load_registry(path)
        attempt = _find_attempt(registry, attempt_id)
        if attempt.get("owner_id") != owner_id or attempt.get("status") not in {
            "reserved",
            "running",
        }:
            raise RuntimeError(f"Attempt cannot be failed: {attempt_id}")
        failure_dir = path.parent / "failure_receipts"
        failure_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", attempt_id)
        failure_path = failure_dir / f"{safe}.json"
        if failure_path.exists():
            raise FileExistsError(f"Refusing to overwrite failure receipt: {failure_path}")
        failure_log_path = Path(
            log_path if log_path is not None else attempt.get("log_path", "")
        ).resolve()
        if not failure_log_path.is_file():
            raise FileNotFoundError(
                "Failed attempt requires its immutable subprocess log"
            )
        core = {
            "schema_version": 1,
            "run_identity": attempt["run_identity"],
            "attempt_id": attempt_id,
            "failure_class": failure_class,
            "failure_code": failure_code,
            "message": str(message)[-12000:],
            "return_code": return_code,
            "preflight_receipt_sha256": registry["preflight_receipt_sha256"],
            "pretraining_master_sha256": registry[
                "pretraining_master_sha256"
            ],
            "pretraining_audit_receipt_sha256": registry[
                "pretraining_audit_receipt_sha256"
            ],
            "registry_sha256_before_failure": registry["registry_sha256"],
            "output_dir": attempt["output_dir"],
            "assigned_gpu_uuid": attempt.get("assigned_gpu_uuid"),
            "worker_nonce": attempt.get("worker_nonce"),
            "child_process_id": attempt.get("child_process_id"),
            "log_path": str(failure_log_path),
            "log_sha256": sha256_file(failure_log_path),
        }
        receipt = {
            **core,
            "failure_receipt_sha256": canonical_json_hash(core),
            "created_at_utc": _utc_now(),
        }
        atomic_write_json(failure_path, receipt)
        attempt.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "failure_class": failure_class,
                "failure_code": failure_code,
                "failure_receipt_path": str(failure_path),
                "failure_receipt_sha256": receipt["failure_receipt_sha256"],
                "log_path": str(failure_log_path),
            }
        )
        registry["last_transition_event"] = {
            "event": "mark_attempt_failed",
            "run_identity": attempt["run_identity"],
            "attempt_id": attempt_id,
            "attempt_number": int(attempt["attempt_number"]),
            "failure_class": failure_class,
            "failure_code": failure_code,
            "failure_receipt_sha256": receipt["failure_receipt_sha256"],
        }
        registry["revision"] = int(registry.get("revision", 0)) + 1
        _save_registry(path, registry)
        return dict(attempt)


def classify_subprocess_failure(text: str, return_code: int | None) -> tuple[str, str]:
    lowered = text.lower()
    infrastructure = (
        ("cuda_driver", ("driver version is insufficient", "cuda driver error")),
        ("gpu_unavailable", ("no cuda gpus are available", "cuda is unavailable")),
        ("filesystem_io", ("no space left on device", "disk quota exceeded")),
        ("host_preemption", ("preempted", "terminated by scheduler")),
        ("process_launch", ("failed to create process", "winerror 2")),
    )
    for code, markers in infrastructure:
        if any(marker in lowered for marker in markers):
            return "infrastructure", code
    model = (
        ("cuda_out_of_memory", ("cuda out of memory",)),
        ("nonfinite_gradient", ("non-finite gradient", "nonfinite gradient")),
        ("nonfinite_loss", ("non-finite", "nan loss", "inf loss")),
        ("determinism_violation", ("deterministic", "nondeterministic")),
        ("budget_gate", ("fixed-budget", "planned-step mismatch")),
    )
    for code, markers in model:
        if any(marker in lowered for marker in markers):
            return "model", code
    del return_code
    return "model", "model_exception"


def validate_complete_run_registry(
    registry_path: str | Path,
    *,
    expected_config_sha256: str,
    expected_pretraining_master_sha256: str,
    expected_pretraining_audit_receipt_sha256: str,
    expected_run_registry_namespace_sha256: str,
    expected_identities: set[str],
    expected_pretraining_audit_receipt_file_sha256: str | None = None,
    require_frozen: bool = True,
) -> dict[str, Any]:
    """Prove every registered slot has one authorized, non-cherry-picked result."""
    import torch

    path = Path(registry_path).resolve()
    registry = _load_registry(path)
    journal = _validate_registry_journal(registry)
    custody = registry.get("run_registry_custody")
    if (
        isinstance(custody, Mapping)
        and custody.get("mode") == "auditor_managed_append_only"
        and journal is None
    ):
        raise PermissionError("Formal registry lacks its external journal chain")
    if require_frozen and registry.get("status") != "complete_frozen":
        raise ValueError("Run registry is not frozen after all registered attempts")
    expected_header = {
        "run_registry_namespace_sha256": str(
            expected_run_registry_namespace_sha256
        ),
        "source_config_sha256": str(expected_config_sha256),
        "pretraining_master_sha256": str(expected_pretraining_master_sha256),
        "pretraining_audit_receipt_sha256": str(
            expected_pretraining_audit_receipt_sha256
        ),
        "expected_run_identities": sorted(expected_identities),
    }
    if expected_pretraining_audit_receipt_file_sha256 is not None:
        expected_header["pretraining_audit_receipt_file_sha256"] = str(
            expected_pretraining_audit_receipt_file_sha256
        )
    for key, value in expected_header.items():
        if registry.get(key) != value:
            raise ValueError(f"Run registry frozen identity mismatch for {key}")
    if set(registry.get("slots", {})) != set(expected_identities):
        raise ValueError("Run registry does not contain exactly every registered slot")
    validate_run_registry_namespace(
        registry["run_registry_namespace_path"],
        run_registry_path=path,
        config_sha256=expected_config_sha256,
        preflight_receipt_sha256=registry["preflight_receipt_sha256"],
        models=sorted(
            {identity.split("::", 1)[0] for identity in expected_identities}
        ),
        seeds=sorted(
            {
                int(identity.rsplit("seed_", 1)[1])
                for identity in expected_identities
            }
        ),
        expected_namespace_sha256=expected_run_registry_namespace_sha256,
    )
    slot_receipts: list[dict[str, Any]] = []
    for identity in sorted(expected_identities):
        attempts = registry["slots"][identity]
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"Run registry slot lacks an attempt: {identity}")
        completed = [row for row in attempts if row.get("status") == "complete"]
        if len(completed) != 1 or attempts[-1] is not completed[0]:
            raise ValueError(
                f"Run registry slot must end in exactly one completed attempt: {identity}"
            )
        prior_failure_hash: str | None = None
        attempt_rows: list[dict[str, Any]] = []
        for index, attempt in enumerate(attempts, start=1):
            if int(attempt.get("attempt_number", -1)) != index:
                raise ValueError(f"Run registry attempt numbering changed: {identity}")
            reservation = validate_run_reservation(
                path,
                attempt_id=attempt["attempt_id"],
                token_path=attempt["reservation_token_path"],
                expected_token_file_sha256=attempt[
                    "reservation_token_file_sha256"
                ],
                model=attempt["model"],
                seed=int(attempt["seed"]),
                output_dir=attempt["output_dir"],
                allowed_statuses={"failed", "complete"},
            )
            if index == 1 and attempt.get("retry_authorization") is not None:
                raise ValueError("First registered attempt cannot be a retry")
            if index > 1:
                authorization = attempt.get("retry_authorization")
                if not isinstance(authorization, dict) or not prior_failure_hash:
                    raise ValueError("Infrastructure retry lacks linked authorization")
                authorization_path = Path(authorization["path"]).resolve()
                expected_auth_sha = authorization.get("expected_external_file_sha256")
                loaded = _load_retry_authorization(
                    authorization_path,
                    expected_file_sha256=str(expected_auth_sha),
                    identity=identity,
                    prior_failure=_load_failure_receipt(
                        attempts[index - 2]["failure_receipt_path"]
                    ),
                    preflight_receipt_sha256=registry[
                        "preflight_receipt_sha256"
                    ],
                )
                if loaded.get("authorization_sha256") != authorization.get("sha256"):
                    raise ValueError("Retry authorization canonical hash mismatch")
            status = attempt.get("status")
            if status == "failed":
                failure = _load_failure_receipt(attempt["failure_receipt_path"])
                if (
                    failure.get("failure_receipt_sha256")
                    != attempt.get("failure_receipt_sha256")
                    or failure.get("attempt_id") != attempt["attempt_id"]
                    or failure.get("failure_class") != "infrastructure"
                ):
                    raise ValueError(
                        "Only externally authorized infrastructure failures may precede completion"
                    )
                prior_failure_hash = failure["failure_receipt_sha256"]
            elif status == "complete":
                checkpoint_path = Path(attempt["checkpoint_path"]).resolve()
                run_state_path = Path(attempt["run_state_path"]).resolve()
                if (
                    not checkpoint_path.is_file()
                    or sha256_file(checkpoint_path) != attempt["checkpoint_sha256"]
                    or not run_state_path.is_file()
                    or sha256_file(run_state_path) != attempt["run_state_sha256"]
                ):
                    raise ValueError("Completed registry payload hash mismatch")
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
                for key, value in {
                    "formal_run_registry_namespace_sha256": registry[
                        "run_registry_namespace_sha256"
                    ],
                    "formal_run_registry_instance_id": registry[
                        "run_registry_instance_id"
                    ],
                    "formal_initial_run_registry_sha256": registry[
                        "initial_registry_sha256"
                    ],
                    "formal_run_registry_custody_sha256": canonical_json_hash(
                        registry["run_registry_custody"]
                    ),
                    "formal_run_attempt_id": attempt["attempt_id"],
                    "formal_run_reservation_token_sha256": reservation["token"][
                        "reservation_token_sha256"
                    ],
                    "formal_run_reservation_token_file_sha256": reservation[
                        "reservation_token_file_sha256"
                    ],
                    "formal_run_reservation_nonce": reservation["token"][
                        "reservation_nonce"
                    ],
                }.items():
                    if checkpoint.get(key) != value:
                        raise ValueError(
                            f"Completed checkpoint differs from registry attempt for {key}"
                        )
            else:
                raise ValueError(f"Run registry has an open/unknown attempt: {identity}")
            attempt_rows.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "status": status,
                    "reservation_token_file_sha256": reservation[
                        "reservation_token_file_sha256"
                    ],
                    "reservation_token_path": reservation[
                        "reservation_token_path"
                    ],
                    "reservation_token_sha256": reservation["token"][
                        "reservation_token_sha256"
                    ],
                    "reservation_nonce": reservation["token"][
                        "reservation_nonce"
                    ],
                    "failure_receipt_sha256": attempt.get(
                        "failure_receipt_sha256"
                    ),
                    "retry_authorization": attempt.get("retry_authorization"),
                    "checkpoint_sha256": attempt.get("checkpoint_sha256"),
                    "run_state_sha256": attempt.get("run_state_sha256"),
                }
            )
        complete = completed[0]
        slot_receipts.append(
            {
                "run_identity": identity,
                "complete_attempt_id": complete["attempt_id"],
                "checkpoint_path": complete["checkpoint_path"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "attempt_count": len(attempt_rows),
                "attempt_history_sha256": canonical_json_hash(attempt_rows),
                "attempts": attempt_rows,
            }
        )
    core = {
        "schema_version": 1,
        "run_registry_namespace_path": registry[
            "run_registry_namespace_path"
        ],
        "run_registry_namespace_sha256": registry[
            "run_registry_namespace_sha256"
        ],
        "run_registry_instance_id": registry["run_registry_instance_id"],
        "initial_registry_sha256": registry["initial_registry_sha256"],
        "run_registry_custody": registry.get("run_registry_custody"),
        "run_registry_custody_sha256": (
            canonical_json_hash(registry["run_registry_custody"])
            if isinstance(registry.get("run_registry_custody"), Mapping)
            else None
        ),
        "run_registry_path": str(path),
        "run_registry_sha256": registry["registry_sha256"],
        "run_registry_file_sha256": sha256_file(path),
        "run_registry_journal": journal,
        "run_registry_journal_sha256": (
            canonical_json_hash(journal) if journal is not None else None
        ),
        "source_config_sha256": registry["source_config_sha256"],
        "pretraining_master_sha256": registry["pretraining_master_sha256"],
        "pretraining_audit_receipt_sha256": registry[
            "pretraining_audit_receipt_sha256"
        ],
        "pretraining_audit_receipt_file_sha256": registry.get(
            "pretraining_audit_receipt_file_sha256"
        ),
        "expected_run_count": len(expected_identities),
        "slots": slot_receipts,
    }
    return {**core, "run_registry_receipt_sha256": canonical_json_hash(core)}


def freeze_complete_run_registry(
    registry_path: str | Path,
    *,
    expected_config_sha256: str,
    expected_pretraining_master_sha256: str,
    expected_pretraining_audit_receipt_sha256: str,
    expected_run_registry_namespace_sha256: str,
    expected_identities: set[str],
    expected_pretraining_audit_receipt_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Close the ledger exactly once before checkpoint release selection."""
    path = Path(registry_path).resolve()
    with _exclusive_lock(path):
        registry = _load_registry(path)
        if registry.get("status") == "active":
            # Validate the active content first; the only mutation below is the
            # irreversible status transition that prevents later attempts.
            validate_complete_run_registry(
                path,
                expected_config_sha256=expected_config_sha256,
                expected_pretraining_master_sha256=expected_pretraining_master_sha256,
                expected_pretraining_audit_receipt_sha256=(
                    expected_pretraining_audit_receipt_sha256
                ),
                expected_run_registry_namespace_sha256=(
                    expected_run_registry_namespace_sha256
                ),
                expected_identities=expected_identities,
                expected_pretraining_audit_receipt_file_sha256=(
                    expected_pretraining_audit_receipt_file_sha256
                ),
                require_frozen=False,
            )
            registry["status"] = "complete_frozen"
            registry["frozen_at_utc"] = _utc_now()
            registry["last_transition_event"] = {
                "event": "freeze_complete_registry",
                "run_identity": None,
                "attempt_id": None,
            }
            registry["revision"] = int(registry.get("revision", 0)) + 1
            _save_registry(path, registry)
        elif registry.get("status") != "complete_frozen":
            raise ValueError("Run registry has an invalid terminal status")
    return validate_complete_run_registry(
        path,
        expected_config_sha256=expected_config_sha256,
        expected_pretraining_master_sha256=expected_pretraining_master_sha256,
        expected_pretraining_audit_receipt_sha256=(
            expected_pretraining_audit_receipt_sha256
        ),
        expected_run_registry_namespace_sha256=(
            expected_run_registry_namespace_sha256
        ),
        expected_identities=expected_identities,
        expected_pretraining_audit_receipt_file_sha256=(
            expected_pretraining_audit_receipt_file_sha256
        ),
        require_frozen=True,
    )


def build_pipeline_plan(
    *,
    enable_test_evaluation: bool,
    expected_master_release_sha256: str | None,
    expected_test_release_sha256: str | None,
) -> list[str]:
    plan = [
        "tests",
        "controller_grid",
        "preflight",
        "profile",
        "pretraining_master_freeze",
        "pretraining_gate",
        "formal_training",
        "checkpoint_release",
    ]
    anchors = (expected_master_release_sha256, expected_test_release_sha256)
    if enable_test_evaluation:
        if not all(value and len(str(value)) == 64 for value in anchors):
            raise PermissionError(
                "Test evaluation requires explicit 64-hex master and test-release anchors"
            )
        for value in anchors:
            int(str(value), 16)
        plan.append("test_evaluation")
    elif any(anchors):
        raise PermissionError(
            "Release anchors alone do not enable test evaluation; pass the explicit enable flag"
        )
    return plan


def _load_pipeline_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or _state_hash(payload) != payload.get(
        "pipeline_state_sha256"
    ):
        raise ValueError("Pipeline state hash/schema mismatch")
    return payload


def _write_pipeline_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    state["pipeline_state_sha256"] = _state_hash(state)
    atomic_write_json(path, state)


def _new_pipeline_state(
    path: Path,
    plan: list[str],
    config_sha256: str,
    launch_authorizations: Mapping[str, Any],
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Pipeline state already exists: {path}")
    state = {
        "schema_version": 1,
        "pipeline_run_id": str(uuid.uuid4()),
        "runner_process_id": os.getpid(),
        "source_config_sha256": config_sha256,
        "launch_authorizations": dict(launch_authorizations),
        "status": "running",
        "current_stage": None,
        "stage_order": plan,
        "stages": {name: {"status": "pending"} for name in plan},
        "estimated_seconds_remaining": None,
        "created_at_utc": _utc_now(),
    }
    _write_pipeline_state(path, state)
    return state


def _stage_start(path: Path, state: dict[str, Any], name: str, command: list[str]) -> None:
    state["current_stage"] = name
    state["stages"][name] = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "command": command,
    }
    _write_pipeline_state(path, state)


def _stage_finish(
    path: Path, state: dict[str, Any], name: str, *, evidence: Mapping[str, Any] | None = None
) -> None:
    row = state["stages"][name]
    row["status"] = "complete"
    row["completed_at_utc"] = _utc_now()
    if evidence:
        row["evidence"] = dict(evidence)
    state["current_stage"] = None
    _write_pipeline_state(path, state)


def _run_command(
    command: list[str],
    log_path: Path,
    cwd: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existed = log_path.exists()
    with log_path.open("ab") as handle:
        if existed:
            handle.write(
                f"\n--- detached runner resume {_utc_now()} ---\n".encode("utf-8")
            )
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=(dict(env) if env is not None else None),
        )
        handle.flush()
        os.fsync(handle.fileno())
    if completed.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        raise subprocess.CalledProcessError(completed.returncode, command, output=text)
    return completed


def _sample_gpu_temperatures(
    config: Mapping[str, Any], gpu_uuids: Iterable[str]
) -> dict[str, float]:
    """Read temperatures with the byte-anchored NVSMI binary."""
    trusted = config.get("trusted_nvidia_smi")
    if not isinstance(trusted, Mapping):
        raise ValueError("Thermal sampling lacks a trusted nvidia-smi identity")
    executable = Path(str(trusted.get("executable_path", ""))).resolve()
    expected_sha256 = str(trusted.get("executable_sha256", "")).lower()
    if (
        len(expected_sha256) != 64
        or not executable.is_file()
        or sha256_file(executable).lower() != expected_sha256
    ):
        raise PermissionError("Thermal sampler executable differs from its byte anchor")
    completed = subprocess.run(
        [
            str(executable),
            "--query-gpu=uuid,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("Read-only GPU temperature query failed")
    requested = {_canonical_gpu_uuid(value) for value in gpu_uuids}
    observed: dict[str, float] = {}
    for raw_line in completed.stdout.splitlines():
        if not raw_line.strip():
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) != 2:
            raise ValueError("GPU temperature query row is malformed")
        gpu_uuid = _canonical_gpu_uuid(fields[0])
        temperature = float(fields[1])
        if not math.isfinite(temperature):
            raise ValueError("GPU temperature query is non-finite")
        if gpu_uuid in observed:
            raise ValueError("GPU temperature query contains a duplicate UUID")
        observed[gpu_uuid] = temperature
    if not requested.issubset(set(observed)):
        raise ValueError("GPU temperature query omitted an assigned GPU")
    return {gpu_uuid: observed[gpu_uuid] for gpu_uuid in sorted(requested)}


def _remaining_training_seconds(
    registry: Mapping[str, Any],
    *,
    gpu_by_seed: Mapping[str, str],
    projected_seconds_per_run: Mapping[str, int],
) -> int:
    per_gpu: dict[str, int] = {gpu_uuid: 0 for gpu_uuid in set(gpu_by_seed.values())}
    for identity in registry["expected_run_identities"]:
        if any(
            row.get("status") == "complete"
            for row in registry["slots"].get(identity, [])
        ):
            continue
        model, seed_text = identity.split("::seed_", 1)
        gpu_uuid = gpu_by_seed[str(int(seed_text))]
        per_gpu[gpu_uuid] += int(projected_seconds_per_run[model])
    return max(per_gpu.values(), default=0)


def _completed_run_count(registry: Mapping[str, Any]) -> int:
    return sum(
        1
        for attempts in registry["slots"].values()
        if any(row.get("status") == "complete" for row in attempts)
    )


def run_formal_training_scheduler(
    *,
    config: dict[str, Any],
    config_path: Path,
    data_root: Path,
    parameter_registry_path: Path,
    run_registry_path: Path,
    work: Path,
    logs: Path,
    owner_id: str,
    python: str,
    pretraining_master_sha256: str,
    pretraining_audit_path: Path,
    pretraining_audit_sha256: str,
    pretraining_audit_file_sha256: str,
    training_gpu_plan: Mapping[str, Any],
    retry_authorizations: Mapping[str, tuple[Path, str]],
    projected_seconds_per_run: Mapping[str, int],
    state_path: Path,
    state: dict[str, Any],
    popen_factory: Any = subprocess.Popen,
    temperature_sampler: Any = _sample_gpu_temperatures,
    sleep_function: Any = time.sleep,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Run the exact registered grid with one child per assigned physical GPU."""
    gpu_by_seed = dict(training_gpu_plan["training_gpu_by_seed"])
    max_parallel = int(training_gpu_plan["max_parallel_training_runs"])
    thermal = dict(training_gpu_plan["thermal_guard"])
    if max_parallel != 2:
        raise ValueError("Formal scheduler plan no longer permits two workers")
    registry = _load_registry(run_registry_path)
    if (
        registry.get("training_gpu_plan_sha256")
        != training_gpu_plan["training_gpu_plan_sha256"]
        or registry.get("training_gpu_by_seed") != gpu_by_seed
    ):
        raise ValueError("Active run registry differs from the frozen GPU plan")
    tasks = [
        (str(model), int(seed))
        for model in config["matching"]["registered_models"]
        for seed in config["training"]["seeds"]
    ]
    total_runs = len(tasks)
    pending: list[tuple[str, int]] = []
    for model, seed in tasks:
        identity = f"{model}::seed_{seed}"
        if not any(
            row.get("status") == "complete"
            for row in registry["slots"].get(identity, [])
        ):
            pending.append((model, seed))

    active: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    stop_dispatch = False
    thermal_paused = False
    last_temperature_sample: dict[str, float] | None = None
    state_persistence_error: Exception | None = None

    def update_state(*, last_completed: str | None = None) -> None:
        snapshot = _load_registry(run_registry_path)
        stage = state["stages"]["formal_training"]
        stage.update(
            {
                "completed_runs": _completed_run_count(snapshot),
                "total_runs": total_runs,
                "active_workers": [
                    {
                        "run_identity": row["identity"],
                        "assigned_gpu_uuid": gpu_uuid,
                        "child_process_id": int(row["process"].pid),
                        "worker_nonce": row["worker_nonce"],
                    }
                    for gpu_uuid, row in sorted(active.items())
                ],
                "dispatch_stopped_after_failure": bool(stop_dispatch),
                "thermal_dispatch_paused": bool(thermal_paused),
                "latest_gpu_temperatures_c": last_temperature_sample,
            }
        )
        if last_completed is not None:
            stage["last_completed_run_identity"] = last_completed
        state["estimated_seconds_remaining"] = _remaining_training_seconds(
            snapshot,
            gpu_by_seed=gpu_by_seed,
            projected_seconds_per_run=projected_seconds_per_run,
        )
        _write_pipeline_state(state_path, state)

    def safe_update_state(*, last_completed: str | None = None) -> None:
        """Turn a state-write failure into drain-only mode instead of orphaning workers."""
        nonlocal state_persistence_error, stop_dispatch
        if state_persistence_error is not None:
            return
        try:
            update_state(last_completed=last_completed)
        except Exception as exc:
            state_persistence_error = exc
            stop_dispatch = True
            failures.append(
                f"pipeline-state persistence failed: {type(exc).__name__}: {exc}"
            )

    def stop_after_dispatch_error(
        *,
        identity: str,
        exc: Exception,
        reservation: Mapping[str, Any] | None = None,
        log_path: Path | None = None,
        failure_code: str = "process_launch",
    ) -> None:
        """Stop new launches while preserving an immutable attempt failure when possible."""
        nonlocal stop_dispatch
        stop_dispatch = True
        text = f"{type(exc).__name__}: {exc}"
        if reservation is not None:
            destination = log_path or (
                logs
                / (
                    f"train_{reservation['model']}_seed_{reservation['seed']}_"
                    f"attempt_{int(reservation['attempt_number']):03d}.dispatch_error.log"
                )
            )
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("ab") as handle:
                    handle.write((text + "\n").encode("utf-8", errors="replace"))
                    handle.flush()
                    os.fsync(handle.fileno())
                mark_run_failed(
                    run_registry_path,
                    attempt_id=str(reservation["attempt_id"]),
                    owner_id=owner_id,
                    failure_class="infrastructure",
                    failure_code=failure_code,
                    message=text,
                    return_code=None,
                    log_path=destination,
                )
            except Exception as ledger_error:
                text += f"\nFailure-ledger error: {ledger_error}"
        failures.append(f"{identity} dispatch: {text}")

    while pending or active:
        completed_gpus: list[str] = []
        for gpu_uuid, worker in list(active.items()):
            process = worker["process"]
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle = worker["log_handle"]
            log_handle.flush()
            os.fsync(log_handle.fileno())
            log_handle.close()
            identity = worker["identity"]
            reservation = worker["reservation"]
            log_path = worker["log_path"]
            completion_error: Exception | None = worker.get("registration_error")
            try:
                if completion_error is None and int(return_code) == 0:
                    run_state_path = Path(reservation["output_dir"]) / "run_state.json"
                    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
                    mark_run_complete(
                        run_registry_path,
                        attempt_id=reservation["attempt_id"],
                        owner_id=owner_id,
                        checkpoint_path=run_state["best_checkpoint"],
                    )
                elif completion_error is None:
                    completion_error = subprocess.CalledProcessError(
                        int(return_code), worker["command"]
                    )
            except Exception as exc:
                completion_error = exc
            if completion_error is not None:
                stop_dispatch = True
                text = str(completion_error)
                if log_path.is_file():
                    text += "\n" + log_path.read_text(
                        encoding="utf-8", errors="replace"
                    )[-12000:]
                failure_class, failure_code = classify_subprocess_failure(
                    text,
                    int(return_code) if return_code is not None else None,
                )
                if worker.get("registration_error") is not None:
                    failure_class, failure_code = "infrastructure", "process_launch"
                try:
                    mark_run_failed(
                        run_registry_path,
                        attempt_id=reservation["attempt_id"],
                        owner_id=owner_id,
                        failure_class=failure_class,
                        failure_code=failure_code,
                        message=text,
                        return_code=(
                            int(return_code) if return_code is not None else None
                        ),
                        log_path=log_path,
                    )
                except Exception as ledger_error:
                    text += f"\nFailure-ledger error: {ledger_error}"
                failures.append(f"{identity}: {text}")
            completed_gpus.append(gpu_uuid)
            safe_update_state(
                last_completed=(identity if completion_error is None else None)
            )
        for gpu_uuid in completed_gpus:
            active.pop(gpu_uuid, None)
        if completed_gpus:
            safe_update_state()

        if failures:
            stop_dispatch = True
            safe_update_state()
            if not active:
                raise RuntimeError(
                    "Formal training stopped after child failure; no automatic retry. "
                    + " | ".join(failures)
                )

        if pending and not stop_dispatch:
            try:
                last_temperature_sample = temperature_sampler(
                    config, set(gpu_by_seed.values())
                )
                dispatch_allowed, thermal_paused = thermal_dispatch_gate(
                    last_temperature_sample,
                    was_paused=thermal_paused,
                    stop_c=float(thermal["stop_c"]),
                    resume_c=float(thermal["resume_c"]),
                )
            except Exception as exc:
                stop_dispatch = True
                failures.append(f"thermal admission failed: {exc}")
                dispatch_allowed = False
            safe_update_state()
        else:
            dispatch_allowed = False

        launched = False
        while (
            pending
            and not stop_dispatch
            and dispatch_allowed
            and len(active) < max_parallel
        ):
            next_index = next(
                (
                    index
                    for index, (_, seed) in enumerate(pending)
                    if gpu_by_seed[str(seed)] not in active
                ),
                None,
            )
            if next_index is None:
                break
            model, seed = pending.pop(next_index)
            identity = f"{model}::seed_{seed}"
            assigned_gpu = gpu_by_seed[str(seed)]
            retry = retry_authorizations.get(
                re.sub(r"[^A-Za-z0-9_.-]+", "_", identity)
            )
            base = work / "formal_training" / model / f"seed_{seed}"
            try:
                reservation = reserve_run(
                    run_registry_path,
                    model=model,
                    seed=seed,
                    owner_id=owner_id,
                    output_dir=base,
                    assigned_gpu_uuid=assigned_gpu,
                    retry_authorization_path=(retry[0] if retry else None),
                    retry_authorization_expected_sha256=(
                        retry[1] if retry else None
                    ),
                )
            except Exception as exc:
                stop_after_dispatch_error(identity=identity, exc=exc)
                break
            attempt_dir = Path(reservation["output_dir"])
            expected_attempt_dir = base / f"attempt_{reservation['attempt_number']:03d}"
            log_path = logs / (
                f"train_{model}_seed_{seed}_attempt_"
                f"{reservation['attempt_number']:03d}.log"
            )
            if log_path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                log_path = log_path.with_name(
                    f"{log_path.stem}.resume_{stamp}{log_path.suffix}"
                )
            if attempt_dir != expected_attempt_dir:
                stop_after_dispatch_error(
                    identity=identity,
                    exc=ValueError(
                        "Reserved attempt output directory changed unexpectedly"
                    ),
                    reservation=reservation,
                    log_path=log_path,
                    failure_code="filesystem_io",
                )
                break
            command = [
                python,
                "-m",
                "liquid_revision",
                "closed-loop-train",
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--registry",
                str(parameter_registry_path),
                "--model",
                model,
                "--seed",
                str(seed),
                "--output-dir",
                str(attempt_dir),
                "--device",
                "cuda:0",
                "--expected-pretraining-master-sha256",
                pretraining_master_sha256,
                "--pretraining-audit-receipt",
                str(pretraining_audit_path),
                "--expected-pretraining-audit-receipt-sha256",
                pretraining_audit_sha256,
                "--expected-pretraining-audit-receipt-file-sha256",
                pretraining_audit_file_sha256,
                "--run-registry",
                str(run_registry_path),
                "--run-attempt-id",
                reservation["attempt_id"],
                "--run-reservation-token",
                reservation["reservation_token_path"],
                "--expected-run-reservation-token-sha256",
                reservation["reservation_token_file_sha256"],
            ]
            if reservation.get("resumed_existing_attempt") and attempt_dir.exists():
                command.append("--resume")
            worker_nonce = uuid.uuid4().hex
            child_environment = os.environ.copy()
            child_environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": assigned_gpu,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "LIQUID_FORMAL_WORKER_NONCE": worker_nonce,
                }
            )
            log_handle = None
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("ab")
                if log_path.stat().st_size:
                    log_handle.write(
                        f"\n--- formal child launch {_utc_now()} ---\n".encode(
                            "utf-8"
                        )
                    )
                    log_handle.flush()
                mark_run_prelaunch(
                    run_registry_path,
                    attempt_id=reservation["attempt_id"],
                    owner_id=owner_id,
                    log_path=log_path,
                    worker_nonce=worker_nonce,
                )
            except Exception as exc:
                if log_handle is not None and not log_handle.closed:
                    log_handle.close()
                stop_after_dispatch_error(
                    identity=identity,
                    exc=exc,
                    reservation=reservation,
                    log_path=log_path,
                )
                break
            try:
                process = popen_factory(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=child_environment,
                )
            except Exception as exc:
                log_handle.write((str(exc) + "\n").encode("utf-8", errors="replace"))
                log_handle.flush()
                os.fsync(log_handle.fileno())
                log_handle.close()
                stop_after_dispatch_error(
                    identity=identity,
                    exc=exc,
                    reservation=reservation,
                    log_path=log_path,
                )
                break
            worker = {
                "identity": identity,
                "process": process,
                "worker_nonce": worker_nonce,
                "reservation": reservation,
                "command": command,
                "log_path": log_path,
                "log_handle": log_handle,
            }
            active[assigned_gpu] = worker
            try:
                launch_receipt = create_worker_launch_receipt(
                    run_registry_path,
                    reservation=reservation,
                    command=command,
                    child_process_id=int(process.pid),
                    worker_nonce=worker_nonce,
                    child_environment=child_environment,
                )
                mark_run_running(
                    run_registry_path,
                    attempt_id=reservation["attempt_id"],
                    owner_id=owner_id,
                    log_path=log_path,
                    child_process_id=int(process.pid),
                    worker_nonce=worker_nonce,
                    worker_launch_receipt_path=launch_receipt["_validated_path"],
                    worker_launch_receipt_file_sha256=launch_receipt[
                        "_validated_file_sha256"
                    ],
                )
            except Exception as exc:
                worker["registration_error"] = exc
                stop_dispatch = True
                failures.append(f"{identity} worker registration: {exc}")
            launched = True
            safe_update_state()

        if active or (pending and (thermal_paused or stop_dispatch)):
            sleep_function(float(poll_interval_seconds))
        elif pending and not launched:
            raise RuntimeError("Formal scheduler cannot dispatch a pending run")

    _checkpoint_paths_from_registry(run_registry_path)


def _validate_profile(path: Path, config: dict[str, Any], registry_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("config_sha256") != canonical_json_hash(config)
        or payload.get("parameter_registry_sha256") != registry_sha256
        or len(payload.get("rows", [])) != 7
    ):
        raise ValueError("Training profile is absent, stale, or incomplete")
    names = {row.get("registered_name") for row in payload["rows"]}
    if names != set(config.get("matching", {}).get("registered_models", [])):
        raise ValueError("Training profile does not cover the seven registered models")
    if any(float(row.get("seconds_per_optimizer_step", 0)) <= 0 for row in payload["rows"]):
        raise ValueError("Training profile contains an invalid throughput")
    csv_path = path.with_suffix(".csv")
    if not csv_path.is_file() or sha256_file(csv_path) != payload.get("files", {}).get("csv"):
        raise ValueError("Training profile CSV hash mismatch")
    return payload


def _validate_unanchored_pretraining_master(
    path: Path,
    *,
    preflight_receipt_sha256: str,
    config_sha256: str,
    run_registry_namespace_path: Path,
    run_registry_path: Path,
    models: Iterable[str],
    seeds: Iterable[int],
    training_gpu_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    core = _without(payload, "pretraining_master_sha256", "created_at_utc")
    if canonical_json_hash(core) != payload.get("pretraining_master_sha256"):
        raise ValueError("Pretraining master self-hash mismatch")
    if (
        payload.get("source_config_sha256") != config_sha256
        or payload.get("preflight_receipt_sha256") != preflight_receipt_sha256
        or payload.get("approved_for_external_audit_anchor") is not True
        or int(payload.get("test_episodes_used", -1)) != 0
    ):
        raise ValueError("Pretraining master does not bind the completed preflight")
    namespace = validate_run_registry_namespace(
        run_registry_namespace_path,
        run_registry_path=run_registry_path,
        config_sha256=config_sha256,
        preflight_receipt_sha256=preflight_receipt_sha256,
        models=models,
        seeds=seeds,
        expected_namespace_sha256=payload.get(
            "run_registry_namespace_sha256"
        ),
        training_gpu_plan=training_gpu_plan,
    )
    if (
        payload.get("run_registry_namespace_path")
        != str(run_registry_namespace_path.resolve())
        or payload.get("run_registry_path") != str(run_registry_path.resolve())
        or payload.get("expected_run_identities")
        != namespace["expected_run_identities"]
    ):
        raise ValueError("Pretraining master run-registry namespace is stale")
    validate_pretraining_run_registry_binding(
        run_registry_path,
        pretraining_master=payload,
    )
    return payload


def _checkpoint_paths_from_registry(path: Path) -> list[str]:
    registry = _load_registry(path)
    expected = set(registry["expected_run_identities"])
    if set(registry["slots"]) != expected:
        raise ValueError("Formal run registry does not contain every expected slot")
    paths: list[str] = []
    for identity in sorted(expected):
        complete = [
            row for row in registry["slots"][identity] if row.get("status") == "complete"
        ]
        if len(complete) != 1:
            raise ValueError(f"Formal run slot lacks exactly one completion: {identity}")
        paths.append(str(complete[0]["checkpoint_path"]))
    return paths


def _run_pipeline(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config, config_sha256 = load_config(config_path)
    registered_seeds = [int(seed) for seed in config["training"]["seeds"]]
    raw_gpu_assignments = list(args.training_gpu_assignment or [])
    formal_required = bool(config.get("formal", {}).get("required", False))
    if formal_required and not raw_gpu_assignments:
        raise PermissionError(
            "Formal training requires an explicit runtime seed=GPU-UUID plan"
        )
    training_gpu_by_seed = (
        parse_training_gpu_assignments(raw_gpu_assignments, seeds=registered_seeds)
        if raw_gpu_assignments
        else {}
    )
    if formal_required and int(args.max_parallel_training_runs) != 2:
        raise ValueError("Formal dual-GPU training requires max parallel runs = 2")
    if training_gpu_by_seed:
        thermal_dispatch_gate(
            {gpu_uuid: 0.0 for gpu_uuid in set(training_gpu_by_seed.values())},
            was_paused=False,
            stop_c=float(args.thermal_stop_c),
            resume_c=float(args.thermal_resume_c),
        )
    runtime_gpu_plan_request = {
        "training_gpu_by_seed": training_gpu_by_seed,
        "max_parallel_training_runs": int(args.max_parallel_training_runs),
        "thermal_guard": {
            "stop_c": float(args.thermal_stop_c),
            "resume_c": float(args.thermal_resume_c),
        },
    }
    work = Path(args.work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state).resolve()
    plan = build_pipeline_plan(
        enable_test_evaluation=args.enable_test_evaluation,
        expected_master_release_sha256=args.expected_master_release_sha256,
        expected_test_release_sha256=args.expected_test_release_sha256,
    )
    claim_custody_supplied = any(
        (
            args.test_claim_custody_receipt,
            args.expected_test_claim_custody_receipt_sha256,
        )
    )
    claim_custody_authorized = all(
        (
            args.test_claim_custody_receipt,
            args.expected_test_claim_custody_receipt_sha256,
        )
    )
    if args.enable_test_evaluation and not claim_custody_authorized:
        raise PermissionError(
            "Test evaluation requires the external claim-custody receipt and its file anchor"
        )
    if claim_custody_supplied and not args.enable_test_evaluation:
        raise PermissionError(
            "Test claim-custody authorization is rejected unless evaluation is explicit"
        )
    claim_custody_path = (
        Path(args.test_claim_custody_receipt).resolve()
        if claim_custody_authorized
        else None
    )
    claim_custody_file_sha256 = (
        str(args.expected_test_claim_custody_receipt_sha256)
        if claim_custody_authorized
        else None
    )
    if claim_custody_authorized:
        if len(claim_custody_file_sha256) != 64:
            raise PermissionError("Test claim-custody anchor must be SHA-256")
        int(claim_custody_file_sha256, 16)
        if (
            not claim_custody_path.is_file()
            or sha256_file(claim_custody_path) != claim_custody_file_sha256
        ):
            raise PermissionError(
                "Test claim-custody receipt differs from its external anchor"
            )
    supplied_pretraining = any(
        (
            args.expected_pretraining_master_sha256,
            args.pretraining_audit_receipt,
            args.expected_pretraining_audit_receipt_sha256,
            args.expected_pretraining_audit_receipt_file_sha256,
        )
    )
    pretraining_authorized = all(
        (
            args.expected_pretraining_master_sha256,
            args.pretraining_audit_receipt,
            args.expected_pretraining_audit_receipt_sha256,
            args.expected_pretraining_audit_receipt_file_sha256,
        )
    )
    if supplied_pretraining and not pretraining_authorized:
        raise PermissionError(
            "Pretraining authorization requires master anchor, audit receipt, and external receipt anchor"
        )
    pretraining_master_sha256 = (
        str(args.expected_pretraining_master_sha256) if pretraining_authorized else None
    )
    pretraining_audit_path = (
        Path(args.pretraining_audit_receipt).resolve()
        if pretraining_authorized
        else None
    )
    pretraining_audit_sha256 = (
        str(args.expected_pretraining_audit_receipt_sha256)
        if pretraining_authorized
        else None
    )
    pretraining_audit_file_sha256 = (
        str(args.expected_pretraining_audit_receipt_file_sha256)
        if pretraining_authorized
        else None
    )
    if pretraining_authorized:
        if len(pretraining_master_sha256) != 64:
            raise PermissionError("Pretraining master anchor must be 64 hexadecimal characters")
        int(pretraining_master_sha256, 16)
        if len(pretraining_audit_sha256) != 64:
            raise PermissionError(
                "Pretraining audit-receipt anchor must be 64 hexadecimal characters"
            )
        int(pretraining_audit_sha256, 16)
        if len(pretraining_audit_file_sha256) != 64:
            raise PermissionError(
                "Pretraining audit-receipt file anchor must be 64 hexadecimal characters"
            )
        int(pretraining_audit_file_sha256, 16)
        if not pretraining_audit_path.is_file():
            raise FileNotFoundError("Independent pretraining audit receipt is absent")
        if sha256_file(pretraining_audit_path) != pretraining_audit_file_sha256:
            raise PermissionError(
                "Independent audit receipt differs from its external file anchor"
            )
        audit_payload = json.loads(
            pretraining_audit_path.read_text(encoding="utf-8")
        )
        if (
            audit_payload.get("pretraining_audit_receipt_sha256")
            != pretraining_audit_sha256
        ):
            raise PermissionError(
                "Independent audit receipt differs from its external anchor"
            )
    launch_authorizations = {
        "expected_pretraining_master_sha256": pretraining_master_sha256,
        "pretraining_audit_receipt_path": (
            str(pretraining_audit_path) if pretraining_audit_path else None
        ),
        "pretraining_audit_receipt_file_sha256": (
            pretraining_audit_file_sha256
        ),
        "expected_pretraining_audit_receipt_sha256": pretraining_audit_sha256,
        "expected_pretraining_audit_receipt_file_sha256": (
            pretraining_audit_file_sha256
        ),
        "test_evaluation_enabled": bool(args.enable_test_evaluation),
        "expected_master_release_sha256": args.expected_master_release_sha256,
        "expected_test_release_sha256": args.expected_test_release_sha256,
        "test_claim_custody_receipt_path": (
            str(claim_custody_path) if claim_custody_path else None
        ),
        "expected_test_claim_custody_receipt_sha256": (
            claim_custody_file_sha256
        ),
        "runtime_training_gpu_plan_request": runtime_gpu_plan_request,
    }
    if state_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume for existing pipeline state: {state_path}")
        state = _load_pipeline_state(state_path)
        prior_process_id = int(state.get("runner_process_id", -1))
        if prior_process_id != os.getpid() and _pid_is_alive(prior_process_id):
            raise RuntimeError(
                f"Pipeline state is still owned by live process {prior_process_id}"
            )
        existing_authorizations = state.get("launch_authorizations", {})
        authorization_upgrade = (
            state.get("status") == "awaiting_external_test_authorization"
            and bool(args.enable_test_evaluation)
            and state.get("stage_order", []) + ["test_evaluation"] == plan
            and existing_authorizations.get("expected_pretraining_master_sha256")
            == launch_authorizations["expected_pretraining_master_sha256"]
            and existing_authorizations.get("pretraining_audit_receipt_file_sha256")
            == launch_authorizations["pretraining_audit_receipt_file_sha256"]
            and existing_authorizations.get(
                "expected_pretraining_audit_receipt_sha256"
            )
            == launch_authorizations.get(
                "expected_pretraining_audit_receipt_sha256"
            )
            and existing_authorizations.get("runtime_training_gpu_plan_request")
            == launch_authorizations.get("runtime_training_gpu_plan_request")
        )
        pretraining_authorization_upgrade = (
            state.get("status") == "awaiting_external_pretraining_audit"
            and pretraining_authorized
            and state.get("stage_order") == plan
            and existing_authorizations.get("expected_pretraining_master_sha256") is None
            and existing_authorizations.get("pretraining_audit_receipt_file_sha256") is None
            and existing_authorizations.get(
                "expected_pretraining_audit_receipt_sha256"
            ) is None
            and existing_authorizations.get(
                "expected_pretraining_audit_receipt_file_sha256"
            ) is None
            and existing_authorizations.get("test_evaluation_enabled")
            == launch_authorizations.get("test_evaluation_enabled")
            and existing_authorizations.get("expected_master_release_sha256")
            == launch_authorizations.get("expected_master_release_sha256")
            and existing_authorizations.get("expected_test_release_sha256")
            == launch_authorizations.get("expected_test_release_sha256")
            and existing_authorizations.get("runtime_training_gpu_plan_request")
            == launch_authorizations.get("runtime_training_gpu_plan_request")
        )
        if state["source_config_sha256"] != config_sha256:
            raise ValueError("Existing pipeline state differs from the requested config")
        if authorization_upgrade:
            state["stage_order"] = plan
            state["stages"]["test_evaluation"] = {"status": "pending"}
            state["launch_authorizations"] = launch_authorizations
        elif pretraining_authorization_upgrade:
            state["launch_authorizations"] = launch_authorizations
        elif (
            state["stage_order"] != plan
            or existing_authorizations != launch_authorizations
        ):
            raise ValueError("Existing pipeline state differs from the requested design")
        if state.get("status") == "complete" and "test_evaluation" not in plan:
            return 0
        state["runner_process_id"] = os.getpid()
        state["resumed_at_utc"] = _utc_now()
        state["status"] = "running"
        state.pop("failure", None)
        _write_pipeline_state(state_path, state)
    else:
        state = _new_pipeline_state(
            state_path, plan, config_sha256, launch_authorizations
        )
    owner_id = state["pipeline_run_id"]
    logs = work / "launcher_logs"
    data_root = Path(args.data_root).resolve()
    registry_path = Path(args.parameter_registry).resolve()
    preflight_path = Path(args.preflight_receipt).resolve()
    run_registry_path = Path(args.run_registry).resolve()
    profile_path = Path(args.profile_output).resolve()
    device = str(args.device)
    python = sys.executable

    def completed(name: str) -> bool:
        return state["stages"][name].get("status") == "complete"

    try:
        if not completed("tests"):
            command = [python, "-m", "pytest", "-q"]
            _stage_start(state_path, state, "tests", command)
            _run_command(command, logs / "tests.log", PROJECT_ROOT)
            _stage_finish(
                state_path,
                state,
                "tests",
                evidence={"log_sha256": sha256_file(logs / "tests.log")},
            )

        controller_path = _resolve(config["controller_selection"]["artifact"])
        if not completed("controller_grid"):
            if controller_path.exists():
                command = ["validate-existing-controller-grid"]
                _stage_start(state_path, state, "controller_grid", command)
                artifact = validate_controller_selection(config)
            else:
                command = [
                    python,
                    "-m",
                    "liquid_revision",
                    "closed-loop-select-controller",
                    "--config",
                    str(config_path),
                    "--output",
                    str(controller_path),
                    "--device",
                    device,
                ]
                _stage_start(state_path, state, "controller_grid", command)
                _run_command(command, logs / "controller_grid.log", PROJECT_ROOT)
                artifact = validate_controller_selection(config)
            _stage_finish(
                state_path,
                state,
                "controller_grid",
                evidence={
                    "controller_selection_sha256": artifact[
                        "controller_selection_sha256"
                    ]
                },
            )
        if not completed("preflight"):
            if preflight_path.exists():
                command = ["validate-existing-preflight"]
                _stage_start(state_path, state, "preflight", command)
                receipt = validate_preflight_receipt(
                    config=config,
                    data_root=data_root,
                    registry_path=registry_path,
                    receipt_path=preflight_path,
                    device=device,
                )
            else:
                command = [
                    python,
                    "-m",
                    "liquid_revision.closed_loop.preflight",
                    "--config",
                    str(config_path),
                    "--data-root",
                    str(data_root),
                    "--registry",
                    str(registry_path),
                    "--output",
                    str(preflight_path),
                    "--device",
                    device,
                    "--raw-log-root",
                    str(work),
                ]
                _stage_start(state_path, state, "preflight", command)
                _run_command(command, logs / "preflight.log", PROJECT_ROOT)
                receipt = validate_preflight_receipt(
                    config=config,
                    data_root=data_root,
                    registry_path=registry_path,
                    receipt_path=preflight_path,
                    device=device,
                )
            _stage_finish(
                state_path,
                state,
                "preflight",
                evidence={
                    "preflight_receipt_sha256": receipt[
                        "preflight_receipt_sha256"
                    ]
                },
            )
        receipt = load_preflight_receipt(preflight_path)

        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
        if not completed("profile"):
            if profile_path.exists():
                command = ["validate-existing-profile"]
                _stage_start(state_path, state, "profile", command)
                profile = _validate_profile(
                    profile_path, config, registry_payload["registry_sha256"]
                )
            else:
                command = [
                    python,
                    "-m",
                    "liquid_revision",
                    "closed-loop-profile-training",
                    "--config",
                    str(config_path),
                    "--registry",
                    str(registry_path),
                    "--output",
                    str(profile_path),
                    "--device",
                    device,
                    "--steps",
                    str(args.profile_steps),
                    "--warmup-steps",
                    str(args.profile_warmup_steps),
                ]
                _stage_start(state_path, state, "profile", command)
                _run_command(command, logs / "profile.log", PROJECT_ROOT)
                profile = _validate_profile(
                    profile_path, config, registry_payload["registry_sha256"]
                )
            projected_hours = sum(
                float(row["projected_training_hours_per_seed"])
                * len(config["training"]["seeds"])
                for row in profile["rows"]
            )
            state["estimated_seconds_remaining"] = int(projected_hours * 3600)
            _stage_finish(
                state_path,
                state,
                "profile",
                evidence={
                    "profile_sha256": sha256_file(profile_path),
                    "projected_formal_training_hours": projected_hours,
                },
            )
        profile = _validate_profile(
            profile_path, config, registry_payload["registry_sha256"]
        )
        projected_seconds_per_run = {
            str(row["registered_name"]): int(
                math.ceil(float(row["projected_training_hours_per_seed"]) * 3600)
            )
            for row in profile["rows"]
        }

        pretraining_master_value = config.get("formal", {}).get(
            "pretraining_master_artifact"
        )
        if not pretraining_master_value:
            raise ValueError("Formal config lacks formal.pretraining_master_artifact")
        pretraining_master_path = _resolve(pretraining_master_value)
        run_namespace_value = config.get("formal", {}).get(
            "run_registry_namespace_artifact"
        )
        if not run_namespace_value:
            raise ValueError(
                "Formal config lacks formal.run_registry_namespace_artifact"
            )
        run_namespace_path = _resolve(run_namespace_value)
        training_gpu_plan: dict[str, Any] | None = None
        if formal_required:
            frozen_environment = validate_environment_freeze(config)
            if frozen_environment is None:
                raise ValueError("Formal training lacks a frozen environment")
            training_gpu_plan = create_training_gpu_plan(
                run_namespace_path.with_name("training_gpu_plan.json"),
                config_sha256=config_sha256,
                preflight_receipt_sha256=receipt["preflight_receipt_sha256"],
                run_registry_path=run_registry_path,
                models=config["matching"]["registered_models"],
                seeds=registered_seeds,
                assignments=training_gpu_by_seed,
                max_parallel_training_runs=int(args.max_parallel_training_runs),
                thermal_stop_c=float(args.thermal_stop_c),
                thermal_resume_c=float(args.thermal_resume_c),
                environment=frozen_environment,
            )
            state["runtime_training_gpu_plan"] = {
                "path": training_gpu_plan["_validated_path"],
                "training_gpu_plan_sha256": training_gpu_plan[
                    "training_gpu_plan_sha256"
                ],
                "file_sha256": training_gpu_plan["_validated_file_sha256"],
            }
            _write_pipeline_state(state_path, state)
        if not completed("pretraining_master_freeze"):
            command = [
                python,
                "-m",
                "liquid_revision",
                "closed-loop-freeze-pretraining",
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--registry",
                str(registry_path),
                "--preflight-receipt",
                str(preflight_path),
                "--run-registry-namespace",
                str(run_namespace_path),
                "--run-registry",
                str(run_registry_path),
                "--output",
                str(pretraining_master_path),
            ]
            _stage_start(
                state_path, state, "pretraining_master_freeze", command
            )
            if not pretraining_master_path.exists():
                freeze_environment = os.environ.copy()
                if training_gpu_plan is not None:
                    freeze_environment.update(
                        {
                            TRAINING_GPU_PLAN_ENV_PATH: training_gpu_plan[
                                "_validated_path"
                            ],
                            TRAINING_GPU_PLAN_ENV_FILE_SHA256: training_gpu_plan[
                                "_validated_file_sha256"
                            ],
                        }
                    )
                _run_command(
                    command,
                    logs / "pretraining_master_freeze.log",
                    PROJECT_ROOT,
                    env=freeze_environment,
                )
            pretraining_master_unanchored = _validate_unanchored_pretraining_master(
                pretraining_master_path,
                preflight_receipt_sha256=receipt["preflight_receipt_sha256"],
                config_sha256=config_sha256,
                run_registry_namespace_path=run_namespace_path,
                run_registry_path=run_registry_path,
                models=config["matching"]["registered_models"],
                seeds=config["training"]["seeds"],
                training_gpu_plan=training_gpu_plan,
            )
            _stage_finish(
                state_path,
                state,
                "pretraining_master_freeze",
                evidence={
                    "pretraining_master_sha256": pretraining_master_unanchored[
                        "pretraining_master_sha256"
                    ],
                    "requires_external_anchor_and_independent_audit": True,
                },
            )
        else:
            pretraining_master_unanchored = _validate_unanchored_pretraining_master(
                pretraining_master_path,
                preflight_receipt_sha256=receipt["preflight_receipt_sha256"],
                config_sha256=config_sha256,
                run_registry_namespace_path=run_namespace_path,
                run_registry_path=run_registry_path,
                models=config["matching"]["registered_models"],
                seeds=config["training"]["seeds"],
                training_gpu_plan=training_gpu_plan,
            )

        if not pretraining_authorized:
            state["status"] = "awaiting_external_pretraining_audit"
            state["current_stage"] = None
            state["estimated_seconds_remaining"] = None
            _write_pipeline_state(state_path, state)
            return 0
        assert pretraining_master_sha256 is not None
        assert pretraining_audit_path is not None
        if (
            pretraining_master_unanchored["pretraining_master_sha256"]
            != pretraining_master_sha256
        ):
            raise PermissionError(
                "External pretraining anchor differs from the frozen master"
            )
        if not completed("pretraining_gate"):
            command = ["validate-external-pretraining-master-and-audit"]
            _stage_start(state_path, state, "pretraining_gate", command)
            pretraining_master, pretraining_audit = validate_pretraining_master_release(
                config,
                data_root=data_root,
                expected_master_sha256=pretraining_master_sha256,
                audit_receipt_path=pretraining_audit_path,
                expected_audit_receipt_sha256=pretraining_audit_sha256,
                expected_audit_receipt_file_sha256=(
                    pretraining_audit_file_sha256
                ),
                expected_identities={
                    "run_registry_path": str(run_registry_path),
                    "environment_sha256": receipt["environment_sha256"],
                    "controller_selection_sha256": receipt[
                        "controller_selection_sha256"
                    ],
                    "trajectory_manifest_sha256": receipt[
                        "trajectory_manifest_sha256"
                    ],
                    "parameter_registry_sha256": receipt[
                        "parameter_registry_sha256"
                    ],
                    "implementation_sha256": receipt[
                        "identities"
                    ]["source_implementation_sha256"],
                    "tuning_selection_sha256": receipt[
                        "tuning_selection_sha256"
                    ],
                },
            )
            _stage_finish(
                state_path,
                state,
                "pretraining_gate",
                evidence={
                    "pretraining_master_sha256": pretraining_master[
                        "pretraining_master_sha256"
                    ],
                    "pretraining_audit_receipt_sha256": pretraining_audit[
                        "pretraining_audit_receipt_sha256"
                    ],
                },
            )
        else:
            pretraining_master, pretraining_audit = validate_pretraining_master_release(
                config,
                data_root=data_root,
                expected_master_sha256=pretraining_master_sha256,
                audit_receipt_path=pretraining_audit_path,
                expected_audit_receipt_sha256=pretraining_audit_sha256,
                expected_audit_receipt_file_sha256=(
                    pretraining_audit_file_sha256
                ),
                expected_identities={
                    "run_registry_path": str(run_registry_path),
                    "environment_sha256": receipt["environment_sha256"],
                    "controller_selection_sha256": receipt[
                        "controller_selection_sha256"
                    ],
                    "trajectory_manifest_sha256": receipt[
                        "trajectory_manifest_sha256"
                    ],
                    "parameter_registry_sha256": receipt[
                        "parameter_registry_sha256"
                    ],
                    "implementation_sha256": receipt[
                        "identities"
                    ]["source_implementation_sha256"],
                    "tuning_selection_sha256": receipt[
                        "tuning_selection_sha256"
                    ],
                },
            )

        activate_pretraining_run_registry(
            run_registry_path,
            pretraining_master=pretraining_master,
            pretraining_audit_receipt=pretraining_audit,
        )
        if not completed("formal_training"):
            _stage_start(
                state_path,
                state,
                "formal_training",
                ["registered-7x5-training-via-run-ledger"],
            )
            authorizations: dict[str, tuple[Path, str]] = {}
            for value in (args.infrastructure_retry_authorization or []):
                if "@" not in value:
                    raise PermissionError(
                        "Retry authorization must be supplied as PATH@EXTERNAL_FILE_SHA256"
                    )
                raw_path, external_sha256 = value.rsplit("@", 1)
                if len(external_sha256) != 64:
                    raise PermissionError("Retry authorization anchor must be SHA-256")
                int(external_sha256, 16)
                authorization_path = Path(raw_path).resolve()
                authorizations[authorization_path.stem] = (
                    authorization_path,
                    external_sha256,
                )
            if training_gpu_plan is None:
                raise PermissionError("Formal training lacks its runtime GPU plan")
            run_formal_training_scheduler(
                config=config,
                config_path=config_path,
                data_root=data_root,
                parameter_registry_path=registry_path,
                run_registry_path=run_registry_path,
                work=work,
                logs=logs,
                owner_id=owner_id,
                python=python,
                pretraining_master_sha256=pretraining_master_sha256,
                pretraining_audit_path=pretraining_audit_path,
                pretraining_audit_sha256=pretraining_audit_sha256,
                pretraining_audit_file_sha256=pretraining_audit_file_sha256,
                training_gpu_plan=training_gpu_plan,
                retry_authorizations=authorizations,
                projected_seconds_per_run=projected_seconds_per_run,
                state_path=state_path,
                state=state,
            )
            _stage_finish(
                state_path,
                state,
                "formal_training",
                evidence={"run_registry_sha256": _load_registry(run_registry_path)["registry_sha256"]},
            )

        checkpoint_paths = _checkpoint_paths_from_registry(run_registry_path)
        checkpoint_release_path = _resolve(
            config["evaluation"]["checkpoint_release_artifact"]
        )
        if not completed("checkpoint_release"):
            command = [
                python,
                "-m",
                "liquid_revision",
                "closed-loop-freeze-checkpoints",
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--registry",
                str(registry_path),
                "--run-registry",
                str(run_registry_path),
                "--output",
                str(checkpoint_release_path),
                "--expected-pretraining-master-sha256",
                pretraining_master_sha256,
                "--expected-pretraining-audit-receipt-sha256",
                pretraining_audit_sha256,
                "--expected-pretraining-audit-receipt-file-sha256",
                pretraining_audit_file_sha256,
                "--pretraining-audit-receipt",
                str(pretraining_audit_path),
            ]
            for checkpoint in checkpoint_paths:
                command.extend(["--checkpoint", checkpoint])
            _stage_start(state_path, state, "checkpoint_release", command)
            if checkpoint_release_path.exists():
                trajectory = load_trajectory_manifest(
                    data_root / "trajectory_manifest.json", expected_config=config
                )
                release_payload = json.loads(
                    checkpoint_release_path.read_text(encoding="utf-8")
                )
                release = validate_checkpoint_release(
                    config,
                    trajectory,
                    checkpoint_paths,
                    expected_master_release_sha256=release_payload[
                        "checkpoint_release_sha256"
                    ],
                )
            else:
                _run_command(command, logs / "checkpoint_release.log", PROJECT_ROOT)
                trajectory = load_trajectory_manifest(
                    data_root / "trajectory_manifest.json", expected_config=config
                )
                release_payload = json.loads(
                    checkpoint_release_path.read_text(encoding="utf-8")
                )
                release = validate_checkpoint_release(
                    config,
                    trajectory,
                    checkpoint_paths,
                    expected_master_release_sha256=release_payload[
                        "checkpoint_release_sha256"
                    ],
                )
            _stage_finish(
                state_path,
                state,
                "checkpoint_release",
                evidence={
                    "checkpoint_release_sha256": release[
                        "checkpoint_release_sha256"
                    ]
                },
            )

        if "test_evaluation" not in plan:
            state["status"] = "awaiting_external_test_authorization"
            state["current_stage"] = None
            state["estimated_seconds_remaining"] = None
            _write_pipeline_state(state_path, state)
            return 0

        if not completed("test_evaluation"):
            release = json.loads(checkpoint_release_path.read_text(encoding="utf-8"))
            if release.get("checkpoint_release_sha256") != args.expected_master_release_sha256:
                raise PermissionError("External master release anchor does not match")
            test_release_path = _resolve(config["evaluation"]["test_release_artifact"])
            test_release = json.loads(test_release_path.read_text(encoding="utf-8"))
            if test_release.get("test_release_sha256") != args.expected_test_release_sha256:
                raise PermissionError("External test-release anchor does not match")
            trajectory = load_trajectory_manifest(
                data_root / "trajectory_manifest.json", expected_config=config
            )
            checkpoint_release = validate_checkpoint_release(
                config,
                trajectory,
                checkpoint_paths,
                expected_master_release_sha256=args.expected_master_release_sha256,
            )
            output = work / "formal_evaluation"
            validate_test_release(
                config,
                checkpoint_release,
                expected_master_release_sha256=args.expected_master_release_sha256,
                expected_test_release_sha256=args.expected_test_release_sha256,
                evaluation_output_dir=output,
                test_claim_custody_receipt_path=claim_custody_path,
                expected_test_claim_custody_receipt_sha256=(
                    claim_custody_file_sha256
                ),
            )
            command = [
                python,
                "-m",
                "liquid_revision",
                "closed-loop-evaluate",
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output),
                "--device",
                device,
                "--expected-master-release-sha256",
                args.expected_master_release_sha256,
                "--expected-test-release-sha256",
                args.expected_test_release_sha256,
                "--test-claim-custody-receipt",
                str(claim_custody_path),
                "--expected-test-claim-custody-receipt-sha256",
                claim_custody_file_sha256,
            ]
            for checkpoint in checkpoint_paths:
                command.extend(["--checkpoint", checkpoint])
            # A formal claim is written before local output creation.  Pipeline
            # resume must therefore propagate even when a crash occurred in
            # that narrow pre-output window and the output directory is absent.
            if args.resume:
                command.append("--resume")
            _stage_start(state_path, state, "test_evaluation", command)
            _run_command(command, logs / "test_evaluation.log", PROJECT_ROOT)
            _stage_finish(
                state_path,
                state,
                "test_evaluation",
                evidence={"output_directory": str(output)},
            )
        else:
            # A completed pipeline resume is still a deep, read-only
            # verification of every shard/manifest/checkpoint.  Missing or
            # re-hashed payloads must fail instead of trusting pipeline_state.
            if not args.resume:
                raise FileExistsError(
                    "Completed evaluation may only be reopened with --resume"
                )
            output = work / "formal_evaluation"
            command = [
                python,
                "-m",
                "liquid_revision",
                "closed-loop-evaluate",
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output),
                "--device",
                device,
                "--expected-master-release-sha256",
                args.expected_master_release_sha256,
                "--expected-test-release-sha256",
                args.expected_test_release_sha256,
                "--test-claim-custody-receipt",
                str(claim_custody_path),
                "--expected-test-claim-custody-receipt-sha256",
                claim_custody_file_sha256,
                "--resume",
            ]
            for checkpoint in checkpoint_paths:
                command.extend(["--checkpoint", checkpoint])
            _run_command(
                command,
                logs / "test_evaluation_resume_verify.log",
                PROJECT_ROOT,
            )
        state["status"] = "complete"
        state["current_stage"] = None
        state["estimated_seconds_remaining"] = 0
        _write_pipeline_state(state_path, state)
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "failed_at_utc": _utc_now(),
        }
        current = state.get("current_stage")
        if current and state["stages"][current].get("status") == "running":
            state["stages"][current]["status"] = "failed"
            state["stages"][current]["failed_at_utc"] = _utc_now()
            state["stages"][current]["failure"] = str(exc)
        _write_pipeline_state(state_path, state)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--parameter-registry", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--run-registry", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--training-gpu-assignment",
        action="append",
        default=[],
        metavar="SEED=GPU-UUID",
        help=(
            "Runtime-only physical GPU assignment; repeat once for every "
            "registered training seed"
        ),
    )
    parser.add_argument("--max-parallel-training-runs", type=int, default=2)
    parser.add_argument("--thermal-stop-c", type=float, default=82.0)
    parser.add_argument("--thermal-resume-c", type=float, default=70.0)
    parser.add_argument("--expected-pretraining-master-sha256")
    parser.add_argument("--pretraining-audit-receipt")
    parser.add_argument("--expected-pretraining-audit-receipt-sha256")
    parser.add_argument("--expected-pretraining-audit-receipt-file-sha256")
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--profile-warmup-steps", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--infrastructure-retry-authorization", action="append", default=[]
    )
    parser.add_argument("--enable-test-evaluation", action="store_true")
    parser.add_argument("--expected-master-release-sha256")
    parser.add_argument("--expected-test-release-sha256")
    parser.add_argument("--test-claim-custody-receipt")
    parser.add_argument("--expected-test-claim-custody-receipt-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    return _run_pipeline(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
