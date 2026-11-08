"""Fail-closed provenance and release gates for the offline benchmark.

The confirmatory workflow has two deliberately separate stages:

1. train and validation-select every registered checkpoint; and
2. only after the complete checkpoint set is hashed and released, authorize one
   fixed-checkpoint test sweep.

This module contains no test-set metric computation.  It only freezes and
verifies identities and cryptographic hashes.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from liquid_revision.closed_loop.formal import (
    _physical_gpu_inventory,
    current_run_accelerator_identity,
)
from liquid_revision.io_utils import (
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
    source_tree_manifest,
)
from liquid_revision.prepare_data import verify_prepared_data

from .metrics import count_trainable_parameters
from .models import build_offline_model, resolve_hidden_dim


FORMAL_SCHEMA_VERSION = 1
PACKAGE_DISTRIBUTIONS = (
    "liquid-revision",
    "numpy",
    "PyYAML",
    "torch",
    "ncps",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def formal_enabled(config: Mapping[str, Any]) -> bool:
    return bool(dict(config.get("formal", {})).get("enabled", False))


def _run_identifier(dataset: str, model: str, seed: int, config_hash: str) -> str:
    return f"{dataset}-{model}-seed{int(seed)}-{config_hash[:10]}"


def expected_run_identities(
    config: Mapping[str, Any], config_hash: str
) -> list[dict[str, Any]]:
    experiment = dict(config.get("experiment", {}))
    models_config = dict(config.get("models", {}))
    datasets = [
        str(value)
        for value in experiment.get("datasets", ["har", "physionet2012"])
    ]
    models = [
        str(value)
        for value in models_config.get("names", ["lstm", "ode-rnn", "ltc", "cfc"])
    ]
    seeds = [int(value) for value in experiment.get("seeds", [101, 202, 303, 404, 505])]
    if not datasets or not models or not seeds:
        raise ValueError("formal datasets, models, and seeds must all be non-empty")
    if len(datasets) != len(set(datasets)) or len(models) != len(set(models)) or len(
        seeds
    ) != len(set(seeds)):
        raise ValueError("formal datasets, models, and seeds must not contain duplicates")
    identities = [
        {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "run_id": _run_identifier(dataset, model, seed, config_hash),
        }
        for dataset in datasets
        for model in models
        for seed in seeds
    ]
    expected_count = int(
        dict(config.get("formal", {})).get("expected_checkpoint_count", len(identities))
    )
    if expected_count != len(identities):
        raise ValueError(
            "formal.expected_checkpoint_count does not match datasets x models x seeds: "
            f"{expected_count} != {len(identities)}"
        )
    return identities


def stable_environment_fingerprint() -> dict[str, Any]:
    """Return a CVD-invariant host/software identity suitable for hashing.

    The global freeze deliberately uses ``nvidia-smi`` physical inventory and
    never process-visible CUDA enumeration.  The selected visible/physical GPU
    mapping belongs to each run and is recorded by
    :func:`current_run_accelerator_identity`.
    """

    packages: dict[str, str | None] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    payload = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "physical_gpu_inventory": _physical_gpu_inventory(),
        "byteorder": sys.byteorder,
    }
    return {**payload, "sha256": canonical_json_hash(payload)}


def _protocol_identity(config: Mapping[str, Any], config_path: Path) -> list[dict[str, str]]:
    configured = dict(config.get("formal", {})).get(
        "protocol_files", ["../docs/PROTOCOL.md", "../docs/ANALYSIS_PLAN.md"]
    )
    identities: list[dict[str, str]] = []
    for value in configured:
        path = Path(str(value))
        if not path.is_absolute():
            path = (config_path.parent / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"registered protocol file is missing: {path}")
        identities.append({"path": str(path), "sha256": sha256_file(path)})
    if not identities:
        raise ValueError("formal.protocol_files must not be empty")
    return identities


def _prepared_data_identity(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    data_config = dict(config.get("data", {}))
    root = Path(str(data_config.get("root", "revision_data")))
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    verification = verify_prepared_data(root)
    manifest_path = root / "manifest.json"
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "verification": verification,
        "verification_sha256": canonical_json_hash(verification),
    }


def _split_identity_from_files(
    config: Mapping[str, Any], config_path: Path
) -> dict[str, dict[str, Any]]:
    from .splits import load_split_manifest

    data_config = dict(config.get("data", {}))
    manifest_dir = Path(str(data_config.get("manifest_dir", "manifests/offline")))
    if not manifest_dir.is_absolute():
        manifest_dir = (config_path.parent / manifest_dir).resolve()
    names = {"har": "har_split.json", "physionet2012": "physionet2012_split.json"}
    selected = [
        str(value)
        for value in dict(config.get("experiment", {})).get(
            "datasets", ["har", "physionet2012"]
        )
    ]
    result: dict[str, dict[str, Any]] = {}
    for dataset in selected:
        if dataset not in names:
            raise ValueError(f"unsupported formal dataset: {dataset}")
        path = manifest_dir / names[dataset]
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = load_split_manifest(path)
        result[dataset] = {
            "path": str(path.resolve()),
            "file_sha256": sha256_file(path),
            "manifest_digest": manifest.digest,
            "split_seed": int(manifest.split_seed),
        }
    return result


def _input_dim(train_arrays: Any, channels: list[str]) -> int:
    feature_dim = int(train_arrays.values.shape[-1])
    widths = {"values": feature_dim, "mask": feature_dim, "age": feature_dim, "dt": 1}
    unknown = [name for name in channels if name not in widths]
    if unknown:
        raise ValueError(f"unknown registered feature channels: {unknown}")
    return int(sum(widths[name] for name in channels))


def build_parameter_registry(
    datasets: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    """Build the task-specific parameter registry from training arrays only."""

    model_config = dict(config.get("models", {}))
    channels = [
        str(value)
        for value in dict(config.get("features", {})).get(
            "channels", ["values", "mask", "age", "dt"]
        )
    ]
    models = [str(value) for value in model_config.get("names", [])]
    tolerance = float(model_config.get("parameter_matching_tolerance_percent", 5.0))
    if tolerance < 0:
        raise ValueError("parameter matching tolerance must be non-negative")
    output_dims = {"har": 6, "physionet2012": 1}
    entries: list[dict[str, Any]] = []
    for dataset_name, data in sorted(datasets.items()):
        if dataset_name not in output_dims:
            raise ValueError(f"unsupported parameter-registry dataset: {dataset_name}")
        input_dim = _input_dim(data["train"], channels)
        counts: dict[str, int] = {}
        widths: dict[str, int] = {}
        for model_name in models:
            hidden_dim = resolve_hidden_dim(model_config, dataset_name, model_name)
            model = build_offline_model(
                model_name,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dims[dataset_name],
                ode_max_step=float(model_config.get("ode_max_step", 1.0)),
                ltc_ode_unfolds=int(model_config.get("ltc_ode_unfolds", 6)),
                cfc_mode=str(model_config.get("cfc_mode", "default")),
            )
            widths[model_name] = hidden_dim
            counts[model_name] = count_trainable_parameters(model)
        if "lstm" not in counts:
            raise ValueError("the formal parameter registry requires the LSTM target")
        target = counts["lstm"]
        for model_name in models:
            deviation = 100.0 * abs(counts[model_name] - target) / target
            if deviation > tolerance + 1e-12:
                raise ValueError(
                    f"parameter mismatch for {dataset_name}/{model_name}: "
                    f"{deviation:.6f}% > {tolerance:.6f}%"
                )
            entries.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "input_dim": input_dim,
                    "output_dim": output_dims[dataset_name],
                    "hidden_dim": widths[model_name],
                    "trainable_parameters": counts[model_name],
                    "lstm_target_parameters": target,
                    "deviation_percent": deviation,
                    "within_tolerance": True,
                }
            )
    core = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "config_sha256": config_hash,
        "feature_channels": channels,
        "tolerance_percent": tolerance,
        "entries": entries,
    }
    return {**core, "parameter_registry_sha256": canonical_json_hash(core)}


def _write_once_or_verify(path: Path, payload: Mapping[str, Any], hash_key: str) -> None:
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != dict(payload):
            raise ValueError(f"frozen artifact identity mismatch; refusing overwrite: {path}")
        core = {key: value for key, value in observed.items() if key not in {hash_key, "created_at_utc"}}
        if observed.get(hash_key) != canonical_json_hash(core):
            raise ValueError(f"frozen artifact self-hash mismatch: {path}")
        return
    atomic_write_json(path, dict(payload))


def create_formal_freeze(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_hash: str,
    datasets: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """Freeze all non-outcome identities before confirmatory training begins."""

    if not formal_enabled(config):
        return {"enabled": False}
    output_root.mkdir(parents=True, exist_ok=True)
    expected = expected_run_identities(config, config_hash)
    registry = build_parameter_registry(datasets, config, config_hash)
    registry_path = output_root / "parameter_registry.json"
    _write_once_or_verify(
        registry_path, registry, "parameter_registry_sha256"
    )
    source = source_tree_manifest(Path(__file__).resolve().parents[1])
    environment = stable_environment_fingerprint()
    requested_device = str(
        dict(config.get("experiment", {})).get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    if requested_device.startswith("cuda") and not environment["physical_gpu_inventory"]:
        raise RuntimeError(
            "formal CUDA freeze requires a physical nvidia-smi inventory independent "
            "of CUDA_VISIBLE_DEVICES"
        )
    core = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_hash,
        "source_tree": source,
        "environment": environment,
        "prepared_data": _prepared_data_identity(config, config_path),
        "split_manifests": _split_identity_from_files(config, config_path),
        "protocol_files": _protocol_identity(config, config_path),
        "parameter_registry_path": str(registry_path.resolve()),
        "parameter_registry_file_sha256": sha256_file(registry_path),
        "parameter_registry_sha256": registry["parameter_registry_sha256"],
        "expected_runs": expected,
        "expected_checkpoint_count": len(expected),
        "test_metrics_computed": False,
    }
    payload = {
        **core,
        "formal_freeze_sha256": canonical_json_hash(core),
        "created_at_utc": utc_now(),
    }
    path = output_root / "formal_freeze.json"
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        observed_core = {
            key: value
            for key, value in observed.items()
            if key not in {"formal_freeze_sha256", "created_at_utc"}
        }
        if observed.get("formal_freeze_sha256") != canonical_json_hash(observed_core):
            raise ValueError("existing formal freeze has an invalid self-hash")
        if observed_core != core:
            raise ValueError("current code/environment/data/protocol identity differs from formal freeze")
        return observed
    atomic_write_json(path, payload)
    return payload


def verify_formal_freeze(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    """Rehash every frozen identity without loading prediction outcomes."""

    path = output_root / "formal_freeze.json"
    if not path.is_file():
        raise PermissionError("formal_freeze.json is absent; test access is forbidden")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    frozen_core = {
        key: value
        for key, value in frozen.items()
        if key not in {"formal_freeze_sha256", "created_at_utc"}
    }
    if frozen.get("formal_freeze_sha256") != canonical_json_hash(frozen_core):
        raise ValueError("formal freeze self-hash mismatch")
    registry_path = Path(str(frozen["parameter_registry_path"]))
    current_static = {
        "config_path": str(config_path.resolve()),
        "config_sha256": config_hash,
        "source_tree": source_tree_manifest(Path(__file__).resolve().parents[1]),
        "environment": stable_environment_fingerprint(),
        "prepared_data": _prepared_data_identity(config, config_path),
        "split_manifests": _split_identity_from_files(config, config_path),
        "protocol_files": _protocol_identity(config, config_path),
        "parameter_registry_path": str(registry_path.resolve()),
        "parameter_registry_file_sha256": sha256_file(registry_path),
        "expected_runs": expected_run_identities(config, config_hash),
    }
    for key, value in current_static.items():
        if frozen.get(key) != value:
            raise ValueError(f"formal freeze identity changed at {key}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_core = {
        key: value for key, value in registry.items() if key != "parameter_registry_sha256"
    }
    if registry.get("parameter_registry_sha256") != canonical_json_hash(registry_core):
        raise ValueError("parameter registry self-hash mismatch")
    if frozen.get("parameter_registry_sha256") != registry["parameter_registry_sha256"]:
        raise ValueError("parameter registry semantic hash differs from formal freeze")
    if int(frozen.get("expected_checkpoint_count", -1)) != len(current_static["expected_runs"]):
        raise ValueError("formal freeze expected-checkpoint count is inconsistent")
    if bool(frozen.get("test_metrics_computed")):
        raise ValueError("formal training freeze must not contain test metrics")
    return frozen


def completion_payload(
    *, result_path: Path, best_path: Path, latest_path: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    core = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "run_id": result["run_id"],
        "dataset": result["dataset"],
        "model": result["model"],
        "seed": int(result["seed"]),
        "status": result["status"],
        "config_sha256": result["config_sha256"],
        "formal_freeze_sha256": result.get("formal_freeze_sha256"),
        "optimizer_steps_planned": int(result["optimizer_steps_planned"]),
        "optimizer_steps_completed": int(result["optimizer_steps_completed"]),
        "test_metrics_computed": bool(result.get("test_metrics_computed", False)),
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "best_checkpoint_path": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "latest_checkpoint_path": str(latest_path.resolve()),
        "latest_checkpoint_sha256": sha256_file(latest_path),
    }
    return {**core, "completion_sha256": canonical_json_hash(core)}


def write_completion_artifact(
    *, result_path: Path, best_path: Path, latest_path: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    payload = completion_payload(
        result_path=result_path, best_path=best_path, latest_path=latest_path, result=result
    )
    atomic_write_json(result_path.parent / "completion.json", payload)
    return payload


def verify_completed_run(
    *,
    run_dir: Path,
    identity: Mapping[str, Any],
    config_hash: str,
    formal_freeze_sha256: str | None,
) -> dict[str, Any]:
    """Rehash a completed checkpoint before reuse or release."""

    result_path = run_dir / "result.json"
    completion_path = run_dir / "completion.json"
    if not result_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"run is not completion-gated: {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion_core = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    if completion.get("completion_sha256") != canonical_json_hash(completion_core):
        raise ValueError(f"completion self-hash mismatch: {run_dir}")
    for key in ("run_id", "dataset", "model", "seed"):
        expected = identity[key]
        observed = result.get(key)
        if key == "seed":
            observed = int(observed)
        if observed != expected or completion.get(key) != expected:
            raise ValueError(f"completed run identity mismatch at {key}: {run_dir}")
    if result.get("config_sha256") != config_hash or completion.get("config_sha256") != config_hash:
        raise ValueError(f"completed run config mismatch: {run_dir}")
    if formal_freeze_sha256 is not None and (
        result.get("formal_freeze_sha256") != formal_freeze_sha256
        or completion.get("formal_freeze_sha256") != formal_freeze_sha256
    ):
        raise ValueError(f"completed run formal-freeze mismatch: {run_dir}")
    if result.get("status") != "checkpoint_frozen" or completion.get("status") != "checkpoint_frozen":
        raise ValueError(f"run is not checkpoint_frozen: {run_dir}")
    if bool(result.get("test_metrics_computed")) or bool(completion.get("test_metrics_computed")):
        raise ValueError(f"training-stage artifact contains test metrics: {run_dir}")
    planned = int(result.get("optimizer_steps_planned", -1))
    completed = int(result.get("optimizer_steps_completed", -2))
    if planned <= 0 or planned != completed:
        raise ValueError(f"optimizer budget incomplete: {run_dir}")
    paths_and_hashes = (
        (result_path, "result_sha256"),
        (run_dir / "best.pt", "best_checkpoint_sha256"),
        (run_dir / "latest.pt", "latest_checkpoint_sha256"),
    )
    for path, key in paths_and_hashes:
        if not path.is_file() or sha256_file(path) != completion.get(key):
            raise ValueError(f"completed artifact rehash failed at {path}")
    expected_paths = {
        "result_path": str(result_path.resolve()),
        "best_checkpoint_path": str((run_dir / "best.pt").resolve()),
        "latest_checkpoint_path": str((run_dir / "latest.pt").resolve()),
    }
    for key, expected_path in expected_paths.items():
        if completion.get(key) != expected_path:
            raise ValueError(f"completed artifact path mismatch at {key}: {run_dir}")
    if result.get("best_checkpoint_sha256") != completion["best_checkpoint_sha256"]:
        raise ValueError(f"result/completion checkpoint hash mismatch: {run_dir}")
    return {"result": result, "completion": completion}


def publish_checkpoint_release(
    *,
    config: Mapping[str, Any],
    config_hash: str,
    output_root: Path,
    formal_freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a release only after the exact complete checkpoint set verifies."""

    identities = expected_run_identities(config, config_hash)
    entries: list[dict[str, Any]] = []
    freeze_hash = str(formal_freeze["formal_freeze_sha256"])
    for identity in identities:
        run_dir = output_root / "runs" / str(identity["run_id"])
        verified = verify_completed_run(
            run_dir=run_dir,
            identity=identity,
            config_hash=config_hash,
            formal_freeze_sha256=freeze_hash,
        )
        completion = verified["completion"]
        entries.append(
            {
                **dict(identity),
                "result_path": completion["result_path"],
                "result_sha256": completion["result_sha256"],
                "completion_path": str((run_dir / "completion.json").resolve()),
                "completion_file_sha256": sha256_file(run_dir / "completion.json"),
                "completion_sha256": completion["completion_sha256"],
                "best_checkpoint_path": completion["best_checkpoint_path"],
                "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
            }
        )
    if len(entries) != len(identities):
        raise AssertionError("checkpoint release is incomplete")
    core = {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "config_sha256": config_hash,
        "formal_freeze_sha256": freeze_hash,
        "expected_checkpoint_count": len(identities),
        "released_checkpoint_count": len(entries),
        "test_metrics_computed": False,
        "entries": entries,
    }
    payload = {
        **core,
        "checkpoint_release_sha256": canonical_json_hash(core),
        "created_at_utc": utc_now(),
    }
    path = output_root / "checkpoint_release.json"
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        observed_core = {
            key: value
            for key, value in observed.items()
            if key not in {"checkpoint_release_sha256", "created_at_utc"}
        }
        if observed.get("checkpoint_release_sha256") != canonical_json_hash(observed_core):
            raise ValueError("existing checkpoint release self-hash mismatch")
        if observed_core != core:
            raise ValueError("checkpoint release differs from rehashed completed artifacts")
        return observed
    atomic_write_json(path, payload)
    return payload


def verify_checkpoint_release(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    """Return a test authorization only after rehashing freeze and all runs."""

    freeze = verify_formal_freeze(
        config=config,
        config_path=config_path,
        config_hash=config_hash,
        output_root=output_root,
    )
    path = output_root / "checkpoint_release.json"
    if not path.is_file():
        raise PermissionError("all checkpoints are not released; test access is forbidden")
    release = json.loads(path.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in release.items()
        if key not in {"checkpoint_release_sha256", "created_at_utc"}
    }
    if release.get("checkpoint_release_sha256") != canonical_json_hash(core):
        raise ValueError("checkpoint release self-hash mismatch")
    expected = expected_run_identities(config, config_hash)
    observed_ids = [entry.get("run_id") for entry in release.get("entries", [])]
    expected_ids = [entry["run_id"] for entry in expected]
    if observed_ids != expected_ids or len(observed_ids) != int(
        release.get("expected_checkpoint_count", -1)
    ):
        raise ValueError("checkpoint release does not contain the exact registered run set")
    if release.get("formal_freeze_sha256") != freeze["formal_freeze_sha256"]:
        raise ValueError("checkpoint release is bound to a different formal freeze")
    for identity, entry in zip(expected, release["entries"], strict=True):
        run_dir = output_root / "runs" / identity["run_id"]
        verified = verify_completed_run(
            run_dir=run_dir,
            identity=identity,
            config_hash=config_hash,
            formal_freeze_sha256=freeze["formal_freeze_sha256"],
        )
        completion_path = run_dir / "completion.json"
        completion = verified["completion"]
        checks = {
            "result_sha256": completion["result_sha256"],
            "completion_file_sha256": sha256_file(completion_path),
            "completion_sha256": completion["completion_sha256"],
            "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        }
        for key, value in checks.items():
            if entry.get(key) != value:
                raise ValueError(f"checkpoint release rehash mismatch at {identity['run_id']}:{key}")
    return {
        "authorized": True,
        "formal_freeze_sha256": freeze["formal_freeze_sha256"],
        "checkpoint_release_sha256": release["checkpoint_release_sha256"],
        "release": release,
    }
