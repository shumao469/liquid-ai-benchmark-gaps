"""Immutable episode-level train/validation/test manifests and data materialization."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..data.closed_loop import (
    _make_environment,
    generate_closed_loop_dataset,
    load_closed_loop_dataset,
)
from ..io_utils import (
    atomic_write_json,
    canonical_json_hash,
    sha256_file,
    source_tree_manifest,
)
from ..perturbations import PerturbationSpec, make_perturbation_schedule


DEFAULT_SPLITS = {
    "train": {"episodes": 2000, "episode_seed_start": 100_000, "schedule_seed_start": 400_000},
    "val": {"episodes": 200, "episode_seed_start": 200_000, "schedule_seed_start": 500_000},
    "test": {"episodes": 200, "episode_seed_start": 300_000, "schedule_seed_start": 600_000},
}

RUNTIME_COMPATIBILITY_REVIEW = (
    Path(__file__).resolve().parents[3] / "private-runtime-evidence"
    / "shared_runtime_compatibility_review.json"
)
ORIGINAL_FULL_REPLAY_REVIEW = RUNTIME_COMPATIBILITY_REVIEW.with_name("provenance_repair_review.json")
ORIGINAL_FULL_REPLAY_REVIEW_SHA256 = "c179a367954158be5098f0fdda732e5ed4a7f1ab981eeb229e4c9c75178f65c2"


def _validated_runtime_compatibility(
    directory: Path, complete: dict[str, Any], config: dict[str, Any],
    current_implementation: dict[str, Any],
) -> dict[str, Any]:
    """Reuse original bytes only under an exact externally reviewed runtime successor.

    Generation implementation is retained, not relabelled as the new runtime.
    The original full replay still runs; only its implementation field can be
    normalized after this narrowly bound review, never its arrays or semantics.
    """
    anchor = complete.get("runtime_compatibility_review", {})
    path = Path(str(anchor.get("path", ""))).resolve()
    if (path != RUNTIME_COMPATIBILITY_REVIEW.resolve() or not path.is_file()
            or sha256_file(path) != anchor.get("file_sha256")):
        raise ValueError("Runtime compatibility requires the fixed externally anchored review")
    review = json.loads(path.read_text(encoding="utf-8"))
    if (review.get("status") != "PASS"
            or review.get("purpose") != "independent_runtime_only_data_reuse_review"
            or review.get("numeric_design_unchanged") is not True
            or review.get("regeneration_performed") is not False
            or review.get("config_sha256") != canonical_json_hash(config)
            or review.get("current_runtime_implementation") != current_implementation
            or review.get("original_generation_implementation_sha256")
            != complete.get("dataset_implementation_sha256")
            or review.get("dataset_sha256") != complete.get("dataset_sha256")):
        raise ValueError("Runtime compatibility review does not bind this exact data/runtime design")
    policy = review.get("runtime_policy", {})
    policy_path = Path(str(policy.get("path", ""))).resolve()
    expected_policy = Path(__file__).resolve().parents[3] / "control/shared_gpu_training_policy_20261105.json"
    if (policy_path != expected_policy.resolve() or not policy_path.is_file()
            or sha256_file(policy_path) != policy.get("file_sha256")
            or canonical_json_hash(json.loads(policy_path.read_text(encoding="utf-8")))
            != policy.get("policy_sha256")):
        raise ValueError("Reviewed shared runtime policy changed")
    prior = review.get("prior_full_replay_review", {})
    prior_path = Path(str(prior.get("path", "")))
    if (prior_path.resolve() != ORIGINAL_FULL_REPLAY_REVIEW.resolve()
            or prior.get("file_sha256") != ORIGINAL_FULL_REPLAY_REVIEW_SHA256
            or not prior_path.is_file() or sha256_file(prior_path) != prior.get("file_sha256")
            or json.loads(prior_path.read_text(encoding="utf-8")).get("status") != "PASS"):
        raise ValueError("Original independent full-replay review is not preserved")
    excluded = {"config_sha256", "source_trajectory_manifest_sha256", "manifest_sha256", "created_at_utc"}
    for split in ("train", "val"):
        parent = review.get("original_dataset_manifests", {}).get(split, {})
        parent_path = Path(str(parent.get("path", "")))
        expected_parent = (Path(__file__).resolve().parents[3] /
            f"reports/shared_gpu_20261105/preserved_quiet_namespace/runs/closed_loop_confirmatory/data/{split}/manifest.json")
        if (parent_path.resolve() != expected_parent.resolve() or not parent_path.is_file()
                or sha256_file(parent_path) != parent.get("file_sha256")):
            raise ValueError("Original generation manifest no longer matches its independent anchor")
        original = json.loads(parent_path.read_text(encoding="utf-8"))
        current = json.loads((directory / split / "manifest.json").read_text(encoding="utf-8"))
        if ({k: v for k, v in original.items() if k not in excluded}
                != {k: v for k, v in current.items() if k not in excluded}
                or original.get("implementation", {}).get("aggregate_sha256")
                != review["original_generation_implementation_sha256"]
                or sha256_file(directory / split / "demonstrations.npz") != review["dataset_sha256"][split]):
            raise ValueError("Reused dataset bytes, generation implementation or numerical metadata changed")
    return review


def _split_entries(name: str, mapping: dict[str, Any]) -> list[dict[str, int | str]]:
    count = int(mapping["episodes"])
    if count <= 0:
        raise ValueError(f"splits.{name}.episodes must be positive")
    episode_start = int(mapping["episode_seed_start"])
    schedule_start = int(mapping["schedule_seed_start"])
    return [
        {
            "trajectory_id": f"{name}-{index:06d}",
            "episode_seed": episode_start + index,
            "schedule_seed": schedule_start + index,
        }
        for index in range(count)
    ]


def _freeze_training_corruptions(
    splits: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, Any] | None:
    corruption = config.get("training_corruption")
    if corruption is None:
        return None
    corruption = dict(corruption)
    clean_fraction = float(corruption.get("clean_fraction", 0.5))
    probability_range = tuple(
        float(value)
        for value in corruption.get("dropout_probability_range", [0.05, 0.20])
    )
    burst_lengths = tuple(int(value) for value in corruption.get("burst_lengths", [2, 3]))
    seed = int(corruption.get("seed", 20261104))
    if not 0 <= clean_fraction <= 1:
        raise ValueError("training_corruption.clean_fraction must be in [0,1]")
    if len(probability_range) != 2 or not 0 <= probability_range[0] <= probability_range[1] <= 1:
        raise ValueError("Invalid training corruption dropout range")
    if not burst_lengths or any(length <= 0 for length in burst_lengths):
        raise ValueError("training corruption burst lengths must be positive")
    base_spec = dict(config.get("dataset", {}).get("perturbation", {}))
    summaries: dict[str, dict[str, int]] = {}
    for split_offset, split_name in enumerate(("train", "val")):
        rows = splits[split_name]
        rng = np.random.default_rng(seed + split_offset)
        order = rng.permutation(len(rows))
        clean_count = int(round(clean_fraction * len(rows)))
        corrupted = len(rows) - clean_count
        iid_count = corrupted // 2
        labels = np.full(len(rows), "burst", dtype=object)
        labels[order[:clean_count]] = "clean"
        labels[order[clean_count : clean_count + iid_count]] = "iid"
        for row_index, row in enumerate(rows):
            kind = str(labels[row_index])
            spec = dict(base_spec)
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
            target_fraction = 0.0
            if kind != "clean":
                target_fraction = float(rng.uniform(*probability_range))
            if kind == "iid":
                spec["iid_dropout_probability"] = target_fraction
            elif kind == "burst":
                length = int(rng.choice(burst_lengths))
                spec["burst_length_steps"] = length
                spec["burst_stationary_probability"] = target_fraction
            row["training_corruption"] = {
                "kind": kind,
                "target_missing_fraction": target_fraction,
                "spec": spec,
            }
        summaries[split_name] = {
            kind: sum(row["training_corruption"]["kind"] == kind for row in rows)
            for kind in ("clean", "iid", "burst")
        }
    return {
        "seed": seed,
        "clean_fraction": clean_fraction,
        "dropout_probability_range": list(probability_range),
        "burst_lengths": list(burst_lengths),
        "counts": summaries,
        "test_corruption_frozen_separately": True,
    }


def build_trajectory_manifest(
    config: dict[str, Any],
    *,
    formal_artifact_identities: dict[str, str] | None = None,
) -> dict[str, Any]:
    split_config = dict(config.get("splits") or DEFAULT_SPLITS)
    if set(split_config) != {"train", "val", "test"}:
        raise ValueError("splits must contain exactly train, val, and test")
    splits = {
        name: _split_entries(name, dict(split_config[name]))
        for name in ("train", "val", "test")
    }
    corruption_manifest = _freeze_training_corruptions(splits, config)
    all_episode_seeds = [int(row["episode_seed"]) for rows in splits.values() for row in rows]
    all_schedule_seeds = [int(row["schedule_seed"]) for rows in splits.values() for row in rows]
    if len(all_episode_seeds) != len(set(all_episode_seeds)):
        raise ValueError("Episode seeds overlap across splits")
    if len(all_schedule_seeds) != len(set(all_schedule_seeds)):
        raise ValueError("Schedule seeds overlap across splits")
    core = {
        "schema_version": 1,
        "split_unit": "trajectory_episode",
        "frozen_before_training": True,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "splits": splits,
        "source_config_sha256": canonical_json_hash(config),
        "training_corruption_manifest": corruption_manifest,
        "formal_artifact_identities": formal_artifact_identities,
    }
    return {
        **core,
        "manifest_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _split_dataset_config(
    config: dict[str, Any],
    split_name: str,
    rows: list[dict[str, Any]],
    *,
    trajectory_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive the only admissible materialization config for one frozen split."""
    if split_name not in {"train", "val"}:
        raise ValueError("Only train and validation datasets may be materialized before test release")
    derived = json.loads(json.dumps(config))
    derived["dataset"] = dict(derived.get("dataset", {}))
    default_spec = derived["dataset"].get("perturbation", {})
    derived["dataset"].update(
        {
            "episodes": len(rows),
            "episode_seeds": [int(row["episode_seed"]) for row in rows],
            "schedule_seeds": [int(row["schedule_seed"]) for row in rows],
            "trajectory_ids": [str(row["trajectory_id"]) for row in rows],
            "episode_perturbations": [
                row.get("training_corruption", {}).get("spec", default_spec)
                for row in rows
            ],
            "source_split": split_name,
            "source_trajectory_rows": rows,
            "source_trajectory_manifest_sha256": trajectory_manifest_sha256,
        }
    )
    return derived


def _manifest_core(payload: dict[str, Any], hash_key: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {hash_key, "created_at_utc"}
    }


def create_trajectory_manifest(config: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite split manifest: {destination}")
    manifest = build_trajectory_manifest(config)
    atomic_write_json(destination, manifest)
    return manifest


def load_trajectory_manifest(
    path: str | Path, *, expected_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    source = Path(path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    core = _manifest_core(manifest, "manifest_sha256")
    observed = canonical_json_hash(core)
    if observed != manifest.get("manifest_sha256"):
        raise ValueError(f"Trajectory manifest hash mismatch: {source}")
    if set(manifest.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("Trajectory manifest must contain exactly train, val, and test")
    counts = manifest.get("counts", {})
    for name in ("train", "val", "test"):
        rows = manifest["splits"][name]
        if int(counts.get(name, -1)) != len(rows) or not rows:
            raise ValueError(f"Trajectory manifest count mismatch for {name}")
        expected_ids = [f"{name}-{index:06d}" for index in range(len(rows))]
        if [str(row.get("trajectory_id")) for row in rows] != expected_ids:
            raise ValueError(f"Trajectory manifest ID ordering mismatch for {name}")
    episode_sets = {
        name: {int(row["episode_seed"]) for row in rows}
        for name, rows in manifest["splits"].items()
    }
    if any(
        episode_sets[left] & episode_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Trajectory manifest contains cross-split leakage")
    schedule_sets = {
        name: {int(row["schedule_seed"]) for row in rows}
        for name, rows in manifest["splits"].items()
    }
    if any(
        schedule_sets[left] & schedule_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Trajectory manifest contains cross-split schedule leakage")
    if expected_config is not None:
        from .formal import formal_artifact_identities

        expected = build_trajectory_manifest(
            expected_config,
            formal_artifact_identities=formal_artifact_identities(expected_config),
        )
        expected_core = _manifest_core(expected, "manifest_sha256")
        if core != expected_core:
            raise ValueError(
                "Trajectory manifest differs from the deterministic rebuild of the frozen config"
            )
    return manifest


def prepare_split_datasets(config: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Atomically generate all split datasets from one frozen manifest."""
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite closed-loop data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    from .formal import formal_artifact_identities

    frozen_identities = formal_artifact_identities(config)
    manifest = build_trajectory_manifest(
        config,
        formal_artifact_identities=frozen_identities,
    )
    staging.mkdir(parents=False, exist_ok=False)
    try:
        atomic_write_json(staging / "trajectory_manifest.json", manifest)
        dataset_hashes: dict[str, str] = {}
        dataset_manifest_hashes: dict[str, str] = {}
        implementation_hashes: set[str] = set()
        # Test rows are committed above but intentionally not materialized.
        # No test schedule, state target or NPZ may be opened before the
        # all-checkpoint release and separate test authorization.
        for split_name in ("train", "val"):
            rows = manifest["splits"][split_name]
            split_config = _split_dataset_config(
                config,
                split_name,
                rows,
                trajectory_manifest_sha256=manifest["manifest_sha256"],
            )
            generated = generate_closed_loop_dataset(split_config, staging / split_name)
            dataset_hashes[split_name] = generated["dataset_sha256"]
            dataset_manifest_hashes[split_name] = generated["manifest_sha256"]
            implementation_hashes.add(generated["implementation"]["aggregate_sha256"])
        if len(implementation_hashes) != 1:
            raise RuntimeError("Train and validation datasets used different implementations")
        completion_core = {
            "schema_version": 2,
            "status": "complete",
            "source_config_sha256": canonical_json_hash(config),
            "trajectory_manifest_sha256": manifest["manifest_sha256"],
            "materialized_splits": ["train", "val"],
            "test_rows_committed_not_materialized": True,
            "deterministic_full_replay_required_before_training": True,
            "dataset_sha256": dataset_hashes,
            "dataset_manifest_sha256": dataset_manifest_hashes,
            "dataset_implementation_sha256": next(iter(implementation_hashes)),
            "formal_artifact_identities": frozen_identities,
        }
        completion = {
            **completion_core,
            "completion_sha256": canonical_json_hash(completion_core),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "COMPLETE.json", completion)
        os.replace(staging, destination)
        return completion
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _canonical_episode_specs(
    rows: list[dict[str, Any]], default_spec: dict[str, Any]
) -> list[dict[str, Any]]:
    return json.loads(
        json.dumps(
            [
                asdict(
                    PerturbationSpec.from_mapping(
                        row.get("training_corruption", {}).get("spec", default_spec)
                    )
                )
                for row in rows
            ]
        )
    )


def _verify_split_semantics(
    *,
    split: str,
    directory: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any] | None,
    completion: dict[str, Any],
    full_replay: bool,
    runtime_compatibility: dict[str, Any] | None = None,
) -> str:
    arrays, dataset_manifest = load_closed_loop_dataset(directory / split)
    if dataset_manifest.get("source_split") != split:
        raise ValueError(f"Dataset source split mismatch for {split}")
    if (
        dataset_manifest.get("source_trajectory_manifest_sha256")
        != completion["trajectory_manifest_sha256"]
    ):
        raise ValueError(f"Dataset trajectory-manifest binding mismatch for {split}")
    if dataset_manifest.get("source_trajectory_rows") != rows:
        raise ValueError(f"Dataset trajectory-row mapping mismatch for {split}")
    if dataset_manifest.get("source_trajectory_rows_sha256") != canonical_json_hash(rows):
        raise ValueError(f"Dataset trajectory-row hash mismatch for {split}")
    expected_ids = [str(row["trajectory_id"]) for row in rows]
    expected_episode_seeds = [int(row["episode_seed"]) for row in rows]
    expected_schedule_seeds = [int(row["schedule_seed"]) for row in rows]
    if dataset_manifest.get("trajectory_ids") != expected_ids:
        raise ValueError(f"Dataset trajectory ID mismatch for {split}")
    if dataset_manifest.get("episode_seeds") != expected_episode_seeds:
        raise ValueError(f"Dataset episode seed mismatch for {split}")
    if dataset_manifest.get("schedule_seeds") != expected_schedule_seeds:
        raise ValueError(f"Dataset schedule seed mismatch for {split}")

    dataset_config = dict((config or {}).get("dataset", {}))
    default_spec = dataset_config.get("perturbation", {})
    expected_specs = _canonical_episode_specs(rows, default_spec)
    if dataset_manifest.get("episode_perturbation_specs") != expected_specs:
        raise ValueError(f"Dataset perturbation spec mismatch for {split}")
    if int(dataset_manifest.get("episodes", -1)) != len(rows):
        raise ValueError(f"Dataset episode count mismatch for {split}")
    expected_split_config: dict[str, Any] | None = None
    if config is not None:
        expected_split_config = _split_dataset_config(
            config,
            split,
            rows,
            trajectory_manifest_sha256=completion["trajectory_manifest_sha256"],
        )
        if dataset_manifest.get("config_sha256") != canonical_json_hash(
            expected_split_config
        ):
            raise ValueError(f"Dataset config mismatch for {split}")

    if (
        dataset_manifest.get("manifest_sha256")
        != completion["dataset_manifest_sha256"].get(split)
    ):
        raise ValueError(f"Completion marker manifest hash mismatch for {split}")
    if (
        dataset_manifest.get("implementation", {}).get("aggregate_sha256")
        != completion.get("dataset_implementation_sha256")
    ):
        raise ValueError(f"Dataset implementation mismatch for {split}")
    if dataset_manifest.get("dataset_sha256") != completion["dataset_sha256"].get(
        split
    ):
        raise ValueError(f"Completion marker dataset hash mismatch for {split}")

    metadata = dataset_manifest["episode_metadata"]
    offsets = arrays["episode_offsets"]
    control_dt = float(dataset_manifest.get("environment", {}).get("control_dt", 0.0))
    horizon = int(dataset_manifest.get("environment", {}).get("horizon_steps", 0))
    if control_dt <= 0 or horizon <= 0:
        raise ValueError(f"Dataset environment timing is invalid for {split}")
    plant_faults = bool(
        dataset_manifest.get("supervision", {}).get(
            "plant_perturbations_during_rollout", False
        )
    )

    environment = None
    if config is not None:
        environment = _make_environment(dict(config.get("environment", {})))
    try:
        for index, row in enumerate(rows):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            steps = stop - start
            episode_meta = metadata[index]
            if (
                episode_meta.get("trajectory_id") != row["trajectory_id"]
                or int(episode_meta.get("episode_seed", -1)) != int(row["episode_seed"])
                or int(episode_meta.get("schedule", {}).get("seed", -1))
                != int(row["schedule_seed"])
            ):
                raise ValueError(f"Dataset episode provenance mismatch for {split}[{index}]")
            spec = PerturbationSpec.from_mapping(expected_specs[index])
            schedule = make_perturbation_schedule(
                horizon_steps=horizon,
                nominal_dt=control_dt,
                spec=spec,
                seed=int(row["schedule_seed"]),
            )
            expected_schedule_metadata = json.loads(json.dumps(schedule.metadata()))
            if episode_meta.get("schedule") != expected_schedule_metadata:
                raise ValueError(f"Dataset schedule metadata mismatch for {split}[{index}]")
            if not np.array_equal(
                arrays["sensor_mask"][start:stop],
                schedule.sensor_mask[:steps].astype(np.float32),
            ):
                raise ValueError(f"Dataset sensor schedule mismatch for {split}[{index}]")
            if not np.array_equal(
                arrays["observed_timestamp_dt"][start:stop, 0],
                schedule.observed_timestamp_dt[:steps],
            ):
                raise ValueError(f"Dataset timestamp schedule mismatch for {split}[{index}]")
            expected_gain = (
                schedule.actuator_gain[:steps]
                if plant_faults
                else np.ones(steps, dtype=np.float32)
            )
            expected_force = (
                schedule.external_force[:steps]
                if plant_faults
                else np.zeros((steps, 2), dtype=np.float32)
            )
            if not np.array_equal(arrays["actuator_gain"][start:stop], expected_gain):
                raise ValueError(f"Dataset actuator schedule mismatch for {split}[{index}]")
            if not np.array_equal(arrays["external_force"][start:stop], expected_force):
                raise ValueError(f"Dataset force schedule mismatch for {split}[{index}]")
            if not np.array_equal(
                arrays["expert_estimated_state"][start:stop],
                arrays["target_state"][start:stop],
            ):
                raise ValueError(f"Dataset state supervision mismatch for {split}[{index}]")
            if environment is not None:
                initial, _ = environment.reset(seed=int(row["episode_seed"]))
                if not np.allclose(
                    arrays["clean_observation"][start],
                    np.asarray(initial, dtype=np.float32),
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise ValueError(
                        f"Dataset initial state does not match frozen episode seed for {split}[{index}]"
                    )
                if json.loads(json.dumps(environment.trajectory_parameters)) != episode_meta.get(
                    "trajectory"
                ):
                    raise ValueError(
                        f"Dataset trajectory parameters do not match frozen episode seed for {split}[{index}]"
                    )
    finally:
        if environment is not None:
            environment.close()

    if full_replay:
        if expected_split_config is None:
            raise ValueError("Deterministic dataset replay requires the frozen config")
        # Self-hashes prove accidental integrity, not semantic correctness.  A
        # fresh independent replay is therefore compared across every array so
        # an interior transition cannot be changed and then made self-consistent
        # merely by re-hashing the NPZ, manifest, and completion marker.
        with tempfile.TemporaryDirectory(
            prefix=f".{split}-deterministic-replay-", dir=directory.parent
        ) as temporary:
            replay_root = Path(temporary) / split
            generate_closed_loop_dataset(expected_split_config, replay_root)
            replay_arrays, replay_manifest = load_closed_loop_dataset(replay_root)
            if set(replay_arrays) != set(arrays):
                raise ValueError(f"Deterministic replay array names differ for {split}")
            for name in sorted(arrays):
                if not np.array_equal(arrays[name], replay_arrays[name]):
                    raise ValueError(
                        f"Dataset differs from deterministic full replay for {split}:{name}"
                    )
            excluded = {
                "manifest_sha256",
                "created_at_utc",
                "dataset_sha256",
            }
            observed_semantics = {
                key: value for key, value in dataset_manifest.items() if key not in excluded
            }
            replay_semantics = {
                key: value for key, value in replay_manifest.items() if key not in excluded
            }
            if runtime_compatibility is not None:
                if replay_semantics.get("implementation") != runtime_compatibility["current_runtime_implementation"]:
                    raise ValueError("Replay runtime source differs from the independent compatibility review")
                observed_semantics["implementation"] = runtime_compatibility["current_runtime_implementation"]
            if observed_semantics != replay_semantics:
                raise ValueError(f"Dataset manifest differs from deterministic replay for {split}")
    return str(dataset_manifest["dataset_sha256"])


def verify_split_datasets(
    root: str | Path,
    *,
    expected_config: dict[str, Any] | None = None,
    full_replay: bool | None = None,
) -> dict[str, Any]:
    directory = Path(root).resolve()
    complete = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
    completion_core = _manifest_core(complete, "completion_sha256")
    if complete.get("schema_version") != 2 or canonical_json_hash(
        completion_core
    ) != complete.get("completion_sha256"):
        raise ValueError("Closed-loop data completion marker hash mismatch")
    if (
        complete.get("materialized_splits") != ["train", "val"]
        or complete.get("test_rows_committed_not_materialized") is not True
        or complete.get("deterministic_full_replay_required_before_training") is not True
        or (directory / "test").exists()
    ):
        raise ValueError("Test data must remain committed but unmaterialized before release")
    if set(complete.get("dataset_sha256", {})) != {"train", "val"} or set(
        complete.get("dataset_manifest_sha256", {})
    ) != {"train", "val"}:
        raise ValueError("Completion marker must bind exactly train and validation datasets")
    manifest = load_trajectory_manifest(
        directory / "trajectory_manifest.json", expected_config=expected_config
    )
    if complete.get("trajectory_manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Completion marker and split manifest disagree")
    runtime_compatibility = None
    if expected_config is not None:
        from .formal import formal_artifact_identities

        if complete.get("source_config_sha256") != canonical_json_hash(expected_config):
            raise ValueError("Completion marker source config mismatch")
        if complete.get("formal_artifact_identities") != formal_artifact_identities(
            expected_config
        ):
            raise ValueError("Completion marker formal identities mismatch")
        current_implementation = source_tree_manifest(Path(__file__).resolve().parents[1])
        if (
            complete.get("dataset_implementation_sha256")
            != current_implementation["aggregate_sha256"]
        ):
            runtime_compatibility = _validated_runtime_compatibility(
                directory, complete, expected_config, current_implementation
            )
    replay_enabled = expected_config is not None if full_replay is None else bool(full_replay)
    if replay_enabled and expected_config is None:
        raise ValueError("full_replay=True requires expected_config")
    verified = {
        split: _verify_split_semantics(
            split=split,
            directory=directory,
            rows=manifest["splits"][split],
            config=expected_config,
            completion=complete,
            full_replay=replay_enabled,
            runtime_compatibility=runtime_compatibility,
        )
        for split in ("train", "val")
    }
    return {
        "status": "verified",
        "materialized_splits": ["train", "val"],
        "test_rows_committed_not_materialized": True,
        "dataset_sha256": verified,
        "completion_sha256": complete["completion_sha256"],
        "deterministic_full_replay_verified": replay_enabled,
    }
