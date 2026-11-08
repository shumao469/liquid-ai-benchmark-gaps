"""Parameter-count registration and paired NCP/null-topology construction."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..io_utils import atomic_write_json, canonical_json_hash, sha256_file
from ..models import build_model, make_degree_matched_wiring_pair
from ..models.topology import make_flat_ncp_wiring


REGISTERED_MODELS = (
    "lstm",
    "odernn",
    "ctrnn",
    "ltc",
    "cfc",
    "ncp_cfc",
    "random_sparse_cfc",
)


def parameter_count_audit(model) -> dict[str, Any]:
    """Count nominal allocated and effective-active trainable scalars.

    For a dense model, every scalar in a ``requires_grad`` parameter is both
    nominally allocated and effective-active.  Direct GraphCfC instead counts
    mask-allowed weight scalars in each of its four affine heads (``ff1``,
    ``ff2``, ``time_a``, and ``time_b``), plus all trainable biases and any
    other trainable scalars.  Mask-forbidden allocated weight slots are
    excluded from the effective-active count because the frozen mask makes
    their forward contribution and gradient identically zero.  This is a
    scalar parameter-capacity count, not an edge or node-degree count; nominal
    allocated trainable scalars and effective-active trainable scalars are
    reported separately.
    """
    nominal = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    effective = nominal
    masked_heads: dict[str, dict[str, int]] = {}
    if getattr(model, "topology_audit", {}).get("dense_sparse_common_cell"):
        cell = model.core.rnn_cell._layers[0]
        mask = cell.sparsity_mask
        if mask is None:
            raise RuntimeError("Graph-CfC parameter audit requires a frozen mask")
        active_per_head = int(torch.count_nonzero(mask).item())
        for name in ("ff1", "ff2", "time_a", "time_b"):
            weight = getattr(cell, name).weight
            allocated = int(weight.numel())
            if tuple(weight.shape) != tuple(mask.shape):
                raise RuntimeError(f"Graph mask shape mismatch for {name}")
            effective -= allocated - active_per_head
            masked_heads[name] = {
                "allocated_weights": allocated,
                "active_weights": active_per_head,
                "forbidden_zero_gradient_weights": allocated - active_per_head,
            }
    masked_allocated = int(
        sum(row["allocated_weights"] for row in masked_heads.values())
    )
    masked_active = int(sum(row["active_weights"] for row in masked_heads.values()))
    return {
        "nominal_allocated_trainable_parameters": nominal,
        "effective_active_trainable_parameters": int(effective),
        "masked_heads": masked_heads,
        "matching_basis": "effective_active_trainable_parameters",
        "adamw_moment_state_scalars": int(2 * nominal),
        "parameter_plus_adamw_moment_scalars": int(3 * nominal),
        "masked_affine_allocated_weight_slots_per_recurrent_update": masked_allocated,
        "masked_affine_active_weight_slots_per_recurrent_update": masked_active,
        "masked_affine_dense_compute_overhead_ratio": (
            float(masked_allocated / masked_active) if masked_active else None
        ),
    }


def _closed_loop_implementation_manifest() -> dict[str, Any]:
    """Hash only code that can affect a closed-loop model/checkpoint.

    Offline benchmark modules are intentionally excluded so an independent
    offline edit cannot invalidate an already frozen closed-loop registry.
    """
    root = Path(__file__).resolve().parents[1]
    candidates = list((root / "closed_loop").rglob("*.py"))
    candidates += list((root / "models").rglob("*.py"))
    candidates += list((root / "envs").rglob("*.py"))
    candidates += [
        root / "data" / "closed_loop.py",
        root / "controllers.py",
        root / "perturbations.py",
        root / "evaluation.py",
        root / "config.py",
        root / "io_utils.py",
    ]
    files = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(set(candidates))
        if path.is_file() and "__pycache__" not in path.parts
    }
    return {
        "root": str(root),
        "file_count": len(files),
        "aggregate_sha256": canonical_json_hash(files),
        "files": files,
    }


def _base_model_config(config: dict[str, Any], hidden_dim: int) -> dict[str, Any]:
    source = dict(config.get("models", {}))
    return {
        **source,
        "input_dim": int(source.get("input_dim", 19)),
        "state_dim": int(source.get("state_dim", 4)),
        "hidden_dim": int(hidden_dim),
    }


def _candidate(
    registered_name: str, config: dict[str, Any], hidden_dim: int
):
    model_config = _base_model_config(config, hidden_dim)
    if registered_name in {"ncp_cfc", "random_sparse_cfc"}:
        matching = dict(config.get("matching", {}))
        wiring, topology_audit = make_flat_ncp_wiring(
            hidden_dim=hidden_dim,
            input_dim=int(model_config["input_dim"]),
            output_dim=int(model_config["state_dim"]),
            sparsity=float(matching.get("ncp_sparsity", 0.5)),
            seed=int(matching.get("ncp_seed", 20261102)),
        )
        model_config["topology"] = {
            "kind": "autoncp",
            "sparsity": float(matching.get("ncp_sparsity", 0.5)),
            "seed": int(matching.get("ncp_seed", 20261102)),
        }
        model_config["_wiring_override"] = wiring
        model_config["_topology_audit"] = topology_audit
        return build_model("cfc", model_config)
    model_config["topology"] = {"kind": "dense", "sparsity": 0.0, "seed": 0}
    return build_model(registered_name, model_config)


def _candidate_parameter_audit(
    registered_name: str, config: dict[str, Any], hidden_dim: int
) -> dict[str, Any]:
    if registered_name not in {"ncp_cfc", "random_sparse_cfc"}:
        return parameter_count_audit(_candidate(registered_name, config, hidden_dim))
    model_config = _base_model_config(config, hidden_dim)
    mode = str(model_config.get("liquid", {}).get("cfc_mode", "default"))
    if mode not in {"default", "no_gate"}:
        raise ValueError("Effective graph-CfC matching is registered for four-head modes")
    matching = dict(config.get("matching", {}))
    _, topology = make_flat_ncp_wiring(
        hidden_dim=hidden_dim,
        input_dim=int(model_config["input_dim"]),
        output_dim=int(model_config["state_dim"]),
        sparsity=float(matching.get("ncp_sparsity", 0.5)),
        seed=int(matching.get("ncp_seed", 20261102)),
    )
    active_per_head = int(topology["recurrent_edges"] + topology["sensory_edges"])
    allocated_per_head = int(
        hidden_dim * (int(model_config["input_dim"]) + hidden_dim)
    )
    heads = {
        name: {
            "allocated_weights": allocated_per_head,
            "active_weights": active_per_head,
            "forbidden_zero_gradient_weights": allocated_per_head - active_per_head,
        }
        for name in ("ff1", "ff2", "time_a", "time_b")
    }
    nominal = int(4 * (allocated_per_head + hidden_dim))
    effective = int(4 * (active_per_head + hidden_dim))
    masked_allocated = int(4 * allocated_per_head)
    masked_active = int(4 * active_per_head)
    return {
        "nominal_allocated_trainable_parameters": nominal,
        "effective_active_trainable_parameters": effective,
        "masked_heads": heads,
        "matching_basis": "effective_active_trainable_parameters",
        "adamw_moment_state_scalars": int(2 * nominal),
        "parameter_plus_adamw_moment_scalars": int(3 * nominal),
        "masked_affine_allocated_weight_slots_per_recurrent_update": masked_allocated,
        "masked_affine_active_weight_slots_per_recurrent_update": masked_active,
        "masked_affine_dense_compute_overhead_ratio": float(
            masked_allocated / masked_active
        ),
    }


def _best_width(
    registered_name: str,
    config: dict[str, Any],
    target: int,
    minimum: int,
    maximum: int,
) -> tuple[int, dict[str, Any]]:
    best: tuple[int, int, dict[str, Any]] | None = None
    for width in range(minimum, maximum + 1):
        if registered_name in {"ncp_cfc", "random_sparse_cfc"} and width <= 6:
            continue
        audit = _candidate_parameter_audit(registered_name, config, width)
        count = int(audit["effective_active_trainable_parameters"])
        candidate = (abs(count - target), width, audit)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        # Counts are monotone for every registered implementation. Once the
        # target has been crossed and the error worsens, larger widths cannot
        # improve the match.
        if (
            registered_name not in {"ncp_cfc", "random_sparse_cfc"}
            and count >= target
            and best is not None
            and candidate[0] > best[0]
        ):
            break
    if best is None:
        raise RuntimeError(f"No width candidates for {registered_name}")
    return best[1], best[2]


def build_parameter_registry(config: dict[str, Any]) -> dict[str, Any]:
    matching = dict(config.get("matching", {}))
    if matching.get(
        "basis", "effective_active_trainable_parameters"
    ) != "effective_active_trainable_parameters":
        raise ValueError(
            "Closed-loop matching basis must be effective_active_trainable_parameters"
        )
    target = int(matching.get("target_parameters", 20_000))
    tolerance = float(matching.get("tolerance_fraction", 0.05))
    minimum = int(matching.get("minimum_hidden_dim", 4))
    maximum = int(matching.get("maximum_hidden_dim", 256))
    if target <= 0 or not 0 < tolerance <= 0.20 or minimum <= 0 or maximum < minimum:
        raise ValueError("Invalid parameter-matching configuration")
    requested = tuple(matching.get("registered_models", REGISTERED_MODELS))
    unknown = set(requested) - set(REGISTERED_MODELS)
    if unknown:
        raise ValueError(f"Unknown registered models: {sorted(unknown)}")

    entries: dict[str, dict[str, Any]] = {}
    pair_width: int | None = None
    pair_audit: dict[str, Any] | None = None
    for name in requested:
        if name == "random_sparse_cfc" and pair_width is not None:
            width, count_audit = pair_width, pair_audit
        else:
            width, count_audit = _best_width(name, config, target, minimum, maximum)
            if name == "ncp_cfc":
                pair_width, pair_audit = width, count_audit
        assert count_audit is not None
        count = int(count_audit["effective_active_trainable_parameters"])
        nominal_count = int(count_audit["nominal_allocated_trainable_parameters"])
        error = abs(count - target) / target
        if error > tolerance:
            raise ValueError(
                f"{name} closest count {count} at width {width} exceeds "
                f"the registered +/-{tolerance:.1%} tolerance around {target}"
            )
        entries[name] = {
            "registered_name": name,
            "base_architecture": "cfc" if name.endswith("cfc") else name,
            "hidden_dim": int(width),
            "trainable_parameters": nominal_count,
            "trainable_parameters_semantics": (
                "deprecated_alias_of_nominal_allocated_trainable_parameters"
            ),
            "nominal_allocated_trainable_parameters": nominal_count,
            "effective_active_trainable_parameters": count,
            "parameter_matching_basis": "effective_active_trainable_parameters",
            "masked_parameter_audit": count_audit["masked_heads"],
            "adamw_moment_state_scalars": count_audit[
                "adamw_moment_state_scalars"
            ],
            "parameter_plus_adamw_moment_scalars": count_audit[
                "parameter_plus_adamw_moment_scalars"
            ],
            "masked_affine_allocated_weight_slots_per_recurrent_update": count_audit[
                "masked_affine_allocated_weight_slots_per_recurrent_update"
            ],
            "masked_affine_active_weight_slots_per_recurrent_update": count_audit[
                "masked_affine_active_weight_slots_per_recurrent_update"
            ],
            "masked_affine_dense_compute_overhead_ratio": count_audit[
                "masked_affine_dense_compute_overhead_ratio"
            ],
            "target_parameters": target,
            "relative_error": float((count - target) / target),
            "within_tolerance": True,
            "cell_implementation": (
                "local_full_state_synchronous_direct_graph_cfc"
                if name in {"cfc", "ncp_cfc", "random_sparse_cfc"}
                else None
            ),
            "topology": (
                "autoncp"
                if name == "ncp_cfc"
                else "degree_matched_random"
                if name == "random_sparse_cfc"
                else "dense"
            ),
        }

    pair_names = {"ncp_cfc", "random_sparse_cfc"}
    topology_pair: dict[str, Any] | None = None
    if pair_names <= set(entries):
        if entries["ncp_cfc"]["hidden_dim"] != entries["random_sparse_cfc"]["hidden_dim"]:
            raise AssertionError("Topology pair must use the same hidden width")
        width = int(entries["ncp_cfc"]["hidden_dim"])
        models = dict(config.get("models", {}))
        _, _, topology_pair = make_degree_matched_wiring_pair(
            hidden_dim=width,
            input_dim=int(models.get("input_dim", 19)),
            output_dim=int(models.get("state_dim", 4)),
            sparsity=float(matching.get("ncp_sparsity", 0.5)),
            ncp_seed=int(matching.get("ncp_seed", 20261102)),
            random_seed=int(matching.get("random_topology_seed", 20261103)),
        )
    from .formal import formal_artifact_identities

    frozen_identities = formal_artifact_identities(config)
    core = {
        "schema_version": 1,
        "target_parameters": target,
        "matching_basis": "effective_active_trainable_parameters",
        "tolerance_fraction": tolerance,
        "search_range": [minimum, maximum],
        "entries": entries,
        "topology_pair": topology_pair,
        "topology_seeds": {
            "ncp_seed": int(matching.get("ncp_seed", 20261102)),
            "random_seed": int(matching.get("random_topology_seed", 20261103)),
            "ncp_sparsity": float(matching.get("ncp_sparsity", 0.5)),
        },
        "source_config_sha256": canonical_json_hash(config),
        "implementation": _closed_loop_implementation_manifest(),
        "formal_artifact_identities": frozen_identities,
    }
    return {
        **core,
        "registry_sha256": canonical_json_hash(core),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_csv(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "registered_name",
                    "base_architecture",
                    "topology",
                    "hidden_dim",
                    "trainable_parameters",
                    "trainable_parameters_semantics",
                    "nominal_allocated_trainable_parameters",
                    "effective_active_trainable_parameters",
                    "parameter_matching_basis",
                    "adamw_moment_state_scalars",
                    "parameter_plus_adamw_moment_scalars",
                    "masked_affine_allocated_weight_slots_per_recurrent_update",
                    "masked_affine_active_weight_slots_per_recurrent_update",
                    "masked_affine_dense_compute_overhead_ratio",
                    "target_parameters",
                    "relative_error",
                    "within_tolerance",
                ],
            )
            writer.writeheader()
            for name in entries:
                writer.writerow(
                    {
                        key: value
                        for key, value in entries[name].items()
                        if key in writer.fieldnames
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def create_parameter_registry(
    config: dict[str, Any], output_path: str | Path
) -> dict[str, Any]:
    destination = Path(output_path).resolve()
    csv_path = destination.with_suffix(".csv")
    if destination.exists() or csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite parameter registry: {destination}")
    registry = build_parameter_registry(config)
    atomic_write_json(destination, registry)
    _write_csv(csv_path, registry["entries"])
    return registry


def load_parameter_registry(
    path: str | Path, *, expected_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    source = Path(path).resolve()
    registry = json.loads(source.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in registry.items()
        if key not in {"registry_sha256", "created_at_utc"}
    }
    if canonical_json_hash(core) != registry.get("registry_sha256"):
        raise ValueError(f"Parameter registry hash mismatch: {source}")
    if expected_config is not None:
        expected = build_parameter_registry(expected_config)
        expected_core = {
            key: value
            for key, value in expected.items()
            if key not in {"registry_sha256", "created_at_utc"}
        }
        if core != expected_core:
            raise ValueError(
                "Parameter registry differs from the deterministic rebuild of the frozen config"
            )
    return registry


def _validate_topology_pair_identity(
    rebuilt: dict[str, Any], frozen: Any
) -> None:
    """Validate the frozen pair identity while permitting model-level evidence.

    A parameter registry stores the pair-level topology identity only.  A trained
    checkpoint stores ``model.topology_audit``, which is deliberately a superset:
    it also records the selected graph and evidence that the mask was applied to
    every CfC head.  Requiring whole-dictionary equality therefore makes a valid
    sparse checkpoint impossible to reload.  The deterministic rebuild defines
    the complete pair-identity schema, so every rebuilt field must be present and
    equal; additional checkpoint audit fields are non-identity annotations and
    are ignored here.
    """
    if not isinstance(frozen, dict):
        raise ValueError("Frozen topology pair must be a mapping")
    missing = sorted(set(rebuilt) - set(frozen))
    mismatched = sorted(
        key for key, value in rebuilt.items() if frozen.get(key) != value
    )
    if missing or mismatched:
        raise ValueError("Rebuilt topology pair differs from the frozen registry")


def build_registered_model(
    registered_name: str, config: dict[str, Any], registry: dict[str, Any]
):
    if registry.get("source_config_sha256") != canonical_json_hash(config):
        raise ValueError(
            "Current config differs from the frozen parameter-registry source"
        )
    if registry.get("implementation") is not None:
        current = _closed_loop_implementation_manifest()
        if current["aggregate_sha256"] != registry["implementation"]["aggregate_sha256"]:
            raise ValueError(
                "Current implementation differs from the frozen parameter registry; "
                "create a new registry rather than reusing stale widths"
            )
    if registered_name not in registry["entries"]:
        raise KeyError(f"Model is not in the frozen registry: {registered_name}")
    entry = registry["entries"][registered_name]
    model_config = _base_model_config(config, int(entry["hidden_dim"]))
    if registered_name in {"ncp_cfc", "random_sparse_cfc"}:
        seeds = registry["topology_seeds"]
        ncp, random_wiring, audit = make_degree_matched_wiring_pair(
            hidden_dim=int(entry["hidden_dim"]),
            input_dim=int(model_config["input_dim"]),
            output_dim=int(model_config["state_dim"]),
            sparsity=float(seeds["ncp_sparsity"]),
            ncp_seed=int(seeds["ncp_seed"]),
            random_seed=int(seeds["random_seed"]),
        )
        _validate_topology_pair_identity(audit, registry.get("topology_pair"))
        model_config["topology"] = {
            "kind": "autoncp",
            "sparsity": float(seeds["ncp_sparsity"]),
            "seed": int(seeds["ncp_seed"]),
        }
        model_config["_wiring_override"] = (
            ncp if registered_name == "ncp_cfc" else random_wiring
        )
        model_config["_topology_audit"] = {
            **audit,
            "selected_graph": (
                "ncp" if registered_name == "ncp_cfc" else "degree_matched_random"
            ),
        }
        model = build_model("cfc", model_config)
    else:
        model_config["topology"] = {"kind": "dense", "sparsity": 0.0, "seed": 0}
        model = build_model(str(entry["base_architecture"]), model_config)
    observed_audit = parameter_count_audit(model)
    observed = int(observed_audit["nominal_allocated_trainable_parameters"])
    observed_effective = int(
        observed_audit["effective_active_trainable_parameters"]
    )
    if observed != int(entry["nominal_allocated_trainable_parameters"]):
        raise ValueError(
            f"Parameter count changed for {registered_name}: {observed} != "
            f"{entry['nominal_allocated_trainable_parameters']}"
        )
    if observed_effective != int(entry["effective_active_trainable_parameters"]):
        raise ValueError(
            f"Effective active parameter count changed for {registered_name}: "
            f"{observed_effective} != {entry['effective_active_trainable_parameters']}"
        )
    return model
